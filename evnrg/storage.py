import os
import errno
import uuid
import re
import pathlib
from typing import NamedTuple
import tempfile
import random
import time

from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
    Provider,
    ContainerDoesNotExistError,
    ObjectDoesNotExistError,
    LibcloudError
)
from libcloud.storage.providers import get_driver, DRIVERS
import appdirs
import pandas as pd
import dask

TMP_DIR = appdirs.user_cache_dir('evnrg')
CONFIG_DIR = appdirs.user_config_dir('evnrg')
LOCAL_DATA = appdirs.user_data_dir('evnrg')
LOCAL_STORAGE = os.path.join(LOCAL_DATA, 'local-object-storage')


# These are all aliases for other strings
PROVIDER_ALIASES = {
    # Amazon S3
    's3': Provider.S3,
    'amazon': Provider.S3,

    # Google Cloud Storage
    'gcs': Provider.GOOGLE_STORAGE,
    'google': Provider.GOOGLE_STORAGE,

    # Backblaze B2
    'b2': Provider.BACKBLAZE_B2,
    'backblaze': Provider.BACKBLAZE_B2,

    # Digital Ocean Spaces
    'digitalocean': Provider.DIGITALOCEAN_SPACES,
    'do': Provider.DIGITALOCEAN_SPACES,
    'spaces': Provider.DIGITALOCEAN_SPACES,

    # Microsoft Azure
    'microsoft': Provider.AZURE_BLOBS,
    'azure': Provider.AZURE_BLOBS,
}


class StorageInfo(NamedTuple):
    """A `NamedTuple` that tells the simulation how to handle data.
    
    Attributes:
        key (str): ID or login key for the cloud account. For example,
            Google Cloud Services uses an IAM email address, and
            can be found in the 'client_email' key of a generated
            JSON file. (See [Google Cloud's IAM documentation](https://cloud.google.com/iam/docs/)
            for more.) For local storage, the key is the base path
            that will be used while emulating cloud storage. Defaults
            to `~/.local/share/evnrg/local-object-storage`.
        secret (str): Secret key or passphrase for the cloud account. For
            Goggle Cloud, it will be in the 'private_key' field in the
            JSON file. Defaults to empty string ('').
        bucket (str): The name of the container (or bucket) to look for data
            and store it in. Defaults to 'evnrg-default'. For local storage,
            this folder will exist under the key: `<key>/<bucket>`.
        provider (str): The provider to use. Takes any provider string object
            that [Apache LibCloud](https://libcloud.readthedocs.io/en/latest/supported_providers.html#id195)
            takes. Additional provider shotcuts are provided as well:
            * Google Cloud Storage: 'gcs', 'google'
            * Amazon S3: 's3', 'amazon'
            * Backblaze B2: 'b2', 'backblaze'
            * Digital Ocean Spaces: 'do', 'spaces', 'digitalocean'
            * Microsoft Azure: 'azure', 'microsoft'
            Defaults to 'local'.
        cache_dir (str): The local cache directory for use when dumping
            dataframes for upload, and downloading dataframes for use.
            Defaults to `~/.cache/evnrg`.
        create_bucket (bool): Specifies if a bucket should be created if it
            does not exist. Defaults to `True`.
    """
    key: str = LOCAL_STORAGE
    secret: str = ''
    bucket: str = 'evnrg-default'
    provider: str = 'local'
    cache_dir: str = TMP_DIR
    create_bucket: bool = True

DF_WRITE = {
    'parquet': lambda d, p: d.to_parquet(
        p,
        engine='fastparquet',
        compression='gzip'
    ),
    'csv': lambda d, p: d.to_csv(
        p,
        date_format='%Y-%m-%dT%H:%M:%S'
    ),
    'records.json': lambda d, p: d.to_json(
        p,
        orient='records',
        date_format='iso',
        date_unit='s'
    ),
    'json': lambda d, p: d.to_json(
        p,
        orient='split',
        date_format='iso',
        date_unit='s'
    )
}

DF_READ = {
    'parquet': lambda p, **kwargs: pd.read_parquet(
        p,
        **{'engine': 'fastparquet', **kwargs}
    ),
    'csv':  lambda p, **kwargs: pd.read_csv(
        p,
        **kwargs
    ),
    'records.json': lambda p, **kwargs: pd.read_json(
        p,
        **{'orient': 'records', **kwargs}
    ),
    'json': lambda p, **kwargs: pd.read_json(
        p,
        **{'orient': 'split', **kwargs}
    )
}

class Storage(object):

    __slots__ = (
        'driver',
        'container',
        'temporary_dir'
    )

    def __init__(self, storage_info: StorageInfo):

        if bool(storage_info.provider):
            provider_id = PROVIDER_ALIASES.get(
                storage_info.provider,
                storage_info.provider
            )
            if provider_id not in DRIVERS.keys():
                raise IndexError('Invalid object strage provider string.')

        self.driver = get_driver(provider_id)(
            storage_info.key,
            storage_info.secret
        )

        self.temporary_dir = tempfile.TemporaryDirectory()

        try:
            self.container = self.driver.get_container(
                container_name=storage_info.bucket
            )
        except ContainerDoesNotExistError:
            if storage_info.create_container:
                self.container = self.driver.create_container(
                        container_name=storage_info.bucket
                    )
            else:
                self.container = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.temporary_dir.cleanup()
    
    @property
    def temp_dir(self):
        return self.temporary_dir.name
    
    @classmethod
    def gen_remotepath(cls,
        resource_type: str,
        run_id: str = None,
        base: str = 'results',
        name: str = uuid.uuid4().hex,
        scenario_id: str = None,
        fleet_id: str = None,
        ext: str = None):

        fname = '.'.join(filter(None, (name, ext)))

        return '/'.join(
            filter(
                None,
                (base, run_id, resource_type, scenario_id, fleet_id, fname)
            )
        )
    
    def gen_localpath(self, ext: str = ''):
        return os.path.join(self.temporary_dir.name, uuid.uuid4().hex) + ext
    
    def upload_file(self, 
                    local_path: str, 
                    remote_path: str, 
                    file_type: str,
                    tries: int = 4):
        
        if not os.path.isfile(local_path):
            raise FileNotFoundError(
                'Not found: {}'.format(local_path)
            )
        
        success = False

        for i in range(tries):
            try:
            
                o = self.driver.upload_object(
                    local_path,
                    self.container,
                    remote_path
                )
                if o:
                    success = True
                    break
            except LibcloudError:
                time.sleep(2**(1+i) + random.random())

        return {
            'remote_path': remote_path,
            'file_type': file_type,
            'uploaded': success
        }
    
    def download_file(self, obj_path: str):

        o = None
        try:
            o = self.driver.get_object(self.container.name, obj_path)
        except ObjectDoesNotExistError:
            return None

        if o is None:
            return None

        tmp_name = self.gen_localpath()

        if self.driver.download_object(o, tmp_name, True, True):

            # retry 3 times
            tries = 0
            while not os.path.isfile(tmp_name) and tries < 3:
                self.driver.download_object(o, tmp_name, True, True)
                tries += 1


            if not os.path.isfile(tmp_name) and tries >= 3:
                return None
        
        if os.path.isfile(tmp_name):
            return tmp_name
        return None


    @classmethod
    def upload_df(cls,
        si: StorageInfo,
        df: pd.DataFrame,
        obj_path: str, 
        fmt: str = 'parquet'):
                
        w = DF_WRITE.get(fmt)

        if not w:
            raise KeyError('Invalid format. Valid formats are: {}'.format(', '.join(DF_WRITE.keys())))
        
        with Storage(si) as dh:

            p = dh.gen_localpath(ext=fmt)
            
            w(df, p)

            res = dh.upload_file(
                local_path=p,
                remote_path=obj_path,
                file_type=fmt
            )
            return res
        raise IOError('Could not write/upload dataframe.')
    
    @classmethod
    def upload_fig(cls,
        si: StorageInfo,
        fig,
        obj_path: str,
        fmt: str = 'svg',
        dpi: int = 300):

        if not name:
            name = uuid.uuid4().hex
        
        with Storage(si) as dh:
            p = dh.gen_localpath(ext=fmt)
            fig.savefig(p, dpi=dpi)

            res = dh.upload_file(
                local_path=p,
                remote_path=obj_path,
                file_type=fmt
            )
            return res
        raise IOError('Could not write/upload figure.')
    
    @classmethod
    def read_df(cls,
        si: StorageInfo,
        obj_path: str,
        fmt: str = 'parquet',
        **kwargs):
        
        with Storage(si) as dh:

            fname = dh.download_file(obj_path):

            if not fname:
                raise FileNotFoundError(
                    'Could not download object: {}'.format(obj_path)
                )
            
            read_func = DF_READ.get(fmt)

            if not read_func:
                raise KeyError(
                    'Invalid read function/format. Valid formats are: {}'.format(', '.join(DF_READ.keys()))
                )
            
            return read_func(fname, **kwargs)
            
        raise IOError('Could not read dataframe from: {}'.format(obj_path))
    
    @classmethod
    def read_delayed_dfs(
        cls,
        paths: list,
        fmt: str = 'parquet',
        si: StorageInfo = None,
        use_ddf: bool = True,
        **kwargs):

        dfs = []
        read_func = DF_READ.get(fmt)
        dtypes = None
        if not read_func:
                raise KeyError(
                    'Invalid read function/format. Valid formats are: {}'.format(', '.join(DF_READ.keys()))
                )
        
        if si:
            with Storage(si) as dh:
                for p in paths:
                    fname = dh.download_file(p)

                    if not fname:
                        raise FileNotFoundError(
                            'Could not download object: {}'.format(obj_path)
                        )
                    
                    df = read_func(fname, **kwargs)
                    dfs.append(dask.delayed(df))
                    if not dtypes:
                        dtypes = df.dtypes.to_dict()
        else:
            for p in paths:

                if not os.path.exists(p):
                    raise FileNotFoundError(
                        'Could not read file: {}'.format(p)
                    )
                
                df = read_func(fname, **kwargs)
                dfs.append(dask.delayed(df))

                if not dtypes:
                    dtypes = df.dtypes.to_dict()
        
        if use_ddf:
            return dask.dataframe.from_delayed(dfs, meta=dtypes)
        
        return dfs

    def read_data(self, obj_name: str, fmt: str = 'parquet',
                  read_fn=None, arguments: dict = {}):
        o = None
        try:
            o = self.driver.get_object(self.container.name, obj_name)
        except ObjectDoesNotExistError:
            return None

        if o is None:
            return None

        fname = uuid.uuid4().hex + '-' + obj_name.split('/')[-1]

        tmp_name = os.path.join(self.cache_dir, fname)

        if self.driver.download_object(o, tmp_name, True, True):

            # retry 3 times
            tries = 0
            while not os.path.isfile(tmp_name) and tries < 3:
                self.driver.download_object(o, tmp_name, True, True)
                tries += 1


            if not os.path.isfile(tmp_name) and tries >= 3:
                return None
            
            self.temp.append(tmp_name)
            
            if fmt == 'parquet':
                # Default to fastparquet
                d_ = {'engine': 'fastparquet'}
                d_.update(arguments)
                return pd.read_parquet(tmp_name, **d_)

            elif fmt == 'csv':
                return pd.read_csv(tmp_name, **arguments)
            
            elif fmt == 'json':
                d_ = {'orient': 'split'}
                d_.update(arguments)
                return pd.read_json(tmp_name, **d_)
            
            elif fmt == 'json-records':
                d_ = {'orient': 'records'}
                d_.update(arguments)
                return pd.read_json(tmp_name, **d_)
            
            elif callable(read_fn):
                return read_fn(tmp_name, **arguments)
            
        return None
     
def load_dataframe(obj_path: str, si: StorageInfo, fmt: str = 'parquet'):
   
    with Storage(si) as dh:

        df = dh.read_df(
            obj_path=obj_path,
            fmt=fmt
        )
        if df:
            return df
        raise IOError('Could not load data from path: {}'.format(obj_path))

    return None

def write_data(
    df: pd.DataFrame,
    si: StorageInfo,
    obj_path: str,
    name: str = None,
    fmt = 'records.json'):


    with Storage(si) as dh:

        results = dh.upload_df(
            df,
            obj_path=obj_path,
            name=name,
            fmt=fmt
        )

        return results

def write_plot(
    fig,
    si: StorageInfo,
    obj_path: str,
    name: str = None,
    fmt: str='svg',
    dpi: int = 300):

    with Storage(si) as dh:
        return dh.upload_fig(
            fig=fig,
            obj_path=obj_path,
            name=name,
            fmt=fmt,
            dpi=dpi
        )

