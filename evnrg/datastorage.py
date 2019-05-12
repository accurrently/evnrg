import os
import errno
import uuid
import re
import pathlib
from typing import NamedTuple

from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
    Provider,
    ContainerDoesNotExistError,
    ObjectDoesNotExistError
)
from libcloud.storage.providers import get_driver, DRIVERS
import appdirs
import pandas as pd

from .dataset import DatasetInfo

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

    def gen_temp_path(self, ext: str = None):
        if ext:
            return os.path.join(self.cache_dir, uuid.uuid4().hex) + '.' + ext
        return os.path.join(self.cache_dir, uuid.uuid4().hex)
        


class UploadResult(NamedTuple):
    """Contains data about the uploaded object.

    Attributes:
        uid (str): The object's uuid.
        filetype (str): The object's filetype. Will generally be one of
            `'parquet'`, `'json'`, `'records'`, or `'csv'`.
        cache_path (str): The path to the temporary file.
        remote_path (str): The object's remote object name.
        obj (libcloud.storage.base.Object): The LibCloud storage object.
    """
    uid: str
    filetype: str
    cache_path: str
    remote_path: str
    obj: Object


class DataHandler(object):

    __slots__ = (
        'driver',
        'container',
        'cache_dir',
        'temp'
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

        self.cache_dir = storage_info.cache_dir

        # Attempt to make the dir
        try:
            os.makedirs(self.cache_dir)
        except Exception as e:
            # Path already exists
            pass


        self.temp = []

        # Create the path if it doesn't exist
        pathlib.Path(self.cache_dir).mkdir(parents=True, exist_ok=True) 

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

    @property
    def ready(self):
        return self.container is not None
    
    def use_bucket(self, bucket: str, create: False):
        try:
            self.container = self.driver.get_container(container_name=bucket)
            return True
        except ContainerDoesNotExistError:
            if create:
                try:
                    self.container = self.driver.create_container(
                        container_name=bucket
                    )
                    return True
                except Exception:
                    self.container = None
            else:
                self.container = None
        return False

    def cleanup(self):
        for fn in self.temp:
            if os.path.isfile(fn):
                os.remove(fn)
    
    @classmethod
    def write_parquet(cls, df: pd.DataFrame, cache_dir: str):

        uid = uuid.uuid4().hex
        p = os.path.join(cache_dir, uid) + '.csv'

        df.to_parquet(
            p,
            engine='fastparquet',
            compression='gzip'
        )

        return p
    
    @classmethod
    def write_csv(cls, df: pd.DataFrame, cache_dir: str):

        uid = uuid.uuid4().hex
        p = os.path.join(cache_dir, uid) + '.csv'

        df.to_csv(
            p,
            date_format='%Y-%m-%dT%H:%M:%S'
        )

        return p
    
    @classmethod
    def write_records(cls, df: pd.DataFrame, cache_dir: str):

        uid = uuid.uuid4().hex
        p = os.path.join(cache_dir, uid) + '.records.json'

        df.to_json(
            p,
            orient='records',
            date_format='iso',
            date_unit='s'
        )

        return p
    
    @classmethod
    def write_json(cls, df: pd.DataFrame, cache_dir: str):

        uid = uuid.uuid4().hex
        p = os.path.join(cache_dir, uid) + '.records.json'

        df.to_json(
            p,
            orient='split',
            date_format='iso',
            date_unit='s'
        )

        return p
    
    def upload_file(self, 
                    local_path: str, 
                    remote_path: str, 
                    file_type: str,
                    remove_on_success: bool = True, 
                    use_cleanup: bool = True,
                    tries: int = 3,
                    meta: dict = {}):
        
        if not os.path.isfile(local_path):
            raise FileNotFoundError(
                'Not found: {}'.format(local_path)
            )
        
        out = {
            'remote_path': remote_path,
            'file_type': file_type,
            'uploaded': False
        }

        out = out.update(meta)

        o = None
        for i in range(tries):
            o = self.driver.upload_object(
                local_path,
                self.container,
                remote_path
            )
            if o:
                if remove_on_success:
                    os.remove(local_path)

                out['uploaded'] = True

        if use_cleanup:
            self.temp.append(local_path)
        
        return out


    def upload_df(self, df: pd.DataFrame, obj_path: str, 
                    fmt: str = 'parquet', uid: str = None,
                    remove_on_success: bool = True,
                    use_cleanup: bool = True,
                    meta: dict = {}):

        if not uid:
            uid = uuid.uuid4().hex
        
        remote_base = obj_path.rstrip('/') + '/' + uid

        res = None
        
        w = {
            'parquet': (DataHandler.write_parquet, 'parquet'),
            'csv': (DataHandler.write_csv, 'csv'),
            'records': (DataHandler.write_records, 'records.json'),
            'json': (DataHandler.write_json, 'json')
        }.get(fmt)

        if w:
            wf, ext = w
            p = wf(df, self.cache_dir)

            res = self.upload_file(
                local_path=p,
                remote_path=remote_base + '.' + ext,
                file_type=ext,
                remove_on_success=remove_on_success,
                use_cleanup=use_cleanup,
                meta=meta
            )

        return res
    
    def upload_data(self, df: pd.DataFrame, obj_path: str, 
                    formats: str = 'parquet', uid: str = None,
                    remove_on_success: bool = True,
                    use_cleanup: bool = True,
                    meta: dict = {}):
        
        fmts = re.findall(r'[\w]+', formats)

        if not uid:
            uid = uuid.uuid4().hex
        results = []
        for fmt in fmts:
            results.append(
                self.upload_df(
                    df,
                    obj_path,
                    fmt,
                    uid,
                    remove_on_success,
                    use_cleanup,
                    meta
                )
            )
        return results


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
    
    def read_dataset(self, dataset_info: DatasetInfo, fmt: str = 'parquet',
                     read_fn=None, arguments: dict = {}):
        if not isinstance(dataset_info, DatasetInfo):
            raise TypeError('Must supply a DatasetInfo object!')

        return self.read_data(
            dataset_info.obj_path,
            fmt,
            read_fn,
            arguments
        ) 


class DataResource(object):

    __slots__ = ('handler', 'info')

    def __init__(self, storage_info: StorageInfo):
        self.info = storage_info

    def __enter__(self):

        self.handler = DataHandler(self.info)

        return self.handler

    def __exit__(self):

        self.handler.cleanup()


        

        

        
