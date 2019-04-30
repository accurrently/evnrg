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

    def upload_data(self, df: pd.DataFrame, obj_path: str, 
                    formats: str = 'parquet',
                    keep_temp: bool = False, enable_cleanup: bool = True):

        uid = uuid.uuid4().hex

        fmts = re.findall(r'[\w]+', formats)

        local_path = os.path.join(self.cache_dir, uid)
        remote_base = obj_path.rstrip('/') + '/'

        results = []
        written = []

        if 'parquet' in fmts:
            local_parq = local_path + '.parquet'
            remote_parq = remote_base + 'parquet/' + uid + '.parquet'
            df.to_parquet(
                local_parq,
                engine='fastparquet',
                compression='gzip'
            )
            written.append(local_parq)
            o = self.driver.upload_object(
                        local_parq,
                        self.container,
                        remote_parq
                    )
            if o:
                results.append(
                    UploadResult(
                        uid=uid,
                        filetype='parquet',
                        cache_path=local_parq,
                        remote_path=remote_parq,
                        obj=o
                    )
                )
        if 'json' in fmts:
            local_json = local_path + '.json'
            remote_json = remote_base + 'json/' + uid + '.json'
            df.to_json(
                local_json,
                orient='split',
                date_format='iso',
                date_unit='s'
            )
            written.append(local_json)
            o = self.driver.upload_object(
                        local_json,
                        self.container,
                        remote_json
                    )
            if o:
                results.append(
                    UploadResult(
                        uid=uid,
                        filetype='json',
                        cache_path=local_json,
                        remote_path=remote_json,
                        obj=o
                    )
                )
        if 'records' in fmts:
            local_records = local_path + '.records.json'
            remote_records = remote_base + 'records/' + uid + '.records.json'
            df.to_json(
                local_records,
                orient='records',
                date_format='iso',
                date_unit='s'
            )
            written.append(local_records)
            o = self.driver.upload_object(
                        local_records,
                        self.container,
                        remote_records
                    )
            if o:
                results.append(
                    UploadResult(
                        uid=uid,
                        filetype='records',
                        cache_path=local_records,
                        remote_path=remote_records,
                        obj=o
                    )
                )
        if 'csv' in fmts:
            local_csv = local_path + '.csv'
            remote_csv = remote_base + 'csv/' + uid + '.csv'
            df.to_csv(
                local_csv,
                date_format='%Y-%m-%dT%H:%M:%S'
            )
            written.append(local_csv)
            o = self.driver.upload_object(
                        local_csv,
                        self.container,
                        remote_csv
                    )
            if o:
                results.append(
                    UploadResult(
                        uid=uid,
                        filetype='csv',
                        cache_path=local_csv,
                        remote_path=remote_csv,
                        obj=o
                    )
                )

        # Delete temporary files
        if not keep_temp:
            for to_remove in written:
                if os.path.isfile(to_remove):
                    os.remove(to_remove)

        # Add to file list for cleanup
        if enable_cleanup:
            for to_remove in written:
                if os.path.isfile(to_remove):
                    self.temp.append(to_remove)

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

        fname = obj_name.split('/')[-1]

        tmp_name = os.path.join(TMP_DIR, fname)

        if self.driver.download_object(o, tmp_name, True, True):

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


        

        

        
