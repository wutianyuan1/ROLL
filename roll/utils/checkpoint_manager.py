import contextlib
import hashlib
import os
import shutil
import tempfile
import traceback
from typing import Dict, Optional

from filelock import FileLock
from huggingface_hub import snapshot_download

from roll.utils.logging import get_logger
from roll.utils.upload_utils import FileSystemUploader


logger = get_logger()


@contextlib.contextmanager
def file_lock_context(lock_path: str):
    temp_lock_path = os.path.join(tempfile.gettempdir(), f"{hashlib.md5(lock_path.encode()).hexdigest()}.lock")
    with FileLock(temp_lock_path):
        yield


def download_model(model_name_or_path: str, local_dir: Optional[str] = None):
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    # use_model_scope = os.getenv("USE_MODELSCOPE", "0") == "1"
    # with file_lock_context(model_name_or_path):
    #     if use_model_scope:
    #         from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

    #         return ms_snapshot_download(model_name_or_path, local_dir=local_dir)
    #     return snapshot_download(model_name_or_path, local_dir=local_dir)
    
    # Use local cache path to avoid downloading
    with file_lock_context(model_name_or_path):
        assert model_name_or_path in [f'Qwen/Qwen2.5-{n}B' for n in [7, 14, 32, 72]]
        return f'/tmp/models/{model_name_or_path}'


class CheckpointManager:
    """
    ray.Actor创建到每个node上，负责将本地output_dir的文件上传到远程存储(oss/hdfs)
    """

    def __init__(self, checkpoint_config=None):
        self.checkpoint_config: Dict = checkpoint_config
        self.uploader = None
        logger.info(f"{checkpoint_config}")
        output_dir = self.checkpoint_config.get("output_dir")
        self.uploader = FileSystemUploader(output_dir=output_dir)

    def upload(self, ckpt_id, local_state_path, keep_local_file=False):
        try:
            if not self.uploader:
                logger.warning(f"uploader is None, skip upload...")
                return

            self.uploader.upload(ckpt_id=ckpt_id, local_state_path=local_state_path)
            if not keep_local_file:
                if os.path.isdir(local_state_path):
                    shutil.rmtree(local_state_path, ignore_errors=True)
                else:
                    os.remove(local_state_path)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"upload failed, {e}")
