import uuid, time
from contextlib import contextmanager

def redis_lock(client, name, expire=10, wait_interval=0.1):
    token = uuid.uuid4().hex
    lock_key = f"lock:{name}"
    @contextmanager
    def _lock():
        while not client.set(lock_key, token, nx=True, ex=expire):
            time.sleep(wait_interval)
        try:
            yield
        finally:
            if client.get(lock_key) == token.encode():
                client.delete(lock_key)
    return _lock()
