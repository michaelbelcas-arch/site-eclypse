# atomic_write.py
import os, tempfile, time

class FileLock:
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self.fd = None
    def acquire(self, timeout=10.0, poll=0.05):
        deadline = time.time() + timeout
        while True:
            try:
                # O_EXCL+O_CREAT => échoue si le lock existe
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"Lock timeout: {self.lock_path}")
                time.sleep(poll)
    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            try:
                os.unlink(self.lock_path)
            except FileNotFoundError:
                pass
            self.fd = None
    def __enter__(self):
        self.acquire()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.release()

def _fsync_dir(dirpath: str):
    try:
        dfd = os.open(dirpath, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        # Windows n'a pas d'O_DIRECTORY; on ignore
        pass

def atomic_write_bytes(path: str, data: bytes, mode: int = 0o644):
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    lock = FileLock(str(path) + ".lock")
    with lock:
        fd, tmp_path = tempfile.mkstemp(prefix=".__tmp__", dir=dirpath)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            # Pose les permissions souhaitées (sans affecter umask global)
            try:
                os.chmod(tmp_path, mode)
            except Exception:
                pass
            # Remplacement atomique
            os.replace(tmp_path, path)
            # Fsync du répertoire pour persister le rename
            _fsync_dir(dirpath)
        except Exception:
            # Nettoyage si échec
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise

def atomic_write_text(path: str, text: str, encoding="utf-8", newline="\n"):
    atomic_write_bytes(path, text.encode(encoding))
