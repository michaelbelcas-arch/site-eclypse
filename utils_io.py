# utils_io.py
import os
import hashlib
from dataclasses import dataclass

@dataclass(frozen=True)
class FileSig:
    mtime_ns: int
    size: int
    head_hash: str  # hash des premiers KB pour éviter les faux-positifs

def file_signature(path: str, head_bytes: int = 4096) -> FileSig:
    st = os.stat(path)
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    size = st.st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(min(head_bytes, size)))
    return FileSig(mtime_ns=mtime_ns, size=size, head_hash=h.hexdigest())