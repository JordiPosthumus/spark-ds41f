"""CPU-only loader and accepted-token state walker; NumPy, no torch dependency."""
import hashlib,json,io
from pathlib import Path
import numpy as np

NO_HIT=255
BOUNDARY_IDS=frozenset((1,128822,128825))

def load_tables(path):
    path=Path(path)
    if path.is_file():path=path.parent
    manifest=json.loads((path/'metadata.json').read_text())
    raw=(path/'tables.npz').read_bytes()
    if manifest.get('format_version')!=1:raise ValueError('suppression format mismatch')
    if hashlib.sha256(raw).hexdigest()!=manifest['archive_sha256']:raise ValueError('suppression tables hash mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as data:
        if set(data.files)!=set(manifest['tables']):raise ValueError('suppression table names mismatch')
        tables={k:data[k] for k in data.files}
    for k,a in tables.items():
        spec=manifest['tables'][k]
        if list(a.shape)!=spec['shape'] or str(a.dtype)!=spec['dtype'] or a.nbytes!=spec['bytes']:raise ValueError('suppression table shape/dtype mismatch')
        a.flags.writeable=False
    if tables['lex_table'].shape!=(31,13) or tables['token_prefix_id'].shape!=(129280,):raise ValueError('unexpected compiled vocabulary or lexer')
    return tables

def initial():return (0,0)

def unpack(value):return int(value)&31,(int(value)>>5)&255,(int(value)>>13)&255

def feed_token(tables,lex,utf,token):
    if not 0<=lex<31 or not 0<=utf<tables['bridge_table'].shape[0]:raise ValueError('invalid suppression state')
    if not 0<=token<len(tables['token_prefix_id']):raise ValueError('invalid suppression token')
    words=0;hit=NO_HIT
    value=int(tables['bridge_table'][utf,tables['token_prefix_id'][token]])
    utf=value&511
    for j in range((value>>9)&3):
        lex,n,h=unpack(tables['lex_table'][lex,(value>>(11+4*j))&15])
        if hit==NO_HIT and h!=NO_HIT:hit=words+h
        words+=n
    body=int(tables['token_body_id'][token])
    lex,n,h=unpack(tables['body_table'][lex,body])
    if hit==NO_HIT and h!=NO_HIT:hit=words+h
    words+=n
    if tables['body_nonempty'][body]:utf=int(tables['body_tail'][body])
    return lex,utf,words,hit

def advance(tables,state,token_ids):
    lex,utf=state
    for token in token_ids:lex,utf,_,_=feed_token(tables,lex,utf,int(token))
    return lex,utf


from functools import lru_cache
import os

@lru_cache(maxsize=1)
def get_tables():
    path = os.environ.get("SPARK_DS41F_PHRASE_TABLES")
    if not path:
        raise ValueError("Missing SPARK_DS41F_PHRASE_TABLES")
    return load_tables(path)
