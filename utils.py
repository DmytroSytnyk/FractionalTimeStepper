from typing import Dict, Any
import hashlib
import json
# import sys
import os
def dict2json (dictionary: Dict[str, Any]):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return encoded, dhash.hexdigest()
def save_dict2json (fileprefix: str,dictionary: Dict[str, Any]):
    enc_dict,hash = dict2json(dictionary)
    #print(fileprefix)
    #import pdb; pdb.set_trace()
    with open(fileprefix + hash + '.json','w') as ofile:
        ofile.write(dictionary)
    return hash

