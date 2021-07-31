from collections import defaultdict

import ecdsa as ecdsa
from ecdsa import NIST384p, SigningKey
# from ellipticcurve.ecdsa import Ecdsa
# from ellipticcurve.privateKey import PrivateKey
import ecdsa
import json
from Crypto import Random
def SVkey():
    sk = ecdsa.SigningKey.generate(curve=NIST384p)
    sk_string=sk.to_string()
    vk = sk.get_verifying_key()
    vk_string=vk.to_string()
    return sk_string,vk_string
def makejson():
    port=50051
    keylist=defaultdict(list)
    for i in range(4):
        t=SVkey()
        keylist[port].append(t[0].hex())
        keylist[port].append(t[1].hex())
        port+=2
    json_str = json.dumps(keylist, indent=4)
    with open('svkey.json', 'w') as json_file:
        json_file.write(json_str)
    return