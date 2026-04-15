import re

def get_support_bucket(fname):
    try:
        return tuple(map(float, re.findall(r"support-((?:0|1)\.\d+)-((?:0|1)\.\d+)-", fname)[0]))
    except:
        print(fname)
        raise ValueError