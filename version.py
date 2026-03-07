import binascii

VERSION = "4.0.0"

def get_version_hash() -> str:
    crc = binascii.crc32(VERSION.encode('utf-8')) & 0xffffffff
    return hex(crc)[2:]
