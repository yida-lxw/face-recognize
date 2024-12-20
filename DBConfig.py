# -*- coding: utf-8 -*-
from ObjectIterator import ObjectIterator, IteratorType


class DBConfig:
    def __init__(self, host:str, port:int, username:str, password:str, db:str,
                 maxConnectionSize:int, initConnectionSize:int, maxIdleSize:int, charset:str="utf8",
                 blockingIfNoConnection:bool=True, iteratorType:IteratorType=IteratorType.DICT):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.db = db
        self.charset = charset
        self.maxConnectionSize = maxConnectionSize
        self.initConnectionSize = initConnectionSize
        self.maxIdleSize = maxIdleSize
        self.blockingIfNoConnection = blockingIfNoConnection
        self.iteratorType = iteratorType

    def __iter__(self):
        field_keys = list(self.__dict__.keys())
        field_dict = self.__dict__
        final_field_dict = {}
        final_field_keys = []
        for key, value in field_dict.items():
            if type(value) == IteratorType:
                continue
            final_field_keys.append(key)
            final_field_dict.update({key: value})
        return ObjectIterator(field_dict=final_field_dict, field_keys=final_field_keys, iteratorType=self.iteratorType)


    # getter
    def getHost(self):
        return self.host

    def getPort(self):
        return self.port

    def getUsername(self):
        return self.username

    def getPassword(self):
        return self.password

    def getDb(self):
        return self.db

    def getCharset(self):
        return self.charset

    def getBlockingIfNoConnection(self):
        return self.blockingIfNoConnection

    def getMaxConnectionSize(self):
        return self.maxConnectionSize

    def getInitConnectionSize(self):
        return self.initConnectionSize

    def getMaxIdleSize(self):
        return self.maxIdleSize

    # setter
    def setHost(self, host:str):
        self.host = host
        return self

    def setPort(self, port:int):
        self.port = port
        return self

    def setUsername(self, username:str):
        self.username = username
        return self

    def setPassword(self, password:str):
        self.password = password
        return self

    def setDb(self, db:str):
        self.db = db
        return self

    def setCharset(self, charset:str):
        self.charset = charset
        return self

    def setBlockingIfNoConnection(self, blockingIfNoConnection:bool):
        self.blockingIfNoConnection = blockingIfNoConnection
        return self

    def setMaxConnectionSize(self, maxConnectionSize:int):
        self.maxConnectionSize = maxConnectionSize
        return self

    def setInitConnectionSize(self, initConnectionSize:int):
        self.initConnectionSize = initConnectionSize
        return self

    def setMaxIdleSize(self, maxIdleSize:int):
        self.maxIdleSize = maxIdleSize
        return self

    def __eq__(self, other):
        if isinstance(other, DBConfig):
            return (self.host == other.host and self.port == other.port and self.db == other.db and
            self.username == other.username and self.password == other.password and self.charset == other.charset and
            self.blockingIfNoConnection == other.blockingIfNoConnection and self.maxConnectionSize == other.maxConnectionSize and
            self.initConnectionSize == other.initConnectionSize and self.maxIdleSize == other.maxIdleSize)
        return False

    def __hash__(self):
        return hash(self)
