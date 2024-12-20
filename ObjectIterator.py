# -*- coding: utf-8 -*-

from enum import Enum

class IteratorType(Enum):
    TUPLE = 1
    VAL = 2
    DICT = 3


class ObjectIterator(object):
    def __init__(self, field_dict:dict, field_keys:list, index:int=0, iteratorType:IteratorType=IteratorType.TUPLE):
        self.index = index
        self.field_dict = field_dict
        self.field_keys = field_keys
        self.iteratorType = iteratorType

    def __iter__(self):
        return self

    def __next__(self):
        try:
            cur_key =  self.field_keys[self.index]
            cur_val = self.field_dict[cur_key]
            self.index += 1
            if IteratorType.VAL == self.iteratorType:
                return cur_val

            if IteratorType.TUPLE == self.iteratorType:
                return (cur_key, cur_val)

            if IteratorType.DICT == self.iteratorType:
                return {cur_key: cur_val}
        except IndexError:
            self.index = 0
            # Done iterating.
            raise StopIteration