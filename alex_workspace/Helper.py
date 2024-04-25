class Counter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1
        return self.count
    def reset(self):
        self.count =0
# some list assistance

from collections import OrderedDict

class ListHelper:
    @staticmethod
    def unique_list(lst):
        # check if list
        if not isinstance(lst, list):
            return lst
        result = []
        for elem in lst:
            # check if empty
            if isinstance(elem, int):
                continue
            if len(elem) == 0:
                continue
            if elem not in result:
                result.append(elem)
        return result
    @staticmethod
    def remvoe_one_size(lst):
        result = []
        for elem in lst:
            if len(elem) == 1:
                continue
            result.append(elem)
        return result