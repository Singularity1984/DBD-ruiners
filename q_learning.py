import numpy as np
from collections import defaultdict
from config import MAX_Q_TABLE_SIZE


class SmartQTable:
    """Оптимизированная Q-table с LRU кэшированием"""

    def __init__(self, max_size=MAX_Q_TABLE_SIZE):
        self._data = {}
        self.max_size = max_size
        self.access_order = []

    def __getitem__(self, key):
        if key in self._data:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self._data[key]
        else:
            if len(self._data) >= self.max_size:
                lru_key = self.access_order.pop(0)
                del self._data[lru_key]
            self._data[key] = [0.0] * 5
            self.access_order.append(key)
            return self._data[key]

    def __setitem__(self, key, value):
        if key not in self._data and len(self._data) >= self.max_size:
            lru_key = self.access_order.pop(0)
            del self._data[lru_key]

        self._data[key] = value
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def __contains__(self, key):
        return key in self._data

    def clear(self):
        self._data.clear()
        self.access_order.clear()

    def update(self, other):
        for k, v in other.items():
            self[k] = v

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __len__(self):
        return len(self._data)

    def prune_infrequent(self, min_visits=3):
        if len(self._data) < self.max_size * 0.7:
            return 0

        keys_to_remove = [k for k in self.access_order
                          if self.access_order.count(k) < min_visits]

        for key in keys_to_remove:
            if key in self._data:
                del self._data[key]
            while key in self.access_order:
                self.access_order.remove(key)

        return len(keys_to_remove)