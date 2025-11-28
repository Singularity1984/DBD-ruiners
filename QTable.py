import asyncio
from config import *

class AsyncQTable:
    def __init__(self, max_size=MAX_Q_TABLE_SIZE):
        self._data = {}
        self.max_size = max_size
        self.access_order = []
        self._lock = asyncio.Lock()

    async def get(self, key):
        async with self._lock:
            if key in self._data:
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return self._data[key]
            else:
                if len(self._data) >= self.max_size:
                    lru_key = self.access_order.pop(0)
                    del self._data[lru_key]
                self._data[key] = [0.0] * 12
                self.access_order.append(key)
                return self._data[key]

    async def set(self, key, value):
        async with self._lock:
            if key not in self._data and len(self._data) >= self.max_size:
                lru_key = self.access_order.pop(0)
                del self._data[lru_key]
            self._data[key] = value
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

    async def update(self, other_dict):
        async with self._lock:
            for key, value in other_dict.items():
                await self.set(key, value)

    async def prune(self, min_visits=3):
        async with self._lock:
            if len(self._data) < self.max_size * 0.7:
                return 0
            keys_to_remove = [k for k in self.access_order if self.access_order.count(k) < min_visits]
            for key in keys_to_remove:
                if key in self._data:
                    del self._data[key]
                while key in self.access_order:
                    self.access_order.remove(key)
            return len(keys_to_remove)

    def __len__(self):
        return len(self._data)