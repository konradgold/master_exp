

class Registry:
    def __init__(self):
        self._store = {}

    def register(self):
        def decorator(cls):
            self._store[cls.__name__] = cls
            return cls
        return decorator

    def get(self, key):
        return self._store.get(key)

    def keys(self):
        return self._store.keys()

    