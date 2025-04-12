import numpy as np
class Array:
    def __init__(self, dtype):
        self.array = np.array([], dtype=dtype)
        self.length = 0
        self.temp_array = np.empty(100, dtype=dtype)
    def append(self, el):
        if self.length == 100:
            self.array = np.append(self.array, self.temp_array)
            self.length = 0
        self.temp_array[self.length] = el
        self.length += 1

    def build(self):
        self.array = np.append(self.array, self.temp_array[:self.length])
        return self.array
