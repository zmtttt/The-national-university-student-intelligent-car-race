import collections


class FixedQueue:
    def __init__(self, size=3, class_num=10):
        self.size = size
        self.class_num = class_num
        self.deque = [collections.deque(size * [0], self.size) for i in range(self.class_num)]
        self.positive_limit = 2

    def append(self, item, index=0):
        for deque in self.deque:
            deque.append(0)
        self.deque[int(index)][-1] = item

    def clear(self):
        self.deque = [collections.deque(self.size * [0], self.size) for i in range(self.class_num)]


    def sum(self):
        return sum(self.deque)

    def roadsign_valid(self):
        index = 0
        for q in self.deque:
            s = sum(q)
            if s > self.positive_limit:
                return True, index
            index += 1
        return False, -1