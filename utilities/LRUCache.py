class DoubleLinkedNode:
    def __init__(self, key, value, pre=None, next=None):
        self.key = key
        self.value = value
        self.pre = pre
        self.next = next

    def __repr__(self):
        return f"DoubleLinkedNode key={self.key} value={self.value}"


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.head = DoubleLinkedNode("head", "head")
        self.tail = DoubleLinkedNode("tail", "tail")
        self.head.next = self.tail
        self.tail.pre = self.head
        self.cache = {}

    def get(self,key):
        if key not in self.cache:
            return None
        else:
            node = self.cache[key]
            self.__moveToHead(node)
            return node.value

    def put(self, key, value):
        if key in self.cache:
            node: DoubleLinkedNode = self.cache[key]
            node.value = value
            self.__moveToHead(node)
        else:
            node = DoubleLinkedNode(key,value)
            self.cache[key] = node
            self.__appendToHead(node)
            self.size += 1
            if self.size > self.capacity:
                deletedNode = self.__deleteTail()
                self.cache.pop(deletedNode.key)
                self.size -= 1

    def __deleteTail(self):
        return self.__deleteNode(self.tail.pre)

    def __moveToHead(self, node):
        self.__deleteNode(node)
        self.__appendToHead(node)

    def __deleteNode(self, node: DoubleLinkedNode) -> DoubleLinkedNode:
        node.next.pre = node.pre
        node.pre.next = node.next
        return node

    def __appendToHead(self, node: DoubleLinkedNode):
        node.next = self.head.next
        node.pre = self.head
        self.head.next.pre = node
        self.head.next = node

    def __len__(self):
        return len(self.cache)

    def __bool__(self):
        return bool(self.cache)
