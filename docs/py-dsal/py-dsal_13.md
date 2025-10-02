# python 数据结构与算法 14 队列的 python 实现

队列的 python 实现

在建立抽象数据类型之后，可以建立一个类来实现队列的。和以前一样，我们采用 python 内置的列表作为工具来建立队列类。

队列也是有序的，所以需要决定队列的哪一头作为队列的前端和尾端。在下面的实现代码中，我们约定列表的 0 位置是队列的尾部，这样的好处是，可以直接使用列表的 insert 方法在队尾加入数据，使用 pop 方法在队列的前端（这时是列表的最后一个数据）删除数据。从性能上分析，这意思着 endueue 是 O(n),而出队是 O(1)。

**Listing 1**

```py
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

```

以下是测试代码

q=Queue()

q.isEmpty()

q.enqueue('dog')

q.enqueue(4)

q=Queue()

q.isEmpty()

q.enqueue(4)

q.enqueue('dog')

q.enqueue(True)

运行代码之后，可以在控制台测试以下功能：

>>>q.size()

3

>>>q.isEmpty()

False

>>>q.enqueue(8.4)

>>>q.dequeue()

4

>>>q.dequeue()

'dog'

>>>q.size()

2