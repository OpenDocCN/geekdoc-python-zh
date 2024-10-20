# Python 队列模块

> 原文：<https://www.askpython.com/python-modules/python-queue>

在本文中，我们将研究 Python 队列模块，它是队列数据结构的接口。

* * *

## Python 队列

队列是一种数据结构，其中要插入的第一个元素也是弹出的第一个元素。这就像现实生活中的排队一样，第一个排队的也是第一个出来的。

在 Python 中，我们可以使用`queue`模块来创建对象队列。

这是标准 Python 库的一部分，所以不需要使用`pip`。

使用以下方式导入模块:

```py
import queue

```

要创建一个队列对象，我们可以使用以下方法实例化它:

```py
q = queue.Queue()

```

默认情况下，它的容量为 0，但是如果您想要显式地提及它，您可以使用:

```py
q = queue.Queue(max_capacity)

```

## Queue.get()和 Queue.put()方法

我们可以使用`queue.get()`和`queue.put()`方法在队列中插入和检索值。

让我们创建一个队列，插入从 1 到 5 的数字。

```py
import queue

# Instantiate the Queue object
q = queue.Queue()

# Insert elements 1 to 5 in the queue
for i in range(1, 6):
    q.put(i)

print('Now, q.qsize() =', q.qsize())

# Now, the queue looks like this:
# (First) 1 <- 2 <- 3 <- 4 <- 5
for i in range(q.qsize()):
    print(q.get())

```

**输出**

```py
Now, q.qsize() = 5
1
2
3
4
5

```

如您所见，输出显示第一个索引确实是 1，所以这是队列的顶部。其余的元素以类似的方式跟随它。

## 清空 Python 队列

我们可以使用`q.empty()`清空一个队列对象。这会将大小设置为 0，并清空队列。

```py
import queue

# Instantiate the Queue object
q = queue.Queue()

# Insert elements 1 to 5 in the queue
for i in range(1, 6):
    q.put(i)

print('Now, q.qsize() =', q.qsize())

# Empty queue
q.empty()

print('After emptying, size =', q.qsize())

for i in range(q.qsize()):
    print(q.get())

```

**输出**

```py
Now, q.qsize() = 5
After emptying, size = 0

```

虽然大多数典型的队列实现都有一个`pop`(或`dequeue`)操作，但是`queue`模块没有这个方法。

因此，如果您想从队列中弹出元素，您必须自己使用不同的队列类。一个简单的解决方案是使用 Python 的 list。

我们将使用`list.append(value)`向队列中插入元素，因为插入发生在最后，并且使用`list.pop(0)`移除元素，因为第一个元素被移除。

```py
class MyQueue():
    # Using Python Lists as a Queue
    def __init__(self):
        self.queue = []

    def enqueue(self, value):
        # Inserting to the end of the queue
        self.queue.append(value)

    def dequeue(self):
         # Remove the furthest element from the top,
         # since the Queue is a FIFO structure
         return self.queue.pop(0)

my_q = MyQueue()

my_q.enqueue(2)
my_q.enqueue(5)
my_q.enqueue(7)

for i in my_q.queue:
    print(i)

print('Popped,', my_q.dequeue())

for i in my_q.queue:
    print(i)

```

**输出**

```py
2
5
7
Popped, 2
5
7

```

我们已经用一个`dequeue`操作编写了自己的队列类！现在，我们将向您展示如何使用其他模块来使用其他类型的队列。

* * *

## Python 中的优先级队列

优先级队列是一种基于项目的**优先级**添加到队列中的队列，优先级通常是一个整数值。

优先级较低的项目具有较高的优先级，位于队列的最前面，而其他项目位于后面。

`queue`模块也支持优先级队列结构，所以让我们看看如何使用它。

```py
import queue

priority_q = queue.PriorityQueue()

priority_q.put((1, 'Hello'))
priority_q.put((3, 'AskPython'))
priority_q.put((2, 'from'))

for i in range(priority_q.qsize()):
    print(priority_q.get())

```

输出

```py
(1, 'Hello')
(2, 'from')
(3, 'AskPython')

```

正如您所看到的，元素是根据它们的优先级插入的。

* * *

## Python 堆队列

我们还可以使用`heapq`模块来实现我们的优先级队列。

```py
>>> import heapq
>>> q = []
>>> heapq.heappush(q, (1, 'hi'))
>>> q
[(1, 'hi')]
>>> heapq.heappush(q, (3, 'AskPython'))
>>> q
[(1, 'hi'), (3, 'AskPython')]
>>> heapq.heappush(q, (2, 'from'))
>>> q
[(1, 'hi'), (3, 'AskPython'), (2, 'from')]
>>> heapq.heappop(q)
(1, 'hi')
>>> heapq.heappop(q)
(2, 'from')
>>> heapq.heappop(q)
(3, 'AskPython')
>>> heapq.heappop(q)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index out of range

```

因此，我们创建了一个优先级队列，并从中弹出，直到它为空。使用下面的程序也可以达到同样的效果

```py
import heapq

q = []

heapq.heappush(q, (2, 'from'))
heapq.heappush(q, (1, 'Hello'))
heapq.heappush(q, (3, 'AskPython'))

while q:
    # Keep popping until the queue is empty
    item = heapq.heappop(q)
    print(item)

```

**输出**

```py
(1, 'Hello')
(2, 'from')
(3, 'AskPython')

```

* * *

## 结论

在本文中，我们学习了如何在 Python 中实现和使用不同的队列。

## 参考

*   [Python 队列文档](https://docs.python.org/3/library/queue.html)
*   [关于 Python 队列的 JournalDev 文章](https://docs.python.org/3/library/queue.html)

* * *