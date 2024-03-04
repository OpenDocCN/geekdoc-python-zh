# Python 中的 Deque 模块介绍

> 原文：<https://www.pythonforbeginners.com/deque/introduction-to-deque-module-in-python>

双端队列是一种线性数据结构，我们可以在它的两端插入或删除元素，即它支持后进先出(LIFO)操作以及先进先出(FIFO)操作。dequee 模块是 python 中集合库的一部分，我们可以使用 dequee 模块中内置的方法来执行插入、删除或计算 dequee 中元素的数量。在本文中，我们将尝试通过用 python 实现程序来理解 deque 模块。

## 如何使用 Python 中的 deque 模块？

为了在 python 中使用 deque，我们将首先使用 import 语句导入模块，如下所示。

```py
from collections import deque
```

导入 deque 模块后，我们可以使用 deque()方法声明一个 deque 对象。dequee()方法将 list 或 tuple 之类的可迭代对象作为输入参数，创建一个 dequee 对象并返回对 dequee 对象的引用。这可以如下实现。

```py
from collections import deque
myDeque= deque([1,2,3,4,5])
print("The created deque is:")
print(myDeque)
```

输出:

```py
The created deque is:
deque([1, 2, 3, 4, 5])
```

## 如何在一个队列中插入元素？

我们可以使用 collections.deque 模块中定义的几种方法在 deque 中插入元素。

要在队列的开头插入一个元素，我们可以使用 appendleft()方法。appendleft()方法在 dequee 对象上调用时接受一个元素作为输入，并将该元素添加到 dequee 的开头。这可以从下面的例子中看出。

```py
print("Deque before insertion:")
print(myDeque)
myDeque.appendleft(0)
print("Deque after insertion of element 0 at start:")
print(myDeque)
```

输出:

```py
Deque before insertion:
deque([1, 2, 3, 4, 5])
Deque after insertion of element 0 at start:
deque([0, 1, 2, 3, 4, 5])
```

要在队列末尾插入一个元素，我们可以使用 append()方法。append()方法在 dequee 对象上调用时接受一个元素作为输入，并将该元素添加到 dequee 的末尾。这可以从下面的例子中看出。

```py
print("Deque before insertion:")
print(myDeque)
myDeque.append(6)
print("Deque after insertion of element 6 at end:")
print(myDeque)
```

输出:

```py
Deque before insertion:
deque([0, 1, 2, 3, 4, 5])
Deque after insertion of element 6 at end:
deque([0, 1, 2, 3, 4, 5, 6])
```

我们也可以在一个队列中同时插入多个条目。要在队列的开始插入多个元素，我们可以使用 extendleft()方法。在 dequee 上调用 extendleft()方法时，该方法将 list 或 tuple 等 iterable 作为输入，并按照与输入中传递的顺序相反的顺序将元素添加到 dequee 的开头，如下所示。

```py
print("Deque before insertion:")
print(myDeque)
myDeque.extendleft([-2,-3,-4])
print("Deque after insertion of multiple elements at start:")
print(myDeque)
```

输出:

```py
Deque before insertion:
deque([0, 1, -1, 2, 3, 4, 5, 6])
Deque after insertion of multiple elements at start:
deque([-4, -3, -2, 0, 1, -1, 2, 3, 4, 5, 6])
```

要在队列末尾插入多个元素，我们可以使用 extend()方法。extend()方法在 dequee 上调用时将 list 或 tuple 等可迭代对象作为输入，并将元素添加到 dequee 的末尾，如下所示。

```py
print("Deque before insertion:")
print(myDeque)
myDeque.extend([-5,-6,-7])
print("Deque after insertion of multiple elements at end:")
print(myDeque)
```

输出:

```py
Deque before insertion:
deque([-4, -3, -2, 0, 1, -1, 2, 3, 4, 5, 6])
Deque after insertion of multiple elements at end:
deque([-4, -3, -2, 0, 1, -1, 2, 3, 4, 5, 6, -5, -6, -7])
```

我们还可以使用 insert()方法在队列中的特定位置插入一个元素。在 deque 上调用 insert()方法时，该方法将必须插入元素的索引作为第一个参数，将元素本身作为第二个参数，并在指定的索引处插入元素，如下所示。

```py
print("Deque before insertion:")
print(myDeque)
myDeque.insert(2,-1)
print("Deque after insertion of element -1 at index 2:")
print(myDeque)
```

输出:

```py
Deque before insertion:
deque([0, 1, 2, 3, 4, 5, 6])
Deque after insertion of element -1 at index 2:
deque([0, 1, -1, 2, 3, 4, 5, 6])
```

## 如何在 Python 中删除 deque 中的元素？

我们可以从队列的开始和结尾删除一个元素。为了从队列的开始处删除一个元素，我们使用 popleft()方法。对 dequee 调用 popleft()方法时，会删除 dequee 中的第一个元素，并返回该元素的值。如果队列为空，popleft()方法将引发一个名为`IndexError`的异常，并显示一条消息“从空队列中弹出”。为了处理这个异常，我们可以使用除了之外的 [python try 来使用异常处理。这可以如下实现。](https://www.pythonforbeginners.com/error-handling/python-try-and-except)

```py
print("Deque before deletion of leftmost element:")
print(myDeque)
try:
    myDeque.popleft()
except IndexError:
    print("Deque is empty")
print("Deque after deletion of leftmost element:")
print(myDeque)
```

输出:

```py
Deque before deletion of leftmost element:
deque([-3, -2, 0, 1, -1, 2, 3, 4, 5, 6, -5, -6, -7])
Deque after deletion of leftmost element:
deque([-2, 0, 1, -1, 2, 3, 4, 5, 6, -5, -6, -7])
```

要从队列末尾删除一个元素，我们可以使用 pop()方法。在 dequee 上调用 pop()方法时，会删除 dequee 中的最后一个元素，并返回该元素的值。如果队列为空，pop()方法将引发一个名为`IndexError`的异常，并显示一条消息“从空队列中弹出”。为了处理这个异常，我们可以使用 python try except 进行异常处理。这可以如下实现。

```py
print("Deque before deletion of rightmost element:")
print(myDeque)
try:
    myDeque.pop()
except IndexError:
    print("Deque is empty")
print("Deque after deletion of rightmost element:")
print(myDeque)
```

输出:

```py
Deque before deletion of rightmost element:
deque([-2, 0, 1, -1, 2, 3, 4, 5, 6, -5, -6, -7])
Deque after deletion of rightmost element:
deque([-2, 0, 1, -1, 2, 3, 4, 5, 6, -5, -6])
```

我们还可以使用 remove()方法从队列中删除任何特定的元素。在 deque 上调用 remove 方法时，该方法将元素作为输入，并删除该元素的第一个匹配项。当要删除的元素不在 dequee 中时，remove 方法会引发 ValueError 异常，并显示消息“dequee . remove(x):x 不在 dequee 中”。为了处理这个异常，我们可以使用 python 中的 try except 块来进行异常处理。这可以如下实现。

```py
print("Deque before deletion of element -1:")
print(myDeque)
try:
    myDeque.remove(-1)
except ValueError:
    print("Value is not present in deque")
print("Deque after deletion of element -1:")
print(myDeque)
```

输出:

```py
Deque before deletion of element -1:
deque([-2, 0, 1, -1, 2, 3, 4, 5, 6, -5, -6])
Deque after deletion of element -1:
deque([-2, 0, 1, 2, 3, 4, 5, 6, -5, -6])
```

## 统计特定元素在队列中的出现次数

为了计算一个元素在队列中出现的次数，我们可以使用 count()方法。在 dequee 上调用 count()方法时，该方法将元素作为输入，返回元素在 dequee 中出现的次数。这可以从下面的例子中看出。

```py
print("Deque is:")
print(myDeque)
n=myDeque.count(2)
print("Number of occurrences element 2 in deque is:")
print(n)
```

输出:

```py
Deque is:
deque([-2, 0, 1, 2, 3, 4, 5, 6, -5, -6])
Number of occurrences element 2 in deque is:
1
```

## 反转队列中的元素

我们可以使用 reverse()方法反转 deque 中的元素。在 deque 上调用 reverse()方法时，会按如下方式反转元素的顺序。

```py
print("Deque is:")
print(myDeque)
myDeque.reverse()
print("Reversed deque is:")
print(myDeque)
```

输出:

```py
Deque is:
deque([-2, 0, 1, 2, 3, 4, 5, 6, -5, -6])
Reversed deque is:
deque([-6, -5, 6, 5, 4, 3, 2, 1, 0, -2])
```

## 搜索队列中的元素

我们可以使用 index()方法搜索队列中的元素。在 deque 上调用 index()方法时，该方法将元素、开始搜索的 start index 和停止搜索的 end index 分别作为第一、第二和第三个参数，并返回元素所在的第一个索引。当元素不存在时。index()方法引发一个`ValueError`，并显示一条消息，指出元素不在队列中。我们可以使用 try except 块来处理异常，并给出如下正确的输出。

```py
print("Deque is:")
print(myDeque)
try:
    n=myDeque.index(4,0,9)
    print("Element 4 is at index")
    print(n)
except ValueError:
    print("Value is not present in deque")
```

输出:

```py
Deque is:
deque([-6, -5, 6, 5, 4, 3, 2, 1, 0, -2])
Element 4 is at index
4
```

## 如何在 Python 中旋转一个 deque？

我们可以使用 rotate()方法将队列向右旋转指定的步数。在 dequee 上调用 rotate()方法时，将 dequee 必须旋转的步数作为输入，并按如下方式旋转数组。

```py
print("Deque is:")
print(myDeque)
myDeque.rotate(3)
print("deque after rotationg three steps rightwards is:")
print(myDeque)
```

输出:

```py
Deque is:
deque([-6, -5, 6, 5, 4, 3, 2, 1, 0, -2])
deque after rotationg three steps rightwards is:
deque([1, 0, -2, -6, -5, 6, 5, 4, 3, 2])
```

## 结论

在本文中，我们介绍了 python 中的 deque 模块及其方法。为了更深入地了解它，并理解 dequee 与其他数据结构(如 [python dictionary](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/) 、list 和 set)的不同之处，请将代码复制到您的 IDE 中，并尝试 dequee 操作。请继续关注更多内容丰富的文章。