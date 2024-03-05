# Python 的 deque:实现高效的队列和堆栈

> [https://realython . com/python-deque/](https://realpython.com/python-deque/)

如果你经常用 Python 处理列表，那么你可能知道当你需要在列表的左端**弹出**和**追加**项时，它们的执行速度不够快。Python 的 [`collections`](https://docs.python.org/3/library/collections.html#module-collections) 模块提供了一个名为 [`deque`](https://docs.python.org/3/library/collections.html#collections.deque) 的类，该类是专门设计来提供快速且节省内存的方法来从底层数据结构的两端追加和弹出项目。

Python 的`deque`是一个低级且高度优化的[双端队列](https://realpython.com/queue-in-python/#deque-double-ended-queue)，对于实现优雅、高效且 python 化的[队列和堆栈](https://realpython.com/queue-in-python/)非常有用，它们是计算中最常见的列表式数据类型。

在本教程中，您将学习:

*   如何在你的代码中创建和使用 Python 的 **`deque`**
*   如何高效地从一个`deque`的两端**追加**和**弹出**项
*   如何利用`deque`搭建高效的**队列**和**栈**
*   当值得用 **`deque`** 代替 **`list`** 时

为了更好地理解这些主题，您应该了解使用 Python [列表](https://realpython.com/python-lists-tuples/)的基础知识。对[队列](https://realpython.com/python-data-structures/#queues-fifos)和[栈](https://realpython.com/how-to-implement-python-stack/)有一个大致的了解也是有益的。

最后，您将编写几个示例，带您了解一些常见的`deque`用例，它是 Python 最强大的数据类型之一。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## Python 的`deque` 入门

在 Python 列表的右端追加和弹出项目通常是高效的操作。如果你用[大 O 符号](https://en.wikipedia.org/wiki/Big_O_notation)表示[时间复杂度](https://wiki.python.org/moin/TimeComplexity)，那么你可以说它们是 *O* (1)。但是，当 Python 需要重新分配内存来增加底层列表以接受新项目时，这些操作会更慢，并且会变成 *O* ( *n* )。

此外，在 Python 列表的左端追加和弹出项目是效率低下的操作，速度为 *O* ( *n* )。

因为 Python 列表为两种操作都提供了 [`.append()`](https://realpython.com/python-append/) 和`.pop()`，所以它们可以用作[栈](https://en.wikipedia.org/wiki/Stack_(abstract_data_type))和[队列](https://en.wikipedia.org/wiki/Queue_(abstract_data_type))。但是，您之前看到的性能问题会显著影响应用程序的整体性能。

Python 的 [`deque`](https://docs.python.org/3/library/collections.html?highlight=collections#collections.deque) 是在 [Python 2.4](https://docs.python.org/3/whatsnew/2.4.html#new-improved-and-deprecated-modules) 中第一个添加到 [`collections`](https://realpython.com/python-collections-module/) 模块的数据类型。这种数据类型是专门为克服 Python 列表中的`.append()`和`.pop()`的效率问题而设计的。

Deques 是类似序列的数据类型，被设计为对**堆栈**和**队列**的概括。它们支持对数据结构两端的高效内存和快速追加和弹出操作。

**注:** `deque`读作“甲板”这个名字代表[**d**double-**e**nd**que**UE](https://en.wikipedia.org/wiki/Double-ended_queue)。

在一个`deque`对象两端的追加和弹出操作是稳定和同样有效的，因为 deques 是作为一个[双向链表](https://realpython.com/linked-lists-python/#how-to-use-doubly-linked-lists)被[实现的](https://github.com/python/cpython/blob/23acadcc1c75eb74b2459304af70d97a35001b34/Modules/_collectionsmodule.c#L34)。此外，deques 上的 append 和 pop 操作也是[线程安全](https://en.wikipedia.org/wiki/Thread_safety)和内存高效的。这些特性使得 deques 对于在 Python 中创建自定义堆栈和队列特别有用。

如果您需要保留最后看到的项目的列表，Deques 也是一种方法，因为您可以限制 deques 的最大长度。如果您这样做，那么一旦 deque 已满，当您在另一端追加新项目时，它会自动丢弃一端的项目。

下面总结一下`deque`的主要特点:

*   存储任何[数据类型](https://realpython.com/python-data-types/)的项目
*   是一种可变的数据类型
*   通过`in`操作员支持[会员操作](https://realpython.com/python-boolean/#the-in-operator)
*   支持[分度](https://realpython.com/python-lists-tuples/#list-elements-can-be-accessed-by-index)，如`a_deque[i]`所示
*   不支持切片，就像在`a_deque[0:2]`中
*   支持操作序列和可迭代的内置函数，如 [`len()`](https://realpython.com/len-python-function/) 、 [`sorted()`](https://realpython.com/python-sort/) 、 [`reversed()`](https://realpython.com/python-reverse-list/) 等
*   不支持[就地](https://en.wikipedia.org/wiki/In-place_algorithm)排序
*   支持正向和反向迭代
*   支持用 [`pickle`](https://realpython.com/python-pickle-module/) 酸洗
*   确保两端的快速、内存高效和线程安全的弹出和追加操作

创建`deque`实例是一个简单的过程。你只需要从`collections`中导入`deque`，并使用可选的`iterable`作为参数调用它:

>>>

```py
>>> from collections import deque

>>> # Create an empty deque
>>> deque()
deque([])

>>> # Use different iterables to create deques
>>> deque((1, 2, 3, 4))
deque([1, 2, 3, 4])

>>> deque([1, 2, 3, 4])
deque([1, 2, 3, 4])

>>> deque(range(1, 5))
deque([1, 2, 3, 4])

>>> deque("abcd")
deque(['a', 'b', 'c', 'd'])

>>> numbers = {"one": 1, "two": 2, "three": 3, "four": 4}
>>> deque(numbers.keys())
deque(['one', 'two', 'three', 'four'])

>>> deque(numbers.values())
deque([1, 2, 3, 4])

>>> deque(numbers.items())
deque([('one', 1), ('two', 2), ('three', 3), ('four', 4)])
```

如果您实例化`deque`而没有提供一个`iterable`作为参数，那么您会得到一个空的 deque。如果您提供并输入`iterable`，那么`deque`会用其中的数据初始化新实例。使用 [`deque.append()`](https://docs.python.org/3/library/collections.html#collections.deque.append) 从左到右进行初始化。

`deque`初始化器接受以下两个可选参数:

1.  **`iterable`** 持有提供初始化数据的 iterable。
2.  **`maxlen`** 保存一个整数[数字](https://realpython.com/python-numbers/)，它指定了队列的最大长度。

如前所述，如果你不提供一个`iterable`，那么你会得到一个空的队列。如果您为 [`maxlen`](https://docs.python.org/3/library/collections.html#collections.deque.maxlen) 提供一个值，那么您的 deque 将只存储最多`maxlen`个项目。

最后，您还可以使用无序的可迭代对象，比如[集合](https://realpython.com/python-sets/)，来初始化您的队列。在这种情况下，最终队列中的项目不会有预定义的顺序。

[*Remove ads*](/account/join/)

## 高效弹出和追加项目

`deque`和`list`最重要的区别是前者允许你在序列的两端执行有效的追加和弹出操作。`deque`类实现专用的 [`.popleft()`](https://docs.python.org/3/library/collections.html#collections.deque.popleft) 和 [`.appendleft()`](https://docs.python.org/3/library/collections.html#collections.deque.appendleft) 方法，这些方法直接在序列的左端操作:

>>>

```py
>>> from collections import deque

>>> numbers = deque([1, 2, 3, 4])
>>> numbers.popleft()
1
>>> numbers.popleft()
2
>>> numbers
deque([3, 4])

>>> numbers.appendleft(2)
>>> numbers.appendleft(1)
>>> numbers
deque([1, 2, 3, 4])
```

在这里，您使用`.popleft()`和`.appendleft()`分别删除和添加值到`numbers`的左端。这些方法是针对`deque`的设计，你在`list`里是找不到的。

就像`list`，`deque`也提供了`.append()`和 [`.pop()`](https://docs.python.org/3/library/collections.html#collections.deque.pop) 方法来操作序列的右端。然而，`.pop()`表现不同:

>>>

```py
>>> from collections import deque

>>> numbers = deque([1, 2, 3, 4])
>>> numbers.pop()
4

>>> numbers.pop(0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: pop() takes no arguments (1 given)
```

这里，`.pop()`删除并返回队列中的最后一个值。该方法不接受索引作为参数，所以您不能使用它从您的 deques 中删除任意项。您只能使用它来移除和返回最右边的项目。

正如您之前了解到的，`deque`被实现为一个**双向链表**。因此，给定 deque 中的每一项都有一个指向序列中下一个和上一个项的引用([指针](https://realpython.com/pointers-in-python/))。

双向链表使得从任意一端追加和弹出条目变得简单而高效。这是可能的，因为只有指针需要更新。因此，这两个操作的性能相似。它们的性能也是可预测的，因为不需要重新分配内存和移动现有项目来接受新项目。

从常规 Python 列表的左端追加和弹出项目需要移动所有项目，这最终是一个 *O* ( *n* )操作。此外，向列表右端添加条目通常需要 Python 重新分配内存，并将当前条目复制到新的内存位置。之后，它可以添加新的项目。这个过程需要更长的时间来完成，追加操作从 *O* (1)转到 *O* ( *n* )。

考虑以下将项目追加到序列左端的性能测试，`deque`对`list`:

```py
# time_append.py

from collections import deque
from time import perf_counter

TIMES = 10_000
a_list = []
a_deque = deque()

def average_time(func, times):
    total = 0.0
    for i in range(times):
        start = perf_counter()
        func(i)
        total += (perf_counter() - start) * 1e9
    return total / times

list_time = average_time(lambda i: a_list.insert(0, i), TIMES)
deque_time = average_time(lambda i: a_deque.appendleft(i), TIMES)
gain = list_time / deque_time

print(f"list.insert() {list_time:.6} ns")
print(f"deque.appendleft() {deque_time:.6} ns  ({gain:.6}x faster)")
```

在这个脚本中，`average_time()`计算执行一个函数(`func`)给定数量的`times`所花费的平均时间。如果您从命令行运行脚本，那么您会得到以下输出:

```py
$ python time_append.py
list.insert()      3735.08 ns
deque.appendleft() 238.889 ns  (15.6352x faster)
```

在这个具体的例子中，`deque`上的`.appendleft()`比`list`上的`.insert()`快好几倍。注意`deque.appendleft()`是 *O* (1)，表示执行时间不变。但是，列表左端的`list.insert()`是 *O* ( *n* )，这意味着执行时间取决于要处理的项目数。

在本例中，如果您增加`TIMES`的值，那么您将获得`list.insert()`更高的时间测量值，但是`deque.appendleft()`的结果稳定(不变)。如果您想对 deques 和 lists 的 pop 操作进行类似的性能测试，那么您可以扩展下面的练习模块，并在完成后将您的结果与真正的 Python 的结果进行比较。



作为一个练习，您可以修改上面的脚本来计时`deque.popleft()`对`list.pop(0)`的操作，并评估它们的性能。



这里有一个测试`deque.popleft()`和`list.pop(0)`操作性能的脚本:

```py
# time_pop.py

from collections import deque
from time import perf_counter

TIMES = 10_000
a_list = [1] * TIMES
a_deque = deque(a_list)

def average_time(func, times):
    total = 0.0
    for _ in range(times):
        start = perf_counter()
        func()
        total += (perf_counter() - start) * 1e9
    return total / times

list_time = average_time(lambda: a_list.pop(0), TIMES)
deque_time = average_time(lambda: a_deque.popleft(), TIMES)
gain = list_time / deque_time

print(f"list.pop(0) {list_time:.6} ns")
print(f"deque.popleft() {deque_time:.6} ns  ({gain:.6}x faster)")
```

如果您在您的计算机上运行这个脚本，那么您将得到类似如下的输出:

```py
list.pop(0)     2002.08 ns
deque.popleft() 326.454 ns  (6.13282x faster)
```

同样，从底层序列的左端移除项目时，`deque`比`list`快。尝试改变`TIMES`的值，看看会发生什么！

`deque`数据类型旨在保证序列两端的有效追加和弹出操作。它非常适合处理需要用 Python 实现队列和堆栈数据结构的问题。

## 访问`deque` 中的随机项目

Python 的`deque`返回可变序列，其工作方式与列表非常相似。除了允许您有效地添加和弹出项目之外，deques 还提供了一组类似列表的方法和其他类似序列的操作来处理任意位置的项目。以下是其中的一些:

| [计]选项 | 描述 |
| --- | --- |
| [T2`.insert(i, value)`](https://docs.python.org/3/library/collections.html#collections.deque.insert) | 将项目`value`插入到索引`i`处的队列中。 |
| [T2`.remove(value)`](https://docs.python.org/3/library/collections.html#collections.deque.remove) | 删除第一次出现的`value`，如果`value`不存在，则提升 [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) 。 |
| [T2`a_deque[i]`](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations) | 从队列中检索索引`i`处的项目。 |
| [T2`del a_deque[i]`](https://docs.python.org/3/library/stdtypes.html#immutable-sequence-types) | 从队列中移除索引`i`处的项目。 |

您可以使用这些方法和技术来处理`deque`对象中任何位置的项目。下面是如何做到这一点:

>>>

```py
>>> from collections import deque

>>> letters = deque("abde")

>>> letters.insert(2, "c")
>>> letters
deque(['a', 'b', 'c', 'd', 'e'])

>>> letters.remove("d")
>>> letters
deque(['a', 'b', 'c', 'e'])

>>> letters[1]
'b'

>>> del letters[2]
>>> letters
deque(['a', 'b', 'e'])
```

这里，首先将`"c"`插入`letters`的`2`位置。然后使用`.remove()`将`"d"`从队列中移除。Deques 还允许**索引**访问项目，您在这里使用它来访问索引`1`处的`"b"`。最后，您可以使用`del` [关键字](https://realpython.com/python-keywords/)从队列中删除任何现有的条目。注意`.remove()`允许您通过值删除项目*，而`del`通过索引*删除项目*。*

即使`deque`对象支持索引，它们也不支持**切片**。换句话说，您不能使用[切片语法](https://docs.python.org/3/whatsnew/2.3.html?highlight=slicing#extended-slices)，`[start:stop:step]`从现有的队列中提取[切片](https://docs.python.org/3/glossary.html#term-slice)，就像您对常规列表所做的那样:

>>>

```py
>>> from collections import deque

>>> numbers = deque([1, 2, 3, 4, 5])

>>> numbers[1:3]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: sequence index must be integer, not 'slice'
```

Deques 支持索引，但有趣的是，它们不支持切片。当你试图从一个队列中得到一片时，你得到一个`TypeError`。一般来说，在链表上执行切片是低效的，所以这个操作是不可用的。

到目前为止，你已经看到了`deque`和`list`非常相似。然而，`list`是基于[数组](https://en.wikipedia.org/wiki/Array_data_structure)，而`deque`是基于一个双向链表。

在作为双向链表实现的`deque`背后有一个隐藏的成本:访问、插入和删除任意项不是有效的操作。为了执行它们，[解释器](https://realpython.com/interacting-with-python/#starting-the-interpreter)必须遍历队列，直到找到想要的条目。所以，它们是 O*(*n*)而不是 *O* (1)操作。*

下面是一个脚本，展示了在处理任意项目时 deques 和 lists 的行为:

```py
# time_random_access.py

from collections import deque
from time import perf_counter

TIMES = 10_000
a_list = [1] * TIMES
a_deque = deque(a_list)

def average_time(func, times):
    total = 0.0
    for _ in range(times):
        start = perf_counter()
        func()
        total += (perf_counter() - start) * 1e6
    return total / times

def time_it(sequence):
    middle = len(sequence) // 2
    sequence.insert(middle, "middle")
    sequence[middle]
    sequence.remove("middle")
    del sequence[middle]

list_time = average_time(lambda: time_it(a_list), TIMES)
deque_time = average_time(lambda: time_it(a_deque), TIMES)
gain = deque_time / list_time

print(f"list {list_time:.6} μs ({gain:.6}x faster)")
print(f"deque {deque_time:.6} μs")
```

这个脚本对在队列和列表中间插入、删除和访问项目进行计时。如果您运行该脚本，您将得到如下所示的输出:

```py
$ python time_random_access.py
list  63.8658 μs (1.44517x faster)
deque 92.2968 μs
```

Deques 不像列表那样是随机存取的数据结构。因此，从队列中间访问元素比在列表中做同样的事情效率更低。这里的要点是，deques 并不总是比 lists 更有效。

Python 的`deque`针对序列两端的操作进行了优化，因此在这方面它们一直比列表好。另一方面，列表更适合随机访问和固定长度的操作。下面是 deques 和 lists 在性能方面的一些差异:

| 操作 | `deque` | `list` |
| --- | --- | --- |
| 通过索引访问任意项目 | *O* ( *n* | *O*① |
| 在左端弹出和追加项目 | *O*① | *O* ( *n* |
| 在右端弹出和追加项目 | *O*① | *O* (1) +重新分配 |
| 在中间插入和删除项目 | *O* ( *n* | *O* ( *n* |

在列表的情况下，当解释器需要增加列表来接受新的条目时，`.append()`的分摊性能会受到内存重新分配的影响。此操作需要将所有当前项目复制到新的内存位置，这会显著影响性能。

这个总结可以帮助您为手头的问题选择合适的数据类型。但是，在从列表切换到 deques 之前，一定要对代码进行概要分析。两者都有各自的性能优势。

[*Remove ads*](/account/join/)

## 用`deque` 构建高效队列

正如您已经了解到的，`deque`被实现为一个双端队列，它提供了对**堆栈**和**队列**的一般化。在本节中，您将学习如何使用`deque`以优雅、高效和 Pythonic 式的方式在底层实现您自己的队列[抽象数据类型(ADT)](https://en.wikipedia.org/wiki/Abstract_data_type) 。

**注意:**在 Python 标准库中，你会找到 [`queue`](https://docs.python.org/3/library/queue.html#module-queue) 。该模块实现了多生产者、多消费者队列，允许您在多个线程之间安全地交换信息。

如果您正在使用队列，那么最好使用那些高级抽象而不是`deque`，除非您正在实现自己的数据结构。

队列是项目的集合。您可以通过在一端添加项目并从另一端删除项目来修改队列。

队列以**先进先出** ( [先进先出](https://en.wikipedia.org/wiki/FIFO_(computing_and_electronics)))的方式管理他们的项目。它们就像一个管道，你在管道的一端推入新的项目，从另一端弹出旧的项目。将一个项目添加到队列的一端被称为**入队**操作。从另一端移除一个项目称为**出列**。

为了更好地理解排队，以您最喜欢的餐馆为例。餐馆里有一长串人等着餐桌点餐。通常，最后到达的人会站在队伍的最后。一有空桌，排在队伍最前面的人就会离开。

下面是如何使用一个基本的`deque`对象来模拟这个过程:

>>>

```py
>>> from collections import deque

>>> customers = deque()

>>> # People arriving
>>> customers.append("Jane")
>>> customers.append("John")
>>> customers.append("Linda")

>>> customers
deque(['Jane', 'John', 'Linda'])

>>> # People getting tables
>>> customers.popleft()
'Jane'
>>> customers.popleft()
'John'
>>> customers.popleft()
'Linda'

>>> # No people in the queue
>>> customers.popleft()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: pop from an empty deque
```

在这里，首先创建一个空的`deque`对象来表示到达餐馆的人的队列。要让一个人入队，您可以使用 [`.append()`](https://docs.python.org/3/library/collections.html#collections.deque.append) ，它会将单个项目添加到右端。要让一个人出列，可以使用 [`.popleft()`](https://docs.python.org/3/library/collections.html#collections.deque.popleft) ，它移除并返回队列左端的单个项目。

酷！您的队列模拟有效！然而，由于`deque`是一个泛化，它的 [API](https://en.wikipedia.org/wiki/API) 与典型的队列 API 不匹配。例如，你有了`.append()`，而不是`.enqueue()`。你还有`.popleft()`而不是`.dequeue()`。此外，`deque`还提供了其他几种可能不符合您特定需求的操作。

好消息是，您可以创建带有您需要的功能的定制队列类，除此之外别无其他。为此，您可以在内部使用一个队列来存储数据，并在您的自定义队列中提供所需的功能。您可以把它看作是[适配器设计模式](https://sourcemaking.com/design_patterns/adapter)的一个实现，其中您将 deque 的接口转换成看起来更像队列接口的东西。

例如，假设您需要一个只提供以下功能的自定义队列抽象数据类型:

*   入队项目
*   出队项目
*   返回队列的长度
*   支持成员资格测试
*   支持正向和反向迭代
*   提供用户友好的字符串表示

在这种情况下，您可以编写如下所示的`Queue`类:

```py
# custom_queue.py

from collections import deque

class Queue:
    def __init__(self):
        self._items = deque()

    def enqueue(self, item):
        self._items.append(item)

    def dequeue(self):
        try:
            return self._items.popleft()
        except IndexError:
            raise IndexError("dequeue from an empty queue") from None

    def __len__(self):
        return len(self._items)

    def __contains__(self, item):
        return item in self._items

    def __iter__(self):
        yield from self._items

    def __reversed__(self):
        yield from reversed(self._items)

    def __repr__(self):
        return f"Queue({list(self._items)})"
```

这里，`._items`保存了一个`deque`对象，允许您存储和操作队列中的项目。`Queue`使用`deque.append()`实现`.enqueue()`来将项目添加到队列的末尾。它还使用`deque.popleft()`实现了`.dequeue()`,以有效地从队列的开头移除项目。

[特殊方法](https://docs.python.org/3/glossary.html#term-special-method)支持以下功能:

| 方法 | 支持 |
| --- | --- |
| [T2`.__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) | 带`len()`的长度 |
| [T2`.__contains__()`](https://docs.python.org/3/reference/datamodel.html#object.__contains__) | 使用`in`进行成员资格测试 |
| [T2`.__iter__()`](https://docs.python.org/3/reference/datamodel.html#object.__iter__) | 正常迭代 |
| [T2`.__reversed__()`](https://docs.python.org/3/reference/datamodel.html#object.__reversed__) | 反向迭代 |
| [T2`.__repr__()`](https://docs.python.org/3/reference/datamodel.html#object.__repr__) | 字符串表示 |

理想情况下，`.__repr__()`应该返回一个表示有效 Python 表达式的字符串。该表达式将允许您使用相同的值明确地重新创建对象。

然而，在上面的例子中，意图是使用方法的[返回](https://realpython.com/python-return-statement/)值来优雅地在[交互外壳](https://realpython.com/interacting-with-python/)上显示对象。通过接受一个初始化 iterable 作为`.__init__()`的参数并从中构建实例，可以从这个特定的字符串表示中构建`Queue`实例。

有了这些最后的添加，您的`Queue`类就完成了。要在代码中使用该类，您可以执行如下操作:

>>>

```py
>>> from custom_queue import Queue

>>> numbers = Queue()
>>> numbers
Queue([])

>>> # Enqueue items
>>> for number in range(1, 5):
...     numbers.enqueue(number)
...
>>> numbers
Queue([1, 2, 3, 4])

>>> # Support len()
>>> len(numbers)
4

>>> # Support membership tests
>>> 2 in numbers
True
>>> 10 in numbers
False

>>> # Normal iteration
>>> for number in numbers:
...     print(f"Number: {number}")
...
1
2
3
4
```

作为练习，您可以测试剩余的特性并实现其他特性，比如支持相等测试、移除和访问随机项等等。来吧，试一试！

[*Remove ads*](/account/join/)

## 探索`deque`的其他特性

除了您到目前为止已经看到的特性，`deque`还提供了其他特定于其内部设计的方法和属性。它们为这种多用途的数据类型增加了新的有用的功能。

在本节中，您将了解 deques 提供的其他方法和属性，它们是如何工作的，以及如何在您的代码中使用它们。

### 限制最大项数:`maxlen`

`deque`最有用的特性之一是在实例化类时，可以使用`maxlen`参数指定给定队列的**最大长度**。

如果您为`maxlen`提供一个值，那么您的队列将只存储最多`maxlen`个项目。在这种情况下，你有一个**有界的德奎**。一旦有界的 deque 充满了指定数量的项目，在任一端添加新项目都会自动删除并丢弃另一端的项目:

>>>

```py
>>> from collections import deque

>>> four_numbers = deque([0, 1, 2, 3, 4], maxlen=4) # Discard 0
>>> four_numbers
deque([1, 2, 3, 4], maxlen=4)

>>> four_numbers.append(5)  # Automatically remove 1
>>> four_numbers
deque([2, 3, 4, 5], maxlen=4)

>>> four_numbers.append(6)  # Automatically remove 2
>>> four_numbers
deque([3, 4, 5, 6], maxlen=4)

>>> four_numbers.appendleft(2) # Automatically remove 6
>>> four_numbers
deque([2, 3, 4, 5], maxlen=4)

>>> four_numbers.appendleft(1)  # Automatically remove 5
>>> four_numbers
deque([1, 2, 3, 4], maxlen=4)

>>> four_numbers.maxlen
4
```

如果输入 iterable 中的条目数大于`maxlen`，那么`deque`将丢弃最左边的条目(本例中为`0`)。一旦队列已满，在任何一端追加一个项目都会自动删除另一端的项目。

请注意，如果您没有为`maxlen`指定一个值，那么它默认为 [`None`](https://realpython.com/null-in-python/) ，并且队列可以增长到任意数量的项目。

有了限制最大项数的选项，您就可以使用 deques 来跟踪给定对象或事件序列中的最新元素。例如，您可以跟踪银行帐户中的最后五笔交易、编辑器中最后十个打开的文本文件、浏览器中的最后五页等等。

注意，`maxlen`在您的 deques 中是一个只读属性，它允许您检查 deques 是否已满，就像在`deque.maxlen == len(deque)`中一样。

最后，您可以将`maxlen`设置为任意正整数，表示您希望存储在特定队列中的最大项数。如果你给`maxlen`提供一个负值，那么你会得到一个`ValueError`。

### 旋转项目:`.rotate()`

deques 的另一个有趣的特性是可以通过在非空的 deques 上调用 [`.rotate()`](https://docs.python.org/3/library/collections.html#collections.deque.rotate) 来旋转它们的元素。这个方法将一个整数`n`作为参数，并将项目`n`向右旋转一步。换句话说，它以循环方式将`n`项目从右端移动到左端。

`n`的默认值为`1`。如果你给`n`提供一个负值，那么旋转向左:

>>>

```py
>>> from collections import deque

>>> ordinals = deque(["first", "second", "third"])

>>> # Rotate items to the right
>>> ordinals.rotate()
>>> ordinals
deque(['third', 'first', 'second'])

>>> ordinals.rotate(2)
>>> ordinals
deque(['first', 'second', 'third'])

>>> # Rotate items to the left
>>> ordinals.rotate(-2)
>>> ordinals
deque(['third', 'first', 'second'])

>>> ordinals.rotate(-1)
>>> ordinals
deque(['first', 'second', 'third'])
```

在这些例子中，你使用`.rotate()`和不同的`n`值旋转`ordinals`几次。如果你调用`.rotate()`而没有参数，那么它依赖于`n`的默认值，并向右旋转队列`1`的位置。使用负的`n`调用该方法允许您向左旋转项目。

### 一次添加多个项目:`.extendleft()`

像常规列表一样，deques 提供了一个 [`.extend()`](https://docs.python.org/3/library/collections.html#collections.deque.extend) 方法，该方法允许您使用一个`iterable`作为参数向 deques 的右端添加几个项目。此外，deques 有一个名为 [`extendleft()`](https://docs.python.org/3/library/collections.html#collections.deque.extendleft) 的方法，它将一个`iterable`作为参数，并将其项目一次性添加到目标 deques 的左端:

>>>

```py
>>> from collections import deque

>>> numbers = deque([1, 2])

>>> # Extend to the right
>>> numbers.extend([3, 4, 5])
>>> numbers
deque([1, 2, 3, 4, 5])

>>> # Extend to the left
>>> numbers.extendleft([-1, -2, -3, -4, -5])
>>> numbers
deque([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
```

用`iterable`调用`.extendleft()`将目标队列向左扩展。在内部，`.extendleft()`执行一系列单独的`.appendleft()`操作，从左到右处理输入的 iterable。这最终会以相反的顺序将项目添加到目标队列的左端。

[*Remove ads*](/account/join/)

## 使用`deque` 的类序列特征

由于 deques 是可变序列，它们实现了几乎所有与[序列](https://docs.python.org/3/library/stdtypes.html?highlight=built#common-sequence-operations)和[可变序列](https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types)相同的方法和操作。到目前为止，您已经了解了其中的一些方法和操作，比如`.insert()`、索引、成员测试等等。

以下是您可以对`deque`对象执行的其他操作的几个例子:

>>>

```py
>>> from collections import deque

>>> numbers = deque([1, 2, 2, 3, 4, 4, 5])

>>> # Concatenation
>>> numbers + deque([6, 7, 8])
deque([1, 2, 2, 3, 4, 4, 5, 6, 7, 8])

>>> # Repetition
>>> numbers * 2
deque([1, 2, 2, 3, 4, 4, 5, 1, 2, 2, 3, 4, 4, 5])

>>> # Common sequence methods
>>> numbers = deque([1, 2, 2, 3, 4, 4, 5])
>>> numbers.index(2)
1
>>> numbers.count(4)
2

>>> # Common mutable sequence methods
>>> numbers.reverse()
>>> numbers
deque([5, 4, 4, 3, 2, 2, 1])

>>> numbers.clear()
>>> numbers
deque([])
```

您可以使用加法[运算符](https://realpython.com/python-operators-expressions/) ( `+`)来连接两个现有的队列。另一方面，乘法运算符(`*`)返回一个新的 deque，相当于重复原始 deque 任意次。

关于其他排序方法，下表提供了一个总结:

| 方法 | 描述 |
| --- | --- |
| [T2`.clear()`](https://docs.python.org/3/library/collections.html#collections.deque.clear) | 从队列中删除所有元素。 |
| [T2`.copy()`](https://docs.python.org/3/library/collections.html#collections.deque.copy) | 创建一个 deque 的浅表副本。 |
| [T2`.count(value)`](https://docs.python.org/3/library/collections.html#collections.deque.count) | 计算`value`在队列中出现的次数。 |
| [T2`.index(value)`](https://docs.python.org/3/library/collections.html#collections.deque.index) | 返回`value`在队列中的位置。 |
| [T2`.reverse()`](https://docs.python.org/3/library/collections.html#collections.deque.reverse) | 在适当的位置反转队列的元素，然后返回`None`。 |

这里，`.index()`还可以带两个可选参数:`start`和`stop`。它们允许您将搜索限制在`start`当天或之后和`stop`之前的那些项目。如果`value`没有出现在当前的队列中，该方法将引发一个 [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) 。

与列表不同，deques 不包含一个`.sort()`方法来对序列进行排序。这是因为对链表进行排序是一个低效的操作。如果您需要对一个队列进行排序，那么您仍然可以使用`sorted()`。

## 将 Python 的`deque`付诸行动

您可以在很多用例中使用 deques，比如实现队列、堆栈和[循环缓冲区](https://en.wikipedia.org/wiki/Circular_buffer)。您还可以使用它们来维护撤销-重做历史，将传入的请求排队到 [web 服务](https://en.wikipedia.org/wiki/Web_service)，保存最近打开的文件和网站的列表，在多线程之间安全地交换数据，等等。

在接下来的几节中，您将编写几个小例子来帮助您更好地理解如何在代码中使用 deques。

### 保存页面历史记录

用一个`maxlen`来限制项目的最大数量使得`deque`适合解决几个问题。例如，假设你正在构建一个应用程序，[从搜索引擎和社交媒体网站搜集](https://realpython.com/python-web-scraping-practical-introduction/)数据。有时，您需要跟踪应用程序请求数据的最后三个站点。

要解决这个问题，您可以使用一个`maxlen`为`3`的队列:

>>>

```py
>>> from collections import deque

>>> sites = (
...     "google.com",
...     "yahoo.com",
...     "bing.com"
... )

>>> pages = deque(maxlen=3)
>>> pages.maxlen
3

>>> for site in sites:
...     pages.appendleft(site)
...

>>> pages
deque(['bing.com', 'yahoo.com', 'google.com'], maxlen=3)

>>> pages.appendleft("facebook.com")
>>> pages
deque(['facebook.com', 'bing.com', 'yahoo.com'], maxlen=3)

>>> pages.appendleft("twitter.com")
>>> pages
deque(['twitter.com', 'facebook.com', 'bing.com'], maxlen=3)
```

在这个例子中，`pages`保存了您的应用程序最近访问的三个站点的列表。一旦`pages`满了，向队列的一端添加一个新站点会自动丢弃另一端的站点。此行为使您的列表与您最近使用的三个站点保持一致。

请注意，您可以将`maxlen`设置为任意正整数，表示要存储在当前队列中的项数。例如，如果你想保存一个十个站点的列表，那么你可以将`maxlen`设置为`10`。

### 线程间共享数据

Python 的`deque`在你编写[多线程](https://realpython.com/intro-to-python-threading/)应用时也很有用，正如 [Raymond Hettinger](https://twitter.com/raymondh) 所描述的，他是 Python 的核心开发者，也是`deque`和`collections`模块的创建者:

> 在 CPython 中，队列的`.append()`、`.appendleft()`、`.pop()`、`.popleft()`和`len(d)`操作是线程安全的。([来源](https://bugs.python.org/msg199368))

因此，您可以安全地在不同的线程中同时从队列的两端添加和删除数据，而没有数据损坏或其他相关问题的风险。

为了尝试一下`deque`在多线程应用中的工作方式，启动您最喜欢的[代码编辑器](https://realpython.com/python-ides-code-editors-guide/)，创建一个名为`threads.py`的新脚本，并向其中添加以下代码:

```py
# threads.py

import logging
import random
import threading
import time
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(message)s")

def wait_seconds(mins, maxs):
    time.sleep(mins + random.random() * (maxs - mins))

def produce(queue, size):
    while True:
        if len(queue) < size:
            value = random.randint(0, 9)
 queue.append(value)            logging.info("Produced: %d -> %s", value, str(queue))
        else:
            logging.info("Queue is saturated")
        wait_seconds(0.1, 0.5)

def consume(queue):
    while True:
        try:
 value = queue.popleft()        except IndexError:
            logging.info("Queue is empty")
        else:
            logging.info("Consumed: %d -> %s", value, str(queue))
        wait_seconds(0.2, 0.7)

logging.info("Starting Threads...\n")
logging.info("Press Ctrl+C to interrupt the execution\n")

shared_queue = deque()

threading.Thread(target=produce, args=(shared_queue, 10)).start()
threading.Thread(target=consume, args=(shared_queue,)).start()
```

这里，`produce()`将一个`queue`和一个`size`作为自变量。然后它在一个 [`while`循环](https://realpython.com/python-while-loop/)中使用 [`random.randint()`](https://docs.python.org/3/library/random.html#random.randint) 连续产生[个随机](https://realpython.com/python-random/)数，并将它们存储在一个名为`shared_queue`的[全局](https://realpython.com/python-scope-legb-rule/#modules-the-global-scope)队列中。由于将项目附加到 deque 是一个线程安全的操作，所以您不需要使用[锁](https://en.wikipedia.org/wiki/Lock_(computer_science))来保护其他线程的共享数据。

助手函数`wait_seconds()`模拟`produce()`和`consume()`都代表长时间运行的操作。它返回一个在给定的秒数范围`mins`和`maxs`之间的随机等待时间值。

在`consume()`中，您在一个循环中调用`.popleft()`来系统地从`shared_queue`中检索和移除数据。您将对`.popleft()`的调用包装在一个 [`try` … `except`](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions) 语句中，以处理共享队列为空的情况。

注意，虽然您在全局[名称空间](https://realpython.com/python-namespaces-scope/)中定义了`shared_queue`，但是您可以通过`produce()`和`consume()`中的局部变量来访问它。直接访问全局变量会有更多的问题，肯定不是最佳实践。

脚本中的最后两行创建并启动单独的线程来并发执行`produce()`和`consume()`。如果您从命令行运行该脚本，那么您将得到类似如下的输出:

```py
$ python threads.py
Starting Threads...

Press Ctrl+C to interrupt the execution

Produced: 1 -> deque([1])
Consumed: 1 -> deque([])
Queue is empty
Produced: 3 -> deque([3])
Produced: 0 -> deque([3, 0])
Consumed: 3 -> deque([0])
Consumed: 0 -> deque([])
Produced: 1 -> deque([1])
Produced: 0 -> deque([1, 0])
 ...
```

生产者线程将数字添加到共享队列的右端，而消费者线程从左端消费数字。要中断脚本执行，您可以按键盘上的 `Ctrl` + `C` 。

最后可以用`produce()`和`consume()`里面的时间间隔来玩一点。更改您传递给`wait_seconds()`的值，观察当生产者比消费者慢时程序的行为，反之亦然。

[*Remove ads*](/account/join/)

### 模拟`tail`命令

您将在这里编写的最后一个示例模拟了 [`tail`命令](https://en.wikipedia.org/wiki/Tail_(Unix))，该命令在 [Unix](https://en.wikipedia.org/wiki/Unix) 和[类 Unix](https://en.wikipedia.org/wiki/Unix-like)操作系统上可用。该命令在命令行接受一个文件路径，并将该文件的最后十行输出到系统的标准输出。您可以使用`-n`、`--lines`选项调整需要`tail`打印的行数。

这里有一个小的 Python 函数，它模拟了`tail`的核心功能:

>>>

```py
>>> from collections import deque

>>> def tail(filename, lines=10):
...     try:
...         with open(filename) as file:
...             return deque(file, lines) ...     except OSError as error:
...         print(f'Opening file "{filename}" failed with error: {error}')
...
```

在这里，你定义`tail()`。第一个参数`filename`将目标文件的路径保存为一个[字符串](https://realpython.com/python-strings/)。第二个参数，`lines`，代表您希望从目标文件的末尾检索的行数。注意`lines`默认为`10`来模拟`tail`的默认行为。

**注意:**这个例子的最初想法来自于`deque`上的 Python 文档。查看关于 [`deque`食谱](https://docs.python.org/3/library/collections.html#deque-recipes)的部分以获得更多的例子。

突出显示的行中的队列最多只能存储您传递给`lines`的项目数。这保证了您从输入文件的末尾获得所需的行数。

正如您之前看到的，当您创建一个有界的 deque 并用一个 iterable 初始化它时，iterable 包含的条目比允许的多(`maxlen`),`deque`构造函数会丢弃输入中所有最左边的条目。正因为如此，您最终得到了目标文件的最后一行`maxlen`。

## 结论

[队列](https://realpython.com/python-data-structures/#queues-fifos)和[栈](https://realpython.com/how-to-implement-python-stack/)是编程中常用的**抽象数据类型**。它们通常需要对底层数据结构的两端进行有效的**弹出**和**追加**操作。Python 的 [`collections`](https://docs.python.org/3/library/collections.html#module-collections) 模块提供了一个名为 [`deque`](https://docs.python.org/3/library/collections.html#collections.deque) 的数据类型，它是专门为两端的快速和内存高效的追加和弹出操作而设计的。

使用`deque`，您可以以优雅、高效和 Pythonic 化的方式在底层编写自己的队列和堆栈。

**在本教程中，您学习了如何:**

*   在你的代码中创建并使用 Python 的 **`deque`**
*   高效地**从序列的两端用`deque`追加**和**弹出**项
*   使用`deque`在 Python 中构建高效的**队列**和**栈**
*   决定什么时候用 **`deque`** 代替 **`list`**

在本教程中，您还编写了一些例子，帮助您了解 Python 中的一些常见用例。*****