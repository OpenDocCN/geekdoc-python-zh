# 如何使用 Python iter()方法？

> 原文：<https://www.askpython.com/python/python-iter-function>

在本文中，我们将看看如何使用 Python iter()函数。

通常，我们可能想要使用[迭代器](https://www.askpython.com/python/built-in-methods/python-iterator)，它是一个本身加载 Python 对象的对象。但是，与数组或列表不同，迭代器只是在需要时加载对象。

这被称为*延迟加载*，或者基于流的加载。这是非常有用的，如果你想节省内存，而不是一次加载整个对象，如果你的对象非常大！

* * *

## Python iter()的基本语法

我们可以使用`iter()`函数生成一个可迭代对象的迭代器，比如字典、[列表](https://www.askpython.com/python/list/iterate-through-list-in-python)、集合等。

使用 Python `iter()`函数的基本语法如下:

```py
iterator = iter(iterable)

```

这将从`iterable`对象生成一个`iterator`。

我们可以简单地使用`next(iterator)`一个接一个地加载对象，直到我们得到`StopIteration`异常。

另外，注意我们**不能**使用同一个迭代器再次遍历 iterable。我们必须在迭代之前使用 Python `iter()`生成另一个迭代器！

* * *

## 使用 Python ITER()——一个简单的例子

下面是一个使用`iter()`的简单例子。我们将获取一个包含 10 个元素的列表，并逐个加载它们。

```py
a = [i for i in range(10)]

iterator = iter(a)

while True:
    try:
        out = next(iterator) # Load the next element
        print(f"Iterator loaded {out}")
    except StopIteration:
        # End of loading. Break out of the while loop
        print("End of iterator loading!")
        break

```

**输出**

```py
Iterator loaded 0
Iterator loaded 1
Iterator loaded 2
Iterator loaded 3
Iterator loaded 4
Iterator loaded 5
Iterator loaded 6
Iterator loaded 7
Iterator loaded 8
Iterator loaded 9
End of iterator loading!

```

正如你所看到的，事实上，它从列表中一个接一个地加载元素，直到我们捕捉到`StopIteration`异常！

* * *

## 将 Python iter()用于自定义对象

正如我前面提到的，我们可以在任何对象上使用 Python iter()，只要它是可迭代的。

这也适用于自定义对象，只要它满足一些条件。

但是 Python 中任何对象成为 iterable 的条件是什么呢？

*   该对象的类必须有`__iter__()`方法。
*   对象的类必须有`__next__()`方法。另外，如果达到终止条件，建议您也引发一个`StopIteration`异常。

现在，Python `iter()`方法将构造迭代器并调用`__iter__()`方法。类似地，`next(iterator)`将在幕后调用`__next__()`方法。

**注意**:如果这个类没有**没有**这些方法，那么它必须至少有`__getitem()__`方法，整数参数从 0 开始。否则，我们会得到一个`TypeError`异常。

现在让我们为一个自定义对象写一个类，它生成整数直到一个极限。

```py
class MyClass():
    def __init__(self, max_val):
        # Set the maximum limit (condition)
        # max_val must be a natural number
        assert isinstance(max_val, int) and max_val >= 0
        self.max_val = max_val
    def __iter__(self):
        # Called when the iterator is generated
        # Initialise the value to 0
        self.value = 0
        return self
    def __next__(self):
        # Called when we do next(iterator)
        if self.value >= self.max_val:
            # Terminating condition
            raise StopIteration
        self.value += 1
        # Return the previously current value
        return self.value - 1

# Set the limit to 10
my_obj = MyClass(10)

# An iterator to the object
my_iterator = iter(my_obj)

while True:
    try:
        val = next(my_obj)
        print(f"Iterator Loaded {val}")
    except StopIteration:
        print("Iterator loading ended!")
        break

```

**输出**

```py
Iterator Loaded 0
Iterator Loaded 1
Iterator Loaded 2
Iterator Loaded 3
Iterator Loaded 4
Iterator Loaded 5
Iterator Loaded 6
Iterator Loaded 7
Iterator Loaded 8
Iterator Loaded 9
Iterator loading ended!

```

如您所见，我们确实能够在自定义对象上使用`iter()`函数。`__iter__()`方法创建迭代器对象，然后我们使用`__next__()`更新它。

终止条件是当当前值大于最大值时，此时我们引发一个`StopIteration`异常。

* * *

## 用 iter()生成值直到一个 sentinel 值

我们可以再传递一个参数给 Python `iter()`。第二个参数称为`sentinel`元素。

如果我们传递这个 sentinel 元素，[迭代器](https://www.askpython.com/python-modules/python-itertools-module)将继续生成值，直到生成的值等于这个 sentinel 值，之后`StopIteration`将被引发。

在此之后，迭代器生成将自动停止！

如果您有来自函数的顺序数据，这将非常有用。函数也是必要的，因为如果我们使用 sentinel 参数，第一个参数**必须是可调用的。**

```py
iterator = iter(callable_object, sentinel)

```

这里，`iterator`是一个迭代器，它会一直调用`callable_object`，直到返回值等于`sentinel`。

在这里，`callable_object`可以是函数，方法，甚至是 Lambda！

让我们举一个简单的例子，使用 Lambda 作为可调用函数。

我们将接受一个字符串作为输入，并将其传递给 lambda 函数，并一直生成值，直到出现一个*换行符* sentinel 元素(' \n ')。

```py
a = "This is a long string consisting of two lines.\nThis is the second line.\nThis is the third line."

start = 0
size = 1

def func(a):
    return a[start: start+size]

iterator = iter(lambda: func(a), '\n')

# Will generate values until '\n'
for out in iterator:
    print(f"Iterator loaded {out}")
    start += size

print("Encountered Newline!")

```

输出

```py
Iterator loaded T
Iterator loaded h
Iterator loaded i
Iterator loaded s
Iterator loaded
Iterator loaded i
Iterator loaded s
Iterator loaded
Iterator loaded a
Iterator loaded
Iterator loaded l
Iterator loaded o
Iterator loaded n
Iterator loaded g
Iterator loaded
Iterator loaded s
Iterator loaded t
Iterator loaded r
Iterator loaded i
Iterator loaded n
Iterator loaded g
Iterator loaded
Iterator loaded c
Iterator loaded o
Iterator loaded n
Iterator loaded s
Iterator loaded i
Iterator loaded s
Iterator loaded t
Iterator loaded i
Iterator loaded n
Iterator loaded g
Iterator loaded
Iterator loaded o
Iterator loaded f
Iterator loaded
Iterator loaded t
Iterator loaded w
Iterator loaded o
Iterator loaded
Iterator loaded l
Iterator loaded i
Iterator loaded n
Iterator loaded e
Iterator loaded s
Iterator loaded .
Encountered Newline!

```

正如您所看到的，迭代器一直生成值，直到遇到换行符！您也可以使用一个`while`循环并捕获`StopIteration`异常来完成相同的程序。

如果您想处理函数返回的输出块，这实际上非常有用，所以一定要注意`iter()`的 sentinel 参数！

* * *

## 结论

在本文中，我们研究了如何使用 Python 中的`iter()`函数为各种对象生成可迭代对象。

## 参考

*   iter()上的 Python 官方文档
*   关于迭代器的 AskPython 文章

* * *