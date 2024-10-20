# Python 迭代器

> 原文：<https://www.askpython.com/python/built-in-methods/python-iterator>

Python 迭代器是一个可以被迭代的对象。任何支持这种迭代的对象都被称为**迭代器**。

现在，你可能会感到困惑。存在另一个名为 **iterable** 的对象。那是什么？让我们来看看。

* * *

## 迭代器和可迭代对象

任何 iterable 都是可以被迭代的对象。也就是说，我们可以使用迭代器遍历这个对象。

可迭代对象的例子有元组、列表、字符串、数组等。

为了从一个*可迭代的*构造一个**迭代器**，我们可以使用`iter()`方法。

```py
iterable_list = [1, 2, 3]
iterable_string = "Hello"

iterator_1 = iter(iterable_list)
iterator_2 = iter(iterable_string)

print(iterator_1, type(iterator_1))
print(iterator_2, type(iterator_2))

```

输出

```py
<list_iterator object at 0x7f887b01aed0> <class 'list_iterator'>
<str_iterator object at 0x7f887b01af50> <class 'str_iterator'>

```

输出显示我们已经创建了两个迭代器；一个用于列表，一个用于字符串。

现在让我们看看迭代器对象支持的方法。

* * *

## Python 迭代器方法

迭代器对象有两个特殊的方法可以使用，称为 **iter()** 和 **next()** 。

前面使用了`iter()`方法，从 iterable 中获取 Python **迭代器**对象。

现在，为了遍历迭代器，我们可以使用`next()`方法来获取 iterable 中的下一个元素。

格式:

```py
next_iterable_object = next(iterator)

```

当没有更多的元素可去时，这将终止并引发一个`StopIteration` [异常](https://www.askpython.com/python/python-exception-handling)。

为了说明这一切，让我们打印列表迭代器的所有元素。

```py
>>> iterable_list = [1, 2, 3]
>>> iterator_1 = iter(iterable_list)
>>> next(iterator_1)
1
>>> next(iterator_1)
2
>>> next(iterator_1)
3
>>> next(iterator_1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration

```

如您所见，当您超出 iterable 的长度时，这确实会引发`StopIteration`异常。

现在，让我们进入下一步:制作我们自己的迭代器！

* * *

## 用 Python 构建我们自己的迭代器

任何迭代器对象都有可数的元素，可以被遍历。但是怎么才能做出自己的迭代器呢？我们需要创建自己的类。

在 Python 中，构建任何迭代器都涉及到一个叫做**迭代器协议**的协议。

这个协议包含两个特定的方法，称为`__iter__()`和`__next__()`，类似于一般的迭代器方法，但是因为它们在一个类中，所以用这个符号作为前缀和后缀，以示区别。

`iter()`和`next()`方法在内部调用这些方法，因此，为了生成迭代器，我们需要在我们的类中定义我们自己的`__iter__()`和`__next__()`方法。

让我们创建一个简单的迭代器，它只遍历一个列表，如果元素数大于 5，就会引发一个 **StopIteration** 异常。

在我们的`next()`方法中，我们还将打印到目前为止迭代的元素数量。

```py
class MyClass():
    def __init__(self):
        self.counter = 0
        # Set the limit
        self.limit = 5

    def __iter__(self):
        # The iter() method internally calls this
        print('Call to __iter__()')
        return self

    def __next__(self):
        print('Call to __next__()')
        if self.counter > self.limit:
            raise StopIteration
        else:
            # Increment counter and return it
            self.counter += 1
            return self.counter

# Create the iterable
my_obj = MyClass()

# Create the iterator object from the iterable
my_it = iter(my_obj)

for i in range(7):
    print(next(my_it))

```

**输出**

```py
Call to __iter__()
Call to __next__()
1
Call to __next__()
2
Call to __next__()
3
Call to __next__()
4
Call to __next__()
5
Call to __next__()
6
Call to __next__()
Traceback (most recent call last):
  File "iter.py", line 29, in <module>
    print(next(my_it))
  File "iter.py", line 15, in __next__
    raise StopIteration
StopIteration

```

这里，它打印从 1 到 6 的数字，下一个调用将触发`StopIteration`异常，因为我们已经超出了限制。

我们已经制作了自己的迭代器！

* * *

## 结论

我希望你现在对迭代器有了很好的理解，并且在阅读完这篇文章后，关于这个概念的任何疑问都被清除了。如果没有，请在下面的评论区问他们！

* * *

## 参考

*   关于迭代器的 JournalDev 文章

* * *