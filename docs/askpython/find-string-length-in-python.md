# 在 Python 中查找字符串长度

> 原文：<https://www.askpython.com/python/string/find-string-length-in-python>

我们可以在 Python 中使用内置的`len()`函数找到[字符串](https://www.askpython.com/python/python-data-types#python-string)长度。让我们看看这个函数是如何工作的，并且让我们尝试使用`len()`来查找各种类型的 Python 字符串的长度。

* * *

## 使用 len()

我们来看一些简单的例子来说明`len()`。

```py
>>> a = "Hello from AskPython"
>>> print(len(a))
20

```

这会打印 20，因为这是字符串中的字符数。因此，我们可以使用`len()`找到长度。

即使字符串有特殊字符，只要它能以某种 Unicode 格式编码，我们就能计算它的长度。

```py
>>> a = 'AåBç'
>>> print(len(a))
4

```

对于带有特殊转义字符的字符串(它们以反斜杠`(\)`为前缀)，只有字符会被计入长度，而反斜杠不会。例子包括(`\n`、`\t`、`\'`等)

```py
>>> a = 'A\t\t'
>>> print(len(a))
3

>>> b = 'A\n\nB'
>>> print(len(b))
4

>>> c = 'A\'B'
>>> print(len(c))
3

```

对于原始字符串，由于它们将反斜杠(`\`)视为文字，所以反斜杠将计入字符串的长度。

```py
>>> s = r'A\t\t'
>>> print(len(s))
5

```

* * *

## 透镜的工作()

当我们使用 String 对象调用`len()`函数时，String 对象的`__len__()`方法被调用。

```py
>> a = "Hello from AskPython"
>>> a.__len__()
20

```

为了证明这一点，让我们在自定义类上实现我们自己的`len()`。既然`__len__()`作用于对象，我们必须继承类`object`。

```py
class Student(object):
    def __init__(self, name):
        self.name = name

    def __len__(self):
        print("Invoking the __len__() method on the Student Object to find len()...")
        count = 0
        for i in self.name:
            count += 1
        return count

a = Student("Amit")
print(len(a))

```

由于`len()`方法调用了`__len__()`，我们将遍历该函数，它计算 iterable 中对象的数量。因为我们传递了一个字符串，我们将简单地得到长度，结果是 4！

**输出**

```py
Invoking the __len__() method on the Student Object to find len()...
4

```

因此，我们为类`Student`实现了自己的`len()`方法！很神奇，不是吗？

* * *

## 参考

*   关于字符串长度的 JournalDev 文章

* * *