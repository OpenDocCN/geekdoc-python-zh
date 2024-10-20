# 在 Python 中使用 id()函数

> 原文：<https://www.askpython.com/python/id-function-in-python>

大家好！在今天的文章中，我们将看看 Python 中的 id()函数。

id()函数返回任何 [Python 对象](https://www.askpython.com/python/oops/python-classes-objects)的身份。这将返回不同对象的整数标识号。

底层的 CPython 实现使用`id()`函数作为对象在内存中的地址。

让我们用一些例子来更好地理解这一点。

* * *

## Python 中 id()函数的基本语法

这个函数接受任何 Python 对象——无论是整数、浮点、字符串、列表、类、函数、lambda 等等，并返回一个整数 id。

```py
val = id(object)

```

* * *

## 在 Python 中使用 id()

对象的 id 对于 Python 缓存这些变量的值很有用。这种使用`id()`检索缓存值的机制让 Python 性能更好！

这在多个变量引用同一个对象的情况下也很有帮助。

```py
a = 1233.45
b = a

print(id(a))
print(id(b))

```

**输出**

```py
2775655780976
2775655780976

```

在这种情况下，Python 更容易跟踪被引用的对象，因此 a 的 id()将与 b 的 id 相同。

现在让我们试着在一些简单的 Python 对象上使用它。

```py
print(id(103)) # Int

print(id(104))

print(id(10.25)) # Float

print(id('Hello from AskPython')) # String

print(id([1, 2, 3])) # List

print(id(lambda x: x * x)) # Lambda

```

**输出**

```py
1658621232
1658621264
2775655780976
2775665230232
2775665206344
2775656111776

```

你可以观察到，对于整数 103 和 104，它们的 ID 号只有 32 的区别。这是有道理的！为什么？

还记得我们提到过`id()`是指对象的地址吗？

id(104)是整数 103 之后的下一个地址块。因为 Python 中的整数存储为 4 个字节，所以这代表 32 位，这正好是它们的 id 号之间的差异。

因此 Python 将所有整数的列表存储在顺序块中，这些块的间距相等。有道理？

现在，让我们在琴弦上测试它们:

```py
# strings
s1 = 'ABC'
s2 = 'ABC'
print(id(s1))
print(id(s2))

```

**输出**

```py
2775656418080
2775656418080

```

正如您所观察到的，Python 确实缓存了字符串以节省内存！

请记住，缓存只能在不可变的 Python 对象上工作，比如 integer、string 和 floats。[元组](https://www.askpython.com/python/tuple/python-tuple)、[列表](https://www.askpython.com/python/list/concatenate-multiple-lists-in-python)等都是可变对象，这里缓存不起作用！

为了证明这一点，让我们检查具有相同元素的两个列表的 id:

```py
>>> l1 = [1, 2, 3, 4]
>>> l2 = [1, 2, 3 ,4]
>>> id(l1)
2775665206344
>>> id(l2)
2775665185224

```

这里，由于列表是可变的，所以不涉及任何缓存。

## 在自定义对象上使用 id()

我们还可以在自定义对象上使用 id()函数。

让我们举一个简单的例子:

```py
class Student():
    def __init__(self, name, id):
        self.name = name
        self.id = id

s = Student('Amit', 10)
t = Student('Rahul', 20)

print(id(s))

print(id(t))

```

**输出**

```py
2775665179336
2775665179448

```

这指的是存储对象的内存地址，这对于两个实例来说显然是不同的！

* * *

## 结论

在本文中，我们学习了在 Python 中使用 id()函数。这表示 Python 对象的底层内存地址，这在缓存不可变对象时很有用。

## 参考

*   关于 Python id()函数的 JournalDev 文章

* * *