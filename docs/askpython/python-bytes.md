# Python 字节()

> 原文：<https://www.askpython.com/python/built-in-methods/python-bytes>

Python **bytes()** 是一个内置函数，它返回一个 bytes 对象，该对象是一个不可变的整数序列，范围为 0 < = x < 256。根据作为源传递的对象类型，它相应地初始化 byte 对象。

让我们看看如何在本文中使用这个函数。

* * *

## 句法

这需要三个可选参数，即:

*   `source` - >初始化字节数组的源
*   `encoding`->`source`字符串的编码(可以是**-8**等)。
*   `errors` - >对源字符串编码失败时函数的行为。

```py
byte_array = bytes(source, encoding, errors)

```

因为这三个参数都是可选的，所以我们可以传递一个空字符串来生成一个空字节数组(大小为 0 的字节数组)。

根据`source`参数的类型，适当的字节数组将被初始化。

*   如果`source`是一个字符串，Python **bytes()** 会用`[str.encode()](https://www.askpython.com/python/string/python-encode-and-decode-functions)`把字符串转换成字节。因此，我们还必须提供**编码**和可选的**错误**，因为`encode()`正被用于处理字符串。
*   如果`source`是一个整数，Python **bytes()** 会创建一个给定整数大小的数组，全部初始化为 *NULL* 。
*   如果`source`属于`Object`类，对象的只读缓冲区将用于初始化字节数组。
*   如果`source`是可迭代的，那么*必定*是 0 < = x < 256 范围内的整数的可迭代的，这些整数作为数组的初始内容。

如果`source`是`None`，这将给出一个`TypeError`，因为它不能将一个`None`对象转换成一个字节数组。

为了更好地理解该函数，我们来看一些例子。

* * *

## 使用 Python 字节()

### 没有和没有参数

```py
b = bytes()
print(b)
c = bytes(None)
print(c)

```

**输出**

```py
b''
TypeError: cannot convert 'NoneType' object to bytes

```

### 使用源字符串

任何没有编码的字符串都会引发一个`TypeError`。

类似地，试图修改`bytes`对象也会给出相同的异常，因为它本质上是不可变的。

```py
try:
    a = bytes('Hello from AskPython')
except TypeError:
    print('We need to specify string encoding always!')

b = bytes('Hello from AskPython', 'UTF-8')
print(type(b), b)

try:
    b[0] = 10
except TypeError:
    print('byte objects are immutable!')

```

**输出**

```py
We need to specify string encoding always!
<class 'bytes'> b'Hello from AskPython'
byte objects are immutable!

```

### 使用源整数

整数零初始化数组中的字节元素对象。

```py
a = bytes(10)
print(type(a), a)

```

**输出**

```py
<class 'bytes'> b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

```

如您所见，bytes 对象是一个由 10 个元素组成的零初始化数组。

### 源是可迭代的

这用`len(iterable)`个元素初始化数组，每个元素的值都等于 iterable 上相应的元素。

字节数组值可以通过普通迭代访问，但不能修改，因为它们是不可变的。

```py
a = bytes([1, 2, 3])
print(type(a), a)
print('Length =', len(a))

# To access the byte array values, we can iterate through it!
for byte_obj in a:
    print(byte_object)

```

**输出**

```py
<class 'bytes'> b'\x01\x02\x03'
Length = 3
1
2
3

```

iterable 上的任何其他内容都将导致一个`TypeError`

```py
>>> a = bytes([1, 2, 3, 'Hi'])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'str' object cannot be interpreted as an integer

```

* * *

## 结论

在本文中，我们学习了 Python **bytes()** 函数，它可以将合适的对象转换成字节数组。

* * *

## 参考

*   字节数为()的 JournalDev 文章

* * *