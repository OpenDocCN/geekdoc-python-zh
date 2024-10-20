# Python 结构模块

> 原文：<https://www.askpython.com/python-modules/python-struct-module>

Python struct [模块](https://www.askpython.com/python-modules)用于提供一个简单的 Python 接口来访问和操作 C 的结构数据类型。如果您需要处理 C 代码，并且没有时间用 C 语言编写工具，因为它是一种低级语言，那么这可能是一个方便的工具。

这个模块可以将 Python 值转换成 C 结构，反之亦然。C 结构被用作 Python [bytes](https://www.askpython.com/python/built-in-methods/python-bytes) 对象，因为 C 中没有被称为对象的东西；只有字节大小的数据结构。

让我们来理解如何使用这个模块来拥有一个到 C 结构的 Python 接口。

* * *

## Python 结构模块方法

在本模块中，由于我们关注的是 C 结构，所以让我们来看看本模块为我们提供的一些功能。

### struct.pack()

这用于将元素打包成 Python 字节串(字节对象)。由于存储模式是基于字节的，基于 C 的程序可以使用来自 Python 程序的`pack()`的输出。

格式: **struct.pack(格式，v1，v2，…)**

`v1`、`v2`、…是将被打包到字节对象中的值。它们表示 C 结构的字段值。因为具有`n`字段的 C 结构必须正好具有`n`值，所以参数必须与格式要求的值完全匹配。

这里，`format`指的是包装的格式。这是必要的，因为我们需要指定字节串的数据类型，因为它用于 C 代码。下表列出了`format`最常见的值。我们需要每个值一种格式来指定它的数据类型。

| **格式** | **C Datatype** | **Python 类型** |
| `c` | 茶 | 长度为 1 的字符串 |
| `?` | _Bool | 弯曲件 |
| `h` | 短的 | 整数 |
| `l` | 长的 | 整数 |
| `i` | （同 Internationalorganizations）国际组织 | 整数 |
| `f` | 漂浮物 | 漂浮物 |
| `d` | 两倍 | 漂浮物 |
| `s` | char[] | 线 |

让我们用一些例子来理解这一点。

下面的代码片段使用`pack()`将 3 个整数 1、2 和 3 存储在一个字节对象中。由于一个整数的大小在我的机器上是 4 字节，所以你看到 3 个 4 字节的块，对应 c 中的 3 个整数。

```py
import struct

# We pack 3 integers, so 'iii' is required
variable = struct.pack('iii', 1, 2, 3)
print(type(variable), variable)

variable_2 = struct.pack('iic', 1, 2, b'A')
print('\n', variable_2)

```

**输出**

```py
<class 'bytes'> b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00'
b'\x01\x00\x00\x00\x02\x00\x00\x00A'

```

如果没有传递适当的类型，Python struct 模块将引发异常`struct.error`。

```py
import struct

# Error!! Incorrect datatype assignment
variable = struct.pack('ccc', 1, 2, 3)
print(type(variable), variable)

```

**输出**

```py
struct.error: char format requires a bytes object of length 1

```

* * *

### struct.unpack()

Python struct 模块的这个函数根据适当的格式将打包的值解包为其原始表示。这将返回一个元组，其大小等于传递的值的数量，因为 byte 对象被解包以给出元素。

格式: **struct.unpack(格式，字符串)**

这将根据`format`格式说明符解包字节`string`。

这是`struct.pack()`的反转。让我们用它产生一个旧的字节字符串，并尝试用`unpack()`取回传递给它的 python 值。

```py
import struct

byte_str = b'\x01\x00\x00\x00\x02\x00\x00\x00A'

# Using the same format specifier as before, since
# we want to get Python values for the same byte-string
tuple_vals = struct.unpack('iic', byte_str)
print(tuple_vals)

```

**输出**

```py
(1, 2, b'A')

```

正如您所看到的，如果我们对`pack()`和`unpack()`使用相同的格式说明符，我们确实可以从这个元组中打包我们的旧 Python 值。

* * *

### struct.calcsize()

此函数使用给定的格式说明符返回结构的字符串表示形式的总大小，以检索数据的类型并计算大小。

格式: **struct.calcsize(fmt)**

```py
import struct

print('C Integer Size in Bytes:', struct.calcsize('i'))
print('Size of 3 characters in Bytes:', struct.calcsize('ccc'))

```

**输出**

```py
C Integer Size in Bytes: 4
Size of 3 characters in Bytes: 3

```

* * *

### **struct.pack_into()**

该函数用于将值打包到 Python 字符串缓冲区，可在`ctypes`模块中获得。

格式: **struct.pack_into(fmt，buffer，offset，v1，v2，…)**

这里，`fmt`指的是格式说明符，一如既往。`buffer`是字符串缓冲区，现在将包含指定的打包值。您还可以从基本地址中指定一个`offset`位置，从这里开始打包。

这不会返回任何值，只是将值存储到`buffer`字符串中。

```py
import struct 
import ctypes 

# We will create a string buffer having a size
# equal to that of a struct with 'iic' values.
buf_size = struct.calcsize('iic') 

# Create the string buffer
buff = ctypes.create_string_buffer(buf_size) 

# struct.pack() returns the packed data 
struct.pack_into('iic', buff, 0, 1, 2, b'A')

print(buff)

# Display the contents of the buffer
print(buff[:])

```

**输出**

```py
<ctypes.c_char_Array_9 object at 0x7f4bccef1040>
b'\x01\x00\x00\x00\x02\x00\x00\x00A'

```

事实上，我们在缓冲字符串中获得了打包的值。

* * *

### struct.unpack_from()

与`unpack()`类似，有一个对应项用于从缓冲字符串中解包值。这与`struct.pack_into()`相反。

格式: **struct.unpack_from(fmt，buffer，offset)**

这将返回一组值，类似于`struct.unpack()`。

```py
import struct 
import ctypes 

# We will create a string buffer having a size
# equal to that of a struct with 'iic' values.
buf_size = struct.calcsize('iic') 

# Create the string buffer
buff = ctypes.create_string_buffer(buf_size) 

# struct.pack() returns the packed data 
struct.pack_into('iic', buff, 0, 1, 2, b'A')

print(struct.unpack_from('iic', buff, 0))

```

**输出**

```py
(1, 2, b'A')

```

* * *

## 结论

在本文中，我们学习了如何使用 Python struct 模块来处理 C 类型的结构对象。

## 参考

*   关于 Python 结构模块的 JournalDev 文章

* * *