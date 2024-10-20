# Python 字符串到字节，字节到字符串

> 原文：<https://www.askpython.com/python/string/python-string-bytes-conversion>

在本文中，我们将了解一下 **[Python 字符串](https://www.askpython.com/python/string/python-string-functions)到字节以及 Python 字节到字符串**的转换。类型的 Python 转换已经获得了相当大的重要性，因为它的数据特征是在各种操作中以不同的形式被使用。

字符串到字节和字节到字符串的 Python 转换有其自身的重要性，因为它在文件处理等时是必需的。

## Python 字符串到字节

以下任一方法都可用于将 Python 字符串转换为字节:

*   使用`bytes()`方法
*   使用`encode()`方法

### 1.使用 bytes()方法将 Python 字符串转换为字节

Python 的 **CPython 库**为我们提供了`bytes()`函数将字符串转换成字节。

**语法:**

```py
bytes(input_string, 'utf-8')

```

**注**:**UTF-8**格式用于编码目的。

**举例:**

```py
inp = "Engineering Discipline"

print("Input String:\n")
print(str(inp))

opt = bytes(inp, 'utf-8') 

print("String after getting converted to bytes:\n")
print(str(opt))
print(str(type(opt)))

```

**输出:**

```py
Input String:

Engineering Discipline
String after getting converted to bytes:

b'Engineering Discipline'
<class 'bytes'>

```

* * *

### 2.使用 encode()方法将 Python 字符串转换为字节

Python 的 **`encode()`** 方法也可以用来将字符串转换成字节格式。

**语法:**

```py
input_string.encode('utf-8')

```

**举例:**

```py
inp = "Engineering Discipline"

print("Input String:\n")
print(str(inp))

opt = inp.encode('utf-8')

print("String after getting converted to bytes:\n")
print(str(opt))
print(str(type(opt)))

```

**输出:**

```py
Input String:

Engineering Discipline
String after getting converted to bytes:

b'Engineering Discipline'
<class 'bytes'>

```

* * *

## Python 字节到字符串

Python 的**字节类**内置了 **`decode()`** 方法将 Python 字节转换成字符串。

**语法**:

```py
string.decode('utf-8')

```

**举例:**

```py
inp = "Engineering Discipline"

print("Input String:\n")
print(str(inp))

opt = inp.encode('utf-8')

print("String after getting converted to bytes:\n")
print(str(opt))
print(str(type(opt)))

original = opt.decode('utf-8')
print("The decoded String i.e. byte to converted string:\n")
print(str(original))

```

在上面的例子中，我们已经使用 encode()方法将输入字符串转换为字节。之后，decode()方法将编码后的输入转换为原始字符串。

**输出:**

```py
Input String:

Engineering Discipline
String after getting converted to bytes:

b'Engineering Discipline'
<class 'bytes'>
The decoded String i.e. byte to converted string:

Engineering Discipline

```

### 熊猫字节到字符串

**[熊猫模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)** 获得了 **`Series.str.decode()`** 的方法将编码数据即字节格式的数据转换为字符串格式。

**语法:**

```py
input_string.decode(encoding = 'UTF-8')

```

**举例:**

```py
import pandas

inp = pandas.Series([b"b'Jim'", b"b'Jonny'", b"b'Shawn'"]) 

print("Encoded String:")
print(inp) 

opt = inp.str.decode(encoding = 'UTF-8') 
print("\n")
print("Decoded String:")
print(opt) 

```

在上面的例子中，我们假设数据是编码格式的。此外，对数据执行操作。

**输出:**

```py
Encoded String:
0    b"b'Jim'"
1    b"b'Jonny'"
2    b"b'Shawn'"
dtype: object

Decoded String:
0    b'Jim'
1    b'Jonny'
2    b'Shawn'
dtype: object
​

```

* * *

## 结论

在本文中，我们已经理解了 Python 字符串到字节的转换，反之亦然，这也思考了编码和解码的概念。

* * *

## 参考

**Python 字符串到字节，字节到字符串–journal dev**