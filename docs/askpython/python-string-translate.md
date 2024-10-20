# Python 字符串翻译()

> 原文：<https://www.askpython.com/python/string/python-string-translate>

Python 的[字符串](https://www.askpython.com/python/string)类`str.translate()`的内置方法用于将字符串映射到翻译表。 **translate()** 方法只是使用转换表规则对其进行转换。

让我们进一步了解本文中的方法。

* * *

## Python 字符串翻译()的语法

由于 **translate()** 函数返回一个新的字符串，字符串中的每个字符都用给定的翻译表替换，所以它的输入是一个字符串和一个表，返回一个字符串。

但是，由于这是一个属于`str`类的方法，我们只使用输入字符串对象调用它们。

```py
out_str = in_str.translate(translation_table)

```

这里，`translation_table`是包含相关映射的翻译表。

虽然我们知道什么是翻译表，但我们还不知道它是如何构造的。让我们也理解这一点。

* * *

### 使用 maketrans()构造转换表

我们可以使用`maketrans()`方法制作转换表。

因为任何映射都可以作为翻译表，所以现有的 [Python 字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)也可以成为翻译表。

#### **maketrans():** 的语法

我们总是使用`str.maketrans()`调用 maketrans()，其中`str`是表示 Python 的 string 类的内置关键字。

如果只有一个参数，它必须是一个字典映射(`dict_map` ) Unicode 序数(整数)或字符(长度为 1 的字符串)到 Unicode 序数、字符串(任意长度)或`None`。字符键将被转换成序数。

```py
translation = str.maketrans(dict_map)

```

如果有两个参数，它们必须是等长的字符串(`str_x`和`str_y`)，在结果字典中，x 中的每个字符都会映射到 y 中相同位置的字符。

```py
translation = str.maketrans(str_x, str_y)

```

如果有第三个参数，它必须是一个字符串(`str_z`)，其字符将被映射到结果中的`None`。

```py
translation = str.maketrans(str_x, str_y, str_z)

```

* * *

### 进行翻译

在使用 **maketrans()** 构建转换表之后，我们现在可以使用 **python string translate()** 执行从输入字符串到输出字符串的转换。

让我们举例说明使用`maketrans()`的所有三种方式，让你的理解更加完整。

#### 使用字典映射

```py
# Input string
inp_str = 'Hello from AskPython'

# Create a Dictionary for the mapping
dict_map = dict()

# We shift every letter from A-z by 9 units
# using ord() to get their ASCII values
for i in range(ord('A'), ord('z')):
    dict_map[chr(i)] = chr(i + 9)

# Convert the dictionary mapping to a translation table
translation = str.maketrans(dict_map)

# Perform the translation
out_str = inp_str.translate(translation)

print(out_str)

```

输出将是包含右移 9 位的每个字符的字符串。

**输出**:

```py
Qnuux o{xv J|tYA}qxw

```

* * *

#### 使用直接映射

下面是一个更简单的例子，它利用直接映射并使用两个参数构造一个转换表，然后将它们传递给输入字符串 translate()函数。

```py
s = 'ABCDBCA'

translation = s.maketrans('A', 'a')
print(s.translate(translation))

translation = s.maketrans('ABCD', 'abcd')
print(s.translate(translation))

```

这会将第一个' *A* 字符翻译成' **a** ，然后将' *ABCD* 转换成' **abcd** ，如下所示。

**输出**:

```py
aBCDBCa
abcdbca

```

同样，由于该选项指定两个字符串的长度相同，任何其他操作都会引发一个`ValueError`异常。

```py
s = 'ABCDBCA'

translation = s.maketrans('Ab', 'a')
print(s.translate(translation))

```

**输出**:

```py
ValueError: the first two maketrans arguments must have equal length

```

* * *

#### 使用无映射

下面是一个更简单的例子，它通过使用 3 个参数调用`maketrans()`来利用`None`映射。

```py
# Input string
inp_str = 'Hello from AskPython'

translation = str.maketrans('As', 'ab', 'e')

# Perform the translation
out_str = inp_str.translate(translation)

print(out_str)

```

这将把‘As’替换为‘ab’，同时把‘e’替换为`None`，有效地从字符串中删除该字符。

**输出**:

```py
Hllo from abkPython

```

* * *

#### 直接构造一个字典映射

我们可以在调用函数时构造一个直接映射到`maketrans()`的字典。

```py
# Input string
inp_str = 'Hello from AskPython'

translation = str.maketrans({ord('A'): 'a', ord('s'): 'b', ord('e'): None})

# Perform the translation
out_str = inp_str.translate(translation)

print(out_str)

```

这将给出与前面代码片段相同的输出，因为它们有相同的翻译表。

**输出**:

```py
Hllo from abkPython

```

**注意**:我们可以将一个 Unicode 值(映射字典中的`ord()`键)映射到任意字符串的*，这与将字符串作为参数传递给`maketrans()`不同。因此，像`{ord('A'): 'hello'}`这样的映射是有效的，并且用“ *hello* ”替换“ **A** ”。*

* * *

## 结论

在本文中，我们学习了如何使用 python string translate()方法基于 python 中的翻译表执行字符串翻译。我们还研究了如何使用各种方法构建这样一个表。

## 参考

*   【maketrans()的 Python 文档
*   [Python 文档](https://docs.python.org/3/library/stdtypes.html#str.translate)用于翻译()
*   journal dev translate()上的文章

* * *