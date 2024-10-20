# 在 Python 中修剪字符串的 3 种方法

> 原文：<https://www.askpython.com/python/string/trim-a-string-in-python>

修剪一个字符串是什么意思，如何在 Python 中修剪一个字符串？当你删除文本字符串周围的空白时，用专业术语来说，我们称之为修剪字符串。在本文中，我们将介绍在 Python 中修剪字符串的 3 种方法。

* * *

## 技巧 1:在 Python 中修剪字符串的 strip()

Python 的`string.strip()`函数基本上删除了特定字符串中所有的**前导**以及**尾随空格**。因此，我们可以使用这个方法在 Python 中完全修剪一个字符串。

**语法:**

```py
string.strip(character)

```

*   `character`:是一个**可选参数**。如果传递给 strip()函数，它将从字符串的两端删除特定的传递字符。

**举例:**

```py
inp_str = "     [email protected]"
print("Input String:")
print(inp_str)
res = inp_str.strip()
print("\nString after trimming extra leading and trailing spaces:")
print(res)

```

**输出:**

```py
Input String:
     [email protected]

String after trimming extra leading and trailing spaces:
[email protected]

```

**例 2:**

```py
inp_str = "@@Python [email protected]@@@"
print("Input String:")
print(inp_str)
res = inp_str.strip('@')
print("\nString after trimming extra leading and trailing spaces:")
print(res)

```

在上面的代码片段中，我们将'**@【T1]'作为字符传递给 strip()函数，以便从两端进行修剪。**

**输出:**

```py
Input String:
@@Python [email protected]@@@

String after trimming extra leading and trailing spaces:
Python JournalDev

```

* * *

### NumPy strip()方法

[Python NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-arrays) 模块内置了`numpy.core.defchararray.strip()` 方法，其功能类似于 Python **string.strip()** 函数。

该方法可用于修剪 Python 中嵌入数组或任何其他可迭代对象的字符串。

**语法:**

```py
numpy.core.char.strip(array, chars=value)

```

*   `array`:需要进行修剪的输入数组。
*   `chars`:可选参数。如果传递给 numpy.strip()函数，则从数组每个元素的两端修剪特定字符。

**举例:**

```py
import numpy

arr = numpy.array([' JournalDev', 'Python  '])
print("Input Array:")
print(arr)
res = numpy.char.strip(arr)
print("Array after performing strip():")
print(res)

```

**输出:**

```py
Input Array:
[' JournalDev' 'Python  ']
Array after performing strip():
['JournalDev' 'Python']

```

**例 2:**

```py
import numpy

arr = numpy.array([' JournalDev', 'Python  '])
print("Input Array:")
print(arr)
res = numpy.char.strip(arr, chars='Python')
print("Array after performing strip():")
print(res)

```

**输出:**

```py
Input Array:
[' JournalDev' 'Python  ']
Array after performing strip():
[' JournalDev' '  ']

```

* * *

## 技术 2: Python lstrip()

Python `string.lstrip()`函数修剪特定输入字符串中的所有前导空格。

**语法:**

```py
string.lstrip(character)

```

*   `character`:是一个**可选参数**。如果传递给 lstrip()函数，它将从输入字符串的开头删除特定的传递字符。

**举例:**

```py
inp_str = "    [email protected]  **"
print("Input String:")
print(inp_str)
res = inp_str.lstrip()
print("\nString after trimming Extra leading spaces:")
print(res)

```

**输出:**

```py
Input String:
    [email protected]  **

String after trimming Extra leading spaces:
[email protected]  **

```

**例题** **2:**

```py
inp_str = "****[email protected]*"
print("Input String:")
print(inp_str)
res = inp_str.lstrip("*")
print("\nString after trimming Extra leading characters:")
print(res)

```

如上所示，lstrip()函数只从输入字符串的**前导部分**的前部开始修剪' ***** '。

**输出:**

```py
Input String:
****[email protected]*

String after trimming Extra leading characters:
[email protected]*

```

* * *

### NumPy lstrip()方法

Python NumPy 模块具有与`string.lstrip()`函数功能相同的`numpy.core.defchararray.lstrip()` 方法。

该函数**从输入数组的每个元素中删除所有前导空格或特殊字符**。

**语法**:

```py
numpy.char.lstrip(array, chars=value)

```

**例 1:**

```py
import numpy

arr = numpy.array(['@@!JournalDev', '@%*Python  '])
print("Input Array:")
print(arr)
res = numpy.char.lstrip(arr, chars="!%@*")
print("Array after performing lstrip():")
print(res)

```

**输出:**

```py
Input Array:
['@@!JournalDev' '@%*Python  ']
Array after performing lstrip():
['JournalDev' 'Python  ']

```

**例 2:**

```py
import numpy

arr = numpy.array(['  JournalDev', ' Python'])
print("Input Array:")
print(arr)
res = numpy.char.lstrip(arr)
print("Array after performing lstrip():")
print(res)

```

**输出:**

```py
Input Array:
['  JournalDev' ' Python']
Array after performing lstrip():
['JournalDev' 'Python']

```

* * *

## 技术 3: Python rstrip()

Python `string.rstrip()`方法从特定的输入字符串中删除所有的尾随空格。

**语法:**

```py
string.rstrip(character)

```

*   `character`:是一个**可选参数**。如果传递给 rstrip()函数，它将从输入字符串的末尾删除传递的字符。

**举例:**

```py
inp_str = "[email protected]   "
print("Input String:")
print(inp_str)
print("Length of Input String:")
print(len(inp_str))
res = inp_str.rstrip()
print("\nString after trimming Extra trailing spaces:")
print(res)
print("Length of Input String after removing extra trailing spaces:")
print(len(res))

```

我们已经使用了`string.len()`函数来获得修剪前后的字符串长度。这有助于我们理解末尾多余的空格已经被删除。

**输出:**

```py
Input String:
[email protected]   
Length of Input String:
20

String after trimming Extra trailing spaces:
[email protected]
Length of Input String after removing extra trailing spaces:
17

```

**例 2:**

```py
inp_str = "[email protected]****"
print("Input String:")
print(inp_str)
print("Length of Input String:")
print(len(inp_str))
res = inp_str.rstrip("*")
print("\nString after trimming Extra trailing characters:")
print(res)
print("Length of Input String after removing extra trailing spaces:")
print(len(res))

```

**输出:**

```py
Input String:
[email protected]****
Length of Input String:
21

String after trimming Extra trailing characters:
[email protected]
Length of Input String after removing extra trailing spaces:
17

```

* * *

### NumPy rstrip()方法

Python NumPy 模块有`numpy.core.defchararray.rstrip(array, chars)`方法从输入数组的每个元素中删除所有尾随空格。

**语法:**

```py
numpy.char.rstrip(array, chars=value)

```

**举例:**

```py
import numpy

arr = numpy.array(['  JournalDev  ', ' Python    '])
print("Input Array:")
print(arr)
res = numpy.char.rstrip(arr)
print("Array after performing rstrip():")
print(res)

```

**输出:**

```py
Input Array:
['  JournalDev  ' ' Python    ']
Array after performing rstrip():
['  JournalDev' ' Python']

```

**例 2:**

```py
import numpy

arr = numpy.array(['  JournalDev****', ' Python!!'])
print("Input Array:")
print(arr)
res = numpy.char.rstrip(arr, chars="*!")
print("Array after performing rstrip():")
print(res)

```

在上例中，我们已经通过了' ***！**到 **numpy.rstrip()** 的作用是作为要修剪的字符。这些字符从数组的每个元素的后端开始修剪。

**输出:**

```py
Input Array:
['  JournalDev****' ' Python!!']
Array after performing rstrip():
['  JournalDev' ' Python']

```

* * *

## Python 修剪字符串一目了然！

*   在 Python 中，修剪字符串意味着从输入字符串的开头和结尾删除多余的空格或特定的一组字符。
*   在 Python 中可以使用三个内置函数来修剪一个字符串:分别是: **strip()、lstrip()、rstrip()** 方法。
*   python**strip . strip()方法**从特定字符串的前端和后端删除空格。
*   **string.lstrip()方法**从字符串中移除所有前导空格。
*   **string.rstrip()方法**从一个字符串中删除所有的尾随空格。

* * *

## 结论

因此，在本文中，我们已经了解了在 Python 中修剪字符串的不同方法。

* * *

## 参考

*   Python 修剪字符串 JournalDev