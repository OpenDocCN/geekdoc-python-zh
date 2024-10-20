# 了解 Python f 字符串

> 原文：<https://www.askpython.com/python/string/python-f-string>

Python 中的字符串格式化可以用 f-string 实现。因此，在本文中，我们将关注 Python f-string 的实现。

* * *

## Python 中 f-string 的必要性

Python f-string 基本上服务于字符串格式化的目的。在“f-strings”出现之前，我们有以下几种在 Python 中格式化字符串的方法:

**1。****Python“%”运算符** —

**缺点** : Python `% operator`不能和对象、属性一起使用。

**2。** Python format()函数—

**缺点**:`string.format() function`可以克服“%”操作符的缺点，但是它被证明是一种**冗长的**格式。

因此， **Python f-string** 应运而生，其中**可以用更简单和最少的语法**对字符串进行插值和格式化。Python 解释器在运行时格式化**字符串**。

* * *

## Python f-string 的使用示例

f 字符串也称为格式化字符串，用于**文字字符串插值**，即注入字符串并格式化特定字符串。

**语法:**

```py
f'{string}'

```

**例子:** **f-string 以 string 为可迭代**

```py
str1 = 'Python'

str4 = 'JournalDev'

res = f'{str4} provides tutorials on {str1}'

print(res)

```

如上所述，f-string 用于在字符串语句之间**注入**或**插入**输入字符串 **str1** 和 **str4** 。

**输出:**

```py
JournalDev provides tutorials on Python

```

* * *

### 带有原始字符串的 Python f 字符串

Python 原始字符串基本上将被视为**‘转义序列’的特殊字符视为文字字符**。当我们希望转义序列，即 **'\n'** 或**反斜杠(\)** 作为字符的文字序列时，就使用它。

**语法:Python 原始字符串**

```py
r'string'

```

Python f 字符串可以很好地与原始字符串同时工作。

**语法:f 字符串和原始字符串**

```py
fr'string or {string}'

```

**举例:**

```py
str1 = 'Python'

str4 = 'JournalDev'

res = fr'{str4} \n and AskPython provides tutorials on {str1}'

print(res)      

```

在上面的示例中，' \n '被视为文字字符。

**输出:**

```py
JournalDev \n and AskPython provides tutorials on Python

```

* * *

### 用 f 字符串调用函数

Python f-strings 使我们能够在其中调用[函数](https://www.askpython.com/python/python-functions)。因此，在一定程度上优化了代码。同样的方法可以用于在 f 字符串分支中创建 lambda 函数。

**语法:**

```py
f'{func()}'

```

**举例:**

```py
def mult(a,b):
    res = a*b
    return res

mult_res = f'{mult(10,20)}'
print(mult_res)

```

**输出:**

```py
200

```

* * *

### 带空白/空白的 Python f 字符串

Python f-strings 也可以处理空格或空白字符。它**忽略了尾随和前导空格**和文本字符串之间的**空格不变**和**保留**。

**举例:**

```py
mult_res = f'  Multiplication: { 10 * 10 }  '
print(mult_res)

```

**输出:**

```py
  Multiplication: 100  

```

* * *

### 带表达式的 Python f 字符串

Python f-string 可以处理**表达式**。因此，基本操作可以直接在 f 弦中执行。

**语法:**

```py
f'{expression'}

```

**举例:**

```py
x = 10
y = 5
print(f'Result: {x/y} ')

```

**输出:**

```py
Result: 2.0 

```

* * *

### 带有 Python 字典的 f 字符串

众所周知， [Python 字典](https://www.askpython.com/python/dictionary)数据结构使用键值对。Python f-string 也可以和字典一起放入框架中。

**语法:**

```py
f"{dict['key']}"

```

**举例:**

```py
info = {'age':'21', 'city':'Pune'}

print(f"{info['city']}")

```

**输出:**

```py
Pune

```

* * *

## 结论

因此，在本文中，我们已经理解了 f-string 使用各种可重复项和表达式的必要性和工作原理。

* * *

## 参考

*   Python f 字符串— JournalDev
*   [字符串插值— PEP 498](https://peps.python.org/pep-0498/)