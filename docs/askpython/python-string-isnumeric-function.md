# Python 字符串是 numeric()函数

> 原文：<https://www.askpython.com/python/string/python-string-isnumeric-function>

如果发现输入字符串中的所有字符都是数字类型，Python String `**isnumeric()**`函数返回 **True** ，否则返回 **False** 。

数字字符可以是以下类型:

*   数字字符=十进制
*   数字 _ 字符=数字
*   数字 _ 字符=数字

**语法:**

```py
input_string.isnumeric()
```

**isnumeric()参数**:isnumeric()函数不接受任何参数作为输入。

* * *

## Python isnumeric()示例

**例 1:**

```py
string = '124678953'

print(string.isnumeric())

```

**输出:**

```py
True
```

**例 2:**

```py
string = 'Divas Dwivedi . 124678953'

print(string.isnumeric())

```

**输出:**

```py
False
```

**例 3:** 特殊 Unicode 字符作为输入字符串

```py
string1 = '\u00B23455'
print(string1)
print(string1.isnumeric())

```

**输出:**

```py
²3455
True
```

**例 4:**

```py
string = "000000000001" 
if string.isnumeric() == True:
    print("Numeric input")
else:  
    print("Not numeric")  
str = "000-9999-0110" 
if str.isnumeric() == True:
    print("Numeric input")
else:  
    print("Non numeric input")

```

**输出:**

```py
Numeric input
Non numeric input
```

* * *

## 访问所有 Unicode 数字字符

`**unicodedata**`模块用于获取所有的数字 Unicode 字符。

```py
import unicodedata

count_numeric = 0
for x in range(2 ** 16):
    str = chr(x)
    if str.isnumeric():
        print(u'{:04x}: {} ({})'.format(x, str, unicodedata.name(str, 'UNNAMED')))
        count_numeric = count_numeric + 1
print(f'Count of Numeric Unicode Characters = {count_numeric}')

```

**输出:**

```py
0030: 0 (DIGIT ZERO)
0031: 1 (DIGIT ONE)
0032: 2 (DIGIT TWO)
.....
ff15: ５ (FULLWIDTH DIGIT FIVE)
ff16: ６ (FULLWIDTH DIGIT SIX)
ff17: ７ (FULLWIDTH DIGIT SEVEN)
ff18: ８ (FULLWIDTH DIGIT EIGHT)
ff19: ９ (FULLWIDTH DIGIT NINE)
Count of Numeric Unicode Characters = 800
```

* * *

## 结论

因此，在本文中，我们研究并实现了 Python String 的 isnumeric()函数。

* * *

## 参考

*   Python 是一个数字函数
*   [Python 字符串文档](https://docs.python.org/3/library/string.html)