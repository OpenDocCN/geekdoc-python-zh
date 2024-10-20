# Python 中的字符串连接

> 原文：<https://www.askpython.com/python/string/string-concatenation-in-python>

串联是一种将字符串连接在一起形成一个新字符串的方法。

以下是在 Python 中执行字符串串联的方法:

1.  通过使用`+`运算符
2.  通过使用`join()`方法
3.  通过使用`%`运算符
4.  通过使用`format()`功能
5.  由`Literal String Interpolation`
6.  通过使用`IO Module`中的`StringIO`
7.  通过使用 **`+=`** 连接运算符

* * *

## 1.使用+运算符的字符串串联

**语法** : string1 + string2

```py
str1 = 'Safa'
str2 = 'Mulani'

result = str1 + str2

print(result)

```

**输出** : SafaMulani

* * *

## 2.使用 join()方法连接 Python 字符串

**语法** : join(string1，string2)

```py
str1 = 'Safa'
str2 = 'Mulani'

print(" ".join([str1, str2]))

```

**输出**:萨法·穆拉尼

* * *

## 3.使用%运算符的字符串串联

**语法** : %(string1，string2)

```py
str1 = 'Safa'
str2 = 'Mulani'
result = "%s %s" % (str1, str2)
print('Concatenated String =', result)

```

**输出**:串接字符串= Safa Mulani

* * *

## 4.使用 format()函数连接字符串

**语法**:格式(string1，string2)

```py
str1 = 'Safa'
str2 = 'Mulani'
res = "{} {}".format(str1, str2)
print('Concatenated String=', res)

```

**输出**:串接字符串= Safa Mulani

* * *

## 5.使用文字字符串插值的字符串连接

Python 3.6+版本允许我们将 f-string 用于文字字符串插值中引入的字符串连接。

**语法**:f“{ string 1 } { string 2 }”

```py
str1 = 'Safa'
str2 = 'Mulani'
res = f'{str1} {str2}'
print('Concatenated String =', res)

```

**输出**:串接字符串= Safa Mulani

* * *

## 6.使用 IO 模块中的 StringIO 连接字符串

```py
from io import StringIO

result = StringIO()

result.write('Safa ')

result.write('Mulani ')

result.write('Engineering ')

print(result.getvalue())

```

**输出** : Safa Mulani 工程

* * *

## 7.使用+=连接运算符

**语法** : string1 += string2

```py
str1 = 'Safa '

str2 = 'Mulani'

str1 += str2

print(str1)

```

**输出**:萨法·穆拉尼

* * *

## 参考

*   Python 字符串串联
*   Python 运算符