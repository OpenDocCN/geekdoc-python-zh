# Python 中的 Unicode 解释了 unicodedata 模块

> 原文：<https://www.askpython.com/python-modules/unicode-in-python-unicodedata>

嘿伙计们！在本教程中，我们将学习 Python 中的 Unicode 以及 Unicode 的字符属性。那么，我们开始吧。

## 什么是 Unicode？

Unicode 将每个字符和符号与一个称为代码点的唯一数字相关联。它支持世界上所有的书写系统，并确保可以使用任何语言组合来检索或组合数据。

码点是十六进制编码中范围从 0 到 0x10FFFF 的整数值。

要开始在 Python 中使用 Unicode 字符，我们需要理解字符串模块是如何解释字符的。

## Python 中如何解释 ASCII 和 Unicode？

Python 为我们提供了一个 *string* 模块，其中包含了各种操作字符串的函数和工具。它属于 ASCII 字符集。

```py
import string

print(string.ascii_lowercase) 
print(string.ascii_uppercase)
print(string.ascii_letters)
print(string.digits)
print(string.hexdigits)
print(string.octdigits)
print(string.whitespace)  
print(string.punctuation)

```

输出:

```py
ABCDEFGHIJKLMNOPQRSTUVWXYZ
abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
0123456789
0123456789abcdefABCDEF
01234567

!"#$%&'()*+,-./:;<=>[email protected][\]^_`{|}~

```

我们可以使用 **chr()** 内置函数创建单字符 Unicode 字符串。它只接受一个整数作为参数，并返回给定字符的 unicode。

类似地，odr()是一个内置函数，它接受一个单字符 Unicode 字符串作为输入，并返回代码点值。

```py
chr(57344)
ord('\ue000')

```

输出:

```py
'\ue000'
57344

```

## Python 中的字符编码是什么意思？

字符串是 Unicode 码位的序列。这些码点被转换成字节序列，以便有效存储。这个过程叫做字符编码。

有许多编码，如 UTF 8，UTF-16，ASCII 等。

默认情况下，Python 使用 UTF-8 编码。

## 什么是 UTF 8 编码？

UTF 8 是最流行和最常用的字符编码。UTF 代表 *Unicode 转换格式*，‘8’表示编码中使用了*的 8 位值*。

它取代了 ASCII(美国信息交换标准码),因为它提供了更多的字符，可以用于世界各地的不同语言，不像 ASCII 只限于拉丁语言。

UTF 8 字符集的前 128 个码位也是有效的 ASCII 字符。UTF-8 中的字符长度可以是 1 到 4 个字节。

## 使用 Python encode()函数对 UTF 8 中的字符进行编码

[encode()方法](https://www.askpython.com/python/string/python-encode-and-decode-functions)将任何字符从一种编码转换成另一种编码。编码函数的语法如下所示

```py
string.encode(encoding='UTF-8',errors='strict')

```

**参数**:

*   ***编码*** 是 python 支持使用的编码。
*   ***错误***–不同错误类型列表如下

1.  **strict-** 默认错误为 *strict* ，失败时引发 UnicodeDecode 错误。
2.  **忽略**–忽略结果中不可解码的 unicode。
3.  **替换**–用“？”替换不可解码的 unicode
4.  **xmlcharrefreplace-** 插入 xlm 字符引用来代替不可解码的 unicode。
5.  **backslashreplace-** 插入\uNNNN 转义序列来代替不可解码的 unicode。
6.  **namereplace-** 在不可解码的 unicode 位置插入\N{…}转义序列。

## 如何在 Python 中通过 encode()函数使用 Unicode？

现在让我们开始理解字符串编码函数如何允许我们在 Python 中创建 unicode 字符串。

### 1.将字符串编码为 UTF-8 编码

```py
string = 'örange'
print('The string is:',string)
string_utf=string.encode()
print('The encoded string is:',string_utf)

```

**输出:**

```py
The string is: örange
The encoded string is: b'\xc3\xb6range'

```

### 2.带错误参数的编码

让我们对德语单词 wei 进行编码，它的意思是白色。

```py
string = 'weiß'

x = string.encode(encoding='ascii',errors='backslashreplace')
print(x)

x = string.encode(encoding='ascii',errors='ignore')
print(x)

x = string.encode(encoding='ascii',errors='namereplace')
print(x)

x = string.encode(encoding='ascii',errors='replace')
print(x)

x = string.encode(encoding='ascii',errors='xmlcharrefreplace')
print(x)

x = string.encode(encoding='UTF-8',errors='strict')
print(x)

```

输出:

```py
b'wei\\xdf'
b'wei'
b'wei\\N{LATIN SMALL LETTER SHARP S}'
b'wei?'
b'weiß'
b'wei\xc3\x9f'

```

## 在 python 中使用 unicode 的 unicodedata 模块

***unicodedata*** 模块为我们提供了 ***Unicode 字符数据库(UCD)*** ，它定义了所有 Unicode 字符的所有字符属性。

让我们看看模块中定义的所有函数，并用一个简单的例子来解释它们的功能。通过使用以下函数，我们可以在 Python 中高效地使用 Unicode。

### 1. **unicodedata.lookup(名称)**

这个函数根据给定的名字查找字符。如果找到该字符，则返回相应的字符。如果找不到，则引发 Keyerror。

```py
import unicodedata 

print (unicodedata.lookup('LEFT CURLY BRACKET')) 
print (unicodedata.lookup('RIGHT SQUARE BRACKET')) 
print (unicodedata.lookup('ASTERISK'))
print (unicodedata.lookup('EXCLAMATION MARK'))

```

输出:

```py
{
]
*
!

```

### 2. **unicodedata.name(chr[，default])**

该函数以字符串形式返回分配给字符 *chr* 的名称。如果没有定义名称，它将返回默认值，否则将引发 Keyerror。

```py
import unicodedata 

print (unicodedata.name(u'%')) 
print (unicodedata.name(u'|')) 
print (unicodedata.name(u'*')) 
print (unicodedata.name(u'@'))

```

输出:

```py
PERCENT SIGN
VERTICAL LINE
ASTERISK
COMMERCIAL AT

```

### 3. **unicodedata.decimal(chr[，default])**

该函数返回分配给字符 *chr* 的十进制值。如果没有定义值，则返回默认值，否则将引发 Keyerror，如下例所示。

```py
import unicodedata

print (unicodedata.decimal(u'6'))
print (unicodedata.decimal(u'b')) 

```

输出:

```py
6
Traceback (most recent call last):
  File "D:\DSCracker\DS Cracker\program.py", line 4, in <module>
    print (unicodedata.decimal(u'b')) 
ValueError: not a decimal

```

### 4. **unicodedata.digit(chr[，default])**

该函数将分配给字符 *chr* 的数字值作为整数返回。需要注意的一点是，这个函数接受单个字符作为输入。在本例的最后一行，我使用了“20 ”,函数抛出一个错误，指出它不能接受一个字符串作为输入。

```py
import unicodedata 

print (unicodedata.decimal(u'9')) 
print (unicodedata.decimal(u'0')) 
print (unicodedata.decimal(u'20'))

```

输出:

```py
9
0
Traceback (most recent call last):
  File "D:\DSCracker\DS Cracker\program.py", line 5, in <module>
    print (unicodedata.decimal(u'20'))
TypeError: decimal() argument 1 must be a unicode character, not str

```

### 5. **unicodedata.numeric(chr[，default])**

该函数返回分配给字符 *chr* 的整数数值。如果没有定义值，则返回默认值，否则将引发 ValueError。

```py
import unicodedata 

print (unicodedata.decimal(u'1'))
print (unicodedata.decimal(u'8'))
print (unicodedata.decimal(u'123'))

```

输出:

```py
1
8
Traceback (most recent call last):
  File "D:\DSCracker\DS Cracker\program.py", line 5, in <module>
    print (unicodedata.decimal(u'123')) 
TypeError: decimal() argument 1 must be a unicode character, not str

```

### 6.**unicode data . category(chr)**

该函数以字符串形式返回分配给角色 *chr* 的一般类别。它返回字母“L ”,大写字母“u ”,小写字母“L”。

```py
import unicodedata 

print (unicodedata.category(u'P')) 
print (unicodedata.category(u'p')) 

```

输出:

```py
Lu
Ll

```

### 7. **unicodedata .双向(chr)**

该函数以字符串形式返回分配给字符 chr 的双向类。如果没有定义这样的值，此函数将返回一个空字符串。

AL 表示阿拉伯字母，AN 表示阿拉伯数字，L 表示从左到右等等。

```py
import unicodedata 

print (unicodedata.bidirectional(u'\u0760'))

print (unicodedata.bidirectional(u'\u0560')) 

print (unicodedata.bidirectional(u'\u0660')) 

```

输出:

```py
AL
L
AN

```

### 8.**unicode data . combining(chr)**

这个函数以字符串形式返回分配给给定字符 *chr* 的规范组合类。如果没有定义组合类，则返回 0。

```py
import unicodedata 

print (unicodedata.combining(u"\u0317"))

```

输出:

```py
220

```

### 9.unicodedata.mirrored(chr)

这个函数以整数的形式返回一个分配给给定角色 *chr* 的*镜像*属性。如果字符在双向文本中被识别为'*镜像*，则返回 *1* ，否则返回 *0* 。

```py
import unicodedata 

print (unicodedata.mirrored(u"\u0028"))
print (unicodedata.mirrored(u"\u0578"))

```

输出:

```py
1
0

```

### 10. **unicodedata.normalize(form，unistr)**

使用此函数返回 Unicode 字符串 unistr 的常规形式。格式的有效值为“NFC”、“NFKC”、“NFD”和“NFKD”。

```py
from unicodedata import normalize 

print ('%r' % normalize('NFD', u'\u00C6')) 
print ('%r' % normalize('NFC', u'C\u0367')) 
print ('%r' % normalize('NFKD', u'\u2760')) 

```

输出:

```py
'Æ'
'Cͧ'
'❠'

```

## 结论

在本教程中，我们学习了 unicode 和定义 unicode 特征的 unicodedatabase 模块。希望你们都喜欢。敬请关注🙂

## 参考

[Unicode 官方文档](https://docs.python.org/3/howto/unicode.html#:~:text=Python's%20string%20type%20uses%20the,character%20its%20own%20unique%20code.)

[单播码数据库](https://docs.python.org/3/library/unicodedata.html)