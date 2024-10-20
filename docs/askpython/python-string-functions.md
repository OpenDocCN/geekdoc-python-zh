# Python 字符串函数

> 原文：<https://www.askpython.com/python/string/python-string-functions>

Python 为几乎所有要在字符串上执行的操作都提供了内置函数。简单地说，这些是根据它们的使用频率和操作来分类的。它们如下:

## Python 字符串函数分类

*   基本功能
*   高级功能
*   杂项(这些函数不是专门针对字符串的，但是可以在字符串操作中使用)

* * *

## 基本字符串函数

| [大写()](https://www.askpython.com/python/string/python-string-capitalize-function) | 它将字符串的第一个字符转换为大写 | `str_name.capitalize()` |
| casefold() | 它将任何字符串转换为小写，而不考虑其大小写 | `str_name.casefold()` |
| 居中() | 它用于将字符串居中对齐 | `str_name.center(Length,character)` |
| 计数() | 它用于计算特定值在字符串中出现的次数 | `str_name.count(value,start,end)` |
| endswith() | 它检查字符串是否以指定的值结尾，然后返回 True | `str_name.endswith(value,start,end)` |
| 查找() | 它用于查找字符串中指定值的存在 | `str_name.find(value,start,end)` |
| 索引() | 它用于查找字符串中指定值的第一个匹配项 | `str_name.index(value,start,end)` |
| isalnum() | 它检查是否所有的字符都是字母数字，然后返回真 | `str_name.isalnum()` |
| isalpha() | 它检查是否所有的字符都是字母(a-z ),然后返回 True | `str_name.isalpha()` |
| 十进制格式() | 它检查所有字符是否都是小数(0-9)，然后返回 True | `str_name.isdecimal()` |
| isdigit() | 它检查是否所有的字符都是数字，然后返回真 | `str_name.isdigit()` |
| islower() | 它检查是否所有的字符都是小写的，然后返回 True | `str_name.islower()` |
| isnumeric() | 它检查是否所有的字符都是数字(0-9)，然后返回 True | `str_name.isnumeric()` |
| isspace() | 它检查是否所有字符都是空白，然后返回 True | `str_name.isspace()` |
| isupper() | 它检查是否所有的字符都是大写的，然后返回 True | `str_name.isupper()` |
| 下部() | 它用于将所有字符转换成小写 | `str_name.lower()` |
| 分区() | 它用于将字符串拆分成一个由三个元素组成的元组 | `str_name.partition(value)` |
| 替换() | 它用于将指定的单词或短语替换为字符串中的另一个单词或短语 | `str_name.replace(oldvalue,newvalue,count)` |
| 拆分() | 它用于将一个字符串拆分成一个列表 | `str_name.split(separator,maxsplit)` |
| 分割线() | 它用于拆分字符串并制作一个列表。在换行符处拆分。 | `str_name.splitlines(keeplinebreaks)` |
| 开始于() | 它检查字符串是否以指定的值开始，然后返回 True | `str_name.startswith(value,start,end)` |
| 条状() | 它用于从两端删除参数中指定的字符 | `str_name.strip(characters`) |
| 交换情况() | 它用于将大写字符串转换成小写字符串，反之亦然 | `str_name.swapcase()` |
| 标题() | 它将每个单词的首字母转换成大写 | `str_name.title()` |
| 上部() | 它用于将字符串中的所有字符转换为大写 | `str_name.upper()` |

* * *

## 高级 Python 字符串函数

| 编码() | 它用于返回编码字符串 | `str_name.encode(encoding=*encoding*, errors=*errors*)` |
| expandtabs() | 它用于设置或固定字符或字母之间的制表符间距 | `str_name.expandtabs(tabsize)` |
| 格式() | 它用执行时的值替换在{}中写入的变量名 | `str_name.format(value1,value2...)` |
| format_map() | 它用于格式化给定的字符串并返回 | `str_name.format_map(mapping)` |
| isidentifier() | 它检查字符是字母数字(a-z)和(0-9)还是下划线(_)并返回 True | `str_name.isidentifier()` |
| isprintable() | 它检查是否所有字符都是可打印的，然后返回 True | `str_name.isprintable()` |
| istitle() | 它检查单词的所有首字母是否都是大写的，然后返回 True | `str_name.istitle()` |
| 加入() | 它接受单词作为 iterable，并将它们连接成一个字符串 | `str_name.join(iterable)` |
| 光源() | 它返回一个左对齐的字符串，其最小值作为宽度给出 | `str_name.ljust(length,character)` |
| lstrip() | 它根据给定的参数从左端移除字符 | `str_name.lstrip(characters)` |
| maketrans() | 它创建一个可用于翻译的映射表 | `str_name.maketrans(x,y,z)` |
| rsplit() | 它用于从右端分割字符串 | `str_name.rsplit(separator,maxsplit)` |
| rfind() | 它搜索指定的值，并找到其最后一个值的位置 | `str_name.rfind(value,start,end)` |
| rindex() | 它搜索指定的值，并找到其最后一个值的位置 | `str_name.rindex(value,start,end)` |
| rjust() | 它返回一个右对齐的字符串，其最小值作为宽度给出 | `str_name.rjust(length,character)` |
| rpartition() | 它查找指定字符串的最后一个匹配项，并将该字符串拆分为由三个元素组成的元组 | `str_name.rpartition(value)` |
| rstrip() | 它根据给定的参数从右端移除字符 | `str_name.rstrip(characters)` |
| 翻译() | 它用于获取翻译后的字符串 | `str_name.translate(table)` |
| zfill() | 它返回一个新字符串，在该字符串的左边填充了“0”字符 | `str_name.zfill(len)` |

* * *

## 处理字符串的各种函数

| ascii() | 它返回包含对象的可打印形式的字符串，并忽略字符串中的非 ASCII 值 | `ascii(object)` |
| 布尔() | 它返回布尔值，即对象的真或假 | `bool(value)` |
| bytearray() | 它返回一个对象，该对象包含通过输入提供的字节数组 | `bytearray(source,encoding,errors)` |
| 字节() | 它返回不可修改的字节对象，是一个范围从 0 到 255 的整数序列 | `bytes(source,encoding,errors)` |
| 枚举() | 它用于向 iterable 添加计数器，然后返回它的值 | `enumerate(iterable,start=0)` |
| 浮动() | 它从给定的参数中返回浮点数 | `float(argument)` |
| 哈希() | 如果适用，它返回对象的哈希值 | `hash(object)` |
| id() | 它返回一个对象的特定标识，它是一个唯一的整数 | `id(object)` |
| int() | 它从给定的输入中返回一个整数对象，并且返回对象的基数总是 10 | `int(x=0,base=10)` |
| len() | 它返回序列的长度，即对象中的项目数 | `len(sequence)` |
| 地图() | 它用于将给定的函数应用于 iterable 的每一项，iterable 可以是元组、列表等。并且还返回包含结果值的列表 | `map(function, iterable, ...)` |
| 订单() | 它接受单个 Unicode 字符的字符串参数，并返回其对应的 Unicode 点 | `ord(character)` |
| 打印() | 它将提供的对象打印到任何输出设备 | `print*(object(s)*,separator=*separator*, end=*end*,file=*file*,flush=*flush*)` |
| 切片() | 它创建一个对象，该对象表示由其范围(开始、停止、步进)指定的一组索引 | `slice(stop)
slice(start,stop,step)` |
| 类型() | 它返回对象的类型 | `type(object)
type(name,bases,dict)` |

* * *

## 参考

[Python 官方文档](https://docs.python.org/2.4/lib/string-methods.html)