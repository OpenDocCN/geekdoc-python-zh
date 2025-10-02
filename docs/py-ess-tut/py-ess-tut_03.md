# 第三章 使用字符串

> 来源：[`www.cnblogs.com/Marlowes/p/5312236.html`](http://www.cnblogs.com/Marlowes/p/5312236.html)
> 
> 作者：Marlowes

读者已经知道了什么是字符串，也知道如何创建它们。利用索引和分片访问字符串中的单个字符也已经不在话下了。那么本章将会介绍如何使用字符串格式化其他的值(如打印特殊格式的字符串)，并简单了解一下利用字符串的分割、连接、搜索等方法能做些什么。

## 3.1 基本字符串操作

所有标准的序列操作(索引、分片、乘法、判断成员资格、求长度、取最小值和最大值)对字符串同样适用，上一章已经讲述了这些操作。但是，请记住字符串都是不可变的。因此，如下所示的项或分片赋值都是不合法的：

```py
>>> website = "http://www.python.org"
>>> website[-3:] = "com" Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  TypeError: 'str' object does not support item assignment 
```

## 3.2 字符串格式化：精简版

如果初次接触 Python 编程，那么 Python 提供的所有字符串格式化功能可能用不到太多。因此，这里只简单介绍一些主要的内容。如果读者对细节感兴趣，可以参见下一章，否则可以直接阅读 3.4 节。

字符串格式化使用字符串格式化操作符(这个名字还是很恰当的)即百分号`%`来实现。

*注：`%`也可以用作模运算(求余)操作符。*

在`%`的左侧放置一个字符串(格式化字符串)，而右侧则放置希望被格式化的值。可以使用一个值，如一个字符串或者数字，也可以使用多个值的元组或者下一章将会讨论的字典(如果希望格式化多个值的话)，这部分内容将在下一章进行讨论。一般情况下使用元组：

```py
>>> format = "Hello, %s. %s enough for ya?"
>>> values = ("world", "Hot")
>>> print format % values
Hello, world. Hot enough for ya? 
```

*注：如果使用列表或者其他序列代替元组，那么序列会被解释为一个值。只有元组和字典(将在第四章讨论)可以格式化一个以上的值。*

格式化字符串的`%s`部分称为*转换说明符*(conversion specifier)，它们标记了需要插入转换值的位置。`s`表示值会被格式化为字符串——如果不是字符串，则会用`str`将其转换为字符串。这个方法对大多数值都有效。其他转换说明符请参见本章后面的表 3-1.

*注：如果要在格式化字符串里面包括百分号，那么必须使用`%%`，这样 Python 就不会将百分号误认为是转换说明符了。*

如果要格式化实数(浮点数)，可以使用`f`说明转换说明符的类型，同时提供所需要的*精度*：一个句点再加上希望保留的小数位数。因为格式化转换说明符总是以表示类型的字符结束，所以精度应该放在类型字符前面：

```py
>>> format = "Pi with three decimals: %.3f"
>>> from math import pi
>>> print format % pi
Pi with three decimals: 3.142 
```

**模板字符串**

string 模块提供另外一种格式化值的方法：模板字符串。它的工作方式类似于很多 UNIX Shell 里的变量替换。如下所示，`substitute`这个模板方法会用传递进来的关键字参数`foo`替换字符串中的`$foo`(有关关键字参数的详细信息，请参看第六章)：

```py
>>> from string import Template
>>> s = Template("$x, glorious $x!")
>>> s.substitute(x="slurm") 'slurm, glorious slurm!' 
```

如果替换字段是单词的一部分，那么参数名就必须用括号括起来，从而准确指明结尾：

```py
>>> s = Template("It's ${x}tastic!")
>>> s.substitute(x="slurm") "It's slurmtastic!" 
```

可以使用`$$`插入美元符号：

```py
>>> s = Template("Make $$ selling $x!")
>>> s.substitute(x="slurm") 'Make $ selling slurm!' 
```

除了关键字参数之外，还可以使用字典变量提供值/名称对(参见第四章)。

```py
>>> s = Template("A $thing must never $action.") >>> d = {}
>>> d["thing"] = "gentleman"
>>> d["action"] = "show his socks"
>>> s.substitute(d)
'A gentleman must never show his socks.' 
```

方法`safe_substitute`不会因缺少值或者不正确使用`$`字符而出错(更多信息请参见 Python 库参考手册的 4.1.2 节)。

## 3.3 字符串格式化：完整版

格式化操作符的右操作数可以是任意类型，如果是元组或者映射类型(如字典)，那么字符串格式化将会有所不同。我们尚未涉及映射(如字典)，在此先了解一下元组。第四章还会详细介绍映射的格式化。

如果右操作数是元组的话，则其中的每一个元素都会被单独格式化，每个值都需要一个对应的转换说明符。

*注：如果需要转换的元组作为转换表达式的一部分存在，那么必须将它用圆括号括起来，以避免出错。*

```py
>>> "%s plus %s equals %s" % (1, 1, 2)
'1 plus 1 equals 2'
# Lacks parentheses!
>>> "%s plus %s equals %s" % 1, 1, 2  
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  TypeError: not enough arguments for format string 
```

基本的转换说明符(与此相对应的是完整的转换说明符，也就是包括映射键的说明符，详细内容参见第四章)包括以下部分。注意，这些项的顺序是至关重要的。

(1)`%`字符：标记转换说明符的开始。

(2)转换标志(可选)： `-` 表示左对齐； `+` 表示在转换值之前要加上正负号；`" "` (空白字符)表示整数之前保留空格；`0`表示转换值若位数不够则用`0`填充。

(3)最小字段宽度(可选)：转换后的字符串至少应该具有该值指定的宽度。如果是`*`，则宽度会从值元组中读出。

(4)点(`.`)后跟精度值(可选)：如果转化的是实数，精度值就表示出现在小数点后的位数。如果转换的是字符串，那么该数字就表示*最大字段宽度*。如果是*，那么精度将会从元组中读出。

(5)转换类型：参见表 3-1。

表 3-1 字符串格式化转换类型

```py
d， i　　　　　　带符号的十进制整数
o　　　　　　　　不带符号的八进制
u　　　　　　　　不带符号的十进制
x　　　　　　　　不带符号的十六进制(小写)
X　　　　　　　　不带符号的十六进制(大写)
e　　　　　　　　科学计数法表示的浮点数(小写)
E　　　　　　　　科学计数法表示的浮点数(大写)
f， F　　　　　　十进制浮点数
g　　　　　　　　如果指数大于-4 或者小于精度值则和 e 相同，其他情况与 f 相同
G　　　　　　　　如果指数大于-4 或者小于精度值则和 E 相同，其他情况与 F 相同
C　　　　　　　　单字符(接受整数或者单字符字符串)
r　　　　　　　　字符串(使用 repr 转换任意 Python 对象)
s　　　　　　　　字符串(使用 str 转换任意 Python 对象) 
```

接下来几个小节将对转换说明符的各个元素进行详细讨论。

### 3.3.1 简单转换

简单的转换只需要写出转换类型，使用起来很简单：

```py
>>> "Price of eggs: $%d" % 42
'Price of eggs: $42'
>>> "Hexadecimal price of eggs: %x" % 42
'Hexadecimal price of eggs: 2a'
>>> from math import pi >>> "Pi: %f..." %pi
'Pi: 3.141593...'
>>> "Very inexact estimate of pi: %i" % pi
'Very inexact estimate of pi: 3'
>>> "Using str: %s" % 42L
'Using str: 42'
>>> "Using repr: %r" % 42L
'Using repr: 42L' 
```

### 3.3.2 字段宽度和精度

转换说明符可以包括字段宽度和精度。字段宽度是转换后的值所保留的最小字符个数，精度(对于数字转换来说)则是结果中应该包含的小数位数，或者(对于字符串转换来说)是转换后的值所能包含的最大字符个数。

这两个参数都是整数(首先是字段宽度，然后是精度)，通过点号(`.`)分隔。虽然两个都是可选的参数，但如果只给出精度，就必须包含点号：

```py
>>> "%10f" % pi  # 字段宽 10
' 3.141593'
>>> "%10.2f" % pi  # 字段宽 10，精度 2
' 3.14'
>>> "%.2f" % pi  # 精度 2
'3.14'
>>> "%.5s" % "Guido van Rossum"
'Guido' 
```

可以使用`*`(星号)作为字段宽度或者精度(或者两者都是用`*`)，此时数值会从元组参数中读出：

```py
>>> "%.*s" % (5, "Guido van Rossum")
'Guido' 
```

### 3.3.3 符号、对齐和用 0 填充

在字段宽度和精度值之前还可以放置一个“标志”，该标志可以是零、加号、减号或空格。零表示数字将会用`0`进行填充。

```py
>>> "%010.2f" % pi
'0000003.14' 
```

注意，在`010`中开头的那个`0`并不意味着字段宽度说明符为八进制数，它只是个普通的 Python 数值。当使用`010`作为字段宽度说明符的时候，表示字段宽度为`10`，并且用`0`进行填充空位，而不是说字段宽度为`8`：

```py
>>> 010
8 
```

减号(`-`)用来左对齐数值：

```py
>>> "%-10.2f" % pi
'3.14 ' 
```

可以看到，在数字的右侧多出了额外的空格。

而空白(`" "`)意味着在正数前加上空格。这在需要对齐正负数时会很有用：

```py
>>> print ("%+5d" % 10) + "\n" + ("%+5d" % -10) +10
-10 
```

代码清单 3-1 中的代码将使用星号字段宽度说明符来格式化一张包含水果价格的表格，表格的总宽度由用户输入。因为是由用户提供信息，所以就不能在转换说明符中将字段宽度硬编码。使用星号运算符就可以从转换元组中读出字段宽度。

```py
 1 #!/usr/bin/env python
 2 # coding=utf-8
 3
 4 # 使用给定的宽度打印格式化后的价格列表
 5
 6 width = input("Please enter width: ")
 7
 8 price_width = 10
 9 item_width = width - price_width
10
11 header_format = "%-*s%*s"
12 format = "%-*s%*.2f"
13
14 print "=" * width 15
16 print header_format % (item_width, "Item", price_width, "Price")
17
18 print "-" * width 19
20 print format % (item_width, "Apples", price_width, 0.4)
21 print format % (item_width, "Pears", price_width, 0.5)
22 print format % (item_width, "Cantaloupes", price_width, 1.92)
23 print format % (item_width, "Dried Apricots (16 oz.)", price_width, 8) 24 print format % (item_width, "Prunes (4 lbs.)", price_width, 12)
25
26 print "=" * width 
```

Code_Listing 3-1

以下是程序运行示例：

```py
Please enter width: 35
===================================
Item                          Price
-----------------------------------
Apples                         0.40
Pears                          0.50
Cantaloupes                    1.92
Dried Apricots (16 oz.)        8.00
Prunes (4 lbs.)               12.00
=================================== 
```

## 3.4 字符串方法

前面几节已经介绍了很多列表的方法，字符串的方法还要丰富得多，这是因为字符串从`string`模块中“继承”了很多方法，而在早期版本的 Python 中，这些方法都是作为函数出现的(如果真的需要的话，还是能找到这些函数的)。

因为字符串的方法是实在太多，在这里只介绍一些特别有用的。全部方法请参见附录 B。在字符串的方法描述中，可以在本章找到关联到其他方法的参考(用“请参见”标记)，或请参见附录 B。

**但是字符串未死**

尽管字符串方法完全来源于`string`模块，但是这个模块还包括一些不能作为字符串方法使用的常量和函数。`maketrans`函数就是其中之一，后面会将它和`translate`方法一起介绍。下面是一些有用的字符串常量(对于此模块的更多介绍，请参见[Python 库参考手册](http://python.org/doc/lib/module-string.html)的 4.1 节)。

√ string.digits：包含数字 0~9 的字符串。

√ string.letters：包含所有字母(大写或小写)的字符串。

√ string.lowercase：包含所有小写字母的字符串。

√ string.printable：包含所有可打印字符的字符串。

√ string.punctuation：包含所有标点的字符串。

√ string.uppercase：包含所有大写字母的字符串。

字母字符串常量(例如`string.letters`)与地区有关(也就是说，其具体值取决于 Python 所配置的语言)(在 Python3.0 中，`string.letters`和其相关内容都会被移除。如果需要则应该使用`string.ascii_letters`常量代替)。如果可以确定自己使用的 ASCII，那么可以在变量中使用`ascii_`前缀，例如`string.ascii_letters`。

### 3.4.1 `find`

`find`方法可以在一个较长的字符串中查找子串。它返回子串所在位置的最左端索引。如果没有找到则返回`-1`。

```py
>>> "With a moo-moo here, and a moo-moo there".find("moo") 7
>>> title = "Monty Python's Flying Circus"
>>> title.find("Monty")
0 >>> title.find("Python")  # 找到字符串
6
>>> title.find("Flying") 15
>>> title.find("Zirquss")  # 未找到字符串
-1 
```

在第二章中我们初始了成员资格，我们在`subject`中使用了`"$$$"`表达式建立了一个垃圾邮件过滤器。也可以使用`find`方法(Python2.3 以前的版本中也可用，但是`in`操作符只能用来查找字符串中的单个字符)：

```py
>>> subject = "$$$ Get rich now!!! $$$"
>>> subject.find("$$$")
0 
```

*注：字符串的`find`方法并不返回布尔值。如果返回的是`0`，则证明在索引`0`位置找到了子串。*

这个方法还可以接收可选的起始点和结束点参数：

```py
>>> subject = "$$$ Get rich now!!! $$$"
>>> subject.find("$$$")
0 >>> subject.find("$$$", 1)  # 只提供起始点
20
>>> subject.find("!!!") 16
>>> subject.find("!!!", 0, 16)  # 提供起始点和结束点
-1 
```

注意，由起始和终止值指定的范围(第二个和第三个参数)包含第一个索引，但不包含第二个索引。这在 Python 中是个惯例。

附录 B：`rfind`、`index`、`rindex`、`count`、`startswith`、`endswith`。

### 3.4.2 `join`

`join`方法是非常重要的字符串方法，它是`split`方法的逆方法，用来连接序列中的元素：

```py
>>> seq = [1, 2, 3, 4, 5]
>>> sep = "+"
>>> sep.join(seq)  # 连接数字列表
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  TypeError: sequence item 0: expected string, int found
>>> seq = ["1", "2", "3", "4", "5"]
>>> sep.join(seq)  # 连接字符串列表
'1+2+3+4+5'
>>> dirs = "", "usr", "bin", "env"
>>> "/".join(dirs)
'/usr/bin/env'
>>> print "C:" + "\\".join(dirs)
C:\usr\bin\env 
```

可以看到，需要被连接的序列元素都必须是字符串。注意最后两个例子中使用了目录的列表，而在格式化时，根据 UNIX 和 DOS/Windows 的约定，使用了不同的分隔符号(在 DOS 版本中还增加了驱动器名)。

请参见：`split`。

### 3.4.3 `lower`

`lower`方法返回字符串的小写字母版。

```py
>>> "Trondheim Hammer Dance".lower()
'trondheim hammer dance' 
```

如果想要编写“不区分大小写”的代码的话，那么这个方法就派上用场了——代码会忽略大小写状态。例如，如果想在列表中查找一个用户名是否存在：列表包含字符串`"gumby"`，而用户输入的是`"Gumby"`，就找不到了：

```py
>>> if "Gumby" in ["gumby", "smith", "jones"]:
        print "Found it!"
...
>>> 
```

如果存储的是`"Gumby"`而用户输入`"gumby"`甚至是`"GUMBY"`，结果也是一样的。解决方法就是在存储和搜索时把所有名字都转换为小写。代码如下：

```py
>>> name = "Gumby"
>>> names = ["gumby", "smith", "jones"]
>>> if name.lower() in names:
        print "Found it!"
...
Found it! 
```

请参见：`translate`。

附录 B：`islower`、`capitalize`、`swapcase`、`title`、`istitle`、`upper`、`isupper`。

**标题转换**

和`lower`方法相关的是`title`方法(参见附录 B)，它会将字符串转换为标题——也就是所有单词的首字母大写，而其他字母小写。但是它使用的单词划分方法可能会得到并不自然的结果：

```py
>>> "that's all folks".title()
"That'S All Folks" 
```

再介绍另外一个`string`模块的`capwords`函数：

```py
>>> import string
>>> string.capwords("that's all, folks")
"That's All, Folks" 
```

当然，如果要得到正确首字母大写的标题(这要根据你的风格而定，可能要小写冠词、连词及 5 个字母以下的介词等)，那么还是得自己把握。

### 3.4.4 `replace`

`replace`方法返回某字符串的所有匹配项均被替换之后得到字符串。

```py
>>> "This is a test".replace("is", "eez")
'Theez eez a test' 
```

如果曾经用过文字处理程序中的“查找并替换”功能的话，就不会质疑这个方法的用处了。

请参见：`translate`。

附录 B：`expandtabs`。

### 3.4.5 `split`

这是一个非常重要的字符串方法，它是`join`的逆方法，用来将字符串分隔成序列。

```py
>>> "1+2+3+4+5".split("+")
['1', '2', '3', '4', '5']
>>> "/usr/bin/env".split("/")
['', 'usr', 'bin', 'env']
>>> "Using the default".split()
['Using', 'the', 'default'] 
```

注意，如果不提供任何分隔符，程序会把所有空格作为分隔符(空格、制表、换行等)。

请参见：`join`。

附录 B：`rsplit`、`splitlines`。

### 3.4.6 `strip`

`strip`方法返回去除两侧(不包括内部)空格的字符串：

```py
>>> " internal whitespace is kept ".strip()
'internal whitespace is kept' 
```

它和`lower`方法一起使用的话就可以很方便的对比输入的和存储的值。让我们回到`lower`部分中的用户名的例子，假设用户在输入名字时无意中在名字后面加上了空格：

```py
>>> names = ["gumby", "smith", "jones"]
>>> name = "gumby "
>>> if name in names:
        print "Found it!"
...
>>> if name.strip() in names:
        print "Found it!"
...
Found it! 
```

也可以指定需要去除的字符，将它们列为参数即可。

```py
>>> "*** SPAM * for * everyone!!! ***".strip(" *!")
'SPAM * for * everyone' 
```

这个方法只会去除两侧的字符，所以字符串中的星号没有被去掉。

附录 B：`lstrip`、`rstrip`。

### 3.4.7 `translate`

`translate`方法和`replace`方法一样，可以替换字符串中的某些部分，但是和前者不同的是，`translate`方法只处理单个字符。它的优势在于可以同时进行多个替换，有些时候比`replace`效率高得多。

使用这个方法的方式有很多(比如替换换行符或者其他因平台而异的特殊字符)。但是让我们考虑一个简单的例子(很简单的例子)：假设需要将纯正的英文文本转换为带有德国口音的版本。为此，需要把字符`c`替换为`k`把`s`替换为`z`。

在使用`translate`转换之前，需要先完成一张*转换表*。转换表中是以某字符替换某字符的对应关系。因为这个表(事实上是字符串)有多达 256 个项目，我们还是不要自己写了，使用`string`模块里面的`maketrans`函数就行。

`maketrans`函数接受两个参数：两个等长的字符串，表示第一个字符串中的每个字符都用第二个字符串中相同位置的字符替换。明白了吗？来看一个简单的例子，代码如下：

```py
>>> from string import maketrans
>>> table = maketrans("cs", "kz") 
```

**转换表中都有什么**

转换表是包含替换 ASCII 字符集中 256 个字符的替换字母的字符串。

```py
>>> table = maketrans("cs", "kz")
>>> len(table) 256
>>> table[97:123]
'abkdefghijklmnopqrztuvwxyz'
>>> maketrans("", "")[97:123]
'abcdefghijklmnopqrstuvwxyz' 
```

正如你看到的，我已经把小写字母部分的表提取出来了。看一下这个表和空转换(没有改变任何东西)中的字母表。空转换包含一个普通的字母表，而在它前面的代码中，字母`c`和`s`分别被替换为`k`和`z`。

创建这个表以后，可以将它用作`translate`方法的参数，进行字符串的转换如下：

```py
>>> "this is an incredible test".translate(table)
'thiz iz an inkredible tezt' 
```

`translate`的第二个参数是可选的，这个参数是用来指定需要删除的字符。例如，如果想要模拟一句语速超快的德国语，可以删除所有空格：

```py
>>> "this is an incredible test".translate(table, " ") 'thizizaninkredibletezt' 
```

请参见：`replace`、`lower`。

## 3.5 小结

本章介绍了字符串的两种非常重要的使用方式。

**字符串格式化**：求模操作符(`%`)可以用来将其他值转换为包含转换标志的字符串，例如`%s`。它还能用来对值进行不同方式的格式化，包括左右对齐、设定字段宽度以及精度值，增加符号(正负号)或者左填充数字`0`等。

**字符串方法**：字符串有很多方法。有些非常有用(比如`split`和`join`)，有些则用的很少(比如`istitle`或者`capitalize`)。

### 3.5.1 本章的新函数

本章新涉及的函数如表 3-2 所示。

表 3-2 本章的新函数

```py
string.capwords(s[, sep])   使用 split 函数分隔字符串 s(以 sep 为分隔符)，使用 capitalize 函数将分割得到的各单词首字母大写，并且使用 join 函数以 sep 为分隔符将各单词连接起来。
string.maketrans(from, to)  创建用于转换的转换表。 
```

### 3.5.2 接下来学什么

列表、字符串和字典是 Python 中最重要的 3 种数据类型。列表和字符串已经学习过了，那么下面是什么呢？下一章中的主要内容是字典，以及字典如何支持索引以及其他方式的键(比如字符串和元组)。字典也提供了一些方法，但是数量没有字符串多。