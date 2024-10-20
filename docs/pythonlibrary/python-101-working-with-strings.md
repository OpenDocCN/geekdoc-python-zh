# Python 101 -使用字符串

> 原文：<https://www.blog.pythonlibrary.org/2020/04/07/python-101-working-with-strings/>

当你编程的时候，你会经常用到字符串。字符串是由单引号、双引号或三引号括起来的一系列字母。Python 3 将 string 定义为“文本序列类型”。您可以使用内置的`str()`函数将其他类型转换为字符串。

在本文中，您将学习如何:

*   创建字符串
*   字符串方法
*   字符串格式
*   串并置
*   字符串切片

让我们从学习创建字符串的不同方法开始吧！

### 创建字符串

以下是创建字符串的一些示例:

```py
name = 'Mike'
first_name = 'Mike'
last_name = "Driscoll"
triple = """multi-line
string"""
```

当您使用三重引号时，您可以在字符串的开头和结尾使用三个双引号或三个单引号。另外，请注意，使用三重引号允许您创建多行字符串。字符串中的任何空白也将包括在内。

下面是一个将整数转换为字符串的示例:

```py
>>> number = 5
>>> str(number)
'5'
```

在 Python 中，反斜杠可以用来创建转义序列。这里有几个例子:

*   `\b` -退格键
*   `\n` -换行
*   `\r` - ASCII 回车
*   `\t` -标签

如果您阅读 Python 的文档，还可以了解其他几个。

您还可以使用反斜杠来转义引号:

```py
>>> 'This string has a single quote, \', in the middle'
"This string has a single quote, ', in the middle"
```

如果上面的代码中没有反斜杠，您将收到一个`SyntaxError`:

```py
>>> 'This string has a single quote, ', in the middle'
Traceback (most recent call last):
  Python Shell, prompt 59, line 1
invalid syntax: <string>, line 1, pos 38
```

这是因为字符串在第二个单引号处结束。通常最好是混合使用双引号和单引号来解决这个问题:

```py
>>> "This string has a single quote, ', in the middle"
"This string has a single quote, ', in the middle"
```

在这种情况下，您使用双引号创建字符串，并将单引号放入其中。这在处理缩略词时尤其有用，比如“不要”、“不能”等。

现在让我们继续，看看你能用什么方法来处理字符串！

### 字符串方法

在 Python 中，一切都是对象。当你学习内省的时候，你会在第 18 章学到这是多么有用。现在，只要知道字符串有可以调用的方法(或函数)就行了。

这里有三个例子:

```py
>>> name = 'mike'
>>> name.capitalize()
'Mike'
>>> name.upper()
'MIKE'
>>> 'MIke'.lower()
'mike'
```

方法名给了你一个线索，让你知道它们是做什么的。例如，`.capitalize()`会将字符串中的第一个字母改为大写字母。

要获得可以访问的方法和属性的完整列表，可以使用 Python 内置的`dir()`函数:

```py
>>> dir(name)
['__add__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
'__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__',
'__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__',
'__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'capitalize',
'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 'index',
'isalnum', 'isalpha', 'isascii', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable',
'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace',
'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip',
'swapcase', 'title', 'translate', 'upper', 'zfill']
```

清单的前三分之一是特殊方法，有时被称为“dunder 方法”(又名双下划线方法)或“魔术方法”。您现在可以忽略这些，因为它们更多地用于中级和高级用例。上面列表中开头没有双下划线的条目可能是您最常用的条目。

您会发现`.strip()`和`.split()`方法在解析或操作文本时特别有用。

您可以使用`.strip()`及其变体、`.rstrip()`和`.lstrip()`来去除字符串中的空白，包括制表符和换行符。这在您读取需要解析的文本文件时特别有用。

事实上，您通常会从字符串中去掉行尾字符，然后对结果使用`.split()`来解析出子字符串。

让我们做一个小练习，学习如何解析字符串中的第二个单词。

首先，这里有一个字符串:

```py
>>> my_string = 'This is a string of words'
'This is a string of words'
```

现在要得到字符串的各个部分，你可以调用`.split()`，就像这样:

```py
>>> my_string.split()
['This', 'is', 'a', 'string', 'of', 'words']
```

结果是字符串的`list`。现在通常你会把这个结果赋给一个变量，但是为了演示的目的，你可以跳过这部分。

相反，既然您现在知道结果是一个字符串，那么您可以使用列表切片来获得第二个元素:

```py
>>> 'This is a string of words'.split()[1]
'is'
```

记住，在 Python 中，列表元素从 0(零)开始，所以当你告诉它你想要元素 1(一)时，它是列表中的第二个元素。

当在工作中进行字符串解析时，我个人发现您可以非常有效地使用`.strip()`和`.split()`方法来获得几乎任何您需要的数据。偶尔您会发现您可能还需要使用正则表达式(regex)，但是大多数时候这两种方法就足够了。

### 字符串格式

字符串格式化或字符串替换是指将一个字符串插入到另一个字符串中。这在你需要做模板的时候特别有用，比如套用信函。但是您将大量使用字符串替换来调试输出、打印到标准输出等等。

Python 有三种不同的方法来完成字符串格式化:

*   使用%方法
*   使用`.format()`
*   使用格式化字符串(f 字符串)

这本书将重点放在 f 弦上，也不时使用`.format()`。但是理解这三者是如何工作的是有好处的。

让我们花一些时间来学习更多关于字符串格式的知识。

### 使用%s 格式化字符串(printf 样式)

使用`%`方法是 Python 最古老的字符串格式化方法。它有时被称为“printf 风格的字符串格式化”。如果您过去使用过 C 或 C++，那么您可能已经熟悉了这种类型的字符串替换。为了简洁起见，您将在这里学习使用`%`的基本知识。

*注意:这种格式很难处理，并且会导致常见的错误，比如无法正确显示 Python 元组和字典。在这种情况下，最好使用其他两种方法中的任何一种。*

使用`%`符号最常见的情况是使用`%s`，这意味着使用`str()`将任何 Python 对象转换成字符串。

这里有一个例子:

```py
>>> name = 'Mike'
>>> print('My name is %s' % name)
My name is Mike
```

在这段代码中，使用特殊的`%s`语法将变量`name`插入到另一个字符串中。为了让它工作，你需要在字符串外面使用`%`,后跟你想要插入的字符串或变量。

下面是第二个例子，展示了您可以将一个`int`传入一个字符串，并让它自动为您转换:

```py
>>> age = 18
>>> print('You must be at least %s to continue' % age)
You must be at least 18 to continue
```

当你需要转换一个对象但不知道它是什么类型时，这种事情特别有用。

您还可以使用多个变量进行字符串格式化。事实上，有两种方法可以做到这一点。

这是第一个:

```py
>>> name = 'Mike'
>>> age = 18
>>> print('Hello %s. You must be at least %i to continue!' % (name, age))
Hello Mike. You must be at least 18 to continue!
```

在这个例子中，您创建了两个变量并使用了`%s`和`%i`。`%i`表示你要传递一个整数。要传入多个项，可以使用百分号，后跟要插入的项的元组。

你可以用名字来说明这一点，就像这样:

```py
>>> print('Hello %(name)s. You must be at least %(age)i to continue!' % {'name': name, 'age': age})
Hello Mike. You must be at least 18 to continue!
```

当`%`符号右侧的参数是字典(或另一种映射类型)时，字符串中的格式必须引用字典中带括号的键。换句话说，如果你看到`%(name)s`，那么在`%`右边的字典一定有一个`name`键。

如果没有包含所有必需的密钥，您将收到一条错误消息:

```py
>>> print('Hello %(name)s. You must be at least %(age)i to continue!' % {'age': age})
Traceback (most recent call last):
   Python Shell, prompt 23, line 1
KeyError: 'name'
```

有关使用 printf 样式的字符串格式的更多信息，请参见以下链接:

[https://docs . python . org/3/library/stdtypes . html # printf-style-string-formatting](https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting)

现在让我们继续使用`.format()`方法。

### 使用格式化字符串。格式()

Python 字符串支持`.format()`方法已经很久了。虽然这本书将重点关注 f 弦的使用，但你会发现`.format()`仍然很受欢迎。

有关格式化如何工作的完整详细信息，请参见以下内容:

[https://docs.python.org/3/library/string.html#formatstrings](https://docs.python.org/3/library/string.html#formatstrings)

让我们来看几个简短的例子，看看`.format()`是如何工作的:

```py
>>> age = 18
>>> name = 'Mike'
>>> print('Hello {}. You must be at least {} to continue!'.format(name, age))
Hello Mike. You must be at least 18 to continue!
```

此示例使用位置参数。Python 寻找`{}`的两个实例，并相应地插入变量。如果您没有传入足够的参数，您将收到如下错误:

```py
>>> print('Hello {}. You must be at least {} to continue!'.format(age))
Traceback (most recent call last):
    Python Shell, prompt 33, line 1
IndexError: tuple index out of range
```

这个错误表明您在`.format()`调用中没有足够的项目。

您也可以按照与上一节类似的方式使用命名参数:

```py
>>> age = 18
>>> name = 'Mike'
>>> print('Hello {name}. You must be at least {age} to continue!'.format(name=name, age=age))
Hello Mike. You must be at least 18 to continue!
```

您可以通过名称传入参数，而不是将字典传递给`.format()`。事实上，如果您尝试传入一个字典，您将会收到一个错误:

```py
>>> print('Hello {name}. You must be at least {age} to continue!'.format({'name': name, 'age': age}))
Traceback (most recent call last):
  Python Shell, prompt 34, line 1
KeyError: 'name'
```

不过，有一个解决方法:

```py
>>> print('Hello {name}. You must be at least {age} to continue!'.format(**{'name': name, 'age': age}))
Hello Mike. You must be at least 18 to continue!
```

这看起来有点奇怪，但是在 Python 中，当您看到像这样使用双星号(`**`)时，这意味着您正在向函数传递命名参数。所以 Python 正在为你把字典转换成`name=name, age=age`。

您也可以使用`.format()`在字符串中多次重复一个变量:

```py
>>> name = 'Mike'
>>> print('Hello {name}. Why do they call you {name}?'.format(name=name))
Hello Mike. Why do they call you Mike?
```

这里你在字符串中引用了两次`{name}`，你可以用`.format()`来替换它们。

如果需要，您也可以使用数字来插值:

```py
>>> print('Hello {1}. You must be at least {0} to continue!'.format(name, age))
Hello 18\. You must be at least Mike to continue!
```

因为 Python 中的大多数东西都是从 0(零)开始的，所以在这个例子中，您最终将`age`传递给了`{1}`，将`name`传递给了`{0}`。

使用`.format()`时，一种常见的编码方式是创建一个格式化字符串，并将其保存到一个变量中以备后用:

```py
>>> age = 18
>>> name = 'Mike'
>>> greetings = 'Hello {name}. You must be at least {age} to continue!'
>>> greetings.format(name=name, age=age)
'Hello Mike. You must be at least 18 to continue!'
```

这允许你在程序中重用`greetings`并传递更新后的`name`和`age`的值。

您还可以指定字符串宽度和对齐方式:

```py
>>> '{:<20}'.format('left aligned')
'left aligned        '
>>> '{:>20}'.format('right aligned')
'       right aligned'
>>> '{:^20}'.format('centered')
'      centered      '
```

默认为左对齐。冒号(`:`)告诉 Python 您将应用某种格式。在第一个示例中，您指定字符串左对齐，宽度为 20 个字符。第二个例子也是 20 个字符宽，但它是右对齐的。最后，`^`告诉 Python 将字符串放在 20 个字符的中间。

如果你想像前面的例子那样传入一个变量，你可以这样做:

```py
>>> '{name:^20}'.format(name='centered')
'      centered      '
```

请注意，`name`必须在`{}`内的`:`之前。

至此，您应该非常熟悉`.format()`的工作方式。

让我们继续前进到 f 弦！

### 用 f 字符串格式化字符串

格式化的字符串文字或 string 是在开头有一个“f”的字符串，在它们里面有包含表达式的花括号，很像您在上一节中看到的那些。这些表达式告诉 f 字符串需要对插入的字符串进行的任何特殊处理，比如对齐、浮点精度等。

f 字符串是在 Python 3.6 中添加的。你可以通过点击这里查看 PEP 498 来了解更多关于它和它是如何工作的:

[https://www.python.org/dev/peps/pep-0498/](https://www.python.org/dev/peps/pep-0498/)

f 字符串中包含的表达式在运行时进行计算。如果 f-string 包含一个表达式，那么它就不能作为函数、方法或类的文档字符串。原因是文档字符串是在函数定义时定义的。

让我们来看一个简单的例子:

```py
>>> name = 'Mike'
>>> age = 20
>>> f'Hello {name}. You are {age} years old'
'Hello Mike. You are 20 years old'
```

在这里，您通过在字符串开头的单引号、双引号或三引号前加上“f”来创建 f 字符串。然后在字符串内部，使用花括号`{}`，将变量插入字符串。

但是，您的花括号必须包含一些内容。如果您创建一个带有空括号的 f 字符串，您将得到一个错误:

```py
>>> f'Hello {}. You are {} years old'
SyntaxError: f-string: empty expression not allowed
```

f 弦可以做`%s`和`.format()`都做不到的事情。因为 f 字符串是在运行时计算的，所以可以在其中放入任何有效的 Python 表达式。

例如，您可以增加`age`变量:

```py
>>> age = 20
>>> f'{age+2}'
'22'
```

或者调用一个方法或函数:

```py
>>> name = 'Mike'
>>> f'{name.lower()}'
'mike'
```

您也可以直接在 f 字符串中访问字典值:

```py
>>> sample_dict = {'name': 'Tom', 'age': 40}
>>> f'Hello {sample_dict["name"]}. You are {sample_dict["age"]} years old'
'Hello Tom. You are 40 years old'
```

但是，f 字符串表达式中不允许使用反斜杠:

```py
>>> print(f'My name is {name\n}')
SyntaxError: f-string expression part cannot include a backslash
```

但是您可以在 f 字符串中的表达式之外使用反斜杠:

```py
>>> name = 'Mike'
>>> print(f'My name is {name}\n')
My name is Mike
```

另一件不能做的事情是在 f 字符串的表达式中添加注释:

```py
>>> f'My name is {name # name of person}'
SyntaxError: f-string expression part cannot include '#'
```

在 Python 3.8 中，f-strings 增加了对`=`的支持，这将扩展表达式的文本，以包括表达式的文本加上等号，然后是求值的表达式。这听起来有点复杂，所以让我们看一个例子:

```py
>>> username = 'jdoe'
>>> f'Your {username=}'
"Your username='jdoe'"
```

这个例子演示了表达式中的文本`username=`被添加到输出中，后面是引号中的实际值`username`。

f 弦非常有力，非常有用。如果你明智地使用它们，它们会大大简化你的代码。你绝对应该给他们一个尝试。

让我们看看你还能用绳子做些什么吧！

### 串并置

字符串还允许连接，这是一个将两个字符串连接成一个字符串的时髦词。

要将字符串连接在一起，可以使用`+`符号:

```py
>>> first_string = 'My name is'
>>> second_string = 'Mike'
>>> first_string + second_string
'My name isMike'
```

哎呀！看起来字符串以一种奇怪的方式合并了，因为你忘了在`first_string`的末尾加一个空格。您可以像这样更改它:

```py
>>> first_string = 'My name is '
>>> second_string = 'Mike'
>>> first_string + second_string
'My name is Mike'
```

另一种合并字符串的方法是使用`.join()`方法。`.join()`方法接受一个可迭代的字符串，比如一个列表，并将它们连接在一起。

```py
>>> first_string = 'My name is '
>>> second_string = 'Mike'
>>> ''.join([first_string, second_string])
'My name is Mike'
```

这将使字符串紧挨着彼此连接。你可以在要连接的字符串中放一些东西:

```py
>>> '***'.join([first_string, second_string])
'My name is ***Mike'
```

在这种情况下，它会将第一个字符串连接到`***`加上第二个字符串。

通常情况下，您可以使用 f 字符串而不是串联或`.join()`,这样代码会更容易理解。

### 切串

字符串切片的工作方式与 Python 列表非常相似。让我们以“迈克”为例。字母“M”在位置 0，字母“e”在位置 3。

如果你想抓取字符 0-3，你可以使用这个语法:`my_string[0:4]`

这意味着您希望子字符串从位置 0 开始，但不包括位置 4。

这里有几个例子:

```py
>>> 'this is a string'[0:4]
'this'
>>> 'this is a string'[:4]
'this'
>>> 'this is a string'[-4:]
'ring'
```

第一个示例从字符串中获取前四个字母并返回它们。如果你愿意，你可以去掉默认的零，而使用 **[:4]** ，这就是第二个例子所做的。

您也可以使用负的位置值。所以 **[-4:]** 的意思是你想从字符串的末尾开始，得到字符串的最后四个字母。

您应该自己尝试切片，看看还能想到什么其他切片。

### 总结

Python 字符串功能强大，非常有用。可以使用单引号、双引号或三引号来创建它们。字符串是对象，所以有方法。您还了解了字符串连接、字符串切片和三种不同的字符串格式化方法。

最新的字符串格式是 f 字符串。它也是格式化字符串的最强大和当前首选的方法。

### 相关阅读

*   Python 101 - [了解字典](https://www.blog.pythonlibrary.org/2020/03/31/python-101-learning-about-dictionaries/)
*   python 101-[了解元组](https://www.blog.pythonlibrary.org/2020/03/26/python-101-learning-about-tuples/)
*   Python 101: [了解列表](https://www.blog.pythonlibrary.org/2020/03/10/python-101-learning-about-lists/)