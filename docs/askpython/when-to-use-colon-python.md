# Python 中什么时候用冒号(:)？

> 原文：<https://www.askpython.com/python/examples/when-to-use-colon-python>

众所周知，Python 是一种易于使用和实现的语言，语法上有很多自由。这使得它甚至对初学者来说都是完美的，可以让应用程序这样的东西高效地设计。但是，像其他语言一样，它也有一些基本的规则和规定，整个代码运行和依赖这些规则和规定。因此，在这篇文章中，我们将学习它们。这对我们的编程很重要。

## Python 和 PEP8 简介

现在。为了让事情更清楚，让我们看看 Python 的特性。

1.  **面向对象**
2.  **多范例**
3.  **跨平台**

python 与众不同的主要原因是它运行在一个**解释器**上。这将逐行运行代码，然后执行。

### PEP8 的精髓

我们可以说 **Python 增强建议(PEP)** 是一本官方书籍或一套规则，告诉我们如何编写最好的 Python 代码。它还提供了一套编程时的限制或**不要做的**事情。其中一些如下:

1.  模块不应有短的**小写**名称。
2.  类名应该采用 **CapWords** 风格。
3.  大多数变量和函数名应该是**小写 _ 带 _ 下划线。**
4.  ***常量应该大写加下划线——这有助于识别它们。***
5.  在参数和运算符之间使用足够的空格，以使代码更具可读性。

要获得更多 PEP 信息，我们可以打开 Python shell，并在其中键入以下命令:

```py
>>> import this

```

**输出:**

```py
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

```

## 冒号在 Python 中的重要性

所以，在一些解释语言中，花括号并不重要。相反，我们与**【上校】**一起工作。与 JavaScript 和 Ruby 等其他语言不同，Python 拥有大多数重要的语法，其中冒号非常重要。以下是这些问题的清单:

1.  列表切片。
2.  字符串切片。
3.  在字典中用于插入键值对。
4.  函数声明
5.  循环声明
6.  条件语句
7.  增加函数的可读性。
8.  类声明。

**注意:在列表和字符串中，索引从 0 开始。更多参考请阅读本文:[https://www.askpython.com/python/list/negative-indexing](https://www.askpython.com/python/list/negative-indexing)**

### 1.列表切片

列表是在 Python 中实现和学习的重要数据结构。它就像一个动态数组，我们可以在其中插入多种数据类型的元素。在列表中，我们使用冒号来检索特定的元素。他们研究索引数字。所以，我们可以用它们来得到关于索引位置的元素。

**获取元素的语法:**

```py
a = [0 : n -1] # indexing in lists starts from zero till n -1 

```

**举例:**

```py
>>> a = [5, -9, 8, 33, 4, 64] # declaring a list
>>> a[:] # retrieving all the elements
[5, -9, 8, 33, 4, 64]
>>> a[2:3] # retrieing the elements from third index and ignoring the fourth index
[8]
>>> a[4:] # retrieving the elements above fifth till last index
[4, 64]
>>> a[1:] # retrieving the elements above second index
[-9, 8, 33, 4, 64]

```

### 2.字符串切片

字符串是 Python 中的另一种数据类型，它可以将一长串句子括在引号中。在旧的编程范式中，字符串是一系列字符。Python 遵循同样的方法从字符串中检索单个字符。它们是不可变的(不可编辑的),但是我们可以获取字符。在内存中，它们存储在一个字符数组中。因此，为了得到它们，使用冒号:

又念:[如何在 Python 中切片字符串？](https://www.askpython.com/python/string/slice-strings-in-python)

**获取字符的语法:**

```py
s = [0 : n -1] # indexing in strings starts from zero till n -1 

```

**举例:**

```py
>>> s = "abcdefgh"
>>> s[2:4] # fetch characters from third till fifth index
'cd'
>>> s[::-1] # reverse the string
'hgfedcba'
>>> s[:-2] # print the string ignoring the second and last characters
'abcdef'
>>> s[1:5]  # print the elements from second till 
'bcde'

```

### 3.在用于插入键值对的字典中

Python 中的字典是一个无序的键值对集合。它们是像 Java 中的 hashmaps 一样的基本数据结构之一。但是，它们的语法声明有很大不同。

**声明字典的语法:**

```py
d = {key_1 : value_1, key_2 : value_2, key_3 : value_3, ..., key_N : value_N}

```

正如我们所见，结肠在这里是一个重要的实体。如果没有这个符号，字典在 Python 中就无法存在。

**举例:**

```py
 >>> d = {"a" : 1, "b" : 2, "c" : 3, "d" : 4} # declaring a dictionary

```

### 4.函数声明

函数的一般语法包含冒号。这是因为 Python 使用缩进**(空白代码空间)**而不是花括号 **"{ }"** 来保留函数下的代码块。 ***在函数和括号之后，我们需要用冒号开始编写函数内部的代码。***

**语法:**

```py
def func_name(param_1, .., param_2): ->  colon used here
    # fuction code lies below colon and is considered as its code body

    return # value

```

### 5.循环声明

Python 中的循环是连续执行一段代码直到代码满足特定条件的语句。因此，为了执行 for()或 while()循环，我们使用冒号。冒号下面的所有代码都被认为是循环的一部分，当且仅当它采用适当的缩进。

**举例:**

```py
for i in range(4): # -> colon is used here
    print(i) 

#Output:
# 0
# 1 
# 2
# 3

```

所以，正如我们看到的，冒号下面的代码给出了从 0 到 3 的数字。同样，我们可以在 while 循环中使用它。

```py
i = 0
while i != 5: # -> colon is used here
    i += 1
    print(i)
# output
# 1
# 2
# 3
# 4
# 5

```

### 6.使用条件语句

条件语句是特殊的代码块。它们是决策语句，当代码块中的表达式计算结果为 true 时，执行代码块。我们也用冒号。它们被放在条件之后，解释器识别出缩进的代码在条件块之下。

```py
if condition: # -> colon is used here
    # code body

else: # -> colon is used here
    # code body

```

### 7.为了增加函数的可读性

这在某种程度上推进了 python 主题。初学者可以简单地忽略这个。就像在静态类型编程语言中，我们需要指定变量和函数的数据类型和返回类型一样，Python 也允许相同的情况，但使用不同的语法:

假设我们声明了一个函数，我们需要显式地提到数据类型。有一个简单的方法可以做到:

1.  在函数内部声明参数时，使用带冒号的数据类型，然后是参数名。
2.  然后，为了说明函数返回的内容，使用箭头运算符(->)在括号后插入数据类型。

```py
def add(a : int, b : int)->int:
    c = a + b
    print(c)
    return c

add(3, 4)

# outputs 7

```

参数名和数据类型之间有一个冒号。

### 8.用于声明类

Python 是一种面向对象的语言。因此，要声明类，我们需要使用冒号。冒号决定了变量的范围和类的功能。这是为了通知解释器一个类的实体位于冒号下面。这里有一个简单的例子:

**代码:**

```py
class Sample:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def printVal(self):
        print(self.a, self.b)

sample = Sample(3, 4)
sample.printVal()  

# outputs: 3, 4

```

因此，在这个类中，我们插入一个冒号。之后缩进四个空格。这将确保所有内容都在类范围内。所以，为了让事情更清楚，我们可以声明一个构造函数 **__init__()** 方法和 **printVal()** 方法。

### 使用冒号时出现的错误和常见错误

当我们声明一个函数或者一个循环或者任何需要冒号的代码块时，有一条重要的规则我们需要遵守。如果我们做不到这一点，事情就会出错，代码最终会给出一个错误。

***我们给冒号的时候，总要记得给一个缩进/空格。这定义了该父代码下的进一步代码的范围。***

现代代码编辑器有内置的自动缩进设置。但是，在使用记事本时，我们需要更加小心。

**示例–使用函数:**

```py
def sayHello():
print("Hello world")

sayHello()

```

**示例–for 循环:**

```py
for i in range(0, 3):

print(I)

```

**示例–如果有条件:**

```py
if (i % 2 == 0):
print(i)

```

**输出:**

```py
IndentationError: expected an indented block

```

## 结论

因此，这样，我们可以认为冒号是 Python 语法树的重要部分或核心部分。记住，如果我们在代码中漏掉了哪怕一个冒号，我们都会陷入很大的麻烦。所以，如果你是 Python 编程的新手，我推荐你认真阅读这篇文章。