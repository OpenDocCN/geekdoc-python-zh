# 学习 Python 编写有效代码的初学者技巧

> 原文：<https://www.pythoncentral.io/python-starter-tips/>

## **Python 入门**

如果你是 Python 的新手，你可能会发现这篇文章非常有用。在这里，您将了解一些常见的 Python 入门提示和技巧，它们将使您能够编写简单高效的代码。

使用 Python 3 . 5 . 2 版本创建了本教程。

### **列表初始化**

列表是 Python 中最常用的数据结构之一。如果你在过去已经声明了列表，你应该会这样做:

```py
alist = list()

  (or) 

  alist = []
```

如果你想将一个列表初始化为五个 0，你应该这样做:

```py
alist = [0,0,0,0,0]
```

上面提到的方法对于短列表来说已经足够好了，但是如果你想将一个列表初始化为 20 个 0 呢？键入 0 二十次不是一种有效的方式。所以相反，你可以这样写:

```py
alist = [0] * 20

print(alist)

Output: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
```

### **Randint 命令**

通常你会被要求生成随机数。Python 通过引入 randint 命令使这变得非常容易。该命令从您指定的范围内随机选择一个数字，使该过程快速而简单。R andint 需要从‘随机’库中导入，其中 可以这样做:

```py
from random import randint 
```

使用 randint 的sy 税为:

```py
randint(<from_range>,<to_range>)
```

举个例子，如果你想打印 0 到 9(包括 0 和 9)之间的随机整数，可以这样做:

```py
from random import randint 

print(randint(0, 9))

Output: prints an integer between 0 and 9
```

注意，每次执行上面的命令，都会得到一个 0 到 9 之间不同的整数。

### **键入命令**

当你接收来自用户的输入或者处理来自其他程序的输入时，知道你正在处理的输入的数据类型是非常有用的。这使您可以更好地控制可以执行的操作。 类型 命令标识变量的数据类型。

语法如下:

```py
type(<variable_name>)
```

例如，如果你有一个名为 的列表变量 ，它是一个列表，那么执行下面的命令将返回:

```py
alist = list()

type(alist)

Output: <class 'list'>
```

### **剥离命令**

这是一个非常有用的命令，用于格式化以字符串形式接收的输入。 strip 命令删除字符串前后的空格。语法是:

```py
<string>.strip()
```

比如你想去掉一个字符串前后的空格，应该这样做:

```py
sample = “   Python “

sample.strip()

Output: Python
```

注意:只有前后的空格被删除，而不是两个单词之间的空格。例如:

```py
sample = “      I love Python       “

sample.strip()

Output: “I love Python” 
```

使用 lstrip/rstrip 命令可以执行更多的操作，例如分别在字符串的左侧/右侧进行剥离。一定要靠自己进一步探索！

### **计数**

我相信你会熟悉使用 作为 关键字的向前计数，它看起来像这样:

```py
for i in range(0,5):

print(i)

Output: prints 0,1,2,3,4
```

然而，还有更多以 为 的关键词。它还能让你一步一步地数，甚至倒着数。语法 如下:

```py
 for i in range (<from_value>,<to_value>,<step>):

print(i)
```

例如，如果你想计算 0 到 10 之间的每一秒的数字，应该这样写:

```py
 for i in range(0,10,2):

print(i)

Output: prints 0,2,4,6,8
```

上面的命令将打印 0，2，4，6，8。请注意，如果没有指定步长，默认情况下步长为 1。

反向计数，命令如下:

```py
for i in range(10,0,-1):

print(i)

Output: prints 10,9,8,7,6,5,4,3,2,1
```

如果步骤被指定为-2，那么上面的命令将打印 10，8，6，4，2。

### **一般提示**

其他一些通用提示:

*   注释——尽可能地写注释，因为这将有助于你和他人更好地理解代码。单行注释可以这样写:

```py
#<Single line comment>
```

段落注释可以这样写:

```py
“”” 

<Paragraph comment>

“””
```

*   命名惯例——在命名变量时，要格外小心使用相关的名称。当其他人阅读您的代码时，在名称中指定数据类型会非常有用。如果您正在初始化一个名为“first name”的字符串，请确保在名称中包含数据类型，如“strFirstName”。这样，任何阅读您的代码的人都会立即理解变量“strFirstName”是 string 类型的。
*   行号——如果您使用 Python IDLE，请确保使用 ALT+G 命令到达特定的行。这有助于根据行号跟踪一行，当您有几行代码并抛出错误时，这是一个救命稻草。

### **结论**

这些是从 Python 中精选出来的一些技巧，希望它们能让你的编码变得简单一些。还有很多东西需要学习，因为 Python 有大量的库，使得编码变得非常简单。所以继续探索，快乐的蟒蛇！