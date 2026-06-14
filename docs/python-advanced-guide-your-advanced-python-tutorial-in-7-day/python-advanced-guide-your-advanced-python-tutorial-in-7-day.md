

# PYTHON 高级指南

Alec Dennis

# PYTHON 高级指南

你的高级 Python 教程
7 天掌握。从中级到高级的
分步指南。
（2022 速成课程）

Alec Dennis

# PYTHON 高级指南

第 1 章：面向对象编程 5

多态 7

封装 10

第 2 章 12

必备编程工具 bash 脚本

Python 中的正则表达式 14

Python 包管理器 16

版本控制 16

融会贯通 17

第 3 章 19

## 文件操作

- 创建新文件 20
- 二进制文件究竟是什么？ 22
- 打开你的文件 23

### 第 4 章 26

### 异常处理

- 处理零除错误异常 27
- 阅读异常错误回溯信息 31
- 使用异常防止程序崩溃 32

Else 代码块 34

静默失败 37

如何处理文件未找到异常错误 38

检查文件是否存在 38

Try 和 Except 39

练习

练习 40

结语 41

## 第一章
面向对象编程

我们现在将探讨面向对象编程的四个概念及其在 Python 中的应用。

### 继承

第一个主要概念被称为“继承”。这指的是一个对象从另一个对象派生的能力。以跑车为例。所有跑车都是车辆，但并非所有车辆都是跑车。此外，所有轿车都是车辆，但并非所有车辆都是轿车，而且轿车绝对不是跑车，尽管它们都是车辆。

基本上，这个面向对象编程原则指出，对象可以也应该被分解成尽可能小且精确的概念。

```python
class Vehicle(object):
    def __init__(self, makeAndModel, prodYear, airConditioning):
        self.makeAndModel = makeAndModel
        self.prodYear = prodYear
        self.airConditioning = airConditioning
        self.doors = 4
```

这在 Python 中是通过派生类来实现的。

假设我们创建了一个名为 SportsCar 的新类。
现在，构建一个名为 SportsCar 的新类，但不是从 object 派生，而是从 Vehicle 派生。

```python
class SportsCar(Vehicle):
    def __init__(self, makeAndModel, prodYear, airConditioning):
        self.makeAndModel = makeAndModel
        self.prodYear = prodYear
        self.airConditioning = airConditioning
        self.doors = 4
```

我们这里不需要 honk 函数；只需要构造函数。现在声明一辆跑车。我打算用法拉利。

```python
ferrari = SportsCar("Ferrari LaFerrari", 2016, True)
```

现在通过调用来测试这个

```python
ferrari.honk()
```

之后，保存并运行。一切应该顺利进行。
为什么会这样？这是由于继承的概念，它指出子类从父类继承函数和类变量。这个概念很容易理解。下一个概念有点更难。

### 多态

多态的概念是，根据情况的不同，相同的过程可以以不同的方式执行。在 Python 中，这可以通过两种方式实现：方法重载和方法重写。

重载方法意味着用不同的参数定义两次相同的函数。例如，我们可以为我们的 Vehicle 类提供两个不同的初始化程序。它目前假定一辆车有四个门。如果我们想指定车门的数量，我们可以在当前初始化函数下面添加一个新的初始化函数，带有一个 doors 参数，如下所示（较新的在底部）：

```python
def __init__(self, makeAndModel, prodYear, airConditioning):
    self.makeAndModel = makeAndModel
    self.prodYear = prodYear
    self.airConditioning = airConditioning
    self.doors = 4

def __init__(self, makeAndModel, prodYear, airConditioning, doors):
    self.makeAndModel = makeAndModel
    self.prodYear = prodYear
    self.airConditioning = airConditioning
    self.doors = doors
```

现在，有人在创建 Vehicle 类的实例时可以选择是否定义车门的数量。如果他们不定义，则假定车门数量为四。

方法重写发生在子类使用自己的代码覆盖父类的方法时。

例如，创建一个名为 Moped 的新类，它扩展自 Vehicle。将车门设置为零（这很荒谬），空调设置为假。唯一重要的参数是品牌/型号和制造年份。它应该看起来像这样：

```python
class Moped(Vehicle):
    def __init__(self, makeAndModel, prodYear):
        self.makeAndModel = makeAndModel
        self.prodYear = prodYear
        self.airConditioning = False
        self.doors = 0

    def honk(self):
        print("%s says: Beep! Beep!" % self.makeAndModel)
```

现在，如果我们创建一个 Moped 类的实例并调用 honk() 方法，它会鸣笛。但众所周知，轻便摩托车不鸣笛，它们发出哔哔声。所以让我们用我们自己的方法覆盖父类的 honk 方法。这非常简单。我们只需在子类中重新定义该函数：

我是 2.99 亿美国人中的一员，即使性命攸关也说不出轻便摩托车的品牌和型号，但你可以通过声明一个实例来自己测试这是否有效。

### 抽象

抽象是面向对象编程的下一个关键主题。这是指程序员和用户应该远离计算机内部工作原理的理念。这有两个好处。

第一个是它减少了固有的安全问题和灾难性系统故障的风险，无论是人为造成的还是其他原因。通过将程序员与计算机的内部工作（如内存和 CPU）隔离，在某些情况下甚至与操作系统隔离，造成不可修复损害的风险很低。

抽象的第二个好处是它自然地使语言更容易理解、阅读和学习。虽然它通过移除用户对完整计算机架构的一些控制而降低了语言的强度，但这换来了用该语言快速有效地编写代码的能力，而无需浪费时间处理内存地址之类的琐事。

这些在 Python 中是适用的，因为，嗯，这很简单。你无法深入计算机的细节，无法处理内存分配，甚至无法精确分配数组大小，但这是一种权衡，换来的是极高的可读性、在非常安全的环境中高度安全的语言，以及编程的易用性。比较以下 C 代码片段：

```c
#include <stdio.h>
int main(void) {
    printf("hello world");
    return 0;
}
```

与执行相同操作的 Python 代码：

```python
print("hello world")
## 就这样。仅此而已。
```

抽象对于当今生产的绝大多数程序来说通常是一个净优势，这就是为什么 Python 和其他面向对象编程语言如此受欢迎的原因。

### 封装

封装是面向对象编程中最后一个重要的概念。这是最容易解释的。这是指应该将共同的数据组合在一起，程序应该是模块化的。我不会详细说明，因为这是一个非常简单的概念。类是封装最简洁的例子：共同的特征和方法被绑定在一个连贯的结构下，使得创建该类型的对象非常容易，而无需为每个实例生成大量超级特定的变量。

所以，这就是全部了。我们终于到达了 Python 之旅的终点。首先，我想感谢你从头到尾阅读了《Python 初学者：Python 编程终极指南》。希望它内容丰富，并为你提供了实现目标所需的所有工具，无论目标是什么。

下一步是将这些知识付诸实践。你刚刚做出了人生中最明智的决定之一，学习了 Python 的基础知识，无论是作为爱好还是职业转变，你现在的目标应该是找到在日常生活中使用它的方法，让生活更轻松或完成你一直想做的任务。

## 第二章
#### 基本编程工具：Bash 脚本

Bash 脚本是一种数据文件，包含一系列命令，这些命令通常你可以手动输入，但使用脚本可以节省时间。请注意，在编程中，任何通常在命令行输入的代码都可以放入脚本中并原样执行。同样，任何可能包含在脚本中的代码通常也可以原样执行。

内存中可能同时运行着多个进程来体现一个程序。例如，你可以使用两个终端同时运行命令提示符。在这种情况下，系统将同时运行两个命令提示符进程。当它们执行完毕后，系统可以终止它们，之后就不会再有代表命令提示符的进程了。

你可以使用终端来运行 Bash 脚本，它会为你提供一个 shell。当你启动一个脚本时，它不会在当前进程中执行，而是会启动一个新的进程并在其中执行。然而，作为编程初学者，你不需要担心这个脚本的机制，因为运行 Bash 可能非常简单。

你可能还会遇到一些关于脚本执行的教程，这本质上是同一件事。在运行脚本之前，请确保它具有执行权限，否则软件会返回错误信息。

下面是一个 Bash 脚本示例：

```
#!/bin/bash
# declare STRING variable
STRING="Hello Python"
#print variable on a screen
echo $STRING
```

你可以使用 755 权限快捷方式来修改脚本，确保它可以与他人共享执行。

#### Python 中的正则表达式

正则表达式（RegEx）是一种指定文本字符串的模式，允许你构建用于管理、匹配和定位文本的模式。Python 是使用正则表达式的一种编程语言示例。正则表达式也可用于在文本编辑器和命令行中搜索文件中的文本。

当你第一次看到正则表达式时，你可能会认为它是一种新的编程语言。然而，如果你处理文本或需要解析大量数据，了解正则表达式可以为你节省无数小时。

在 Python 中，`re` 模块提供了完整的正则表达式功能。如果在使用或编译正则表达式时出现错误，它还会引发 `re.error` 异常。在 Python 中使用正则表达式时，你必须熟悉两个基本函数。但首先，你应该理解不同的字符在正则表达式中具有不同的含义。为了让你在使用正则表达式时不会感到困惑，我们将使用术语 `r'expression'` 来指代原始字符串（Raw Strings）。

`search` 和 `match` 函数是 Python 正则表达式中最重要的。

`search` 函数在字符串中查找正则表达式模式的第一次出现，可以包含可选的标志。`search` 函数包含以下参数（语法如下）：

-   字符串 - 将被搜索以匹配其中的模式
-   模式 - 要匹配的正则表达式
-   标志 - 可以使用按位运算指定的修饰符

如果成功，`re.search` 函数将返回一个匹配对象；否则，它将返回 `None` 对象。要发现匹配的表达式，请使用匹配对象的 `groups()` 或 `groups(num)` 函数。

这是一个使用 `search` 函数的代码示例：

```
import re

#Check if the string starts with "The" and ends with "Spain":
txt = "The rain in Spain"
x = re.search("^The.*Spain$", txt)

if (x):
    print("YES! We have a match!")
else:
    print("No match")
```

输出将是：

```
YES! We have a match!
```

同时，`match` 函数将尝试使用适当的标志将正则表达式模式与字符串匹配。`match` 函数的语法如下：

`match` 函数有以下可用参数：

-   字符串 - 将被搜索以匹配字符串开头的模式
-   模式 - 要匹配的正则表达式
-   标志 - 可以使用按位运算指定的修饰符

#### Python 包管理器

包管理器是编程中使用的工具，用于以有组织的方式自动化安装、配置、升级和卸载特定语言系统的程序。

它也被称为包管理系统，因为它处理包含软件名称、版本号、用途以及语言正常运行所需依赖项列表的数据文件的分发和归档。

当你使用包管理器时，元数据通常保存在本地数据库中，以防止代码不兼容和权限缺失。

Python 中的一个实用程序可用于查找、安装、升级和删除 Python 包。它还可以确定系统上安装的包的最新版本，并从远程或本地服务器升级现有包。

Python 包管理器不是免费的，只能通过 ActivePython 访问。它还使用存储库，即包含各种类型模块的预安装包集合。

#### 源代码控制

编程中的源代码控制（也称为版本控制或修订控制）管理代码的更改，这些更改通过称为修订号或简称修订的字母或数字代码来标识。例如，修订 1 指的是初始代码集，而修订 2 指的是第一次更改。每次编辑都会附带时间戳以及进行修改的人员的身份。修订是必要的，以便可以恢复或比较代码。

与团队合作时，源代码控制至关重要。你可以使用不同的视图将你的代码更改与其他开发者的代码更改合并，这些视图显示详细的更改，然后将正确的代码合并到主代码分支中。

无论你使用 Python 还是其他语言，源代码控制对于编码项目都至关重要。请记住，每个编码项目都应从使用源代码控制系统（如 Mercurial 或 Git）开始。

自编程诞生以来，出现了各种源代码控制技术。以前，专门的控制系统提供了针对大型编码项目和特定项目流程量身定制的功能。然而，现在无论你是在进行单个项目还是作为大型团队的一部分，都可以使用开源解决方案进行源代码控制。

在你早期的 Python 编程中，最好使用开源版本控制系统。你可以使用 Mercurial 或 Git，它们都是开源的，用于源代码控制分发。

Subversion 也可用，可用于集中系统以检查文件并减少合并冲突。

#### 综合运用

编程工具将使你的任务更轻松。本节介绍的工具将为你节省大量时间，使沟通更容易，并使你的编码更顺畅。总之，我们发现了以下几点：

-   Bash 脚本可以为你节省大量编码时间，同时使你的代码行更有条理、更易读。
-   正则表达式（RegEx）可以帮助你识别、搜索和匹配代码中的文本字符串，使你不必逐行浏览或自己评估每段代码。
-   包管理器将自动化系统，使你可以轻松安装、升级、配置或删除某些程序，这些程序将帮助你进行开发。
-   源代码控制对于维护修订至关重要，无论你是独自工作还是与团队合作，这样你就可以恢复更改或比较修订。

没有这些工具你仍然可以编写脚本，但它们会让你的生活更轻松。

## 第三章
### 处理文件

在使用 Python 时，我们需要关注的下一件事是确保我们知道如何处理和操作文件。你可能正在处理数据并希望保存它，同时确保它可供你以后根据需要拾取和使用。在如何保存数据、以后如何找到它以及它在代码中如何反应方面，你确实有多种选择。

当你处理文件时，你会看到数据保存在磁盘上，或者你可以在代码中根据需要重复使用。本章将教我们更多关于如何管理我们需要执行的一些工作，以确保文件正常运行等内容。

现在，我们将切换到Python语言的文件模式，这将为你提供更多选项。一个简单的思考方式是想象自己正在处理一个Word文档。你可能会在某个时候尝试保存你正在处理的文档之一，以免丢失，以便以后能找到。在Python中，这些类型的文件将是可比较的。然而，你不会像在Word中那样存储页面，而是保存代码的各个部分。

在处理文件时，你会看到有一些程序或方法可供选择。这些选项包括：

-   关闭你正在处理的文件。
-   创建一个新文件进行处理。
-   查找或重新定位文件到新位置，以便更容易找到。
-   在先前创建的文件上编写新的代码部分。

#### 创建新文件

我们首先要看的是生成一个文件。如果我们没有一个文件来协助我们，就很难完成许多其他任务。如果你想创建一个新文件并随后向其中添加代码，你必须首先确保文件在IDLE中打开。然后，在编写代码时，你可以选择要使用的模式。

在Python中生成文件时，你会看到有三个选项可供选择。我们将在这里介绍的三种基本模式是追加（a）、模式（x）和写入（w）。

当你希望打开一个文件并对其进行更改时，你应该使用写入模式。这是三种模式中最直接的一种。写入方法将使你更容易设置和运行代码的适当部分。写入函数将易于使用，并允许你对文件进行任何修改或更改。你可以向文件添加新信息，更改现有内容，并执行各种其他操作。如果你想看看使用写入方法可以对这部分代码做什么，请打开你的编译器并运行以下代码：

```python
#file handling operations
#writing to a new file hello.txt
f = open('hello.txt', 'w', encoding = 'utf-8' )
f.write("Hello Python Developers!")
f.write("Welcome to Python World")
f.flush()
f.close()
```

接下来，我们将讨论你可以对正在处理的目录做些什么。当前目录始终是默认目录。你可以浏览并修改存储代码信息的目录，但你必须在开始时这样做，否则它最终不会出现在你想要的位置。

无论你在编写代码时使用的是哪个目录，如果你想以后找到该文件，都需要返回到该目录。如果你希望它出现在不同的目录中，请确保在保存文件和代码之前将其移动到那里。使用我们上面提到的选项，当你转到当前目录（或你为本次尝试指定的目录）时，你将能够打开文件并查看你写入的消息。

我们为此编写了一小段代码。当然，随着时间的推移，你将开发更复杂的代码。对于这些脚本，有时你会想要更改或覆盖该文件中的部分信息。这在Python中是可行的，只需要对你使用的语法进行微小的调整。以下是一个你可以用这个做什么的例子：

```python
#file handling operations
#writing to a new file hello.txt
f = open('hello.txt', 'w', encoding = 'utf-8')
f.write("Hello Python Developers!")
f.write("Welcome to Python World")
mylist = ["Apple", "Orange", "Banana"]
#writelines() is used to write multiple lines into the file
f.write(mylist)
```

由于你只需要添加一个新行，前面的示例非常适合对先前处理过的文件进行微小修改。此示例不需要第三行，因为它只包含一些简单的单词，但你可以使用上面的语法并根据需要进行修改，向程序添加任何你想要的内容。

#### 什么是二进制文件？

在继续之前，我们应该考虑的另一件事是将代码中的一些文件和数据打印为二进制文件的想法。这可能看起来是一项艰巨的任务，但Python会帮助你。你只需要将数据转换为音频或图像文件而不是文本文件即可实现这一点。

Python允许你将任何代码转换为二进制文件。无论它过去是什么类型的文件都没有关系。但是，你必须确保正确处理数据，以便以后更容易以你选择的方式显示。确保这对你成功工作所需的语法如下：

```python
# write binary data to a file
# writing the file hello.dat write binary mode
f = open('hello.dat', 'wb')
# writing as byte strings
f.write("I am writing data in binary file!\n")
f.write("Let's write another list\n")
f.close()
```

如果你花时间在文件中使用此代码，它将帮助你创建所需的二进制文件。一些程序员选择采用这种策略，因为它有助于他们组织工作，并在需要时更容易检索信息。

#### 打开你的文件

到目前为止，我们已经处理了创建新文件并保存它，以及处理二进制文件。在这些示例中，我们介绍了一些处理文件的基础知识，以便你可以让它们为你工作，并在需要时访问它们。

既然你已经完成了这一部分，现在是时候学习如何打开文件并使用它，以及随时对其进行修改了。当你打开该文件时，再次使用它会容易得多。当你准备好检查打开和使用文件所需的步骤时，你需要以下语法。

```python
# read binary data to a file
# writing the file hello.dat write append binary mode
with open("hello.dat", 'rb') as f:
    data = f.read()
    text = data.decode('utf-8')
    print(text)
```

将此放入系统后，你将获得的输出如下：

```
Hello,world!
This is a demo using with
This file contains three lines
Hello world
This is a demo using with
This file contains three lines
Seeking out a file you need
```

最后，我们将看看如何找到你可能需要的一些文件，用于这种类型的编程语言。我们已经讨论了如何创建文件、如何以不同方式存储它们、如何打开和重写它们以及如何定位文件。然而，在某些情况下，你可以将其中一个文件移动到新位置。

例如，如果你正在处理一个文件，并且你发现事情没有按照你希望的方式显示，那么是时候改变这一点了。也许你没有正确拼写标识符的名称，或者目录不在你想要的位置，在这种情况下，查找选项可能是找到这个缺失文件并进行必要更改的最佳方法，以便以后更容易找到。

你将能够使用此方法修改文件的位置，确保它始终位于正确的位置，甚至在需要时更容易找到。你只需要使用类似于上面所示的语法来帮助你进行这些修改。

通过本章讨论的所有不同方法，你将能够在代码中执行各种操作。无论你是想创建一个新文件、更改代码、移动文件还是做其他事情，你都可以使用本章涵盖的代码来完成所有这些操作。

## 第四章
### 异常处理

异常处理等同于错误管理。它有三个功能。

1.  它使你能够调试程序。
2.  它使你的软件在遇到错误或异常时能够继续执行。
3.  它使你能够构建自定义错误，以辅助调试、移除和控制 Python 的一些怪异行为，并使你的程序按预期运行。

#### 处理零除错误异常

异常处理可以是一个简单或困难的过程，这取决于你希望程序如何运行以及你有多大的创造力。你可能对“创造力”这个词感到惊讶。编程不都是关于逻辑的吗？不是。

编程的主要目标是解决问题。问题的解决方案不仅需要推理，还需要想象力。你是否听过“跳出框框思考”这个说法？破坏程序的异常很不方便，通常被称为 bug。这类问题的解决方案往往难以捉摸。你必须找到解决方案，否则可能不得不从头开始重写应用程序。

例如，当你进行除法运算时，你有一个计算器软件，包含以下代码块：

```
>>> def div(dividend, divisor):
        print(dividend / divisor)
>>> div(5, 0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in div
ZeroDivisionError: division by zero
```

当然，除以零在数学上是不可能的操作。因此，当这种情况发生时，Python 会终止程序，因为它不知道你打算完成什么。它不知道任何有效的答案或回应。

话虽如此，这里的问题是错误会完全终止你的程序。你有两种选择来处理这个异常。首先，你可以确保这样的操作不会在你的程序中发生。其次，你可以让操作和错误发生，同时指示 Python 继续你的应用程序。

以下是第一种解决方案的样子：

```
>>> def div(dividend, divisor):
    if (divisor != 0):
        print(dividend / divisor)
    else:
        print("Cannot Divide by Zero.")
>>> div(5, 0)
Cannot Divide by Zero.
```

以下是第二种解决方案的样子：

```
>>> def div(dividend, divisor):
    try:
        print(dividend / divisor)
    except:
        print("Cannot Divide by Zero.")
>>> div(5, 0)
Cannot Divide by Zero.
```

请记住处理错误和异常的两个主要解决方案。一、首先避免犯错。二、处理错误带来的后果。

#### 使用 Try-Except 块

在前面的例子中，使用了 try except 块来管理错误。然而，你或你的用户仍然可能对你的解决方案造成破坏。例如：

```
>>> def div(dividend, divisor):
    try:
        print(dividend / divisor)
    except :
        print("Cannot Divide by Zero.")
>>> div(5, "a")
Cannot Divide by Zero.
```

为“except”块编写的语句不足以解释输入引起的错误。当用一个数除以一个字符串时，不会显示“Cannot Divide by Zero.”警告。

要使此工作正常，你必须首先了解如何正确使用 unless 块。首先，通过提供特定的异常，你可以描述它将收集和响应的错误。例如：

```
>>> def div(dividend,
            divisor):
    try:
        print(dividend / divisor)
    except ZeroDivisionError:
        print("Cannot Divide by Zero.")

>>> div(5, 0)
Cannot Divide by Zero.

>>> div(5, "a")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in div
TypeError: unsupported operand type(s) for /: 'int' and 'str'
```

现在提供了要处理的错误。当程序遇到给定的错误时，它将运行捕获该错误的“except”块中的语句。如果没有指定 except 块来捕获进一步的失败，Python 将介入，停止程序并抛出异常。

但这怎么可能发生呢？在示例中未指定错误时，它处理了所有情况。是的，这是正确的。当“except”块中没有定义要搜索的错误时，它将捕获任何错误。例如：

```
>>> def div(dividend, divisor):
    try:
        print(dividend / divisor)
    except:
        print("An error happened.")
>>> div(5, 0)
An error happened.
>>> div(5, "a")
An error happened.
```

如果你不知道可能遇到什么类型的错误，这是使用“unless”块的更好方法。

#### 阅读异常错误回溯

错误处理最关键的方面是了解如何阅读回溯消息。这并不难做到。回溯消息是：

```
<Traceback Stack Header>
<File Name>, <Line Number>, <Function/Module>
<Exception>: <Exception Description>
```

以下是你需要记住的事项：
根据回溯堆栈头，发生了一个错误。

- 文件名表示包含问题的文件的名称。因为书中的示例是使用解释器编码的，所以文件名始终是“stdin>”或标准输入。
- 行号指定文件中发生问题的具体行号。因为示例是通过解释器运行的，所以它总是说行。如果错误位于代码块或模块中，它将报告相对于代码块或模块的语句行号。
- 函数/模块部分指定哪个函数或模块负责该语句。如果代码块没有标识符或语句在代码块外声明，则默认值为 module>。
- 异常指定发生的错误类型。有些是内置类（例如，ZeroDivisionError、TypeError 等），而有些只是错误（例如，SyntaxError）。它们可以在你的 unless 块上使用。
- 异常描述提供了有关错误如何发生的额外信息。描述的格式可能因错误而异。

#### 使用异常防止崩溃

无论如何，你需要做的就是找出可以使用哪些异常，只需制造一个错误。例如，使用前面示例中的 TypeError，你也可以捕获该问题并用适当的句子响应。

```
>>> def div(dividend, divisor):
    try:
        print(dividend / divisor)
    except ZeroDivisionError:
        print("Cannot Divide by Zero.")
    except TypeError:
        print("Cannot Divide by Anything Other Than a Number.")
    except:
        print("An unknown error has been detected.")
>>> div(5, 0)
Cannot Divide by Zero.
>>> div(5, "a")
Cannot Divide by Anything Other Than a Number.
>>> div(undeclaredVariable / 20)
An unknown error has been detected.
```

然而，以这种方式捕获问题仍然可能很困难。它确实允许你避免事故或停止，但你不知道发生了什么。要了解未知问题，请使用 as 关键字将异常数据分配给一个变量。变量 detail 是此目的的常用约定。

例如：

```
>>> def div(dividend, divisor):
    try:
        print(dividend / divisor)
    except Exception as detail:
        print("An error has been detected.")
        print(detail)
        print("Continuing with the program.")
>>> div(5, 0)
An error has been detected.
Division by zero
Continuing with the program.
>>> div(5, "a")
An error has been detected.
unsupported operand type(s) for /: 'int' and 'str'
Continuing with the program.
```

#### Else 块

有时错误发生在代码块的中间。使用 try 和 except，你可以捕获错误。如果发生错误，你可能不想执行该代码块中的任何语句。例如：

```
>>> def div(dividend, divisor):
    try:
        quotient = dividend / divisor
    except Exception as detail:
        print("An error has been detected.")
```

#### 静默失败

静默失败，有时也称为悄无声息地失败，是一个在错误和异常处理过程中经常使用的编程术语。

静默失败是指程序在某个时刻失败，但从未向用户发出警报的状态。

静默失败是指解析器、运行时开发环境或编译器未能发出错误或异常，并继续执行程序的情况。这通常会导致不可预见的后果。

当程序员忽略或绕过异常时，他可能会导致静默失败。或者，他公开地隐藏异常并设计变通方法，以确保即使发生错误，程序也能继续正常运行。他这样做可能有多种原因，例如该错误不会破坏程序，或者用户不需要知道该问题。

#### 如何处理 FileNotFoundError 异常错误

你有时会遇到 FileNotFoundError。处理此类问题取决于你打开文件的目标或目的。以下是此错误的一些最常见原因：

- 你没有将目录和文件名作为字符串传递。
- 你拼错了目录和文件名。
- 你没有指定目录。
- 你没有包含正确的文件扩展名。
- 文件不存在。

处理 FileNotFoundError 问题的第一步是确保这些常见原因都不是罪魁祸首。一旦你完成了这一步，你将需要决定如何处理该问题，这完全取决于你最初访问文件的原因。

#### 检查文件是否存在

同样，处理异常有两种方法：预防性和反应性。预防性方法是首先确定文件是否存在。

你需要使用 Python 安装中自带的 `os` 模块来完成此操作。然后你可以使用 `path` 模块中的 `isfile()` 函数。`path` 模块的文件名由操作系统决定（UNIX 使用 `posixpath`，Windows 使用 `ntpath`，旧版 MacOS 使用 `macpath`）。例如：

```python
>>> from os import path
>>> path.isfile("random.txt")
False
>>> path.isfile("sampleFile.txt")
True
```

#### Try 和 Except

你也可以使用 `try`、`except` 和 `else` 块来完成此操作，虽然这比较麻烦。

```python
>>> def openFile(filename):
    try:
        x = open(filename, "r")
    except FileNotFoundError:
        print("The file '" + filename + "' does not exist.")
    except FileNotFound:
        print("The file '" + filename + "' does exist.")
>>> openFile("random.txt")
```

#### 创建新文件

如果文件不存在，并且你的目的是覆盖任何现有文件，你应该使用 "w" 或 "w+" 访问模式。如果文件不存在，访问模式会为你创建它。例如：

```python
>>> x = open("new.txt", "w")
>>> x.tell()
0
```

如果你打算读写，请改用 "w+" 访问模式。

#### 练习

尝试通过找到至少十个不同的异常来破坏你的 Python 程序。

然后创建一个循环。

在循环中创建十个语句，这些语句将在一个 try 块内生成十个不同的异常中的每一个。

每次循环迭代时，导致异常的语句之后的语句应导致另一个异常，依此类推。

为每个错误创建一个单独的 except 块。

#### 解答

#### 结论

感谢你一直坚持到最后！
你可以使用许多不同的编程语言，但对于大多数新程序员来说，Python 是最好的语言之一，它提供了你在刚开始使用这种编程语言时所寻求的强大功能和易用性。本指南介绍了 Python 的工作原理以及你可以用它完成的一些编码类型。

除了看到许多如何用 Python 编写代码和用这种语言构建你自己的一些程序的例子之外，我们还花了一些时间研究如何在机器学习、人工智能和数据分析领域使用 Python。这些是日益流行的技术主题和方面，许多程序员正试图了解更多关于它们的知识。而且，在本指南的帮助下，即使你是 Python 初学者，你也能够处理所有这些内容。
当你准备好了解更多关于如何使用 Python 编程语言以及如何确保你可以将 Python 用于数据分析、人工智能和机器学习时，请务必重新阅读本指南以开始学习。