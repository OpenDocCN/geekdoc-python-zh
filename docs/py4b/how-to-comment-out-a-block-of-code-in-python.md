# 如何用 Python 注释掉一段代码

> 原文：<https://www.pythonforbeginners.com/comments/how-to-comment-out-a-block-of-code-in-python>

用 Python 编程是令人兴奋的。编写代码并与他人分享会带来令人惊奇的事情。但是在我们的程序成长之前，我们需要确保它们易于阅读。这就是程序员学习如何注释代码块的原因。

为什么让我们的代码可读很重要？简单的答案是，代码被阅读的次数比它被编写的次数多。为了确保我们的代码是可维护的，我们需要让其他人清楚发生了什么。

因此，注释是编写可读代码的必要部分。写注释让我们有机会解释一段代码是做什么的。我们用它们来解释程序中容易混淆或模糊的部分。

为了测试的目的，我们也可以使用注释来删除程序的一部分。通过屏蔽掉一行代码，我们可以防止它被编译。这允许我们测试替代逻辑或排除程序故障。

## 为什么写评论很重要

随着项目的增长，有必要进行扩展。这意味着更大的代码库和更大的团队。为了让团队正常运作，每个人都需要步调一致。

早期糟糕的选择会导致代码难以维护。如果没有注释来帮助破译代码，对于新开发人员来说，跟上速度可能是一个挑战。你不希望你的同事绞尽脑汁试图弄清楚一些糟糕命名的变量是做什么的。

即使你是一个人工作，以单人模式编写软件，在程序中加入一些注释也是一个好主意。为什么？很有可能，当你一两个月后回到这个项目时，你不会记得所有的事情是如何运作的。对于跨越多个文件的大型程序来说尤其如此。

## 什么是代码块？

一般来说，代码块指的是组合在一起的多个相似的代码。这可以包括几个陈述以及注释。在 Python 中，代码块位于相同的缩进级别。

#### 示例 1:在 Python 中识别代码块

```py
def print_upper_case(name):
    # these statements make up a single block
    upper = name.upper()
    print(upper)

print_upper_case("taylor") # outside of the block 
```

**输出**

```py
TAYLOR
```

在这个例子中，我们已经确定了位于 **print_upper_case()** 函数下面的代码块。这段代码以一个注释开始，后面是另外两个语句。底部的函数调用在前面提到的代码块之外。

### 我们真的需要多行注释吗？

有时候注释掉整个代码块会很有用。例如，如果我们需要对一部分代码进行故障诊断，并且我们想看看如果某个块不执行会发生什么。

在这种情况下，注释掉整个代码块会很方便。这样我们就不会丢失已经写好的东西。

此外，虽然保持评论简短是明智的，但有时我们需要不止一行来表达我们的需求。在这种情况下，最好使用块注释。

## 使用#来注释代码块

用 Python 注释掉一段代码最直接的方法是使用 **#** 字符。任何以 hashtag 开头的 Python 语句都将被编译器视为注释。

无论是在一行中还是在其他地方，你都可以有无限多的块注释。如果我们需要进行多行注释，这将非常有用。

#### 示例 1:编写消息

```py
def message_writer(msg):
    # We may need to do it this way in the future
    #new_msg = "Message Writer says: "
    #new_msg += msg
    #print(new_msg)

    print("Message Writer says: " + msg)

message_writer("Lovin' it!") 
```

**输出**

```py
Lovin' it!
```

在上面的例子中，我们使用了块注释对编译器暂时隐藏一些 Python 语句。通过在每个语句前添加一个 **#** ，我们有效地从代码中删除了它。

也许我们有一个开发人员使用的 Python 程序。这是一个系统的设置文件。根据他们的需要，可能需要从程序中删除一些代码行。使用块注释意味着我们可以给这些开发人员几个选项，他们可以通过简单地取消注释这些语句来有效地“打开”。

## 如何使用文档字符串进行块注释

虽然块注释在技术上允许我们进行多行注释，但是使用它们会很麻烦。如果代码块长于几行，尤其如此。添加和删除标签并不好玩。

幸运的是，有另一种方法可以在 Python 中创建多行注释。我们可以使用 docstrings(文档字符串)来实现这一点。

文档字符串允许我们快速注释掉一段代码。我们可以使用三重引号在 Python 中创建一个 docstring。这个方法得到了 Python 的创造者 Guido Van Rossum 的认可。这里有一个关于使用 docstrings 在 Guido 的 Twitter 页面上发表评论的引用:

> Python 提示:可以使用多行字符串作为多行注释。除非用作文档字符串，否则它们不会生成代码！🙂
> 
> *-Guido Van Rossum*

重要的是要记住，文档字符串并不是真正的注释，它们是没有赋给变量的字符串。未赋值的字符串在运行时会被忽略，所以在我们的目的中，它们将起到注释的作用。

然而，docstrings 在 Python 中还有另外一个用途。放在函数或类声明之后，docstring 将作为与该对象相关联的一段文档。在这种情况下，文档字符串作为一种快速生成 API 的方法(**应用程序**程序**接口 **I** )。**

## Python 中注释掉代码的替代方法

块注释和文档字符串是在 Python 中创建注释的唯一方法。如果你像我一样，这两种方法都不是你想要的。

幸运的是，我们不必完全依赖 Python 的工具。使用今天的技术，点击鼠标就可以创建多行注释。

### 使用 IDE 或文本编辑器

除非你是在记事本中写代码，否则你可能会使用一些工具来注释掉一段代码。虽然这个方法对 Python 来说并不明确，但它在现实世界中是一种常见的做法。

使用 IDE(**I**integrated**D**development**E**environment)，可以一次注释整行代码。在许多情况下，可以从工具栏访问这些特殊功能。这些特性将特定于每个 IDE，因此您需要根据您使用什么软件来编写 Python 代码进行一些研究。

大多数文本编辑器都有一个特性，允许你一次注释掉几行代码。如果您搜索一个专门为开发人员创建的文本编辑器，它可能会有这个特性。

## 在 Python 中使用多行注释的示例

为了帮助说明何时以及如何使用块注释和文档字符串，我们还包括了几个例子。

```py
def add_square(x,y):
    a = x*x
    b = y*y

    return a + b

print(add_square(2,2)) 
```

我们编写的 **add_spare()** 函数使用了一段代码，需要几行代码来完成所需的计算。这个代码块可以用一行代码重写。我们已经注释掉了旧的行，这样你就可以比较这两种方法。

```py
def add_square(x,y):
    #a = x*x
    #b = y*y
    #return a + b

    return (x*x) + (y*y)

print(add_square(2,2)) 
```

文档字符串用于定义函数、类或模块。我们可以使用 **__doc__** 属性读取文档字符串。

```py
def product(x,y):
    '''Returns the product of the given arguments.'''
    return x*y

x = 4
y = 3

print("{} x {} = {}".format(x,y,product(x,y)))
print(product.__doc__) 
```

**输出**

```py
4 x 3 = 12
```

另一方面，字符串看起来像文档字符串，但它们的工作方式不同。相反，当程序运行时，它们会被忽略。因此，我们可以将它们用作多行注释。

```py
"""
The Fibonacci series is a sequence of numbers where each
term is the sum of the previous two terms.

Starting with 0 and 1, we can compute the nth term in the sequence.
"""

def fib(n):
    a,b = 0,1
    for i in range(n):
        a,b = b,a+b
        print(a)
    return a

print(fib(10)) 
```

#### 用 Python 注释的技巧:

*   保持评论简洁明了。
*   始终保持尊重和乐于助人。
*   记住，最好的代码通过为变量、函数和类使用好的名字来解释自己。

## 摘要

虽然 Python 确实不像其他编程语言那样支持多行注释，但是使用 **#** 字符仍然可以创建这些类型的注释。块注释是在 Python 中创建多行注释的标准方式。它们也可以用来注释掉程序中的一段代码。

如果块注释不够，还可以使用 docstrings 创建多行注释。除非以特殊方式使用，否则文档字符串不会生成任何代码。

最后，可以使用现代 IDE 注释掉一段代码。这种方法将取决于您使用的软件，但它通常是一个相当简单的过程。一些程序甚至有热键专门用于注释和取消注释代码块。

## 相关职位

*   除了语句之外， [Python 如何尝试捕捉代码中的错误](https://www.pythonforbeginners.com/error-handling/python-try-and-except)
*   如何使用一个 [Python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)来存储键值对