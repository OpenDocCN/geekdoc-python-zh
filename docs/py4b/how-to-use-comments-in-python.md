# Python 中如何使用注释？

> 原文：<https://www.pythonforbeginners.com/comments/how-to-use-comments-in-python>

用 Python 写代码可能很有挑战性。这就是为什么在给变量和函数命名时，尽可能清晰是很重要的。一个好的名字会解释变量的用途，或者一个函数的作用。然而，好的命名约定只能让你到此为止。如果我们面对的是复杂的逻辑或令人困惑的算法呢？或者如果我们需要和另一个程序员分享和解释我们的意图呢？在代码中为自己或他人留下注释会很有帮助。这就是 Python 注释的用途。

有了注释，我们可以使我们的程序更容易理解，也更容易在以后进行修改。Python 注释会被解释器忽略，所以它们不会影响代码的运行。

#### 我们将涉及的主题:

1.  为什么写评论很重要
2.  如何使用注释改进您的 Python 代码
3.  用 Python 注释的最佳实践

## 什么时候写 Python 注释？

让我们面对现实吧，编程并不容易；我们都需要一些帮助。注释允许程序员添加一些额外的信息来解释代码中发生了什么。

如果你在做一个大项目，很难跟踪所有正在进行的事情。在长时间缺席后重返项目时尤其如此。

这种情况比你想象的更常见。程序员经常被要求同时处理许多项目。

回到一个项目往往是痛苦的。如果代码清晰，遵循样式指南，并利用恰当的注释，更新项目会容易得多。

当涉及到团队工作时，评论甚至更为重要。破译你自己的代码已经够难了，更别说是同事或陌生人写的了。遵循一些基本准则将有助于你写出清晰、有效的评论。

## 如何用 Python 写评论？

注释通常以块的形式编写，并解释其后的代码。通常，注释用于解释复杂的公式、算法或设计决策。

**定义**:注释是程序员可读的代码，被计算机忽略。这是给我们自己或其他从事项目的程序员的一个信息。

**用法**:使用注释来解释函数、算法、数学、设计选择或任何你将来可能需要参考的东西。

**语法**:Python 中的大多数注释都是使用 **#** 字符编写的。Python 在程序运行时会忽略任何跟在散列符号( **#)** 后面的代码。

Python 中有三种不同类型的注释:block、inline 和 docstring。知道何时使用每一种都很重要。例如，docstrings 可以用来自动生成项目的文档。

## 用 Python 编写单行注释

单行注释，在 Python 中称为*块*注释，以 **#** 字符开始。块注释用于解释注释后面的代码。

记住评论要尽可能的短。很容易做过头，在程序越多单词越清晰的印象下添加对所有事情的评论。那不是真的。

俗话说，简洁是智慧的灵魂。注释代码也是如此。保持简单是个好主意。

### 示例 1: Python 块注释

```py
# this is a block comment in python, it uses a single line

# a list of keywords provided by the client
keywords = ["tv shows", "best streaming tv","hit new shows"] 
```

### 例子 2:勾股定理

```py
import math

# the sides of a triangle opposite the hypotenuse
a = 2
b = 3

# the Pythagorean Theorem
c = math.sqrt(a**2 + b**2)

print(c) 
```

**输出**

```py
3.605551275463989
```

## **行内注释**

为了节省空间，有时把注释放在你描述的代码的同一行是个好主意。这就是所谓的*内联*注释。

行内注释也用 **#** 字符书写。它们被放在代码语句的末尾。当您想要包含有关变量名的更多信息，或者更详细地解释一个函数时，行内注释非常有用。

### 示例 3:用 Python 编写行内注释

```py
income = 42765 # annual taxable income in USD
tax_rate = 0.22 # 22% annual tax rate
print("You owe ${}".format(income*tax_rate)) 
```

## 用 Python 编写多行注释

在 Python 中，编写超过一行的注释的最直接方法是使用多个块注释。Python 会忽略散列符号( **#)** ，留给你的是多行注释。

### 示例 4:多行注释

```py
chessboard = {}
# we need to use a nested for loop to build a grid of tiles
# the grid of tiles will serve as a basis for the game board
# 0 will stand for an empty space
for i in range(1,9):
    for j in range(1,9):
        chessboard[i,y] = 0 
```

不幸的是，使用这种方法需要在每行之前添加一个 hashtag。

虽然从技术上来说你不能写一个超过一行的注释，但是 Python docstrings 提供了一个绕过这个规则的方法。

文档字符串意味着与函数和类相关联。它们用于为您的项目创建文档，因为 docstring 将与它们所在的函数相关联。

文档字符串使用两组三重引号来表示它们之间的文本是特殊的。这些三重引号应该放在函数和类之后。它们用于记录 Python 代码。

### 示例 5:使用带有函数的 docstring

```py
def string_reverse(a_string):
    """A helper function to reverse strings

    Args:
        a_string (sring):

    Returns:
        string: A reversed copy of the input string
    """
    reverse = ""
    i = len(a_string)
    while i > 0:
        reverse += a_string[i-1]
        i = i-1

    return reverse

# driver code
print(string_reverse("Is this how you turn a phrase?")) 
```

**输出**

```py
?esarhp a nrut uoy woh siht sI
```

我们还可以使用这些三重引号在 Python 中创建注释。因为它们是作为字符串读取的，所以运行时不会影响我们的程序。

```py
""" this is a multiline comment in Python

    it's basically just a string

    but you have to be careful when using this method.
""" 
```

建议阅读:要阅读更多计算机科学主题，您可以阅读这篇关于使用 ASP.net 的[动态基于角色授权的文章。你也可以阅读这篇关于使用 Asp.net](https://codinginfinite.com/dynamic-role-based-authorization-asp-net-core-assign-database/)记录[用户活动的文章。](https://codinginfinite.com/user-activity-logging-asp-net-core-mvc-application/)

## 为自己做评论

有些评论留给开发者自己。例如，你可以留下一个评论来提醒自己，你在之前的一个工作日没有完成的事情，你在哪里停下来了。

大型程序很难管理。当您打开一个项目时，尤其是如果它是您已经有一段时间没有查看的项目，您有时会发现您需要“重新加载”代码。本质上，你需要提醒自己该项目的细节。

在这方面，注释可以帮助节省时间。一些恰当的评论可以帮助你避免那些令人挠头的时刻，比如你问“我到底在这里做什么？”

## 为他人评论

无论何时你在一个团队中工作，无论是 Python 还是其他，交流都是至关重要的。这里的困惑只会导致浪费时间和沮丧。

这就是为什么程序员应该用注释来解释他们的代码。遵循风格指南并适当地评论你的作品将确保每个人都在同一页上。

如果你对机器学习感兴趣，你可以阅读这篇关于机器学习中[回归的文章。您可能还会喜欢这篇关于使用 python 中的 sklearn 进行](https://codinginfinite.com/regression-in-machine-learning-with-examples/)[多项式回归的文章。](https://codinginfinite.com/polynomial-regression-using-sklearn-module-in-python/)

## 用 Python 注释的最佳实践

虽然 Python 注释通常使程序更容易理解，但情况并非总是如此。如果一个人希望写出富有成效的评论，他应该遵守一些不成文的规则。

的确，添加更多的注释并不一定能提高代码的可读性。事实上，评论太多和太少一样糟糕。那么你怎么知道评论多少，什么时候评论，或者什么时候不评论呢？

**评论指南:**

1.  避免冗余
2.  要尊重
3.  少即是多

知道什么时候评论是不必要的是一项需要掌握的重要技能。通常，你会看到学生在变量中重复函数名，好像重新写一遍会让他们的意思更清楚。

不要做像这样不必要的评论:

```py
 # the name of the author
author_name = "Le Guin, Ursula K" 
```

让我们看一个更长的例子，它有多个注释和一个 docstring。使用注释导航冒泡排序算法。

### 示例 7:注释冒泡排序算法

```py
def BubbleSort(nums):
    """A method for quickly sorting small arrays."""
    total_nums = len(nums)

    for i in range(total_nums):
        for j in range(0, total_nums-i-1):
            # traverse the array backwards
            # swap elements that are out of order
            if nums[j] > nums[j+1]:
                temp = nums[j]
                nums[j] = nums[j+1]
                nums[j+1] = temp

nums = [100,7,19,83,63,97,1]

BubbleSort(nums)
print(nums) 
```

**输出**

```py
[1, 7, 19, 63, 83, 97, 100]
```

## 有用的链接

希望我已经给你留下了正确注释 Python 代码的重要性的印象。这会节省你的时间，使你的工作更容易。
想吃更多吗？通过点击这些链接到更多的 *Python 初学者*教程，提高你的 Python 编码技能。

*   [如何使用 Python 字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)
*   [了解 Python 列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)