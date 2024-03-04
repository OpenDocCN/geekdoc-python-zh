# Python 字符串串联和格式化

> 原文：<https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python>

任何语言都需要完成的一个常见任务是合并或组合[字符串](https://www.pythonforbeginners.com/basics/strings)。这个过程被称为连接。

描述它的最好方式是，你把两个独立的字符串——由解释器存储——合并成一个。

例如，一个字符串是“hello”，另一个是“world”当你使用串联来组合它们时，它就变成了一个字符串，或者“hello world”。

这篇文章将描述如何在 Python 中连接字符串。有不同的方法可以做到这一点，我们将讨论最常见的方法。之后，我们将探索格式化，以及它是如何工作的。

### 串联

在 Python 中，有几种方法可以连接或组合字符串。创建的新字符串被称为字符串对象。显然，这是因为 Python 中的一切都是对象——这就是为什么 Python 是一种面向对象的语言。

为了将两个字符串合并成一个对象，可以使用“+”运算符。编写代码时，应该是这样的:

```py
*str1 = “Hello”*
*str2 = “World”*
*str1 + str2*
```

这段代码的最后一行是连接，当解释器执行它时，将创建一个新的字符串。

需要注意的一点是，Python 不能连接字符串和整数。这被认为是两种不同类型的对象。所以，如果你想合并两者，你需要把整数转换成字符串。

以下示例显示了当您尝试合并 string 和 integer 对象时会发生什么。它是从大卫·鲍尔的个人网站上复制过来的[。](http://davidbau.com/python/slides/slide6.html)

```py
*>>> print ‘red’ + ‘yellow’*
*Redyellow*
*>>> print ‘red’ * 3*
*Redredred*
*>>> print ‘red’ + 3*
*Traceback* *(most recent call last):*
*File “”, line 1, in*
*TypeError**: cannot concatenate ‘**str**’ and ‘**int**’ objects*
*>>>*
```

作为参考，“> > >”字符表示解释器请求命令的位置。

请注意，将字符串“red”乘以三次会得出值“redred”。这很重要，所以一定要记下来。

此外，我们可以看到，当试图结合'红色'和 3 解释抛出了一个错误。这是我们试图连接一个字符串和整数对象的地方，但是失败了。

通俗地说，字符串可以是任何记录的字符，但它最常用于存储单词和信息。另一方面，整数是没有小数点的记录数值**。Python 不能将单词和数字相加。从这个角度来看，错误发生的原因是有道理的。**

为了实现这一点，我们可以使用适当的函数将数字转换成字符串。其代码应该是这样的:

```py
*>>> print ‘red’ +* *str**(**3)*
*red3*
*>>>*
```

我们使用的方法是 *str* *(* *)* 函数。注意解释器是如何简单地将两个对象组合起来，并在被要求打印数据时将它们吐出来的？

### Python 中的字符串格式

在 Python 中，我们可以利用两种不同的字符串插值方法。

字符串插值是一个术语，用于描述对包含在一个或多个占位符中的字符串值进行求值的过程。简而言之，它帮助开发者进行[字符串格式化](https://www.pythonforbeginners.com/basics/strings-formatting)和连接。

希望您自己更熟悉这个术语，因为它是任何编程语言的关键元素，尤其是 Python。

### 使用%运算符的字符串格式

在我们深入了解之前，理解% string 运算符在 Python 3.1 及更高版本中将被弃用(不再使用)是很重要的。最终，它将从该语言的未来版本中完全删除。

然而，熟悉这种方法仍然是一个好主意，也是常见的做法。

理解如何使用操作符的最好方法是查看活动代码。

```py
*x* *= ‘apple**s**’*
*y* *= ‘lemon**s**’*
*z* *= “**In* *the basket are %s and %s” % (**x,y**)*
```

此示例代码将按照我们设置的顺序，用相应的字符串替换“%s”运算符值。当您打印“z”字符串对象时——在执行上述代码后——它将返回以下内容:

```py
*In the basket are apples and lemons*
```

### 用{ }运算符设置字符串格式

当您使用花括号或 **{}** 操作符时，它们充当您希望存储在字符串中的变量的占位符。为了将变量传递给字符串，你必须调用 ***格式(* *)*** 方法。

使用***(**)***格式的一个好处是，您不必在连接数据之前将整数转换成字符串。它会自动为你做到这一点。这就是为什么它是首选运算符方法的一个原因。

再一次，让我们来看一些展示这一点的代码:

```py
*Fname* *= “John”*
*Lname* *= “Doe”*
*Age = “24”* 
*p**rint “{} {} is {} years* *old.“* *format(**fname**,* *lname**, age)*
```

这样做的目的是获取适当的值，并将它们作为变量存储在相应的字符串中。

***格式(* *)*** 方法的另一个有用的特性是，您实际上不必按照您希望变量显示的顺序将输入提供给解释器，只要您像这样对占位符进行编号:

```py
*print “{0} {1} is {2} years old.”* *format(**fname**,* *lname**, age)*
```

### 在 Python 中使用 Join 方法

Python 中的 join 方法用于连接字符串列表。

例如

```py
*>>>* *‘* *‘* *.join([‘the’, ‘quick’, ‘brown’, ‘fox’, ‘jumps’, ‘over’, ‘the’, ‘lazy’, ‘dog’])*
*‘the quick brown fox jumps over the lazy dog’*
```

让我们创建一个新的名单，其中包括一些伟大的乐队，但这一次让我们做不同的事情。

```py
*>>> music = [“Metallica”, “Rolling Stones”, “ACDC”, “Black Sabbath”, “**Shinedown**”]*
```

这个代码片段将使用我们指定的变量创建一个名为“music”的字符串或列表。

您可以通过加入一个空格来加入新列表，如下所示:

```py
*>>> print* *‘ ’**.join(music)*
```

您也可以通过在新的一行开始代码来加入它，如:

```py
*>>> print “*
*“.join**(music)*
```

方法没有对错，用你喜欢的方法。

### 相关职位

[Python 字符串](https://www.pythonforbeginners.com/basics/strings)

[反转列表和字符串](https://www.pythonforbeginners.com/code-snippets-source-code/reverse-loop-on-a-list)

[字符串操作](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)