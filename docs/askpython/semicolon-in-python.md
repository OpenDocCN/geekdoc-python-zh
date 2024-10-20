# Python 中的分号

> 原文：<https://www.askpython.com/python/examples/semicolon-in-python>

先说一下 Python 中分号的用法。分号(；)是结束或中断当前语句。

在像 C、C++和 Java 这样的编程语言中，使用分号是终止代码行所必需的。然而，Python 的情况并非如此。那么在 Python 编程中使用分号有什么不同吗？让我们找出答案。

* * *

## Python 中为什么允许分号？

Python 不需要分号来终止语句。如果您希望将多个语句放在同一行中，可以使用分号来分隔语句。

在 **Python** 中的一个**分号**表示分离，而不是终止。它允许你在同一行写多条语句。这种语法也使得在一条语句的末尾加上一个分号**成为合法的**。所以，实际上是两个语句，第二个是空的。

## 如何用 Python 打印分号？

让我们看看当我们试图在 Python 中将分号打印为常规字符串时会发生什么

```py
>>> print(";")

```

输出:

```py
;

```

它不加区别地对待分号并打印出来。

## 用分号分隔语句

现在，让我们看看如何使用分号来拆分 Python 中的语句。在这种情况下，我们将尝试在同一行中使用分号放置两个以上的语句。

语法:

```py
statement1; statement2

```

例如:

下面是 Python 中没有分号的三个语句

```py
>>> print('Hi')
>>> print('Hello')
>>> print('Hola!')

```

现在让我们使用同样的带有分号的三个语句

```py
print('Hi'); print('Hello'); print('Hola!')

```

输出:

```py
Hi
Hello
Hola!

```

正如您所看到的，在我们用分号将它们分开之后，Python 分别执行这三个语句。如果不使用它，解释器会给出一个 err.r

* * *

## 在 Python 中使用分号和循环

在像' [For loop](https://www.askpython.com/python/python-for-loop) 这样的循环中，如果整个语句以循环开始，并且您使用分号来形成像循环体一样的连贯语句，则可以使用分号。

示例:

```py
for i in range (4): print ('Hi') ; print('Hello')

```

输出:

```py
Hi
Hello
Hi
Hello
Hi
Hello
Hi
Hello

```

如果你使用分号来分隔一个普通表达式和一个块语句，比如循环，Python 会抛出一个错误。

示例:

```py
print('Hi') ; for i in range (4): print ('Hello')

```

输出:

```py
Invalid Syntax

```

* * *

## 结论

这就把我们带到了这个关于在 Python 中使用分号的简短模块的结尾。让我们用两点来总结本教程:

*   Python 中的分号主要用于分隔写在一行上的多个语句。
*   分号用于书写次要语句并保留一点空间——如 name = Marie 年龄= 23；打印(姓名、年龄)

使用分号是非常“非 pythonic 化”的，除非绝对必须使用，否则最好避免使用。