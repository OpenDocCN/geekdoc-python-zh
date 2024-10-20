# Python 3.10 中的案例/开关

> 原文：<https://www.blog.pythonlibrary.org/2021/09/16/case-switch-comes-to-python-in-3-10/>

**Python 3.10** 正在增加一个叫做结构模式匹配的新特性，这个特性在 [PEP 634](https://www.python.org/dev/peps/pep-0634/) 中有定义，在 [PEP 636](https://www.python.org/dev/peps/pep-0636/) 中有关于这个主题的教程。结构化模式匹配为 Python 带来了 case / switch 语句。新语法超越了某些语言用于 case 语句的语法。

本教程的目的是让你熟悉 Python 3.10 中可以使用的新语法。但是在您深入了解 Python 的最新版本之前，让我们回顾一下在 3.10 发布之前您可以使用什么

## 3.10 之前的 Python

Python 一直有几种解决方案，可以用来代替 case 或 switch 语句。一个通俗的例子就是用 Python 的`if` - `elif` - `else`就像这个 [StackOverflow 回答](https://stackoverflow.com/a/60236/393194)中提到的。在该答案中，它显示了以下示例:

```py
if x == 'a':
    # Do the thing
elif x == 'b':
    # Do the other thing
if x in 'bc':
    # Fall-through by not using elif, but now the default case includes case 'a'!
elif x in 'xyz':
    # Do yet another thing
else:
    # Do the default
```

这是使用 case 语句的一个非常合理的替代方法。

你会在 [StackOverflow](https://stackoverflow.com/a/30881320/393194) 和其他网站上找到的另一个常见解决方案是使用 Python 的字典来做这样的事情:

```py
choices = {'a': 1, 'b': 2}
result = choices.get(key, 'default')
```

还有其他解决方案在字典内部使用 lambdas 或者在字典内部使用函数。这些也是有效的解决方案。

在 Python 3.10 发布之前，使用`if` - `elif` - `else`很可能是最常见的，也通常是最可读的解决方案。

## 结构模式匹配入门

Python 新的结构模式匹配使用了两个新的关键字:

*   **匹配**(不是开关！)
*   **案例**

要了解如何使用这段代码，请看下面基于 Guido 的教程的例子:

```py
>>> status_code = 400
>>> match status_code:
...     case 400:
...         print("bad request")
...     case 200:
...         print("good")
...     case _:
           print("Something else bad happened")
bad request

```

这段代码接受 status_code，并告诉 Python 将其与其中一种情况进行匹配。如果大小写是 _(下划线)，则没有找到大小写，这是默认大小写。最后一个 case 语句有时被称为“失败”情况。

## 组合文字

通过组合要比较的文字，可以稍微简化 case 语句。例如，您可能想要检查模式 status_code 是否匹配多个文字。为此，您可以像这样修改代码:`case 400|401|403`

这里有一个完整的例子:

```py
>>> status_code = 400 
>>> match status_code: 
...     case 400|401|403 : 
...         print("bad request") 
...     case 200: 
...         print("good")
...     case _:
            print("Something else bad happened") bad request
bad request
```

是不是很酷？

## 包扎

结构模式匹配是一个激动人心的新特性，仅在 Python 3.10 和更新版本中可用。这是一个强大的新功能，有很多有趣的用途。这些用例可以用 Python 现有的特性来解决吗？可能吧，但这让事情变得更简单了！

## 相关文章

*   Guido 的[模式匹配教程](https://github.com/gvanrossum/patma/blob/master/README.md#tutorial)
*   当[模式匹配被添加到 Python 3.10](https://brennan.io/2021/02/09/so-python/) 中时，堆栈溢出用户欢欣鼓舞