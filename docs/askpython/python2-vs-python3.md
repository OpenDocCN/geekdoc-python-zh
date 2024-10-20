# Python2 与 python 3——简单比较

> 原文：<https://www.askpython.com/python/python2-vs-python3>

你好，初学者！你一定听说过 Python2 对 Python3，有些人用 2 版本，有些人用 3 版本。今天就让我们了解一下 Python 两个版本的区别。

## Python2 和 Python3 的主要区别

让我们通过理解一些最常用的函数以及它们在两个版本中的不同之处，来了解 Python 2.x 和 Python 3.x 的区别。

### 1.`print`声明

| **Python 版本** | **语法** |
| Python2 | 打印“我是 Python2 版” |
| Python3 | 打印(“我是 Python3 版”) |

通常，上述两种语法的输出是完全相同的。但是 Python3 中括号的使用使得用户更容易阅读。

### 2.`input`声明

所有程序都需要用户输入，只有在这里把它添加到列表中才有意义。让我们看看如何在 Python2 和 Python3 中使用输入法。

| **Python 版本** | **语法** |
| Python2 | raw_input():用于字符串
input():用于整数 |
| Python3 | input():用于所有需要的输入 |

### 3.`variables`在打印报表时

我们如何在 Python2 和 Python3 之间使用格式字符串方法在[打印语句](https://www.askpython.com/python/built-in-methods/python-print-function)中使用变量？

| **Python 版本** | **语法** |
| Python2 | msg = "Hello"
print("输入的消息是% " % msg) |
| Python3 | msg = "Hello"
print("输入的消息是{0} ")。格式(消息)) |

### 4.错误处理

在 python3 中，程序员需要在 [`except`块](https://www.askpython.com/python/python-exception-handling)中添加`as`作为额外的关键字。

| **Python 版本** | **语法** |
| Python2 | 尝试:
//代码
除<错误>，错误:
//代码 |
| Python3 | 试:
//代码
除<错误>为 err:
//代码 |

### 5.Python 中的除法运算

在 Python2 的情况下，除法运算产生一个整数。另一方面，Python3 在[除法运算](https://www.askpython.com/python/examples/python-division-operation)后返回一个浮点值。

### 6.迭代函数

在 Python2 中，`[xrange()](https://www.askpython.com/python/built-in-methods/python-xrange-method)`用于迭代，而在 Python3 中，新的高级[函数`range()`用于迭代。](https://www.askpython.com/python/built-in-methods/python-range-method)

## Python2 和 Python3 哪个好？

现在大多数开发者都在创建与 Python 3 严格兼容的库。它也比 Python2 更容易编码和理解

此外，在 Python3 中，字符串以 Unicode 的形式存储，这比 Python2 中使用的 ASCII 码更加通用。最后，Python3 消除了开发冲突，因为它允许[输入 Python2 不支持的](https://www.askpython.com/python/python-data-types)。

除此之外，Python 3 支持所有现代编程，如人工智能、机器学习和数据科学概念。

简单来说: **Python2 是过去，Python3 是未来！**

## 结论

谈到更喜欢哪个版本，Python2 还是 Python3，我们可以断定 Python 3 是直接的赢家。另外，如果你是一个新的程序员，我会建议你选择 Python3。