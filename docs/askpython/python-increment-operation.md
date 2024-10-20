# Python 增量运算

> 原文：<https://www.askpython.com/python/examples/python-increment-operation>

如何执行 Python 增量操作？如果您来自 C++或 Java 之类的语言，您可能想尝试将类似的增量功能扩展到 Python。

但是，正如我们将在本文中看到的，这并不完全相同。让我们看看如何在 Python 中尝试使用 Increment ( `++`)操作的类似功能。

* * *

## Python 增量

在讨论确切的区别之前，我们先来看看如何在 Python 中增加一个[变量](https://www.askpython.com/python/python-variables)。

下面的代码展示了几乎所有程序员如何在 Python 中递增整数或类似的变量。

```py
>>> a = 10
>>> print(a)
10

>>> a += 1
>>> print(a)
11

>>> a += 100
>>> print(a)
111

>>> b = 'Hello'
>>> b += 1
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can only concatenate str (not "int") to str

>>> b += ' from AskPython'
>>> b
'Hello from AskPython'

```

我们已经在连续的步骤中增加了整数变量`a`。此外，由于`+` [操作符](https://www.askpython.com/python/python-operators)也代表字符串的连接，我们也可以在适当的位置追加到字符串中！

我们可以使用`a++`将`a`后加 1 吗？

```py
>>> a++
  File "<stdin>", line 1
    a++
      ^
SyntaxError: invalid syntax

```

这里有一个问题。Python 的设计不允许使用`++`“操作符”。在 C++ / Java 中被称为增量运算符的`++`术语在 Python 中没有一席之地。

## Python 中为什么没有++运算符？

如果你想更详细地理解这一点，你需要有一些编程语言设计的背景。

Python 中不包含`++`操作符的选项是一个*设计决策*。负责用 Python 语言创建要素的人认为没有必要引入 CPP 风格的增量运算符。

当 Python 解释器从我们的输入中解析`a++`符号时，它以如下方式被解释:

*   由于二元运算符`+`是加法运算符，`a++`将被视为`a`、`+`和`+`。但是 Python 希望在第一个`+`操作符后有一个数字。因此，它会在`a++`上给出一个语法错误，因为第二个`+`不是一个数字。

类似地，预增量`++a`，将被这样处理:

*   Python 中的一元`+`运算符指的是 identity 运算符。这只是返回它后面的整数。这就是为什么它是整数上的一个*恒等运算*
*   比如`+5`的值简单来说就是`5`，对于`+-5`来说就是`-5`。这是一个一元运算符，适用于实数
*   `++a`将被解析为+和`+a`，但是第二个`+a`再次被处理为`(+a)`，也就是简单的`a`
*   因此，`+(+(a))`简单地计算为`a`。

因此，即使我们想将`a`的值增加 1，我们也不能使用`++`符号来实现，因为这种运算符并不存在。

因此，我们必须使用`+=`操作符来完成这种增量。

```py
a += 1
a -= 1

```

同样的逻辑也适用于减量操作。

## +=运算如何求值？

你可能认为既然有一个`=`符号，它可能是一个赋值语句。

但是，这不是一个常规的赋值语句。这被称为**扩充赋值语句**。

在常规赋值语句中，先计算右边，然后再把它赋给左边。

```py
# 2 + 3 is evaluated to 5, before assigning to a
a = 2 + 3

```

但是，在这个扩充的赋值语句中，先计算左侧，然后再计算右侧。这样做是为了将更新后的值就地写入左侧*。*

```py
*# Reads the value of a, before adding 3 to it in-place
a += 3* 
```

*这是增加变量的唯一方法，不需要使用像`a = a + 1`这样的重新赋值语句。但是在这里，总的来说，选项并不重要，因为解释器会在运行时优化代码。*

* * *

## *结论*

*在本文中，我们了解了如何在 Python 中使用 increment 操作，以及为什么不支持`++`和`--`操作符。*

## *参考*

*   *关于在 Python 上使用增量的 StackOverflow 问题*
*   *[Python 文档](https://docs.python.org/3/reference/simple_stmts.html#augmented-assignment-statements)关于赋值语句*

* * *