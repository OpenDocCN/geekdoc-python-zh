# Python 中的 Currying 初学者入门

> 原文：<https://www.askpython.com/python/examples/currying-in-python>

在本文中，我们将尝试理解" ***库里** "* 的概念，它的优点，以及它在 python 中的实现。Currying 实际上是为了纪念数学家和逻辑学家 Haskell Curry 而命名的。它是功能设计模式之一。它主要用于解决问题和根据数学函数概念设计的程序。

## 什么是设计模式？

设计模式为常见或重复出现的问题提供了标准解决方案。设计模式的使用非常方便，有助于开发人员提高他们正在处理的代码的可读性。

## 什么是 Currying？

使用这样一种功能设计模式主要用于将带有多个参数的函数简化为一个函数链，每个函数链都带有一个参数。

**例如:**

```py
function_mult(1,2,3) ---> function_mult(1)(2)(3)

```

请考虑左边的函数，它执行乘法，有三个参数 1、2 和 3，然后根据所有三个参数的值生成输出。

在执行 Currying 之后，该函数被改变为具有单个参数的函数，该函数接受第一个参数(这里是 1)并返回接受第二个参数(这里是 2)的新函数，该函数然后再次返回接受第三个参数(这里是 3)的新函数，然后产生最终输出。

这就是 currying 如何将一个具有多个参数的函数变成一个由单个参数的多个函数组成的链。

## 我们如何在 Python 中实现 Currying？

为了理解 currying 的实现，让我们首先定义一个具有多个参数的函数。

考虑下面的函数，它对所提供的参数执行乘法运算。

```py
def mult(x, y, z):
  return x * y * z

ans = mult(10, 20, 30)
print(ans)

```

**输出:**

Six thousand

currying 的第一步是将多个参数绑定在一起。考虑函数有 *n* 个参数，我们需要绑定所有这些参数，为此我们用第一个参数固定函数，并创建一个接受(n–1)个参数的新函数。现在我们继续创建新的函数，直到函数的参数个数为 1。

在 python 中，我们使用 functools 中的标准 Python 函数 partial

```py
from functools import partial

mult_10 = partial(mult, 10)
mult_10_20 = partial(mult_10, 20)
print(mult_10_20(30))

```

**输出:**

Six thousand

## 使用装饰器涂抹

使用装饰器可以更有效地实现 Currying。装饰器将代码或功能包装在函数周围，以增强函数的功能。为此，我们使用不同的标准 python 函数。(*了解更多关于装修工 **[点击这里](https://www.askpython.com/python/examples/decorators-in-python)** )*

首先我们使用*签名*，它有助于记录传递给函数的参数数量。

*部分*函数在这里有助于将带有 *n* 个参数的函数派生为带有较少参数的函数

```py
from inspect import signature
def curry(func):

  def inner(arg):

    #checking if the function has one argument,
    #then return function as it is
    if len(signature(func).parameters) == 1:
      return func(arg)

    return curry(partial(func, arg))

  return inner

#using cuury function on the previous example function
@curry
def mult(x, y, z):
  return x * y * z

print(mult(10)(20)(30))

```

**输出:**

Six thousand

## 开始奉承

Curry 是一个功能性的设计模式。使用 curry 的主要目的是将给定的函数简化为一系列绑定函数，这使得开发人员的工作更加方便，并且使得解决方案更具可读性。在本文中，我们试图理解 Python 语言中 Curry 的工作和实现。

### 参考

*   [Stackoverflow 关于何时应该使用奉承的问题](https://stackoverflow.com/questions/24881604/when-should-i-use-function-currying)
*   [https://en.wikipedia.org/wiki/Currying](https://en.wikipedia.org/wiki/Currying)