# 为 Python 函数计时

> 原文：<https://www.pythoncentral.io/time-a-python-function/>

在之前的一篇文章中([在 Python 中测量时间- time.time() vs time.clock()](https://www.pythoncentral.io/measure-time-in-python-time-time-vs-time-clock/ "Measure Time in Python - time.time() vs time.clock()") )，我们学习了如何使用模块`timeit`对程序进行基准测试。因为我们在那篇文章中计时的程序只包括原始语句而不是函数，所以我们将探索如何在 Python 中实际计时一个函数。

## 不带参数的 Python 函数计时

模块函数`timeit.timeit(stmt, setup, timer, number)`接受四个参数:

*   `stmt`哪个是你要衡量的语句；它默认为“通过”。
*   `setup`是在运行`stmt`之前运行的代码；它默认为“通过”。
*   哪一次？计时器对象；它通常有一个合理的默认值，所以你不必担心它。
*   `number`这是您希望运行`stmt`的执行次数。

其中`timeit.timeit()`函数返回执行代码所用的秒数。

现在假设你想测量一个函数`costly_func`，实现如下:

```py

def costly_func():

   return map(lambda x: x^2, range(10))

```

您可以使用 timeit 来测量它的执行时间:

```py

>>> import timeit

>>> def costly_func():

...     return map(lambda x: x^2, range(10))

...

>>> # Measure it since costly_func is a callable without argument

>>> timeit.timeit(costly_func)

2.547558069229126

>>> # Measure it using raw statements

>>> timeit.timeit('map(lambda x: x^2, range(10))')

2.3258371353149414

```

请注意，我们使用了两种方法来测量这个函数。第一种方式传入 Python 可调用函数`costly_func`，而第二种方式传入原始 Python 语句`costly_func`。虽然第一种方式的时间开销比第二种方式稍大，但我们通常更喜欢第一种方式，因为它可读性更好，也更容易维护。

## 用参数计时 Python 函数

我们可以使用 decorators 来度量带有参数的函数。假设我们的`costly_func`是这样定义的:
【python】
def cost _ func(lst):
返回映射(lambda x: x^2，lst)

您可以使用如下定义的装饰器来度量它:
【python】
def wrapper(func，*args，* * kwargs):
def wrapped():
return func(* args，**kwargs)
return wrapped

现在您使用这个装饰器将带参数的`costly_func`包装成不带参数的函数，以便将其传递给`timeit.timeit`。

```py

>>> def wrapper(func, *args, **kwargs):

...     def wrapped():

...         return func(*args, **kwargs)

...     return wrapped

...

>>> def costly_func(lst):

...     return map(lambda x: x^2, lst)

...

>>> short_list = range(10)

>>> wrapped = wrapper(costly_func, short_list)

>>> timeit.timeit(wrapped, number=1000)

0.0032510757446289062

>>> long_list = range(1000)

>>> wrapped = wrapper(costly_func, long_list)

>>> timeit.timeit(wrapped, number=1000)

0.14835596084594727

```

## 计时来自另一个模块的 Python 函数

假设您在另一个模块`mymodule`中定义了函数`costly_func`，既然它不是本地可访问的，那么您如何测量它的时间呢？嗯，你可以把它导入到本地名称空间或者使用`setup`参数。

```py

# mymodule.py

def costly_func():

    return map(lambda x: x^2, range(1000))

```

```py

>>> timeit.timeit('costly_func()', setup='from mymodule import costly_func', number=1000)

0.15768003463745117

# OR just import it in the local namespace

>>> from mymodule import costly_func

>>> timeit.timeit(costly_func, number=1000)

0.15358710289001465

```

## 总结和提示

在本文中，我们学习了如何使用`timeit.timeit`测量 Python 函数的执行时间。通常，我们更喜欢将 Python 函数作为可调用对象导入并传递到`timeit.timeit`中，因为这样更易于维护。此外，请记住，默认的执行次数是 1000000，对于某些复杂的函数，这可能会大大增加总执行时间。