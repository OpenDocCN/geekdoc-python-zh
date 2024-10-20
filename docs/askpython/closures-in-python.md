# Python 中的闭包——实用参考

> 原文：<https://www.askpython.com/python/oops/closures-in-python>

在本教程中，我们将看到 Python 中的闭包是什么，它们何时存在，以及我们如何使用它们。

为了理解闭包的概念，我们需要理解一些基本的概念，比如嵌套函数和自由变量。

然后我们会看到闭包在 Python 中的实现，闭包存在的条件，以及使用闭包的好处。

## Python 中的嵌套函数是什么？

Python 中的嵌套函数是定义在另一个函数内部的函数。下面是一个嵌套函数的例子。

这里 **nested_function()** 定义在 **outer_function()** 的局部作用域内，除非在函数调用时被 **outer_function** 返回，否则只能在同一作用域内调用。

```py
#nested function example

def outer_function():
    x=10
    print("It is outer function which encloses a nested function")
    def nested_function():
        print("I am in nested function and I can access x from my enclosing function's scope. Printing x")
        print(x)
    nested_function()
#Execution
outer_function() 

```

输出:

```py
It is outer function which encloses a nested function
I am in nested function and I can access x from my enclosing function's scope. Printing x
10

```

我们可以看到嵌套函数可以从它的封闭范围访问变量。当 **outer_function** 被调用时，它定义 **nested_function** 并最终调用它打印 x 的值

需要记住的重要一点是，函数是 python 中的第一类对象；即函数可以作为参数传递，从其他函数返回，并赋给任何变量。

## 什么是自由变量？

一个变量只能在它被定义的范围内被访问，也就是说，如果我们在一个函数或块内声明一个变量，那么它只能在那个函数或块内使用。否则将会出现名称错误。

当一个变量在一个没有定义的函数或代码块中使用时，这个变量被称为自由变量。

在上面的例子中, **x** 是自由变量。这里 **nested_function** 可以引用 **x** ，因为一个函数可以访问定义它的作用域中定义的变量。

## Python 中的闭包是什么？

*Python 中的闭包用于[面向对象编程](https://www.askpython.com/python/oops/object-oriented-programming-python)，通过它，嵌套函数记住并访问定义它的函数范围内的变量。*

闭包在实现中使用嵌套函数和自由变量。

执行嵌套函数时，外部函数不必是活动的*，即*外部函数范围内的变量可能不在内存中，但嵌套函数可以访问它。

这样，数据被附加到代码上，甚至不存在于内存中，然后被嵌套函数使用。

## Python 中闭包存在的条件是什么？

从上面的描述中，我们可以很容易地发现 Python 中闭包的存在。

*   我们需要嵌套函数。
*   嵌套函数需要引用其外部作用域(即外部函数)中定义的变量。
*   闭包存在的第三个也是最重要的条件是外部函数必须返回嵌套函数。

## Python 中闭包的例子

让我们看一个 Python 中闭包的例子。假设我们希望有一个函数对传递给它的数字进行计算并打印结果。

```py
#closure example
def generate_number():
    print("I am in generate_number function and will return the inner_function when called")
    x=999
    y=100
    def inner_function(number):
        result=(number*x)%y
        print("I am in inner function and printing the result")
        print(result)
    return inner_function

#execute
print("Calling generate_number")
do_something = generate_number()
print("Calling do_something")
do_something(77)

```

输出

```py
Calling generate_number
I am in generate_number function and will return the inner_function when called
Calling do_something
I am in inner function and printing the result
23

```

在上面的例子中，

*   函数 **generate_number()** 被定义，它有两个变量和一个函数 **inner_function** 被定义在其作用域内。
*   **inner_function** 可以访问函数 **generate_number** 范围内的变量 **x** 和 **y** 。它执行计算并打印结果。
*   现在在执行过程中，当我们调用 **generate_number()** 函数时，它完成它的执行并将 **inner_function** 返回给变量 **do_something** 。
*   此时， **generate_number** 的执行结束，其作用域从内存*中被清空(参见 [Python 垃圾收集](https://www.askpython.com/python-modules/garbage-collection-in-python) )* 。
*   现在 **do_something** 变量开始作为一个函数。
*   当我们调用这个函数时，它执行 **inner_function** 并打印结果。

这里需要注意的是， **inner_function** 正在执行，而 **generate_number** 已经完成了它的执行。

因此，变量 **x** 和 **y** 不在内存中，内部函数仍然可以使用这些变量。

这表明**数据已经附加到代码而不是内存中。**这就是闭包的本质。

## 为什么要在 Python 中使用闭包？

如果我们想避免使用全局变量，可以使用 Python 中的闭包，因此可以用于数据隐藏。当装饰器被实现时，闭包的一个非常好的用法就完成了。

## 结论

好了，今天就到这里。我们涵盖了很多 Python 的基础和高级教程，以满足您的需求。如果你是初学者，试试这篇 Python 初学者教程。快乐学习！🙂