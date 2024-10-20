# Python 中的变量范围

> 原文：<https://www.askpython.com/python/variable-scopes-in-python>

变量范围是一个有趣、有用且易于理解的概念，python 程序员在处理变量和函数之前必须了解这个概念。在本教程中，我们将讨论什么是作用域，python 具有的作用域的类型，并且我们将理解关键字`global`和`nonlocal`。

## 什么是范围？

一个[函数](https://www.askpython.com/python/examples/functional-programming-introduction)、类或任何[用户定义变量](https://www.askpython.com/python/python-variables)的范围是可以访问它的代码区域。在函数内部声明的变量只能在函数内部访问，而不能在函数外部访问，所以函数就是变量的作用域。

例如在下面的代码中:

```py
def funcx():
    x = 1
    print(x)
    return

def funcy():
    y = 2
    print(y)
    return

def funcz():
    z = 3
    print(z)
    return

```

变量`x`在`funcx`中声明，所以`funcx`是变量的作用域。同样，`y`的范围是`funcy`，而`z`的范围是`funcz`。

## 理解全局、局部和非局部变量范围

在 python 中，每个变量都有一个变量作用域，也就是说，有一个明确定义的变量使用范围。根据变量的使用场合，变量可以有不同类型的作用域，我们一个一个来说。

### 1。全局变量范围

如果一个变量可以从 python 文件中的任何地方访问，那么这个变量就是全局变量。参见下面的代码:

```py
x = 10
print(x)
def func():
    print(x)
    return

func()

```

变量`x`在任何函数外部声明。这意味着可以在整个代码的任何地方访问它。在上面的例子中，`x`在函数`func`之外以及在`func`之内被访问。

输出:

```py
Global: 10
Local: 10
```

注意:在函数中操作一个全局变量有点复杂，我们将在后面用关键字`global`讨论它。

### 2。局部变量范围

如果一个变量是在函数内部声明的，那么它就在局部范围内。这将使得该变量只能在该特定函数内部被访问，如果没有同名的全局变量，任何在函数外部访问这样一个变量的尝试都会导致错误。

```py
def func():
    x = 10
    print(x)

```

在上面的函数中，变量`x`是在函数内部创建的，所以`x`是`func`的局部变量。试图访问`func`外的`x`将导致错误。

### 3。非局部变量范围

为了理解非局部范围，我们需要举一个例子:

```py
def outer_func():
    x = 10
    print("Outside:", x)

    def inner_func():
        print("Inside:", x)
        return

    inner_func()
    return

outer_func()

```

在函数`outer_func`中，我们有一个变量`x`，所以很明显，`x`对于`outer_func`是局部的。但是对于`inner_func`，`x`是非本地的，意味着`x`对于`inner_func`来说不是本地的，但是也不是全局的。

我们可以从`inner_func`访问`x`作为非局部变量。这是输出结果:

```py
Outside: 10
Inside: 10
```

注意:从`inner_func`操作`x`稍微复杂一点，我们将在讨论非本地关键字时看到。

## 操作全局和非局部变量

我们已经看到，我们可以在函数中访问一个全局的和非局部的变量，但是如果我们直接试图在函数中操作这个变量，那么就会导致错误。让我们看一个例子:

```py
x = 10
def func():
    x += 1
    print(x)
    return

func()

```

现在，从逻辑上讲，我们应该能够增加`x`，因为它是一个全局变量，可以在任何地方访问，但这是实际的输出:

```py
---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
<ipython-input-33-07e029a18d76> in <module>
      5     return
      6 
----> 7 func()

<ipython-input-33-07e029a18d76> in func()
      1 x = 10
      2 def func():
----> 3     x += 1
      4     print(x)
      5     return

UnboundLocalError: local variable 'x' referenced before assignment
```

上面写着，`UnboundLocalError: local variable 'x' referenced before assignment`。

Python 假设`x`是本地的，并告诉我们在引用它之前给它赋值。我们知道打印`x`会起作用，所以这意味着如果一个全局变量在函数内部被直接改变，就会发生这样的错误。

### Python 中的全局关键字

为了避免所讨论的错误，我们可以对全局变量范围使用`global`关键字:

```py
x = 10
def func():
    global x
    x += 1
    print(x)
    return

func()

```

我们可以看到，我们在函数内部将`x`声明为全局的，并告诉 Python`x`已经在全局范围内声明，我们将使用那个`x`。输出:

```py
11
```

所以这次它打印了修改后的`x`的值。

### Python 中的外地关键字

对于非局部变量范围，我们将使用`nonlocal`关键字来避免所讨论的错误，如下所示:

```py
def outer_func():
    x = 10

    def inner_func():
        nonlocal x
        x += 1
        print(x)

    inner_func()
    return

outer_func()

```

我们告诉 Python`x`在`inner_func`函数内部是非局部的。(`global`不起作用，因为 x 不是全局的)。

输出:

```py
11
```

## 结论

在本教程中，我们讨论了 python 中作用域的含义，进一步讨论了什么是全局、局部和非局部变量以及如何使用它们。我们使用了两个关键字:`global`和`nonlocal`，我们看到它们在 python 代码中工作，并输出结果。