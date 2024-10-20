# python–检查变量是否为字符串

> 原文：<https://www.askpython.com/python/examples/python-check-if-variable-is-a-string>

因为 Python 不支持静态类型检查(即编译类型的类型检查)，如果你想检查一个 Python 变量或对象是否是一个字符串；我们需要使用某些方法。

让我们了解一下检查一个*字符串*类型对象的一些方法。

* * *

## 1.使用 isinstance()方法

**isinstance** (object，type)方法检查`object`是否属于`type`，如果该条件成立，则返回 **True** ，否则返回 **False** 。

常见的类型有: **int** ， **str** ， **list** ， **object** 等。

由于我们要检查`str`类型，我们将使用`isinstance(object, str)`并检查它是否是`True`。

让我们通过一个例子来理解这一点。

```py
a = 123

b = 'Hello'

print('Is a an instance of str?', isinstance(a, str))
print('Is b an instance of str?', isinstance(b, str))

```

**输出**

```py
Is a an instance of str? False
Is b an instance of str? True

```

## 2.使用类型(对象)方法

这类似于`isinstance()`方法，但是它显式地返回对象的类型。

让我们用它来检查返回的类型是否是`str`。

示例:

```py
a = 123

b = 'Hello'

print('type(a) =', type(a))
print('type(b) =', type(b))

print('Is a of type string?', type(a) == str)
print('Is b of type string?', type(b) == str)

```

**输出**

```py
type(a) = <class 'int'>
type(b) = <class 'str'>
Is a of type string? False
Is b of type string? True

```

* * *

## 检查函数参数是否为字符串

我们可以应用上述任何一种方法在任何函数中引入“检查”条件，这允许我们仅在输入是字符串时才执行函数的主体。

让我们用一个简单的例子来理解这一点。

```py
a = 123

b = 'Hello'

def capitalize_name(inp):
    # We can also use "if isinstance(inp, str)"
    if type(inp) != str:
        print('Input must be a string')
    else:
        print(inp.upper())

capitalize_name(a)
capitalize_name(b)

```

**输出**

```py
Input must be a string
HELLO

```

我们的函数现在在进入主体之前，明确地检查参数是否是一个字符串。由于 Python 的动态类型检查，这些类型检查有可能避免不必要的运行时错误。

我们没有从函数中抛出任何错误。但是，在实际编程中，大多数函数参数类型验证都会抛出`TypeError`异常。

* * *

## 结论

在本文中，我们学习了如何使用 Python 的`isinstance()`、`type()`方法并检查输入是否是字符串。我们还将此应用于一个函数，只接受字符串形式的参数。

现在，在运行时可能会导致更少的错误，这种格式对于一个好的 Python 程序来说是非常必要的。希望这篇文章对你有用！

## 参考

*   关于变量类型检查的 JournalDev 文章