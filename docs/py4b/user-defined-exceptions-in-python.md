# Python 中用户定义的异常

> 原文：<https://www.pythonforbeginners.com/exceptions/user-defined-exceptions-in-python>

当我们用 python 为现实生活中的应用程序编写程序时，对程序中变量的取值有许多限制。例如，年龄不能为负值。当一个人输入一个负值的年龄时，程序会显示一条错误信息。但是这些类型的约束不能在 python 程序中自动应用。为了处理这些类型的错误并对值施加约束，我们在 python 中使用用户定义的异常。在这篇文章中，我们将看看在 python 中实现用户定义的异常的不同方法。

## python 中用户定义的异常是什么？

python 中用户定义的异常是由程序员创建的，用来对程序中变量的取值施加约束。Python 有许多内置的异常，当程序中出现错误时会引发这些异常。当程序进入不希望的状态时，在显示程序执行过程中发生了哪个内置异常后，程序会自动终止。我们可以通过使用用户定义的异常来强制约束，从而阻止程序进入不希望的状态。

用户定义的异常可以通过显式引发异常、使用 assert 语句或为用户定义的异常定义自定义类来实现。在本文中，我们将逐一研究每种方法。

## 在条件语句后显式使用 raise 关键字

内置异常由 python 中的程序自动引发，但我们也可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块和 raise 关键字引发内置异常。通过使用 raise 关键字显式引发内置异常，我们可以在程序中的任何地方使用它们来强制约束变量值。

例如，假设我们必须根据一个人的年龄来计算他的出生年份，我们可以这样做:

```py
age= 10
print("Age is:")
print(age)
yearOfBirth= 2021-age
print("Year of Birth is:")
print(yearOfBirth)
```

输出:

```py
Age is:
10
Year of Birth is:
2011
```

在这种情况下，程序通过从当前年份中减去年龄给出了正确的输出。现在假设我们在输入中给年龄一个负值，比如-10。

```py
age= -10
print("Age is:")
print(age)
yearOfBirth= 2021-age
print("Year of Birth is:")
print(yearOfBirth)
```

输出:

```py
Age is:
-10
Year of Birth is:
2031
```

当我们为年龄提供一个负数时，程序仍然可以正常工作，但是输出的结果在逻辑上是不正确的，因为没有人能知道他的出生年份。

为了防止这种年份，我们可以检查输入的 age 中给出的值是否为负，然后我们可以使用 raise 关键字强制程序引发一个异常，如下所示。

raise 语句的语法是`raise ExceptionName`。当出现错误时，except 块中的代码应该处理该异常，否则将导致程序出错。

```py
try:
    age= -10
    print("Age is:")
    print(age)
    if age<0:
        raise ValueError
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except ValueError:
    print("Input Correct age.")
```

输出:

```py
Age is:
-10
Input Correct age.
```

这里我们可以看到，对于 10 岁的孩子，程序成功地处理了这个案例。让我们检查当给出正确的年龄值时，它是否给出正确的出生年份。

```py
try:
    age= 10
    print("Age is:")
    print(age)
    if age<0:
        raise ValueError
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except ValueError:
    print("Input Correct age.")
```

输出:

```py
Age is:
10
Year of Birth is:
2011
```

因此，我们可以看到，在 try except 块中使用 raise 语句后，程序在两种情况下都给出了正确的输出。现在我们将看到如何使用 assert 语句来实施相同的约束。

## 使用断言语句在 python 中实现用户定义的异常

我们可以使用 assert 语句在 python 中实现对变量值的约束。当不满足 assert 语句中给定的条件时，程序在输出中给出`AssertionError`。

python 中 assert 语句的语法是`assert condition`，其中`condition`可以是计算结果为`True`或`False`的任何条件语句。当 assert 语句中的条件求值为`True`时，程序正常工作。当 assert 语句中的条件评估为`False`时，程序给出`AssertionError`。

为了强制实施年龄应该大于零的约束，我们可以使用如下程序所示的 assert 语句。

```py
age= 10
print("Age is:")
print(age)
assert age>0
yearOfBirth= 2021-age
print("Year of Birth is:")
print(yearOfBirth)
```

输出:

```py
Age is:
10
Year of Birth is:
2011
```

当年龄输入为负值时，条件将评估为假，程序将因`AssertionError`而结束。

```py
age= -10
print("Age is:")
print(age)
assert age>0
yearOfBirth= 2021-age
print("Year of Birth is:")
print(yearOfBirth)
```

输出:

```py
Age is:
-10
Traceback (most recent call last):

  File "<ipython-input-9-214b1ab4dfa4>", line 4, in <module>
    assert age>0

AssertionError
```

当 assert 语句中的条件不满足时，我们还可以给出一条要打印的消息。用 assert 语句打印消息的语法是`assert condition, message`。`message`应该是字符串常量。每当 assert 语句中的`condition`计算为`False`时，程序将产生一个`AssertionError`，并打印`message`。

```py
age= -10
print("Age is:")
print(age)
assert age>0 , "Age should be positive integer"
yearOfBirth= 2021-age
print("Year of Birth is:")
print(yearOfBirth)
```

输出:

```py
Age is:
-10
Traceback (most recent call last):

  File "<ipython-input-10-82c11649bdd0>", line 4, in <module>
    assert age>0 , "Age should be positive integer"

AssertionError: Age should be positive integer
```

在输出中，我们可以看到“年龄应该是正整数”的消息也和`AssertionError`一起打印出来。

如果想禁止程序过早退出，我们也可以使用 try except 块来处理`AssertionError`。可能有这样的情况，当 [python 写入文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)操作已经执行，如果程序过早退出，写入文件的数据将不会被保存。为了保存写入文件的数据，我们需要在程序退出前关闭文件。为此，我们必须使用 try except 块和 assert 语句，如下所示。

```py
try:
    age= -10
    print("Age is:")
    print(age)
    assert age>0
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except AssertionError:
    print("Input Correct age.") 
```

输出:

```py
Age is:
-10
Input Correct age.
```

在输出中，我们可以看到当 age 小于 0 时，`AssertionError`已经发生，但是已经被 except 块中的代码处理。如果我们想要执行任何文件操作，那么我们可以在 except 块中实现代码，或者我们可以使用 finally 块来实现代码。

## 为用户定义的异常定义自定义类

为了创建一个用户定义的异常，我们创建一个具有期望异常名称的类，它应该继承异常类。之后，我们可以根据实现约束的需要在代码中的任何地方引发异常。

为了生成一个用户定义的异常，我们在满足特定条件时使用“raise”关键字。然后由代码的 except 块处理该异常。然后我们使用 pass 语句。pass 语句用于表明我们不会在自定义异常类中实现任何东西。它被用作一个占位符，什么也不做，但我们仍然必须使用它，因为如果我们将自定义类的主体留空，python 解释器将显示我们的代码中有错误。

示例:

```py
 class NegativeAgeError(Exception):
    pass
try:
    age= -10
    print("Age is:")
    print(age)
    if age<0:
        raise NegativeAgeError
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except NegativeAgeError:
    print("Input Correct age.")
```

输出:

```py
Age is:
-10
Input Correct age.
```

这里我们可以看到，当 age 的值小于零时，程序中的 try 块使用 raise 关键字抛出`NegativeAgeError`。然后由 except 块处理该异常。如果我们将年龄的正确值作为输入，它将正常打印出生年份。

```py
 class NegativeAgeError(Exception):
    pass
try:
    age= 10
    print("Age is:")
    print(age)
    if age<=0:
        raise NegativeAgeError
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except NegativeAgeError:
    print("Input Correct age.")
```

输出:

```py
Age is:
10
Year of Birth is:
2011
```

## 结论:

在本文中，我们看到了如何使用自定义异常处理方法(如 assert 关键字、raise 关键字和自定义异常类)在 python 中实现对变量值的约束。我们也可以使用用户定义的异常在我们的程序中实现不同的现实约束，这样程序在语法上是正确的，逻辑上是合理的。敬请关注更多文章。