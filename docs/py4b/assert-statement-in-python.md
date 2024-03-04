# Python 中的 Assert 语句

> 原文：<https://www.pythonforbeginners.com/basics/assert-statement-in-python>

调试是软件开发人员旅程的重要部分之一。Python 编程语言还为调试程序提供了各种构造。一个这样的构造是 assert 语句。在本文中，我们将讨论什么是 assert 语句，它是如何工作的，以及我们如何在 python 中使用 assert 语句。

## Python 中的 assert 语句是什么？

在 Python 中，Assert 语句是一个用于在程序中强制执行某些条件的构造。python 中 assert 语句的语法如下。

```py
assert  condition, message
```

*   这里**断言**是一个关键字。
*   **条件**包含需要为真的条件语句。
*   **消息**是当条件为假时将显示的语句。在断言语句中使用**消息**是可选的。

## 断言语句是如何工作的？

assert 语句通过使用 AssertionError 异常来工作。每当 assert 语句中给出的条件评估为真时，程序正常工作。

```py
name = "Bill"
age = 20
assert age >= 0
print("{} is {} years old.".format(name,age)) 
```

输出:

```py
Bill is 20 years old.
```

相反，如果 assert 语句中的条件评估为 False，则 AssertionError 异常发生，程序终止。

```py
name = "Bill"
age = -20
assert age >= 0
print("{} is {} years old.".format(name,age)) 
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    assert age >= 0
AssertionError 
```

如果我们在 assert 语句中传递一条消息，那么在 AssertionError 发生时也会打印这条消息。

```py
name = "Bill"
age = -20
assert age >= 0, "Age cannot be negative."
print("{} is {} years old.".format(name,age)) 
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    assert age >= 0, "Age cannot be negative."
AssertionError: Age cannot be negative.
```

因此，我们可以说，当 assert 语句中的条件为真时，assert 语句什么也不做。如果 assert 语句中的条件为 False，python 解释器将使用适当的消息引发 AssertionError。

## python 中如何使用 assert 语句？

不应该在 python 程序中使用 assert 语句来实现任何业务逻辑。assert 语句仅用于调试 python 程序。当程序没有产生预期的输出时，可以使用 assert 语句来检测错误发生的位置。

例如，假设您的程序因为像“年龄”这样的变量有负值而遇到错误，这不是我们想要的情况。

为了检测错误发生的位置，可以在使用“age”变量的每个语句之前放置一个 assert 语句。在每个 assert 语句中，您可以检查“age”的值是否大于零。您还可以添加一条消息来标识 IndexError 发生的位置，如下所示。

```py
name = "Bill"
age = 10
assert age >= 0, "Age takes negative before first print statement."
print("{} is {} years old.".format(name, age))
name = "Sam"
age = -20
assert age >= 0, "Age takes negative before second print statement."
print("{} is {} years old.".format(name, age))
```

输出:

```py
Bill is 10 years old.
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 7, in <module>
    assert age >= 0, "Age takes negative before second print statement."
AssertionError: Age takes negative before second print statement.
```

在上面的例子中，程序提前终止。为了避免这种情况，可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块。在这里，您可以在 try 块中编写代码，并处理由 except 块中的 assert 语句引发的 AssertionError，如下所示。

```py
try:
    name = "Bill"
    age = 10
    assert age >= 0, "Age takes negative before first print statement."
    print("{} is {} years old.".format(name, age))
    name = "Sam"
    age = -20
    assert age >= 0, "Age takes negative before second print statement."
    print("{} is {} years old.".format(name, age))
except AssertionError as e:
    print(e.args) 
```

输出:

```py
Bill is 10 years old.
('Age takes negative before second print statement.',)
```

## 结论

在本文中，我们讨论了 python 中的 assert 语句。我们还讨论了 assert 语句是如何工作的，以及如何在调试程序时使用它们。要了解更多关于错误的信息，您可以阅读这篇关于 python 中的[异常的文章。](https://www.pythonforbeginners.com/error-handling/pythons-built-in-exceptions)