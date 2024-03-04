# 如何用 Python 写注释

> 原文：<https://www.pythonforbeginners.com/strings/how-to-write-comments-in-python>

python 中的注释是源代码的一部分，不被 python 解释器执行。注释对应用程序没有贡献任何功能，对应用程序的用户没有任何意义。但是注释对程序员有很大的帮助。注释增加了源代码的可读性和可理解性，并帮助程序员重构或调试代码，同时为应用程序中的新功能添加源代码。在本文中，我们将看到如何使用不同的方式在 python 中编写注释。

## 如何用 Python 写单行注释？

我们可以用# symbol 写一行 python 注释。在 python 中，任何写在# till 换行符之后的东西，包括符号本身，都被认为是注释。每当遇到换行符时，单行注释就会终止。我们可以在程序的任何地方使用#符号开始一行注释，在符号变成注释后使用整个语句。

下面是一个示例代码，其中一行注释写在一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中。

```py
myDict={"name":"PythonForBeginners.com",
        #This is a single line comment inside a python dictionary
        "acronym":"PFB"
        }
print(myDict["acronym"])
```

我们也可以在 python 语句后开始单行注释。

```py
 myDict={"name":"PythonForBeginners.com",
        #This is a single line comment inside a python dictionary
        "acronym":"PFB"
        }
print(myDict["acronym"]) #This is single line comment which starts after a python statement 
```

## 如何用 Python 写多行注释？

理论上，python 没有多行注释的语法，但是我们可以使用其他方式，比如多行字符串和单行注释来编写 python 中的多行注释。

我们可以使用单行注释在 python 中编写多行注释，方法是在连续的行上编写单行注释，如下例所示。

```py
#This is multiline comment
#written using single line comments
#for a demonstration
myDict={"name":"PythonForBeginners.com",
        "acronym":"PFB"
        }
print(myDict["acronym"]) 
```

我们也可以在 python 语句后开始多行注释，同时用#符号实现多行注释，因为它必然表现为单行注释。

```py
#This is multiline comment
#written using single line comments
#for a demonstration
myDict={"name":"PythonForBeginners.com",
        "acronym":"PFB"
        }
print(myDict["acronym"]) #This is multiline comment after a python statement
#written using single line comments
#for a demonstration
```

为了用 python 写多行注释，我们也可以使用多行字符串。如果我们没有给任何变量分配一个多行字符串，解释器将分析和评估该多行字符串，但是不会为该多行字符串生成字节码，因为没有地址可以分配给它们。实际上，多行字符串将作为多行注释工作。以下是使用多行字符串编写的多行注释的示例。

```py
"""This is multiline comment
written using multi line strings
for a demonstration"""
myDict={"name":"PythonForBeginners.com",
        "acronym":"PFB"
        }
print(myDict["acronym"])
```

我们必须记住，使用多行字符串编写的多行注释不是真正的注释，它们只是未赋值的字符串常量。因此，它们应该遵循适当的缩进，并且只能从新行开始。如果我们试图在任何 python 语句后使用多行字符串编写多行注释，将会导致语法错误。

```py
myDict={"name":"PythonForBeginners.com",
        "acronym":"PFB"
        }
print(myDict["acronym"])"""This is not a multiline comment and 
will cause syntax error in the program"""
```

## 结论

在本文中，我们看到了如何使用#符号和多行字符串在 python 中编写注释。更多文章敬请关注。