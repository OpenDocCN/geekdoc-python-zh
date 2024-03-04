# 为什么我们需要 Python 中的注释

> 原文：<https://www.pythonforbeginners.com/comments/why-we-need-comments-in-python>

注释是包含在源代码中但对程序逻辑没有贡献的语句。一个 [python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)不是由解释器执行的，只有当我们可以访问源代码时才能访问它。现在的问题是，尽管它们对程序逻辑没有贡献，但是为什么注释会包含在源代码中。在这篇文章中，我们将尝试了解为什么我们需要 python 中的注释。让我们开始吧。

## 我们使用注释来指定源代码的元数据。

当我们出于商业目的编写任何计算机程序时，我们通常会包括程序员的姓名、源代码创建的日期和时间。一般来说，当根据合同开发程序时，源代码的使用条件、源代码的许可和版权以及程序的一般描述也包含在源文件中。为了在源代码中包含这些信息，我们使用 python 注释。元数据包含在源代码文件开头的头注释中。下面是一个标题注释的例子。

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:59:11 2021

@author: aditya1117
This code has been written to demonstrate the use of comments in python to specify metadata of the sorce code
""" 
```

当任何人访问源代码文件时，他们可以读取文件的元数据，然后他们可以知道他们是否被允许重用或修改源代码。如果对源代码进行了任何更改，也应该在元数据中提及，以便其他人可以获得相关信息。

## 我们使用注释来记录源代码。

对于源代码文档，我们使用注释来描述程序中使用的每个函数和方法的规范。我们包括输入参数、预期输出和方法或函数的一般描述，这样当有人试图使用程序中的函数和方法时，他可以从文档中得到想法。当用任何编程语言引入一个框架或库时，必须提供适当的源代码文档。下面是一个文档示例，描述了一个将数字及其平方作为键值对添加到 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)的函数。

```py
#The function adds a number and its square to a python dictionary as key value pair.
def add_square_to_dict(x,mydict):
    a=x*x
    mydict[str(x)]=a
    return mydict 
```

## 我们使用注释来阐明为什么在源代码中写了一个特定的语句。

一般来说，程序中使用的函数和语句对于它们在程序中的使用应该是不言自明的，但是每当看起来不清楚为什么要在源代码中编写语句时，我们就需要注释来说明为什么要在源代码中编写语句。

## 我们使用注释来指定有助于调试代码和改进程序功能的指令。

当编写程序时，根据代码中需要的功能，对代码进行许多修改。为了能够重构和修改代码进行改进，程序员必须知道源代码中包含的函数、方法、类等的所有属性。程序员必须清楚地理解代码中包含的每个语句，以便能够修改它。为了向程序员提供所有这些信息，我们需要注释来指定源代码中函数和方法的所有属性。

## 我们使用注释来指定对源代码所做的更改。

无论何时对程序的源代码进行任何更改，都必须在源代码中使用注释来指定，并且应该包括关于为什么进行更改以及谁进行了更改的信息。

## 结论

在本文中，我们已经看到了为什么我们需要 python 中的注释的各种原因。请继续关注更多内容丰富的文章。