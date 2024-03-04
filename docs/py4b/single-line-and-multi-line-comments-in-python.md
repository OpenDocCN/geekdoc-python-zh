# Python 中的单行和多行注释

> 原文：<https://www.pythonforbeginners.com/comments/single-line-and-multi-line-comments-in-python>

注释是一段代码，当程序被执行时，它不会被编译器或解释器执行。只有当我们能够访问源代码时，才能阅读注释。注释用于解释源代码，使代码更具可读性和可理解性。在本文中，我们将看到如何使用 python 中的不同方法编写单行和多行注释。

## python 中的单行注释是什么？

单行注释是那些在 python 中没有换行或者换行的注释。一个 [python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)通过用一个`#`初始化注释文本来编写，并在遇到行尾时终止。下面的例子显示了一个程序中的单行注释，其中定义了一个函数，将一个数字及其平方作为键值对添加到 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中。

```py
 #This is a single line comment in python
def add_square_to_dict(x,mydict):
    a=x*x
    mydict[str(x)]=a
    return mydict
```

我们也可以在另一个语句后添加单行注释。

```py
#This is a single line comment in python
print("Pythonforbeginners.com") #This is also a python comment
```

## 什么是多行注释？

顾名思义，多行注释可以扩展到多行。但是 python 没有多行注释的语法。我们可以使用单行注释或三重引用的 python 字符串在 python 中实现多行注释。

## 如何使用#符号实现多行注释？

为了使用`#`符号实现多行注释，我们可以简单地将多行注释的每一行描述为单行注释。然后我们可以用符号`#`开始每一行，我们可以实现多行注释。

```py
#This is a multiline comment in python
#and expands to more than one line
print("Pythonforbeginners.com")
```

当使用`#`符号编写多行注释时，我们也可以在任何 python 语句后开始多行注释。

```py
#This is a multiline comment in python
#and expands to more than one line
print("Pythonforbeginners.com") #This is also a python comment
#and it also expands to more than one line.
```

## 如何使用三重引号字符串实现多行注释？

python 中的多行字符串如果没有赋给变量，可以用作多行注释。当字符串没有被赋给任何变量时，它们会被解释器解析和评估，但不会生成字节码，因为没有地址可以赋给字符串。实际上，未赋值的多行字符串作为多行注释工作。

```py
"""This is 
a 
multiline comment in python
which expands to many lines"""
```

在这里，我们必须记住，多行注释只是字符串常量，没有被赋予任何变量。所以它们必须有正确的意图，不像带有`#`符号的单行注释，这样可以避免语法错误。

此外，使用三重引号的多行注释应该总是以换行符开始，这与单行注释不同。

```py
 #This is a multiline comment in python
#and expands to more than one line
"""This is 
a 
multiline comment in python
which expands to many lines"""
print("Pythonforbeginners.com") """This is not 
a 
multiline comment in python
and will cause syntax error"""
```

## 结论

在本文中，我们看到了如何用 python 编写单行和多行注释。我们还看到了如何使用字符串编写多行注释。请继续关注更多内容丰富的文章。