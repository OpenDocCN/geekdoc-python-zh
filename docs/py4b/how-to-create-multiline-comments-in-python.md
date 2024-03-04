# 如何在 Python 中创建多行注释

> 原文：<https://www.pythonforbeginners.com/comments/how-to-create-multiline-comments-in-python>

注释通过提供有关代码的适当信息，用于增强源代码的可读性和可理解性。注释不是由解释器或编译器执行的，它们在代码中使用只是为了帮助程序员。在本文中，我们将了解如何在 python 中创建多行注释。

## python 中的多行注释是什么？

正如我们在名字中看到的，多行注释是扩展到多行的 [python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)，即多行注释是那些在源代码中扩展到两行或更多行的注释。从理论上讲，python 中没有多行注释的语法，但是我们可以使用单行注释或三重引号字符串实现多行注释。我们将逐一查看实现多行注释的两种方法。

## 如何使用# symbol 在 python 中创建多行注释？

正如我们所知，python 中单行注释是通过在注释前添加#符号来实现的，只要出现换行符，注释就会终止，我们可以在多行注释的每一行的开头添加#符号。这样，我们可以使用单行注释实现多行注释。这可以看如下。示例中的函数将一个数字及其平方作为键值对添加到一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中。

```py
#This is a multiline comment in python
#and expands to more than one line 

def add_square_to_dict(x,mydict):
    #This is an example of block of code where the block consists of an entire function
    a=x*x
    mydict[str(x)]=a
    return mydict
```

在 python 中，我们可以在源代码中的任何语句后开始一行注释。类似地，当我们使用# symbol 实现多行注释时，我们可以在代码中的任何语句后开始多行注释。

```py
#This is a multiline comment in python
#and expands to more than one line
print("Pythonforbeginners.com") #This is also a python comment
#and it also expands to more than one line.
```

## 如何在 python 中使用多行字符串创建多行注释？

要在 python 中使用多行字符串作为多行注释，我们可以使用多行字符串，而不用将它赋给任何变量。这样，当解释器执行代码时，不会为未赋值的字符串生成字节码，因为没有地址可以分配给它。这样，多行字符串将充当多行注释。这可以看如下。

```py
"""This is 
a 
multiline comment in python
which expands to many lines"""
print("PythonforBeginners.com")
```

当我们用#符号实现多行注释时，我们可以在 python 中的语句后开始多行注释。但是如果我们使用多行字符串来实现多行注释，我们就不能在任何语句后开始多行注释。如果这样做，就会导致错误。这可以看如下。

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

当我们使用#符号实现注释时，我们不必关心缩进，但是当我们使用多行字符串作为注释时，我们必须遵循正确的缩进，否则将会出现错误。这可以看如下。

```py
 #This is a multiline comment in python
#and expands to more than one line wthout caring for indentation
"""This is 
a 
multiline comment in python
which expands to many lines and should follow indentation"""
print("Pythonforbeginners.com")
```

## 结论

在本文中，我们看到了用 python 创建多行注释的各种方法。我们还看到了如何使用多行注释，以便它们不会在程序中导致错误。请继续关注更多内容丰富的文章。