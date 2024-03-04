# 在 Python 中注释掉多行的快捷方式

> 原文：<https://www.pythonforbeginners.com/comments/shortcut-to-comment-out-multiple-lines-in-python>

在测试或调试代码时，我们经常需要注释掉 python 中的代码块。当一个块被转换成 python 注释时，它不会影响程序的输出，而是帮助确定哪个函数或块在程序中产生错误。本文讨论了在不同的 python IDEs 中一次注释掉多行代码的快捷方式。让我们逐一查看每个 IDE 的示例。

## 在 Spyder IDE 中注释掉多行的快捷方式

在 spyder python IDE 中，我们可以通过选择一行代码，然后使用组合键`ctrl+1`来注释该行代码。这将把选中的单行变成注释，如下所示。示例中给出的函数将一个数字及其平方作为键值对添加到一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中。

```py
print("This line will be commented out.")
def add_square_to_dict(x,mydict):
    a=x*x
    mydict[str(x)]=a
    return mydict 
```

按下`ctrl+1`后:

```py
#print("This line will be commented out.")
def add_square_to_dict(x,mydict):
    a=x*x
    mydict[str(x)]=a
    return mydict 
```

在 spyder IDE 中注释掉多行代码的快捷方式是先选中所有需要注释掉的行，然后按组合键`ctrl+4`。这将把所有选中的行变成一个 [python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)，如下所示。

```py
 class MyNumber():
      """This is the docstring of this class.

      It describes what this class does and all its attributes."""
      def __init__(self, value):
          self.value=value
      def increment(self):
          """This is the docstring for this method.

          It describes what the method does, what are its calling conventions and
          what are its side effects"""
          self.value=self.value+1
          return self.value
  print (MyNumber.increment.__doc__)
```

按下 ctrl+4 后:

```py
# =============================================================================
# 
#   class MyNumber():
#       """This is the docstring of this class.
#       
#       It describes what this class does and all its attributes."""
#       def __init__(self, value):
#           self.value=value
#       def increment(self):
#           """This is the docstring for this method.
#           
#           It describes what the method does, what are its calling conventions and
#           what are its side effects"""
#           self.value=self.value+1
#           return self.value
#   print (MyNumber.increment.__doc__)
# =============================================================================
```

### 如何在 Spyder IDE 中取消代码注释？

*   当代码行被注释掉时，我们可以在选择代码行后使用`ctrl+1`取消注释。
*   在 Spyder 的某些版本中,`ctrl+5`可以用来取消代码行的注释。

## 注释掉空闲多行的快捷方式

要注释掉空闲状态下的代码块，我们必须首先选择该行，然后按组合键`ctrl+D`。这将注释掉选定的代码行，如下所示。

```py
 class MyNumber():
         """This is the docstring of this class.

         It describes what this class does and all its attributes."""
         def __init__(self, value):
             self.value=value
         def increment(self):
             """This is the docstring for this method.

             It describes what the method does, what are its calling conventions and
             what are its side effects"""
             self.value=self.value+1
             return self.value
     print (MyNumber.increment.__doc__)
```

按下`ctrl+D`后，代码行将被注释掉，如下所示。

```py
## class MyNumber():
##         """This is the docstring of this class.
##         
##         It describes what this class does and all its attributes."""
##         def __init__(self, value):
##             self.value=value
##         def increment(self):
##             """This is the docstring for this method.
##             
##             It describes what the method does, what are its calling conventions and
##             what are its side effects"""
##             self.value=self.value+1
##             return self.value
##     print (MyNumber.increment.__doc__) 
```

### 取消空闲代码的注释

要取消 IDLE 中代码行的注释，我们只需选择代码行，然后按下`ctrl+shift+d`。这将取消对选定行的注释。

## 在 Jupyter 笔记本中注释掉多行的快捷方式

我们可以使用`ctrl+/`注释掉 Jupyter Notebook 中选中的 python 代码行。这将选定的代码行变成注释，如下所示。

```py
 class MyNumber():
         """This is the docstring of this class.

         It describes what this class does and all its attributes."""
         def __init__(self, value):
             self.value=value
         def increment(self):
             """This is the docstring for this method.

             It describes what the method does, what are its calling conventions and
             what are its side effects"""
             self.value=self.value+1
             return self.value
     print (MyNumber.increment.__doc__)
```

按下`ctrl+/`后，代码会被注释掉，如下图所示。

```py
 #      class MyNumber():
#          """This is the docstring of this class.

#          It describes what this class does and all its attributes."""
#          def __init__(self, value):
#              self.value=value
#          def increment(self):
#              """This is the docstring for this method.

#              It describes what the method does, what are its calling conventions and
#              what are its side effects"""
#              self.value=self.value+1
#              return self.value
#      print (MyNumber.increment.__doc__)
```

### 取消 Jupyter 笔记本中代码的注释

要取消 Jupyter 笔记本中所选行的注释，我们只需再次按下`ctrl+/`。这将取消 Jupyter 笔记本中代码的注释。

## 在 PyCharm 中注释掉多行

如果我们必须在 Pycharm 中注释掉多行代码，我们可以选择要注释掉的行，然后按`ctrl+shift+/`。在这之后，这些行将从代码中被注释掉。

### 取消 PyCharm 中代码的注释

要取消 PyCharm 中代码的注释，我们只需选择代码行，然后再次按下`ctrl+shift+/`。在此之后，评论将被转化为代码。

## 结论

在本文中，我们看到了在 python 不同的 ide 中一次注释掉多行的快捷方式，如 spyder、IDLE、Jupyter Notebook 和 PyCharm。

要了解更多关于 python 编程的知识，可以阅读这篇关于 python 中的[字符串操作的文章。](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)你可能也会喜欢这篇关于 [python simplehttpserver](https://www.pythonforbeginners.com/modules-in-python/how-to-use-simplehttpserver) 的文章。

请继续关注更多内容丰富的文章。