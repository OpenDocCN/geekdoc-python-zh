# 用 Python 注释掉一段代码

> 原文：<https://www.pythonforbeginners.com/code/comment-out-a-block-of-code-in-python>

在测试或调试 python 程序时，可能会有很多情况，我们希望代码中的某个语句或代码中的某个语句块不被执行。为此，我们注释掉代码块。在本文中，我们将看到如何使用 [python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)注释掉 python 中的代码块。首先，我们将了解代码块在 python 中的含义，然后我们将了解如何注释掉它们。

## 什么是代码块？

代码块是任何编程语言中源代码中的一组语句，一旦满足某个条件或调用某个函数，就应该执行这些语句。

在 python 中，通过对语句应用适当的缩进来创建代码块，以创建代码块并将它们与其他代码块分开。

python 中的代码块可以由属于控制语句、函数或类的语句组成。

例如，下面是 if-else 控制语句中的一段代码。

```py
number =int(input())
#This an example of block of code where a block consists of control statement(If else in this case)
if (number%2==0):
    print("number is even")
else:
    print("number is odd")
```

函数中的代码块可能如下所示。以下函数将一个数字及其平方作为键值对添加到 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中。

```py
 def add_square_to_dict(x,mydict):
    #This is an example of block of code where the block consists of an entire function
    a=x*x
    mydict[str(x)]=a
    return mydict 
```

类中的代码块可能如下所示。

```py
 class number:
    #This is an example of block of code where the block consists of an entire class

    def __init__(self,value):
        self.value =value
    def increment(self):
        self.value=self.value+1
```

通常，一个代码块会扩展到多行。所以要用 python 注释掉一段代码，我们必须使用多行注释。首先我们将看到多行注释的工作，然后我们将尝试注释掉一段代码。

## 在 python 中处理多行注释

在 python 中，多行注释本质上不是作为一个构造包含在内的。但是单行注释和多行字符串可以用来在 python 中实现多行注释。

我们可以使用单行注释实现多行注释，只要遇到换行符就插入#号。这样，多行注释就被描述为一系列单行注释，如下所示。

```py
 def add_square(x,y):
    a=x*x
    b=y*y
    #This is a multi line comment
    #implemented using # sign
    return a+b
```

如果我们不把多行字符串赋给任何变量，我们也可以把它们作为多行注释。当字符串没有被赋给任何变量时，它们会被解释器解析和评估，但不会生成字节码，因为没有地址可以赋给字符串。这将影响字符串作为注释的工作。在这个方法中，可以使用三重引号声明多行注释，如下所示。

```py
def add_square(x,y):
    a=x*x
    b=y*y
    """This is a multiline comment
    implemented with the help of 
    triple quoted strings"""
    return a+b
```

## 使用# sign 注释掉 python 中的一段代码

我们可以在 python 中注释掉一个代码块，方法是在特定代码块中的每个语句的开头放置一个#符号，如下所示。

```py
number =int(input())
#if (number%2==0):
 #   print("number is even")
#else:
#    print("number is odd") 
```

## 使用多行字符串注释掉 python 中的代码块

我们可以使用多行字符串方法注释掉一段代码，如下所示。

```py
number =int(input())
"""if (number%2==0):
   print("number is even")
else:
  print("number is odd")
"""
```

虽然这种方法可行，但是我们不应该使用多行字符串作为注释来注释掉代码块。原因是多行注释被用作文档字符串来记录源代码。

## 结论

我们可以使用# symbol 注释掉 python 中的一段代码，但是我们应该避免使用多行字符串来注释掉这段代码。请继续关注更多内容丰富的文章。