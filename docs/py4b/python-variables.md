# Python 中的变量

> 原文：<https://www.pythonforbeginners.com/basics/python-variables>

## 变量

您可以使用任何字母、特殊字符“_”和提供的每个数字
而不是以它开头。

在 Python 中有特殊含义的空格和符号，如“+”和“-”是不允许的
。

我通常使用小写字母，单词之间用下划线隔开，以提高可读性。

记住变量名是区分大小写的。

Python 是动态类型的，这意味着你不必声明每个变量是什么类型。

在 Python 中，变量是文本和数字的存储占位符。

它必须有一个名称，以便您能够再次找到它。

变量总是被赋予等号，后跟
变量的值。

Python 有一些保留字，不能用作变量名。

这些变量在程序中被引用来得到它的值。

变量的值可以在以后更改。

**将值 10 存储在名为 foo** 的变量中

```py
foo = 10
```

**将 foo+10 的值存储在名为 bar** 的变量中

```py
bar = foo + 10
```

## 一些不同变量类型的列表

```py
 x = 123 			# integer
x = 123L			# long integer
x = 3.14 			# double float
x = "hello" 			# string
x = [0,1,2] 			# list
x = (0,1,2) 			# tuple
x = open(‘hello.py’, ‘r’) 	# file

You can also assign a single value to several variables simultaneously multiple 
assignments.

Variable a,b and c are assigned to the same memory location,with the value of 1
a = b = c = 1 
```

#### 例子

```py
length = 1.10
width  = 2.20
area   = length * width
print "The area is: " , area

This will print out: The area is:  2.42 
```

##### 更多阅读

[http://python.org/](https://python.org/ "python")