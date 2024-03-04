# 在 Python 中将整数转换为字符串

> 原文：<https://www.pythonforbeginners.com/basics/convert-integer-to-string-in-python>

Python 字符串是为模式匹配等操作进行数据分析时最常用的数据类型之一。在本文中，我们将使用不同的方法在 python 中将整数转换成字符串。

## 使用 str()函数将整数转换为字符串

将整数转换为字符串的最简单方法是使用 str()函数。str()函数将整数作为输入，并返回其字符串表示形式，如下所示。

```py
myInt = 1117
myStr = str(myInt)
print("The integer myInt is:", myInt)
print("The string myStr is:", myStr)
```

输出:

```py
The integer myInt is: 1117
The string myStr is: 1117
```

我们可以检查输入变量和输出变量的类型，以确认整数是否已经转换为字符串。为此，我们将使用 type()函数。type 函数将 python 对象作为输入，并返回输入对象的数据类型，如下所示。

```py
myInt = 1117
myStr = str(myInt)
print("The data type of myInt is:", type(myInt))
print("The data type of myStr is:", type(myStr))
```

输出:

```py
The data type of myInt is: <class 'int'>
The data type of myStr is: <class 'str'>
```

## 使用字符串格式将整数转换为字符串

字符串格式化是一种将变量或另一个字符串插入预定义字符串的方法。我们也可以使用字符串格式将整数转换成字符串。在本文中，我们将使用“%s”操作符以及 format()方法将整数转换为字符串。

“%s”运算符用于格式化字符串中的值。一般用来避免[字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)。但是，我们可以使用这个操作符将整数转换成字符串。为此，首先我们将创建一个空字符串，并在空字符串中放置一个%s 占位符。之后，我们可以指定需要转换成字符串的整数。在程序执行期间，python 解释器会将整数转换为字符串，如下例所示。

```py
myInt = 1117
myStr = "%s" % myInt
print("myInt is:",myInt)
print("The data type of myInt is:", type(myInt))
print("myStr is:",myStr)
print("The data type of myStr is:", type(myStr))
```

输出:

```py
myInt is: 1117
The data type of myInt is: <class 'int'>
myStr is: 1117
The data type of myStr is: <class 'str'>
```

代替“%s”操作符，我们也可以使用 format()方法来执行转换。为此，我们可以在空字符串中放置一个{}占位符。之后，我们可以在空字符串上调用 format 方法，将整数作为 format()方法的输入。这将把整数转换成字符串，如下所示。

```py
myInt = 1117
myStr = "{}".format(myInt)
print("myInt is:",myInt)
print("The data type of myInt is:", type(myInt))
print("myStr is:",myStr)
print("The data type of myStr is:", type(myStr))
```

输出:

```py
myInt is: 1117
The data type of myInt is: <class 'int'>
myStr is: 1117
The data type of myStr is: <class 'str'>
```

## 使用 F 弦进行转换

f 字符串用于将值或表达式嵌入到字符串中。我们也可以用 f 字符串把一个整数转换成一个字符串。

使用 f 字符串的语法类似于 format()方法的语法。唯一的区别是我们可以将变量直接放入占位符中。这使得代码更具可读性。要将整数变量 n 放入字符串，我们只需将 n 放入占位符{}中，如下所示。

```py
f"This is a string containing {n}"
```

要使用 f 字符串将整数转换为字符串，我们将声明一个空字符串，其中只有一个整数占位符。这样，在运行时，整数将被转换为字符串。这可以从下面的例子中看出。

```py
myInt = 1117
myStr = f"{myInt}"
print("myInt is:",myInt)
print("The data type of myInt is:", type(myInt))
print("myStr is:",myStr)
print("The data type of myStr is:", type(myStr))
```

输出:

```py
myInt is: 1117
The data type of myInt is: <class 'int'>
myStr is: 1117
The data type of myStr is: <class 'str'>
```

## 结论

在本文中，我们看到了在 python 中将整数转换成字符串的不同方法。我们使用了内置的 str()方法、字符串格式以及 f 字符串。要了解更多关于字符串的内容，你可以阅读这篇关于 [python 字符串拆分](https://www.pythonforbeginners.com/dictionary/python-split)操作的文章。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 来编写本文中使用的程序，以使程序更加健壮，并以系统的方式处理错误。