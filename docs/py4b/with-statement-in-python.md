# Python 中的 With Open 语句

> 原文：<https://www.pythonforbeginners.com/files/with-statement-in-python>

在 Python 中，可以使用 open()方法访问文件。但是，直接使用 open()方法需要使用 close()方法显式关闭文件。相反，您可以使用 python 中的 with Open 语句创建上下文。它返回一个 file 对象，该对象具有获取信息和操作打开的文件的方法和属性。

## Python 中的 With 语句

使用“With”语句，可以获得更好的语法和异常处理。

" with 语句通过封装常见的准备和清理任务简化了异常处理."

此外，它会自动关闭文件。with 语句提供了一种确保始终使用清理的方法。

如果没有 with 语句，我们将编写如下内容:

```py
file = open("welcome.txt")

data = file.read()

print data

file.close()  # It's important to close the file when you're done with it 
```

在上面的代码中，我们需要使用 close()方法显式关闭文件。

## Python 中 Open()函数的 With 语句用法

使用 with 打开文件非常简单:with open(filename) as file:

```py
with open("welcome.txt") as file: # Use file to refer to the file object

   data = file.read()

   do something with data 
```

以写入模式打开 output.txt

```py
with open('output.txt', 'w') as file:  # Use file to refer to the file object

    file.write('Hi there!') 
```

在上面的代码中，您可以看到我们已经使用 with open 语句打开了 output.txt 文件。该语句返回一个分配给变量“file”的文件指针。现在，我们可以在 with 语句的上下文中对 file 对象执行任何操作。一旦执行完所有语句，并且执行到达 with context 块的末尾，python 解释器将自动关闭该文件。此外，如果程序在 with 块中遇到任何异常，python 中的 with open 上下文会在终止程序之前关闭文件。这样，即使程序突然终止，文件中的数据也保持安全。

请注意，我们不必编写“file.close()”。会被自动调用。

## 结论

在本文中，我们讨论了如何使用 with open 语句而不是 open()
方法在 python 中打开文件。要了解更多关于 python 编程的知识，你可以阅读这篇关于 Python 中的[字符串操作的文章。你可能也会喜欢这篇关于](https://www.pythonforbeginners.com/basics/string-manipulation-in-python) [python 的文章，如果是简写](https://avidpython.com/python-basics/python_if_else_shorthand/)。