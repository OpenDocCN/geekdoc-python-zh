# Python 中的文件处理备忘单

> 原文：<https://www.pythonforbeginners.com/cheatsheet/python-file-handling>

## 文件处理

Python 中的文件处理不需要导入模块。

## 文件对象

相反，我们可以使用内置对象“文件”。默认情况下，该对象提供了操作文件所需的基本函数和方法。在读取、追加或写入文件之前，首先必须使用 Python 的内置 open()函数。在这篇文章中，我将描述如何使用文件对象的不同方法。

## 打开()

open()函数用于打开我们系统中的文件，filename 是要打开的文件的名称。模式指示文件将如何打开:“r”表示读取,“w”表示写入,“a”表示追加。open 函数有两个参数，文件名和打开文件的模式。默认情况下，当只传递文件名时，open 函数以读取模式打开文件。

## 例子

这个小脚本将打开(hello.txt)并打印内容。这将把文件信息存储在文件对象“文件名”中。

```py
 filename = "hello.txt"
file = open(filename, "r")
for line in file:
   print line, 
```

## 阅读()

read 函数包含不同的方法，read()、readline()和 readlines()

```py
 read()		#return one big string
readline	#return one line at a time
read-lines 	#returns a list of lines 
```

## 写()

此方法将一系列字符串写入文件。

```py
 write ()	#Used to write a fixed sequence of characters to a file

writelines()	#writelines can write a list of strings. 
```

## 追加()

append 函数用于追加到文件中，而不是覆盖它。要追加到现有文件，只需在追加模式(“a”)下打开文件:

## 关闭()

当您处理完一个文件时，使用 close()关闭它并释放被打开的文件占用的所有系统资源

## 文件处理示例

让我们展示一些例子

```py
 To open a text file, use:
fh = open("hello.txt", "r")

To read a text file, use:
fh = open("hello.txt","r")
print fh.read()

To read one line at a time, use:
fh = open("hello".txt", "r")
print fh.readline()

To read a list of lines use:
fh = open("hello.txt.", "r")
print fh.readlines()

To [write to a file](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python), use:
fh = open("hello.txt","w")
write("Hello World")
fh.close()

To [write to a file](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python), use:
fh = open("hello.txt", "w")
lines_of_text = ["a line of text", "another line of text", "a third line"]
fh.writelines(lines_of_text)
fh.close()

To append to file, use:
fh = open("Hello.txt", "a")
write("Hello World again")
fh.close()

To close a file, use
fh = open("hello.txt", "r")
print fh.read()
fh.close() 
```