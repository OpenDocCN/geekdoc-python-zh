# Python 中的文件处理

> 原文：<https://www.pythonforbeginners.com/filehandling/file-handling-in-python>

在实际应用中，我们经常需要从文件中读取数据，并将数据写入文件。在本文中，我们将学习 python 中的文件处理，并将使用 python 中的不同函数实现不同的操作，如 [python 读取文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)，写入文件和追加文件。

## 用 Python 打开文件

要在 python 中打开一个文件，我们可以使用 open()函数。通常有两个输入参数传递给 open()函数。第一个参数是需要打开的文件名，第二个参数是打开文件的模式。最常见的模式参数定义如下。

*   “r”模式用于以只读模式打开文件。
*   “w”模式用于以写模式打开文件。它还会用相同的名称覆盖现有文件，或者创建一个新文件(如果它不存在)。
*   “a”模式用于以追加模式打开文件。如果文件不存在，它会创建一个新文件，但如果文件已经存在，它不会覆盖该文件。
*   “b”模式用于以二进制模式打开文件。

open()函数在成功打开文件时返回一个 file 对象。可以通过指定文件名和模式打开文件，如下所示。

```py
myFile=open("filename.txt",mode="r") 
```

我们还可以将文件的编码指定为第三个输入参数。默认情况下，编码取决于操作系统。我们可以如下明确指定编码。

```py
myFile=open("filename.txt",mode="r",encoding="UTF-8")
```

## 关闭文件

在 python 中，我们必须在程序终止之前使用 close()方法关闭所有打开的文件。对 file 对象调用 close()方法时，会关闭文件。**在进行文件操作时，我们必须使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 进行异常处理，并调用 finally 块中的 close()方法，这样即使在程序执行过程中出现任何错误，文件也会被关闭。**打开的文件可以使用 close()方法关闭，如下所示。

```py
myFile.close()
```

## 从文件中读取

打开文件后，我们可以使用 read()方法从文件中读取数据。read()方法将一个可选参数作为输入，指定要从文件中读取的字符数。如果在 file 对象上调用 read()方法而不带任何参数，它将读取整个文件并以文本字符串的形式返回。这可以从下面的例子中看出。

```py
try:
    myFile=open("/home/aditya1117/filename.txt",mode="r")
    print(myFile.read())
except Exception as e:
    print(str(e))
finally:
    myFile.close()
```

输出:

```py
This is a sample text file.
This file is for PythonForBeginners.com.
This file has been written by Aditya Raj for demonstration.
```

我们可以从文件中读取一定数量的字符，如下所示。

```py
try:
    myFile=open("/home/aditya1117/filename.txt",mode="r")
    print(myFile.read(21))
except Exception as e:
    print(str(e))
finally:
    myFile.close()
```

输出:

```py
This is a sample text 
```

我们也可以使用 readline()方法逐行读取文件，如下所示。

```py
try:
    myFile=open("/home/aditya1117/filename.txt",mode="r")
    print(myFile.readline())
except Exception as e:
    print(str(e))
finally:
    myFile.close()
```

输出:

```py
This is a sample text file. 
```

我们还可以使用 for 循环迭代文件。通过这种方式，迭代器逐行迭代文件对象。这可以看如下。

```py
try:
    myFile=open("/home/aditya1117/filename.txt",mode="r")
    for line in myFile:
        print(line)
except Exception as e:
    print(str(e))
finally:
    myFile.close()
```

输出:

```py
This is a sample text file.

This file is for PythonForBeginners.com.

This file has been written by Aditya Raj for demonstration. 
```

## 用 Python 写文件

我们可以使用 write()方法将任何字符串写入打开的文件，如下所示。

```py
try:
    myFile=open("/home/aditya1117/filename1.txt",mode="w")
    myFile.write("This string is being added to the file.")
except Exception as e:
    print(str(e))
finally:
    myFile.close()
```

当文件以“w”模式打开时，它会覆盖现有文件。如果我们想保持以前的数据不变，并向其中添加新数据，我们可以使用“a”表示的追加模式打开文件，如下所示。

```py
try:
    myFile=open("/home/aditya1117/filename1.txt",mode="a")
    myFile.write("This string is being appended to the file.")
except Exception as e:
    print(str(e))
finally:
    myFile.close()
```

## 结论

在本文中，我们已经理解了 python 中文件处理的概念。我们已经看到了如何打开一个文件，从中读取数据，将数据写入文件，然后关闭文件。请继续关注更多内容丰富的文章。