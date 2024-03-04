# 使用 Python 中的 os 模块进行文件处理

> 原文：<https://www.pythonforbeginners.com/filehandling/file-handling-using-os-module-in-python>

在 python 中，您可能已经使用了内置函数来对文件执行操作。在本文中，我们将尝试使用操作系统模块来实现文件处理，以使用 python 中的不同函数来执行文件操作，如 [python 读取文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)、写入文件和追加文件。

## Python 中如何使用 os 模块实现文件处理？

os 模块可用于使用 python 对文件系统执行不同的操作。与内置函数不同，os 模块以字节的形式向文件系统读写数据。像 open()、write()、read()和 close()这样的方法也是在 os 模块中定义的，与内置函数相比具有不同的规范和属性。要使用这些方法，我们可以按如下方式导入 os 模块。

```py
import os
```

## 使用操作系统模块打开文件

要使用 os 模块在 python 中打开文件，我们可以使用 open()方法。open()方法接受两个参数作为输入。第一个参数是需要打开的文件名，第二个参数是打开文件的模式。最常见的模式参数定义如下。

*   os。O_RDONLY 模式用于以只读模式打开文件。
*   os。O_WRONLY 模式用于以只写模式打开文件。
*   os。O_RDWR 模式用于打开文件进行读写。
*   os。O_APPEND 模式用于以附加模式打开文件，当文件以这种模式打开时，数据在每次写操作时被附加到文件中。
*   os。O_CREAT 模式用于创建一个文件，然后在该文件不存在时打开它。

open()方法在成功打开文件时返回一个文件描述符。可以通过指定文件名和模式打开文件，如下所示。

```py
 myFile=os.open("filename.txt",os.O_RDONLY)
```

在管道运算符“|”的帮助下，我们可以在打开文件时同时使用一种或多种模式，如下所示。

```py
 myFile=os.open("filename.txt",os.O_RDONLY|os.O_CREAT)
```

## 关闭文件

在 python 中，我们必须在程序终止之前关闭所有打开的文件，使用 close()方法来执行成功的写操作。os 模块中的 close()方法也可以用于同样的目的。close()方法将文件描述符作为输入参数，并在成功完成时关闭文件。

**在进行文件操作时，我们必须使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 进行异常处理，并调用 finally 块中的 close()方法，这样即使在文件执行过程中出现任何错误，文件也会被关闭。**打开的文件可以使用 close()方法关闭，如下所示。

```py
 os.close(myFile)
```

## 使用操作系统模块读取文件

我们可以使用 read()方法通过 python 中的 os 模块来读取文件。read()方法将文件描述符作为其第一个参数，将要读取的字节数作为第二个参数，并返回一个字节字符串，需要使用 decode()方法对该字符串进行解码以获得实际数据。

在字节字符串上调用 decode()方法时，它采用要解码的文件的编码类型，并以字符串格式返回数据。

我们可以使用 read()和 decode()方法从文件中读取数据，如下所示。

```py
import os
try:
    myFile=os.open("/home/aditya1117/filename.txt",os.O_RDONLY)
    myData=os.read(myFile,105)
    myStr=myData.decode("UTF-8")
    print(myStr)
except Exception as e:
    print(str(e))
finally:
    os.close(myFile)
```

输出:

```py
This is a sample text file.
This file is for PythonForBeginners.com.
This file has been written by Aditya
```

## 使用操作系统模块写入文件

我们可以使用 write()方法通过 os 模块将数据写入文件系统。write()方法以文件描述符作为第一个参数，以字节格式写入文件的数据作为第二个参数。它从文件描述符的开始处将数据写入文件。成功完成写入操作后，它会返回写入文件的字节数。

为了将我们的数据转换成字节格式，我们将使用 encode()方法。在任何对象上调用 encode()方法时，都将编码格式作为输入，并以字节格式返回数据。

首先，我们将在操作系统中打开文件。O_WRONLY 或 os。O_RDWR 模式，然后我们可以使用 encode()方法和 write()方法，使用 python 中的 os 模块将任何数据写入文件系统，如下所示。

```py
import os
try:
    myFile=os.open("/home/aditya1117/filename1.txt",os.O_WRONLY)
    myStr="Hi This is Pythonforbeginners.com"
    myData=myStr.encode("UTF-8")
    os.write(myFile,myData)
except Exception as e:
    print(str(e))
finally:
    os.close(myFile)
```

要执行追加操作，我们只需使用 os 打开文件。O_APPEND 与 os 一起。Owronly 模式，然后我们可以使用 write()方法将数据追加到文件中，如下所示。

```py
import os

try:
    myFile = os.open("/home/aditya1117/filename2.txt", os.O_WRONLY | os.O_APPEND)
    myStr = "Hi! This is Pythonforbeginners.com"
    myData = myStr.encode("UTF-8")
    os.write(myFile, myData)
except Exception as e:
    print(str(e))
finally:
    os.close(myFile)
```

## 结论

在本文中，我们使用 os 模块在 python 中实现了对文件的不同操作。请继续关注更多内容丰富的文章。