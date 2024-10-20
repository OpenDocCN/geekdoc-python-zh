# Python IO 模块:完整的实用参考

> 原文：<https://www.askpython.com/python-modules/python-io-module>

大家好！在今天的文章中，我们将学习如何使用 Python IO 模块。

当你想执行与文件相关的 I/O 操作(如文件读/写)时，这个模块非常有用

虽然您可以使用普通的`read()`和`write()`方法来读/写文件，但是这个模块在这些操作方面给了我们更多的灵活性。

为了更好地理解这个模块，我们举几个例子。

## Python IO 模块

这个模块是标准库的一部分，所以没有必要使用 [pip](https://www.askpython.com/python-modules/python-pip) 单独安装它。

要导入 io 模块，我们可以执行以下操作:

```py
import io

```

在`io`模块中，有两个常用类对我们非常有用:

*   **字节序** - >对字节数据的 I/O 操作
*   **StringIO** - >对字符串数据的 I/O 操作

我们可以使用`io.BytesIO`和`io.StringIO`来访问这些类。

让我们一个一个来看看。

* * *

## Python BytesIO 类

在这里，我们可以用字节(`b''`)的形式保存我们的数据。当我们使用`io.BytesIO`时，数据保存在内存缓冲区中。

我们可以使用构造函数获得字节流的实例:

```py
import io
bytes_stream = io.BytesIO(b'Hello from Journaldev\x0AHow are you?')

```

请注意，我们正在传递一个字节字符串(以`b`为前缀)。

现在，`bytes_stream`只是一个字节流的句柄。

为了实际打印缓冲区内的数据，我们需要使用`bytes_stream.getvalue()`。

```py
import io
bytes_stream = io.BytesIO(b'Hello from Journaldev\x0AHow are you?')
print(bytes_stream.getvalue())

```

这里，`getvalue()`从句柄获取字节串的值。

由于字节字符串`\x0A`是换行符(' \n ')的 ASCII 表示，我们得到以下输出:

**输出**

```py
b'Hello from Journaldev\nHow are you?'

```

现在，每当我们完成工作时，关闭我们的缓冲句柄总是一个好习惯。

这也是为了确保我们释放了分配给缓冲区的所有内存。

要关闭缓冲区，请使用:

```py
bytes_stream.close()

```

现在让我们看看 StringIO 类。

* * *

## Python StringIO 类

与`io.BytesIO`类似，`io.StringIO`类可以从 StringIO 缓冲区读取与字符串相关的数据。

```py
import io

string_stream = io.StringIO("Hello from Journaldev\nHow are you?")

```

我们可以使用`string_stream.read()`从字符串缓冲区读取，使用`string_stream.write()`写入。这非常类似于从文件中读取/写入！

我们可以使用`getvalue()`打印内容。

```py
import io

string_stream = io.StringIO("Hello from Journaldev\nHow are you?")

# Print old content of buffer
print(f'Initially, buffer: {string_stream.getvalue()}')

# Write to the StringIO buffer
string_stream.write('This will overwrite the old content of the buffer if the length of this string exceeds the old content')

print(f'Finally, buffer: {string_stream.getvalue()}')

# Close the buffer
string_stream.close()

```

**输出**

```py
Initially, buffer: Hello from Journaldev
How are you?
Finally, buffer: This will overwrite the old content of the buffer if the length of this string exceeds the old content

```

由于我们正在写入同一个缓冲区，新的内容显然会覆盖旧的内容！

### 从 StringIO 缓冲区读取

与写入类似，我们也可以使用`buffer.read()`从 StringIO 缓冲区读取。

```py
import io

input = io.StringIO('This goes into the read buffer.')
print(input.read())

```

**输出**

```py
This goes into the read buffer.

```

如您所见，内容现在在读缓冲区中，它是使用`buffer.read()`打印的。

## 使用 io 读取文件

我们也可以使用`io.open()`方法直接从文件中读取，类似于从文件对象中读取。

这里，该模块为我们提供了缓冲与非缓冲读取的选择。

例如，下面将通过设置`buffering = SIZE`使用缓冲读取来读取文件。如果`SIZE` = 0，这将意味着没有缓冲！

假设`sample.txt`有以下内容:

```py
Hello from JournalDev!
How are you?
This is the last line.

```

```py
import io

# Read from a text file in binary format using io.open()
# We read / write using a buffer size of 5 bytes
file = io.open("sample.txt", "rb", buffering = 5)

print(file.read())

# Close the file
file.close()

```

**输出**

```py
b'Hello from JournalDev!\nHow are you?\nThis is the last line.\n'

```

如您所见，文件已被成功读取！这里，`io`将使用大约 5 个字节的缓冲区读取文件。

* * *

## 使用 io.open()与 os.open()

io.open()函数是执行 I/O 操作的首选方法，因为它是一个高级 Pythonic 接口。

相反，`os.open()`将执行对`open()`函数的系统调用。这将返回一个文件描述符，它不能像`io`句柄对象一样使用。

由于`io.open()`是`os.open()`的包装函数，使用这样的包装函数通常是个好习惯，因为它们会自动为您处理许多错误。

* * *

## 结论

在本文中，我们学习了如何使用 Python IO 模块，它有两个主要的类——`io.BytesIO`和`io.StringIO`,用于在缓冲区中读写字节和字符串数据。

## 参考

*   IO 模块上的 [Python 文档](https://docs.python.org/3/library/io.html)
*   关于 Python IO 模块的 JournalDev 文章

* * *