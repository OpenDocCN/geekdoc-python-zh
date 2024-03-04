# Python 中的字节

> 原文：<https://www.pythonforbeginners.com/basics/bytes-in-python>

你一定学过 python 中不同的数据类型，比如字符串和[数字数据类型](https://www.pythonforbeginners.com/basics/numeric-types-python)，比如整数和浮点数。在本文中，您将了解另一种称为字节的数据类型。您将学习 python 中字节背后的基本概念，并对字节实现不同类型的操作来理解这些概念。

## Python 中的字节是什么？

通常，当我们在辅助存储器中保存任何数据时，它都根据某种类型的编码进行编码，例如字符串的 ASCII、UTF-8 和 UTF-16，图像的 PNG、JPG 和 JPEG，以及音频文件的 mp3 和 wav，并被转换成字节对象。当我们使用 python 读文件操作再次访问数据时，它被解码成相应的文本、图像或音频。字节对象包含机器可读的数据，我们可以将字节对象直接存储到二级存储器中。

在 python 中，我们可以从列表、字符串等其他数据中显式地创建字节对象。

## 如何在 Python 中创建字节？

要创建字节对象，我们可以使用 bytes()函数。bytes()函数将三个参数作为输入，它们都是可选的。必须转换成字节的对象作为第一个参数传递。仅当第一个参数是字符串时，才使用第二个和第三个参数。在这种情况下，第二个参数是字符串的编码，第三个参数是编码失败时执行的错误响应的名称。bytes()函数返回一个不可变的字节对象。在接下来的小节中，我们将通过从不同的数据对象创建 bytes 对象来理解 bytes()函数的工作原理。

## 创建一个给定大小的字节对象

要创建任意给定大小的 bytes 对象，我们将把大小作为输入传递给 bytes()方法，然后创建一个所需大小的 bytes 对象，它被初始化为全零。这可以从下面的例子中理解。

```py
bytes_obj = bytes(10)
print("The bytes object is:", bytes_obj)
print("Size of the bytes object is:", len(bytes_obj) )
```

输出:

```py
The bytes object is: b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
Size of the bytes object is: 10
```

## 将字符串转换为字节

要将字符串转换为 bytes 对象，我们将把字符串作为第一个输入，把编码作为第二个输入传递给 bytes()函数。错误响应还有第三个参数，但为了简单起见，此时可以忽略。该函数返回一个包含编码字符串的 bytes 对象。这可以这样理解。

```py
myString = "Pythonforbeginners.com"
print("The given string is:" , myString)
bytes_obj = bytes(myString , "UTF-8")
print("The bytes object is:", bytes_obj)
print("Size of the bytes object is:", len(bytes_obj) )
```

输出:

```py
The given string is: Pythonforbeginners.com
The bytes object is: b'Pythonforbeginners.com'
Size of the bytes object is: 22
```

## 将列表转换为字节

我们还可以使用 bytes()函数将任何可迭代对象(如 list 或 tuple)转换为 bytes 对象。要执行这个操作，我们只需将 iterable 对象传递给 bytes()函数，该函数返回相应的 bytes 对象。请记住，字节对象是不可变的，不能被修改。我们可以使用 bytes()函数将列表转换成字节，如下所示。

```py
myList = [1,2,3,4,5]
print("The given list is:" , myList)
bytes_obj = bytes(myList)
print("The bytes object is:", bytes_obj)
print("Size of the bytes object is:", len(bytes_obj) )
```

输出:

```py
The given list is: [1, 2, 3, 4, 5]
The bytes object is: b'\x01\x02\x03\x04\x05'
Size of the bytes object is: 5
```

记住**传递给 bytes()函数的列表应该只包含元素。用浮点数或字符串传递 s list 会导致 bytes()函数抛出 TypeError。**

## 结论

在本文中，我们已经了解了什么是 bytes 对象，以及如何使用 bytes()方法从 iterables 和 strings 创建 bytes 对象。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。