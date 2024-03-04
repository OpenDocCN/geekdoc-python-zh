# Python 中的 Bytearray

> 原文：<https://www.pythonforbeginners.com/data-types/bytearray-in-python>

你一定学过 python 中不同的数据类型，比如字符串和[数字数据类型](https://www.pythonforbeginners.com/basics/numeric-types-python)，比如整数和浮点数。在本文中，您将了解 python 编程语言中另一种称为 bytearray 的数据类型。您将学习 python 中 bytearray 背后的基本概念，并对 bytearray 对象实现不同类型的操作来理解这些概念。

## Python 中的 bytearray 是什么？

python 中的 bytearray 是一个字节数组，可以保存机器可读格式的数据。当任何数据被保存在辅助存储器中时，它根据特定类型的编码被编码，例如对于字符串是 ASCII、UTF-8 和 UTF-16，对于图像是 PNG、JPG 和 JPEG，对于音频文件是 mp3 和 wav，并且被转换成字节对象。当我们使用 [python read file](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python) 操作再次访问数据时，它被解码成相应的文本、图像或音频。因此，byte 对象包含机器可读的数据，bytearray 对象是字节数组。

## 如何在 Python 中创建 bytearray 对象？

我们可以使用 bytearray()方法在 python 中创建一个 bytearray 对象。bytearray()函数将三个参数作为输入，它们都是可选的。必须转换为 bytearray 的对象作为第一个参数传递。仅当第一个参数是字符串时，才使用第二个和第三个参数。在这种情况下，第二个参数是字符串的编码格式，第三个参数是编码失败时执行的错误响应的名称。函数的作用是:返回一个 bytearray 对象。在接下来的小节中，我们将通过从不同的数据对象创建 bytes 对象来理解 bytearray()函数的工作原理。

## 创建 bytearray 对象

要创建给定大小的 bytearray 对象，我们可以将所需的 bytearray 大小作为 bytearray()函数的输入。成功执行后，它返回给定大小的 bytearray 对象，初始化为零，如下所示。

```py
myObj=bytearray(10)
print("The bytearray object is:",myObj)
print("Length of the bytearray object is:",len(myObj))
```

输出:

```py
The bytearray object is: bytearray(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
Length of the bytearray object is: 10
```

为了将字符串转换为 bytearray 对象，我们将字符串作为第一个输入，将编码类型作为第二个输入参数传递给 bytearray()函数。然后，它返回字符串的 bytearray，如下所示。

```py
myString="pythonforbeginners.com"
print("The string is:",myString)
myObj=bytearray(myString,"UTF-8")
print("The bytearray object is:",myObj)
```

输出:

```py
The string is: pythonforbeginners.com
The bytearray object is: bytearray(b'pythonforbeginners.com')
```

我们可以使用 python 中的 bytearray()函数将整数列表转换为 bytearray。bytearray()函数将 0 到 255 之间的整数列表作为输入，并返回相应的 bytearray 对象，如下所示。

```py
myList=[1,2,56,78,90]
print("The List is:",myList)
myObj=bytearray(myList)
print("The bytearray object is:",myObj)
```

输出:

```py
The List is: [1, 2, 56, 78, 90]
The bytearray object is: bytearray(b'\x01\x028NZ')
```

对于不在 0 到 255 之间的整数值，bytearray 函数按如下方式引发 ValueError。

```py
myList=[1,2,56,78,900]
print("The List is:",myList)
myObj=bytearray(myList)
print("The bytearray object is:",myObj)
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/main.py", line 3, in <module>
    myObj=bytearray(myList)
ValueError: byte must be in range(0, 256)
The List is: [1, 2, 56, 78, 900]
```

我们可以如下使用 [python try-except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 处理上述异常。

```py
myList=[1,2,56,78,900]
print("The List is:",myList)
try:
    myObj=bytearray(myList)
    print("The bytearray object is:",myObj)
except Exception as e:
    print(str(e)) 
```

输出:

```py
The List is: [1, 2, 56, 78, 900]
byte must be in range(0, 256)
```

## 对 bytearray 对象的操作

虽然 byte 对象是不可变的，但是 bytearray 对象是可变的，可以被修改，它们的行为几乎和 python 列表一样。以下是对 bytearray 对象的一些常见操作。

Bytearray 支持索引和切片。我们可以使用索引来获取特定索引处的数据，或者我们可以对 bytearray 进行切片来获取两个索引之间的数据，如下所示。

```py
myList=[1,2,56,78,90]
print("The List is:",myList)
myObj=bytearray(myList)
print("The bytearray object is:",myObj)
sliced_obj=myObj[0:2]
indexed_obj=myObj[1]
print("Sliced part of bytearray is:",sliced_obj)
print("Data at index 1 of bytearray is:",indexed_obj) 
```

输出:

```py
The List is: [1, 2, 56, 78, 90]
The bytearray object is: bytearray(b'\x01\x028NZ')
Sliced part of bytearray is: bytearray(b'\x01\x02')
Data at index 1 of bytearray is: 2
```

由于 bytearray 对象是可变的，我们也可以使用索引和切片来修改 bytearray 对象，如下所示。

```py
myList=[1,2,56,78,90]
print("The List is:",myList)
myObj=bytearray(myList)
print("The bytearray object is:",myObj)
myObj[0:2]=[15,16]
myObj[4]=34
print("The modified bytearray object is:",myObj)
```

输出:

```py
The List is: [1, 2, 56, 78, 90]
The bytearray object is: bytearray(b'\x01\x028NZ')
The modified bytearray object is: bytearray(b'\x0f\x108N"')
```

我们还可以使用 insert()方法将数据插入 bytearray 对象的给定索引处，如下所示。

```py
myList=[1,2,56,78,90]
print("The List is:",myList)
myObj=bytearray(myList)
print("The bytearray object is:",myObj)
myObj.insert(1,23)
print("The modified bytearray object is:",myObj)
```

输出:

```py
The List is: [1, 2, 56, 78, 90]
The bytearray object is: bytearray(b'\x01\x028NZ')
The modified bytearray object is: bytearray(b'\x01\x17\x028NZ')
```

我们可以使用 append()方法将数据追加到 bytearray 对象中，如下所示。

```py
myList=[1,2,56,78,90]
print("The List is:",myList)
myObj=bytearray(myList)
print("The bytearray object is:",myObj)
myObj.append(105)
print("The modified bytearray object is:",myObj)
```

输出:

```py
The List is: [1, 2, 56, 78, 90]
The bytearray object is: bytearray(b'\x01\x028NZ')
The modified bytearray object is: bytearray(b'\x01\x028NZi')
```

我们还可以使用 del 语句删除特定索引处或两个索引之间的数据，如下所示。

```py
myList=[1,2,56,78,90]
print("The List is:",myList)
myObj=bytearray(myList)
print("The bytearray object is:",myObj)
del myObj[0]
del myObj[1:3]
print("The modified bytearray object is:",myObj)
```

输出:

```py
The List is: [1, 2, 56, 78, 90]
The bytearray object is: bytearray(b'\x01\x028NZ')
The modified bytearray object is: bytearray(b'\x02Z')
```

## 结论

在本文中，我们研究了 python 中的 bytearray 数据结构。我们还将不同的数据类型转换为 bytearray，并使用内置函数执行了诸如将数据追加到 bytearray、插入数据和从 bytearray 中删除数据之类的操作。请继续关注更多内容丰富的文章。