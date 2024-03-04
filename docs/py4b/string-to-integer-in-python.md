# Python 中的字符串到整数

> 原文：<https://www.pythonforbeginners.com/strings/string-to-integer-in-python>

在用 Python 编程的过程中，我们经常需要在 Python 中将一个字符串转换成一个整数。这是因为 Python 中的标准输入总是被读取为独立于输入类型的字符串。为了在我们的程序中使用整数数据，当它作为空格分隔的整数被传递时，我们需要在使用 [Python string split](https://www.pythonforbeginners.com/dictionary/python-split) 操作分割它们之后将字符串输入转换成整数。在这篇文章中，我们将看看如何将一个字符串正确无误地转换成整数，并用 Python 实现这些程序。

## Python 中如何把字符串转换成整数？

在 Python 中我们可以使用 int()函数将一个字符串转换成整数。必须转换为整数的字符串作为输入参数传递给 int()函数，如果作为输入传递的字符串格式正确，并且在字符串转换为整数的过程中没有发生错误，则该函数返回相应的整数值。我们可以使用 int()函数将字符串转换为整数，如下所示。

```py
print("Input String is:")
myInput= "1117"
print(myInput)
print("Output Integer is:")
myInt=int(myInput)
print(myInt)
```

输出:

```py
Input String is:
1117
Output Integer is:
1117
```

当输入字符串的格式不正确时，int()函数会引发 ValueError。这可以从下面的例子中看出。

```py
print("Input String is:")
myInput= "aditya1117"
print(myInput)
print("Output Integer is:")
myInt=int(myInput)
print(myInt)
```

输出:

```py
Input String is:
aditya1117
Output Integer is:
Traceback (most recent call last):

  File "<ipython-input-3-c8793975130e>", line 5, in <module>
    myInt=int(myInput)

ValueError: invalid literal for int() with base 10: 'aditya1117'
```

## 将字符串转换为整数时，哪些输入值会出现错误？

在几种情况下，int()函数在将字符串转换为整数时会引发 ValueError。下文讨论了一些案例。

当我们传递一个包含字母而不是数字的字符串时，将会发生 ValueError，并且输入字符串不会被转换成整数。这可以从下面的例子中看出。

```py
print("Input String is:")
myInput= "aditya1117"
print(myInput)
print("Output Integer is:")
myInt=int(myInput)
print(myInt)
```

输出:

```py
Input String is:
aditya1117
Output Integer is:
Traceback (most recent call last):

  File "<ipython-input-10-c8793975130e>", line 5, in <module>
    myInt=int(myInput)

ValueError: invalid literal for int()
```

当传递的字符串包含任何空格字符和数字文字时，将发生 ValueError，并且输入字符串不会转换为整数。这可以从下面的例子中看出。

```py
print("Input String is:")
myInput= "11 17"
print(myInput)
print("Output Integer is:")
myInt=int(myInput)
print(myInt)
```

输出:

```py
Input String is:
11 17
Output Integer is:
Traceback (most recent call last):

  File "<ipython-input-4-46d411efb04b>", line 5, in <module>
    myInt=int(myInput)

ValueError: invalid literal for int() with base 10: '11 17'
```

当传递的字符串包含任何标点符号(如句点字符。)或逗号(，)以及数字文字，将出现 ValueError，并且输入字符串不会转换为整数。这可以从下面的例子中看出。

```py
 print("Input String is:")
myInput= "11.17"
print(myInput)
print("Output Integer is:")
myInt=int(myInput)
print(myInt)
```

输出:

```py
Input String is:
11.17
Output Integer is:
Traceback (most recent call last):

  File "<ipython-input-5-97993fa7ba5b>", line 5, in <module>
    myInt=int(myInput)

ValueError: invalid literal for int() with base 10: '11.17'
```

## 在将字符串转换为整数时如何避免 ValueError？

在 Python 中将字符串转换为整数时，我们可以先发制人地检查传递的字符串是否只包含数字，这样我们就可以避免错误的发生，或者我们可以使用 Python try except 来处理由 int()函数引发的 ValueError。下面讨论了这两种方法。

我们可以使用 isdigit()方法来检查一个字符串是否只包含数字字符。对字符串调用 isdigit()方法时，如果字符串仅由数字组成，则返回 true。否则返回 false。这可以如下实现。

```py
print("Input String is:")
myInput= "1117"
print(myInput)
if myInput.isdigit():
    print("Output Integer is:")
    myInt=int(myInput)
    print(myInt)
else:
    print("Input cannot be converted into integer.")
```

输出:

```py
Input String is:
1117
Output Integer is:
1117
```

如果输入字符串包含数字以外的字符，输出将如下所示。

```py
 print("Input String is:")
myInput= "aditya1117"
print(myInput)
if myInput.isdigit():
    print("Output Integer is:")
    myInt=int(myInput)
    print(myInt)
else:
    print("Input cannot be converted into integer.") 
```

输出:

```py
Input String is:
aditya1117
Input cannot be converted into integer.
```

为了在 ValueError 发生后对其进行处理，我们可以使用异常处理，使用 [Python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 来处理 ValueError 并向用户显示正确的消息，如下所示。

```py
 print("Input String is:")
myInput= "1117"
print(myInput)
try:
    print("Output Integer is:")
    myInt=int(myInput)
    print(myInt)
except ValueError:
    print("Input cannot be converted into integer.")
```

输出:

```py
Input String is:
1117
Output Integer is:
1117
```

如果输入字符串包含数字以外的字符，输出将如下所示。

```py
print("Input String is:")
myInput= "aditya1117"
print(myInput)
try:
    myInt=int(myInput)
    print("Output Integer is:")
    print(myInt)
except ValueError:
    print("Input cannot be converted into integer.")
```

输出:

```py
Input String is:
aditya1117
Input cannot be converted into integer.
```

## 结论

在本文中，我们看到了如何在 Python 中将字符串转换为整数，以及在转换过程中会出现什么问题。我们还看到了如何避免和处理在字符串到整数的转换过程中由 int()函数引发的 ValueError。请继续关注更多内容丰富的文章。