# Python 中检查字符串是否为空的 4 种方法

> 原文：<https://www.askpython.com/python/examples/check-whether-string-is-empty>

这个标题听起来有点奇怪，因为有人可能认为我们可以在 len()或 not 操作符的帮助下简单地检查 spring 是否为空。但我们需要记住，它将空格作为字符串中的一个字符，并将字符串显示为非空字符串。在本文中，我们将学习可以用来检查字符串是否为空的方法。我们开始吧！

## 方法来检查 Python 中的字符串是否为空

让我们看看 Python 中检查字符串是否为空的 4 种不同方法。我们将通过一个例子来探究每种方法，并显示输出，以帮助您自己执行同样的操作。

### 1.使用 NOT 运算符

此方法将带有空格的字符串视为非空字符串。它将字符串中的空格算作一个字符。我们应该知道带空格的字符串是一个空字符串，并且大小不为零，但是这个方法忽略了这个事实。

**例如**

```py
str1 = ""
str2 = "  "

if(not str1):
    print ("Yes the string is empty")
else :
    print ("No the string is not empty")

if(not str2):
    print ("Yes the string is empty")
else :
    print ("No the string is not empty"

```

**输出:**

```py
Yes the string is empty
No the string is not empty

```

你可以看到它把带空格的字符串打印成了一个非空字符串。

### 2.使用 len()函数

与 not 操作符一样，这也将带有空格的字符串视为非空字符串。此方法检查非空的零长度字符串。

**例如:**

```py
str1 = ""
str2 = "  "

if(len(str1) == 0):
    print ("Yes the string is empty ")
else :
    print ("No the string is not empty")

if(len(str2) == 0):
    print ("Yes the strinf is empty")
else :
    print ("No the string is not empty")

```

**输出:**

```py
Yes the string is empty 
No the string is not empty

```

### 3.使用 not+str.strip()方法

这个方法没有忽略空+非零长度字符串的事实。因此，此方法可以用于检查空的零长度字符串。它寻找一个空的非零长度的字符串。

**例如:**

```py
str1 = ""
str2 = "  "

if(not (str1 and str1.strip())):
    print ("Yes the string is empty")
else :
    print ("No the string is not empty")

if(not(str2 and str2.strip())):
    print ("Yes the string is empty")
else :
    print ("No the string is not empty")

```

**输出:**

```py
Yes the string is empty
Yes the string is empty

```

### 4.使用 not str.isspace 方法

这个方法和上面的方法类似。这种方法被认为是更健壮的，因为它执行剥离操作，如果字符串包含大量空格，该操作将承担计算责任。

```py
str1 = ""
str2 = "  "

if(not (str1 and not str1.isspace())):
    print ("Yes the string is empty")
else :
    print ("No the string is not empty")

if(not (str2 and not str2.isspace())):
    print ("Yes the string is empty")
else :
    print ("No the string is not empty")

```

**输出:**

```py
Yes the string is empty
Yes the string is empty

```

## 结论

所以在这篇文章中，我们学习了很多不同的方法来检查空字符串。虽然每种方法都有自己的缺点，但您可以根据自己的适合程度来使用它们。