# 如何在 Python 中反转字符串

> 原文：<https://www.pythonforbeginners.com/basics/how-to-reverse-a-string-in-python>

我们可以在 python 中对字符串执行很多操作。这些操作包括分割字符串、字符串连接、反转字符串、分割字符串等等。在本文中，我们将使用不同的方法在 python 中反转一个字符串。

## 使用切片反转字符串

在 python 中反转字符串的第一个也是最简单的方法是使用切片。我们可以使用语法`string_name[ start : end : interval ]`创建任何字符串的切片，其中 start 和 end 是必须从字符串切片的子字符串的开始和结束索引。interval 参数用于以指定的间隔选择字符。

要使用切片来反转整个字符串，我们将把开始和结束参数留空。这将导致从整个字符串创建切片。interval 参数将被指定为-1，以便连续的字符以相反的顺序包含在分片的字符串中。这可以从下面的例子中理解。

```py
myString = "PythonForBeginners"
reversedString = myString[:: -1]
print("Original String is:", myString)
print("Reversed String is:", reversedString)
```

输出:

```py
Original String is: PythonForBeginners
Reversed String is: srennigeBroFnohtyP
```

## 使用 for 循环反转字符串

要使用 for 循环反转字符串，我们将首先计算字符串的长度。然后，我们将创建一个新的空字符串。然后，我们将从末尾到开始逐个字符地访问字符串，并将它添加到新创建的字符串中。这将导致创建一个新字符串，该字符串中的字符与原始字符串的顺序相反。从下面的例子可以很容易理解这一点。

```py
myString = "PythonForBeginners"
reversedString = ""
strlen = len(myString)
for count in range(strlen):
    character = myString[strlen - 1 - count]
    reversedString = reversedString + character

print("Original String is:",myString)
print("Reversed String is:", reversedString)
```

输出:

```py
Original String is: PythonForBeginners
Reversed String is: srennigeBroFnohtyP
```

## 使用 Python 列表反转字符串

我们也可以在 python 中使用 list 来反转一个字符串。在这个方法中，首先，我们将把字符串转换成一个字符列表，并创建一个新的空字符串。之后，我们将从列表的末尾逐个取出字符，然后我们将执行[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)来将字符添加到新字符串中。

这样，将创建一个新字符串，它将是原始字符串的反向。这可以如下进行。

```py
myString = "PythonForBeginners"
charList=[]
charList[:0] = myString
reversedString = ""

strlen = len(charList)
for count in range(strlen):
    character = charList.pop()
    reversedString = reversedString+character

print("Original String is:", myString)
print("Reversed String is:", reversedString) 
```

输出:

```py
Original String is: PythonForBeginners
Reversed String is: srennigeBroFnohtyP
```

## 使用递归反转字符串

为了使用递归来反转一个字符串，我们将使用下面的过程。

假设我们定义了一个函数 reverseString(input_string)来反转字符串。首先我们将检查 input_string 是否为空，如果是，那么我们将返回 input_string。否则，我们将从 input_string 中取出最后一个字符，并为剩余部分调用 reverseString()函数，并将它的输出连接在 input_string 的最后一个字符之后，如下所示。

```py
myString = "PythonForBeginners"

def reverseString(input_string):
    strlen = len(input_string)
    if strlen == 0:
        return ""
    else:
        last = input_string[-1]
        rest = input_string[0:strlen - 1]
        return last + reverseString(rest)

reversedString = reverseString(myString)
print("Original String is:", myString)
print("Reversed String is:", reversedString)
```

输出:

```py
Original String is: PythonForBeginners
Reversed String is: srennigeBroFnohtyP
```

## 使用 reversed()方法

reversed()方法返回作为输入参数传递给它的任何可迭代对象的反向迭代器。我们可以使用反向迭代器来创建输入字符串的反向。我们将首先创建一个空字符串，然后通过迭代 reversed()函数返回的反向迭代器以逆序添加字符，如下所示。

```py
myString = "PythonForBeginners"
reverse_iterator = reversed(myString)
reversedString = ""
for char in reverse_iterator:
    reversedString = reversedString + char

print("Original String is:", myString)
print("Reversed String is:", reversedString)
```

输出:

```py
Original String is: PythonForBeginners
Reversed String is: srennigeBroFnohtyP
```

## 结论

在本文中，我们研究了在 python 中反转字符串的不同方法。。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。