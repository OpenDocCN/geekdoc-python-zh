# Python 中字符串中字符的位置

> 原文：<https://www.pythonforbeginners.com/basics/position-of-a-character-in-a-string-in-python>

在 Python 中进行[文本分析时，在给定字符串中搜索字符是最常见的任务之一。本文讨论了如何在 python 中找到字符串中字符的位置。](https://www.pythonforbeginners.com/basics/text-analysis-in-python)

## 使用 for 循环查找字符串中字符的位置

为了找到一个字符在字符串中的位置，我们将首先使用`len()`函数找到字符串的长度。`len()`函数将一个字符串作为其输入参数，并返回该字符串的长度。我们将字符串的长度存储在一个变量`strLen`中。

获得字符串的长度后，我们将使用`range()`函数创建一个包含从 0 到`strLen-1`的值的 range 对象。`range()`函数将变量`strLen`作为其输入参数，并返回一个 range 对象。

接下来，我们将使用一个 [python for 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)和 range 对象来遍历字符串字符。当迭代时，我们将检查当前字符是否是我们正在寻找位置的字符。为此，我们将使用索引操作符。如果当前字符是我们正在寻找的字符，我们将打印该字符的位置，并使用 python 中的 break 语句退出循环。否则，我们将移动到下一个字符。

在执行 for 循环之后，如果字符出现在输入字符串中，我们将得到字符的位置，如下所示。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
range_obj=range(strLen)
character="F"
for index in range_obj:
    if myStr[index]==character:
        print("The character {} is at index {}".format(character,index))
        break
```

输出:

```py
The input string is: Python For Beginners
The character F is at index 7
```

### 使用 for 循环查找字符串中出现的所有字符

为了找到字符串中出现的所有字符，我们将从 for 循环中移除 [break 语句](https://www.pythonforbeginners.com/basics/break-and-continue-statements)。由于这个原因，for 循环将检查所有的字符，并打印出它们的位置，如果这个字符是我们正在寻找的。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
range_obj=range(strLen)
character="n"
for index in range_obj:
    if myStr[index]==character:
        print("The character {} is at index {}".format(character,index))
```

输出:

```py
The input string is: Python For Beginners
The character n is at index 5
The character n is at index 15
The character n is at index 16
```

## 使用 for 循环查找字符串中字符的最右边的索引

为了找到字符最右边的位置，我们需要修改我们的方法。为此，我们将创建一个名为`position`的变量来存储字符串中字符的位置，而不是打印位置。在遍历字符串中的字符时，只要找到所需的字符，我们就会不断更新`position`变量。

在执行 for 循环之后，我们将获得字符串中最右边的字符的索引。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
range_obj=range(strLen)
character="n"
position=-1
for index in range_obj:
    if myStr[index]==character:
        position=index
print("The character {} is at rightmost index {}".format(character,position))
```

输出:

```py
The input string is: Python For Beginners
The character n is at rightmost index 16
```

上述方法给出了字符从左侧开始的位置。如果想得到字符串中最右边的字符的位置，可以使用下面的方法。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
range_obj=range(strLen)
character="n"
for index in range_obj:
    if myStr[strLen-index-1]==character:
        print("The character {} is at position {} from right.".format(character,index+1))
        break
```

输出:

```py
The input string is: Python For Beginners
The character n is at position 4 from right.
```

在这个例子中，我们使用了[字符串索引](https://www.pythonforbeginners.com/strings/string-indexing-in-python)操作符来访问字符串右侧的元素。因此，我们可以用最少的 for 循环执行次数得到任意给定字符的最右边的位置。

## 使用 While 循环查找字符串中字符的位置

代替 for 循环，您可以使用 while 循环来查找字符串中某个字符的位置。为此，我们将使用以下步骤。

*   首先，我们将定义一个名为`position`的变量，并将其初始化为-1。
*   然后，我们将使用`len()`函数找到字符串的长度。
*   现在，我们将使用 while 来遍历字符串中的字符。如果位置变量变得大于或等于字符串的长度，我们将定义退出条件。
*   在 while 循环中，我们将首先递增`position`变量。接下来，我们将检查当前位置的字符是否是我们要寻找的字符。如果是，我们将打印字符的位置，并使用 break 语句退出 while 循环。否则，我们将移动到下一个字符。

在执行 while 循环之后，我们将获得字符串中第一个出现的字符的索引。您可以在下面的代码中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
character="n"
position=-1
while position<strLen-1:
    if myStr[position+1]==character:
        print("The character {} is at position {}.".format(character,position+1))
        break
    position+=1
```

输出:

```py
The input string is: Python For Beginners
The character n is at position 5.
```

### 使用 While 循环查找字符串中出现的所有字符

如果要查找字符串中出现的所有字符，可以从 while 循环中删除 break 语句。此后，程序将打印给定字符串中字符的所有位置，如下所示。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
character="n"
position=-1
while position<strLen-1:
    if myStr[position+1]==character:
        print("The character {} is at position {}.".format(character,position+1))
    position+=1
```

输出:

```py
The input string is: Python For Beginners
The character n is at position 5.
The character n is at position 15.
The character n is at position 16.
```

## 使用 While 循环查找字符串中字符的最右边的索引

为了使用 Python 中的 while 循环找到字符最右边的位置，我们将创建一个名为`rposition`的变量来存储字符串中字符的位置，而不是打印位置。当使用 While 循环遍历字符串的字符时，每当我们找到所需的字符时，我们将不断更新`rposition`变量。

在 while 循环执行之后，我们将获得字符串中最右边的字符的位置。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
character="n"
position=-1
rposition=0
while position<strLen-1:
    if myStr[position+1]==character:
        rposition=position+1
    position+=1
print("The character {} is at rightmost position {}.".format(character,rposition))
```

输出:

```py
The input string is: Python For Beginners
The character n is at rightmost position 16.
```

上述方法给出了字符从左侧开始的位置。如果想得到字符串中最右边的字符的位置，可以使用下面的方法。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
character="n"
position=0
while position<strLen-1:
    if myStr[strLen-position-1]==character:
        print("The character {} is at position {} from right.".format(character,position+1))
        break
    position+=1
```

输出:

```py
The input string is: Python For Beginners
The character n is at position 4 from right.
```

## 使用 Find()方法查找字符在字符串中的位置

要查找字符串中某个字符的位置，也可以使用`find()`方法。在字符串上调用`find()`方法时，它将一个字符作为输入参数。执行后，它返回字符串中第一个出现的字符的位置。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
character="n"
position=myStr.find(character)
print("The character {} is at position {}.".format(character,position))
```

输出:

```py
The input string is: Python For Beginners
The character n is at position 5.
```

建议阅读:[用 Python 创建聊天应用](https://codinginfinite.com/python-chat-application-tutorial-source-code/)

## 使用 Index()方法查找字符串中某个字符的索引

`index()`方法用于查找字符串中某个字符的索引。在字符串上调用`index()`方法时，它将一个字符作为输入参数。执行后，它返回字符串中第一个出现的字符的位置。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
character="n"
position=myStr.index(character)
print("The character {} is at index {}.".format(character,position))
```

输出:

```py
The input string is: Python For Beginners
The character n is at index 5.
```

## 使用 rfind()方法在 Python 中查找字符串的最右边的索引

要找到 python 字符串中某个字符的最右边位置，也可以使用`rfind()`方法。除了返回字符串中输入字符最右边的位置之外，`rfind()` 方法的工作方式与`find()`方法类似。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
strLen=len(myStr)
character="n"
position=myStr.rfind(character)
print("The character {} is at rightmost position {} .".format(character,position))
```

输出:

```py
The input string is: Python For Beginners
The character n is at rightmost position 16 .
```

## 结论

在本文中，我们讨论了在 Python 中查找字符串中字符位置的不同方法。要了解关于这个主题的更多信息，您可以阅读这篇关于如何[在字符串](https://www.pythonforbeginners.com/basics/find-all-occurrences-of-a-substring-in-a-string-in-python)中找到一个子字符串的所有出现的文章。

您可能也会喜欢这篇关于 [python simplehttpserver](https://www.pythonforbeginners.com/modules-in-python/how-to-use-simplehttpserver) 的文章。

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！