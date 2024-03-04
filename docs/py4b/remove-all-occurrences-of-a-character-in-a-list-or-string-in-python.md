# 在 Python 中移除列表或字符串中出现的所有字符

> 原文：<https://www.pythonforbeginners.com/basics/remove-all-occurrences-of-a-character-in-a-list-or-string-in-python>

在自然语言处理、数据科学和数据挖掘等领域，我们需要处理大量的文本数据。为此，我们通常在 Python 中使用字符串和列表。给定一个字符列表或字符串，我们有时需要从列表或字符串中删除一个或所有出现的字符。在本文中，我们将讨论在 python 中从列表或[字符串中删除所有出现的字符的不同方法。](https://www.pythonforbeginners.com/strings/remove-commas-from-string-in-python)

## 使用 pop()方法从列表中移除元素

给定一个列表，我们可以使用`pop()`方法从列表中删除一个元素。当在列表上调用`pop()` 方法时，从列表中删除最后一个元素。执行后，它返回删除的元素。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", myList)
x = myList.pop()
print("The popped element is:", x)
print("The modified list is:", myList)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The popped element is: 7
The modified list is: [1, 2, 3, 4, 5, 6]
```

在上面的例子中，您可以看到在执行了`pop()`方法之后，列表的最后一个元素已经从列表中删除了。

但是，如果输入列表为空，程序将运行到`IndexError`。这意味着你试图从一个空列表中弹出一个元素。例如，看看下面的例子。

```py
myList = []
print("The original list is:", myList)
x = myList.pop()
print("The popped element is:", x)
print("The modified list is:", myList)
```

输出:

```py
The original list is: []
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    x = myList.pop()
IndexError: pop from empty list
```

您可以观察到程序遇到了一个`IndexError`异常，并显示消息“`IndexError: pop from empty list`”。

您还可以使用`pop()`方法从列表中的特定索引中删除元素。为此，您需要提供元素的索引。

当在列表上调用时，`pop()`方法将需要删除的元素的索引作为其输入参数。执行后，它从给定的索引中移除元素。`pop()`方法也返回被移除的元素。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8]
print("The original list is:", myList)
x = myList.pop(3)
print("The popped element is:", x)
print("The modified list is:", myList)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7, 8]
The popped element is: 4
The modified list is: [1, 2, 3, 5, 6, 7, 8]
```

这里，我们使用`pop()`方法弹出了列表中索引 3 处的元素。

## 使用 remove()方法从列表中删除元素

如果不知道要删除的元素的索引，可以使用 `remove()` 方法。当在列表上调用时，`remove()`方法将一个元素作为它的输入参数。执行后，它从列表中移除第一个出现的 input 元素。`remove()` 方法不返回任何值。换句话说，它返回`None`。

您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 3, 4,3, 5,3, 6, 7, 8]
print("The original list is:", myList)
myList.remove(3)
print("The modified list is:", myList)
```

输出:

```py
The original list is: [1, 2, 3, 4, 3, 5, 3, 6, 7, 8]
The modified list is: [1, 2, 4, 3, 5, 3, 6, 7, 8]
```

在上面的例子中，您可以看到在执行了`remove()`方法之后，第一次出现的元素 3 被从列表中删除。

如果`remove()`方法的输入参数中给出的值不在列表中，程序将运行到如下所示的`ValueError`异常。

```py
myList = []
print("The original list is:", myList)
myList.remove(3)
print("The modified list is:", myList)
```

输出:

```py
The original list is: []
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    myList.remove(3)
ValueError: list.remove(x): x not in list
```

在上面的例子中，列表是空的。因此，数字 3 不是列表中的一个元素。因此，当我们调用`remove()`方法时，程序运行到`ValueError`异常，并显示消息“`ValueError: list.remove(x): x not in list`”。

到目前为止，我们已经讨论了如何从列表中删除一个元素。现在让我们来讨论如何在 python 中删除字符列表中某个字符的所有出现。

## 从列表中删除所有出现的字符

给定一个字符列表，我们可以使用 for 循环和`append()`方法删除所有出现的值。为此，我们将使用以下步骤。

*   首先，我们将创建一个名为`outputList`的空列表。为此，您可以使用方括号或`list()` 构造函数。
*   创建完`outputList`后，我们将使用 for 循环遍历输入的字符列表。
*   在遍历列表元素时，我们将检查是否需要删除当前字符。
*   如果是，我们将使用[继续语句](https://www.pythonforbeginners.com/basics/break-and-continue-statements)移动到下一个字符。否则，我们将使用 `append()`方法将当前字符追加到`outputList`中。

在执行 for 循环之后，我们将在 output list 中获得字符的输出列表。在 outputList 中，除了那些我们需要从原始列表中删除的字符之外，所有的字符都将出现。您可以在下面的 python 程序中观察到这一点。

```py
myList = ['p', 'y', 'c', 't', 'c', 'h', 'o', 'n', 'f', 'c', 'o', 'r', 'b', 'e', 'g', 'c', 'i', 'n', 'n', 'c', 'e', 'r',
          's']
print("The original list is:", myList)
outputList = []
charToDelete = 'c'
print("The character to be removed is:", charToDelete)
for character in myList:
    if character == charToDelete:
        continue
    outputList.append(character)
print("The modified list is:", outputList)
```

输出:

```py
The original list is: ['p', 'y', 'c', 't', 'c', 'h', 'o', 'n', 'f', 'c', 'o', 'r', 'b', 'e', 'g', 'c', 'i', 'n', 'n', 'c', 'e', 'r', 's']
The character to be removed is: c
The modified list is: ['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

### 使用列表理解从列表中删除一个字符的所有出现

不使用 for 循环，我们可以使用[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)和成员操作符“`in`”来删除给定字符的所有出现。

`in`操作符是一个二元操作符，它将一个元素作为第一个操作数，将一个容器对象(如 list)作为第二个操作数。执行后，如果容器对象中存在该元素，它将返回`True`。否则，它返回`False`。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 3, 4,3, 5,3, 6, 7, 8]
print("The list is:", myList)
print(3 in myList)
print(1117 in myList) 
```

输出:

```py
The list is: [1, 2, 3, 4, 3, 5, 3, 6, 7, 8]
True
False
```

使用 list comprehension 和'`in`'操作符，我们可以创建一个新列表，其中包含除了我们需要从原始列表中删除的字符之外的所有字符，如下例所示。

```py
myList = ['p', 'y', 'c', 't', 'c', 'h', 'o', 'n', 'f', 'c', 'o', 'r', 'b', 'e', 'g', 'c', 'i', 'n', 'n', 'c', 'e', 'r',
          's']
print("The original list is:", myList)
charToDelete = 'c'
print("The character to be removed is:", charToDelete)
outputList = [character for character in myList if character != charToDelete]
print("The modified list is:", outputList)
```

输出:

```py
The original list is: ['p', 'y', 'c', 't', 'c', 'h', 'o', 'n', 'f', 'c', 'o', 'r', 'b', 'e', 'g', 'c', 'i', 'n', 'n', 'c', 'e', 'r', 's']
The character to be removed is: c
The modified list is: ['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

### 使用 Remove()方法从列表中移除元素的所有匹配项

我们也可以从原来的列表中删除一个角色的所有实例，而不是创建一个新的列表。为此，我们将使用`remove()`方法和成员操作符“`in`”。

*   为了从给定的字符列表中删除给定元素的所有实例，我们将使用'`in`'操作符检查该字符是否出现在列表中。如果是，我们将使用 remove 方法从列表中删除该字符。
*   我们将使用一个 [while 循环](https://www.pythonforbeginners.com/loops/python-while-loop)来反复检查该字符的存在，以将其从列表中删除。
*   在删除给定字符的所有匹配项后，程序将从 while 循环中退出。

这样，我们将通过修改原始列表得到输出列表，如下所示。

```py
myList = ['p', 'y', 'c', 't', 'c', 'h', 'o', 'n', 'f', 'c', 'o', 'r', 'b', 'e', 'g', 'c', 'i', 'n', 'n', 'c', 'e', 'r',
          's']
print("The original list is:", myList)
charToDelete = 'c'
print("The character to be removed is:", charToDelete)
while charToDelete in myList:
    myList.remove(charToDelete)
print("The modified list is:", myList)
```

输出:

```py
The original list is: ['p', 'y', 'c', 't', 'c', 'h', 'o', 'n', 'f', 'c', 'o', 'r', 'b', 'e', 'g', 'c', 'i', 'n', 'n', 'c', 'e', 'r', 's']
The character to be removed is: c
The modified list is: ['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

### 使用 filter()函数删除所有出现的字符

我们也可以使用`filter()`函数从字符列表中删除所有出现的字符。

`filter()`函数将另一个函数比如`myFun`作为它的第一个输入参数，将一个类似 list 的容器对象作为它的第二个参数。这里，`myFun`应该把容器对象的一个元素作为它的输入参数。执行后，它应该返回`True`或`False`。

如果对于输入中给定的容器对象的任何元素,`myFun`的输出是`True`,则该元素包含在输出中。否则，元素不会包含在输出中。

要使用`filter()`方法删除列表中给定项目的所有出现，我们将遵循以下步骤。

*   首先，我们将定义一个函数`myFun`，它将一个字符作为输入参数。如果输入的字符等于我们需要删除的字符，它返回`False`。否则，它应该返回`True`。
*   定义了`myFun`之后，我们将把`myFun`作为第一个参数，把字符列表作为第二个输入参数传递给`filter()`函数。
*   执行后，`filter()` 函数将返回一个 iterable 对象，其中包含没有从列表中删除的字符。
*   为了将 iterable 对象转换成列表，我们将把 iterable 对象传递给`list()`构造函数。这样，我们将在删除所有出现的指定字符后得到列表。

您可以在下面的示例中观察整个过程。

```py
def myFun(character):
    charToDelete = 'c'
    return charToDelete != character

myList = ['p', 'y', 'c', 't', 'c', 'h', 'o', 'n', 'f', 'c', 'o', 'r', 'b', 'e', 'g', 'c', 'i', 'n', 'n', 'c', 'e', 'r',
          's']
print("The original list is:", myList)
charToDelete = 'c'
print("The character to be removed is:", charToDelete)
outputList=list(filter(myFun,myList))
print("The modified list is:", outputList)
```

输出:

```py
The original list is: ['p', 'y', 'c', 't', 'c', 'h', 'o', 'n', 'f', 'c', 'o', 'r', 'b', 'e', 'g', 'c', 'i', 'n', 'n', 'c', 'e', 'r', 's']
The character to be removed is: c
The modified list is: ['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

不用定义函数`myFun`，我们可以创建一个 [lambda 函数](https://www.pythonforbeginners.com/basics/lambda-function-in-python)，并将其传递给过滤函数，以从列表中删除所有的字符实例。你可以这样做。

```py
myList = ['p', 'y', 'c', 't', 'c', 'h', 'o', 'n', 'f', 'c', 'o', 'r', 'b', 'e', 'g', 'c', 'i', 'n', 'n', 'c', 'e', 'r',
          's']
print("The original list is:", myList)
charToDelete = 'c'
print("The character to be removed is:", charToDelete)
outputList = list(filter(lambda character: character != charToDelete, myList))
print("The modified list is:", outputList)
```

输出:

```py
The original list is: ['p', 'y', 'c', 't', 'c', 'h', 'o', 'n', 'f', 'c', 'o', 'r', 'b', 'e', 'g', 'c', 'i', 'n', 'n', 'c', 'e', 'r', 's']
The character to be removed is: c
The modified list is: ['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

到目前为止，我们已经讨论了不同的方法来删除列表中出现的所有字符。现在我们将讨论如何从 python 字符串中删除特定字符的出现。

## 在 Python 中移除字符串中出现的所有字符

给定一个输入字符串，我们可以使用不同的字符串方法和正则表达式方法从字符串中删除一个或所有出现的字符。让我们逐一讨论它们。

### 在 Python 中使用 for 循环删除字符串中出现的所有字符

要使用 for 循环从字符串中删除所有出现的特定字符，我们将遵循以下步骤。

*   首先，我们将创建一个名为`outputString`的空字符串来存储输出字符串。
*   之后，我们将遍历原始字符串的字符。
*   在迭代字符串的字符时，如果我们找到了需要从字符串中删除的字符，我们将使用 continue 语句移动到下一个字符。
*   否则，我们将把当前字符连接到`outputString`。

在使用上述步骤的 for 循环迭代到字符串的最后一个字符之后，我们将在名为`outputString`的新字符串中获得输出字符串。您可以在下面的代码示例中观察到这一点。

```py
myStr = "pyctchonfcorbegcinncers"
print("The original string is:", myStr)
charToDelete = 'c'
print("The character to delete:", charToDelete)
outputString = ""
for character in myStr:
    if character == charToDelete:
        continue
    outputString += character

print("The modified string is:", outputString)
```

输出:

```py
The original string is: pyctchonfcorbegcinncers
The character to delete: c
The modified string is: pythonforbeginners
```

### 使用列表理解从 Python 中的字符串中移除所有出现的字符

不使用 for 循环，我们可以使用 list comprehension 和 `join()`方法从给定的字符串中删除特定值的出现。

*   首先，我们将使用 list comprehension 为字符串创建一个不需要删除的字符列表，如下所示。

```py
myStr = "pyctchonfcorbegcinncers"
print("The original string is:", myStr)
charToDelete = 'c'
print("The character to delete:", charToDelete)
myList = [character for character in myStr if character != charToDelete]
print("The list of characters is:")
print(myList)
```

输出:

```py
The original string is: pyctchonfcorbegcinncers
The character to delete: c
The list of characters is:
['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

*   获得列表后，我们将使用`join()`方法创建输出列表。当对特殊字符调用`join()`方法时，该方法将包含字符或字符串的 iterable 对象作为其输入参数。执行后，它返回一个字符串。输出字符串包含输入 iterable 对象的字符，由调用 join 方法的特殊字符分隔。
*   我们将使用空字符`“”`作为特殊字符。我们将调用空字符上的`join()`方法，将上一步获得的列表作为它的输入参数。在执行了`join()`方法之后，我们将得到想要的输出字符串。您可以在下面的示例中观察到这一点。

```py
myStr = "pyctchonfcorbegcinncers"
print("The original string is:", myStr)
charToDelete = 'c'
print("The character to delete:", charToDelete)
myList = [character for character in myStr if character != charToDelete]
print("The list of characters is:")
print(myList)
outputString = "".join(myList)
print("The modified string is:", outputString)
```

输出:

```py
The original string is: pyctchonfcorbegcinncers
The character to delete: c
The list of characters is:
['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
The modified string is: pythonforbeginners
```

### 在 Python 中使用 split()方法删除字符串中出现的所有字符

我们还可以使用`split()`方法从给定的字符串中删除一个字符的所有出现。在字符串上调用`split()`方法时，它将分隔符作为输入参数。执行后，它返回由分隔符分隔的子字符串列表。

要从给定的字符串中删除给定字符的所有出现，我们将使用以下步骤。

*   首先，我们将在原始字符串上调用`split()` 方法。w 将把需要删除的字符作为输入参数传递给`split()`方法。我们将把`split()`方法的输出存储在`myList`中。
*   获得`myList`后，我们将调用空字符串上的`join()`方法，并将`myList`作为 `join()` 方法的输入参数。
*   在执行了`join()` 方法之后，我们将得到想要的输出。所以，我们将它存储在一个变量`outputString`中。

您可以在下面的示例中观察整个过程。

```py
myStr = "pyctchonfcorbegcinncers"
print("The original string is:", myStr)
charToDelete = 'c'
print("The character to delete:", charToDelete)
myList = myStr.split(charToDelete)
print("The list is:")
print(myList)
outputString = "".join(myList)
print("The modified string is:", outputString)
```

输出:

```py
The character to delete: c
The list is:
['py', 't', 'honf', 'orbeg', 'inn', 'ers']
The modified string is: pythonforbeginners
```

### 在 Python 中，使用 filter()函数移除字符串中出现的所有字符

在 Python 中，我们还可以将 `filter()` 函数与`join()`方法和 lambda 函数结合使用，来删除字符串中出现的字符。

`filter()`函数将另一个函数比如说`myFun`作为它的第一个输入参数，将一个类似 string 的可迭代对象作为它的第二个输入参数。这里，`myFun`应该把字符串对象的字符作为它的输入参数。执行后，它应该返回`True`或`False`。

如果对于输入中给定的字符串对象的任何字符，`myFun`的输出为`True`，则该字符包含在输出中。否则，字符不会包含在输出中。

要删除字符串中给定字符的所有出现，我们将遵循以下步骤。

*   首先，我们将定义一个函数`myFun`，它将一个字符作为输入参数。如果输入字符等于必须删除的字符，它返回`False`。否则，它应该返回`True`。
*   定义了`myFun`之后，我们将把`myFun`作为第一个参数，把字符串作为第二个输入参数传递给`filter()`函数。
*   执行后，`filter()`函数将返回一个 iterable 对象，其中包含没有从字符串中删除的字符。
*   我们将通过将 iterable 对象传递给`list()`构造函数来创建一个字符列表。
*   一旦我们得到了字符列表，我们将创建输出字符串。为此，我们将调用空字符串上的`join()`方法，将字符列表作为其输入参数。
*   在执行了`join()`方法之后，我们将得到想要的字符串。

您可以在下面的代码中观察整个过程。

```py
def myFun(character):
    charToDelete = 'c'
    return charToDelete != character

myStr = "pyctchonfcorbegcinncers"
print("The original string is:", myStr)
charToDelete = 'c'
print("The character to delete:", charToDelete)
myList = list(filter(myFun, myStr))
print("The list is:")
print(myList)
outputString = "".join(myList)
print("The modified string is:", outputString) 
```

输出:

```py
The original string is: pyctchonfcorbegcinncers
The character to delete: c
The list is:
['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
The modified string is: pythonforbeginners
```

不用定义函数`myFun`，我们可以创建一个等价的 lambda 函数，并将它传递给 filter 函数，从字符串中删除该字符的所有实例。你可以这样做。

```py
myStr = "pyctchonfcorbegcinncers"
print("The original string is:", myStr)
charToDelete = 'c'
print("The character to delete:", charToDelete)
myList = list(filter(lambda character: character != charToDelete, myStr))
print("The list is:")
print(myList)
outputString = "".join(myList)
print("The modified string is:", outputString)
```

输出:

```py
The original string is: pyctchonfcorbegcinncers
The character to delete: c
The list is:
['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
The modified string is: pythonforbeginners
```

### 在 Python 中使用 replace()方法删除字符串中出现的所有字符

在字符串上调用`replace()`方法时，将需要替换的字符作为第一个参数。在第二个参数中，它用将要替换的字符代替第一个参数中给出的原始字符。

执行后，`replace()`方法返回作为输入给出的字符串的副本。在输出字符串中，所有字符都被替换为新字符。

为了从字符串中删除一个给定字符的所有出现，我们将对字符串调用`replace()`方法。我们将传递需要删除的字符作为第一个输入参数。在第二个输入参数中，我们将传递一个空字符串。

执行后，所有出现的字符将被替换为空字符串。因此，我们可以说这个字符已经从字符串中删除了。

您可以在下面的示例中观察整个过程。

```py
myStr = "pyctchonfcorbegcinncers"
print("The original string is:", myStr)
charToDelete = 'c'
print("The character to delete:", charToDelete)
outputString = myStr.replace(charToDelete, "")
print("The modified string is:", outputString)
```

输出:

```py
The original string is: pyctchonfcorbegcinncers
The character to delete: c
The modified string is: pythonforbeginners
```

### 在 Python 中，使用 translate()方法移除字符串中出现的所有字符

我们也可以使用 `translate()`方法从字符串中删除字符。当在字符串上调用`translate()`方法时，它将一个翻译表作为输入参数。执行后，它根据转换表返回修改后的字符串。

可以使用`maketrans()`方法创建转换表。在字符串上调用`maketrans()`方法时，将需要替换的字符作为第一个参数，新字符作为第二个参数。执行后，它返回一个转换表。

我们将使用以下步骤从字符串中删除给定的字符。

*   首先，我们将对输入字符串调用`maketrans()`方法。我们将需要删除的字符作为第一个输入参数，一个空格字符作为第二个输入参数传递给`maketrans()`方法。这里，我们不能将空字符传递给`maketrans()`方法，这样它就可以将字符映射到空字符串。这是因为两个字符串参数的长度应该相同。否则，`maketrans()`方法会出错。
*   执行后，`maketrans()` 方法将返回一个转换表，将我们需要删除的字符映射到一个空格字符。
*   一旦我们得到了翻译表，我们将调用输入字符串上的`translate()`方法，将翻译表作为它的输入参数。
*   执行后， `translate()`方法将返回一个字符串，其中我们需要删除的字符被替换为空格字符。
*   为了从字符串中删除空格字符，我们将首先在`translate()`方法的输出上调用`split()`方法。在这一步之后，我们将得到一个子字符串列表。
*   现在，我们将在一个空字符串上调用`join()`方法。这里，我们将把子字符串列表作为输入传递给`join()`方法。
*   在执行了`join()`方法之后，我们将得到想要的字符串。

您可以在下面的代码中观察整个过程。

```py
myStr = "pyctchonfcorbegcinncers"
print("The original string is:", myStr)
charToDelete = 'c'
print("The character to delete:", charToDelete)
translationTable = myStr.maketrans(charToDelete, " ")
outputString = "".join(myStr.translate(translationTable).split())
print("The modified string is:", outputString)
```

输出:

```py
The original string is: pyctchonfcorbegcinncers
The character to delete: c
The modified string is: pythonforbeginners
```

### 在 Python 中使用正则表达式删除字符串中出现的所有字符

[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)提供了一种最有效的操作字符串或文本数据的方法。

要从字符串中删除一个字符，我们可以使用在`re`模块中定义的`sub()`方法。 `sub()`方法将需要替换的字符比如说`old_char`作为它的第一个输入参数。它将新字符`new_char`作为第二个输入参数，输入字符串作为第三个参数。执行后，它将输入字符串中的`old_char`替换为`new_char`，并返回一个新的字符串。

要从给定的字符串中删除一个字符的所有出现，我们将使用以下步骤。

*   我们将需要删除的字符作为第一个输入参数`old_char`传递给`sub()`方法。
*   作为第二个参数`new_char`，我们将传递一个空字符串。
*   我们将输入字符串作为第三个参数传递给`sub()`方法。

执行后，`sub()` 方法将返回一个新的字符串。在新字符串中，需要删除的字符将被空字符串字符替换。因此，我们将得到期望的输出字符串。

您可以在下面的代码中观察到这一点。

```py
import re

myStr = "pyctchonfcorbegcinncers"
print("The original string is:", myStr)
charToDelete = 'c'
print("The character to delete:", charToDelete)
outputString = re.sub(charToDelete, "", myStr)
print("The modified string is:", outputString)
```

输出:

```py
The original string is: pyctchonfcorbegcinncers
The character to delete: c
The modified string is: pythonforbeginners
```

## 结论

在本文中，我们讨论了从列表中删除一个字符的所有匹配项的不同方法。同样，我们已经讨论了不同的方法来删除字符串中出现的所有字符。对于列表，我建议您使用带有`remove()`方法的方法。对于字符串，您可以使用`replace()`方法或`re.sub()`方法，因为这是删除列表中的所有字符或 python 中的字符串的最有效方法

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！