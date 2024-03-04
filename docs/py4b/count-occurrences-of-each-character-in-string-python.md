# 在 Python 中计算字符串中每个字符的出现次数

> 原文：<https://www.pythonforbeginners.com/basics/count-occurrences-of-each-character-in-string-python>

[字符串操作](https://www.pythonforbeginners.com/basics/python-string-methods-for-string-manipulation)是文本数据分析的关键组成部分之一。在分析文本数据时，我们可能需要计算文本中字符的出现频率。在本文中，我们将讨论 Python 中计算字符串中每个字符出现次数的不同方法。

## 使用 For 循环和 set()函数计算字符串中每个字符的出现次数

在 Python 中，我们可以使用 for 循环和 `set()` 函数来计算字符串中每个字符的出现次数。为此，我们将使用以下步骤。

*   首先，我们将创建一个包含输入字符串中所有字符的集合。为此，我们将使用`set()` 函数。`set()`函数将 iterable 对象作为其输入参数，并返回 iterable 对象的所有元素的集合。
*   创建字符集后，我们将使用嵌套的 for 循环来计算字符串中每个字符的出现次数。
*   在外部 for 循环中，我们将遍历集合中的元素。在 for 循环中，我们将定义一个变量`countOfChar`，并将其初始化为 0。
*   然后，我们将使用另一个 for 循环迭代输入字符串。
*   在内部 for 循环中，如果我们在字符串中找到集合的当前字符，我们将把`countOfChar`加 1。否则，我们将移动到字符串中的下一个字符。
*   在执行内部 for 循环后，我们将获得单个字符的计数。我们将使用 print 语句打印它。然后，我们将使用外部 for 循环移动到集合中的下一个字符。

执行 for 循环后，将打印字符串中每个字符出现的次数。您可以在下面的示例中观察到这一点。

```py
input_string = "Pythonforbeginners is a great source to get started with Python."
print("The input string is:", input_string)
mySet = set(input_string)
for element in mySet:
    countOfChar = 0
    for character in input_string:
        if character == element:
            countOfChar += 1
    print("Count of character '{}' is {}".format(element, countOfChar))
```

输出:

```py
The input string is: Pythonforbeginners is a great source to get started with Python.
Count of character 'o' is 5
Count of character 'a' is 3
Count of character 'c' is 1
Count of character 'e' is 6
Count of character 'd' is 1
Count of character 't' is 8
Count of character 'r' is 5
Count of character 'y' is 2
Count of character 'n' is 4
Count of character 'u' is 1
Count of character 's' is 4
Count of character 'g' is 3
Count of character 'w' is 1
Count of character '.' is 1
Count of character 'h' is 3
Count of character ' ' is 9
Count of character 'P' is 2
Count of character 'b' is 1
Count of character 'i' is 3
Count of character 'f' is 1
```

如果你想存储字符的频率，你可以使用一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)。为了存储频率，我们将首先创建一个名为`countOfChars`的空字典。

计算完一个字符的计数后，我们将把这个字符作为键，把计数作为值添加到字典中。您可以在下面的代码中观察到这一点。

```py
input_string = "Pythonforbeginners is a great source to get started with Python."
print("The input string is:", input_string)
mySet = set(input_string)
countOfChars = dict()
for element in mySet:
    countOfChar = 0
    for character in input_string:
        if character == element:
            countOfChar += 1
    countOfChars[element] = countOfChar
print("Count of characters is:")
print(countOfChars)
```

输出:

```py
The input string is: Pythonforbeginners is a great source to get started with Python.
Count of characters is:
{'s': 4, 'P': 2, 'b': 1, '.': 1, 'd': 1, 'c': 1, 'g': 3, 'r': 5, 'i': 3, 'o': 5, 'u': 1, 'a': 3, 'f': 1, 'e': 6, 'n': 4, 'y': 2, ' ': 9, 'w': 1, 't': 8, 'h': 3}
```

## 使用 Python 中的 Count()方法计算字符串中每个字符的出现次数

字符串中的`count()`方法用于统计字符串中某个字符出现的频率。当在字符串上调用时，`count()`方法将一个字符作为它的输入参数。执行后，它返回作为输入参数给出的字符的频率。

为了使用`count()`方法计算字符串中每个字符的出现次数，我们将使用以下步骤。

*   首先，我们将使用`set()`函数在输入字符串中创建一组字符。
*   之后，我们将使用 for 循环遍历集合中的元素。
*   在 for 循环中，我们将调用输入字符串上的`count()`方法，将集合中的当前元素作为其输入参数。执行后，`count()`方法将返回集合中当前元素的出现次数。我们将使用 print 语句打印该值。

在执行 for 循环后，将打印所有字符的频率。您可以在下面的示例中观察到这一点。

```py
input_string = "Pythonforbeginners is a great source to get started with Python."
print("The input string is:", input_string)
mySet = set(input_string)
countOfChars = dict()
for element in mySet:
    countOfChar = input_string.count(element)
    countOfChars[element] = countOfChar
    print("Count of character '{}' is {}".format(element, countOfChar))
```

输出:

```py
The input string is: Pythonforbeginners is a great source to get started with Python.
Count of character 'e' is 6
Count of character 'n' is 4
Count of character 'a' is 3
Count of character '.' is 1
Count of character 'h' is 3
Count of character 'r' is 5
Count of character 'f' is 1
Count of character 'y' is 2
Count of character 's' is 4
Count of character 't' is 8
Count of character 'w' is 1
Count of character 'i' is 3
Count of character 'd' is 1
Count of character 'g' is 3
Count of character 'u' is 1
Count of character 'c' is 1
Count of character 'o' is 5
Count of character 'P' is 2
Count of character 'b' is 1
Count of character ' ' is 9
```

您还可以将字符的频率存储在字典中，如下所示。

```py
input_string = "Pythonforbeginners is a great source to get started with Python."
print("The input string is:", input_string)
mySet = set(input_string)
countOfChars = dict()
for element in mySet:
    countOfChar = input_string.count(element)
    countOfChars[element] = countOfChar
print("Count of characters is:")
print(countOfChars)
```

输出:

```py
The input string is: Pythonforbeginners is a great source to get started with Python.
Count of characters is:
{'t': 8, 'o': 5, 'P': 2, 'n': 4, 'f': 1, 'e': 6, 'g': 3, 'c': 1, '.': 1, 's': 4, 'w': 1, 'y': 2, ' ': 9, 'u': 1, 'i': 3, 'd': 1, 'a': 3, 'h': 3, 'r': 5, 'b': 1}
```

上述方法具有很高的时间复杂度。如果字符串中有 N 个不同的字符，并且字符串长度为 M，则执行的时间复杂度将为 M*N。因此，如果必须分析数千个字符的字符串，则不建议使用这些方法。为此，我们可以使用下面几节中讨论的其他方法。

## 使用 [Python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)来统计字符串中每个字符的出现次数

python 中的字典存储键值对。为了使用字典计算 Python 中字符串中每个字符的出现次数，我们将使用以下方法。

*   首先，我们将创建一个名为`countOfChars`的空字典来存储字符及其频率。
*   现在，我们将使用 for 循环迭代输入字符串。
*   在迭代过程中，我们将使用成员操作符检查当前字符是否存在于字典中。
*   如果该字符出现在字典中，我们将把与该字符相关的值增加 1。否则，我们将把这个字符作为一个键添加到字典中，1 作为它的关联值。

在执行 for 循环之后，我们将获得`countOfChars`字典中每个字符的计数。您可以在下面的示例中观察到这一点。

```py
input_string = "Pythonforbeginners is a great source to get started with Python."
print("The input string is:", input_string)
countOfChars = dict()
for character in input_string:
    if character in countOfChars:
        countOfChars[character] += 1
    else:
        countOfChars[character] = 1
print("The count of characters in the string is:")
print(countOfChars)
```

输出:

```py
The input string is: Pythonforbeginners is a great source to get started with Python.
The count of characters in the string is:
{'P': 2, 'y': 2, 't': 8, 'h': 3, 'o': 5, 'n': 4, 'f': 1, 'r': 5, 'b': 1, 'e': 6, 'g': 3, 'i': 3, 's': 4, ' ': 9, 'a': 3, 'u': 1, 'c': 1, 'd': 1, 'w': 1, '.': 1}
```

不使用 if else 语句，可以使用 [python try-except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块来计算字符串中字符的出现次数。

*   在 for 循环中，我们将在 try 块中将字典中与当前字符相关联的值增加 1。如果该字符在字典中不存在，程序将引发一个 KeyError 异常。
*   在 except 块中，我们将捕获 KeyError 异常。这里，我们将把字符作为一个键分配给字典，1 作为它的关联值。

在执行 for 循环之后，我们将获得`countOfChars`字典中每个字符的计数。您可以在下面的示例中观察到这一点。

```py
input_string = "Pythonforbeginners is a great source to get started with Python."
print("The input string is:", input_string)
countOfChars = dict()
for character in input_string:
    try:
        countOfChars[character] += 1
    except KeyError:
        countOfChars[character] = 1
print("The count of characters in the string is:")
print(countOfChars)
```

输出:

```py
The input string is: Pythonforbeginners is a great source to get started with Python.
The count of characters in the string is:
{'P': 2, 'y': 2, 't': 8, 'h': 3, 'o': 5, 'n': 4, 'f': 1, 'r': 5, 'b': 1, 'e': 6, 'g': 3, 'i': 3, 's': 4, ' ': 9, 'a': 3, 'u': 1, 'c': 1, 'd': 1, 'w': 1, '.': 1}
```

如果我们有一个很大的输入字符串，与字符串的长度相比，只有很少的不同字符，那么使用 try-except 块的方法效果最好。如果输入字符串很小，并且输入字符串的长度不是非常大于不同字符的总数，那么这种方法会比较慢。这是因为处理异常是一项开销很大的操作。

如果程序非常频繁地引发 [KeyError 异常](https://www.pythonforbeginners.com/basics/python-keyerror)，将会降低程序的性能。因此，您应该根据输入字符串长度和字符串中不同字符的数量，选择使用 if-else 语句还是 try-except 块。

还可以避免同时使用 if else 语句和 try-except 块。为此，我们需要使用以下方法。

*   首先，我们将使用`set()`函数在原始字符串中创建一组字符。
*   然后，我们将使用集合的元素作为键，使用 0 作为关联值来初始化字典`countOfChars`。
*   现在，我们将使用 for 循环遍历输入字符串的字符。在迭代过程中，我们将把与当前字符相关的值加 1。

在执行 for 循环后，我们将获得字符串中每个字符的出现次数。您可以在下面的示例中观察到这一点。

```py
input_string = "Pythonforbeginners is a great source to get started with Python."
print("The input string is:", input_string)
mySet = set(input_string)
countOfChars = dict()
for element in mySet:
    countOfChars[element] = 0
for character in input_string:
    countOfChars[character] += 1
print("The count of characters in the string is:")
print(countOfChars)
```

输出:

```py
The input string is: Pythonforbeginners is a great source to get started with Python.
The count of characters in the string is:
{'d': 1, 'r': 5, 'y': 2, 'a': 3, 'P': 2, 'i': 3, 's': 4, ' ': 9, 'f': 1, '.': 1, 'h': 3, 't': 8, 'g': 3, 'c': 1, 'u': 1, 'e': 6, 'n': 4, 'w': 1, 'o': 5, 'b': 1} 
```

## 使用集合计算字符串中每个字符的出现次数。Counter()函数

集合模块为我们提供了各种函数来处理集合对象，如列表、字符串、集合等。`Counter()`函数就是其中之一。它用于计算集合对象中元素的出现频率。

`Counter()`函数接受一个集合对象作为它的输入参数。执行后，它返回一个[集合计数器](https://www.pythonforbeginners.com/collection/python-collections-counter)对象。计数器对象以字典的形式包含所有字符及其频率。

要计算 Python 中一个字符串中每个字符的出现次数，只需将它传递给`Counter()`函数并打印输出，如下例所示。

```py
from collections import Counter
input_string = "Pythonforbeginners is a great source to get started with Python."
print("The input string is:", input_string)
countOfChars = Counter(input_string)
print("The count of characters in the string is:")
print(countOfChars)
```

输出:

```py
The input string is: Pythonforbeginners is a great source to get started with Python.
The count of characters in the string is:
Counter({' ': 9, 't': 8, 'e': 6, 'o': 5, 'r': 5, 'n': 4, 's': 4, 'h': 3, 'g': 3, 'i': 3, 'a': 3, 'P': 2, 'y': 2, 'f': 1, 'b': 1, 'u': 1, 'c': 1, 'd': 1, 'w': 1, '.': 1})
```

## 使用 Collections.defaultdict()

集合模块为我们提供了一个具有增强特性的字典对象。这叫`defaultdict`。如果您试图修改与字典中不存在的键相关联的值，`defaultdict`对象不会引发 KeyError 异常。相反，它自己创建一个密钥，然后继续执行语句。

例如，如果在一个简单的字典中没有名为 `“Aditya”`的键，而我们执行了操作`myDict[“Aditya”]+=1`，程序将会遇到一个 KeyError 异常。另一方面，`defaultdict`对象将首先在字典中创建键`“Aditya”`，并将成功执行上述语句。但是，我们需要帮助 defaultdict 对象创建键的默认值。

`defaultdict()`函数接受另一个函数，比如说`fun1`作为它的输入参数。每当 defaultdict 对象需要创建一个带有默认值的新键时，它就会执行`fun1`，并使用`fun1`返回的值作为新键的关联值。在我们的例子中，我们需要一个字符计数的默认值 0，我们将把`int()`函数作为输入参数传递给`defaultdict()`函数。

为了使用 Python 中的 defaultdict 对象计算字符串中每个字符的出现次数，我们将使用以下步骤。

*   首先，我们将使用`collections.defaultdict()`函数创建一个 defaultdict 对象。这里，我们将把`int()`函数作为输入参数传递给`defaultdict()`函数。
*   然后，我们将使用 for 循环遍历输入字符串的字符。
*   在迭代过程中，我们将不断增加与 defaultdict 对象中每个字符相关的值。

在执行 for 循环后，我们将获得 defaultdict 对象中每个字符的计数。您可以在下面的示例中观察到这一点。

```py
from collections import defaultdict
input_string = "Pythonforbeginners is a great source to get started with Python."
print("The input string is:", input_string)
countOfChars = defaultdict(int)
for character in input_string:
    countOfChars[character] += 1
print("The count of characters in the string is:")
print(countOfChars)
```

输出:

```py
The input string is: Pythonforbeginners is a great source to get started with Python.
The count of characters in the string is:
defaultdict(<class 'int'>, {'P': 2, 'y': 2, 't': 8, 'h': 3, 'o': 5, 'n': 4, 'f': 1, 'r': 5, 'b': 1, 'e': 6, 'g': 3, 'i': 3, 's': 4, ' ': 9, 'a': 3, 'u': 1, 'c': 1, 'd': 1, 'w': 1, '.': 1})
```

## 结论

在本文中，我们讨论了 python 中计算字符串中每个字符出现次数的不同方法。在所有这些方法中，我建议您使用使用集合模块的方法，因为这些是最有效的方法。

我希望你喜欢阅读这篇文章。想了解更多关于 python 编程的知识，可以阅读这篇关于 Python 中[字典理解的文章。你可能也会喜欢这篇关于机器学习中](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)[回归的文章](https://codinginfinite.com/regression-in-machine-learning-with-examples/)。你也可以看看这篇关于[数据分析师 vs 数据科学家](https://www.codeconquest.com/blog/data-analyst-vs-data-scientist-skills-education-responsibilities-and-salaries/)的文章。

请继续关注更多内容丰富的文章。快乐学习！