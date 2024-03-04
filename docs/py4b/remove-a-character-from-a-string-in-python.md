# 在 Python 中从字符串中移除字符

> 原文：<https://www.pythonforbeginners.com/basics/remove-a-character-from-a-string-in-python>

我们在 Python 中使用字符串来操作文本数据。在分析文本数据时，我们可能需要从数据中删除一些字符。在本文中，我们将讨论在 Python 中从字符串中删除字符的不同方法。

## 在 Python 中使用 For 循环从字符串中删除字符

我们使用 for 循环来遍历 iterable 对象的元素。我们将使用以下方法，通过 Python 中的 for 循环从字符串中删除一个字符。

*   首先，我们将定义一个名为`newStr`的空字符串来存储输出字符串。
*   现在，我们将使用 for 循环遍历输入字符串的字符。
*   在迭代过程中，如果我们发现一个字符不等于我们想要删除的字符，我们将把这个字符附加到`newStr`。
*   如果我们找到需要删除的字符，我们跳过它。

在执行 for 循环之后，我们将在变量`newStr`中获得输出字符串。您可以在下面的示例中观察到这一点。

```py
input_string = "Adcictcya"
char_to_remove = "c"
newStr = ""
for character in input_string:
    if character != char_to_remove:
        newStr += character

print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

在这里，您可以看到我们已经从输入字符串'`Adcictcya`'中删除了字符'`c`'，以生成输出字符串'`Aditya`'。

## 在 Python 中使用列表理解从字符串中移除字符

在 Python 中，我们可以使用 [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 从输入字符串中删除一个字符，而不是使用 for 循环。为此，我们将使用以下方法。

*   首先，我们将使用 list comprehension 来创建一个输入字符串的字符列表，在此之前，我们将删除想要删除的字符。我们将把列表存储在变量`myList`中。
*   在创建了`myList`之后，我们将定义一个名为`newStr`的空字符串来存储输出字符串。
*   现在，我们将使用 for 循环遍历列表中的元素。
*   在迭代过程中，我们将使用[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)来连接`myList`到`newStr`的每个元素。
*   在执行 for 循环后，我们将在变量`newStr`中获得所需的输出字符串。您可以在下面的示例中观察到这一点。

```py
input_string = "Adcictcya"
char_to_remove = "c"
myList = [character for character in input_string if character != char_to_remove]
newStr = ""
for character in myList:
    newStr += character

print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

创建`myList`后，我们可以使用`join()`方法代替 for 循环来创建输出字符串。

当在分隔符字符串上调用时，`join()`方法将 iterable 对象作为其输入参数。执行后，它返回一个字符串，其中 iterable 对象的所有元素都由分隔符分隔。

我们将使用下面的方法，使用 list comprehension 和 `join()`方法从 python 中的字符串中删除一个字符。

*   首先，我们将使用 list comprehension 来创建一个输入字符串的字符列表，在此之前，我们将删除想要删除的字符。我们将把列表存储在变量`myList`中。
*   现在，我们将定义一个空字符串作为分隔符。
*   定义分隔符后，我们将调用分隔符上的 `join()`方法。这里，我们将把`myList`作为输入参数传递给 `join()` 方法。
*   执行后，`join()` 方法将返回所需的字符串。

您可以在下面的示例中观察到这一点。

```py
input_string = "Adcictcya"
char_to_remove = "c"
myList = [character for character in input_string if character != char_to_remove]
separator = ""
newStr = separator.join(myList)

print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

## 在 Python 中使用 split()方法从字符串中移除字符

方法`split()`用于[将一个字符串](https://www.pythonforbeginners.com/dictionary/python-split)分割成子字符串。当在字符串上调用时，它将一个字符作为其输入参数。执行后，它返回在给定字符处拆分的原始字符串的子字符串列表。

要使用`split()`方法从 Python 中的字符串中删除一个字符，我们将使用以下步骤。

*   首先，我们将对输入字符串调用`split()`方法。这里，我们将需要从字符串中删除的字符作为输入参数进行传递。
*   执行后，`split()`方法将返回一个列表。我们将列表存储在`myList`中。
*   现在，我们将定义一个名为`newStr`的空字符串来存储输出字符串。
*   之后，我们将使用 for 循环遍历列表中的元素。
*   在迭代过程中，我们将使用字符串连接来连接`myList`到`newStr`的每个元素。

在执行 for 循环后，我们将在变量`myStr`中获得所需的输出字符串。您可以在下面的示例中观察到这一点。

```py
input_string = "Adcictcya"
char_to_remove = "c"
myList = input_string.split(char_to_remove)
newStr = ""
for element in myList:
    newStr += element

print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

不使用 for 循环和字符串连接来创建来自`myList`的输出字符串，我们可以使用如下的`join()`方法。

*   首先，我们将对输入字符串调用`split()`方法。这里，我们将需要从字符串中删除的字符作为输入参数进行传递。
*   执行后，`split()`方法将返回一个列表。我们将列表存储在`myList`中。
*   现在，我们将定义一个空字符串作为分隔符。
*   定义分隔符后，我们将调用分隔符上的`join()` 方法。这里，我们将把`myList`作为输入参数传递给`join()` 方法。
*   执行后， `join()`方法将返回输出字符串。

您可以在下面的示例中观察整个过程。

```py
input_string = "Adcictcya"
char_to_remove = "c"
myList = input_string.split(char_to_remove)
separator = ""
newStr = separator.join(myList)

print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

## 在 Python 中使用 filter()函数从字符串中移除字符

`filter()`函数用于根据条件过滤可迭代对象的元素。`filter()` 函数将另一个函数作为它的第一个输入参数，将一个 iterable 对象作为它的第二个输入参数。作为第一个输入参数给出的函数必须将 iterable 对象的元素作为其输入，并为每个元素返回 True 或 False。

执行后，`filter()` 函数返回一个迭代器，其中包含输入参数中给定的 iterable 对象的所有元素，对于这些元素，第一个输入参数中给定的函数返回 True。

要使用 filter 函数从给定的字符串中删除一个字符，我们将使用以下步骤。

*   首先，我们将定义一个函数`myFun`，它把给定字符串的字符作为它的输入参数。如果输入参数中给出的字符是需要删除的字符，则返回 False。否则，它返回 True。
*   在定义了`myFun`之后，我们将把`myFun`和输入字符串分别作为第一和第二输入参数传递给`filter()`函数。
*   执行后，filter 函数返回一个包含字符串字符的迭代器。我们将使用`list()`函数将迭代器转换成一个列表。我们将列表存储在`myList`中。
*   现在，我们将定义一个名为`newStr`的空字符串来存储输出字符串。
*   之后，我们将使用 for 循环遍历列表中的元素。
*   在迭代过程中，我们将使用字符串连接来连接`myList`到`newStr`的每个元素。

在执行 for 循环后，我们将在变量`newStr`中获得所需的输出字符串。您可以在下面的示例中观察到这一点。

```py
def myFun(character):
    char_to_remove = "c"
    return character != char_to_remove

input_string = "Adcictcya"
char_to_remove = "c"
filter_object = filter(myFun, input_string)
myList = list(filter_object)
newStr = ""
for element in myList:
    newStr += element

print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

您还可以使用 join()方法通过以下步骤创建输出字符串。

*   首先，我们将定义一个函数 myFun，它将给定字符串的字符作为输入参数。如果输入参数中给出的字符是需要删除的字符，则返回 False。否则，它返回 True。
*   在定义了`myFun`之后，我们将把`myFun`和输入字符串分别作为第一和第二输入参数传递给`filter()`函数。
*   执行后，filter 函数返回一个包含字符串字符的迭代器。我们将使用`list()`函数将迭代器转换成一个列表。我们将列表存储在`myList`中。
*   现在，我们将定义一个空字符串作为分隔符。
*   定义分隔符后，我们将调用分隔符上的 `join()`方法。这里，我们将把`myList`作为输入参数传递给 `join()`方法。
*   执行后，`join()` 方法将返回输出字符串。

您可以在下面的示例中观察整个过程。

```py
def myFun(character):
    char_to_remove = "c"
    return character != char_to_remove

input_string = "Adcictcya"
char_to_remove = "c"
filter_object = filter(myFun, input_string)
myList = list(filter_object)
separator = ""
newStr = separator.join(myList)

print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

不使用`myFun`，你可以使用一个[λ表达式](https://www.pythonforbeginners.com/basics/lambda-function-in-python)和`filter()`函数，如下所示。

```py
input_string = "Adcictcya"
char_to_remove = "c"
filter_object = filter(lambda character:character!=char_to_remove, input_string)
myList = list(filter_object)
separator = ""
newStr = separator.join(myList)

print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

这里，lambda 表达式执行与前面代码中的`myFun`相同的任务。

## 在 Python 中使用 replace()方法从字符串中移除字符

`replace()`方法用于将字符串中的一个字符替换为另一个字符。您也可以使用 `replace()`方法从字符串中删除一个字符。在字符串上调用时，`replace()`方法将需要替换的字符作为第一个参数，新字符作为第二个输入参数。执行后，它返回修改后的字符串。

为了使用`replace()` 方法从字符串中删除一个字符，我们将对原始字符串调用`replace()` 方法。这里，将需要删除的字符作为第一个输入参数，一个空字符串作为第二个输入参数传递给`replace()`方法。

执行后， `replace()` 方法将返回输出字符串。您可以在下面的示例中观察到这一点。

```py
input_string = "Adcictcya"
char_to_remove = "c"
newStr = input_string.replace(char_to_remove, "")

print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

## 在 Python 中使用 translate()方法从字符串中移除字符

`translate()`方法也用于替换字符串中的字符。当在字符串上调用时，`translate()`方法将一个翻译表作为它的输入参数。执行后，它返回修改后的字符串。

要从给定的字符串中删除一个字符，我们首先要做一个翻译表。为此，我们可以使用`maketrans()`方法。

在字符串上调用`maketrans()`方法时，将一个字符串作为第一个参数，另一个包含相同数量字符的字符串作为第二个输入参数中的第一个参数。执行后，它返回一个转换表。

*   为了制作从给定字符串中删除字符的转换表，我们将需要删除的字符作为第一个输入参数，一个空格字符作为第二个输入参数传递给`maketrans()`方法。执行后，`maketrans()`方法将返回一个转换表，将需要删除的字符映射到一个空白字符。
*   现在，我们将调用输入字符串上的 `translate()` 方法，将翻译表作为其输入参数。在执行了`translate()`方法之后，我们将得到一个字符串，其中需要删除的字符被一个空白字符替换。
*   接下来，我们将使用`split()` 方法将`translate()`方法返回的字符串拆分成一个列表。我们将把列表存储在一个名为`myList`的变量中。
*   接下来，我们将定义一个名为`newStr`的空字符串来存储输出字符串。
*   之后，我们将使用 for 循环遍历列表中的元素。
*   在迭代过程中，我们将使用字符串连接来连接`myList`到`newStr`的每个元素。

在执行 for 循环后，我们将在变量`newStr`中获得所需的输出字符串。您可以在下面的示例中观察到这一点。

```py
input_string = "Adcictcya"
char_to_remove = "c"
translation_table = input_string.maketrans(char_to_remove, " ")
tempStr = input_string.translate(translation_table)
myList = tempStr.split()
newStr = ""
for element in myList:
    newStr += element
print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr) 
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

您可以使用 join()方法从`myList`创建输出字符串，而不是使用 for 循环和字符串串联。为此，您可以使用以下步骤。

*   首先，将定义一个空字符串作为分隔符。
*   定义分隔符后，我们将调用分隔符上的`join()`方法。这里，我们将把`myList`作为输入参数传递给`join()`方法。
*   执行后，`join()`方法将返回输出字符串。

您可以在下面的示例中观察整个过程。

```py
input_string = "Adcictcya"
char_to_remove = "c"
translation_table = input_string.maketrans(char_to_remove, " ")
tempStr = input_string.translate(translation_table)
myList = tempStr.split()
separator=""
newStr = separator.join(myList)
print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

使用`translate()`方法的方法只适用于不包含空白字符的字符串。因此，如果输入字符串中有空白字符，就不能使用这种方法。

## 在 Python 中使用正则表达式从字符串中移除字符

[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)为我们提供了 Python 中[字符串操作的各种函数。在 Python 中，还可以使用正则表达式从字符串中移除字符。为此，我们将使用`sub()` 函数。](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)

`sub()`函数用于替换字符串中的字符。它需要三个输入参数。第一个输入参数是需要替换的字符。第二个参数是替换字符。最后，第三个输入参数是原始字符串。执行后，`sub()`函数返回新的字符串。

要使用`sub()`函数从字符串中删除一个字符，我们将使用以下步骤。

*   在`sub()`函数的第一个参数中，我们将传递需要删除的字符。
*   我们将把一个空字符串作为第二个输入参数，把原始字符串作为第三个输入参数传递给`sub()`函数。
*   在执行了`sub()`函数之后，我们将得到想要的输出字符串。

您可以在下面的示例中观察到这一点。

```py
import re

input_string = "Adcictcya"
char_to_remove = "c"
newStr = re.sub(char_to_remove, "", input_string)
print("The input string is:", input_string)
print("The character to delete is:", char_to_remove)
print("The output string is:", newStr)
```

输出:

```py
The input string is: Adcictcya
The character to delete is: c
The output string is: Aditya
```

## 结论

在本文中，我们讨论了在 Python 中从字符串中删除字符的不同方法。在所有这些方法中，使用`re.sub()`函数和 `replace()`方法的方法是最有效的。因此，在 Python 中，应该使用这两种方法从字符串中删除字符。

要了解更多关于 python 编程的知识，您可以阅读这篇关于如何在 Python 中从列表或字符串中删除所有出现的字符的文章。你可能也会喜欢这篇关于机器学习中[回归的文章](https://codinginfinite.com/regression-in-machine-learning-with-examples/)。你也可以看看这篇关于[数据分析师 vs 数据科学家](https://www.codeconquest.com/blog/data-analyst-vs-data-scientist-skills-education-responsibilities-and-salaries/)的文章。

请继续关注更多内容丰富的文章。快乐学习！