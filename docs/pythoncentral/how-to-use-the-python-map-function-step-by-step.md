# 如何使用 Python 地图功能(逐步)

> 原文：<https://www.pythoncentral.io/how-to-use-the-python-map-function-step-by-step/>

映射是一种允许开发人员处理和转换 iterable 中的元素而无需显式声明 for 循环的方法。这就是 map()函数的用武之地。它允许开发人员将 iterable 的每一项转换成新的 iterable。

map()函数是支持函数式编程风格的两个内置 Python 方法之一。

在本文中，我们将讨论 map()函数的工作原理，如何使用该函数来转换各种可迭代对象，以及如何将该函数与其他 Python 工具相结合来执行更复杂的转换。

请记住，这篇文章假设读者理解了 for 循环、函数和可迭代的工作原理。

## **什么是 map()函数？**

Python 程序员有时会遇到这样的问题，他们需要对一个 iterable 的所有元素进行相同的操作，并从中创建一个新的 iterable。

通常，开发人员会为此使用 for 循环，但是使用 map()函数也有助于达到同样的效果。

与 for 循环非常相似，map()函数也遍历所提供的 iterable 的元素。然后它输出一个迭代器，其中的转换函数已经应用于所提供的 iterable 的每个元素。它具有以下签名:

```py
map(function, iterable[, iterable1, iterable2,..., iterableN])
```

分解一下:

*函数* 应用于*iterable*的每个元素，之后创建一个新的迭代器，按需产生转换后的项。 *函数* 可以是接受与传递给 map()函数的 iterables 数目相同的参数数目的任何方法。

需要注意的是，签名中的第一个参数是一个函数对象。换句话说，它要求你传递一个函数而不调用它。第一个参数是转换函数，负责将原始元素转换成转换后的元素。

参数可以是任何类型的 Python 函数，而不仅仅是内置函数，包括类、用户定义函数和 lambda 函数。

现在，我们用一个例子来理解 map()函数。假设您需要一个程序，它接受一个数值列表作为输入，并输出一个包含原始列表中每个数字的平方值的列表。

如果您决定对此使用 for 循环，解决方案将如下所示:

```py
numberList = [3, 6, 9]
squareList = []

for num in numberList:
squareList.append(num ** 2)

squareList

# Output: [9, 36, 81]
```

上面代码中的 for 循环遍历 numberList 中的所有数字，并对每个值应用指数运算。结果存储在 squareList 列表中。

然而，如果您使用 map()函数来开发一个解决方案，实现将如下所示:

```py
def square(number):
return number ** 2

numberList = [3, 6, 9]

squareList = map(square, numberList)

list(squareList)
# Output: [9, 36, 81]
```

在上面的代码中，square()是转换函数，它将列表中的每个数字“映射”到各自的平方值。调用 map()函数会将定义的 square()函数应用于 numberList 中的值。结果是一个保存平方值的迭代器。

然后，在 map()上调用 list()函数，创建一个保存平方值的列表对象。

map()函数使用 C 语言，比 Python 中的 for 循环更高效。但除了效率之外，使用该功能还有另一个优势——更低的内存消耗。

Python for 循环将整个列表存储在系统内存中，但 map()函数并非如此。map()函数按需为列表元素提供服务，一次只在系统内存中存储一个元素。

-

**注意:** 在 Python 2.x 中 map()函数返回一个列表，但在 Python 3.x 中，它返回一个 map 对象。map 对象充当迭代器，根据需要返回列表元素。因此，您需要使用 list()方法来创建 list 对象。

-

在 int()函数的帮助下，你也可以使用 map()函数将一个列表中的所有元素从字符串转换成整数。你可以这样做:

```py
stringNumbers = ["32", "58", "93”]
integers = map(int, stringNumbers)
list(integers)
# Output: [32, 58, 93]
```

在上面的例子中，map()函数将 int()方法应用于 stringNumbers 列表中的每一项。调用 list()函数背后的想法是，它将耗尽 map()函数返回的迭代器。

然后 list()函数将迭代器转换成一个列表对象。请记住，在此过程中不会修改原始序列。

## **map()如何与不同的功能配合使用**

如前所述，map()适用于所有类型的 Python 可调用程序，前提是可调用程序接受参数并能返回有用的值。例如，您可以使用实现 __call__()方法的类和实例，以及类、静态和实例方法。

这里有一个例子供你考虑:

```py
numberList= [-2, -1, 0, 1, 2]
#Making the negative values positive
absoluteValues = list(map(abs, numberList))
list(map(float, numberList))
# It prints [-2.0, -1.0, 0.0, 1.0, 2.0]
words = ["map", "with", "different", "functions"]
list(map(len, words))
# It prints [3,4,9, 9]
```

涉及 map()函数的一个更常见的趋势是使用 lambda 函数作为主要参数。当开发人员需要提供一个基于表达式的函数来映射()时，lambda 函数特别有用。

让我们后退一步，用 lambda 函数重新实现前面讨论的平方值示例。这是如何工作的:

```py
numberList = [5, 10]
squareList = map(lambda num: num ** 2, numberList)
list(squareList)
#Output: [25,100]
```

从上面的例子可以清楚地看出，lambda 函数是与 map()结合使用的最有用的函数，并且经常被用作 map()函数的第一个参数，以快速处理和转换 iterables。

## **使用 map()处理多个输入的可重复项**

如果您向 map()提供几个 iterables，转换函数将需要与 iterables 传递的一样多的参数。函数的每次迭代都会将每个 iterable 中的一个值作为参数传递给 *函数* 。当到达最短迭代的末尾时，迭代停止。

例如，考虑以下 map()与 pow()函数的使用:

```py
listOne = [2, 4, 6]
listTwo = [1, 2, 3, 7]

>>> list(map(pow, listOne, listTwo))
# Output: [2, 16, 216]
```

如果你熟悉 pow()函数，你会知道它接受两个参数——我们称之为 a 和 b——并返回 a 的 b 次方。

在上面的例子中，a 的值是 2，b 在第一次迭代中是 1。到了第二次迭代，a 值是 4，b 是 2，依此类推。观察输出，您会看到输出的 iterable 与最短的 iterable 一样长，在本例中是 listOne。

使用这种技术，你可以通过各种数学运算来合并包含数值的可重复项。

## **使用 map()转换字符串的可重复项**

如果你的一些解决方案涉及到处理包含字符串对象的 iterables，你可能需要使用某种转换函数来转换这些对象。在这些情况下，map()函数可以派上用场。

在本节中，我们将讨论两种不同的场景，在这两种场景中，map()对于转换 string 对象的 iterables 非常有用。

### **使用 str 类的方法**

当开发人员需要操作字符串时，他们通常使用内置 str 类下的方法。这些方法有助于将字符串转换成新字符串。如果您正在处理包含字符串的 iterables，并且希望以相同的方式转换所有的字符串，那么您可以使用 map()函数:

```py
stringList = ["this", "is", "a", "string"]
list(map(str.capitalize, stringList))

list(map(str.upper, stringList))

list(map(str.lower, stringList))

# Output: 
# [‘This', 'Is, A', 'String']
# ['THIS, 'IS', 'A', 'STRING']
# ['this', ‘is', a', 'string']
```

您可以使用 str 和 map()方法对 stringList 列表中的每个元素执行各种转换。在大多数情况下，开发人员会找到不接受任何附加参数的有用方法。这些方法包括上面代码中的方法和其他一些方法，如 str.swapcase()和 str.title()。

有时，开发人员也使用接受额外参数的方法，比如 str.strip()函数，它接受一个去掉空格的 char 参数。这里有一个如何使用这种方法的例子:

```py
spacedList = ["These ", "  strings", "have   ", " whitespaces   "]
list(map(str.strip, spacedList))

#Output: ['These', 'strings', 'have', 'whitespaces']
```

如上面的代码所示，str.strip()方法使用默认的 char 值，map()方法从 spacedList 的元素中删除空白。

然而，如果你需要给方法提供参数，并且不想依赖默认值，使用 lambda 函数是正确的方法。

让我们来看另一个例子，它使用 str.strip()方法从列表元素中删除省略号而不是空格:

```py
dotList = ["These..", "...strings", "have....", "..dots.."]
list(map(lambda s: s.strip("."), dotList))

# Output: ['These', 'strings', 'have', 'dots']
```

这里，lambda 函数调用。strip()(在“s”对象上),从字符串中移除点前面和后面的点。

当开发人员需要处理含有尾随空格或其他需要从文本中删除的不需要的字符的文本文件时，这种处理字符串的方法非常方便。

在这种情况下，开发人员也可以使用不带自定义 char 参数的 str.strip()方法——它从文本中删除换行符。

### **从字符串中删除标点符号**

文本处理问题以各种形式出现，在某些情况下，当文本被拆分成单词时，不需要的标点符号会留在文档中。

解决这个问题的一个简单方法是创建一个自定义函数，通过包含常用标点符号的正则表达式从单个单词中删除这些标点符号。

使用 sub()方法可以实现这个函数——这是 Python 标准库的“re”模块下的一个正则表达式函数。这里有一个示例解决方案:

```py
import re

def punctationRemove(word):
return re.sub(r'[!?.:;,"()-]', "", word)

punctationRemove("...Python!")
# Output: 'Python'
```

函数有一个正则表达式模式，用于保存英语中常见的标点符号。注意撇号(')没有出现在正则表达式中，因为保留像“I'll”这样的缩写是有意义的。

代码中的 re()函数通过用空字符串替换匹配的标点符号来删除标点符号。最后，它返回单词的干净版本。

但是这个转换函数只是解决方案的一半——您现在需要使用 map()函数将这个函数应用于文本文件中的所有单词。一种方法是将字符串提供给变量，比如“text”，然后将 text.split()函数的返回值存储在 words 变量中。

如果你打印单词变量，你会发现它是一个列表，将字符串中的单词作为单独的字符串项保存。此时您也许能够推断出最后一步——运行 map 函数，并将标点符号移除函数和单词列表传递给它。

记得把这个 map 函数放在 list()函数中，打印出去掉标点符号的单词列表。

您提供给程序的单词可能带有标点符号。对 map()函数的调用将标点符号移除()函数应用于每个单词，移除所有标点符号。因此，第二个列表包含所有被清理的单词。

## **用 map()变换数字项**

在处理包含数值的可重复项时，map()方法很有可能派上用场。有了它，开发人员可以执行广泛的算术和数学运算，包括将字符串转换为整数或浮点数等。

使用幂运算符是最常见的数学运算之一，用于转换包含数值的可重复项。下面是一个带有转换函数的程序的例子，它接受一个数字并返回这个数字的平方和立方:

```py
def exponent(x):
return x ** 2, x ** 3

numList = [3, 4, 5, 6]

list(map(exponent, numList))
# Output: [(9, 27), (16, 64), (25, 125), (36, 216)]
```

exponent()函数从用户处接受一个数字，并返回平方值和立方值。注意，输出包括元组，因为 Python 将返回值作为元组来处理。对该函数的每次调用都返回一个二元组。

接下来，当 exponent()函数作为参数传递给 map()方法时，它返回一个保存已处理数字的元组列表。

这只是 map()方法可能实现的数学转换的一个例子。使用 map()可以完成的其他任务包括从 iterable 的每个元素中添加和减去特定的常数。但是您也可以使用像 sin()和 sqrt()这样的函数，它们是 Python 标准库的数学模块和 map()的一部分。

下面是一个如何将 factorial()方法与 map()一起使用的示例:

```py
import math

numbersList = [1, 2, 3, 4]

list(map(math.factorial, numbersList))

# Output: [1, 2, 6, 24]
```

在这个例子中，numbersList 列表被转换成一个新的列表，其中保存了原始列表中数字的阶乘值。在代码中使用 map()函数没有任何限制。确保你对这个函数有所思考，并想出你自己的程序来巩固你对它的理解。

如果你觉得这篇文章有帮助，你可能想看看这篇关于 FastAPI 的文章。