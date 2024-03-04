# 用 Python 从字符串中提取唯一字符

> 原文：<https://www.pythonforbeginners.com/basics/extract-unique-characters-from-string-in-python>

在 Python 中，字符串被广泛用于[文本分析。本文讨论如何在 Python 中从字符串中提取唯一字符。](https://www.pythonforbeginners.com/basics/text-analysis-in-python)

## 使用 for 循环从 Python 中的字符串提取唯一字符

为了使用 for 循环从字符串中提取唯一的字符，我们将首先定义一个空列表来存储输出字符。然后，我们将使用 for 循环遍历字符串字符。迭代时，我们将检查当前字符是否在列表中。如果是，我们将移动到下一个字符。否则，我们将使用`append()`方法将当前字符添加到列表中。在列表上调用 `append()`方法时，该方法将字符作为其输入参数，并将其附加到列表的末尾。

在执行 for 循环后，我们将在列表中获得字符串的所有唯一字符。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
output=[]
for character in myStr:
    if character not in output:
        output.append(character)
print("The output is:",output)
```

输出:

```py
The input string is: Python For Beginners
The output is: ['P', 'y', 't', 'h', 'o', 'n', ' ', 'F', 'r', 'B', 'e', 'g', 'i', 's']
```

在上面的例子中，我们使用了字符串“Python For 初学者”。在执行 for 循环之后，我们从字符串中获得唯一字符的列表。

## 使用集合从 Python 中的字符串提取唯一字符

[Python 集合](https://www.pythonforbeginners.com/basics/set-operations-in-python)数据结构用于将唯一的不可变对象存储在一起。因为 python 字符串和单个字符都是不可变的，所以我们可以使用集合从 Python 中的字符串中提取字符。

为了在 Python 中从字符串中提取唯一的字符，我们将首先创建一个空集。然后，我们将使用 for 循环遍历字符串字符。迭代时，我们将使用`add()` 方法将当前字符添加到集合中。在集合上调用`add()`方法时，该方法将字符作为其输入参数，如果该字符不在集合中，则将它添加到集合中。

在执行 for 循环之后，我们将获得字符串集合中的所有唯一字符。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
output=set()
for character in myStr:
    output.add(character)
print("The output is:",output)
```

输出:

```py
The input string is: Python For Beginners
The output is: {'F', 'e', 'y', 't', 'h', 'P', 'i', 'g', ' ', 'o', 's', 'r', 'n', 'B'}
```

除了使用 for 循环和`add()`方法，您还可以在集合上使用`update()`方法从 Python 中的字符串中提取唯一的字符。在 set 对象上调用`update()`方法时，该方法将 iterable 对象作为其输入参数。执行后，它将 iterable 对象的所有元素添加到集合中。由于字符串是一个可迭代的对象，您可以使用`update()`方法从字符串中提取唯一的字符。

为了使用`update()`方法从字符串中提取唯一的字符，我们将首先使用`set()`函数创建一个空集。然后，我们将在空集上调用`update()`方法，并将字符串作为输入参数传递给`update()`方法。执行后，我们将得到一个包含字符串所有唯一字符的集合。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
output=set()
output.update(myStr)
print("The output is:",output)
```

输出:

```py
The input string is: Python For Beginners
The output is: {'F', 'e', 'y', 't', 'h', 'P', 'i', 'g', ' ', 'o', 's', 'r', 'n', 'B'}
```

您也可以直接将输入字符串传递给`set()`函数，以创建一组字符串的唯一字符，如下所示。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
output=set(myStr)
print("The output is:",output)
```

输出:

```py
The input string is: Python For Beginners
The output is: {'F', 'e', 'y', 't', 'h', 'P', 'i', 'g', ' ', 'o', 's', 'r', 'n', 'B'}
```

建议阅读:[用 Python 创建聊天应用](https://codinginfinite.com/python-chat-application-tutorial-source-code/)

## 使用 Counter()方法从 Python 中的字符串中提取唯一字符

通过所有独特的字符，您还可以找到字符串中字符的频率。您可以使用 collections 模块中定义的 Counter 函数来执行此任务。

`Counter()`函数将一个 iterable 对象作为其输入参数，并返回一个 Counter 对象。计数器对象包含 iterable 对象的所有独特元素及其频率。

为了从一个字符串中提取所有独特的字符及其频率，我们将把这个字符串作为输入传递给`Counter()`方法。执行后，我们将获得字符串的所有唯一字符及其频率。您可以在下面的示例中观察到这一点。

```py
from collections import Counter
myStr="Python For Beginners"
print("The input string is:",myStr)
output=Counter(myStr)
print("The output is:",output)
```

输出:

```py
The input string is: Python For Beginners
The output is: Counter({'n': 3, 'o': 2, ' ': 2, 'r': 2, 'e': 2, 'P': 1, 'y': 1, 't': 1, 'h': 1, 'F': 1, 'B': 1, 'g': 1, 'i': 1, 's': 1})
```

## 结论

在本文中，我们讨论了在 Python 中从字符串中提取唯一字符的不同方法。

要了解更多关于 python 编程的知识，你可以阅读这篇关于[字符串操作](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)的文章。您可能也会喜欢这篇关于 [python simplehttpserver](https://www.pythonforbeginners.com/modules-in-python/how-to-use-simplehttpserver) 的文章。

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！