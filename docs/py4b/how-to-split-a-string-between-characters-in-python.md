# 如何在 Python 中分割字符间的字符串

> 原文：<https://www.pythonforbeginners.com/basics/how-to-split-a-string-between-characters-in-python>

在这篇 Python 拆分字符串的指南中，我们将探索使用该语言精确拆分字符串的各种方法。当我们在 Python 中分割字符之间的字符串时，可以从整体中提取字符串的一部分(也称为子串)。

学习如何拆分字符串对任何 Python 程序员都是有用的。无论您打算将 Python 用于 web 开发、数据科学还是自然语言处理，拆分字符串都将是一个常规操作。我们将遵循几个在 Python 中获取子字符串的过程。首先，我们来看看*拼接符号*和**分割**()函数。之后，我们将研究更高级的技术，比如 *regex* 。

## 用切片符号在字符之间分割字符串

谈到拆分字符串，对于 Python 开发人员来说，*切片符号*是一个显而易见的选择。使用切片符号，我们可以找到字符串的一个子部分。

**示例:用切片符号分割字符串**

```py
text = """BERNARDO
Well, good night.
If you do meet Horatio and Marcellus,
The rivals of my watch, bid them make haste."""

speaker = text[:8]

print(speaker) 
```

**输出**

```py
BERNARDO
```

## 按字符位置拆分字符串

要使用这种方法，我们需要知道要分割的子串的开始和结束位置。我们可以使用 **index** ()方法来查找字符串中某个字符的索引。

**举例:如何找到字符串中某个字符的索引**

```py
sentence = "Jack and Jill went up the hill."

index1 = sentence.index("J",0)
print(index1)

index2 = sentence.index("J",1)
print(index2) 
```

**输出**

```py
0
9 
```

## 使用 split()的快速指南

Python 标准库附带了一个用于拆分字符串的函数: **split** ()函数。这个函数可以用来分割字符之间的字符串。split()函数有两个参数。第一个叫做*分隔符*，它决定使用哪个字符来分割字符串。

split()函数返回原始字符串的子字符串列表。通过向 split()函数传递不同的值，我们可以用多种方式拆分字符串。

## 使用 split()函数拆分字符串

我们可以通过使用 **split** ()函数中的*分隔符*来指定拆分字符串的字符。默认情况下，split()将使用空格作为分隔符，但是如果我们愿意，我们可以自由地提供其他字符。

**示例:用空格分割字符串**

```py
sentence = "The quick brown fox jumps over the lazy dog."

# split a string using whitespace
words = sentence.split()

print(words) 
```

**输出**

```py
['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog.']
```

**示例:拆分由逗号分隔的字符串**

```py
rainbow = "red,orange,yellow,green,blue,indigo,violet"

# use a comma to separate the string
colors = rainbow.split(',')

print(colors) 
```

**输出**

```py
['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
```

## 使用带多个参数的 split()

使用 split()函数，我们还可以控制拆分多少行文本。这个函数接受第二个参数: *maxsplit* 。这个变量告诉 split()函数要执行多少次拆分。

**示例:拆分多行文本**

```py
text = """HORATIO
Before my God, I might not this believe
Without the sensible and true avouch
Of mine own eyes."""

lines = text.split(maxsplit=1)

print(lines) 
```

**输出**

```py
['HORATIO', 'Before my God, I might not this believe\nWithout the sensible and true avouch\nOf mine own eyes.']
```

因为我们将 maxsplit 的值设置为 1，所以文本被分成两个子字符串。

## 如何在两个相同的字符之间拆分字符串

如果我们有一个由多个相同字符分割的文本，我们可以使用 split()函数来分隔字符之间的字符串。

**示例:使用符号分隔字符串**

```py
nums = "1--2--3--4--5"

nums = nums.split('--')

print(nums) 
```

**输出**

```py
['1', '2', '3', '4', '5']
```

## 如何找到两个符号之间的字符串

我们可以将 index()函数与切片符号结合起来，从字符串中提取一个子字符串。index()函数将给出子串的开始和结束位置。一旦我们知道了符号的位置(本例中是**、**的**、T3)，我们将使用切片符号提取字符串。**

**示例:用 index()函数提取子串**

```py
# extract the substring surrounded by $'s from the text
text = "Lorem ipsum dolor sit amet, $substring$ adipisicing elit."

start = text.index('$')
end = text.index('$',start+1)

substring = text[start+1:end]
print(f"Start: {start}, End: {end}")
print(substring) 
```

**输出**

```py
Start: 28, End: 38
substring 
```

## 如何使用正则表达式在字符间拆分字符串

正则表达式是在字符串或文本中搜索模式的一种便捷方式。因为正则表达式模式(regex)非常通用，所以可以用来创建非常有针对性的搜索。

Python 自带了**re***库。使用 regex，我们可以仔细搜索文本，查找特定的单词、短语，甚至是特定长度的单词。*

**示例:使用正则表达式搜索字符串**

```py
import re

text="""The Fulton County Grand Jury said Friday an investigation
of Atlanta's recent primary election produced "no evidence" that
any irregularities took place."""

# search the text for words that are 14 characters long
match= re.search(r"\w{14}", text)
print(match.group())

# search the text for the word "Atlanta"
atlanta = re.search(r"Atlanta",text)
print(atlanta.group()) 
```

**输出**

```py
irregularities
Atlanta 
```

**示例:使用正则表达式查找日期**

```py
sentence= "Tony was born on May 1st 1972."

date= re.search(r"\d{4}",sentence)

print(date.group()) 
```

**输出**

```py
1972
```

在上面的例子中，我们使用了 **search()** 方法来使用正则表达式模式查找子串。这个方法有两个参数。第一个是我们的 regex 模式，第二个是我们想要执行搜索的字符串。

正则表达式使用特殊字符和数字创建目标搜索。例如，我们的第一个例子使用特殊字符 **\w** 来搜索单词。

**正则表达式的特殊字符**:

*   **/w**–搜索字母数字字符(单词)
*   **/d**–搜索数字字符(0-9)
*   **/s**–搜索空白字符

**示例:查找字符串是否以带有正则表达式的单词开头**

```py
speech= """HAMLET
O God, your only jig-maker. What should a man do
but be merry? for, look you, how cheerfully my
mother looks, and my father died within these two hours."""

match= re.search(r"^HAMLET",speech)
print("HAMLET" in match.group()) 
```

**输出**

真实的

此外，我们可以使用 regex 来查找两个字符之间的字符串。在下一个例子中，我们将使用 regex 模式来查找方括号中的字符串。

**示例:查找两个特殊字符之间所有字符的正则表达式**

```py
speech="""KING CLAUDIUS
[Aside] O, 'tis too true!
How smart a lash that speech doth give my conscience!"""

match = re.search(r"(?<=\[).+?(?=\])",speech)
print(match.group()) 
```

**输出**

```py
Aside
```

正则表达式包括许多元字符。涵盖所有这些超出了本教程的范围，但这里有一些以上的例子。

**更多正则表达式元字符**

*   **\**–用于转义特殊字符(例如， **[** 字符)
*   **。**–通配符(匹配除换行符以外的任何字符)
*   **+**–匹配多次出现
*   **？**–将前面的字符标记为可选

## 使用 Slice 对象分割字符串

Python slice 对象用于分割序列，如字符串或列表。slice 对象告诉 Python 如何对序列进行切片。

切片对象有三个**参数**:开始、停止和步进。前两个参数告诉 Python 从哪里开始和结束切片，而*步骤*参数描述了每个步骤之间的增量。

用一个切片对象我们可以得到字符之间的子串。要创建切片对象，使用**切片**()函数。该函数返回一个新的切片对象，该对象可以应用于字符串或其他序列。

**示例:使用 Slice 对象获取子串**

```py
text="""To be, or not to be, that is the question,
Whether 'tis nobler in the mind to suffer"""

x = slice(19)
words = text[x]
print(words) 
```

**输出**

```py
To be, or not to be
```

## 摘要

本指南探讨了几种在字符之间拆分字符串的技术。这项任务最简单的解决方案通常是切片标记法，但这并不总是正确的。根据您的需要，可能需要使用其他 Python 方法来实现您的目标。

以下是我们所涉及主题的快速回顾:

*   使用 split()函数，我们可以将字符串分割成子字符串。
*   如果您需要非常有针对性的搜索，请尝试使用正则表达式。
*   切片对象是对字符串进行切片的另一种选择。
*   切片符号是一种在字符之间拆分字符串的快速方法。

您可以将每个选项视为 Python 开发人员工具箱中的一个工具。记住使用合适的工具来完成这项工作，你就会走上正轨。

## 相关职位

如果你觉得这个指南很有帮助，并且渴望学习更多的 Python 编程，请查看 Python 初学者的链接。

*   使用 [Python 写文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)保存文本文件
*   如何用[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)在 Python 中连接字符串