# Python 中单引号和双引号的区别

> 原文：<https://www.askpython.com/python/string/difference-between-single-and-double-quotes-in-python>

一个[字符串](https://www.askpython.com/python/string/strings-in-python)是一个字符序列。Python 中允许**开始**和**结束**一个带单引号和双引号的字符串文字。python 编程中有两种表示字符串的方法。

在本文中，您将借助一个示例(即代码及其输出)看到两个引号之间的区别。

## Python 中单引号有什么用？

单引号用于标记新闻标题中的引用或直接引用。

当用 Python 编程时，我们通常对字符串文字使用单引号。例如-*‘我的标识符’*。让我们通过 Python 中的代码用一个例子来理解。

**注意:**当您知道字符串中可能包含双引号时，请始终使用单引号。

### Python 中单引号的用法示例

下面是单引号的实现代码。

```py
word = 'Ask?'
print(word)
sentence = 'Python Programming'
print(sentence)
name = '"Hi" ABC'
print(name)
congrat = 'We congrat's you.'
print(congrat)

```

**输出**

```py
Ask?
Python Programming
"Hi" ABC
Invalid Syntax

```

## Python 中的双引号是用来做什么的？

双引号是用来引起直接(逐字)引用的。例如，“我希望你会在这里，”他说。在 Python 编程中，我们使用双引号来表示字符串。让我们通过 python 中的代码用一个例子来理解。

注意:当你知道你的字符串中会有单引号时，用双引号将你的字符串括起来

### 密码

```py
wish = "Hello World!"
print(wish)
hey = "AskPython says "Hi""
print(hey)
famous ="'Taj Mahal' is in Agra."
print(famous)

```

**输出**

```py
Hello World!
Invalid Syntax
'Taj Mahal' is in Agra.

```

## Python 中单引号和双引号的主要区别

| **单引号** | **双引号** |
| 表示为“” | 表示为" " |
| 任何类似标识符的东西都用单引号括起来。 | 双引号通常用于文本。 |
| 单引号用于正则表达式、字典键或 SQL。 | 双引号用于表示字符串。 |
| ‘我们‘欢迎’你。’ | “你好，是我。” |

## 奖金 Python 中的三重引号

如果您必须使用可能同时包含单引号和双引号的字符串，该怎么办？为此，Python 允许使用三重引号。下面是一个简单的例子。三重引号还允许您向 Python 变量添加多行字符串，而不是仅限于单行。

### 三重引号示例

```py
sentence1 = '''He asked, "did you speak with him?"'''
print(sentence1)
sentence2 = '''"That's great", she said.'''
print(sentence2)

```

**输出:**

```py
He asked, "did you speak with him?"
"That's great", she said.

```

如您所见，Python 现在理解了双引号和单引号是字符串的一部分，不需要转义。

## 结论

为了结束这个简单的话题，我想说 Python 中单引号和双引号的区别并不大。这完全取决于我们使用单引号和双引号的情况。

作为程序员，您可以决定什么最适合您的字符串声明。如果有疑问，请使用三重引号，这样就不会对字符串中包含的内容有任何问题。