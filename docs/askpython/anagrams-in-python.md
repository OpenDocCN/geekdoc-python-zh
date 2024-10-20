# Python 中的字谜简介

> 原文：<https://www.askpython.com/python/examples/anagrams-in-python>

你好。今天我们将学习如何在 Python 中实现一个有趣的主题，叫做字谜。让我们先了解一下什么是变位词。

## 什么是变位词？

字谜是单词和句子背后有趣的悬念。如果一个特定单词或句子的所有字母经过重新排列后都可以形成其他单词或句子，那么它们彼此都是变位词。

一些变位词的例子有“sram”和“mars”，“top”和“otp”等等。但是现在下一个问题是为什么要学习字谜呢？

字谜对作家真的很有帮助，因为它们给写作增加了一层额外的悬念，而且它们是一种聪明有趣的方式，让写作变得有趣。使用变位词真的很有趣。

## 在 Python 中检查变位词

让我们看看如何使用简单的算法识别 Python 中的变位词。

### 检验两个单词是否为变位词的算法

下面的步骤显示了如何检查两个字符串是否是变位词。

```py
STEP 1: Take input of the 1st string
STEP 2: Take input of the 2nd string
STEP 3: Sort all the letters of both the strings
STEP 4: Check if after sorting both the strings match.
if they match: Anagram
if not: Not an Anagram

```

### 检查两个字符串是否是变位词的程序

```py
s1 = input()
s2 = input()
s1 = sorted(s1)
s2 = sorted(s2)
if(s1==s2):
    print("Anagram")
else:
    print("Not an Anagram")

```

一些示例字符串的结果如下所示。首先检查的琴弦是`tac`和`cat`，以及`tic`和`cat`。我们可以清楚地看到，第一对是变位词，而第二对不是变位词。

```py
tac
cat
Anagram

```

```py
tic
cat
Not an Anagram

```

## 结论

恭喜你！我们学习了字谜以及如何在 Python 编程语言中实现它们。我希望现在你对字谜很清楚，并能自己实现它！

编码快乐！感谢您的阅读！