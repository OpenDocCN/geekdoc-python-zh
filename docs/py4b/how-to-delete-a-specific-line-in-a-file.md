# 如何删除文件中的特定行

> 原文：<https://www.pythonforbeginners.com/files/how-to-delete-a-specific-line-in-a-file>

因为 Python 没有提供删除文件中特定行的直接方法，所以有必要找到我们自己的方法。

在本指南中，我们将介绍几种使用 Python 从文本文件中移除行的方法。我们将看到如何根据行在文档中的位置删除行，以及如何删除与字符串匹配的内容。

我们还将介绍使用定制逻辑解决更具挑战性的问题的例子。不管我们是处理简单的文本文件，还是更复杂的逗号分隔文件(CSV)，这些技术都将帮助您管理您的数据。

我们可以使用 Python 以一种内存高效的方式处理大文件和小文件。

## 使用数字删除一行

在我们的第一个例子中，我们将根据一行在文件中的位置来删除它。从保存在我们计算机上的随机生成的姓名列表开始，我们将使用 Python 根据姓名在列表中出现的顺序从列表中删除一个姓名。

该文件名为 *names.txt* ，保存在与我们的 python 文件相同的目录中。我们的目标是删除文件中的第 7 行。

在 Python 中，我们可以使用带有语句的**来安全地打开文件。打开文件后，我们将使用 **readlines()** 方法来检索包含文件内容的列表。**

名单看完就这么多。接下来，我们将使用另一个带有语句的**再次打开文件，这次是在*写模式*下。**

使用 for 循环遍历文件中的行，我们还利用一个变量来跟踪当前的行号。当我们到达我们想要删除的行时，一个 **if** 语句确保我们跳过该行。

让我们再复习一遍这些步骤:

1.  以读取模式打开文件
2.  阅读文件内容
3.  以写模式打开文件
4.  使用 for 循环读取每一行并将其写入文件
5.  当我们到达我们想要删除的行时，跳过它

因为我们使用带有语句的 Python **来处理文件，所以在我们完成后没有必要关闭它。Python 为我们处理了这一点。**

*names.txt*

#### 示例 1:根据指定的行号删除一行

```py
def remove_line(fileName,lineToSkip):
    """ Removes a given line from a file """
    with open(fileName,'r') as read_file:
        lines = read_file.readlines()

    currentLine = 1
    with open(fileName,'w') as write_file:
        for line in lines:
            if currentLine == lineToSkip:
                pass
            else:
                write_file.write(line)

            currentLine += 1

# call the function, passing the file and line to skip
remove_line("names.txt",7) 
```

通过将我们的逻辑包装在一个函数中，我们可以通过调用 remove_lines()并传递文件名和我们想要删除的行号来轻松地从文件中删除一行。

如果我们计划不止一次地使用一个 Python 代码块，将它包装在一个函数中是一个好主意。这样做会节省我们的时间和精力。

## 通过匹配内容删除一行

我们已经看到了如何根据行的位置从文件中删除内容。现在我们来看看如何删除匹配给定字符串的行。

我们有一本童谣目录，但有人对我们搞了点恶作剧。具有讽刺意味的是，他们在我们的文件中添加了“此行不属于”这一行！

没必要惊慌。我们可以使用 Python 轻松地消除这种危害。

在我们的 Python 代码中，我们将首先读取名为 *itsy_bitsy.txt* 的文件，并将其内容存储在名为 *lines* 的变量中。

就像前面的例子一样，我们将使用 Python **和**语句来打开文件。为了找到匹配的行，我们需要删除 **readlines()** 加在每个字符串末尾的换行符。

我们可以使用 **strip()** 函数删除换行符。这是一个内置函数，可以删除字符串开头或结尾的字符。

当找到匹配的内容时，我们将使用 if 语句传递它，有效地将它从旧文件中删除。

小小的蜘蛛爬上了水龙卷。
大雨
降临，把蜘蛛冲走了。太阳出来了，这条线不属于这里，它擦干了所有的雨水，小蜘蛛又爬上了壶嘴。

#### 示例 2:匹配内容并将其从文件中移除

```py
with open("itsy_bitsy.txt", 'r') as file:
    lines = file.readlines()

# delete matching content
content = "This line doesn't belong"
with open("itsy_bitsy.txt", 'w') as file:
    for line in lines:
        # readlines() includes a newline character
        if line.strip("\n") != content:
            file.write(line) 
```

## 使用自定义逻辑删除 Python 中的一行

在处理文件数据时，我们经常需要定制的解决方案来满足我们的需求。在下面的例子中，我们将探索使用自定义逻辑来解决各种数据问题。

通过定制我们的解决方案，有可能解决更困难的问题。例如，如果我们想从文件中删除一行，但只知道它的一部分，该怎么办？

即使我们只知道一个单词，我们也可以使用 Python 找到需要删除的行。通过利用 Python 的内置方法，我们将看到如何用 Python 代码解决定制挑战。

### 删除带有特定字符串的行

在下一个练习中，我们将了解如何删除包含字符串一部分的行。基于从前面的例子中获得的知识，可以删除包含给定子串的行。

在 Python 中， **find()** 方法可用于在字符串中搜索子串。如果字符串包含子字符串，函数将返回表示其位置的索引。否则，该方法返回-1。

在一个名为 *statements.txt* 的文本文件中，我们有一个随机生成的句子列表。我们需要删除包含给定子串的任何句子。

通过使用 find()，我们将知道一行是否包含我们正在寻找的字符串。如果有，我们会从文件中删除。

下面是使用 **find():** 的语法

```py
mystring.find(substring)
```

*statement . txt*
他没有注意到关于香蕉的警告。我的朋友带着苹果去了市场。
她买了一个种植桃子的农场。山那边有一个可爱的葡萄园。她非常喜欢她的新车。

#### 示例 3:删除包含给定字符串的行

```py
# remove a line containing a string
with open("statements.txt",'r') as file:
    lines = file.readlines()

with open("statements.txt",'w') as file:
    for line in lines:
        # find() returns -1 if no match is found
        if line.find("nuts") != -1:
            pass
        else:
            file.write(line) 
```

### 删除文件中最短的一行

我们再来看看 *statement.txt* 。已经做了一些改变。

*statement . txt*
*他没有理会关于香蕉的警告。我的朋友带着苹果去市场。她买了一个种植桃子的农场。他声称看到了不明飞行物。山那边有一个可爱的葡萄园。除了椰子，这个岛上几乎没有什么可吃的。*

我们增加了一些新的台词。这一次，我们需要删除文档中最短的一行。我们可以通过使用 **len()** 方法来找到每一行的长度。

通过比较线路的长度，可以找到最短的线路。随后，我们可以使用一个带有语句的**打开并从文件中删除该行。**

#### 示例 4:使用 len()方法删除文件中最短的一行

```py
# remove the shortest line from statements.txt
with open("statements.txt",'r') as read_file:
    lines = read_file.readlines()

shortest = 1000 # used to compare line length
lineToDelete = "" # the line we want to remove

for line in lines:
    if len(line) < shortest:
        shortest = len(line)
        lineToDelete = line

with open("statements.txt",'w') as write_file:
    for line in lines:
        if line == lineToDelete:
            pass
        else:
            write_file.write(line) 
```

## 摘要

在这篇文章中，我们介绍了几种在 Python 中从文件中删除行的方法。我们看到，我们可以使用 for 循环根据行在文件中的位置删除行。

我们还可以通过比较字符串来删除匹配内容的文件，或者使用 **==** 操作符，或者使用 **find()** 方法。

这些只是在 Python 中从文件中删除行的一些方法。

## 相关职位

如果您想了解更多关于使用 Python 处理字符串和文件数据的信息，请点击下面的链接。

*   用 [Python 字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)连接字符串
*   如何使用 Python 字典进行更好的数据管理
*   使用 [Python 列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)来简化你的代码