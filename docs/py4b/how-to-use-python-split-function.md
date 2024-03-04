# 如何使用 Python Split 函数

> 原文：<https://www.pythonforbeginners.com/strings/how-to-use-python-split-function>

Python 提供了一个用于拆分字符串的内置方法。使用 **split()** 函数，我们可以将单个字符串分解成一个字符串列表。

使用 split()是将大型字符串分解成更易于管理的部分的一种简单而有效的方法。

如果我们的任务是处理文本文件，这可能是一个有用的方法，我们将在后面看到。

## Split 函数的定义和语法

**split()** 方法使用分隔符将字符串分隔成单个单词，也称为*分隔符*。Python 使用空白作为默认分隔符，但是您可以自由地提供一个替代符。

分隔符可以是任何东西，但通常是用来分隔字符串中单词的字符。例如，逗号通常用于拆分列表。

```py
# the following string uses a comma as the separator.
groceries = “Bread,Milk,Eggs,Bananas,Coffee”
```

split()函数将返回一个字符串列表。默认情况下，对返回列表的长度没有限制。但是你可以用 *maxsplit* 设置来改变它。

**语法:**

```py
my_string.split(separator,maxsplit)
```

**分隔符:**这是 Python 用来分割字符串的分隔符。如果没有提供分隔符，则使用空白作为分隔符。

**maxsplit:** 该设置用于决定字符串应该被拆分多少次。默认情况下，该列表没有限制。

## 为什么要在 Python 中使用 split()函数？

使用 split()函数有很多好的理由。无论是使用逗号分隔值(CSV)文件，还是试图将一个大字符串分解成小部分，split()都提供了一种在 Python 中分割字符串的有效方法。

split()函数将把一个字符串分解成单词列表。这些以 Python 字符串数组的形式返回给我们。下面是使用 split()的分步指南:

1.  创建一个字符串。
2.  使用 Python split()函数
3.  打印结果

```py
gandalf = "You shall not pass."
gandalf.split() # The string will be split based on whitespace.
```

这是我们分开绳子时的样子。

```py
['You', 'shall', 'not', 'pass.']
```

使用 **split()** 函数和 [python 字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)来快速连接和拆分字符串。split()函数对于快速解析逗号分隔值(CSV)文件也很有用。

## Python 分割函数的示例

当我们使用 split()函数分割一个字符串时，Python 返回一个新的字符串数组。该数组保存拆分后的字符串。

比如说。如果我们有一个类似这样的字符串:

```py
frosty_string = "Some say the world will end in fire."
```

我们可以看到字符串中的单词由空格分隔。用这个字符串运行 split()将返回一个新数组，其中包含罗伯特·弗罗斯特的一首著名诗歌的开头一行的单词。

### 示例 1:使用 Python split()和一个基本字符串

```py
# this line is from Fire and Ice by Robert Frost
frosty_string = "Some say the world will end in fire."
words = frosty_string.split()
print(words)
```

**输出**

```py
['Some', 'say', 'the', 'world', 'will', 'end', 'in', 'fire.']
```

还可以使用 split()将列表分成预定义数量的部分。正如我们在它的定义中看到的，split()函数有一个可选的第二个参数。通过设置 *maxpslit* ，我们可以告诉 Python 将列表分成两部分。

### 示例 2:使用 Python Split 将列表分成两部分

```py
fruit = "Apples Oranges"
# setting maxsplit to 1 will return a list with 2 parts
two_max = fruit.split(' ', 1)
print(two_max) 
```

**输出**

```py
['Apples', 'Oranges']
```

## 使用逗号分割字符串数据

我们已经看到了 split()函数的基本用法。默认情况下，split()使用空格来分割字符串。虽然空格适用于某些数据，但其他字符通常更有意义。通常使用逗号来分隔字符串中的值。

使用逗号，我们可以很容易地快速拆分大量的字符串数据。在以下示例中，包含通用摇滚乐队中所有乐器的字符串被拆分为一个新列表。

### 示例 3:使用逗号作为分隔符

```py
# use a comma as the separator
band = "Guitar,Piano,Drums,Trumpet,Bass Guitar,Vocals"
instruments = band.split(',')
print(instruments) 
```

**输出**

```py
['Guitar', 'Piano', 'Drums', 'Trumpet', 'Bass Guitar', 'Vocals']
```

使用逗号作为分隔符可以很容易地将字符串转换成列表。将字符串转换成列表(反之亦然)是一种存储和读取数据的便捷方法。

## 用 Split()函数读取文本文件

我在电脑上创建了一个名为 *chess.txt* 的文本文件。该文件与我的 Python 文件位于同一文件夹中。

这个文本文件包含一行，列出了国际象棋游戏中使用的棋子。请注意，逗号用于分隔各部分的名称。

国王、王后、车、骑士、主教、棋子

利用 split()，我们可以很容易地将游戏棋子分开。我们将使用 Python 的 **open()** 方法来读取文件，并在分割字符串之前提取文本行。

### 示例 4:使用 split()分隔文本文件中的单词

```py
file = open("chess.txt")

pieces = file.read().split(',')

print(pieces) 
```

**输出**

```py
['King', 'Queen', 'Rook', 'Knight', 'Bishop', 'Pawn\n']
```

## 将 Strip()与 Split()函数一起使用

回头看看上一个例子，您可能会注意到我们的列表有些奇怪。我们在某个地方发现了一些迷失的角色。那个 **\n** 是哪里来的？

这是新的行字符。它告诉 Python 我们已经到达了文本文件的行尾。

我们可以利用 Python 的 **strip()** 方法把新的行字符从我们的列表末尾去掉。使用这种方法将确保我们不会从文件中提取任何讨厌的隐藏字符，例如 **\n** 字符**。**

```py
file = open("chess.txt")
pieces = file.read().strip().split(',')
print(pieces)
file.close() 
```

**输出**

```py
['King', 'Queen', 'Rook', 'Knight', 'Bishop', 'Pawn']
```

## 拆分多行文本

上面的例子是相对基础的。毕竟，我们只有一行文本要处理。但是如果我们有用逗号分隔的数据行呢？让我们看看读取一个简单的数据库可能是什么样子。

我们将为一位需要计算班上每个学生平均成绩的老师编写一个程序。

我们的客户需要跟踪他/她的学生的作业。具体来说，他/她想要一种读取存储在文本文件中的数据的方法。这个名为 *grades.txt* 的文件包含了老师的学生的成绩。

我们的 teachers_pet.py 程序将从一个文件中读取信息，并构建一个包含成绩数据的字符串。

一旦我们有了成绩字符串，我们就可以使用 split()函数来拆分成绩并将它们存储在一个列表中。之后，我们需要使用 **float()** 方法将字符串转换成数字。

虽然这确实是一个复杂的过程，但是这个例子展示了在 Python 中处理数字和字符串的各种方法。

**grades.txt**
诺亚 81、94、100、65
伊利亚 80、84、72、79
卢卡斯 95、80、89、89
艾玛 95、80、80、80、77
艾娃 90、84、85、80
阿米莉亚 100、100、95、0

### 例 5:教师的宠物项目

```py
# teachers_pet.py
file = open("grades.txt", 'r')

grade_data = []

while(True):
    line = file.readline()
    if not line:
        break

    grade_data.append(line.split(','))

print(grade_data)
file.close()

# this function will loop through the student's grades
def calculate_averages(grades_data):
    for grades in grade_data:
        total = 0
        # the student's name occupies the first spot
        for i in range(1,len(grades)):
            total += float(grades[i].strip()))
        avg = total/float(len(grades)-1)
        print("{} has an average of {}.".format(grades[0],avg))

calculate_averages(grade_data) 
```

**输出**

```py
Noah has an average of 85.0
Elijah has an average of 78.75
Lucas has an average of 88.25
Emma has an average of 83.0
Ava has an average of 84.75
Amelia has an average of 73.75 
```

我们的节目有两部分。在第一部分中，我们用 open()方法从文件中读取数据。一行一行地，每个学生的成绩被添加到一个名为 *grades_data* 的列表中。

我们利用 **float()** 将字符串转换成数字。我们还将使用 **strip()** 删除任何不必要的字符，例如 **\n** 字符。

请记住，成绩列表中的第一个元素是学生的姓名。

我们还需要一个函数来计算和打印成绩平均值。我们在**calculation _ averages()**中这样做。这个方便的功能将查看成绩数据，并找到每个学生的平均绩点。

## 进一步研究的教训

Python 附带了许多可以改进代码的内置函数。split()函数就是这样一个工具，可以用来分解困难的字符串问题。

使用分隔符分隔字符串允许您将字符串数据转换为字符串列表，这样更易于管理。split()函数也是一种快速分解从文件中读取的文本数据的方法。

通过访问以下教程，了解关于 Python 以及如何提高编程技能的更多信息。

*   [了解如何使用注释来改进您的 Python 代码](https://www.pythonforbeginners.com/comments/comments-in-python)
*   [如何使用 Python 串联来连接字符串](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)