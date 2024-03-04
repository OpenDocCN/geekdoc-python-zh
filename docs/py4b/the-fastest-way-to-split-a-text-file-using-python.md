# 使用 Python 分割文本文件的最快方法

> 原文：<https://www.pythonforbeginners.com/files/the-fastest-way-to-split-a-text-file-using-python>

Python 是世界上最流行的编程语言之一。它受欢迎的一个原因是 Python 使得处理数据变得容易。

从文本文件中读取数据是 Python 中的例行任务。在这篇文章中，我们将看看使用 Python 读取和分割文本文件的最快方法。拆分数据会将文本转换为列表，使其更易于处理。

我们还将介绍 Python 中拆分文本文件的一些其他方法，并解释这些方法如何以及何时有用。

在下面的例子中，我们将看到 Python 如何帮助我们掌握阅读文本数据。利用 Python 的许多内置函数将简化我们的任务。

## 介绍 split()方法

在 Python 中拆分文本的最快方法是使用 **split()** 方法。这是一个内置的方法，可用于将字符串分成各个部分。

split()方法将返回一个字符串中的元素列表。默认情况下，Python 使用空格来分割字符串，但是您可以提供一个分隔符并指定使用什么字符。

例如，逗号(，)常用来分隔字符串数据。逗号分隔值(CSV)文件就是这种情况。无论您选择什么作为分隔符，Python 都将使用它来分割字符串。

### 用 split()方法拆分文本文件

在第一个例子中，我们有一个雇员数据的文本文件，包括雇员的姓名、电话号码和职业。我们需要编写一个 Python 程序来读取这些随机生成的信息，并将数据分成列表。

*employee _ data . txt*
Lana Anderson 485-3094-88 电工
Elian Johnston 751-5845-87 室内设计师
Henry Johnston 777-6561-52 天文学家
Dale Johnston 248-1843-09 记者
Luke Owens 341-7471-63 教师
Amy Perry 494-3532-17 电工
Chloe Baker 57

在使用 Python **with** 语句打开数据文件后，我们可以使用 for 循环遍历文件内容。读取数据后，使用 split()方法将文本分成单词。

在我们的例子中，文本使用空格分隔，这是 split()方法的默认行为。

#### 示例 1:使用 Python 拆分员工数据

```py
with open("employee_data.txt",'r') as data_file:
    for line in data_file:
        data = line.split()
        print(data) 
```

**输出**

```py
['Lana', 'Anderson', '485-3094-88', 'Electrician']
['Elian', 'Johnston', '751-5845-87', 'Interior', 'Designer']
['Henry', 'Johnston', '777-6561-52', 'Astronomer']
['Dale', 'Johnston', '248-1843-09', 'Journalist']
['Luke', 'Owens', '341-7471-63', 'Teacher']
['Amy', 'Perry', '494-3532-17', 'Electrician']
['Chloe', 'Baker', '588-7165-01', 'Interior', 'Designer'] 
```

### 用逗号分割字符串

我们为 split()方法提供了一个可选的分隔符来指定用哪个字符来分割字符串。默认分隔符是空白。

在下一个例子中，我们将使用逗号来分割从文件中读取的测试分数数据。

*grades.txt*
珍妮特，100，50，69
托马斯，99，76，100
凯特，102，78，65

#### 示例 2:用逗号分割分数

```py
with open("grades.txt",'r') as file:
    for line in file:
        grade_data = line.strip().split(',')
        print(grade_data) 
```

这里使用了 **strip()** 方法来删除行尾的换行符(\n)。

**输出**

```py
['Janet', '100', '50', '69']
['Thomas', '99', '76', '100']
['Kate', '102', '78', '65'] 
```

### 用 splitlines()拆分文本文件

方法的作用是获取一个文本文件中的行列表。在接下来的例子中，我们将假设我们运行一个专门面向戏剧公司的网站。我们从文本文件中读取脚本数据，并将其推送到公司的网站上。

罗密欧，罗密欧，为什么你是罗密欧？
否定你的父亲，拒绝你的名字。
或者如果你不愿意，只要发誓做我的爱人，我就不再是凯普莱特家族的人了。

我们可以读取文件，并用 splitlines()方法将这些行分割成一个列表。然后，可以使用 for 循环来打印文本数据的内容。

#### 示例 3:使用 splitlines()读取文本文件

```py
with open("juliet.txt",'r') as script:
    speech = script.read().splitlines()

for line in speech:
    print(line) 
```

## 使用生成器拆分文本文件

在 Python 中，生成器是一个特殊的例程，可用于创建数组。生成器类似于返回数组的函数，但它一次返回一个元素。

生成器使用 **yield** 关键字。当 Python 遇到 yield 语句时，它存储函数的状态，直到稍后再次调用生成器。

在下一个例子中，我们将使用一个生成器来读取莎士比亚的*罗密欧与朱丽叶*中罗密欧的著名演讲的开头。使用 yield 关键字可以确保在每次迭代中保存 while 循环的状态。这在处理大文件时会很有用。

*romeo.txt*
但是柔柔，那边窗户透进来的是什么光？
是东方，朱丽叶是太阳。
起来，美丽的太阳，杀死嫉妒的月亮，
她已经病入膏肓，悲伤苍白
你，她的女仆，远比她美丽。

#### 示例 4:使用生成器拆分文本文件

```py
def generator_read(file_name):
    file = open(file_name,'r')
    while True:
        line = file.readline()
        if not line:
            file.close()
            break
        yield line

file_data = generator_read("romeo.txt")
for line in file_data:
    print(line.split()) 
```

## 用列表理解读取文件数据

Python 列表理解为处理列表提供了一个优雅的解决方案。我们可以利用较短的语法来编写带有列表理解的代码。此外，列表理解语句通常更容易阅读。

在前面的例子中，我们不得不使用 for 循环来读取文本文件。我们可以使用列表理解将 for 循环换成一行代码。

**列表理解语法:**
*my_list =【表达式】for *元素*in*List**

一旦通过列表理解获得了数据，我们就使用 split()方法来分离这些行，并将它们添加到一个新的列表中。

使用前一个例子中的同一个 *romeo.txt* 文件，让我们看看 list comprehension 如何提供一种更优雅的方法来分割 Python 中的文本文件。

#### 示例 5:使用列表理解来读取文件数据

```py
with open("romeo.txt",'r') as file:
    lines = [line.strip() for line in file]

for line in lines:
    print(line.split()) 
```

# 将一个文本文件分割成多个较小的文件

如果我们有一个大文件，我们想分裂成较小的文件呢？我们使用 for 循环和切片在 Python 中分割一个大文件。

通过列表切片，我们告诉 Python 我们想要处理给定列表中特定范围的元素。这是通过提供切片的起点和终点来完成的。

在 Python 中，列表可以用冒号分割。在下面的例子中，我们将使用列表切片将一个文本文件分割成多个更小的文件。

### 用列表切片分割文件

可以使用 Python 列表切片来分割列表。为此，我们首先使用 **readlines()** 方法读取文件。接下来，文件的上半部分被写入一个名为 *romeo_A.txt* 的新文件。我们将在这个 for 循环中使用列表切片将原始文件的前半部分写入一个新文件。

使用第二个 for 循环，我们将把剩余的文本写到另一个文件中。为了执行切片，我们需要使用 **len()** 方法来查找原始文件中的总行数。

最后， **int()** 方法用于将除法结果转换为整数值。

#### 示例 6:将单个文本文件拆分成多个文本文件

```py
with open("romeo.txt",'r') as file:
    lines = file.readlines()

with open("romeo_A.txt",'w') as file:
    for line in lines[:int(len(lines)/2)]:
        file.write(line)

with open("romeo_B.txt",'w') as file:
    for line in lines[int(len(lines)/2):]:
        file.write(line) 
```

在与 romeo.txt 相同的目录下运行该程序将创建以下文本文件。

*romeo_A.txt*
但是柔柔，那边窗户透进来的是什么光？
是东方，朱丽叶是太阳。

*罗密欧 _B.txt*
起来吧，美丽的太阳，杀死嫉妒的月亮，
她已经病入膏肓，悲伤苍白
你，她的女仆，远比她美丽。

## 相关职位

我们已经看到了如何使用 split()方法来分割文本文件。此外，我们的示例展示了 split()如何与 Python 生成器和 list comprehension 配合使用，以更优雅地读取大文件。

利用 Python 的许多内置方法，如 split()和 readlines()，我们可以更快地处理文本文件。使用这些工具将节省我们的时间和精力。

如果你真的想掌握 Python，花些时间学习如何使用这些方法来准备你自己的解决方案是个好主意。

如果你想学习更多关于 Python 编程的知识，请访问下面的 Python 初学者教程。

*   Python 注释如何决定你程序的成败
*   用 [Python 列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)加速您的代码