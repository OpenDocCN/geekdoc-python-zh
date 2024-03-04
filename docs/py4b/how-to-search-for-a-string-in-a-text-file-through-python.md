# 如何通过 Python 在文本文件中搜索字符串

> 原文：<https://www.pythonforbeginners.com/strings/how-to-search-for-a-string-in-a-text-file-through-python>

在这篇文章中，我们将探索在文本文件中搜索字符串的各种方法。使用 Python 3 编写的例子，我们将演示如何打开和读取文本文件和 CSV 文件。打开后，我们可以在这些文件中搜索特定的字符串。

文本文件可以保存许多不同类型的数据。需要在这些文件中搜索字符串的情况并不少见。Python 标准库包括许多函数和模块，简化了在文本文件中搜索字符串的过程。

## 使用 **readlines** ()方法在文本文件中搜索字符串

在我们可以在文本文件中搜索字符串之前，我们需要读取文件的内容。 *readlines* ()方法读取一个文件的数据并返回它所包含的行的列表。列表中的每个元素将包含文本文件中的一行。

在我们的第一个例子中，我们将创建一个新的文本文件，并在其中搜索特定的字符串。我们将使用 readlines()方法来读取文件的数据。然后我们将这些数据分配给一个名为 *lines 的变量。*

有了行列表，我们可以使用 for 循环遍历文本文件，搜索包含字符串“magician”的任何行。如果找到一个，就将整行打印到终端。

**示例:使用 readlines()** 在文本文件中查找字符串

```py
# create a new text file and write some text
text_file = open("text_file.txt",'w')
text_file.write("The magician appeared without a sound.")
text_file.close()

# safely open the text file we created and search for a string
with open("text_file.txt",'r') as text_file:
    lines = text_file.readlines()
    for line in lines:
        if "magician" in line:
            print(line) 
```

**输出**

```py
The magician appeared without a sound.
```

## 使用 **read** ()方法在文本文件中搜索字符串

在 Python 中，我们可以使用的另一种从文本文件中读取数据的方法是 *read* ()。这个方法返回一个包含文件内容的字符串。

使用 read()方法，我们将在下面的文本文件中搜索单词“魔术师”如果我们要找的单词在文本文件中，程序会通过在控制台上打印“True”来让我们知道。

*魔术师 _ 故事. txt*
魔术师无声无息的出现了。她紧张地笑了笑。她听到时钟滴答作响
，意识到自从她最后一次见到魔术师之后，他已经变了形。朱莉娅觉得她的心跳到了嗓子眼。

**示例:使用 read()方法搜索文本文件**

```py
story_file = open("magic_story.txt",'r')
story_text = story_file.read()
story_file.close()

if "magician" in story_text:
    print("True") 
```

**输出**

```py
True
```

## 统计字符串在文本文件中出现的次数

接下来，使用前面例子中的故事，我们将演示如何计算一个单词在文本文件中出现的次数。有时，一个文本文件包含一个单词或短语的许多实例。使用 Python，可以计算一个字符串在文本文档中出现的次数。

在下面的例子中，一个带有语句的 Python **用于打开文本文件并读取其内容。以这种方式打开文件可以确保文件被安全地处理和正确地关闭。随着文件的打开，数据通过 *readlines* ()方法被转换成一个字符串列表。

最后，使用一个 *f 字符串*打印结果。**

**示例:计算包含特定字符串的文本行数**

```py
with open("magic_story.txt",'r') as text_file:
    lines = text_file.readlines()
    count = 0
    for line in lines:
        if "magician" in line:
            count += 1

    print(f"Found 'magician' {count} times.") 
```

**输出**

```py
Found 'magician' 2 times.
```

## 在文本文件中查找包含字符串的所有行

到目前为止，我们已经了解了如何在文本文件中搜索字符串。但是如果我们需要更多的信息呢？假设我们想要记录包含字符串的文本行。我们如何用 Python 做到这一点？

从文件中提取行的最简单的方法可能是使用列表。对于下一个例子，我们将在文档中搜索一个字符串，并使用 *append* ()方法将包含我们正在寻找的字符串的任何行添加到一个列表中。

append()方法用于向列表中添加元素。一旦一个元素被添加到列表中，就可以使用括号来访问它。通过将元素的索引放在列表变量后面的括号中，我们可以获得对该元素的引用。

我们还需要使用*剥离*()方法来移除每行末尾的换行符。在这个练习中，我们将使用以下文本，节选自 T.S .艾略特的*《透明人》*。

*hollow_men.txt*
欲望
与效力
与存在
之间的痉挛
本质
与下凡
之间落下阴影

**示例:使用列表存储文本行**

```py
file = open("hollow_men.txt",'r')
lines = file.readlines()
file.close()

lines_we_want = []

for i in range(len(lines)):
    if "Between" in lines[i]:
        lines_we_want.append(lines[i].strip("\n"))

print(lines_we_want)
```

**输出**

```py
['Between the desire', 'Between the potency', 'Between the essence']
```

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 使用 **find** ()方法在文本文件中搜索字符串

*find* ()方法返回一个字符串第一个实例的位置。如果没有找到字符串，这个方法返回值-1。我们可以用这个方法来检查一个文件是否包含一个字符串。

**示例:使用 find()方法**

```py
text = "Hello World"

print(text.find("Hello"))
print(text.find("World"))
print(text.find("Goodbye"))
```

**输出**

```py
0
6
-1
```

首先，我们可以用一些示例文本创建一个新的文本文件。其次，我们将使用带有语句的**打开文件。读取文件内容后，我们将数据存储为一个新变量。最后，使用 find()方法，我们将尝试定位单词“魔术师”的第一个实例并打印结果。**

**示例:使用 find()方法在文本文件中搜索字符串**

```py
# create a new texts file and write some text
text_file = open("new_file.txt",'w')
text_file.write("The magician appeared without a sound.")
text_file.close()

# safely open the text file we created and search for a string
with open("new_file.txt",'r') as text_file:
    text_data = text_file.read()
    print(text_data.find("magician")) 
```

**输出**

```py
4
```

## 如何在 CSV 文件中搜索字符串数据

CSV 文件以纯文本形式存储数据。这些文件通常包含由逗号分隔的字符串或数字。这就是为什么我们称它们为逗号分隔值文件。这些 CSV 文件使得程序交换信息更加容易。

Python 附带了一个用于读取 CSV 文件的模块。 **csv** 模块包括从这些文件类型中提取数据的特殊方法。其中一种方法是*读本*()。

假设我们的员工交给我们一个文件，其中包含一个网站的用户数据列表。我们的工作是在这些数据中搜索特定的用户。我们将首先搜索一个名为“the_magician”的用户。

*account_info.csv*
用户名，邮箱，加入日期
cranky_jane，[【邮箱保护】](/cdn-cgi/l/email-protection)，01-21-2021
the_magician，[【邮箱保护】](/cdn-cgi/l/email-protection)，10-10-2020
军士 _ 佩珀，[【邮箱保护】](/cdn-cgi/l/email-protection)，05-15-2020

在找到用户句柄之前，我们需要读取文件数据。通过将 *reader* ()方法与 for 循环相结合，可以迭代包含在文本文件中的 CSV 数据行。如果我们遇到一行包含我们要找的用户名，我们将打印该行的数据。

**示例:在 CSV 文件中搜索字符串**

```py
import csv

# the string we want to find in the data
username = "the_magician"

with open("account_info.csv",'r') as data_file:
    # create a reader object from the file data
    reader = csv.reader(data_file)

    # search each row of the data for the username
    for row in reader:
        if username in row:
            print(row) 
```

**输出**

```py
['the_magician', '[[email protected]](/cdn-cgi/l/email-protection)', '10-10-2020']
```

如果我们想在 CSV 文件中搜索多个用户名，该怎么办？我们可以通过创建一个我们想要搜索的用户名列表来做到这一点。将每一行与用户名列表进行比较，就有可能找到我们要查找的帐户数据。

**示例:在 CSV 文件中搜索字符串列表**

```py
import csv

# the strings we want to find in the data
usernames = ["the_magician","cranky_jane"]

with open("account_info.csv",'r') as data_file:
    # create a reader object from the file data
    reader = csv.reader(data_file)

    # search each row of the data for the username
    for row in reader:
        for username in usernames:
            if username in row:
                print(row) 
```

**输出**

```py
['cranky_jane', '[[email protected]](/cdn-cgi/l/email-protection)', '01-21-2021']
['the_magician', '[[email protected]](/cdn-cgi/l/email-protection)', '10-10-2020'] 
```

## 结论

有了合适的工具，完成一项工作就容易多了。这就是为什么了解 Python 的许多特性非常重要。为了在文本文件中搜索字符串，我们需要学习像 *readlines* ()和 *find* ()这样的方法。这些都是通用的方法，非常适合解决许多不同的问题。

## 相关职位

如果您想了解更多关于 Python 开发的知识，请点击这些链接，查看 Python 为初学者提供的其他优秀教程。

*   使用 [Python 读取文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)方法打开文本文件
*   如何给程序添加一个 [Python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)