# 如何在 Python 中清除文本文件

> 原文：<https://www.pythonforbeginners.com/basics/how-to-clear-a-text-file-in-python>

虽然程序经常被用来创建文件，但有时程序员也需要擦除文件数据。幸运的是，有多种方法可以清除文件中的文本。在这篇文章中，我们将使用一些简单的例子来演示如何在 Python 中清除文本文件。

通过使用 Python 的一些标准工具，我们可以打开、读取和清除文本文件。我们还将学习如何使用切片符号从文本文件中删除特定的行。

Python 简化了[文件处理](https://www.pythonforbeginners.com/filehandling/file-handling-in-python)的过程，产生了更简洁的程序。作为学习本指南的结果，您将理解 Python 如何清除文本文件和删除数据行。

## 在写入模式下使用 open()函数清除文本文件

以**写**模式打开文件将自动删除文件内容。 *open* ()函数有两个参数，我们想要打开的文本文件和我们打开它的模式。

以写模式打开文件会清除其中的数据。此外，如果指定的文件不存在，Python 将创建一个新文件。删除文件最简单的方法是使用 open()并在写入模式下将它赋给一个新变量。

```py
file_to_delete = open("info.txt",'w')
file_to_delete.close() 
```

带有语句的 [Python **简化了异常处理。使用**和**在*写*模式下打开一个文件也将清除其数据。pass 语句完成了该示例。**](https://www.pythonforbeginners.com/files/with-statement-in-python)

```py
# clear the data in the info file
with open("info.txt",'w') as file:
    pass 
```

## 如何用 truncate()方法清除文件

truncate()方法减小文档的大小。此方法采用一个可选参数来设置文件的新大小。如果没有提供参数，则文件保持不变。

试图将文件截断到大于其原始大小可能会产生意外的结果。例如，如果截断大小超过原始文件大小，程序可能会在文件末尾添加空白。

我们将使用一个演示文本文件来测试 Python 截断文件的能力:

*info.txt*
吉多·范·罗苏姆在 1991 年创造了 Python。
Python 是一种通用编程语言。
Python 的优势之一是可读性。

```py
with open("info.txt",'r+') as file:
    file.truncate(16)

# read the file’s contents
with open("info.txt", 'r') as file:
    lines = file.readlines()
    for line in lines:
        print(line) 
```

**输出**

```py
Guido van Rossum
```

在上面的例子中，truncate()方法将文件缩减为文本的前 16 个字符。请注意，我们以读写模式打开了该文件。为了使 truncate 方法起作用，这是必要的。

通过将值 **0** 传递给 truncate()方法，可以完全清除文本文件。

```py
with open("example.txt",'r+') as file:
    file.truncate(0) 
```

上面的 Python 代码将清除一个文本文件的内容。以这种方式使用 truncate()方法会将文件大小减小到 0，并删除文件中包含的所有内容。

## 使用 Python 列表切片清除文本文件

使用 Python 切片符号，可以检索列表、字符串或元组的子集。使用这个 Python 特性，我们可以定义给定子集的开始和结束索引。

切片表示法使用一种特殊的语法来查找一系列值的子集。以下示例显示了切片标记法的工作原理。通过定义切片的开始、结束和步骤，我们将获得原始列表的子集。

```py
nums = [1,2,3,4,5,6,7,8,9]
sub = nums[1:8:2]
print(sub) 
```

**输出**

```py
[2, 4, 6, 8]
```

接下来，我们将使用上一节中相同的 info.txt 文档来演示如何使用切片符号来清除文本文件中的行。方法将返回文档中文本行的列表。

提取这些行之后，我们可以使用切片符号来创建一个新的列表，用它来覆盖旧文件。只有第一行文本会保留下来。其他的将从文本文件中清除。

```py
# read the file's contents
read_file = open("info.txt",'r')
lines = read_file.readlines()
read_file.close()

write_file = open("info.txt",'w')

# slice the file's content
lines = lines[:1]

for line in lines:
    write_file.write(line)
    print(line)

write_file.close() 
```

**输出**

```py
Guido van Rossum created Python in 1991.
```

## 如何从文本文件中删除特定的行

使用一些常见的 Python 函数，我们还可以从文本文件中清除特定的数据行。如果我们事先知道要删除哪些行，这将特别有帮助。在这种情况下，我们可以使用切片符号来检索文件的子集。

通过用文件的数据子集覆盖文件，我们可以删除文本行。下面的例子摘自美国作家兰斯顿·休斯的诗《推迟的梦》。

*dream.txt*
延期的梦会怎么样？
它会像阳光下的葡萄干一样干枯吗
？

### 清除文本文件的第一行

为了清除文件的第一行文本，我们需要使用一些 Python 的文件处理方法。首先，我们将使用 *readlines* ()获取文件的文本数据列表。用 *seek* ()方法，我们可以手动重新定位文件指针。

其次，我们可以使用*截断*()的方法来调整文件的大小。第三，我们将向文件中写入一个新的行列表。使用切片符号，可以省略原始文件的第一行。

```py
with open("dream.txt",'r+') as file:
    # read the lines
    lines = file.readlines()

    # move to the top of the file
    file.seek(0)
    file.truncate()

    file.writelines(lines[1:])

with open("dream.txt",'r') as file:
    lines = file.readlines()

for line in lines:
    print(line) 
```

**输出**

```py
Does it dry up

Like a raisin in the sun? 
```

### 从文本文件中清除多行

也可以从文件中删除多行文本。使用切片表示法，我们可以创建任何我们想要的数据子集。使用这种覆盖原始文件的方法，我们可以有效地清除文件中任何不需要的数据。

通过改变切片的开始和结束索引，我们将得到 *dream.txt* 文件的不同子集。也可以组合分片列表。在创建列表子集时，使用这些工具可以提供很多灵活性。

```py
with open("dream.txt",'r+') as file:
    # read the lines
    lines = file.readlines()

    # move to the top of the file
    file.seek(0)
    file.truncate()

    file.writelines(lines[:3])

with open("dream.txt",'r') as file:
    lines = file.readlines()

for line in lines:
    print(line) 
```

**输出**

```py
What happens to a dream deferred?
```

## 如何使用字符串清除文本文件

如果我们想从包含某个单词、短语或数字的字符串中删除一行，该怎么办？我们可以用一个 Python **if** 语句来检查每行的字符串数据。如果我们正在寻找的字符串在该行中，我们可以避免将它包含在输出文件中。

*employee.txt*
姓名:杰米·詹姆森
年龄:37
职业:文案
起始日期:2019 年 4 月 3 日

在这个例子中，我们需要打开文件三次。首先，我们将打开文件提取其内容。接下来，我们将把数据写入一个新文件，清除我们不想要的行。最后，我们需要打开文件并阅读它，以证明文件得到了正确处理。

```py
data_file  = open("employee.txt",'r')
lines = data_file.readlines()
data_file.close()

write_file = open("employee.txt",'w')
for line in lines:
    if "Age" not in line:
        write_file.write(line)

write_file.close()

read_file = open("employee.txt",'r')
lines = read_file.readlines()
read_file.close()

for line in lines:
    print(line) 
```

**输出**

```py
Name: James Jameson

Occupation: Copywriter

Starting Date: April 3, 2019 
```

## 摘要

在这篇文章中，我们深入探讨了如何在 Python 中清除文本文件。我们已经看到 Python 语言为完成这项任务提供了许多选项。使用每次安装 Python 时都标配的一些方法和函数，我们能够清除整个文件以及删除文件中的特定行。

虽然像切片符号这样的工具对于清除文本文件非常有用，但是它们在其他领域也很方便。这样，您学习的每一个 Python 新特性都将提高您的整体编码能力。

## 相关职位

如果你觉得这篇文章很有帮助，并且想学习更多关于 Python 编程的知识，请点击这些链接，阅读我们 Python 初学者团队的更多精彩文章。无论是文件处理还是数据管理，你都会学到在当今快速变化的数字世界中取得成功所需的技能。

*   [使用 Python try 进行高级错误处理，除了](https://www.pythonforbeginners.com/error-handling/python-try-and-except)
*   [Python 列表理解简化数据管理](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)