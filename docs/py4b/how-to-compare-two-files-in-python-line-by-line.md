# 如何在 Python 中逐行比较两个文件

> 原文：<https://www.pythonforbeginners.com/basics/how-to-compare-two-files-in-python-line-by-line>

本教程研究了在 Python 中比较两个文件的各种方法。我们将讨论[读取两个文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)并逐行比较它们，以及使用可用的模块来完成这个常见的任务。

在 Python 中有很多方法可以比较两个文件。Python 附带了用于此目的的模块，包括 filecmp 和 difflib 模块。

以下 Python 3 示例对比了确定两个文件是否包含相同数据的各种方法。我们将使用 Python 3 内置的函数和模块，因此不需要下载额外的包。

## 逐行比较两个文本文件

我们可以使用 *open* ()函数读取文件中包含的数据来比较两个文本文件。open()函数将在本地目录中查找一个文件，并尝试读取它。

对于这个例子，我们将比较两个包含电子邮件数据的文件。我们被告知，这两份邮件列表可能并不相同。我们将让 Python 为我们检查文件。使用 *readlines* ()方法，可以从文本文件中提取行。

*emails_A.txt*
[【邮件保护】](/cdn-cgi/l/email-protection)
[【邮件保护】](/cdn-cgi/l/email-protection)
[【邮件保护】](/cdn-cgi/l/email-protection)
[【邮件保护】](/cdn-cgi/l/email-protection)
[【邮件保护】](/cdn-cgi/l/email-protection)

*emails_B.txt*
[【邮件保护】](/cdn-cgi/l/email-protection)
[【邮件保护】](/cdn-cgi/l/email-protection)
[【邮件保护】](/cdn-cgi/l/email-protection)
[【邮件保护】](/cdn-cgi/l/email-protection)
[【邮件保护】](/cdn-cgi/l/email-protection)

一旦数据被提取，一个 **[for 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)** 用于逐行比较文件。如果这些行不匹配，用户会收到一条消息，告诉他们不匹配发生在哪里。我们将包括数据本身，这样用户可以很容易地跟踪不同的行。

#### **示例:使用 Python 比较电子邮件列表**

```py
file1 = open("emails_A.txt",'r')
file2 = open("emails_B.txt",'r')

file1_lines = file1.readlines()
file2_lines = file2.readlines()

for i in range(len(file1_lines)):
    if file1_lines[i] != file2_lines[i]:
        print("Line " + str(i+1) + " doesn't match.")
        print("------------------------")
        print("File1: " + file1_lines[i])
        print("File2: " + file2_lines[i])

file1.close()
file2.close() 
```

**输出**

```py
Line 1 doesn't match.
------------------------
File1: [[email protected]](/cdn-cgi/l/email-protection)

File2: [[email protected]](/cdn-cgi/l/email-protection)

Line 3 doesn't match.
------------------------
File1: [[email protected]](/cdn-cgi/l/email-protection)

File2: [[email protected]](/cdn-cgi/l/email-protection)

Line 4 doesn't match.
------------------------
File1: [[email protected]](/cdn-cgi/l/email-protection)

File2: [[email protected]](/cdn-cgi/l/email-protection) 
```

## 使用 filecmp 模块比较文件

filecmp 模块包括用于在 Python 中处理文件的函数。具体来说，这个模块用于比较两个或多个文件之间的数据。我们可以使用 *filecmp.cmp* ()方法来实现这一点。如果文件匹配，这个方法将返回*真*，否则返回*假*。

这个例子使用了三个文件。第一个和第三个是一样的，而第二个略有不同。我们将使用 filecmp.cmp()方法通过 Python 来比较这些文件。

*标点 1.txt*
吃你的晚餐。
我要感谢我的父母、珍妮特和上帝。
对不起我关心你。她真的很喜欢烹饪、她的家庭和她的猫。

*标点符号 2.txt*
吃。你是晚餐！我要感谢我的父母、珍妮特和上帝。
对不起。我关心你。她真的很喜欢烹饪她的家人和她的猫。

*标点 3.txt*
吃你的晚餐。
我要感谢我的父母、珍妮特和上帝。
对不起我关心你。她真的很喜欢烹饪、她的家庭和她的猫。

在使用 filecmp 模块之前，我们需要导入它。我们还需要导入 **os** 模块，这将允许我们使用目录中的路径加载文件。在本例中，使用了一个自定义函数来完成比较。

在我们比较文件之后，我们可以看到数据是否匹配，最后，我们将提醒用户结果。

#### **示例:用 filecmp.cmp()** 比较两个文件

```py
import filecmp
import os

# notice the two backslashes
file1 = "C:\\Users\jpett\\Desktop\\PythonForBeginners\\2Files\\punctuation1.txt"
file2 = "C:\\Users\jpett\\Desktop\\PythonForBeginners\\2Files\\punctuation2.txt"
file3 = "C:\\Users\jpett\\Desktop\\PythonForBeginners\\2Files\\punctuation3.txt"

def compare_files(file1,file2):
    compare = filecmp.cmp(file1,file2)

    if compare == True:
        print("The files are the same.")
    else:
        print("The files are different.")

compare_files(file1,file2)
compare_files(file1,file3) 
```

**输出**

```py
The files are different.
The files are the same. 
```

## 使用 difflib 模块比较两个文件

difflib 模块对于比较文本和找出它们之间的差异很有用。这个 Python 3 模块预打包了该语言。它包含许多有用的功能来比较正文。

首先，我们将使用 *unified_diff* ()函数来查明两个数据文件之间的不匹配。这些文件包含虚构学生的信息，包括他们的姓名和平均绩点。

其次，我们将比较这些学生记录，并检查学生的成绩从 2019 年到 2020 年是如何变化的。我们可以使用 unified_diff()函数来实现这一点。下面的例子使用带有语句的**来读取文件数据。通过使用 Python with 语句，我们可以安全地打开和读取文件。**

*student_gpa_2019.txt*
切尔西-沃克 3.3
卡洛琳-贝内特 2.8
加里-霍姆斯 3.7
拉斐尔-罗杰斯 3.6
帕特里克-尼尔森 2.1

*student_gpa_2020.txt*
切尔西-沃克 3.6
卡洛琳-贝内特 2.7
加里-霍姆斯 3.7
拉斐尔-罗杰斯 3.7
帕特里克-尼尔森 2.1

#### **示例:比较学生的 GPA**

```py
import difflib

with open("student_gpa_2019.txt",'r') as file1:
    file1_contents = file1.readlines()
with open("student_gpa_2020.txt",'r') as file2:
    file2_contents = file2.readlines()

diff = difflib.unified_diff(
    file1_contents, file2_contents, fromfile="file1.txt",
    tofile="file2.txt", lineterm='')

for line in diff:
    print(line) 
```

**输出**

```py
--- file1.txt
+++ file2.txt
@@ -1,5 +1,5 @@
-Chelsea        Walker 3.3
-Caroline       Bennett 2.8
+Chelsea        Walker 3.6
+Caroline       Bennett 2.7
 Garry  Holmes 3.7
-Rafael Rogers 3.6
+Rafael Rogers 3.7
 Patrick        Nelson 2.1
```

查看输出，我们可以看到 difflib 模块不仅仅是逐行比较文本文件。unified_diff()函数还提供了一些关于所发现的差异的上下文。

## 比较两个。Python 中的 csv 文件逐行

逗号分隔值文件用于在程序之间交换数据。Python 也提供了处理这些文件的工具。通过使用 **csv** 模块，我们可以快速访问 csv 文件中的数据。

使用 csv 模块，我们将比较两个数据文件并识别不匹配的行。这些文件包含员工记录，包括每个员工的名字、姓氏和电子邮件。这些数据是随机生成的，但我们会假装我们的员工迫切需要我们来完成比较。

*employeesA.csv*
【名】【姓】【电子邮件】
【大卫】【克劳福德】[【电子邮件保护】](/cdn-cgi/l/email-protection)
【莎拉】【佩恩】[【电子邮件保护】](/cdn-cgi/l/email-protection)
【罗伯特】【库珀】[【电子邮件保护】](/cdn-cgi/l/email-protection)
【艾达】【亚历山大】[【电子邮件保护】](/cdn-cgi/l/email-protection)
【瓦莱里娅】【道格拉斯】

*employees sb . CSV*
【名】【姓】【电子邮件】
【安德鲁】【克劳福德】[【电子邮件保护】](/cdn-cgi/l/email-protection)
【莎拉】【佩恩】[【电子邮件保护】](/cdn-cgi/l/email-protection)
【罗伯特】【库珀】[【电子邮件保护】](/cdn-cgi/l/email-protection)
【阿加塔】【安德森】[【电子邮件保护】](/cdn-cgi/l/email-protection)
【麦莉】【福尔摩斯】

一旦我们有了雇员数据，我们就可以使用 reader()函数读取它。包含在 csv 模块中的 reader()函数可以解释 csv 数据。收集完数据后，我们可以使用 Python 将数据转换成列表。

最后，使用 for 循环，我们将比较两个列表的元素。每个元素将保存雇员数据文件中的一行。这样，我们可以遍历列表并发现哪些行不相同。

Python 程序将逐行比较这些文件。因此，我们可以识别雇员数据文件之间的所有差异。

#### **示例:使用 csv 模块比较员工数据文件**

```py
import csv

file1 = open("employeesA.csv",'r')
file2 = open("employeesB.csv",'r')

data_read1= csv.reader(file1)
data_read2 = csv.reader(file2)

# convert the data to a list
data1 = [data for data in data_read1]
data2 = [data for data in data_read2]

for i in range(len(data1)):
    if data1[i] != data2[i]:
        print("Line " + str(i) + " is a mismatch.")
        print(f"{data1[i]} doesn't match {data2[i]}")

file1.close()
file2.close() 
```

**输出**

```py
Line 1 is a mismatch.
['David', 'Crawford', '[[email protected]](/cdn-cgi/l/email-protection)'] doesn't match ['Andrew', 'Crawford', '[[email protected]](/cdn-cgi/l/email-protection)']
Line 4 is a mismatch.
['Aida', 'Alexander', '[[email protected]](/cdn-cgi/l/email-protection)'] doesn't match ['Agata', 'Anderson', '[[email protected]](/cdn-cgi/l/email-protection)']
Line 5 is a mismatch.
['Valeria', 'Douglas', '[[email protected]](/cdn-cgi/l/email-protection)'] doesn't match ['Miley', 'Holmes', '[[email protected]](/cdn-cgi/l/email-protection)'] 
```

## 最后

Python 提供了许多比较两个文本文件的工具，包括 csv 文件。在这篇文章中，我们讨论了 Python 3 中的许多函数和模块。此外，我们已经看到了如何使用它们在 Python 中逐行比较文件。

通过发现新的模块，我们可以编写使我们的生活更容易的程序。我们日常使用的许多程序和 web 应用程序都是由 Python 驱动的。

## 相关职位

*   [如何使用 python split()函数拆分文本](https://www.pythonforbeginners.com/dictionary/python-split)
*   [Python 列表理解及使用方法](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)