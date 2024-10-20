# 如何使用 Python 从文本文件中提取电子邮件

> 原文：<https://www.askpython.com/python/examples/extract-emails-from-text-file>

在这篇文章中，我们将看到如何使用 Python 从文本文件中提取电子邮件。为了使事情更容易使用，我们将使用一些正则表达式。这些是一些特殊的字符方程，它们在很长一段时间里甚至在计算机出现之前就被用于字符串操作。

## 在 Python 中使用正则表达式

术语**正则表达式**在我们需要操作一个字符串并为我们的输出创建一个好的格式时意义重大。**“re”**模块是 Python 中的内置模块。在小节中，我们将看到基本操作，然后转向主题。

### 正则表达式的应用

为了更清楚地了解这一点，下面是一些应用:

1.  *在字符串中寻找特定的模式。*
2.  匹配句子中的特定关键字或字母。
3.  从长文本中提取有用的符号或模式。
4.  *执行复杂的字符串操作。*

## 关于正则表达式 Python 库的小教程

正则表达式允许我们匹配给定文本中的特定模式。因此，为了使事情变得简单，我们将在这个主题中了解它们。不仅用于电子邮件提取，还用于大数据中文本的 **ETL(提取、转换和加载)处理。**

有四个基本函数可以对字符串执行四种基本操作:

1.  *match():匹配文本开头的特定字符串模式。*
2.  *find():在给定文本中查找字符串模式。*
3.  *findall():查找整个文本中所有匹配的字符串。*
4.  *finditer()* : *找到一个匹配的模式，并将其作为一个 iterable 返回。*

### 特殊字符匹配的限制

有一组特殊字符不参与匹配，而是帮助查找字符串中的复杂模式。以下是这些问题的清单:

1.  方括号: **[ ]**
2.  圆括号: **( )**
3.  花括号: **{ }**
4.  管道: **|**
5.  反斜杠: **\**
6.  问号:**？**
7.  加号: **+**
8.  点运算符:**.”**
9.  异或(XOR)运算符: **^**
10.  与符号: **$**
11.  星号或星形运算符: *****

要记住的要点:还要注意，每当匹配一个模式时，我们必须在声明一个字符串之前，用字母“r”将其指定为一个原始字符串。这使得 Python 的 RegEx 引擎避免了任何类型的错误。例如:myPattern = r"myString "。

### 编译正则表达式

开始字符串操作的第一件事是，我们需要将表达式编译到我们的系统中。这将创建一个对象，帮助我们调用上述四个函数。为了编译表达式，我们使用了 **re.compile()** 函数，并将我们的模式插入到该函数中。将旗帜设置为 **re。UNICODE** 。

**代码:**

```py
import re
myPattern = re.compile("python", flags = re.UNICODE)
print(type(myPattern)) 

```

**输出:**

```py
<class 're.Pattern'>

```

现在我们已经成功地创建了一个模式对象。仅使用它，我们将调用函数并执行所有操作。

#### match()函数

如果字符串的起始字符与模式匹配，该函数将创建一个对象。

**代码:**

```py
match = myPattern.match("python")  
print(match.group())

```

**输出:**

```py
python

```

我们可以指定是否调用组函数。因此，当一个模式匹配我们的样本字符串时，对象就被创建了。我们可以使用 **span()** 函数来检查匹配的索引。

```py
print("The pattern matches upto {0}".format(match.span()))

```

```py
The pattern matches upto (0, 6)

```

**请记住，如果函数找不到任何匹配，则不会创建任何对象。**我们得到一个 NoneType 作为返回答案。`match() function`以元组的形式返回匹配的字符串索引位置。它还有两个额外的参数，即:

1.  pos:匹配文本/字符串的起始位置/索引。
2.  endpos:起始文本的结束位置/索引。

**举例:**

```py
match = myPattern.match("hello python", pos = 6)  
print(match.group())
print("The pattern matches upto {0}".format(match.span()))

# output
python
The pattern matches upto (6, 12)

```

### 高级匹配实体

有时我们的字符串可能包含一些数字、数字、空格、字母数字字符等。所以，为了让事情更可靠，re 有一些签名。我们需要在原始字符串中指定它们。

1.  \d:匹配 0 到 9 的整数字符。
2.  \D:匹配从 **0 到 9 的非整数字符。**
3.  \s:对于任何空白字符。 **"\n "、" \t "、" \r"**
4.  \S:对于任何非空白字符。
5.  \w:匹配字母数字字符。
6.  \W:匹配任何非字母数字字符。

#### **匹配功能的标志:**

当我们执行某种复杂的文本分析时，标志证明是一个额外的帮手。下面是一些标志的列表:

1.  **关于。ASCII 或 re。A** :针对所有 ASCII 码字符，如: **\w，\w，\b，\d，\D，\s 和\S .**
2.  **关于。调试**:显示所有的调试信息。
3.  **关于。忽略 CASE 或 re。I** :该标志执行不区分大小写的匹配。
4.  **关于。多线或环形。M** :匹配开始或结束模式后，立即进行换行。

想了解更多关于**旗帜**的信息，请点击此链接:【https://docs.python.org/3/library/re.html#flags】T2

#### search()函数

search 函数在字符串中搜索特定的模式/单词/字母/字符，如果找到该模式，则返回对象。

```py
import re

pattern = r"rain rain come soon, come fast, make the land green";
mySearch = re.search("rain", pattern, re.IGNORECASE))
print("Successfully found, ", mySearch.group(), " from", mySearch.start(), " to ",mySearch.end())

#output
Successfully found "rain"  from 0  to  4

```

## 使用 RegEx 模块提取电子邮件

我们正在学习所有的基础知识，现在是时候迎接更大的挑战了。让我们在一段代码中实现文件读取和正则表达式的知识，并从该文件中提取一些电子邮件地址。

**样本文件:**

```py
Hello my name is Tom the cat.
I like to play and work with my dear friend jerry mouse. 
We both have our office and email addresses also. 
They are [email protected], [email protected] 
Our friend spike has also joined us in our company.
His email address is [email protected] 
We all entertaint the children through our show. 

```

下面是包含三个电子邮件地址的简单文件。这也使事情变得更复杂，但是我们的代码应该使它们更简单。利用正则表达式的上述知识，我们可以很好地实现它。

对此的常规表达是:**【0-9a-zA-z】[【电子邮件保护】](/cdn-cgi/l/email-protection)【0-9a-zA-z】+\。[0-9a-zA-z]+"**

**代码:**

```py
import re

try:
    file = open("data.txt")
    for line in file:
        line = line.strip()
        emails = re.findall("[0-9a-zA-z][email protected][0-9a-zA-z]+\.[0-9a-zA-z]+", line)
        if(len(emails) > 0):
            print(emails)

except FileNotFoundError as e:
    print(e)

```

**说明:**

1.  该模式表示:**提取以字母数字字符开始并带有一个“@”符号的文本，然后它再次带有字母数字字符并带有一个点“.”并且在点号之后，文本再次具有相同类型的字符。**
2.  不要直接取点号，**而是用反斜杠“\”包含它**，指定我们正在使用的 python regex 引擎的点号。照原样使用它将指定我们在模式中采用除换行符之外的每个字符。
3.  然后将示例文本包含在一个文件中。
4.  以阅读模式打开文件。
5.  用行变量实现 for 循环。它读取文本中的每一行。
6.  然后剥去线条，提取文本的每一部分。
7.  创建一个 **findall()** 函数的对象，并在其中包含我们的模式表达式，然后包含 line 变量。这段代码将文本的每一条与模式匹配。
8.  模式匹配后，它就打印出来。
9.  外层代码只是一个处理错误的 try-catch 块。

**输出:**

```py
['[email protected]', '[email protected]']
['[email protected]']

```

## 结论

因此，我们使用几行代码实现了一个智能脚本，从给定的文本中提取电子邮件。