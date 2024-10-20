# Python replace()函数

> 原文：<https://www.askpython.com/python/string/python-replace-function>

在本文中，我们将了解 Python 的****replace()函数**的功能。**

* * *

## Python 用字符串替换()函数

Python 内置了 string.replace()函数，用另一个字符串替换一个字符串的移植。

`string.replace() function`接受要替换的字符串和要用来替换旧字符串的新字符串。

**语法:**

```py
string.replace("old string","new string", count)

```

*   `old string`:要替换的字符串。
*   `new string`:您希望放在旧字符串位置的字符串的新部分。
*   `count`:表示我们希望字符串被新字符串替换的次数。

**示例 1:** 用传递给函数的新字符串替换旧字符串

```py
inp_str = "Python with AskPython"
res = inp_str.replace("AskPython", "AP")
print("Original String:",inp_str)
print("Replaced String:",res)

```

在上面的代码片段中，我们将字符串“AskPython”替换为“AP”。

**输出:**

```py
Original String: Python with AskPython
Replaced String: Python with AP

```

## 替换指定数量的实例

现在让我们使用 count 参数来指定要替换的字符串的实例数。

**示例 2:** 使用`count`作为 replace()函数的参数

```py
inp_str = "abcdaaseweraa"
res = inp_str.replace("a", "x",2)
print("Original String:",inp_str)
print("Replaced String:",res)

```

在本例中，我们将输入字符串作为“abcdaaseweraa”传递。此外，我们传递了原字符串中的字符“a ”,以替换为字符“x”。

这里，计数被设置为 2，即只有前两个遇到的字符“a”将被字符“x”替换。其余遇到的“a”将被取消标记，并保持不变。

**输出:**

```py
Original String: abcdaaseweraa
Replaced String: xbcdxaseweraa

```

* * *

## Python replace()函数与熊猫模块

replace()函数也可以用来替换 csv 或文本文件中的一些字符串。

Python Pandas 模块在处理数据集时非常有用。`pandas.str.replace() function`用于在变量或数据列中用另一个字符串替换一个字符串。

**语法:**

```py
dataframe.str.replace('old string', 'new string')

```

我们将在以下示例中使用以下数据集:

![Input Data Set](img/e9ef7e3a9109a9640822f41355ef6397.png)

**Input Data Set**

**举例:**

```py
import pandas
df = pandas.read_csv("C:/IMDB_data.csv", sep=",",encoding='iso-8859-1')
df['Language']=df['Language'].str.replace("English","Hindi")

```

在上面的代码片段中，`pandas.read_csv() function`用于导入和加载数据集。

从上面的数据集中可以看出，我们选择了“语言”列，以便用“印地语”替换“英语”。

**输出:**

![Output-replace() function](img/7f83802f74d1d6c49c3838bb01e4e94d.png)

**Output-replace() function**

* * *

## 结论

因此，正如我们在上面看到的，Python replace()函数在替换大型数据集的一部分字符串时非常有用。

我强烈建议读者阅读 [Pandas 教程](https://www.askpython.com/python-modules/pandas),进一步了解如何在 Python 中与 CSV 文件交互。

* * *

## 参考

*   JournalDev 上的 Python replace()函数文章