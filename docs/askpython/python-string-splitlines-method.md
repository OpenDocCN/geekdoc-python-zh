# 如何使用 Python 字符串 splitlines()方法

> 原文：<https://www.askpython.com/python/string/python-string-splitlines-method>

## 介绍

今天在本教程中，我们将讨论**Python 字符串 splitlines()方法**。

首先，让我们看看这个方法的基本定义。

## Python 字符串 splitlines()方法

Python [string](https://www.askpython.com/python/string) `splitlines()`是一个内置方法，它返回字符串中的行列表，在行边界处断开。换行符不包括在结果列表中，除非`keepends`被称为**真**。

下面给出了在 Python 中使用`splitlines()`方法的语法。

```py
str.splitlines([keepends])

```

这里，

*   str 是一个字符串对象，我们需要把它分成一系列的行，
*   **保持端点**当设置`True`时，线条边界包含在结果列表元素中。否则，不包括换行符。

线边界字符及其各自的描述如下表所示。

### 线边界表

| 性格；角色；字母 | Python 中的表示 |
| \n | 换行 |
| \r | 回车 |
| \r\n | 回车+换行 |
| \v 或\x0b | 线条制表(Python 3.2 以后) |
| \f 或\x0c | 表单提要(Python 3.2 以后) |
| \x1c | 文件分隔符 |
| \x1d | 组分隔符 |
| \x1e | 记录分离器 |
| \x85 | 下一行(C1 控制代码) |
| **\** u2028 | 行分隔符 |
| \u2029 | 段落分隔符 |

## 在 Python 中使用 splitlines()方法

既然我们已经介绍了 Python 中`splitlines()`方法的基本定义和语法，让我们看一些例子。这有助于我们对题目有一个清晰的理解。

### 没有保留端

如前所述，不提及 keepends 参数将导致创建一个分割线列表**，不包括**换行符或边界字符。

看看下面的例子。

```py
#String initialisation
string1 = "Tim\nCharlie\nJohn\nAlan"
string2 = "Welcome\n\nto\r\nAskPython\t!"
string3 = "Keyboard\u2028Monitor\u2029\x1cMouse\x0cCPU\x85Motherboard\x1eSpeakers\r\nUPS"

#without keepends
print(string1.splitlines())
print(string2.splitlines())
print(string3.splitlines())

```

**输出**:

```py
['Tim', 'Charlie', 'John', 'Alan']
['Welcome', '', 'to', 'AskPython\t!']
['Keyboard', 'Monitor', '', 'Mouse', 'CPU', 'Motherboard', 'Speakers', 'UPS']

```

这里，

*   我们声明了三个字符串，其中包含由不同的换行符分隔的各种单词，
*   我们将它们中的每一个传递给内置的`splitlines()`方法，并将 **keepends** 设置为默认值 **(false)** 。并打印拆分行的结果列表。

从输出中我们可以看到，由于没有设置 **keepends** ，所有被分割的行都不包含行边界或边界字符。对于 string2，`'\t'`包含在单词`'Askpython\t'`中，因为它不是边界字符(不在表中)。

因此，输出是合理的。

### 带保持端

如果我们将**保持端**参数称为`True`，那么现在分裂的线路将**包括**各自的线路断路器。

让我们通过将 Python string `splitlines()`方法中的 **keepends** 参数设置为`True`来修改我们之前的代码(没有 keepends)。仔细查看输出，并尝试注意与前一个输出相比的变化。

```py
#String initialisation
string1 = "Tim\nCharlie\nJohn\nAlan"
string2 = "Welcome\n\nto\r\nAskPython\t!"
string3 = "Keyboard\u2028Monitor\u2029\x1cMouse\x0cCPU\x85Motherboard\x1eSpeakers\r\nUPS"

#with keepends
print(string1.splitlines(keepends=True))
print(string2.splitlines(keepends=True))
print(string3.splitlines(keepends=True))

```

**输出**:

```py
['Tim\n', 'Charlie\n', 'John\n', 'Alan']
['Welcome\n', '\n', 'to\r\n', 'AskPython\t!']
['Keyboard\u2028', 'Monitor\u2029', '\x1c', 'Mouse\x0c', 'CPU\x85', 'Motherboard\x1e', 'Speakers\r\n', 'UPS']

```

正如所料，对于相同的字符串,`splitlines()`输出包括所有的边界字符。

## 结论

因此，在本教程中，我们开始了解内置的 Python string `splitlines()`方法，它做什么以及如何工作。

关于这个话题的任何问题，请在下面的评论中发表。

## 参考

*   关于 [string splitlines()](https://docs.python.org/3/library/stdtypes.html?highlight=splitlines#str.splitlines) 的 Python 文档，
*   Python 字符串 split lines()–日志开发帖子，
*   [如何在新行字符上拆分 python 字符串](https://stackoverflow.com/questions/24237524/how-to-split-a-python-string-on-new-line-characters)–堆栈溢出问题。