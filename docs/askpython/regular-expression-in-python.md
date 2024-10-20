# Python 中的正则表达式

> 原文：<https://www.askpython.com/python/regular-expression-in-python>

Python 中正则表达式用于匹配模式和字符串。从形式上讲，**正则表达式**是定义搜索模式的一系列字符。Python 正则表达式是匹配文本模式的强大方法。模块`re`，正则表达式的缩写，是为我们提供正则表达式所有特性的 Python 模块。

## 1.使用 Python 的 re 模块

让我们看看 Python re 模块的一些常见例子。是内置的 [Python 模块](https://www.askpython.com/python-modules/python-modules)，我们不需要安装。

### 1.1)重新搜索()

`re.search(pattern, str)`用于在`str`(搜索字符串)中搜索序列`pattern`，它是一个正则表达式，如果找到该模式，则返回一个匹配项。

让我们看一个同样的例子

```py
import re

str = 'This is a sample text which we use to search a pattern within the text.'

pat = r'text'

match = re.search(pat, str)

if match is None:
    print('Pattern not found')
else:
    print('Pattern found!')
    print('Match object', match)

```

输出

```py
Pattern found!
Match object <re.Match object; span=(17, 21), match='text'>

```

如您所见，输出显示确实存在模式匹配，我们在`str`中搜索简单的单词`text`，span 表示匹配的索引。也就是说，`str[17]`到`str[20]`被匹配，对应的是子串`text`，和预期的一样。但这只给出了第一个匹配。

### 1.2)查找所有()

为了给出所有匹配字符串的列表，我们使用`re.findall(pat, str)`返回所有匹配字符串的列表(可以为空)。

```py
>>> matches = re.findall(pat, str)
>>> print(matches)
['text', 'text']

```

`re.findall()`是一个提取模式的非常强大的特性，它可以用于任何东西，比如在文件中搜索。

```py
import re
with open('text.txt', 'r') as f:
    matches = re.findall(r'pattern', f.read())
print(matches)

```

## 2.Python 中正则表达式的规则

在我们进一步讨论之前，我们先来看看正则表达式遵循的某些规则，这些规则是创建模式字符串所必需的。

### 2.1)标识符

这些是模式标识符和每个标识符遵循的规则。

| **图案** | **规则** |
| \d | 匹配任何数字 |
| \D | 匹配除数字以外的任何内容 |
| \s | 匹配单个空格 |
| \S | 匹配除空格以外的任何内容 |
| \w | 匹配任何字母 |
| \W | 匹配除字母以外的任何内容 |
| 。 | 匹配除换行符(\n)以外的任何字符 |
| \. | 匹配句号 |
| \b | 单词周围的空间(单词边界) |

### 2.2)修改器

除了标识符之外，正则表达式还需要遵循某些操作符/修饰符。

| 修改 | **规则** |
| * | 匹配前面的字符/标识符的零次或多次出现 |
| + | 匹配一个或多个事件 |
| ？ | 匹配 0 或 1 次重复/出现 |
| $ | 在字符串末尾执行匹配 |
| ^ | 在字符串开头执行匹配 |
| {1,3} | 如果重复次数在 1 到 3 次之间，则匹配 |
| {3} | 如果重复次数正好是 3 次，则匹配 |
| {3,} | 匹配 3 次或更多次 |
| [a-z] | 匹配从 a 到 z 的任何单个字符 |

下面是一个使用上述规则的例子。

以下模式匹配一个或多个`are`单词，后跟一个空格，其后必须有一个或多个任何字母数字字符、逗号或空格的匹配。下面的匹配在最近的句号处停止，因为它不包括在组中。

```py
import re

str = 'There are 10,000 to 20000 students in the college. This can mean anything.\n'

pat = r'are{1,}\s[a-z0-9,\s]+'

match = re.search(pat, str)
matches = re.findall(pat, str)

if match is None:
    print('Pattern not found')
else:
    print('Pattern found!')
    print('Match object', match)
    print('Listing all matches:', matches)

```

输出

```py
Pattern found!
Match object <re.Match object; span=(6, 49), match='are 10,000 to 20000 students in the college'>
Listing all matches: ['are 10,000 to 20000 students in the college']

```

## 3.结论

我们学习了正则表达式的基础知识，以及如何使用 Python 的`re`模块来实现这一功能，使用正则表达式规则来匹配模式。

## 4.参考

*   关于正则表达式的 pythonprogramming.net 文章:[https://python programming . net/Regular-Expressions-regex-tutorial-python-3/](https://pythonprogramming.net/regular-expressions-regex-tutorial-python-3/)

*   维基百科文章:[https://en.wikipedia.org/wiki/Regular_expression](https://en.wikipedia.org/wiki/Regular_expression)