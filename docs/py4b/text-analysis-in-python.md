# Python 中的文本分析

> 原文：<https://www.pythonforbeginners.com/basics/text-analysis-in-python>

分析文本数据是在自然语言处理、[机器学习](https://codinginfinite.com/machine-learning-an-introduction/)和相关领域工作的人生活中最常见的任务之一。我们需要找到模式，搜索特定的字符串，[用另一个字符](https://www.pythonforbeginners.com/basics/replace-characters-in-a-string-in-python)替换一个字符，并执行许多这样的任务。本文讨论如何在 Python 中使用正则表达式进行文本分析。本文讨论了各种概念，如正则表达式函数、模式和字符类以及量词。

## Python 中用于文本分析的正则表达式函数

Python 为我们提供了实现和使用正则表达式的 re 模块。re 模块包含各种函数，可以帮助您在 Python 中进行文本分析。让我们逐一讨论这些功能。

### match()函数

`re.match()`函数用于检查字符串是否以某种模式开始。`re.match()`函数将一个模式作为其第一个输入参数，将一个输入字符串作为其第二个输入参数。执行后，如果输入字符串不是以给定的模式开头，它将返回`None`。

如果输入字符串以给定的模式开始，`match()`函数返回一个 match 对象，它包含字符串中模式的跨度。您可以在下面的示例中观察到这一点。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="ACAA"
match_obj=re.match(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("Match object is",match_obj)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: ACAA
Match object is <re.Match object; span=(0, 4), match='ACAA'>
```

在这里，您可以观察到 match 对象包含匹配的字符串以及它在原始字符串中的位置。

您也可以使用`group()`方法在匹配对象中打印图案。在 match 对象上调用`group()`方法时，将从字符串中返回匹配的文本，如下例所示。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="ACAA"
match_obj=re.match(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("Text in match object is",match_obj.group())
print("Span of text is:",match_obj.span())
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: ACAA
Text in match object is ACAA
Span of text is: (0, 4) 
```

在本例中，我们还使用`span()` 方法打印了匹配的子字符串的位置。在 match 对象上调用 `span()`方法时，它返回一个元组，该元组包含原始字符串中匹配子字符串的开始和结束索引。

### search()函数

`search()`函数用于检查模式是否存在于输入字符串中。`search()`函数将一个模式作为其第一个输入参数，将一个字符串作为其第二个输入参数。执行后，如果给定的模式在字符串中不存在，它返回`None`。

如果模式出现在字符串中，`search()`函数返回一个 match 对象，其中包含模式在字符串中第一次出现的位置。您可以在下面的示例中观察到这一点。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="AA"
print("The text is:",text)
print("The pattern is:",pattern)
output=re.search(pattern,text)
print("The output is:",output) 
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: AA
The output is: <re.Match object; span=(2, 4), match='AA'>
```

可以看到`search()`函数是对`match()`函数的升级。但是，它只检查给定模式在字符串中的第一次出现。为了[在一个字符串中找到给定模式](https://www.pythonforbeginners.com/basics/find-all-occurrences-of-a-substring-in-a-string-in-python)的所有出现，我们可以使用`findall()`函数。

### findall()函数

re 模块中的`findall()` 函数用于查找字符串中遵循给定模式的所有子字符串。`findall()`函数将一个模式作为其第一个输入参数，将一个字符串作为其第二个输入参数。执行后，如果给定的模式在字符串中不存在，它将返回一个空列表。

如果输入字符串包含遵循给定模式的子字符串，`findall()`函数将返回一个包含所有匹配给定模式的子字符串的列表。您可以在下面的示例中观察到这一点。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="AA"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output) 
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: AA
The output is: ['AA', 'AA', 'AA', 'AA', 'AA']
```

这里，`findall()`函数返回了原始字符串中所有匹配子字符串的列表。

### finditer()函数

`findall()` 函数返回匹配给定模式的所有子字符串的列表。如果我们想要访问子字符串的匹配对象来找到它们的跨度，我们不能使用`findall()` 函数来完成。为此，您可以使用`finditer()`功能。

`finditer()` 函数将一个 regex 模式作为其第一个输入参数，将一个字符串作为其第二个输入参数。执行后，它返回一个 iterable 对象，该对象包含非重叠模式匹配的 match 对象。您可以在下面的示例中观察到这一点。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="AA"
output=re.finditer(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:")
for match_object in output:
    print("Text:",match_object.group(), "Span:",match_object.span())
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: AA
The output is:
Text: AA Span: (2, 4)
Text: AA Span: (4, 6)
Text: AA Span: (11, 13)
Text: AA Span: (17, 19)
Text: AA Span: (42, 44)
```

在上面的例子中，`finditer()` 函数返回匹配子字符串的匹配对象列表。我们已经使用`group()` 和`span()`方法获得了匹配的子字符串的文本和位置。

### split()函数

re 模块中的`split()`函数用于按照给定的模式将一个字符串分割成子字符串。`split()` 函数将一个模式作为其第一个输入参数，将一个字符串作为其第二个输入参数。执行后，它在发现模式的位置拆分给定的字符串，并返回如下所示的子字符串列表。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="AA"
output=re.split(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: AA
The output is: ['AC', '', 'BCBCB', 'ABAC', 'DABCABDBACABACDBADBCACB', 'BCDDDDCABACBCDA']
```

正如您在上面的示例中所看到的，原始字符串是从出现两个 a 的位置拆分出来的。拆分后，剩余子字符串的列表由`split()`函数返回。

如果模式出现在字符串的开头，那么由`split()`函数返回的列表也包含一个空字符串。您可以在下面的示例中观察到这一点。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="AC"
output=re.split(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: AC
The output is: ['', 'AAAABCBCBAAAB', 'AADABCABDB', 'AB', 'DBADBC', 'BAABCDDDDCAB', 'BCDA'] 
```

如果输入字符串不包含给定的模式，则由`split()`函数返回的列表包含输入字符串作为其唯一的元素，如下例所示。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="AY"
output=re.split(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: AY
The output is: ['ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA']
```

如您所见，输入字符串中没有出现`"AY"`。因此，输入字符串没有被分割，输出列表只包含一个字符串。

### sub()函数

re 模块中的`sub()`函数用于在 Python 中用另一个字符串替换一个子字符串。`sub()`函数的语法如下。

```py
re.sub(old_pattern, new_pattern, input_string)
```

`re.sub()`函数接受三个输入参数。执行后，它将`input_string`中的`old_pattern`替换为`new_pattern`，并返回修改后的字符串，如下例所示。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="A"
replacement="S"
output=re.sub(pattern,replacement,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The new pattern is:",replacement)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: A
The new pattern is: S
The output is: SCSSSSBCBCBSSSBSCSSDSBCSBDBSCSBSCDBSDBCSCBSSBCDDDDCSBSCBCDS
```

到目前为止，我们已经讨论了 Python 中的一些正则表达式函数。对于使用 Python 中正则表达式的文本分析，我们还需要了解一些正则表达式工具来创建分析所需的模式。让我们逐一讨论。

建议阅读:如果你对数据挖掘和数据分析感兴趣，你可以阅读这篇关于使用 Python 中的 sklearn 模块进行 k-means 聚类的文章。

## Python 中用于文本分析的正则表达式模式和字符类

使用 python 执行文本分析时，您可能需要同时检查两个或更多模式。在这种情况下，您可以使用字符类。

### 在 Python 中使用正则表达式匹配单个模式

例如，如果给你一个包含学生成绩的字符串，你需要检查字符串中 A 的数量，你可以使用如下所示的`findall()` 函数和字符 `‘A’` 模式来完成。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="A"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: A
The output is: ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
```

这里，字符`"A"`的所有出现都由`findall()`函数返回。

### 在 Python 中使用正则表达式匹配多个模式

现在，假设您需要检查字符串中有多少个 A 或 B，您不能在`findall()`函数中使用模式`“AB”`。模式`“AB”`匹配紧跟着 b 的 A

为了匹配 A 或 B，我们将使用字符类。因此，我们将使用模式`“[AB]”`。在这里，当我们将`AB`放在模式字符串内的方括号中时，它表现为一个集合。因此，模式匹配 A 或 b。但是，它不会匹配 AB。您可以在下面的示例中观察到这一点。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="[AB]"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: [AB]
The output is: ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'B', 'A', 'B', 'A', 'B', 'A'] 
```

在上面的例子中， `findall()`函数搜索 A 或 B 并返回所有匹配的子字符串。

现在，你可以找到模式“A”、“B”和“AB”。如果你必须找到 A 后面跟着 B 或者 C 呢？换句话说，你必须匹配模式 `“AB”` 或 `“AC”`。在这种情况下，我们可以使用`“[A][BC]”`模式。这里，我们将 A 放在一个单独的方括号中，因为它是一个强制字符。在下面的方括号中，我们把 BC 放了出来，在模式中只考虑其中的一个字符。因此，模式将匹配 AB 以及 AC。您可以在下面的示例中观察到这一点。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="[A][BC]"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: [A][BC]
The output is: ['AC', 'AB', 'AB', 'AC', 'AB', 'AB', 'AC', 'AB', 'AC', 'AC', 'AB', 'AB', 'AC']
```

您也可以使用管道操作符|来创建模式`“AB|AC”`。这里，管道操作符作为 or 操作符工作，模式将匹配 AB 和 AC。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="AB|AC"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: AB|AC
The output is: ['AC', 'AB', 'AB', 'AC', 'AB', 'AB', 'AC', 'AB', 'AC', 'AC', 'AB', 'AB', 'AC']
```

### 使用 Python 中的正则表达式匹配除一个模式之外的所有模式

假设您想要查找输入字符串中除 A 级之外的所有等级。在这种情况下，我们将在字符 A 之前的方括号中引入脱字符，如`“[^A]”`所示。此模式将匹配除 A 之外的所有模式，如下所示。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="[^A]"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: [^A]
The output is: ['C', 'B', 'C', 'B', 'C', 'B', 'B', 'C', 'D', 'B', 'C', 'B', 'D', 'B', 'C', 'B', 'C', 'D', 'B', 'D', 'B', 'C', 'C', 'B', 'B', 'C', 'D', 'D', 'D', 'D', 'C', 'B', 'C', 'B', 'C', 'D']
```

记住模式`“^A”`不会给出相同的结果。当方括号中没有插入符号字符时，这意味着模式应该从紧跟插入符号字符的字符开始。因此，模式`“^A”` 将检查以 a 开头的模式。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="^A"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: ^A
The output is: ['A']
```

## 使用正则表达式量词根据模式中的字符数匹配模式

如果你想匹配两个连续的 A，你可以使用模式`“AA”`或`“[A][A]”`。如果您必须在一个大文本中匹配 100 个连续的 A 会怎么样？在这种情况下，您不能手动创建 100 个字符的模式。在这种情况下，正则表达式量词可以帮助您。

正则表达式量词用于指定模式中连续字符的数量。表示为`pattern{m}`。这里，模式是我们正在寻找的模式，m 是该模式的重复次数。例如，您可以使用量词来表示连续的 A，如下所示。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="A{4}"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: A{4}
The output is: ['AAAA']
```

在上面的例子中，您需要确保花括号中不包含任何空格字符。它应该只包含用作量词的数字。否则，你得不到想要的结果。您可以在下面的示例中观察到这一点。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA"
pattern="A{4 }"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBADBCACBAABCDDDDCABACBCDA
The pattern is: A{4 }
The output is: []
```

现在，假设您想要匹配任何最小为 2 个 A，最大为 6 个 A 的模式。在这种情况下，您可以使用语法模式{m，n}来表示该模式。这里，m 是模式连续出现的下限，n 是上限。

例如，您可以使用模式`“A{2,6}”`匹配 2 到 6 个 A，如下例所示。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBAAAAAAADBCACBAABCDDDDCABACBCDA"
pattern="A{2,6}"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBAAAAAAADBCACBAABCDDDDCABACBCDA
The pattern is: A{2,6}
The output is: ['AAAA', 'AAA', 'AA', 'AAAAAA', 'AA'] 
```

同样，您需要确保括号内没有空格。否则，程序不会给出预期的结果。

你也可以一次重复使用不同的模式。例如，如果您想匹配 2 到 5 A，然后是 1 到 2 B，您可以使用如下所示的模式`“A{2,5}B{1,2}”`。

```py
import re
text="ACAAAABCBCBAAABACAADABCABDBACABACDBAAAAAAADBCACBAABCDDDDCABACBCDA"
pattern="A{2,5}B{1,2}"
output=re.findall(pattern,text)
print("The text is:",text)
print("The pattern is:",pattern)
print("The output is:",output)
```

输出:

```py
The text is: ACAAAABCBCBAAABACAADABCABDBACABACDBAAAAAAADBCACBAABCDDDDCABACBCDA
The pattern is: A{2,5}B{1,2}
The output is: ['AAAAB', 'AAAB', 'AAB'] 
```

## 结论

在本文中，我们讨论了 Python 中用于文本分析的一些正则表达式函数。要了解更多关于 python 中的文本分析，您可以阅读这篇关于[删除字符串](https://www.pythonforbeginners.com/basics/remove-all-occurrences-of-a-character-in-a-list-or-string-in-python)中出现的所有字符的文章。你可能也会喜欢这篇关于用 Python 理解[列表的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)

请继续关注更多内容丰富的文章。

快乐学习！