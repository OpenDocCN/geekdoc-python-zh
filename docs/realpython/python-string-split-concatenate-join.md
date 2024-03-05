# 在 Python 中拆分、连接和连接字符串

> 原文：<https://realpython.com/python-string-split-concatenate-join/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解:[**Python 中的拆分、串联、连接字符串**](/courses/splitting-concatenating-and-joining-strings-python/)

生活中很少有保证:死亡、税收、程序员需要处理字符串。字符串可以有多种形式。它们可以是非结构化的文本、用户名、产品描述、数据库列名，或者任何我们用语言描述的东西。

由于字符串数据几乎无处不在，当涉及到字符串时，掌握交易工具很重要。幸运的是，Python 使得字符串操作变得非常简单，尤其是与其他语言甚至更老版本的 Python 相比。

在本文中，你将学习一些最基本的字符串操作:拆分、连接和连接。你不仅将学会如何使用这些工具，还将对它们的工作原理有更深的理解。

***参加测验:****通过我们的交互式“Python 中字符串的拆分、连接和联接”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-split-strings/)

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 拆分字符串

在 Python 中，字符串被表示为 [`str`](https://docs.python.org/3/library/stdtypes.html#str) 对象，这些对象是**不可变的**:这意味着内存中表示的对象不能被直接改变。这两个事实可以帮助你学习(然后记住)如何使用`.split()`。

您猜到字符串的这两个特性与 Python 中的拆分功能有什么关系了吗？如果您猜测`.split()`是一个 **[实例方法](https://docs.python.org/3/tutorial/classes.html#instance-objects)** ，因为字符串是一种特殊的类型，那么您就猜对了！在其他一些语言中(比如 Perl)，原始字符串作为独立的`.split()`函数的输入，而不是字符串本身调用的方法。

**注意:调用字符串方法的方式**

像`.split()`这样的字符串方法在这里主要显示为在字符串上调用的实例方法。它们也可以被称为静态方法，但这并不理想，因为它更“罗嗦”为了完整起见，这里有一个例子:

```py
# Avoid this:
str.split('a,b,c', ',')
```

与首选用途相比，这种方法体积庞大且不实用:

```py
# Do this instead:
'a,b,c'.split(',')
```

关于 Python 中实例、类和静态方法的更多信息，请查看我们的[深度教程](https://realpython.com/instance-class-and-static-methods-demystified/)。

字符串不变性呢？这应该提醒你，字符串方法不是**就地操作**，而是它们在内存中返回一个*新的*对象。

**注:就地操作**

就地操作是直接改变调用它们的对象的操作。一个常见的例子是在[列表](https://realpython.com/python-lists-tuples/)上使用的 [`.append()`方法](https://realpython.com/python-append/):当你在一个列表上调用`.append()`时，该列表通过将`.append()`的输入添加到同一个列表中而被直接改变。

[*Remove ads*](/account/join/)

### 无参数分割

在深入探讨之前，我们先看一个简单的例子:

>>>

```py
>>> 'this is my string'.split()
['this', 'is', 'my', 'string']
```

这实际上是一个`.split()`调用的特例，我选择它是因为它简单。如果没有指定任何分隔符，`.split()`会将任何空格作为分隔符。

对`.split()`的裸调用的另一个特性是，它会自动删除开头和结尾的空白，以及连续的空白。比较调用以下不带分隔符参数的字符串上的`.split()`和使用`' '`作为分隔符参数的情况:

>>>

```py
>>> s = ' this   is  my string '
>>> s.split()
['this', 'is', 'my', 'string']
>>> s.split(' ')
['', 'this', '', '', 'is', '', 'my', 'string', '']
```

首先要注意的是，这展示了 Python 中字符串的不变性:对`.split()`的后续调用处理原始字符串，而不是第一次调用`.split()`的列表结果。

您应该看到的第二件事——也是最主要的一件事——是简单的`.split()`调用提取了句子中的单词，并丢弃了任何空白。

### 指定分隔符

另一方面,`.split(' ')`,就更加字面了。当有前导或尾随分隔符时，您将得到一个空字符串，您可以在结果列表的第一个和最后一个元素中看到它。

如果有多个连续的分隔符(例如“this”和“is”之间以及“is”和“my”之间)，第一个分隔符将被用作分隔符，随后的分隔符将作为空字符串进入结果列表。

**注:`.split()`** 调用中的分隔符

虽然上面的例子使用单个空格字符作为输入到`.split()`的分隔符，但是您并不局限于用作分隔符的字符类型或字符串长度。唯一的要求是您的分隔符是一个字符串。你可以使用从`"..."`到`"separator"`的任何东西。

### 用最大分割限制分割

`.split()`还有一个可选参数叫做`maxsplit`。默认情况下，`.split()`会在被调用时进行所有可能的拆分。然而，当您给`maxsplit`赋值时，将只进行给定数量的分割。使用我们之前的示例字符串，我们可以看到`maxsplit`在运行:

>>>

```py
>>> s = "this is my string"
>>> s.split(maxsplit=1)
['this', 'is my string']
```

正如您在上面看到的，如果您将`maxsplit`设置为`1`，第一个空白区域被用作分隔符，其余的被忽略。让我们做些练习来检验一下到目前为止我们所学的一切。



给一个负数作为`maxsplit`参数会怎么样？



`.split()`将在所有可用的分隔符上分割字符串，这也是当`maxsplit`未设置时的默认行为。



您最近收到了一个格式非常糟糕的逗号分隔值(CSV)文件。您的工作是将每一行提取到一个列表中，列表中的每个元素代表该文件的列。是什么让它格式不好？“地址”字段包含多个逗号，但需要在列表中表示为单个元素！

假设您的文件已经作为以下多行字符串加载到内存中:

```py
Name,Phone,Address
Mike Smith,15554218841,123 Nice St, Roy, NM, USA
Anita Hernandez,15557789941,425 Sunny St, New York, NY, USA
Guido van Rossum,315558730,Science Park 123, 1098 XG Amsterdam, NL
```

您的输出应该是一个列表列表:

```py
[
    ['Mike Smith', '15554218841', '123 Nice St, Roy, NM, USA'],
    ['Anita Hernandez', '15557789941', '425 Sunny St, New York, NY, USA'],
    ['Guido van Rossum', '315558730', 'Science Park 123, 1098 XG Amsterdam, NL']
]
```

每个内部列表代表我们感兴趣的 CSV 的行，而外部列表将它们放在一起。



这是我的解决方案。有几种方法可以解决这个问题。重要的是，您使用了带有所有可选参数的`.split()`,并获得了预期的输出:

```py
input_string = """Name,Phone,Address
Mike Smith,15554218841,123 Nice St, Roy, NM, USA
Anita Hernandez,15557789941,425 Sunny St, New York, NY, USA
Guido van Rossum,315558730,Science Park 123, 1098 XG Amsterdam, NL"""

def string_split_ex(unsplit):
    results = []

    # Bonus points for using splitlines() here instead, 
    # which will be more readable
    for line in unsplit.split('\n')[1:]:
        results.append(line.split(',', maxsplit=2))

    return results

print(string_split_ex(input_string))
```

我们在这里调用`.split()`两次。第一种用法可能看起来有点吓人，但是不要担心！我们会一步一步来，你会习惯这些表达方式。让我们再来看看第一个`.split()`呼叫:`unsplit.split('\n')[1:]`。

第一个元素是`unsplit`，它只是指向你的输入字符串的[变量](https://realpython.com/python-variables/)。然后我们有我们的`.split()`呼叫:`.split('\n')`。在这里，我们拆分一个叫做**换行符**的特殊字符。

`\n`是做什么的？顾名思义，它告诉正在读取字符串的任何东西，它后面的每个字符都应该显示在下一行。在像我们的`input_string`这样的多行字符串中，每行的末尾都有一个隐藏的`\n`。

最后一部分可能是新的:`[1:]`。到目前为止，这条语句在内存中给了我们一个新的列表，并且`[1:]`看起来像一个列表索引符号，它确实是——有点像！这个扩展的索引符号给了我们一个[列表片](https://www.oreilly.com/learning/how-do-i-use-the-slice-notation-in-python)。在这种情况下，我们获取索引`1`处的元素及其之后的所有内容，丢弃索引`0`处的元素。

总之，我们遍历一个字符串列表，其中每个元素表示多行输入字符串中除第一行之外的每一行。

在每个字符串中，我们使用`,`作为拆分字符再次调用`.split()`，但是这一次我们使用`maxsplit`只拆分前两个逗号，保持地址不变。然后，我们将该调用的结果附加到名副其实的`results`数组中，并将其返回给调用者。

[*Remove ads*](/account/join/)

## 串联和连接字符串

另一个基本的字符串操作与拆分字符串相反:字符串**连接**。如果你没见过这个词，不用担心。这只是“粘在一起”的一种花哨说法

### 用`+`运算符连接

有几种方法可以做到这一点，这取决于你想要达到的目标。最简单也是最常见的方法是使用加号(`+`)将多个字符串加在一起。只需在想要连接的任意多的字符串之间放置一个`+`:

>>>

```py
>>> 'a' + 'b' + 'c'
'abc'
```

为了与数学主题保持一致，您也可以将字符串相乘来重复它:

>>>

```py
>>> 'do' * 2
'dodo'
```

记住，字符串是不可变的！如果您连接或重复存储在变量中的字符串，您必须将新字符串赋给另一个变量才能保留它。

>>>

```py
>>> orig_string = 'Hello'
>>> orig_string + ', world'
'Hello, world'
>>> orig_string
'Hello'
>>> full_sentence = orig_string + ', world'
>>> full_sentence
'Hello, world'
```

如果我们没有不可变的字符串，`full_sentence`将会输出`'Hello, world, world'`。

另一点需要注意的是，Python 不做隐式字符串转换。如果你试图将一个字符串和一个非字符串类型连接起来，Python [会抛出一个`TypeError`](https://realpython.com/python-exceptions/) :

>>>

```py
>>> 'Hello' + 2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: must be str, not int
```

这是因为您只能将字符串与其他字符串连接起来，如果您来自像 JavaScript 这样试图进行隐式类型转换的语言，这可能是一种新的行为。

### 用`.join()` 在 Python 中从列表到字符串

还有另一种更强大的方法将字符串连接在一起。在 Python 中，你可以用`join()`方法从一个列表变成一个字符串。

这里的常见用例是，当您有一个由字符串组成的 iterable(如列表)时，您希望将这些字符串组合成一个字符串。和`.split()`、`.join()`一样，是一个字符串[实例方法](https://realpython.com/instance-class-and-static-methods-demystified/)。如果你所有的字符串都在一个 iterable 中，你在哪个上面调用`.join()`？

这是一个有点棘手的问题。请记住，当您使用`.split()`时，您是在想要拆分的字符串或字符上调用它。相反的操作是`.join()`，所以您可以在想要用来将字符串的 iterable 连接在一起的字符串或字符上调用它:

>>>

```py
>>> strings = ['do', 're', 'mi']
>>> ','.join(strings)
'do,re,mi'
```

在这里，我们用逗号(`,`)连接`strings`列表的每个元素，并在其上调用`.join()`而不是`strings`列表。



如何使输出文本更具可读性？



你可以做的一件事是增加间距:

>>>

```py
>>> strings = ['do', 're', 'mi']
>>> ', '.join(strings)
'do, re, mi'
```

通过在连接字符串中添加一个空格，我们极大地提高了输出的可读性。这是您在连接字符串以提高可读性时应该始终牢记的事情。

`.join()`的聪明之处在于，它将你的“joiner”插入到你想要连接的 iterable 的字符串之间，而不是仅仅在 iterable 的每个字符串的末尾添加 joiner。这意味着如果你传递一个大小为`1`的 iterable，你将看不到你的 joiner:

>>>

```py
>>> 'b'.join(['a'])
'a'
```



使用我们的[网页抓取教程](https://realpython.com/python-web-scraping-practical-introduction/)，你已经建立了一个伟大的天气抓取工具。但是，它在列表的列表中加载字符串信息，每个列表都包含您要写入 CSV 文件的唯一一行信息:

```py
[
    ['Boston', 'MA', '76F', '65% Precip', '0.15 in'],
    ['San Francisco', 'CA', '62F', '20% Precip', '0.00 in'],
    ['Washington', 'DC', '82F', '80% Precip', '0.19 in'],
    ['Miami', 'FL', '79F', '50% Precip', '0.70 in']
]
```

您的输出应该是如下所示的单个字符串:

```py
"""
Boston,MA,76F,65% Precip,0.15in
San Francisco,CA,62F,20% Precip,0.00 in
Washington,DC,82F,80% Precip,0.19 in
Miami,FL,79F,50% Precip,0.70 in
"""
```



对于这个解决方案，我使用了 list comprehension，这是 Python 的一个强大特性，允许您快速构建列表。如果你想学习更多关于它们的知识，可以看看这篇[伟大的文章](https://dbader.org/blog/list-dict-set-comprehensions-in-python)，它涵盖了 Python 中所有可用的理解。

下面是我的解决方案，从一个列表列表开始，以一个字符串结束:

```py
input_list = [
    ['Boston', 'MA', '76F', '65% Precip', '0.15 in'],
    ['San Francisco', 'CA', '62F', '20% Precip', '0.00 in'],
    ['Washington', 'DC', '82F', '80% Precip', '0.19 in'],
    ['Miami', 'FL', '79F', '50% Precip', '0.70 in']
]

# We start with joining each inner list into a single string
joined = [','.join(row) for row in input_list]

# Now we transform the list of strings into a single string
output = '\n'.join(joined)

print(output)
```

这里我们使用`.join()`不是一次，而是两次。首先，我们在列表理解中使用它，它将每个内部列表中的所有字符串组合成一个字符串。接下来，我们用我们之前看到的换行符`\n`连接这些字符串。最后，我们简单地打印结果，这样我们就可以验证它是否如我们所期望的那样。

[*Remove ads*](/account/join/)

## 将所有这些联系在一起

虽然对 Python 中最基本的字符串操作(拆分、连接和连接)的概述到此结束，但仍有大量的字符串方法可以让您更轻松地操作字符串。

一旦你掌握了这些基本的字符串操作，你可能想学习更多。幸运的是，我们有许多很棒的教程来帮助您完全掌握 Python 支持智能字符串操作的特性:

*   [Python 3 的 f-Strings:一种改进的字符串格式化语法](https://realpython.com/python-f-strings/)
*   [Python 字符串格式化最佳实践](https://realpython.com/python-string-formatting/)
*   [Python 中的字符串和字符数据](https://realpython.com/python-strings/)

***参加测验:****通过我们的交互式“Python 中字符串的拆分、连接和联接”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-split-strings/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解:[**Python 中的拆分、串联、连接字符串**](/courses/splitting-concatenating-and-joining-strings-python/)*******