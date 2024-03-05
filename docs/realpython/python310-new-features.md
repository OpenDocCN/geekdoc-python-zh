# Python 3.10:很酷的新特性供您尝试

> 原文：<https://realpython.com/python310-new-features/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Python 3.10 中很酷的新特性**](/courses/cool-new-features-python-310/)

[Python 3.10 出来了！](https://www.python.org/downloads/release/python-3100/)自 2020 年 5 月以来，志愿者一直致力于新版本的工作，为您带来更好、更快、更安全的 Python。截至[2021 年 10 月 4 日](https://www.python.org/dev/peps/pep-0619/)，第一个正式版本面世。

Python 的每个新版本都带来了大量的变化。你可以在[文档](https://docs.python.org/3.10/whatsnew/3.10.html)中读到所有这些。在这里，您将了解到最酷的新功能。

在本教程中，您将了解到:

*   使用更有用、更精确的**错误消息**进行调试
*   使用**结构模式匹配**处理数据结构
*   添加可读性更强、更具体的**类型提示**
*   使用`zip()`时检查序列的**长度**
*   计算**多变量统计**

要自己尝试新功能，您需要运行 Python 3.10。可以从 [Python 主页](https://www.python.org/downloads/)获取。或者，你可以[使用 Docker](https://realpython.com/python-versions-docker/) 和[最新的 Python 镜像](https://hub.docker.com/_/python/)。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

**额外学习材料:**查看[真实 Python 播客第 81 集](https://realpython.com/podcasts/rpp/81/)，了解 Python 3.10 技巧，并与*真实 Python* 团队成员进行讨论。

## 更好的错误消息

Python 经常被称赞为用户友好的编程语言。虽然这是真的，但 Python 的某些部分可以更友好。Python 3.10 提供了大量更精确、更有建设性的错误消息。在这一部分，您将看到一些最新的改进。完整列表可在[文档](https://docs.python.org/3.10/whatsnew/3.10.html#better-error-messages)中找到。

回想一下用 Python 编写你的第一个 [Hello World](https://www.scriptol.com/programming/hello-world.php) 程序:

```py
# hello.py

print("Hello, World!)
```

也许你创建了一个文件，给`print()`添加了著名的调用，保存为`hello.py`。然后你运行这个程序，渴望称自己是一个真正的 Pythonista。然而，有些事情出错了:

```py
$ python hello.py
 File "/home/rp/hello.py", line 3
 print("Hello, World!)
 ^
SyntaxError: EOL while scanning string literal
```

代码中有一个`SyntaxError`。`EOL`，那到底是什么意思？你回到你的代码，在盯着看了一会儿和搜索了一会儿之后，你意识到在你的字符串的末尾少了一个引号。

Python 3.10 中最具影响力的改进之一是针对许多常见问题的更好、更精确的错误消息。如果您在 Python 3.10 中运行有问题的 Hello World，您将获得比 Python 早期版本更多的帮助:

```py
$ python hello.py
 File "/home/rp/hello.py", line 3
 print("Hello, World!)
 ^
SyntaxError: unterminated string literal (detected at line 3)
```

错误信息仍然有点技术性，但是神秘的`EOL`已经消失了。相反，消息告诉你你需要终止你的字符串！许多不同的错误信息都有类似的改进，您将在下面看到。

一个 [`SyntaxError`](https://realpython.com/invalid-syntax-python/) 是一个错误，当你的代码被解析时，甚至在它开始执行之前。语法错误可能很难调试，因为解释器会提供不精确或者有时甚至是误导性的错误消息。以下代码缺少终止字典的花括号:

```py
 1# unterminated_dict.py
 2
 3months = {
 4    10: "October",
 5    11: "November",
 6    12: "December"
 7
 8print(f"{months[10]} is the tenth month")
```

本应在第 7 行的右花括号丢失是一个错误。如果使用 Python 3.9 或更低版本运行此代码，您将看到以下错误消息:

```py
 File "/home/rp/unterminated_dict.py", line 8
    print(f"{months[10]} is the tenth month")
    ^
SyntaxError: invalid syntax
```

错误消息突出显示了第 8 行，但是第 8 行没有语法问题！如果你经历过 Python 中的语法错误，你可能已经知道诀窍是查看 Python 抱怨的那一行之前的行*。在这种情况下，您要在第 7 行寻找丢失的右大括号。*

在 Python 3.10 中，相同的代码显示了一个更有用、更精确的错误消息:

```py
 File "/home/rp/unterminated_dict.py", line 3
    months = {
             ^
SyntaxError: '{' was never closed
```

这直接将您指向有问题的字典，并允许您立即修复问题。

还有其他一些方法可以弄乱字典语法。一个典型的例子是忘记了其中一项后面的逗号:

```py
 1# missing_comma.py
 2
 3months = {
 4    10: "October" 5    11: "November",
 6    12: "December",
 7}
```

在这段代码中，第 4 行末尾缺少一个逗号。Python 3.10 就如何修复代码给了你一个清晰的建议:

```py
 File "/home/real_python/missing_comma.py", line 4
    10: "October"
        ^^^^^^^^^
SyntaxError: invalid syntax. Perhaps you forgot a comma?
```

您可以添加缺少的逗号，让您的代码立即备份并运行。

另一个常见的错误是在比较值时使用赋值运算符(`=`)而不是等式比较运算符(`==`)。以前，这只会引起另一个`invalid syntax`消息。在 Python 的最新版本中，您会得到更多的建议:

>>>

```py
>>> if month = "October":
  File "<stdin>", line 1
    if month = "October":
       ^^^^^^^^^^^^^^^^^
SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '='?
```

解析器建议您使用[比较运算符](https://realpython.com/python-operators-expressions/#comparison-operators)或[赋值表达式运算符](https://realpython.com/python-walrus-operator/)来代替。

请注意 Python 3.10 错误消息中的另一个漂亮的改进。最后两个例子展示了 carets ( `^^^`)是如何突出整个冒犯性的表达的。以前，单个插入符号(`^`)仅仅表示一个大概的位置。

现在，最后一个错误消息改进是，如果您拼错了属性或名称，属性和名称错误现在可以提供建议:

>>>

```py
>>> import math
>>> math.py
AttributeError: module 'math' has no attribute 'py'. Did you mean: 'pi'? 
>>> pint
NameError: name 'pint' is not defined. Did you mean: 'print'? 
>>> release = "3.10"
>>> relaese
NameError: name 'relaese' is not defined. Did you mean: 'release'?
```

请注意，这些建议对内置名称和您自己定义的名称都有效，尽管它们可能[不是在所有环境中都可用](https://docs.python.org/3.10/whatsnew/3.10.html#attributeerrors)。如果你喜欢这类建议，看看 [BetterErrorMessages](https://github.com/SylvainDe/DidYouMean-Python) ，它提供了更多类似的建议。

您在本节中看到的改进只是许多错误消息的一部分。新的 Python 将比以前更加用户友好，并且希望新的错误消息将节省您的时间和挫折。

[*Remove ads*](/account/join/)

## 结构模式匹配

Python 3.10 中最大的新特性，可能在[争议](https://lwn.net/Articles/845480/)和[潜在影响](https://en.wikipedia.org/wiki/Pattern_matching)方面，是**结构模式匹配**。它的引入有时被称为`switch ... case`来到 Python，但是你会看到结构模式匹配比那要强大得多。

您将看到三个不同的例子，它们共同强调了为什么这个特性被称为结构模式匹配，并向您展示了如何使用这个新特性:

1.  检测和解构数据中不同的**结构**
2.  使用不同种类的**模式**
3.  **匹配**文字模式

结构化模式匹配是对 Python 语言的全面补充。为了让您体验如何在自己的项目中利用它，接下来的三小节将深入探讨一些细节。如果你愿意，你还会看到一些可以帮助你更深入探索的链接。

### 解构数据结构

结构模式匹配的核心是定义数据结构可以匹配的模式。在这一节中，您将学习一个实际的例子，在这个例子中，您将处理结构不同的数据，尽管它们的含义是相同的。您将定义几个模式，根据哪个模式匹配您的数据，您将适当地处理您的数据。

这一节对可能的模式的解释会稍微少一点。相反，它会试图给你一个可能性的印象。下一节将后退一步，更详细地解释这些模式。

是时候匹配你的第一个图案了！下面的例子使用了一个`match ... case`块，通过从一个`user`数据结构中提取名字来查找用户的名字:

>>>

```py
>>> user = {
...     "name": {"first": "Pablo", "last": "Galindo Salgado"},
...     "title": "Python 3.10 release manager",
... }

>>> match user: ...     case {"name": {"first": first_name}}: ...         pass
...

>>> first_name
'Pablo'
```

您可以在突出显示的行中看到工作中的结构模式匹配。`user`是一个有用户信息的小字典。`case`行指定了一个与`user`匹配的模式。在本例中，您正在寻找一个带有`"name"`键的字典，它的值是一个新字典。这个嵌套字典有一个名为`"first"`的键。对应的值被绑定到变量`first_name`。

举个实际的例子，假设您正在处理用户数据，而底层数据模型会随着时间的推移而变化。因此，您需要能够处理同一数据的不同版本。

在下一个例子中，您将使用来自 [randomuser.me](https://randomuser.me) 的数据。这是一个生成随机用户数据的很好的 API，您可以在测试和开发过程中使用它。API 也是一个随着时间而改变的 API 的例子。你仍然可以访问 API 的[旧版本](https://randomuser.me/documentation#previous)。

您可以展开下面折叠的部分，查看如何使用 [`requests`](https://realpython.com/python-requests/) 通过 API 获得不同版本的用户数据:



您可以使用`requests`从 API 中获得一个随机用户，如下所示:

```py
# random_user.py

import requests

def get_user(version="1.3"):
    """Get random users"""
    url = f"https://randomuser.me/api/{version}/?results=1"
    response = requests.get(url)
    if response:
        return response.json()["results"][0]
```

`get_user()`随机获取一个 [JSON](https://realpython.com/python-json/) 格式的用户。注意`version`参数。在早期版本如`"1.1"`和当前版本`"1.3"`之间，返回数据的结构有了很大的变化，但是在每种情况下，实际的用户数据都包含在`"results"`数组内的一个列表中。该函数返回列表中的第一个也是唯一一个用户。

在撰写本文时，API 的最新版本是 1.3，数据具有以下结构:

```py
{ "gender":  "female", "name":  { "title":  "Miss", "first":  "Ilona", "last":  "Jokela" }, "location":  { "street":  { "number":  4473, "name":  "Mannerheimintie" }, "city":  "Harjavalta", "state":  "Ostrobothnia", "country":  "Finland", "postcode":  44879, "coordinates":  { "latitude":  "-6.0321", "longitude":  "123.2213" }, "timezone":  { "offset":  "+5:30", "description":  "Bombay, Calcutta, Madras, New Delhi" } }, "email":  "ilona.jokela@example.com", "login":  { "uuid":  "632b7617-6312-4edf-9c24-d6334a6af52d", "username":  "brownsnake482", "password":  "biatch", "salt":  "ofk518ZW", "md5":  "6d589615ca44f6e583c85d45bf431c54", "sha1":  "cd87c931d579bdff77af96c09e0eea82d1edfc19", "sha256":  "6038ede83d4ce74116faa67fb3b1b2e6f6898e5749b57b5a0312bd46a539214a" }, "dob":  {  "date":  "1957-05-20T08:36:09.083Z",  "age":  64  },   "registered":  { "date":  "2006-07-30T18:39:20.050Z", "age":  15 }, "phone":  "07-369-318", "cell":  "048-284-01-59", "id":  { "name":  "HETU", "value":  "NaNNA204undefined" }, "picture":  { "large":  "https://randomuser.me/api/portraits/women/28.jpg", "medium":  "https://randomuser.me/api/portraits/med/women/28.jpg", "thumbnail":  "https://randomuser.me/api/portraits/thumb/women/28.jpg" }, "nat":  "FI" }
```

在不同版本之间变化的成员之一是`"dob"`，出生日期。注意，在 1.3 版本中，这是一个有两个成员的 JSON 对象，`"date"`和`"age"`。

**注意:**默认情况下， [randomuser.me](https://randomuser.me) 返回一个随机用户。通过将[种子](https://randomuser.me/documentation#seeds)设置为`310`，您可以获得与本例中完全相同的用户:

```py
url = f"https://randomuser.me/api/{version}/?results=1&seed=310"
```

通过将`&seed=310`添加到 URL 来设置种子。API 返回的完整对象还包含一些名为`"info"`的成员中的元数据。这些元数据将包括数据的版本以及用于创建随机用户的种子。

将上面的结果与 1.1 版本的随机用户进行比较:

```py
{ "gender":  "female", "name":  { "title":  "miss", "first":  "ilona", "last":  "jokela" }, "location":  { "street":  "7336 myllypuronkatu", "city":  "kurikka", "state":  "central ostrobothnia", "postcode":  53740 }, "email":  "ilona.jokela@example.com", "login":  { "username":  "blackelephant837", "password":  "sand", "salt":  "yofk518Z", "md5":  "b26367ea967600d679ee3e0b9bda012f", "sha1":  "87d2910595acba5b8e8aa8b00a841bab08580e2f", "sha256":  "73bd0d205d0dc83ae184ae222ff2e9de5ea4039119a962c4f97fabd5bbfa7aca" }, "dob":  "1966-04-17 11:57:01",   "registered":  "2005-08-10 10:15:01", "phone":  "04-636-931", "cell":  "048-828-40-15", "id":  { "name":  "HETU", "value":  "366-9204" }, "picture":  { "large":  "https://randomuser.me/api/portraits/women/24.jpg", "medium":  "https://randomuser.me/api/portraits/med/women/24.jpg", "thumbnail":  "https://randomuser.me/api/portraits/thumb/women/24.jpg" }, "nat":  "FI" }
```

注意，在这个旧格式中，`"dob"`成员的值是一个普通的字符串。

在本例中，您将处理每个用户的出生日期(`dob`)信息。这些数据的结构在不同版本的随机用户 API 之间发生了变化:

```py
#  Version  1.1 "dob":  "1966-04-17 11:57:01" #  Version  1.3 "dob":  {"date":  "1957-05-20T08:36:09.083Z",  "age":  64}
```

注意，在 1.1 版本中，出生日期被表示为一个简单的字符串，而在 1.3 版本中，它是一个 JSON 对象，有两个成员:`"date"`和`"age"`。假设您想要查找一个用户的年龄。根据数据的结构，您可能需要根据出生日期计算年龄，或者查找年龄(如果已经有年龄的话)。

**注意:**`age`的值在下载数据时是准确的。如果存储数据，这个值最终会过时。如果这是一个问题，您应该基于`date`计算当前年龄。

传统上，您会用一个`if`测试来检测数据的结构，可能是基于`"dob"`字段的类型。在 Python 3.10 中，您可以采用不同的方法。现在，您可以使用结构模式匹配来代替:

```py
 1# random_user.py (continued)
 2
 3from datetime import datetime
 4
 5def get_age(user):
 6    """Get the age of a user"""
 7    match user: 8        case {"dob": {"age": int(age)}}: 9            return age
10        case {"dob": dob}: 11            now = datetime.now()
12            dob_date = datetime.strptime(dob, "%Y-%m-%d %H:%M:%S")
13            return now.year - dob_date.year
```

`match ... case`构造是 Python 3.10 中的新特性，也是执行结构化模式匹配的方式。您从一个`match`语句开始，该语句指定了您想要匹配的内容。在这个例子中，这就是`user`数据结构。

一个或几个`case`语句跟在`match`后面。每一个`case`描述了一种模式，它下面的缩进块说明了如果有匹配会发生什么。在本例中:

*   **第 8 行**匹配一个带有`"dob"`键的字典，其值是另一个带有名为`"age"`的整数(`int`)项的字典。`age`这个名字抓住了它的价值。

*   **第 10 行**匹配任何带有`"dob"`键的字典。名字`dob`抓住了它的价值。

模式匹配的一个重要特征是最多匹配一个模式。因为第 10 行的模式匹配任何带有`"dob"`的字典，所以第 8 行更具体的模式排在最前面是很重要的。

**注意:**第 13 行的年龄计算不是很精确，因为它忽略了日期。您可以通过显式比较月份和日期来检查用户今年是否已经庆祝了生日，从而改进这一点。然而，更好的解决方案是使用 [dateutil](https://dateutil.readthedocs.io/) 包中的 [`relativedelta`](https://realpython.com/python-datetime/#doing-arithmetic-with-python-datetime) 。使用`relativedelta`可以直接计算年份。

在仔细研究模式的细节以及它们是如何工作的之前，试着用不同的数据结构调用`get_age()`来看看结果:

>>>

```py
>>> import random_user

>>> users11 = random_user.get_user(version="1.1")
>>> random_user.get_age(users11)
55

>>> users13 = random_user.get_user(version="1.3")
>>> random_user.get_age(users13)
64
```

您的代码可以正确计算两个版本的用户数据的年龄，这两个版本的用户数据具有不同的出生日期。

仔细看看那些图案。第一个模式`{"dob": {"age": int(age)}}`，匹配版本 1.3 的用户数据:

```py
{
    ...
    "dob": {"date": "1957-05-20T08:36:09.083Z", "age": 64},
    ...
}
```

第一种模式是嵌套模式。外面的花括号表示需要一个带有键`"dob"`的字典。对应的值应该是字典。这个嵌套字典必须匹配子模式`{"age": int(age)}`。换句话说，它需要一个整数值的`"age"`键。该值被绑定到名称`age`。

第二种模式`{"dob": dob}`，匹配旧版本 1.1 的用户数据:

```py
{
    ...
    "dob": "1966-04-17 11:57:01",
    ...
}
```

第二种模式比第一种模式简单。同样，花括号表示它将匹配一个字典。但是，任何带有`"dob"`键的字典都会被匹配，因为没有指定其他限制。该键的值被绑定到名称`dob`。

主要的收获是，您可以使用最熟悉的符号来描述数据的结构。然而，一个显著的变化是你可以使用像`dob`和`age`这样的名字，它们还没有被定义。相反，当模式匹配时，来自数据的值被**绑定**到这些名称。

在这个例子中，您已经探索了结构模式匹配的一些功能。在下一节中，您将更深入地了解细节。

[*Remove ads*](/account/join/)

### 使用不同种类的模式

您已经看到了如何使用模式有效地解开复杂数据结构的例子。现在，您将后退一步，看看构成这一新功能的构件。许多事情凑在一起使它起作用。事实上，有三个描述结构化模式匹配的 [Python 增强提案](https://www.python.org/dev/peps/pep-0001/#what-is-a-pep)(pep):

1.  **人教版 634:** [规格](https://www.python.org/dev/peps/pep-0634/)
2.  **PEP 635:** [动机与理](https://www.python.org/dev/peps/pep-0635/)
3.  **PEP 636:** [教程](https://www.python.org/dev/peps/pep-0636/)

如果您对以下内容感兴趣，这些文档将为您提供大量背景和细节。

模式是结构模式匹配的核心。在本节中，您将了解一些不同种类的现有模式:

*   **映射模式**像字典一样匹配映射结构。
*   **序列模式**匹配序列结构，如元组和列表。
*   **捕获模式**将值绑定到名称。
*   **AS 模式**将子模式的值绑定到名称。
*   **OR 模式**匹配几个不同子模式中的一个。
*   **通配符模式**匹配任何内容。
*   **类模式**匹配类结构。
*   **值模式**匹配存储在属性中的值。
*   **文字模式**匹配文字值。

在前一节的示例中，您已经使用了其中的几个。特别是，您使用了**映射模式**来解开存储在字典中的数据。在本节中，您将了解其中一些是如何工作的。所有的细节都可以在上面提到的 PEPs 中找到。

一个**捕获模式**用于捕获一个模式的匹配，并将其绑定到一个名称。考虑下面的[递归](https://realpython.com/python-recursion/)函数，它对一系列数字求和:

```py
 1def sum_list(numbers):
 2    match numbers:
 3        case []:
 4            return 0
 5        case [first, *rest]: 6            return first + sum_list(rest)
```

第 3 行的第一个`case`匹配空列表并返回`0`作为其总和。第 5 行的第二个`case`使用一个**序列模式**和两个捕获模式来匹配带有一个或多个元素的列表。列表中的第一个元素被捕获并绑定到名称`first`。第二种捕获模式`*rest`，使用[解包语法](https://realpython.com/python-kwargs-and-args/#unpacking-with-the-asterisk-operators)来匹配任意数量的元素。`rest`将绑定到包含除第一个元素之外的所有`numbers`元素的列表。

`sum_list()`通过递归相加列表中的第一个数字和其余数字的和来计算数字列表的和。您可以按如下方式使用它:

>>>

```py
>>> sum_list([4, 5, 9, 4])
22
```

4 + 5 + 9 + 4 的和被正确地计算为 22。作为一个练习，您可以尝试跟踪对`sum_list()`的递归调用，以确保您理解代码是如何对整个列表求和的。

**注意:**捕获模式本质上是给变量赋值。然而，一个限制是只允许未被删除的名字。换句话说，您不能使用一个捕获模式来直接分配给一个[类或实例属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)。

`sum_list()`处理对一列数字求和。观察如果你试图对任何不是列表的东西求和会发生什么:

>>>

```py
>>> print(sum_list("4594"))
None

>>> print(sum_list(4594))
None
```

将字符串或数字传递给`sum_list()`会返回`None`。发生这种情况是因为没有匹配的模式，执行在`match`块之后继续。那正好是函数的结尾，所以`sum_list()`隐式[返回`None`](https://realpython.com/python-return-statement/#implicit-return-statements) 。

不过，通常情况下，您希望在匹配失败时得到提醒。例如，您可以添加一个 catchall 模式作为最终案例，通过引发一个错误来处理这个问题。您可以使用下划线(`_`)作为**通配符模式**，它可以匹配任何内容，而不必绑定到名称。您可以向`sum_list()`添加一些错误处理，如下所示:

```py
def sum_list(numbers):
    match numbers:
        case []:
            return 0
        case [first, *rest]:
            return first + sum_list(rest)
 case _:            wrong_type = numbers.__class__.__name__
            raise ValueError(f"Can only sum lists, not {wrong_type!r}")
```

最后的`case`将匹配与前两个模式不匹配的任何内容。这将引发一个描述性错误，例如，如果您试图计算`sum_list(4594)`。当您需要提醒用户某些输入与预期不符时，这很有用。

不过，你的模式仍然不是万无一失的。考虑一下，如果您尝试对一系列字符串求和，会发生什么情况:

>>>

```py
>>> sum_list(["45", "94"])
TypeError: can only concatenate str (not "int") to str
```

基本情况返回`0`，因此求和只适用于可以用数字相加的类型。Python 不知道如何将数字和文本字符串相加。您可以使用**类模式**将您的模式限制为仅匹配整数:

```py
def sum_list(numbers):
    match numbers:
        case []:
            return 0
 case [int(first), *rest]:            return first + sum_list(rest)
        case _:
            raise ValueError(f"Can only sum lists of numbers")
```

在`first`前后添加`int()`可以确保只有值是整数时模式才匹配。不过，这可能限制太多了。您的函数应该能够将[整数](https://realpython.com/python-numbers/#integers)和[浮点数](https://realpython.com/python-numbers/#floating-point-numbers)相加，那么在您的模式中您怎么能允许这样呢？

为了检查几个子模式中是否至少有一个匹配，您可以使用一个**或模式**。OR 模式由两个或更多子模式组成，如果至少有一个子模式匹配，则模式匹配。当第一个元素是类型`int`或类型`float`时，您可以使用它来匹配:

```py
def sum_list(numbers):
    match numbers:
        case []:
            return 0
 case [int(first) | float(first), *rest]:            return first + sum_list(rest)
        case _:
            raise ValueError(f"Can only sum lists of numbers")
```

您可以使用管道符号(`|`)来分隔 OR 模式中的子模式。您的函数现在允许对一系列浮点数求和:

>>>

```py
>>> sum_list([45.94, 46.17, 46.72])
138.82999999999998
```

在结构模式匹配中有很多能力和灵活性，甚至比你目前所看到的还要多。本概述中未涵盖的一些内容包括:

*   使用[防护装置](https://www.python.org/dev/peps/pep-0635/#guards)限制图案
*   使用[作为模式](https://www.python.org/dev/peps/pep-0635/#as-patterns)来捕获子模式的值
*   使用[类模式](https://www.python.org/dev/peps/pep-0636/#matching-positional-attributes)匹配自定义[枚举](https://docs.python.org/3/library/enum.html)和[数据类](https://realpython.com/python-data-classes/)

如果您感兴趣，也可以查看文档以了解更多关于这些特性的信息。在下一节中，您将了解文字模式和值模式。

[*Remove ads*](/account/join/)

### 匹配文字模式

**文字模式**是一种匹配文字对象的模式，比如显式字符串或数字。在某种意义上，这是最基本的一种模式，允许你模仿其他语言中的`switch ... case`语句。以下示例匹配特定的名称:

```py
def greet(name):
    match name:
 case "Guido":            print("Hi, Guido!")
        case _:
            print("Howdy, stranger!")
```

第一个`case`匹配文字字符串`"Guido"`。在这种情况下，只要`name`不是`"Guido"`，就使用`_`作为通配符来打印通用问候。这种文字模式有时可以代替`if ... elif ... else`结构，并且可以扮演与其他一些语言中的`switch ... case`相同的角色。

结构模式匹配的一个限制是不能直接匹配存储在变量中的值。假设您已经定义了`bdfl = "Guido"`。像`case bdfl:`这样的图案不会与`"Guido"`相配。相反，这将被解释为匹配任何内容的捕获模式，并将该值绑定到`bdfl`，有效地覆盖旧值。

但是，您可以使用一个**值模式**来匹配存储的值。值模式看起来有点像捕获模式，但是它使用了一个预先定义的带点的名称，该名称包含将要匹配的值。

**注:**一个**带点的名字**是一个名字里面带一个点(`.`)。实际上，这将引用类的属性、类的实例、枚举或模块。

例如，您可以使用一个[枚举](https://docs.python.org/3/library/enum.html)来创建这样的点名称:

```py
import enum

class Pythonista(str, enum.Enum):
    BDFL = "Guido"
    FLUFL = "Barry"

def greet(name):
    match name:
 case Pythonista.BDFL:            print("Hi, Guido!")
        case _:
            print("Howdy, stranger!")
```

第一种情况现在使用一个值模式来匹配`Pythonista.BDFL`，也就是`"Guido"`。请注意，您可以在值模式中使用任何带点的名称。例如，您可以使用常规类或模块来代替枚举。

要查看如何使用文字模式的更大的例子，考虑一下 FizzBuzz 的游戏。这是一个数数游戏，你应该根据以下规则用单词替换一些数字:

*   你用 **fizz** 代替能被 **3** 整除的数字。
*   你用**蜂音**代替能被 **5** 整除的数字。
*   你用 **fizzbuzz** 替换能被 **3** 和 **5** 整除的数字。

FizzBuzz 有时用于在编程教育中引入条件句，并作为面试中的筛选问题。尽管解决方案很简单，乔尔·格鲁什已经写了一本关于不同游戏编程方式的完整的《T2》书。

Python 中的典型解决方案将如下使用`if ... elif ... else`:

```py
def fizzbuzz(number):
    mod_3 = number % 3
    mod_5 = number % 5

    if mod_3 == 0 and mod_5 == 0:
        return "fizzbuzz"
    elif mod_3 == 0:
        return "fizz"
    elif mod_5 == 0:
        return "buzz"
    else:
        return str(number)
```

[`%`运算符](https://realpython.com/python-modulo-operator/)计算模数，你可以用它来[测试整除性](https://realpython.com/python-modulo-operator/#python-modulo-operator-in-practice)。即如果两个数 *a* 和 *b* 的 *a* 模数 *b* 为 0，那么 *a* 可被 *b* 整除。

在`fizzbuzz()`中，你计算`number % 3`和`number % 5`，然后用它们来测试 3 和 5 的整除性。请注意，您必须首先测试 3 和 5 的整除性。否则，能被 3 和 5 整除的数字将被`"fizz"`或`"buzz"`的情况所覆盖。

您可以检查您的实现是否给出了预期的结果:

>>>

```py
>>> fizzbuzz(3)
fizz

>>> fizzbuzz(14)
14

>>> fizzbuzz(15)
fizzbuzz

>>> fizzbuzz(92)
92

>>> fizzbuzz(65)
buzz
```

你可以自己确认 3 能被 3 整除，65 能被 5 整除，15 能被 3 和 5 整除，而 14 和 92 不能被 3 和 5 整除。

在一个`if ... elif ... else`结构中，你要多次比较一个或几个变量，使用模式匹配来重写是非常简单的。例如，您可以执行以下操作:

```py
def fizzbuzz(number):
    mod_3 = number % 3
    mod_5 = number % 5

    match (mod_3, mod_5):
        case (0, 0):
            return "fizzbuzz"
        case (0, _):
            return "fizz"
        case (_, 0):
            return "buzz"
        case _:
            return str(number)
```

您在`mod_3`和`mod_5`上都匹配。然后，每个`case`模式匹配相应值上的文字数字`0`或通配符`_`。

将这个版本与前一个版本进行比较和对比。注意图案`(0, 0)`如何对应于测试`mod_3 == 0 and mod_5 == 0`，而`(0, _)`如何对应于`mod_3 == 0`。

正如您在前面看到的，您可以使用 OR 模式来匹配几个不同的模式。例如，由于`mod_3`只能取值`0`、`1`和`2`，所以可以用`case (1, 0) | (2, 0)`代替`case (_, 0)`。记住`(0, 0)`已经讲过了。

**注意:**如果你一直在其他语言中使用`switch ... case`，你应该记得在 Python 的模式匹配中没有 [fallthrough](https://en.wikipedia.org/wiki/Switch_statement#Fallthrough) 。这意味着最多会执行一个`case`，即第一个匹配的`case`。这与 C 和 Java 等语言不同。您可以使用或模式来处理大部分失败的效果。

Python 核心开发者[有意识地选择](https://www.python.org/dev/peps/pep-3103/)不在语言中包含`switch ... case`语句。然而，有一些第三方包可以做到，比如 [switchlang](https://pypi.org/project/switchlang/) ，它增加了一个`switch`命令，也适用于早期版本的 Python。

[*Remove ads*](/account/join/)

## 类型联合、别名和保护

可靠地说，每一个新的 Python 版本都会给静态类型系统带来一些改进。Python 3.10 也不例外。事实上，这个新版本附带了四个不同的关于打字的 pep:

1.  **人教版 604:** [允许编写工会类型为`X | Y`](https://www.python.org/dev/peps/pep-0604)
2.  **PEP 613:** [显式类型别名](https://www.python.org/dev/peps/pep-0613)
3.  **PEP 647:** [自定义类型守卫](https://www.python.org/dev/peps/pep-0647/)
4.  **PEP 612:** [参数说明变量](https://www.python.org/dev/peps/pep-0612/)

PEP 604 可能是这些变化中应用最广泛的，但是在这一节中你将得到每个特性的简要概述。

您可以使用**联合类型**来声明一个变量可以有几种不同类型中的一种。例如，您已经能够键入 hint a 函数来计算一组数字、浮点数或整数的平均值，如下所示:

```py
from typing import List, Union

def mean(numbers: List[Union[float, int]]) -> float:
    return sum(numbers) / len(numbers)
```

注释`List[Union[float, int]]`意味着`numbers`应该是一个列表，其中每个元素要么是浮点数，要么是整数。这工作得很好，但是符号有点冗长。另外，你需要从`typing`导入`List`和`Union`。

**注:**`mean()`的实现看起来很简单，但实际上有几个[的死角](https://www.python.org/dev/peps/pep-0450/#rationale)会失败。如果需要计算手段，就用 [`statistics.mean()`](https://docs.python.org/3/library/statistics.html#statistics.mean) 代替。

在 Python 3.10 中，可以用更简洁的`float | int`代替`Union[float, int]`。结合在类型提示中使用`list`而不是`typing.List`的能力，这是 [Python 3.9](https://realpython.com/python39-new-features/#type-hint-lists-and-dictionaries-directly) 引入的。然后，您可以简化代码，同时保留所有类型信息:

```py
def mean(numbers: list[float | int]) -> float:
    return sum(numbers) / len(numbers)
```

现在,`numbers`的注释更容易阅读，并且作为一个额外的好处，你不需要从`typing`导入任何东西。

联合类型的一个特殊情况是当一个变量可以有一个特定的类型或者是`None`。您可以将这样的**可选类型**注释为`Union[None, T]`，或者等效地，为某些类型`T`注释为 [`Optional[T]`](https://realpython.com/python-type-checking/#the-optional-type) 。可选类型没有新的特殊语法，但是可以使用新的联合语法来避免导入`typing.Optional`:

```py
address: str | None
```

在本例中，`address`可以是`None`或字符串。

您还可以在运行时在`isinstance()`或`issubclass()`测试中使用新的联合语法:

>>>

```py
>>> isinstance("mypy", str | int)
True

>>> issubclass(str, int | float | bytes)
False
```

传统上，您使用元组一次测试几种类型——例如，`(str, int)`而不是`str | int`。这种旧语法仍然有效。

**类型别名**允许你快速[定义新的别名](https://realpython.com/python-type-checking/#type-aliases)来代替更复杂的类型声明。例如，假设你用一组花色和等级串和一副牌的列表来表示一张扑克牌。然后一副牌被提示为`list[tuple[str, str]]`。

为了简化类型注释，可以按如下方式定义类型别名:

```py
Card = tuple[str, str]
Deck = list[Card]
```

这通常没问题。然而，类型检查器通常不可能知道这样的语句是类型别名还是普通全局变量的定义。为了帮助类型检查器，或者更确切地说，帮助类型检查器帮助您，您现在可以显式地注释类型别名:

```py
from typing import TypeAlias

Card: TypeAlias = tuple[str, str]
Deck: TypeAlias = list[Card]
```

添加`TypeAlias`注释向类型检查者和任何阅读您代码的人阐明了意图。

**类型守卫**用于缩小联合类型。下面的函数接受一个字符串或`None`，但总是返回一组表示扑克牌的字符串:

```py
def get_ace(suit: str | None) -> tuple[str, str]:
 if suit is None:        suit = "♠"
    return (suit, "A")
```

突出显示的行作为类型保护，静态类型检查器能够意识到`suit`在返回时必然是一个字符串。

目前，类型检查器只能使用[几种不同的构造](https://www.python.org/dev/peps/pep-0647/#motivation)以这种方式缩小联合类型。使用新的 [`typing.TypeGuard`](https://docs.python.org/3.10/library/typing.html#typing.TypeGuard) ，您可以注释自定义函数，这些函数可用于缩小联合类型:

```py
from typing import Any, TypeAlias, TypeGuard

Card: TypeAlias = tuple[str, str]
Deck: TypeAlias = list[Card]

def is_deck_of_cards(obj: Any) -> TypeGuard[Deck]:
    # Return True if obj is a deck of cards, otherwise False
```

根据`obj`是否代表`Deck`对象，`is_deck_of_cards()`应该返回`True`或`False`。然后，您可以使用 guard 函数，类型检查器将能够正确地缩小类型范围:

```py
def get_score(card_or_deck: Card | Deck) -> int:
 if is_deck_of_cards(card_or_deck):        # Calculate score of a deck of cards
    ...
```

在`if`块内部，类型检查器知道`card_or_deck`实际上属于类型`Deck`。详见 [PEP 647](https://www.python.org/dev/peps/pep-0647/) 。

最后一个新的类型化特征是**参数规格变量**，它与[类型变量](https://realpython.com/python-type-checking/#type-variables)相关。考虑一下[装饰师](https://realpython.com/primer-on-python-decorators/)的定义。一般来说，它看起来像下面这样:

```py
import functools
from typing import Any, Callable, TypeVar

R = TypeVar("R")

def decorator(func: Callable[..., R]) -> Callable[..., R]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        ...
    return wrapper
```

注释意味着装饰器返回的函数是一个可调用的函数，带有一些参数和与传递给装饰器的函数相同的返回类型`R`。函数头中的[省略号](https://realpython.com/python-ellipsis/) ( `...`)正确地允许任意数量的参数，并且每个参数可以是任意类型。但是，没有验证返回的 callable 是否与传入的函数具有相同的参数。实际上，这意味着类型检查器不能正确地检查修饰函数。

不幸的是，您不能使用`TypeVar`作为参数，因为您不知道函数将有多少个参数。在 Python 3.10 中，你可以访问 [`ParamSpec`](https://docs.python.org/3.10/library/typing.html#typing.ParamSpec) 来正确地输入提示这些类型的调用。`ParamSpec`的工作方式与`TypeVar`相似，但同时代表几个参数。你可以如下重写你的装饰器来利用`ParamSpec`:

```py
import functools
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def decorator(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        ...
    return wrapper
```

注意，当你注释`wrapper()`时，你也使用`P`。你也可以使用新的 [`typing.Concatenate`](https://docs.python.org/3.10/library/typing.html#typing.Concatenate) 给`ParamSpec`添加类型。详情和示例参见[文档](https://docs.python.org/3.10/library/typing.html)和 [PEP 612](https://www.python.org/dev/peps/pep-0612/) 。

[*Remove ads*](/account/join/)

## 更严格的序列压缩

[`zip()`](https://realpython.com/python-zip-function/) 是 Python 中的[内置函数](https://docs.python.org/3/library/functions.html#built-in-functions)，可以组合多个序列中的元素。Python 3.10 引入了新的`strict`参数，它增加了一个运行时测试来检查所有被压缩的序列是否具有相同的长度。

例如，考虑下面的[乐高](https://www.lego.com/)套装表:

| 名字 | 设定数目 | 片 |
| --- | --- | --- |
| [卢浮宫](https://brickset.com/sets/21024-1/Louvre) | Twenty-one thousand and twenty-four | Six hundred and ninety-five |
| [对角巷](https://brickset.com/sets/75978-1/Diagon-Alley) | Seventy-five thousand nine hundred and seventy-eight | Five thousand five hundred and forty-four |
| [美国宇航局阿波罗土星五号](https://brickset.com/sets/92176-1/NASA-Apollo-Saturn-V) | Ninety-two thousand one hundred and seventy-six | One thousand nine hundred and sixty-nine |
| [千年隼](https://brickset.com/sets/75192-1/Millennium-Falcon) | Seventy-five thousand one hundred and ninety-two | Seven thousand five hundred and forty-one |
| [纽约市](https://brickset.com/sets/21028-1/New-York-City) | Twenty-one thousand and twenty-eight | Five hundred and ninety-eight |

用普通 Python 表示这些数据的一种方法是将每一列作为一个列表。它可能看起来像这样:

>>>

```py
>>> names = ["Louvre", "Diagon Alley", "Saturn V", "Millennium Falcon", "NYC"]
>>> set_numbers = ["21024", "75978", "92176", "75192", "21028"]
>>> num_pieces = [695, 5544, 1969, 7541, 598]
```

请注意，您有三个独立的列表，但是它们的元素之间存在隐式的对应关系。名字(`"Louvre"`)、第一套号(`"21024"`)和第一件数(`695`)都描述了第一套乐高积木。

**注意:** [pandas](https://realpython.com/pandas-dataframe/) 非常适合处理和操作这类表格数据。但是，如果您正在进行较小的计算，您可能不希望在项目中引入如此大的依赖性。

`zip()`可用于并行迭代这三个列表:

>>>

```py
>>> for name, num, pieces in zip(names, set_numbers, num_pieces):
...     print(f"{name} ({num}): {pieces} pieces")
...
Louvre (21024): 695 pieces
Diagon Alley (75978): 5544 pieces
Saturn V (92176): 1969 pieces
Millennium Falcon (75192): 7541 pieces
NYC (21028): 598 pieces
```

请注意每一行是如何从所有三个列表中收集信息并显示某个特定集合的信息的。这是一种非常常见的模式，在许多不同的 Python 代码中使用，包括标准库中的[。](https://www.python.org/dev/peps/pep-0618/#examples)

您还可以添加`list()`来将所有三个列表的内容收集到一个元组的嵌套列表中:

>>>

```py
>>> list(zip(names, set_numbers, num_pieces))
[('Louvre', '21024', 695),
 ('Diagon Alley', '75978', 5544),
 ('Saturn V', '92176', 1969),
 ('Millennium Falcon', '75192', 7541),
 ('NYC', '21028', 598)]
```

请注意嵌套列表与原始表非常相似。

使用`zip()`的负面影响是很容易引入难以发现的细微错误。请注意，如果您的列表中有一项缺失，会发生什么情况:

>>>

```py
>>> set_numbers = ["21024", "75978", "75192", "21028"]  # Saturn V missing

>>> list(zip(names, set_numbers, num_pieces))
[('Louvre', '21024', 695),
 ('Diagon Alley', '75978', 5544),
 ('Saturn V', '75192', 1969),
 ('Millennium Falcon', '21028', 7541)]
```

所有关于纽约市布景的信息都消失了！此外，土星五号和千年隼的设定数字是错误的。如果数据集较大，这种错误可能很难发现。即使你发现有问题，也不容易诊断和解决。

问题是您假设三个列表具有相同数量的元素，并且每个列表中的信息顺序相同。在`set_numbers`被破坏后，这个假设不再成立。

[PEP 618](https://www.python.org/dev/peps/pep-0618/) 为`zip()`引入了一个新的`strict`关键字参数，你可以用它来确认所有的序列都有相同的长度。在您的示例中，它会引发一个错误，提醒您列表已损坏:

>>>

```py
>>> list(zip(names, set_numbers, num_pieces, strict=True)) Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: zip() argument 2 is shorter than argument 1
```

当迭代到达纽约市乐高集合时，第二个参数`set_numbers`已经用尽，而第一个参数`names`中仍有元素。您的代码会因错误而失败，而不是静静地给出错误的结果，您可以采取措施来查找并修复错误。

有些情况下，您希望组合长度不等的序列。展开下面的方框，查看`zip()`和`itertools.zip_longest()`如何处理这些问题:



跟随习语的[将乐高积木分成两个一组:](https://realpython.com/python-itertools/#what-is-itertools-and-why-should-you-use-it)

>>>

```py
>>> num_per_group = 2
>>> list(zip(*[iter(names)] * num_per_group))
[('Louvre', 'Diagon Alley'), ('Saturn V', 'Millennium Falcon')]
```

一共有五套，一个不平均分成对的数。在这种情况下，`zip()`的默认行为，即最后一个元素被删除，可能是有意义的。您也可以在这里使用`strict=True`,但是当您的列表不能被分成对时，这将会产生一个错误。第三个选项，在这种情况下可能是最好的，是使用来自 [`itertools`](https://realpython.com/python-itertools/) 标准库中的 [`zip_longest()`](https://docs.python.org/3/library/itertools.html#itertools.zip_longest) 。

顾名思义，`zip_longest()`组合序列，直到最长的序列用完。如果你用`zip_longest()`来划分乐高积木，纽约市没有任何配对就变得更加明显了:

>>>

```py
>>> from itertools import zip_longest

>>> list(zip_longest(*[iter(names)] * num_per_group, fillvalue=""))
[('Louvre', 'Diagon Alley'),
 ('Saturn V', 'Millennium Falcon'),
 ('NYC', '')]
```

注意，`'NYC'`和一个空字符串一起出现在最后一个元组中。您可以使用`fillvalue`参数控制为缺失值填充什么。

虽然`strict`并没有给`zip()`增加任何新的功能，但是它可以帮助你避免那些难以发现的错误。

[*Remove ads*](/account/join/)

## `statistics`模块中的新功能

随着 2014 年 [Python 3.4](https://www.python.org/downloads/release/python-340/) 的发布， [`statistics`](https://docs.python.org/3/library/statistics.html) 模块被添加到标准库中。`statistics`的目的是使[的统计计算](https://realpython.com/python-statistics/)达到 Python 中图形计算器的[级别。](https://www.python.org/dev/peps/pep-0450/)

**注意:** `statistics`不是为了提供专用的数值数据类型或全功能的统计建模而设计的。如果标准库不能满足您的需求，可以看看第三方包，如 [NumPy](https://realpython.com/numpy-tutorial/) 、 [SciPy](https://docs.scipy.org/doc/scipy/reference/stats.html) 、 [pandas](https://realpython.com/pandas-python-explore-dataset/) 、 [statsmodels](https://www.statsmodels.org/) 、 [PyMC3](http://docs.pymc.io/) 、 [scikit-learn](https://scikit-learn.org/) 或 [seaborn](https://seaborn.pydata.org/) 。

Python 3.10 为`statistics`增加了一些多变量函数:

*   **`correlation()`** 计算两个变量的皮尔逊[相关](https://realpython.com/numpy-scipy-pandas-correlation-python/)系数
*   **`covariance()`** 计算样本[协方差](https://en.wikipedia.org/wiki/Covariance)为两个变量
*   **`linear_regression()`** 计算斜率和截距[进行线性回归](https://realpython.com/linear-regression-in-python/)

您可以使用每个函数来描述两个变量之间关系的某个方面。例如，假设您有一组博客文章的数据，即每篇博客文章的字数和每篇文章在一段时间内的浏览量:

>>>

```py
>>> words = [7742, 11539, 16898, 13447, 4608, 6628, 2683, 6156, 2623, 6948]
>>> views = [8368, 5901, 3978, 3329, 2611, 2096, 1515, 1177, 814, 467]
```

你现在想调查字数和浏览量之间是否有任何(线性)关系。在 Python 3.10 中，可以用新的 [`correlation()`](https://docs.python.org/3.10/library/statistics.html#statistics.correlation) 函数计算`words`和`views`之间的**相关性**:

>>>

```py
>>> import statistics

>>> statistics.correlation(words, views)
0.454180067865917
```

两个变量之间的相关性总是一个介于-1 和 1 之间的数。如果接近 0，那么它们之间几乎没有对应关系，而接近-1 或 1 的相关性表明这两个变量的行为倾向于相互跟随。在这个例子中，0.45 的相关性表明有一种趋势，即有更多单词的帖子有更多的浏览量，尽管这不是一个强有力的联系。

**注意:**俗话说[相关性并不意味着因果关系](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation)记住这一点很重要。即使你发现两个变量密切相关，你[也不能断定](https://tylervigen.com/spurious-correlations)一个是另一个的原因。

还可以计算出`words`和`views`之间的**协方差**。协方差是两个变量之间联合可变性的另一个度量。可以用 [`covariance()`](https://docs.python.org/3.10/library/statistics.html#statistics.covariance) 来计算:

>>>

```py
>>> import statistics

>>> statistics.covariance(words, views)
5292289.977777777
```

与相关性相反，协方差是一个绝对度量。它应该在变量本身的可变性的背景下进行解释。实际上，你可以通过每个变量的[标准差](https://en.wikipedia.org/wiki/Standard_deviation)来归一化协方差，以恢复[皮尔逊相关系数](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient):

>>>

```py
>>> import statistics

>>> cov = statistics.covariance(words, views)
>>> σ_words, σ_views = statistics.stdev(words), statistics.stdev(views)
>>> cov / (σ_words * σ_views)
0.454180067865917
```

请注意，这与您之前的相关系数完全匹配。

查看两个变量之间线性对应关系的第三种方式是通过简单的线性回归。你通过计算两个数字*斜率*和*截距*来做[线性回归](https://en.wikipedia.org/wiki/Simple_linear_regression)，这样(平方)误差在近似*视图数量* = *斜率* × *字数* + *截距*中最小化。

在 Python 3.10 中，可以使用 [`linear_regression()`](https://docs.python.org/3.10/library/statistics.html#statistics.linear_regression) :

>>>

```py
>>> import statistics

>>> statistics.linear_regression(words, views)
LinearRegression(slope=0.2424443064354672, intercept=1103.6954940247645)
```

基于这种回归，一篇 10，074 字的帖子预计会有大约 0.2424 × 10074 + 1104 = 3546 次浏览。但是，正如你之前看到的，字数和浏览量之间的相关性相当弱。因此，你不应该期望这个预测非常准确。

`LinearRegression`对象是一个名为元组的[。这意味着您可以解开斜率并直接截取:](https://realpython.com/python-namedtuple/)

>>>

```py
>>> import statistics

>>> slope, intercept = statistics.linear_regression(words, views)
>>> slope * 10074 + intercept
3546.0794370556605
```

在这里，您使用`slope`和`intercept`来预测一篇 10，074 个单词的博客帖子的浏览量。

如果你做大量的统计分析，你仍然想使用一些更高级的包，比如 pandas 和 statsmodels。然而，随着 Python 3.10 中对`statistics`的新添加，您有机会更容易地进行基本分析，而无需引入第三方依赖。

[*Remove ads*](/account/join/)

## 其他非常酷的功能

到目前为止，您已经看到了 Python 3.10 中最大、最有影响力的新特性。在这一节中，您将看到新版本带来的其他一些变化。如果你对这个新版本的所有变化感到好奇，可以查看一下[文档](https://docs.python.org/3.10/whatsnew/3.10.html)。

### 默认文本编码

打开文本文件时，用于解释字符的默认编码取决于系统。特别是使用了 [`locale.getpreferredencoding()`](https://docs.python.org/3.10/library/locale.html#locale.getpreferredencoding) 。在 Mac 和 Linux 上，这通常会返回`"UTF-8"`，而在 Windows 上的结果更加多样。

因此，当您尝试打开文本文件时，应该始终指定一种编码:

```py
with open("some_file.txt", mode="r", encoding="utf-8") as file:
    ...  # Do something with file
```

如果没有明确指定编码，将使用首选的区域设置编码，并且您可能会遇到在一台计算机上可以读取的文件在另一台计算机上无法打开的情况。

Python 3.7 引入了 [UTF-8 模式](https://docs.python.org/3.10/library/os.html#utf8-mode)，它允许你强制你的程序使用独立于地区编码的 UTF-8 编码。您可以通过给`python`可执行文件提供`-X utf8`命令行选项或者通过设置`PYTHONUTF8`环境变量来启用 UTF-8 模式。

在 Python 3.10 中，您可以激活一个警告，在没有指定编码的情况下打开一个文本文件时向您发出警告。考虑下面的脚本，它没有指定编码:

```py
# mirror.py

import pathlib
import sys

def mirror_file(filename):
 for line in pathlib.Path(filename).open(mode="r"):        print(f"{line.rstrip()[::-1]:>72}")

if __name__ == "__main__":
    for filename in sys.argv[1:]:
        mirror_file(filename)
```

该程序将一个或多个文本文件回显到控制台，但每一行都是相反的。在[编码警告](https://docs.python.org/3.10/library/io.html#io-encoding-warning)启用的情况下运行程序本身:

```py
$ python -X warn_default_encoding mirror.py mirror.py
/home/rp/mirror.py:7: EncodingWarning: 'encoding' argument not specified
 for line in pathlib.Path(filename).open(mode="r"): yp.rorrim #

 bilhtap tropmi
 sys tropmi

 :)emanelif(elif_rorrim fed
 :)"r"=edom(nepo.)emanelif(htaP.bilhtap ni enil rof
 )"}27>:]1-::[)(pirtsr.enil{"f(tnirp

 :"__niam__" == __eman__ fi
 :]:1[vgra.sys ni emanelif rof
 )emanelif(elif_rorrim
```

注意印在控制台上的`EncodingWarning`。命令行选项`-X warn_default_encoding`激活它。如果您在打开文件时指定了编码，例如`encoding="utf-8"`，警告将会消失。

有时您希望使用用户定义的本地编码。您仍然可以通过显式使用`encoding="locale"`来这样做。然而，建议尽可能使用 UTF-8。你可以查看 [PEP 597](https://www.python.org/dev/peps/pep-0597/) 了解更多信息。

### 异步迭代

[异步编程](https://realpython.com/python-async-features/)是一个强大的编程范例，从版本 3.5 开始，Python [就提供了这种编程范例。你可以通过使用`async`关键字或者](https://www.python.org/dev/peps/pep-0492/)[以`.__a`](https://www.python.org/dev/peps/pep-0492/#why-magic-methods-start-with-a) 开始的[特殊方法](https://docs.python.org/3/reference/datamodel.html#special-method-names)来识别一个异步程序，比如 [`.__aiter__()`](https://docs.python.org/3/reference/datamodel.html#object.__aiter__) 或者 [`.__aenter__()`](https://docs.python.org/3/reference/datamodel.html#object.__aenter__) 。

Python 3.10 中新增了两个异步[内置函数](https://docs.python.org/3/library/functions.html#built-in-functions):[`aiter()`](https://docs.python.org/3.10/library/functions.html#aiter)和 [`anext()`](https://docs.python.org/3.10/library/functions.html#anext) 。实际上，这些函数调用了`.__aiter__()`和`.__anext__()`特殊方法——类似于常规的`iter()`和`next()`——所以没有添加新功能。这些都是方便的函数，使您的代码更具可读性。

换句话说，在最新版本的 Python 中，以下语句——其中`things`是一个[异步可迭代](https://www.python.org/dev/peps/pep-0492/#asynchronous-iterators-and-async-for)——是等价的:

>>>

```py
>>> it = things.__aiter__()
>>> it = aiter(things)
```

无论哪种情况，`it`最终都是一个异步迭代器。展开下面的方框，查看使用`aiter()`和`anext()`的完整示例:



下面的程序计算几个文件中的行数。在实践中，您使用 Python 迭代文件的能力来计算行数。该脚本使用异步迭代来同时处理几个文件。

注意，运行这段代码之前需要安装带有 [`pip`](https://realpython.com/what-is-pip/) 的第三方 [`aiofiles`](https://pypi.org/project/aiofiles/) 包:

```py
# line_count.py

import asyncio
import sys
import aiofiles

async def count_lines(filename):
    """Count the number of lines in the given file"""
    num_lines = 0

    async with aiofiles.open(filename, mode="r") as file:
 lines = aiter(file)        while True:
            try:
 await anext(lines)                num_lines += 1
            except StopAsyncIteration:
                break

    print(f"{filename}: {num_lines}")

async def count_all_files(filenames):
    """Asynchronously count lines in all files"""
    tasks = [asyncio.create_task(count_lines(f)) for f in filenames]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(count_all_files(filenames=sys.argv[1:]))
```

`asyncio`用于为每个文件名创建并运行一个异步任务。`count_lines()`异步打开一个文件，并使用`aiter()`和`anext()`遍历它，以计算行数。

参见 [PEP 525](https://www.python.org/dev/peps/pep-0525/) 了解更多关于异步迭代的信息。

[*Remove ads*](/account/join/)

### 上下文管理器语法

[上下文管理器](https://realpython.com/python-with-statement/)非常适合管理你的程序中的资源。然而，直到最近，它们的语法还包含了一个不常见的赘疣。你[没有被允许](https://www.python.org/dev/peps/pep-0617/#some-rules-are-not-actually-ll-1)使用括号来打断长`with`语句，就像这样:

```py
with (
    read_path.open(mode="r", encoding="utf-8") as read_file,
    write_path.open(mode="w", encoding="utf-8") as write_file,
):
    ...
```

在 Python 的早期版本中，这会导致一个`invalid syntax`错误消息。相反，如果您想要控制换行的位置，您需要使用反斜杠(`\`):

```py
with read_path.open(mode="r", encoding="utf-8") as read_file, \
     write_path.open(mode="w", encoding="utf-8") as write_file:
    ...
```

虽然在 Python 中带反斜杠的[显式行延续](https://realpython.com/python-program-structure/#explicit-line-continuation)是可能的，但 PEP 8 [不鼓励它](https://www.python.org/dev/peps/pep-0008/#maximum-line-length)。[黑色](https://black.readthedocs.io/)格式化工具[完全避免了](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)反斜杠。

在 Python 3.10 中，现在允许在`with`语句周围添加括号，以满足您的需求。特别是如果你同时使用几个上下文管理器，就像上面的例子一样，这有助于提高代码的可读性。Python 的[文档](https://docs.python.org/3.10/whatsnew/3.10.html#parenthesized-context-managers)展示了这种新语法的一些其他可能性。

一个小的**有趣的事实**:带括号的`with`语句实际上在 [CPython](https://realpython.com/cpython-source-code-guide/) 的 3.9 版本中工作。随着 [Python 3.9](https://realpython.com/python39-new-features/) 中 [PEG 解析器](https://realpython.com/python39-new-features/#a-more-powerful-python-parser)的引入，它们的实现几乎是免费的。之所以称之为 Python 3.10 特性，是因为在 Python 3.9 中使用 PEG 解析器是自愿的，而 Python 3.9 和旧的 LL(1)解析器不支持带括号的`with`语句。

### 现代安全的 SSL

安全性很有挑战性！一个很好的经验法则是避免使用自己的安全算法，而是依赖已有的包。

Python 针对 [`hashlib`](https://docs.python.org/3/library/hashlib.html) 、 [`hmac`](https://docs.python.org/3/library/hmac.html) 、 [`ssl`](https://docs.python.org/3/library/ssl.html) 标准库模块中暴露的不同密码特性，使用 [OpenSSL](https://www.openssl.org/) 。您的系统可以管理 OpenSSL，或者 Python 安装程序可以包含 OpenSSL。

Python 3.9 支持使用任何 [OpenSSL 版本](https://en.wikipedia.org/wiki/OpenSSL#Major_version_releases) 1.0.2 LTS、1.1.0 和 1.1.1 LTS。OpenSSL 1.0.2 LTS 和 OpenSSL 1.1.0 都已过期，因此 Python 3.10 将只支持 OpenSSL 1.1.1 LTS，如下表所示:

| openssl 版本 | Python 3.9 | Python 3.10 | 寿命结束 |
| --- | --- | --- | --- |
| LTS | ✔ | 一千 | 2019 年 12 月 20 日 |
| 1.1.0 | ✔ | 一千 | 2019 年 9 月 10 日 |
| LTS | ✔ | ✔ | 2023 年 9 月 11 日 |

这种对旧版本支持的终止只会影响到您，如果您需要在旧操作系统上升级系统 Python 的话。如果你使用 macOS 或者 Windows，或者如果你安装来自 python.org 的 Python 或者使用 T2 的 Conda，你将不会看到任何变化。

不过， [Ubuntu](https://ubuntu.com/) 18.04 LTS 用的是 OpenSSL 1.1.0，而[红帽企业版 Linux](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux) (RHEL) 7 和 [CentOS](https://www.centos.org/) 7 都用的是 OpenSSL 1.0.2 LTS。如果你需要在这些系统上运行 Python 3.10，你应该考虑使用[python.org](https://www.python.org)或 Conda 安装程序自己安装。

放弃对旧版本 OpenSSL 的支持将使 Python 更加安全。这也将有助于 Python 开发人员，因为代码将更容易维护。最终，这将有助于您，因为您的 Python 体验将更加健壮。详见 [PEP 644](https://www.python.org/dev/peps/pep-0644/) 。

### 关于你的 Python 解释器的更多信息

[`sys`](https://docs.python.org/3/library/sys.html) 模块包含许多关于您的系统、当前 Python 运行时和当前正在执行的脚本的信息。比如，你可以用 [`sys.path`](https://docs.python.org/3/library/sys.html#sys.path) 查询 [Python 寻找模块](https://realpython.com/python-import/#pythons-import-path)的路径，用 [`sys.modules`](https://docs.python.org/3/library/sys.html#sys.modules) 查看[在当前会话中已经导入](https://realpython.com/python-import/#import-internals)的所有模块。

在 Python 3.10 中，`sys`有两个新属性。首先，您现在可以获得标准库中所有模块的名称列表:

>>>

```py
>>> import sys

>>> len(sys.stdlib_module_names)
302

>>> sorted(sys.stdlib_module_names)[-5:]
['zipapp', 'zipfile', 'zipimport', 'zlib', 'zoneinfo']
```

在这里，您可以看到标准库中大约有 300 个模块，其中几个以字母`z`开头。请注意，只列出了顶级模块和包。像 [`importlib.metadata`](https://realpython.com/python38-new-features/#importlibmetadata) 这样的子包没有单独的条目。

你可能不会经常使用 [`sys.stdlib_module_names`](https://docs.python.org/3.10/library/sys.html#sys.stdlib_module_names) 。尽管如此，这个列表与类似的自省特性很好地结合在一起，比如 [`keyword.kwlist`](https://docs.python.org/3/library/keyword.html#keyword.kwlist) 和 [`sys.builtin_module_names`](https://docs.python.org/3/library/sys.html#sys.builtin_module_names) 。

新属性的一个可能的用例是确定当前导入的哪些模块是第三方依赖项:

>>>

```py
>>> import pandas as pd
>>> import sys

>>> {m for m in sys.modules if "." not in m} - sys.stdlib_module_names
{'__main__', 'numpy', '_cython_0_29_24', 'dateutil', 'pytz',
 'six', 'pandas', 'cython_runtime'}
```

您可以通过查看`sys.modules`中名称中没有点的名字来找到导入的顶级模块。通过将它们与标准库模块名称进行比较，您会发现 [`numpy`](https://realpython.com/numpy-array-programming/) 、 [`dateutil`](https://realpython.com/python-packages/#dateutil-for-working-with-dates-and-times) 、 [`pandas`](https://realpython.com/python-pandas-tricks/) 是本例中导入的一些第三方模块。

另一个新属性是 [`sys.orig_argv`](https://docs.python.org/3.10/library/sys.html#sys.orig_argv) 。这与 [`sys.argv`](https://docs.python.org/3/library/sys.html#sys.argv) 有关，它保存了在程序启动时赋予它的[命令行参数](https://realpython.com/python-command-line-arguments/#the-sysargv-array)。相比之下，`sys.orig_argv`列出了传递给`python`可执行文件本身的命令行参数。考虑下面的例子:

```py
# argvs.py

import sys

print(f"argv: {sys.argv}")
print(f"orig_argv: {sys.orig_argv}")
```

这个脚本回显了`orig_argv`和`argv`列表。运行它以查看信息是如何捕获的:

```py
$ python -X utf8 -O argvs.py 3.10 --upgrade
argv: ['argvs.py', '3.10', '--upgrade']
orig_argv: ['python', '-X', 'utf8', '-O', 'argvs.py', '3.10', '--upgrade']
```

本质上，所有参数——包括 Python 可执行文件的名称——都以`orig_argv`结尾。这与`argv`相反，后者只包含不被`python`本身处理的参数。

同样，这个特性你不会经常用到。如果你的程序需要关心它是如何运行的，你通常最好依靠已经公开的信息，而不是试图解析这个列表。例如，只有当您的脚本没有使用优化标志`-O`运行时，您才可以选择使用[严格`zip()`模式](#stricter-zipping-of-sequences)，如下所示:

```py
list(zip(names, set_numbers, num_pieces, strict=__debug__))
```

当解释器启动时，设置 [`__debug__`](https://docs.python.org/3/library/constants.html#__debug__) 标志。如果指定了 [`-O`](https://docs.python.org/3/using/cmdline.html#cmdoption-o) 或 [`-OO`](https://docs.python.org/3/using/cmdline.html#cmdoption-oo) 运行`python`，则为`False`，否则为`True`。使用`__debug__`通常比`"-O" not in sys.orig_argv`或一些类似的构造更好。

对于`sys.orig_argv`来说,[的一个激励用例](https://bugs.python.org/issue23427#msg371028)是，你可以用它来生成一个新的 Python 进程，其命令行参数与你当前的进程相同或有所修改。

[*Remove ads*](/account/join/)

### 未来注释

[注解](https://realpython.com/python-type-checking/#annotations)是在 Python 3 中引入的，为您提供了一种将元数据附加到变量、函数参数和返回值的方法。它们最常用于向代码中添加类型提示。

注释的一个挑战是它们必须是有效的 Python 代码。首先，这使得[很难键入提示](https://realpython.com/python37-new-features/#typing-enhancements)递归类。 [PEP 563](https://www.python.org/dev/peps/pep-0563/) 引入了[推迟注释评估](https://realpython.com/python-news-april-2021/#pep-563-pep-649-and-the-future-of-python-type-annotations)，使得用尚未定义的名字进行注释成为可能。从 Python 3.7 开始，您可以使用 [`__future__`](https://docs.python.org/3/library/__future__.html) 导入来激活注释的延迟求值:

```py
from __future__ import annotations
```

其意图是推迟评估将在将来的某个时候成为默认。在 [2020 Python 语言峰会](https://pyfound.blogspot.com/2020/04/the-2020-python-language-summit.html)之后，决定在 Python 3.10 中实现这一点。

然而，在更多的测试之后，很明显，延迟评估对于在运行时使用注释的项目来说效果不好。FastAPI 和 T2 Pydantic 和 T4 的关键人物表达了他们的担忧。在最后一刻，我们决定为 Python 3.11 重新安排这些更改。

为了简化向未来行为的过渡，Python 3.10 也做了一些改变。最重要的是，新增了 [`inspect.get_annotations()`](https://docs.python.org/3.10/library/inspect.html#inspect.get_annotations) 功能。您应该调用这个函数在运行时访问注释:

>>>

```py
>>> import inspect

>>> def mean(numbers: list[int | float]) -> float:
...     return sum(numbers) / len(numbers)
...

>>> inspect.get_annotations(mean)
{'numbers': list[int | float], 'return': <class 'float'>}
```

查看[注解最佳实践](https://docs.python.org/3.10/howto/annotations.html)了解详情。

## 如何在运行时检测 Python 3.10

Python 3.10 是 Python 的第一个版本，拥有两位数的次要版本号。虽然这主要是一个有趣的事实，并表明 Python 3 已经存在了相当长的时间，但它也有一些实际的后果。

当你的代码需要在运行时基于 Python 的版本做一些特定的事情时，到目前为止，你已经完成了版本字符串的[字典式的](https://docs.python.org/3/reference/expressions.html#value-comparisons)比较。虽然这从来都不是好的做法，但还是有可能做到以下几点:

```py
# bad_version_check.py

import sys

# Don't do the following
if sys.version < "3.6":
    raise SystemExit("Only Python 3.6 and above is supported")
```

在 Python 3.10 中，这段代码会引发`SystemExit`并停止你的程序。这是因为，作为字符串，`"3.10"`小于`"3.6"`。

比较版本号的正确方法是使用数字元组:

```py
# good_version_check.py

import sys

if sys.version_info < (3, 6):
    raise SystemExit("Only Python 3.6 and above is supported")
```

[`sys.version_info`](https://docs.python.org/3/library/sys.html#sys.version_info) 是一个可以用来比较的元组对象。

如果您在代码中进行这种比较，您应该用 [flake8-2020](https://pypi.org/project/flake8-2020/) 检查您的代码，以确保您正确处理版本:

```py
$ python -m pip install flake8-2020

$ flake8 bad_version_check.py good_version_check.py
bad_version_check.py:3:4: YTT103 `sys.version` compared to string
 (python3.10), use `sys.version_info`
```

随着`flake8-2020`扩展被激活，你会得到一个关于用`sys.version_info`替换`sys.version`的建议。

[*Remove ads*](/account/join/)

## 那么，该不该升级到 Python 3.10 呢？

现在，您已经看到了 Python 最新版本中最酷的特性。现在的问题是你是否应该升级到 Python 3.10，如果是，你应该什么时候升级。考虑升级到 Python 3.10 时，需要考虑两个不同的方面:

1.  您是否应该升级您的环境，以便使用 Python 3.10 解释器运行您的代码？
2.  你应该使用 Python 3.10 的新特性来编写你的代码吗？

显然，如果您想测试结构化模式匹配或您在这里读到的任何其他很酷的新特性，您需要 Python 3.10。可以将最新版本与您当前的 Python 版本并行安装。一个简单的方法是使用像 [pyenv](https://realpython.com/intro-to-pyenv/) 或 [Conda](https://realpython.com/python-windows-machine-learning-setup/) 这样的环境管理器。你也可以[使用 Docker](https://realpython.com/python-versions-docker/) 运行 Python 3.10，而不用在本地安装。

Python 3.10 已经通过了大约五个月的 beta 测试，所以开始使用它进行自己的开发应该不会有什么大问题。您可能会发现您的一些依赖项没有立即提供 Python 3.10 的[轮子，这使得它们的安装更加麻烦。但是一般来说，使用最新的 Python 进行本地开发是相当安全的。](https://realpython.com/python-wheels/)

与往常一样，在升级生产环境之前，您应该小心谨慎。警惕测试你的代码在新版本上运行良好。特别是，你要留意那些被[弃用的](https://docs.python.org/3.10/whatsnew/3.10.html#deprecated)或被[移除的](https://docs.python.org/3.10/whatsnew/3.10.html#removed)特性。

您是否可以在代码中开始使用这些新特性取决于您的用户群和代码运行的环境。如果您能保证 Python 3.10 是可用的，那么使用新的联合类型语法或任何其他新特性都没有危险。

如果你分发的应用或库被其他人使用，你可能要保守一点。目前， [Python 3.6](https://www.python.org/dev/peps/pep-0494) 是官方支持的最老的 Python 版本。它将于 2021 年 12 月寿终正寝，之后 [Python 3.7](https://www.python.org/dev/peps/pep-0537) 将是支持的最低版本。

该文档包括一个关于将代码移植到 Python 3.10 的有用指南。查看更多详情！

## 结论

新 Python 版本的发布总是值得庆祝的。即使你不能马上开始使用这些新功能，它们也会在几年内广泛应用，成为你日常生活的一部分。

在本教程中，您已经看到了一些新功能，如:

*   更友好的**错误消息**
*   强大的**结构模式匹配**
*   **类型提示**改进
*   更安全的**序列组合**
*   新增**统计功能**

要了解更多 Python 3.10 技巧以及与真正的 Python 团队成员的讨论，请查看[真正的 Python 播客第 81 集](https://realpython.com/podcasts/rpp/81/)。

体验新功能的乐趣！请在下面的评论中分享你的经历。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Python 3.10 中很酷的新特性**](/courses/cool-new-features-python-310/)************