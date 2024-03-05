# 如何在 Python 中替换字符串

> 原文：<https://realpython.com/replace-string-python/>

如果你正在寻找在 Python 中删除或替换全部或部分[字符串](https://realpython.com/python-strings/)的方法，那么本教程就是为你准备的。你将获得一个虚构的聊天室脚本，并使用 **`.replace()`方法**和 **`re.sub()`函数**对其进行[净化](https://en.wikipedia.org/wiki/Sanitization_(classified_information))。

在 Python 中，`.replace()`方法和`re.sub()`函数通常用于通过移除字符串或子字符串或替换它们来清理文本。在本教程中，您将扮演一家公司的开发人员，该公司通过一对一的文本聊天提供技术支持。你的任务是创建一个脚本来净化聊天，删除任何[的个人数据](https://en.wikipedia.org/wiki/Personal_data)，并用表情符号替换任何脏话。

你只会得到一份非常简短的聊天记录:

```py
[support_tom] 2022-08-24T10:02:23+00:00 : What can I help you with?
[johndoe] 2022-08-24T10:03:15+00:00 : I CAN'T CONNECT TO MY BLASTED ACCOUNT
[support_tom] 2022-08-24T10:03:30+00:00 : Are you sure it's not your caps lock?
[johndoe] 2022-08-24T10:04:03+00:00 : Blast! You're right!
```

尽管这份文字记录很短，但它是代理一直在进行的典型聊天类型。它有用户标识符、 [ISO 时间戳](https://en.wikipedia.org/wiki/ISO_8601)和消息。

在这种情况下，客户`johndoe`提交了投诉，公司的政策是整理和简化记录，然后将其传递给独立评估。净化信息是你的工作！

**示例代码:** [点击这里下载免费的示例代码](https://realpython.com/bonus/replace-string-python-code/)，您将使用它来替换 Python 中的字符串。

你要做的第一件事就是注意任何脏话。

## 如何移除或替换 Python 字符串或子字符串

在 Python 中替换字符串最基本的方法是使用`.replace()` string 方法:

>>>

```py
>>> "Fake Python".replace("Fake", "Real")
'Real Python'
```

如您所见，您可以将`.replace()`链接到任何字符串上，并为该方法提供两个参数。第一个是要替换的字符串，第二个是替换。

**注意:**虽然 Python shell 显示了`.replace()`的结果，但是字符串本身保持不变。通过将字符串赋给变量，可以更清楚地看到这一点:

>>>

```py
>>> name = "Fake Python"
>>> name.replace("Fake", "Real")
'Real Python'

>>> name
'Fake Python'

>>> name = name.replace("Fake", "Real")
'Real Python'

>>> name
'Real Python'
```

请注意，当您简单地调用`.replace()`时，`name`的值不会改变。但是当你把`name.replace()`的结果赋给`name`变量的时候，`'Fake Python'`就变成了`'Real Python'`。

现在是时候将这些知识应用到文字记录中了:

>>>

```
>>> transcript = """\
... [support_tom] 2022-08-24T10:02:23+00:00 : What can I help you with?
... [johndoe] 2022-08-24T10:03:15+00:00 : I CAN'T CONNECT TO MY BLASTED ACCOUNT
... [support_tom] 2022-08-24T10:03:30+00:00 : Are you sure it's not your caps lock?
... [johndoe] 2022-08-24T10:04:03+00:00 : Blast! You're right!"""

>>> transcript.replace("BLASTED", "😤")
[support_tom] 2022-08-24T10:02:23+00:00 : What can I help you with?
[johndoe] 2022-08-24T10:03:15+00:00 : I CAN'T CONNECT TO MY 😤 ACCOUNT
[support_tom] 2022-08-24T10:03:30+00:00 : Are you sure it's not your caps lock?
[johndoe] 2022-08-24T10:04:03+00:00 : Blast! You're right!
```py

将脚本加载为一个[三重引用字符串](https://realpython.com/python-data-types/#triple-quoted-strings)，然后对其中一个脏话使用`.replace()`方法就可以了。但是还有另一个不会被取代的脏话，因为在 Python 中，字符串需要与*完全匹配*:

>>>

```
>>> "Fake Python".replace("fake", "Real")
'Fake Python'
```py

如您所见，即使一个字母的大小写不匹配，它也会阻止任何替换。这意味着如果你使用的是`.replace()`方法，你将需要不同次数的调用它。在这种情况下，您可以继续另一个对`.replace()`的呼叫:

>>>

```
>>> transcript.replace("BLASTED", "😤").replace("Blast", "😤")
[support_tom] 2022-08-24T10:02:23+00:00 : What can I help you with?
[johndoe] 2022-08-24T10:03:15+00:00 : I CAN'T CONNECT TO MY 😤 ACCOUNT
[support_tom] 2022-08-24T10:03:30+00:00 : Are you sure it's not your caps lock?
[johndoe] 2022-08-24T10:04:03+00:00 : 😤! You're right!
```py

成功！但是你可能会想，对于像通用转录消毒剂这样的东西来说，这不是最好的方法。你会想要有一个替换列表，而不是每次都必须键入`.replace()`。

[*Remove ads*](/account/join/)

## 设置多个替换规则

您还需要对文字稿进行一些替换，以使其成为独立审查可接受的格式:

*   缩短或删除时间戳
*   将用户名替换为*代理*和*客户端*

现在你开始有更多的字符串需要替换，链接`.replace()`会变得重复。一个想法是保存一个元组的 T2 列表，每个元组中有两个条目。这两项对应于您需要传递给`.replace()`方法的参数——要替换的字符串和替换字符串:

```
# transcript_multiple_replace.py

REPLACEMENTS = [
    ("BLASTED", "😤"),
    ("Blast", "😤"),
    ("2022-08-24T", ""),
    ("+00:00", ""),
    ("[support_tom]", "Agent "),
    ("[johndoe]", "Client"),
]

transcript = """
[support_tom] 2022-08-24T10:02:23+00:00 : What can I help you with?
[johndoe] 2022-08-24T10:03:15+00:00 : I CAN'T CONNECT TO MY BLASTED ACCOUNT
[support_tom] 2022-08-24T10:03:30+00:00 : Are you sure it's not your caps lock?
[johndoe] 2022-08-24T10:04:03+00:00 : Blast! You're right!
"""

for old, new in REPLACEMENTS:
    transcript = transcript.replace(old, new)

print(transcript)
```py

在这个版本的清理脚本中，您创建了一个替换元组列表，这为您提供了一个快速添加替换的方法。如果有大量替换，您甚至可以从外部 [CSV 文件](https://realpython.com/python-csv/)中创建这个元组列表。

然后迭代替换元组列表。在每次迭代中，对字符串调用`.replace()`，用从每个替换元组中解包的`old`和`new`变量填充参数。

**注意:**在这种情况下，`for`循环中的解包在功能上与使用索引相同:

```
for replacement in replacements:
    new_transcript = new_transcript.replace(replacement[0], replacement[1])
```py

如果您对解包感到困惑，那么查看 Python 列表和元组教程中关于解包的[部分。](https://realpython.com/python-lists-tuples/#tuple-assignment-packing-and-unpacking)

这样，您就大大提高了脚本的整体可读性。如果需要，还可以更容易地添加替换。运行这个脚本显示了一个更加清晰的脚本:

```
$ python transcript_multiple_replace.py
Agent  10:02:23 : What can I help you with?
Client 10:03:15 : I CAN'T CONNECT TO MY 😤 ACCOUNT
Agent  10:03:30 : Are you sure it's not your caps lock?
Client 10:04:03 : 😤! You're right!
```py

这是一份非常干净的成绩单。也许这就是你所需要的。但是如果你内心的自动机不开心，可能是因为仍然有一些事情困扰着你:

*   如果有另一种使用 *-ing* 或不同大小写的变体，如 *BLAst* ，替换脏话将不起作用。
*   从时间戳中删除日期目前仅适用于 2022 年 8 月 24 日。
*   移除完整的时间戳将涉及到为每一个可能的时间建立替换对——这不是你太热衷于做的事情。
*   在*代理*后添加空格来排列你的列是可行的，但不是很通用。

如果这些是您关心的问题，那么您可能希望将注意力转向正则表达式。

## 利用`re.sub()`制定复杂的规则

每当你想做一些稍微复杂一些或者需要一些[通配符](https://en.wikipedia.org/wiki/Wildcard_character)的替换时，你通常会想把注意力转向[正则表达式](https://realpython.com/regex-python/)，也称为**正则表达式**。

Regex 是一种小型语言，由定义模式的字符组成。这些模式或正则表达式通常用于在*查找*和*查找和替换*操作中搜索字符串。许多编程语言都支持正则表达式，它被广泛使用。Regex 甚至会给你[超能力](https://xkcd.com/208/)。

在 Python 中，利用正则表达式意味着使用`re`模块的 [`sub()`函数](https://docs.python.org/3/library/re.html#re.sub)并构建自己的正则表达式模式:

```
# transcript_regex.py

import re

REGEX_REPLACEMENTS = [
    (r"blast\w*", "😤"),
    (r" [-T:+\d]{25}", ""),
    (r"\[support\w*\]", "Agent "),
    (r"\[johndoe\]", "Client"),
]

transcript = """
[support_tom] 2022-08-24T10:02:23+00:00 : What can I help you with?
[johndoe] 2022-08-24T10:03:15+00:00 : I CAN'T CONNECT TO MY BLASTED ACCOUNT
[support_tom] 2022-08-24T10:03:30+00:00 : Are you sure it's not your caps lock?
[johndoe] 2022-08-24T10:04:03+00:00 : Blast! You're right!
"""

for old, new in REGEX_REPLACEMENTS:
    transcript = re.sub(old, new, transcript, flags=re.IGNORECASE)

print(transcript)
```py

虽然您可以将`sub()`函数与`.replace()`方法混合使用，但是本例只使用了`sub()`，所以您可以看到它是如何使用的。您会注意到，现在只需使用一个替换元组就可以替换脏话的所有变体。类似地，对于完整的时间戳，您只使用一个正则表达式:

```
$ python transcript_regex.py
Agent  : What can I help you with?
Client : I CAN'T CONNECT TO MY 😤 ACCOUNT
Agent  : Are you sure it's not your caps lock?
Client : 😤! You're right!
```

现在你的成绩单已经完全清理干净了，所有的噪音都被去除了！那是怎么发生的？这就是 regex 的魔力。

**的第一个正则表达式模式**，`"blast\w*"`，利用了`\w` [特殊字符](https://www.regular-expressions.info/characters.html)，它将匹配[字母数字字符](https://en.wikipedia.org/wiki/Alphanumericals)和下划线。直接在`*` [后面加上量词](https://www.regular-expressions.info/refrepeat.html)将匹配`\w`的零个或多个字符。

第一个模式的另一个重要部分是,`re.IGNORECASE`标志使它不区分大小写。所以现在，任何包含`blast`的子串，无论大小写，都会被匹配和替换。

**注:**`"blast\w*"`模式相当宽泛，也会将`fibroblast`修改为`fibro😤`。它也不能识别这个词的礼貌用法。刚好符合人物。也就是说，你想要审查的典型脏话并不真的有礼貌的替代含义！

**第二个正则表达式模式**使用[字符集](https://www.regular-expressions.info/charclass.html)和量词来替换时间戳。你经常一起使用字符集和量词。例如，`[abc]`的正则表达式模式将匹配一个字符`a`、`b`或`c`。在匹配`a`、`b`或`c`的零个或多个字符的**之后直接加上一个`*`。**

尽管有更多的量词。如果您使用`[abc]{10}`，它将精确匹配`a`、`b`或`c`的任意顺序和任意组合的十个字符。还要注意重复字符是多余的，所以`[aa]`相当于`[a]`。

对于时间戳，您使用一个扩展字符集`[-T:+\d]`来匹配您可能在时间戳中找到的所有可能字符。与量词`{25}`配对，这将匹配任何可能的时间戳，至少到 10，000 年。

**注意:**特殊字符`\d`，匹配任何数字字符。

时间戳正则表达式模式允许您以时间戳格式选择任何可能的日期。鉴于《纽约时报》对于这些抄本的独立审查员来说并不重要，您可以用一个空字符串来替换它们。可以编写一个更高级的正则表达式，在删除日期的同时保留时间信息。

第三个正则表达式模式用于选择任何以关键字`"support"`开头的用户字符串。请注意，您[对](https://en.wikipedia.org/wiki/Escape_character) ( `\`)方括号(`[`)进行了转义，因为否则该关键字将被解释为字符集。

最后，**最后一个正则表达式模式**选择客户端用户名字符串并用`"Client"`替换它。

**注意:**虽然深入了解这些正则表达式模式的细节会很有趣，但本教程不是关于正则表达式的。通读 [Python 正则表达式教程](https://realpython.com/regex-python/)可以获得关于这个主题的很好的入门。此外，您可以利用神奇的 [RegExr](https://regexr.com/) 网站，因为 regex 很复杂，所有级别的 regex 向导都依赖于像 RegExr 这样方便的工具。

RegExr 特别好，因为您可以复制和粘贴 regex 模式，它会通过解释为您分解它们。

使用 regex，您可以大幅减少必须写出的替换数量。也就是说，您可能仍然需要想出许多模式。鉴于 regex 不是可读性最好的语言，拥有大量模式很快就会变得难以维护。

幸运的是，`re.sub()`有一个巧妙的技巧，允许您对替换的工作方式有更多的控制，并且它提供了一个更易维护的架构。

[*Remove ads*](/account/join/)

## 使用带有`re.sub()`的回调来获得更多控制

Python 和`sub()`的一个锦囊妙计是，你可以传入一个[回调函数](https://en.wikipedia.org/wiki/Callback_(computer_programming))，而不是替换字符串。这使您可以完全控制如何匹配和替换。

为了开始构建这个版本的脚本清理脚本，您将使用一个基本的 regex 模式来看看如何使用带有`sub()`的回调:

```py
# transcript_regex_callback.py

import re

transcript = """
[support_tom] 2022-08-24T10:02:23+00:00 : What can I help you with?
[johndoe] 2022-08-24T10:03:15+00:00 : I CAN'T CONNECT TO MY BLASTED ACCOUNT
[support_tom] 2022-08-24T10:03:30+00:00 : Are you sure it's not your caps lock?
[johndoe] 2022-08-24T10:04:03+00:00 : Blast! You're right!
"""

def sanitize_message(match):
    print(match)

re.sub(r"[-T:+\d]{25}", sanitize_message, transcript)
```

您使用的 regex 模式将匹配时间戳，并且您不是提供替换字符串，而是传入对`sanitize_message()`函数的引用。现在，当`sub()`找到一个匹配时，它将调用`sanitize_message()`,使用一个匹配对象作为参数。

由于`sanitize_message()`只是打印它作为参数接收的对象，当运行它时，您会看到匹配对象被打印到控制台:

```py
$ python transcript_regex_callback.py
<re.Match object; span=(15, 40), match='2022-08-24T10:02:23+00:00'>
<re.Match object; span=(79, 104), match='2022-08-24T10:03:15+00:00'>
<re.Match object; span=(159, 184), match='2022-08-24T10:03:30+00:00'>
<re.Match object; span=(235, 260), match='2022-08-24T10:04:03+00:00'>
```

一个[匹配对象](https://docs.python.org/3/library/re.html#match-objects)是`re`模块的构件之一。更基本的`re.match()`函数返回一个匹配对象。`sub()`不返回任何匹配对象，而是在后台使用它们。

因为您在回调中获得了这个 match 对象，所以您可以使用其中包含的任何信息来构建替换字符串。一旦构建好了，就返回新的字符串，`sub()`将用返回的字符串替换匹配的字符串。

## 将回调应用到脚本

在您的脚本清理脚本中，您将利用 match 对象的`.groups()`方法来返回两个捕获组的内容，然后您可以在它自己的函数中清理每个部分或者丢弃它:

```
# transcript_regex_callback.py

import re

ENTRY_PATTERN = (
    r"\[(.+)\] "  # User string, discarding square brackets
    r"[-T:+\d]{25} "  # Time stamp
    r": "  # Separator
    r"(.+)"  # Message
)
BAD_WORDS = ["blast", "dash", "beezlebub"]
CLIENTS = ["johndoe", "janedoe"]

def censor_bad_words(message):
    for word in BAD_WORDS:
        message = re.sub(rf"{word}\w*", "😤", message, flags=re.IGNORECASE)
    return message

def censor_users(user):
    if user.startswith("support"):
        return "Agent"
    elif user in CLIENTS:
        return "Client"
    else:
        raise ValueError(f"unknown client: '{user}'")

def sanitize_message(match):
    user, message = match.groups()
    return f"{censor_users(user):<6} : {censor_bad_words(message)}"

transcript = """
[support_tom] 2022-08-24T10:02:23+00:00 : What can I help you with?
[johndoe] 2022-08-24T10:03:15+00:00 : I CAN'T CONNECT TO MY BLASTED ACCOUNT
[support_tom] 2022-08-24T10:03:30+00:00 : Are you sure it's not your caps lock?
[johndoe] 2022-08-24T10:04:03+00:00 : Blast! You're right!
"""

print(re.sub(ENTRY_PATTERN, sanitize_message, transcript))
```py

不需要有很多不同的正则表达式，你可以有一个匹配整行的顶级正则表达式，用括号(`()`)把它分成捕获组。捕获组对实际的匹配过程没有影响，但它们会影响由匹配产生的匹配对象:

*   `\[(.+)\]`匹配方括号中的任何字符序列。捕获组挑选出用户名字符串，例如`johndoe`。
*   `[-T:+\d]{25}`匹配您在上一节中探索的时间戳。因为您不会在最终的脚本中使用时间戳，所以不会用括号来捕获它。
*   `:`匹配一个字面冒号。冒号用作消息元数据和消息本身之间的分隔符。
*   `(.+)`匹配任何字符序列，直到行尾，这将是消息。

通过调用`.groups()`方法，捕获组的内容将作为 match 对象中的单独项可用，该方法返回匹配字符串的元组。

**注意:**条目正则表达式定义使用 Python 的隐式字符串连接:

```
ENTRY_PATTERN = (
    r"\[(.+)\] "  # User string, discarding square brackets
    r"[-T:+\d]{25} "  # Time stamp
    r": "  # Separator
    r"(.+)"  # Message
)
```py

从功能上来说，这与将所有内容写成一个字符串是一样的:`r"\[(.+)\] [-T:+\d]{25} : (.+)"`。将较长的正则表达式模式组织在单独的行上，可以将它分成块，这不仅使它更易读，而且还允许您插入注释。

这两个组是用户字符串和消息。`.groups()`方法将它们作为一组字符串返回。在`sanitize_message()`函数中，首先使用解包将两个字符串赋给变量:

```
def sanitize_message(match):
 user, message = match.groups()    return f"{censor_users(user):<6} : {censor_bad_words(message)}"
```py

请注意这种体系结构如何在顶层允许非常广泛和包容的正则表达式，然后让您在替换回调中用更精确的正则表达式补充它。

`sanitize_message()`函数使用两个函数来清除用户名和不良单词。它还使用 [f 弦](https://realpython.com/python-f-strings/)来调整消息。注意`censor_bad_words()`如何使用动态创建的正则表达式，而`censor_users()`依赖于更基本的字符串处理。

这看起来像是一个很好的净化脚本的第一个原型！输出非常干净:

```
$ python transcript_regex_callback.py
Agent  : What can I help you with?
Client : I CAN'T CONNECT TO MY 😤 ACCOUNT
Agent  : Are you sure it's not your caps lock?
Client : 😤! You're right!
```

不错！使用带有回调的`sub()`可以让您更加灵活地混合和匹配不同的方法，并动态构建正则表达式。当你的老板或客户不可避免地改变他们对你的要求时，这种结构也给你最大的发展空间！

[*Remove ads*](/account/join/)

## 结论

在本教程中，您已经学习了如何在 Python 中替换字符串。一路走来，您已经从使用基本的 Python `.replace()` string 方法发展到使用带有`re.sub()`的回调来实现绝对控制。您还研究了一些正则表达式模式，并将它们分解成更好的架构来管理替换脚本。

有了所有这些知识，您已经成功地清理了一份聊天记录，现在可以进行独立审查了。不仅如此，您的脚本还有很大的发展空间。

**示例代码:** [点击这里下载免费的示例代码](https://realpython.com/bonus/replace-string-python-code/)，您将使用它来替换 Python 中的字符串。***