# Python 的真实世界正则表达式

> 原文：<https://www.pythoncentral.io/real-world-regular-expressions-for-python/>

我们已经在这一系列文章中讨论了很多内容，所以现在让我们把它们放在一起，并通过一个实际应用程序来工作。

一个常见的任务是解析 Windows INI 文件，这些文件是键/值对，分成几个部分，如下所示:

```py

[Section 1]

val1=hello world

val2=42

[Section 2]

val1=foo!

```

让我们首先编写一段 Python 代码，逐行读入测试文件:

```py

for lineBuf in open('test.ini', 'r'):

    print(lineBuf)

```

现在，我们将通过编写一些正则表达式来计算每行内容，从而扩展这一点。

## 识别章节标题

我们要做的第一件事是编写一个正则表达式，它将识别部分标题，即以方括号开始和结束的行。我们可以这样写一个正则表达式:`^\[(.+)\]$`

用简单的英语说:

*   匹配`^`(行首)。
*   匹配一个`[`字符(转义，因为`[`通常在正则表达式中有特殊含义)。
*   匹配在组中捕获的一个或多个字符(部分名称)。
*   匹配一个`]`字符(其实没必要转义这个)。
*   匹配`$`(行尾)。

如果我们更新代码来使用这个正则表达式:

```py

sectionRegEx = re.compile(r'^\[(.+)\]$')

for lineBuf in open('test.ini', 'r'):

    mo = sectionRegEx.search(lineBuf)

    if mo:

        print('Found a section: [%s]' % mo.group(1))

```

我们得到这样的输出:

```py

Found a section: [Section 1]

Found a section: [Section 2]

```

似乎工作正常！

## 处理节标题中的空白

处理节标题中的空白会很方便，所以如果有人给我们一个如下所示的 INI 文件:

```py

[Section 1]

val1=hello world

val2=42

[  Section 2 ]  junk here!

val1=foo!

```

我们将能够正确处理奇怪的第二部分标题。现在，我们的代码没有找到它，所以让我们更新正则表达式来处理它:`^\s*\[\s*(.+?)\s*\]`

用简单的英语说:

*   匹配`^`(行首)。
*   匹配`\s*`(零个或多个空白字符)。
*   匹配一个`[`字符。
*   匹配`\s*`(零个或多个空白字符)。
*   匹配一个或多个字符(部分名称)。
*   匹配`\s*`(零个或多个空白字符)。
*   匹配一个`]`字符。

请注意，我们必须使`+`字符(捕获部分名称)非贪婪，以防止它匹配任何可能出现在结束`]`之前的尾随空格。我们也在结束`]`后停止匹配，因为我们不关心在它之后的线上是否有任何东西。

现在，我们的代码识别了格式古怪的节名:

```py

sectionRegEx = re.compile(r'^\s*\[\s*(.+?)\s*\]')

for lineBuf in open('test.ini', 'r'):

    mo = sectionRegEx.search(lineBuf)

    if mo :

        print('Found a section: [%s]' % mo.group(1))

```

这为我们提供了以下输出:

```py

Found a section: [Section 1]

Found a section: [Section 2]

```

找到了第二个部分的标题，并清除了它的名称。

## 识别键/值对

下一步是编写一个识别键/值对的正则表达式，可能是这样的:`^(.+)=(.+)$`

用简单的英语说:

*   匹配`^`(行首)。
*   匹配在组中捕获的一个或多个字符(密钥名)。
*   匹配`=`字符。
*   匹配在组中捕获的一个或多个字符(键值)。
*   匹配`$`(行尾)。

同样，我们希望这个正则表达式处理无关的空白，所以让我们把它改写成这样:`^\s*(.+?)\s*=\s*(.+?)\s*$`

我们更新后的代码现在看起来像这样:

```py

sectionRegEx = re.compile(r'^\s*\[\s*(.+?)\s*\]')

keyValRegEx = re.compile(r'^\s*(.+?)\s*=\s*(.+?)\s*$')

for lineBuf in open('test.ini', 'r'):

    mo = sectionRegEx.search(lineBuf)

    if mo:

        print('Found a section: [%s]' % mo.group(1))

    mo = keyValRegEx.search(lineBuf)

    if mo:

        print('{%s} = {%s}' % (mo.group(1), mo.group(2)))

```

当我们打印键名和值时，我们用花括号将它们括起来，这样我们就可以看到它们是否被正确地剪裁了。

如果我们给它以下测试输入:

```py

[Section 1]

val1=hello world

val2  =  42 = forty-two

[  Section 2 ]  junk here!

val1=foo!

```

我们得到以下输出:

```py

Found a section: [Section 1]

{val1} = {hello world}

{val2} = {42 = forty-two}

Found a section: [Section 2]

{val1} = {foo!}

```