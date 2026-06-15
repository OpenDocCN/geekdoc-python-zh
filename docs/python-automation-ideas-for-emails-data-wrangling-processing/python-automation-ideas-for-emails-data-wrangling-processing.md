

### PYTHON

#### 自动化：

电子邮件、数据整理、Excel处理、报告生成、网络爬虫等创意方案

劳埃德·奥乔亚

让我们开启自动化之旅

- 使用第三方工具——parse
- 介绍正则表达式
- 深入了解正则表达式
- 添加命令行参数

我们将从创建一个独立的工作环境开始。

#### 激活虚拟环境

使用Python时，第一步最好是明确定义工作环境。

这有助于你脱离操作系统解释器和环境，并正确定义将要使用的依赖项。不这样做往往会导致混乱的局面。记住，*显式优于隐式！*

> ![icon](icon.png) *显式优于隐式*是《Python之禅》中被引用最多的部分之一。《Python之禅》是Python的一系列通用准则，旨在阐明何为Pythonic风格。完整的《Python之禅》可以通过在Python解释器中调用`import this`来查看。

这在两种情况下尤为重要：

- 当在同一台计算机上处理多个项目时，因为它们可能具有在某个时刻冲突的不同依赖项。例如，同一模块的两个版本无法安装在同一环境中。
- 当在一个将在不同计算机上使用的项目上工作时，例如，在个人笔记本电脑上开发一些代码，最终将在远程服务器上运行。

开发者之间的一个常见笑话是，对bug的回应是“它在我的机器上运行”，意思是它在他们的笔记本电脑上似乎可以工作，但在生产服务器上却不行。尽管大量因素可能导致此错误，但一个好的做法是创建一个可自动复制的环境，减少对实际使用哪些依赖项的不确定性。

使用`venv`模块可以轻松实现这一点，它设置了一个本地虚拟环境。安装的依赖项都不会与机器上安装的Python解释器共享，从而创建一个隔离的环境。

在Python 3中，`venv`工具作为标准库的一部分安装。在之前的版本中并非如此，你必须安装外部的`virtualenv`包。

### 准备工作

要创建一个新的虚拟环境，请执行以下操作：

1. 转到包含项目的主目录：

   ```
   $ cd my-directory
   ```

2. 输入以下命令：

   ```
   $ python3 -m venv .venv
   ```

   这将创建一个名为`.venv`的子目录，其中包含虚拟环境。
   包含虚拟环境的目录可以位于任何位置。将其保留在同一根目录下便于管理，并在前面添加一个点可以避免在运行`ls`或其他命令时显示它。

3. 在激活虚拟环境之前，检查`pip`中安装的版本。这因你的操作系统和已安装的包而异。稍后可能会升级。同时，检查引用的Python解释器，它将是主要的操作系统解释器：

   ```
   $ pip --version
   pip 10.0.1 from /usr/local/lib/python3.7/site-packages/pip (python 3.7)
   $ which python3
   /usr/local/bin/python3
   ```

> 请注意，`which`可能在你的shell中不可用。例如，在Windows中，可以使用`where`。

现在你的虚拟环境已准备就绪。

### 操作步骤...

1. 如果你使用Linux或macOS，通过运行以下命令激活虚拟环境：

   ```
   $ source .venv/bin/activate
   ```

   根据你的操作系统（例如Windows）和shell（例如fish），你可能需要不同的命令。请在此处查看Python文档中venv的文档：https://docs.python.org/3/library/venv.html。
   你会注意到shell提示符将显示(.venv)，表明虚拟环境已激活。

2. 注意，使用的Python解释器是虚拟环境内的那个，而不是“准备工作”部分步骤3中的通用操作系统解释器。检查虚拟环境内的位置：

   ```
   (.venv) $ which python
   /root_dir/.venv/bin/python
   (.venv) $ which pip
   /root_dir/.venv/bin/pip
   ```

3. 升级pip的版本，然后检查版本：

   ```
   (.venv) $ pip install --upgrade pip
   ...
   Successfully installed pip-20.0.2
   (.venv) $ pip --version
   pip 20.0.2 from /root_dir/.venv/lib/python3.8/site-packages/pip (python 3.8)
   ```

> 另一种方法是运行`python -m ensurepip -U`，这将确保pip已安装。

4. 退出环境并运行pip以检查版本，这将返回之前的环境。检查pip版本和Python解释器，以显示激活虚拟环境目录之前的现有目录，如“准备工作”部分步骤3所示。请注意，它们是不同的pip版本：

   ```
   (.venv) $ deactivate
   $ which python3
   /usr/local/bin/python3
   $ pip --version
   pip 10.0.1 from /usr/local/lib/python3.8/site-packages/pip (python 3.8)
   ```

### 工作原理...

请注意，在虚拟环境中，你可以使用`python`而不是`python3`，尽管`python3`也可用。这将使用环境中定义的Python解释器。

在某些系统（如Linux）中，你可能需要使用`python3.8`而不是`python3`。请验证你使用的Python解释器是3.8或更高版本。

在虚拟环境中，“操作步骤”部分的*步骤3*安装了最新版本的`pip`，而不会影响外部安装。

虚拟环境包含`.venv`目录中的所有Python数据，`activate`脚本将所有环境变量指向那里。最棒的是，它可以非常容易地删除和重新创建，消除了在独立沙箱中进行实验的恐惧。

请记住，目录名称会显示在提示符中。如果你需要区分环境，请使用描述性的目录名称，例如`.my_automate_recipe`，或使用`-prompt`选项。

### 更多内容...

要删除虚拟环境，请停用它并删除目录：

```
(.venv) $ deactivate
$ rm -rf .venv
```

`venv`模块有更多选项，可以通过`-h`标志显示：

```
$ python3 -m venv -h
usage: venv [-h] [--system-site-packages]
            [--symlinks | --copies] [--clear]
            [--upgrade] [--without-pip]
            [--prompt PROMPT]
            ENV_DIR [ENV_DIR ...]

Creates virtual Python environments in one or more target directories.

positional arguments:
  ENV_DIR               A directory to create the
                        environment in.
```

可选参数：

| 参数 | 描述 |
| --- | --- |
| -h, --help | 显示此帮助信息并退出 |
| --system-site-packages | 允许虚拟环境访问系统site-packages目录。 |
| --symlinks | 尝试使用符号链接而不是复制，当符号链接不是平台的默认设置时。 |
| --copies | 尝试使用复制而不是符号链接，即使符号链接是平台的默认设置。 |
| --clear | 如果环境目录已存在，则在创建环境之前删除其内容。 |
| --upgrade | 升级环境目录以使用此版本的Python，假设Python已就地升级。 |
| --without-pip | 跳过在虚拟环境中安装或升级pip（默认情况下pip是引导安装的） |
| --prompt PROMPT | 为此环境提供替代的提示符前缀。 |

一旦创建了环境，你可能希望激活它，例如通过在其bin目录中source一个activate脚本。

处理虚拟环境的一种便捷方式，特别是如果你经常需要在它们之间切换，是使用`virtualenvwrapper`模块：

1. 要安装它，请运行：

   ```bash
   $ pip install virtualenvwrapper
   ```

2. 然后，将以下变量添加到你的shell启动脚本中，这些通常是`.bashrc`或`.bash_profile`。虚拟环境将安装在`WORKON_HOME`目录下，而不是与项目相同的目录，如前所示：

   ```bash
   export WORKON_HOME=~/.virtualenvs
   source /usr/local/bin/virtualenvwrapper.sh
   ```

Source启动脚本或打开一个新终端将允许你创建新的虚拟环境：

```
$ mkvirtualenv automation_cookbook
...
Installing setuptools, pip, wheel...done.
(automation_cookbook) $ deactivate
$ workon automation_cookbook
(automation_cookbook) $
```

更多信息，请查看`virtualenvwrapper`的文档：https://virtualenvwrapper.readthedocs.io/en/latest/index.html。

> 另一个定义环境的工具是Poetry (https://python-poetry.org/)。此工具旨在创建具有清晰依赖项的一致环境，并提供用于升级和管理依赖包的命令。查看它是否对你的用例有用。

在`workon`后按`Tab`键会自动补全可用环境的命令。

### 另请参阅

- 本章稍后介绍的*安装第三方包*方案。
- 本章稍后介绍的*使用第三方工具——parse*方案。

### 安装第三方包

Python 最强大的能力之一，就是能够使用令人印象深刻的第三方包目录，这些包覆盖了不同领域的大量功能，从专门执行数值运算、机器学习和网络通信的模块，到命令行便捷工具、数据库访问、图像处理等等！

其中大部分包都可在官方的 Python 包索引（https://pypi.org/）上找到，该索引拥有超过 20 万个现成可用的包。在本书中，我们将安装其中一些。通常，在尝试解决问题时，花一点时间研究外部工具是值得的。很可能已经有人创建了一个能解决全部或至少部分问题的工具。

比查找和安装包更重要的是，要跟踪正在使用哪些包。这极大地有助于**可复现性**，即在任何情况下都能从头开始重建整个环境的能力。

### 准备工作

起点是找到一个对我们的项目有用的包。

一个很棒的包是 `requests`，这是一个处理 HTTP 请求的模块，以其简单直观的接口和出色的文档而闻名。可以在这里查看文档：https://requests.readthedocs.io/en/master/。

在本书中，我们将使用 `requests` 来处理 HTTP 连接。

下一步是选择要使用的版本。在这种情况下，最新版本（撰写本文时为 2.22.0）将是完美的。如果未指定模块的版本，默认情况下将安装最新版本，这可能导致不同环境之间出现不一致，因为新版本会不断发布。

我们还将使用出色的 `delorean` 模块进行时间处理（版本 1.0.0：http://delorean.readthedocs.io/en/latest/）。

### 操作步骤...

1. 在我们的 `main` 目录中创建一个 `requirements.txt` 文件，该文件将指定我们项目的所有依赖项。让我们从 `delorean` 和 `requests` 开始：

    ```
    delorean==1.0.0
    requests==2.22.0
    ```

2. 使用 `pip` 命令安装所有依赖项：

    ```
    $ pip install -r requirements.txt
    ...
    Successfully installed babel-2.8.0 certifi-2019.11.28
    chardet-3.0.4 delorean-1.0.0 humanize-0.5.1 idna-2.8 python-
    dateutil-2.8.1 pytz-2019.3 requests-2.22.0 six-1.14.0
    tzlocal-2.0.0 urllib3-1.25.7
    ```

    使用 `pip list` 显示已安装的可用模块：

    ```
    $ pip list
    Package            Version
    ------------------ ----------
    Babel              2.8.0
    certifi            2019.11.28
    chardet            3.0.4
    Delorean           1.0.0
    humanize           2.0.0
    idna               2.8
    pip                19.2.3
    python-dateutil    2.8.1
    pytz               2019.3
    requests           2.22.0
    setuptools         41.2.0
    six                1.14.0
    tzlocal            2.0.0
    urllib3            1.25.8
    ```

3. 现在，您可以在使用虚拟环境时使用这两个模块：

    ```
    $ python
    Python 3.8.1 (default, Dec 27 2019, 18:05:45)
    [Clang 11.0.0 (clang-1100.0.33.16)] on darwin
    Type "help", "copyright", "credits" or "license" for more
    information.
    >>> import delorean
    >>> import requests
    ```

### 工作原理...

`requirements.txt` 文件指定了模块和版本，pip 会在 `pypi.org` 上进行搜索。

请注意，从头开始创建一个新的虚拟环境并运行以下命令将完全重建您的环境，这使得可复现性变得非常直接：

```
$ pip install -r requirements.txt
```

请注意，*操作步骤*部分的第 2 步会自动安装作为依赖项的其他模块，例如 `urllib3`。

### 更多内容...

如果任何模块需要更改为不同的版本（因为有新版本可用），请使用 `requirements` 进行更改，然后再次运行 `install` 命令：

```
$ pip install -r requirements.txt
```

当需要包含新模块时，这也适用。

在任何时候，都可以使用 `freeze` 命令来显示所有已安装的模块。`freeze` 以与 `requirements.txt` 兼容的格式返回模块，从而可以生成包含我们当前环境的文件：

```
$ pip freeze > requirements.txt
```

这将包括依赖项，因此预计文件中会有更多模块。

找到优秀的第三方模块有时并不容易。搜索特定功能可能效果很好，但有时，一些很棒的模块会带来惊喜，因为它们能完成您从未想过的事情。一个很棒的精选列表是 Awesome Python（https://awesome-python.com/），它涵盖了针对常见 Python 用例的许多优秀工具，例如加密、数据库访问、日期和时间处理等。

在某些情况下，安装包可能需要额外的工具，例如编译器或支持某些功能的特定库（例如，特定的数据库驱动程序）。如果是这种情况，文档将解释依赖项。

### 另请参阅

- 本章前面介绍的*激活虚拟环境*方法。
- 本章后面介绍的*使用第三方工具 – parse*方法，以了解如何使用已安装的第三方模块。

### 使用格式化值创建字符串

处理创建文本和文档时的基本能力之一是能够将值正确格式化为结构化字符串。Python 擅长提供良好的默认值，例如正确渲染数字，但有很多选项和可能性。

我们将以表格为例，讨论创建格式化文本时的一些常见选项。

### 准备工作

Python 中格式化字符串的主要工具是 `format` 方法。它使用一种定义的迷你语言来渲染变量，如下所示：

```
result = template.format(*parameters)
```

`template` 是一个基于迷你语言进行解释的字符串。最简单的情况下，模板将花括号之间的值替换为参数。这里有几个例子：

```
>>> 'Put the value of the string here: {}'.format('STRING')
"Put the value of the string here: STRING"
>>> 'It can be any type ({}) and more than one ({})'.format(1.23, 'STRING')
"It can be any type (1.23) and more than one (STRING)"
>>> 'Specify the order: {1}, {0}'.format('first', 'second')
'Specify the order: second, first'
>>> 'Or name parameters: {first}, {second}'.format(second='SECOND', first='FIRST')
'Or name parameters: FIRST, SECOND'
```

在 95% 的情况下，这种格式化就足够了；保持简单是件好事！但在复杂的情况下，例如自动对齐字符串和创建美观的文本表格时，`format` 迷你语言有更多选项。

### 操作步骤...

1. 编写以下脚本 `recipe_format_strings_step1.py` 以打印对齐的表格：

    ```
    # INPUT DATA
    data = [
        (1000, 10),
        (2000, 17),
        (2500, 170),
        (2500, -170),
    ]

    # Print the header for reference
    print('REVENUE | PROFIT | PERCENT')

    # This template aligns and displays the data in the proper format
    TEMPLATE = '{revenue:>7,} | {profit:>+6} | {percent:>7.2%}'

    # Print the data rows
    for revenue, profit in data:
        row = TEMPLATE.format(revenue=revenue, profit=profit,
                              percent=profit / revenue)
        print(row)
    ```

2. 运行它以显示以下对齐的表格。请注意，`PERCENT` 被正确显示为百分比：

    ```
    REVENUE | PROFIT | PERCENT
      1,000 |    +10 |   1.00%
      2,000 |    +17 |   0.85%
      2,500 |   +170 |   6.80%
      2,500 |   -170 |  -6.80%
    ```

### 工作原理...

`TEMPLATE` 常量定义了三列，每列由名为 `revenue`、`profit` 和 `percent` 的参数定义。这使得将模板应用于 `format` 调用变得明确而直接。

在参数名称之后，有一个冒号分隔格式定义。请注意，所有内容都在花括号内。在所有列中，格式规范将宽度设置为七个字符，以确保所有列具有相同的宽度，并使用 > 符号将值右对齐：

- `revenue` 使用 `,` 符号添加千位分隔符 —— `[{revenue:>7,}]`。
- `profit` 为正值添加 `+` 号。负值会自动添加 `-` 号 —— `[{profit:>+7}]`。
- `percent` 显示精度为两位小数的百分比值 —— `[{percent:>7.2%}]`。这是通过 `.2`（精度）和添加 `%` 符号来实现百分比的。

### 更多内容...

您可能也见过使用 `%` 运算符的 Python 格式化。虽然它适用于简单的格式化，但不如格式化迷你语言灵活，因此不建议使用。

Python 3.6 以来的一个很棒的新特性是使用 f-string，它使用定义的变量执行格式化操作：

```
>>> param1 = 'first'
>>> param2 = 'second'
>>> f'Parameters {param1}:{param2}'
'Parameters first:second'
```

这简化了大量代码，并允许我们创建非常具有描述性和可读性的代码。

使用 f-string 时要小心，确保字符串在适当的时间被替换。一个常见问题是，定义为要渲染的变量尚未定义。例如，前面定义的 `TEMPLATE` 不会被定义为 f-string，因为 `revenue` 和其他参数在那时不可用。在字符串定义作用域内定义的所有变量（包括局部和全局变量）都将可用。

如果需要写入花括号，则需要重复两次。请注意，每次重复将显示为一个花括号，加上一个用于值替换的花括号，总共三个花括号：

```
>>> value = 'VALUE'
>>> f'This is the value, in curly brackets {{{value}}}'
'This is the value, in curly brackets {VALUE}'
```

### 让我们开始自动化之旅

这使我们能够创建元模板——即生成模板的模板。在某些情况下，这很有用，但它们会很快变得复杂。请谨慎使用，因为很容易生成难以阅读的代码。

> 表示具有特殊含义的字符通常需要某种特殊的定义方式，例如，像我们在这里看到的那样，通过重复花括号。这被称为“转义”，它是任何代码表示中的常见过程。

Python 格式规范迷你语言比这里显示的选项更多。

由于该语言力求非常简洁，有时可能难以确定符号的位置。你有时可能会问自己这样的问题，比如“+符号是在宽度参数之前还是之后？”请仔细阅读文档，并记住始终在格式规范前包含冒号。

请参阅 Python 网站上的完整文档和示例：https://docs.python.org/3/library/string.html#formatspec 或这个精彩的网页——https://pyformat.info——其中展示了大量示例。

### 另请参阅

- 第5章《生成精彩报告》中的“模板报告”食谱，以了解更多高级模板技术。
- 本章稍后介绍的“操作字符串”食谱，以了解更多关于处理文本的内容。

### 操作字符串

处理文本时，通常需要对其进行操作和处理；即，能够将其连接、分割成规则的块，或将其更改为大写或小写。我们稍后将讨论更高级的文本解析和分离方法；然而，在许多情况下，将段落分成行、句子甚至单词是很有用的。其他时候，单词可能需要删除某些字符，或者需要将单词替换为规范版本，以便能够将其与预定值进行比较。

### 准备工作

我们将定义一段基本文本并将其转换为其主要组成部分；然后，我们将重建它。例如，一份报告需要转换为新格式才能通过电子邮件发送。

本例中我们将使用的输入格式如下：

> 第二季度结束后，我们公司 CASTAÑACORP 的收入增长了 7.47%。这符合年度目标。销售的主要驱动力是在我们市场部监督下设计的新包装。我们的支出得到了控制，仅增长了 0.7%，尽管董事会认为需要进一步削减。评估结果令人满意，对下一季度的预测是乐观的。董事会预计利润至少增加 200 万美元。

我们需要编辑文本以消除任何对数字的引用。它需要通过在每个句点后添加新行、以 80 个字符对齐，并出于兼容性原因转换为 ASCII 来进行适当格式化。

文本将存储在解释器中的 `INPUT_TEXT` 变量中。

### 如何操作...

1. 输入文本后，将其分割成单个单词：

```
>>> INPUT_TEXT = '''
...     AFTER THE CLOSE OF THE SECOND QUARTER, OUR COMPANY,
...     CASTAÑACORP
...     HAS ACHIEVED A GROWTH IN THE REVENUE OF 7.47%. THIS IS IN
...     LINE
...     ...
...     '''
>>> words = INPUT_TEXT.split()
```

2. 将任何数字替换为 'X' 字符：

```
>>> redacted = [''.join('X' if w.isdigit() else w for w in word)
for word in words]
```

3. 将文本转换为纯 ASCII（注意公司名称包含字母 ñ，它不是 ASCII）：

```
>>> ascii_text = [word.encode('ascii', errors='replace').
decode('ascii')
...                 for word in redacted]
```

4. 将单词分组为 80 个字符的行：

```
>>> newlines = [word + '\n' if word.endswith('.') else word for
word in ascii_text]
>>> LINE_SIZE = 80
>>> lines = []
>>> line = ''
>>> for word in newlines:
...     if line.endswith('\n') or len(line) + len(word) + 1 >
LINE_SIZE:
...         lines.append(line)
...         line = ''
...     line = line + ' ' + word
```

5. 将所有行格式化为标题，并将它们连接成单个文本：

```
>>> lines = [line.title() for line in lines]
>>> result = '\n'.join(lines)
```

6. 打印结果：

```
>>> print(result)
```

After The Close Of The Second Quarter, Our Company, Casta?Acorp
Has Achieved A Growth In The Revenue Of X.Xx%. This Is In Line
With The Objectives For The Year. The Main Driver Of The Sales
Has Been The New Package Designed Under The Supervision Of Our
Marketing Department. Our Expenses Has Been Contained, Increasing
Only By X.X%, Though The Board Considers It Needs To Be Further
Reduced. The Evaluation Is Satisfactory And The Forecast For The
Next Quarter Is Optimistic.

### 工作原理...

每个步骤都对文本执行特定的转换：

- 第一步将文本按默认分隔符（空格和换行符）分割。这将其分割成单个单词，没有行或多空格作为分隔。
- 要替换数字，我们遍历每个单词的每个字符。对于每个字符，如果它是数字，则返回 'x' 代替。这是通过两个列表推导式完成的，一个用于运行列表，另一个用于每个单词，仅在存在数字时替换它们 - ['x' if w.isdigit() else w for w in word]。注意单词被重新连接在一起。
- 每个单词被编码为 ASCII 字节序列，然后解码回 Python 字符串类型。注意使用 errors 参数来强制替换未知字符，例如 ñ。

> 字符串和字节之间的区别起初并不直观，特别是如果你从不需要担心多种语言或编码转换。在 Python 3 中，字符串（内部 Python 表示）和字节之间有很强的分离。因此，适用于字符串的大多数工具在字节对象中不可用。除非你清楚为什么需要字节对象，否则始终使用 Python 字符串。如果需要执行像本任务中的转换，请在同一行中编码和解码，以便将对象保持在舒适的 Python 字符串领域内。如果你有兴趣了解更多关于编码的知识，可以参考这篇简短的文章：https://eli.thegreenplace.net/2012/01/30/the-bytesstr-dichotomy-in-python-3 以及这篇更长、更详细的文章：http://www.diveintopython3.net/strings.html。

- 下一步为所有以句点结尾的单词添加一个额外的换行符（\n 字符）。这标记了不同的段落。之后，它创建一行并逐个添加单词。如果额外的单词使其超过 80 个字符，它将完成该行并开始新行。如果该行已经以换行符结尾，它也会完成该行并开始另一行。注意添加了一个额外的空格来分隔单词。
- 最后，每行都被格式化为标题（每个单词的首字母大写），所有行通过换行符连接。

### 更多内容...

可以在字符串上执行的其他一些有用操作如下：

- 字符串可以像任何其他列表一样进行切片。这意味着 `"word"[0:2]` 将返回 `"wo"`。
- 使用 `.splitlines()` 用换行符分隔行。
- 有 `.upper()` 和 `.lower()` 方法，它们返回一个副本，其中所有字符都设置为大写或小写。它们的使用与 `.title()` 非常相似：

```python
>>> 'UPPERCASE'.lower()
'uppercase'
```

- 对于简单的替换（例如，将所有 *A 替换为 B* 或 *将 mine \ 替换为 ours*），使用 `.replace()`。此方法对于非常简单的情况很有用，但替换可能很容易变得棘手。注意替换的顺序以避免冲突和大小写敏感性问题。注意以下示例中的错误替换：

```python
>>> 'One ring to rule them all, one ring to find them, One ring to bring them all and in the darkness bind them.'.replace('ring', 'necklace')
'One necklace to rule them all, one necklace to find them, One necklace to bnecklace them all and in the darkness bind them.'
```

这类似于我们在正则表达式匹配代码中意外部分时会遇到的问题。稍后会有更多示例。有关更多信息，请参阅正则表达式食谱。

> 要换行文本，可以使用标准库中包含的 `textwrap` 模块，而不是手动计算字符。在此处查看文档：https://docs.python.org/3/library/textwrap.html。

如果你处理多种语言或任何非英语输入，学习 Unicode 和编码的基础知识非常有用。简而言之，鉴于世界上所有不同语言中存在大量字符，包括与拉丁字母无关的字母，例如中文或阿拉伯文，有一个标准试图涵盖所有这些字符，以便计算机能够正确理解它们。Python 3 大大改善了这种情况，使字符串的内部对象能够处理所有这些字符。Python 使用的默认编码，也是最常见和兼容的编码，目前是 UTF-8。

一篇关于 UTF-8 基础知识的好文章是这篇博客文章：https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/。

处理编码在从外部文件读取时仍然很重要，因为这些文件可能采用不同的编码（例如，CP-1252 或 windows-1252，这是旧版 Microsoft 系统常见的编码，或者 ISO 8859-15，这是行业标准）。

### 另请参阅

- 本章前面介绍的*使用格式化值创建字符串*方法，以了解字符串创建的基础知识。
- 本章后面介绍的*正则表达式简介*方法，以学习如何检测和提取文本中的模式。
- 本章后面介绍的*深入正则表达式*方法，以进一步了解正则表达式。
- *第 4 章，搜索和读取本地文件*中的*处理编码*方法，以了解不同类型的编码。

### 从结构化字符串中提取数据

在许多自动化任务中，我们需要处理以已知格式结构化的输入文本，并提取相关信息。例如，电子表格可能在一段文本中定义了一个百分比（例如 37.4%），我们希望以数字格式检索它，以便稍后应用（0.374，作为浮点数）。

在这个方法中，我们将学习如何处理包含产品内联信息的销售日志，例如产品是否已售出、其价格、获得的利润以及其他信息。

### 准备工作

假设我们需要解析存储在销售日志中的信息。我们将使用具有以下结构的销售日志：

```
[<ISO 格式的时间戳>] - SALE - PRODUCT: <产品 ID> - PRICE: $<销售价格>
```

让我们开始我们的自动化之旅

例如，一个特定的日志可能如下所示：

```
[2018-05-05T10:58:41.504054] - SALE - PRODUCT: 1345 - PRICE: $09.99
```

请注意，价格有一个前导零。所有价格都将有两位美元和两位美分。

> 标准 ISO 8601 定义了表示时间和日期的标准方式。它在计算领域被广泛使用，几乎任何计算机语言都可以解析和生成它。

在开始之前，我们需要激活虚拟环境：

```
$ source .venv/bin/activate
```

### 如何操作...

1. 在 Python 解释器中，进行以下导入。请记住按照*创建虚拟环境*方法中描述的那样激活您的虚拟环境：
    ```python
    >>> import delorean
    >>> from decimal import Decimal
    ```

2. 输入要解析的日志：
    ```python
    >>> log = '[2018-05-05T11:07:12.267897] - SALE - PRODUCT: 1345 - PRICE: $09.99'
    ```

3. 将日志拆分为其各个部分，这些部分由 ` - ` 分隔（注意破折号前后的空格）。我们忽略 SALE 部分，因为它不提供任何相关信息：
    ```python
    >>> divide_it = log.split(' - ')
    >>> timestamp_string, _, product_string, price_string = divide_it
    ```

4. 将时间戳解析为 datetime 对象：
    ```python
    >>> timestamp = delorean.parse(timestamp_string.strip('[]'))
    ```

5. 将 product_id 解析为整数：
    ```python
    >>> product_id = int(product_string.split(':')[-1])
    ```

6. 将价格解析为 Decimal 类型：
    ```python
    >>> price = Decimal(price_string.split('$')[-1])
    ```

7. 现在您拥有了所有原生 Python 格式的值：
    ```python
    >>> timestamp, product_id, price
    (Delorean(datetime=datetime.datetime(2018, 5, 5, 11, 7, 12, 267897), timezone='UTC'), 1345, Decimal('9.99'))
    ```

### 工作原理...

其基本工作原理是隔离每个元素，然后将它们解析为适当的类型。第一步是将完整日志拆分为更小的部分。` - ` 字符串是一个很好的分隔符，因为它将其拆分为四个部分——一个时间戳部分、一个仅包含单词 SALE 的部分、产品部分和价格部分。

对于时间戳，我们需要隔离 ISO 格式，它在日志中位于方括号内。这就是为什么时间戳需要去除方括号。我们使用 delorean 模块（前面已介绍）将其解析为 datetime 对象。

单词 SALE 被忽略。那里没有相关信息。

要隔离产品 ID，我们在冒号处分割产品部分。然后，我们将最后一个元素解析为整数：
```python
>>> product_string.split(':')
['PRODUCT', ' 1345']
>>> int(' 1345')
1345
```

要分割价格，我们使用美元符号作为分隔符，并将其解析为 Decimal 字符：
```python
>>> price_string.split('$')
['PRICE: ', '09.99']
>>> Decimal('09.99')
Decimal('9.99')
```

如下一节所述，不要将此值解析为 float 类型，因为它会改变精度。

### 更多内容...

这些日志元素可以组合成一个单一对象，有助于解析和聚合它们。例如，我们可以在 Python 代码中以以下方式定义一个类：
```python
class PriceLog(object):
    def __init__(self, timestamp, product_id, price):
        self.timestamp = timestamp
        self.product_id = product_id
        self.price = price
    def __repr__(self):
        return '<PriceLog ({}, {}, {})>'.format(self.timestamp,
                                               self.product_id,
                                               self.price)

    @classmethod
    def parse(cls, text_log):
        '''
        从格式为
        [<时间戳>] - SALE - PRODUCT: <产品 ID> - PRICE: $<价格>
        的文本日志解析为 PriceLog 对象
        '''
        divide_it = text_log.split(' - ')
        tmp_string, _, product_string, price_string = divide_it
        timestamp = delorean.parse(tmp_string.strip('[]'))
        product_id = int(product_string.split(':')[-1])
        price = Decimal(price_string.split('$')[-1])
        return cls(timestamp=timestamp, product_id=product_id,
                   price=price)
```

因此，解析可以如下进行：
```python
>>> log = '[2018-05-05T12:58:59.998903] - SALE - PRODUCT: 897 - PRICE: $17.99'
>>> PriceLog.parse(log)
<PriceLog (Delorean(datetime=datetime.datetime(2018, 5, 5, 12, 58, 59, 998903), timezone='UTC'), 897, 17.99)>
```

避免对价格使用 float 类型。浮点数存在精度问题，在聚合多个价格时可能会产生奇怪的错误，例如：
```python
>>> 0.1 + 0.1 + 0.1
0.30000000000000004
```

尝试以下两个选项以避免任何问题：

- **使用整数美分作为基本单位：** 这意味着将货币输入乘以 100 并将其转换为整数（或所用货币的任何正确分数单位）。在显示时，您可能仍希望更改基本单位。
- **解析为 Decimal 类型：** Decimal 类型保持固定精度，并按预期工作。您可以在 Python 文档 https://docs.python.org/3.8/library/decimal.html 中找到有关 Decimal 类型的更多信息。

如果您使用 Decimal 类型，请直接从字符串将结果解析为 Decimal。如果先将其转换为 float，您可能会将精度错误带入新类型。

### 另请参阅

- 本章前面介绍的*创建虚拟环境*方法，以了解如何启动具有已安装模块的虚拟环境。
- 本章后面介绍的*使用第三方工具——parse*方法，以进一步了解如何使用第三方工具处理文本。
- 本章后面介绍的*正则表达式简介*方法，以学习如何检测和提取文本中的模式。
- 本章后面介绍的*深入正则表达式*方法，以进一步了解正则表达式。

### 使用第三方工具——parse

虽然手动解析数据（如前一个方法所示）对于小字符串非常有效，但要调整精确公式以适应各种输入可能非常费力。如果输入有时多了一个破折号怎么办？或者它有一个可变长度的头部，取决于其中一个字段的大小怎么办？

一个更高级的选项是使用正则表达式，正如我们将在下一个方法中看到的那样。但 Python 中有一个很棒的模块叫做 parse (https://github.com/richardj0n3s/parse)，它允许我们反转格式字符串。这是一个强大的工具，易于使用，并大大提高了代码的可读性。

### 准备工作

将 `parse` 模块添加到我们虚拟环境中的 `requirements.txt` 文件，并重新安装依赖项，如 *创建虚拟环境* 配方中所示。

`requirements.txt` 文件应如下所示：

```
delorean==1.0.0
requests==2.22.0
parse==1.14.0
```

然后，在虚拟环境中重新安装模块：

```
$ pip install -r requirements.txt
...
Collecting parse==1.14.0
  Downloading https://files.pythonhosted.org/packages/4a/ea/9a16ff916752241aa80f1a5ec56dc6c6defc5d0e70af2d16904a9573367f/parse-1.14.0.tar.gz
...
Installing collected packages: parse
  Running setup.py install for parse ... done
Successfully installed parse-1.14.0
```

### 如何做...

1. 导入 `parse` 函数：
   ```python
   >>> from parse import parse
   ```

2. 定义要解析的日志，格式与 *从结构化字符串中提取数据* 配方中的相同：
   ```python
   >>> LOG = '[2018-05-06T12:58:00.714611] - SALE - PRODUCT: 1345 - PRICE: $09.99'
   ```

3. 分析它，并像尝试打印它时那样进行描述，如下所示：
   ```python
   >>> FORMAT = '[{date}] - SALE - PRODUCT: {product} - PRICE: ${price}'
   ```

4. 运行 `parse` 并检查结果：
   ```python
   >>> result = parse(FORMAT, LOG)
   >>> result
   <Result () {'date': '2018-05-06T12:58:00.714611', 'product': '1345', 'price': '09.99'}>
   >>> result['date']
   '2018-05-06T12:58:00.714611'
   >>> result['product']
   '1345'
   >>> result['price']
   '09.99'
   ```

5. 注意结果都是字符串。定义要解析的类型：
   ```python
   >>> FORMAT = '[{date:ti}] - SALE - PRODUCT: {product:d} - PRICE: ${price:05.2f}'
   ```

6. 再次解析：
   ```python
   >>> result = parse(FORMAT, LOG)
   >>> result
   <Result () {'date': datetime.datetime(2018, 5, 6, 12, 58, 0, 714611), 'product': 1345, 'price': 9.99}>
   >>> result['date']
   datetime.datetime(2018, 5, 6, 12, 58, 0, 714611)
   >>> result['product']
   1345
   >>> result['price']
   9.99
   ```

7. 为价格定义一个自定义类型，以避免浮点类型的问题：
   ```python
   >>> from decimal import Decimal
   >>> def price(string):
   ...     return Decimal(string)
   ...
   >>> FORMAT = '[{date:ti}] - SALE - PRODUCT: {product:d} - PRICE: ${price:price}'
   >>> parse(FORMAT, LOG, {'price': price})
   <Result () {'date': datetime.datetime(2018, 5, 6, 12, 58, 0, 714611), 'product': 1345, 'price': Decimal('9.99')}>
   ```

### 工作原理...

`parse` 模块允许我们定义一个格式字符串，该字符串在解析值时反转 `format` 方法。我们在创建字符串时讨论的许多概念都适用——将值放在括号中，在冒号后定义类型，等等。

默认情况下，如 *步骤 4* 所示，值被解析为字符串。这是分析文本时的一个良好起点。值可以被解析为更有用的原生类型，如 *如何做* 部分的 *步骤 5 和 6* 所示。请注意，虽然大多数解析类型与 Python 格式规范迷你语言中的类型相同，但也有一些其他可用类型，例如用于 ISO 格式时间戳的 `ti`。

> 虽然我们在本书中更宽泛地使用时间戳作为“日期和时间”的替代，但在最严格的意义上，它应该只用于数字格式，例如 *Unix 时间戳* 或 *纪元*，定义为自特定时间以来的秒数。

无论如何，使用包含其他格式的时间戳是很常见的，因为它是一个清晰易懂的概念，但在与他人共享信息时，请务必就格式达成一致。

如果原生类型不够用，我们可以定义自己的解析，如 *如何做* 部分的 *步骤 7* 所示。请注意，`price` 函数的定义接收一个字符串并返回适当的格式，在本例中是 `Decimal` 类型。

*从结构化字符串中提取数据* 配方的 *更多内容* 部分中描述的所有关于浮点数和价格信息的问题也适用于此。

### 更多内容...

时间戳也可以转换为 `delorean` 对象以保持一致性。此外，`delorean` 对象携带时区信息。添加与前一个配方相同的结构，得到以下对象，该对象能够解析日志：

```python
import parse
from decimal import Decimal
import delorean

class PriceLog(object):
    def __init__(self, timestamp, product_id, price):
        self.timestamp = timestamp
        self.product_id = product_id
        self.price = price

    def __repr__(self):
        return '<PriceLog ({}, {}, {})>'.format(self.timestamp,
                                               self.product_id,
                                               self.price)

    @classmethod
    def parse(cls, text_log):
        '''
        Parse from a text log with the format
        [<Timestamp>] - SALE - PRODUCT: <product id> - PRICE: $<price>
        to a PriceLog object
        '''
        def price(string):
            return Decimal(string)
        def isodate(string):
            return delorean.parse(string)
        FORMAT = ('[{timestamp:isodate}] - SALE - PRODUCT: {product:d}'
                  ' - PRICE: ${price:price}')
        formats = {'price': price, 'isodate': isodate}
        result = parse.parse(FORMAT, text_log, formats)
        return cls(timestamp=result['timestamp'],
                   product_id=result['product'],
                   price=result['price'])
```

因此，解析它会返回类似的结果：

```python
>>> log = '[2018-05-06T14:58:59.051545] - SALE - PRODUCT: 827 - PRICE: $22.25'
>>> PriceLog.parse(log)
<PriceLog (Delorean(datetime=datetime.datetime(2018, 6, 5, 14, 58, 59, 51545), timezone='UTC')), 827, 22.25)>
```

此代码包含在 GitHub 文件中，https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter01/price_log.py

所有支持的解析类型可以在文档中找到：https://github.com/r1chardj0n3s/parse#format-specification。

### 另请参阅

- 本章前面介绍的 *从结构化字符串中提取数据* 配方，了解如何使用简单过程从文本中获取信息。
- 本章后面介绍的 *正则表达式简介* 配方，了解如何从文本中检测和提取模式。
- 本章后面介绍的 *深入正则表达式* 配方，以进一步了解正则表达式。

### 正则表达式简介

**正则表达式**，或 **regex**，是一种用于 *匹配* 文本的模式。换句话说，它允许我们定义一个 **抽象字符串**（通常是一种结构化文本的定义），以检查其他字符串是否匹配。

最好用一个例子来描述它们。想象定义一个文本模式，即 *一个以大写 A 开头，之后仅包含小写 "n" 和 "a" 的单词。* 让我们展示一些可能的比较和结果：

| 要比较的文本 | 结果 |
|---|---|
| Anna | 匹配 |
| Bob | 不匹配（没有初始 A） |
| Alice | 不匹配（初始 A 后的 I 不是 n 或 a） |
| James | 不匹配（没有初始 A） |
| Aaan | 匹配 |
| Ana | 匹配 |
| Annnn | 匹配 |
| Aaaan | 匹配 |
| ANNA | 不匹配（N 不是 n 或 a） |

表 1.1：模式匹配示例

如果这听起来很复杂，那是因为它确实如此。正则表达式可能非常复杂，因为它们可能极其错综复杂且难以理解。但它们也非常有用，因为它们允许我们执行极其强大的模式匹配。

正则表达式的一些常见用途包括：

- **验证输入数据：** 例如，一个仅包含数字、破折号和括号的电话号码。
- **字符串解析：** 从结构化字符串中检索数据，例如日志或 URL。这与前面配方中描述的内容类似。
- **抓取：** 在长文本中查找某些内容的出现。例如，查找网页中的所有电子邮件。
- **替换：** 查找并用其他内容替换一个或多个单词。例如，将 *the owner* 替换为 *John Smith*。

### 准备工作

处理正则表达式的 Python 模块名为 `re`。我们将要介绍的主要函数是 `re.search()`，它返回一个 `match` 对象，其中包含与模式匹配的信息。

> 由于正则表达式模式也定义为字符串，我们将通过在它们前面加上 `r` 前缀来区分它们，例如 `r'pattern'`。这是 Python 将文本标记为原始字符串字面量的方式，这意味着其中的字符串按字面意思处理，没有任何转义。这意味着 `\` 用作反斜杠，而不是转义序列。例如，没有 `r` 前缀时，`\n` 表示换行符。

有些字符是特殊的，指的是诸如 *字符串结尾*、*任何数字*、*任何字符*、*任何空白字符* 等概念。

最简单的形式就是字面字符串。例如，正则表达式模式 `r'LOG'` 匹配字符串 `'LOGS'`，但不匹配字符串 `'NOT A MATCH'`。如果没有匹配，`re.search` 返回 `None`。如果有匹配，它返回一个特殊的 `Match` 对象：

```
>>> import re
>>> re.search(r'LOG', 'LOGS')
<_sre.SRE_Match object; span=(0, 3), match='LOG'>
>>> re.search(r'LOG', 'NOT A MATCH')
>>>
```

### 如何操作...

1. 导入 `re` 模块：
   ```python
   >>> import re
   ```

2. 然后，匹配一个不在字符串开头的模式：
   ```python
   >>> re.search(r'LOG', 'SOME LOGS')
   <_sre.SRE_Match object; span=(5, 8), match='LOG'>
   ```

3. 匹配一个仅在字符串开头的模式。注意 `^` 字符：
   ```python
   >>> re.search(r'^LOG', 'LOGS')
   <_sre.SRE_Match object; span=(0, 3), match='LOG'>
   >>> re.search(r'^LOG', 'SOME LOGS')
   >>>
   ```

4. 匹配一个仅在字符串结尾的模式。注意 `$` 字符：
   ```python
   >>> re.search(r'LOG$', 'SOME LOG')
   <_sre.SRE_Match object; span=(5, 8), match='LOG'>
   >>> re.search(r'LOG$', 'SOME LOGS')
   >>>
   ```

5. 匹配单词 'thing'（不包括 things），但不匹配 something 或 anything。注意第二个模式开头的 `\b`：
   ```python
   >>> STRING = 'something in the things she shows me'
   >>> match = re.search(r'thing', STRING)
   >>> STRING[:match.start()], STRING[match.start():match.end()], STRING[match.end():]
   ('some', 'thing', ' in the things she shows me')
   >>> match = re.search(r'\bthing', STRING)
   >>> STRING[:match.start()], STRING[match.start():match.end()], STRING[match.end():]
   ('something in the ', 'thing', 's she shows me')
   ```

6. 匹配一个仅包含数字和破折号的模式（例如电话号码）。检索匹配的字符串：
   ```python
   >>> re.search(r'[0123456789-]+', 'the phone number is 1234-567-890')
   <_sre.SRE_Match object; span=(20, 32), match='1234-567-890'>
   >>> re.search(r'[0123456789-]+', 'the phone number is 1234-567-890').group()
   '1234-567-890'
   ```

7. 简单地匹配一个电子邮件地址：
   ```python
   >>> re.search(r'\S+@\S+', 'my email is email.123@test.com').group()
   'email.123@test.com'
   ```

### 工作原理...

`re.search` 函数匹配一个模式，无论它在字符串中的位置如何。如前所述，如果未找到模式，它将返回 `None`，否则返回一个 `Match` 对象。

使用了以下特殊字符：

- `^`：标记字符串的开头
- `$`：标记字符串的结尾
- `\b`：标记单词的开头或结尾
- `\s`：标记任何非空白字符，包括 `*` 或 `$` 等字符

更多特殊字符将在下一个食谱 *深入正则表达式* 中展示。

在 *如何操作* 部分的 *步骤 6* 中，模式 `r'[0123456789-]+'` 由两部分组成。第一部分在方括号内，匹配 0 到 9 之间的任何单个字符（任何数字）和破折号（`-`）字符。之后的 `+` 号表示该字符可以出现一次或多次。这在正则表达式中称为 **量词**。这使得它可以匹配任何数字和破折号的组合，无论长度如何。

*步骤 7* 再次使用 `+` 号来匹配 `@` 之前和之后所需的任意多个字符。在这种情况下，字符匹配是 `\s`，它匹配任何非空白字符。

请注意，这里描述的简单电子邮件模式 *非常简单*，因为它会匹配无效的电子邮件，例如 `john@smith@test.com`。对于大多数用途，更好的正则表达式是 `r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"`。你可以访问 http://emailregex.com/ 找到它，以及更多信息的链接。

请注意，解析包含边界情况的有效电子邮件实际上是一个困难且具有挑战性的问题。前面的正则表达式应该适用于本书涵盖的大多数用途，但在像 Django 这样的通用框架项目中，电子邮件验证是一个非常长且难以阅读的正则表达式。

生成的匹配对象返回匹配模式开始和结束的位置（使用 `start` 和 `end` 方法），如 *步骤 5* 所示，它将字符串分割成匹配的部分，显示了两种匹配模式之间的区别。

> *步骤 5* 中显示的差异非常常见。试图捕获 GP（如全科医生）最终可能会捕获茄子和风笛！同样，`things\b` 不会捕获 things。请务必测试并进行适当的调整，例如捕获 `\bGP\b` 以仅匹配单词 GP。

可以通过调用 `group()` 来检索特定的匹配模式，如 *步骤 6* 所示。请注意，结果始终是一个字符串。可以使用我们之前见过的任何方法进行进一步处理，例如通过破折号将电话号码分成组：

```python
>>> match = re.search(r'[0123456789-]+', 'the phone number is 1234-567-890')
>>> [int(n) for n in match.group().split('-')]
[1234, 567, 890]
```

### 更多内容...

处理正则表达式可能既困难又复杂。请留出时间测试你的匹配，并确保它们按预期工作，以避免出现令人不快的意外。

> "有些人，当遇到问题时，会想 '我知道，我会用正则表达式。' 现在他们有两个问题了。"

– Jamie Zawinski

正则表达式在保持非常简单时效果最佳。通常，如果有特定的工具可以完成，优先使用它而不是正则表达式。一个非常清晰的例子是 HTML 解析；请参考 *第 3 章，构建你的第一个网页抓取应用程序*，以了解实现此目的的更好工具。

一些文本编辑器也允许我们使用正则表达式进行搜索。虽然大多数是面向编写代码的编辑器，例如 Vim、BBEdit 或 Notepad++，但它们也存在于更通用的工具中，例如 MS Office、Open Office 或 Google Documents。但要小心，因为特定的语法可能略有不同。

你可以使用一些工具交互式地检查你的正则表达式。一个免费在线可用的好工具是 https://regex101.com/，它显示每个元素并解释正则表达式。仔细检查你使用的是 Python 风格：

### 第 1 章

![图 1.1：使用 RegEx101 的示例](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_32_0.png)

图 1.1：使用 RegEx101 的示例

请注意，上图中的 **EXPLANATION** 框描述了 `\b` 匹配单词边界（单词的开头或结尾），而 *thing* 字面匹配这些字符。

正则表达式在某些情况下可能非常慢，甚至容易受到所谓的 **正则表达式拒绝服务攻击** 的影响，这是一种旨在混淆特定正则表达式的字符串，使其花费大量时间。在最坏的情况下，它甚至可能阻塞计算机。虽然自动化任务可能不会让你遇到这些问题，但要注意正则表达式处理时间过长的情况。

### 另请参阅

- 本章前面介绍的 *从结构化字符串中提取数据* 食谱，以学习从文本中提取信息的简单技术。
- 本章前面介绍的 *使用第三方工具 – parse* 食谱，以使用第三方工具从文本中提取信息。
- 本章后面介绍的 *深入正则表达式* 食谱，以进一步了解正则表达式。

### 深入正则表达式

在这个食谱中，我们将学习更多关于如何处理正则表达式的知识。在介绍基础知识之后，我们将更深入地研究模式元素，介绍组作为检索和解析字符串的更好方法，学习如何搜索同一字符串的多个出现，并处理较长的文本。

### 如何操作...

1. 导入 re：
   ```python
   >>> import re
   ```

2. 将电话模式作为组的一部分（在括号内）进行匹配。注意使用 `\d` 作为任何数字的特殊字符：
   ```python
   >>> match = re.search(r'the phone number is ([\d-]+)', '37: the phone number is 1234-567-890')
   >>> match.group()
   'the phone number is 1234-567-890'
   >>> match.group(1)
   '1234-567-890'
   ```

3. 编译一个模式，并使用 yes|no 选项捕获一个不区分大小写的模式：
   ```python
   >>> pattern = re.compile(r'The answer to question (\w+) is (yes|no)', re.IGNORECASE)
   >>> pattern.search('Naturally, the answer to question 3b is YES')
   <_sre.SRE_Match object; span=(10, 42), match='the answer to question 3b is YES'>
   >>> pattern.search('Naturally, the answer to question 3b is YES').groups()
   ('3b', 'YES')
   ```

### 工作原理...

引入的新特殊字符如下：

- \d：标记任意数字（0 到 9）。
- \s：标记任意空白字符，包括制表符和其他空白特殊字符。请注意，这与上一个食谱中介绍的 \S 相反。
- \w：标记任意字母（包括数字，但不包括句点等字符）。
- .（点）：标记任意字符。

> 请注意，相同字母的大写或小写表示相反的匹配，例如，\d 匹配数字，而 \D 匹配非数字。

要定义组，请将定义的组放在括号中。组可以单独检索。这使得它们非常适合匹配包含可变部分的更大模式，以便在下一步中处理，如步骤 2 所示。请注意与上一个食谱中步骤 6 模式的区别。在这种情况下，模式不仅是数字，还包括前缀文本，即使我们随后只提取数字：

```
>>> re.search(r'the phone number is ([\d-]+)', '37: the phone number is 1234-567-890')
<_sre.SRE_Match object; span=(4, 36), match='the phone number is 1234-567-890'>
>>> _.group(1)
'1234-567-890'
>>> re.search(r'[0123456789-]+', '37: the phone number is 1234-567-890')
<_sre.SRE_Match object; span=(0, 2), match='37'>
>>> _.group()
'37'
```

请记住，组 0（.group() 或 .group(0)）始终是整个匹配。其余的组按它们出现的顺序排列。

模式也可以被编译。如果模式需要反复匹配，这可以节省一些时间。要使用这种方式，请编译模式，然后使用该对象执行搜索，如步骤 3 和 4 所示。可以添加一些额外的标志，例如使模式不区分大小写。

步骤 4 的模式需要一点信息。它由两个组组成，由单个字符分隔。特殊字符 "."（点）意味着它匹配所有内容。在我们的示例中，它匹配句点、空格和逗号。第二个组是定义选项的直接选择，在本例中是美国州缩写。

第一个组以大写字母（[A-Z]）开头，并接受字母或空格的任意组合（[\w\s]+?），但不接受句点或逗号等标点符号。这匹配城市，包括由多个单词组成的城市。

最后的 +? 使字母的匹配非贪婪，匹配尽可能少的字符。这避免了一些问题，例如当城市之间没有标点符号时。看看我们没有为第二次匹配包含非贪婪限定符的结果，以及它如何包含两个元素：

```
>>> PATTERN = re.compile(r'([A-Z][\w\s]+?).(TX|OR|OH|MI)')
>>> TEXT = 'the jackalopes are the team of Odessa,TX while the knights are native of Corvallis OR and the mud hens come from Toledo,OH; the whitecaps have their base in Grand Rapids,MI'
>>> list(PATTERN.finditer(TEXT))[1]
<re.Match object; span=(73, 122), match='Corvallis OR and the mud hens come from Toledo,OH'>
```

请注意，此模式从任何大写字母开始，并持续匹配直到找到一个州，除非被标点符号分隔，这可能不是预期的，例如：

```
>>> re.search(r'([A-Z][\w\s]+?).(TX|OR|OH|MI)', 'This is a test, Escanaba MI')
<_sre.SRE_Match object; span=(16, 27), match='Escanaba MI'>
>>> re.search(r'([A-Z][\w\s]+?).(TX|OR|OH|MI)', 'This is a test with Escanaba MI')
<_sre.SRE_Match object; span=(0, 31), match='This is a test with Escanaba MI'>
```

步骤 4 还向你展示了如何在长文本中查找多个出现。虽然存在 .findall() 方法，但它不返回完整的匹配对象，而 .finditer() 会。正如现在在 Python 3 中常见的，.finditer() 返回一个迭代器，可以在 for 循环或列表推导式中使用。请注意，.search() 只返回模式的第一个出现，即使出现更多匹配：

```
>>> PATTERN.search(TEXT)
<_sre.SRE_Match object; span=(31, 40), match='Odessa,TX'>
>>> PATTERN.findall(TEXT)
[('Odessa', 'TX'), ('Corvallis', 'OR'), ('Toledo', 'OH')]
```

### 更多内容...

如果大小写互换，特殊字符可以反转。例如，我们使用的特殊字符的反向如下：

- \D：标记任意非数字。
- \W：标记任意非字母。
- \B：标记不在单词开头或结尾的位置。例如，r'thing\B' 将匹配 things 但不匹配 thing。

> 最常用的特殊字符通常是 \d（数字）和 \w（字母和数字），因为它们标记了要搜索的常见模式。

组也可以被命名。这使得它们更明确，但代价是组在以下形式中更冗长——(?P<groupname>PATTERN)。可以通过名称 .group(groupname) 或调用 .groupdict() 来引用组，同时保持其数字位置。

例如，步骤 4 的模式可以描述如下：

```
>>> PATTERN = re.compile(r'(?P<city>[A-Z][\w\s]+?). (?P<state>TX|OR|OH|MI)')
>>> match = PATTERN.search(TEXT)
>>> match.groupdict()
{'city': 'Odessa', 'state': 'TX'}
>>> match.group('city')
'Odessa'
>>> match.group('state')
'TX'
>>> match.group(1), match.group(2)
('Odessa', 'TX')
```

正则表达式是一个非常广泛的主题。有专门的技术书籍致力于它们，它们可能深奥得令人望而生畏。Python 文档是一个很好的参考（https://docs.python.org/3/library/re.html），可以了解更多。

如果你一开始感到有点畏惧，这是一种完全自然的感觉。仔细分析每个模式，将它们分成更小的部分，它们就会开始变得有意义。不要害怕运行正则表达式交互式分析器！

正则表达式可以非常强大和通用，但它们可能不是你试图实现目标的合适工具。我们已经看到了一些注意事项和具有微妙之处的模式。根据经验，如果一个模式开始变得复杂，就是时候寻找不同的工具了。也请记住之前的食谱以及它们提供的选项，例如 parse。

### 另请参阅

- 本章前面介绍的“介绍正则表达式”食谱，以了解使用正则表达式的基础知识。
- 本章前面介绍的“使用第三方工具 – parse”食谱，以了解从文本中提取信息的不同技术。

### 添加命令行参数

许多任务最好构建为命令行界面，接受不同的参数来改变其工作方式，例如，从提供的 URL 或其他 URL 抓取网页。Python 在标准库中包含了一个强大的 argparse 模块，可以以最小的努力创建丰富的命令行参数解析。

### 准备工作

在脚本中使用 `argparse` 的基本用法可以通过三个步骤展示：

1. 定义你的脚本将接受的参数，生成一个新的解析器。
2. 调用定义的解析器，返回一个包含所有结果参数的对象。
3. 使用参数调用脚本的入口点，该入口点将应用定义的行为。

尝试为你的脚本使用以下通用结构：

```
IMPORTS
def main(main parameters):
    DO THINGS

if __name__ == '__main__':
    DEFINE ARGUMENT PARSER
    PARSE ARGS
    VALIDATE OR MANIPULATE ARGS, IF NEEDED
    main(arguments)
```

`main` 函数使得很容易知道代码的入口点是什么。`if` 语句下的部分仅在文件被直接调用时执行，但在导入时不执行。我们将在所有步骤中遵循这一点。

### 操作步骤...

1. 创建一个脚本，该脚本将接受一个整数作为位置参数，并将打印该数量的井号符号。`recipe_cli_step1.py` 脚本如下，但请注意我们遵循了前面介绍的结构，并且 `main` 函数只是打印参数：

```
import argparse

def main(number):
    print('#' * number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int, help='A number')
```

### 让我们开始自动化之旅

```python
args = parser.parse_args()

main(args.number)
```

2.  调用脚本并检查参数是如何呈现的。不带参数调用脚本会显示自动帮助信息。使用自动参数 `-h` 可以显示扩展帮助：

```bash
$ python3 recipe_cli_step1.py
usage: recipe_cli_step1.py [-h] number
recipe_cli_step1.py: error: the following arguments are required: number

$ python3 recipe_cli_step1.py -h
usage: recipe_cli_step1.py [-h] number

positional arguments:
  number      A number

optional arguments:
  -h, --help  show this help message and exit
```

3.  带额外参数调用脚本，其行为符合预期：

```bash
$ python3 recipe_cli_step1.py 4
####

$ python3 recipe_cli_step1.py not_a_number
usage: recipe_cli_step1.py [-h] number
recipe_cli_step1.py: error: argument number: invalid int value: 'not_a_number'
```

4.  修改脚本以接受一个可选参数，用于指定要打印的字符。默认值将是 `"#"`。`recipe_cli_step2.py` 脚本将如下所示：

```python
import argparse

def main(character, number):
    print(character * number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int, help='A number')
    parser.add_argument('-c', type=str, help='Character to print',
                        default='#')

    args = parser.parse_args()
    main(args.c, args.number)
```

5.  帮助信息已更新，并且使用 `-c` 标志允许我们打印不同的字符：

```bash
$ python3 recipe_cli_step2.py -h
usage: recipe_cli_step2.py [-h] [-c C] number

positional arguments:
  number    A number

optional arguments:
  -h, --help show this help message and exit
  -c C Character to print

$ python3 recipe_cli_step2.py 4
####

$ python3 recipe_cli_step2.py 5 -c m
mmmmm
```

6.  添加一个标志，当其存在时改变行为。`recipe_cli_step3.py` 脚本如下：

```python
import argparse

def main(character, number):
    print(character * number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int, help='A number')
    parser.add_argument('-c', type=str, help='Character to print',
                        default='#')
    parser.add_argument('-U', action='store_true', default=False,
                        dest='uppercase',
                        help='Uppercase the character')
    args = parser.parse_args()

    if args.uppercase:
        args.c = args.c.upper()
    main(args.c, args.number)
```

7.  调用它时，如果添加了 `-U` 标志，字符将被大写：

```bash
$ python3 recipe_cli_step3.py 4 -c f
ffff
$ python3 recipe_cli_step3.py 4 -c f -U
FFFF
```

### 工作原理...

如“如何做”部分的步骤1所述，参数通过 `.add_arguments` 添加到解析器中。一旦所有参数定义完毕，调用 `parse_args()` 将返回一个包含结果的对象（如果出错则退出）。

每个参数都应添加帮助描述，但它们的行为可能有很大不同：

-   如果一个参数以 `-` 开头，它被视为可选参数，如步骤4中的 `-c` 参数。否则，它是一个位置参数，如步骤1中的 `number` 参数。
-   为了清晰起见，始终为可选参数定义默认值。如果你不定义，它将是 `None`，但这可能会引起混淆。
-   记住始终添加一个带有参数描述的 `help` 参数；帮助信息是自动生成的，如步骤2所示。
-   如果存在类型，它将被验证，例如步骤3中的 `number`。默认情况下，类型将是字符串。
-   `store_true` 和 `store_false` 操作可用于生成标志，即不需要任何额外参数的参数。将相应的默认值设置为相反的布尔值。这在步骤6和7中的 `U` 参数中得到了演示。
-   `args` 对象中属性的名称默认是参数的名称（如果存在破折号，则去掉）。你可以使用 `dest` 来更改它。例如，在步骤6中，命令行参数 `-U` 被描述为 `uppercase`。

在使用短参数（如单个字母）时，为内部使用更改参数名称非常有用。一个好的命令行界面会使用 `-c`，但在内部，使用更详细的标签（如 `configuration_file`）可能是个好主意。记住，显式优于隐式！

-   一些参数可以与其他参数协同工作，如 *步骤3* 所示。执行所有必要的操作，将 `main` 函数作为清晰简洁的参数传递。例如，在 *步骤3* 中，只传递了两个参数，但其中一个可能已被修改。

### 更多内容...

你也可以使用双破折号创建长参数，例如：

```python
parser.add_argument('-v', '--verbose', action='store_true',
                    default=False,
                    help='Enable verbose output')
```

这将同时接受 `-v` 和 `--verbose`，并将存储名称 `verbose`。

添加长名称是使界面更直观、更易于记忆的好方法。用过几次后，很容易记住有一个 `verbose` 选项，并且它以 `v` 开头。

处理命令行参数时的主要不便可能是参数过多。这会造成混淆。尝试使你的参数尽可能独立，不要在它们之间建立太多依赖关系；否则，处理组合可能会很棘手。

特别是，尽量不要创建超过两个的位置参数，因为它们没有助记符。位置参数也接受默认值，但在大多数情况下，这不会是预期的行为。

有关高级详细信息，请参阅 `argparse` 的 Python 文档 (https://docs.python.org/3/library/argparse.html)。

### 另请参阅

-   本章前面介绍的 *创建虚拟环境* 配方，了解如何创建一个安装第三方模块的环境。
-   本章前面介绍的 *安装第三方包* 配方，了解如何在虚拟环境中安装和使用外部模块。

# 2
### 让任务自动化变得简单

为了正确地自动化任务，我们需要一种方法让它们在适当的时间自动执行。需要手动启动的任务并不是真正完全自动化的。

然而，为了能够在处理更紧迫问题的同时让它们在后台运行，任务需要能够以 *即发即忘* 模式运行。我们应该能够监控它是否正确执行，确保我们捕获相关信息（例如，如果出现有趣的事情则接收通知），并知道运行期间是否发生了任何错误。

确保软件以高可靠性持续运行实际上是一件非常重要的事情。这是一个需要专业知识和人员的领域，他们通常被称为系统管理员、运维人员或 **SRE（站点可靠性工程）**。像亚马逊和谷歌这样的大型运营需要在确保一切 24/7 正常运行方面进行巨额投资。

本书的目标比这要小得多，因为大多数软件不需要这种高可用性。设计每年停机时间少于几秒的系统具有挑战性，但以合理的可靠性执行任务则容易得多。但是，请注意需要进行维护，并相应地进行规划。

在本章中，我们将介绍以下配方：

-   准备任务
-   设置 cron 作业
-   捕获错误和问题
-   发送电子邮件通知

我们将从回顾在自动化任务之前应该如何准备任务开始。

### 准备任务

一切都始于精确定义需要执行的工作，并以不需要人工干预即可运行的方式进行设计。

一些理想的特征点如下：

1.  **单一、清晰的入口点**：对于如何启动任务没有困惑。
2.  **清晰的参数**：如果有任何参数，它们应该尽可能明确。
3.  **无交互性**：不能停止执行以向用户请求信息。
4.  **结果应被存储**：以便在运行时间之外进行检查。
5.  **清晰的结果**：当我们自己监督程序执行时，我们可以接受更详细的结果，例如未标记的数据或额外的调试信息。然而，对于自动化任务，最终结果应尽可能简洁明了。
6.  **错误应被记录**：以分析出了什么问题。

命令行程序已经具备了许多这些特征。它总是有一个清晰的入口点，带有定义的参数，并且结果可以被存储，即使只是文本格式。并且可以通过确保有一个澄清参数的配置文件和一个输出文件来改进。

请注意，第6点是 *捕获错误和问题* 配方的目标，将在那里介绍。

为了避免交互性，请不要使用任何等待用户输入的函数，例如 `input`。记住要删除调试器断点！

### 准备工作

我们将从遵循一个结构开始，其中主函数将作为入口点，所有参数都提供给它。

这与 *第1章，让我们开始自动化之旅* 中的 *添加命令行参数* 配方中介绍的基本结构相同。

定义一个包含所有显式参数的主函数，可以满足第1点（单一、清晰的入口）和第2点（清晰的参数）。第3点（无交互性）也不难实现。

为了改进第2点（清晰的参数）和第5点（清晰的结果），我们将探讨如何从文件中读取配置，并将结果存储到另一个文件中。另一种选择是发送通知，例如电子邮件，这将在本章后面介绍。

### 如何操作...

1. 准备以下通过乘法计算两个数的命令行程序，并将其保存为 `prepare_task_step1.py`：
    ```python
    import argparse

    def main(number, other_number):
        result = number * other_number
        print(f'The result is {result}')

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-n1', type=int, help='A number', default=1)
        parser.add_argument('-n2', type=int, help='Another number', default=1)

        args = parser.parse_args()

        main(args.n1, args.n2)
    ```
    运行 `prepare_task_step1.py` 来计算两个数的乘积：
    ```bash
    $ python3 prepare_task_step1.py -n1 3 -n2 7
    The result is 21
    ```

2. 更新文件，定义一个包含两个参数的配置文件，并将其保存为 `prepare_task_step3.py`。请注意，定义配置文件会覆盖任何命令行参数：
    ```python
    import argparse
    import configparser

    def main(number, other_number):
        result = number * other_number
        print(f'The result is {result}')

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-n1', type=int, help='A number', default=1)
        parser.add_argument('-n2', type=int, help='Another number', default=1)

        parser.add_argument('--config', '-c', type=argparse.FileType('r'),
                            help='config file')

        args = parser.parse_args()
        if args.config:
            config = configparser.ConfigParser()
            config.read_file(args.config)
            # Transforming values into integers
            args.n1 = int(config['ARGUMENTS']['n1'])
            args.n2 = int(config['ARGUMENTS']['n2'])

        main(args.n1, args.n2)
    ```

3. 创建配置文件 `config.ini`。查看 `ARGUMENTS` 部分以及 `n1` 和 `n2` 的值：
    ```ini
    [ARGUMENTS]
    n1=5
    n2=7
    ```

4. 使用配置文件运行命令。请注意，如步骤2所述，配置文件会覆盖命令行参数：
    ```bash
    $ python3 prepare_task_step3.py -c config.ini
    The result is 35
    $ python3 prepare_task_step3.py -c config.ini -n1 2 -n2 3
    The result is 35
    ```

5. 添加一个参数以将结果存储到文件中，并将其保存为 `prepare_task_step6.py`：
    ```python
    import argparse
    import sys
    import configparser

    def main(number, other_number, output):
        result = number * other_number
        print(f'The result is {result}', file=output)

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-n1', type=int, help='A number', default=1)
        parser.add_argument('-n2', type=int, help='Another number', default=1)

        parser.add_argument('--config', '-c', type=argparse.FileType('r'),
                            help='config file')
        parser.add_argument('-o', dest='output', type=argparse.FileType('w'),
                            help='output file',
                            default=sys.stdout)

        args = parser.parse_args()
        if args.config:
            config = configparser.ConfigParser()
            config.read_file(args.config)
            # Transforming values into integers
            args.n1 = int(config['ARGUMENTS']['n1'])
            args.n2 = int(config['ARGUMENTS']['n2'])

        main(args.n1, args.n2, args.output)
    ```

6. 运行结果以检查其是否将输出发送到定义的文件。请注意，结果文件之外没有输出：
    ```bash
    $ python3 prepare_task_step6.py -n1 3 -n2 5 -o result.txt
    $ cat result.txt
    The result is 15
    $ python3 prepare_task_step6.py -c config.ini -o result2.txt
    $ cat result2.txt
    The result is 35
    ```

### 工作原理...

请注意，`argparse` 模块允许我们使用 `argparse.FileType` 类型将文件定义为参数，并自动打开它们。这非常方便，如果文件路径指向无效位置，将会引发错误。

请记住以正确的模式打开文件。在*步骤5*中，配置文件以读取模式（`r`）打开，输出文件以写入模式（`w`）打开，如果文件存在，这将覆盖该文件。你可能会发现追加模式（`a`）很有用，它会将下一段数据添加到现有文件的末尾。

`configparser` 模块使我们能够轻松使用配置文件。如*步骤2*所示，文件的解析很简单，如下所示：
```python
config = configparser.ConfigParser()
config.read_file(file)
```
然后，`config` 将可以像字典一样访问。它将配置文件的节作为键，内部是另一个包含每个配置值的字典。因此，`ARGUMENTS` 节中的值 `n2` 可以通过 `config['ARGUMENTS']['n2']` 访问。

请注意，值始终以字符串形式存储，需要转换为其他类型，例如整数。

如果你需要获取布尔值，请不要执行 `value = bool(config[raw_value])`，因为无论是什么，任何字符串都会被转换为 `True`；例如，字符串 `False` 是一个真字符串，因为它不是空的。使用空字符串也是一个糟糕的选择，因为它们非常令人困惑。请改用 `.getboolean` 方法，例如 `value = config.getboolean(raw_value)`。对于整数和浮点值，有类似的 `getint()` 和 `getfloat()` 方法。

Python 3 允许我们向 `print` 函数传递一个 `file` 参数，该参数将写入该文件。*步骤5*展示了将所有打印信息重定向到文件的用法。

请注意，默认参数是 `sys.stdout`，它会将值打印到终端（标准输出）。这意味着在不使用 `-o` 参数的情况下调用脚本将在屏幕上显示信息，这在开发和调试脚本时很有帮助：
```bash
$ python3 prepare_task_step6.py -c config.ini
The result is 35
$ python3 prepare_task_step6.py -c config.ini -o result.txt
$ cat result.txt
The result is 35
```

### 更多内容...

请参阅 Python 官方文档中 `configparser` 的完整文档：https://docs.python.org/3/library/configparser.html。

在大多数情况下，这个配置解析器应该足够了，但如果需要更强大的功能，你可以使用 YAML 文件作为配置文件。YAML 文件（https://learn.getgrav.org/advanced/yaml）作为配置文件非常常见。它们结构良好，可以直接解析，并考虑各种数据类型：

1. 将 PyYAML 添加到 `requirements.txt` 文件中：
   `PyYAML==5.3`
2. 在虚拟环境中安装依赖项：
   `$ pip install -r requirements.txt`
3. 创建 `prepare_task_yaml.py` 文件：
    ```python
    import yaml
    import argparse
    import sys

    def main(number, other_number, output):
        result = number * other_number
        print(f'The result is {result}', file=output)

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-n1', type=int, help='A number', default=1)
        parser.add_argument('-n2', type=int, help='Another number', default=1)
        parser.add_argument('-c', dest='config', type=argparse.FileType('r'),
                            help='config file in YAML format', default=None)
        parser.add_argument('-o', dest='output', type=argparse.FileType('w'),
                            help='output file', default=sys.stdout)

        args = parser.parse_args()
        if args.config:
            config = yaml.load(args.config, Loader=yaml.FullLoader)
            # No need to transform values
            args.n1 = config['ARGUMENTS']['n1']
            args.n2 = config['ARGUMENTS']['n2']

        main(args.n1, args.n2, args.output)
    ```
    > 请注意，PyYAML 的 `yaml.load()` 函数需要一个 `Loader` 参数。这是为了避免在 YAML 文件来自不受信任的来源时执行任意代码。除非你需要一组 YAML 语言特性，否则请始终使用 `yaml.SafeLoader`。如果来自 YAML 文件的任何部分数据来自不受信任的来源（例如用户输入），切勿使用 `yaml.SafeLoader` 以外的加载器。有关更多信息，请参阅本文：https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation。

4. 定义配置文件 `config.yaml`：
    ```yaml
    ARGUMENTS:
      n1: 7
      n2: 4
    ```

### 设置定时任务

Cron 是一种古老但可靠的命令执行方式。自 20 世纪 70 年代在 Unix 中出现以来，它一直是系统管理中执行维护任务的常用工具，例如释放磁盘空间、轮转日志文件、进行备份以及其他常见的重复性操作。

本方案专门针对 Unix 和类 Unix 操作系统，因此适用于 Linux 和 macOS。虽然在 Windows 中也可以安排任务，但其方式非常不同，使用的是任务计划程序，此处不作介绍。如果你能访问 Linux 服务器，这将是安排周期性任务的好方法。

其主要优点如下：

-   它几乎存在于所有 Unix 或 Linux 系统中，并且配置为自动运行。
-   它易于使用，尽管一开始可能有点令人困惑。
-   它广为人知。几乎所有参与管理任务的人都对其使用方法有大致了解。
-   它允许轻松执行周期性命令，且精度良好。

然而，它也有一些缺点，包括：

-   默认情况下，它可能不会提供太多反馈。检索输出、记录执行情况和错误至关重要。
-   任务应尽可能自包含，以避免环境变量问题，例如使用错误的 Python 解释器或执行路径错误。
-   它是 Unix 特有的。
-   仅支持固定的周期性时间。
-   它不控制同时运行多少个任务。每次倒计时结束时，它都会创建一个新任务。例如，一个需要 1 小时完成的任务，如果安排每 45 分钟运行一次，将会有 15 分钟的重叠时间，此时两个任务将同时运行。

不要低估最后一点的影响。同时运行多个耗时任务可能会对性能产生不良影响。耗时任务重叠可能导致竞态条件，使每个任务都阻止其他任务完成！请为任务留出充足的完成时间并密切关注它们。请记住，在同一主机上运行的任何其他程序的性能都可能受到影响，这可能包括任何服务，例如 Web 服务器、数据库和电子邮件。检查任务将执行的主机的负载情况，以避免意外情况。

### 准备工作

我们将创建一个名为 `cron.py` 的脚本：

```python
import argparse
import sys
from datetime import datetime
import configparser

def main(number, other_number, output):
    result = number * other_number
    print(f'[{datetime.utcnow().isoformat()}] The result is {result}',
          file=output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.
ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', '-c', type=argparse.
FileType('r'),
                        help='config file',
                        default='/etc/automate.ini')
    parser.add_argument('-o', dest='output', type=argparse.
FileType('w'),
                        help='output file',
                        default=sys.stdout)

    args = parser.parse_args()
    if args.config:
        config = configparser.ConfigParser()
        config.read_file(args.config)
        # Transforming values into integers
        args.n1 = int(config['ARGUMENTS']['n1'])
        args.n2 = int(config['ARGUMENTS']['n2'])

    main(args.n1, args.n2, args.output)
```

请注意以下细节：

1.  配置文件默认为 `/etc/automate.ini`。请重用上一个方案中的 `config.ini`。
2.  输出中添加了时间戳。这将明确显示任务何时运行。
3.  结果被添加到文件中，如文件以 `a` 模式打开所示。
4.  `ArgumentDefaultsHelpFormatter` 参数在使用 `-h` 参数打印帮助信息时，会自动添加有关默认值的信息。

检查任务是否产生预期结果，以及是否可以记录到已知文件：

```
$ python3 cron.py
[2020-01-15 22:22:31.436912] The result is 35
$ python3 cron.py -o /path/automate.log
$ cat /path/automate.log
[2020-01-15 22:28:08.833272] The result is 35
```

### 操作步骤...

1.  获取 Python 解释器的完整路径。这是你虚拟环境中的解释器：
    ```
    $ which python
    /your/path/.venv/bin/python
    ```

2.  准备要执行的 cron 任务。获取完整路径并检查是否可以无任何问题地执行。执行几次：
    ```
    $ /your/path/.venv/bin/python /your/path/cron.py -o /path/automate.log
    $ /your/path/.venv/bin/python /your/path/cron.py -o /path/automate.log
    ```

3.  检查结果是否已正确添加到结果文件：
    ```
    $ cat /path/automate.log
    [2020-01-15 22:28:08.833272] The result is 35
    [2020-01-15 22:28:10.510743] The result is 35
    ```

4.  编辑 crontab 文件，设置任务每 5 分钟运行一次：
    ```
    $ crontab -e
    ```
    ```
    */5 * * * * /your/path/.venv/bin/python /your/path/cron.py -o /path/automate.log
    ```

    请注意，这将使用你的默认命令行编辑器打开一个编辑终端。

    如果你尚未设置默认命令行编辑器，则默认情况下很可能是 Vim。如果你没有 Vim 经验，这可能会令人困惑。按 *I* 开始插入文本，完成后按 *Esc*。然后，使用 *wq* 保存文件并退出。有关 Vim 的更多信息，请参阅此介绍：https://null-byte.wonderhowto.com/how-to/intro-vim-unix-text-editor-every-hacker-should-be-familiar-with-0174674。

    有关如何更改默认命令行编辑器的信息，请参阅以下链接：https://www.a2hosting.com/kb/developer-corner/linux/setting-the-default-text-editor-in-linux。

5.  检查 crontab 内容。请注意，这会显示 crontab 内容，但不会设置为编辑：
    ```
    $ crontab -l
    */5 * * * * /your/path/.venv/bin/python /your/path/cron.py -o /path/automate.log
    ```

6.  等待并检查结果文件，查看任务执行情况：
    ```
    $ tail -F /path/automate.log
    [2020-01-17 21:20:00.611540] The result is 35
    [2020-01-17 21:25:01.174835] The result is 35
    [2020-01-17 21:30:00.886452] The result is 35
    ```

### 工作原理...

crontab 行由描述任务运行频率的部分（前六个元素）加上任务本身组成。前六个元素中的每一个代表不同的时间单位。其中大多数是星号，表示 *任意*：

```
* * * * * *
| | | | | |
| | | | | +-- 年                (范围: 1900-3000)
| | | | +---- 星期几             (范围: 1-7, 1 代表星期一)
| | | +------ 月份               (范围: 1-12)
| | +-------- 日期               (范围: 1-31)
| +---------- 小时               (范围: 0-23)
+------------ 分钟               (范围: 0-59)
```

因此，我们的行 `*/5 * * * * *` 表示在所有小时、所有天……所有年份中，每当分钟数能被 5 整除时执行。

以下是一些示例：

```
30 15 * * * * 表示“每天 15:30”
30 * * * * * 表示“每小时的 30 分钟”
0,30 * * * * * 表示“每小时的 0 分钟和 30 分钟”
*/30 * * * * * 表示“每半小时”
0 0 * * 1 * 表示“每周一 00:00”
```

### 另请参阅

-   *第 1 章，开始我们的自动化之旅* 中的 *命令行参数* 方案，以获取有关命令行参数的更多信息。
-   本章稍后介绍的 *发送电子邮件通知* 方案，以查看更完整的自动化任务示例。
-   *第 13 章，调试技术* 中的 *使用断点调试* 方案，以学习如何在自动执行代码之前对其进行调试。

### 轻松实现任务自动化

不要过度猜测。可以参考 https://crontab.guru/ 这样的速查表获取示例和调整方法。大多数常见用法都会在那里直接描述。你也可以编辑一个公式，然后获得一段描述性的文本，说明它将如何运行。

在描述完如何运行 cron 任务后，请包含执行任务的命令行，该命令行已在“如何操作...”部分的步骤2中准备好。

> 请注意，任务描述中包含了所有相关文件的完整路径——解释器、脚本和输出文件。这消除了与路径相关的所有歧义，并降低了可能出现错误的概率。一个非常常见的错误是 cron 无法确定这三个元素中的一个或多个。

### 更多内容...

默认输出（标准输出）的描述可能有点冗长。当调用 `python3 cron.py -h` 时，它会显示为：

```
-o OUTPUT   output file (default: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>)
```

这是标准输出（stdout）的描述。可以使用 `ArgumentParser` 中的 `formatter_class` 参数来更改参数的格式。这意味着你可以使用一个继承自可用默认格式化器的自定义格式化器来调整值的显示方式。请参阅 https://docs.python.org/2/library/argparse.html#formatter-class 上的文档。

如果在执行 crontab 时出现任何问题，你应该会收到一封系统邮件。这将在终端中显示为如下消息：

```
You have mail.
$
This can be read with mail:
$ mail
Mail version 8.1 6/6/93. Type ? for help.
"/var/mail/jaime": 1 message 1 new
>N 1 jaime@Jaimes-iMac-5K Fri Jun 17 21:15 20/914 "Cron <jaime@Jaimes-iM"
? 1
Message 1:
...
```

```
/usr/local/Cellar/python/3.8.1/Frameworks/Python.framework/Versions/3.8/Resources/Python.app/Contents/MacOS/Python: can't open file 'cron.py':
[Errno 2] No such file or directory
```

在下一个食谱中，我们将探索独立捕获错误的方法，以便任务能够顺利运行。

### 另请参阅

- *第1章，开启我们的自动化之旅* 中的 *添加命令行选项* 食谱，以了解命令行选项的基本概念。
- 本章接下来的 *捕获错误和问题* 食谱，学习如何存储执行过程中发生的事件。

### 捕获错误和问题

自动化任务的主要特点是其 *即发即忘* 的特性。我们不会主动查看结果，而是让它在后台运行。

本书中的大多数食谱都涉及外部信息，例如网页或其他报告，因此在运行时发现意外问题的可能性很高。本食谱将介绍一个自动化任务，该任务会将意外行为安全地存储到日志文件中，以便事后检查。

### 准备工作

作为起点，我们将使用一个任务，该任务将根据命令行中描述的两个数字进行除法运算。

这个任务与本章前面 *准备任务* 食谱的 *工作原理* 部分步骤5中介绍的任务非常相似。但是，我们不是将两个数字相乘，而是将它们相除。

### 如何操作...

1. 创建 `task_with_error_handling_step1.py` 文件，如下所示：

```python
import argparse
import sys

def main(number, other_number, output):
    result = number / other_number
    print(f'The result is {result}', file=output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n1', type=int, help='A number', default=1)
    parser.add_argument('-n2', type=int, help='Another number', default=1)
    parser.add_argument('-o', dest='output', type=argparse.FileType('w'),
                        help='output file', default=sys.stdout)

    args = parser.parse_args()

    main(args.n1, args.n2, args.output)
```

2. 执行几次以查看它对两个数字进行除法运算：

```
$ python3 task_with_error_handling_step1.py -n1 3 -n2 2
The result is 1.5
$ python3 task_with_error_handling_step1.py -n1 25 -n2 5
The result is 5.0
```

3. 检查除以0会产生错误，并且错误不会记录在结果文件中：

```
$ python task_with_error_handling_step1.py -n1 5 -n2 1 -o result.txt
$ cat result.txt
The result is 5.0
$ python task_with_error_handling_step1.py -n1 5 -n2 0 -o result.txt
Traceback (most recent call last):
  File "task_with_error_handling_step1.py", line 20, in <module>
    main(args.n1, args.n2, args.output)
  File "task_with_error_handling_step1.py", line 6, in main
    result = number / other_number
ZeroDivisionError: division by zero
$ cat result.txt
```

4. 创建 `task_with_error_handling_step4.py` 文件：

```python
import argparse
import sys
import logging

LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
LOG_LEVEL = logging.DEBUG


def main(number, other_number, output):
    logging.info(f'Dividing {number} between {other_number}')
    result = number / other_number
    print(f'The result is {result}', file=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n1', type=int, help='A number',
                        default=1)
    parser.add_argument('-n2', type=int, help='Another number', default=1)

    parser.add_argument('-o', dest='output', type=argparse.FileType('w'),
                        help='output file', default=sys.stdout)
    parser.add_argument('-l', dest='log', type=str, help='log file',
                        default=None)

    args = parser.parse_args()
    if args.log:
        logging.basicConfig(format=LOG_FORMAT, filename=args.log,
                            level=LOG_LEVEL)
    else:
        logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)

    try:
        main(args.n1, args.n2, args.output)
    except Exception as exc:
        logging.exception("Error running task")
        exit(1)
```

5. 运行它以检查它是否显示正确的 INFO 和 ERROR 日志，以及是否将其存储在日志文件中：

```
$ python3 task_with_error_handling_step4.py -n1 5 -n2 0
2020-01-19 14:25:28,849 root INFO Dividing 5 between 0
2020-01-19 14:25:28,849 root ERROR division by zero
Traceback (most recent call last):
  File "task_with_error_handling_step4.py", line 31, in <module>
    main(args.n1, args.n2, args.output)
  File "task_with_error_handling_step4.py", line 10, in main
    result = number / other_number
ZeroDivisionError: division by zero
$ python3 task_with_error_handling_step4.py -n1 5 -n2 0 -l error.log
$ python3 task_with_error_handling_step4.py -n1 5 -n2 0 -l error.log
$ cat error.log
2020-01-19 14:26:15,376 root INFO Dividing 5 between 0
2020-01-19 14:26:15,376 root ERROR division by zero
Traceback (most recent call last):
  File "task_with_error_handling_step4.py", line 33, in <module>
    main(args.n1, args.n2, args.output)
  File "task_with_error_handling_step4.py", line 11, in main
    result = number / other_number
ZeroDivisionError: division by zero
2020-01-19 14:26:19,960 root INFO Dividing 5 between 0
2020-01-19 14:26:19,961 root ERROR division by zero
Traceback (most recent call last):
  File "task_with_error_handling_step4.py", line 33, in <module>
    main(args.n1, args.n2, args.output)
  File "task_with_error_handling_step4.py", line 11, in main
    result = number / other_number
ZeroDivisionError: division by zero
```

### 工作原理...

为了正确捕获任何意外异常，主函数应该包装在 `try-except` 块中，如 *如何操作...* 部分步骤4中所实现的那样。将其与步骤1不包装代码的方式进行比较：

```python
try:
    main(...)
except Exception as exc:
    # Something went wrong
    logging.exception("Error running task")
    exit(1)
```

请注意，记录异常对于获取有关错误原因的信息非常重要。

这种异常被戏称为“宝可梦异常”，因为它可以捕获“所有”异常。它将在最高级别捕获任何意外错误。不要在代码的其他区域使用它，因为捕获所有内容可能会隐藏意外错误。至少，任何意外异常都应该被记录，以便进行进一步分析。

额外的步骤，通过使用 `exit(1)` 调用以状态1退出，通知操作系统我们的脚本出了问题。

`logging` 模块允许我们进行日志记录。请注意基本配置，它包括一个可选的日志存储文件、格式以及要显示的日志级别。

可用的日志级别从不太关键到更关键依次是：DEBUG、INFO、WARNING、ERROR 和 CRITICAL。日志级别将设置记录消息所需的最低严重性。例如，如果严重性设置为 WARNING，则 INFO 日志将不会被存储。

创建日志很容易。你可以通过调用 `logging.<logging level>` 方法（其中 logging level 是 DEBUG、INFO 等）来完成此操作。例如：

```python
>>> import logging
>>> logging.basicConfig(level=logging.INFO)
>>> logging.warning('a warning message')
WARNING:root:a warning message
>>> logging.info('an info message')
INFO:root:an info message
```

### 轻松实现任务自动化

```python
>>> logging.debug('a debug message')
>>>
```

请注意，严重性低于 INFO 的日志不会显示。使用级别定义来调整显示信息的多少。这可能会改变，例如，DEBUG 日志可能仅在开发任务时使用，而在运行时不显示。请注意，`task_with_error_handling_step4.py` 默认将日志级别定义为 DEBUG。

良好的日志级别定义是显示相关信息同时减少噪音的关键。有时设置起来并不容易，但特别是当涉及多人时，尝试就 WARNING 与 ERROR 的确切含义达成一致，以避免误解。

`logging.exception()` 是一个特殊情况，它将创建一个 ERROR 日志，但也会包含有关异常的信息，例如堆栈跟踪。

记得检查日志以发现错误。一个有用的提醒是在结果文件中添加注释，如下所示：

```python
try:
    main(args.n1, args.n2, args.output)
except Exception as exc:
    logging.exception(exc)
    print('There has been an error. Check the logs', file=args.output)
```

### 更多内容...

Python 日志模块具有许多功能，包括以下内容：

- 它提供了对日志格式的进一步调整，例如，包含生成日志的文件和行号。
- 它定义了不同的日志记录器对象，每个对象都有自己的配置，例如日志级别和格式。这允许我们以不同的方式将日志发布到不同的系统，不过，通常使用单个日志记录器对象以保持简单。
- 它将日志发送到多个地方，例如标准输出和文件，甚至远程日志记录器。
- 它自动轮换日志，在一定时间或大小后创建新的日志文件。这对于按天或周组织日志非常方便。它还允许压缩或删除旧日志。日志积累时会占用空间。
- 它从文件中读取标准日志配置。

与其创建复杂的日志记录规则，不如尝试使用适当的级别广泛记录日志，然后按级别进行过滤。

有关详细信息，请参阅 Python 模块文档 https://docs.python.org/3.8/library/logging.html，或教程 https://docs.python.org/3.8/howto/logging.html。

### 另请参阅

- *第 1 章，开始我们的自动化之旅* 中的 *添加命令行选项* 配方，描述命令行选项的基本元素。
- 本章前面介绍的 *准备任务* 配方，了解设计自动化任务时应遵循的策略。

### 发送电子邮件通知

电子邮件已成为日常生活中不可或缺的工具。可以说，如果自动化任务检测到某些内容，这是发送通知的最佳场所。另一方面，电子邮件收件箱已经充斥着垃圾邮件，因此请小心。

垃圾邮件过滤器也是现实。请注意向谁发送电子邮件以及要发送的电子邮件数量。电子邮件服务器或地址可能会被标记为垃圾邮件来源，所有电子邮件可能会被互联网悄悄丢弃。

本配方将向您展示如何使用现有电子邮件帐户发送单个电子邮件。

这种方法适用于作为自动化任务结果发送给少数人的备用电子邮件，但仅限于此。有关发送电子邮件（包括群组）的更多想法，请参阅 *第 9 章，处理通信渠道*。

### 准备工作

对于此配方，我们需要一个有效的电子邮件帐户设置，包括以下内容：

- 使用 SMTP 的有效电子邮件服务器；SMTP 是标准电子邮件协议
- 要连接的端口
- 地址
- 密码

这四个元素应该足以发送电子邮件。

一些电子邮件服务，例如 Gmail，会鼓励您设置 2FA，这意味着仅凭密码不足以发送电子邮件。通常，它们会允许您创建一个特定的应用程序密码，绕过 2FA 请求。请查看您的电子邮件提供商信息以获取选项。

您使用的电子邮件服务应在其文档中说明 SMTP 服务器是什么以及使用哪个端口。这也可以从电子邮件客户端中检索，因为它们是相同的参数。请查看您的提供商文档。在以下示例中，我们将使用 Gmail 帐户。

### 操作步骤...

1. 创建 `email_task.py` 文件，如下所示：

```python
import argparse
import configparser

import smtplib
from email.message import EmailMessage

def main(to_email, server, port, from_email, password):
    print(f'With love, from {from_email} to {to_email}')

    # Create the message
    subject = 'With love, from ME to YOU'
    text = '''This is an example test'''
    msg = EmailMessage()
    msg.set_content(text)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    # Open communication and send
    server = smtplib.SMTP_SSL(server, port)
    server.login(from_email, password)
    server.send_message(msg)
    server.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('email', type=str, help='destination email')
    parser.add_argument('-c', dest='config', type=argparse.FileType('r'),
                        help='config file', default=None)

    args = parser.parse_args()
    if not args.config:
        print('Error, a config file is required')
        parser.print_help()
        exit(1)

    config = configparser.ConfigParser()
    config.read_file(args.config)

    main(args.email,
         server=config['DEFAULT']['server'],
         port=config['DEFAULT']['port'],
         from_email=config['DEFAULT']['email'],
         password=config['DEFAULT']['password'])
```

2. 创建一个名为 `email_conf.ini` 的配置文件，其中包含您电子邮件帐户的详细信息。例如，对于 Gmail 帐户，请填写以下模板。模板可在 GitHub 上找到 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter02/email_conf.ini，但请务必用您的数据填写：

```ini
[DEFAULT]
email = EMAIL@gmail.com
server = smtp.gmail.com
port = 465
password = PASSWORD
```

3. 确保该文件不能被系统上的其他用户读取或写入，将文件权限设置为仅允许我们的用户。600 权限意味着文件所有者具有读写访问权限，其他人没有访问权限：
   ```
   $ chmod 600 email_conf.ini
   ```
4. 运行脚本发送测试电子邮件：
   ```
   $ python3 email_task.py -c email_conf.ini destination_email@server.com
   ```
5. 检查目标电子邮件的收件箱；应收到一封主题为 "With love, from ME to YOU" 的电子邮件。

### 工作原理...

前面的脚本中有两个关键步骤——消息的生成和发送。

消息主要需要包含收件人和发件人电子邮件地址，以及主题。如果内容是纯文本，如本例所示，调用 `.set_content()` 就足够了。然后可以发送整个消息。

从技术上讲，可以发送一封标记为来自与您用于发送的电子邮件地址不同的电子邮件。但不建议这样做，因为您的电子邮件提供商可能会认为这是试图冒充其他电子邮件。您可以使用回复标头允许回复到不同的帐户。

发送电子邮件需要您连接到指定的服务器并启动 SMTP 连接。SMTP 是电子邮件通信的标准。

步骤非常简单——配置服务器、登录、发送准备好的消息，然后退出。

如果您需要发送多条消息，您可以登录、发送多封电子邮件，然后退出，而不是每次都连接。

### 更多内容...

如果目标是更大的操作，例如营销活动，甚至是生产电子邮件，例如确认用户电子邮件，请参阅第 9 章，处理通信渠道。

本示例中使用的电子邮件消息内容非常简单，但实际邮件可以比这复杂得多。

收件人字段可以包含多个收件人。用逗号分隔，如下所示：

```
message['To'] = ','.join(recipients)
```

邮件可以定义为HTML格式，同时提供纯文本替代版本，并可包含附件。基本操作是设置一个`MIMEMultipart`对象，然后附加构成邮件的各个MIME部分：

```
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

message = MIMEMultipart()
part1 = MIMEText('some text', 'plain')
message.attach(part1)
with open('path/image', 'rb') as image:
    part2 = MIMEImage(image.read())
message.attach(part2)
```

最常见的SMTP连接是SMTP_SSL，它更安全，因为与服务器的所有通信都是加密的，并且始终需要登录名和密码。但是，也存在普通的、未经身份验证的SMTP——请查阅您的电子邮件提供商文档。

请记住，本示例旨在用于简单通知。如果附加不同的信息，邮件可能会变得相当复杂。如果您的目标是向客户或任何普通群体发送电子邮件，请尝试使用第9章“处理通信渠道”中的思路。

### 另请参阅

- *第1章，开始我们的自动化之旅*中的*添加命令行选项*示例，以了解命令行选项的基本概念。
- 本章前面介绍的*准备任务*示例，以了解设计自动化任务时应遵循的策略。

## 3 构建您的第一个网络爬虫应用

互联网和**万维网（WWW）**可能是当今最突出的信息来源。其中大部分信息可以通过HTTP获取。HTTP最初是为了共享超文本页面而发明的（因此得名**超文本传输协议**），这开启了万维网。

这个过程在我们每次请求网页时都会发生，因此几乎每个人都应该很熟悉。但我们也可以通过编程方式执行这些操作，以自动检索和处理信息。Python在其标准库中有一个HTTP客户端，但出色的`requests`模块使得获取网页变得非常容易。在本章中，我们将了解如何操作。

在本章中，我们将涵盖以下示例：

- 下载网页
- 解析HTML
- 爬取网络
- 订阅源
- 访问Web API
- 与表单交互
- 使用Selenium进行高级交互
- 访问受密码保护的页面
- 加速网络爬取

### 下载网页

下载网页的基本能力涉及对URL发起HTTP GET请求。这是任何网络浏览器的基本操作。

让我们快速回顾一下此操作的不同部分，因为它有三个不同的元素：

1. 使用HTTP协议。这涉及请求的结构方式。
2. 使用GET方法，这是最常见的HTTP方法。我们将在访问Web API示例中看到更多内容。
3. 一个完整的URL，描述页面的地址，包括服务器（例如：mypage.com）和路径（例如：/page）。

该请求将通过互联网路由到服务器并由服务器处理，然后将发回响应。此响应将包含一个状态码（如果一切正常，通常为200）和一个包含结果的主体，通常是一个包含HTML页面的文本。

大部分操作由用于执行请求的HTTP客户端自动处理。我们将在本示例中了解如何发出简单请求以获取网页。

> HTTP请求和响应也可以包含头部。头部包含有关请求本身的重要信息，例如请求的总大小、内容的格式、请求的日期以及使用的浏览器或服务器。

### 准备工作

使用出色的`requests`模块，获取网页非常简单。安装该模块：

```
$ echo "requests==2.23.0" >> requirements.txt
$ source .venv/bin/activate
(.venv) $ pip install -r requirements.txt
```

我们将下载位于http://www.columbia.edu/~fdc/sample.html的页面，因为它是一个简单的HTML页面，在文本模式下易于阅读。

### 操作步骤...

1. 导入requests模块：
    ```python
    >>> import requests
    ```

2. 使用以下URL向服务器发出请求，这将花费一两秒钟：
    ```python
    >>> url = 'http://www.columbia.edu/~fdc/sample.html'
    >>> response = requests.get(url)
    ```

3. 检查返回对象的状态码：
    ```python
    >>> response.status_code
    200
    ```

4. 检查结果的内容：
    ```python
    >>> response.text
    '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n<html>\n<head>\n
    ...

    完整主体

    ...

    <!-- 关闭上面开始的 <html> -->\n'
    ```

5. 检查发送和返回的头部：
    ```python
    >>> response.request.headers
    {'User-Agent': 'python-requests/2.22.0', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive'}
    >>> response.headers
    {'Date': 'Fri, 24 Jan 2020 19:04:12 GMT', 'Server': 'Apache', 'Last-Modified': 'Wed, 11 Dec 2019 12:46:44 GMT', 'Accept-Ranges': 'bytes', 'Vary': 'Accept-Encoding,User-Agent', 'Content-Encoding': 'gzip', 'Content-Length': '10127', 'Keep-Alive': 'timeout=15, max=100', 'Connection': 'Keep-Alive', 'Content-Type': 'text/html', 'Set-Cookie': 'BIGipServer-CUIT-www.columbia.edu-80-pool=1311259520.20480.0000; expires=Sat, 25-Jan-2020 01:04:12 GMT; path=/; HttpOnly'}
    ```

### 工作原理...

`requests`的操作非常简单；使用`GET`方法（本例中）对URL执行请求。这将返回一个可以分析的`result`对象。主要元素是`status_code`和主体内容，后者可以表示为`text`。

完整的请求可以在`request`属性中检查：

```
>>> response.request
<PreparedRequest [GET]>
>>> response.request.url
'http://www.columbia.edu/~fdc/sample.html'
```

完整的`requests`模块文档可以在此处找到：`https://requests.readthedocs.io/en/master/`。

在本章中，我们将展示`requests`库的更多功能。

### 更多内容...

所有HTTP状态码可以在此网页查看：`https://httpstatuses.com/`。它们也在`http.HTTPStatus`枚举中用方便的常量名称描述，例如`OK`、`NOT_FOUND`或`FORBIDDEN`。

> 最著名的错误状态码可以说是404，当URL描述的资源未找到时返回。尝试执行`requests.get('http://www.columbia.edu/invalid')`来体验一下。

状态码的一般结构是：

1. 1XX - 关于协议具体信息。
2. 2XX - 成功。
3. 3XX - 重定向。例如：URL不再有效，可在其他地方找到。应包含新的URL。
4. 4XX - 客户端错误。发送到服务器的信息存在某些错误（如格式错误）或客户端存在错误（例如，需要身份验证才能访问URL）。
5. 5XX – 服务器错误。服务器端出现错误；例如，服务器可能不可用或处理请求时可能存在错误。

请求可以使用**HTTPS（安全HTTP）**协议。它与HTTP相同，但确保请求和响应的内容是私密的。`requests`透明地处理它。

> 任何处理私人信息的网站都应使用HTTPS，以确保信息没有泄露。HTTP容易受到窃听。在可用的地方使用HTTPS。

### 另请参阅

- *第1章，开始我们的自动化之旅*中的*安装第三方包*示例，以了解安装外部模块的基础知识。
- 本章后面的*解析HTML*示例，以了解如何处理从服务器返回的信息。

### 解析HTML

下载原始文本或二进制文件是一个良好的起点，但网络的主要语言是HTML。

HTML是一种结构化语言，定义文档的不同部分，如标题和段落。HTML也是分层的，定义子元素。将原始文本解析为结构化文档的能力基本上就是从网页自动提取信息的能力。例如，如果某些文本包含在特定的HTML元素中，如`class div`或标题`h3`标签之后，则可能是相关的。

### 准备工作

我们将使用出色的`Beautiful Soup`模块将HTML文本解析为可分析的内存对象。我们需要使用最新版本的`beautifulsoup4`包以兼容Python 3。将该包添加到您的`requirements.txt`并在虚拟环境中安装依赖项：

```
$ echo "beautifulsoup4==4.8.2" >> requirements.txt
$ pip install -r requirements.txt
```

构建你的第一个网页抓取应用

### 如何操作...

1. 导入 BeautifulSoup 和 requests：
    ```python
    >>> import requests
    >>> from bs4 import BeautifulSoup
    ```

2. 设置要下载页面的 URL 并获取它：
    ```python
    >>> URL = 'http://www.columbia.edu/~fdc/sample.html'
    >>> response = requests.get(URL)
    >>> response
    <Response [200]>
    ```

3. 解析下载的页面：
    ```python
    >>> page = BeautifulSoup(response.text, 'html.parser')
    ```

4. 获取页面的标题。可以看到它与浏览器中显示的相同：
    ```python
    >>> page.title
    <title>Sample Web Page</title>
    >>> page.title.string
    'Sample Web Page'
    ```

5. 查找页面中所有的 h3 元素，以确定现有的章节：
    ```python
    >>> page.find_all('h3')
    [<h3><a name="contents">CONTENTS</a></h3>, <h3><a name="basics">1. Creating a Web Page</a></h3>, <h3><a name="syntax">2. HTML Syntax</a></h3>, <h3><a name="chars">3. Special Characters</a></h3>, <h3><a name="convert">4. Converting Plain Text to HTML</a></h3>, <h3><a name="effects">5. Effects</a></h3>, <h3><a name="lists">6. Lists</a></h3>, <h3><a name="links">7. Links</a></h3>, <h3><a name="tables">8. Tables</a></h3>, <h3><a name="install">9. Installing Your Web Page on the Internet</a></h3>, <h3><a name="more">10. Where to go from here</a></h3>]
    ```

6. 提取“特殊字符”章节的文本。当遇到下一个 `<h3>` 标签时停止：
    ```python
    >>> link_section = page.find('h3', attrs={'id': 'chars'})
    >>> section = []
    >>> for element in link_section.next_elements:
    ...     if element.name == 'h3':
    ...         break
    ...     section.append(element.string or '')
    ...
    >>> result = ''.join(section)
    >>> result
    '3. Special Characters\n\nHTML special "character entities" start with ampersand (&) and end with semicolon (;)'
    ```

### 工作原理...

第一步是下载页面。然后，可以解析原始文本，如*步骤 3*所示。生成的 `page` 对象包含解析后的信息。

> `html.parser` 解析器是默认的，但对于某些操作，它可能会有问题。例如，对于大型页面，它可能很慢，并且在渲染高度动态的网页时可能会遇到问题。你可以使用其他解析器，例如 `lxml`，它快得多，或者 `html5lib`，它更接近浏览器的操作方式。它们是外部模块，需要添加到 `requirements.txt` 文件中。

`BeautifulSoup` 允许我们搜索 HTML 元素。它可以用 `.find()` 搜索 HTML 元素的第一个出现，或者用 `.find_all()` 返回一个列表。在*步骤 5*中，它搜索了一个具有特定属性 `id=chars` 的特定标签 `<a>`。之后，它持续迭代 `.next_elements`，直到找到下一个 `h3` 标签，该标签标记了章节的结束。

每个元素的文本被提取出来，最后组合成一个单独的文本。注意 `or` 的使用，它避免了存储当元素没有文本时返回的 `None`。

> HTML 非常通用，可以有多种结构。本配方中介绍的情况是典型的，但划分章节的其他选项可以是将相关章节分组在一个大的 `<div>` 标签或其他元素中，甚至是纯文本。在找到提取网页精华内容的特定过程之前，需要进行一些实验。不要害怕尝试！

### 更多内容...

正则表达式可以用作 `.find()` 和 `.find_all()` 方法的输入。例如，此搜索使用 `h2` 和 `h3` 标签：

```python
>>> page.find_all(re.compile('^h(2|3)'))
[<h2>Sample Web Page</h2>, <h3 id="contents">CONTENTS</h3>, <h3 id="basics">1. Creating a Web Page</h3>, <h3 id="syntax">2. HTML Syntax</h3>, <h3 id="chars">3. Special Characters</h3>, <h3 id="convert">4. Converting Plain Text to HTML</h3>, <h3 id="effects">5. Effects</h3>, <h3 id="lists">6. Lists</h3>, <h3 id="links">7. Links</h3>, <h3 id="tables">8. Tables</h3>, <h3 id="viewing">9. Viewing Your Web Page</h3>, <h3 id="install">10. Installing Your Web Page on the Internet</h3>, <h3 id="more">11. Where to go from here</h3>]
```

另一个有用的 `find` 参数是使用 `class_` 参数包含 CSS 类。这将在本书后面展示。

完整的 Beautiful Soup 文档可以在这里找到：https://www.crummy.com/software/BeautifulSoup/bs4/doc/。

### 另请参阅

- *第 1 章，让我们开始自动化之旅*中的*安装第三方包*配方，了解如何安装外部模块。
- 本章前面的*下载网页*配方，了解请求网页的基础知识。

### 网络爬虫

鉴于超链接页面的性质，从已知位置开始并跟随链接到其他页面，是网络抓取工具库中一个非常重要的工具。

为此，我们爬取一个页面，寻找一个短语，并打印任何包含它的段落。我们只搜索属于单个站点的页面，例如：仅以 www.somesite.com 开头的 URL。我们不会跟随指向外部站点的链接。

### 准备工作

本配方建立在迄今为止介绍的概念之上，因此它将涉及下载和解析页面以搜索链接，然后继续下载。

> 在爬取网络时，请记住在下载时设置限制。爬取过多页面非常容易。正如任何检查维基百科的人可以确认的那样，互联网是潜在无限的。

我们将使用一个准备好的示例，该示例可在 GitHub 仓库中找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter03/test_site。下载整个站点并运行包含的脚本：

```
$ python simple_delay_server.py
```

这将在 URL http://localhost:8000 上提供该站点。你可以在浏览器中找到它。这是一个包含三个条目的简单博客。

### 构建你的第一个网页抓取应用

大部分内容并不有趣，但我们添加了几个包含关键词 `python` 的段落：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_79_0.png)

### 如何操作...

1. 完整的脚本 `crawling_web_step1.py` 可在 GitHub 上的以下链接找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter03/crawling_web_step1.py。这里显示了最相关的部分：

    ```python
    def process_link(source_link, text):
        logging.info(f'Extracting links from {source_link}')
        parsed_source = urlparse(source_link)
        result = requests.get(source_link)
        # Error handling. See GitHub for details
        ...
        page = BeautifulSoup(result.text, 'html.parser')
        search_text(source_link, page, text)
        return get_links(parsed_source, page)

    def get_links(parsed_source, page):
        '''Retrieve the links on the page'''
        links = []
        for element in page.find_all('a'):
            link = element.get('href')
            # Validate is a valid link. See GitHub for details
            ...
            links.append(link)
        return links
    ```

2. 搜索对 python 的引用，返回包含它的 URL 和段落的列表。注意，由于链接损坏，有几个错误：
    ```
    $ python crawling_web_step1.py http://localhost:8000/ -p python
    Link http://localhost:8000/: --> A smaller article , that contains a reference to Python
    Link http://localhost:8000/files/5eabef23f63024c20389c34b94dee593-1.html: --> A smaller article , that contains a reference to Python
    Link http://localhost:8000/files/33714fc865e02aeda2dabb9a42a787b2-0.html: --> This is the actual bit with a python reference that we are interested in.
    Link http://localhost:8000/files/archive-september-2018.html: --> A smaller article , that contains a reference to Python
    Link http://localhost:8000/index.html: --> A smaller article , that contains a reference to Python
    ```

3. 另一个很好的搜索词是 crocodile。试试看：
    ```
    $ python crawling_web_step1.py http://localhost:8000/ -p crocodile
    ```

### 工作原理...

让我们看看脚本的每个组成部分：

1. 一个在主函数中遍历所有找到的链接的循环：
    ```python
    def main(base_url, to_search):
        checked_links = set()
        to_check = [base_url]
        max_checks = 10
    ```

### 构建你的第一个网络爬虫应用

```python
while to_check and max_checks:
    link = to_check.pop(0)
    links = process_link(link, text=to_search)
    checked_links.add(link)
    for link in links:
        if link not in checked_links:
            checked_links.add(link)
            to_check.append(link)

max_checks -= 1
```

请注意，这里有一个10页的检索限制，代码正在检查任何要添加的新链接是否尚未被添加。

> 请注意，这两个元素充当了脚本的限制条件。我们不会重复下载同一个链接，并且会在某个点停止。

2. 下载并解析链接，在 `process_link` 函数中：

```python
def process_link(source_link, text):
    logging.info(f'Extracting links from {source_link}')
    parsed_source = urlparse(source_link)
    result = requests.get(source_link)
    if result.status_code != http.client.OK:
        logging.error(f'Error retrieving {source_link}: {result}')
        return []

    if 'html' not in result.headers['Content-type']:
        logging.info(f'Link {source_link} is not an HTML page')
        return []

    page = BeautifulSoup(result.text, 'html.parser')
    search_text(source_link, page, text)

    return get_links(parsed_source, page)
```

这里的代码下载文件并检查状态是否正确，以跳过诸如断开链接之类的错误。此代码还检查类型（如 `Content-Type` 中所述）是否为HTML页面，以跳过PDF和其他格式。最后，它将原始HTML解析为一个 `BeautifulSoup` 对象。
代码还使用 `urlparse` 解析源链接，以便稍后在*步骤4*中，它可以跳过所有对外部源的引用。`urlparse` 将URL分解为其组成元素：

```python
>>> from urllib.parse import urlparse
>>> urlparse('http://localhost:8000/files/b93bec5d9681df87e6e8d5703ed7cd81-2.html')
ParseResult(scheme='http', netloc='localhost:8000', path='/files/b93bec5d9681df87e6e8d5703ed7cd81-2.html', params='', query='', fragment='')
```

3. 代码在 `search_text` 函数中查找要搜索的文本：

```python
def search_text(source_link, page, text):
    '''Search for an element with the searched text and print it'''
    for element in page.find_all(text=re.compile(text, flags=re.IGNORECASE)):
        print(f'Link {source_link}: --> {element}')
```

这会在解析后的对象中搜索指定的文本。请注意，搜索是作为 `regex` 进行的，并且仅在页面的文本中进行。它会打印出匹配的结果，包括 `source_link`，引用找到匹配项的URL：

```python
for element in page.find_all(text=re.compile(text)):
    print(f'Link {source_link}: --> {element}')
```

4. `get_links` 函数检索页面上的所有链接：

```python
def get_links(parsed_source, page):
    '''Retrieve the links on the page'''
    links = []
    for element in page.find_all('a'):
        link = element.get('href')
        if not link:
            continue

        # Avoid internal, same page links
        if link.startswith('#'):
            continue

        if link.startswith('mailto:'):
            # Ignore other links like mailto
            # More cases like ftp or similar may be included
            continue

        # Always accept local links
        if not link.startswith('http'):
            netloc = parsed_source.netloc
            scheme = parsed_source.scheme
            path = urljoin(parsed_source.path, link)
            link = f'{scheme}://{netloc}{path}'

        # Only parse links in the same domain
        if parsed_source.netloc not in link:
            continue

        links.append(link)

    return links
```

这会在解析后的页面中搜索所有 `<a>` 元素并检索 `href` 元素，但仅限于具有此类 `href` 元素且是完整URL（以 `http` 开头）或本地链接的元素。这会移除不是URL的链接，例如 `#` 链接或页面内部的链接。

> 请记住，某些引用可能具有其他效果，例如 `mailto:` 协议。有一个检查可以避免 `mailto:` 协议，但也可能存在像 `ftp` 或 `irc` 这样的情况，尽管在实践中很少见。

进行额外的检查以确保链接与原始链接具有相同的源；只有这样，它们才会被注册为有效链接。`netloc` 属性检测链接是否来自与*步骤2*中生成的解析URL相同的URL域。

> 我们不会跟踪指向不同地址的链接（例如，http://www.google.com 的链接）。

最后，返回链接，它们将被添加到*步骤1*中描述的循环中。

### 还有更多...

可以实施进一步的过滤器；例如，所有以 `.pdf` 结尾的链接都可以被丢弃，因为它们可能指向PDF文件：

```python
# In get_links
if link.endswith('pdf'):
    continue
```

也可以使用 `Content-Type` 来确定以不同方式解析返回的对象。请记住，如果不发出请求，`Content-Type` 将不可用，这意味着代码无法在不请求链接的情况下跳过它们。PDF结果（`Content-Type: application/pdf`）将没有有效的 `response.text` 对象可供解析，但PDF结果可以通过其他方式解析。其他类型也是如此，例如CSV文件（`Content-Type: text/csv`）或可能需要解压缩的ZIP文件（`Content-Type: application/zip`）。我们将在后面看到如何处理这些。

### 另请参阅

- 本章前面的*下载网页*食谱，了解请求网页的基础知识。
- 本章前面的*解析HTML*食谱，了解如何解析HTML中的元素。

### 订阅源

RSS可能是互联网最大的秘密。它的黄金时代似乎是在2000年代，它使得订阅网站变得容易。它存在于许多网站中，并且非常有用。

从本质上讲，RSS是一种呈现一系列有序引用（通常是文章，但也包括其他元素，如播客剧集或YouTube发布）和发布时间的方式。这提供了一种非常自然的方式来了解自上次检查以来有哪些新文章，以及呈现一些关于它们的结构化数据，如标题和摘要。

在本食谱中，我们将介绍 `feedparser` 模块，并确定如何从RSS源获取数据。

> RSS不是唯一可用的源格式。还有一种叫做Atom的格式，但Atom和RSS或多或少是相同的。`feedparser` 也能够解析Atom，因此两种格式可以以相同的方式处理。

### 准备工作

我们需要将 `feedparser` 依赖项添加到我们的 `requirements.txt` 文件中并重新安装它：

```bash
$ echo "feedparser==5.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

源URL几乎可以在所有处理发布的页面上找到，包括博客、新闻、播客等。有时它们很容易找到，但有时它们有点隐蔽。搜索 `feed` 或 `RSS`。

大多数报纸和通讯社的RSS源按主题划分。对于我们的示例，我们将解析 **纽约时报** 主页源，`https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml`。主页源页面上还有更多可用的源：`https://archive.nytimes.com/www.nytimes.com/services/xml/rss/index.html`。

> 请注意，源可能受使用条款和条件的约束。以纽约时报为例，条款和条件在主页源页面末尾有描述。

请注意，此源变化相当频繁，这意味着链接的条目将与本书中的示例不同。

### 如何做...

1. 导入 `feedparser` 模块，以及 `datetime`、`delorean` 和 `requests`：

    ```python
    >>> import feedparser
    >>> import datetime
    >>> import delorean
    >>> import requests
    ```

2. 解析源（它将自动下载）并检查其最后更新时间。源信息，如源的标题，可以在 `feed` 属性中获取：

    ```python
    >>> rss = feedparser.parse('http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml')
    >>> rss.channel.updated
    'Friday, 24 Jan 2020 19:42:27 +0000'
    ```

3. 获取小于或等于6小时的条目：

    ```python
    >>> time_limit = delorean.parse(rss.channel.updated) - datetime.timedelta(hours=6)
    >>> entries = [entry for entry in rss.entries if delorean.parse(entry.published) > time_limit]
    ```

4. 一些返回的条目将比6小时更旧：

    ```python
    >>> len(entries)
    28
    >>> len(rss.entries)
    54
    ```

5. 检索有关条目的信息，例如 `title`。完整的条目URL可作为 `link` 获取。探索此特定源中的可用信息：

    ```python
    >>> entries[18]['title']
    'These People Really Care About Fonts'
    >>> entries[18]['link']
    'https://www.nytimes.com/2020/01/24/style/typography-font-design.html?emc=rss&partner=rss'
    >>> requests.get(entries[18].link)
    <Response [200]>
    ```

构建你的第一个网络爬虫应用

### 工作原理...

解析后的 `feed` 对象包含了条目的信息，以及关于 `feed` 本身的通用信息，例如其更新时间。`feed` 信息可以在 `feed` 属性中找到：

```
>>> rss.feed.title
'NYT > Top Stories'
```

每个条目都像一个字典一样工作，因此字段很容易获取。它们也可以作为属性访问，但将它们视为键可以让我们获取所有可用字段：

```
>>> entries[5].keys()
dict_keys(['title', 'title_detail', 'links', 'link', 'id', 'guidislink', 'media_content', 'summary', 'summary_detail', 'media_credit', 'credit', 'content', 'authors', 'author', 'author_detail', 'published', 'published_parsed', 'tags'])
```

处理 feed 的基本策略是解析它们并遍历条目，对它们是否有趣进行快速检查，例如，通过检查 `description` 或 `summary`。如果条目看起来值得，可以通过 `link` 字段完全下载它们。然后，为了避免重新检查条目，存储最新的发布日期，下次只检查更新的条目。

### 更多内容...

完整的 `feedparser` 文档可以在这里找到：`https://pythonhosted.org/feedparser/`。

可用的信息可能因 feed 而异。在《纽约时报》的例子中，有一个包含标签信息的 `tag` 字段，但这不是标准的。至少，条目会有标题、描述和链接。

> RSS feed 也是策划你自己的新闻来源选择的好方法。有很多优秀的 feed 阅读器可以做到这一点。

### 另请参阅

- *第 1 章，让我们开始自动化之旅* 中的 *安装第三方包* 配方，以了解安装外部模块的基础知识。
- 本章前面的 *下载网页* 配方，以了解更多关于发出请求和获取远程页面的信息。

### 访问 Web API

丰富的接口可以通过 Web 创建，允许通过 HTTP 进行强大的交互。最常见的接口是通过使用 JSON 的 RESTful API。这些基于文本的接口易于理解和编程，并使用通用的 **语言无关** 技术，这意味着它们可以在任何具有 HTTP 客户端模块的编程语言中访问，当然也包括 Python。

> 也使用 JSON 以外的格式，例如 XML。但 JSON 是一种非常简单且可读的格式，可以很好地转换为 Python 字典（以及其他语言的等效物）。目前，JSON 是 RESTful API 中最常见的格式。在此处了解更多关于 JSON 的信息：https://www.json.org/。

RESTful 的严格定义需要一些特定的特征，但非正式的 RESTful 定义是通过 HTTP URL 描述资源的系统。这意味着每个 URL 代表一个特定的资源，例如报纸上的一篇文章或房地产网站上的一个属性。然后可以通过 HTTP 方法（`GET` 查看，`POST` 创建，`PUT`/`PATCH` 编辑，`DELETE` 删除）来操作资源。

> 适当的 RESTful 接口需要具有某些特征。它们是一种创建接口的方式，不仅限于 HTTP 接口。你可以在此处阅读更多相关信息：https://codewords.recurse.com/issues/five/what-restful-actually-means。

使用 `requests` 与 RESTful 接口非常容易，因为它们包含原生的 JSON 支持。

### 准备工作

为了演示如何操作 RESTful API，我们将使用示例站点 https://jsonplaceholder.typicode.com/。它模拟了一个包含帖子、评论和其他常见资源的常见情况。我们将使用帖子和评论。要使用的 URL 如下：

```
# 所有帖子的集合
/posts
# 单个帖子。X 是帖子的 ID
/posts/X
# 帖子 X 的评论
/posts/X/comments
```

该站点为每个 URL 返回正确的结果。非常方便！

> 因为它是一个测试站点，数据不会被创建，但站点会返回所有正确的响应。

### 如何做...

1. 导入 requests：
   ```
   >>> import requests
   ```
2. 获取所有帖子的列表并显示最新的帖子：
   ```
   >>> result = requests.get('https://jsonplaceholder.typicode.com/posts')
   >>> result
   <Response [200]>
   >>> result.json()
   # 100 个帖子的列表未在此显示
   >>> result.json()[-1]
   {'userId': 10, 'id': 100, 'title': 'at nam consequatur ea labore ea harum', 'body': 'cupiditate quo est a modi nesciunt soluta\nipsa voluptas error itaque dicta in\nautem qui minus magnam et distinctio eum\naccusamus ratione error aut'}
   ```
3. 创建一个新帖子。查看新创建资源的 URL。调用也返回该资源：
   ```
   >>> new_post = {'userId': 10, 'title': 'a title', 'body':
   'something something'}
   >>> result = requests.post('https://jsonplaceholder.typicode.com/posts',
   json=new_post)
   >>> result
   <Response [201]>
   >>> result.json()
   {'userId': 10, 'title': 'a title', 'body': 'something something',
   'id': 101}
   >>> result.headers['Location']
   'http://jsonplaceholder.typicode.com/posts/101'
   ```
   注意，用于创建资源的 POST 请求返回 201，这是已创建的正确状态码。
4. 使用 GET 获取现有帖子：
   ```
   >>> result = requests.get('https://jsonplaceholder.typicode.com/posts/2')
   >>> result
   <Response [200]>
   >>> result.json()
   {'userId': 1, 'id': 2, 'title': 'qui est esse', 'body': 'est rerum tempore vitae\nsequi sint nihil reprehenderit dolor beatae ea dolores neque\nfugiat blanditiis voluptate porro vel nihil molestiae ut reiciendis\nqui aperiam non debitis possimus qui neque nisi nulla'}
   ```
5. 使用 PATCH 更新其值。检查返回的资源：
   ```
   >>> update = {'body': 'new body'}
   >>> result = requests.patch('https://jsonplaceholder.typicode.com/posts/2', json=update)
   >>> result
   <Response [200]>
   >>> result.json()
   {'userId': 1, 'id': 2, 'title': 'qui est esse', 'body': 'new body'}
   ```

### 工作原理...

通常访问两种资源——单个资源（https://jsonplaceholder.typicode.com/posts/X）和集合（https://jsonplaceholder.typicode.com/posts）

- 集合接受 GET 来检索集合的所有成员，接受 POST 来创建新资源
- 单个元素接受 GET 来获取元素，接受 PUT 和 PATCH 来编辑，接受 DELETE 来删除元素

所有可用的 HTTP 方法都可以在 `requests` 中调用。在前面的配方中，我们使用了 `.get()`，但 `.post()`、`.patch()`、`.put()` 和 `.delete()` 也是可用的。

返回的响应对象有一个 `.json()` 方法，用于从 JSON 解码结果。

同样，要发送信息，有一个 `json` 参数可用。这会将字典编码为 JSON 并将其发送到服务器。数据需要遵循资源的格式，否则可能会引发错误。

> GET 和 DELETE 不需要数据，而 PATCH、PUT 和 POST 需要通过请求体发送数据。

引用的资源将被返回，其 URL 在头部中可用。这在创建新资源时很有用，因为其 URL 事先未知。

> PATCH 和 PUT 之间的区别在于，后者替换整个资源，而前者执行部分更新。

### 更多内容...

RESTful API 非常强大，但也具有巨大的可变性。请查看特定 API 的文档以了解其详细信息。

### 另请参阅

- 本章前面的 *下载网页* 配方，以了解请求网页的基础知识
- *第 1 章，让我们开始自动化之旅* 中的 *安装第三方包* 配方，以了解安装外部模块的基础知识

### 与表单交互

网页中常见的元素是表单。表单是向网页发送值的一种方式，例如，在博客文章上创建新评论或提交购买。

浏览器呈现表单，以便你可以输入值并在按下提交或等效按钮后以单一操作发送它们。我们将在本配方中了解如何以编程方式创建此操作。

> 请注意，向网站发送数据通常比从其接收数据更微妙。例如，向网站发送自动评论非常符合 **垃圾邮件** 的定义。这意味着自动化可能更困难，因为它涉及考虑安全措施。请仔细检查你试图实现的是一个有效、合乎道德的用例。

### 准备工作

我们将针对测试服务器 https://httpbin.org/forms/post 进行操作，该服务器允许我们发送测试表单并返回提交的信息。

> 请注意，URL https://httpbin.org/forms/post 渲染表单，但内部调用 URL https://httpbin.org/post 来发送信息。我们将在本配方中使用这两个 URL。

### 构建你的第一个网页抓取应用

以下是一个用于订购披萨的示例表单：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_93_0.png)

你可以手动填写表单，并查看它以JSON格式返回的信息，包括所使用的浏览器等额外信息。

以下是生成的网页表单前端：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_94_0.png)

图 3.3：已填写的表单

以下截图显示了生成的网页表单后端：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_95_0.png)

图 3.4：返回的JSON内容

我们需要分析HTML以查看表单接受的数据。源代码如下：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_95_1.png)

图 3.5：源代码

检查输入的名称：`custname`、`custtel`、`custemail`、`size`（单选选项）、`topping`（多选复选框）、`delivery`（时间）和`comments`。

### 如何操作...

1. 导入`requests`、`BeautifulSoup`和`re`模块：
```python
>>> import requests
>>> from bs4 import BeautifulSoup
>>> import re
```

2. 获取表单页面，解析它，并打印输入字段。检查提交URL是否为`/post`（而不是`/forms/post`）：
```python
>>> response = requests.get('https://httpbin.org/forms/post')
>>> page = BeautifulSoup(response.text)
>>> form = page.find('form')
>>> {field.get('name') for field in form.find_all(re.compile('input|textarea'))}
{'delivery', 'topping', 'size', 'custemail', 'comments', 'custtel', 'custname'}
```
请注意，`textarea`是有效的输入，并在HTML格式中定义。

3. 将要提交的数据准备为字典。检查值是否与表单中定义的一致：
```python
>>> data = {'custname': "Sean O'Connell", 'custtel': '123-456-789', 'custemail': 'sean@oconnell.ie', 'size': 'small', 'topping': ['bacon', 'onion'], 'delivery': '20:30', 'comments': ''}
```

4. 提交值并检查响应是否与浏览器中返回的相同：
```python
>>> response = requests.post('https://httpbin.org/post', data)
>>> response
<Response [200]>
>>> response.json()
{'args': {}, 'data': '', 'files': {}, 'form': {'comments': '', 'custemail': 'sean@oconnell.ie', 'custname': "Sean O'Connell", 'custtel': '123-456-789', 'delivery': '20:30', 'size': 'small', 'topping': ['bacon', 'onion']}, 'headers': {'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate', 'Connection': 'close', 'Content-Length': '140', 'Content-Type': 'application/x-www-form-urlencoded', 'Host': 'httpbin.org', 'User-Agent': 'python-requests/2.22.0'}, 'json': None, 'origin': '89.100.17.159', 'url': 'https://httpbin.org/post'}
```

### 工作原理...

Requests直接编码并以配置的格式发送数据。默认情况下，它以`application/x-www-form-urlencoded`格式发送POST数据。

> 将requests的操作与*访问Web API*食谱进行比较，后者使用`json`参数以JSON格式显式发送数据。这意味着`Content-Type`是`application/json`而不是`application/x-www-form-urlencoded`。

这里的关键方面是尊重表单的格式以及可能返回错误的值（如果错误），通常是400错误，表示客户端存在问题。

### 更多内容...

除了遵循表单格式和输入有效值外，使用表单时的主要问题是防止垃圾邮件和滥用行为的多种方式。

你通常需要确保在提交表单之前已下载该表单，以避免提交多个表单或**跨站请求伪造（CSRF）**。

> CSRF，即利用你的浏览器已认证的事实，从一个页面向另一个页面发起恶意调用，是一个严重的问题——例如，你可能认为你正在进入一个关于可爱小狗的网站，但实际上它利用你已登录银行页面的事实，代表你执行金融操作：例如将你的储蓄转移到一个遥远的账户。这里有一个关于CSRF的很好描述：https://stackoverflow.com/a/33829607。浏览器中的新技术默认有助于解决这些CSRF问题。

要获取特定的令牌，你需要首先下载表单，如食谱所示，获取CSRF令牌的值，然后重新提交。请注意，令牌可能有不同的名称；这只是一个例子：
```python
>>> form.find(attrs={'name': 'token'}).get('value')
'ABCEDF12345'
```

### 另请参阅

- 本章前面的*下载网页*食谱，以了解请求网页的基础知识。
- 本章前面的*解析HTML*食谱，以跟进服务器返回信息的结构化。

### 使用Selenium进行高级交互

有时，非真实的东西无法工作。Selenium是一个用于在Web浏览器中实现自动化的项目。它被构想为一种自动测试的方式，但也可以用于自动化与网站的交互。

Selenium可以控制Safari、Chrome、Firefox、Internet Explorer或Microsoft Edge，尽管它需要为每种情况安装特定的驱动程序。我们将使用Chrome。

### 准备工作

我们需要为Chrome安装正确的驱动程序，称为`chromedriver`。可在此处获取：https://sites.google.com/a/chromium.org/chromedriver/。它适用于大多数平台。它还要求你已安装Chrome：https://www.google.com/chrome/。

将`selenium`模块添加到`requirements.txt`并安装它：
```
$ echo "selenium==3.141.0" >> requirements.txt
$ pip install -r requirements.txt
```

### 如何操作...

1. 导入Selenium，启动浏览器并加载表单页面。将打开一个页面以反映操作：
```python
>>> from selenium import webdriver
>>> browser = webdriver.Chrome()
>>> browser.get('https://httpbin.org/forms/post')
```
> 注意Chrome中的横幅，显示它正由自动化测试软件控制。

2. 在**Customer name**字段中添加一个值。记住它被称为`custname`：
```python
>>> custname = browser.find_element_by_name("custname")
>>> custname.clear()
>>> custname.send_keys("Sean O'Connell")
```
表单将更新：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_99_0.png)

3. 将披萨大小设置为中等：
```python
>>> for size_element in browser.find_elements_by_name("size"):
...     if size_element.get_attribute('value') == 'medium':
...         size_element.click()
...
>>>
```
这将设置**Pizza Size**单选按钮。

4. 添加培根和奶酪：
```python
>>> for topping in browser.find_elements_by_name('topping'):
...     if topping.get_attribute('value') in ['bacon', 'cheese']:
...         topping.click()
...
>>>
```
最后，复选框将显示为已标记：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_100_0.png)

5. 提交表单。页面将提交，结果将显示：
```python
>>> browser.find_element_by_tag_name('form').submit()
```
表单将被提交，服务器的结果将显示：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_100_1.png)

6. 关闭浏览器：
```python
>>> browser.quit()
```

### 工作原理...

*如何操作...*部分中的步骤1展示了如何创建Selenium页面并转到特定URL。

Selenium的工作方式与Beautiful Soup类似：你选择一个元素，然后操作它。Selenium中的选择器与Beautiful Soup中的选择器工作方式类似，最常见的是`find_element_by_id`、`find_element_by_class_name`、`find_element_by_name`、`find_element_by_tag_name`和`find_element_by_css_selector`。

有等效的`find_elements_by_x`操作，它们通过除第一个找到的元素之外的其他属性返回列表（例如`find_elements_by_tag_name`、`find_elements_by_name`等）。这在检查元素是否存在时也很有用。如果没有元素，`find_element`将引发错误，而`find_elements`将返回一个空列表。

可以通过`.get_attribute()`获取HTML属性（例如表单元素上的值）或`.text`来获取元素上的数据。

可以通过模拟发送按键来输入文本（使用`.send_keys()`方法）、发送点击（使用`.click()`）或提交表单（使用`.submit()`）来操作元素。请注意，`.click()`将以与鼠标点击相同的方式选择/取消选择。

最后，步骤6关闭浏览器。

### 更多内容...

以下是Python Selenium文档：http://selenium-python.readthedocs.io/。

对于每个元素，可以提取额外信息，例如`.is_displayed()`或`.is_selected()`。可以使用`.find_element_by_link_text()`和`.find_element_by_partial_link_text()`搜索文本。

有时，打开浏览器可能不方便。另一种选择是以无头模式启动浏览器并从那里操作它，如下所示：
```python
>>> from selenium.webdriver.chrome.options import Options
>>> chrome_options = Options()
```

### 访问受密码保护的页面

有时网页并非对公众开放，而是以某种方式受到保护。最简单的保护形式是使用基本HTTP身份验证，它几乎集成到所有Web服务器中，并实现了用户/密码模式。

### 准备工作

我们可以在 `https://httpbin.org` 测试这种身份验证。

它有一个路径 `/basic-auth/{user}/{password}`，该路径强制进行身份验证，并指定了用户名和密码。这对于理解身份验证的工作原理非常方便。

### 如何操作...

1. 导入 `requests`：
   ```python
   >>> import requests
   ```

2. 使用错误的凭据向URL发起GET请求。注意，我们将URL中的凭据设置为 `user` 和 `psswd`：
   ```python
   >>> requests.get('https://httpbin.org/basic-auth/user/psswd',
                    auth=('user', 'psswd'))
   <Response [200]>
   ```

3. 使用错误的凭据会返回401状态码（未授权）：
   ```python
   >>> requests.get('https://httpbin.org/basic-auth/user/psswd',
                    auth=('user', 'wrong'))
   <Response [401]>
   ```

4. 凭据也可以直接作为URL的一部分传递，在服务器之前用冒号和@符号分隔，如下所示：
   ```python
   >>> requests.get('https://user:psswd@httpbin.org/basic-auth/user/psswd')
   <Response [200]>
   >>> requests.get('https://user:wrong@httpbin.org/basic-auth/user/psswd')
   <Response [401]>
   ```

### 工作原理...

由于HTTP基本身份验证无处不在，`requests` 对其的支持非常简单。

“如何操作...”部分的步骤2和4展示了如何提供正确的密码。步骤3展示了密码错误时会发生什么。

> 请记住始终使用HTTPS，以确保密码的发送是保密的。如果使用HTTP，密码将在互联网上明文发送，可能被监听元素捕获。

### 更多内容...

将用户名和密码添加到URL中在浏览器中也有效。尝试直接访问页面，会看到一个要求输入用户名和密码的对话框：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_104_0.png)

图3.9：用户凭据页面

当使用包含用户名和密码的URL（`https://user:psswd@httpbin.org/basic-auth/user/psswd`）时，对话框不会出现，并且会自动进行身份验证。

如果需要访问多个页面，可以在 `requests` 中创建一个会话并设置身份验证参数，以避免在每个地方都输入它们：

```python
>>> s = requests.Session()
>>> s.auth = ('user', 'psswd')
>>> s.get('https://httpbin.org/basic-auth/user/psswd')
<Response [200]>
```

### 另请参阅

- 本章前面的 *下载网页* 食谱，了解请求网页的基础知识。
- 本章前面的 *访问Web API* 食谱，了解如何访问位于身份验证墙后面的API。

### 加速网络爬取

从网页下载信息所花费的大部分时间通常是在等待。一个请求从我们的计算机发送到远程服务器进行处理，在响应生成并返回到我们的计算机之前，我们对此无能为力。

在执行本书中的食谱时，你会注意到 `requests` 调用中存在等待，通常是一到两秒钟。但计算机在等待时可以做其他事情，包括同时发出更多请求。在这个食谱中，我们将看到如何并行下载一组页面，并等待它们全部完成。我们将使用一个故意设置得很慢的服务器来展示为什么值得正确处理这个问题。

### 准备工作

我们将获取代码来爬取和搜索关键词，利用Python 3的 `futures` 功能同时下载多个页面。

`future` 是一个表示值承诺的对象。这意味着你在代码在后台执行时立即收到一个对象——只有在专门请求其 `.result()` 时，代码才会等待结果可用。

> 如果结果在那时已经可用，那会更快。可以将此操作想象成在做其他任务时把东西放进洗衣机。有可能在我们完成其他家务时，洗衣已经完成了。

要生成一个 `future`，你需要一个后台引擎，称为 **执行器**。创建后，向其 `submit` 一个函数和参数以获取一个 `future`。获取结果可以尽可能延迟，允许连续生成多个 `futures`；然后我们可以等待所有任务完成并并行执行它们。这是创建一个、等待其完成、再创建另一个等操作的替代方案。

有几种方法可以创建执行器；在这个食谱中，我们将使用 `ThreadPoolExecutor`，它使用线程。

我们将使用一个准备好的示例，可在以下GitHub仓库中找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter03/test_site。下载整个站点并运行包含的脚本：

```
$ python simple_delay_server.py -d 2
```

这将在URL http://localhost:8000 上提供该站点。你可以在浏览器中查看它。这是一个简单的博客，有三个条目。大部分内容并不有趣，但我们添加了几个包含关键词python的段落。参数 -d 2 使服务器故意变慢，模拟不良连接。

### 如何操作...

1. 编写以下脚本，speed_up_step1.py。完整代码可在GitHub的Chapter03目录中找到（https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter03/speed_up_step1.py）。这里只展示最相关的部分。它基于 crawling_web_step1.py：

```python
...
def process_link(source_link, text):
    ...
    return source_link, get_links(parsed_source, page)
...

def main(base_url, to_search, workers):
    checked_links = set()
    to_check = [base_url]
    max_checks = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while to_check:
            futures = [executor.submit(process_link, url, to_search)
                      for url in to_check]
            to_check = []
            for data in concurrent.futures.as_completed(futures):
                link, new_links = data.result()

                checked_links.add(link)
                for link in new_links:
                    if link not in checked_links and link not in to_check:
                        to_check.append(link)

            max_checks -= 1
            if not max_checks:
                return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ...
    parser.add_argument('-w', type=int, help='Number of workers',
                        default=4)
    args = parser.parse_args()

    main(args.u, args.p, args.w)
```

2. 注意 `main` 函数中的差异。此外，添加了一个额外的参数（并发工作线程数），并且 `process_link` 函数现在返回源链接。

3. 运行 `crawling_web_step1.py` 脚本以获取时间基准。注意，为了清晰起见，此处已移除输出：

```
$ time python crawling_web_step1.py http://localhost:8000/
... REMOVED OUTPUT
real    0m12.221s
user    0m0.160s
sys     0m0.034s
```

4. 使用一个工作线程运行新脚本，这将使其比原始脚本更慢：

```
$ time python speed_up_step1.py -w 1
... REMOVED OUTPUT
real    0m16.403s
user    0m0.181s
sys     0m0.068s
```

5. 增加工作线程数：

```
$ time python speed_up_step1.py -w 2
... REMOVED OUTPUT
real    0m10.353s
```

### 工作原理...

创建并发请求的主引擎是主函数。请注意，其余代码基本保持不变（除了在 `process_link` 函数中返回源链接）。

> 这种改动在适配并发时其实很常见。并发任务需要返回所有相关数据，因为它们无法依赖有序的上下文。

以下是处理并发引擎的相关代码部分：

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    while to_check:
        futures = [executor.submit(process_link, url, to_search)
                  for url in to_check]
        to_check = []
        for data in concurrent.futures.as_completed(futures):
            link, new_links = data.result()
            checked_links.add(link)
            for link in new_links:
                if link not in checked_links and link not in to_check:
                    to_check.append(link)
        max_checks -= 1
```

构建你的第一个网页抓取应用

```python
if not max_checks:
    return
```

`with` 上下文管理器创建了一个工作线程池，并指定了线程数量。在其中，创建了一个包含所有待检索 URL 的 futures 列表。`.as_completed()` 函数返回已完成的 futures，然后需要进行一些处理来获取新发现的链接，并检查它们是否需要被添加到待检索列表中。这个过程类似于 *抓取网页* 食谱中介绍的方法。

这个过程会重复进行，直到检索到足够的链接或没有链接可检索。请注意，链接是分批检索的；第一次处理基础链接并检索所有链接。在第二次迭代中，将请求所有这些链接。一旦它们全部下载完成，就会处理新的一批。

> 处理并发请求时，请记住它们在两次执行之间可能会改变顺序。如果一个请求花费的时间稍多或稍少，可能会影响检索信息的顺序。由于我们在下载 10 页后停止，这也意味着这 10 页可能是不同的。

### 更多内容...

Python 中完整的 `futures` 文档可以在这里找到：`https://docs.python.org/3/library/concurrent.futures.html`。

> 正如你在 *如何做...* 部分的 *步骤 4* 和 *步骤 5* 中看到的，正确确定工作线程的数量可能需要一些测试。某些数字可能会因为管理开销的增加而使过程变慢。不要害怕尝试！

在 Python 世界中，还有其他方法可以进行并发 HTTP 请求。有一个原生的请求模块允许我们使用 `futures`，叫做 `requests-futures`。可以在这里找到：`https://github.com/ross/requests-futures`。

另一种选择是使用异步编程。这种工作方式最近受到了很多关注，因为它在处理许多并发调用的情况下可以非常高效，但生成的编码方式与传统方式不同，需要一些时间来适应。Python 包含了 `asyncio` 模块来以这种方式工作，并且有一个很好的模块叫做 `aiohttp` 用于处理 HTTP 请求。你可以在以下位置找到更多关于 `aiohttp` 的信息：`https://aiohttp.readthedocs.io/en/stable/client_quickstart.html`。

关于异步编程的优秀介绍可以在这篇文章中找到：https://djangostars.com/blog/asynchronous-programming-in-python-asyncio/。

### 另请参阅

- 本章前面的 *抓取网页* 食谱，了解此食谱的效率较低的替代方案。
- 本章前面的 *下载网页* 食谱，学习请求网页的基础知识。

# 4 搜索和读取本地文件

在本章中，我们将介绍从文件中读取信息的基本操作，从搜索和存储在不同目录和子目录中的文件开始。然后，我们将描述一些最常见的文件类型以及如何读取它们，包括原始文本文件、PDF 和 Word 文档等格式。

最后一个食谱将在目录树中递归地搜索不同种类文件中的一个单词。

在本章中，我们将涵盖以下食谱：

- 爬取和搜索目录
- 读取文本文件
- 处理编码
- 读取 CSV 文件
- 读取日志文件
- 读取文件元数据
- 读取图像
- 读取 PDF 文件
- 读取 Word 文档
- 扫描文档中的关键字

我们将从访问目录树中的所有文件开始。

### 爬取和搜索目录

在这个食谱中，我们将学习如何递归扫描目录以获取其中包含的所有文件。这将包括子目录中的所有文件。匹配的文件可以是特定类型的，比如文本文件，或者每一个文件。

这通常是处理文件时的起始操作，用于检测所有现有的文件。

### 准备工作

让我们从创建一个包含一些文件信息的测试目录开始：

```bash
$ mkdir dir
$ touch dir/file1.txt
$ touch dir/file2.txt
$ mkdir dir/subdir
$ touch dir/subdir/file3.txt
$ touch dir/subdir/file4.txt
$ touch dir/subdir/file5.pdf
$ touch dir/file6.pdf
```

所有文件都将是空的；我们在这个食谱中仅使用它们来发现它们。请注意，有四个文件具有 `.txt` 扩展名，两个文件具有 `.pdf` 扩展名。

> 这些文件也可以在 GitHub 仓库中找到：
https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter04/documents/dir。

进入创建的 `dir` 目录

```bash
$ cd dir
```

### 如何做...

1. 打印 `dir` 目录和子目录中的所有文件名：

```python
>>> import os
>>> for root, dirs, files in os.walk('.'):
...     for file in files:
...         print(file)
...
file1.txt
file2.txt
file6.pdf
file3.txt
file4.txt
file5.pdf
```

2. 打印文件的完整路径，与根目录连接：

```python
>>> for root, dirs, files in os.walk('.'):
...     for file in files:
...         full_file_path = os.path.join(root, file)
...         print(full_file_path)
...
./dir/file1.txt
./dir/file2.txt
./dir/file6.pdf
./dir/subdir/file3.txt
./dir/subdir/file4.txt
./dir/subdir/file5.pdf
```

3. 只打印 .pdf 文件：

```python
>>> for root, dirs, files in os.walk('.'):
...     for file in files:
...         if file.endswith('.pdf'):
...             full_file_path = os.path.join(root, file)
...             print(full_file_path)
...
./dir/file6.pdf
./dir/subdir/file5.pdf
```

4. 只打印包含偶数的文件：

```python
>>> import re
>>> for root, dirs, files in os.walk('.'):
...     for file in files:
...         if re.search(r'[13579]', file):
...             full_file_path = os.path.join(root, file)
...             print(full_file_path)
...
./dir/file1.txt
./dir/subdir/file3.txt
./dir/subdir/file5.pdf
```

### 工作原理...

`os.walk()` 遍历整个目录及其下的所有子目录，返回所有文件。对于每个目录，它返回一个包含该目录、其下的任何子目录以及所有文件的元组：

```python
>>> for root, dirs, files in os.walk('.'):
...     print(root, dirs, files)
...
. ['dir'] []
./dir ['subdir'] ['file1.txt', 'file2.txt', 'file6.pdf']
./dir/subdir [] ['file3.txt', 'file4.txt', 'file5.pdf']
```

`os.path.join()` 函数允许我们连接两个路径，例如基础路径和文件。

由于路径作为纯字符串返回，因此可以进行任何类型的过滤，如步骤 3 所示。在步骤 4 中，可以使用正则表达式的全部功能进行过滤。

在下一个食谱中，我们将处理文件的内容，而不仅仅是文件名。

### 更多内容...

在这个食谱中，返回的文件没有以任何方式打开或修改。此操作是只读的。文件可以按照以下食谱中的描述打开。

> 请注意，在遍历目录时更改其结构可能会影响结果。如果需要在遍历目录树时进行一些文件维护，比如复制或移动文件，最好将其存储在不同的目录中。

### `os.path` 模块

`os.path` 模块还有其他有趣的函数。我们讨论了 `.join()`，但其他包含的实用工具包括：

- `os.path.abspath()`，返回文件的绝对路径。
- `os.path.split()`，在目录和文件之间分割路径：
  ```python
  >>> os.path.split('/a/very/long/path/file.txt')
  ('/a/very/long/path', 'file.txt')
  ```
- `os.path.exists()`，返回文件在文件系统上是否存在。

关于 `os.path` 的完整文档可以在这里找到：https://docs.python.org/3/library/os.path.html。另一个模块 `pathlib` 可以用于更高级的、面向对象的访问：https://docs.python.org/3/library/pathlib.html。

如 *步骤 4* 所示，可以使用多种过滤方式。*第 1 章，让我们开始自动化之旅* 中展示的所有字符串操作和技巧都可用。

### 另请参阅

- *第 1 章，让我们开始自动化之旅* 中的 *介绍正则表达式* 配方，学习如何使用正则表达式进行过滤。
- 本章后面的 *读取文本文件* 配方，用于打开找到的文件并读取其内容。

### 读取文本文件

在搜索特定文件后，下一个典型操作是打开它并读取其内容。文本文件非常简单但功能强大。它们以纯文本形式存储数据，没有复杂的二进制格式。

Python 原生提供文本文件支持，很容易将其视为可以用 Python 字符串表示的行的集合。

### 准备工作

我们将读取 `zen_of_python.txt` 文件，其中包含 Tim Peters 的 *Python 之禅*，这是一系列非常恰当地描述了 Python 设计原则的格言。

它在 GitHub 仓库中可用：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter04/documents/zen_of_python.txt：

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

Python 之禅在 PEP-20 中有描述：https://www.python.org/dev/peps/pep-0020/。

> 在任何 Python 解释器中输入 `import this` 都会显示 Python 之禅。

### 如何做...

1. 打开并逐行打印整个文件（结果未显示）：
   ```python
   >>> with open('zen_of_python.txt') as file:
   ...     for line in file:
   ...         print(line)
   ...
   [RESULT NOT DISPLAYED]
   ```

2. 打开文件并打印任何包含字符串 `should` 的行：
   ```python
   >>> with open('zen_of_python.txt', 'r') as file:
   ...     for line in file:
   ...         if 'should' in line.lower():
   ...             print(line)
   ...
   Errors should never pass silently.

   There should be one-- and preferably only one --obvious way to do it.
   ```

3. 打开文件并打印包含单词 `better` 的第一行：
   ```python
   >>> with open('zen_of_python.txt', 'rt') as file:
   ...     for line in file:
   ...         if 'better' in line.lower():
   ...             print(line)
   ...             break
   ...
   Beautiful is better than ugly.
   ```

### 工作原理...

要打开文件，请使用 `open()` 函数。这返回一个 `file` 对象，然后可以对其进行迭代以逐行返回，如 *如何做...* 部分的 *步骤 1* 所示。注意它以文本模式打开文件。

`with` 上下文管理器是处理文件的一种非常方便的方式。它会在使用完毕（离开代码块）后关闭文件。即使引发异常，它也会这样做。

*步骤 2* 展示了如何根据对我们任务适用的行来迭代和过滤这些行。这些行作为字符串返回，可以按照 *第 1 章，让我们开始自动化之旅* 和 *第 3 章，构建你的第一个网页抓取应用* 中的配方所述的多种方式进行过滤。

如 *步骤 3* 所示，可能不需要读取整个文件。因为逐行迭代文件时会边读边处理，所以你可以随时停止，避免读取文件的其余部分。对于像我们示例这样的小文件，这不太相关，但对于长文件，这可以减少内存使用和时间。

### 更多内容...

`with` 上下文管理器是处理文件的首选方式，但不是唯一的方式。你也可以像这样手动打开和关闭它们：

```python
>>> file = open('zen_of_python.txt')
>>> content = file.read()
>>> file.close()
```

注意 `.close()` 方法，以确保文件关闭并释放与打开文件相关的资源。`.read()` 方法一次性读取整个文件，而不是逐行读取。

> `.read()` 方法还接受一个以字节为单位的 `size` 参数，该参数限制读取的数据大小。例如，`file.read(1024)` 将返回最多 1KB 的信息。下一次调用 `.read()` 将从该点继续。

文件以特定模式打开。模式定义了读/写的组合，以及是将数据视为文本还是二进制数据。默认情况下，文件以只读和文本模式打开，分别描述为 "r"（步骤 2）或 "rt"（步骤 3）。

更多模式将在其他配方中探讨。

### 另请参阅

- 本章前面的 *爬取和搜索目录* 配方，用于查找稍后将被读取的文件。
- 本章后面的 *处理编码* 配方，学习如何处理以非标准方式编码的文件。

### 处理编码

文本文件可以以不同的编码存在。近年来，情况已大大改善，因为有几种编码相当标准，但在处理不同系统时仍然存在兼容性问题。

> 文件中的原始数据和 Python 中的字符串对象之间存在差异。字符串对象已从文件包含的任何编码转换为本机 Unicode 字符串。一旦处于此格式，它可能需要以不同的编码存储。默认情况下，Python 使用操作系统定义的编码，在现代操作系统中是 UTF-8。这是一种高度兼容的编码，但根据你的具体要求，你可能需要以不同的编码保存文件。

### 准备工作

我们在 GitHub 仓库中准备了两个文件，它们以两种不同的编码存储字符串 20£：一种是常用的 UTF-8，另一种是 ISO 8859-1，一种不同的常见编码。这些准备好的文件在 GitHub 的 Chapter04/documents 目录下可用，文件名为 example_iso.txt 和 example_utf8.txt：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter04/documents。

我们将使用 Beautiful Soup 模块，该模块在第 3 章，构建你的第一个网页抓取应用的 *解析 HTML* 配方中已介绍过。

### 如何做...

1. 打开 example_utf8.txt 文件并显示其内容：
   ```python
   >>> with open('example_utf8.txt') as file:
   ...     print(file.read())
   ...
   20£
   ```

2. 尝试打开 example_iso.txt 文件，这将引发异常：
   ```python
   >>> with open('example_iso.txt') as file:
   ...     print(file.read())
   ...
   Traceback (most recent call last):
     ...
   UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3 in position 2: invalid start byte
   ```

3. 使用正确的编码打开 example_iso.txt 文件：
   ```python
   >>> with open('example_iso.txt', encoding='iso-8859-1') as file:
   ...     print(file.read())
   ```

### 搜索和阅读本地文件

...

20£

4. 打开 utf8 文件并将其内容保存到 iso-8859-1 文件中：

```python
>>> with open('example_utf8.txt') as file:
...     content = file.read()
>>> with open('example_output_iso.txt', 'w',
encoding='iso-8859-1') as file:
...     file.write(content)
...
4
```

5. 最后，以正确的格式从新文件中读取，以确保其已正确保存：

```python
>>> with open('example_output_iso.txt', encoding='iso-8859-1') as file:
...     print(file.read())
...
20£
```

### 工作原理...

“如何操作”部分中的步骤 1 和步骤 2 非常直接。在步骤 3 中，我们添加了一个额外的参数 `encoding`，以指定文件需要以不同于 UTF-8 的编码方式打开。

> Python 开箱即用地支持许多标准编码。在此处查看所有编码及其别名：https://docs.python.org/3/library/codecs.html#standard-encodings。

在步骤 4 中，我们创建了一个新的 ISO-8859-1 文件，并像往常一样写入。注意 `"w"` 参数，它指定以写入模式和文本模式打开文件。

步骤 5 是确认文件已正确保存。

### 更多内容...

本方法假设我们知道文件的编码。但有时，我们并不确定。`Beautiful Soup` 是一个用于解析 HTML 的模块，它可以尝试检测特定文件的编码。

> 自动检测文件的编码可能是，嗯，不可能的。编码可能有无限多种！但常用的编码子集应该能覆盖 90% 的现实情况。只需记住，最简单的确认方法是询问文件的创建者。

为此，我们需要使用 `'rb'` 参数以二进制格式打开文件进行读取。然后，我们将二进制内容传递给 `Beautiful Soup` 的 `UnicodeDammit` 模块，如下所示：

```python
>>> from bs4 import UnicodeDammit
>>> with open('example_output_iso.txt', 'rb') as file:
...     content = file.read()
...
>>> suggestion = UnicodeDammit(content)
>>> suggestion.original_encoding
'iso-8859-1'
>>> suggestion.unicode_markup
'20£\n'
```

然后可以推断出编码。虽然 `.unicode_markup` 返回解码后的字符串，但最好仅使用此建议来获取编码，然后以正确的编码重新以文本模式打开文件。

### 另请参阅

- *第 1 章，开始我们的自动化之旅* 中的 *操作字符串* 方法，以了解更多关于如何编辑字符串的信息。
- *第 3 章，构建你的第一个网页抓取应用* 中的 *解析 HTML* 方法，以了解更多关于 Beautiful Soup 的信息。

### 搜索和阅读本地文件

### 读取 CSV 文件

一些文本文件包含以逗号分隔的表格数据。这是一种创建结构化数据的便捷方式，而不是使用专有的、更复杂的二进制格式，如 Excel 或其他格式。这些文件被称为 **逗号分隔值** 或 CSV 文件，大多数电子表格软件包允许我们直接处理它们。

### 准备工作

我们使用前 10 部电影的影院上座率数据准备了一个 CSV 文件，如本页所述：http://www.mrob.com/pub/film-video/topadj.html。

我们将表格的前 10 个元素复制到电子表格程序（Numbers）中，并将文件导出为 CSV。该文件可在 GitHub 仓库的 Chapter04/documents 目录中找到，名为 `top_films.csv`。

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_123_0.png)

### 如何操作...

1. 导入 csv 模块：

```python
>>> import csv
```

2. 打开文件，创建一个读取器，并遍历它以显示所有行的表格数据（仅显示三行）：

```python
>>> with open('top_films.csv') as file:
...     data = csv.reader(file)
...     for row in data:
...         print(row)
...
['Rank', 'Admissions\n(millions)', 'Title (year) (studio)', 'Director(s)']
['1', '225.7', 'Gone With the Wind (1939)\xa0(MGM)', 'Victor Fleming, George Cukor, Sam Wood']
['2', '194.4', 'Star Wars (Ep. IV: A New Hope) (1977)\xa0(Fox)', 'George Lucas']
...
['10', '118.9', 'The Lion King (1994)\xa0(BV)', 'Roger Allers, Rob Minkoff']
```

3. 打开文件并使用 `DictReader` 来结构化数据，包括表头：

```python
>>> with open('top_films.csv') as file:
...     data = csv.DictReader(file)
...     structured_data = [row for row in data]
...
>>> structured_data[0]
{'Rank': '1', 'Admissions\n(millions)': '225.7', 'Title (year) (studio)': 'Gone With the Wind (1939)\xa0(MGM)', 'Director(s)': 'Victor Fleming, George Cukor, Sam Wood'}
```

4. `structured_data` 中的每个项目都是一个完整的字典，包含每个值：

```python
>>> structured_data[0].keys()
dict_keys(['Rank', 'Admissions\n(millions)', 'Title (year) (studio)', 'Director(s)'])
>>> structured_data[0]['Rank']
'1'
>>> structured_data[0]['Director(s)']
'Victor Fleming, George Cukor, Sam Wood'
```

### 工作原理...

注意文件需要被读取，并且我们使用了 `with` 上下文块。这确保了文件在块结束时被关闭。

如“如何操作”部分步骤 2 所示，csv.reader 类允许我们通过将返回的代码行细分为列表来结构化它们，遵循表格数据的格式。注意所有值都被描述为字符串。csv.reader 不理解第一行是否是表头。

为了更结构化地读取文件，在步骤 3 中，我们使用了 csv.DictReader。默认情况下，它将第一行读取为定义适用字段的表头，然后将每一行转换为包含这些字段的字典。

> 有时，就像本例一样，文件中描述的字段名称可能有点冗长。不要害怕将字典作为额外步骤转换为更易于管理的字段名称。

### 更多内容...

CSV 的文件结构解释非常宽松。数据可以有多种存储方式。这在 csv 模块中表示为方言。例如，值可以用逗号、分号或制表符分隔。可以通过调用 csv.list_dialect() 来显示默认接受的方言列表。

> 默认情况下，方言将是 Excel，这是最常见的。即使是其他电子表格也通常使用它。

方言也可以通过 Sniffer 类从文件本身推断出来。Sniffer 类分析文件的样本（或整个文件）并返回一个方言对象，以便以正确的方式读取。

注意文件是以无换行符方式打开的，以避免对其做出任何假设：

```python
>>> with open('top_films.csv', newline='') as file:
...     dialect = csv.Sniffer().sniff(file.read())
```

然后可以在打开读取器时使用该方言。再次注意换行符，因为方言将正确分割行：

```python
>>> with open('top_films.csv', newline='') as file:
...     reader = csv.reader(file, dialect)
...     for row in reader:
...         print(row)
```

完整的 csv 模块文档可在此处找到：https://docs.python.org/3/library/csv.html。

### 另请参阅

- 本章前面的 *处理编码* 方法，以了解如何处理编码。
- 本章前面的 *读取文本文件* 方法，以了解更多关于打开和读取文件的信息。

### 读取日志文件

另一种常见的结构化文本文件格式是 **日志文件**。日志文件由日志行组成，每行是具有特定格式的文本行，描述一个事件。

> 日志仅在同一文件或文件类型中是结构化的。格式可能非常不同，没有通用的结构或语法。每个应用程序都可以并且将会使用不同的格式。

通常，每个日志都会有一个事件发生的时间，因此文件是这些事件的有序集合。

### 准备工作

包含五个销售日志的 example_log.log 文件可从 GitHub 仓库此处获取：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter04/documents/example_logs.log。

格式为：

```
[<Timestamp in iso format>] - SALE - PRODUCT: <product id> - PRICE: $<price of the sale>
```

我们将使用 Chapter01/price_log.py 文件将每个日志处理为一个对象。Chapter04/documents 目录中有一个副本，以简化导入过程。

### 搜索和读取本地文件

### 如何操作...

1. 导入 PriceLog：
    ```python
    >>> from price_log import PriceLog
    ```
2. 打开日志文件并解析所有日志：
    ```python
    >>> with open('example_logs.log') as file:
    ...     logs = [PriceLog.parse(log) for log in file]
    ...
    >>> len(logs)
    5
    >>> logs[0]
    <PriceLog (Delorean(datetime=datetime.datetime(2018, 6, 17, 22, 11, 50, 268396), timezone='UTC'), 1489, 9.99)>
    ```
3. 计算所有销售的总收入：
    ```python
    >>> total = sum(log.price for log in logs)
    >>> total
    Decimal('47.82')
    ```
4. 确定每个 product_id 的销售数量：
    ```python
    >>> from collections import Counter
    >>> counter = Counter(log.product_id for log in logs)
    >>> counter
    Counter({1489: 2, 4508: 1, 8597: 1, 3086: 1})
    ```
5. 过滤日志以查找所有销售产品 ID 为 1489 的记录：
    ```python
    >>> logs = [log for log in logs if log.product_id == 1489]
    >>> len(logs)
    2
    >>> logs[0].product_id, logs[0].timestamp
    (1489, Delorean(datetime=datetime.datetime(2018, 6, 17, 22, 11, 50, 268396), timezone='UTC'))
    >>> logs[1].product_id, logs[1].timestamp
    (1489, Delorean(datetime=datetime.datetime(2018, 6, 17, 22, 11, 50, 268468), timezone='UTC'))
    ```

### 工作原理...

由于每个条目都是单行，我们打开文件并逐行处理，解析每一行。解析代码位于 `price_log.py` 中。有关解析过程的更多细节，请查看该文件。

在 *如何操作...* 部分的 *步骤 2* 中，我们打开文件并处理每一行，创建一个包含所有已处理日志的日志列表。然后，我们可以进行聚合操作，如后续步骤所示。

*步骤 3* 展示了如何聚合所有值。在这个例子中，是将日志文件中所有售出物品的价格相加，以获得总收入。

*步骤 4* 使用 `Counter` 来确定文件日志中每个项目的数量。这返回一个类似字典的对象，其中包含要计数的值及其出现的次数。

过滤也可以采用逐行处理的方式，如 *步骤 5* 所示。这与本章其他食谱中进行的过滤类似。

### 更多内容...

请记住，一旦获得了所需的所有数据，就可以停止处理文件。如果文件非常大（日志文件通常如此），这可能是一个好策略。

`Counter` 是一个快速统计列表的绝佳工具。更多详情请参阅 Python 文档：`https://docs.python.org/3/library/collections.html#counter-objects`。你可以通过调用以下方法获取排序后的项目：

```python
>>> counter.most_common()
[(1489, 2), (4508, 1), (8597, 1), (3086, 1)]
```

### 另请参阅

- *第 1 章，开始我们的自动化之旅* 中的 *使用第三方工具——parse* 食谱。
- 本章前面的 *读取文本文件* 食谱，以了解更多关于打开和读取文件的信息。

### 搜索和读取本地文件

### 读取文件元数据

文件元数据是与特定文件相关的所有信息，但不包括数据内容本身。最明显的是文件名，但还有更多可用参数，例如文件大小、创建日期或其权限。

浏览这些数据很重要，例如，用于过滤早于某个日期的文件，或查找所有大于某个 KB 值的文件。在本食谱中，我们将了解如何在 Python 中访问文件元数据。

### 准备工作

我们将使用 `zen_of_python.txt` 文件，该文件可在 GitHub 仓库中找到 (https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter04/documents/zen_of_python.txt)。如你所见，通过使用 `ls` 命令，该文件大小为 856 字节，在本例中，它创建于 6 月 14 日：

```
$ ls -lrt zen_of_python.txt
-rw-r--r--@ 1 jaime staff 856 14 Jun 21:22 zen_of_python.txt
```

在你的计算机上，日期可能会有所不同，具体取决于你下载代码的时间。

### 如何操作...

1. 导入 `os` 和 `datetime`：
    ```python
    >>> import os
    >>> from datetime import datetime
    ```
2. 获取 `zen_of_python.txt` 文件的统计信息：
    ```python
    >>> stats = os.stat(('zen_of_python.txt'))
    >>> stats
    os.stat_result(st_mode=33188, st_ino=15822537, st_dev=16777224, st_nlink=1, st_uid=501, st_gid=20, st_size=856, st_atime=1529461935, st_mtime=1529007749, st_ctime=1529007757)
    ```
3. 获取文件大小（以字节为单位）：
    ```python
    >>> stats.st_size
    856
    ```
4. 获取文件最后修改的时间：
    ```python
    >>> datetime.fromtimestamp(stats.st_mtime)
    datetime.datetime(2018, 6, 14, 21, 22, 29)
    ```
5. 获取文件最后访问的时间：
    ```python
    >>> datetime.fromtimestamp(stats.st_atime)
    datetime.datetime(2018, 6, 20, 3, 32, 15)
    ```

### 工作原理...

`os.stats` 返回一个 stats 对象，该对象表示存储在文件系统中的元数据。元数据包括：

- 文件大小（以字节为单位），如 *如何操作...* 部分的步骤 3 所示，使用 `st_size`。
- 文件内容最后修改的时间，如步骤 4 所示，使用 `st_mtime`。
- 文件最后读取（访问）的时间，如步骤 5 所示，使用 `st_atime`。

时间以时间戳形式返回，因此在步骤 4 和步骤 5 中，我们从时间戳创建一个 datetime 对象，以便更好地访问数据。

所有这些值都可用于过滤文件并访问有意义的文件。

> 注意，你不需要使用 `open()` 打开文件来读取其元数据。通过读取修改时间来检测文件是否在已知时间后被更改，将比比较其内容更快，因此你可以利用这一点进行比较。

### 更多内容...

为了逐个获取统计信息，`os.path` 中也提供了便捷函数，遵循 `get<value>` 模式：

```python
>>> os.path.getsize('zen_of_python.txt')
856
>>> os.path.getmtime('zen_of_python.txt')
1529531584.0
>>> os.path.getatime('zen_of_python.txt')
1529531669.0
```

该值以 UNIX 时间戳格式（自 1970 年 1 月 1 日以来的秒数）指定。

### 搜索和读取本地文件

> 请注意，调用所有这三个函数将比单次调用 `os.stats` 并处理结果更慢。此外，返回的 `stats` 可以被检查以检测可用的元数据。

本食谱中描述的值适用于所有文件系统，但根据特定平台，还有更多可用的值。

例如，要获取文件的创建日期，你可以使用 macOS 的 `st_birthtime` 参数或 Windows 的 `st_mtime`。

> `st_mtime` 始终可用，但其含义在不同系统之间有所不同。在 Unix 系统中，当内容被修改时它会更改，因此它不是可靠的创建时间。

`os.stat` 将跟随符号链接。如果你想获取符号链接的统计信息，请使用 `os.lstat()`。

你可以在此处查看所有可用统计信息的完整文档：https://docs.python.org/3/library/os.html - os.stat_result。

### 另请参阅

- 本章前面的 *读取文本文件* 食谱，以了解打开和读取文件的基础知识。
- 本章后面的 *读取图像* 食谱，以了解如何读取和处理图像文件。

### 读取图像

可能最常见的非文本数据是图像数据。图像有其自己的一组特定元数据，可以读取这些元数据来过滤值或执行其他操作。

主要挑战在于处理多种格式和不同的元数据定义。我们将在本食谱中展示如何从 JPEG 和 PNG 获取信息，以及相同的信息如何以不同的方式编码。

### 准备工作

可以说，处理图像的最佳通用工具包是 Pillow。这个库允许你轻松读取最常见格式的文件，并对其进行操作。Pillow 最初是 PIL（Python Imaging Library）的一个分支，PIL 是一个几年前就已停滞的先前模块。

我们还将使用 `xmltodict` 模块将一些 XML 数据转换为更方便的字典。我们将把这两个模块添加到 `requirements.txt` 并在虚拟环境中重新安装它们：

```
$ echo "Pillow==7.0.0" >> requirements.txt
$ echo "xmltodict==0.12.0" >> requirements.txt
$ pip install -r requirements.txt
```

照片文件中的元数据信息在 EXIF（可交换图像文件）格式中定义。EXIF 是一个用于存储图片信息的标准，包括拍摄照片的相机、拍摄时间、描述位置的 GPS、曝光、焦距、颜色信息等。

> 你可以在此处获得一个很好的概述：https://www.slrphotographyguide.com/what-is-exif-metadata/。所有信息都是可选的，但几乎所有数码相机和处理软件都会存储一些数据。由于隐私问题，其中一部分（如确切位置）可以被禁用。

本食谱将使用以下图像，可在 GitHub 仓库中下载 (https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter04/images)：

- photo-dublin-a1.jpg
- photo-dublin-a2.png
- photo-dublin-b.png

其中两个，`photo-dublin-a1.jpg` 和 `photo-dublin-a2.png`，是同一场景，但第一个是未经处理的照片，第二个经过修饰以轻微改变颜色，并且也被缩放了。注意一个是 JPEG 格式，另一个是 PNG 格式。另一个，`photo-dublin-b.png`，是一张不同的照片。两张照片都是在都柏林用同一部手机相机在两个不同的日子拍摄的。

### 搜索和读取本地文件

虽然 Pillow 能直接理解 JPG 文件如何存储 EXIF 信息，但 PNG 文件存储的是 XMP 信息——这是一种更通用的标准，可以在其中包含 EXIF 数据。

> 关于 XMP 的更多信息可以在此处获取：https://www.adobe.com/devnet/xmp.html。在大多数情况下，它定义了一个 XML 树结构，其原始形式相对可读。

更复杂的是，XMP 使用 RDF 进行编码，而 RDF 是一个描述如何编码 XML 树的标准。

> 如果 EXIF、XMP 和 RDF 听起来令人困惑，嗯，那是因为它们确实如此。XMP 使用 RDF 存储 EXIF 信息。最终，它们之间的差异并不十分相关，最好的方法是找出有趣的部分。我们可以使用 Python 自省工具检查名称的具体细节，并准确检查数据的结构以及我们正在寻找的参数名称。

由于 GPS 信息以不同的格式存储，我们在 GitHub 仓库中包含了一个名为 `gps_conversion.py` 的文件：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter04/gps_conversion.py。该文件包含 `exif_to_decimal` 和 `rdf_to_decimal` 函数，它们会将两种格式都转换为十进制数以便进行比较。

### 如何操作...

1.  导入本食谱中要使用的模块和函数：
    ```python
    >>> from PIL import Image
    >>> from PIL.ExifTags import TAGS, GPSTAGS
    >>> import xmltodict
    >>> from gps_conversion import exif_to_decimal, rdf_to_decimal
    ```
2.  打开第一张照片：
    ```python
    >>> image1 = Image.open('images/photo-dublin-a1.jpg')
    ```
3.  获取文件的宽度、高度和格式：
    ```python
    >>> image1.height
    3024
    >>> image1.width
    4032
    >>> image1.format
    'JPEG'
    ```
4.  检索图像的 EXIF 信息并将其转换为方便的字典。显示相机、使用的镜头以及拍摄时间：
    ```python
    >>> exif_info_1 = {TAGS.get(tag, tag): value
            for tag, value in image1._getexif().items()}
    >>> exif_info_1['Model']
    'iPhone X'
    >>> exif_info_1['LensModel']
    'iPhone X back dual camera 4mm f/1.8'
    >>> exif_info_1['DateTimeOriginal']
    '2018:04:21 12:07:55'
    ```
5.  打开第二张图像并获取 XMP 信息：
    ```python
    >>> image2 = Image.open('images/photo-dublin-a2.png')
    >>> image2.height
    1512
    >>> image2.width
    2016
    >>> image2.format
    'PNG'
    >>> xmp_info = xmltodict.parse(image2.info['XML:com.adobe.xmp'])
    ```
6.  获取 RDF 描述字段，其中包含我们正在寻找的所有值。检索型号（一个 TIFF 值）、镜头型号（一个 EXIF 值）和创建日期（一个 XMP 值）。检查这些值是否与步骤 4 中的相同，即使文件不同：
    ```python
    >>> rdf_info_2 = xmp_info['x:xmpmeta']['rdf:RDF']
    ['rdf:Description']
    >>> rdf_info_2['tiff:Model']
    'iPhone X'
    >>> rdf_info_2['exifEX:LensModel']
    'iPhone X back dual camera 4mm f/1.8'
    >>> rdf_info_2['xmp:CreateDate']
    '2018-04-21T12:07:55'
    ```
7.  获取两张图片中的 GPS 信息，将它们转换为等效格式，并检查它们是否相同。注意分辨率不同，但它们在小数点后第四位是匹配的：
    ```python
    >>> gps_info_1 = {GPSTAGS.get(tag, tag): value
            for tag, value in exif_info_1['GPSInfo'].items()}
    >>> exif_to_decimal(gps_info_1)
    ('N53.34690555555556', 'W6.247797222222222')
    >>> rdf_to_decimal(rdf_info_2)
    ('N53.346905', 'W6.247796666666667')
    ```
8.  打开第三张图像并获取创建日期和 GPS 信息，检查它是否与其他照片不匹配，尽管很接近（第二位和第三位小数不同）：
    ```python
    >>> image3 = Image.open('photo-dublin-b.png')
    >>> xmp_info = xmltodict.parse(image3.info['XML:com.adobe.xmp'])
    >>> rdf_info_3 = xmp_info['x:xmpmeta']['rdf:RDF']
    ['rdf:Description']
    >>> rdf_info_3['xmp:CreateDate']
    '2018-03-08T18:16:57'
    >>> rdf_to_decimal(rdf_info_3)
    ('N53.34984166666667', 'W6.260388333333333')
    ```

### 工作原理...

Pillow 能够解释大多数常见图像格式的文件，如“如何操作...”部分的步骤 2 所示。

Image 对象包含有关文件大小和格式的基本信息，如步骤 3 所示。`info` 属性包含依赖于格式的信息。

JPG 文件的 EXIF 元数据可以使用 `_getexif()` 方法解析，但随后需要正确翻译，因为它使用原始二进制定义。例如，数字 42,036 对应于 `LensModel` 属性。幸运的是，`PIL.ExifTags` 模块中定义了所有标签。我们在步骤 4 中将字典转换为可读的标签，以获得更易读的字典。

步骤 5 打开一个 PNG 格式，它具有与大小相关的相同属性，但元数据以 XML/RDF 格式存储，需要借助 `xmltodict` 进行解析。步骤 6 展示了如何导航此元数据以提取与 JPG 格式中相同的信息。数据是相同的，因为两个文件都来自同一张原始图片，即使图像不同。

> Xmltodict 在尝试解析非 XML 格式的数据时存在一些问题。请检查输入是否为有效的 XML。

步骤 7 提取了两张图像的 GPS 信息，它们以不同的方式存储，并显示它们是相同的（尽管由于编码方式不同，精度不同）。

> 并非每张图像都必然包含位置信息或其他元数据。根据格式和生成文件的相机，此信息可能会被更改、删除或以不同方式存储。

步骤 8 显示了另一张照片的信息。

### 更多内容...

Pillow 还具有许多围绕修改图片的功能。调整文件大小或进行简单修改（如旋转）非常容易。您可以在此处找到完整的 Pillow 文档：https://pillow.readthedocs.io。

> Pillow 允许对图像进行大量操作。不仅是简单的操作，如调整大小或转换格式，还包括裁剪图像、应用颜色滤镜或生成动画 GIF 等。如果您对使用 Python 进行图像处理感兴趣，它绝对是需要掌握的模块。

食谱中的 GPS 坐标以 **DMS（度、分、秒）** 和 **DDM（度、十进制分）** 表示，并转换为 **DD（十进制度）**。您可以在此处了解更多关于不同 GPS 格式的信息：http://www.ubergizmo.com/how-to/read-gps-coordinates/。如果您好奇，您还可以在那里找到如何搜索图片的确切位置。

读取图像文件的一个更高级用途是尝试对其进行 **OCR（光学字符识别）** 处理。这意味着自动检测图像中的文本并提取和处理它。开源模块 `tesseract` 允许您执行此操作，并且可以与 Python 和 Pillow 一起使用。

您需要在系统上安装 `tesseract`（https://github.com/tesseract-ocr/tesseract/wiki）和 `pytesseract` Python 模块（使用 `pip install pytesseract`）。

您可以从 GitHub 仓库下载一个包含清晰文本的文件，名为 `photo-text.jpg`，地址为 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter04/images/photo-text.jpg：

```python
>>> from PIL import Image
>>> import pytesseract
>>> pytesseract.image_to_string(Image.open('photo-text.jpg'))
'Automate!'
```

如果图像中的文本不是很清晰，或者与图像混合，或者使用特殊字体，OCR 可能会很困难。在 `photo-dublin-a-text.jpg` 文件中有一个这样的例子（可在 GitHub 仓库 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter04/images/photo-dublin-a-text.jpg 中找到），其中包含图片上的文本：

```python
>>> pytesseract.image_to_string(Image.open('photo-dublin-a-text.jpg'))
'fl\n\nAutomat'
```

有关 Tesseract 的更多信息，请访问以下链接：

https://github.com/tesseract-ocr/tesseract

https://github.com/madmaze/pytesseract

> 正确导入文件进行 OCR 可能需要初始图像处理以获得更好的结果。图像处理超出了本书的目标范围，但您可以使用 OpenCV，它比 Pillow 更强大。您可以处理文件，然后使用 Pillow 打开它：http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html。

### 另请参阅

-   本章前面的“读取文本文件”食谱，了解打开和读取文件的基础知识。
-   本章前面的“读取文件元数据”食谱，了解如何从文件中获取额外信息。
-   本章前面的“爬取和搜索目录”食谱，了解如何在目录中搜索和查找文件。

### 读取 PDF 文件

文档的一种常见格式是 **PDF（便携式文档格式）**。它最初是一种为任何打印机描述文档的格式，因此 PDF 是一种确保文档打印效果与显示完全一致的格式。它已成为共享文档的强大标准，尤其是那些最终版本且旨在只读的文档。

### 准备工作

在本节中，我们将使用 `PyPDF2` 模块。我们需要将其添加到虚拟环境中：

```
$ echo "PyPDF2==1.26.0" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 的 `Chapter03/documents` 目录中，我们准备了两个文档 `document-1.pdf` 和 `document-2.pdf`，用于本节。请注意，它们主要包含 Lorem Ipsum 文本，这只是一个占位符。

> Lorem Ipsum 文本通常用于设计中，以便在设计前无需创建内容即可展示文本。你可以在此处了解更多相关信息：https://loremipsum.io/。

它们都是相同的测试文档，但第二个文档只能使用密码打开。密码是 `automate`。

### 如何操作...

1. 导入模块：
   ```python
   >>> from PyPDF2 import PdfFileReader
   ```

2. 打开 `document-1.pdf` 文件并创建一个 PDF 文档对象。请注意，文件需要在整个读取过程中保持打开状态：
   ```python
   >>> file = open('document-1.pdf', 'rb')
   >>> document = PdfFileReader(file)
   ```

3. 获取文档的页数并检查其是否加密：
   ```python
   >>> document.numPages
   3

   >>> document.isEncrypted
   False
   ```

4. 从文档信息中获取创建日期（2018年6月24日 11:15:18），并发现它是使用 Mac Quartz PDFContext 创建的：
   ```python
   >>> document.documentInfo['/CreationDate']
   'D:20180624111518Z00\'00\''
   >>> document.documentInfo['/Producer']
   'Mac OS X 10.13.5 Quartz PDFContext'
   ```

5. 获取第一页并读取其上的文本：
   ```python
   >>> document.pages[0].extractText()
   '!A VERY IMPORTANT DOCUMENT \nBy James McCormac CEO Loose Seal Inc '
   ```

6. 对第二页执行相同操作（此处已编辑）：
   ```python
   >>> document.pages[1].extractText()
   '"!This is an example of a test document that is stored in PDF format. It contains some \nsentences to describe what it is and the it has lore ipsum text.\n!"\nLorem ipsum dolor sit amet, consectetur adipiscing elit. ...$'
   ```

7. 关闭文件并打开 document-2.pdf：
   ```python
   >>> file.close()
   >>> file = open('document-2.pdf', 'rb')
   >>> document = PdfFileReader(file)
   ```

8. 检查文档是否加密（它需要密码），如果我们尝试访问其内容，则会引发错误：
   ```python
   >>> document.isEncrypted
   True
   >>> document.numPages
   ...
   PyPDF2.utils.PdfReadError: File has not been decrypted
   ```

9. 解密文件并访问其内容：
   ```python
   >>> document.decrypt('automate')
   1
   >>> document.numPages
   3
   >>> document.pages[0].extractText()
   'A VERY IMPORTANT DOCUMENT \nBy James McCormac CEO Loose Seal Inc\n'
   ```

10. 关闭文件以进行清理：
    ```python
    >>> file.close()
    ```

### 工作原理...

一旦文档打开，如 *如何操作...* 部分的 *步骤 1* 和 *步骤 2* 所示，`document` 对象提供了对文档的访问。

一些有用的属性是页数（在 `.numPages` 中可用）和每一页（在 `.pages` 中可用），可以像列表一样访问。

其他可访问的数据存储在 `.documentInfo` 中，其中存储了关于创建者和创建时间的元数据。

> `.documentInfo` 中的信息是可选的，有时不是最新的。它在很大程度上取决于用于生成 PDF 的工具。

每个 `page` 对象都可以通过调用 `.extractText()` 来获取其文本，这将返回页面中包含的所有文本，如 *步骤 5* 和 *步骤 6* 中所示。此方法尝试提取所有文本，但它有一些限制。对于结构良好的文本，例如我们的示例，它效果很好，并且生成的文本可以干净地处理。当处理多列文本或位于奇怪位置的文本时，可能会使处理变得复杂。

> 请注意，PDF 文件需要在整个操作过程中保持打开状态，而不是使用 `with` 上下文运算符。离开 `with` 块后，文件将被关闭。

*步骤 8* 和 *步骤 9* 展示了如何处理加密文件。你可以使用 `.isEncrypted` 检测文件是否加密，然后使用 `.decrypt` 方法提供正确的密码进行解密。

### 更多内容...

PDF 是一种非常灵活的格式，广泛用于各种目的，但这也意味着它可能难以解析和处理。

虽然大多数 PDF 文件包含文本信息，但它们包含文本图像的情况并不少见。当文档被扫描时，就会发生这种情况。在这种情况下，信息存储为图像集合，而不是结构化文本。这使得提取文本数据变得困难；我们可能最终不得不求助于 OCR 等方法将图像解析为文本。

PyPDF2 没有提供处理图像的良好接口。你可能需要将 PDF 转换为图像集合，然后使用其他工具（如 Pillow）处理图像。有关 OCR 和 Pillow 使用的想法，请参阅读取图像部分。大多数 PDF 阅读器都可以做到这一点，或者你可以使用命令行工具，如 pdftoppm (https://linux.die.net/man/1/pdftoppm) 或 QPDF（见下文）。

某些文件加密方法可能不被 PyPDF2 理解。它将生成 NotImplementedError: only algorithm code 1 and 2 are supported。如果发生这种情况，你需要在外部解密 PDF，然后在解密后打开它。你可以使用 QPDF 创建一个没有密码的副本，如下所示：

```
$ qpdf --decrypt --password=PASSWORD encrypted.pdf output-decrypted.pdf
```

QPDF 的完整文档可在此处获取：http://qpdf.sourceforge.net/files/qpdf-manual.html。QPDF 在大多数包管理器中也可用。

> QPDF 能够进行大量转换和深入分析 PDF。Python 中也有一个名为 pikepdf (https://pikepdf.readthedocs.io/en/stable/) 的库的绑定。这个包比 PyPDF2 更复杂，对于文本提取来说不那么直接，但它对于其他操作（如从 PDF 中提取图像）可能很有用。

### 另请参阅

- 本章前面的读取文本文件部分，了解打开和读取文件的基础知识。
- 本章前面的爬取和搜索目录部分，了解如何在目录中搜索和查找文件。

### 读取 Word 文档

Word 文档（.docx）是另一种常见的文档类型，主要存储文本。它们通常由 Microsoft Office 生成，但其他工具也可以生成兼容文件。它们可能是共享需要编辑的文件的最常见格式，但它们也常用于分发文档。

我们将在本节中了解如何从 Word 文档中提取文本信息。

### 准备工作

我们将使用 python-docx 模块来读取和处理 Word 文档：

```
$ echo "python-docx==0.8.10" >> requirements.txt
$ pip install -r requirements.txt
```

我们准备了一个测试文件，可在 GitHub 的 Chapter04/documents 目录中找到，名为 document-1.docx，我们将在本节中使用它。请注意，此文档遵循与读取 PDF 文件部分测试文档中描述的相同 Lorem Ipsum 模式。

### 如何操作...

1. 导入 python-docx：
   ```python
   >>> import docx
   ```

2. 打开 document-1.docx 文件：
   ```python
   >>> doc = docx.Document('document-1.docx')
   ```

3. 检查存储在 core_properties 中的一些元数据属性：
   ```python
   >>> doc.core_properties.title
   'A very important document'
   >>> doc.core_properties.keywords
   'lorem ipsum'
   >>> doc.core_properties.modified
   datetime.datetime(2018, 6, 24, 15, 1, 7)
   ```

4. 检查段落数量：
   ```python
   >>> len(doc.paragraphs)
   58
   ```

### 搜索和读取本地文件

5.  遍历段落以检测包含文本的段落。请注意，并非所有文本都显示在此处：

```python
>>> for index, paragraph in enumerate(doc.paragraphs):
...     if paragraph.text:
...         print(index, paragraph.text)
...
30 A VERY IMPORTANT DOCUMENT
31 By James McCormac
32 CEO Loose Seal Inc
34
...
56 TITLE 2
57 ...
```

6.  获取第30和31段的文本，它们对应于第一页的标题和副标题：

```python
>>> doc.paragraphs[30].text
'A VERY IMPORTANT DOCUMENT'
>>> doc.paragraphs[31].text
'By James McCormac'
```

7.  每个段落都有“runs”，即具有不同属性的文本部分。检查第一个文本段落和run是否为粗体，第二个是否为斜体：

```python
>>> doc.paragraphs[30].runs[0].italic
>>> doc.paragraphs[30].runs[0].bold
True
>>> doc.paragraphs[31].runs[0].bold
>>> doc.paragraphs[31].runs[0].italic
True
```

8.  在此文档中，大多数段落只有一个run，但第48段是一个很好的不同run示例。显示其文本和不同的样式。例如，单词Word是粗体，ipsum是斜体：

```python
>>> [run.text for run in doc.paragraphs[48].runs]
['This is an example of a test document that is stored in ',
'Word', ' format', '. It contains some ', 'sentences', ' to
describe what it is and it has ', 'lore', 'm', ' ipsum', ' text.']
>>> run1 = doc.paragraphs[48].runs[1]
>>> run1.text
'Word'
>>> run1.bold
True
>>> run2 = doc.paragraphs[48].runs[8]
>>> run2.text
' ipsum'
>>> run2.italic
True
```

### 工作原理...

Word文档最重要的特点是数据按段落而非页面结构化。字体大小、行距和其他因素可能会导致页数发生变化。

大多数段落通常也是空的，或者只包含换行符、制表符或其他空白字符。检查段落是否为空并跳过它是一个好主意。

在*操作步骤...*部分，*步骤2*打开文件，*步骤3*展示了如何访问核心属性。这些是在Word中定义为文档元数据的属性，例如作者或创建日期。

> 这些信息需要谨慎对待，因为许多生成Word文档的工具（但不是Microsoft Office）不一定会填写它。在使用该信息之前，请仔细检查。

可以迭代文档的段落并以原始格式提取其文本，如*步骤6*所示。这是不包含样式信息的信息，通常是自动处理数据最有用的格式。

如果需要样式信息，可以使用runs，如*步骤7*和*步骤8*所示。每个段落可以包含一个或多个runs，这些runs是共享相同样式的较小文本单元。例如，如果一个句子是*Word1* *word2* **word3**，将会有三个runs，一个包含斜体文本（Word1），另一个包含下划线（word2），另一个包含粗体（word3）。此外，可能还有包含普通文本的中间runs，仅包含空格，总共五个runs。

可以单独检测粗体、斜体或下划线等属性的样式。

> runs的划分可能相当复杂。由于编辑器的工作方式，出现半词、一个词被分成两个runs（有时具有相同属性）的情况并不少见。不要依赖runs的数量，而要分析内容。特别是，如果您试图确保具有特定样式的部分是否被分成两个或多个runs，请仔细检查。*步骤8*中的单词lore和m（它应该是一个单词lorem）就是一个很好的例子。

请注意，由于Word文档由许多来源生成，许多属性可能未设置，留给工具指定要使用的具体信息。例如，保留默认字体非常常见，这意味着runs上的字体信息可能留空。

### 更多内容...

更多样式信息可以在font属性下找到，例如small_caps或size：

```python
>>> run2.font.cs_italic
True
>>> run2.font.size
152400
>>> run2.font.small_caps
```

通常，专注于原始文本而不关注样式信息是解析的正确方法。但有时，段落中的粗体单词可能具有特殊意义。它可能是标题或某些特别有意义的文本。因为它被突出显示，所以它很可能就是您要找的！在分析文档时请记住这一点。

您可以在此处找到完整的python-docx文档：https://python-docx.readthedocs.io/en/latest/.

### 另请参阅

- 本章前面的“读取文本文件”食谱，了解打开和读取文本文件的基础知识。
- 本章前面的“读取PDF文件”食谱，了解如何处理其他类型的文档文件。

### 扫描文档中的关键词

在这个食谱中，我们将应用前面食谱中的所有经验，并在目录中的所有文件中搜索特定关键词。这是本章其余食谱的总结，包括一个搜索不同类型文件的脚本。

### 准备工作

确保在`requirements.txt`文件中包含以下模块，并将其安装到您的虚拟环境中：

```
beautifulsoup4==4.8.2
Pillow==7.0.0
PyPDF2==1.26.0
python-docx==0.8.10
```

检查要搜索的目录是否包含以下文件（所有文件均可在https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter04/documents/获取）。请注意，为简单起见，`file5.pdf`和`file6.pdf`是`document-1.pdf`的副本。`file1.txt`到`file4.txt`是空文件：

```
├── dir
│   ├── file1.txt
│   ├── file2.txt
│   ├── file6.pdf
│   └── subdir
│       ├── file3.txt
│       ├── file4.txt
│       └── file5.pdf
├── document-1.docx
├── document-1.pdf
├── document-2-1.pdf
├── document-2.pdf
├── example_iso.txt
├── example_output_iso.txt
├── example_utf8.txt
├── top_films.csv
└── zen_of_python.txt
```

我们准备了一个脚本`scan.py`，它将在所有`.txt`、`.csv`、`.pdf`和`.docx`文件中搜索一个单词。该脚本位于GitHub仓库的`Chapter04`目录中。

### 操作步骤...

1.  参阅帮助`-h`了解如何使用`scan.py`脚本：

   ```
   $ python scan.py -h
   usage: scan.py [-h] [-w W]

   optional arguments:
     -h, --help show this help message and exit
     -w W Word to search
   ```

2.  搜索单词`the`，它存在于大多数文件中：

   ```
   $ python scan.py -w the
   >>> Word found in ./documents/top_films.csv
   >>> Word found in ./documents/zen_of_python.txt
   >>> Word found in ./documents/document-1.pdf
   >>> Word found in ./documents/dir/file6.pdf
   >>> Word found in ./documents/dir/subdir/file5.pdf
   ```

3.  搜索单词`lorem`，仅存在于PDF和`.docx`文件中：

   ```
   $ python scan.py -w lorem
   >>> Word found in ./documents/document-1.pdf
   >>> Word found in ./documents/document-1.docx
   >>> Word found in ./documents/dir/file6.pdf
   >>> Word found in ./documents/dir/subdir/file5.pdf
   ```

4.  搜索单词`20€`，仅存在于两个ISO文件（具有不同编码）和UTF8文件中：

   ```
   $ python scan.py -w 20€
   >>> Word found in ./documents/example_iso.txt
   >>> Word found in ./documents/example_output_iso.txt
   >>> Word found in ./documents/example_utf8.txt
   ```

5.  搜索不区分大小写。搜索单词`BETTER`，仅存在于`zen_of_python.txt`文件中：

   ```
   $ python scan.py -w BETTER
   >>> Word found in ./documents/zen_of_python.txt
   ```

### 工作原理...

`scan.py`文件包含以下元素：

1.  一个解析输入参数并为命令行创建帮助的入口点。
2.  一个遍历目录并分析找到的每个文件的主函数。根据它们的扩展名，它决定是否有可用的函数来处理和搜索它们。
3.  一个`EXTENSION`字典，将扩展名与搜索它们的函数配对。
4.  `search_txt`、`search_csv`、`search_pdf`和`search_docx`函数，用于处理和搜索每种文件类型的所需单词。

请记住，文件扩展名只是文件名的结尾，只是文件格式的一个提示。因此，应该谨慎对待它们。在Python标准库中，有一个函数`mimetypes.guess_type`，可以对文件类型进行有根据的猜测。请在此处查看Python文档：https://docs.python.org/3.8/library/mimetypes.html。

比较不区分大小写，因此搜索词被转换为小写，并且在所有比较中，文本都被转换为小写。

每个搜索函数都有其自身的特点：

1.  `search_txt`首先打开文件以确定其编码，使用`UnicodeDammit`，然后打开文件并逐行读取。一旦找到单词，它就会停止并返回成功。
2.  `search_csv`以CSV格式打开文件，不仅逐行迭代，还逐列迭代。一旦找到单词，它就会返回。
3.  `search_pdf`打开文件，如果是加密的则退出。如果不是，它会逐页进行，提取文本并与单词进行比较。一旦找到匹配项，它就会返回。
4.  `search_docx`打开文件并遍历其所有段落以查找匹配项。一旦找到匹配项，函数就会返回。

### 更多内容...

可以实施一些额外的想法：

- 可以添加更多的搜索函数。在本章中，我们介绍了日志文件和图像，以及文本文件。

搜索与读取本地文件

- 类似的结构也可用于搜索文件并仅返回最近的10个。
- `search_csv` 未进行方言嗅探检测。此功能也可添加。
- 读取过程是相当顺序化的。应能并行读取文件，通过分析它们来加快返回速度。请注意，并行读取文件可能导致排序问题，因为文件处理顺序可能不一致。

### 另请参阅

- 本章前面的 *爬取与搜索目录* 配方，了解如何在目录中搜索和查找文件。
- 本章前面的 *读取文本文件* 配方，了解打开和读取基本文本文件的基础知识。
- 本章前面的 *处理编码* 配方，了解如何以不同编码打开文件。
- 本章前面的 *读取CSV文件* 配方，了解如何读取CSV文件。
- 本章前面的 *读取PDF文件* 配方，了解如何打开和读取PDF文档。
- 本章前面的 *读取Word文档* 配方，了解读取Word文档的基础知识。

## 5 生成精彩报告

本章将介绍如何编写自动化文档并执行相关操作，例如处理不同格式的模板。我们将涵盖从纯文本等简单选项到Markdown等更丰富可能性的选项。我们还将介绍Word和PDF等标准格式。这两种格式可以说是全球范围内共享文档和报告最常用的方式。

本章将涵盖以下配方：

- 创建纯文本简单报告
- 使用报告模板
- Markdown格式化文本
- 编写基础Word文档
- Word文档样式设置
- Word文档结构生成
- Word文档添加图片
- 编写简单PDF文档
- PDF结构化
- PDF报告聚合
- PDF水印与加密

我们将从最简单的纯文本报表生成开始。

### 创建纯文本简单报告

创建报告最简单的方式是生成纯文本并将其存储在文件中。尽管与我们稍后将看到的其他格式相比，这可能显得过于简单，但请不要低估其实用性。纯文本是最容易共享的格式，因为它几乎适用于所有环境，并且文本信息在表示信息方面可以发挥很大作用。

### 准备工作

在本配方中，我们将生成一份关于上月观看电影数量和总时长的简短文本格式报告。内部，待表示的原始数据将以Python字典的形式存在。报告还将包含生成日期以供参考。

### 操作步骤...

1. 导入datetime：
   ```python
   >>> from datetime import datetime
   ```

2. 创建包含文本格式报告的模板：
   ```python
   >>> TEMPLATE = '''
   Movies report
   --------------

   Date: {date}
   Movies seen in the last 30 days: {num_movies}
   Total minutes: {total_minutes}
   '''
   ```

3. 创建包含待存储值的字典。请注意，这是要在报告中呈现的数据：
   ```python
   >>> data = {
   ...     'date': datetime.utcnow(),
   ...     'num_movies': 3,
   ...     'total_minutes': 376,
   ... }
   ```

4. 组合报告文本，将数据添加到模板中：
   ```python
   >>> report = TEMPLATE.format(**data)
   ```

5. 创建以当前日期命名的新文件并存储报告：
   ```python
   >>> FILENAME_TMPL = "{date}_report.txt"
   >>> filename = FILENAME_TMPL.format(date=data['date'].strftime('%Y-%m-%d'))
   >>> filename
   2020-01-26_report.txt
   >>> with open(filename, 'w') as file:
   ...     file.write(report)
   ```

6. 检查新创建的报告：
   ```
   $ cat 2020-01-26_report.txt
   Movies report
   -----------------

   Date: 2020-01-26 23:40:08.737671
   Movies seen in the last 30 days: 3
   Total minutes: 376
   ```

### 工作原理...

*操作步骤...* 部分的步骤2和步骤3设置了一个简单的模板并添加了包含所有数据的字典。然后，在*步骤4*中，这两者被组合成特定的报告。

> 在*步骤4*中，字典与模板结合。请注意，字典中的键对应于模板中的参数。这里的技巧是在`format`调用中使用双星号来解包字典，将每个键作为参数传递给`format()`。

在*步骤5*中，生成的报告（一个字符串）被存储在新创建的文件中。我们使用`with`上下文管理器配合`open()`，如前面章节所述。在这种情况下，我们生成一个新文件来写入数据。关闭`with`块后，文件被正确关闭，数据被存储在磁盘上。

> 打开模式决定了如何打开文件，是读取还是写入，以及文件是文本还是二进制。`w`模式打开文件以进行写入，如果文件已存在则覆盖它。请小心不要误删现有文件！

步骤6检查文件是否已用正确的数据创建。

### 更多内容...

文件名使用动态日期创建，以最小化覆盖现有文件的概率。选择从年份开始到日期结束的日期格式，以便文件按正确顺序排序。

> 日期格式YYYY-MM-DD在ISO 8601标准中有介绍，该标准描述了不同的日期和时间格式化方式。这是一种易于解析且在Python中受支持的标准格式。

`with`上下文管理器将关闭文件，即使发生异常。如果出现错误，`write`调用将引发`IOError`异常。

> 写入磁盘时的一些常见问题包括权限问题、硬盘空间不足或路径问题（例如，尝试写入不存在的目录）。

请注意，文件可能在关闭或显式刷新之前不会完全提交到磁盘。通常，操作系统会处理这个问题，但如果您尝试两次打开同一个文件，一次用于读取，一次用于写入，请记住这一点。

> 如果程序在将数据刷新到硬盘之前突然结束，这可能会导致错误，使数据似乎消失。如果需要，调用`file.flush()`强制将数据提交到磁盘。这在多次写入同一文件时很有用。请注意，在`with`块结束时，文件将自动刷新并关闭。

### 另请参阅

- 本章后面的 *使用报告模板* 配方，了解HTML模板。
- 本章后面的 *Markdown格式化文本* 配方，了解Markdown。
- 本章后面的 *PDF报告聚合* 配方，了解如何生成PDF报告。

### 使用报告模板

虽然纯文本可以传达大量信息，但要生成更好的报告，我们需要一个可以为文本添加样式的系统。粗体文本、项目符号和图像等细节可以产生很大差异。由于所有浏览器都使用HTML，因此以这种格式生成报告是一个不错的选择。每个人都熟悉浏览器渲染文本。

HTML是一种非常灵活的格式，可用于渲染富文本和报告。虽然HTML模板可以作为纯纯文本管理，但这样做非常容易出错且繁琐。有一些工具允许您更好地处理结构化文本并定义模板。

这还将模板与代码分离，将数据的生成与数据的表示分开。模板的样式可以由专业设计师单独完成，使其看起来很棒。

### 准备工作

本配方中使用的工具Jinja2读取包含模板的文件并应用上下文。上下文包含要显示的数据。

我们应该首先安装模块：

```
$ echo "jinja2==2.11.1" >> requirements.txt
$ pip install -r requirements.txt
```

Jinja2使用自己的语法，这是HTML和Python的混合体。它针对HTML文档，因此可以轻松执行正确转义特殊字符等操作。

在GitHub仓库中，https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter05，我们包含了一个名为jinja_template.html的模板文件，其中包含要使用的模板。

### 操作步骤...

1. 导入Jinja2 Template和datetime：
   ```python
   >>> from jinja2 import Template
   >>> from datetime import datetime
   ```

2. 将模板从文件读入内存：
   ```python
   >>> with open('jinja_template.html') as file:
   ...     template = Template(file.read())
   ```

生成出色的报告

3.  创建包含要显示数据的上下文：

```python
>>> context = {
...     'date': datetime.now(),
...     'movies': ['Casablanca', 'The Sound of Music', 'Vertigo'],
...     'total_minutes': 404,
... }
```

4.  渲染模板并将结果写入新文件 `report.html`：

```python
>>> with open('report.html', 'w') as file:
...     file.write(template.render(context))
```

5.  退出 Python 解释器，并在浏览器中打开 `report.html` 文件：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_155_0.png)

图 5.1：渲染后的 report.html

### 工作原理...

*如何操作...* 部分中的步骤 2 和 4 非常直接：它们读取模板，然后渲染并保存生成的报告。

如步骤 3 和 4 所示，主要任务是创建一个包含要显示信息的上下文字典。然后模板会渲染这些信息，如步骤 5 所示。让我们来看一下 `jinja_template.html`：

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <title> Movies Report</title>
</head>
<body>
    <h1>Movies Report</h1>
    <p>Date {{date}}</p>
    <p>Movies seen in the last 30 days: {{movies|length}}</p>
    <ol>
        {% for movie in movies %}
        <li>{{movie}}</li>
        {% endfor %}
    </ol>
    <p>Total minutes: {{total_minutes}} </p>
</body>
</html>
```

其中大部分是替换花括号之间定义的上下文值，例如 `{{total_minutes}}`。

注意标签 `{% for ... %}` / `{% endfor %}`，它定义了一个循环。这允许以一种非常 Pythonic 的方式生成列表中的多行或多个元素。

可以对变量应用过滤器来修改它们。在这个例子中，`length` 过滤器应用于 `movies` 列表，通过管道符号 `|` 获取其大小，如 `{{movies|length}}` 所示。

### 更多内容...

除了 `{% for %}` 标签外，还有一个 `{% if %}` 标签，允许条件显示：

```jinja
{% if movies|length > 5 %}
    Wow, so many movies this month!
{% else %}
    Regular number of movies
{% endif %}
```

已经定义了许多过滤器（完整列表请参见此处：http://jinja.pocoo.org/docs/2.11/templates/#list-of-builtin-filters）。也可以定义自定义过滤器。

> 请注意，你可以使用过滤器在模板中添加大量处理和逻辑。虽然少量使用没问题，但请尽量限制模板中的逻辑量。大多数用于显示数据的计算都应事先完成，让模板只负责显示值。这使得上下文非常直接，并简化了模板，便于进行更改。

处理 HTML 文件时，最好对变量进行自动转义。这意味着特殊字符可能会被解释为 HTML 语法的一部分，而不是纯文本；例如，`<` 字符将被替换为等效的 HTML 代码，以便在 HTML 页面上正确显示。为此，请使用 `autoescape` 参数创建模板。查看此处的区别：

```python
>>> Template('{{variable}}', autoescape=False).render({'variable': '<'})
'<'
>>> Template('{{variable}}', autoescape=True).render({'variable': '<'})
'&lt;'
```

转义可以使用 `e` 过滤器（表示 *escape*）应用于每个变量，也可以使用 `safe` 过滤器（表示可以安全地按原样渲染）来取消应用。

Jinja2 模板是可扩展的，这意味着你可以创建一个 `base_template.html` 文件，然后扩展它，更改其中的一些元素。你也可以包含其他文件，对不同部分进行分区和分离。更多详细信息请参阅完整文档。

> Jinja2 非常强大，允许我们创建复杂的 HTML 模板，也可以用于其他格式，如 LaTeX 或 JavaScript，尽管这需要进行配置。我鼓励你阅读完整文档，了解其所有功能！

完整的 Jinja2 文档可在此处找到：http://jinja.pocoo.org/docs/2.11/.

### 另请参阅

-   本章前面的 *创建纯文本文本报告* 配方，了解创建纯文本文本报告的基础知识。
-   本章后面的 *在 Markdown 中格式化文本* 配方，了解 Markdown，一种替代的模板格式。

### 在 Markdown 中格式化文本

Markdown 是一种非常流行的标记语言，用于创建可以转换为样式化 HTML 的纯文本。这是一种很好的结构化文档的方式，使得文档在纯文本格式下仍然易于阅读，同时又能在 HTML 中正确设置样式。

在这个配方中，我们将看到如何使用 Python 将 Markdown 文档转换为样式化的 HTML。

### 准备工作

我们应该首先安装 `mistune` 模块，它将把 Markdown 文档编译成 HTML：

```bash
$ echo "mistune==0.8.4" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 仓库中，有一个名为 `markdown_template.md` 的模板文件，其中包含要生成的报告模板。

### 如何操作...

1.  导入 `mistune` 和 `datetime`：

```python
>>> import mistune
>>> from datetime import datetime
```

2.  从文件中读取模板：

```python
>>> with open('markdown_template.md') as file:
...     template = file.read()
```

3.  设置要包含在报告中的数据的上下文：

```python
>>> context = {
...     'date': datetime.now(),
...     'pmovies': ['Casablanca', 'The Sound of Music', 'Vertigo'],
...     'total_minutes': 404,
... }
```

4.  由于电影需要以项目符号列表的形式显示，我们将把列表转换为合适的 Markdown 项目符号列表。同时，我们将存储电影的数量：

```python
>>> context['num_movies'] = len(context['pmovies'])
>>> context['movies'] = '\n'.join('* {}'.format(movie) for movie in context['pmovies'])
```

5.  渲染模板并将生成的 Markdown 编译为 HTML：

```python
>>> md_report = template.format(**context)
>>> report = mistune.markdown(md_report)
```

6.  最后，将生成的报告存储在 `report.html` 文件中：

```python
>>> with open('report.html', 'w') as file:
...     file.write(report)
```

7.  在浏览器中打开 `report.html` 文件以检查结果：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_159_0.png)

### 工作原理...

*如何操作...* 部分中的步骤 2 和 3 准备了模板和要显示的数据。在步骤 4 中，生成了额外的信息——电影数量，这是从 `movies` 元素派生出来的。然后，`movies` 元素从 Python 列表转换为有效的 Markdown 元素。注意换行符和开头的 `*`，它们将被渲染为项目符号：

```python
>>> '\n'.join('* {}'.format(movie) for movie in context['pmovies'])
'* Casablanca\n* The Sound of Music\n* Vertigo'
```

在步骤 5 中，生成了 Markdown 格式的模板。这种原始形式的格式非常易读，这是 Markdown 的优点：

```markdown
# Movies Report
======

Date: 2018-06-29 20:47:18.930655

Movies seen in the last 30 days: 3

* Casablanca
* The Sound of Music
* Vertigo

Total minutes: 404
```

然后，在步骤 6 中，使用 mistune 将报告转换为 HTML 并存储到文件中。

### 更多内容...

学习 Markdown 非常有用，因为它被许多常见网页支持，作为一种启用易于读写且可以渲染为样式化格式的文本输入方式。一些例子包括 GitHub、Stack Overflow 和大多数博客平台。

> 实际上不止一种 Markdown。这是因为官方定义有限或模糊，并且人们对其澄清或扩展的兴趣不大。这导致了几种略有不同的实现，例如 GitHub Flavored Markdown、MultiMarkdown 和 CommonMark。

Markdown 中的文本相当易读，但如果你需要交互式地查看其外观，可以使用 Dillinger 在线编辑器：https://dillinger.io/。

Mistune 的完整文档可在此处找到：http://mistune.readthedocs.io/en/latest/。

完整的 Markdown 语法可在 https://daringfireball.net/projects/markdown/syntax 找到，一个包含最常用元素的优秀备忘单可在 https://www.markdownguide.org/cheat-sheet/ 找到。

生成出色的报告

### 另请参阅

-   本章前面的*在纯文本中创建简单报告*方法，了解创建纯文本报告的基础知识。
-   本章前面的*使用模板生成报告*方法，了解如何直接在HTML中创建模板。

### 编写基本的Word文档

微软（MS）Office是最常见的软件之一，其中MS Word几乎是可编辑文档的事实标准。通过自动化脚本可以生成docx文档，这有助于以易于在许多企业中共享的格式分发报告。

在本方法中，我们将学习如何通过编程方式生成完整的Word文档。

### 准备工作

我们将使用python-docx模块来处理Word文档：

```
$ echo "python-docx==0.8.10" >> requirements.txt
$ pip install -r requirements.txt
```

### 操作步骤...

1.  导入python-docx和datetime：
    ```python
    >>> import docx
    >>> from datetime import datetime
    ```

2.  定义包含要存储在报告中的数据的上下文：
    ```python
    >>> context = {
    ...     'date': datetime.now(),
    ...     'movies': ['Casablanca', 'The Sound of Music', 'Vertigo'],
    ...     'total_minutes': 404,
    ... }
    ```

3.  创建一个新的docx文档并添加一个标题，电影报告：
    ```python
    >>> document = docx.Document()
    >>> document.add_heading('Movies Report', 0)
    ```

4.  添加一个描述日期的段落，日期使用斜体：
    ```python
    >>> paragraph = document.add_paragraph('Date: ')
    >>> paragraph.add_run(str(context['date'])).italic = True
    ```

5.  在另一个段落中添加关于观看电影数量的信息：
    ```python
    >>> paragraph = document.add_paragraph('Movies seen in the last 30 days: ')
    >>> paragraph.add_run(str(len(context['movies']))).italic = True
    ```

6.  将每部电影添加为项目符号：
    ```python
    >>> for movie in context['movies']:
    ...     document.add_paragraph(movie, style='List Bullet')
    ```

7.  添加总分钟数并保存文件，如下所示：
    ```python
    >>> paragraph = document.add_paragraph('Total minutes: ')
    >>> paragraph.add_run(str(context['total_minutes'])).italic = True
    >>> document.save('word-report.docx')
    ```

8.  关闭解释器并打开`word-report.docx`文件进行检查：

    ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_162_0.png)

    图5.3：word-report.docx的内容

### 工作原理...

Word文档的基础是它被划分为段落，每个段落又被划分为文本块。文本块是段落中共享相同样式的一部分。

*操作步骤...*部分中的步骤1和步骤2是导入和定义将要存储在报告中的数据的准备工作。

在*步骤3*中，创建了文档并添加了带有适当标题的标题。这会自动设置文本样式。

*步骤4*介绍了处理段落的方法。基于文本创建了一个具有默认样式的新段落，但可以添加新的文本块来更改它。在这里，我们添加了第一个包含文本*Date*的文本块，但之后又添加了另一个包含具体时间的文本块，并将格式更改为*斜体*。

在*步骤5*和*步骤6*中，我们可以看到关于电影的信息。第一部分存储电影数量，格式与*步骤4*类似。之后，电影被逐一添加到报告中，并将样式设置为项目符号，样式为"*List Bullet*"。

最后，*步骤7*以与*步骤4*类似的方式存储所有电影的总运行时间，并将文档保存到文件中。

### 更多内容...

如果需要在文档中引入额外的行用于格式化目的，请添加空段落。

由于MS Word格式的工作方式，没有简单的方法来确定文档将有多少页。你可能需要对尺寸进行一些测试，尤其是在生成动态文档时。

> 即使你生成docx文件，也不一定需要MS Office。还有其他应用程序可以打开和处理这些文件，包括免费的替代品，如LibreOffice。

完整的python-docx文档可在此处获取：https://python-docx.readthedocs.io/en/latest/.

### 另请参阅

-   本章后面的*设置Word文档样式*方法，了解如何格式化文档。
-   本章后面的*在Word文档中生成结构*方法，了解如何在Word文档中创建节和其他分隔符。

### 设置Word文档样式

Word文档可以包含几乎没有格式的文本，但我们也可以添加样式来帮助我们理解显示的内容。Word有一组预定义的样式，可用于变化文档并突出显示其重要部分。

### 准备工作

我们将使用python-docx模块来处理Word文档：

```
$ echo "python-docx==0.8.10" >> requirements.txt
$ pip install -r requirements.txt
```

### 操作步骤...

1.  导入python-docx模块：
    ```python
    >>> import docx
    ```

2.  创建一个新文档：
    ```python
    >>> document = docx.Document()
    ```

3.  添加一个段落，以不同方式突出显示一些单词（斜体、粗体和下划线）：
    ```python
    >>> p = document.add_paragraph('This shows different kinds of emphasis: ')
    >>> p.add_run('bold').bold = True
    >>> p.add_run(', ')
    <docx.text.run.Run object at ...>
    >>> p.add_run('italics').italic = True
    >>> p.add_run(' and ')
    <docx.text.run.Run object at ...>
    >>> p.add_run('underline').underline = True
    >>> p.add_run('.')
    <docx.text.run.Run object at ...>
    ```

4.  创建一些段落并使用默认样式设置它们的样式，例如List Bullet、List Number或Quote：
    ```python
    >>> document.add_paragraph('a few', style='List Bullet')
    <docx.text.paragraph.Paragraph object at ...>
    >>> document.add_paragraph('bullet', style='List Bullet')
    <docx.text.paragraph.Paragraph object at ...>
    >>> document.add_paragraph('points', style='List Bullet')
    <docx.text.paragraph.Paragraph object at ...>
    >>>
    >>> document.add_paragraph('Or numbered', style='List Number')
    <docx.text.paragraph.Paragraph object at ...>
    >>> document.add_paragraph('that will', style='List Number')
    <docx.text.paragraph.Paragraph object at ...>
    >>> document.add_paragraph('that keep', style='List Number')
    <docx.text.paragraph.Paragraph object at ...>
    >>> document.add_paragraph('count', style='List Number')
    <docx.text.paragraph.Paragraph object at ...>
    >>>
    >>> document.add_paragraph('And finish with a quote', style='Quote')
    <docx.text.paragraph.Paragraph object at 0x10d2336d8>
    ```

5.  创建一个具有不同字体和大小的段落。我们将使用Arial字体和25磅的大小。该段落将右对齐：
    ```python
    >>> from docx.shared import Pt
    >>> from docx.enum.text import WD_ALIGN_PARAGRAPH
    >>> p = document.add_paragraph('This paragraph will have a manual styling and right alignment')
    >>> p.runs[0].font.name = 'Arial'
    >>> p.runs[0].font.size = Pt(25)
    >>> p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    ```

6.  保存文档：
    ```python
    >>> document.save('word-report-style.docx')
    ```

7.  打开word-report-style.docx文档以验证其内容：

    ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_166_0.png)

    图5.4：最终的word-report-style.docx文档

### 工作原理...

在*步骤1*创建文档后，*操作步骤...*部分的*步骤2*添加了一个包含多个文本块的段落。在Word中，一个段落可以包含多个文本块，这些文本块是可能具有不同样式的较小部分。通常，与单个单词相关的任何格式更改都将应用于文本块，而影响整个段落的更改将应用于`paragraph`对象。

每个文本块默认使用`Normal`样式创建。`.bold`、`.italic`或`.underline`的任何属性都可以更改为`True`以设置文本块是否应采用适当的样式或这些样式的组合。`False`值将停用它，而`None`值将应用配置的默认值。

> 请注意，此协议中的正确单词是*italic*，而不是*italics*。将属性设置为italics不会有任何效果，但也不会显示错误。

*步骤4*展示了如何应用一些段落的默认样式；在这种情况下，用于显示项目符号、编号列表和引号。还有更多包含的样式，可以在以下文档的此页面上找到：https://python-docx.readthedocs.io/en/latest/user/styles-understanding.html?highlight=List%20Bullet#paragraph-styles-in-default-template。尝试找出哪些最适合你的文档。

生成出色的报告

运行对象的 `.font` 属性在*步骤 5*中展示。这允许你手动设置特定的字体和大小。请注意，大小需要使用正确的 `Pt`（磅）对象来指定。

> 磅在常规 Word 文档中很常见，但有时很难知道确切的大小。不要害怕进行一些测试和实验。

段落的对齐方式在 `paragraph` 对象中设置，并使用一个常量来定义是左对齐、右对齐、居中还是两端对齐。所有对齐选项可以在这里找到：https://python-docx.readthedocs.io/en/latest/api/enum/WdAlignParagraph.html。

最后，*步骤 7* 将文件保存到文件系统。

### 还有更多...

`font` 属性也可用于设置文本的更多属性，例如小型大写字母、阴影、浮雕或删除线。所有可能性的完整范围在以下文档中展示：https://python-docx.readthedocs.io/en/latest/api/text.html#docx.text.run.Font。

另一个可用的选项是更改文本颜色：

```
>>> from docx.shared import RGBColor
>>> DARK_BLUE = RGBColor.from_string('1b3866')
>>> run.font.color.rgb = DARK_BLUE
```

颜色可以用通常的十六进制格式从字符串描述。尝试将所有颜色定义为命名常量，以确保它们都一致，并且在报告中最多限制使用三种颜色，以免分散对内容的注意力。

> 你可以使用在线颜色选择器，例如这个：https://www.w3schools.com/colors/colors_picker.asp。记住不要使用开头的 #。如果你需要生成调色板，使用像 https://coolors.co/ 这样的工具来生成好的组合是个好主意。

完整的 python-docx 文档可以在这里找到：https://python-docx.readthedocs.io/en/latest/.

### 另请参阅

- 本章前面的*编写基本 Word 文档*食谱，学习创建 Word 文档的基础知识。
- 接下来的*在 Word 文档中生成结构*食谱，学习如何在 Word 文档中创建节和其他分隔符。

### 在 Word 文档中生成结构

要创建适当的专业报告，它们需要被正确地结构化。MS Word 文档没有页面的概念，因为它以段落为单位工作。但我们可以引入分节符和节来适当地划分文档。

我们将在本食谱中了解如何创建结构化的 Word 文档，引入分节符来创建节。

### 准备工作

我们将使用 python-docx 模块来处理 Word 文档：

```
$ echo "python-docx==0.8.10" >> requirements.txt
$ pip install -r requirements.txt
```

### 如何操作...

1. 导入 python-docx 模块：
    ```
    >>> import docx
    ```
2. 创建一个新文档：
    ```
    >>> document = docx.Document()
    ```
3. 创建一个带有换行符的段落：
    ```
    >>> p = document.add_paragraph('This is the start of the paragraph')
    >>> run = p.add_run()
    >>> run.add_break(docx.text.run.WD_BREAK.LINE)
    >>> p.add_run('And now this in a different line')
    >>> p.add_run('. Even if it's on the same paragraph.')
    ```
4. 创建一个分页符并写一个段落：
    ```
    >>> document.add_page_break()
    >>> document.add_paragraph('This appears in a new page')
    ```
5. 创建一个新节，该节将是横向页面：
    ```
    >>> section = document.add_section(docx.enum.section.WD_SECTION.NEW_PAGE)
    >>> section.orientation = docx.enum.section.WD_ORIENT.LANDSCAPE
    >>> section.page_height, section.page_width = section.page_width, section.page_height
    >>> document.add_paragraph('This is part of a new landscape section')
    ```
6. 创建另一个节，恢复为纵向方向：
    ```
    >>> section = document.add_section(docx.enum.section.WD_SECTION.NEW_PAGE)
    >>> section.orientation = docx.enum.section.WD_ORIENT.PORTRAIT
    >>> section.page_height, section.page_width = section.page_width, section.page_height
    >>> document.add_paragraph('In this section, recover the portrait orientation')
    ```
7. 保存文档：
    ```
    >>> document.save('word-report-structure.docx')
    ```
8. 通过打开文档并检查生成的节来检查结果：

    ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_170_0.png)
    图 5.5：渲染的第一页

    检查新页面：

    ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_170_1.png)
    图 5.6：渲染的新文件

    检查横向节：

    ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_171_0.png)
    图 5.7：渲染的横向节

    然后，恢复为纵向方向：

    ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_172_0.png)
    图 5.8：恢复为纵向方向后的新页面

### 工作原理...

在*如何操作...*部分的*步骤 2*中创建文档后，我们为第一个节添加一个段落。请注意，文档以一个节开始。该段落在段落中间引入了一个换行符。

> 💡 段落中的换行符和新段落之间有一个小的区别，尽管对于大多数用途来说，它们非常相似。尝试实验一下。

在*步骤 3*中引入了分页符，没有更改节。

*步骤 4*在新页面上创建一个新节。*步骤 5*还将该节中的页面方向更改为横向。在*步骤 6*中，引入了一个新节，方向恢复为纵向。

> 请注意，当更改方向时，我们还需要交换宽度和高度。每个新节都继承前一个节的属性，因此这种交换也需要在*步骤 6*中进行。

最后，文档在*步骤 7*中保存。

### 还有更多...

一个节规定了页面布局，包括页面的方向和大小。页面大小可以使用长度选项来更改，例如 `Inches` 或 `Cm`：

```
>>> from docx.shared import Inches, Cm
>>> section.page_height = Inches(10)
>>> section.page_width = Cm(20)
```

页面边距也可以用同样的方式定义：

```
>>> section.left_margin = Inches(1.5)
>>> section.right_margin = Cm(2.81)
>>> section.top_margin = Inches(1)
>>> section.bottom_margin = Cm(2.54)
```

节也可以强制不仅在下一页开始，而且在下一个奇数页开始，这在双面打印时看起来会更好：

```
>>> document.add_section(docx.enum.section.WD_SECTION.ODD_PAGE)
```

完整的 `python-docx` 文档可以在这里找到：https://python-docx.readthedocs.io/en/latest/.

### 另请参阅

- 本章前面的*编写基本 Word 文档*食谱，学习创建 Word 文档的基础知识。
- 本章前面的*设置 Word 文档样式*食谱，学习如何为文档添加格式。

### 向 Word 文档添加图片

Word 文档可以包含图像以显示图表或任何其他类型的额外信息。添加图像是创建丰富报告的好方法。

> 任何有经验的 Word 用户都会知道，正确定位图像可能会有多令人沮丧，因为周围的环境可能会使其改变。请记住，虽然以编程方式定位图像会有所帮助，因为它将被包含在特定位置，但更改周围的段落可能会改变其渲染方式。

在本食谱中，我们将学习如何将现有的图像文件附加到 Word 文档。

### 准备工作

我们将使用 `python-docx` 模块来处理 Word 文档：

```
$ echo "python-docx==0.8.10" >> requirements.txt
$ pip install -r requirements.txt
```

我们需要准备一张要包含在文档中的图像。我们将使用 GitHub 上的文件 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter04/images/photo-dublin-a1.jpg，它展示了都柏林的景色。你可以在命令行下载它，像这样：

```
$ wget https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter04/images/photo-dublin-a1.jpg
```

### 如何操作...

1. 导入 `python-docx` 模块：
```python
>>> import docx
```

2. 创建一个新文档：
```python
>>> document = docx.Document()
```

3. 创建一个包含一些文本的段落：
```python
>>> document.add_paragraph('This is a document that includes a picture taken in Dublin')
```

4. 添加图片：
```python
>>> image = document.add_picture('photo-dublin-a1.jpg')
```

5. 按比例缩放图片以适应页面（14 x 10 厘米）：
```python
>>> from docx.shared import Cm
>>> image.width = Cm(14)
>>> image.height = Cm(10)
```

6. 图片已被添加到一个新段落中。将其居中对齐并添加描述性文本：
```python
>>> paragraph = document.paragraphs[-1]
>>> from docx.enum.text import WD_ALIGN_PARAGRAPH
>>> paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
>>> paragraph.add_run().add_break()
>>> paragraph.add_run('A picture of Dublin')
```

7. 添加一个包含额外文本的新段落，并保存文档：
```python
>>> document.add_paragraph('Keep adding text after the image')
<docx.text.paragraph.Paragraph object at XXX>
>>> document.save('report.docx')
```

8. 检查结果：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_176_0.png)

图 5.9：包含图片的 report.docx 文档

### 工作原理...

前几个步骤（*如何操作...* 部分的 *步骤 1 到 3*）创建了文档并添加了一些文本。

*步骤 4* 从文件中添加图片，而 *步骤 5* 将其调整为合适的大小。默认情况下，图片太大。

生成精彩报告

> 调整大小时请注意图片的比例。注意，你也可以使用其他度量单位，例如 *Inch*，它同样在 *shared* 模块中定义。

插入图片也会创建一个新段落，因此可以为该段落设置样式以对齐图片或添加更多文本，例如引用或描述。在 *步骤 6* 中，通过 `document.paragraph` 属性获取该段落。获取最后一个段落并适当地设置样式，将其居中对齐。添加一个新行和一个包含描述性文本的 `run`。

*步骤 7* 在图片后添加额外文本并保存文档。

### 更多内容...

可以更改图片的大小，但正如我们之前所见，如果大小发生变化，需要计算图片的比例。如果像 *如何操作...* 部分的 *步骤 5* 中那样通过近似值进行调整，调整可能最终并不完美。

> 注意，图片的比例并非完美的 10:14，而是 10:13.33。对于照片来说，这可能已经足够好，但对于比例变化敏感的图像（如图表），则可能需要格外小心。

要获得正确的比例，请用高度除以宽度，然后进行适当的缩放：

```python
>>> image = document.add_picture('photo-dublin-a1.jpg')
>>> image.height / image.width
0.75
>>> RELATION = image.height / image.width
>>> image.width = Cm(12)
>>> image.height = Cm(12 * RELATION)
```

如果需要将值转换为特定尺寸，可以使用 `cm`、`inches`、`mm` 或 `pt` 属性：

```python
>>> image.width.cm
12.0
>>> image.width.mm
120.0
>>> image.width.inches
4.724409448818897
>>> image.width.pt
340.15748031496065
```

完整的 python-docx 文档可在此处获取：https://python-docx.readthedocs.io/en/latest/.

### 另请参阅

- 本章前面的 *编写基本 Word 文档* 配方，以了解处理 Word 文档的基础知识。
- 本章前面的 *设置 Word 文档样式* 配方，以了解如何为文档添加丰富的格式。
- 本章前面的 *在 Word 文档中生成结构* 配方，以了解如何向文档添加节和其他分隔符。

### 编写简单的 PDF 文档

PDF 文件是共享报告的常见格式。PDF 文档的主要特点是它们精确定义了文档的外观和打印方式，并且在生成后是只读的。这使得它们在传输信息时非常直接明了。

在这个配方中，我们将看到如何使用 Python 编写一个简单的 PDF 报告。

### 准备工作

我们将使用 fpdf 模块来创建 PDF 文档：

```bash
$ echo "fpdf==1.7.2" >> requirements.txt
$ pip install -r requirements.txt
```

### 如何操作...

1. 导入 fpdf 模块：
```python
>>> import fpdf
```

2. 创建一个文档：
```python
>>> document = fpdf.FPDF()
```

3. 定义标题的字体和颜色，并添加第一页：
```python
>>> document.set_font('Times', 'B', 14)
>>> document.set_text_color(19, 83, 173)
>>> document.add_page()
```

4. 编写文档标题：
```python
>>> document.cell(0, 5, 'PDF test document')
>>> document.ln()
```

5. 编写一个长段落：
```python
>>> document.set_font('Times', '', 12)
>>> document.set_text_color(0)
>>> document.multi_cell(0, 5, 'This is an example of a long paragraph. ' * 10)
[]
>>> document.ln()
```

6. 编写另一个长段落：
```python
>>> document.multi_cell(0, 5, 'Another long paragraph. Lorem ipsum dolor sit amet, consectetur adipiscing elit.' * 20)
[]
```

7. 保存文档：
```python
>>> document.output('report.pdf')
''
```

8. 检查 report.pdf 文档：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_179_0.png)

### 工作原理...

`fpdf` 模块创建一个 PDF 文档并允许我们向其中写入内容。

> 由于 PDF 的特殊性，思考它的最佳方式是想象一个光标在文档中书写并移动到下一个位置，类似于打字机。

首先的操作是指定要使用的字体和大小，然后添加第一页。这在 *步骤 3* 中完成。第一个字体是粗体（第二个参数 'B'），并且比文档其余部分的字体更大。这使其成为标题。颜色也通过 `.set_text_color` 以 RGB 分量进行设置。

> 文本也可以用 *I* 设置为斜体，用 <u>U</u> 设置为下划线。你可以组合它们，因此 **BI** 将产生同时为粗体和斜体的文本。

`.cell` 调用创建一个包含指定文本的文本框。前两个参数是宽度和高度。宽度 0 使用从左边距到右边距的整个空间。高度 5（毫米）对于 12 号字体是合适的。调用 `.ln` 引入一个新行。

要编写多行段落，我们使用 `.multi_cell` 方法。它的参数与 `.cell` 相同。在 *步骤 5* 和 *步骤 6* 中编写了两个段落。注意之前字体的更改，以区分标题和报告正文。`.set_text_color` 使用单个参数调用，以灰度设置颜色。在这种情况下，使用 0 表示黑色。

> 对长文本使用 `cell` 会使其超出边距并超出页面。仅将其用于适合单行的文本。你可以使用 `.get_string_width` 找到字符串的大小。

文档在 *步骤 7* 中保存到磁盘。

### 更多内容...

如果 `multi_cell` 操作占用了页面上的所有可用空间，则会自动添加页面。调用 `.add_page` 将移动到新页面。

生成精彩报告

你可以使用任何默认字体（Courier、Helvetica 和 Times），或使用 `.add_font` 添加额外字体。有关更多详细信息，请查看文档：http://pyfpdf.readthedocs.io/en/latest/reference/add_font/index.html.

> Symbol 和 ZapfDingbats 字体也可用，但代表符号。如果你需要显示特殊字符，这可能很有用，但在使用之前请务必测试结果。其余的默认字体应满足你对衬线、无衬线和固定宽度情况的需求。在 PDF 中，使用的字体将嵌入文档中，因此它们将始终正确显示。

在整个文档中保持高度一致，至少在相同大小的文本之间保持一致。定义一个你满意的常量，并将其用于整个内容：

```python
>>> BODY_TEXT_HEIGHT = 5
>>> document.multi_cell(0, BODY_TEXT_HEIGHT, text)
```

默认情况下，文本将是两端对齐的，但这可以更改。使用 align 参数，可选 J（两端对齐）、C（居中）、R（右对齐）或 L（左对齐）。例如，这会产生左对齐的文本：

```python
>>> document.multi_cell(0, BODY_TEXT_HEIGHT, text, align='L')
```

完整的 FPDF 文档可在此处找到：http://pyfpdf.readthedocs.io/en/latest/index.html.

### 参见

- 接下来的 *构建 PDF 结构*，以了解如何向 PDF 文档添加分隔符。
- 本章后面的 *聚合 PDF 报告*，以了解如何将不同的 PDF 文档合并为一个。
- 本章后面的 *为 PDF 添加水印和加密*，以了解如何为文档添加安全措施。

### 构建 PDF 结构

在创建 PDF 时，可以自动生成一些元素，以更好地呈现和结构化你的元素。在这个配方中，我们将看到如何添加页眉和页脚，以及如何创建指向文档其他部分的链接。

### 准备工作

我们将使用 fpdf 模块来创建 PDF 文档：

```
$ echo "fpdf==1.7.2" >> requirements.txt
$ pip install -r requirements.txt
```

### 如何操作...

`structuring_pdf.py` 脚本可在 GitHub 上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter05/structuring_pdf.py。以下展示了最相关的部分：

```python
import fpdf
from random import randint

class StructuredPDF(fpdf.FPDF):
    LINE_HEIGHT = 5

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        page_number = 'Page {number}/{nb}'.format(number=self.page_no())
        self.cell(0, self.LINE_HEIGHT, page_number, 0, 0, 'R')

    def chapter(self, title, paragraphs):
        self.add_page()
        link = self.title_text(title)
        page = self.page_no()
        for paragraph in paragraphs:
            self.multi_cell(0, self.LINE_HEIGHT, paragraph)
            self.ln()

        return link, page

    def title_text(self, title):
        self.set_font('Times', 'B', 15)
        self.cell(0, self.LINE_HEIGHT, title)
        self.set_font('Times', '', 12)
        self.line(10, 17, 110, 17)
        link = self.add_link()
        self.set_link(link)
        self.ln()
        self.ln()

        return link

    def get_full_line(self, head, tail, fill):
        ...

    def toc(self, links):
        self.add_page()
        self.title_text('Table of contents')
        self.set_font('Times', 'I', 12)

        for title, page, link in links:
            line = self.get_full_line(title, page, '.')
            self.cell(0, self.LINE_HEIGHT, line, link=link)
            self.ln()

LOREM_IPSUM = ...

def main():
    document = StructuredPDF()
    document.alias_nb_pages()
    links = []
    num_chapters = randint(5, 40)
    for index in range(1, num_chapters):
        chapter_title = 'Chapter {}'.format(index)
        num_paragraphs = randint(10, 15)
        link, page = document.chapter(chapter_title,
                                      [LOREM_IPSUM] * num_paragraphs)
        links.append((chapter_title, page, link))

    document.toc(links)
    document.output('report.pdf')
```

1.  运行该脚本。它将生成 `report.pdf` 文件，其中包含一些章节和一个目录。请注意，它生成的具体内容是随机的，因此每次运行时具体数字都会有所不同：

```
$ python3 structuring_pdf.py
```

检查结果。这是一个示例：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_184_0.png)

检查末尾的目录：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_185_0.png)

### 工作原理...

让我们看看脚本的每个元素。

`StructuredPDF` 定义了一个继承自 `FPDF` 的类。这对于重写 `footer` 方法很有用，该方法在生成时为每个页面创建页脚。它还有助于简化 `main` 中的代码。

`main` 函数创建文档。它启动文档并添加每个章节，收集它们的链接信息。最后，它调用 `toc` 方法，使用链接信息生成目录。

> 要存储的文本是通过乘以 `LOREM_IPSUM` 文本生成的，这是一个占位符。

`chapter` 方法首先打印一个标题部分，然后添加每个段落。它收集章节开始的页码和 `title_text` 方法返回的链接并返回它们。

`title_text` 方法以更大、更粗的字体写入文本。然后，它添加一条线以将标题与章节正文分开。它在以下几行中生成并设置一个指向当前页面的 `link` 对象：

```python
link = self.add_link()
self.set_link(link)
```

此链接将用于目录中，以添加一个指向此章节的可点击元素。

`footer` 方法自动为每页添加页脚。它设置较小的字体，并添加包含当前页码（通过 `page_no` 获取）的文本，并使用 `{nb}`，它将被替换为总页数。

> 在 main 中调用 `.alias_nb_pages()` 确保在生成文档时 `{nb}` 被替换。

最后，目录在 `toc` 方法中生成。它写入标题并添加所有已收集的引用链接，这些链接作为链接、页码和章节名称，这是所需的所有信息。

### 更多内容...

注意使用 `randint` 为文档添加一些随机性。这个调用在 Python 标准库中可用，返回一个在定义的最大值和最小值之间的随机数。两者都包含在内。

`get_full_line` 方法为目录生成适当大小的行。它接受一个起始值（章节名称）和结束值（页码），并添加填充字符（点）的数量，直到行具有适当的宽度（120 毫米）。

为了计算文本的大小，脚本调用 `get_string_width`，它考虑了字体和大小。

链接对象可用于指向特定页面，而不是当前页面，也可以不指向页面的开头。要调整调用，请使用 `set_link(link, y=place, page=num_page)`。查看文档：http://pyfpdf.readthedocs.io/en/latest/reference/set_link/index.html。

> 调整某些元素可能需要一定程度的反复试验；例如，定位线条。稍长或稍短的线条可能取决于个人品味。不要害怕尝试和检查，直到产生所需的效果。

完整的 FPDF 文档可以在这里找到：http://pyfpdf.readthedocs.io/en/latest/index.html。

### 另请参阅

-   本章前面的 *编写简单的 PDF 文档* 食谱，了解如何处理 PDF 文档的基础知识。
-   本章后面的 *聚合 PDF 报告* 食谱，了解如何将多个文档合并为一个。
-   本章后面的 *水印和加密 PDF* 食谱，了解如何为文档添加安全措施。

### 聚合 PDF 报告

在这个食谱中，我们将看到如何将两个 PDF 合并为一个。我们将把一个报告的页面添加到另一个报告的末尾。

### 准备工作

我们将使用 PyPDF2 模块。Pillow 和 pdf2image 也是脚本使用的依赖项：

```
$ echo "PyPDF2==1.26.0" >> requirements.txt
$ echo "pdf2image==1.11.0" >> requirements.txt
$ echo "Pillow==7.0.0" >> requirements.txt
$ pip install -r requirements.txt
```

为了使 pdf2image 正常工作，它需要安装 pdftoppm，因此请在此处查看有关如何为不同平台安装它的说明：https://github.com/Belval/pdf2image#first-you-need-pdftoppm。

我们需要两个 PDF 来合并它们。对于这个食谱，我们将使用两个 PDF：一个是由 `structuring_pdf.py` 脚本生成的 `report.pdf` 文件，可在 GitHub 上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter05/structuring_pdf.py，另一个（`report2.pdf`）是通过以下命令进行水印处理后的文件：

```
$ python watermarking_pdf.py report.pdf -u automate_user -o report2.pdf
```

这使用了 *水印和加密 PDF* 食谱中描述的水印脚本 `watermarking_pdf.py`，该脚本可在 GitHub 上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter05/watermarking_pdf.py。

### 如何操作...

1.  导入 PyPDF2 并创建输出 PDF：
    ```python
    import PyPDF2
    output_pdf = PyPDF2.PdfFileWriter()
    ```
2.  读取第一个文件并创建一个读取器：
    ```python
    file1 = open('report.pdf', 'rb')
    pdf1 = PyPDF2.PdfFileReader(file1)
    ```
3.  将所有页面附加到输出 PDF：
    ```python
    output_pdf.appendPagesFromReader(pdf1)
    ```
4.  打开第二个文件，创建一个读取器，并将页面附加到输出 PDF：
    ```python
    file2 = open('report2.pdf', 'rb')
    pdf2 = PyPDF2.PdfFileReader(file2)
    output_pdf.appendPagesFromReader(pdf2)
    ```
5.  创建输出文件并保存：
    ```python
    with open('result.pdf', 'wb') as out_file:
        output_pdf.write(out_file)
    ```
6.  关闭两个源文件：
    ```python
    file1.close()
    file2.close()
    ```
7.  检查输出文件并确认它包含两个 PDF 的页面。

### 工作原理...

PyPDF2 允许我们为每个源文件创建一个读取器，并将其所有页面添加到新创建的 PDF 写入器中。请注意，文件是以二进制模式（`rb`）打开的。

生成精彩报告

> 输入文件需要保持打开状态，直到结果保存完毕。这是由于页面复制的工作方式所致。如果文件处于打开状态，生成的文件可能会被存储为空文件。

PDF 写入器最终保存到一个新文件中。请注意，文件需要以二进制写入模式（`wb`）打开才能写入。

### 更多内容...

`.appendPagesFromReader` 对于添加所有页面非常方便，但也可以使用 `.addPage` 逐个添加页面。例如，要添加第三页，代码如下所示：

```
>>> page = pdf1.getPage(3)
>>> output_pdf.addPage(page)
```

PyPDF2 的完整文档可在此处找到：https://pythonhosted.org/PyPDF2/。

### 另请参阅

- 本章前面的 *编写简单 PDF 文档* 配方，了解如何处理 PDF 文档的基础知识。
- 本章前面的 *构建 PDF 结构* 配方，了解如何向 PDF 文档添加分隔符。
- 本章后面的 *水印和加密 PDF* 配方，了解如何为文档添加安全措施。

### 水印和加密 PDF

PDF 文件具有一些有趣的安全措施，以限制文档的分发。我们可以加密内容，要求用户输入密码才能阅读。我们还将了解如何添加水印，以清晰地标记文档为非公开分发，并在泄露时了解其来源。

### 准备工作

我们将使用 `pdf2image` 模块将 PDF 文档转换为 PIL 图像。Pillow 是先决条件。我们还将使用 PyPDF2：

```
$ echo "pdf2image==1.11.0" >> requirements.txt
$ echo "Pillow==7.0.0" >> requirements.txt
$ echo "PyPDF2==1.26.0" >> requirements.txt
$ pip install -r requirements.txt
```

为了使 `pdf2image` 正常工作，它需要安装 `pdftoppm`，因此请在此处查看有关如何为不同平台安装它的说明：https://github.com/Belval/pdf2image#。

我们还需要一个 PDF 文件来进行水印和加密。我们将使用由 `structuring_pdf.py` 脚本生成的 `report.pdf` 文件，该脚本在 *构建 PDF 结构* 配方中描述，可在 GitHub 上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter05/structuring_pdf.py。

### 如何操作...

1. 脚本 `watermarking_pdf.py` 可在 GitHub 上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter05/watermarking_pdf.py。最相关的部分显示如下：

```
def encrypt(out_pdf, password):
    output_pdf = PyPDF2.PdfFileWriter()

    in_file = open(out_pdf, "rb")
    input_pdf = PyPDF2.PdfFileReader(in_file)
    output_pdf.appendPagesFromReader(input_pdf)
    output_pdf.encrypt(password)

    # 中间文件
    with open(INTERMEDIATE_ENCRYPT_FILE, "wb") as out_file:
        output_pdf.write(out_file)

    in_file.close()

    # 重命名中间文件
    os.rename(INTERMEDIATE_ENCRYPT_FILE, out_pdf)

def create_watermark(watermarked_by):
    mask = Image.new('L', WATERMARK_SIZE, 0)
    draw = ImageDraw.Draw(mask)
    font = ImageFont.load_default()
    text = 'WATERMARKED BY {}\n{}'.format(watermarked_by, datetime.now())
    draw.multiline_text((0, 100), text, 55, font=font)

    watermark = Image.new('RGB', WATERMARK_SIZE)
    watermark.putalpha(mask)
    watermark = watermark.resize((1950, 1950))
    watermark = watermark.rotate(45)
    # 裁剪至仅水印部分
    bbox = watermark.getbbox()
    watermark = watermark.crop(bbox)

    return watermark
```

```
def apply_watermark(watermark, in_pdf, out_pdf):
    # 从 PDF 转换为图像
    images = convert_from_path(in_pdf)
    ...
    # 在每一页上粘贴水印
    for image in images:
        image.paste(watermark, position, watermark)

    # 保存生成的 PDF
    images[0].save(out_pdf, save_all=True, append_images=images[1:])
```

2. 使用以下命令为 PDF 文件添加水印：

```
$ python watermarking_pdf.py report.pdf -u automate_user -o out.pdf
Creating a watermark
Watermarking the document
```

3. 检查文档是否在 `out.pdf` 的所有页面上添加了带有 `automate_user` 和时间戳的水印。

4. 使用以下命令进行水印和加密。请注意，加密可能需要一些时间：

```
$ python watermarking_pdf.py report.pdf -u automate_user -o out.pdf -p secretpassword
Creating a watermark
Watermarking the document
Encrypting the document
```

5. 打开生成的 `out.pdf` 文件，检查是否需要输入 `secretpassword` 密码。时间戳也将是新的。

### 工作原理...

`watermarking_pdf.py` 脚本首先使用 `argparse` 从命令行获取参数，然后将其传递给一个主函数，该函数调用其他三个函数：`create_watermark`、`apply_watermark`，以及如果使用了密码，则调用 `encrypt`。

`create_watermark` 生成带有水印的图像。它使用 Pillow 的 `Image` 类创建一个灰度图像（模式 L）并绘制文本。然后，此图像作为 alpha 通道应用于新图像，使图像半透明，从而显示要水印的文本。

> Alpha 通道使白色（颜色 0）完全透明，黑色（颜色 255）完全不透明。在这种情况下，背景是白色（完全透明），文本颜色为 55，使其半透明。

然后将图像旋转 45 度并裁剪，以减少可能出现的透明背景。这使图像居中并允许更好的定位。

在下一步中，`apply_watermark` 使用 `pdf2image` 模块将 PDF 转换为一系列 PIL 图像。它计算应用水印的位置，然后粘贴水印。

> 图像需要通过其左上角定位。这位于文档的一半减去水印的一半处，高度和宽度均如此。请注意，脚本假设文档的所有页面大小相同。

最后，将结果保存为 PDF。注意 `save_all` 参数，它允许我们保存多页 PDF。

如果传递了密码，则调用 `encrypt` 函数。它使用 `PdfFileReader` 打开输出 PDF，并使用 `PdfFileWriter` 创建一个新的中间 PDF。输出 PDF 的所有页面都添加到新 PDF 中，PDF 被加密，然后使用 `os.rename` 将中间 PDF 重命名为输出 PDF。

### 更多内容...

作为水印过程的一部分，请注意页面从文本转换为图像。这增加了额外的保护，因为文本无法直接作为文本提取。在保护文件时，这是一个好主意，因为它将阻止直接复制/粘贴。

> 然而，这不是一个巨大的安全措施，因为文本可能通过 OCR 工具提取。但它可以防止随意提取文本。输出文件大小也大得多，约 30 MB。这也使得加密和解密速度变慢。

PIL 的默认字体可能有点粗糙。如果 TrueType 或 OpenType 文件可用，可以通过调用以下命令添加和使用另一种字体：

```
font = ImageFont.truetype('my_font.ttf', SIZE)
```

请注意，这可能需要安装 FreeType 库，通常作为 `libfreetype` 包的一部分提供。更多文档可在 https://www.freetype.org/ 获取。根据字体和大小，您可能需要调整尺寸。

`pdf2image` 的完整文档可在 https://github.com/Belval/pdf2image 找到，PyPDF2 的完整文档在 https://pythonhosted.org/PyPDF2/，Pillow 的完整文档在 https://pillow.readthedocs.io/en/5.2.x/。

### 另请参阅

- 本章前面的 *编写简单 PDF 文档* 配方，了解如何向 PDF 文档添加分隔符。
- 本章前面的 *构建 PDF 结构* 配方，了解如何向 PDF 文档添加分隔符。
- 本章前面的 *聚合 PDF 报告* 配方，了解如何将多个文档合并为一个。

### 电子表格的乐趣

电子表格是计算领域中最通用且无处不在的工具之一。其直观的表格和单元格方式，被几乎所有使用计算机进行日常操作的人所采用。但它们允许你应用复杂操作，包括使用宏语言。甚至有个流传的笑话说，某个地方的整个复杂业务都由单个电子表格管理和描述。它们是功能极其强大的工具。

这使得自动化读取和写入电子表格的能力变得如此有趣。我们将在本章中了解如何处理电子表格，主要是最常见的格式——Excel。最后一个食谱将涵盖一个免费替代方案——LibreOffice，特别是如何在其中使用Python作为脚本语言。

> Python相比使用电子表格套件中包含的特定工具具有优势。首先，它比像VBA这样的自定义工具更通用，后者仅适用于单个应用程序套件。你还可以利用其庞大的可用库来执行操作，无论是在能力方面（例如，使用统计库或专门的数学库）还是在性能方面。此外，在Python中，代码可读且易于理解，与其他替代方案相比。在*第7章，清理和处理数据*中，我们将介绍一些可以帮助你提高处理电子表格文件和流程生产力的技术。

在本章中，我们将涵盖以下食谱：

- 编写CSV电子表格
- 更新CSV文件
- 读取Excel电子表格
- 更新Excel电子表格
- 在Excel电子表格中创建新工作表
- 在Excel中创建图表
- 处理Excel中的单元格格式
- 在LibreOffice中创建宏

让我们首先看看CSV文件。

### 编写CSV电子表格

CSV文件是高度兼容格式的简单电子表格。它们是包含表格数据的文本文件，以逗号分隔（因此得名逗号分隔值），采用简单的表格格式。CSV文件可以使用Python的标准库创建，并且可以被各种电子表格软件读取。

### 准备工作

对于这个食谱，只需要Python的标准库。一切开箱即用！

### 如何操作...

1. 导入csv模块：
   ```python
   >>> import csv
   ```

2. 定义数据将如何排序以及要存储的数据的表头：
   ```python
   >>> HEADER = ('Admissions', 'Name', 'Year')
   >>> DATA = [
   ... (225.7, 'Gone With the Wind', 1939),
   ... (194.4, 'Star Wars', 1977),
   ... (161.0, 'ET: The Extra-Terrestrial', 1982)
   ... ]
   ```

3. 将数据写入CSV文件：
   ```python
   >>> with open('movies.csv', 'w', newline='') as csvfile:
   ...     movies = csv.writer(csvfile)
   ...     movies.writerow(HEADER)
   ...     for row in DATA:
   ...         movies.writerow(row)
   ```

4. 在电子表格软件中检查生成的CSV文件。在下面的截图中，该文件使用LibreOffice软件显示：

   ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_198_0.png)

### 工作原理...

在*如何操作...*部分的*步骤1*和*2*准备工作之后，*步骤3*是保存`movies.csv`文件的部分。

它以写入（`w`）模式打开一个新文件`movies.csv`。`csvfile`中的文件句柄对象引用该文件，`csv.writer()`函数使用它来创建CSV文件。所有这些都发生在`with`块中，因此当块完成时，它会关闭文件。

> 注意`newline=''`参数。这样做是为了让`writer`控制`newline`格式，避免不兼容问题，例如添加两次换行符。这通常不是问题，但有时CSV格式可能要求换行符是特定字符，与默认字符不同。例如，如果处理要在不同操作系统上使用的文件，就可能发生这种情况。最好在CSV配置中显式处理它。

写入器使用`.writerow()`方法逐行写入元素。第一行是表头，其余行是数据行。

### 更多内容...

提供的代码以默认CSV方言存储数据。CSV方言定义了每行数据的分隔符（逗号或其他字符）、如何转义字符、定义新条目的换行符（也称为行终止符）等。

> 转义是存储可能被解释为语法一部分的字符的过程；例如，存储包含逗号或引号的文本列。在这种情况下，方言将决定如何存储它，通常是通过添加特殊字符，例如，`\,`表示字面逗号。

如果需要调整方言，可以在写入器调用中定义每个参数。有关所有可定义参数的列表，请参考以下链接：https://docs.python.org/3/library/csv.html#dialects-and-formatting-parameters。

> CSV文件越简单越好。如果要存储的数据很复杂，也许最好的替代方案不是CSV文件。但CSV文件在处理表格数据时非常有用。它们几乎可以被所有程序理解，即使在底层处理它们也很容易。

完整的csv模块文档可以在这里找到：https://docs.python.org/3/library/csv.html。

### 另请参阅

- 第4章，搜索和读取本地文件中的读取CSV文件食谱。
- 下一节中的更新CSV文件食谱。

### 更新CSV文件

鉴于CSV文件是简单的文本文件，更新其内容的最佳解决方案是读取它们，将它们处理成内部Python对象，进行更改，然后以相同格式将结果覆盖回去。在这个食谱中，我们将看到如何做到这一点。

### 准备工作

在这个食谱中，我们将使用GitHub上提供的`movies.csv`文件，地址为https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter06/movies.csv。它包含以下数据：

| Admissions | Name | Year |
|---|---|---|
| 225.7 | Gone With the Wind | 1939 |
| 194.4 | Star Wars | 1968 |
| 161.0 | ET: The Extra-Terrestrial | 1982 |

图6.2：电影数据

注意`Star Wars`的`Year`不正确（该电影于1977年上映）。我们将在本食谱中更改它。

### 如何操作...

1. 导入`csv`模块并定义文件名：
   ```python
   >>> import csv
   >>> FILENAME = 'movies.csv'
   ```

2. 使用`DictReader`读取文件内容，并将此内容转换为有序行列表：
   ```python
   >>> with open(FILENAME, newline='') as file:
   ...     data = [row for row in csv.DictReader(file)]
   ```

3. 检查获得的数据。将适当的值从1968更改为1977：
   ```python
   >>> data
   [{'Admissions': '225.7', 'Name': 'Gone With the Wind', 'Year': '1939'}, {'Admissions': '194.4', 'Name': 'Star Wars', 'Year': '1968'}, {'Admissions': '161.0', 'Name': 'ET: The Extra-Terrestrial', 'Year': '1982'}]
   >>> data[1]['Year']
   '1968'
   >>> data[1]['Year'] = '1977'
   ```

4. 再次打开文件并存储值：
   ```python
   >>> HEADER = data[0].keys()
   >>> with open(FILENAME, 'w', newline='') as file:
   ...     writer = csv.DictWriter(file, fieldnames=HEADER)
   ...     writer.writeheader()
   ...     writer.writerows(data)
   ```

5. 在电子表格软件中检查结果。结果类似于*编写CSV电子表格*食谱的*步骤4*中显示的结果。

### 工作原理...

在*如何操作...*部分的*步骤2*中导入`csv`模块后，我们从文件中提取所有数据。文件在`with`块中打开。`DictReader`方便地将其转换为字典列表，表头作为键，单元格内容作为值。

然后可以操作和修改格式方便的数据。我们在*步骤3*中更改了数据以修复`Year`问题。

> 在这个食谱中，我们通过直接访问行号来更改值，但在更一般的情况下，可能需要搜索要更改的特定行或多行。

*步骤4*覆盖文件，并使用`DictWriter`存储数据。`DictWriter`要求我们通过要求`fieldnames`来定义列上的字段。为了获取它们，我们检索其中一行的键并将其存储在`HEADER`中。

文件再次以`w`模式打开以覆盖它。`DictWriter`首先使用`.writeheader`存储表头，然后使用单个调用`.writerows()`存储所有行。

> 行也可以通过调用`.writerow()`方法逐个添加。

关闭`with`块后，文件已存储，可以检查以验证其正确性。

### 更多内容...

对于熟悉的数据源，CSV文件的方言通常是已知的，但情况并非总是如此，特别是当文件来自未知来源时。在这种情况下，`Sniffer`类可以提供帮助。它分析文件的样本（或整个文件）并返回一个猜测的`dialect`对象：

```python
>>> with open(FILENAME, newline='') as file:
...     dialect = csv.Sniffer().sniff(file.read())
```

然后可以在打开文件时将该方言传递给 `DictReader` 类。文件需要打开两次进行读取。

> 记得在 `DictWriter` 类上也使用该方言，以相同的格式保存文件。

`csv` 模块的完整文档可在此处找到：[https://docs.python.org/3.7/library/csv.html](https://docs.python.org/3.7/library/csv.html)。

### 另请参阅

- *第4章，搜索和读取本地文件* 中的 *读取 CSV 文件* 配方。
- 本章前面的 *写入 CSV 电子表格* 配方。

### 读取 Excel 电子表格

MS Office 可以说是办公套件软件中最常见的，使其格式几乎成为标准。MS Excel 可能是最常见的电子表格应用程序，Excel 格式也是最常见的电子表格格式，被许多其他电子表格应用程序所效仿。

在本配方中，我们将了解如何使用 Python 中的 `openpyxl` 模块从 Excel 电子表格中获取信息。

### 准备工作

我们将使用 `openpyxl` 模块。我们应该安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "openpyxl==3.0.3" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 仓库中，有一个名为 `movies.xlsx` 的 Excel 电子表格，其中包含按观影人数排名的前 10 部电影的信息。该文件可在此处找到：[https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter06/movies.xlsx](https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter06/movies.xlsx)。

信息来源可在 [http://www.mrob.com/pub/film-video/topadj.html](http://www.mrob.com/pub/film-video/topadj.html) 找到。

### 如何操作...

1. 导入 openpyxl 模块：
   ```python
   >>> import openpyxl
   ```

2. 将文件加载到内存中：
   ```python
   >>> xlsfile = openpyxl.load_workbook('movies.xlsx')
   ```

3. 列出所有工作表并获取第一个，这是唯一包含数据的工作表：
   ```python
   >>> xlsfile.sheetnames
   ['Sheet1']
   >>> sheet = xlsfile['Sheet1']
   ```

4. 获取单元格 B4 和 D4 的值（E.T. 的观影人数和导演）：
   ```python
   >>> sheet['B4'].value
   161
   >>> sheet['D4'].value
   'Steven Spielberg'
   ```

5. 获取行数和列数。超出此范围的任何单元格将返回 None 作为值：
   ```python
   >>> sheet.max_row
   11
   >>> sheet.max_column
   4
   >>> sheet['A12'].value
   >>> sheet['E1'].value
   ```

### 工作原理...

在步骤 1 中导入模块后，*如何操作...* 部分中的步骤 2 将文件加载到内存中的 Workbook 对象中。每个工作簿可以包含一个或多个工作表，每个工作表包含单元格。

为了确定可用的工作表，在步骤 3 中，我们获取所有工作表名称（本例中只有一个），然后像字典一样访问工作表以检索 Worksheet 对象。

Worksheet 然后可以直接通过其名称访问所有单元格，例如 A4 或 C3。每个单元格将返回一个 Cell 对象。`.value` 属性存储单元格中的值。

> 在本章的其余配方中，我们将看到 Cell 对象的更多属性。继续阅读！

使用 `max_columns` 和 `max_rows` 可以获取数据存储的区域。这允许我们在数据范围内进行搜索。

> Excel 将列定义为字母（A、B、C 等），行定义为数字（1、2、3 等）。请记住始终先设置列，然后设置行（D1，而不是 1D），否则将引发错误。

区域外的单元格可以访问，但不会返回数据。它们可用于写入新数据。

### 更多内容...

单元格也可以通过 `sheet.cell(column, row)` 检索。两个元素的索引都从 1 开始。

数据区域内的所有单元格都可以从工作表中迭代，例如：

```python
>>> for row in sheet:
...     for cell in row:
...         # 对单元格进行操作
```

这将返回一个包含所有单元格的列表的列表，逐行排列：A1、A2、A3 ... B1、B2、B3，依此类推。

> 你可以通过迭代 `sheet.columns` 来检索单元格的列：A1、B1、C1，依此类推，A2、B2、C2，依此类推。

检索单元格时，你可以通过 `.coordinate`、`.row` 和 `.column` 找到其位置：

```python
>>> cell.coordinate
'D4'
>>> cell.column
'D'
>>> cell.row
4
```

完整的 openpyxl 文档可在此处找到：https://openpyxl.readthedocs.io/en/stable/index.html。

### 另请参阅

- 下一节中的 *更新 Excel 电子表格* 配方。
- 本章后面的 *在 Excel 电子表格中创建新工作表* 配方。
- 本章后面的 *在 Excel 中创建图表* 配方。
- 本章后面的 *在 Excel 中处理单元格格式* 配方。

### 更新 Excel 电子表格

在本配方中，我们将了解如何更新现有的 Excel 电子表格。这包括更改单元格中的原始值以及设置在打开电子表格时将被计算的公式。我们还将了解如何向单元格添加注释。

### 准备工作

我们将使用 openpyxl 模块。我们应该安装该模块，将其添加到我们的 requirements.txt 文件中，如下所示：

```
$ echo "openpyxl==3.0.3" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 仓库中，有一个名为 movies.xlsx 的 Excel 电子表格，其中包含按观影人数排名的前 10 部电影的信息。

该文件可在此处找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter06/movies.xlsx。

### 如何操作...

1. 导入 openpyxl 模块和 Comment 类：
   ```python
   >>> import openpyxl
   >>> from openpyxl.comments import Comment
   ```

2. 将文件加载到内存中并获取工作表：
   ```python
   >>> xlsfile = openpyxl.load_workbook('movies.xlsx')
   >>> sheet = xlsfile['Sheet1']
   ```

3. 获取单元格 D4 的值（E.T. 的导演）：
   ```python
   >>> sheet['D4'].value
   'Steven Spielberg'
   ```

4. 将值更改为仅 Spielberg：
   ```python
   >>> sheet['D4'].value = 'Spielberg'
   ```

5. 向该单元格添加注释：
   ```python
   >>> sheet['D4'].comment = Comment('Changed text automatically',
       'User')
   ```

6. 添加一个新元素，用于获取 Admission 列中所有值的总和：
   ```python
   >>> sheet['B12'] = '=SUM(B2:B11)'
   ```

7. 将电子表格保存到 movies_comment.xlsx 文件：
   ```python
   >>> xlsfile.save('movies_comment.xlsx')
   ```

8. 检查生成的文件，其中包含注释和 B 列总和的计算结果（位于 A12）：

   ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_206_0.png)

   图 6.3：单元格现在显示一个注释

### 工作原理...

在 *如何操作...* 部分中，在 *步骤 1* 中进行导入并在 *步骤 2* 中读取电子表格后，我们在 *步骤 3* 中选择要更改的单元格。

在 *步骤 4* 中通过赋值更新值。通过用新的 `Comment` 类实例覆盖 `.comment` 属性，向单元格添加了注释。请注意，需要提供进行注释的用户名。

值也可以包含公式的描述。在 *步骤 6* 中，我们向单元格 *B12* 添加了一个新公式。该值在 *步骤 8* 中打开文件时被计算并显示。

> 公式的值不是在 Python 中计算的，而是在 Excel 中计算的。这意味着公式可能包含错误，或者由于生成的电子表格中的错误而显示意外结果。请务必仔细检查公式是否正确。

最后，在 *步骤 7* 中，通过调用 XLSX 文件对象的 `.save` 方法将电子表格保存到磁盘。生成的文件名可以与输入文件相同以覆盖它。

可以通过外部访问文件来查看注释和值。

### 更多内容...

你可以将数据存储为多种数据类型，它将被转换为 Excel 的适当类型。例如，存储 `datetime` 将以适当的日期格式存储。`float` 或其他数字格式也是如此。

如果你不需要更改电子表格中的任何值，可以以只读模式打开它：

```python
>>> xlsfile = openpyxl.load_workbook('movies.xlsx', read_only=True)
>>> xlsfile['Sheet1']['A1'].value = '37%'
...
AttributeError: Cell is read only
```

向自动生成的单元格添加注释有助于审查生成的文件，清楚地说明每个特定值是如何生成的。

虽然可以添加公式来自动生成 Excel 文件，但调试结果可能很棘手。生成结果时，通常最好在 Python 中进行计算，并将结果以原始格式存储。

### 在 Excel 电子表格中创建新工作表

在本教程中，我们将演示如何从头开始创建一个新的 Excel 电子表格，并处理该电子表格中的多个工作表，包括创建它们。

### 准备工作

我们将使用 `openpyxl` 模块。我们需要安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "openpyxl==3.0.3" >> requirements.txt
$ pip install -r requirements.txt
```

我们将在新文件中存储关于观影人数最多的电影的信息。数据提取自此处：http://www.mrob.com/pub/film-video/topadj.html。

### 如何操作...

1. 导入 `openpyxl` 模块：
   ```python
   >>> import openpyxl
   ```

2. 创建一个新的 Excel 文件。这会创建一个默认的工作表，名为 Sheet：
   ```python
   >>> xlsfile = openpyxl.Workbook()
   >>> xlsfile.sheetnames
   ```
   ```
   >>> sheet = xlsfile['Sheet']
   ```

3. 从数据源向此工作表添加关于观影人数的数据。为简单起见，只添加前三条：
   ```python
   >>> data = [
   ...     (225.7, 'Gone With the Wind', 'Victor Fleming'),
   ...     (194.4, 'Star Wars', 'George Lucas'),
   ...     (161.0, 'ET: The Extraterrestrial', 'Steven Spielberg'),
   ... ]
   >>> for row, (admissions, name, director) in enumerate(data, 1):
   ...     sheet['A{}'.format(row)].value = admissions
   ...     sheet['B{}'.format(row)].value = name
   ```

4. 创建一个新工作表：
   ```python
   >>> sheet = xlsfile.create_sheet("Directors")
   >>> sheet
   <Worksheet "Directors">
   >>> xlsfile.sheetnames
   ['Sheet', 'Directors']
   ```

5. 为每部电影添加导演姓名：
   ```python
   >>> for row, (admissions, name, director) in enumerate(data, 1):
   ...     sheet['A{}'.format(row)].value = director
   ...     sheet['B{}'.format(row)].value = name
   ```

6. 将文件保存为 `movie_sheets.xlsx`：
   ```python
   >>> xlsfile.save('movie_sheets.xlsx')
   ```

7. 打开 `movie_sheets.xlsx` 文件，检查它是否包含两个工作表，并且信息正确，如下图所示：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_210_0.png)

图 6.4：movie_sheets.xlsx 的内容

### 工作原理...

在 *如何操作...* 部分，在 *步骤 1* 中导入模块后，我们在 *步骤 2* 中创建了一个新的电子表格。这是一个仅包含默认工作表的新电子表格。

要存储的数据在 *步骤 3* 中定义。请注意，它包含将放入两个工作表的信息（两个表中都有电影名称；第一个表中有观影人数，第二个表中有导演姓名）。在此步骤中，第一个工作表被填充。

> 注意值的存储方式。正确的单元格被定义为 A 列或 B 列以及正确的行（行从 1 开始）。`enumerate` 函数返回一个元组，第一个元素是索引，第二个是枚举的参数（在本例中，是一个包含三个值的元组）。

之后，在 *步骤 4* 中，使用名称 `Directors` 创建了新工作表。`.create_sheet` 返回新工作表。

`Directors` 工作表中的信息在 *步骤 5* 中存储，文件在 *步骤 6* 中保存。

### 更多内容...

现有工作表的名称可以通过 `.title` 属性进行更改：

```python
>>> sheet = xlsfile['Sheet']
>>> sheet.title = 'Admissions'
>>> xlsfile.sheetnames
['Admissions', 'Directors']
```

请注意，更改后将无法通过 `xlsfile['Sheet']` 访问该工作表。该名称已不存在！

活动工作表（即打开文件时将显示的工作表）可以通过 `.active` 属性获取，并通过 `._active_sheet_index` 进行更改。索引从 0 开始，对应第一个工作表：

```python
>>> xlsfile.active
<Worksheet "Admissions">
>>> xlsfile._active_sheet_index
0
>>> xlsfile._active_sheet_index = 1
>>> xlsfile.active
<Worksheet "Directors">
```

工作表也可以使用 `.copy_worksheet` 进行复制。请注意，某些数据（例如图表）不会被复制。所有单元格数据都将被复制：

```python
new_copied_sheet = xlsfile.copy_worksheet(source_sheet)
```

如果需要复制图表，请记住，如果需要，可以通过代码多次复制它们。

完整的 openpyxl 文档可以在此处找到：https://openpyxl.readthedocs.io/en/stable/index.html。

### 另请参阅

- 本章前面的 *读取 Excel 电子表格* 教程。
- 本章前面的 *更新 Excel 电子表格* 教程。
- 下一节中的 *在 Excel 中创建图表* 教程。
- 本章后面的 *在 Excel 中处理单元格格式* 教程。

### 在 Excel 中创建图表

电子表格包含许多处理数据的工具，包括以彩色图表呈现数据。让我们看看如何以编程方式向 Excel 电子表格添加图表。

### 准备工作

我们将使用 `openpyxl` 模块。我们需要安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "openpyxl==3.0.3" >> requirements.txt
$ pip install -r requirements.txt
```

我们将在新文件中存储关于观影人数最多的电影的信息。数据提取自此处：http://www.mrob.com/pub/film-video/topadj.html。

### 如何操作...

1. 导入 `openpyxl` 模块并创建一个新的 Excel 文件：
   ```python
   >>> import openpyxl
   >>> from openpyxl.chart import BarChart, Reference
   >>> xlsfile = openpyxl.Workbook()
   ```

2. 从数据源向此工作表添加关于观影人数的数据。为简单起见，只添加前三条：
   ```python
   >>> data = [
   ...     ('Name', 'Admissions'),
   ...     ('Gone With the Wind', 225.7),
   ...     ('Star Wars', 194.4),
   ...     ('ET: The Extraterrestrial', 161.0),
   ... ]
   >>> sheet = xlsfile['Sheet']
   >>> for row in data:
   ...     sheet.append(row)
   ```

3. 创建一个 `BarChart` 对象并填充基本信息：
   ```python
   >>> chart = BarChart()
   >>> chart.title = "Admissions per movie"
   >>> chart.y_axis.title = 'Millions'
   ```

4. 创建对数据的引用并将数据添加到图表：
   ```python
   >>> data = Reference(sheet, min_row=2, max_row=4, min_col=1, max_col=2)
   >>> chart.add_data(data, from_rows=True, titles_from_data=True)
   ```

5. 将图表添加到工作表并保存文件：
   ```python
   >>> sheet.add_chart(chart, "A6")
   >>> xlsfile.save('movie_chart.xlsx')
   ```

6. 在电子表格中检查生成的图表，如下图所示：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_213_0.png)

### 工作原理...

在 *如何操作...* 部分，在 *步骤 1* 和 *步骤 2* 中准备好数据后，数据已准备好在 A1:B4 范围内。请注意，A1 和 B1 都包含一个标题，该标题不应在图表中使用。

在 *步骤 3* 中，我们设置新图表并包含基本数据，例如标题和 *y 轴* 的单位。

步骤 4 通过 `Reference` 对象创建一个引用框，从第 2 行第 1 列到第 4 行第 2 列，这是我们的数据所在的区域，不包括标题。数据通过 `.add_data()` 方法添加到图表中。`from_rows` 参数使每一行成为一个不同的数据系列。另一个参数 `titles_from_data` 使用第一列来命名系列。

图表在步骤 5 中被添加到单元格 A6，并保存到磁盘。

### 更多内容...

可以创建多种图表，包括条形图、折线图、面积图（填充线与轴之间区域的折线图）、饼图或散点图（XY 图，其中一个值对另一个值进行绘图）。每种图表都有一个等效的类，例如 `PieChart` 或 `LineChart`。

同时，每种图表都可以有不同的类型。例如，`BarChart` 的默认类型是 `column`，垂直打印条形，但也可以通过选择不同的类型来垂直打印：

```python
>>> chart.type = 'bar'
```

查看 `openpyxl` 文档以了解所有可用的组合。

- 除了从数据中提取 x 轴标签外，还可以使用 `set_categories` 显式设置它们。例如，将步骤 4 与以下代码进行比较：
  ```python
  data = Reference(sheet, min_row=2, max_row=4, min_col=2, max_col=2)
  labels = Reference(sheet, min_row=2, max_row=4, min_col=1, max_col=1)
  chart.add_data(data, from_rows=False, titles_from_data=False)
  chart.set_categories(labels)
  ```

范围也可以使用描述区域的文本标签输入，而不是使用 `Reference` 对象：

```python
chart.add_data('Sheet!B2:B4', from_rows=False, titles_from_data=False)
chart.set_categories('Sheet!A2:A4')
```

如果数据范围需要以编程方式创建，这种描述方式可能更难处理。

电子表格的乐趣

在 Excel 中正确地定义图表有时可能很困难。Excel 从特定范围提取数据的方式可能令人费解。请记住留出时间进行试错，并处理各种差异。例如，在步骤 4 中，我们定义了三个各含一个数据点的系列，而在前面的代码中，我们定义了一个包含三个数据点的系列。其中大多数差异都很细微。最后，最重要的一点是最终图表的呈现效果。尝试不同的图表类型并了解它们的区别。

完整的 openpyxl 文档可以在这里找到：https://openpyxl.readthedocs.io/en/stable/index.html。

### 另请参阅

- 本章前面的“读取 Excel 电子表格”配方。
- 本章前面的“更新 Excel 电子表格”配方。
- 上一节中的“在 Excel 电子表格中创建新工作表”配方。
- 下一节中的“处理 Excel 中的单元格格式”配方。

### 处理 Excel 中的单元格格式

在电子表格中呈现信息不仅仅是将其组织到单元格中或以图表形式进行图形化显示。它还涉及更改格式以突出显示重要细节。在本配方中，我们将了解如何操作单元格格式以增强结果并以最佳方式呈现数据。

### 准备工作

我们将使用 openpyxl 模块。我们应该安装该模块，将其添加到我们的 requirements.txt 文件中，如下所示：

```
$ echo "openpyxl==3.0.3" >> requirements.txt
$ pip install -r requirements.txt
```

我们将把关于观影人数最多的电影的信息存储在一个新文件中。数据提取自这里：http://www.mrob.com/pub/film-video/topadj.html。

### 如何做...

1. 导入 openpyxl 模块并创建一个新的 Excel 文件：

```
>>> import openpyxl
>>> from openpyxl.styles import Font, PatternFill, Border, Side
>>> xlsfile = openpyxl.Workbook()
```

2. 从数据源将关于观影人数的数据添加到此工作表中。为简单起见，只添加前四个：

```
>>> data = [
...     ('Name', 'Admissions'),
...     ('Gone With the Wind', 225.7),
...     ('Star Wars', 194.4),
...     ('ET: The Extraterrestrial', 161.0),
...     ('The Sound of Music', 156.4),
... ]
>>> sheet = xlsfile['Sheet']
>>> for row in data:
...     sheet.append(row)
```

3. 定义用于设置电子表格样式的颜色：

```
>>> BLUE = '0033CC'
>>> LIGHT_BLUE = 'E6ECFF'
>>> WHITE = 'FFFFFF'
```

4. 定义蓝色背景和白色字体的标题：

```
>>> header_font = Font(name='Tahoma', size=14, color=WHITE)
>>> header_fill = PatternFill("solid", fgColor=BLUE)
>>> for row in sheet['A1:B1']:
...     for cell in row:
...         cell.font = header_font
...         cell.fill = header_fill
```

5. 定义列的交替模式，并在标题后的每一行添加边框：

```
>>> white_side = Side(border_style='thin', color=WHITE)
>>> blue_side = Side(border_style='thin', color=BLUE)
>>> alternate_fill = PatternFill("solid", fgColor=LIGHT_BLUE)
>>> border = Border(bottom=blue_side, left=white_side,
...                  right=white_side)
>>> for row_index, row in enumerate(sheet['A2:B5']):
...     for cell in row:
...         cell.border = border
...         if row_index % 2:
...             cell.fill = alternate_fill
```

6. 将文件保存为 movies_format.xlsx：

```
>>> xlsfile.save('movies_format.xlsx')
```

7. 检查生成的文件：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_217_0.png)

### 工作原理...

在*如何做...*部分的*步骤 1*中，我们导入了 openpyxl 模块并创建了一个新的 Excel 文件。在*步骤 2*中，我们将数据添加到第一个工作表。*步骤 3*也是一个准备步骤，用于定义要使用的颜色。颜色以十六进制格式定义，这在网页设计领域很常见。

要查找颜色的定义，网上有很多颜色选择器，甚至操作系统中也内置了。像 https://coolors.co/ 这样的工具对于定义要使用的调色板很有用。

在*步骤 4*中，我们准备格式来定义标题。标题将使用不同的字体（Tahoma）、更大的字号（14pt），并且是蓝色背景上的白色文字。为此，我们准备了一个包含字体、字号和前景色的 `Font` 对象，以及一个包含背景色的 `PatternFill` 对象。

在创建 `header_font` 和 `header_fill` 之后的循环将字体和填充应用到相应的单元格。

请注意，即使只涉及一行，迭代一个范围也总是先返回行，然后返回单元格。

在*步骤 5*中，为行添加了边框并应用了交替背景。边框定义为蓝色顶部和底部，白色左侧和右侧。填充的创建方式与*步骤 4*类似，但使用的是浅蓝色。背景仅应用于偶数行。

请注意，单元格的上边框是其上方单元格的下边框，反之亦然。这意味着在循环中可能会覆盖边框。

文件最终在*步骤 6*中保存。

### 更多内容...

有多种选项可用于设置文本样式，例如粗体、斜体、删除线或下划线。定义字体，如果需要更改其任何元素，请重新分配它。并请记住检查该字体在系统中是否可用。

创建填充也有多种方式。`PatternFill` 接受多种图案，但最有用的是 `solid`。`GradientFill` 也可用于应用双色渐变。

最好将自己限制在使用 `PatternFill` 的纯色填充。你可以调整颜色以最好地代表你想要的效果。请记住包含 `style='solid'`，否则颜色可能不会显示。

也可以定义条件格式，但对于自动生成的电子表格，在 Python 中尝试定义逻辑，然后根据结果应用适当的静态格式会更简单。

数字格式可以正确设置，例如：

```
cell.style = 'Percent'
```

这将把值 0.37 显示为 37%。

完整的 `openpyxl` 文档可以在这里找到：https://openpyxl.readthedocs.io/en/stable/index.html。

### 另请参阅

- 本章前面的*读取 Excel 电子表格*配方。
- 本章前面的*更新 Excel 电子表格*配方。
- 本章前面的*在 Excel 电子表格中创建新工作表*配方。
- 上一节中的*在 Excel 中创建图表*配方。

### 在 LibreOffice 中创建宏

LibreOffice 是一个免费的生产力套件，是 MS Office 和其他办公软件包的替代品。它包括一个名为 `writer` 的文本编辑器和一个名为 `Calc` 的电子表格程序等。`Calc` 理解常规的 Excel 格式，并且它也可以通过其 UNO API 在内部完全进行脚本化。UNO 接口允许以编程方式访问该套件，并且可以通过不同的语言（如 Java）进行访问。

其中一种可用的语言是 Python，这使得在套件格式中生成非常复杂的应用程序变得非常容易，因为它可以使用完整的 Python 标准库。

使用完整的 Python 标准库可以访问诸如加密、打开外部文件（包括 zip 文件）以及连接远程数据库等元素。此外，你可以利用 Python 语法，避免处理 LibreOffice BASIC。

在本配方中，我们将了解如何将一个外部 Python 文件作为宏添加到电子表格中，该宏将更改电子表格的内容。

### 准备工作

需要安装 LibreOffice。它可以从 https://www.libreoffice.org/ 获取。

下载并安装后，需要配置它以允许执行宏：

1. 转到 **设置** | **安全** 以查找 **宏安全** 详细信息：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_220_0.png)

2. 打开 **宏安全** 并选择 **中** 以允许执行我们的宏。这将在允许我们运行宏之前显示一个警告：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_221_0.png)

图 6.8：将宏安全级别设置为中

要将宏插入文件，我们将使用一个名为 `include_macro.py` 的脚本，该脚本可在此处获取：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter06/include_macro.py。

包含宏的脚本也可在此处作为 `libreoffice_script.py` 获取：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter06/libreoffice_script.py。

要放入脚本的文件名为 `movies.ods`，也可在此处获取：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter06/movies.ods。

它包含一个以 .ods 格式（LibreOffice 格式）存储的表格，其中有 10 部观影人数最多的电影。数据提取自这里：http://www.mrob.com/pub/film-video/topadj.html。

### 如何操作...

1.  使用 `include_macro.py` 脚本将 `libreoffice_script.py` 文件附加到 `movies.ods` 宏文件：

```
$ python include_macro.py -h
usage: It inserts the macro file "script" into the file
"spreadsheet" in .ods format. The resulting file is located in the
macro_file directory, that will be created
[-h] spreadsheet script

positional arguments:
spreadsheet File to insert the script
script Script to insert in the file

optional arguments:
-h, --help show this help message and exit
```

```
$ python include_macro.py movies.ods libreoffice_script.py
```

2.  在 LibreOffice 中打开生成的文件 `macro_file/movies.ods`。注意它会显示一个启用宏的警告（点击 **启用**）。转到 **工具 | 宏 | 运行宏**：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_222_0.png)

3.  在 movies.ods | libreoffice_script 下选择 ObtainAggregated，然后点击运行。这将计算总入场人次并将其存储在单元格 B12 中。它在 A15 中添加了一个“总计”标签：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_223_0.png)

图 6.10：在单元格 B12 中计算的总入场人次

4.  重复步骤 2 和 3 再次运行。现在它运行所有聚合，但将 B12 相加并在 B13 中得到结果：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_223_1.png)

图 6.11：再次运行宏会将 B 列中的所有单元格求和，这次包括单元格 B12

### 工作原理...

步骤 1 中的主要工作由 `include_macro.py` 脚本完成。它将文件复制到 `macro_file` 子目录中，以避免修改输入文件。

在内部，.ods 文件是一个具有特定结构的 zip 文件。该脚本利用 Python 的 zip 文件模块将脚本添加到内部的适当子目录中。它还修改了 `manifest.xml` 文件，以便 LibreOffice 知道文件内部有一个脚本。

步骤 3 中执行的宏定义在 `libreoffice_script.py` 中，包含一个函数：

```
def ObtainAggregated(*args):
    # get the doc from the scripting context
    # which is made available to all scripts
    desktop = XSCRIPTCONTEXT.getDesktop()
    model = desktop.getCurrentComponent()
    # get the first sheet
    sheet = model.Sheets.getByIndex(0)

    # Find the admissions column
    MAX_ELEMENT = 20
    for column in range(0, MAX_ELEMENT):
        cell = sheet.getCellByPosition(column, 0)
        if 'Admissions' in cell.String:
            break
    else:
        raise Exception('Admissions not found')

    accumulator = 0.0
    for row in range(1, MAX_ELEMENT):
        cell = sheet.getCellByPosition(column, row)
        value = cell.getValue()
        if value:
            accumulator += cell.getValue()
        else:
            break

    cell = sheet.getCellByPosition(column, row)
    cell.setValue(accumulator)
    cell = sheet.getCellRangeByName("A15")
    cell.String = 'Total'
    return None
```

XSCRIPTCONTEXT 变量是自动创建的，允许获取当前组件，并从中获取第一个工作表。之后，遍历工作表以通过 .getCellByPosition 查找“入场人次”列，并使用 .String 属性获取字符串值。使用相同的方法，通过 .getValue 提取数值来聚合该列中的所有值。

> 由于循环会遍历该列直到找到空单元格，因此第二次执行时，它将聚合 B12 中的值，即上一次执行的聚合值。这样做是为了展示宏可以多次执行并产生不同的结果。

也可以通过 .getCellRangeByName 按字符串位置引用单元格，将“总计”存储在单元格 A15 中。

### 更多内容...

Python 解释器嵌入在 LibreOffice 中，这意味着如果 LibreOffice 发生变化，特定版本也可能改变。在撰写本书时的最新版 LibreOffice（6.4.0）中，包含的版本是 Python 3.7.6。

UNO 接口非常完整，允许你访问许多高级元素。不幸的是，文档并不完善，获取和使用它可能既复杂又耗时。文档以 Java 或 C++ 定义，有 LibreOffice BASIC 或其他语言的示例，但 Python 的示例很少。完整文档可在 https://api.libreoffice.org/ 找到，参考手册在这里：https://api.libreoffice.org/docs/idl/ref/index.html。

> 例如，可以创建复杂的图表，甚至是交互式对话框来询问和处理用户的响应。论坛和旧答案中有大量信息。BASIC 代码在大多数情况下也可以适配到 Python。

LibreOffice 是之前名为 OpenOffice 项目的分支。UNO 在分叉时已经可用，这意味着在互联网上搜索 OpenOffice 时会找到一些相关参考。

请记住，LibreOffice 能够读取和写入 Excel 文件。某些功能可能不是 100% 兼容。例如，可能存在格式问题。

> 出于同样的原因，完全有可能使用本章其他配方中描述的工具生成 Excel 格式的文件，然后用 LibreOffice 打开它。这可能是一个好方法，因为 openpyxl 的文档更好。

调试有时也可能很棘手。请记住在使用新代码重新打开文件之前，确保文件已完全关闭。

UNO 还能够与 LibreOffice 套件的其他部分协同工作，例如在 Writer 中创建其他类型的文档，如文本文档（类似于 MS Word）。

### 另请参阅

-   本章前面的 *编写 CSV 电子表格* 配方。
-   本章前面的 *更新 Excel 电子表格* 配方。

# 7 清理和处理数据

一些自动化任务需要处理大量数据。随着数据量的增长，会出现两个新的、不同的问题：处理任务耗时过长，以及输入数据质量问题导致更多问题。

这两个问题在处理大量数据的数据科学领域广为人知，但即使在较小规模下也可能出现。

输入数据的质量与数据源的数量密切相关。通常，来自单一来源的数据会更一致，但使用单一来源是有限制的。即使数据来自同一来源，它仍可能包含不一致或错误。

> 差异的一些例子可能是区域性的，例如日期格式或货币、额外信息、同一概念的不同名称（包括拼写差异）、拼写错误、数据质量普遍较差且有错误……这个列表很长！

为了进行公平比较，输入数据可能需要清理。这可能是一项艰巨的任务，需要多次迭代才能完善流程，特别是如果数据随时间变化的话。我们将探讨一些处理此任务的技术。

关于时间，对于某些自动化情况，这不是问题。如果一个自动化的后台邮件程序在夜间编写并发送每日更新邮件，以便人们上班时阅读，那么它花费两小时还是两分钟没有区别。但如果有人在等待及时的结果，这可能会非常低效。

特别是在实际代码开发和测试期间，等待时间至关重要，因为它不仅会占用本可用于开发功能的时间，还会破坏你的专注力和注意力，而这些是开发过程的关键。

有几种技巧和技术可以加速代码执行。主要的两个是：避免多次执行相同的操作；以及通过将任务分成更小的块来并行化任务。

> 关于如何减少计算机任务时间的最佳通用建议是学习计算机科学，特别是算法和数据结构。这超出了本书的范围，而且可能是一个漫长的过程。但不要害怕向程序员同行寻求帮助！甚至可以在线求助！

第 3 章“构建你的第一个网络爬虫应用程序”介绍了代码并行化作为“加速网络爬虫”配方的一部分。我们将在本章中继续探讨这个概念。

在本章中，我们将涵盖以下配方：

-   准备 CSV 电子表格
-   根据位置追加货币
-   标准化日期格式
-   聚合结果
-   并行处理数据
-   使用 Pandas 处理数据

让我们从在表格文件中准备基础信息开始。

### 准备 CSV 电子表格

正如我们在上一章中看到的，CSV 文件是包含表格数据的文件，这些数据被定义为具有定义列的行集合，以逗号分隔。它们是所有类型数据的一种非常常见的格式。我们将在本配方中了解如何从日志文件中提取数据并将信息存储在 CSV 文件中。

### 准备工作

我们将使用与*第1章，开启自动化之旅*中*从结构化字符串中提取数据*食谱类似的日志格式：

```
[<Timestamp>] - SALE - PRODUCT: <product id> - PRICE: <price>
```

每一行将代表一条销售日志。

我们将使用 `parse` 模块。我们需要安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "parse==1.14.0" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 仓库中，有一些待处理的日志文件，其结构如下：

```
sale_logs/
  OH
    logs.txt
  ON
    logs.txt
```

代码可在 GitHub 仓库中找到，地址为 `https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter07`。

### 如何操作...

1.  检查 `sale_logs/OH/logs.txt` 文件的内容：

```
$ head sale_logs/OH/logs.txt
[08-27-2019 18:39:41] - SALE - PRODUCT: 12346 - PRICE: 02.99
[08-27-2019 19:39:41] - SALE - PRODUCT: 12346 - PRICE: 02.99
[08-27-2019 20:39:41] - SALE - PRODUCT: 12346 - PRICE: 02.99
[08-27-2019 21:39:41] - SALE - PRODUCT: 12346 - PRICE: 02.99
[08-27-2019 22:39:41] - SALE - PRODUCT: 12345 - PRICE: 09.99
[08-27-2019 23:39:41] - SALE - PRODUCT: 12345 - PRICE: 07.99
[08-28-2019 00:39:41] - SALE - PRODUCT: 12346 - PRICE: 02.99
[08-28-2019 01:39:41] - SALE - PRODUCT: 12346 - PRICE: 02.99
[08-28-2019 02:39:41] - SALE - PRODUCT: 12346 - PRICE: 02.99
[08-28-2019 03:39:41] - SALE - PRODUCT: 12346 - PRICE: 02.99
```

2.  使用 `logs_to_csv.py` 脚本导入数据并将其转换为 CSV 文件。该脚本将位置作为输入添加：

```
$ python logs_to_csv.py sale_logs/OH/logs.txt output_1_OH.csv -l OH
```

3.  在电子表格中检查生成的 CSV 文件。在以下截图中，该文件使用 LibreOffice 软件显示：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_231_0.png)

图 7.1：产品数据截图

请注意 CSV 如何包含位置信息，此处标记为 OH。同时请注意时间戳的格式。

### 工作原理...

让我们看一下在*步骤 2*中使用的 `logs_to_csv.py` 脚本。

它从 `price_log.py` 文件导入，该文件包含日志的解析逻辑。这与*第1章，开启自动化之旅*中*从结构化字符串中提取数据*食谱中的代码类似。该代码添加了一个包含所有行的表头，并添加了一个位置列。让我们来看一下：

```
class PriceLog(object):

    def __init__(self, location, timestamp, product_id, price):
        self.timestamp = timestamp
        self.product_id = product_id
        self.price = price
        self.location = location

    @classmethod
    def parse(cls, location, text_log):
        '''
        Parse from a text log with the format
        [<Timestamp>] - SALE - PRODUCT: <product id> - PRICE: <price>
        to a PriceLog object

        It requires a location
        '''
        def price(string):
            return Decimal(string)

        FORMAT = ('[{timestamp}] - SALE - PRODUCT: {product:d} - '
                  'PRICE: {price:price}')

        formats = {'price': price}
        result = parse.parse(FORMAT, text_log, formats)

        return cls(location=location, timestamp=result['timestamp'],
                   product_id=result['product'],
                   price=result['price'])

    @classmethod
    def header(cls):
        return ['LOCATION', 'TIMESTAMP', 'PRODUCT', 'PRICE']

    def row(self):
        return [self.location, self.timestamp, self.product_id,
                self.price]
```

该文件通过 `logs_to_csv.py` 文件中的 `log_to_csv` 函数转换为 CSV 格式：

```
def log_to_csv(input_file, output_file, location):
    logs = [PriceLog.parse(location, line) for line in input_file]

    # Save into csv format
    writer = csv.writer(output_file)
    writer.writerow(PriceLog.header())
    writer.writerows(l.row() for l in logs)
```

`logs_to_csv.py` 文件的其余部分处理参数解析，这在*第2章，轻松实现任务自动化*中有详细描述。

生成的文件包含相同的信息，但 CSV 是一种更易于理解、处理和添加额外信息的格式，正如我们将在后续食谱中看到的。

### 更多内容...

本食谱的主要目标是将数据从文本文件导入 CSV，而不对原始数据施加任何格式限制。例如，日期采用特定格式。时间戳格式会因文件而异，正如我们将在后续食谱中看到的。

位置的描述也特意保持开放，因为我们将在后续食谱中使用它来设置不同类型的位置。

如果日志文件非常大，读取整个文件然后保存可能会导致内存效率低下。在这种情况下，你可以一次处理一行，如下所示：

```
def log_to_csv(input_file, output_file, location):
    writer = csv.writer(output_file)
    writer.writerow(PriceLog.header())

    # Read and save line by line
    for line in input_file:
        log = PriceLog.parse(location, line)
        writer.writerow(log.row())
```

这种方法也可以分批进行，一次读取和写入多行。这可以提高吞吐量，尽管可能需要一些实验来找到最佳解决方案。

### 另请参阅

- *第4章，搜索和读取本地文件*中的*读取 CSV 文件*食谱。
- *第1章，开启自动化之旅*中的*从结构化字符串中提取数据*食谱。
- *第2章，轻松实现任务自动化*中的*准备任务*食谱。

### 根据位置追加货币

上一个食谱生成的 CSV 文件不包含货币信息，即使位置可能指示使用不同货币的不同地点。在本食谱中，我们将处理一个 CSV 文件以添加额外信息：价格所使用的货币，以及转换为美元。

### 准备工作

我们将使用上一个食谱生成的 CSV 文件，该文件接收并转换日志为以下格式：

```
[<Timestamp>] - SALE - PRODUCT: <product id> - PRICE: <price>
```

每一行将代表一条销售日志。

我们将使用 `parse` 模块。我们需要安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "parse==1.14.0" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 仓库中，有一些待处理的日志文件，其结构如下：

```
sale_logs/
  OH
    logs.txt
  ON
    logs.txt
```

代码可在 GitHub 仓库中找到，地址为 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter07。

### 如何操作...

1.  使用 `logs_to_csv.py` 脚本导入数据并将其转换为 CSV 文件。该脚本将位置作为输入添加。为 OH 和 ON 日志创建文件：

```
$ python logs_to_csv.py sale_logs/OH/logs.txt output_1_OH.csv -l OH
$ python logs_to_csv.py sale_logs/ON/logs.txt output_1_ON.csv -l ON
```

2.  使用 `location_price.py` 脚本处理两个生成的文件：

```
$ python location_price.py output_1_OH.csv output_2_OH.csv
$ python location_price.py output_1_ON.csv output_2_ON.csv
```

3.  在电子表格中检查生成的 CSV 文件。在以下截图中，这些文件使用 LibreOffice 软件显示：

| LOCATION | TIMESTAMP | PRODUCT | PRICE | COUNTRY | CURRENCY | USD |
|---|---|---|---|---|---|---|
| ON | 2018-08-27 19:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-27 20:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-27 21:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-27 22:05:55+00:00 | 12345 | 10.5 | CANADA | CAD | 8 |
| ON | 2018-08-27 23:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 00:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 01:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 02:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 03:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 04:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 05:05:55+00:00 | 12345 | 10.5 | CANADA | CAD | 8 |
| ON | 2018-08-28 06:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 07:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 08:05:55+00:00 | 12345 | 13.5 | CANADA | CAD | 10 |
| ON | 2018-08-28 09:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 10:05:55+00:00 | 12345 | 10.5 | CANADA | CAD | 8 |
| ON | 2018-08-28 11:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 12:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 13:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 14:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |
| ON | 2018-08-28 15:05:55+00:00 | 12346 | 3.99 | CANADA | CAD | 3 |

图 7.2a：加拿大产品数据截图

### 工作原理...

让我们来看一下 `location_price.py` 脚本，它在*步骤 2*中使用。

脚本文件的末尾处理参数解析，这在*第 2 章，轻松实现任务自动化*中有详细描述。

`main` 函数读取输入的 CSV 文件，并对每一行调用 `add_price_by_location` 函数。然后保存文件：

```python
def main(input_file, output_file):
    reader = csv.DictReader(input_file)
    result = [add_price_by_location(row) for row in reader]

    # Save into csv format
    header = result[0].keys()
    writer = csv.DictWriter(output_file, fieldnames=header)
    writer.writeheader()
    writer.writerows(result)
```

### 清理和处理数据

它使用 CSV 的 `DictReader` 类将每一行转换为一个字典。然后将该字典传递进行处理。处理后的行使用 `DictWriter` 类存储到输出 CSV 文件中。输出文件的字段标题从第一个输出行的字典键中获取。

`DictReader` 和 `DictWriter` 的用法已在*第 6 章，电子表格的乐趣*中描述过。

最有趣的代码在 `add_price_by_location` 中。这段代码将根据位置代码检测国家（美国或加拿大）。让我们来看一下：

```python
US_LOCATIONS = ['AL', 'AK', ..., 'WY', 'DC']
CAD_LOCATIONS = ['AB', 'BC', ..., 'NU', 'YT']
CAD_TO_USD = Decimal(0.76)

def add_price_by_location(row):
    location = row['LOCATION']
    if location in US_LOCATIONS:
        row['COUNTRY'] = 'USA'
        row['CURRENCY'] = 'USD'
        row['USD'] = Decimal(row['PRICE'])
    elif location in CAD_LOCATIONS:
        row['COUNTRY'] = 'CANADA'
        row['CURRENCY'] = 'CAD'
        row['USD'] = Decimal(row['PRICE']) * CAD_TO_USD
    else:
        raise Exception('Location not found')

    return row
```

根据位置，它推导出货币。请注意，加拿大省份和地区以及美国州的两个字母代码在两个列表中描述。每个的完整描述在 `CAD_LOCATIONS` 和 `US_LOCATIONS` 数组中。如果检测到的位置是美国州，则货币设置为美元；对于加拿大的所有省份和地区，则设置为加元。

> 美国州和加拿大省份拥有不同的唯一代码，使其易于区分，这非常方便。在其他系统中，这可能更复杂。

它向字典中添加了新的键，用于国家、货币和价格的美元等值。然后这三个键作为列添加到输出 CSV 中。

> 自 Python 3.7 起，字典保持插入顺序。这意味着当键被呈现时，最后添加到字典的键将最后被检索。在之前的版本中并非如此，我们必须依赖 OrderedDict，这是一种保留键顺序的特殊字典。我们利用了这种行为来生成标题，因为任何新引入的元素都将作为新列存储在末尾，保持旧列的顺序。

生成的文件以通用货币提供每条销售日志的信息，使比较更容易。

### 还有更多...

正如我们之前评论的，处理和区分位置和货币可能不像这里展示的那样直接。不同的国家可能有不同的表示地点的方式，在某些情况下，用于表示不同国家位置的代码可能会重叠。

国家作为一列添加，以便以后使用。

货币汇率被定义为常量，但这只是一个近似值。可能需要从外部来源获取它。例如，www.exchangerate-api.com 提供了一个免费的 API 来集成汇率转换，如下所示：

```python
>>> import requests
>>> result = requests.get('https://api.exchangerate-api.com/v4/latest/CAD')
>>> result.json()['rates']['USD']
0.755081
```

更复杂的设置可能需要访问特定的汇率来源或能够检索历史汇率信息。

> 大多数汇率来源对调用次数有限制，或按访问收费。请务必避免持续轮询数据，并将汇率存储以供本地使用。

### 另请参阅

- 上一节的*准备 CSV 电子表格*配方。
- *第 4 章，搜索和读取本地文件*中的*读取 CSV 文件*配方。
- *第 2 章，轻松实现任务自动化*中的*准备任务*配方。

### 标准化日期格式

日志中的日期时间格式因位置而异。在加拿大日志中，格式是标准的 ISO 8601，格式为 YYYY-MM-DD。来自美国的日志使用 MM-DD-YYYY 格式。在这个配方中，我们将添加一个新列，使用标准格式来统一两个日期。

### 准备工作

我们将使用上一个配方生成的 CSV 文件，该文件接收并转换以下格式的日志：

```
[<Timestamp>] - SALE - PRODUCT: <product id> - PRICE: <price>
```

每一行将代表一条销售日志。

我们将使用 `parse` 模块。我们应该安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "parse==1.14.0" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 仓库中，有一些要处理的日志文件，结构如下：

```
sale_logs/
  OH
    logs.txt
  ON
    logs.txt
```

代码可以在 GitHub 仓库中找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter07。

### 如何操作...

1. 使用 `logs_to_csv.py` 脚本导入数据并将其转换为 CSV 文件。该脚本将位置作为输入添加。为 OH 和 OT 日志创建文件：

```
$ python logs_to_csv.py sale_logs/OH/logs.txt output_1_OH.csv -l OH
$ python logs_to_csv.py sale_logs/ON/logs.txt output_1_ON.csv -l ON
```

2. 使用 `location_price.py` 脚本处理两个生成的文件：

```
$ python location_price.py output_1_OH.csv output_2_OH.csv
$ python location_price.py output_1_ON.csv output_2_ON.csv
```

3. 使用 `standard_date.py` 脚本处理两个文件：

```
$ python standard_date.py output_2_OH.csv output_3_OH.csv
$ python standard_date.py output_2_ON.csv output_3_ON.csv
```

4. 在电子表格中检查生成的 CSV 文件。在以下截图中，文件使用 LibreOffice 软件显示：

| | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| 1 | LOCATION | TIMESTAMP | PRODUCT | PRICE | COUNTRY | CURRENCY | USD | STD_TIMESTAMP |
| 2 | OH | 08-27-2019 18:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-27T17:39:41+00:00 |
| 3 | OH | 08-27-2019 19:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-27T18:39:41+00:00 |
| 4 | OH | 08-27-2019 20:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-27T19:39:41+00:00 |
| 5 | OH | 08-27-2019 21:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-27T20:39:41+00:00 |
| 6 | OH | 08-27-2019 22:39:41 | 12345 | 9.99 | USA | USD | 9.99 | 2019-08-27T21:39:41+00:00 |
| 7 | OH | 08-27-2019 23:39:41 | 12345 | 7.99 | USA | USD | 7.99 | 2019-08-27T22:39:41+00:00 |
| 8 | OH | 08-28-2019 00:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-27T23:39:41+00:00 |
| 9 | OH | 08-28-2019 01:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T00:39:41+00:00 |
| 10 | OH | 08-28-2019 02:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T01:39:41+00:00 |
| 11 | OH | 08-28-2019 03:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T02:39:41+00:00 |
| 12 | OH | 08-28-2019 04:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T03:39:41+00:00 |
| 13 | OH | 08-28-2019 05:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T04:39:41+00:00 |
| 14 | OH | 08-28-2019 06:39:41 | 12345 | 7.99 | USA | USD | 7.99 | 2019-08-28T05:39:41+00:00 |
| 15 | OH | 08-28-2019 07:39:41 | 12345 | 9.99 | USA | USD | 9.99 | 2019-08-28T06:39:41+00:00 |
| 16 | OH | 08-28-2019 08:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T07:39:41+00:00 |
| 17 | OH | 08-28-2019 09:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T08:39:41+00:00 |
| 18 | OH | 08-28-2019 10:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T09:39:41+00:00 |
| 19 | OH | 08-28-2019 11:39:41 | 12345 | 9.99 | USA | USD | 9.99 | 2019-08-28T10:39:41+00:00 |
| 20 | OH | 08-28-2019 12:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T11:39:41+00:00 |
| 21 | OH | 08-28-2019 13:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T12:39:41+00:00 |
| 22 | OH | 08-28-2019 14:39:41 | 12346 | 2.99 | USA | USD | 2.99 | 2019-08-28T13:39:41+00:00 |

### 清理和处理数据

| | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| 1 | 位置 | 时间戳 | 产品 | 价格 | 国家 | 货币 | 美元 | 标准时间戳 |
| 2 | ON | 2018-08-27 19:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-27 19:05:55+00:00 |
| 3 | ON | 2018-08-27 20:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-27 20:05:55+00:00 |
| 4 | ON | 2018-08-27 21:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-27 21:05:55+00:00 |
| 5 | ON | 2018-08-27 22:05:55+00:00 | 12345 | 10.5 | 加拿大 | CAD | 8 | 2018-08-27 22:05:55+00:00 |
| 6 | ON | 2018-08-27 23:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-27 23:05:55+00:00 |
| 7 | ON | 2018-08-28 00:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 00:05:55+00:00 |
| 8 | ON | 2018-08-28 01:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 01:05:55+00:00 |
| 9 | ON | 2018-08-28 02:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 02:05:55+00:00 |
| 10 | ON | 2018-08-28 03:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 03:05:55+00:00 |
| 11 | ON | 2018-08-28 04:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 04:05:55+00:00 |
| 12 | ON | 2018-08-28 05:05:55+00:00 | 12345 | 10.5 | 加拿大 | CAD | 8 | 2018-08-28 05:05:55+00:00 |
| 13 | ON | 2018-08-28 06:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 06:05:55+00:00 |
| 14 | ON | 2018-08-28 07:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 07:05:55+00:00 |
| 15 | ON | 2018-08-28 08:05:55+00:00 | 12345 | 13.5 | 加拿大 | CAD | 10 | 2018-08-28 08:05:55+00:00 |
| 16 | ON | 2018-08-28 09:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 09:05:55+00:00 |
| 17 | ON | 2018-08-28 10:05:55+00:00 | 12345 | 10.5 | 加拿大 | CAD | 8 | 2018-08-28 10:05:55+00:00 |
| 18 | ON | 2018-08-28 11:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 11:05:55+00:00 |
| 19 | ON | 2018-08-28 12:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 12:05:55+00:00 |
| 20 | ON | 2018-08-28 13:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 13:05:55+00:00 |
| 21 | ON | 2018-08-28 14:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 14:05:55+00:00 |
| 22 | ON | 2018-08-28 15:05:55+00:00 | 12346 | 3.99 | 加拿大 | CAD | 3 | 2018-08-28 15:05:55+00:00 |

图 7.3：标准化时间戳一致性的截图

请注意，**STD_TIMESTAMP** 列中的时间格式统一，而 **TIMESTAMP** 列中的时间格式则不统一。

### 工作原理...

让我们来看一下在*步骤 3*中使用的 `standard_date.py` 脚本。

脚本文件的末尾处理参数解析，这在*第 2 章，让任务自动化变得简单*中有详细描述。

主函数读取输入的 CSV 文件，并为每一行调用 `add_std_timestamp` 函数。然后将结果写入输出文件：

```
def main(input_file, output_file):
    reader = csv.DictReader(input_file)
    result = [add_price_by_location(row) for row in reader]

    # Save into csv format
    header = result[0].keys()
    writer = csv.DictWriter(output_file, fieldnames=header)
    writer.writeheader()
    writer.writerows(result)
```

它使用 CSV 的 `DictReader` 类将每一行转换为字典，然后传递进行处理。输出的 CSV 文件使用 `DictWriter` 存储结果行。输出文件的字段标题从第一个输出行的字典键中获取。

`DictReader` 和 `DictWriter` 的用法在*第 6 章，电子表格的乐趣*中已有描述。

每一行都在 `add_std_timestamp` 函数中被修改，具体取决于日志写入的国家。让我们来看一下：

```
def add_std_timestamp(row):
    country = row['COUNTRY']
    if country == 'USA':
        # No change
        row['STD_TIMESTAMP'] = american_format(row['TIMESTAMP'])
    elif country == 'CANADA':
        # No change
        row['STD_TIMESTAMP'] = row['TIMESTAMP']
    else:
        raise Exception('Country not found')

    return row
```

根据来源国家，此函数转换日期并创建新的标准化时间戳。

> 请注意，国家信息已在上一个配方中插入到数据中，它是从位置代码推导出来的。在处理的早期阶段显式存储国家信息，可以简化日期处理脚本，因为数据已经计算好了。这在本例中看起来很直接，但在处理数据时，在不同阶段重复操作是很常见的，因为开发时通常只关注代码的一小部分，而不是整个代码。

该函数向字典添加一个新键，其值为 ISO 8601 格式的标准时间戳。在加拿大生成的日志已经具有此格式，但在美国生成的日志需要进行转换。转换在 `american_format` 函数中完成：

```
def american_format(timestamp):
    ...
    '''
    Transform from MM-DD-YYYY HH:MM:SS to iso 8601
    '''
    FORMAT = '%m-%d-%Y %H:%M:%S'

    parsed_tmp = datetime.strptime(timestamp, FORMAT)
    time_with_tz = parsed_tmp.astimezone(timezone.utc)
    isotimestamp = time_with_tz.isoformat()

    return isotimestamp
```

时间戳使用标准 Python 库的 `datetime.strptime` 进行解析，格式为 "%m-%d-%Y %H:%M:%S"，对应于 MM-DD-YYYY HH:MM:SS。生成的 `datetime` 对象随后被添加到 UTC 时区，转换为 ISO 8601 有效的字符串，然后返回。

> 请记住，从 Python 3.7 开始，字典会保持插入顺序。

生成的文件允许我们以相同的格式比较时间。

### 更多内容...

在某些情况下，时间戳格式的检测可能不依赖于国家等其他参数，需要你尝试多种格式以查看哪种合适。当处理多个来源时，这实际上可能成为一个问题，因为单行可能不包含足够的信息。例如，日期 05-06-2019 在国际格式中可能是 6 月 5 日，在美国格式中可能是 5 月 6 日。可能需要分析整个文件，甚至进行猜测，然后再进行验证。

本配方中所有日志文件的时间都以 UTC 时间存储，但并非总是如此。它可能以不同的时区存储。

在我们的例子中，俄亥俄州和安大略省的时区相同，但时区可能因地点而异。根据日志，这可能需要进行调整。

> ISO 8601 格式可能包含时区。例如，以 +00:00 结尾表示时区是 UTC。不要假设时间格式总是包含时区信息。如果不存在，可能会导致数据中的时间不一致。始终包含时区，或在组合不同来源时使用 UTC 时间，以避免混淆。

如果使用不同的时区，*第 1 章，开始我们的自动化之旅*中介绍的 `delorean` 模块有助于以简单的方式定义时间和匹配等效时间：

```
>>> import delorean
>>> timestamp = delorean.parse('2018-08-28 20:05:55+00:00')
>>> timestamp_EST = timestamp.shift('US/Eastern')
>>> timestamp_EST.datetime.isoformat()
'2018-08-28T16:05:55-04:00'
```

完整的 Delorean 文档可以在 https://delorean.readthedocs.io/ 在线找到。

请注意，旧的时间戳仍然保留在 CSV 文件中，而不是被新的时间戳列覆盖。除非有明确的节省空间需求，否则最好保留它以供参考，并在以后检测可能的问题。它还让我们能够轻松地知道何时将此阶段应用于给定的 CSV 文件。

### 另请参阅

- 上一节中的*根据位置追加货币*配方。
- *第 4 章，搜索和读取本地文件*中的*读取 CSV 文件*配方。
- *第 2 章，让任务自动化变得简单*中的*准备任务*配方。

### 聚合结果

数据清理后，我们就可以处理结果了。对于我们的示例，我们将计算按位置和日期划分的平均销售价格，以及数据范围内按位置和日期划分的总销售额。由于我们的数据是按位置存储的，这将分两步完成。首先，我们将按位置创建文件，然后使用位置结果中的日期按日期创建文件。

### 清理与处理数据

### 准备工作

我们将使用上一个食谱中生成的 CSV 文件，该文件接收并转换以下格式的日志：

```
[<Timestamp>] - SALE - PRODUCT: <product id> - PRICE: <price>
```

每一行代表一条销售日志。

我们将使用 `parse` 模块和 `delorean` 模块。我们需要安装这些模块，将它们添加到 `requirements.txt` 文件中，如下所示：

```
$ echo "parse==1.14.0" >> requirements.txt
$ echo "delorean==1.0.0" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 仓库中，有一些待处理的日志文件，结构如下：

```
sale_logs/
  OH
    logs.txt
  ON
    logs.txt
```

代码可在 GitHub 仓库中找到，地址为 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter07。

### 操作步骤...

1.  使用 `logs_to_csv.py` 脚本导入数据并将其转换为 CSV 文件。该脚本将位置作为输入。为 OH 和 ON 日志创建文件：

   ```
   $ python logs_to_csv.py sale_logs/OH/logs.txt output_1_OH.csv -l OH
   $ python logs_to_csv.py sale_logs/ON/logs.txt output_1_ON.csv -l ON
   ```

2.  使用 `location_price.py` 脚本处理两个生成的文件：

   ```
   $ python location_price.py output_1_OH.csv output_2_OH.csv
   $ python location_price.py output_1_ON.csv output_2_ON.csv
   ```

3.  使用 `standard_date.py` 脚本处理两个文件：

   ```
   $ python standard_date.py output_2_OH.csv output_3_OH.csv
   $ python standard_date.py output_2_ON.csv output_3_ON.csv
   ```

4.  使用 `aggregate_by_location.py` 脚本处理两个生成的文件：

   ```
   $ python aggregate_by_location.py output_3_ON.csv aggregate_ON.csv
   $ python aggregate_by_location.py output_3_OH.csv aggregate_OH.csv
   ```

5.  在电子表格中检查生成的 CSV 文件。在以下截图中，文件使用 LibreOffice 软件显示：

| DATE | TOTAL USD | NUMBER | AVERAGE |
|---|---|---|---|
| 2019-08-27 | 32.93 | 7 | 4.7 |
| 2019-08-28 | 114.76 | 24 | 4.78 |
| 2019-08-29 | 111.76 | 24 | 4.66 |
| 2019-08-30 | 113.76 | 24 | 4.74 |
| 2019-08-31 | 135.76 | 24 | 5.66 |
| 2019-09-01 | 112.76 | 24 | 4.7 |
| 2019-09-02 | 126.76 | 24 | 5.28 |
| 2019-09-03 | 114.76 | 24 | 4.78 |
| 2019-09-04 | 115.76 | 24 | 4.82 |
| 2019-09-05 | 100.76 | 24 | 4.2 |
| 2019-09-06 | 119.76 | 24 | 4.99 |

| DATE | TOTAL USD | NUMBER | AVERAGE |
|---|---|---|---|
| 2018-08-27 | 20 | 5 | 4 |
| 2018-08-28 | 113 | 24 | 4.71 |
| 2018-08-29 | 120 | 24 | 5 |
| 2018-08-30 | 120 | 24 | 5 |
| 2018-08-31 | 115 | 24 | 4.79 |
| 2018-09-01 | 106 | 24 | 4.42 |
| 2018-09-02 | 126 | 24 | 5.25 |
| 2018-09-03 | 159 | 24 | 6.62 |
| 2018-09-04 | 114 | 24 | 4.75 |
| 2018-09-05 | 134 | 24 | 5.58 |
| 2018-09-06 | 94 | 24 | 3.92 |

图 7.4：聚合结果截图

### 工作原理...

让我们看看在 *步骤 6* 中使用的 `aggregate_by_location.py` 脚本。

脚本文件的最后一部分处理参数解析，这在 *第 2 章，轻松实现任务自动化* 中有详细描述。

`main` 函数读取输入 CSV 并调用 `calculate_results` 函数来生成聚合报告。然后将结果写入输出文件。`DictReader` 和 `DictWriter` 的使用已在 *第 6 章，电子表格的乐趣* 中描述。

在 `calculate_results` 函数中，聚合操作发生。分析每一行以检查其日期，并将所有具有相同日期的条目聚合：

```python
def calculate_results(reader):
    result = []
    last_date = None
    total_usd = 0
    number = 0

    for row in reader:
        date = parse_iso(row['STD_TIMESTAMP'])
        if not last_date:
            last_date = date

        if last_date < date:
            # 新的一天！
            result.append(line(date, total_usd, number))
            total_usd = 0
            number = 0
            last_date = date

        number += 1
        total_usd += Decimal(row['USD'])

    # 最终结果
    result.append(line(date, total_usd, number))
    return result
```

代码记录最新的日期，并持续聚合直到日期发生变化。每次日期变化时，该行数据就被追加到 `result` 数组中。

> 请注意，这利用了数据已按时间戳排序的事实。在此示例中，日志中的时间戳按来源排序，但在某些场景下可能需要执行一些排序和/或过滤。使用我们在本章中描述的相同技术，如果需要，可以创建一个额外的排序处理步骤。

为了获取日期，使用 `delorean` 模块解析输入的 ISO 格式：

```python
def parse_iso(timestamp):
    # 解析 ISO 格式
    total = delorean.parse(timestamp, dayfirst=False)
    # 仅保留日期
    return total.date
```

`dayfirst=False` 参数确保时间戳被正确解释。

> `delorean` 的新（尚未发布）版本将包含一个特定的 `isofirst` 参数来显式解析 ISO 8601。在撰写本文时，此版本尚未发布，但文档中已显示。

每一行新数据在 `line` 函数中以字典格式配置，如下所示：

```python
def line(date, total_usd, number):
    data = {
        'DATE': date,
        'TOTAL USD': total_usd,
        'NUMBER': number,
        # 四舍五入到两位小数
        'AVERAGE': round(total_usd / number, 2),
    }
    return data
```

每一行包含日期、销售总额（美元）、销售数量和每件商品的平均价格。

> 存储聚合总额和构成该总额的事件数量，使我们能够进一步聚合值并计算平均值。仅平均值本身无法进一步聚合，但聚合总额和事件数量可以，并且可以轻松地从中计算出平均值。

聚合文件以 CSV 格式存储，在 `calculate_results` 将值返回给 `main` 之后。

### 更多内容...

仔细检查输入日期会发现它们并不完全相同。虽然 `output_3_OH.csv` 文件中的格式是 `YYYY-MM-DDTHH:MM:SS+00:00`，但 `output_3_ON.csv` 文件中的格式是 `YYYY-MM-DD HH:MM:SS+00:00`。注意 T 所在位置是空格。ISO 8601 格式使用 T 字符分隔日期和时间，但使用空格分隔也很常见，如 RFC3339 (https://tools.ietf.org/html/rfc3339) 所述，因此大多数软件工具都能解析它，包括 `delorean`。

> 在后期阶段发现数据未完全标准化实际上是一个常见问题。对输入数据要宽容，对数据输出要尽可能严格。

完整的 `delorean` 文档可在线查阅：https://delorean.readthedocs.io/。

每次销售的平均价格四舍五入到两位小数，这使其精确到分，在此特定情况下是合理的。这是通过内置的 `round` 函数完成的，该函数接受一个额外的参数，指定要四舍五入到的小数位数：

```python
>>> round(3.14159)
3
>>> round(3.14159, 4)
3.1416
```

如果你需要更精确地控制向上或向下舍入到下一个整数，可以使用 `math.ceil` 和 `math.floor` 函数：

```python
>>> import math
>>> math.ceil(3.14159)
4
>>> math.floor(3.14159)
3
```

对于除平均值之外更通用的统计运算符，`statistics` 标准库中的 Python 模块提供了诸如 `median()` 和 `quantiles()` 等函数。请查阅文档：https://docs.python.org/3/library/statistics.html。

虽然计算平均值不需要，但其中一些度量（如众数或中位数）可能需要在内存中处理完整数据集。这限制了可以处理的数据量，特别是对于非常大的数据集。

> 关于什么是 *大数据* 的争论很困难，因为没有一个明确的点标志着普通数据变成了大数据。虽然关于大数据的深入讨论超出了本书的范围，但其主要特征是它无法容纳在单台计算机中，需要在不同计算机上进行分布式处理。这大大增加了操作的复杂性，以至于需要专门的技能来处理。在转向该领域之前，请考虑数据是否可以放入具有尽可能多内存的特定服务器中，以及这是否有帮助。这种“用硬件解决问题”的方法可以走得很远，并且通常比重新架构代码所涉及的工作更便宜。

请注意，计算统计量可能是一个困难的挑战，需要特定的知识来避免诸如异常值修改平均值之类的问题，或者更微妙的问题，例如数据的非代表性抽样。在计算复杂内容时，请仔细检查您的假设，并验证结果是否正确表示了所需的度量。这看起来似乎是一项简单的任务，但可能比预期的更棘手。

### 另请参阅

- 上一节中的 *标准化日期格式* 食谱。
- *第 4 章，搜索和读取本地文件* 中的 *读取 CSV 文件* 食谱。
- *第 2 章，轻松实现任务自动化* 中的 *准备任务* 食谱。

清洗与处理数据

### 并行处理数据

上一个食谱中介绍的处理方法效果良好。但它需要逐个处理每个文件。当我们处理的文件数量较少时，这或许没问题，但面对海量文件时，这种方法效率不高。每次我们只使用单个CPU核心，对于这类数值计算任务来说并非最佳选择。

在本食谱中，我们将学习如何并行处理文件，利用计算机的所有核心来加速处理过程，从而大幅提升吞吐量。

### 准备工作

我们将使用上一个食谱生成的CSV文件，该文件接收并转换以下格式的日志：

`[<Timestamp>] - SALE - PRODUCT: <product id> - PRICE: <price>`

每一行代表一条销售日志。

我们将使用 `parse` 模块和 `delorean` 模块。我们需要安装这些模块，将它们添加到 `requirements.txt` 文件中，如下所示：

```
$ echo "parse==1.14.0" >> requirements.txt
$ echo "delorean==1.0.0" >> requirements.txt
$ pip install -r requirements.txt
```

在GitHub仓库中，有一些待处理的日志文件，其结构如下：

```
sale_logs/
  OH
    logs.txt
  ON
    logs.txt
```

代码可在GitHub仓库中找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter07。

### 操作步骤...

1.  使用 `logs_to_csv.py` 脚本导入数据并将其转换为CSV文件。该脚本将位置作为输入参数。分别为OH和ON日志创建文件：

    ```
    $ python logs_to_csv.py sale_logs/OH/logs.txt output_1_OH.csv -l OH
    $ python logs_to_csv.py sale_logs/ON/logs.txt output_1_ON.csv -l ON
    ```

2.  使用 `location_price.py` 脚本处理两个生成的文件：

    ```
    $ python location_price.py output_1_OH.csv output_2_OH.csv
    $ python location_price.py output_1_ON.csv output_2_ON.csv
    ```

3.  使用 `standard_date.py` 脚本处理两个文件：

    ```
    $ python standard_date.py output_2_OH.csv output_3_OH.csv
    $ python standard_date.py output_2_ON.csv output_3_ON.csv
    ```

4.  通过一次调用 `aggregate_by_location_parallel.py` 处理所有文件，并指定要处理的文件：

    ```
    $ python aggregate_by_location_parallel.py "output_3_*.csv"
    Processing output_3_ON.csv
    Processing output_3_OH.csv
    Done with output_3_ON.csv => aggregate_ON.csv
    Done with output_3_OH.csv => aggregate_OH.csv
    ```

5.  在电子表格中检查生成的CSV文件。以下截图使用LibreOffice软件显示文件：

| | A | B | C | D |
|---|---|---|---|---|
| 1 | DATE | TOTAL USD | NUMBER | AVERAGE |
| 2 | 2019-08-28 | 32.93 | 7 | 4.7 |
| 3 | 2019-08-29 | 114.76 | 24 | 4.78 |
| 4 | 2019-08-30 | 111.76 | 24 | 4.66 |
| 5 | 2019-08-31 | 113.76 | 24 | 4.74 |
| 6 | 2019-09-01 | 135.76 | 24 | 5.66 |
| 7 | 2019-09-02 | 112.76 | 24 | 4.7 |
| 8 | 2019-09-03 | 126.76 | 24 | 5.28 |
| 9 | 2019-09-04 | 114.76 | 24 | 4.78 |
| 10 | 2019-09-05 | 115.76 | 24 | 4.82 |
| 11 | 2019-09-06 | 100.76 | 24 | 4.2 |
| 12 | 2019-09-07 | 119.76 | 24 | 4.99 |

| | A | B | C | D |
|---|---|---|---|---|
| 1 | DATE | TOTAL USD | NUMBER | AVERAGE |
| 2 | 2018-08-28 | 20 | 5 | 4 |
| 3 | 2018-08-29 | 113 | 24 | 4.71 |
| 4 | 2018-08-30 | 120 | 24 | 5 |
| 5 | 2018-08-31 | 120 | 24 | 5 |
| 6 | 2018-09-01 | 115 | 24 | 4.79 |
| 7 | 2018-09-02 | 106 | 24 | 4.42 |
| 8 | 2018-09-03 | 126 | 24 | 5.25 |
| 9 | 2018-09-04 | 159 | 24 | 6.62 |
| 10 | 2018-09-05 | 114 | 24 | 4.75 |
| 11 | 2018-09-06 | 134 | 24 | 5.58 |
| 12 | 2018-09-07 | 94 | 24 | 3.92 |

图7.5：本食谱的结果

### 工作原理...

让我们看看在*步骤6*中使用的 `aggregate_by_location_parallel.py` 脚本。

脚本文件的最后一部分处理参数解析，这在*第2章，轻松实现任务自动化*中有详细描述。

`main` 函数检测要聚合的输入文件，然后并行处理它们：

```
def main(input_glob):
    input_files = [filename for filename in glob.glob(input_glob)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(aggregate_filename, filename)
                   for filename in input_files]
        concurrent.futures.wait(futures)
```

该函数首先使用输入glob模式过滤相关文件，并将它们存储在 `input_files` 变量中。这是通过 `glob.glob` 函数完成的，该函数返回与glob模式匹配的文件名。

Glob是常用的模式，用于通过通配符字符（通常是*）匹配一组文件名。例如，glob `*.txt` 将匹配任何扩展名为txt的文件名。

Python在其标准库中包含了一个glob模块。你可以在 https://docs.python.org/3/library/glob.html 查看完整文档。

Glob模式默认在当前目录中搜索。如果需要调整，请记住这一点。

下一步是使用并行执行器为每个文件调用 `aggregate_filename` 函数。我们稍后将描述这个函数，但先看看执行器：

```
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(aggregate_filename, filename)
               for filename in input_files]
    concurrent.futures.wait(futures)
```

首先，我们使用with语句和对 `ProcessPoolExecutor` 的调用来定义执行器。`ProcessPoolExecutor` 创建多个进程工作器，这些工作器将在后台运行提交给执行器的调用。

`ProcessPoolExecutor` 使用在后台创建的独立进程，而不是线程。你也可以使用 `ThreadPoolExecutor`。

由于Python内部结构的一些限制，使用线程对于CPU密集型工作负载（如数值运算）来说不是最优的。线程适用于I/O操作，包括调用外部API和从磁盘读取。

对于这个依赖于数值计算的特定工作负载，进程数（因此工作器数）应与所用CPU的核心数相同。我们在这个例子中假设为四个，这是台式机中的常见数量。

下一步是为每个文件名创建一个 `future`。`future` 是一个对象，它引用将在后台由执行器执行的调用。当它完成时，`future` 对象将存储结果。

> 你可以将 `future` 对象想象成一张代客洗车的票。你提交任务，拿到一张票，然后可以去做其他事情。当你回来时，交回票，如果还有工作未完成，可能需要等待。

最后一步是调用 `concurrent.futures.wait` 等待所有future完成。请注意，在这个特定示例中，future没有需要检查的结果。

我们在*第3章，构建你的第一个网络爬虫应用*的*加速网络爬虫*食谱中介绍了future和执行器。在那个食谱中，我们使用了基于线程的future。

要执行的任务封装了文件的聚合。让我们看看：

```
from aggregate_by_location import main as main_by_file

def aggregate_filename(filename):
    try:
        print(f'Processing {filename}')
        # Obtain the location
        match = re.match(r'output_3_(.*).csv', filename)
        location = match.group(1)
        output_file = f'aggregate_{location}.csv'

        with open(filename) as in_file, open(output_file, 'w') as out_file:
            main_by_file(in_file, out_file)

        print(f'Done with {filename} => {output_file}', flush=True)
    except Exception as exc:
        print(f'Unexpected exception {exc}')
```

其核心是调用 `main_by_file`。这个方法直接从上一节的*聚合结果*食谱中导入。它接收一个输入文件并生成一个输出文件。这些文件使用 `with` 语句以读写访问方式打开。

文件名是预先确定的。输入文件是函数的参数，但输出文件是从输入文件名中获取的。它使用正则表达式 `output_3_(.*).csv` 从文件名中提取位置信息，并将其放入匹配组中：

```
match = re.match(r'output_3_(.*).csv', filename)
location = match.group(1)
output_file = f'aggregate_{location}.csv'
```

输入文件名和生成的输出文件会被打印出来，以便在执行过程中提供反馈。

关于正则表达式的使用，在*第1章，开启我们的自动化之旅*中的*正则表达式入门*和*深入正则表达式*两个食谱中有更详细的描述。

生成的文件与之前*聚合结果*食谱中生成的文件相同，但在此情况下，我们并行处理文件，最多同时处理四个。这显著加快了处理速度。

### 更多内容...

使用并行任务有多个优点，但也有一些注意事项需要考虑。

执行多个任务可能会引发异常，从而停止整个脚本的执行，这与单任务运行时的情况类似。在我们的示例中，`aggregate_filename` 函数包含一个 `try/except` 块，它将捕获任何可能的问题并记录错误。其余文件将继续处理，但至少错误可以被注意到，而不会被静默忽略。

> 这实际上是 Python 之禅的一部分，可以通过调用 `import this` 来查看。

*错误不应该被静默地忽略。*

*除非明确地被静默。*

脚本还可以包含一个额外的检查，以在确定输出文件名后，查看特定文件是否已经创建。这可以通过 `os.path.isfile` 函数来检查：

```
import os.path
...
output_file = f'aggregate_{location}.csv'
if os.path.isfile(output_file):
    # 文件已存在，不覆盖
    return

with open(filename) as in_file ...
```

在处理大量文件时，能够重复执行脚本而无需从头开始是一个巨大的优势。一个常见的问题是处理海量数据时，在处理完成80%时发生错误，然后不得不从头开始重新运行脚本，导致所有工作都得重做。

如果可能，花点时间将部分结果保存到磁盘或其他存储位置，这样它们就可以被跳过，从而加快处理速度。

> 为了开发目的，保持工作数据集较小。这将使您能够快速迭代并减少干扰。同时，将已处理的数据存储在磁盘上。

其中一个用于产生反馈的 `print` 语句添加了 `flush` 参数：

```
print(f'Done with {filename} => {output_file}', flush=True)
```

`flush` 参数将使 `print` 语句立即显示消息。如果未设置 `flush`，或将其设置为 `False`，消息将不会立即打印到屏幕上，而是进入一个中间缓冲区。缓冲区将在操作系统决定打印时被打印，通常是在换行符之后。

这可能会导致一个小的延迟。在某些情况下，这种延迟可能是明显的，例如在执行最后一个任务时。在这种情况下，打印缓冲区可能持有信息，但程序在信息打印之前就退出了。在单任务程序中，退出前缓冲区会被打印，但在多任务应用程序中这并不能保证。

多任务应用程序的并行性质也可能意味着文件处理的顺序在每次运行之间不一致。为了避免问题，每个任务都应该是相互独立的，以避免引入依赖关系。

要了解更多关于 `futures` 和 `executors` 的信息，请查看 Python 文档 https://docs.python.org/3/library/concurrent.futures.html。

### 另请参阅

- 上一节的*聚合结果*食谱。
- *第4章，搜索和读取本地文件*中的*读取 CSV 文件*食谱。
- *第2章，轻松实现任务自动化*中的*准备任务*食谱。
- *第3章，构建你的第一个网页抓取应用程序*中的*加速网页抓取*食谱。

### 使用 Pandas 处理数据

对于某些操作，简单的计算是不够的。有时，操作在计算方式上可能有一些细微差别，并且由于使用某些类型的数据，可能会出现精度问题。

> Python 允许我们自动使用大数字，但在某些语言中，适应大数字可能是一个挑战。计算机中的数字有局限性，例如有限的精度或准确的范围。这些局限性乍一看可能并不明显。

更重要的是，Python 以其卓越的数值计算性能而闻名。与编译语言（如 C++ 甚至 Java）相比，复杂的数学运算将花费更长的时间。

这就是为什么使用专门的包可以极大地提供帮助。它们处理了数据处理的许多复杂性，并且由于针对该目的进行了优化，因此能产生更好的性能。

在这个食谱中，我们将看到如何使用 Pandas 库处理文件，这是一个易于使用的 Python 数据分析库，被科学界广泛使用。

### 准备工作

我们将使用上一个食谱中生成的 CSV 文件，该文件接收并转换以下格式的日志：

`[<Timestamp>] - SALE - PRODUCT: <product id> - PRICE: <price>`

每一行将代表一条销售日志。

我们将使用 pandas 模块。我们应该安装该模块，将其添加到我们的 requirements.txt 文件中，如下所示：

```
$ echo "pandas==1.0.1" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 仓库中，有一些要处理的日志文件，结构如下：

```
sale_logs/
  OH
    logs.txt
  ON
    logs.txt
```

代码可以在 GitHub 仓库中找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter07。

### 如何操作...

1. 使用 logs_to_csv.py 脚本导入数据并将其转换为 CSV 文件。该脚本将位置作为输入。为 OH 和 ON 日志创建文件：
   ```
   $ python logs_to_csv.py sale_logs/OH/logs.txt output_1_OH.csv -l OH
   $ python logs_to_csv.py sale_logs/ON/logs.txt output_1_ON.csv -l ON
   ```
2. 使用 location_price.py 脚本处理两个生成的文件：
   ```
   $ python location_price.py output_1_OH.csv output_2_OH.csv
   $ python location_price.py output_1_ON.csv output_2_ON.csv
   ```
3. 使用 standard_date.py 脚本处理两个文件：
   ```
   $ python standard_date.py output_2_OH.csv output_3_OH.csv
   $ python standard_date.py output_2_ON.csv output_3_ON.csv
   ```
4. 在一次调用 `aggregate_by_location_pandas.py` 中处理所有文件，指定要处理的文件：
   ```
   $ python aggregate_by_location_by_pandas.py output_3_OH.csv aggregate_pd_OH.csv
   $ python aggregate_by_location_by_pandas.py output_3_ON.csv aggregate_pd_ON.csv
   ```
5. 在电子表格中检查生成的 CSV 文件 `aggregate_pd_OH.csv` 和 `aggregate_pd_ON.csv`。在以下截图中，文件使用 LibreOffice 软件显示：

| DATE | TOTAL USD | NUMBER | AVERAGE |
|---|---|---|---|
| 2018-08-28 | 20 | 5 | 4 |
| 2018-08-29 | 113 | 24 | 4.71 |
| 2018-08-30 | 120 | 24 | 5 |
| 2018-08-31 | 120 | 24 | 5 |
| 2018-09-01 | 115 | 24 | 4.79 |
| 2018-09-02 | 106 | 24 | 4.42 |
| 2018-09-03 | 126 | 24 | 5.25 |
| 2018-09-04 | 159 | 24 | 6.62 |
| 2018-09-05 | 114 | 24 | 4.75 |
| 2018-09-06 | 134 | 24 | 5.58 |
| 2018-09-07 | 94 | 24 | 3.92 |

| DATE | TOTAL USD | NUMBER | AVERAGE |
|---|---|---|---|
| 2018-08-28 | 20 | 5 | 4 |
| 2018-08-29 | 113 | 24 | 4.71 |
| 2018-08-30 | 120 | 24 | 5 |
| 2018-08-31 | 120 | 24 | 5 |
| 2018-09-01 | 115 | 24 | 4.79 |
| 2018-09-02 | 106 | 24 | 4.42 |
| 2018-09-03 | 126 | 24 | 5.25 |
| 2018-09-04 | 159 | 24 | 6.62 |
| 2018-09-05 | 114 | 24 | 4.75 |
| 2018-09-06 | 134 | 24 | 5.58 |
| 2018-09-07 | 94 | 24 | 3.92 |

图 7.6：检查食谱的结果

### 清理与处理数据

### 工作原理...

让我们来看一下 `aggregate_by_location_pandas.py` 脚本，它在*第 6 步*中使用。

> 这个方法等同于本章中的*聚合结果*方法。

脚本文件的最后一部分处理参数解析，这在*第 2 章，轻松实现任务自动化*中有详细描述。

在 Pandas 中，基本数据模型称为 `DataFrame`，本质上是一个包含行和列的表格表示。大多数操作都与修改 `DataFrame` 相关。

主函数读取输入 CSV 并调用 `calculate_results` 函数来生成聚合报告。`DictReader` 的用法已在*第 6 章，电子表格的乐趣*中的*更新 CSV 电子表格*方法中描述过。`Calculate_results` 返回一个 `DataFrame`：

```python
def main(input_file, output_file):
    reader = csv.DictReader(input_file)
    result = calculate_results(reader)

    # Save into csv format
    output_file.write(result.to_csv())
```

写入输出文件的操作使用了 Pandas 中 `DataFrame` 可用的 `.to_csv()` 函数。这会生成一个等同于 CSV 格式的文本结果。数据是通过底层的 `output.write()` 调用写入的。

在 `calculate_results` 函数中，聚合操作发生。聚合分三个阶段进行。首先，数据被导入到 `DataFrame` 中，然后对值进行聚合。最后，为了兼容性，数据被四舍五入到两位小数。让我们看一下代码：

```python
def pandas_format(row):
    row['DATE'] = pd.to_datetime(row['STD_TIMESTAMP'])
    row['USD'] = pd.to_numeric(row['USD'])

    return row
```

```python
def calculate_results(reader):
    # Load the data, formatting
    data = pd.DataFrame(pandas_format(r) for r in reader)

    by_usd = data.groupby(data['DATE'].dt.date)['USD']
    result = by_usd.agg(['sum', 'count', 'mean'])

    # Round to 2 digital places
    result = result.round(2)

    # Rename columns
    result = result.rename(columns={
        'sum': 'TOTAL USD',
        'count': 'NUMBER',
        'mean': 'AVERAGE',
    })

    return result
```

第一行将数据导入到 Pandas DataFrame 中。`pandas_format` 函数添加了 `DATE` 列，将标准时间戳转换为 `datetime` 对象，并将 `USD` 列转换为数字格式。这一切都是为 Pandas 处理这些列奠定基础。

聚合的核心发生在这两行：

```python
by_usd = data.groupby(data['DATE'].dt.date)['USD']
result = by_usd.agg(['sum', 'count', 'mean'])
```

第一行将 DataFrame 转换为按 `DATE` 列（这是一个完整的 `datetime` 对象）分组，但仅按其日期（不包含小时信息）进行分组。这会按整天聚合结果。分组后的值仅针对 `USD` 列呈现。

第二行以三种不同的方式聚合结果：`Sum` 用于获取总计结果，`count` 用于获取事件数量，`mean` 用于获取平均值。它们都引用之前选择的 `USD` 列。

`calculate_results` 函数的其余部分更直接。它首先使用 `.round(2)` 将结果四舍五入到两位小数。然后更改列名以与前面的方法保持一致。`.rename` 函数使用字典来定义输入和输出结果。

生成的 CSV 文件与*聚合结果*方法中呈现的文件等同，除了最小的格式差异。

### 更多内容...

`calculate_results` 中应用更改和覆盖结果时显示的模式在 Pandas 中非常常见：

```python
result = by_usd.agg(['sum', 'count', 'mean'])
result = result.round(2)
result = result.rename(...)
```

这个过程也可以轻松地连接成以下形式：

```python
result = by_usd.agg(['sum', 'count', 'mean']).round(2).rename(...)
```

这种工作方式意味着对于每个操作，都会生成数据的副本。对于非常大的数据集，这可能效率不高。你可以通过使用 `inplace` 参数来执行大多数操作而不创建副本。这将修改它们而不使用额外的内存空间或复制数据。

考虑以下示例：

```python
result.rename(columns={
    'sum': 'TOTAL USD',
    'count': 'NUMBER',
    'mean': 'AVERAGE',
}, inplace=True)
```

前面的代码将替换列而不创建副本。它将返回 `None`。这不允许我们使用链式操作，通常被认为是不好的做法。仅在遇到内存问题时使用此方法。

Pandas 是一个庞大而复杂的包，有许多用途，从统计分析到绘图再到复杂数学应用。它在数据科学社区中被广泛使用。Pandas 使用描述所需结果的方法，而不是执行操作的方法。

> 这种方法被称为*声明式*，与命令式相对。*声明式*语言旨在描述结果，即*是什么*，而命令式语言描述*如何做*。声明式语言最常见的例子是 SQL 语言，用于与数据库交互。Python 主要是命令式语言，但正如我们在 Pandas 中看到的，也可以使用声明式方法。

Pandas 经常与 Jupyter Notebook 一起使用。此应用程序允许用户创建丰富的 Python 会话，混合代码执行、文档和图形，形成可通过 Web 浏览器访问的笔记本形式的丰富环境。它能够自动呈现来自 Pandas 或 Matplotlib 等模块的数据。

本项目的目标是允许探索数据，而不是遵循使用自动化工具时更重复的过程，但它是一个值得了解如何使用的绝佳工具。

可以在 `https://jupyter.org/try` 在线测试笔记本。访问 `https://jupyter.org/` 主页查看如何在本地安装并阅读完整文档。

### 清理与处理数据

这是一个显示一些数据的示例会话：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_265_0.png)

你可以在 https://pandas.pydata.org/docs/user_guide/index.html 找到完整的 Pandas 文档。

### 另请参阅

- 本章前面的*聚合结果*方法。
- *第 4 章，搜索和读取本地文件*中的*读取 CSV 文件*方法。
- *第 2 章，轻松实现任务自动化*中的*准备任务*方法。

## 8 开发令人惊叹的图表

图表和图像是以简单易懂的方式呈现复杂数据的绝佳方式。在本章中，我们将利用强大的 matplotlib 库来学习如何创建各种图表。matplotlib 是一个旨在以多种方式显示数据的库，它可以创建令人惊叹的图表，帮助以最佳方式传输和显示信息。

> matplotlib 众所周知，并且与 Python 生态系统中的其他工具配合良好。例如，matplotlib 图表也可以由 Jupyter Notebooks 自动显示，如第 7 章，清理与处理数据中所介绍。

我们将涵盖的图表将从简单的条形图到折线图或饼图，并在同一图表中组合多个绘图、添加注释，甚至绘制地理地图。

本章将涵盖以下方法：

- 绘制简单的销售图表
- 绘制堆叠条形图
- 绘制饼图
- 显示多条线
- 绘制散点图
- 可视化地图
- 添加图例和注释
- 组合图表
- 保存图表

让我们从创建第一个图表开始。

### 绘制简单的销售图表

在这个方法中，我们将看到如何通过绘制与不同时间段销售额成比例的条形来绘制销售图表。

### 准备工作

我们可以使用以下命令在虚拟环境中安装 `matplotlib`：

```bash
$ echo "matplotlib==3.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

在某些操作系统中，这可能需要我们安装额外的软件包；例如，在 Ubuntu 中，可能需要运行 `apt-get install python3-tk`。有关详细信息，请查看 `matplotlib` 文档。

如果你使用的是 macOS，可能会遇到这样的错误：**RuntimeError: Python is not installed as a framework**。请参阅 `matplotlib` 文档了解如何修复：https://matplotlib.org/faq/osx_framework.html。

### 如何做...

1. 导入 `matplotlib`：

```python
>>> import matplotlib.pyplot as plt
```

2. 准备要显示在图表上的数据：

```python
>>> DATA = (
...     ('Q1 2017', 100),
...     ('Q2 2017', 150),
...     ('Q3 2017', 125),
...     ('Q4 2017', 175),
... )
```

3. 将数据拆分为图表可用的格式。这是一个准备步骤：

```python
>>> POS = list(range(len(DATA)))
```

### 工作原理...

导入模块后，数据将在*如何操作...*部分的*步骤 2*中以可用的结构呈现。

### 开发惊艳图表

由于 `matplotlib` 的工作方式，它需要一个 X 分量和一个 Y 分量。在这种情况下，我们的 X 分量只是一个递增的数字序列，其数量与数据点的数量相同。我们将其存储在变量 `POS` 中。这为每个 Y 值定位，并作为时间序列工作。在 `VALUES` 中，我们以序列形式存储销售额的数值，在 `LABELS` 中，我们存储每个数据点的关联标签。所有这些准备工作都在*步骤 3*中完成。

*步骤 4*使用序列 X (`POS`) 和 Y (`VALUES`) 创建条形图。这些定义了我们的条形。为了指定其引用的时间段，我们使用 `.xticks` 以相同的方式在 *x 轴*上为每个值添加标签。为了阐明含义，我们使用 `.ylabel` 添加标签。

要显示生成的图表，*步骤 5*调用 `.show`，这将打开一个包含结果的新窗口。

> 调用 `.show` 会阻塞程序的执行。关闭窗口后，程序将恢复。

### 更多内容...

您可能希望更改值呈现的格式。在我们的示例中，也许数字代表数百万美元。为此，您可以向 *y 轴*添加一个格式化器，这样那里表示的值将应用该格式化器：

```python
>>> from matplotlib.ticker import FuncFormatter

>>> def value_format(value, position):
...     return '$ {}M'.format(int(value))

>>> axes = plt.gca()
>>> axes.yaxis.set_major_formatter(FuncFormatter(value_format))
```

`value_format` 是一个根据数据的值和位置返回值的函数。在这里，它将把值 100 返回为 $100M。

> 值将以浮点数形式检索，需要您将其转换为整数以进行显示。

要应用格式化器，我们需要使用 `.gca`（获取当前坐标轴）检索 `axis` 对象。然后，`.yaxis` 属性为 *y 轴*标签设置格式化器。

条形的颜色也可以通过 `color` 参数确定。颜色可以以多种格式指定，如 https://matplotlib.org/api/colors_api.html 中所述，但我最喜欢的是遵循 XKCD 颜色调查，使用 `xkcd:` 前缀（冒号后无空格）：

```python
>>> plt.bar(POS, VALUES, color='xkcd:moss green')
```

完整的调查可以在这里找到：https://xkcd.com/color/rgb/。

> 大多数常见颜色，如蓝色和红色，也可用于快速测试。不过，它们往往有点亮和刺眼，不适合用于美观的报告。

将颜色与格式化坐标轴结合，我们得到以下结果：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_270_0.png)

### 开发惊艳图表

条形图不需要以时间方式显示信息。正如我们所看到的，`matplotlib` 要求我们指定每个条形的 X 参数。它是一个强大的工具，用于生成各种图表。

> 例如，条形可以排列以显示直方图，例如显示特定身高的人群。条形将从低身高开始，增加到平均大小，然后回落。不要将自己局限于电子表格图表！

完整的 `matplotlib` 文档可以在这里找到：https://matplotlib.org/。

### 另请参阅

- 本章接下来的*绘制堆叠条形*食谱，了解如何在每个条形上绘制更多累积信息。
- 本章后面的*添加图例和注释*食谱，了解如何向图表添加上下文信息。
- 本章后面的*组合图表*食谱，了解如何将多个绘图组合成一个图表。

### 绘制堆叠条形

显示不同类别的一种强大方式是将它们呈现为堆叠条形，以便显示每个类别和总计。我们将在本食谱中了解如何做到这一点。

### 准备工作

我们需要在虚拟环境中安装 `matplotlib`：

```bash
$ echo "matplotlib==3.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

如果您使用 macOS，可能会遇到如下错误：**RuntimeError: Python is not installed as a framework**。请参阅 `matplotlib` 文档了解如何修复：https://matplotlib.org/faq/osx_framework.html。

### 如何操作...

1. 导入 matplotlib：
    ```python
    >>> import matplotlib.pyplot as plt
    ```

2. 准备数据。这代表两个产品的销售额；一个成熟产品和一个新产品：
    ```python
    >>> DATA = (
    ...     ('Q1 2017', 100, 0),
    ...     ('Q2 2017', 105, 15),
    ...     ('Q3 2017', 125, 40),
    ...     ('Q4 2017', 115, 80),
    ... )
    ```

3. 处理数据以准备预期格式：
    ```python
    >>> POS = list(range(len(DATA)))
    >>> VALUESA = [valueA for label, valueA, valueB in DATA]
    >>> VALUESB = [valueB for label, valueA, valueB in DATA]
    >>> LABELS = [label for label, value1, value2 in DATA]
    ```

4. 创建条形图。需要两个绘图：
    ```python
    >>> plt.bar(POS, VALUESB)
    <BarContainer object of 4 artists>
    >>> plt.bar(POS, VALUESA, bottom=VALUESB)
    <BarContainer object of 4 artists>
    >>> plt.ylabel('Sales')
    Text(0, 0.5, 'Sales')
    >>> plt.xticks(POS, LABELS)
    <REDACTED>
    ```

5. 显示图表：
    ```python
    >>> plt.show()
    ```

6. 结果将显示在新窗口中，如下所示：

    ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_273_0.png)

### 工作原理...

导入模块后，数据在*步骤 2*中以可用的结构呈现。

在*步骤 3*中，数据被准备成三个序列：`VALUESA`、`VALUESB` 和 `LABELS`。添加了一个递增的 `POS` 序列，以便条形在 *x 轴*上一个接一个地定位。

*步骤 4*使用序列 *X* (`POS`) 和 *Y* (`VALUESB`) 创建条形图。第二个条形序列 `VALUESA` 使用 `bottom=VALUESB` 参数添加到前一个条形之上。此过程将两个条形堆叠，将整个 `VALUESA` 条形定位在 `VALUESB` 条形之上。

> 注意，我们首先堆叠第二个值 `VALUESB`。第二个值代表引入市场的新产品。`VALUESA` 代表成熟产品，更稳定。此图表更好地显示了新产品的增长。

每个时间段在 *x 轴*上用 `.xticks` 标记。为了阐明含义，我们使用 `.ylabel` 添加标签。

要显示生成的图表，*步骤 5*调用 `.show`，这将打开一个包含结果的新窗口。

> 调用 `.show` 会阻塞程序的执行。关闭窗口后，程序将恢复。

### 更多内容...

另一种呈现堆叠条形的方法是将它们添加为百分比，这样总计不会改变，只有彼此之间的相对大小。

为此，VALUESA 和 VALUESB 需要按以下方式相对于百分比计算：

```python
>>> VALUESA = [100 * valueA / (valueA + valueB) for label, valueA, valueB in DATA]
>>> VALUESB = [100 * valueB / (valueA + valueB) for label, valueA, valueB in DATA]
```

这使得每个值等于总计的百分比，并且总计始终加起来为 100。这会产生以下图形：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_274_0.png)

条形不一定需要堆叠。有时，将条形一个对另一个进行比较可能很有趣。

### 开发惊艳图表

为此，我们需要移动第二个条形序列的位置。我们还需要设置更细的条形以允许空间：

```python
>>> WIDTH = 0.3
>>> plt.bar([p - WIDTH / 2 for p in POS], VALUESA, width=WIDTH)
>>> plt.bar([p + WIDTH / 2 for p in POS], VALUESB, width=WIDTH)
```

注意条形的宽度设置为空间的三分之一，因为我们的参考空间在条形之间为 1。第一个条形向左移动，而第二个条形向右移动以使它们居中。`bottom` 参数已被删除，因此条形不会堆叠：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_275_0.png)

完整的 `matplotlib` 文档可以在这里找到：https://matplotlib.org/。

### 另请参阅

- 本章前面的*绘制简单销售图表*食谱，了解绘制条形图的基础知识。
- 本章后面的*添加图例和注释*食谱，了解如何向图表添加上下文信息。
- 本章后面的*组合图表*食谱，了解如何将多个绘图添加到单个图表中。

### 绘制饼图

饼图！这是商业入门课程的最爱，也是展示百分比的常用方式。本节将介绍如何绘制饼图，其中不同的扇区代表不同的比例。

### 准备工作

我们需要在虚拟环境中安装 `matplotlib`，使用以下命令：

```
$ echo "matplotlib==3.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

如果你使用的是 macOS，可能会遇到如下错误：**RuntimeError: Python is not installed as a framework**。请参阅 `matplotlib` 文档了解如何修复：https://matplotlib.org/faq/osx_framework.html。

### 操作步骤...

1. 导入 `matplotlib`：
   ```python
   >>> import matplotlib.pyplot as plt
   ```

2. 准备数据。这代表了几条产品线：
   ```python
   >>> DATA = (
   ...     ('Common', 100),
   ...     ('Premium', 75),
   ...     ('Luxurious', 50),
   ...     ('Extravagant', 20),
   ... )
   ```

3. 处理数据以准备所需的格式：
   ```python
   >>> VALUES = [value for label, value in DATA]
   >>> LABELS = [label for label, value in DATA]
   ```

4. 创建饼图：
   ```python
   >>> plt.pie(VALUES, labels=LABELS, autopct='%1.1f%%')
   <REDACTED>
   >>> plt.gca().axis('equal')
   (-1.1113861431510297, 1.1005422098873965, -1.125031021533458, 1.1221350517711501)
   ```

5. 显示图表：
   ```python
   >>> plt.show()
   ```

6. 结果将显示在一个新窗口中，如下所示：

   ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_277_0.png)

### 工作原理...

在操作步骤...的第1步中导入了模块，第2步导入了要展示的数据。在第3步中，数据被分离为两个部分：一个 `VALUES` 列表和一个 `LABELS` 列表。

在第4步中，通过添加 `VALUES` 和 `LABELS` 创建了图表。`autopct` 参数格式化数值，使其显示为保留一位小数的百分比。

调用 `axis('equal')` 确保饼图看起来是圆形的，而不是带有一点透视效果而呈现椭圆形。

为了显示生成的图表，第5步调用了 `.show`，这会打开一个包含结果的新窗口。

> 调用 `.show` 会阻塞程序的执行。关闭窗口后，程序将继续运行。

### 更多内容...

饼图在商业图表中有点被过度使用了。大多数情况下，使用带有百分比或数值的条形图会是更好的数据可视化方式，尤其是在显示超过两三个选项时。请尽量限制在报告和数据展示中使用饼图。

可以通过 `startangle` 参数旋转扇区的起始角度，扇区的排列方向由 `counterclock` 定义（默认为 `True`）：

```python
>>> plt.pie(VALUES, labels=LABELS, startangle=90, counterclock=False)
```

标签内的格式可以通过函数设置。由于饼图内的值定义为百分比，找到原始值可能有点棘手。以下代码片段创建了一个以整数百分比为索引的字典，这样我们就可以检索到对应的值。请注意，这假设没有百分比重复。如果出现这种情况，标签可能会略有不准确。在这种情况下，我们可能需要使用到小数点后第一位以获得更好的精度：

```python
>>> from matplotlib.ticker import FuncFormatter

>>> total = sum(value for label, value in DATA)
>>> BY_VALUE = {int(100 * value / total): value for label, value in DATA}

>>> def value_format(percent, **kwargs):
...     value = BY_VALUE[int(percent)]
...     return '{}'.format(value)
```

也可以使用 `explode` 参数分离一个或多个扇区。这指定了扇区与中心的分离程度：

```python
>>> explode = (0, 0, 0.1, 0)
>>> plt.pie(VALUES, labels=LABELS, explode=explode, autopct=value_format,
...         startangle=90, counterclock=False)
```

通过组合所有这些选项，调用 `plt.show()` 时我们得到以下结果：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_279_0.png)

图 8.7：通过分离扇区来突出显示

完整的 `matplotlib` 文档可以在这里找到：`https://matplotlib.org/`。

### 另请参阅

- 本章前面的 *绘制简单销售图表* 小节，以了解绘制条形图的基础知识。
- 上一节的 *绘制堆叠条形图* 小节，了解如何将累积值绘制为条形图。

### 显示多条线

本节将向你展示如何在图表中显示多条线。

### 准备工作

我们需要在虚拟环境中安装 matplotlib：

```
$ echo "matplotlib==3.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

如果你使用的是 macOS，可能会遇到如下错误：**RuntimeError: Python is not installed as a framework**。请参阅 matplotlib 文档了解如何修复：https://matplotlib.org/faq/osx_framework.html。

### 操作步骤...

1. 导入 matplotlib：
   ```python
   >>> import matplotlib.pyplot as plt
   ```

2. 准备数据。这代表了两个产品的销售情况：
   ```python
   >>> DATA = (
   ...     ('Q1 2017', 100, 5),
   ...     ('Q2 2017', 105, 15),
   ...     ('Q3 2017', 125, 40),
   ...     ('Q4 2017', 115, 80),
   ... )
   ```

3. 处理数据以准备所需的格式：
   ```python
   >>> POS = list(range(len(DATA)))
   >>> VALUESA = [valueA for label, valueA, valueB in DATA]
   >>> VALUESB = [valueB for label, valueA, valueB in DATA]
   >>> LABELS = [label for label, value1, value2 in DATA]
   ```

4. 创建折线图。需要两条线：
   ```python
   >>> plt.plot(POS, VALUESA, 'o-')
   [<matplotlib.lines.Line2D object at 0x12e78a2b0>]
   >>> plt.plot(POS, VALUESB, 'o-')
   [<matplotlib.lines.Line2D object at 0x12e7afcd0>]
   >>> plt.ylabel('Sales')
   Text(0, 0.5, 'Sales')
   >>> plt.xticks(POS, LABELS)
   <REDACTED>
   ```

5. 显示图表：
   ```python
   >>> plt.show()
   ```

6. 结果将显示在一个新窗口中：

   ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_281_0.png)

### 工作原理...

在 *操作步骤...* 部分，*第1步* 导入了模块，*第2步* 以格式化的方式展示了要绘制的数据。

在 *第3步* 中，数据被准备成三个序列：`VALUESA`、`VALUESB` 和 `LABELS`。一个递增的 `POS` 序列用于在 *x轴* 上定位每个点。

*第4步* 使用序列 *X* (`POS`) 和 *Y* (`VALUESA`) 创建图表，然后使用 `POS` 和 `VALUESB`。添加了值 `'o-'` 以在每个数据点上绘制一个圆圈，并在它们之间绘制一条完整的线。

> 默认情况下，图表将显示一条实线，每个点上没有标记。如果只使用标记（即 'o'），则不会有线。

每个时间段都用 .xticks 标记在 x 轴上。为了阐明含义，我们用 .ylabel 添加了一个标签。

为了显示生成的图表，第5步调用了 .show，这会打开一个包含结果的新窗口。

> 调用 .show 会阻塞程序的执行。关闭窗口后，程序将继续运行。

### 更多内容...

带线的图表看似简单，却能创造出许多有趣的表示。在展示数学图表时，它可能是最方便的。例如，我们可以用几行代码展示一个显示摩尔定律的图表。

> 摩尔定律是戈登·摩尔的一项观察，即集成电路上的组件数量每2年翻一番。它最初于1965年提出，后来在1975年进行了修正。它似乎与过去40年的技术进步历史速率非常接近。

我们首先创建一条描述理论线的线，数据点从1970年到2013年。从1,000个晶体管开始，每2年翻一番，直到2013年：

```python
>>> POS = [year for year in range(1970, 2013)]
>>> MOORES = [1000 * (2 ** (i * 0.5)) for i in range(len(POS))]
>>> plt.plot(POS, MOORES)
[<matplotlib.lines.Line2D object at 0x12b7c27c0>]
```

根据一些文档，我们从这里提取了一些商用CPU的示例、它们的发布年份以及它们的集成组件数量：http://mercury.pr.erau.edu/~siewerts/cec320/documents/Papers/AHistoryofMicroprocessorTransistorCount.pdf。

### 绘制精美图表

由于数值较大，我们将使用 `1_000_000` 表示一百万，这是 Python 3 支持的写法：

```python
>>> DATA = (
...     ('Intel 4004', 2_300, 1971),
...     ('Motorola 68000', 68_000, 1979),
...     ('Pentium', 3_100_000, 1993),
...     ('Core i7', 731_000_000, 2008),
... )
```

绘制一条带标记的线，将这些点显示在正确的位置。'v' 标记将显示为一个三角形：

```python
>>> data_x = [x for label, y, x in DATA]
>>> data_y = [y for label, y, x in DATA]
>>> plt.plot(data_x, data_y, 'v')
[<matplotlib.lines.Line2D object at 0x12b7c2d60>]
```

为每个数据点在正确的位置添加 CPU 名称的标签：

```python
>>> for label, y, x in DATA:
...     plt.text(x, y, label)
Text(1971, 2300, 'Intel 4004')
Text(1979, 68000, 'Motorola 68000')
Text(1993, 3100000, 'Pentium')
Text(2008, 731000000, 'Core i7')
```

最后，增长在图表中以线性尺度显示没有意义，因此我们将刻度改为对数尺度，这使得指数增长看起来像一条直线。但为了保持量级感，我们添加了网格。调用 `.show` 来显示图表：

```python
>>> plt.gca().grid()
>>> plt.yscale('log')
```

调用 `plt.show()` 时将显示生成的图表：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_284_0.png)

注意直线如何在对数尺度上遵循晶体管数量的倍增，以及处理器如何紧密排列。你可以看到，实际的处理器与摩尔的预测非常相似！

完整的 `matplotlib` 文档可以在这里找到：https://matplotlib.org/。特别地，可以在这里查看可用的线条格式（实线、虚线、点线等）和标记（点、圆、三角形、星形等）：https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html。

### 另请参阅

- 本章后面的 *添加图例和注释* 配方，了解如何向图表添加上下文信息。
- 本章后面的 *组合图表* 配方，了解如何向单个图表添加多个绘图。

### 绘制散点图

散点图是一种将信息显示为具有 X 和 Y 值的点的图表。当呈现二维数据（与之前看到的时间序列相反）以及查看两个变量之间是否存在任何关系时，它们非常有用。在这个配方中，我们将绘制一个图表，将网站上花费的时间与花费的金额进行对比，以查看是否能看到某种模式。

### 准备工作

我们需要在虚拟环境中安装 matplotlib：

```bash
$ echo "matplotlib==3.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

如果你使用的是 macOS，可能会遇到这样的错误：**RuntimeError: Python is not installed as a framework**。请参阅 matplotlib 文档了解如何修复：https://matplotlib.org/faq/osx_framework.html。

由于我们将处理数据点，我们将使用 `scatter.csv` 文件来读取数据。此文件可在 GitHub 上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter08/scatter.csv。

### 如何做...

1. 导入 matplotlib、csv 和 FuncFormatter（用于稍后格式化坐标轴）：

```python
>>> import csv
>>> import matplotlib.pyplot as plt
>>> from matplotlib.ticker import FuncFormatter
```

2. 使用 csv 模块从文件中读取数据，准备数据：

```python
>>> with open('scatter.csv') as fp:
...     reader = csv.reader(fp)
...     data = list(reader)
```

3. 准备绘图数据，然后绘制：

```python
>>> data_x = [float(x) for x, y in data]
>>> data_y = [float(y) for x, y in data]
>>> plt.scatter(data_x, data_y)
<matplotlib.collections.PathCollection object at 0x11ccbda30>
```

4. 通过格式化坐标轴来改进上下文：

```python
>>> def format_minutes(value, pos):
...     return '{}m'.format(int(value))
>>> def format_dollars(value, pos):
...     return '${}'.format(value)
>>> plt.gca().xaxis.set_major_formatter(FuncFormatter(format_minutes))
>>> plt.xlabel('Time in website')
Text(0.5, 0, 'Time in website')
>>> plt.gca().yaxis.set_major_formatter(FuncFormatter(format_dollars))
>>> plt.ylabel('Spending')
Text(0, 0.5, 'Spending')
```

5. 显示图表：

```python
>>> plt.show()
```

6. 结果将显示在新窗口中：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_286_0.png)

### 工作原理...

*如何做...* 部分的步骤 1 导入了我们稍后将使用的模块。

步骤 2 从 CSV 文件中读取数据。数据被转换为列表，以便我们可以多次迭代它，这对于 *步骤 3* 是必要的。

步骤 3 将数据准备成两个数组，然后使用 `.scatter` 进行绘制。`.scatter` 的参数与 `matplotlib` 的其他方法一样，需要 X 和 Y 值的数组。它们的大小必须相同。数据从文件格式转换为 `float` 以确保数字格式。

步骤 4 改进了每个坐标轴上数据的呈现方式。为每个坐标轴创建一个函数，定义该坐标轴上的值应如何显示（以美元或分钟为单位）。该函数接受要显示的值和位置作为输入。通常，位置会被忽略。坐标轴格式化器将被 `.set_major_formatter` 覆盖。注意两个坐标轴都是通过 `.gca`（获取当前坐标轴）返回的。

使用 `.xlabel` 和 `.ylabel` 为坐标轴添加标签。

最后，*步骤 5* 在新窗口中显示图表。分析结果，我们可以说似乎有两种用户：一种花费少于 10 分钟且花费不超过 10 美元，另一种花费更多时间，并且更有可能花费高达 100 美元。

> 请注意，所呈现的数据是合成的，并且是根据预期结果生成的。现实生活中的数据可能看起来更分散。可以使用统计分析来以更高的复杂度确定趋势和模式。

### 更多内容...

散点图不仅可以显示二维点，还可以添加第三维（面积）甚至第四维（颜色）。

要添加这些元素，请使用参数 `s` 表示 `size`（大小），`c` 表示 `color`（颜色）。

> `size` 定义为以点为单位的球的直径的平方。因此，对于直径为 10 的球，将使用 100。`color` 可以使用 `matplotlib` 中任何常见的颜色定义，例如十六进制颜色、RGB 等。有关更多详细信息，请参阅文档：https://matplotlib.org/users/colors.html。

例如，我们可以使用以下方式生成一个包含四个维度的随机图表：

```python
>>> import matplotlib.pyplot as plt
>>> import random
>>> NUM_POINTS = 100
>>> COLOR_SCALE = ['#FF0000', '#FFFF00', '#FFFF00', '#7FFF00', '#00FF00']
>>> data_x = [random.random() for _ in range(NUM_POINTS)]
>>> data_y = [random.random() for _ in range(NUM_POINTS)]
>>> size = [(50 * random.random()) ** 2 for _ in range(NUM_POINTS)]
>>> color = [random.choice(COLOR_SCALE) for _ in range(NUM_POINTS)]
>>> plt.scatter(data_x, data_y, s=size, c=color, alpha=0.5)
<matplotlib.collections.PathCollection object at 0x123552ee0>
>>> plt.show()
```

`COLOR_SCALE` 从绿色到红色，每个点的大小将在 0 到 50 点直径之间。结果应该类似于：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_288_0.png)

请注意，值是随机的，因此每次运行代码都会生成不同的图表。

`alpha` 值使每个点半透明，使我们能够看到它们重叠的位置。此值越高，点的透明度越低。此参数会影响显示的颜色，因为它会将点与背景混合。

> 尽管可以显示两个独立的值（大小和颜色），但它们也可以与其他任何值相关联。例如，使颜色依赖于大小将使所有相同大小的点具有相同的颜色，这可能有助于我们区分数据。请记住，图表的最终目标是使数据易于理解。尝试不同的方法来改进这一点。

完整的 `matplotlib` 文档可以在这里找到：https://matplotlib.org/。

### 另请参阅

- 本章前面的 *显示多条线* 配方，了解如何绘制遵循时间序列的多条线。
- 本章后面的 *添加图例和注释* 配方，了解如何向图表添加上下文信息。

### 可视化地图

显示随区域变化的信息的最佳方式是创建一个呈现信息的地图，同时为数据提供区域位置感。

在这个配方中，我们将利用 `Fiona` 模块导入 GIS 信息，以及 `matplotlib` 来显示信息。我们将显示西欧地图，并用颜色等级显示每个国家的人口。颜色越深，人口越多。

### 准备工作

我们需要在虚拟环境中安装 `matplotlib` 和 `Fiona`：

```bash
$ echo "matplotlib==3.2.1" >> requirements.txt
$ echo "Fiona==1.8.13" >> requirements.txt
$ pip install -r requirements.txt
```

如果你使用的是 macOS，可能会遇到这样的错误：**RuntimeError: Python is not installed as a framework**。请参阅 matplotlib 文档了解如何修复此问题：https://matplotlib.org/faq/osx_framework.html。

地图数据需要下载。幸运的是，地理信息有大量免费可用的数据。在 Google 上搜索应该能快速找到几乎所有你需要的信息，包括关于地区、县、河流或任何其他类型数据的详细信息。

> GIS 信息以不同格式由许多公共组织提供。Fiona 能够理解大多数常见格式并以等效方式处理它们，但存在细微差别。请阅读 Fiona 文档了解更多详情。

本食谱中我们将使用的数据覆盖所有欧洲国家，可在以下 GitHub URL 获取：https://github.com/leakyMirror/map-of-europe/blob/master/GeoJSON/europe.geojson。请注意，它是 GeoJSON 格式，这是一种易于使用的标准。

### 如何操作...

1.  导入模块：

    ```python
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.cm as cm
    >>> import fiona
    ```

2.  加载要显示的国家人口数据：

    ```python
    >>> COUNTRIES_POPULATION = {
    ...     'Spain': 47.2,
    ...     'Portugal': 10.6,
    ...     'United Kingdom': 63.8,
    ...     'Ireland': 4.7,
    ...     'France': 64.9,
    ...     'Italy': 61.1,
    ...     'Germany': 82.6,
    ...     'Netherlands': 16.8,
    ...     'Belgium': 11.1,
    ...     'Denmark': 5.6,
    ...     'Slovenia': 2,
    ...     'Austria': 8.5,
    ...     'Luxembourg': 0.5,
    ...     'Andorra': 0.077,
    ...     'Switzerland': 8.2,
    ...     'Liechtenstein': 0.038,
    ... }
    >>> MAX_POPULATION = max(COUNTRIES_POPULATION.values())
    >>> MIN_POPULATION = min(COUNTRIES_POPULATION.values())
    ```

3.  准备颜色映射表，它将决定每个国家显示的颜色（不同深浅的绿色）。计算每个国家对应的颜色：

    ```python
    >>> colormap = cm.get_cmap('Greens')
    >>> COUNTRY_COLOUR = {
    ...     country_name: colormap(
    ...         (population - MIN_POPULATION) / (MAX_POPULATION - MIN_POPULATION)
    ...     )
    ...     for country_name, population in COUNTRIES_POPULATION.items()
    ... }
    ```

4.  打开文件并读取数据，筛选出我们在步骤 1 中定义了人口的国家：

    ```python
    >>> with fiona.open('europe.geojson') as fd:
    ...     full_data = [data for data in fd
    ...                 if data['properties']['NAME'] in COUNTRIES_POPULATION]
    ```

5.  以正确的颜色绘制每个国家：

    ```python
    >>> for data in full_data:
    ...     country_name = data['properties']['NAME']
    ...     color = COUNTRY_COLOUR[country_name]
    ...     geo_type = data['geometry']['type']
    ...     if geo_type == 'Polygon':
    ...         data_x = [x for x, y in data['geometry']['coordinates'][0]]
    ...         data_y = [y for x, y in data['geometry']['coordinates'][0]]
    ...         plt.fill(data_x, data_y, c=color)
    ...     elif geo_type == 'MultiPolygon':
    ...         for coordinates in data['geometry']['coordinates']:
    ...             data_x = [x for x, y in coordinates[0]]
    ...             data_y = [y for x, y in coordinates[0]]
    ...             plt.fill(data_x, data_y, c=color)
    ```

6.  显示结果：

    ```python
    >>> plt.show()
    ```

7.  结果将显示在一个新窗口中，如下所示：

    ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_292_0.png)

### 工作原理...

在 *如何操作...* 部分的 *步骤 1* 中导入模块后，*步骤 2* 定义了要显示的数据。请注意，名称需要与 GEO 文件中的格式相同。计算最小和最大人口值，以便稍后正确平衡范围。

> 人口数据已四舍五入到一个有效数字，并以百万为单位定义。本食谱仅定义了少数几个国家，但 GIS 文件中还有更多可用数据。

步骤 3 描述了一个 `colormap`，它定义了绿色（Greens）深浅的颜色范围。这是 matplotlib 中的一个标准 `colormap`，但我们可以使用其他颜色，如橙色、红色或 plasma 以实现更冷到热的渐变效果。更多详情请参阅文档：https://matplotlib.org/examples/color/colormaps_reference.html。

`COUNTRY_COLOUR` 字典存储了 `colormap` 为每个国家定义的颜色。人口值被归一化到 0.0（最少人口）到 1.0（最多人口）之间的数字，并传递给 `colormap` 以检索其对应比例尺上的颜色。

然后在步骤 4 中检索 GIS 信息。`fiona` 读取 `europe.geojson` 文件，数据被复制以便在后续步骤中使用。它还进行了筛选，只处理我们定义了人口的国家，这意味着不会绘制额外的国家。

步骤 5 中的循环遍历 `data_list` 中的国家信息列表，然后使用 `.fill` 绘制每个国家的几何图形，该函数绘制一个多边形。每个国家的几何图形要么是单个多边形（`Polygon`），要么是多个（`MultiPolygon`）。在每种情况下，都会绘制相应的多边形，所有多边形颜色相同。这意味着 `MultiPolygon` 会被绘制多次。

> GIS 信息以点的形式存储，坐标描述了点的经纬度。区域（如国家）有一个坐标列表，描述其内部的一个区域。一些地图更精确，有更多的点来定义区域。可能需要多个多边形来定义一个国家，因为某些部分可能彼此分离，岛屿是最明显的例子，但也有飞地。飞地是与主体分离的国家区域，例如阿拉斯加。

最后，通过调用 `.show` 显示数据。

### 更多内容...

利用 GIS 文件中包含的信息，我们可以在地图上添加额外信息。`properties` 对象包含国家名称的信息，但也包含其 ISO 名称、FID 代码以及作为 `LON` 和 `LAT` 的中心位置。我们可以使用此信息通过 `.text` 显示国家名称：

```python
long, lat = data['properties']['LON'], data['properties']['LAT']
iso3 = data['properties']['ISO3']
plt.text(long, lat, iso3, horizontalalignment='center')
```

此代码将位于 *如何操作...* 部分 *步骤 6* 的循环内。

> 如果你分析该文件，你会看到 `properties` 对象包含人口信息，存储为 `POP2005`，因此你可以直接从地图中绘制人口信息。这留作练习。不同的地图文件包含不同的信息，因此请务必尝试以释放所有可能性。

此外，你可能会注意到地图在某些情况下可能会失真。`matplotlib` 会尝试将其呈现在一个方形框中，如果地图不是大致方形的，这将很明显。例如，尝试只显示西班牙、葡萄牙、爱尔兰和英国。我们可以强制图表以相同的间距呈现 1 度纬度和 1 度经度，如果我们绘制的不是靠近极地的东西，这是一个好方法。这可以通过在坐标轴上调用 `.set_aspect` 来实现。当前坐标轴可以通过 `.gca`（**获取当前坐标轴**）获得：

```python
>>> axes = plt.gca()
>>> axes.set_aspect('equal', adjustable='box')
```

此外，为了改善地图的外观，我们可以设置背景颜色以帮助区分背景和前景，并移除坐标轴上的标签，因为打印经纬度可能会分散注意力。移除坐标轴上的标签可以通过使用 `.xticks` 和 `.yticks` 设置空标签来实现。背景颜色由坐标轴的前景颜色决定：

```python
>>> plt.xticks([])
([], <a list of 0 Text major ticklabel objects>)
>>> plt.yticks([])
([], <a list of 0 Text major ticklabel objects>)
>>> axes = plt.gca()
>>> axes.set_facecolor('xkcd:light blue')
```

最后，为了更好地区分不同区域，可以添加围绕每个区域的线条。这可以通过在 `.fill` 之后立即使用相同的数据绘制一条细线来实现。请注意，此代码在 *步骤 2* 中重复了两次：

```python
>>> plt.fill(data_x, data_y, c=color)
[<matplotlib.patches.Polygon object at 0x1161a49d0>]
>>> plt.plot(data_x, data_y, c='black', linewidth=0.2)
[<matplotlib.lines.Line2D object at 0x1161a4b80>]
```

### 绘制精美的图表

将所有这些元素应用到地图后，现在看起来是这样的：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_295_0.png)

生成的代码可在此处的 GitHub 上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter08/visualising_maps.py。

> 正如我们所看到的，地图是作为通用多边形绘制的。不要害怕包含其他几何图形。你可以定义自己的多边形，并使用 `.fill` 或一些额外的标签来打印它们。例如，偏远地区可能需要被移走，以避免地图过大。或者，可以使用矩形在地图的部分区域上打印额外信息。

完整的 Fiona 文档可在此处找到：https://fiona.readthedocs.io/。完整的 matplotlib 文档可在此处找到：https://matplotlib.org/。

### 另请参阅

- 本章接下来的 *添加图例和注释* 配方，了解如何向图表添加上下文信息。
- 本章稍后的 *组合图表* 配方，了解如何向单个图表添加多个绘图。

### 添加图例和注释

在绘制信息密集的图表时，可能需要一个图例来显示相关信息，以提高对所呈现数据的理解，例如什么颜色对应什么概念。在 matplotlib 中，图例可以非常丰富，并且有多种呈现方式。用于吸引对特定点注意力的注释也有助于读者理解图表上显示的信息。

在这个配方中，我们将创建一个包含三个不同组件的图表，并显示一个带有信息的图例以便更好地理解它，同时对我们图表上最有趣的点进行注释。

### 准备工作

我们需要在虚拟环境中安装 matplotlib：

```
$ echo "matplotlib==3.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

如果你使用的是 macOS，你可能会遇到这样的错误：**RuntimeError: Python is not installed as a framework**。请参阅 matplotlib 文档了解如何修复它：https://matplotlib.org/faq/osx_framework.html。

### 操作步骤...

1. 导入 matplotlib：
   ```python
   >>> import matplotlib.pyplot as plt
   ```

2. 准备要在图表上显示的数据以及应显示的图例。每一行由时间标签、产品A的销售额、产品B的销售额和产品C的销售额组成：
   ```python
   >>> LEGEND = ('ProductA', 'ProductB', 'ProductC')
   >>> DATA = (
   ...     ('Q1 2017', 100, 30, 3),
   ...     ('Q2 2017', 105, 32, 15),
   ...     ('Q3 2017', 125, 29, 40),
   ...     ('Q4 2017', 115, 31, 80),
   ... )
   ```

3. 将数据拆分为图表可用的格式。这是一个准备步骤：
   ```python
   >>> POS = list(range(len(DATA)))
   >>> VALUESA = [valueA for label, valueA, valueB, valueC in DATA]
   >>> VALUESB = [valueB for label, valueA, valueB, valueC in DATA]
   >>> VALUESC = [valueC for label, valueA, valueB, valueC in DATA]
   >>> LABELS = [label for label, valueA, valueB, valueC in DATA]
   ```

4. 使用数据创建一个柱状图：
   ```python
   >>> WIDTH = 0.2
   >>> plt.bar([p - WIDTH for p in POS], VALUESA, width=WIDTH)
   <BarContainer object of 4 artists>
   >>> plt.bar([p for p in POS], VALUESB, width=WIDTH)
   <BarContainer object of 4 artists>
   >>> plt.bar([p + WIDTH for p in POS], VALUESC, width=WIDTH)
   <BarContainer object of 4 artists>
   >>> plt.ylabel('Sales')
   Text(0, 0.5, 'Sales')
   >>> plt.xticks(POS, LABELS)
   <REDACTED>
   ```

5. 添加一个注释，显示图表中的最大增长：
   ```python
   >>> plt.annotate('400% growth', xy=(1.2, 18), xytext=(1.3, 40),
   ...              horizontalalignment='center',
   ...              arrowprops=dict(facecolor='black', shrink=0.05))
   Text(1.3, 40, '400% growth')
   ```

6. 添加图例：
   ```python
   >>> plt.legend(LEGEND)
   <matplotlib.legend.Legend object at 0x1153d1e80>
   ```

7. 显示图表：
   ```python
   >>> plt.show()
   ```

8. 结果将显示在一个新窗口中：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_298_0.png)

### 工作原理...

*操作步骤...* 部分的步骤 1 和 2 准备了导入和柱状图将显示的数据。在 *步骤 3* 中，数据被拆分为不同的数组，为 `matplotlib` 准备输入。基本上，每个数据序列都存储在不同的数组中。

*步骤 4* 绘制数据。每个数据序列都调用 `.bar`，指定其位置和值。标签与 `.xticks` 的作用相同。为了将每个柱子围绕标签分开，第一个柱子向左偏移，第三个柱子向右偏移。

在第二季度 `ProductC` 的柱子上方添加了一个注释。请注意，注释包括 `xy` 中的点和 `xytext` 中的文本位置。

在 *步骤 6* 中，添加了图例。请注意，标签需要按照数据输入的相同顺序添加到数据中。图例会自动定位在不覆盖任何数据的区域。`Arrowprops` 告诉箭头指向数据。

最后，在步骤 7 中通过调用 `.show` 绘制图表。

> 调用 `.show` 会阻塞程序的执行。当窗口关闭时，程序将恢复。

### 更多内容...

在大多数情况下，只需调用 `.legend` 即可自动显示图例。如果你需要自定义它们出现的顺序，可以将每个标签引用到特定的元素。例如，这种方式（注意它将 `ProductA` 称为 `valueC` 系列）：
```python
>>> valueA = plt.bar([p - WIDTH for p in POS], VALUESA, width=WIDTH)
>>> valueB = plt.bar([p for p in POS], VALUESB, width=WIDTH)
>>> valueC = plt.bar([p + WIDTH for p in POS], VALUESC, width=WIDTH)
>>> plt.legend((valueC, valueB, valueA), LEGEND)
<matplotlib.legend.Legend object at 0x112273fa0>
```

图例的位置也可以通过 `loc` 参数手动更改。默认情况下，它是 `best`，它会将图例绘制在数据重叠最少的区域（理想情况下没有重叠）。但可以使用 `right`、`upper`、`left` 等值，或者一个特定的 `(X, Y)` 元组。

另一个选项是使用 `bbox_to_anchor` 选项将图例绘制在图表外部。在这种情况下，图例附加到边界框的 `(X, Y)`，其中 0 是图表的左下角，1 是右上角。这可能会导致图例被外部边框裁剪，因此你可能需要使用 `.subplots_adjust` 调整图表的起始和结束位置：
```python
>>> plt.legend(LEGEND, title='Products', bbox_to_anchor=(1, 0.8))
<matplotlib.legend.Legend object at 0x11963b910>
>>> plt.subplots_adjust(right=0.80)
```

调整 `bbox_to_anchor` 和 `.subplots_adjust` 参数需要一些试错才能产生预期的结果。

`.subplots_adjust` 将位置引用为将要显示的轴的位置。这意味着 `right=0.80` 将在绘图右侧留出 20% 的屏幕，而 `left` 的默认值为 0.125，意味着在绘图左侧留出 12.5% 的空间。有关更多详细信息，请参阅文档：https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html。

注释可以采用不同的样式，并且可以通过不同的选项进行调整，例如连接方式等。例如，此代码将创建一个带有 `fancy` 样式的箭头，以曲线方式连接。结果显示在此处：
```python
plt.annotate('400% growth', xy=(1.2, 18), xytext=(1.3, 40),
             horizontalalignment='center',
             arrowprops={'facecolor': 'black',
                         'arrowstyle': "fancy",
                         'connectionstyle': "angle3",
                         })
```

在我们的配方中，我们没有注释到柱子的末端（点 (1.2, 15)），而是稍微靠上一点，以留出一点呼吸空间。

调整要注释的确切点以及文本的位置需要一些测试。文本的位置也是通过寻找最佳位置以避免与柱子重叠来确定的。字体大小和颜色可以在 `.legend` 和 `.annotate` 调用中使用 `fontsize` 和 `color` 参数进行更改。

应用所有这些元素后，图表可能看起来类似于下面的图表。可以通过调用 `adding_legend_and_annotation.py` 脚本来复制此图表，该脚本可在 GitHub 上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter08/adding_legend_and_annotations.py：

绘制精美图表

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_301_0.png)

完整的 matplotlib 文档可在此处找到：https://matplotlib.org/。特别是，图例指南可在此处找到：https://matplotlib.org/users/legend_guide.html#plotting-guide-legend。注释指南可在此处找到：https://matplotlib.org/users/annotations.html。

### 另请参阅

- 本章前面的*绘制堆叠条形图*食谱，了解如何绘制受益于图例的累积值条形图。
- 本章接下来的*组合图表*食谱，了解如何向单个图表添加多个绘图。

### 组合图表

多个绘图可以组合在同一个图表中。在这个食谱中，我们将看到如何在同一个绘图中、两个不同的坐标轴上呈现数据，以及如何向同一个图表添加更多绘图。

### 准备工作

我们需要在虚拟环境中安装 matplotlib：

```
$ echo "matplotlib==3.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

如果你使用的是 macOS，可能会遇到这样的错误：**RuntimeError: Python is not installed as a framework**。请参阅 matplotlib 文档了解如何修复：https://matplotlib.org/faq/osx_framework.html。

### 操作步骤...

1. 导入 matplotlib：
    ```python
    >>> import matplotlib.pyplot as plt
    ```

2. 准备要在图表上显示的数据以及应显示的图例。每一行由时间标签、产品 A 的销售额和产品 B 的销售额组成。注意产品 B 的值比 A 高得多：
    ```python
    >>> DATA = (
    ...     ('Q1 2017', 100, 3000, 3),
    ...     ('Q2 2017', 105, 3200, 5),
    ...     ('Q3 2017', 125, 2900, 7),
    ...     ('Q4 2017', 115, 3100, 3),
    ... )
    ```

3. 将数据准备成独立的数组：
    ```python
    >>> POS = list(range(len(DATA)))
    >>> VALUESA = [valueA for label, valueA, valueB, valueC in DATA]
    >>> VALUESB = [valueB for label, valueA, valueB, valueC in DATA]
    >>> VALUESC = [valueC for label, valueA, valueB, valueC in DATA]
    >>> LABELS = [label for label, valueA, valueB, valueC in DATA]
    ```
    注意，这会展开并为每个值创建一个列表。

    > 也可以使用以下方式展开值：`LABELS, VALUESA, VALUESB, VALUESC = ZIP(*DATA)`。

4. 创建第一个子图：
    ```python
    >>> plt.subplot(2, 1, 1)
    <matplotlib.axes._subplots.AxesSubplot object at 0x115a91cd0>
    ```

5. 创建一个包含 VALUESA 信息的条形图：
    ```python
    >>> valueA = plt.bar(POS, VALUESA)
    >>> plt.ylabel('Sales A')
    Text(0, 0.5, 'Sales A')
    ```

6. 创建一个不同的 y 轴，并将 VALUESB 的信息添加为折线图：
    ```python
    >>> plt.twinx()
    <matplotlib.axes._subplots.AxesSubplot object at 0x118b0c160>
    >>> valueB = plt.plot(POS, VALUESB, 'o-', color='red')
    >>> plt.ylabel('Sales B')
    Text(0, 0.5, 'Sales B')
    >>> plt.xticks(POS, LABELS)
    <REDACTED>
    ```

7. 创建另一个子图并用 VALUESC 填充：
    ```python
    >>> plt.subplot(2, 1, 2)
    <matplotlib.axes._subplots.AxesSubplot object at 0x115abdfd0>
    >>> plt.plot(POS, VALUESC)
    [<matplotlib.lines.Line2D object at 0x118c7c0d0>]
    >>> plt.gca().set_ylim(ymin=0)
    (0.0, 7.2)
    >>> plt.xticks(POS, LABELS)
    <REDACTED>
    ```

8. 显示图表：
    ```python
    >>> plt.show()
    ```

9. 结果将显示在一个新窗口中：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_304_0.png)

### 工作原理...

在*操作步骤...*部分的*步骤 1*中，调用了模块所需的所有导入。

*步骤 2*以可用的格式呈现数据。

*步骤 3*是一个准备步骤，将数据拆分为不同的数组以供后续步骤使用。

*步骤 4*创建一个新的 `.subplot`。这将整个绘图分成两个元素。参数是行数、列数和选定的子图。因此，我们在一列中创建两个子图并在第一个中绘图。

步骤 5 使用 VALUESA 数据在此子图中打印一个 `.bar` 图，并使用 `.ylabel` 将 y 轴标记为 Sales A。

步骤 6 使用 `.twinx` 创建一个新的 y 轴，通过 `.plot` 将 VALUESB 绘制为折线图。标签使用 `.ylabel` 标记为 Sales B。x 轴使用 `.xticks` 标记。

> VALUESB 绘图设置为红色，以避免两个绘图具有相同的颜色。默认情况下，第一种颜色在两种情况下是相同的，这会导致混淆。数据点用 'o' 选项标记。

在步骤 7 中，我们使用 `.subplot` 切换到第二个子图。该绘图将 VALUESC 打印为一条线，并再次使用 `.xticks` 将标签放在 x 轴上，并将 y 轴的最小值设置为 0。然后在步骤 8 中显示图表。

### 更多内容...

通常，具有多个轴的图表阅读起来比较复杂。仅在有充分理由且数据高度相关时才使用它们。

> 默认情况下，折线图中的 y 轴将尝试在最小和最大 Y 值之间呈现信息。截断轴通常不是呈现信息的最佳方式，因为它会扭曲感知的差异。例如，如果图表从 10 到 11，那么 10 到 11 范围内的值变化看起来可能很重要，但这不到 10%。使用 `plt.gca().set_ylim(ymin=0)` 将 y 轴最小值设置为 0 是一个好主意，尤其是在有两个不同轴的情况下。

选择子图的调用将首先按行，然后按列进行，因此 `.subplot(2, 2, 3)` 将选择第一列、第二行的子图。

划分的子图网格可以更改。首先调用 `.subplot(2, 2, 1)` 和 `.subplot(2, 2, 2)`，然后调用 `.subplot(2, 1, 2)`，将创建一个结构，第一行有两个小图，第二行有一个更宽的图。返回将覆盖先前绘制的子图。

完整的 matplotlib 文档可在此处找到：https://matplotlib.org/。特别是，图例指南可在此处找到：https://matplotlib.org/users/legend_guide.html#plotting-guide-legend。注释指南可在此处找到：https://matplotlib.org/users/annotations.html。

### 另请参阅

- 本章前面的*绘制多条线*食谱，展示了在单个图表中显示多个值的替代方法。
- 本章前面的*可视化地图*食谱，了解如何显示具有多种数据的其他类型的丰富图表。

### 保存图表

图表准备好后，我们可以将其存储在硬盘上，以便在其他文档中引用。在这个食谱中，我们将看到如何以不同格式保存图表。

### 准备工作

我们需要在虚拟环境中安装 `matplotlib`：

```
$ echo "matplotlib==3.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

如果你使用的是 macOS，可能会遇到这样的错误：**RuntimeError: Python is not installed as a framework**。请参阅 `matplotlib` 文档了解如何修复：https://matplotlib.org/faq/osx_framework.html。

### 操作步骤...

1. 导入 `matplotlib`：
    ```python
    >>> import matplotlib.pyplot as plt
    ```

2. 准备要在图表上显示的数据并将其拆分为不同的数组：
    ```python
    >>> DATA = (
    ...     ('Q1 2017', 100),
    ...     ('Q2 2017', 150),
    ...     ('Q3 2017', 125),
    ...     ('Q4 2017', 175),
    ... )
    >>> POS = list(range(len(DATA)))
    >>> VALUES = [value for label, value in DATA]
    >>> LABELS = [label for label, value in DATA]
    ```

3. 使用数据创建条形图：
    ```python
    >>> plt.bar(POS, VALUES)
    <BarContainer object of 4 artists>
    >>> plt.xticks(POS, LABELS)
    <REDACTED>
    >>> plt.ylabel('Sales')
    Text(0, 0.5, 'Sales')
    ```

4. 将图表保存到硬盘：
    ```python
    >>> plt.savefig('data.png')
    ```

5. 打开新文件 data.png 以显示图表，如下所示：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_307_0.png)

图 8.17：以 PNG 格式保存的图表

### 工作原理...

在*操作步骤...*部分的步骤 1 和 2 中导入和准备数据后，步骤 3 通过调用 `.bar` 生成图表。添加了 `.ylabel`，并通过 `.xticks` 用适当的时间描述标记 x 轴。

步骤 4 将文件以名称 `data.png` 保存到硬盘。

### 更多内容...

图像的分辨率可以通过 `dpi` 参数确定。这将影响文件的大小。使用 72 到 300 之间的分辨率。较低的分辨率将难以阅读，而较高的分辨率除非图表尺寸巨大，否则没有意义：

```python
>>> plt.savefig('data.png', dpi=72)
```

`matplotlib` 知道如何存储最常见的文件格式，例如 JPEG、PDF 和 PNG。当文件名具有正确的扩展名时，它将自动使用。

> 除非有特定要求，否则请使用 PNG。与其他格式相比，它在存储颜色有限的图表方面非常高效。如果需要查找所有支持的文件，可以调用 `plt.gcf().canvas.get_supported_filetypes()`。

完整的 `matplotlib` 文档可在此处找到：https://matplotlib.org/。特别是，图例指南可在此处找到：https://matplotlib.org/users/legend_guide.html#plotting-guide-legend。注释指南可在此处找到：https://matplotlib.org/users/annotations.html。

### 另请参阅

- 本章前面的*绘制简单销售图表*食谱，了解绘制条形图的基础知识。
- 本章前面的*添加图例和注释*食谱，了解如何向图表添加上下文信息。

## 9 处理通信渠道

处理通信渠道是自动化能带来巨大收益的领域。在本章中，我们将了解如何使用两种最常见的通信渠道——包括新闻通讯在内的电子邮件，以及通过手机发送和接收短信。

多年来，这两个渠道都存在相当多的滥用情况，例如垃圾邮件或未经请求的营销信息，这使得发送者有必要使用外部工具来避免消息被自动过滤器拒绝。我们将在适用的地方介绍适当的注意事项。所介绍的工具具有许多功能，可以帮助你完成特定任务。它们还有优秀的文档，所以不要害怕阅读它。

在本章中，我们将涵盖以下食谱：

- 使用电子邮件模板
- 发送单个电子邮件
- 阅读电子邮件
- 向电子邮件新闻通讯添加订阅者
- 通过电子邮件发送通知
- 生成短信
- 接收短信
- 创建 Telegram 机器人

我们将从了解如何使用模板生成优质电子邮件开始。

### 使用电子邮件模板

要发送电子邮件，我们首先需要生成其内容。在这个食谱中，我们将了解如何生成合适的模板，包括纯文本样式和 HTML 格式。

### 准备工作

我们应该首先安装 `mistune` 模块，它将把 Markdown 文档编译成 HTML。我们还将使用 `jinja2` 模块将 HTML 与我们的文本结合起来：

```
$ echo "mistune==0.8.4" >> requirements.txt
$ echo "jinja2==2.11.1" >> requirements.txt
$ pip install -r requirements.txt
```

在本书的 GitHub 仓库中，有几个我们将使用的模板——位于 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter09/email_template.md 的 `email_template.md`，以及用于样式的模板 `email_styling.html`，位于 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter09/email_styling.html。

### 如何操作...

1. 导入模块：
    ```python
    >>> import mistune
    >>> import jinja2
    ```
2. 从磁盘读取两个模板：
    ```python
    >>> with open('email_template.md') as md_file:
    ...     markdown = md_file.read()

    >>> with open('email_styling.html') as styling_file:
    ...     styling = styling_file.read()
    ```
3. 定义要包含在模板中的数据。模板非常简单，只接受一个 `name` 参数：
    ```python
    >>> data = {'name': 'Seamus'}
    ```
4. 渲染 Markdown 模板。这将生成数据的纯文本版本：
    ```python
    >>> text = markdown.format(**data)
    ```
5. 渲染 Markdown 并添加样式：
    ```python
    >>> html_content = mistune.markdown(text)
    >>> html = jinja2.Template(styling).render(content=html_content)
    ```
6. 将文本和 HTML 版本保存到磁盘以进行检查：
    ```python
    >>> with open('text_version.txt', 'w') as fp:
    ...     fp.write(text)
    >>> with open('html_version.html', 'w') as fp:
    ...     fp.write(html)
    ```
7. 退出解释器并检查文本版本：
    ```
    $ cat text_version.txt
    Hi Seamus:

    This is an email talking about **things**

    ### Very important info

    1. One thing
    2. Other thing
    3. Some extra detail

    Best regards,

    *The email team*
    ```
8. 在浏览器中检查 HTML 版本：

    ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_313_0.png)

    图 9.1：HTML 文件的渲染图像

### 工作原理...

步骤 1 导入了稍后将使用的模块，步骤 2 读取了将要渲染的两个模板。`email_template.md` 是内容的基础，它是一个 Markdown 模板。`email_styling.html` 是一个 HTML 模板，包含基本的 HTML 外围结构和 CSS 样式信息。

> 基本策略是以 Markdown 格式创建内容。这是一个可读的纯文本文件，可以作为电子邮件的一部分发送。然后，该内容可以转换为 HTML，并用一些样式包装以组装成完整的、带样式的 HTML 文件。样式文件 `email_styling.html` 有一个内容区域，我们可以将从 Markdown 渲染的 HTML 放入其中。

步骤 3 定义了将在 `email_template.md` 中渲染的数据。它是一个非常简单的模板，只需要一个名为 `name` 的参数。

在步骤 4 中，Markdown 模板使用 `data` 进行渲染。这将生成电子邮件的纯文本版本。

HTML 版本在步骤 5 中渲染。纯文本版本使用 `mistune` 渲染为 HTML，然后使用 `jinja2` 模板包装在 `email_styling.html` 中。最终版本是一个自包含的 HTML 文档。

最后，我们在步骤 6 中将两个版本，纯文本（作为 `text`）和 HTML（作为 `html`），保存到文件中。步骤 7 和 8 检查存储的值。信息是相同的，但在 HTML 版本中，样式更好。

### 更多内容...

使用 Markdown 使得生成同时包含纯文本和 HTML 版本的电子邮件变得容易。Markdown 在文本格式下相当可读，并且非常自然地渲染成 HTML。也就是说，也可以生成完全不同的 HTML 版本，这将允许更多的自定义并利用 HTML 的特性。

完整的 Markdown 语法可以在 `https://daringfireball.net/projects/markdown/syntax` 找到，一个包含最常用元素的优秀备忘单可以在 `https://www.markdownguide.org/cheat-sheet` 找到。

> 虽然制作电子邮件的纯文本版本并非严格必要，但这是一个好习惯，表明你关心阅读电子邮件的人。大多数电子邮件客户端接受 HTML，但这并非完全通用。

对于 HTML 电子邮件，请注意整个样式表信息应包含在电子邮件中。这意味着 CSS 需要嵌入到 HTML 中。避免对外部资源进行引用，这可能导致电子邮件在某些电子邮件客户端中无法正确渲染，甚至被判定为垃圾邮件。

`email_styling.html` 中的样式基于 modest 样式表，可以在这里找到：`http://markdowncss.github.io/`。还有更多可用的 CSS 样式表，使用 Google 搜索应该能找到更多。请记住删除任何外部引用，如前所述。

可以通过将图像编码为 base64 格式将其包含在 HTML 中，以便直接嵌入到 HTML img 标签中，而不是添加引用：

```python
>>> import base64
>>> with open("image.png", 'rb') as file:
...     encoded_data = base64.b64encode(file)
>>> print("<img src='data:image/png;base64,{data}'/>".format(data=encoded_data))
```

你可以在本文中找到有关此技术的更多信息：https://css-tricks.com/data-uris/。

mistune 完整文档可在 http://mistune.readthedocs.io/en/latest/ 获取，jinja2 文档可在 https://jinja.palletsprojects.com/en/2.11.x/ 找到。

### 另请参阅

- 第 5 章“生成精彩报告”中的“在 Markdown 中格式化文本”食谱，以了解更多关于 Markdown 的信息
- 第 5 章“生成精彩报告”中的“使用模板生成报告”食谱，以了解更多关于 Jinja2 模板的信息
- 本章后面的“发送单个电子邮件”食谱，以跟进如何发送编写的电子邮件

### 发送单个电子邮件

发送电子邮件最基本的方式是使用标准的**简单邮件传输协议 (SMTP)**。SMTP 是互联网上最古老的协议之一。尽管存在更新的专有协议，但 SMTP 是所有电子邮件提供商相互通信的标准，也是电子通信的支柱之一。

SMTP 允许你发送包含多个部分的富电子邮件，这些部分包含不同类型的数据。这些部分可用于添加附件，或生成替代部分，例如同一消息的纯文本和 HTML 版本，以便在兼容的电子邮件客户端上显示。

鉴于 SMTP 是如此强大的标准，它易于使用，并且几乎所有语言和操作系统都支持它。这种易用性也是一个弱点，因为垃圾邮件的数量巨大，电子邮件领域的大公司有强烈的动机禁止来自未经验证来源（如 Python 脚本）的电子邮件。

这就是为什么 SMTP 仅推荐用于非常零星的使用和简单的目的，例如每天向受控地址发送少量电子邮件。

> 不要使用此方法向分发列表或来自随机电子邮件地址的客户群发电子邮件。你可能会因反垃圾邮件规则而被服务提供商封禁。

### 准备工作

对于这个食谱，我们需要一个服务提供商的电子邮件账户。根据你使用的服务提供商，会有一些小的差异，但我们将使用 Gmail 账户，因为它们非常常见且可以免费访问。

由于 Gmail 的安全性，我们需要创建一个特定的应用密码，用于发送电子邮件。请按照此处的说明操作：https://support.google.com/accounts/answer/185833。这将帮助你生成用于此食谱目的的密码。请记住为邮件访问创建它。之后你可以删除密码以将其移除。

我们将使用 `smtplib` 模块，它是 Python 标准库的一部分。

### 处理通信渠道

### 如何操作...

1.  导入 `smtplib` 和 `email` 模块：
    ```python
    >>> import smtplib
    >>> from email.mime.multipart import MIMEMultipart
    >>> from email.mime.text import MIMEText
    ```

2.  设置凭据，将以下内容替换为你自己的。为了测试，我们将发送到同一个电子邮件地址，但你可以使用不同的地址：
    ```python
    >>> USER = 'your.account@gmail.com'
    >>> PASSWORD = 'YourPassword'
    >>> sent_from = USER
    >>> send_to = [USER]
    ```

3.  定义要发送的数据。注意两种替代方案，一种是纯文本，另一种是 HTML：
    ```python
    >>> text = "Hi!\nThis is the text version linking to https://www.packtpub.com/\nCheers!"
    >>> html = """<html><head></head><body>
    ... <p>Hi!<br>
    ... This is the HTML version linking to <a href="https://www.packtpub.com/">Packt</a><br>
    ... </p>
    ... </body></html>
    """
    ```

4.  将消息组合为 MIME 多部分，包括 `subject`、`to` 和 `from`：
    ```python
    >>> msg = MIMEMultipart('alternative')
    >>> msg['Subject'] = 'An interesting email'
    >>> msg['From'] = sent_from
    >>> msg['To'] = ', '.join(send_to)
    ```

5.  填充电子邮件的数据内容部分：
    ```python
    >>> part_plain = MIMEText(text, 'plain')
    >>> part_html = MIMEText(html, 'html')
    >>> msg.attach(part_plain)
    >>> msg.attach(part_html)
    ```

6.  使用 SMTP SSL 协议发送电子邮件：
    ```python
    >>> with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
    ...     server.login(USER, PASSWORD)
    ...     server.sendmail(sent_from, send_to, msg.as_string())
    ```

7.  电子邮件应该已经发送。检查你的电子邮件账户以获取该消息。查看*原始电子邮件*，你将看到完整的原始邮件，其中包含 HTML 和纯文本元素。以下截图中的电子邮件已被编辑：

    ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_318_0.png)

    图 9.2：同时包含纯文本和 HTML 格式的电子邮件

### 工作原理...

在*步骤 1*中，从 `smtplib` 和 `email` 导入相关模块后，*步骤 2*定义了从 Gmail 获取的凭据。

*步骤 3*展示了将要发送的 HTML 和文本。它们是替代方案，因此应该呈现相同的信息，但格式不同。

基本消息信息在*步骤 4*中设置。它指定了电子邮件的主题，以及*发件人*和*收件人*。*步骤 5*添加了多个部分，每个部分都有适当的 `MIMEText` 类型。

> 根据 MIME 格式，最后添加的部分是首选的替代方案，因此我们最后添加 HTML 部分。

*步骤 6*设置与服务器的连接，使用凭据登录，并发送消息。它使用 `with` 上下文来获取连接。请注意，地址 `smtp.gmail.com` 和端口 `465` 是 Gmail 专用的。

如果凭据有误，它将引发一个异常，提示用户名和密码不被接受。

### 更多内容...

请注意，`send_to` 是一个地址列表。你可以将电子邮件发送给多个地址。唯一的注意事项是在*步骤 4*中，它需要被指定为所有地址的逗号分隔值列表。

> 虽然可以将 `sent_from` 标记为与用于发送电子邮件的地址不同的地址，但不建议这样做。这可能被解释为试图伪造电子邮件的来源，并导致电子邮件被标记为垃圾邮件。

这里使用的服务器 `smtp.gmail.com` 是 Gmail 指定的服务器，定义的 SMTPS（安全 SMTP）端口是 `465`。Gmail 也接受端口 `587`，这是标准端口，但需要你通过调用 `.starttls` 来指定会话类型，如以下代码所示：
```python
with smtplib.SMTP('smtp.gmail.com', 587) as server:
    server.starttls()
    server.login(USER, PASSWORD)
    server.sendmail(sent_from, send_to, msg.as_string())
```

如果你对这些差异和两种协议的更多细节感兴趣，可以在这篇文章中找到更多信息：https://www.fastmail.com/help/technical/ssltlsstarttls.html。

完整的 `smtplib` 文档可以在 https://docs.python.org/3/library/smtplib.html 找到，而 `email` 模块，包含有关电子邮件不同格式的信息，包括 MIME 类型的示例，可以在这里找到：https://docs.python.org/3/library/email.html。MIME 类型可用于向电子邮件添加二进制附件。

### 另请参阅

- 本章前面的*使用电子邮件模板*食谱，了解如何组合电子邮件正文
- 本章后面的*通过电子邮件发送通知*食谱，了解如何发送批量电子邮件

### 读取电子邮件

在这个食谱中，我们将看到如何从一个账户读取电子邮件。我们将使用 IMAP4 标准，这是最常用的读取电子邮件的标准。

一旦读取，电子邮件可以被自动处理和分析，以生成诸如智能自动回复、将电子邮件转发到不同目标、聚合结果以进行监控等操作。选项是无限的！

### 准备工作

对于这个食谱，我们需要一个服务提供商的电子邮件账户。根据你使用的提供商，会有一些小的差异，但我们将使用一个 Gmail 账户，因为它非常常见且可以免费访问。

由于 Gmail 的安全性，我们需要创建一个特定的应用程序密码来发送电子邮件。请按照此处的说明操作：https://support.google.com/accounts/answer/185833。这将生成一个用于此食谱目的的密码。请记住为邮件创建它。之后你可以删除密码以将其移除。

我们将使用 `imaplib` 模块，它是 Python 标准库的一部分。

这个食谱将读取最后收到的电子邮件，因此你可以用它来更好地控制将要读取的内容。我们将发送一封简短的电子邮件，看起来像是发送给支持团队的。

### 如何操作...

1.  导入 `imaplib` 和 `email` 模块：
    ```python
    >>> import imaplib
    >>> import email
    >>> from email.parser import BytesParser, Parser
    >>> from email.policy import default
    ```

2.  设置凭据，将以下内容替换为你自己的：
    ```python
    >>> USER = 'your.account@gmail.com'
    >>> PASSWORD = 'YourPassword'
    ```

3.  连接到电子邮件服务器：
    ```python
    >>> mail = imaplib.IMAP4_SSL('imap.gmail.com')
    >>> mail.login(USER, PASSWORD)
    ```

4.  选择收件箱文件夹：
    ```python
    >>> mail.select('inbox')
    ```

5.  读取所有电子邮件 UID 并检索最新收到的电子邮件：
    ```python
    >>> result, data = mail.uid('search', None, 'ALL')
    >>> latest_email_uid = data[0].split()[-1]
    >>> result, data = mail.uid('fetch', latest_email_uid, '(RFC822)')
    >>> raw_email = data[0][1]
    ```

6.  将电子邮件解析为 Python 对象：
    ```python
    >>> email_message = BytesParser(policy=default).parsebytes(raw_email)
    ```

7.  显示电子邮件的主题和发件人：
    ```python
    >>> email_message['subject']
    '[Ref ABCDEF] Subject: Product A'
    >>> email.utils.parseaddr(email_message['From'])
    ('Sender name', 'sender@gmail.com')
    ```

8.  检索文本内容：
    ```python
    >>> email_type = email_message.get_content_maintype()
    >>> if email_type == 'multipart':
    ...     for part in email_message.get_payload():
    ...         if part.get_content_type() == 'text/plain':
    ...             payload = part.get_payload()
    ... elif email_type == 'text':
    ...     payload = email_message.get_payload()
    >>> print(payload)
    Hi:

    I'm having difficulties getting into my account. What was the URL, again?

    Thanks!
        A confused customer
    ```

### 工作原理...

在导入将要使用的模块并定义凭据之后，我们在*步骤 3*中连接到服务器。

*步骤 4*连接到 `inbox`。这是 Gmail 中包含收到的电子邮件的默认文件夹。

> 当然，你可能需要读取不同的文件夹。你可以通过调用 `mail.list()` 获取所有文件夹的列表。

在*步骤 5*中，首先通过调用 `.uid('search', None, "ALL")` 检索收件箱中所有电子邮件的 UID 列表。然后通过 `.uid('fetch', latest_email_uid, '(RFC822)')` 的 `fetch` 操作从服务器再次检索最新收到的电子邮件。这会以 RFC822 格式检索电子邮件，这是标准格式。请注意，检索电子邮件会将其标记为已读。

> `.uid` 命令允许我们调用 IMAP4 命令，返回一个包含结果（OK 或 NO）和数据的元组。如果有错误，它将引发适当的异常。

`BytesParser` 模块用于将原始的 RFC822 电子邮件转换为 Python 对象。这在*步骤 6*中完成。

### 处理通信渠道

元数据（包括主题、发件人和时间戳等详细信息）可以像字典一样访问，如*步骤7*所示。地址可以从原始文本格式中解析出来，使用 `email.utils.parseaddr` 分离出相应部分。

最后，可以展开并提取内容。如果邮件类型是 multipart，则可以通过遍历 `.get_payload()` 来提取每个部分。其中较容易处理的是 `plain/text`，因此假设存在该类型，*步骤8*中的代码将提取它。

邮件正文存储在 `payload` 变量中。

### 更多内容...

在*步骤5*中，我们从收件箱检索了所有邮件，但这并非必要。搜索命令可以通过过滤条件进行参数化，例如，只检索最近一天的邮件：

```python
import datetime
since = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%d-%b-%Y")
result, data = mail.uid('search', None, f'(SENTSINCE {since})')
```

这将根据邮件日期进行搜索。请注意，分辨率是以天为单位的。

通过 `IMAP4` 还可以执行更多操作。有关更多详细信息，请查看 RFC 3501（https://tools.ietf.org/html/rfc3501）和 RFC 6851（https://tools.ietf.org/html/rfc6851）。

> 前述 RFC 描述了 IMAP4 协议，可能有些枯燥。查看 RFC 索引中的可用命令，可以了解该协议的功能，然后你可以搜索如何实现特定命令的示例。

邮件的主题和正文，以及其他元数据（如日期、收件人、发件人等），都可以被解析和处理。例如，本配方中检索到的主题可以按以下方式处理：

```python
>>> import re
>>> re.search(r'\[Ref (\w+)\] Subject: (\w+)', '[Ref ABCDEF] Subject: Product A').groups()
('ABCDEF', 'Product')
```

### 另请参阅

- *第1章，开始我们的自动化之旅*，了解有关正则表达式和其他信息解析方式的更多信息。

### 向电子邮件通讯添加订阅者

一种常见的营销工具是电子邮件通讯。它们是向多个目标发送信息的便捷方式。一个好的通讯系统难以实现，推荐的方式是使用市场上可用的系统。一个著名的是 **MailChimp**（https://mailchimp.com/）。

MailChimp 有很多可能性，但与本书相关的是它的 API，可以通过脚本自动化工具。这个 RESTful API 可以通过 Python 访问。在本配方中，我们将看到如何向现有列表添加更多订阅者。

### 准备工作

由于我们将使用 MailChimp，我们需要拥有一个账户。你可以在 https://login.mailchimp.com/signup/ 创建一个免费账户。

创建账户后，请确保至少有一个我们可以添加订阅者的列表。作为注册过程的一部分，它可能已经为你创建。它将出现在 **Audience -> Manage Audience -> View Audience** 下：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_324_0.png)

**Audience** 将包含已订阅的用户。

对于 API，我们需要一个 API 密钥。转到 **Account -> Extras -> API keys** 并创建一个新的：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_325_0.png)

图 9.4：MailChimp 中 API 密钥的截图

我们将使用 `requests` 模块来访问 API。将其添加到你的虚拟环境中：

```
$ echo "requests==2.23" >> requirements.txt
$ pip install -r requirements.txt
```

MailChimp API 使用你的账户所使用的 **数据中心（DC）** 概念。这可以从你的 API 密钥的最后 4 位数字或 MailChimp 管理站点 URL 的开头获取。例如，us19。

### 如何操作...

1. 导入 `requests` 模块：

```python
>>> import requests
```

2. 定义认证和基础 URL。基础 URL 需要在开头包含你的 dc（例如 us19）：

```python
>>> API_KEY = 'your secret key'
>>> BASE = 'https://<dc>.api.mailchimp.com/3.0'
>>> auth = ('user', API_KEY)
```

3. 获取你所有的列表：

```python
>>> url = f'{BASE}/lists'
>>> response = requests.get(url, auth=auth)
>>> result = response.json()
```

4. 过滤你的列表以获取所需列表的 href：

```python
>>> LIST_NAME = 'Your list name'
>>> this_list = [l for l in result['lists'] if l['name'] == LIST_NAME][0]
>>> list_url = [l['href'] for l in this_list['_links'] if l['rel'] == 'self'][0]
```

5. 使用列表 URL，你可以获取列表成员的 URL：

```python
>>> response = requests.get(list_url, auth=auth)
>>> result = response.json()
>>> result['stats']
{'member_count': 1, 'unsubscribe_count': 0, 'cleaned_count': 0, ...}
>>> members_url = [l['href'] for l in result['_links'] if l['rel'] == 'members'][0]
```

6. 可以通过向 members_url 发送 GET 请求来检索成员列表：

```python
>>> response = requests.get(members_url, auth=auth)
>>> result = response.json()
>>> len(result['members'])
1
```

7. 向列表追加一个新成员：

```python
>>> new_member = {
    'email_address': 'test@test.com',
    'status': 'subscribed',
}
>>> response = requests.post(members_url, json=new_member, auth=auth)
```

8. 使用 GET 检索用户列表，将获取到两个用户：

```python
>>> response = requests.get(members_url, auth=auth)
>>> result = response.json()
>>> len(result['members'])
2
```

### 工作原理...

在步骤1导入 requests 模块后，我们在步骤2定义了连接的基本值：基础 URL 和凭据。请注意，对于认证，我们只需要 API 密钥作为密码，以及任何用户（如 MailChimp 文档所述：https://developer.mailchimp.com/documentation/mailchimp/guides/get-started-with-mailchimp-api-3/）。

步骤3通过调用适当的 URL 检索所有列表。结果以 JSON 格式返回。调用包含带有定义凭据的 auth 参数。所有后续调用都将使用此 auth 参数进行认证。

步骤4展示了如何过滤返回的列表以获取感兴趣列表的 URL。每个返回的调用都包含一个带有相关信息的 _links 列表，使得遍历 API 成为可能。

步骤5调用列表的 URL。这返回列表的信息，包括基本统计信息。通过应用类似于步骤4的过滤，我们检索成员的 URL。

> 由于大小限制并为了显示相关数据，并非所有检索到的元素都已显示。请随意交互式地分析它们并了解它们。数据结构良好，遵循 RESTful 的可发现性原则。Python 的内省使其相当可读和易于理解。

步骤6通过向 members_url 发送 GET 请求检索成员列表，这可以被视为单个用户。这可以在“准备工作”部分的 Web 界面中看到。

步骤7创建一个新用户，并通过 json 参数传递信息向 members_url 发送 POST 请求，以便将其转换为 JSON 格式。更新后的数据在步骤7中检索，显示列表中有一个新用户。

### 更多内容...

完整的 MailChimp API 功能非常强大，可以执行大量任务。请访问完整的 MailChimp 文档以发现所有可能性：https://developer.mailchimp.com/。

> 简要说明一下，这有点超出本书范围，请注意向自动化列表添加订阅者的法律影响。垃圾邮件是一个严重的问题，并且有新的法规来保护客户的权利，例如 GDPR。确保你有用户允许你向他们发送邮件的许可。好消息是 MailChimp 自动实现了工具来帮助解决这个问题，例如自动退订按钮。

通用的 MailChimp 文档也很有趣，展示了很多可能性。MailChimp 能够管理通讯和通用分发列表，但它也可以定制以生成流程、安排发送电子邮件，并根据参数（如生日）自动向你的受众发送消息。

### 另请参阅

- 本章前面的 *发送单个电子邮件* 配方，了解直接发送电子邮件与此类工具之间的区别
- 接下来的 *发送通知电子邮件* 配方，了解如何为特定用户的操作发送定制电子邮件

### 通过电子邮件发送通知

在本配方中，我们将介绍如何向客户发送电子邮件。事务性电子邮件是响应用户操作而发送的，例如确认或警报电子邮件。由于垃圾邮件保护和其他限制，最好借助外部工具来实现此类电子邮件。

在本配方中，我们将使用 **Mailgun**（https://www.mailgun.com），它能够发送此类电子邮件以及通信响应。

### 准备工作

我们需要在 Mailgun 创建一个账户。访问 https://signup.mailgun.com 创建一个。请注意，信用卡信息是可选的。

### 处理通信渠道

注册完成后，前往**域名**页面，你会看到一个沙盒环境。我们可以用它来测试 Mailgun 的功能，不过它只能向已注册的测试邮箱账户发送邮件。API 凭据会显示在那里：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_329_0.png)

图 9.5：Mailgun 中的域名信息

我们需要注册账户，这样我们才能作为*授权收件人*接收邮件。你可以在这里添加：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_329_1.png)

图 9.6：使用 Mailgun 验证账户

要验证账户，请检查授权收件人的邮箱并确认。该邮箱地址现在就可以接收测试邮件了：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_330_0.png)

图 9.7：账户已准备好接收测试邮件

我们将使用 `requests` 模块连接到 Mailgun API。在虚拟环境中安装它：

```
$ echo "requests==2.23" >> requirements.txt
$ pip install -r requirements.txt
```

一切准备就绪，可以发送邮件了，但请注意，你只能向授权收件人发送邮件。要能够向任何地方发送邮件，我们需要设置一个域名。请按照 Mailgun 的文档操作：`https://documentation.mailgun.com/en/latest/quickstart-sending.html#verify-your-domain`。

### 操作步骤...

1. 导入 `requests` 模块：
   ```python
   >>> import requests
   ```

2. 准备凭据，以及收件人和发件人邮箱。注意我们使用了一个模拟的发件人：
   ```python
   >>> KEY = 'YOUR-SECRET-KEY'
   >>> DOMAIN = 'YOUR-DOMAIN.mailgun.org'
   >>> TO = 'YOUR-AUTHORISED-RECEIVER'
   >>> FROM = f'sender@{DOMAIN}'
   >>> auth = ('api', KEY)
   ```

3. 准备要发送的邮件。这里有一个 HTML 版本和一个备选的纯文本版本：
   ```python
   >>> text = "Hi!\nThis is the text version linking to https://www.packtpub.com/\nCheers!"
   >>> html = '''<html><head></head><body>
   ...     <p>Hi!<br>
   ...         This is the HTML version linking to <a href="https://www.packtpub.com/">Packt</a><br>
   ...     </p>
   ... </body></html>'''
   ```

4. 设置要发送给 Mailgun 的数据：
   ```python
   >>> data = {
   ...     'from': f'Sender <{FROM}>',
   ...     'to': f'Jaime Buelta <{TO}>',
   ...     'subject': 'An interesting email!',
   ...     'text': text,
   ...     'html': html,
   ... }
   ```

5. 调用 API：
   ```python
   >>> response = requests.post(f"https://api.mailgun.net/v3/{DOMAIN}/messages", auth=auth, data=data)
   >>> response.json()
   {'id': '<YOUR-ID.mailgun.org>', 'message': 'Queued. Thank you.'}
   ```

6. 检索事件并确认邮件已送达：
   ```python
   >>> response_events = requests.get(f'https://api.mailgun.net/v3/{DOMAIN}/events', auth=auth)
   >>> response_events.json()['items'][0]['recipient'] == TO
   True
   >>> response_events.json()['items'][0]['event']
   'delivered'
   ```

7. 邮件应该会出现在你的收件箱中。由于它是通过沙盒环境发送的，如果它没有直接显示，请务必检查你的垃圾邮件文件夹。

### 工作原理...

步骤 1 导入了 requests 模块以供后续使用。步骤 2 定义了凭据和邮件的基本信息，这些信息应该从 Mailgun 的 Web 界面中提取，如前所示。

步骤 3 定义了将要发送的邮件。步骤 4 按照 Mailgun 期望的方式组织信息。注意 html 和 text 字段。默认情况下，它会将 HTML 设置为首选选项，纯文本选项作为备选。TO 和 FROM 的格式应为 `Name <address>` 格式。你可以在 TO 中使用逗号分隔多个收件人。

步骤 5 进行了 API 调用。这是一个向 messages 端点发起的 POST 调用。数据以标准方式传输，并使用 auth 参数进行基本身份验证。注意步骤 2 中的定义。所有对 Mailgun 的调用都应包含此 auth 参数。它返回一条消息，通知你操作成功，然后消息被加入队列。

在步骤 6 中，通过 GET 请求调用以检索事件。这将显示最近执行的操作，其中最后一个将是最近的发送操作。有关投递状态的信息也可以在这里找到。

### 更多内容...

要发送邮件，你需要设置用于发送的域名，而不是使用沙盒环境。你可以在这里找到如何操作的说明：https://documentation.mailgun.com/en/latest/quickstart-sending.html#verify-your-domain。这需要你更改 DNS 记录以验证你是该域名的合法所有者。这也能提高邮件的送达率。

邮件可以通过以下方式包含附件：

```python
attachments = [
    ("attachment",
        ("attachment1.jpg", open("image.jpg","rb").read())
    ),
    ("attachment",
        ("attachment2.txt", open("text.txt","rb").read())
    )]
response = requests.post(f"https://api.mailgun.net/v3/{DOMAIN}/messages",
                        auth=auth, files=attachments, data=data)
```

注意 (`"attachment"`, (`<filename>`, `<binary data>`) 的结构。

数据可以包含通常的信息，如 `cc` 或 `bcc`，但你也可以使用 `o:deliverytime` 参数将投递延迟最多三天：

```python
import datetime
import email.utils
delivery_time = datetime.datetime.now() + datetime.timedelta(days=1)
data = {
    ...
    'o:deliverytime': email.utils.format_datetime(delivery_time),
}
```

Mailgun 也可用于接收邮件，并在邮件到达时触发流程；例如，根据规则转发邮件。请查阅 Mailgun 文档以了解更多信息。

完整的 Mailgun 文档可以在这里找到：`https://documentation.mailgun.com/en/latest/quickstart.html`。请务必查看他们的*最佳实践*部分（`https://documentation.mailgun.com/en/latest/best_practices.html#email-best-practices`），以了解发送邮件的世界以及如何避免邮件被标记为垃圾邮件。

### 另请参阅

- 本章前面的*使用电子邮件模板*食谱，学习如何使用模板为邮件设置样式
- 本章前面的*发送单个电子邮件*食谱，学习如何直接从 Python 发送邮件，而不是使用外部服务

### 生成短信

最广泛可用的通信渠道之一是短信。短信在分发信息方面非常方便。

> 短信可用于营销目的，也可用作警报或发送通知的方式，或者最近非常常见的是，作为实现双因素认证系统的一种方式。

我们将使用 **Twilio**，一个提供 API 以轻松发送短信的服务。

### 准备工作

我们需要在 https://www.twilio.com/ 为 Twilio 创建一个账户。访问该页面并注册一个新账户。

你需要按照说明设置一个电话号码来接收消息。你需要输入发送到此电话的代码或接听电话以验证电话线路是否正确。

创建一个新项目并查看仪表板。从那里，你将能够创建你的第一个电话号码，以便接收和发送短信：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_334_0.png)

号码配置完成后，它将出现在**所有产品和服务 -> 电话号码**的**活跃号码**部分。

在主仪表板上，查看 `ACCOUNT SID` 和 `AUTH TOKEN`。它们将在后面使用。注意你需要显示 auth token。

我们还需要安装 twilio 模块。将其添加到你的虚拟环境中：

```
$ echo "twilio==6.37.0" >> requirements.txt
$ pip install -r requirements.txt
```

请注意，对于试用账户，接收方电话号码只能是已验证的号码。你可以验证多个号码；请按照 https://support.twilio.com/hc/en-us/articles/223180048-Adding-a-Verified-Phone-Number-or-Caller-ID-with-Twilio 中的文档操作。

### 操作步骤...

1. 从 twilio 模块导入 Client：
   ```python
   >>> from twilio.rest import Client
   ```

2. 设置之前从仪表板获取的身份验证凭据。同时，设置你的 Twilio 电话号码；例如，这里我们设置 +353 12 345 6789，一个虚构的爱尔兰号码。它应该是你所在国家的本地号码：
   ```python
   >>> ACCOUNT_SID = 'Your account SID'
   >>> AUTH_TOKEN = 'Your secret token'
   >>> FROM = '+353 12 345 6789'
   ```

3. 启动客户端以访问 API：
   ```python
   >>> client = Client(ACCOUNT_SID, AUTH_TOKEN)
   ```

4. 向你的授权电话号码发送消息。注意 from_ 末尾的下划线：
   ```python
   >>> message = client.messages.create(body='This is a test message from Python!',
                                      from_=FROM,
                                      to='+your authorised number')
   ```

5. 你将在手机上收到一条短信：

### 工作原理...

使用 Twilio 客户端发送消息非常直接。

在*步骤 1*中，我们导入 `Client`。在*步骤 2*中，我们准备凭据并配置电话号码。

*步骤 3*使用正确的身份验证创建客户端，然后在*步骤 4*中发送消息。

处理通信渠道

> 请注意，在试用账户下工作时，目标号码必须是已验证的号码之一，否则会产生错误。您可以添加更多已验证的号码；请查阅 Twilio 文档。

所有从试用账户发送的消息都会在短信中包含该详情，如步骤 5 所示。

### 还有更多...

在某些地区（撰写本文时为美国和加拿大），短信号码能够发送包含图片的 MMS 消息。要将图片附加到消息中，请添加 `media_url` 参数和要发送的图片 URL：

```
client.messages.create(body='An MMS message',
                     media_url='http://my.image.com/image.png',
                     from_=FROM,
                     to='+your authorised number')
```

该客户端基于 RESTful API，允许您执行多种操作，例如创建新电话号码或先获取一个可用号码然后购买它：

```
available_numbers = client.available_phone_numbers("IE").local.list()
number = available_numbers[0]
new_number = client.incoming_phone_numbers.create(phone_number=number.phone_number)
```

查阅文档以获取更多可用操作；大多数仪表板上的点击操作都可以通过编程方式执行。

> Twilio 还能够执行其他电话服务，例如电话呼叫和文本转语音。请在完整文档中查看。

完整的 Twilio 文档可在此处获取：https://www.twilio.com/docs/.

### 另请参阅

- 接下来的*接收短信*食谱，了解如何接收消息以及发送消息
- 本章后面的*创建 Telegram 机器人*食谱，以进一步扩展您的知识

### 接收短信

短信也可以被接收和自动处理。这使得按需提供信息等服务成为可能（例如，发送 INFO GOALS 以接收足球联赛的结果），以及更复杂的流程，例如在机器人中，可以与用户进行简单对话，从而实现远程配置恒温器等丰富服务。

> 每次 Twilio 向您注册的电话号码之一接收短信时，它都会向一个公开可用的 URL 发起请求。这在 Twilio 服务中进行配置，意味着访问该 URL 时执行的代码应由您控制。这带来了该 URL 需要在互联网上可用的问题。这意味着仅使用您的本地计算机是行不通的，因为它很可能无法从您的本地网络外部寻址。我们将使用 Heroku (http://heroku.com) 来提供可用服务，但也有其他替代方案。Twilio 文档提供了使用 ngrok 的示例，它通过在公共地址和您的本地开发环境之间创建隧道来允许本地开发。更多详情请见此处：https://www.twilio.com/blog/2013/10/test-your-webhooks-locally-with-ngrok.html.

这种操作方式在通信 API 中很常见。值得注意的是，Twilio 有一个用于 WhatsApp 的 beta API，其工作方式类似。更多信息请查阅文档：https://www.twilio.com/docs/sms/whatsapp/quickstart/python.

处理通信渠道

### 准备工作

我们需要在 https://www.twilio.com/ 创建一个 Twilio 账户。有关详细说明，请参阅*发送短信*食谱中的*准备工作*部分。

对于本食谱，我们还需要在 Heroku (https://www.heroku.com/) 上设置一个 Web 服务，以便能够创建一个能够接收发送给 Twilio 的短信的 webhook。因为本食谱的主要目标是短信部分，我们在设置 Heroku 时会保持简洁，但您可以参考其出色的文档。它使用起来相当简单：

1. 在 Heroku 创建一个账户。
2. 您需要安装 Heroku 的命令行界面（所有平台的说明可在 https://devcenter.heroku.com/articles/getting-started-with-python#set-up 找到），然后登录命令行：

   ```
   $ heroku login
   Enter your Heroku credentials.
   Email: your.user@server.com
   Password:
   ```

3. 从 https://github.com/datademofun/heroku-basic-flask 下载一个基本的 Heroku 模板。我们将用它作为服务器的基础。
4. 将 `twilio` 客户端添加到 `requirements.txt` 文件中：

   ```
   $ echo "twilio" >> requirements.txt
   ```

5. 用 GitHub 上 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter09/app.py 中的文件替换 `app.py`。其关键部分是获取请求的正文并将其与一些额外信息一起发回。代码将在后面的*工作原理...*部分显示。

> 在文件 `runtime.txt` 中将 python 解释器替换为最新支持的 python 版本。撰写本文时，该版本为 3.8.3。请查阅 heroku 文档 https://devcenter.heroku.com/articles/python-support#specifying-a-python-version

> 您可以保留现有的 app.py 以查看模板示例以及 Heroku 的工作方式。请查看 https://github.com/datademofun/heroku-basic-flask 上的 README。

6. 完成后，将更改提交到 Git：

   ```
   $ git add .
   $ git commit -m 'first commit'
   ```

7. 在 Heroku 中创建一个新服务。它将随机生成一个新的服务名称（此处我们使用 service-name-12345）。此 URL 可访问：

   ```
   $ heroku create
   Creating app... done, ● SERVICE-NAME-12345
   https://service-name-12345.herokuapp.com/ | https://git.heroku.com/service-name-12345.git
   ```

8. 部署服务。在 Heroku 中，部署服务会将代码推送到远程 Git 服务器：

   ```
   $ git push heroku master
   ...
   remote: Verifying deploy... done.
   To https://git.heroku.com/service-name-12345.git
   b6cd95a..367a994 master -> master
   ```

9. 检查服务是否在 webhook URL 处正常运行。请注意，它在上一步的输出中显示。您也可以在浏览器中检查：

   ```
   $ curl https://service-name-12345.herokuapp.com/
   All working!
   ```

处理通信渠道

### 操作步骤...

1. 转到 Twilio 并访问 **PHONE NUMBER** 部分。配置 webhook URL。这将使每次收到短信时都会调用该 URL。转到 **All Products** 和 **Services** -> **Phone Numbers** 中的 **Active Numbers** 部分，并填写 webhook。注意 webhook 末尾的 `/sms`。点击 **Save**：

   ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_341_0.png)

   图 9.10：在 Twilio 中配置电话号码

2. 服务现在已启动并运行，可以使用了。向您的 Twilio 电话号码发送一条短信；您应该会收到自动回复：

   ![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_342_0.png)

   图 9.11：来自 Twilio 试用账户的短信

   请注意，模糊的部分应替换为您的信息。

> 如果您有试用账户，您只能将消息发回给您授权的电话号码之一，因此您需要从这些号码发送文本。

处理通信渠道

### 工作原理...

步骤 1 设置了 webhook，因此当 Twilio 在电话线上收到短信时，它会调用您的 Heroku 应用程序。

让我们看看 `app.py` 中的代码，了解它是如何工作的。以下是代码，为清晰起见进行了删减；完整文件请查看 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter09/app.py：

```
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

@app.route('/')
def homepage():
    return 'All working!'

@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
    from_number = request.form['From']
    body = request.form['Body']
    resp = MessagingResponse()
    msg = (f'Awwwww! Thanks so much for your message {from_number}, '
           f'"{body}" to you too.')
    resp.message(msg)
    return str(resp)

if __name__ == '__main__':
    app.run()
```

app.py 可以分为三个部分：

- 文件开头的 Python 导入和 Flask 应用程序的启动，这只是设置 Flask
- 对 homepage 的调用，这是为了测试服务器是否正常工作而生成的
- sms_reply，这是魔法发生的地方

`sms_reply` 函数从 `request.form` 字典中获取发送短信的电话号码以及消息正文。然后，在 `msg` 中编写回复内容，将其附加到一个新的 `MessagingResponse` 对象中，并返回该对象。

> 我们将用户的消息作为一个整体来处理，但请记住 *第1章，开启自动化之旅* 中提到的所有文本解析技术。它们都适用于此处，可用于检测预定义操作或进行任何其他文本处理。

返回的值将由 Twilio 发送回发送者，从而产生 *步骤 2* 中看到的结果。

### 更多内容...

为了能够生成自动对话，对话的状态应该被存储。对于高级状态，可能应该将其存储在数据库中，生成一个流程；但对于简单的情况，将信息存储在 `session` 中可能就足够了。会话能够在同一对发送和接收电话号码之间持久化存储信息到 cookie 中，允许你在消息之间检索它。

例如，此修改不仅会返回发送的消息正文，还会返回上一条消息。仅包含相关部分：

```python
app = Flask(__name__)
app.secret_key = b'somethingreallysecret!!!!'
...

@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
    from_number = request.form['From']
    last_message = session.get('MESSAGE', None)
    body = request.form['Body']
    resp = MessagingResponse()
    msg = (f'Awwwww! Thanks so much for your message {from_number}, '
           f'"{body}" to you too. ')
    if last_message:
        msg += f'Not so long ago you said "{last_message}" to me..'
    session['MESSAGE'] = body
    resp.message(msg)
    return str(resp)
```

之前的 `body` 存储在会话的 `MESSAGE` 键中，该会话会被延续。注意需要定义一个密钥才能使用会话数据。阅读此链接获取相关信息：https://flask.palletsprojects.com/en/1.1.x/quickstart/#sessions。有关 Twilio 中 cookie 处理的更多信息，请参阅此处：https://support.twilio.com/hc/en-us/articles/223136287-How-do-Twilio-cookies-work-。

> 要在 Heroku 中部署新版本，请将新的 `app.py` 文件提交到 Git，然后使用 `git push heroku master`。新版本将自动部署！

因为本食谱的主要目标是演示如何回复，所以没有详细描述 Heroku 和 Flask，但它们都有优秀的文档。Heroku 的完整文档可在 https://devcenter.heroku.com/categories/reference 找到，Flask 的文档在此处：http://flask.pocoo.org/docs/。

> 请记住，使用 Heroku 和 Flask 只是本食谱的便利选择。它们是很好且易于使用的工具，但只要你能暴露一个 URL 以便 Twilio 调用，就有很多替代方案。另外，请检查安全措施以确保对此端点的请求来自 Twilio：https://www.twilio.com/docs/usage/security#validating-requests。

Twilio 的完整文档可在此处找到：https://www.twilio.com/docs/。

### 另请参阅

- 本章前面的 *生成短信* 食谱，以掌握 Twilio 的基础知识以及如何接收消息。
- 接下来的 *创建 Telegram 机器人* 食谱，以进一步扩展你的知识并将其应用于类似元素。

### 创建 Telegram 机器人

Telegram Messenger 是一个即时通讯应用程序，对创建机器人有很好的支持。机器人是旨在产生自动对话的小型应用程序。机器人的巨大前景在于，作为机器，它们可以创建任何类型的对话，与人类的对话完全无法区分，并通过图灵测试，但这个目标相当雄心勃勃，在大多数情况下并不现实。

> 图灵测试由艾伦·图灵于 1951 年提出。两名参与者，一个人类和一个 AI 机器/软件程序，通过文本（如在即时通讯应用中）与一名人类评判员交流，由评判员决定哪一个是人类，哪一个不是。如果评判员只能正确猜测 50% 的时间，那么它就不能被轻易区分，因此 AI 通过了测试。这是衡量 AI 的最早尝试之一。

但对于更有限的方法，机器人可以非常有用，类似于电话系统，你需要按 2 查看账户，按 3 报告卡片丢失。在本食谱中，我们将看到如何生成一个简单的机器人，它将显示公司的优惠和活动。

### 准备工作

我们需要为 Telegram 创建一个新的机器人。这是通过一个名为 BotFather 的界面完成的，它是一个 Telegram 特殊频道，允许我们创建一个新机器人。你可以通过此链接访问该频道：https://telegram.me/botfather。通过你的 Telegram 账户访问它。

运行 /start 启动界面，然后使用 /newbot 创建一个新机器人。界面会要求你提供机器人的名称和一个唯一的用户名。

设置完成后，它会给你以下内容：

- 你的机器人 Telegram 频道：https://t.me/<yourusername>。
- 一个允许访问机器人的令牌。复制它，因为稍后会用到。

> 如果你丢失了现有令牌，可以生成一个新令牌。阅读 BotFather 的文档获取更多信息。

我们还需要安装 Python 模块 telepot，它封装了 Telegram 的 RESTful 接口：

```bash
$ echo "telepot==12.7" >> requirements.txt
$ pip install -r requirements.txt
```

从 GitHub 下载 telegram_bot.py 脚本，地址为 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter09/telegram_bot.py。

### 操作步骤...

1. 在 telegram_bot.py 脚本的第 6 行 TOKEN 常量中设置你生成的令牌：
   ```python
   TOKEN = '<YOUR TOKEN>'
   ```
2. 启动机器人：
   ```bash
   $ python telegram_bot.py
   ```
3. 在手机上使用 URL 打开 Telegram 频道并启动它。你可以使用 help、offers 和 events 命令：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_348_0.png)

图 9.12：通过短信发送的营销优惠

### 工作原理...

步骤 1 设置用于你特定频道的令牌。在步骤 2 中，我们在本地启动机器人。

让我们看看 `telegram_bot.py` 中的代码结构：

```python
# IMPORTS

# TOKEN

# Define the information to return per command
def get_help():
def get_offers():
def get_events():
COMMANDS = {
    'help': get_help,
    'offers': get_offers,
    'events': get_events,
}

class MarketingBot(telepot.helper.ChatHandler):
    ...

# Create and start the bot
```

`MarketingBot` 类创建了一个处理与 Telegram 通信的接口：

- 当频道启动时，将调用 `open` 方法
- 当收到消息时，将调用 `on_chat_message` 方法
- 如果一段时间没有响应，将调用 `on_idle`

在每种情况下，都使用 `self.sender.sendMessage` 方法向用户发送消息。大部分有趣的部分发生在 `on_chat_message` 中：

```python
def on_chat_message(self, msg):
    # If the data sent is not text, return an error
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text':
        self.sender.sendMessage("I don't understand you. "
                               "Please type 'help' for options")
        return
```

```python
# Make the commands case insensitive
command = msg['text'].lower()

if command not in COMMANDS:
    self.sender.sendMessage("I don't understand you. "
                           "Please type 'help' for options")
    return

message = COMMANDS[command]()
self.sender.sendMessage(message)
```

首先，它检查收到的消息是否是文本，如果不是，则返回错误消息。它分析收到的文本，如果它是定义的命令之一，则执行相应的函数以检索要返回的文本。

然后，它将消息发送回用户。

*步骤 3* 从用户的角度展示了这是如何工作的，用户正在与机器人交互。

### 更多内容...

你可以使用 `BotFather` 界面向你的 Telegram 频道添加更多信息、头像图片等。

为了简化我们的界面，我们可以创建一个自定义键盘来简化机器人。在定义命令后，在脚本的第 44 行左右创建它：

```python
from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
keys = [[KeyboardButton(text=text)] for text in COMMANDS]
KEYBOARD = ReplyKeyboardMarkup(keyboard=keys)
```

注意它创建了一个包含三行的键盘，每行包含一个命令。然后，在每个 `sendMessage` 调用中添加生成的 `KEYBOARD` 作为 `reply_markup`，如下所示：

```python
message = COMMANDS[command]()
self.sender.sendMessage(message, reply_markup=KEYBOARD)
```

### 处理通信渠道

这将用仅定义的按钮替代键盘，使界面非常直观：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_351_0.png)

图 9.13：带有三个按钮的短信

这些更改可以从 `telegram_bot_custom_keyboard.py` 文件下载，该文件可在 GitHub 上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter09/telegram_bot_custom_keyboard.py。

你可以创建其他类型的自定义界面，例如内联按钮，甚至是一个创建游戏的平台。更多信息请查看 Telegram API 文档。

与 Telegram 的交互也可以通过 webhook 完成，方式类似于 *接收短信* 配方中介绍的方法。请查看 `telepot` 文档中的 Flask 示例：https://github.com/nickoala/telepot/tree/master/examples/webhook。

> 设置 Telegram webhook 可以通过 `telepot` 完成。它要求你的服务位于 HTTPS 地址后面，以确保通信是私密的。对于简单服务来说，这可能有点棘手。你可以在 Telegram 文档中查看设置 webhook 的文档：https://core.telegram.org/bots/api#setwebhook。

完整的 Telegram 机器人 API 可以在这里找到：https://core.telegram.org/bots。

`telepot` 模块的文档可以在这里找到：https://telepot.readthedocs.io/en/latest/。

### 另请参阅

- 本章前面的 *发送短信* 配方，了解 Twilio 的基础知识以及如何接收短信。
- 本章前面的 *接收短信* 配方，了解如何使用 Twilio 接收短信。

# 10
## 为什么不自动化你的营销活动？

在本章中，我们将介绍与营销活动相关的以下配方：

- 检测机会
- 创建个性化优惠券代码
- 通过客户首选渠道发送通知
- 准备销售信息
- 生成销售报告

### 简介

在本章中，我们将创建一个营销活动，并逐步介绍我们将采取的每个自动化步骤。我们将在一个项目中使用本书中的概念和配方，该项目需要不同的步骤。

让我们看一个例子。对于我们的项目，我们公司希望开展一项营销活动，以提高参与度和销售额。这是一个非常值得称赞的努力。为此，我们可以将其分为几个任务：

1. 我们希望检测启动活动的最佳时机，因此我们将从不同来源接收有关关键词的通知，这些关键词将帮助我们做出明智的决定。
2. 该活动将包括生成要发送给潜在客户的个人代码。
3. 我们将通过用户首选的渠道（短信或电子邮件）直接将这些代码发送给用户。
4. 为了监控活动的结果，我们将汇编销售信息。
5. 最后，将生成一份销售报告。

本章将介绍这些任务中的每一个，并提供一个基于本书中介绍的模块和技术的综合解决方案。

> 虽然这些示例是根据实际需求创建的，但请记住，你的特定环境总会给你带来惊喜。不要害怕在你对系统了解更多时进行实验、调整和改进。迭代是创建优秀系统的方法。

让我们开始吧！

### 检测机会

在本章中，我们介绍了一个分为几个任务的营销活动：

1. 检测启动活动的最佳时机。
2. 生成要发送给潜在客户的个人代码。
3. 通过用户首选的渠道（短信或电子邮件）直接将代码发送给用户。
4. 收集活动的结果。
5. 生成一份包含结果分析的销售报告。

本配方涵盖任务 1。

我们的第一个任务是检测启动活动的最佳时机。为此，我们将监控一系列新闻网站，搜索包含我们定义的关键词的新闻。我们将任何匹配这些关键词的文章添加到报告中，并通过电子邮件发送。

### 准备工作

在本配方中，我们将使用本书中之前介绍的几个外部模块：`delorean`、`requests` 和 `BeautifulSoup`。如果它们尚未安装，我们需要将它们添加到我们的虚拟环境中：

```
$ echo "delorean==1.0.0" >> requirements.txt
$ echo "requests==2.23.0" >> requirements.txt
$ echo "beautifulsoup4==4.8.2" >> requirements.txt
$ echo "feedparser==5.2.1" >> requirements.txt
$ echo "jinja2==2.11.1" >> requirements.txt
$ echo "mistune==0.8.4" >> requirements.txt
$ pip install -r requirements.txt
```

你需要制作一个 RSS 源列表，我们将从中检索数据。

> 在我们的示例中，我们使用以下源，它们都是知名新闻网站的技术源：
> * http://feeds.reuters.com/reuters/technologyNews
> * https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml
> * https://feeds.bbci.co.uk/news/science_and_environment/rss.xml

从 GitHub 下载 `search_keywords.py` 脚本，该脚本将执行操作：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/search_keywords.py。你还需要下载电子邮件模板，可以在以下位置找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/email_styling.html 和 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/email_template.md。还有一个配置模板位于：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/config-opportunity.ini。

你需要一个有效的电子邮件服务用户名和密码。请查看 *第 9 章 处理通信渠道* 中的 *发送单个电子邮件* 配方。

### 操作步骤...

1. 创建一个 `config-opportunity.ini` 文件，其格式应如下所示。记得用你的详细信息填充它：

```
[SEARCH]
keywords = keyword, keyword
feeds = feed, feed

[EMAIL]
user = <YOUR EMAIL USERNAME>
password = <YOUR EMAIL PASSWORD>
from = <EMAIL ADDRESS FROM>
to = <EMAIL ADDRESS TO>
```

你可以使用 GitHub 上的模板（https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/config-opportunity.ini）来搜索关键词 cpu 和一些测试源。记得用你自己的账户详细信息填写 EMAIL 字段。

2. 调用脚本以生成电子邮件和报告：

```
$ python search_keywords.py config-opportunity.ini
```

3. 检查 `to` 电子邮件；你应该会收到一份包含找到的文章的报告。请记住，它会因每日新闻而异，但应该与此类似：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_357_0.png)

### 工作原理...

在步骤 1 中为脚本创建适当的配置后，步骤 2 通过调用 `search_keywords.py` 完成网页抓取和发送包含结果的电子邮件。

让我们看一下 `search_keywords.py` 脚本。代码结构分为以下几个部分：

- IMPORTS 部分使所有要使用的 Python 模块可用。它还定义了 EmailConfig 命名元组以帮助处理电子邮件参数。
- READ TEMPLATES 检索电子邮件模板并将其存储在 EMAIL_TEMPLATE 和 EMAIL_STYLING 常量中以供后续使用。
- `__main__` 块通过获取配置参数、解析配置文件，然后调用主函数来启动进程。
- 主函数组合了其他函数。首先，它检索文章，然后获取正文并发送电子邮件。
- `get_articles` 遍历所有源，并丢弃任何超过 1 周的文章。对于剩余的文章，它搜索关键词匹配。所有匹配的文章都会返回，包括链接信息和摘要。
- `compose_email_body` 使用电子邮件模板来编译电子邮件正文。请注意，模板是 Markdown 格式，它会被解析为 HTML，以在纯文本和 HTML 中提供相同的信息。
- `send_email` 获取正文信息以及所需信息（如用户名/密码），并发送电子邮件。

### 更多内容...

从不同来源检索信息的主要挑战之一是在所有情况下解析文本。一些源可能以不同的格式返回信息。

例如，在我们的示例中，你可以看到 Reuters 源摘要包含 HTML 信息，这些信息会在生成的电子邮件中呈现。如果你遇到这类问题，你可能需要进一步处理返回的数据，直到它变得一致。这可能高度依赖于对最终报告的预期质量。

> 在开发自动化任务时，特别是在处理多个输入源时，预计会花费大量时间清理输入以使其一致。然而，要找到一个平衡点，并记住最终接收者。例如，如果电子邮件是发送给你自己或朋友的，你可以比发送给重要客户时更宽松一些。

### 为何不自动化你的营销活动？

另一种可能性是增加匹配的复杂性。在本配方中，检查是通过简单的 `in` 运算符完成的。请记住，*第1章，开启自动化之旅* 中的所有技术都可以供你使用，包括所有正则表达式的功能。

> 此脚本可以通过 cron 作业实现自动化，如 *第2章，轻松实现任务自动化* 中所述。尝试每周运行一次！

### 另请参阅

- *第1章，开启自动化之旅* 中的 *添加命令行参数* 配方，了解命令行参数的详细信息。
- *第1章，开启自动化之旅* 中的 *介绍正则表达式* 配方，了解如何使用正则表达式。
- *第2章，轻松实现任务自动化* 中的 *准备任务* 配方，检查良好自动化任务的结构。
- *第2章，轻松实现任务自动化* 中的 *设置 cron 作业* 配方，了解如何自动重复执行任务。
- *第3章，构建你的第一个网页抓取应用* 中的 *解析 HTML* 配方，了解如何解析返回的 HTML。
- *第3章，构建你的第一个网页抓取应用* 中的 *网络爬取* 配方，了解在检索网络信息时如何跟踪链接。
- *第3章，构建你的第一个网页抓取应用* 中的 *订阅源* 配方，了解处理 RSS 源的基础知识。
- *第9章，处理通信渠道* 中的 *发送个人电子邮件* 配方，了解如何使用 Python 发送电子邮件。

### 创建个性化优惠券代码

在本章中，我们将介绍一个分为多个任务的营销活动：

1. 检测启动活动的最佳时机
2. **生成将发送给潜在客户的个人代码**
3. 通过用户偏好的渠道（短信或电子邮件）直接将代码发送给用户
4. 收集活动结果
5. 生成包含结果分析的销售报告

本配方展示了活动的 *任务2*。

在检测到机会后，我们决定为所有客户生成一个活动。为了定向推广并避免重复，我们将生成100万个唯一优惠券，分为三批：

- 一半的代码将被打印并在营销活动中分发。
- 300,000个代码将被保留，以便在活动达到某些目标后使用。
- 剩余的200,000个将通过短信和电子邮件定向发送给客户。

这些优惠券可以在在线系统中兑换。我们的任务是生成合适的代码，这些代码应满足以下要求：

- 代码需要是唯一的。
- 代码需要可打印且易于阅读，因为有些客户可能会通过电话口述它们。
- 应有一种快速的方法来丢弃伪造的代码。这将避免使用随机代码进行攻击，从而可能导致系统过载。
- 代码应以 CSV 格式呈现以便打印。

### 准备工作

从 GitHub 下载 `create_personalised_coupons.py` 脚本，该脚本将在 CSV 文件中生成优惠券，地址为 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/create_personalised_coupons.py。

### 如何操作...

1. 调用 `create_personalised_coupons.py` 脚本。根据你的处理器速度，运行需要一两分钟。它将在屏幕上显示生成的代码：

```
$ python create_personalised_coupons.py
Code: HWLF-P9J9E-U3
Code: EAUE-FRCWR-WM
Code: PMW7-P39MP-KT
...
```

2. 检查它是否创建了三个 CSV 文件：`codes_batch_1.csv`、`codes_batch_2.csv` 和 `codes_batch_3.csv`，每个文件包含适当数量的代码：

```
$ wc -l codes_batch_*.csv
500000 codes_batch_1.csv
300000 codes_batch_2.csv
200000 codes_batch_3.csv
1000000 total
```

3. 检查每个批次文件是否包含唯一的代码。你的代码将是唯一的，并且与这里显示的代码不同：

```
$ head codes_batch_2.csv
9J9F-M33YH-YR
7WLP-LTJUP-PV
WHFU-THW7R-T9
...
```

### 工作原理...

步骤1调用生成所有代码的脚本，步骤2检查结果是否正确。步骤3显示了代码的存储格式。让我们分析一下 `create_personalised_coupons.py` 脚本。

总结来说，它具有以下结构：

```
# IMPORTS
...
# FUNCTIONS
def random_code(digits)
    ...
def checksum(code1, code2)
    ...
def check_code(code)
    ...
def generate_code()
    ...
# SET UP TASK
...
# GENERATE CODES
...
# CREATE AND SAVE BATCHES
...
```

不同的函数协同工作以创建一个代码。`random_code` 从 `CHARACTERS` 中生成随机字母和数字的组合。该字符串包含所有可供选择的有效字符。

> 字符的选择被定义为易于打印且彼此不易混淆的符号。例如，根据字体的不同，字母 O 和数字 0，或者数字 1 和字母 I 可能难以区分。这可能取决于字体和印刷工艺的具体情况，因此如有必要，请检查印刷测试以调整字符。但避免在印刷形式中使用所有字母和数字，因为这可能会引起混淆。如有必要，请增加代码的长度；例如，如果需要更多代码，或者可识别符号的数量较少。

`checksum` 函数基于两个代码生成一个额外的数字，该数字是从两个代码派生出来的。这个过程被称为 **哈希**，它是计算中一个众所周知的过程，尤其是在密码学中。

> 哈希的基本功能是从一个输入产生一个更小且不可逆的输出，这意味着除非知道输入，否则很难猜测。哈希在计算中有很多常见的应用，通常在底层。例如，Python 字典广泛使用哈希。

在我们的配方中，我们将使用 SHA256，这是 Python `hashlib` 模块中包含的一个著名的快速哈希算法：

```
def checksum(code1, code2):
    m = hashlib.sha256()
    m.update(code1.encode())
    m.update(code2.encode())
    checksum = int(m.hexdigest()[:2], base=16)
    digit = CHARACTERS[checksum % len(CHARACTERS)]
    return digit
```

两个代码被连接作为输入，生成的哈希字符串的两个十六进制数字用于从 `CHARACTERS` 中选择相应的字符。

哈希数字被转换为基数为 10 的数字（因为它们是基数为 16 的），我们应用 `modulo` 运算符来获取一个可用的字符。

这个校验和的目标是能够快速检查一个代码是否看起来正确，并丢弃可能的垃圾信息。我们可以对一个代码再次执行该操作，以查看校验和是否相同。请注意，这不是一个加密哈希，因为在操作的任何阶段都不需要密钥。鉴于这个特定的用例，这种（低）安全级别可能对我们的目的来说已经足够了。

> 密码学是一个更大的主题，确保安全性很强可能很困难。密码学中涉及哈希的主要策略可能是只存储哈希值，以避免以可读格式存储密码。你可以在这里阅读关于该技术的快速介绍：https://crackstation.net/hashing-security.htm。

`generate_code` 函数然后生成一个随机代码，由四位数字、五位数字和两位校验和数字组成，用破折号分隔。第一个校验和数字是按从右到左的顺序使用前九位数字生成的（使用四位字符块作为 `code1`，五位字符块作为 `code2`）。第二个校验和数字是通过反转它们生成的（五位字符块作为 `code1`，四位字符块作为 `code2`）。

`check_code` 函数反转该过程，如果代码正确则返回 `True`，否则返回 `False`。

有了基本元素，脚本首先定义所需的批次：500,000、300,000 和 200,000。

所有代码都在同一个池中生成，称为 `codes`。这是为了避免批次之间的重复。请注意，由于过程的随机性，我们不能排除生成重复代码的可能性，尽管这种可能性很小。我们被允许最多重试三次以避免生成重复代码。代码存储在一个集合累加器中，以保证其唯一性并加快检查代码是否已存在的过程。

> 集合是 Python 在底层使用哈希的另一个地方，因此它对要添加的元素进行哈希，并将其与已存在元素的哈希进行比较。这使得在集合中的检查成为一个非常快速的操作。

为确保流程正确，每个代码都会被验证，并在生成代码时打印以显示进度。这也能让我们检查一切是否按预期工作。

最后，代码被分成适当数量的批次，每个批次保存在一个单独的`.csv`文件中。代码使用`.pop()`从`codes`中逐个移除，直到`batch`达到合适的大小：

```
batch = [(codes.pop(),) for _ in range(batch_size)]
```

注意上一行如何创建一个包含适当行数的批次，每行只有一个元素。每行仍然是一个列表，这对于CSV文件来说是正确的。

然后，创建一个文件，并使用`csv.writer`将代码作为行存储。

作为最后的测试，验证剩余的代码以确保它们为空。

### 还有更多...

在这个食谱中，流程中使用了一种直接的方法。这与*第2章，轻松自动化任务*中的*准备任务运行*食谱中提出的原则相反。请注意，与那里提出的任务相比，此脚本旨在运行一次以生成代码，仅此而已。它还使用定义的常量（如`BATCHES`）进行配置，而不是命令行参数或配置文件。

鉴于这是一个独特的任务，设计为运行一次，花时间将其构建为可重用组件可能不是我们时间的最佳利用方式。

> 过度工程化绝对是可能的，在务实设计和更具前瞻性的方法之间做出选择可能并不容易。要现实地考虑维护成本，并尝试找到自己的平衡点。

同样，这个食谱中关于校验和的设计旨在提供一种最小化的方式来检查代码是完全捏造的还是看起来合法的。鉴于代码将针对系统进行检查，这似乎是一种合理的方法，但请留意您的特定用例。

我们的代码空间由`22个字符 ** 9位数字 = 1,207,269,217,792个可能的代码`组成，这意味着猜测生成的百万个代码之一的概率非常小。产生相同代码两次的可能性也不大，但尽管如此，我们通过最多三次重试来保护我们的代码免受这种情况的影响。

为什么不自动化您的营销活动？

这类检查，以及检查每个代码是否已验证以及我们最终没有剩余代码，在开发此类脚本时非常有用。它确保我们朝着正确的方向前进，并且事情按计划进行。只需注意，在某些条件下`assert`可能不会被执行。

> 如Python文档所述，如果Python代码被优化（使用`-O`命令运行），则`assert`命令将被忽略。请参阅此处的文档：https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement。使用`-O`参数的情况很少见，但如果确实如此，可能会令人困惑。避免严重依赖`assert`。

学习密码学的基础知识并不像您想象的那么困难。有少数基本方案是众所周知的，并且可以轻松学习。一篇很好的介绍文章是https://thebestvpn.com/cryptography/。Python也有大量的加密函数；请参阅https://docs.python.org/3/library/crypto.html处的文档。最好的方法是找到一本好书，并知道虽然这是一个真正掌握起来很困难的主题，但它绝对是可接近的。

### 另请参阅

- *第1章，开始我们的自动化之旅*中的*介绍正则表达式*食谱，以学习如何使用正则表达式。
- *第4章，搜索和读取本地文件*中的*读取CSV文件*食谱，以学习如何处理CSV文件。

### 通过客户首选渠道向其发送通知

在本章中，我们介绍了一个分为几个任务的营销活动：

1. 检测启动活动的最佳时机
2. 生成要发送给潜在客户的个人代码
3. **通过用户首选的渠道（短信或电子邮件）直接将代码发送给用户**
4. 收集活动结果
5. 生成包含结果分析的销售报告

此食谱展示了活动的*任务3*。

一旦我们的代码已创建用于直接营销，我们需要将它们分发给我们的客户。

对于这个食谱，从包含所有客户信息及其首选联系方式的CSV文件输入开始，我们将用之前生成的代码填充文件，然后通过适当的方法发送通知。这将包括促销代码。

### 准备工作

在这个食谱中，我们将使用几个已经介绍过的模块：`delorean`、`requests`和`twilio`。如果它们尚未存在于我们的虚拟环境中，我们需要将它们添加进去：

```
$ echo "delorean==1.0.0" >> requirements.txt
$ echo "requests==2.23.0" >> requirements.txt
$ echo "twilio==6.37.0" >> requirements.txt
$ pip install -r requirements.txt
```

我们需要定义一个`config-channel.ini`文件，其中包含我们用于服务的凭据，Mailgun和Twilio。此文件的模板可以在GitHub上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/config-channel.ini。

> 有关如何获取凭据的信息，请参阅*第9章，处理通信渠道*中的*通过电子邮件发送通知*和*生成短信*食谱。

该文件具有以下格式：

```
[MAILGUN]
KEY = <YOUR KEY>
DOMAIN = <YOUR DOMAIN>
FROM = <YOUR FROM EMAIL>
[TWILIO]
ACCOUNT_SID = <YOUR SID>
AUTH_TOKEN = <YOUR TOKEN>
FROM = <FROM TWILIO PHONE NUMBER>
```

为什么不自动化您的营销活动？

为了描述所有要定位的联系人，我们需要生成一个CSV文件`notifications.csv`，格式如下：

| 姓名 | 联系方式 | 目标 | 状态 | 代码 | 时间戳 |
|---|---|---|---|---|---|
| John Smith | PHONE | +1-555-12345678 | NOT-SENT | | |
| Paul Smith | EMAIL | paul.smith@test.com | NOT-SENT | | |

图10.2：`notifications.csv`的格式

请注意，`Code`列为空，并且所有状态应为`NOT-SENT`或为空。

> 如果您在Twilio和Mailgun中使用测试账户，请注意其限制。例如，Twilio只允许您向已验证的电话号码发送消息。您可以创建一个仅包含两个或三个联系人的小型CSV文件来测试脚本。

要使用的优惠券代码应准备好在CSV文件中。您可以使用`create_personalised_coupons.py`脚本生成多个批次，该脚本可在GitHub上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/create_personalised_coupons.py。

从GitHub下载要使用的脚本`send_notifications.py`：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/send_notifications.py。

### 如何做...

1. 运行`send_notifications.py`以查看其选项和用法：

```
$ python send_notifications.py --help
usage: send_notifications.py [-h] [-c CODES] [--config CONFIG_FILE] notif_file

positional arguments:
  notif_file            notifications file

optional arguments:
-h, --help show this help message and exit
-c CODES, --codes CODES
Optional file with codes. If present, the file will be populated with codes. No codes will be sent
--config CONFIG_FILE config file (default config.ini)
```

2. 将代码添加到notifications.csv文件：

```
$ python send_notifications.py --config config-channel.ini notifications.csv -c codes_batch_3.csv
$ head notifications.csv
Name,Contact Method,Target,Status,Code,Timestamp
John Smith,PHONE,+1-555-12345678,NOT-SENT,CFXK-U37JN-TM,
Paul Smith,EMAIL,paul.smith@test.com,NOT-SENT,HJGX-M97WE-9Y,
...
```

3. 最后，发送通知：

```
$ python send_notifications.py --config config-channel.ini notifications.csv
$ head notifications.csv
Name,Contact Method,Target,Status,Code,Timestamp
John Smith,PHONE,+1-555-12345678,SENT,CFXK-U37JN-TM,2018-08-25T13:08:15.908986+00:00
Paul Smith,EMAIL,paul.smith@test.com,SENT,HJGX-M97WE-9Y,2018-08-25T13:08:16.980951+00:00
...
```

4. 检查电子邮件和手机以验证消息是否已收到。

### 工作原理...

步骤1展示了脚本的使用。总体思路是多次调用它；第一次用代码填充它，第二次发送消息。如果出现错误，可以再次执行脚本，并且只会重试之前未发送的消息。

### 为何不将你的营销活动自动化？

`notifications.csv` 文件包含了将在*步骤 2*中注入的代码。这些代码最终在*步骤 3*中发送。

让我们分析 `send_notifications.py` 的代码。这里只展示最相关的部分：

```python
# IMPORTS

def send_phone_notification(...):
def send_email_notification(...):
def send_notification(...):

def save_file(...):
def main(...):

if __name__ == '__main__':
    # Parse arguments and prepare configuration
    ...
```

主函数逐行遍历文件，并分析每种情况下的操作。如果条目状态为 SENT，则跳过它。如果条目没有代码，则尝试填充它。如果尝试发送，它会附加时间戳以记录发送或尝试发送的时间。

对于每个条目，整个文件会再次保存到名为 `save_file` 的文件中。注意文件光标是如何定位在文件开头的。文件被写入然后刷新到磁盘：

```python
def save_file(notif_file, data):
    """
    Overwrite the file with the new information
    """

    # Start at the start of the file
    notif_file.seek(0)

    header = data[0].keys()
    writer = csv.DictWriter(notif_file, fieldnames=header)
    writer.writeheader()
    writer.writerows(data)

    # Be sure to write to disk
    notif_file.flush()
```

这会在每次条目操作时覆盖文件，而我们无需关闭并重新打开文件。

> 为何要为每个条目写入整个文件？这是一种存储每个操作的简单方法，并允许你重试发送过程。例如，如果某个条目产生意外错误或超时，甚至发生普遍故障，所有进度和之前的代码都将被标记为 SENT，不会再次发送。这意味着可以根据需要重试操作。对于大量条目，这是确保过程中途出现问题不会导致我们向客户重复发送消息的好方法。

对于海量行数，我们可能面临保存文件时出现问题的风险。这可能导致文件在写入过程中因意外错误而损坏，或保存时间过长。如果是这种情况，请将文件拆分为可以独立处理的独立批次。对于非常大的处理过程，可能需要一个保证数据不会损坏的系统，例如使用数据库。

对于每个要发送的代码，`send_notification` 函数决定调用 `send_phone_notification` 或 `send_email_notification`。在两种情况下，它都会附加当前时间。

两个 `send` 函数在无法发送消息时都会返回错误。这允许你在生成的 `notifications.csv` 文件中标记它，并稍后重试。

> `notifications.csv` 文件也可以手动更改。例如，想象一下电子邮件中有一个拼写错误，这就是错误的原因。可以更改它并重试。

`send_email_notification` 基于 Mailgun 接口发送消息。更多信息，请参阅*第 9 章，处理通信渠道*中的*通过电子邮件发送通知*食谱。请注意，此处发送的电子邮件仅包含文本。

`send_phone_notification` 基于 Twilio 接口发送消息。更多信息，请参阅*第 9 章，处理通信渠道*中的*生成短信*食谱。

### 还有更多...

时间戳特意以 ISO 格式写入，因为它是一种可解析的格式。这意味着我们可以轻松地获取回一个合适的对象，如下所示：

```python
>>> import datetime
>>> timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
>>> timestamp
'2018-08-25T14:13:53.772815+00:00'
>>> datetime.datetime.fromisoformat(timestamp)
datetime.datetime(2018, 9, 11, 21, 5, 41, 979567, tzinfo=datetime.timezone.utc)
```

这使你可以轻松地来回解析时间戳。

> ISO 8601 时间格式在大多数编程语言中得到良好支持，并精确定义了时间，因为它包含了时区。如果你可以使用它，它是记录时间的绝佳选择。

`send_notification` 中用于路由通知的策略很有趣：

```python
# Route each of the notifications
METHOD = {
    'PHONE': send_phone_notification,
    'EMAIL': send_email_notification,
}
try:
    method = METHOD[entry['Contact Method']]
    result = method(entry, config)
except KeyError:
    result = 'INVALID_METHOD'
```

`METHOD` 字典将每个可能的 `Contact Method` 分配给一个具有相同定义的函数，该函数同时接受一个条目和一个配置。

然后，根据具体方法，从字典中检索并调用该函数。注意 `method` 变量包含要调用的正确函数。

> 这与其他编程语言中可用的 `switch` 操作类似。也可以通过 `if...else` 块实现这一点。对于像这段代码这样的简单情况，字典方法使代码非常易读。

`invalid_method` 函数用作默认值。如果 `Contact Method` 不是可用的方法之一（`PHONE` 或 `EMAIL`），将引发并捕获 `KeyError`，结果将被定义为 `INVALID_METHOD`。

### 另请参阅

- *第 9 章，处理通信渠道*中的*通过电子邮件发送通知*食谱，了解如何通过 Mailgun 发送电子邮件。
- *第 9 章，处理通信渠道*中的*生成短信*食谱，了解如何使用 Twilio 发送短信。

### 准备销售信息

在本章中，我们介绍了一个分为几个任务的营销活动：

- 检测启动活动的最佳时机
- 生成要发送给潜在客户的单独代码
- 通过用户首选的渠道（短信或电子邮件）直接将代码发送给用户
- **收集活动结果**
- 生成包含结果分析的销售报告

本食谱展示了活动的*任务 4*。

在向用户发送信息后，我们需要从商店收集销售日志，以监控活动进展和影响程度。

销售日志以每个关联商店的单独文件形式报告，因此在本食谱中，我们将看到如何将所有信息聚合到电子表格中，以便将信息作为一个整体进行处理。

### 准备工作

对于本食谱，我们需要安装以下模块：

```bash
$ echo "openpyxl==3.0.3" >> requirements.txt
$ echo "parse==1.15.0" >> requirements.txt
$ echo "delorean==1.0.0" >> requirements.txt
$ pip install -r requirements.txt
```

我们可以从 GitHub 获取本食谱的测试结构和测试日志，地址为 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter10/sales。请下载完整的 sales 目录，其中包含大量测试日志。为了显示结构，我们将使用 tree 命令（http://mama.indstate.edu/users/ice/tree/），该命令在 Linux 中默认安装，在 macOS 中可以使用 brew 安装（https://brew.sh/）。你也可以使用图形工具检查目录。

我们还需要 sale_log.py 模块和 parse_sales_log.py 脚本，可在 GitHub 上获取，地址为 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/parse_sales_log.py。

### 操作步骤...

1. 检查 sales 目录的结构。每个子目录代表一个已提交其指定期间销售日志的商店：

```bash
$ tree sales
sales
├── 345
│   └── logs.txt
├── 438
│   ├── logs_1.txt
│   ├── logs_2.txt
│   ├── logs_3.txt
│   └── logs_4.txt
└── 656
    └── logs.txt
```

2. 检查日志文件：

```bash
$ head sales/438/logs_1.txt
[2018-08-27 21:05:55+00:00] - SALE - PRODUCT: 12346 - PRICE: $02.99 - NAME: Single item - DISCOUNT: 0%
[2018-08-27 22:05:55+00:00] - SALE - PRODUCT: 12345 - PRICE: $07.99 - NAME: Family pack - DISCOUNT: 20%
...
```

3. 调用 `parse_sales_log.py` 脚本生成报告：

```bash
$ python parse_sales_log.py sales -o report.xlsx
```

4. 检查生成的 Excel 结果 `report.xlsx`：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_374_0.png)

图 10.3：report.xlsx 截图

### 为何不将你的营销活动自动化？

### 工作原理...

步骤1和2展示了数据的结构。步骤3调用 `parse_sales_log.py` 来读取所有日志文件并进行解析，然后将它们存储到一个Excel电子表格中。电子表格的内容在步骤4中显示。

让我们看看 `parse_sales_log.py` 的结构：

```python
# IMPORTS
from sale_log import SaleLog

def get_logs_from_file(shop, log_filename):
    with open(log_filename) as logfile:
        logs = [SaleLog.parse(shop=shop, text_log=log)
                for log in logfile]
    return logs

def main(log_dir, output_filename):
    logs = []
    for dirpath, dirnames, filenames in os.walk(log_dir):
        for filename in filenames:
            # The shop is the last directory
            shop = os.path.basename(dirpath)
            fullpath = os.path.join(dirpath, filename)
            logs.extend(get_logs_from_file(shop, fullpath))

    # Create and save the Excel sheet
    xlsfile = openpyxl.Workbook()
    sheet = xlsfile['Sheet']
    sheet.append(SaleLog.row_header())
    for log in logs:
        sheet.append(log.row())
    xlsfile.save(output_filename)

if __name__ == '__main__':
    # PARSE COMMAND LINE ARGUMENTS AND CALL main()
```

命令行参数在第1章《让我们开始自动化之旅》中有解释。请注意，导入部分包含了 `SaleLog`。

主函数遍历整个目录，并通过 `os.walk` 获取所有文件。你可以在第2章《轻松实现任务自动化》中了解更多关于 `os.walk` 的信息。然后，每个文件都被传递给 `get_logs_from_file` 来解析其日志，并将它们添加到全局 `logs` 列表中。

请注意，特定的商店存储在最后一个子目录中，因此使用 `os.path.basename` 来提取它。

一旦日志列表完成，就使用 `openpyxl` 模块创建一个新的Excel工作表。`SaleLog` 模块有一个 `.row_header` 方法来添加第一行，然后所有日志都使用 `.row` 转换为行格式。最后，保存文件。

为了解析日志，我们创建了一个名为 `sale_log.py` 的模块，它抽象了解析和处理行的过程。大部分内容都很直接，并正确地构建了每个不同的参数，但 `parse` 方法需要特别注意：

```python
@classmethod
def parse(cls, shop, text_log):
    '''
    Parse from a text log with the format
    ...
    to a SaleLog object
    '''
    def price(string):
        return Decimal(string)

    def isodate(string):
        return delorean.parse(string)

    FORMAT = ('[{timestamp:isodate}] - SALE - PRODUCT: '
              '{product:d} '
              '- PRICE: ${price:price} - NAME: {name:D} '
              '- DISCOUNT: {discount:d}%')

    formats = {'price': price, 'isodate': isodate}
    result = parse.parse(FORMAT, text_log, formats)

    return cls(timestamp=result['timestamp'],
               product_id=result['product'],
               price=result['price'],
               name=result['name'],
               discount=result['discount'],
               shop=shop)
```

`sale_log.py` 是一个 `classmethod`，这意味着可以通过调用 `SaleLog.parse` 来使用它。它返回该类的一个新元素。

> 类方法被调用时，第一个参数存储的是类，而不是通常存储在self中的对象。惯例是使用 `cls` 来表示它。在末尾调用 `cls(...)` 等同于调用 `SaleFormat(...)`，因此它调用了 `__init__` 方法。

该方法使用 `parse` 模块从模板中检索值。请注意，有两个元素 `timestamp` 和 `price` 具有自定义解析。`delorean` 模块帮助我们解析日期，而价格最好描述为 `Decimal` 以保持适当的精度。自定义过滤器在 `formats` 参数中应用。

### 还有更多...

`Decimal` 类型在Python文档中有详细描述，地址为：https://docs.python.org/3/library/decimal.html。

完整的 `openpyxl` 文档可以在这里找到：https://openpyxl.readthedocs.io/en/stable/。另外，请查看第6章《电子表格的乐趣》，获取更多关于如何使用该模块的示例。

完整的 `parse` 文档可以在这里找到：https://github.com/richardj0n3s/parse。第1章《让我们开始自动化之旅》也更详细地描述了这个模块。

### 另请参阅

-   第1章《让我们开始自动化之旅》中的“使用第三方工具 – parse”配方，以了解更多关于 `parse` 模块的信息。
-   第4章《搜索和读取本地文件》中的“爬取和搜索目录”配方，以了解如何遍历和查找目录中的所有文件。
-   第4章《搜索和读取本地文件》中的“读取文本文件”配方，以了解如何打开文本文件。
-   第6章《电子表格的乐趣》中的“更新Excel电子表格”配方，以了解如何使用Python写入Excel电子表格。

### 生成销售报告

在本章中，我们介绍一个分为几个任务的营销活动：

1.  检测启动活动的最佳时机
2.  生成要发送给潜在客户的个人代码
3.  通过用户偏好的渠道（短信或电子邮件）直接将代码发送给用户
4.  收集活动结果
5.  生成包含结果分析的销售报告

本配方展示了活动的*任务5*。

作为最后一步，所有关于每笔销售的信息都被汇总并显示在销售报告中。

在本配方中，我们将看到如何从电子表格读取、创建PDF并生成图表，以自动生成一份综合报告，从而分析我们活动的表现。

### 准备工作

在本配方中，我们的虚拟环境需要以下模块：

```bash
$ echo "openpyxl==3.0.3" >> requirements.txt
$ echo "fpdf==1.7.2" >> requirements.txt
$ echo "delorean==1.0.0" >> requirements.txt
$ echo "PyPDF2==1.26.0" >> requirements.txt
$ echo "matplotlib==3.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

我们需要 `sale_log.py` 模块，它可以在GitHub上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/sale_log.py。

> 输入电子表格是在上一个配方“准备销售信息”中生成的。有关更多信息，请参阅该配方。

你可以从GitHub下载生成输入电子表格的脚本 `parse_sales_log.py`：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/parse_sales_log.py。

从GitHub下载原始日志文件：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter10/sales。请下载完整的 `sales` 目录。

从GitHub下载 `generate_sales_report.py` 脚本：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter10/generate_sales_report.py。

### 操作步骤...

1.  调用 `parse_sales_log.py` 脚本生成输入文件：

    ```bash
    $ python parse_sales_log.py sales -o report.xlsx
    ```

2.  检查输入文件和 `generate_sales_report.py` 的用法：

    ```bash
    $ ls report.xlsx
    report.xlsx
    $ python generate_sales_report.py --help
    usage: generate_sales_report.py [-h] input_file output_file

    positional arguments:
      input_file
      output_file

    optional arguments:
      -h, --help show this help message and exit
    ```

3.  使用输入文件和输出文件调用 `generate_sales_report.py` 脚本：

    ```bash
    $ python generate_sales_report.py report.xlsx output.pdf
    ```

4.  检查输出文件 `output.pdf`。它将包含三页，第一页是简要摘要，第二页和第三页是显示按天和按商店划分的销售情况的图表。

### 工作原理

步骤1展示了如何使用脚本，步骤2在输入文件上调用它。让我们看看 `generate_sales_report.py` 脚本的基本结构：

```python
# IMPORTS

def generate_summary(logs):

def aggregate_by_day(logs):
def aggregate_by_shop(logs):

def graph(...):

def create_summary_brief(...):

def main(input_file, output_file):
    # open and read input file
    # Generate each of the pages calling the other calls
    # Group all the pdfs into a single file
    # Write the resulting PDF

if __name__ == '__main__':
    # Compile the input and output files from the command line
    # call main
```

有两个关键元素——以不同方式（按商店和按天）聚合日志，以及在每种情况下生成摘要。摘要由 `generate_summary` 生成，它从日志列表生成一个包含其聚合信息的字典。日志的聚合在 `aggregate_by` 函数中以不同的风格完成。

> `generate_summary` 生成一个包含聚合信息的字典，包括开始和结束时间、所有日志的总收入、总单位数、平均折扣，以及按产品划分的相同数据明细。

从末尾开始理解脚本会更好。主函数连接所有不同的操作。读取每个日志并将其转换为原生的 `SaleLog` 对象。

### 为何不将你的营销活动自动化？

然后，它会将每个页面生成为中间 PDF 文件：

- 一份由 `create_summary_brief` 生成的简报，提供所有数据的总体摘要。
- 日志按 `aggregate_by_day` 聚合。生成摘要，并生成图表。
- 日志按 `aggregate_by_shop` 聚合。生成摘要，并生成图表。

所有中间 PDF 页面使用 `PyPDF2` 合并为一个文件。最后，删除中间页面。

`aggregate_by_day` 和 `aggregate_by_shop` 都返回一个包含每个元素摘要的列表。在 `aggregate_by_day` 中，我们使用 `.end_of_day` 来区分一天与另一天，从而检测一天的结束。

`graph` 函数执行以下操作：

1. 准备所有要显示的数据。这包括每个标签（日期或店铺）的单位数量和总收入。
2. 创建一个顶部图表，显示按产品划分的总收入，以堆叠条形图呈现。为了实现这一点，在计算总收入的同时，也计算了基线（下一个堆叠的位置）。
3. 它将图表的底部分成与产品数量一样多的子图，并在每个子图上显示每个标签（日期或店铺）的销售单位数量。

> 为了更好的显示效果，图表被定义为 A4 纸张的大小。它还允许我们使用 `skip_labels`，在第二个图表的 X 轴上每隔 X 个标签打印一个，以避免重叠。这在显示日期时很有用，设置为每周只显示一个标签。

生成的图表保存到文件中。

`create_summary_brief` 使用 `fpdf` 模块保存一个包含总体摘要信息的文本 PDF 页面。

> `create_summary_brief` 中的模板和信息被故意保持简单，以避免使本食谱复杂化，但可以通过更好的描述性文本和格式使其更复杂。有关如何使用 `fpdf` 的更多详细信息，请参阅 *第 5 章，生成精彩报告*。

如前所述，主函数将所有 PDF 页面分组并合并为一个文档，之后删除中间页面：

> 报告生成于 2018-08-29T23:45:21.661291+00:00
> 涵盖数据从 8 月 27 日到 10 月 8 日
>
> 摘要
> -------
> 总收入：$ 14225.0
> 总单位：3000 单位
> 平均折扣：2%
>
> 图 10.4：销售摘要

第二页显示按天划分的销售图表：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_382_0.png)

第三页按店铺划分销售：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_383_0.png)

### 更多内容...

本食谱中包含的报告可以扩展。例如，可以在每页计算平均折扣并显示为折线：

```python
# 生成包含平均折扣的数据系列
discount = [summary['average_discount'] for _, summary in full_summary]
....
# 打印图例
# 在第二个轴上绘制折扣
plt.twinx()
plt.plot(pos, discount,'o-', color='green')
plt.ylabel('平均折扣')
```

不过，请注意不要在单个图表中放置过多信息。这可能会降低可读性。在这种情况下，使用另一个图表可能是更好的显示方式。

> 请注意在创建第二个轴之前打印图例，否则它将只显示第二个轴上的信息。

图表的大小和方向可以决定是使用更多还是更少的标签，以确保它们清晰易读。这在使用 `skip_labels` 避免杂乱时得到了演示。请关注生成的图形，并尝试通过更改大小或在某些情况下限制标签来适应可能出现的问题。

> 例如，一个可能的限制是产品不超过三个，因为在我们的图表中第二行打印四个图表可能会使文本难以辨认。请随意实验并检查代码的限制。

完整的 `matplotlib` 文档可在 https://matplotlib.org/ 找到。`delorean` 文档可在此处找到：https://delorean.readthedocs.io/en/latest/。

`openpyxl` 的所有文档可在 https://openpyxl.readthedocs.io/en/stable/ 获取。PDF 操作模块的完整文档可在 PyPDF2 的 https://pythonhosted.org/PyPDF2/ 和 pyfpdf 的 https://pyfpdf.readthedocs.io/en/latest/ 找到。

> 本食谱利用了 *第 5 章，生成精彩报告* 中关于 PDF 创建和操作、*第 6 章，电子表格的乐趣* 中关于电子表格读取以及 *第 8 章，开发惊艳图表* 中关于图表创建的不同概念和技术。请查看它们以了解更多信息。

### 另请参阅

- *第 5 章，生成精彩报告* 中的 *聚合 PDF 报告* 食谱，了解如何合并多个 PDF 文件。
- *第 6 章，电子表格的乐趣* 中的 *读取 Excel 电子表格* 食谱，了解如何从 Excel 电子表格中获取信息。
- *第 8 章，开发惊艳图表* 中的 *绘制堆叠条形图* 食谱，了解有关如何绘制堆叠条形图的更多信息。
- *第 8 章，开发惊艳图表* 中的 *显示多条线* 食谱，了解如何获取包含多条信息线的单个图表。
- *第 8 章，开发惊艳图表* 中的 *添加图例和注释* 食谱，了解有关向图表添加图例和额外注释的更多信息。
- *第 8 章，开发惊艳图表* 中的 *组合图表* 食谱，了解如何将不同的图表组合成单个图像。
- *第 8 章，开发惊艳图表* 中的 *保存图表* 食谱，了解有关如何以不同格式存储图表的更多信息。

# 11 用于自动化的机器学习

在本章中，我们将涵盖以下食谱：

- 使用 Google Cloud Vision AI 分析图像
- 使用 Google Cloud Vision AI 从图像中提取文本
- 使用 Google Cloud Natural Language 分析文本
- 创建你自己的自定义机器学习模型来分类文本

### 简介

机器学习是一种允许系统在没有明确描述这些模式的情况下被训练来识别模式的技术。机器学习的基础是创建和训练一个模型，一个用训练数据准备好的系统，然后可以自动处理与训练数据相似的新数据。模型从训练数据中学习。

例如，检测电子邮件中垃圾邮件的传统方法是检查可疑的单词或句子。而使用机器学习技术，则是向模型提供垃圾邮件和非垃圾邮件消息的列表，系统会自行调整。它从数据中学习。然后可以将新电子邮件提供给模型以检测它们是否是垃圾邮件。

这种方法也可以用于图像，因此，与其尝试创建一个复杂的形状检测算法来识别狗，不如使用大量的狗图像来训练模型以检测图像中是否有狗。同样的方法可以用于其他领域，例如声音（语音转文本和文本转语音）以及视频。

这种训练被称为“监督”训练，因为训练数据需要事先正确标记。这是目前最成熟和最有用的机器学习类型。还有其他类型的机器学习以无监督的方式工作（训练数据不需要标签），例如，从一组图片中确定哪些图片彼此相关。

机器学习正变得越来越流行。随着时间的推移，机器学习模型变得更有能力并产生更好的结果。这曾经是一个难以进入的复杂领域，但得益于新的云提供商，现成的 API 可供快速利用机器学习的力量。

机器学习可用于许多不同的领域。在本章中，我们将涵盖以下示例：

- 检测图片中的位置
- 查找并提取图像中的文本，包括手写文本
- 检测文本的情感是积极还是消极
- 将文本翻译成其他语言
- 根据之前的示例，确定消息是针对商店内的哪个部门

在本章中，我们将使用公开可用的 Google 资源，特别是他们现成的模型来检测图像和文本中的一般特征。他们公开训练的模型非常强大，允许你检测各种元素。我们还将介绍如何创建和训练一个自定义文本模型，该模型可以将我们自己的标签应用于新文本。

> 机器学习应用根据其复杂性分为三个级别。第一个是应用已有的现成训练模型。第二个是用你自己的数据训练现有模型。第三个是从头开始创建你自己的模型。要达到第三个级别，你需要在机器学习方面具备显著的专业知识，因此超出了本书的范围，但前两个级别是可访问的。本章大部分内容涉及第一个级别，但最后一个食谱涵盖了使用监督训练的第二个级别。

### 使用 Google Cloud Vision AI 分析图像

我们将获取 Google Cloud Vision AI 的基本访问权限，以自动检测图片中可以推断出的广泛类别。这些类别在 API 中被称为标签。这些标签可以识别物体（如盒子）、地点（如风景）、动物物种（如猫）以及其他事物。我们将使用第 4 章“搜索和读取本地文件”中已经展示过的图像。

在本教程中，我们将设置一个 Google Cloud 账户以使用其 API。此过程将作为本章其他教程的基础。

### 准备工作

我们首先需要设置 Google 账户，因此请访问 Google Cloud 网站：https://cloud.google.com/vision：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_388_0.png)

在那里，你可以点击 **免费开始使用** 来设置你的账户：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_389_0.png)

图 11.2：注册页面

作为设置的一部分，你需要提供信用卡信息。你应该会获得一些免费额度，以便有足够的时间进行测试，并且不会产生费用，尽管你会在信用卡账单上看到一个 *Google 临时预授权*：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_390_0.png)

图 11.3：注册后的欢迎屏幕

接下来，你需要访问 **API 和服务** 选项卡：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_391_0.png)

图 11.4：API 和服务页面

并启用 **Vision API**。你可以按名称搜索和筛选，因为有很多不同的 API：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_392_0.png)

图 11.5：搜索 Cloud Vision API

然后，从界面中启用它：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_392_1.png)

图 11.6：API 概览页面，用于启用 API

要访问该 API，你需要创建一组凭据进行身份验证。点击 **创建凭据**：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_393_0.png)

图 11.7：Cloud Vision API 主页面

你需要创建一个服务账户，以便使用 Python 客户端与 Vision API 交互。服务账户是一组旨在用于自动化脚本或“机器人”的凭据，就像我们需要的那样。

前往 **凭据** 页面创建一个新的凭据。点击 **创建凭据** 后，选择 **服务账户**：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_394_0.png)

图 11.8：创建新的服务账户

添加一个描述性名称并创建服务账户。你不需要填写其他两个可选步骤，但它们可以限制可以使用该密钥的用户或授予权限。

最后，点击 **+创建密钥** 按钮并选择 **JSON** 格式来创建一个新的 JSON 密钥：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_395_0.png)

图 11.9：下载 JSON 私钥

这个下载的 JSON 文件将包含访问 API 所需的凭据。在本章中，我们将其称为 `credentials.json`。

> 服务账户密钥可以被删除。密钥被删除后，它将不再有效。理想情况下，应定期更换密钥，以避免因泄露而导致的安全问题。

你还需要启用计费。转到控制台中的 **结算** 部分，`https://console.cloud.google.com/billing`，确保你的项目已启用结算：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_396_0.png)

图 11.10：账户概览

你现在已准备好访问该 API。

> 如果你打算将该系统用于超出一些测试的范围，请花时间探索结算选项。有些选项可以让你限制支出，以避免意外费用。

我们需要添加官方的 Google Cloud Vision 库。我们应该安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "google-cloud-vision==1.0.0" >> requirements.txt
$ pip install -r requirements.txt
```

我们将使用 `image_labels.py` 脚本和 `photo-dublin-b.png` 图像文件，该文件也在第 4 章“搜索和读取本地文件”的 *读取图像* 教程中使用过。你可以从 GitHub 仓库下载它们：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter11。

图像位于 `images` 子目录中。

### 操作步骤...

- 1. 查看 `images/photo-dublin-a2.png` 图像：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_397_0.png)

图 11.11：都柏林海滨照片

- 2. 调用 `image_labels.py` 脚本，传入凭据和 `photo-dublin-a2.png` 文件：

```
$ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python image_labels.py images/photo-dublin-a2.png
Labels for the image and score:
Water 0.9388793110847473
Daytime 0.9213085770606995
River 0.9155402183532715
City 0.9150108098983765
Sky 0.9127334952354431
Waterway 0.9020747542381287
Urban area 0.8954816460609436
Human settlement 0.8528644442558289
Architecture 0.8278814554214478
Metropolitan area 0.8263764381408691
```

请注意，这些标签很好地描述了图片的特征，包括“河流”和“白天”等细节。

### 工作原理...

一个重要的细节是使用 `credentials.json` 凭据文件。注意我们如何在 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量中设置它，以便库可以读取。请记住添加正确的路径来访问该文件。

> 在行首添加变量使其在该命令的环境中可用。这避免了必须在 shell 环境中永久设置它。你也可以使用 `export` 命令来无限期地定义它：

`export GOOGLE_APPLICATION_CREDENTIALS=credentials.json`

如果你使用的是 Windows，你需要使用等效的 `set` 命令来设置环境变量：`set GOOGLE_APPLICATION_CREDENTIALS=credentials.json`

让我们看看步骤 2 中使用的 `image_labels.py` 脚本：

```python
import argparse
from google.cloud import vision

def landmark(client, image):
    print('Landmark detected')
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    for landmark in landmarks:
        print(f'  {landmark.description}')
        for location in landmark.locations:
            coord = location.lat_lng
            print(f'    Latitude {coord.latitude}')
            print(f'  Longitude {coord.longitude}')

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

def main(image_file):
    content = image_file.read()

    client = vision.ImageAnnotatorClient()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print('Labels for the image and score:')

    for label in labels:
        print(label.description, label.score)
        if(label.description == 'Landmark'):
            landmark(client, image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input', type=argparse.FileType('rb'),
                        help='input image')
    args = parser.parse_args()
    main(args.input)
```

该脚本使用 `argparse` 来打开文件并按调用中描述的方式传递。然后，它调用 `main`，该函数使用 Google Cloud Vision API 来检索图像的标签。该过程的核心是：

```python
content = image_file.read()

client = vision.ImageAnnotatorClient()
image = vision.types.Image(content=content)

response = client.label_detection(image=image)
labels = response.label_annotations
print('Labels for the image and score:')

for label in labels:
    print(label.description, label.score)
```

它首先从文件中提取内容。请注意，这是通过 `argparse` 配置为 `rb` 模式提取的二进制内容。

然后将内容发送到 `ImageAnnotatorClient` 以执行 `label_detection`。图像首先需要正确转换为 `vision.types.Image`。

然后打印响应标签的描述和分数。请注意它们是按分数排序的。高分意味着 API 认为该标签是一个很好的匹配。

### 更多内容...

除了特定标签的 `score` 外，接口还会返回其 `topicality`。虽然 `score` 反映了标签适用于图像的置信度，但 `topicality` 反映了标签对整个图像的代表性。通常，它们是相同或非常相似的，但一张风景图像可能显示远处有一座小房子，与 `landscape` 标签相比，它具有较高的 `score` 但较低的 `topicality`。

`label_detection` 接口是最通用的，将返回关于图像的一般信息，但还有其他更具体的接口可用。我们在 `landmark` 函数中添加了地标检测，如果返回了 `landmark` 标签，则会调用该函数：

```python
def landmark(client, image):
    print('Landmark detected')
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
```

### 用于自动化的机器学习

```python
for landmark in landmarks:
    print(f' {landmark.description}')
    for location in landmark.locations:
        coord = location.lat_lng
        print(f'  Latitude {coord.latitude}')
        print(f'  Longitude {coord.longitude}')

if response.error.message:
    raise Exception(
        '{}\nFor more info on error messages, check: '
        'https://cloud.google.com/apis/design/errors'.format(
            response.error.message))
```

如果你使用 `photo-dublin-b.png` 图片（这是都柏林市中心邮政总局大楼的照片）来调用脚本，你将能够触发它：

```
$ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python3 image_labels.py photo-dublin-b.png
Labels for the image and score:
Architecture 0.9421795010566711
Landmark 0.928507924079895
Landmark detected
    General Post Office
    Latitude 53.349369
    Longitude -6.260251
Building 0.8951834440231323
Sky 0.8857545852661133
Classical architecture 0.8762346506118774
Daytime 0.8634399771690369
Town 0.8440616726875305
City 0.82234787940979
Facade 0.8102320432662964
Street 0.758461058139801
```

图像可以以二进制格式发送进行分析，如本例所示；如果图像在网络上可用，也可以作为 URL 发送：

```python
image = vision.types.Image()
image.source.image_uri = uri
```

这可以与*第3章，构建你的第一个网页抓取应用*中的想法相结合，以自动检测网页上的某些类型的图像。例如，你可以检测公司网站上的头像（如标签 Face 所示），或检测车辆目录中的红色汽车图片。

> 有一个特定的 API 用于检测人脸：`face_client.detection(image=image)`，它包含诸如喜悦或悲伤等不同情绪的可能性等详细信息。它的调用方式与 `landmark_detection` 类似。

其他可用功能可以检测图像的主色调、图像中的徽标、露骨（成人）内容，甚至图像是否存在于网络上的某处。请务必查阅文档以了解所有选项。

> 请务必查阅可用类型的文档，以获取从 API 返回的属性。文档地址为 `https://googleapis.dev/python/vision/latest/gapic/v1/types.html`。

你可以访问完整的 Vision API 文档：`https://cloud.google.com/vision/docs/`。Python 客户端文档在此：`https://googleapis.dev/python/vision/latest/index.html`。

### 另请参阅

- 本章后面的*使用 Google Cloud Vision AI 从图像中提取文本*食谱，了解另一种图像分析技术。
- 本章后面的*使用 Google Cloud Natural Language 分析文本*食谱，将类似的方法应用于文本。
- 本章后面的*创建你自己的自定义机器学习模型来分类文本*食谱，了解如何训练你自己的模型。此技术也适用于图像。

### 使用 Google Cloud Vision AI 从图像中提取文本

我们可以利用 Google Cloud 界面的强大功能来检测和提取图像中的文本。这个过程被称为光学字符识别，或 OCR。

### 准备工作

我们需要启用 Google Cloud Vision API 并创建用于操作的凭据，如前一个食谱*使用 Google Cloud Vision AI 分析图像*中所述。我们需要使用生成的 JSON 格式的服务账户密钥。在本章中，我们将称其为 `credentials.json`。

我们需要添加官方的 Google Cloud Vision 库。我们应该安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "google-cloud-vision==1.0.0" >> requirements.txt
$ pip install -r requirements.txt
```

我们将使用 `image_text.py` 脚本以及也在*第4章，搜索和读取本地文件*中使用的 `photo-text.jpg` 和 `dublin-a-text.jpg` 文件。你可以从 GitHub 仓库下载它们：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter11。

图像位于 `images` 子目录中。

### 操作步骤...

1.  查看 `images/photo-text.jpg` 图像：

![带有文本的图像](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_403_0.png)

图 11.12：带有文本的图像

2.  执行 `image_text.py` 脚本，传递凭据和 `images/photo-text.jpg` 文件：

```
$ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python3 image_text.py images/photo-text.jpg
Automate !
```

3.  查看 `images/photo-dublin-a-text.jpg` 文件，它在风景上叠加了相同的文本：

![与上一张图片具有相同文本的风景](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_404_0.png)

图 11.13：与上一张图片具有相同文本的风景

4.  调用 `image_text.py` 脚本，传递凭据和 `images/photo-dublin-a-text.jpg` 文件：

```
$ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python3 image_text.py images/photo-dublin-a-text.jpg
Automate !
```

看看即使文本叠加在图像上，它是如何检测到的。

### 工作原理...

让我们看看在步骤 2 和 4 中使用的 `image_text.py` 脚本：

```python
import argparse
from google.cloud import vision

def main(image_file, verbose):
    content = image_file.read()

    client = vision.ImageAnnotatorClient()
    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:

        for block in page.blocks:

            if verbose:
                print('\nBlock confidence: {}\n'.format(block.confidence))

            if block.confidence < 0.8:
                if verbose:
                    print('Skipping block due to low confidence')
                continue

            for paragraph in block.paragraphs:
                paragraph_text = []
                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    paragraph_text.append(word_text)
                    if verbose:
                        print(f'Word text: {word_text} '
                              f'(confidence: {word.confidence})')
                    for symbol in word.symbols:
                        print(f'\tSymbol: {symbol.text} '
                              f'(confidence: {symbol.confidence})')

    print(' '.join(paragraph_text))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input', type=argparse.FileType('rb'),
                        help='input image')
    parser.add_argument('-v', dest='verbose', help='Print more data',
                        action='store_true')
    args = parser.parse_args()
    main(args.input, args.verbose)
```

最后的代码块使用 `argparse` 库配置命令行参数。它配置了输入文件和 `verbose` 参数。我们将在*更多内容*部分讨论 `verbose` 参数。

`main` 函数中的核心代码如下：

```python
def main(image_file, verbose):
    content = image_file.read()

    client = vision.ImageAnnotatorClient()
    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        ...
        for paragraph in block.paragraphs:
            paragraph_text = []
            for word in paragraph.words:
                word_text = ''.join([symbol.text for symbol in word.symbols])
                paragraph_text.append(word_text)

    print(' '.join(paragraph_text))
```

图像的内容被读取并通过 `document_text_detection` 接口发送到 Vision API。这返回一个结构化的响应，分为块、段落、单词，最后是符号。

置信度低的块会被跳过。这可以防止打印 API 不确定是否正确的元素。

> 置信度级别被任意设置为 0.8。你可以调整这个值，直到找到适合你的值。如果你对任何置信度级别都满意，你可以直接使用 `response.full_text_annotation.text` 调用文本，尽管如果图像中存在可能给文本增加噪声的元素（如 `photo-dublin-a-text.jpg` 图像），这可能会返回错误的结果。

符号被聚合成单词，然后组成段落。每个段落都独立打印。

### 更多内容...

`verbose` 标志显示有关每个符号置信度范围的更多信息。例如，调用：

```
$ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python3 image_text.py images/photo-dublin-a-text.jpg -v
```

```
Block confidence: 0.9900000095367432

Word text: Automate (confidence: 0.9900000095367432)
    Symbol: A (confidence: 0.9900000095367432)
    Symbol: u (confidence: 0.9900000095367432)
    Symbol: t (confidence: 1.0)
    Symbol: o (confidence: 1.0)
    Symbol: m (confidence: 1.0)
    Symbol: a (confidence: 1.0)
    Symbol: t (confidence: 1.0)
    Symbol: e (confidence: 0.9900000095367432)
```

### 使用 Google Cloud Natural Language 分析文本

在本教程中，我们将使用 Google Cloud 界面来评估文本。这与之前*使用 Google Cloud Vision AI 分析图像*的教程类似，但应用于文本。我们将能够检测一段文本的语言及其整体情感倾向（即其积极或消极的程度）。我们还将把所有非英文文本翻译成英文。

### 准备工作

我们需要启用 Google Cloud Natural Language API 并创建用于访问它的凭据。大部分过程与本章前面描述的在 Google Cloud 项目中启用 Vision API 的过程相似，因此我们将使用之前*使用 Google Cloud Vision AI 分析图像*教程中描述的同一个项目。

登录您的账户，前往 API 控制台 https://console.cloud.google.com/apis。确保您使用的是与之前相同的项目，否则 `credentials.json` 文件将无法工作：

点击 **ENABLE APIS AND SERVICES** 并搜索 **Natural Language** 以启用它：

同样，搜索并启用 **Cloud Translation API**：

我们需要使用生成的 JSON 格式的服务账户密钥。在本章中，我们将其称为 `credentials.json`。

我们将使用 `google-cloud-language` 模块。我们应该安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "google-cloud-language==1.3.0" >> requirements.txt
$ echo "google-cloud-translate==2.0.1" >> requirements.txt
$ pip install -r requirements.txt
```

代码可以在 GitHub 仓库中找到，https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter11。在 `/texts/` 子目录中有一些文本示例，主要包含19世纪经典小说开头的不同语言文本。

### 操作步骤...

1.  使用小说《傲慢与偏见》的开头调用 `text_analysis.py` 脚本，并传递凭据：

    ```
    $ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python text_analysis.py texts/pride_and_prejudice.txt
    ```

    文本：It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.

    语言：en
    情感得分（情感的积极程度）：0.699999988079071
    情感强度（情感的强烈程度）：1.5

2.  使用西班牙语小说《女执政官》的开头调用 `text_analysis.py` 脚本，并传递凭据：

    ```
    $ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python text_analysis.py texts/regenta.txt
    ```

    文本：La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo.

    语言：es
    情感得分（情感的积极程度）：0.0
    情感强度（情感的强烈程度）：0.8999999761581421
    英文翻译
    The heroic city napped. The south wind, hot and lazy, pushed the whitish clouds that ripped as they ran north. In the streets there was no more noise than the shrill noise of the swirls of dust, rags, straws and papers that went from stream to stream, from sidewalk to sidewalk, from corner to corner revoking and chasing each other, like butterflies that seek and flee and that the air envelops in its invisible folds. Like mobs of urchins, those crumbs from the garbage, those leftovers from everything gathered in a heap, they stood as if for a moment asleep and they jumped again with a start, dispersing, some climbing the walls to the trembling glass of the lanterns, others to the paper posters badly glued to the corners, and there was a pen that reached a third floor, and sand that was embedded for days, or for years, in the window of a shop window, clinging to a lead.

### 工作原理...

让我们看一下在*步骤 1 和 2*中使用的 `text_analysis.py` 脚本：

```python
import argparse
from google.cloud import language
from google.cloud import translate_v2 as translate
from google.cloud.language import enums
from google.cloud.language import types

def main(image_file):
    content = image_file.read()
    print(f'Text: {content}')
    document = types.Document(content=content,
                              type=enums.Document.Type.PLAIN_TEXT)

    client = language.LanguageServiceClient()

    response = client.analyze_sentiment(document=document)
    lang = response.language
    print(f'Language: {lang}')
    sentiment = response.document_sentiment
    score = sentiment.score
    magnitude = sentiment.magnitude
    print(f'Sentiment Score (how positive the sentiment is): {score}')
    print(f'Sentiment Magnitude (how strong it is): {magnitude}')
    if lang != 'en':
        # Translate into English
        translate_client = translate.Client()
```

### 另请参阅

-   本章前面的*使用 Google Cloud Vision AI 分析图像*教程，了解如何为 Google Cloud 界面创建账户。

### 用于自动化的机器学习

```python
response = translate_client.translate(content,
    target_language='en')
    print('IN ENGLISH')
    print(response['translatedText'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input', type=argparse.FileType('r'),
                        help='input text')
    args = parser.parse_args()
    main(args.input)
```

最后一个代码块使用 `argparse` 库配置命令行参数。它配置输入文件并以文本格式打开。`main` 函数执行操作。

第一阶段是从文件中提取文本。文本被封装成 `types.Document` 类型，将其定义为纯文本。

使用 `LanguageServiceClient` 创建客户端，并调用 `analyze_sentiment` 方法从 Google 服务器获取结果。

此调用返回自动检测到的语言（存储在 `lang` 中）以及文本的情感（存储在 `score` 和 `magnitude` 中）。

> 情感由 `score` 和 `magnitude` 共同构成。`score` 描述文本中聚合情感的积极程度。分数为 -1.0 表示极其负面的情绪，1.0 表示极其正面的情绪。`magnitude` 表示这种情感在文本中的清晰程度，是一个正数。短句可能难以被 API 正确评估。

如果语言不是英语，则通过创建新的 `Client` 并调用 `translate` 进行翻译：

```python
translate_client = translate.Client()
response = translate_client.translate(content,
    target_language='en')
print('IN ENGLISH')
print(response['translatedText'])
```

翻译后的文本将被打印出来。

> Google 翻译有时会因偶尔出现奇怪的翻译而受到一些负面评价，但在大多数情况下，它仍然能够提供相当不错的结果，并且支持的语言数量令人印象深刻。

自动翻译可以作为初始阶段，用于确定哪些部分需要更好的人工翻译，例如习语。

### 还有更多...

不同语言的语言处理质量差异很大。某些语言的支持程度可能不如英语。请查阅文档了解更多详情。

Google 文本 API 还具有其他功能。例如，它们可以根据预定义的类别对文本进行分类。`text_analysis_categories.py` 脚本就实现了这一功能，它可在 GitHub 上找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter11。执行该脚本可以对文本进行分类，例如，对 `texts/category_example.txt` 文本进行分类：

```
$ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python text_analysis_categories.py texts/category_example.txt
Text: This text talks about literature and different authors from the XIX century. It discusses the different styles from different authors in different languages, analysing and comparing them with their historical context.

Categories
Category: /Books & Literature
Confidence: 0.9300000071525574
```

让我们看看 `text_analysis_categories.py` 脚本，它与 `text_analysis.py` 非常相似：

```python
import argparse
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

def main(image_file):
    content = image_file.read()
    print(f'Text: {content}')
    document = types.Document(content=content,
                              type=enums.Document.Type.PLAIN_TEXT)

    client = language.LanguageServiceClient()

    print('Categories')
    response = client.classify_text(document=document)
    if not response.categories:
        print('No categories detected')

    for category in response.categories:
        print(f'Category: {category.name}')
        print(f'Confidence: {category.confidence}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input', type=argparse.FileType('r'),
                        help='input text')
    args = parser.parse_args()
    main(args.input)
```

为了获取类别，它调用 `classify_text`，该方法返回类别以及每个检测到的类别的置信度。完整的类别列表可以在 Google 文档中找到：https://cloud.google.com/natural-language/docs/categories。

> 与情感分析一样，请在文档中检查 API 是否与特定语言兼容，因为 API 可能无法对某些语言进行分类。你可以尝试先翻译成英语，然后再进行分类，但结果的质量可能会有所不同。

在所有这些情况下，文档也可以使用 `enums.Document.Type.PLAIN_TEXT` 类型接受 HTML 文本。如果你想将此食谱与本书中的其他食谱结合使用，例如爬取网络并将有趣的文章翻译成你的母语，或者对来自 RSS 订阅源的博客文章进行分类以仅过滤相关文章，这会很有帮助。

访问 https://cloud.google.com/natural-language/ 查看完整的自然语言 API 文档。Python 客户端的文档可在 https://googleapis.dev/python/translation/latest/index.html 和 https://googleapis.dev/python/language/latest/index.html 获取。

### 另请参阅

- 本章前面的 *使用 Google Cloud Vision AI 分析图像* 食谱，了解如何创建账户以使用 Google Cloud 接口。
- 本章后面的 *创建你自己的自定义机器学习模型来分类文本* 食谱，了解如何训练模型以识别自定义元素。

### 创建你自己的自定义机器学习模型来分类文本

使用默认接口根据情感或通用类别对文本进行分类非常强大，但不允许我们根据自己的规则对不同的文本进行分类。能够创建自己的模型正是机器学习的全部力量所在。

幸运的是，Google 提供了基于我们自己的训练数据集创建和训练模型的能力。这使我们能够生成一组文本并使用我们自己的标签对其进行分类。利用这些数据，我们将准备自己的模型，以便与新的文本进行匹配。

在这个食谱中，我们将看到一个分类发送给一家商店的电子邮件的示例，该商店有两个部门：“电器”和“家具”。我们将创建第三个类别“其他”，用于捕获那些不适合归入前两个类别的电子邮件。

这个过程高度依赖于提供给模型的数据质量。本食谱中给出的示例很简单，以保持其小巧，但它们展示了这种技术的潜力。

让我们看看如何创建和操作我们自己的机器学习模型。

### 准备工作

我们需要启用 Google Cloud 自然语言 API 并创建凭据以进行操作。大部分过程与本章前面描述的在 Google Cloud 项目中启用 Vision API 的过程类似，因此我们可以从前面 *使用 Google Cloud Vision AI 分析图像* 食谱中使用的项目开始。

### 用于自动化的机器学习

登录你的账户，前往 API 仪表板 `https://console.cloud.google.com/apis`。请确保使用与之前相同的项目，否则 `credentials.json` 文件将无法工作。

我们准备了一些用于训练模型的文本。它可在 GitHub 上找到：`https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter11`。

在 `shop_training` 子目录中，有三个子目录（`appliances`、`furniture` 和 `others`），每个子目录包含十个包含电子邮件内容的文本文件。每个子目录对应一个标签，因此每个子目录中的所有电子邮件都应被归类到该标签下。

数据被压缩在一个名为 `shop.zip` 的文件中。它包含相同的信息，但 Google 要求以 zip 格式上传。

> 每个电子邮件文本都模拟了向商店的两个部门之一或其他部门提出的请求。你可以查看它们。

我们需要创建一个新模型。前往 `https://console.cloud.google.com/natural-language` 并选择 **AutoML 文本和文档分类**：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_419_0.png)

现在我们需要用我们的示例添加一个新的数据集：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_420_0.png)

选择我们的数据集用于**单标签分类**（每个文本将有一个单一标签）：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_420_1.png)

要上传文件，请选择一个ZIP文件。你需要创建一个新的存储桶来存储数据。请记住，存储桶名称需要是全局唯一的，因此你需要指定自己的名称。

### 用于自动化的机器学习

按照说明创建一个存储桶。添加ZIP文件：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_421_0.png)

图 11.21：导入训练集

导入数据将需要几分钟时间。一旦导入完成，你将看到所有可用数据作为不同的项目：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_421_1.png)

图 11.22：导入的数据

有了这些数据，我们现在可以训练模型了。点击**训练**选项卡，然后点击**开始训练**，这可能需要*几个小时*，所以请耐心等待。完成后你会收到一封电子邮件：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_422_0.png)

图 11.23：开始训练

> 正如你所看到的，为了正确标注数据，每个标签应有100个或更多的示例。我们的数据没有那么充足。示例越多，模型效果越好，尽管训练时间会更长。请记住，训练模型所需的计算能力将计入你的账户。使用免费账户，额度应该足够进行一些测试，但使用平台时请注意潜在的费用。

训练完成后，你可以查看模型的评估结果。系统会保留一些训练数据作为测试，以评估你的模型效果如何：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_422_1.png)

图 11.24：评估模型的仪表板

这里最重要的两个参数是`精确率`和`召回率`。

> 精确率是模型预测正确标签的能力，而召回率是不分配错误标签的能力。精确率低的模型会有更多的假阳性——即附加了不正确标签的结果。例如，它会在没有微笑时检测到微笑。召回率低的模型会产生假阴性——即本应附加标签的结果没有被附加。例如，它在有微笑时检测不到微笑。

混淆矩阵显示了基于训练数据的检测模式。一个完美的矩阵将显示100%的对角线，这意味着数据被正确分类，没有错误元素。你可以看到这里并非如此。

> 我们66.67%的精确率和召回率参数并不理想。这部分是由于生成的样本数量较少。在实际应用中，目标应达到90%或更高。

下一个选项卡**测试与使用**包含了我们使用所需的所有信息，但要能够调用API，我们需要为我们的服务账户添加适当的权限。

> 复制**测试与使用**选项卡末尾的Python模型引用。它看起来类似于`projects/<PROJECT_ID>/locations/us-central1/models/<ID>`。我们稍后将用它作为`REFERENCE`。

请记住，服务账户与凭据相关联。要添加它们，请访问 https://console.cloud.google.com/iam-admin/：

### 第11章

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_424_0.png)

图 11.25：定义了用户的仪表板

我们需要将服务账户添加为用户，并授予其**AutoML编辑器**角色。转到**服务账户**页面以获取我们创建的服务账户的名称：

> 查看本章的第一个食谱*使用Google Cloud Vision AI分析图像*，了解创建服务账户的过程。

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_424_1.png)

图 11.26：服务账户列表

复制服务账户的名称，返回**IAM**选项卡，使用该名称创建一个新用户。使用顶部的**添加**按钮。输入服务账户名称并添加额外的**AutoML编辑器**角色：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_425_0.png)

图 11.27：为成员添加角色

保存后，模型就可以使用`credentials.json`文件进行身份验证了。记得标注模型的`REFERENCE`，因为稍后会用到。

我们将使用`google-cloud-automl`模块。我们应该安装该模块，将其添加到我们的`requirements.txt`文件中，如下所示：

```
$ echo "google-cloud-automl==0.10.0" >> requirements.txt
$ pip install -r requirements.txt
```

代码可以在GitHub仓库中找到：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter11。有一些名为`example_shopX.txt`的文本示例。我们将使用的脚本是`text_predict.py`。请记住，训练数据位于`shop_training`子目录中，并压缩为`shop.zip`。

`精确率`和`召回率`相互对立，并且可以在一定程度上进行调整。

### 如何操作...

1.  检查来自`example_shop1.txt`的消息，并使用`text_predict.py`、`credentials.json`和模型的`REFERENCE`对其进行分类：

```
$ cat example_shop1.txt
Hello:

Are there any offers in fridges? I'm searching to replace mine.
I live in a fifth floor and the lift is broken, would that be a
problem? I'll be fine with paying an extra.

Thanks a lot,
Carrie
$ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python text_
predict.py -m projects/<project_id>/locations/us-central1/
models/<id> example_shop1.txt
Label: appliances : 0.99986
Label: furniture : 0.00014
Label: others : 0.00000
```

检查结果是否正确标记为`appliances`。

2.  查看来自`example_shop2.txt`的消息，并使用`text_predict.py`、`credentials.json`和模型的`REFERENCE`对其进行分类：

```
$ cat example_shop2.txt
Hello:

Are there any offers in fridges? I'm looking to replace mine
that is old. I live in a fifth floor and the lift is broken, would
that be a problem? I'll be fine with paying an extra.
I think you also have a furniture department, right? What are
the prices for mattresses?

Thanks a lot,
Carrie
$ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python text_
predict.py -m projects/<project_id>/locations/us-central1/
models/<id> example_shop2.txt
Label: furniture : 0.99995
Label: appliances : 0.00005
Label: others : 0.00000
```

检查结果是否正确标记为`furniture`。

3.  查看来自`example_shop3.txt`的消息，并使用`text_predict.py`、`credentials.json`和模型的`REFERENCE`对其进行分类：

```
$ cat example_shop3.txt
Hello:

    I need your full details including your address and phone for an invoice. Can you please send them to me?

    Thanks a lot,
        Carrie
$ GOOGLE_APPLICATION_CREDENTIALS=credentials.json python text_predict.py -m projects/<project_id>/locations/us-central1/models/<id> example_shop3.txt
Label: others : 1.00000
Label: furniture : 0.00000
Label: appliances : 0.00000
```

检查结果是否正确标记为`others`。

### 工作原理...

让我们看一下在所有步骤中使用的`text_predict.py`脚本：

```
import argparse

from google.api_core.client_options import ClientOptions
from google.cloud import automl_v1

def main(content, model_name):
    content = args.input.read()
    options = ClientOptions(api_endpoint='automl.googleapis.com')
    prediction_client = automl_v1.PredictionServiceClient(
        client_options=options
    )
    payload = {'text_snippet': {
        'content': content,
        'mime_type': 'text/plain'}
    }
    params = {}
    request = prediction_client.predict(model_name, payload, params)
    for result in request.payload:
        label = result.display_name
        match = result.classification.score
        print(f'Label: {label} : {match:.5f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input', type=argparse.FileType('r'),
                        help='input text')
    parser.add_argument('-m', dest='model', type=str, help='model ref')
    args = parser.parse_args()

    main(content, args.model)
```

脚本文件的最后一部分处理参数解析，这在*第2章，让任务自动化变得简单*中有详细描述。它接受一个文件和模型的详细信息。

主函数读取输入文本并调用Google AutoML API。数据需要以包含`mime_type`的标准结构发送。`prediction_client.predict`的结果被打印出来。请注意，它是排序的，因此最相关的标签位于开头：

```
for result in request.payload:
    label = result.display_name
    match = result.classification.score
    print(f'Label: {label} : {match:.5f}')
```

分数匹配显示到小数点后5位，使用`{match:.5f}`。

你可以创建自己的消息并测试它们是否被正确标记。

### 更多内容...

将文本分类到不同类别有助于将电子邮件引导到正确的部门，为它们分配不同的优先级，根据消息数据库生成统计信息，或将它们分成不同的组。

### 用于自动化的机器学习

在我们的示例中，我们使用了一个简单的单标签检测模型，但该 API 允许更复杂的过程，例如识别多个标签（而非每个文本一个标签），或识别自定义情感或实体。请务必查阅完整文档：https://cloud.google.com/natural-language/automl/docs。

> 尽可能使用最简单的模型。与所有工具一样，在配置和调整方面存在投资回报。在我们的示例中，简单地按部门划分电子邮件可能比尝试设置多个标签更好。采用合理的方法，在生成复杂模型之前先从小处着手。复杂模型也需要更长时间来调整。

该过程中最重要的步骤是选择一组合适的数据来训练模型。建议每个标签至少使用 100 个样本。数据集的选择也很重要，因为它必须能代表你想要检测的事物；既要包括应该被标记的事物，也要包括不应该被标记的事物。

> 机器学习模型非常依赖于训练集，因此很容易陷入强化训练集中偏见的陷阱。机器学习的核心是利用偏见来获得结果。创建多样化的数据集非常重要。用图片来解释这一点很容易。如果你只使用猫坐着的图片来训练识别猫的模型，它可能无法识别跳跃或奔跑的猫。

最小化隐藏偏见可能很困难。在你的训练数据中包含足够多的罕见匹配案例示例，这样常见案例就不会完全压倒它们。

我们之前讨论过精确率和召回率。虽然它们并非完全相互依赖，但它们确实相关，并且可以相互调整。模型允许通过**置信度阈值**进行一些调整。你可以在模型的 **EVALUATE** 选项卡上更改它：

![](img/f6b6daf89c6a40b2f1ee7cdf5c5dea1b_430_0.png)

图 11.28：可以从 EVALUATE 面板调整置信度阈值

你也可以使用此页面查看不同的标签及其相互关系。

> 💡 如果你想获得良好的结果，调整训练集和调整模型是该过程的重要组成部分。花足够的时间测试和调整你想要在实际操作中使用的模型。

遵循相同的参数，你也可以使用 AutoML Vision 产品，它的工作方式相同，但处理的是图像。你使用一组图片训练模型来识别和分类不同的标签，然后用它来分析新图片。

你可以用它来训练检测图片中的自定义元素。例如，它可以被训练来统计国家公园中不同亚种熊的数量或特定汽车型号。你可以使用 AutoML Vision 来识别特定的表情，如微笑，或更复杂的表情，如左眼眨眼或挑眉。这结合每秒拍摄多张照片的相机，可以在无法使用双手且声音不可行的环境（如工业环境）中实现快速交互界面。其可能性才刚刚被发掘。

你可以在 AutoML Vision 的完整文档中了解更多：https://cloud.google.com/vision/overview/docs#automl-vision。

### 另请参阅

- 本章前面的 *使用 Google Cloud Vision AI 分析图像* 配方，了解如何创建账户以使用 Google Cloud 界面。
- 上一节的 *使用 Google Cloud Natural Language 分析文本* 配方，了解使用预训练模型进行文本分析的其他可能性。

# 12 自动化测试例程

在本章中，我们将涵盖以下配方：

- 编写和执行测试用例
- 测试外部代码
- 使用依赖模拟进行测试
- 使用 HTTP 调用模拟进行测试
- 准备测试场景
- 有选择地运行测试

### 简介

当你的代码和软件复杂度增长时，生成测试以确保你的程序按预期运行，是在崎岖地形上为你提供坚实立足点的最佳工具。

测试本质上是再次检查代码是否有效并按预期执行。这是一个看似简单的陈述，但在实践中可能非常困难。

> 掌握测试能力是一项艰巨的任务，值得用一两本书来阐述。本章介绍的任务从业务导向任务转向软件工程任务，这是一种不同的方法。本章的目标是介绍测试的一些实际方面，以引入这个主题。

### 自动化测试例程

测试最重要的事情是它尝试独立地检查被测试的代码。这涉及到一种心态，即看待代码、输入和输出，并以全新的视角处理任务，不受内部实现的影响。在某些情况下，测试软件的人员可能与最初编写代码的人员不同，以确保对代码应该和不应该做什么有良好的理解。尝试用这种方法设计你的测试，并创建定义良好的接口来使用。

> 总之，测试检查代码是否做了它应该做的事，以及是否没有做它不应该做的事。

句子的第一部分比较容易，但第二部分非常困难甚至不可能。编写测试有成本，包括时间和维护。只有在高度关键的软件中，代码才会被广泛测试以确保没有意外发生。尝试在多少测试适合你的需求方面找到平衡。

测试通常根据它们测试软件的不同部分的数量进行分类。只覆盖代码一小部分（如一个函数）的测试通常称为单元测试，而覆盖整个系统的测试称为系统测试。覆盖不同软件元素集成的测试称为集成测试。这是一个非常流动的定义，因为并非所有人都同意什么是系统，或者单元测试在变成集成测试之前可以有多大，或者集成测试和系统测试之间是否有显著差异。但理解不同的测试覆盖软件的不同区域——有些更大，有些更小——并且有不同的要求是有帮助的。

> 要让一个开发团队在同一个软件上工作，确保其高质量，并持续添加和运行测试的纪律至关重要。这就是为什么有很多类型的测试来帮助完成这项任务。持续集成（CI）包括对软件的每次更改运行测试，并在将其与其他开发者的更改合并之前运行测试的实践。CI 工具允许你在后台自动运行测试，并将任何问题通知开发者，确保没有意外的失败。

在本章中，我们将介绍使用 `pytest` 作为运行各种测试的工具。这是最完整的 Python 测试框架之一。`pytest` 能够轻松设置测试，并提供许多有用的选项来运行测试子集、快速运行测试以及在测试失败时检测问题。

它还有一个广泛的插件生态系统，允许你与其他系统（如数据库和 Web 服务）集成，并通过额外功能扩展 pytest，例如代码覆盖率、并行运行测试和性能基准测试。让我们开始编写和运行一些简单的测试。

### 编写和执行测试用例

在这个配方中，我们将学习如何使用 pytest 库定义和运行测试。

### 准备工作

我们将使用 pytest 模块。我们应该安装该模块，将其添加到我们的 requirements.txt 文件中，如下所示：

```
$ echo "pytest==5.4.1" >> requirements.txt
$ pip install -r requirements.txt
```

我们将使用文件 test_case.py。你可以从 GitHub 仓库下载它：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter12/tests。

### 如何操作...

1. 查看文件 tests/test_case.py，其中包含四个测试的定义：

```
LIST = [1, 2, 3]

def test_one():
    # Nothing happens
    pass

def test_two():
    assert 2 == 1 + 1

def test_three():
    assert 3 in LIST
```

自动测试例程

```python
def test_fail():
    assert 4 in LIST
```

2. 运行 pytest 来执行测试文件：

```
$ pytest tests/test_case.py
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1,
pluggy-0.13.1
rootdir: /Python-Automation-Cookbook/Chapter12
collected 4 items

tests/test_case.py ...F
[100%]

=============================== FAILURES ===============================
_______________________________ test_fail ________________________________

def test_fail():
>       assert 4 in LIST
E       assert 4 in [1, 2, 3]

tests/test_case.py:16: AssertionError
============================= short test summary info =============================
FAILED tests/test_case.py::test_fail - assert 4 in [1, 2, 3]
========================= 1 failed, 3 passed in 0.03s =========================
```

恭喜，你已经运行了你的第一个测试套件。请注意其中一个测试失败了。

### 工作原理...

pytest 允许你简单地将测试定义为函数。

> 并非每个函数都会被解释为测试。默认情况下，只有以 "test" 开头的函数才会被视为测试。这是一个非常方便的默认配置，但可以根据需要进行更改。相关文档可在 https://docs.pytest.org/en/latest/goodpractices.html#test-discovery 找到。

`tests/test_case.py` 中的每个函数都定义了一个测试，正如我们在 *步骤 1* 中看到的那样。

文件 `tests/test_case.py` 中定义的四个测试构成了 pytest 测试的基础：执行代码并包含一个或多个断言来验证代码是否正确。请注意，这些初始测试非常简单。我们将在本章后面看到更复杂的测试：

- `test_one` 没有任何断言，所以它只能通过。
- `test_two` 使用 == 运算符检查加法是否正确。
- `test_three` 使用 `in` 运算符检查一个值是否包含在列表中。
- `test_fails` 使用 `in` 运算符检查一个值是否包含在列表中。它不包含，所以这个测试会失败，如 *步骤 2* 所示。

在 *步骤 2* 中，pytest 在开始时收集所有测试，然后运行它们：

```
collected 4 items

tests/test_case.py ...F
[100%]
```

每个通过的测试用一个点表示，而每个失败的测试显示一个 F。

> 如果终端支持颜色，失败的测试将标记为红色，通过的测试标记为绿色。

自动测试例程

对于失败的测试，会显示断言失败的具体行的详细信息：

```
test_fail

def test_fail():
>       assert 4 in LIST
E       assert 4 in [1, 2, 3]

tests/test_case.py:16: AssertionError
========================= short test summary info ==========================
FAILED tests/test_case.py::test_fail - assert 4 in [1, 2, 3]
```

稍后会显示失败测试、总测试数和所用时间的简要摘要。这些信息对于采取行动并修复代码以使测试通过非常有用。

### 更多内容...

默认情况下，显示的信息几乎不显示通过的测试。这使你可以专注于失败的测试。如果你想显示每个单独的测试，可以在调用 `pytest` 时启用详细标志（`-v` 或 `--verbose`）：

```
$ pytest -v tests/test_case.py
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1, pluggy-0.13.1 -- /usr/local/opt/python@3.8/bin/python3.8
cachedir: .pytest_cache
rootdir: Python-Automation-Cookbook/Chapter12
collected 4 items

tests/test_case.py::test_one PASSED                                [ 25%]
tests/test_case.py::test_two PASSED                                [ 50%]
tests/test_case.py::test_three PASSED                              [ 75%]
tests/test_case.py::test_fail FAILED                               [100%]

============================= FAILURES ===============================
_____________________________ test_fail _______________________________

def test_fail():
>       assert 4 in LIST
E       assert 4 in [1, 2, 3]

tests/test_case.py:18: AssertionError
========================= short test summary info ==========================
FAILED tests/test_case.py::test_fail - assert 4 in [1, 2, 3]
========================= 1 failed, 3 passed in 0.03s =========================
```

Python 中 `assert` 关键字的使用非常灵活且易于理解。测试需要以断言某个表达式求值为 True 的方式来定义。

`pytest` 不仅会显示失败的行，还会提供一些上下文，如我们的示例所示，使问题易于发现。

> 尝试修复错误，使所有测试通过。

你可以在以下链接访问完整的 `pytest` 文档：https://docs.pytest.org/。

### 另请参阅

- 本章后面的 *选择性运行测试* 配方，了解如何仅运行测试的一个子集。
- 本章接下来的 *测试外部代码* 配方，了解如何测试其他模块中的代码。

### 测试外部代码

测试的主要目标是能够检查测试文件边界之外的代码。我们可以在测试中轻松导入代码，然后验证它是否按预期工作。让我们看看如何做到这一点。

### 准备工作

我们将使用 `pytest` 模块。我们应该安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "pytest==5.4.1" >> requirements.txt
$ pip install -r requirements.txt
```

我们将使用测试文件 `tests/test_external.py` 和 `code/external.py`。你可以从 GitHub 仓库 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter12 下载它们，位于 `tests` 和 `code` 子目录下。

目录结构应如下所示：

```
├── code
│   ├── __init__.py
│   └── external.py
├── conftest.py
└── tests
    └── test_external.py
```

### 如何操作...

1. `__init__.py` 文件和 `conftest.py` 是空的，但定义了模块的结构。
2. 检查文件 `code/external.py`，其中包含一个除法函数的定义：

    ```python
    def division(a, b):
        return a / b
    ```

3. 测试文件 `tests/test_external.py` 包含一些关于除法函数的测试：

    ```python
    import pytest
    from code.external import division

    def test_int_division():
        assert 4 == division(8, 2)

    def test_float_division():
        assert 3.5 == division(7, 2)

    def test_division_by_zero():
        with pytest.raises(ZeroDivisionError):
            division(1, 0)
    ```

4. 在 `tests/test_external.py` 上运行 `pytest` 以查看所有测试通过：

    ```
    $ pytest tests/test_external.py
    ============================= test session starts =============================
    platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
    rootdir: Python-Automation-Cookbook/Chapter12
    collected 3 items

    tests/test_external.py ...                                                [100%]

    ============================== 3 passed in 0.01s ===============================
    ```

### 工作原理...

目录结构允许 `pytest` 检测结构中的不同模块：

- `__init__.py` 文件定义了子目录 `code` 包含一个 Python 模块。这是一个标准的 Python 定义。
- 文件 `conftest.py` 包含 `pytest` 的特定信息。即使它是空的，它也定义了测试的根目录。
- 以 `test` 开头的文件被检测为包含测试。在这些文件内部，以 `test` 为前缀的函数被检测并运行。

`code` 模块的内容在 *步骤 1* 中定义。文件 `external.py` 包含 `division` 函数。

在 *步骤 3* 中，定义了测试文件。注意导入：

```python
from code.external import division
```

这允许你使用在测试文件边界之外定义的外部代码。然后在三种情况下验证该函数：

- `test_int_division` 检查 `division` 除两个整数并返回正确的整数结果。
- `test_float_division` 验证除两个整数可以产生浮点数结果。
- `test_division_by_zero` 检查在尝试除以零时是否引发正确的异常。

### 自动化测试流程

前两个测试包含一个简单的断言来检查函数调用的结果，但 `test_division_by_zero` 要求你验证代码是否引发了一个异常。这通过一个使用 `pytest.raises` 的 `with` 代码块来实现：

```python
def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        division(1, 0)
```

如果调用没有产生异常，这个代码块将生成一个断言错误，从而让你检查其行为。请注意，需要将特定的异常作为参数传递。

步骤 4 运行测试，检查所有测试是否正确以及代码是否按预期执行。

### 更多内容...

为了能够定义要捕获的异常，你需要从模块代码或相关库中导入异常定义。

在 `with` 代码块内的任何部分引发异常都会被捕获。请尝试包含尽可能小的、预期会引发异常的调用，以避免捕获无关的异常。出于同样的原因，应声明最精确的异常。

当运行 `pytest` 而不指定文件时，它将尝试检测可能的测试文件并捕获子目录中的所有测试。请确保模块已按照 `__init__.py` 和 `conftest.py` 的定义正确设置。

> 我们将在本章后面更多地了解 `conftest.py`。

### 另请参阅

- 上一节的 *编写和执行测试用例* 配方，以了解定义测试的基础知识。
- 本章后面的 *使用 HTTP 调用模拟进行测试* 配方，以了解如何使用模拟特定库的测试模块。

### 使用依赖模拟进行测试

使用 Python 的最大优势之一是拥有丰富的可用资源库。这包括标准库中的模块，例如用于读写 CSV 文件的 `csv` 模块或用于使用正则表达式的 `re` 库。

其他可能是外部的，例如用于解析 HTML 的 `Beautiful Soup` 或用于生成图表的 `matplotlib`。我们也可以创建自己的库，或者将代码组织在不同的文件中，并以可重用且提高可读性的方式封装功能。

在创建测试时，有时不建议在测试核心中使用外部元素和库调用。例如，为了测试一个报告是否被正确处理，可能需要读取一个包含该报告的 CSV 文件。但通过准备文件来准备测试，在这种情况下会变得繁琐，并且可能偏离测试的实际目标。

在这些情况下，模拟这些依赖项以简化测试或避免外部调用（如网络访问或其他硬件调用）可能是方便的。这种在测试期间替换外部依赖项的模拟被称为 `mock`。

> Mock 与单元测试密切相关，因为它们是小型、专注的测试，覆盖单个代码单元，如一个函数、一个类，甚至一个小模块。这允许你将这些代码单元与外部元素隔离测试，或者更准确地说，完全控制代码访问的外部元素。请记住，一个单元测试在不再是单元测试之前可以有多大，这是有争议的。

这可以通过 Python 标准库中的 `mock` 库来完成，它允许你替换外部依赖项的行为。

### 准备工作

我们将使用 `pytest` 模块。我们应该安装该模块，将其添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "pytest==5.4.1" >> requirements.txt
$ pip install -r requirements.txt
```

我们将使用测试文件 `tests/test_dependencies.py` 和 `code/dependencies.py`。你可以从 GitHub 仓库 `https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter12` 下载它们，位于 `tests` 和 `code` 子目录下。

目录结构应如下所示：

```
├── code
│   ├── __init__.py
│   └── dependencies.py
├── conftest.py
└── tests
    └── test_dependencies.py
```

### 操作步骤...

1. `__init__.py` 文件和 `conftest.py` 是空的，但定义了模块的结构。
2. 查看文件 `code/dependencies.py`，其中包含计算某些形状面积的定义：

```python
PI = 3.14159

def rectangle(sideA, sideB):
    return sideA * sideB

def circle(radius):
    return 2 * PI * radius

def calculate_area(shape, sizeA, sizeB=0):
    if sizeA <= 0:
        raise ValueError('sizeA needs to be positive')

    if sizeB < 0:
        raise ValueError('sizeB needs to be positive')

    if shape == 'SQUARE':
        return rectangle(sizeA, sizeA)

    if shape == 'RECTANGLE':
        return rectangle(sizeA, sizeB)

    if shape == 'CIRCLE':
        return circle(sizeA)

    raise Exception(f'Shape {shape} not defined')
```

3. 查看文件 `tests/test_dependencies.py` 中的测试，验证 `calculate_area` 的计算：

```python
from unittest import mock
from code.dependencies import calculate_area

def test_square():
    result = calculate_area('SQUARE', 2)
    assert result == 4

def test_rectangle():
    result = calculate_area('RECTANGLE', 2, 3)
    assert result == 6

def test_circle_with_proper_pi():
    result = calculate_area('CIRCLE', 2)
    assert result == 12.56636

@mock.patch('code.dependencies.PI', 3)
def test_circle_with_mocked_pi():
    result = calculate_area('CIRCLE', 2)
    assert result == 12

@mock.patch('code.dependencies.rectangle')
def test_circle_with_mocked_rectangle(mocked_rectangle):
    mocked_rectangle.return_value = 12
    result = calculate_area('SQUARE', 2)
    assert result == 12
    mocked_rectangle.assert_called()
```

4. 运行 pytest 来执行测试文件：

```
$ pytest tests/test_dependencies.py
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1,
pluggy-0.13.1
rootdir: Python-Automation-Cookbook/Chapter12
collected 5 items

tests/test_dependencies.py .....
[100%]

============================== 5 passed in 0.10s ===============================
```

### 工作原理...

代码模块的内容在步骤 1 中定义。文件 `dependencies.py` 包含以下元素：

- 变量 `PI` 的定义。
- 两个内部函数，分别计算矩形（基于两条边）和圆形（基于半径）的面积。
- `calculate_area` 函数，它接受多种形状并将请求传递给正确的内部函数。例如，它理解正方形是一种四条边相等的矩形。

在步骤 3 中，定义了测试文件。注意导入，包括 `mock` 模块和要测试的函数：

```python
from unittest import mock
from code.dependencies import calculate_area
```

前三个测试（`test_square`、`test_rectangle` 和 `test_circle_with_proper_pi`）是直接了当的。

测试 `test_circle_with_mocked_pi` 使用 `mock.patch` 装饰器将 `code.dependencies` 模块中的 `PI` 变量替换为 `3`：

```python
@mock.patch('code.dependencies.PI', 3)
def test_circle_with_mocked_pi():
    result = calculate_area('CIRCLE', 2)

    assert result == 12
```

这在测试运行期间更改了 `PI` 常量，从而影响了结果。一旦测试完成，模拟被禁用，变量恢复为之前定义的值。

测试 `test_circle_with_mocked_rectangle` 模拟了 `rectangle` 函数。由于装饰器中没有定义替代品，它作为参数 `mocked_rectangle` 传递给函数：

```python
@mock.patch('code.dependencies.rectangle')
def test_circle_with_mocked_rectangle(mocked_rectangle):
    mocked_rectangle.return_value = 12

    result = calculate_area('SQUARE', 2)

    assert result == 12
    mocked_rectangle.assert_called()
```

在测试期间，该函数被一个 `MagicMock` 对象替换。要指定此对象作为函数被调用时返回的值，请使用属性 `.return_value`。正如我们在结果中看到的，它替换了面积计算。也可以检查模拟是否已被调用，使用 `.assert_called()`。

在步骤 4 中，调用测试以查看所有测试是否按预期工作。

### 更多内容...

Mock 非常灵活，有多种方法可以检查它们是否被调用以及如何被调用。这里列出了一些可能性：

- `.assert_called_once()`：如果未被调用或被调用超过一次，则引发错误。
- `.assert_called_with(args)`：如果未被调用或使用不同的参数调用，则引发错误。参数将根据模拟的最后一次调用进行检查。
- `.call_count`：计算被调用的次数。

### 自动化测试流程

当访问或调用任何属性时，Mock对象会自动创建另一个模拟对象。例如：

```
>>> from unittest.mock import Mock
>>> mock = Mock()
>>> mock.attribute
<Mock name='mock.attribute' id='4337292000'>
>>> mock.other_attribute
<Mock name='mock.other_attribute' id='4337292144'>
>>> mock.other_attribute()
<Mock name='mock.other_attribute()' id='4337353728'>
```

这意味着模拟对象拥有灵活的API，能够轻松适应模块的大多数调用方式。

在我们的示例中，我们模拟了代码中的一个函数和一个常量，但你也可以模拟任何库，无论是Python标准库还是其他已安装的包。

> 请记住，你需要在正确的路径进行模拟。你需要模拟对象被导入的位置，而不是其原始定义的位置。例如，如果你在代码模块中使用 `from extpck import extfunction` 导入，你需要按如下方式模拟：

@mock.patch('code.module.extfunction')

这是一个容易犯的错误，即使是经验丰富的模拟用户也不例外。请记住，你需要模拟的是使用的位置，而不是定义的位置。

如果你需要在模拟对象被调用时引发异常，可以使用 `.side_effect` 属性来实现。这对于测试外部库的错误条件以及确保你的代码能正确处理这些错误非常有用：

```
>>> from unittest.mock import Mock
>>> mock = Mock()
>>> mock.side_effect = Exception('Custom error')
>>> mock()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/Cellar/python@3.8/3.8.2/Frameworks/Python.framework/Versions/3.8/lib/python3.8/unittest/mock.py", line 1081, in __call__
    return self._mock_call(*args, **kwargs)
  File "/usr/local/Cellar/python@3.8/3.8.2/Frameworks/Python.framework/Versions/3.8/lib/python3.8/unittest/mock.py", line 1085, in _mock_call
    return self._execute_mock_call(*args, **kwargs)
  File "/usr/local/Cellar/python@3.8/3.8.2/Frameworks/Python.framework/Versions/3.8/lib/python3.8/unittest/mock.py", line 1140, in _execute_mock_call
    raise effect
Exception: Custom error
```

同样的 `.side_effect` 属性，通过赋值一个迭代器，可以在模拟对象需要为多次调用返回不同结果时使用：

```
>>> from unittest.mock import Mock
>>> mock = Mock()
>>> mock.side_effect = (1, 2, 3)
>>> mock()
1
>>> mock()
2
>>> mock()
3
```

我们提到过 `mock.patch` 可以用作装饰器。这是Python的一个概念，本质上是修改它所装饰的函数，通常是在函数处理之前和/或之后添加额外的功能。

装饰器是一个非常有用的概念，本质上，它们用修改后的版本替换了原函数。你可以从这篇文章中更深入地了解装饰器的工作原理：https://medium.com/hasgeek/python-decorators-demystified-5ab4081fd0fe。

`patch` 也可以用作 `with` 代码块。如果是这种情况，模块在代码块内部会被模拟：

```
>>> from unittest.mock import patch
>>> from code.dependencies import circle
>>> with patch('code.dependencies.PI', 2):
...     print(circle(2))
...
8
>>> circle(2)
12.56636
```

你可以在 https://docs.python.org/3/library/unittest.mock.html 阅读关于模拟对象的完整文档。

### 另请参阅

- 本章前面的 *测试外部代码* 配方，了解如何测试其他模块中的代码。
- 接下来的 *使用HTTP调用模拟进行测试* 配方，了解如何使用模拟特定库的测试模块。

### 使用HTTP调用模拟进行测试

在测试中使用模拟对象是一项常见操作。一些依赖项通常在大多数测试中被模拟。

一个常见的需要模拟的依赖项是外部HTTP调用。在运行测试时执行这些调用成本高、速度慢，并且如果网络连接失败，可能会产生不可靠的结果。

虽然可以通过Python标准库中的 `mock` 库来模拟外部调用，如前面的 *使用依赖模拟进行测试* 配方所示，但也有特定的测试模块允许你模拟HTTP调用和响应。此外，还有特定的库可以模拟其他特定库。这会产生更简单、更好的模拟，因为它们适应了被模拟对象的行为。

我们之前使用过出色的 `requests` 库（在 *第1章，开始我们的自动化之旅* 的 *安装第三方包* 配方中介绍，但也在全书中使用）。我们将研究如何专门模拟这个库。我们将使用测试库 `responses`，它允许我们生成预期的请求及其响应。

> 请注意，这个测试库不是模拟通用的HTTP访问，而是专门模拟 `requests` 模块。

### 准备工作

我们将使用 `pytest` 模块，以及 `requests` 和 `responses` 库。我们应该安装这些模块，将它们添加到我们的 `requirements.txt` 文件中，如下所示：

```
$ echo "pytest==5.4.1" >> requirements.txt
$ echo "requests==2.23.0" >> requirements.txt
$ echo "responses==0.10.12" >> requirements.txt
$ pip install -r requirements.txt
```

我们将使用测试文件 tests/test_requests.py 和 code/code_requests.py。你可以从GitHub仓库 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter12/tests 下载它们，位于 tests 和 code 子目录下。

目录结构应如下所示：

```
├── code
│   ├── __init__.py
│   └── code_requests.py
├── conftest.py
└── tests
    └── test_requests.py
```

### 操作步骤...

1. __init__.py 文件和 conftest.py 文件是空的，但定义了模块的结构。
2. 查看文件 code/code_requests.py，其中包含一个对外部表单 https://httpbin.org/post 的调用，用于订购披萨：

```
import requests
from datetime import datetime, timedelta

RECIPES = {
    'DEFAULT': {
        'size': 'small',
        'topping': ['bacon', 'onion'],
    },
    'SPECIAL': {
        'size': 'large',
        'topping': ['bacon', 'mushroom', 'onion'],
    }
}

def order_pizza(recipe='DEFAULT'):
    delivery_time = datetime.now() + timedelta(hours=1)
    delivery = delivery_time.strftime('%H:%M')

    data = {
        'custname': "Sean O'Connell",
        'custtel': '123-456-789',
        'custemail': 'sean@oconnell.ie',
        # Indicate the time
        'delivery': delivery,
        'comments': ''
    }

    extra_info = RECIPES[recipe]
    data.update(extra_info)
    resp = requests.post('https://httpbin.org/post', data)
    return resp.json()['form']
```

3. 查看文件 tests/test_requests.py 中的测试，检查代码行为是否正确：

```
import pytest
import requests
import responses
import urllib.parse
from code.code_requests import order_pizza


@responses.activate
def test_order_pizza():
    body = {
        'form': {
            'size': 'small',
            'topping': ['bacon', 'onion']
        }
    }
    responses.add(responses.POST, 'https://httpbin.org/post',
                  json=body, status=200)
    result = order_pizza()
    assert result['size'] == 'small'
    # Decode the sent data
    encoded_body = responses.calls[0].request.body
    sent_data = urllib.parse.parse_qs(encoded_body)
    assert sent_data['size'] == ['small']


@responses.activate
def test_order_pizza_timeout():
    responses.add(responses.POST, 'https://httpbin.org/post',
                  body=requests.exceptions.Timeout())

    with pytest.raises(requests.exceptions.Timeout):
        order_pizza()
```

4. 运行 pytest 来执行测试文件：

```
$ pytest tests/test_requests.py
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1,
pluggy-0.13.1
rootdir: Python-Automation-Cookbook-Second-Edition/Chapter12
collected 2 items

tests/test_requests.py ..                                                [100%]

============================== 2 passed in 0.12s ===============================
```

### 工作原理...

代码模块的内容在步骤1中定义。文件 code_requests.py 包含以下元素：

- 一部分导入语句。
- 在 RECIPES 常量中定义了两种披萨。
- order_pizza 函数选择披萨，组合表单数据，然后将其POST到 https://httpbin.org/post。

### 自动化测试流程

> 在第3章《构建你的第一个网络爬虫应用》的*与表单交互*食谱中，我们介绍了 https://httpbin.org/forms/post 中呈现的表单及其POST提交地址 https://httpbin.org/post。该表单用于订购披萨，并以JSON格式返回表单中提交的相同信息。

让我们仔细看看 `order_pizza` 函数。

代码使用当前时间计算配送时间，增加一小时。时间使用 `delivery_time.strftime('%H:%M')` 格式化为 HH:MM 格式。`strftime` 函数用于格式化时间，字符串 '%H:%M' 仅打印时间的小时和分钟：

```
delivery_time = datetime.now() + timedelta(hours=1)
delivery = delivery_time.strftime('%H:%M')
```

完整数据由预定义的 `data` 字典和食谱信息组成。食谱信息从 `RECIPE` 字典中选取，然后使用 `.update()` 添加到 `data` 字典中：

```
data = {
    'custname': "Sean O'Connell",
    'custtel': '123-456-789',
    'custemail': 'sean@oconnell.ie',
    # 指定时间
    'delivery': delivery,
    'comments': ''
}
extra_info = RECIPES[recipe]
data.update(extra_info)
```

最后，使用 `requests.post` 将信息发送到URL。响应数据从JSON解码后，由函数返回：

```
resp = requests.post('https://httpbin.org/post', data)
return resp.json()['form']
```

在步骤3中，定义了测试文件。注意导入了 `responses` 模块。第一个测试 `test_order_pizza` 首先使用装饰器 `@responses.activate` 激活 `responses` 模块：

```
@responses.activate
def test_order_pizza():
    body = {
        'form': {
            'size': 'small',
            'topping': ['bacon', 'onion']
        }
    }
    responses.add(responses.POST, 'https://httpbin.org/post',
                   json=body, status=200)

    result = order_pizza()
    assert result['size'] == 'small'
    # 解码发送的数据
    encoded_body = responses.calls[0].request.body
    sent_data = urllib.parse.parse_qs(encoded_body)
    assert sent_data['size'] == ['small']
```

它首先定义预期的HTTP请求为 https://httpbin.org/post 以及应返回的响应：

```
body = {
    'form': {
        'size': 'small',
        'topping': ['bacon', 'onion']
    }
}
responses.add(responses.POST, 'https://httpbin.org/post',
               json=body, status=200)
```

对 `responses.add` 的调用指定了方法（POST）、URL、JSON格式的响应以及状态码。当我们的代码向给定URL发出请求时，它将接收此信息，而不是进行外部网络调用。

下一个代码块是调用 `order_pizza()` 并断言结果，这很直接。

之后，以下代码块检查由 `responses` 捕获并发送的数据：

```
# 解码发送的数据
encoded_body = responses.calls[0].request.body
sent_data = urllib.parse.parse_qs(encoded_body)
assert sent_data['size'] == ['small']
```

`responses` 库跟踪所有捕获的请求。我们检索第一个请求的主体并将其存储在 `encoded_body` 变量中。此数据已被编码并作为 `POST` 请求的一部分以 `application/x-www-form-urlencoded` 格式发送，这是 `POST` 请求的默认格式。我们使用默认库 `urllib` 和 `parse_qs()` 将其解码为字典。

> `urllib.parse` 的完整文档可在Python官方文档中找到：https://docs.python.org/3/library/urllib.parse.html。

第二个测试 `test_order_pizza_timeout` 展示了在请求特定URL时如何引发异常：

```
@responses.activate
def test_order_pizza_timeout():
    responses.add(responses.POST, 'https://httpbin.org/post',
                  body=requests.exceptions.Timeout())

    with pytest.raises(requests.exceptions.Timeout):
        order_pizza()
```

在这种情况下，`responses.add` 调用在主体中指定了一个 `Exception`，当使用定义的方法请求该URL时将引发该异常。

> 由于异常在请求完成时立即引发，这加速了 `Timeout` 异常的生成，而正常情况下这可能需要几秒或几分钟才能生成。

步骤4运行测试以检查代码是否按预期工作。

### 更多内容...

`responses` 库对于生成错误条件并为它们准备代码非常有用。`STATUS` 参数可用于在外部系统中生成错误代码，例如 "403 Forbidden"、"404 Not Found"、"500 Internal Server error" 和 "503 Service Unavailable"。正确处理可能产生的不同情况并对其做出反应将提高代码的可靠性。

> 请注意，其中一些错误可能在您这边没有任何更改的情况下发生，例如503。在某些情况下，等待并重试的策略可能是合适的，而在其他情况下，适当通知"外部服务不可用"可能更好。不要假设外部服务总是表现完美，因为它们可能（并且将会）出现问题。

一旦启用 `responses.activate` 来捕获所有HTTP请求，任何对意外URL的请求都会引发错误：

```
E       requests.exceptions.ConnectionError: Connection refused by Responses - the call doesn't match any registered mock.
E
E       Request:
E         - POST https://httpbin.org/otherurl
E
E       Available matches:
E         - POST https://httpbin.org/post
```

这使得每个使用 `responses` 的测试都是自包含的，并且意味着它们不会因错误或更改而泄漏任何外部调用。

`pytest` 插件 `pytest-responses` 允许自动为所有测试激活 responses。您可以在此处阅读其文档：https://github.com/getsentry/pytest-responses。

`responses` 库的完整文档可在此处找到：https://github.com/getsentry/responses。

`responses` 并不是唯一一个为测试时模拟某些特定方面而创建的库示例。另一个例子是 `freezegun`，它允许您设置时间。以这个测试为例：

```
import responses
import urllib.parse
from freezegun import freeze_time
from code.code_requests import order_pizza


@responses.activate
@freeze_time("2020-03-17T19:34")
def test_order_time():
    body = {
        'form': {
            'size': 'small',
            'topping': ['bacon', 'onion']
        }
    }
    responses.add(responses.POST, 'https://httpbin.org/post',
                   json=body, status=200)

    order_pizza()
    # 解码发送的数据
    encoded_body = responses.calls[0].request.body
    sent_data = urllib.parse.parse_qs(encoded_body)
    assert sent_data['delivery'] == ['20:34']
```

此测试使用 `freeze_time` 装饰器将时间设置为19:34，无论测试何时运行。请注意，它也设置了日期。

> 该测试在GitHub上以 `tests/test_requests_time.py` 的形式提供。记得使用 `pip` 安装 `freezegun` 包。

`freezegun` 的完整文档可在此处访问：https://github.com/spulec/freezegun。

### 另请参阅

- 本章前面的*测试外部代码*食谱，了解如何测试其他模块中的代码。
- 上一节的*使用依赖模拟进行测试*食谱，了解如何模拟任何类型的包或函数。

### 准备测试场景

测试通常是分批准备的。类似的测试需要类似的设置和清理，它们只在小细节上有所不同。一遍又一遍地重复相同的准备会产生样板代码，并且可读性较差。

> 术语 *样板* 源于19世纪当地报纸印刷由发行公司预先准备好的新闻，这些新闻被印在金属板上。这意味着相同的新闻，以相同的格式，在不同的报纸上重复出现。样板代码是重用的代码，几乎没有变化，在大多数情况下，主要增加混乱。在创建测试时，很容易陷入这种模式，这使得代码变得笨重。

在这个食谱中，我们将看到如何使用pytest夹具准备设置场景来运行测试。

### 准备工作

我们将使用pytest模块。我们应该安装模块，将它们添加到我们的requirements.txt文件中，如下所示：

```
$ echo "pytest==5.4.1" >> requirements.txt
$ pip install -r requirements.txt
```

我们将使用测试文件 tests/test_fixtures.py 和 code/code_fixtures.py。您可以从GitHub仓库 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter12/ 下载它们，位于 tests 和 code 子目录下：

目录结构应如下所示：

```
├── code
│   ├── __init__.py
│   └── code_fixtures.py
├── conftest.py
└── tests
    └── test_fixtures.py
```

### 操作步骤...

1. __init__.py 文件和 conftest.py 是空的，但定义了模块的结构。
2. 检查文件 code/code_fixtures.py，其中包含将数据存储到zip文件并检索的代码：

```
from zipfile import ZipFile
```

### 自动化测试流程

```python
INTERNAL_FILE = 'internal.txt'

def write_zipfile(filename, content):
    with ZipFile(filename, 'w') as zipfile:
        zipfile.writestr(INTERNAL_FILE, content)

def read_zipfile(filename):
    with ZipFile(filename, 'r') as zipfile:
        with zipfile.open(INTERNAL_FILE) as intfile:
            content = intfile.read()

    return content.decode('utf8')
```

3.  检查文件 `tests/test_fixtures.py` 中的测试，验证代码行为是否正确：

```python
import os
import random
import string
from pytest import fixture
from zipfile import ZipFile
from code.code_fixtures import write_zipfile, read_zipfile


@fixture
def fzipfile():
    content_length = 50
    content = ''.join(random.choices(string.ascii_lowercase,
                                     k=content_length))
    fnumber = ''.join(random.choices(string.digits, k=3))

    filename = f'file{fnumber}.zip'

    write_zipfile(filename, content)
    yield filename, content

    os.remove(filename)
```

```python
def test_writeread_zipfile():
    TESTFILE = 'test.zip'
    TESTCONTENT = 'This is a test'
    write_zipfile(TESTFILE, TESTCONTENT)
    content = read_zipfile(TESTFILE)

    assert TESTCONTENT == content
```

```python
def test_readwrite_zipfile(fzipfile):
    filename, expected_content = fzipfile
    content = read_zipfile(filename)

    assert content == expected_content
```

```python
def test_internal_zipfile(fzipfile):
    filename, expected_content = fzipfile
    EXPECTED_LIST = ['internal.txt']

    # Verify only a single file exist in the zipfile
    with ZipFile(filename, 'r') as zipfile:
        assert zipfile.namelist() == EXPECTED_LIST
```

4.  运行 pytest 来执行测试文件：

```
$ pytest tests/test_fixtures.py
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1,
pluggy-0.13.1
rootdir: /Users/jaime/Dropbox/code/Packt/Python-Automation-
Cookbook/Chapter12New
collected 3 items

tests/test_fixtures.py ...
[100%]

============================== 3 passed in 0.01s ===============================
```

5.  检查目录中是否已创建名为 `test.zip` 的新文件：

```
$ ls test.zip
test.zip
```

### 工作原理...

`code` 模块的内容在 *步骤 1* 中定义。

如 *步骤 2* 所述，文件 `code_fixtures.py` 包含两个函数，用于在 zip 文件中存储和保存信息。这使用了标准的 Python `zipfile` 模块来处理 zip 文件：

`write_zipfile` 创建一个包含内部压缩文件 `internal.txt` 的 zip 文件，该文件包含作为参数传递的数据。这是使用 `.writestr()` 方法写入的：

```python
def write_zipfile(filename, content):
    with ZipFile(filename, 'w') as zipfile:
        zipfile.writestr(INTERNAL_FILE, content)
```

`read_zipfile` 读取文件并从内部文件中提取内容：

```python
def read_zipfile(filename):
    with ZipFile(filename, 'r') as zipfile:
        with zipfile.open(INTERNAL_FILE) as intfile:
            content = intfile.read()
    return content.decode('utf8')
```

`read_zipfile` 遵循与写入相同的模式。它在 zip 文件中搜索定义的内部文件并读取其内容。内容需要解码，因为它以 UTF-8 编码。

Zip 文件是压缩文件的集合。我们需要定义至少一个文件来在 zip 文件中存储信息。你可以在此处了解更多关于 `zipfile` 模块的信息：https://docs.python.org/3/library/zipfile.html。

测试在步骤 3 中定义。测试 `test_writeread_zipfile` 首先生成一个文件，然后读取它，测试整个生命周期并确保文件可以被写入然后读取：

```python
def test_writeread_zipfile():
    TESTFILE = 'test.zip'
    TESTCONTENT = 'This is a test'
    write_zipfile(TESTFILE, TESTCONTENT)
    content = read_zipfile(TESTFILE)

    assert TESTCONTENT == content
```

此测试工作正常，但不执行任何清理操作，这会将文件 `test.zip` 留在工作目录中。

对于另外两个测试，使用声明的 fixture `fzipfile`。注意 `@fixture` 装饰器。让我们看看它：

```python
import os
import random
import string
from pytest import fixture

@fixture
def fzipfile():
    content_length = 50
    content = ''.join(random.choices(string.ascii_lowercase,
                                     k=content_length))
    fnumber = ''.join(random.choices(string.digits, k=3))

    filename = f'file{fnumber}.zip'

    write_zipfile(filename, content)
    yield filename, content

    os.remove(filename)
```

该 fixture 生成一些随机内容，形式为 50 个小写字符的字符串。它还生成一个随机文件名。然后它写入这个文件，并 yield 生成的文件名和内容。最后，它通过调用 `os.remove` 删除文件。

### 自动化测试流程

Python 中的 `yield` 关键字允许我们暂停代码执行，返回一个值，然后恢复代码。在 fixture 中，`yield` 之前的所有内容都将在测试开始前执行，`yield` 之后的所有内容都将在测试结束时执行。

> `yield` 通常在充当迭代器的生成器函数中使用，但每次请求新值时，代码会继续执行直到找到另一个 `yield`；例如：

```python
>>> def generator():
...     yield 1
...     yield 2
...     for _ in range(3):
...         yield 3
...
>>> list(generator())
[1, 2, 3, 3, 3]
```

你可以在 https://realpython.com/introduction-to-python-generators/ 了解更多关于生成器的信息。

然后 fixture 执行一些设置（创建一个 zip 文件），返回要使用的值（随机生成的名称和内容），最后进行清理（删除文件）。

另外两个测试使用此 fixture 来设置测试。`test_readwrite_zipfile` 非常直接；它读取由 fixture 创建的文件：

```python
def test_readwrite_zipfile(fzipfile):
    filename, expected_content = fzipfile
    content = read_zipfile(filename)

    assert content == expected_content
```

请注意，fixture `fzipfile` 返回的值是一个包含两个元素的元组：文件名和内容。

`test_internal_zipfile` 使用该 fixture 来检查 zip 文件中是否只有一个文件，并且文件名正确。它打开 zip 文件并使用 `.namelist()` 获取内部文件列表以进行验证：

```python
def test_internal_zipfile(fzipfile):
    filename, expected_content = fzipfile
    EXPECTED_LIST = ['internal.txt']

    # Verify only a single file exist in the zipfile
    with ZipFile(filename, 'r') as zipfile:
        assert zipfile.namelist() == EXPECTED_LIST
```

### 更多内容...

Fixture 可以在多个测试文件之间共享。你不需要导入它们。相反，将它们添加到 `conftest.py` 文件中。你可以在子目录中生成本地的 `conftest.py` 文件，用于仅对该目录本地的 fixture。

> 要使用 fixture，测试中的参数名称需要与 fixture 的名称相同。注意不要无意中覆盖这些名称。

`pytest` 提供了一些内置的 fixture，可用于常见操作。例如，fixture `caplog` 捕获运行测试时发出的日志，而 `tmp_path` 创建一个每个测试唯一的临时子目录。有关更多详细信息，请访问 fixture 文档。

有很多 `pytest` 插件包含用于处理许多常见场景或工具的 fixture；例如，连接数据库、Web 框架、外部 API 等。在创建 fixture 之前，值得在 `pypi.org` 上进行搜索，或者查看 `http://plugincompat.herokuapp.com/` 上的非详尽 `pytest` 插件列表。

当测试失败时，`pytest` 还会为 fixture 上的特定数据添加上下文。这有助于理解行为和调试问题，无论是在 fixture 中还是在被测试的代码中：

```
$ pytest tests/test_fixtures.py
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /Users/jaime/Dropbox/code/Packt/Python-Automation-Cookbook Second-Edition/Chapter12
collected 3 items

tests/test_fixtures.py ..F                                               [100%]

================================= FAILURES ==================================
```

自动测试例程

### test_internal_zipfile

```python
fzipfile = ('file989.zip',
            'ldrnfqwqodcwmkxkehxcaaxzaocxasbduixouchvrzqyfgaaxv')
```

失败

请记住在你的测试夹具中执行清理阶段。正如我们在 `test_writeread_zipfile` 示例中看到的，不进行清理可能导致虚假数据被存储在硬盘驱动器或测试夹具创建虚假数据的任何其他位置。将代码放在 `yield` 关键字之后的夹具中，即使发生异常或其他类型的错误，也一定会被调用。

完整的 `pytest` 夹具文档可在此处获取：https://docs.pytest.org/en/latest/fixture.html。

### 另请参阅

- 本章前面的 *编写和执行测试用例* 配方，以了解定义测试的基础知识。
- 本章前面的 *测试外部代码* 配方，以了解如何测试其他模块中的代码。

### 选择性运行测试

检测并运行项目中所有定义的测试对于验证一切是否正常工作是很好的。但在处理测试时完成的大部分开发工作，仅执行所有测试的一个子集会更有益。

在添加新代码或新测试时，快速迭代测试和代码的特定部分以缩小关注范围至关重要。

在本配方中，我们将了解如何使用 `pytest` 运行可用测试的一个子集，以及在不同场景下应使用哪些参数。

### 准备工作

我们将使用 `pytest` 模块以及其他模块。我们应该通过将它们添加到我们的 `requirements.txt` 文件中来安装模块，如下所示：

```
$ echo "pytest==5.4.1" >> requirements.txt
$ echo "requests==2.23.0" >> requirements.txt
$ echo "responses==0.10.12" >> requirements.txt
$ echo "freezegun==0.3.15" >> requirements.txt
$ pip install -r requirements.txt
```

我们将使用本章前面配方中介绍的测试文件。你可以从 GitHub 仓库 https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/tree/master/Chapter12 下载它们，位于 tests 和 code 子目录下：

```
├── code
│   ├── __init__.py
│   ├── code_fixtures.py
│   ├── code_requests.py
│   ├── dependencies.py
│   └── external.py
├── conftest.py
└── tests
    ├── test_case.py
    ├── test_dependencies.py
    ├── test_external.py
    ├── test_fixtures.py
    ├── test_requests.py
    └── test_requests_time.py
```

### 如何操作...

1. 使用 pytest 运行所有测试：

```
$ pytest
============================= test session starts ==============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /Python-Automation-Cookbook-second-Edition/Chapter12
collected 18 items

tests/test_case.py ...F
[ 22%]
tests/test_dependencies.py .....
[ 50%]
tests/test_external.py ...
[ 66%]
tests/test_fixtures.py ...
[ 83%]
tests/test_requests.py ..
[ 94%]
tests/test_requests_time.py .
[100%]

============================= FAILURES =============================
_____________________________ test_fail _____________________________

>   ???
E   assert 4 in [1, 2, 3]

/Python-Automation-Cookbook-Second-Edition/Chapter12/tests/test_case.py:18: AssertionError
========================= short test summary info ======================
FAILED tests/test_case.py::test_fail - assert 4 in [1, 2, 3]
========================= 1 failed, 17 passed in 0.28s =================
```

请注意 `test_fail` 失败了。

2. 运行 `pytest --collect-only`：

```
$ pytest --collect-only
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /Python-Automation-Cookbook-Second-Edition/Chapter12
collected 18 items
<Module tests/test_case.py>
  <Function test_one>
  <Function test_two>
  <Function test_three>
  <Function test_fail>
<Module tests/test_dependencies.py>
  <Function test_square>
  <Function test_rectangle>
  <Function test_circle_with_proper_pi>
  <Function test_circle_with_mocked_pi>
  <Function test_circle_with_mocked_rectangle>
<Module tests/test_external.py>
    <Function test_int_division>
    <Function test_float_division>
    <Function test_division_by_zero>
<Module tests/test_fixtures.py>
    <Function test_writeread_zipfile>
    <Function test_readwrite_zipfile>
    <Function test_internal_zipfile>
<Module tests/test_requests.py>
    <Function test_order_pizza>
    <Function test_order_pizza_timeout>
<Module tests/test_requests_time.py>
    <Function test_order_time>

========================= no tests ran in 0.23s ==========================
```

3. 运行 pytest -v -k time：

```
$ pytest -v -k time
============================= test session starts ==============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1,
pluggy-0.13.1 -- /usr/local/opt/python@3.8/bin/python3.8
cachedir: .pytest_cache
rootdir: /Python-Automation-Cookbook-Second-Edition/Chapter12
collected 18 items / 16 deselected / 2 selected

tests/test_requests.py::test_order_pizza_timeout PASSED                     [ 50%]
tests/test_requests_time.py::test_order_time PASSED                         [100%]

========================= 2 passed, 16 deselected in 0.22s ==========================
```

4. 运行 pytest -v -k time tests/test_requests.py：

```
$ pytest -v -k time tests/test_requests.py
============================= test session starts ==============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1,
pluggy-0.13.1 -- /usr/local/opt/python@3.8/bin/python3.8
cachedir: .pytest_cache
rootdir: /Python-Automation-Cookbook-Second-Edition/Chapter12
collected 2 items / 1 deselected / 1 selected

tests/test_requests.py::test_order_pizza_timeout PASSED
[100%]

================= 1 passed, 1 deselected in 0.10s =================
```

5. 运行 pytest -v tests/test_requests_time.py::test_order_time：

```
$ pytest -v tests/test_requests_time.py::test_order_time
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1,
pluggy-0.13.1 -- /usr/local/opt/python@3.8/bin/python3.8
cachedir: .pytest_cache
rootdir: /Python-Automation-Cookbook-Second-Edition/Chapter12
collected 1 item

tests/test_requests_time.py::test_order_time PASSED
[100%]

========================== 1 passed in 0.38s ===========================
```

6. 运行 pytest -v --lf：

```
$ pytest -v --lf
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1,
pluggy-0.13.1 -- /usr/local/opt/python@3.8/bin/python3.8
cachedir: .pytest_cache
rootdir: /Python-Automation-Cookbook-Second-Edition/Chapter12
collected 11 items / 10 deselected / 1 selected
run-last-failure: rerun previous 1 failure (skipped 3 files)

tests/test_case.py::test_fail FAILED
[100%]

=============================== FAILURES ===============================
_______________________________ test_fail ________________________________

def test_fail():
    >       assert 4 in LIST
    E       assert 4 in [1, 2, 3]

tests/test_case.py:18: AssertionError
========================= short test summary info ==========================
FAILED tests/test_case.py::test_fail - assert 4 in [1, 2, 3]
========================= 1 failed, 10 deselected in 0.30s =================
```

### 工作原理...

在步骤 1 中，调用 pytest 时没有参数。这将运行子目录下定义的所有测试。

请注意它显示收集了多少测试：

```
collected 18 items
```

步骤 2 展示了如何获取收集到的测试列表，但不运行它们。这是通过 --collect-only 参数完成的。它会展示所有测试但不运行它们。

要仅运行某些测试，步骤 3 展示了如何生成一个匹配字符串来限制要运行的测试。使用 -k 允许按字符串匹配，因此所有匹配字符串 time 的测试都将被执行。

> 选项 -v 使输出详细化，并为每个测试显示一行，而不是每个测试文件一行。

请注意测试总数仍然存在。pytest 收集了所有测试，但随后只运行匹配 -k 参数的那些：

```
collected 18 items / 16 deselected / 2 selected
```

要限制收集的测试数量，可以指定文件路径，如步骤 4 所示，其中将特定文件的路径添加到命令中。在这种情况下，收集的测试数量仅限于位于该文件路径中的测试：

```
$ pytest -v -k time tests/test_requests.py
...
collected 2 items / 1 deselected / 1 selected
```

### 自动化测试流程

第5步展示了如何使用完整文件路径和描述符来指定单个测试。当使用 `-v` 参数时，这很容易从输出中复制：

```
$ pytest -v tests/test_requests_time.py::test_order_time
...
collected 1 item
```

最后，在第6步中，使用了不同的参数 `--lf`。这只会运行上次失败的测试：

```
$ pytest -v --lf
...
collected 11 items / 10 deselected / 1 selected
run-last-failure: rerun previous 1 failure (skipped 3 files)
```

如果上次运行没有测试失败，`--lf` 将运行所有测试。

### 更多内容...

请记住，所有这些参数都可以组合使用。例如，`--collect-only` 可以与 `-k` 一起使用，以检查所选测试是否正确。

同样，可以添加多个文件路径和测试。例如：

```
$ pytest -v tests/test_external.py::test_int_division tests/test_requests.py
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1, pluggy-0.13.1 -- /usr/local/opt/python@3.8/bin/python3.8
cachedir: .pytest_cache
rootdir: /Python-Automation-Cookbook-Second-Edition/Chapter12
collected 3 items
tests/test_external.py::test_int_division PASSED                       [ 33%]
tests/test_requests.py::test_order_pizza PASSED                        [ 66%]
tests/test_requests.py::test_order_pizza_timeout PASSED                [100%]
============================== 3 passed in 0.11s ===============================
```

当测试数量增长时，收集测试可能需要相当长的时间。当测试数量达到数百或数千个时，使用 `-k` 进行收集和过滤可能需要很长时间，以至于延迟测试的执行（半分钟或更长时间）。因此，了解如何通过指定要收集测试的文件来减少收集的测试数量非常重要。

要在检测到失败时停止测试执行，请使用参数 `-x`：

```
$ pytest -v -x
============================= test session starts =============================
platform darwin -- Python 3.8.2, pytest-5.4.1, py-1.8.1, pluggy-0.13.1 -- /usr/local/opt/python@3.8/bin/python3.8
cachedir: .pytest_cache
rootdir: /Python-Automation-Cookbook-Second-Edition/Chapter12
collected 18 items

tests/test_case.py::test_one PASSED                                   [  5%]
tests/test_case.py::test_two PASSED                                   [ 11%]
tests/test_case.py::test_three PASSED                                 [ 16%]
tests/test_case.py::test_fail FAILED                                  [ 22%]

============================== FAILURES =================____________
______________________________ test_fail _______________________________

    def test_fail():
>       assert 4 in LIST
E       assert 4 in [1, 2, 3]

tests/test_case.py:18: AssertionError
========================= short test summary info ======================
FAILED tests/test_case.py::test_fail - assert 4 in [1, 2, 3]
!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
======================== 1 failed, 3 passed in 0.38s =========================
```

在开发过程中，通常会对一个或少数几个测试进行迭代以修复它们。在这些情况下，`--lf` 可以极大地帮助一次性运行整个测试套件，然后仅重复运行失败的测试。随着代码的修复，通过的测试将从执行组中移除，直到最后一个测试通过。然后，`--lf` 将再次运行所有测试。

> 当测试套件中的测试数量较少时，这些建议大多没有必要，因为测试会非常快。随着测试数量的增加，使用这些参数变得更加重要，以避免浪费时间执行与正在开发的代码特定部分没有直接关系的测试。只需记住在最后运行一次整个套件，以确保您的更改没有对代码的意外部分产生破坏性影响。实际上，这种情况发生的次数比开发者希望的要多！

### 另请参阅

- 本章前面的 *编写和执行测试用例*，了解如何定义测试的基础知识。
- 本章前面的 *测试外部代码*，了解如何测试其他模块中的代码。

# 13 调试技术

在本章中，我们将介绍以下内容：

- 学习 Python 解释器基础
- 通过日志记录进行调试
- 使用断点进行调试
- 提高您的调试技能

### 简介

编写代码并不容易。实际上，它非常困难。即使是世界上最好的程序员也无法预见代码的所有可能替代方案和流程。

这意味着执行我们的代码总会产生意外和意想不到的行为。有些会非常明显，而有些则非常微妙，但识别和消除代码中这些缺陷的能力对于构建可靠的软件至关重要。

软件中的这些缺陷被称为 **bug**，因此消除它们被称为 **debugging**。

仅仅通过阅读代码并推理其执行图像来检查代码，永远不足以涵盖非平凡代码的所有可能结果。总会有意外，复杂的代码很难跟踪。这就是为什么通过停止执行并查看当前状态来进行调试的能力很重要。

> 每个人，我是说每个人，都会在代码中引入 bug，然后在之后对它们感到惊讶。有些人将调试描述为在犯罪电影中扮演侦探，而你自己也是凶手。

任何调试过程大致遵循以下路径：

1.  你意识到有一个问题。
2.  你理解正确的行为应该是什么。
3.  你发现为什么当前代码会产生 bug。
4.  你更改代码以产生正确的结果。

95% 的时间里，除了 *第3步* 之外的一切都很直接。*第3步* 是调试过程的主体。

理解 bug 的 *为什么*，其核心遵循科学方法：

1.  测量和观察代码正在做什么。
2.  对为什么会产生这种情况提出假设。
3.  通过专门设计的实验（例如，测试）或检查测试的执行（可视为自然实验）来验证或反驳假设。
4.  使用所得信息来修复 bug 或迭代该过程。

调试是一项技能，因此它会随着时间的推移而提高。实践在培养识别错误的直觉方面起着重要作用，但有一些通用的想法可能对你有所帮助：

-   **分而治之：** 隔离代码的小部分，以便能够理解代码。尽可能简化问题。

> 这种方法有一种形式叫做 **Wolf fence 算法**，由 Eduard Gauss 描述：
>
> “阿拉斯加有一只狼；你怎么找到它？首先，在该州中间建一道围栏，等待狼嚎叫，然后确定它在围栏的哪一边。仅在该侧重复此过程，直到你能看到狼。”

-   **从错误向后追溯：** 如果在特定点有一个明确的错误，那么 bug 很可能位于其周围。从错误开始逐步向后追溯，沿着轨迹直到找到错误的来源。

-   **你可以假设任何你想要的东西，只要证明你的假设是正确的：** 代码非常复杂，无法一次性全部记在脑子里。你需要验证小的假设，当它们组合在一起时，将为检测和修复问题提供坚实的基础。进行小实验，让你能够排除代码中工作的部分，并专注于未测试的部分。

> 或者，用 Sherlock Holmes 的话说：
> *“一旦你排除了不可能，剩下的无论多么不可能，都一定是真相。”*

记住要证明你做出的任何假设。避免未经证实的假设，因为它们会使你偏离 bug 的位置。很容易认为错误发生在代码的一部分，却去看另一部分。

> 这听起来可能有点吓人，但大多数 bug 都相当明显。也许是一个拼写错误，或者一段代码没有为特定值做好准备。尽量保持事情简单。简单的代码更容易分析和调试。

查看 *第12章，自动化测试流程*，了解如何使用测试。测试是帮助你调试、发现问题和添加验证点的绝佳工具。在定义的测试中调试代码允许你创建一个小型环境，你可以专注于预期的输入和输出，并搜索 bug。

在本章中，我们将研究几种调试工具和技术，并将其专门应用于 Python 脚本。这些脚本将包含一些 bug，我们将在本教程中修复它们。

### 学习 Python 解释器基础

在本教程中，我们将介绍 Python 的一些内置功能，用于检查代码、调查发生了什么以及检测何时行为不正常。

> 我们也可以验证事情是否按预期工作。请记住，能够排除代码的一部分作为 bug 的来源是极其重要的。

### 调试技术

在调试过程中，我们通常需要分析来自外部模块或服务的未知元素和对象。Python 中的代码在执行过程中的任何时刻都具有高度的可发现性。这种在代码执行时检查其类型和属性的能力被称为*内省*。

本节中的所有内容都默认包含在 Python 解释器中。

### 如何操作...

1.  导入 pprint：
    ```python
    >>> from pprint import pprint
    ```

2.  创建一个名为 dictionary 的新字典：
    ```python
    >>> dictionary = {'example': 1}
    ```

3.  显示此环境中的全局变量：
    ```python
    >>> globals()
    {...'pprint': <function pprint at 0x100995048>,
    ...'dictionary': {'example': 1}}
    ```

4.  使用 pprint 以可读格式打印全局变量字典：
    ```python
    >>> pprint(globals())
    {'__annotations__': {},
    ...
    'dictionary': {'example': 1},
    'pprint': <function pprint at 0x100995048>}
    ```

5.  显示 dictionary 的所有属性：
    ```python
    >>> dir(dictionary)
    ['__class__', '__contains__', '__delattr__', '__delitem__',
    '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
    '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__',
    '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__',
    '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
    '__reversed__', '__setattr__', '__setitem__', '__sizeof__',
    '__str__', '__subclasshook__', 'clear', 'copy', 'fromkeys', 'get',
    'items', 'keys', 'pop', 'popitem', 'setdefault', 'update',
    'values']
    ```

6.  显示 dictionary 对象的帮助信息：
    ```python
    >>> help(dictionary)
    Help on dict object:

    class dict(object)
     |  dict() -> new empty dictionary
     |  dict(mapping) -> new dictionary initialized from a mapping
     |                     object's
     |                     (key, value) pairs
     |  ...
    ```

### 工作原理...

在步骤 1 中导入 pprint（漂亮打印）后，我们在步骤 2 中创建了一个新的字典进行操作。

步骤 3 展示了全局命名空间包含的内容，其中包括定义的字典和模块。globals() 显示所有导入的模块和其他全局变量。

> 对于局部命名空间，有一个等效的 locals()。

pprint 在步骤 4 中帮助以更可读的格式显示全局变量，增加了更多空格并按行分隔元素。

步骤 5 展示了如何使用 dir() 获取 Python 对象的所有属性名称。请注意，这包括所有双下划线值，例如 __len__。

使用内置的 help() 函数将显示对象的相关信息。

### 更多内容...

特别是，dir() 对于检查未知对象、模块或类非常有用。如果你需要过滤掉默认属性并澄清输出，可以使用列表推导式过滤输出：

```python
>>> [att for att in dir(dictionary) if not att.startswith('__')]
['clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem',
'setdefault', 'update', 'values']
```

同样，如果你正在搜索特定方法（例如以 set 开头的方法），也可以这样做。

help() 将显示函数或类的文档字符串。文档字符串是定义后紧跟的字符串，用于记录函数或类：

```python
>>> def something():
...     '''
...     This is help for something
...     '''
...     pass
...
>>> help(something)
Help on function something in module __main__:

something()
    This is help for something
```

注意 "This is help for something" 字符串是如何在函数定义后显示的。

> 文档字符串通常用三引号括起来，以允许编写多行字符串。Python 会将三引号内的所有内容视为一个大字符串，即使有换行符。你可以使用 " 或 ' 字符，只要使用三个即可。你可以在 https://www.python.org/dev/peps/pep-0257/ 找到更多关于文档字符串的信息。

内置函数的文档可以在 https://docs.python.org/3/library/functions.html#built-in-functions 找到，而 pprint 的完整文档可以在 https://docs.python.org/3/library/pprint.html 找到。

### 另请参阅

-   本章后面的“提升你的调试技能”部分，以获取更多调试工具。
-   本章接下来的“通过日志记录进行调试”部分，学习如何通过设置跟踪来调试元素。

### 通过日志记录进行调试

调试，归根结底，是检测程序内部发生了什么，并找出可能发生的意外或不正确的效果。一种简单但非常有效的方法是在代码的战略部分输出变量和其他信息，以便程序员跟踪程序的流程。

这种方法最简单的形式被称为**打印调试**。这种技术包括在某些点插入打印语句，以在调试时打印变量的值或点。

但将这种技术更进一步，并结合*第 2 章，轻松自动化任务*中介绍的日志技术，允许我们创建程序执行的跟踪。这种跟踪信息在检测运行程序中的问题时非常有用。日志通常也在使用测试框架运行测试时显示。

> *第 12 章，自动测试例程*中介绍的 pytest 会自动显示失败测试的日志。其他测试框架可能需要配置。日志记录在本书的*第 2 章，轻松自动化任务*中介绍。

### 准备工作

从 GitHub 下载 `debug_logging.py` 文件：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter13/debug_logging.py。

这包含一个冒泡排序算法（https://www.studytonight.com/data-structures/bubble-sort）的实现，这是对元素列表进行排序的最简单方法之一。它多次遍历列表，在每次迭代中，检查两个相邻的值并交换它们，使较大的值在较小的值之后。这使得较大的值像气泡一样在列表中上升。

> 冒泡排序是一种简单但朴素的排序实现方式，有更好的替代方案。除非你有极好的理由不这样做，否则请依赖列表中的标准 `.sort` 方法。

运行时，它检查以下列表以验证其正确性：

```python
assert [1, 2, 3, 4, 7, 10] == bubble_sort([3, 7, 10, 2, 4, 1])
```

这个实现中有一个错误，所以我们可以在本节中修复它！

### 如何操作...

1.  运行 `debug_logging.py` 脚本并检查是否失败：
    ```
    $ python debug_logging.py
    INFO:Sorting the list: [3, 7, 10, 2, 4, 1]
    INFO:Sorted list:      [2, 3, 4, 7, 10, 1]
    Traceback (most recent call last):
      File "debug_logging.py", line 17, in <module>
        assert [1, 2, 3, 4, 7, 10] == bubble_sort([3, 7, 10, 2, 4, 1])
    AssertionError
    ```

2.  通过更改 `debug_logging.py` 脚本的第二行来启用调试日志：
    ```python
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO)
    ```
    将前面的行更改为以下行：
    ```python
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)
    ```
    注意不同的级别。

3.  再次运行脚本，包含更多信息：
    ```
    $ python debug_logging.py
    INFO:Sorting the list: [3, 7, 10, 2, 4, 1]
    DEBUG:alist: [3, 7, 10, 2, 4, 1]
    DEBUG:alist: [3, 7, 10, 2, 4, 1]
    DEBUG:alist: [3, 7, 2, 10, 4, 1]
    DEBUG:alist: [3, 7, 2, 4, 10, 1]
    DEBUG:alist: [3, 7, 2, 4, 10, 1]
    DEBUG:alist: [3, 2, 7, 4, 10, 1]
    DEBUG:alist: [3, 2, 4, 7, 10, 1]
    DEBUG:alist: [2, 3, 4, 7, 10, 1]
    DEBUG:alist: [2, 3, 4, 7, 10, 1]
    DEBUG:alist: [2, 3, 4, 7, 10, 1]
    INFO:Sorted list : [2, 3, 4, 7, 10, 1]
    Traceback (most recent call last):
    ```

### 调试技术

### 工作原理...

步骤1展示了脚本，并指出代码存在缺陷，因为它未能正确排序列表。

该脚本已包含一些日志，用于显示起始和最终结果，以及一些调试日志，用于显示每个中间步骤。在步骤2中，我们激活了DEBUG日志的显示，因为在步骤1中只显示了INFO日志。

> 请注意，日志默认显示在标准错误输出中。这在终端中默认显示。如果需要将日志定向到其他位置，例如文件，可以配置不同的处理程序。有关更多详细信息，请参阅Python中的日志记录配置：https://docs.python.org/3/howto/logging.html。

步骤3再次运行脚本，这次显示了额外信息，表明列表中的最后一个元素未排序。

该错误是一个差一错误，这是一种非常常见的错误，因为它应该迭代到列表的整个大小。这在步骤4中得到了修复。

> 检查代码以理解为什么会出现错误。应该比较整个列表，但我们错误地将大小减小了1。

步骤5显示修复后的脚本运行正确。

### 更多内容...

在本教程中，我们预先战略性地放置了调试日志，但在实际调试练习中可能并非如此。作为错误调查的一部分，你可能需要添加更多日志或更改位置。

这种技术的最大优势在于，我们能够看到程序的流程，能够检查从一个代码执行块到下一个块的输出，并理解流程。缺点是我们最终可能会得到一堆文本，而这些文本并未提供关于问题的具体信息。你需要在信息过多和过少之间找到平衡。

如果需要，可以详细记录，但为了减少混乱，尽量避免冗长且令人困惑的日志。保持文本尽可能简短且具有描述性。保持每个日志简洁，因为如果需要，你总是可以创建更多日志。

记得在修复错误后降低日志级别。之后你可能需要删除一些日志，因为它们从长远来看不会有用。

> 这种技术的快速而粗糙的版本是添加打印语句而不是调试日志。虽然有些人对此持保留态度，但它是一种用于调试目的的有价值的技术。但请记住在完成后清理它们。

所有内省工具在生成日志时都可用，因此你可以创建显示例如调用`dir(object)`对象所有结果的日志：

```
logging.debug(f'object {dir(object)}')
```

任何可以显示为字符串的内容都可以在日志中呈现。

### 另请参阅

- 本章前面的*学习Python解释器基础*教程，了解Python内省工具的基础知识。
- 本章后面的*提高调试技能*教程，查看涵盖不同问题的完整调试示例。

### 使用断点调试

Python有一个内置的调试器，称为`pdb`。通过设置断点，可以在任何点停止代码的执行。断点将跳转到命令行模式。从命令行，可以分析当前状态。鉴于Python是解释型的，可以从这个阶段执行任何新代码。这非常灵活，允许你创建灵活的断点并更改当前状态以分析程序的行为。

让我们看看如何操作。

### 准备工作

从GitHub下载`debug_algorithm.py`脚本：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter13/debug_algorithm.py。

在下一节中，我们将详细分析代码的执行。该代码检查数字是否满足某些条件：

```
def valid(candidate):
    if candidate <= 1:
        return False

    lower = candidate - 1
    while lower > 1:
        if candidate / lower == candidate // lower:
            return False
        lower -= 1
    return True

assert not valid(1)
assert valid(3)
assert not valid(15)
assert not valid(18)
assert not valid(50)
assert valid(53)
```

你可能已经认出代码的功能，但请耐心等待，以便我们可以交互式地分析它。

### 操作步骤...

1. 运行代码以查看所有断言是否有效：
   ```
   $ python debug_algorithm.py
   ```
2. 在`while`语句之后，第7行之前添加`breakpoint()`，结果如下：
   ```
   while lower > 1:
       breakpoint()
       if candidate / lower == candidate // lower:
   ```
3. 再次执行代码，看到它在断点处停止，进入交互式Pdb模式：
   ```
   $ python debug_algorithm.py
   > .../debug_algorithm.py(8)valid()
   -> if candidate / lower == candidate // lower:
   (Pdb)
   ```
4. 检查`candidate`的值和两个操作。此行检查将`candidate`除以`lower`是否为整数（浮点数和整数除法相同）：
   ```
   (Pdb) candidate
   3
   (Pdb) candidate / lower
   1.5
   (Pdb) candidate // lower
   1
   ```
5. 使用`n`继续到下一条指令。检查它是否结束`while`循环并返回`True`：
   ```
   (Pdb) n
   > ...debug_algorithm.py(10)valid()
   -> lower -= 1
   (Pdb) n
   > ...debug_algorithm.py(6)valid()
   -> while lower > 1:
   (Pdb) n
   > ...debug_algorithm.py(12)valid()
   -> return True
   (Pdb) n
   --Return--
   > ...debug_algorithm.py(12)valid()->True
   -> return True
   ```
6. 使用`c`继续执行直到找到另一个断点。注意这是下一次调用`valid()`，输入为15：
   ```
   (Pdb) c
   > ...debug_algorithm.py(8)valid()
   -> if candidate / lower == candidate // lower:
   (Pdb) candidate
   15
   (Pdb) lower
   14
   ```
7. 继续运行并检查数字，直到理解`valid`函数的功能。你能找出代码的功能吗？（如果不能，请不要担心，查看下一节。）完成后，使用`q`退出。这会停止执行：
   ```
   (Pdb) q
   ...
   bdb.BdbQuit
   ```

### 工作原理...

正如你可能已经知道的，该代码正在检查一个数字是否为质数。它尝试将该数字除以所有小于它的整数。如果在任何点，该数字能被其中任何一个整除，则返回`False`，因为它不是质数。

> 这实际上是一种非常低效的检查质数的方法，因为它处理大数字时会花费很长时间。不过，对于我们的教学目的来说，它足够快。如果你对寻找质数感兴趣，可以查看数学包，如SymPy（https://docs.sympy.org/latest/modules/ntheory.html?highlight=prime#sympy.ntheory.primetest.isprime）。

在*步骤1*中检查了总体执行后，在*步骤2*中，我们在代码中引入了一个`breakpoint`。

当你在*步骤3*中执行代码时，它将在`breakpoint`位置停止，进入交互模式。

在交互模式下，我们可以检查任何变量的值，以及执行任何类型的操作。如*步骤4*所示，有时通过重现代码行的部分可以更好地分析它。

可以在命令行中检查代码并执行常规操作。可以通过调用`n`（`next`）来执行下一行代码，如*步骤5*中多次实现的，以查看代码的流程。

*步骤6*展示了如何使用`c`（`continue`）命令恢复执行，以便在下一个断点处停止。所有这些操作都可以迭代以查看流程和值，并理解代码在任何点的功能。

执行可以使用`q`（`quit`）停止，如*步骤7*所示。

### 更多内容...

要查看所有可用操作，可以在任何点调用`h`（`help`）。

你可以使用`l`（`list`）命令在任何点检查周围的代码。例如，在*步骤4*中：

```
(Pdb) l
  3     return False
  4
  5     lower = candidate - 1
  6     while lower > 1:
  7         breakpoint()
  8  ->     if candidate / lower == candidate // lower:
  9             return False
 10         lower -= 1
 11
 12     return True
```

另外两个主要的调试器命令是`s`（`step`），它将执行下一步，包括进入新的调用（如*单步进入*），以及`r`（`return`），它将继续执行当前函数直到执行`return`语句，然后停止。请注意，在任何Python函数的末尾，都有一个隐式的`return None`。

你可以使用 `pdb` 命令（`break`）来设置（或禁用）更多断点。你需要指定断点所在的文件和行号，但实际上，直接修改代码并重新运行会更直接且不易出错。

你不仅可以读取变量，还可以覆盖它们。或者创建新变量。或者进行额外的调用。或者任何你能想到的操作。Python 解释器的全部功能都为你所用！用它来检查某事物如何工作，或验证某事是否正在发生。

> 避免使用调试器保留的名称来创建变量，例如将列表命名为 `l`。这会造成混淆，并干扰 `pdb` 命令，有时是以不明显的方式。

`breakpoint()` 函数在 Python 3.7 中引入，如果你使用兼容版本，强烈推荐使用。在之前的版本中，你需要用以下代码替换它：

```
import pdb;
pdb.set_trace()
```

它们的工作方式完全相同。注意同一行中的两条语句，这在 Python 中通常不推荐，但这是将断点保持在单行的好方法。

> 记得在调试完成后移除任何 `breakpoints`！尤其是在提交到版本控制系统（如 Git）时。

你可以在官方 PEP 中阅读更多关于新的 `breakpoint` 调用的信息：https://www.python.org/dev/peps/pep-0553/。

完整的 `pdb` 文档可以在这里找到：https://docs.python.org/3.7/library/pdb.html#module-pdb。它包含了所有调试命令。

### 另请参阅

- 本章前面的 *学习 Python 解释器基础* 配方，以了解 Python 自省工具的基础知识。
- 本章接下来的 *提高你的调试技能* 配方，以查看一个涵盖不同问题的调试示例。

### 提高你的调试技能

在这个配方中，我们将使用一个模拟调用外部服务的小脚本来分析和修复一些错误。我们将展示不同的技术来提高你的调试技能。

该脚本将向一个互联网服务器（httpbin.org，一个测试站点）发送一些个人姓名，然后将它们取回，模拟从外部服务器检索。然后，它将它们拆分为名字和姓氏，并准备按姓氏排序。最后，它将对它们进行排序。

> 我们之前在第 3 章《构建你的第一个网页抓取应用程序》的《与表单交互》配方中使用过这个测试站点。请注意，URL https://httpbin.org/forms/post 渲染表单，但内部调用 URL https://httpbin.org/post 来发送信息。对于这个配方，我们只需要使用第二个 URL。

该脚本包含几个错误，我们将检测并修复它们。

### 准备工作

对于这个配方，我们将使用 `requests` 和 `parse` 模块，并将它们包含在我们的虚拟环境中：

```
$ echo "requests==2.18.3" >> requirements.txt
$ echo "parse==1.8.2" >> requirements.txt
$ pip install -r requirements.txt
```

`debug_skills.py` 脚本可从 GitHub 获取：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter13/debug_skills.py。请注意，它包含错误，我们将作为本配方的一部分进行修复。

### 操作步骤...

1. 运行脚本，它将生成一个错误：

```
$ python debug_skills.py
Traceback (most recent call last):
  File "debug_skills.py", line 26, in <module>
    raise Exception(f'Error accessing server: {result}')
Exception: Error accessing server: <Response [405]>
```

2. 分析状态码。我们得到 405，这意味着我们发送的方法不被允许。我们检查代码并意识到，对于第 24 行的请求，我们使用了 GET，而应该使用 POST（如 URL 中所述）。将代码替换为以下内容：

```
# ERROR Step 2. Using .get when it should be .post
# (old) result = requests.get('http://httpbin.org/post',
json=data)
result = requests.post('http://httpbin.org/post', json=data)
```

为了清晰起见，我们将旧的有错误的代码用 (old) 注释掉。

3. 再次运行代码，它将产生一个不同的错误：

```
$ python debug_skills.py
Traceback (most recent call last):
  File "debug_skills_solved.py", line 34, in <module>
    first_name, last_name = full_name.split()
ValueError: too many values to unpack (expected 2)
```

4. 在第 33 行插入一个断点，即错误之前的一行。再次运行并进入调试模式：

```
$ python debug_skills_solved.py
..debug_skills.py(35)<module>()
-> first_name, last_name = full_name.split()
(Pdb) n
> ...debug_skills.py(36)<module>()
-> ready_name = f'{last_name}, {first_name}'
(Pdb) c
> ...debug_skills.py(34)<module>()
-> breakpoint()
```

运行 `n` 没有产生错误，这意味着它不是第一个值。在 `c` 上运行几次后，我们意识到这不是正确的方法，因为我们不知道哪个输入是产生错误的那个。

5. 相反，用 `try...except` 块包装该行，并在该点产生一个断点：

```
try:
    first_name, last_name = full_name.split()
except:
    breakpoint()
```

6. 我们再次运行代码。这一次，代码在数据产生错误的那一刻停止：

```
$ python debug_skills.py
> ...debug_skills.py(38)<module>()
-> ready_name = f'{last_name}, {first_name}'
(Pdb) full_name
'John Paul Smith'
```

7. 原因现在很清楚了；第 35 行只允许我们拆分两个单词，但如果添加了中间名，则会引发错误。经过一些测试，我们确定用这一行来修复它：

```
# ERROR Step 6 split only two words. Some names has middle names
# (old) first_name, last_name = full_name.split()
first_name, last_name = full_name.rsplit(maxsplit=1)
```

8. 我们再次运行脚本。确保移除 `breakpoint` 和 `try..except` 块。这一次，它生成了一个姓名列表！并且它们按姓氏的字母顺序排序。然而，有几个姓名看起来不正确：

```
$ python debug_skills_solved.py
['Berg, Keagan', 'Cordova, Mai', 'Craig, Michael', 'García, Rocío', 'Mccabe, Fathima', "O'Carroll, Séamus", 'Pate, Poppy-Mae', 'Rennie, Vivienne', 'Smith, John Paul', 'Smyth, John', 'Sullivan, Roman']
```

谁叫 O'Carroll, Séamus？

9. 为了分析这个特定情况但跳过其余部分，我们必须创建一个 `if` 条件，仅在第 33 行为该姓名中断。注意使用 `in` 以避免必须完全正确：

```
full_name = parse.search('"custname": "{name}"', raw_result)['name']
if "O'Carroll" in full_name:
    breakpoint()
```

10. 再次运行脚本。断点在适当的时刻停止：

```
$ python debug_skills.py
> debug_skills.py(38)<module>()
-> first_name, last_name = full_name.rsplit(maxsplit=1)
(Pdb) full_name
"Séamus O'Carroll"
```

11. 在代码中向上移动并检查不同的变量：

```
(Pdb) full_name
"Séamus O'Carroll"
(Pdb) raw_result
'{"custname": "Séamus O\'Carroll"}'
(Pdb) result.json()
{'args': {}, 'data': '{"custname": "Séamus O\'Carroll"}', 'files': {}, 'form': {}, 'headers': {'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate', 'Connection': 'close', 'Content-Length': '37', 'Content-Type': 'application/json', 'Host': 'httpbin.org', 'User-Agent': 'python-requests/2.18.3'}, 'json': {'custname': 'Séamus O'Carroll'}, 'origin': '89.100.17.159', 'url': 'http://httpbin.org/post'}
```

12. 在 `result.json()` 字典中，实际上有一个不同的字段似乎正确地渲染了名称，它叫做 'json'。让我们详细看看；我们可以看到它是一个字典：

```
(Pdb) result.json()['json']
{'custname': "Séamus O'Carroll"}
(Pdb) type(result.json()['json'])
<class 'dict'>
```

13. 现在，我们需要更改代码。不再解析 'data' 中的原始值，而是直接使用结果中的 'json' 字段。这简化了代码，这很棒！

```
# ERROR Step 11. Obtain the value from a raw value. Use
# the decoded JSON instead
# raw_result = result.json()['data']
# Extract the name from the result
# full_name = parse.search('"custname": "{name}"', raw_result)['name']
raw_result = result.json()['json']
full_name = raw_result['custname']
```

14. 再次运行代码。记得移除 `breakpoint`：

```
$ python debug_skills.py
['Berg, Keagan', 'Cordova, Mai', 'Craig, Michael', 'García, Rocío', 'Mccabe, Fathima', "O'Carroll, Séamus", 'Pate, Poppy-Mae', 'Rennie, Vivienne', 'Smith, John Paul', 'Smyth, John', 'Sullivan, Roman']
```

这一次，一切都正确了！你已经成功调试了程序！

### 工作原理...

这个配方的结构分为三个不同的问题。让我们分小块分析它：

1. 第一个错误——对外部服务的错误调用：
   在 `步骤 1` 中显示第一个错误后，我们仔细阅读了结果错误，说明服务器返回了 `405` 状态码。

### 调试技术

这对应于一个 `method not allowed` 错误，表明我们的调用方法不正确。

检查以下代码行：
```python
result = requests.get('http://httpbin.org/post', json=data)
```

这提示我们正在向一个只接受 `POST` 请求的 URL 发送 `GET` 请求，因此我们在 *步骤 2* 中进行修改。

> 请注意，检测此错误无需特定的调试步骤，只需仔细阅读错误消息和代码即可。请记住关注错误消息和日志。通常，这足以发现问题，或至少是解决问题的重要线索。

我们在 *步骤 3* 中运行代码以发现下一个问题。

### 2. 第二个错误 – 中间名处理不当：

在 *步骤 3* 中，我们得到一个“值过多无法解包”的错误。我们可以在 *步骤 4* 中创建一个 `breakpoint` 来分析此时的数据，但发现并非所有数据都会产生此错误。*步骤 4* 中的分析表明，在未产生错误时停止执行可能会非常混乱，因此不得不继续执行直到错误出现。我们知道错误在此时产生，但仅针对特定数据。

由于我们知道错误在某个点产生，我们在 *步骤 5* 中使用 `try..except` 块捕获它。当异常产生时，我们触发 `breakpoint`。

这导致脚本在 *步骤 6* 的执行中，当 `full_name` 为 `'John Paul Smith'` 时停止。这会产生一个错误，因为 `split` 期望两个元素，而不是三个。

这在 *步骤 7* 中得到修复，允许除最后一个单词外的所有内容成为名字的一部分，将任何中间名分组到第一个元素中。这符合我们程序的目的，即按姓氏排序。

> 名字的处理实际上相当复杂。如果你想了解关于名字人们可能做出的大量错误假设，请查看这篇文章：https://www.kalzumeus.com/2010/06/17/falsehoods-programmers-believe-about-names/。

以下代码行使用 `rsplit` 实现了这一点：
```python
first_name, last_name = full_name.rsplit(maxsplit=1)
```

它从右侧开始按单词分割文本，最多进行一次分割，保证最多返回两个元素。

> 例如，如果定义了一个没有姓氏的名字，这可能会产生错误。如果输入数据发生变化，我们可能需要修改代码。

代码修改后，*步骤 8* 再次运行代码以发现下一个错误。

### 3. 第三个错误 – 使用外部服务返回的错误值：

在 *步骤 8* 中运行代码会显示列表且不产生任何错误。但是，通过检查结果，我们可以看到一些名字处理不正确。

我们在 *步骤 9* 中选择一个示例并创建一个条件断点。我们仅在数据满足 `if` 条件时激活 `breakpoint`。

> 在这种情况下，`if` 条件在“O'Carroll”字符串出现时停止，无需使用等于语句使其更严格。对这段代码要务实，因为无论如何你都需要在修复错误后将其删除。

代码在 *步骤 10* 中再次运行。一旦我们验证数据符合预期，我们反向追溯以找到问题的根源。*步骤 11* 分析之前的值和到那时为止的代码，试图找出导致错误值的原因。

然后我们发现我们使用了服务器响应的错误字段。`json` 字段中的值对此任务更合适，并且已经为我们解析好了。*步骤 12* 检查该值并查看应如何使用它。

在 *步骤 13* 中，我们修改代码以使用现有的 `.json()` 方法正确解码 JSON 内容。请注意，`parse` 模块不再需要，并且使用 `json` 方法使代码更简洁。

> 这种使用现有工具的结果比看起来更常见，尤其是在处理外部接口时。我们可能以一种有效的方式使用它，但这可能不是最佳方式。花点时间阅读文档，关注改进并学习如何更好地使用工具。

修复后，代码在 *步骤 14* 中再次运行。最终，代码按预期工作，按姓氏字母顺序排序名字。请注意，另一个包含奇怪字符的名字也得到了修复。

### 更多内容...

修复后的脚本可从 GitHub 获取：https://github.com/PacktPublishing/Python-Automation-Cookbook-Second-Edition/blob/master/Chapter13/debug_skills_fixed.py。你可以下载它并查看差异。

创建条件断点还有其他方法。调试器实际上支持创建仅在满足某些条件时才停止的断点。如果可能，我发现直接使用代码更容易，因为它在运行之间是持久的，并且更容易记住和操作。你可以在 Python pdb 文档中查看如何创建它：https://docs.python.org/3/library/pdb.html#pdbcommand-break。

如第一个错误所示，捕获异常的断点类型展示了在代码中设置条件是多么简单。只是要小心在之后删除它们！

还有其他可用的调试器具有更丰富的功能；例如：

- ipdb (https://github.com/gotcha/ipdb)：添加了制表符补全和语法高亮。
- pudb (https://documen.tician.de/pudb/)：显示一个旧式的、半图形化的、基于文本的界面，风格类似于 90 年代早期的工具，自动显示局部作用域变量。
- web-pdb (https://pypi.org/project/web-pdb/)：打开一个 Web 服务器以访问带有调试器的图形界面。

阅读上述调试器的文档以了解如何安装和运行它们。

> 还有更多可用的调试器。在互联网上搜索会给你更多选项，包括 Python IDE。无论如何，请注意添加依赖项。能够使用默认调试器总是好的。

新的断点命令允许我们使用 PYTHONBREAKPOINT 环境变量轻松地在调试器之间切换；例如：

```
$ PYTHONBREAKPOINT=ipdb.set_trace python my_script.py
```

这会在代码中的任何断点处启动 ipdb。你可以在 breakpoint() 文档中了解更多相关信息，文档地址为：https://www.python.org/dev/peps/pep-0553/#environment-variable。

> 这有一个重要的作用，即通过设置 `PYTHONBREAKPOINT=0` 来禁用所有断点，这是一个确保生产代码永远不会被意外留下的 `breakpoint()` 中断的好工具。

Python `pdb` 文档可在此处找到：https://docs.python.org/3/library/pdb.html。`parse` 模块的完整文档可在 https://github.com/richardj0n3s/parse 找到，而完整的 `requests` 文档可在 https://requests.readthedocs.io/en/master/ 找到。

### 另请参阅

- 本章前面的 *学习 Python 解释器基础* 配方，以了解 Python 中代码自省的基础知识。
- 上一节的 *使用断点进行调试* 配方，以了解设置断点的基础知识。

## 索引

**A**

抽象字符串 28
aiohttp
参考链接 110
注解
添加 297-299
argparse
参考链接 43
异步编程
参考链接 111
Awesome Python
URL 10

在 Excel 电子表格中创建 213-215
保存 307-309
chromedriver
URL 99
颜色选择器
参考链接 169
命令行参数
添加 38-42
逗号分隔值 (CSV) 124
configparser 模块文档
参考链接 51
常量 53
cron 作业
设置 53-56
工作原理 57
Unix/Linux 中的 cron 作业
优点 53
缺点 54
跨站请求伪造 (CSRF)
参考链接 98
密码学 364
CSV 文件
读取 124, 126
更新 200-202
CSV 模块文档
参考链接 127, 200
CSV 电子表格
聚合结果 245-249
基于位置追加货币 235-239
并行处理数据 252-257
使用 Pandas 处理数据 259-263
标准化日期格式 240-243
准备 230-234
写入 198, 199
自定义机器学习模型
为文本分类创建 419-429

**B**

样板代码 459
BotFather 347
断点
参考链接 489
用于调试 485-488
brew
URL 374
冒泡排序算法
参考链接 481
错误 475
内置过滤器
参考链接 157
Python 3 中的 bytes/str 二分法
参考链接 17

**C**

Calc 220
单元格格式
处理 216-219
图表

### D

data
从结构化字符串中提取数据 19-21
data center (DC) 326
DataFrame 262
debugging
关于 475
通过日志记录进行调试 481-483
使用断点进行调试 485-488
debugging, skills
提升调试技能 490-495
Decimal, Degrees (DD) 137
Decimal type, Python
参考链接 378
声明式 265
default text editor, setting in Linux
参考链接 57
Degrees, Decimal, Minutes (DDM) 137
Degrees, Minutes, Seconds (DMS) 137
Delorean module documentation
参考链接 245, 385
dependency mocking
用于执行测试 443-447
方言 126
Dillinger
URL 161
directories
爬取 114-116
搜索 114-116
documents
扫描关键词 147-149

参考链接 122
errors
捕获错误 59-63
转义 14
Excel spreadsheet
处理单元格格式 216-219
创建图表 213-215
创建新工作表 209-211
读取 203, 204
更新 206-208
Exchangeable Image File (EXIF) 133
ExchangeRate-API
URL 239
exif metadata
参考链接 133
external code
测试外部代码 439-441

### E

email
通过电子邮件发送通知 329-333
读取电子邮件 321-323
email newsletter
添加订阅者 325-328
email notifications
发送电子邮件通知 65-68
Email Regex
URL 31
email templates
处理电子邮件模板 312-315
encodings
处理编码问题 120-122

### F

feedparser module documentation
URL 88
feeds
订阅订阅源 86-88
file metadata
读取文件元数据 130, 131
Fiona documentation
参考链接 296
formatted values
用于创建字符串 11-14
formatter_class argument
参考链接 58
forms
与表单交互 93-98
FPDF documentation
参考链接 182
freezegun documentation
参考链接 458

### G

get current axes (gca) 295
Google Cloud Natural Language
用于分析文本 411-417
Google Cloud Vision
用于分析图像 389-401
用于从图像中提取文本 403-408
graphs
组合图表 302-306

### H

哈希 363
Heroku
URL 339
HTML
解析 HTML 75-78
HTTP call mocking
用于执行测试 450-456
HTTP status codes
结构 74
URL 74

### I

images
使用 Google Cloud Vision 分析图像 389-401
读取图像 132-137
使用 Google Cloud Vision 提取文本 403-408
image_text_box.py
参考链接 410
include_macro.py
参考链接 222
individual email
发送个人电子邮件 316-320
集成测试 434
内省 478
ipdb
URL 496

### J

Jinja2 documentation
参考链接 158
JSON
URL 89
JSONPlaceholder
URL 90
Jupyter
URL 265

### L

标签 389
语言无关 89
legends
添加图例 297-299
LibreOffice
创建宏 220-226
URL 221, 226
行终止符 200
log files
读取日志文件 127, 129
Lorem Ipsum text
URL 139

### M

macro
在 LibreOffice 中创建宏 220-226
MailChimp
关于 329
URL 325, 329
Mailgun
URL 329
Mailgun documentation
参考链接 334
maps
可视化地图 290-294
Markdown
格式化文本 159, 160
Markdown syntax
参考链接 161, 315
matplotlib
URL 272
matplotlib, annotations
参考链接 302
matplotlib, colors
参考链接 288
Microsoft (MS) Office 162
Mistune
参考链接 161
mocks 443
movies.ods
参考链接 222
multiple lines
显示多行 280, 282

### N

notifications
向客户发送通知 366-371
通过电子邮件发送通知 329-333

### O

openpyxl
参考链接 378
openpyxl documentation
参考链接 206
opportunities
检测机会 356-359
Optical Character Recognition (OCR) 137, 403
os.path
参考链接 117

### P

Pandas
用于处理 CSV 电子表格中的数据 259-263
Pandas documentation
参考链接 266
parse documentation
参考链接 378
parse module
参考链接 23, 497
使用 23, 25
工作原理 25
parse module, format-specification
参考链接 27
password-protected pages
访问受密码保护的页面 103, 104
pathlib module
参考链接 117
pdb 485
pdb documentation
参考链接 489
Portable Document Format (PDF)
关于 139
加密 PDF 190-194
构建 PDF 结构 182-187
添加水印 190-194
PDF document
编写 PDF 文档 179-181
PDF files
读取 PDF 文件 139-141
PDF reports
汇总 PDF 报告 188-190
pdftoppm
参考链接 142
personalized coupon codes
创建个性化优惠券代码 360-365
pictures
向 Word 文档中添加图片 175-178
pie charts
绘制饼图 277, 278
pikepdf
参考链接 142
Pillow documentation
参考链接 137, 195
Poetry tool 7
pprint documentation
参考链接 480
print debugging 481
problems
捕获问题 59-63
pudb
URL 496
PyPDF2 documentation
参考链接 190
pytest documentation
参考链接 439
pytest library
用于执行测试用例 435-437
用于编写测试用例 435-437
Python, assert commands
参考链接 366
Python, built-in functions documentation
参考链接 480
Python documentation
参考链接 258
references 65
Python Imaging Library (PIL) 133
Python interpreter 477-479
Python Package Index
URL 8
Python Selenium documentation
参考链接 102

### Q

量词 31

### R

regex101
URL 32
正则表达式拒绝服务攻击 33
正则表达式 28
正则表达式, 用途
验证输入数据 28
数据抓取 28
解析字符串 28
替换单词 28
regular expression operations
参考链接 38
regular expressions
关于 28-34
工作原理 30-36
reports
创建纯文本报告 152, 153
使用模板创建报告 155-157
requests-futures
参考链接 110
responses library 456
RESTful 89
RFC3339
URL 250
RGB colors
参考链接 271
RSS feeds
参考链接 357

### S

sales graph
绘制销售图表 268-270
sales information
准备销售信息 373-378
sales report
生成销售报告 379-384
scatter plot
绘制散点图 286-288
Selenium
用于交互 99-101
工作原理 102
shop.zip file 420
Simple Mail Transfer Protocol (SMTP) 316
SMS
接收短信 339-345
SMS messages
生成短信 334-338
smtplib documentation
参考链接 321
somesite
URL 79
stacked bars
绘制堆叠条形图 272-274
堆栈跟踪 64
standard output (stdout) 58
strings
使用格式化值创建字符串 11-14
操作字符串 14-17
参考链接 17
structure
在 Word 文档中生成结构 169-174
structured strings
从结构化字符串中提取数据 19-21
subscribers
向电子邮件通讯添加订阅者 325-328
监督训练 388
系统测试 434

### T

tasks
准备任务 46-50
Telegram bot
创建 Telegram 机器人 347-351
参考链接 353
teleport module documentation
参考链接 353
templates
使用模板创建报告 155-157
test cases
使用 pytest 库执行测试用例 435-437
使用 pytest 库编写测试用例 435-437
testing scenarios
准备测试场景 458-464
tests
使用依赖模拟执行测试 442-447
使用 HTTP 调用模拟执行测试 450-456
选择性运行测试 466-472
text
使用 Google Cloud Natural Language 分析文本 411-417
在 Markdown 中格式化文本 159, 160
text_analysis_categories.py script
参考链接 417
生成结构 169-174
设置样式 165-167
编写文本 162, 164
text classification
为文本分类创建自定义机器学习模型 419-429
write_zipfile 462
text files
读取文本文件 117-119
textwrap module documentation
参考链接 18
third-party packages
安装第三方包 8-10
Twilio
URL 335
Twilio documentation
参考链接 339
单元测试 434
urllib.parse documentation
参考链接 456

### V

Vim
参考链接 56
virtual environment
激活虚拟环境 2-4
工作原理 5
virtualenvwrapper module documentation
参考链接 7

### W

web
爬取网页 79-84
web APIs
访问 Web API 89-92
web pages
下载网页 72-74
web-pdb
URL 496
web scraping
加速网页抓取 106-110
Wolf fence algorithm 476
Word documents
向 Word 文档中添加图片 175-178
读取 Word 文档 143-146

### X

XMP
参考链接 134

### Y

YAML files
参考链接 51
yield 464

# 目录

前言 v

## 第 1 章：开启我们的自动化之旅 1

- 激活虚拟环境 2
- 安装第三方包 8
- 使用格式化值创建字符串 11
- 操作字符串 14
- 从结构化字符串中提取数据 19
- 使用第三方工具——parse 23
- 介绍正则表达式 28
- 深入了解正则表达式 34
- 添加命令行参数 38

## 第 2 章：轻松实现任务自动化 45

- 准备任务 46
- 设置 cron 作业 53
- 捕获错误和问题 59
- 发送电子邮件通知 65

## 第 3 章：构建你的第一个网页抓取应用 71

- 下载网页 72
- 解析 HTML 75
- 爬取网页 79
- 订阅订阅源 85
- 访问 Web API 89
- 与表单交互 93
- 使用 Selenium 进行高级交互 99
- 访问受密码保护的页面 103

# 目录

- **第4章：搜索和读取本地文件** 113
  - 爬取和搜索目录 114
  - 读取文本文件 117
  - 处理编码问题 120
  - 读取CSV文件 124
  - 读取日志文件 127
  - 读取文件元数据 130
  - 读取图像 132
  - 读取PDF文件 139
  - 读取Word文档 143
  - 扫描文档中的关键词 147
- **第5章：生成出色的报告** 151
  - 创建纯文本文本报告 152
  - 使用模板生成报告 155
  - 在Markdown中格式化文本 159
  - 编写基础Word文档 162
  - 设置Word文档样式 165
  - 在Word文档中生成结构 169
  - 向Word文档添加图片 175
  - 编写简单的PDF文档 179
  - 构建PDF结构 182
  - 汇总PDF报告 188
  - 为PDF添加水印和加密 190
- **第6章：电子表格的乐趣** 197
  - 编写CSV电子表格 198
  - 更新CSV文件 200
  - 读取Excel电子表格 203
  - 更新Excel电子表格 206
  - 在Excel电子表格中创建新工作表 209
  - 在Excel中创建图表 213
  - 处理Excel中的单元格格式 216
  - 在LibreOffice中创建宏 220
- **第7章：数据清洗与处理** 229
  - 准备CSV电子表格 230
  - 根据位置追加货币信息 235
  - 标准化日期格式 240
  - 汇总结果 245
  - 并行处理数据 252
  - 使用Pandas处理数据 259
- **第8章：开发精美的图表** 267
  - 绘制简单的销售图表 268
  - 绘制堆叠条形图 272
  - 绘制饼图 277
  - 显示多条数据线 280
  - 绘制散点图 286
  - 可视化地图 290
  - 添加图例和注释 297
  - 组合图表 302
  - 保存图表 307
- **第9章：处理通信渠道** 311
  - 使用电子邮件模板 312
  - 发送单个电子邮件 316
  - 读取电子邮件 321
  - 向电子邮件通讯添加订阅者 325
  - 通过电子邮件发送通知 329
  - 生成短信消息 334
  - 接收短信 339
  - 创建Telegram机器人 347
- **第10章：为何不自动化你的营销活动？** 355
  - 简介 355
  - 发现机会 356
  - 创建个性化优惠券代码 360
  - 通过客户首选渠道发送通知 366
  - 准备销售信息 373
  - 生成销售报告 379
- **第11章：用于自动化的机器学习** 387
  - 简介 387
  - 使用Google Cloud Vision AI分析图像 389
  - 使用Google Cloud Vision AI从图像中提取文本 403
  - 使用Google Cloud Natural Language分析文本 411
  - 创建自定义机器学习模型以对文本进行分类 419
- **第12章：自动化测试流程** 433
  - 简介 433
  - 编写和执行测试用例 435
  - 测试外部代码 439
  - 使用依赖模拟进行测试 442
  - 使用HTTP调用模拟进行测试 450
  - 准备测试场景 458
  - 选择性运行测试 466
- **第13章：调试技术** 475
  - 简介 475
  - 学习Python解释器基础 477
  - 通过日志记录进行调试 481
  - 使用断点进行调试 485
  - 提升调试技能 490
- **您可能感兴趣的其他书籍** 499
- **索引** 503

## 前言

我们可能都在花时间做一些价值不大的小型手动任务。这些任务可能包括：浏览信息源以寻找零星的相关信息、反复操作电子表格生成相同的图表，或者逐个搜索文件直到找到所需的数据。其中一些——很可能是大部分——任务实际上是可以自动化的。虽然前期需要投入，但对于那些需要重复执行的任务，我们可以让计算机来完成这类琐碎的工作，从而将我们自己的精力集中在人类擅长的事情上——基于结果进行高层次的分析和决策。本书将解释如何使用Python语言来自动化常见的商业任务，这些任务如果由计算机执行，速度可以大大提升。

鉴于Python的表达力和易用性，开始编写小程序来执行这些操作，并将它们组合成更集成的系统，其简单程度令人惊讶。在整本书中，我们将展示一些简单易懂的示例，这些示例可以根据您的具体需求进行调整，并且我们将把它们组合起来执行更复杂的操作。我们将执行一些常见的操作，例如通过网络爬虫发现机会、分析信息以生成带有图表的自动电子表格报告、通过自动生成的电子邮件进行沟通、通过短信获取通知，以及学习如何在您专注于其他更重要事情时运行任务。

虽然本书需要一些Python知识，但它是为非程序员编写的，提供了清晰且具有指导性的示例，这些示例将帮助读者提高技能，同时专注于特定的日常目标。

### 本书适合谁

本书适合Python初开发者，不一定是专业开发人员，他们希望利用并扩展自己的知识来自动化任务。书中的大多数示例都针对市场营销、销售和其他非技术领域。读者需要了解一点Python语言，包括其基本概念。

### 本书涵盖的内容

第1章，*开始我们的自动化之旅*，介绍了一些贯穿全书的基础内容。它描述了如何通过虚拟环境安装和管理第三方工具、如何进行有效的字符串操作、如何使用命令行参数，并介绍了正则表达式和其他文本处理方法。

第2章，*让任务自动化变得简单*，展示了如何准备和自动运行任务。它涵盖了如何编程任务使其在预定时间执行，而不是手动运行；如何获得自动运行任务的结果通知；以及如何在自动化流程中出现错误时获得通知。

第3章，*构建你的第一个网络爬虫应用*，探讨了如何发送网络请求以与外部网站进行通信，支持多种格式，如原始HTML内容、结构化订阅源、RESTful API，甚至自动化浏览器以无需人工干预执行步骤。它还涵盖了如何处理结果以提取相关信息。

第4章，*搜索和读取本地文件*，解释了如何搜索本地文件和目录并分析其中存储的信息。您将学习如何筛选不同编码的相关文件，以及读取多种常见格式的文件，如CSV、PDF、Word文档，甚至图像。

第5章，*生成出色的报告*，探讨了如何以多种格式展示文本格式的信息。这包括创建模板以生成文本文件，以及创建格式丰富且样式正确的Word和PDF文档。

第6章，*电子表格的乐趣*，探讨了如何读写CSV格式的电子表格；功能丰富的Microsoft Excel电子表格（包括格式和图表）；以及作为Microsoft Excel免费替代品的LibreOffice电子表格。

第7章，*数据清洗与处理*，介绍了处理多个信息源和在处理前清洗数据的技术。您将学习如何进行批处理以加速处理大量数据，包括使用Pandas等特定数据分析库。

第8章，*开发精美的图表*，解释了如何制作精美的图表，包括常见的饼图、折线图和条形图，以及其他高级案例，如堆叠条形图甚至地图。它还解释了如何组合和设置多个图表的样式，以生成丰富的图形，并以易于理解的格式显示相关信息。

第9章，*处理通信渠道*，解释了如何通过多个渠道发送消息，利用外部工具来完成大部分繁重工作。本章深入介绍了单独和*批量*发送和接收电子邮件、通过短信进行通信，以及在Telegram中创建机器人。

第10章，*为何不自动化你的营销活动？*，结合了书中包含的不同示例来生成完整的营销活动，包括发现机会、生成促销活动、与潜在客户沟通以及分析和报告促销活动产生的销售等步骤。本章展示了如何组合不同的元素来创建强大的系统。

第11章，*用于自动化的机器学习*，解释了如何使用Google的机器学习API进行文本分析、检测图像地标的位置以及从图像中提取文本。本章包括创建和训练一个模型，用于根据文本判断电子邮件应分配给哪个部门。

第12章，*自动化测试流程*，探讨了编写和执行测试以验证代码是否按预期运行。为此，介绍了测试框架`pytest`以及处理测试的常见情况。

第13章，*调试技术*，介绍了不同的方法和技巧，以帮助调试过程并确保软件质量。它利用了Python强大的内省能力和其开箱即用的调试工具来修复问题并生成可靠的自动化软件。

### PYTHON

#### 自动化：

电子邮件、数据整理、Excel处理、报告生成、网络爬虫等创意方案

LLOYD OCHOA