# 记录 Python 代码:完整指南

> 原文：<https://realpython.com/documenting-python-code/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**记录 Python 代码:完整指南**](/courses/documenting-python-code/)

欢迎阅读完整的 Python 代码文档指南。无论你是在记录一个小脚本还是一个大项目，无论你是一个初学者还是经验丰富的 python 爱好者，这个指南将涵盖你需要知道的一切。

我们将本教程分为四个主要部分:

1.  **[为什么编写代码文档如此重要](#why-documenting-your-code-is-so-important) :** 文档及其重要性介绍
2.  **[注释与记录代码](#commenting-vs-documenting-code) :** 注释和记录的主要区别，以及使用注释的适当时间和方式
3.  **[使用文档字符串记录您的 Python 代码库](#documenting-your-python-code-base-using-docstrings) :** 深入探究类、类方法、函数、模块、包和脚本的文档字符串，以及在每个文档字符串中应该找到的内容
4.  **[记录您的 Python 项目](#documenting-your-python-projects) :** 必要的元素以及它们应该包含哪些内容

随意从头到尾通读这篇教程，或者跳到你感兴趣的部分。它的设计是双向的。

## 为什么记录代码如此重要

如果你正在阅读本教程，希望你已经知道了编写代码文档的重要性。如果没有，那么让我引用圭多在最近的一次 PyCon 上对我说的话:

> "读代码比写代码更常见."
> 
> -*基多·范罗斯*

当你写代码时，你是为两个主要的读者写的:你的用户和你的开发者(包括你自己)。这两种受众同等重要。如果你像我一样，你可能会打开旧的代码库，对自己说:“我到底在想什么？”如果你在阅读自己的代码时有问题，想象一下当你的用户或其他开发者试图使用你的代码或向你的代码贡献 T1 时，他们会经历什么。

相反，我敢肯定你遇到过这样的情况，你想用 Python 做点什么，却发现了一个看起来很棒的库，可以完成这项工作。然而，当你开始使用这个库的时候，你会寻找一些例子，文章，甚至是关于如何做一些具体事情的官方文档，但是却不能立即找到解决方案。

搜索之后，您开始意识到缺少文档，甚至更糟，完全丢失了。这是一种令人沮丧的感觉，让你不敢使用这个库，不管代码有多棒或多高效。Daniele Procida 对这种情况做了最好的总结:

> “你的软件有多好并不重要，因为如果文档不够好，人们就不会使用它。
> 
> — *[丹尼尔·普罗西达](https://www.divio.com/en/blog/documentation/)*

在本指南中，您将从头开始学习如何正确地记录您的 Python 代码，从最小的脚本到最大的 [Python 项目](https://realpython.com/intermediate-python-project-ideas/)，以帮助防止您的用户因过于沮丧而无法使用或贡献您的项目。

[*Remove ads*](/account/join/)

## 注释与记录代码

在我们讨论如何记录 Python 代码之前，我们需要区分记录和注释。

一般来说，注释是向开发人员描述你的代码。预期的主要受众是 Python 代码的维护者和开发者。结合编写良好的代码，注释有助于引导读者更好地理解您的代码及其目的和设计:

> “代码告诉你怎么做；评论告诉你为什么。”
> 
> ——*[杰夫·阿特伍德](https://blog.codinghorror.com/code-tells-you-how-comments-tell-you-why/)(又名编码恐怖)*

记录代码就是向用户描述它的用途和功能。虽然它可能对开发过程有所帮助，但主要的目标受众是用户。下一节描述如何以及何时对代码进行注释。

### 注释代码的基础知识

在 Python 中，注释是使用井号(`#`)创建的，应该是不超过几个句子的简短语句。这里有一个简单的例子:

```py
def hello_world():
    # A simple comment preceding a simple print statement
    print("Hello World")
```

根据 [PEP 8](http://pep8.org/#maximum-line-length) ，评论的最大长度应该是 72 个字符。即使您的项目将最大行长度更改为大于建议的 80 个字符，也是如此。如果注释将超过注释字符限制，使用多行注释是合适的:

```py
def hello_long_world():
    # A very long statement that just goes on and on and on and on and
    # never ends until after it's reached the 80 char limit
    print("Hellooooooooooooooooooooooooooooooooooooooooooooooooooooooo World")
```

注释你的代码有多种用途，包括:

*   **计划和评审:**当你开发代码的新部分时，首先使用注释作为计划或概述代码部分的方法可能是合适的。请记住，一旦实际的编码已经实现并经过审查/测试，就要删除这些注释:

    ```py
    # First step
    # Second step
    # Third step` 
    ```

*   **代码描述:**注释可以用来解释特定代码段的意图:

    ```py
    # Attempt a connection based on previous settings. If unsuccessful,
    # prompt user for new settings.` 
    ```

*   **算法描述:**当使用算法时，尤其是复杂的算法时，解释算法如何工作或如何在代码中实现会很有用。描述为什么选择了一个特定的算法而不是另一个算法可能也是合适的。

    ```py
    # Using quick sort for performance gains` 
    ```

*   **标记:**标记的使用可以用来标记代码中已知问题或改进区域所在的特定部分。一些例子有:`BUG`、`FIXME`和`TODO`。

    ```py
    # TODO: Add condition for when val is None` 
    ```

对你的代码的注释应该保持简短和集中。尽可能避免使用长注释。此外，你应该使用杰夫·阿特伍德建议的[以下四条基本规则:](https://blog.codinghorror.com/when-good-comments-go-bad/)

1.  尽可能让注释靠近被描述的代码。不在描述代码附近的注释会让读者感到沮丧，并且在更新时很容易被忽略。

2.  不要使用复杂的格式(如表格或 ASCII 数字)。复杂的格式会导致内容分散注意力，并且随着时间的推移很难维护。

3.  不要包含多余的信息。假设代码的读者对编程原则和语言语法有基本的了解。

4.  设计自己注释的代码。理解代码最简单的方法就是阅读它。当你使用清晰、易于理解的概念设计代码时，读者将能够很快理解你的意图。

记住注释是为读者设计的，包括你自己，帮助引导他们理解软件的目的和设计。

### 通过类型提示注释代码(Python 3.5+)

Python 3.5 中添加了类型提示，它是帮助代码读者的一种额外形式。事实上，它把杰夫的第四个建议从上面提到了下一个层次。它允许开发人员设计和解释他们的部分代码，而无需注释。这里有一个简单的例子:

```py
def hello_name(name: str) -> str:
    return(f"Hello {name}")
```

通过检查类型提示，您可以立即看出函数期望输入`name`是类型`str`或[字符串](https://realpython.com/python-strings/)。您还可以看出函数的预期输出也将是类型`str`，或者字符串。虽然类型提示有助于减少注释，但是要考虑到这样做也可能会在您创建或更新项目文档时增加额外的工作量。

你可以从 Dan Bader 制作的视频[中了解更多关于类型提示和类型检查的信息。](https://www.youtube.com/watch?v=2xWhaALHTvU)

[*Remove ads*](/account/join/)

## 使用文档字符串记录您的 Python 代码库

既然我们已经学习了注释，让我们深入研究一下如何记录 Python 代码库。在本节中，您将了解 docstrings 以及如何在文档中使用它们。本节进一步分为以下小节:

1.  **[文档字符串背景](#docstrings-background) :** 关于文档字符串如何在 Python 内部工作的背景
2.  **[Docstring 类型](#docstring-types) :** 各种 Docstring“类型”(函数、类、类方法、[模块、包](https://realpython.com/python-modules-packages/)和脚本)
3.  **[文档字符串格式](#docstring-formats) :** 不同的文档字符串“格式”(Google、NumPy/SciPy、reStructuredText 和 Epytext)

### 文档字符串背景

记录 Python 代码都是以文档字符串为中心的。这些是内置的字符串，如果配置正确，可以帮助您的用户和您自己处理项目的文档。除了 docstring，Python 还有一个内置函数`help()`，它将对象 docstring 打印到控制台。这里有一个简单的例子:

>>>

```py
>>> help(str)
Help on class str in module builtins:

class str(object)
 |  str(object='') -> str
 |  str(bytes_or_buffer[, encoding[, errors]]) -> str
 |
 |  Create a new string object from the given object. If encoding or
 |  errors are specified, then the object must expose a data buffer
 |  that will be decoded using the given encoding and error handler.
 |  Otherwise, returns the result of object.__str__() (if defined)
 |  or repr(object).
 |  encoding defaults to sys.getdefaultencoding().
 |  errors defaults to 'strict'.
 # Truncated for readability
```

这个输出是如何产生的？因为 Python 中的一切都是对象，所以可以使用`dir()`命令检查对象的目录。让我们这样做，看看有什么发现:

>>>

```py
>>> dir(str)
['__add__', ..., '__doc__', ..., 'zfill'] # Truncated for readability
```

在这个目录输出中，有一个有趣的属性，`__doc__`。如果你检查一下那处房产，你会发现:

>>>

```py
>>> print(str.__doc__)
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors are specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
```

瞧啊。您已经找到了文档字符串存储在对象中的位置。这意味着您可以直接操作该属性。但是，内置有一些限制:

>>>

```py
>>> str.__doc__ = "I'm a little string doc! Short and stout; here is my input and print me for my out"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can't set attributes of built-in/extension type 'str'
```

可以操作任何其他自定义对象:

```py
def say_hello(name):
    print(f"Hello {name}, is it me you're looking for?")

say_hello.__doc__ = "A simple function that says hello... Richie style"
```

>>>

```py
>>> help(say_hello)
Help on function say_hello in module __main__:

say_hello(name)
 A simple function that says hello... Richie style
```

Python 还有一个简化文档字符串创建的特性。不是直接操作`__doc__`属性，而是将字符串直接放在对象的下面，这将自动设置`__doc__`值。下面是上面的同一个例子所发生的情况:

```py
def say_hello(name):
    """A simple function that says hello... Richie style"""
    print(f"Hello {name}, is it me you're looking for?")
```

>>>

```py
>>> help(say_hello)
Help on function say_hello in module __main__:

say_hello(name)
 A simple function that says hello... Richie style
```

这就对了。现在你了解了 docstrings 的背景。现在是时候了解不同类型的文档字符串以及它们应该包含什么信息了。

### 文档字符串类型

在 [PEP 257](https://www.python.org/dev/peps/pep-0257/) 中描述了 Docstring 约定。它们的目的是为您的用户提供对象的简要概述。它们应该保持足够的简洁，以便于维护，但仍然足够详细，以便新用户理解它们的目的以及如何使用文档对象。

在所有情况下，文档字符串都应该使用三重双引号(`"""`)字符串格式。无论文档字符串是否是多行的，都应该这样做。至少，docstring 应该是对您所描述的内容的快速总结，并且应该包含在一行中:

```py
"""This is a quick summary line used as a description of the object."""
```

多行文档字符串用于进一步阐述摘要之外的对象。所有多行文档字符串都有以下部分:

*   单行摘要行
*   摘要前的空行
*   对 docstring 的任何进一步阐述
*   另一个空行

```py
"""This is the summary line

This is the further elaboration of the docstring. Within this section,
you can elaborate further on details as appropriate for the situation.
Notice that the summary and the elaboration is separated by a blank new
line.
"""

# Notice the blank line above. Code should continue on this line.
```

所有文档字符串的最大字符长度应该与注释相同(72 个字符)。文档字符串可以进一步分为三个主要类别:

*   **类文档字符串:**类和类方法
*   **包和模块文档字符串:**包、模块和函数
*   **脚本文件字符串:**脚本和函数

#### 类文档字符串

类文档字符串是为类本身以及任何类方法创建的。文档字符串紧跟在缩进一级的类或类方法之后:

```py
class SimpleClass:
    """Class docstrings go here."""

    def say_hello(self, name: str):
        """Class method docstrings go here."""

        print(f'Hello {name}')
```

类文档字符串应包含以下信息:

*   对其目的和行为的简要总结
*   任何公共方法，以及简短的描述
*   任何类属性(特性)
*   任何与子类化的[接口](https://realpython.com/python-interface/)相关的东西，如果这个类打算被子类化的话

[类构造函数](https://realpython.com/python-class-constructor/)的参数应该记录在`__init__`类方法 docstring 中。各个方法应该使用各自的文档字符串进行记录。类方法 docstrings 应包含以下内容:

*   对该方法及其用途的简要描述
*   传递的任何参数(必需的和可选的)，包括关键字参数
*   标记任何被认为是可选的或具有默认值的参数
*   执行该方法时出现的任何副作用
*   出现的任何异常
*   对何时可以调用该方法有任何限制吗

让我们举一个表示动物的数据类的简单例子。这个类将包含一些类属性、实例属性、一个`__init__`和一个[实例方法](https://realpython.com/instance-class-and-static-methods-demystified/):

```py
class Animal:
    """
 A class used to represent an Animal

 ...

 Attributes
 ----------
 says_str : str
 a formatted string to print out what the animal says
 name : str
 the name of the animal
 sound : str
 the sound that the animal makes
 num_legs : int
 the number of legs the animal has (default 4)

 Methods
 -------
 says(sound=None)
 Prints the animals name and what sound it makes
 """

    says_str = "A {name} says {sound}"

    def __init__(self, name, sound, num_legs=4):
        """
 Parameters
 ----------
 name : str
 The name of the animal
 sound : str
 The sound the animal makes
 num_legs : int, optional
 The number of legs the animal (default is 4)
 """

        self.name = name
        self.sound = sound
        self.num_legs = num_legs

    def says(self, sound=None):
        """Prints what the animals name is and what sound it makes.

 If the argument `sound` isn't passed in, the default Animal
 sound is used.

 Parameters
 ----------
 sound : str, optional
 The sound the animal makes (default is None)

 Raises
 ------
 NotImplementedError
 If no sound is set for the animal or passed in as a
 parameter.
 """

        if self.sound is None and sound is None:
            raise NotImplementedError("Silent Animals are not supported!")

        out_sound = self.sound if sound is None else sound
        print(self.says_str.format(name=self.name, sound=out_sound))
```

#### 包和模块文档字符串

包文档字符串应该放在包的`__init__.py`文件的顶部。这个 docstring 应该列出由包导出的模块和子包。

模块文档字符串类似于类文档字符串。不再记录类和类方法，而是记录模块和其中的任何函数。模块文档字符串甚至在任何导入之前就被放在文件的顶部。模块文档字符串应包括以下内容:

*   模块及其用途的简要描述
*   模块导出的任何类、异常、函数和任何其他对象的列表

模块函数的 docstring 应该包含与类方法相同的项目:

*   对该功能及其用途的简要描述
*   传递的任何参数(必需的和可选的)，包括关键字参数
*   标记所有被认为是可选的参数
*   执行函数时出现的任何副作用
*   出现的任何异常
*   对何时调用该函数有任何限制吗

#### 脚本文件字符串

脚本被视为从控制台运行的单个文件可执行文件。脚本的 Docstrings 放在文件的顶部，应该记录得足够好，以便用户能够充分理解如何使用脚本。当用户错误地传入一个参数或使用`-h`选项时，它应该可以用于它的“用法”消息。

如果您使用 [`argparse`](https://realpython.com/command-line-interfaces-python-argparse/) ，那么您可以省略特定于参数的文档，假设它已经被正确地记录在`argparser.parser.add_argument`函数的`help`参数中。建议在`argparse.ArgumentParser`的构造函数中使用`__doc__`作为`description`参数。查看我们关于[命令行解析库](https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/)的教程，了解更多关于如何使用`argparse`和其他常见命令行解析器的细节。

最后，任何自定义或第三方导入都应该在 docstrings 中列出，以便让用户知道运行脚本可能需要哪些包。下面是一个简单打印电子表格列标题的脚本示例:

```py
"""Spreadsheet Column Printer

This script allows the user to print to the console all columns in the
spreadsheet. It is assumed that the first row of the spreadsheet is the
location of the columns.

This tool accepts comma separated value files (.csv) as well as excel
(.xls, .xlsx) files.

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

 * get_spreadsheet_cols - returns the column headers of the file
 * main - the main function of the script
"""

import argparse

import pandas as pd

def get_spreadsheet_cols(file_loc, print_cols=False):
    """Gets and prints the spreadsheet's header columns

 Parameters
 ----------
 file_loc : str
 The file location of the spreadsheet
 print_cols : bool, optional
 A flag used to print the columns to the console (default is
 False)

 Returns
 -------
 list
 a list of strings used that are the header columns
 """

    file_data = pd.read_excel(file_loc)
    col_headers = list(file_data.columns.values)

    if print_cols:
        print("\n".join(col_headers))

    return col_headers

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_file',
        type=str,
        help="The spreadsheet file to pring the columns of"
    )
    args = parser.parse_args()
    get_spreadsheet_cols(args.input_file, print_cols=True)

if __name__ == "__main__":
    main()
```

[*Remove ads*](/account/join/)

### 文档字符串格式

您可能已经注意到，在本教程给出的所有示例中，有一些特定的格式带有公共元素:`Arguments`、`Returns`和`Attributes`。有一些特定的文档字符串格式可用于帮助文档字符串解析器和用户拥有熟悉和已知的格式。本教程示例中使用的格式是 NumPy/SciPy 样式的文档字符串。一些最常见的格式如下:

| 格式化类型 | 描述 | 由 Sphynx 支持 | 形式规范 |
| --- | --- | --- | --- |
| [谷歌文档字符串](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) | 谷歌推荐的文档格式 | 是 | 不 |
| [重组文本](http://docutils.sourceforge.net/rst.html) | 官方 Python 文档标准；不适合初学者，但功能丰富 | 是 | 是 |
| [NumPy/SciPy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) | NumPy 的 reStructuredText 和 Google Docstrings 的组合 | 是 | 是 |
| [Epytext](http://epydoc.sourceforge.net/epytext.html) | Epydoc 的 Python 改编版；非常适合 Java 开发人员 | 不是正式的 | 是 |

docstring 格式的选择取决于您，但是您应该在整个文档/项目中坚持使用相同的格式。下面是每种类型的示例，让您了解每种文档格式的外观。

#### 谷歌文档字符串示例

```py
"""Gets and prints the spreadsheet's header columns

Args:
 file_loc (str): The file location of the spreadsheet
 print_cols (bool): A flag used to print the columns to the console
 (default is False)

Returns:
 list: a list of strings representing the header columns
"""
```

#### 重构文本示例

```py
"""Gets and prints the spreadsheet's header columns

:param file_loc: The file location of the spreadsheet
:type file_loc: str
:param print_cols: A flag used to print the columns to the console
 (default is False)
:type print_cols: bool
:returns: a list of strings representing the header columns
:rtype: list
"""
```

#### NumPy/SciPy 文档字符串示例

```py
"""Gets and prints the spreadsheet's header columns

Parameters
----------
file_loc : str
 The file location of the spreadsheet
print_cols : bool, optional
 A flag used to print the columns to the console (default is False)

Returns
-------
list
 a list of strings representing the header columns
"""
```

#### Epytext 示例

```py
"""Gets and prints the spreadsheet's header columns

@type file_loc: str
@param file_loc: The file location of the spreadsheet
@type print_cols: bool
@param print_cols: A flag used to print the columns to the console
 (default is False)
@rtype: list
@returns: a list of strings representing the header columns
"""
```

## 记录您的 Python 项目

Python 项目有各种形状、大小和用途。您记录项目的方式应该适合您的具体情况。记住你的项目的用户是谁，并适应他们的需求。根据项目类型，推荐文档的某些方面。项目及其文件的总体[布局](https://realpython.com/python-application-layouts/)应如下:

```py
project_root/
│
├── project/  # Project source code
├── docs/
├── README
├── HOW_TO_CONTRIBUTE
├── CODE_OF_CONDUCT
├── examples.py
```

项目通常可以细分为三种主要类型:私有、共享和公共/开源。

### 私人项目

私人项目是仅供个人使用的项目，通常不会与其他用户或开发人员共享。对于这种类型的项目，文档可能很简单。根据需要，可以添加一些推荐的部件:

*   **自述:**项目及其目的的简要概述。包括安装或运营项目的任何特殊要求。
*   **`examples.py` :** 一个 Python 脚本文件，给出了如何使用这个项目的简单例子。

记住，即使私人项目是为你个人设计的，你也被认为是一个用户。思考任何可能让你感到困惑的事情，并确保在注释、文档字符串或自述文件中记录下来。

[*Remove ads*](/account/join/)

### 共享项目

共享项目是指在开发和/或使用项目的过程中，您与其他人进行协作的项目。项目的“客户”或用户仍然是你自己和那些使用项目的少数人。

文档应该比私人项目更严格，主要是为了帮助项目的新成员，或者提醒贡献者/用户项目的新变化。以下是一些建议添加到项目中的部件:

*   **自述:**项目及其目的的简要概述。包括安装或运行项目的任何特殊要求。此外，添加自上一版本以来的任何主要更改。
*   **`examples.py` :** 一个 Python 脚本文件，给出了如何使用项目的简单例子。
*   如何贡献:这应该包括项目的新贡献者如何开始贡献。

### 公共和开源项目

公共和开放源代码项目是那些旨在与一大群用户共享的项目，并且可能涉及大型开发团队。这些项目应该像项目本身的实际开发一样优先考虑项目文件。以下是一些建议添加到项目中的部件:

*   **自述:**项目及其目的的简要概述。包括安装或运行项目的任何特殊要求。此外，添加自上一版本以来的任何主要变化。最后，添加进一步的文档、错误报告和项目的任何其他重要信息的链接。丹·巴德为[整理了一个很棒的教程](https://dbader.org/blog/write-a-great-readme-for-your-github-project)，告诉你应该在你的自述文件中包含哪些内容。

*   如何贡献:这应该包括项目的新贡献者如何提供帮助。这包括开发新功能、修复已知问题、添加文档、添加新测试或报告问题。

*   **行为准则:**定义了其他贡献者在开发或使用你的软件时应该如何对待彼此。这也说明了如果代码被破坏会发生什么。如果你正在使用 Github，可以用推荐的措辞生成一个行为准则[模板](https://help.github.com/articles/adding-a-code-of-conduct-to-your-project/)。特别是对于开源项目，考虑添加这个。

*   **License:** 一个描述您的项目正在使用的许可证的纯文本文件。特别是对于开源项目，考虑添加这个。

*   **docs:** 包含更多文档的文件夹。下一节将更全面地描述应该包含的内容以及如何组织该文件夹的内容。

#### `docs`文件夹的四个主要部分

Daniele Procida 在 PyCon 2017 上发表了精彩的[演讲](https://www.youtube.com/watch?v=azf6yzuJt54)和随后的[博客文章](https://www.divio.com/en/blog/documentation/)关于记录 Python 项目。他提到，所有项目都应该有以下四个主要部分来帮助你集中精力工作:

*   教程:带领读者完成一个项目(或有意义的练习)的一系列步骤的课程。面向用户的学习。
*   **How-To Guides** :引导读者完成解决常见问题所需步骤的指南(面向问题的食谱)。
*   参考文献:澄清和阐明某一特定主题的解释。面向理解。
*   **解释**:机器的技术描述以及如何操作(关键类、功能、API 等)。想想百科文章。

下表显示了所有这些部分之间的相互关系及其总体目的:

|  | 在我们学习的时候最有用 | 当我们编码时最有用 |
| --- | --- | --- |
| **实际步骤** | *教程* | *操作指南* |
| **理论知识** | *解释* | *参考* |

最后，您希望确保您的用户能够获得他们可能有的任何问题的答案。通过以这种方式组织你的项目，你将能够容易地回答这些问题，并且以一种他们能够快速浏览的格式。

### 文档工具和资源

记录您的代码，尤其是大型项目，可能会令人望而生畏。幸运的是，有一些工具和参考资料可以帮助您入门:

| 工具 | 描述 |
| --- | --- |
| [斯芬克斯](http://www.sphinx-doc.org/en/stable/) | 自动生成多种格式文档的工具集 |
| [Epydoc](http://epydoc.sourceforge.net/) | 一个基于文档字符串为 Python 模块生成 API 文档的工具 |
| [阅读文档](https://readthedocs.org/) | 为您自动构建、版本控制和托管您的文档 |
| [脱氧核糖核酸](https://www.doxygen.nl/manual/docblocks.html) | 一个用于生成支持 Python 和多种其他语言的文档的工具 |
| [MkDocs](https://www.mkdocs.org/) | 使用 Markdown 语言帮助构建项目文档的静态站点生成器。查看[使用 MkDocs](https://realpython.com/python-project-documentation-with-mkdocs/) 构建您的 Python 项目文档以了解更多信息。 |
| [pycco](https://pycco-docs.github.io/pycco/) | 一个“快速而肮脏”的文档生成器，并排显示代码和文档。查看[我们关于如何使用它的教程，了解更多信息](https://realpython.com/generating-code-documentation-with-pycco/)。 |
| [T2`doctest`](https://docs.python.org/3/library/doctest.html) | 一个标准库模块，用于将使用示例作为自动化测试运行。查看 [Python 的 doctest:立即记录并测试您的代码](https://realpython.com/python-doctest/) |

除了这些工具，还有一些额外的教程、视频和文章，在您记录项目时会很有用:

1.  [卡罗尔·威林-实用狮身人面像- PyCon 2018](https://www.youtube.com/watch?v=0ROZRNZkPS8)
2.  [Daniele Procida -文档驱动的开发 Django 项目的经验教训- PyCon 2016](https://www.youtube.com/watch?v=bQSR1UpUdFQ)
3.  [Eric Holscher -用 Sphinx 记录您的项目&阅读文档- PyCon 2016](https://www.youtube.com/watch?v=hM4I58TA72g)
4.  [Titus Brown，Luiz Irber——创建、构建、测试和记录 Python 项目:2016 年 PyCon 实用指南](https://youtu.be/SUt3wT43AeM?t=6299)
5.  [重组文本正式文档](http://docutils.sourceforge.net/rst.html)
6.  斯芬克斯的重组文本引物

有时候，最好的学习方法是模仿别人。下面是一些很好地使用文档的项目的例子:

*   **姜戈:** [文档](https://docs.djangoproject.com/en/2.0/) ( [来源](https://github.com/django/django/tree/master/docs))
*   **请求:** [文档](https://requests.readthedocs.io/en/master/) ( [来源](https://github.com/requests/requests/tree/master/docs))
*   **点击:** [文档](http://click.pocoo.org/dev/) ( [来源](https://github.com/pallets/click/tree/master/docs))
*   **熊猫:** [Docs](http://pandas.pydata.org/pandas-docs/stable/) ( [来源](https://github.com/pandas-dev/pandas/tree/master/doc))

[*Remove ads*](/account/join/)

## 我从哪里开始？

项目文档有一个简单的进程:

1.  没有文档
2.  一些文件
3.  完整的文档
4.  良好的文档
5.  出色的文档

如果你不知道你的文档下一步该做什么，看看你的项目相对于上面的进展现在处于什么位置。你有任何文件吗？如果没有，那就从那里开始。如果您有一些文档，但是缺少一些关键的项目文件，可以从添加这些文件开始。

最后，不要因为记录代码所需的工作量而气馁或不知所措。一旦你开始记录你的代码，继续下去会变得更容易。如果您有任何问题，请随时发表评论，或者在社交媒体上联系真正的 Python 团队，我们会提供帮助。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**记录 Python 代码:完整指南**](/courses/documenting-python-code/)*******