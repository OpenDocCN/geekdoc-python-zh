# Python Idle:初学者指南

> 原文：<https://www.pythonforbeginners.com/basics/python-idle-a-beginners-guide>

如果您在这里，您最近已经在您的机器上安装了 python 或者您想要这样做。Python IDLE 是你学习 Python 的第一批软件之一。本文将讨论开始使用 Python IDLE 所需要了解的一切。

## Python 空闲是什么？

当您在 Windows 或 Mac 上安装 python 时，安装会附带 IDLE。IDLE 代表集成开发和学习环境。看名字就能猜到，IDLE 帮助你入门学习 python。如果你是初学者，它可以成为你学习 python 最好的工具之一。

### 作为交互式 Python 解释器的 Python IDLE

IDLE 可以作为交互式解释器工作。您可以逐行编写 python 命令，它会执行这些命令并显示结果。您也可以将交互式 python 解释器称为 python shell。python shell 旨在读取 python 命令、评估语句、打印结果，并重复相同的过程，直到退出。

如果您想了解变量、赋值、数学和位运算，或者 python 中对列表和字符串等数据结构的运算，您可以使用 python IDLE 逐行执行语句。

如果需要，也可以在 python IDLE 中编写小块代码。但是，我会建议你使用 python 文件在 python 程序中编写代码块，然后执行它。

### Python 空闲时作为文件编辑器

在编写函数或循环等代码块时，可以创建扩展名为. py 的 python 脚本文件。

所有 python 文件的扩展名都是. py。您可以使用一个 python 文件一次执行任意数量的语句。使用 Python IDLE，您可以非常容易地创建和修改现有的 Python 文件。

## 作为集成开发环境的 Python IDLE

集成开发环境(IDE)为您提供了不同的工具和功能，帮助您以简单的方式编写程序。Python IDLE 还提供了不同的特性，比如语法突出显示、自动缩进、代码完成、突出显示和代码重构。本文将讨论如何在 python IDLE 中使用所有这些特性。

## 如何安装 Python IDLE？

当您在 Windows 机器或 MacOS 上安装 python 时，python IDLE 与 Python 安装捆绑在一起。你不需要单独安装 IDLE。

在 Linux 机器上，您可以按照以下步骤使用 apt-get 安装 python IDLE。

首先，运行以下更新命令来更新存储库。

```py
sudo apt-get update
```

这里，sudo 命令用于以超级用户的身份运行该语句，您需要有一个管理员密码来执行该命令。

在 update 语句之后，可以使用下面的语句安装 python idle。

```py
sudo apt-get install idle
```

执行上述语句后，IDLE 将被安装在您的 Linux 机器上。您可以通过在搜索栏中搜索应用程序来验证这一点，如下所示。

![Python IDLE Examples](img/5d821fa90c67feb05404e796aec41cf1.png)



IDLE Examples

或者，您可以在命令行终端中运行命令“idle”。执行该命令后，将启动 python IDLE。

![Python IDLE Examples](img/3e63e4cadb989ea2d1f39115f184c730.png)



IDLE Examples

在 Windows 机器上，你可以在开始菜单中搜索空闲，然后启动应用程序。

## Python IDLE 入门

一旦您开始空闲，将出现以下屏幕。

![Python IDLE Examples](img/61628506391454ac742fcfac4f9bc097.png)



Python IDLE Examples

在屏幕上，你可以看到下面的文字。

```py
Python 3.8.10 (default, Mar 15 2022, 12:22:08) 
[GCC 9.4.0] on linux 
Type "help", "copyright", "credits" or "license()" for more information.
```

让我们输入“`help`”，看看会发生什么。

![Python IDLE Examples](img/807501d34bb153d5dce7dd22959e3443.png)



IDLE Examples

程序建议输入“help()”。让我们这样做吧。

输入`help()`后，出现以下画面。

![Python IDLE Examples](img/b46b7f3d43b197dd76526cab543ac8d0.png)



IDLE Examples

在帮助菜单中，您可以键入任何内容来了解它在 python 中的含义。例如，让我们在帮助菜单中键入+号。执行后，将显示以下输出。

![Python IDLE Examples](img/00ce91417531241b713cc1f26ae21c15.png)



IDLE Examples

同样，当我们在帮助菜单中键入任何有效的文字时，相应的文档将会显示出来，以帮助您理解这些功能。

要从帮助菜单返回空闲编辑器窗口，您可以键入“`ctrl+D`”。它会带你到主屏幕。

以类似的方式来帮助，你可以键入'`copyright`'来查看谁拥有如下所示的闲置的版权。

![Python IDLE Examples](img/1f371e0ec85c3da33ea57b6b3535e205.png)



IDLE Examples

您也可以通过在如下所示的空闲中键入'`credits`'来查看演职员表。

![Python IDLE Examples](img/8b890725d6ee7fd9226093f6d8118023.png)



IDLE Examples

了解了这些特性之后，现在让我们来讨论如何在 python IDLE 中执行不同的操作。

## 使用 IDLE 作为交互式 Python 解释器

可以使用 python IDLE 作为交互解释器，一条一条执行语句。您可以执行数学运算、比较、接受用户输入以及许多其他任务。让我们讨论其中的一些。

### 空闲时的数学运算

你可以像在计算器中一样将两个数相加。要执行任何数学运算，你只需要输入数学表达式，然后按回车键。解释器将显示输出。您可以在下图中观察到这一点。

![Python IDLE Examples](img/a78298ddd28baecc6efeb504f3c6fd94.png)



IDLE Examples

这里，我们进行了各种数学运算。您可以观察到，当我们在输入一个表达式后按 enter 键时，输出会立即显示在 IDLE 中的下一行。

### 空闲时的比较操作

您还可以使用比较运算符来比较空闲。如果表达式计算结果为`True`，解释器将打印`True`。否则，将显示`False`。

您还可以结合数学表达式和比较运算符来生成输出。

![Python IDLE Examples](img/91b82f4e607b09d8f1ddac0de3aea1ad.png)



IDLE Examples

### 对空闲变量的操作

除了直接对数字执行运算，您还可以定义变量来存储值。

python 中的变量可以用来存储任何数字、字符串或其他数据类型。例如，您可以定义两个值分别为 10 和 20 的变量`num1`和`num2`，如下所示。

![](img/25ad74486e154fe616c966509a91aace.png)



IDLE Examples

只需在如下所示的提示符下键入变量名，就可以打印变量中的值。

![](img/5365de16d8727f4e85f2995975098232.png)



IDLE Examples

现在，您可以使用`num1`和`num2`将 10 和 20 相加，如下所示。

![](img/c632ec6c510c684be6b6e5b77b74f0df.png)



IDLE Examples

你也可以将`num1`和`num2`的和赋给另一个变量，如下所示。

![](img/0d86fa9585cfaf3195956c0473cda85b.png)



IDLE Examples

这里，我们将`num1`和`num2`的和赋给了`num3`。然后，我们打印出`num3`中的值。

### 空闲时打印语句

除了数值之外，您还可以为变量分配文本值。在 python 中，文本被存储为一个[字符串。为了定义字符串变量，我们将文本用单引号或双引号括起来，如下图所示。](https://www.pythonforbeginners.com/basics/python-string-methods-for-string-manipulation)

![](img/16c88c6e6ce57fc2550eff724b6ae64a.png)



IDLE Examples

这里，我们创建了一个名为`website`的变量，并为其赋值为“`Pythonforbeginners`”。您可以通过在如上所示的提示中键入变量名来打印变量中的值。

您可以使用`print()`函数打印任何值，而不是直接在提示中键入变量名。`print()`函数将一个值或变量作为其输入参数，并在终端中打印出来，如下所示。

![](img/84647df60e0531e5e2b505ad1d12af99.png)



IDLE Examples

您也可以通过在如下所示的`print()`函数中用逗号分隔来打印多个值。

![](img/97d1a9ec13a91a1d51c182ca1e3d48fa.png)



IDLE Examples

这里，值由空格字符分隔。

通过使用`print()` 函数中的`sep`参数，可以使用逗号分隔 print 语句中的值。`sep`参数将一个字符或字符串作为输入参数，并使用该字符分隔传递给`print()`函数的值，如下所示。

![](img/fb9522c0058e85ccd16ba74a64772eda.png)



IDLE Examples

### 在空闲时接受用户输入

您也可以使用`input()` 功能从用户处获取用户输入。`input()`函数将代表用户提示的字符串作为其输入参数。执行时，它接受用户输入，直到用户按下 enter 键。一旦用户按下回车键，`input()`函数就会以字符串的形式返回用户输入的所有字符。您可以在下面的示例中观察到这一点。

![](img/119a80c299b9ed777e695ead9cec0381.png)



IDLE Examples

在上图中，我们首先使用`input()`函数获取用户输入。我们还提示用户输入他的名字。一旦用户输入值，该值就被分配给变量名。然后，我们打印了存储在变量名中的值。

## 使用 Python IDLE 作为文件编辑器

要一次执行多条语句，需要在扩展名为. py 的文件中创建一个 python 脚本。

要创建一个 python 文件，可以从 IDLE 右上角的`File`菜单中选择`New File`选项。或者，你可以键入`Ctrl+N`打开一个新的无标题文件。

创建无标题文件后，将出现以下屏幕。

![](img/a314615b407983c2eb4cb8a4d15bae27.png)



IDLE Examples

您可以编写如下所示的代码。

![](img/1320dfb74057a3d7b5eeb63390723b0e.png)



IDLE Examples

在上面的代码中，程序要求用户输入他的姓名和年龄。之后，它计算用户的出生年份并打印结果。

`input()`函数总是返回一个字符串值。因此，我们在第四行代码中使用了 `int()` 函数，将输入的年龄转换为整数。

在最后一个 print 语句中，我们使用了`str()`函数将整数转换成字符串，这样解释器就可以执行[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)并打印结果。

编写完代码后，您需要运行 python 脚本。为此，首先需要用. py 扩展名保存文件。您可以进入`File-> Save`选项，通过给出文件名来保存文件。之后，你可以点击`Run-> Run Module`来运行你的 python 脚本。

点击`Run module`，程序将在空闲状态下执行，输出如下图所示。

![](img/180be9caab0f90ef0d1f42e44e37722c.png)



IDLE Examples

## Python 空闲时的其他操作

*   没有办法清除空闲的终端。您需要关闭 IDLE 并重新启动它来清除屏幕。
*   您可以通过点击`Options-> Configure IDLE`并选择首选功能来更改字体、缩进宽度、高亮显示首选项、快捷键和其他功能。
*   点击`Help-> Python Docs`可以阅读 python 官方文档。当您点击 python 文档按钮时，您将被重定向到安装在您计算机上的 Python 版本的官方文档。
*   要在程序完成前停止执行，可以键入 ctrl+C。
*   要关闭 python IDLE，可以在键盘上键入`ctrl+D`。

## 结论

在本文中，我们讨论了有助于您开始使用 python IDLE 的各种操作。想要了解更多关于 python 编程的知识，你可以注册这个 [python 初学者课程](https://www.pythonforbeginners.com/courses/python-3-for-beginners)。