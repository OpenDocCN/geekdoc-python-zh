# Flake8 简介

> 原文：<https://www.blog.pythonlibrary.org/2019/08/13/an-intro-to-flake8/>

Python 有几个 linters，可以用来帮助您找到代码中的错误或关于样式的警告。例如，最彻底的棉绒之一是 [Pylint](https://www.blog.pythonlibrary.org/2012/06/12/pylint-analyzing-python-code/) 。

Flake8 被描述为一种用于样式指南实施的工具。它也是围绕 [PyFlakes](https://www.blog.pythonlibrary.org/2012/06/13/pyflakes-the-passive-checker-of-python-programs/) 、 [pycodestyle](https://pypi.org/project/pycodestyle/) 和内德·巴奇尔德[麦凯布剧本](https://pypi.org/project/mccabe/)的包装。您可以使用 Flake8 作为一种 lint 您的代码并强制 PEP8 合规性的方法。

* * *

### 装置

使用 pip 时，安装 Flake8 相当容易。如果您想将 flake8 安装到默认的 Python 位置，可以使用以下命令:

```py

python -m pip install flake8

```

既然安装了 Flake8，那就来学习一下如何使用吧！

* * *

### 入门指南

下一步是在您的代码基础上使用 Flake8。让我们写一小段代码来运行 Flake8。

将以下代码放入名为`hello.py:`的文件中

```py

from tkinter import *

class Hello:
    def __init__(self, parent):
        self.master = parent
        parent.title("Hello Tkinter")

        self.label = Label(parent, text="Hello there")
        self.label.pack()

        self.close_button = Button(parent, text="Close",
                                   command=parent.quit)
        self.close_button.pack()

    def greet(self):
        print("Greetings!")

root = Tk()
my_gui = Hello(root)
root.mainloop()

```

在这里，您使用 Python 的 [tkinter](https://docs.python.org/3/library/tkinter.html) 库编写了一个小型 GUI 应用程序。网上很多 tkinter 代码用的是`from tkinter import * pattern, which is something that you should normally avoid. The reason being that you don't know everything you are importing and you could accidentally overwrite an import with your own variable name.`

让我们针对这个代码示例运行 Flake8。

打开您的终端并运行以下命令:

```py

flake8 hello.py

```

您应该会看到以下输出:

 `tkkkk.py:1:1: F403 'from tkinter import *' used; unable to detect undefined names
tkkkk.py:3:1: E302 expected 2 blank lines, found 1
tkkkk.py:8:22: F405 'Label' may be undefined, or defined from star imports: tkinter
tkkkk.py:11:29: F405 'Button' may be undefined, or defined from star imports: tkinter
tkkkk.py:18:1: E305 expected 2 blank lines after class or function definition, found 1
tkkkk.py:18:8: F405 'Tk' may be undefined, or defined from star imports: tkinter
tkkkk.py:20:16: W292 no newline at end of file`

以“F”开头的项目是 PyFlake [错误代码](http://flake8.pycqa.org/en/latest/user/error-codes.html)，指出代码中的潜在错误。其他错误来自 [pycodestyle](https://pycodestyle.readthedocs.io/en/latest/intro.html#error-codes) 。您应该查看这两个链接，以了解完整的错误代码列表及其含义。

您还可以针对一个文件目录运行 Flake8，而不是一次运行一个文件。

如果您想要限制要捕获的错误类型，可以执行如下操作:

```py

flake8 --select E123 my_project_dir

```

这将只显示在指定目录中的任何文件的 E123 错误，而忽略所有其他类型的错误。

要获得 Flake8 可以使用的命令行参数的完整列表，请查看它们的[文档](http://flake8.pycqa.org/en/latest/user/options.html)。

最后，Flake8 允许你改变它的[配置](http://flake8.pycqa.org/en/latest/user/configuration.html)。例如，你的公司可能只遵循 [PEP8](https://www.python.org/dev/peps/pep-0008/) 的部分内容，所以你不希望 Flake8 标记你的公司不关心的东西。Flake8 也支持使用插件。

* * *

### 包扎

Flake8 可能就是您用来帮助保持代码整洁和没有错误的工具。如果你使用持续集成系统，比如 TravisCI 或 Jenkins，你可以将 Flake8 和 [Black](https://www.blog.pythonlibrary.org/2019/07/16/intro-to-black-the-uncompromising-python-code-formatter/) 结合起来，自动格式化你的代码并标记错误。

这绝对是一个值得一试的工具，而且比 PyLint 噪音小得多。试试看，看你怎么想！

* * *

### 相关阅读

*   Flake8 网站
*   py flakes "[Python 程序的被动检查器](https://www.blog.pythonlibrary.org/2012/06/13/pyflakes-the-passive-checker-of-python-programs/)
*   Black 简介[不妥协的 Python 代码格式化程序](https://www.blog.pythonlibrary.org/2019/07/16/intro-to-black-the-uncompromising-python-code-formatter/)
*   PyLint: [分析 Python 代码](https://www.blog.pythonlibrary.org/2012/06/12/pylint-analyzing-python-code/)