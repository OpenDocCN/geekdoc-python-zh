# Black 简介——不妥协的 Python 代码格式化程序

> 原文：<https://www.blog.pythonlibrary.org/2019/07/16/intro-to-black-the-uncompromising-python-code-formatter/>

有几个 Python 代码检查器可用。例如，许多开发人员喜欢使用 [Pylint](https://www.pylint.org/) 或 [Flake8](http://flake8.pycqa.org/en/latest/) 来检查他们代码中的错误。这些工具使用静态代码分析来检查代码中的错误或命名问题。Flake8 还会检查你的代码，看看你是否遵守了 Python 的风格指南 [PEP8](https://www.python.org/dev/peps/pep-0008/) 。

然而，有一个新的工具你可以使用，叫做[黑色](https://github.com/python/black)。Black 是一个 Python 代码格式化程序。它会根据黑色代码的风格重新格式化你的整个文件，非常接近 PEP8。

* * *

### 装置

安装黑色很容易。您可以使用 pip 来实现这一点:

```py
pip install black

```

你也可以按照这些[指令](https://github.com/python/black#editor-integration)配置流行的文本编辑器和 ide 来使用黑色。

既然黑装了，那就试一试吧！

* * *

### 使用黑色

Black 要求你有一些代码来运行它。让我们创建一个有许多参数的简单函数，然后在该脚本上运行 Black。

这里有一个例子:

```py
def long_func(x, param_one=None, param_two=[], param_three={}, param_four=None, param_five="", param_six=123456):
    print("This function has several params")

```

现在，在您的终端中，尝试对您的代码文件运行`black`，如下所示:
 `black long_func.py`

当您运行这个命令时，您应该看到下面的输出:
 `reformatted long_func.py
All done!
1 file reformatted.`

这意味着您的文件已被重新格式化，以遵循黑色标准。

让我们打开文件，看看它是什么样子的:

```py
def long_func(
    x,
    param_one=None,
    param_two=[],
    param_three={},
    param_four=None,
    param_five="",
    param_six=123456,
):
    print("This function has several params")

```

如您所见，Black 已将每个参数放在各自的行上。

* * *

### 检查文件格式

如果您不希望 Black 更改您的文件，但您想知道 Black 是否认为某个文件应该更改，您可以使用以下命令标志之一:

*   `--check` -检查文件是否应该重新格式化，但不实际修改文件
*   `--diff` -写出 Black 对文件的不同处理，但不修改文件

我喜欢用这些来测试我的文件，看看 Black 会如何重新格式化我的代码。我没有用黑色很长时间，所以这让我看看我是否喜欢黑色将要做的事情，而不实际做任何事情。

* * *

### 包扎

我喜欢黑色。我认为这真的很有用，尤其是在一个组织中实施某种 Python 风格的时候。请注意，黑色默认为 88 个字符，但您可以使用`-l`更改

或者`--line-length`

如果需要的话。在[项目的页面](https://github.com/python/black)上还列出了一些其他有用的选项。如果有机会，我觉得你应该给布莱克一个尝试！

* * *

### 相关阅读

*   PyLint: [分析 Python 代码](https://www.blog.pythonlibrary.org/2012/06/12/pylint-analyzing-python-code/)
*   py flakes "[Python 程序的被动检查器](https://www.blog.pythonlibrary.org/2012/06/13/pyflakes-the-passive-checker-of-python-programs/)
*   Python 101: [第 32 集—静态代码分析](https://www.blog.pythonlibrary.org/2018/11/07/python-101-episode-32-static-code-analysis/)