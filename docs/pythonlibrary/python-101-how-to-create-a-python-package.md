# Python 101 -如何创建 Python 包

> 原文：<https://www.blog.pythonlibrary.org/2021/09/23/python-101-how-to-create-a-python-package/>

当您创建一个 Python 文件时，您正在创建一个 Python 模块。您创建的任何 Python 文件都可以由另一个 Python 脚本导入。因此，根据定义，它也是一个 Python 模块。如果您有两个或更多相关的 Python 文件，那么您可能有一个 Python 包。

一些组织将他们所有的代码留给自己。这就是所谓的**闭源**。Python 是一种**开源**语言，你可以从 **Python 包索引(PyPI)** 获得的大部分 Python 模块和包也都是免费和开源的。共享您的包或模块的最快方法之一是将它上传到 Python 包索引或 Github 或两者。

在本文中，您将了解以下主题:

*   创建模块
*   创建包
*   为 PyPI 打包项目
*   创建项目文件
*   正在创建`setup.py`
*   上传到 PyPI

这个过程的第一步是理解创建一个可重用模块是什么样子的。我们开始吧！

## 创建模块

您创建的任何 Python 文件都是可以导入的模块。您可以通过向任何文章的代码文件夹中添加一个新文件，并尝试在其中导入一个模块，用本书中的一些示例来尝试一下。例如，如果您有一个名为`a.py`的 Python 文件，然后创建一个名为`b.py`的新文件，您可以通过使用`import a`将`a`导入到`b`中。

当然，这是一个愚蠢的例子。相反，您将创建一个简单的模块，其中包含一些基本的算术函数。您可以将文件命名为`arithmetic.py`,并将以下代码添加到其中:

```py
# arithmetic.py

def add(x, y):
    return x + y

def divide(x, y):
    return x / y

def multiply(x, y):
    return x * y

def subtract(x, y):
    return x - y

```

这段代码很幼稚。例如，您根本没有错误处理。这意味着你可以除以零，并引发一个异常。您还可以向这些函数传递不兼容的类型，比如字符串和整数——这将导致引发不同类型的异常。

但是，出于学习的目的，这段代码已经足够了。您可以通过创建一个测试文件来证明它是可导入的。创建一个名为`test_arithmetic.py`的新文件，并将以下代码添加到其中:

```py
# test_arithmetic.py

import arithmetic
import unittest

class TestArithmetic(unittest.TestCase):

    def test_addition(self):
        self.assertEqual(arithmetic.add(1, 2), 3)

    def test_subtraction(self):
        self.assertEqual(arithmetic.subtract(2, 1), 1)

    def test_multiplication(self):
        self.assertEqual(arithmetic.multiply(5, 5), 25)

    def test_division(self):
        self.assertEqual(arithmetic.divide(8, 2), 4)

if __name__ == '__main__':
    unittest.main()

```

将测试保存在与模块相同的文件夹中。现在，您可以使用以下命令运行这段代码:

```py
$ python3 test_arithmetic.py 
....
----------------------------------------------------------------------
Ran 4 tests in 0.000s

OK

```

这表明您可以将`arithmetic.py`作为一个模块导入。这些测试还显示了代码工作的基本功能。您可以通过测试被零除以及混合字符串和整数来增强这些测试。这些类型的测试目前会失败。一旦您有一个失败的测试，您可以遵循测试驱动的开发方法来修复问题。

现在让我们来看看如何制作一个 Python 包！

## 创建包

Python 包是您计划与其他人共享的一个或多个文件，通常通过将其上传到 Python 包索引(PyPI)来实现。包通常是通过命名文件目录而不是文件本身来制作的。然后，在该目录中，您将有一个特殊的`__init__.py`文件。当 Python 看到`__init__.py`文件时，它知道这个文件夹可以作为一个包导入。

有几种方法可以将`arithmetic.py`转换成一个包。最简单的方法是将代码从`arithmetic.py`移到`arithmetic/__init__.py`:

*   创建文件夹`arithmetic`
*   将`arithmetic.py`移动/复制到`arithmetic/__init__.py`
*   如果在上一步中使用了“复制”,则删除`arithmetic.py`
*   运行`test_arithmetic.py`

最后一步非常重要！如果您的测试仍然通过，那么您就知道从模块到包的转换成功了。要测试你的包，如果你在 Windows 上打开一个命令提示符，如果你在 Mac 或 Linux 上打开一个终端。然后导航到包含`arithmetic`文件夹的文件夹，但不在其中。现在你应该和你的`test_arithmetic.py`文件在同一个文件夹中。此时你可以运行`python test_arithmetic.py`，看看你的努力是否成功。

简单地将所有代码放在一个`__init__.py`文件中似乎很愚蠢，但是对于长达几千行的文件来说，这确实很好。

将`arithmetic.py`转换成一个包的第二种方式与第一种类似，但是涉及到使用比`__init__.py`更多的文件。在实代码中，函数/类等。在中，每个文件都会以某种方式分组——可能一个文件用于所有的定制异常，一个文件用于公共工具，一个文件用于主要功能。

对于我们的例子，您只需将`arithmetic.py`中的四个函数拆分到它们自己的文件中。继续将每个函数从`__init__.py`移到它自己的文件中。您的文件夹结构应该如下所示:

```py
arithmetic/
    __init__.py
    add.py
    subtract.py
    multiply.py
    divide.py

```

对于`__init__.py`文件，您可以添加以下代码:

```py
# __init__.py
from .add import add
from .subtract import subtract
from .multiply import multiply
from .divide import divide

```

现在你已经做了这些改变，你的下一步应该是什么？希望你说，“运行我的测试！”如果您的测试仍然通过，那么您没有破坏您的 API。

现在，如果你碰巧和你的`test_arithmetic.py`文件在同一个文件夹中，你的`arithmetic`包只对你的其他 Python 代码可用。要使它在 Python 会话或其他 Python 代码中可用，您可以使用 Python 的`sys`模块将您的包添加到 Python 搜索路径。当您使用`import`关键字时，Python 使用搜索路径来查找模块。通过打印出`sys.path`，你可以看到 Python 搜索了哪些路径。

假设你的`arithmetic`文件夹在这个位置:`/Users/michael/packages/arithmetic`。要将它添加到 Python 的搜索路径中，您可以这样做:

```py
import sys

sys.path.append("/Users/michael/packages/arithmetic")
import arithmetic

print(arithmetic.add(1, 2))

```

这将把`arithmetic`添加到 Python 的路径中，这样你就可以导入它，然后在你的代码中使用这个包。然而，那真的很尴尬。如果你能使用`pip`来安装你的软件包就好了，这样你就不用一直摆弄路径了。

让我们看看接下来该怎么做！

## 为 PyPI 打包项目

在为 Python 包索引(PyPI)创建包时，您将需要一些附加文件。这里有一个很好的教程，介绍了创建包并上传到 PyPI 的过程:

*   [https://packaging.python.org/tutorials/packaging-projects/](https://packaging.python.org/tutorials/packaging-projects/)

官方包装说明建议您建立如下目录结构:

```py
my_package/
    LICENSE
    README.md
    arithmetic/
        __init__.py
        add.py
        subtract.py
        multiply.py
        divide.py
    setup.py
    tests/

```

`tests`文件夹可以是空的。这是您将包含包测试的文件夹。大多数开发人员使用 Python 的`unittest`或`pytest`框架进行测试。对于本例，您可以将文件夹留空。

让我们继续，在下一节中学习您需要创建的其他文件！

## 创建项目文件

在**许可证**文件中，您可以提到您的软件包拥有什么许可证。这告诉软件包的用户他们能做什么，不能做什么。您可以使用许多不同的许可证。GPL 和 MIT 许可证只是两个流行的例子。

**README.md** 文件是对你的项目的描述，用 Markdown 编写。您将希望在这个文件中写下您的项目，并包括它可能需要的关于依赖项的任何信息。您可以给出安装说明以及软件包的示例用法。Markdown 是相当多才多艺的，甚至让你做语法突出！

你需要提供的另一个文件是`setup.py`。该文件更复杂，因此您将在下一节中了解这一点。

## 正在创建`setup.py`

有一个名为`setup.py`的特殊文件，用作 Python 发行版的构建脚本。它由`setuptools`使用，它为您进行实际的构建。如果你想了解更多关于`setuptools`的信息，那么你应该看看以下内容:

*   [https://setuptools.readthedocs.io/en/latest/](https://setuptools.readthedocs.io/en/latest/)

你可以使用`setup.py`来创建一个 Python **轮子**。wheel 是一个 ZIP 格式的存档文件，有一个特殊格式的名称和一个`.whl`扩展名。它包含安装软件包所需的一切。你可以把它想象成你的代码的压缩版本，`pip`可以帮你解压并安装。车轮遵循 PEP 376，您可以在这里阅读:

*   [https://www.python.org/dev/peps/pep-0376/](https://www.python.org/dev/peps/pep-0376/)

一旦你读完了所有的文档(如果你想的话)，你就可以创建你的`setup.py`并把这段代码添加进去:

```py
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arithmetic-YOUR-USERNAME-HERE", # Replace with your own username
    version="0.0.1",
    author="Mike Driscoll",
    author_email="driscoll@example.com",
    description="A simple arithmetic package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/driscollis/arithmetic",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

```

这段代码的第一步是导入`setuptools`。然后，将您的`README.md`文件读入一个您很快就会用到的变量。最后一位是大部分代码。这里你调用`setuptools.setup()`，它可以接受很多不同的参数。上面的例子只是你可以传递给这个函数的一个例子。要查看完整列表，您需要访问此处:

*   [https://packaging . python . org/guides/distributing-packages-using-setup tools/](https://packaging.python.org/guides/distributing-packages-using-setuptools/)

大多数论点都是不言自明的。让我们把注意力集中在更迟钝的人身上。`packages`参数是您的包所需的包的列表。在这种情况下，您使用`find_packages()`自动为您找到必要的包。`classifiers`参数用于向`pip`传递额外的元数据。例如，这段代码告诉`pip`这个包是 Python 3 兼容的。

现在你已经有了一个`setup.py`，你已经准备好创建一个 Python 轮子了！

## 生成 Python 轮子

`setup.py`用于创建 Python 轮子。确保你已经安装了最新版本的`setuptools`和`wheel`总是一个好主意，所以在你创建你自己的轮子之前，你应该运行下面的命令:

```py
python3 -m pip install --user --upgrade setuptools wheel

```

如果有比您当前安装的版本更新的版本，这将更新软件包。现在你已经准备好自己创建一个轮子了。打开命令提示符或终端应用程序，导航到包含您的`setup.py`文件的文件夹。然后运行以下命令:

```py
python3 setup.py sdist bdist_wheel

```

该命令将输出大量文本，但一旦完成，您会发现一个名为`dist`的新文件夹，其中包含以下两个文件:

*   `arithmetic_YOUR_USERNAME_HERE-0.0.1-py3-none-any.whl`
*   `arithmetic-YOUR-USERNAME-HERE-0.0.1.tar.gz`

`tar.gz`是一个源文件，这意味着它包含了您的包的 Python 源代码。如果需要，您的用户可以使用源归档文件在他们自己的机器上构建包。`whl`格式是一个归档文件，由`pip`用来在用户的机器上安装你的软件包。

如果需要，您可以使用`pip`直接安装车轮:

```py
python3 -m pip install arithmetic_YOUR_USERNAME_HERE-0.0.1-py3-none-any.whl

```

但是通常的方法是将您的包上传到 Python 包索引(PyPI ),然后安装它。接下来，让我们来看看如何在 PyPI 上获得您的惊喜套餐吧！

## 上传到 PyPI

将包上传到 PyPI 的第一步是在 *Test PyPI* 上创建一个帐户。这允许您测试您的包是否可以上传到测试服务器上，并从该测试服务器安装。要创建帐户，请转到以下 URL 并按照该页面上的步骤操作:

*   [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)

现在您需要创建一个 PyPI API 令牌。这将允许您安全地上传软件包。要获得 API 令牌，您需要转到这里:

*   [https://test.pypi.org/manage/account/#api-tokens](https://test.pypi.org/manage/account/#api-tokens)

您可以限制令牌的范围。但是，您不需要为这个令牌这样做，因为您正在为一个新项目创建它。确保在关闭页面之前复制令牌并将其保存在某个地方。一旦页面关闭，您将无法再次检索令牌。您将需要创建一个新的令牌。

现在您已经注册并拥有了 API 令牌，您将需要获得`twine`包。您将使用`twine`将您的包上传到 PyPI。要安装`twine`，可以这样使用`pip`:

```py
python3 -m pip install --user --upgrade twine

```

安装完成后，您可以使用以下命令将您的包上传到测试 PyPI:

```py
python3 -m twine upload --repository testpypi dist/*

```

注意，您需要从包含`setup.py`文件的文件夹中运行这个命令，因为它正在复制`dist`文件夹中的所有文件来测试 PyPI。当您运行这个命令时，它会提示您输入用户名和密码。对于用户名，您需要使用`__token__`。密码是以`pypi-`为前缀的令牌值。

当此命令运行时，您应该会看到类似如下的输出:

```py
Uploading distributions to https://test.pypi.org/legacy/
Enter your username: [your username]
Enter your password:
Uploading arithmetic_YOUR_USERNAME_HERE-0.0.1-py3-none-any.whl
100%|?????????????????????| 4.65k/4.65k [00:01<00:00, 2.88kB/s]
Uploading arithmetic_YOUR_USERNAME_HERE-0.0.1.tar.gz
100%|?????????????????????| 4.25k/4.25k [00:01<00:00, 3.05kB/s]

```

至此，您应该能够在 Test PyPI 上查看您的包，网址如下:

*   [https://test.pypi.org/project/arithmetic_YOUR_USERNAME_HERE](https://test.pypi.org/project/arithmetic_YOUR_USERNAME_HERE)

现在您可以使用下面的命令从 Test PyPI 测试安装您的包:

```py
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps arithmetic-YOUR-USERNAME-HERE

```

如果一切正常，您现在应该已经在系统上安装了`arithmetic`包。当然，本教程向您展示了如何打包测试 PyPI。一旦您验证了它可以工作，那么您将需要执行以下操作来安装到真正的 PyPI:

*   为包装选择一个容易记忆且独特的名称
*   在[https://pypi.org](https://pypi.org)注册账户
*   使用`twine upload dist/*`上传您的包裹，并输入您在 real PyPI 上注册的帐户的凭证。当上传到真正的 PyPI 时，您不需要使用`--repository`标志，因为该服务器是默认的
*   使用`pip install your_unique_package_name`从真实的 PyPI 安装你的包

现在您知道如何在 Python 包索引上分发您自己创建的包了！

## 包扎

Python 模块和包是您在程序中导入的内容。从很多方面来说，它们是你的程序的组成部分。在本文中，您了解了以下内容:

*   创建模块
*   创建包
*   为 PyPI 打包项目
*   创建项目文件
*   正在创建`setup.py`
*   上传到 PyPI

至此，您不仅知道了什么是模块和包，还知道了如何通过 Python 包索引来分发它们。现在，您和任何其他 Python 开发人员都可以下载并安装您的包。恭喜你！你现在是一个包维护者了！

## 相关阅读

这篇文章基于 **Python 101 第二版**中的一章，你可以在 [Leanpub](https://leanpub.com/py101) 或[亚马逊](https://amzn.to/2Zo1ARG)上购买。

如果你想学习更多的 Python，那么看看这些教程:

*   python 101—[如何处理图像](https://www.blog.pythonlibrary.org/2021/09/14/python-101-how-to-work-with-images/)

*   python 101-[记录你的代码](https://www.blog.pythonlibrary.org/2021/09/12/documenting-code/)

*   Python 101: [使用 JSON 的介绍](https://www.blog.pythonlibrary.org/2020/09/15/python-101-an-intro-to-working-with-json/)

*   python 101-[创建多个流程](https://www.blog.pythonlibrary.org/2020/07/15/python-101-creating-multiple-processes/)