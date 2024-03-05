# Pipenv:新 Python 打包工具指南

> 原文：<https://realpython.com/pipenv-guide/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。与书面教程一起观看，加深理解: [**与 Pipenv**](/courses/working-with-pipenv/) 一起工作

Pipenv 是 Python 的一个打包工具，它使用[`pip`](https://realpython.com/what-is-pip/)`virtualenv`和传统的`requirements.txt`解决了一些与典型工作流相关的常见问题。

除了解决一些常见问题之外，它还将开发过程整合并简化为一个命令行工具。

本指南将介绍 Pipenv 解决了哪些问题，以及如何用 Pipenv 管理 Python 依赖关系。此外，它还将介绍 Pipenv 如何与之前的[包装](https://realpython.com/python-modules-packages/)配送方法相适应。

**免费奖励:** ，向您展示如何使用 Pip、PyPI、Virtualenv 和需求文件等工具避免常见的依赖管理问题。

## Pipenv 解决的问题

为了理解 Pipenv 的好处，了解一下 Python 中当前的打包和依赖管理方法是很重要的。

先说一个处理第三方包的典型情况。然后，我们将构建部署完整 Python 应用程序的方法。

[*Remove ads*](/account/join/)

### 依赖关系管理用`requirements.txt`

想象一下你正在做一个 [Python 项目](https://realpython.com/intermediate-python-project-ideas/)，它使用了像`flask`这样的第三方包。您需要指定该需求，以便其他开发人员和自动化系统可以运行您的应用程序。

所以您决定在一个`requirements.txt`文件中包含`flask`依赖项:

```py
flask
```

太好了，在本地一切都很好，在你的应用程序上破解了一段时间后，你决定把它转移到生产中。这就是事情变得有点可怕的地方…

上面的`requirements.txt`文件没有指定使用哪个版本的`flask`。在这种情况下，`pip install -r requirements.txt`会默认安装最新版本。这是可以的，除非在最新的版本中有接口或行为的变化会破坏我们的应用程序。

为了这个例子，让我们假设一个新版本的`flask`发布了。然而，它并不向后兼容您在开发过程中使用的版本。

现在，假设您将应用程序部署到生产环境中，并执行`pip install -r requirements.txt`。 [Pip](https://realpython.com/what-is-pip/) 获得最新的、不向后兼容的`flask`版本，就这样，您的应用程序中断了…生产。

“但是，嘿，它在我的机器上工作了！”—我自己也去过，感觉不是很棒。

此时，您知道您在开发期间使用的版本`flask`工作正常。所以，为了解决问题，你应该在你的`requirements.txt`中尽量具体一点。您向`flask`依赖项添加了一个*版本说明符*。这也被称为*钉住*依赖关系:

```py
flask==0.12.1
```

将`flask`依赖固定到一个特定的版本可以确保一个`pip install -r requirements.txt`设置你在开发过程中使用的`flask`的精确版本。但是真的吗？

请记住，`flask`本身也有依赖项(由`pip`自动安装)。然而，`flask`本身并没有为它的依赖项指定确切的版本。比如它允许任何版本的`Werkzeug>=0.14`。

再一次，为了这个例子，让我们假设一个新版本的`Werkzeug`发布了，但是它给你的应用程序引入了一个大错误。

当您这次在生产中执行`pip install -r requirements.txt`时，您将获得`flask==0.12.1`,因为您已经锁定了那个需求。然而，不幸的是，你将得到最新的，有缺陷的版本`Werkzeug`。同样，产品在生产中出现故障。

这里真正的问题是**构建不是确定性的**。我的意思是，给定相同的输入(`requirements.txt`文件)，pip 并不总是产生相同的环境。目前，您无法在生产中轻松复制您的开发机器上的确切环境。

这个问题的典型解决方案是使用`pip freeze`。此命令允许您获取当前安装的所有第三方库的确切版本，包括自动安装的子依赖项 pip。因此，您可以冻结开发中的一切，以确保您在生产中拥有相同的环境。

执行`pip freeze`会导致可以添加到`requirements.txt`的固定依赖关系:

```py
click==6.7
Flask==0.12.1
itsdangerous==0.24
Jinja2==2.10
MarkupSafe==1.0
Werkzeug==0.14.1
```

有了这些固定的依赖项，您可以确保安装在生产环境中的包与开发环境中的包完全匹配，这样您的产品就不会意外中断。不幸的是，这种“解决方案”导致了一系列全新的问题。

既然您已经指定了每个第三方包的确切版本，您有责任保持这些版本最新，即使它们是`flask`的子依赖项。如果在`Werkzeug==0.14.1`中发现了一个安全漏洞，包的维护者立即在`Werkzeug==0.14.2`中打了补丁，那该怎么办？你真的需要升级到`Werkzeug==0.14.2`，以避免任何由`Werkzeug`早期未打补丁版本引起的安全问题。

首先，你需要意识到你的版本有问题。然后，您需要在有人利用安全漏洞之前，在您的生产环境中获得新版本。因此，您必须手动更改您的`requirements.txt`来指定新版本`Werkzeug==0.14.2`。正如您在这种情况下所看到的，保持必要更新的责任落在了您的身上。

事实是，只要不破坏您的代码，您真的不在乎安装了什么版本的`Werkzeug`。事实上，您可能希望获得最新版本，以确保您获得错误修复、安全补丁、新功能、更多优化等等。

真正的问题是:**“您如何在不承担更新子依赖项版本的责任的情况下，允许您的 Python 项目的确定性构建？”**

> 剧透:简单的答案是使用 Pipenv。

[*Remove ads*](/account/join/)

### 开发具有不同依赖关系的项目

让我们稍微转换一下话题，来谈谈当你在多个项目中工作时出现的另一个常见问题。想象一下`ProjectA`需要`django==1.9`，但是`ProjectB`需要`django==1.10`。

默认情况下，Python 试图将所有第三方包存储在系统范围的位置。这意味着每次你想在`ProjectA`和`ProjectB`之间切换时，你必须确保安装了正确版本的`django`。这使得在项目之间切换很痛苦，因为您必须卸载并重新安装软件包来满足每个项目的需求。

标准的解决方案是使用一个拥有自己的 Python 可执行文件和第三方包存储的虚拟环境。这样，`ProjectA`和`ProjectB`就充分分开了。现在，您可以轻松地在项目之间切换，因为它们不共享同一个包存储位置。`PackageA`可以在自己的环境中拥有它所需要的`django`的任何版本，`PackageB`可以拥有它所需要的完全独立的版本。一个非常常见的工具是`virtualenv`(或者 Python 3 中的`venv`)。

Pipenv 具有内置的虚拟环境管理功能，因此您只需使用一个工具来管理您的软件包。

### 依赖性解析

我说的依赖解析是什么意思？假设您有一个类似这样的`requirements.txt`文件:

```py
package_a
package_b
```

假设`package_a`有一个子依赖`package_c`,`package_a`需要这个包的一个特定版本:`package_c>=1.0`。反过来，`package_b`具有相同的子依赖关系，但是需要`package_c<=2.0`。

理想情况下，当你试图安装`package_a`和`package_b`时，安装工具会查看对`package_c`(即`>=1.0`和`<=2.0`)的需求，并选择一个满足这些需求的版本。您希望该工具能够解决依赖性，以便您的程序最终能够正常工作。这就是我所说的“依赖性解决方案”

不幸的是，pip 本身目前没有真正的依赖性解决方案，但是有一个[开放问题](https://github.com/pypa/pip/issues/988)来支持它。

pip 处理上述场景的方式如下:

1.  它安装`package_a`并寻找满足第一个需求的`package_c`版本(`package_c>=1.0`)。

2.  然后 Pip 安装最新版本的`package_c`来满足这个需求。假设`package_c`的最新版本是 3.1。

这是麻烦(潜在的)开始的地方。

如果 pip 选择的`package_c`版本不符合未来需求(如`package_b`需要`package_c<=2.0`，安装将失败。

这个问题的“解决方案”是在`requirements.txt`文件中指定子依赖项(`package_c`)所需的范围。这样，pip 可以解决这种冲突，并安装满足这些要求的软件包:

```py
package_c>=1.0,<=2.0
package_a
package_b
```

就像以前一样，你现在直接关注子依赖项(`package_c`)。问题是，如果`package_a`在你不知道的情况下改变了他们的需求，你指定的需求(`package_c>=1.0,<=2.0`)可能不再被接受，安装可能会再次失败。真正的问题是，再一次，你要负责保持子依赖的最新需求。

理想情况下，您的安装工具应该足够智能，能够安装满足所有需求的包，而无需您显式指定子依赖版本。

## Pipenv 简介

现在我们已经解决了这些问题，让我们看看 Pipenv 如何解决它们。

首先，让我们安装它:

```py
$ pip install pipenv
```

一旦你做到了这一点，你就可以有效地忘记`pip`，因为 Pipenv 本质上是一个替代品。它还引入了两个新文件， [`Pipfile`](https://github.com/pypa/pipfile) (意在取代`requirements.txt`)和`Pipfile.lock`(支持确定性构建)。

Pipenv 在底层使用了`pip`和`virtualenv`,但是通过一个命令行界面简化了它们的使用。

[*Remove ads*](/account/join/)

### 用法示例

让我们从创建令人惊叹的 Python 应用程序开始。首先，在虚拟环境中生成一个 shell 来隔离这个应用程序的开发:

```py
$ pipenv shell
```

如果虚拟环境不存在，这将创建一个虚拟环境。Pipenv 在默认位置创建所有虚拟环境。如果你想改变 Pipenv 的默认行为，有一些[环境变量用于配置](https://docs.pipenv.org/advanced/#configuration-with-environment-variables)。

您可以分别使用参数`--two`和`--three`强制创建 Python 2 或 3 环境。否则，Pipenv 将使用`virtualenv`找到的任何缺省值。

> 旁注:如果您需要一个更具体的 Python 版本，您可以提供一个`--python`参数来指定您需要的版本。比如:`--python 3.6`

现在你可以安装你需要的第三方包了，`flask`。哦，但是你知道你需要的是版本`0.12.1`而不是最新的版本，所以请具体说明:

```py
$ pipenv install flask==0.12.1
```

您应该会在终端中看到如下内容:

```py
Adding flask==0.12.1 to Pipfile's [packages]...
Pipfile.lock not found, creating...
```

您会注意到创建了两个文件，一个`Pipfile`和`Pipfile.lock`。我们一会儿会仔细看看这些。让我们安装另一个第三方软件包，`numpy`，进行一些数字运算。您不需要特定的版本，所以不要指定:

```py
$ pipenv install numpy
```

如果你想直接从版本控制系统(VCS)安装一些东西，你可以！您可以像使用`pip`一样指定位置。例如，要从版本控制安装`requests`库，请执行以下操作:

```py
$ pipenv install -e git+https://github.com/requests/requests.git#egg=requests
```

注意上面的`-e`参数，使安装可编辑。目前，[这是 Pipenv 进行子依赖解析所需要的](https://pipenv.readthedocs.io/en/latest/basics/#a-note-about-vcs-dependencies)。

假设您也有这个令人敬畏的应用程序的一些单元测试，并且您想要使用 [`pytest`](https://realpython.com/pytest-python-testing/) 来运行它们。您在生产中不需要`pytest`,因此您可以使用`--dev`参数指定该依赖项仅用于开发:

```py
$ pipenv install pytest --dev
```

提供`--dev`参数将把依赖关系放在`Pipfile`中的一个特殊的`[dev-packages]`位置。只有当您用`pipenv install`指定了`--dev`参数时，这些开发包才会被安装。

不同的部分将开发所需的依赖项与基本代码实际工作所需的依赖项分开。一般来说，这可以通过附加的需求文件来完成，比如`dev-requirements.txt`或者`test-requirements.txt`。现在，所有的事情都整合在一个`Pipfile`的不同部门下。

好了，让我们假设您已经在本地开发环境中做好了一切准备，并准备将其推向生产。要做到这一点，您需要锁定您的环境，以便确保您在生产中拥有相同的环境:

```py
$ pipenv lock
```

这将创建/更新您的`Pipfile.lock`，您将永远不需要(也不打算)手动编辑它。您应该始终使用生成的文件。

现在，一旦您在生产环境中获得了代码和`Pipfile.lock`,您应该安装记录的最后一个成功的环境:

```py
$ pipenv install --ignore-pipfile
```

这告诉 Pipenv 忽略`Pipfile`进行安装，并使用`Pipfile.lock`中的内容。给定这个`Pipfile.lock`，Pipenv 将创建与您运行`pipenv lock`时完全相同的环境，包括子依赖项和所有内容。

锁文件通过获取环境中所有包版本的快照(类似于`pip freeze`的结果)来实现确定性构建。

现在让我们假设另一个开发人员想要对您的代码进行一些添加。在这种情况下，他们将获得代码，包括`Pipfile`，并使用以下命令:

```py
$ pipenv install --dev
```

这将安装开发所需的所有依赖项，包括常规依赖项和您在`install`期间用`--dev`参数指定的依赖项。

> 当 Pipfile 中没有指定一个确切的版本时，`install`命令为依赖关系(和子依赖关系)提供了更新版本的机会。

这是一个重要的注意事项，因为它解决了我们之前讨论的一些问题。为了演示，假设您的一个依赖项的新版本变得可用。因为您不需要这个依赖项的特定版本，所以您不需要在`Pipfile`中指定一个确切的版本。当您`pipenv install`时，新版本的依赖项将被安装到您的开发环境中。

现在，您对代码进行更改，并运行一些测试来验证一切仍按预期运行。(你有单元测试，对吗？)现在，就像以前一样，您使用`pipenv lock`锁定您的环境，并且将使用依赖关系的新版本生成更新的`Pipfile.lock`。和以前一样，您可以用锁文件在生产中复制这个新环境。

正如您在这个场景中看到的，您不再需要强制使用您并不真正需要的确切版本来确保您的开发和生产环境是相同的。您也不需要一直更新您“不关心”的子依赖项 Pipenv 的这个工作流，结合您出色的测试，解决了手动进行所有依赖管理的问题。

[*Remove ads*](/account/join/)

### Pipenv 的依赖性解决方法

Pipenv 将尝试安装满足核心依赖项所有要求的子依赖项。但是，如果存在相互冲突的依赖关系(`package_a`需要`package_c>=1.0`，但是`package_b`需要`package_c<1.0`)，Pipenv 将无法创建锁文件，并将输出如下错误:

```py
Warning: Your dependencies could not be resolved. You likely have a mismatch in your sub-dependencies.
  You can use $ pipenv install --skip-lock to bypass this mechanism, then run $ pipenv graph to inspect the situation.
Could not find a version that matches package_c>=1.0,package_c<1.0
```

正如警告所说，您还可以显示一个依赖关系图来了解您的顶级依赖关系及其子依赖关系:

```py
$ pipenv graph
```

该命令将打印出一个树状结构，显示您的依赖关系。这里有一个例子:

```py
Flask==0.12.1
  - click [required: >=2.0, installed: 6.7]
  - itsdangerous [required: >=0.21, installed: 0.24]
  - Jinja2 [required: >=2.4, installed: 2.10]
    - MarkupSafe [required: >=0.23, installed: 1.0]
  - Werkzeug [required: >=0.7, installed: 0.14.1]
numpy==1.14.1
pytest==3.4.1
  - attrs [required: >=17.2.0, installed: 17.4.0]
  - funcsigs [required: Any, installed: 1.0.2]
  - pluggy [required: <0.7,>=0.5, installed: 0.6.0]
  - py [required: >=1.5.0, installed: 1.5.2]
  - setuptools [required: Any, installed: 38.5.1]
  - six [required: >=1.10.0, installed: 1.11.0]
requests==2.18.4
  - certifi [required: >=2017.4.17, installed: 2018.1.18]
  - chardet [required: >=3.0.2,<3.1.0, installed: 3.0.4]
  - idna [required: >=2.5,<2.7, installed: 2.6]
  - urllib3 [required: <1.23,>=1.21.1, installed: 1.22]
```

从`pipenv graph`的输出中，可以看到我们之前安装的顶层依赖项(`Flask`、`numpy`、`pytest`和`requests`)，下面可以看到它们所依赖的包。

此外，您可以反转树来显示需要它的父级的子依赖关系:

```py
$ pipenv graph --reverse
```

当您试图找出冲突的子依赖关系时，这种反向树可能更有用。

### Pipfile

[Pipfile](https://github.com/pypa/pipfile) 打算取代`requirements.txt`。Pipenv 目前是使用`Pipfile`的参考实现。看来很有可能 [`pip`本身就能够处理这些文件](https://github.com/pypa/pipfile#pip-integration-eventual)。还有，值得注意的是 [Pipenv 甚至是 Python 自己推荐的官方包管理工具](https://packaging.python.org/tutorials/managing-dependencies/#managing-dependencies)。

`Pipfile`的语法是 [TOML](https://realpython.com/python-toml/) ，文件被分成几个部分。`[dev-packages]`用于开发包，`[packages]`用于最低需求包，`[requires]`用于其他需求，比如特定版本的 Python。请参见下面的示例文件:

```py
[[source]] url  =  "https://pypi.python.org/simple" verify_ssl  =  true name  =  "pypi" [dev-packages] pytest  =  "*" [packages] flask  =  "==0.12.1" numpy  =  "*" requests  =  {git = "https://github.com/requests/requests.git", editable = true} [requires] python_version  =  "3.6"
```

理想情况下，您的`Pipfile`中不应该有任何子依赖项。我的意思是你应该只包含你实际导入和使用的包。不需要仅仅因为`chardet`是`requests`的一个子依赖项，就将`chardet`保留在你的`Pipfile`中。(Pipenv 会自动安装。)T4 应该传达你的包所需要的顶层依赖关系。

### Pipfile.lock

该文件通过指定重现环境的确切要求来实现确定性构建。它包含了包和哈希的精确版本，以支持更安全的验证， [`pip`现在也支持](https://pip.pypa.io/en/stable/reference/pip_install/#hash-checking-mode)。一个示例文件可能如下所示。注意，这个文件的语法是 JSON，我已经用`...`排除了文件的一部分:

```py
{ "_meta":  { ... }, "default":  { "flask":  { "hashes":  [ "sha256:6c3130c8927109a08225993e4e503de4ac4f2678678ae211b33b519c622a7242", "sha256:9dce4b6bfbb5b062181d3f7da8f727ff70c1156cbb4024351eafd426deb5fb88" ], "version":  "==0.12.1" }, "requests":  { "editable":  true, "git":  "https://github.com/requests/requests.git", "ref":  "4ea09e49f7d518d365e7c6f7ff6ed9ca70d6ec2e" }, "werkzeug":  { "hashes":  [ "sha256:d5da73735293558eb1651ee2fddc4d0dedcfa06538b8813a2e20011583c9e49b", "sha256:c3fd7a7d41976d9f44db327260e263132466836cef6f91512889ed60ad26557c" ], "version":  "==0.14.1" } ... }, "develop":  { "pytest":  { "hashes":  [ "sha256:8970e25181e15ab14ae895599a0a0e0ade7d1f1c4c8ca1072ce16f25526a184d", "sha256:9ddcb879c8cc859d2540204b5399011f842e5e8823674bf429f70ada281b3cc6" ], "version":  "==3.4.1" }, ... } }
```

请注意为每个依赖项指定的确切版本。甚至像`werkzeug`这样不在我们的`Pipfile`中的子依赖项也出现在这个`Pipfile.lock`中。哈希用于确保您检索到的包与开发时相同。

再次值得注意的是，您不应该手动更改这个文件。它是用`pipenv lock`生成的。

[*Remove ads*](/account/join/)

### Pipenv 额外功能

使用以下命令在默认编辑器中打开第三方包:

```py
$ pipenv open flask
```

这将在默认编辑器中打开`flask`包，或者您可以指定一个带有`EDITOR`环境变量的程序。比如我用[崇高文字](https://www.sublimetext.com/)，所以我只设置`EDITOR=subl`。这使得深入研究您正在使用的包的内部变得非常简单。

* * *

您可以在虚拟环境中运行命令，而无需启动 shell:

```py
$ pipenv run <insert command here>
```

* * *

检查您环境中的安全漏洞(以及 [PEP 508](https://www.python.org/dev/peps/pep-0508/) 要求):

```py
$ pipenv check
```

* * *

现在，让我们说你不再需要一个包。您可以卸载它:

```py
$ pipenv uninstall numpy
```

此外，假设您想要从虚拟环境中完全清除所有已安装的软件包:

```py
$ pipenv uninstall --all
```

您可以用`--all-dev`替换`--all`来删除开发包。

* * *

当顶层目录中存在一个`.env`文件时，Pipenv 支持自动加载环境变量。这样，当你`pipenv shell`打开虚拟环境时，它从文件中加载你的环境变量。`.env`文件只包含键值对:

```py
SOME_ENV_CONFIG=some_value
SOME_OTHER_ENV_CONFIG=some_other_value
```

* * *

最后，这里有一些快速的命令来找出东西在哪里。如何找到您的虚拟环境:

```py
$ pipenv --venv
```

如何找到您的项目主页:

```py
$ pipenv --where
```

[*Remove ads*](/account/join/)

## 包装分发

您可能会问，如果您打算将代码作为一个包分发，这一切是如何工作的。

### 是的，我需要将我的代码打包分发

Pipenv 如何处理`setup.py`文件？

这个问题有很多微妙之处。首先，当您使用`setuptools`作为构建/发布系统时，一个`setup.py`文件是必要的。这已经成为事实上的标准有一段时间了，但是[最近的变化](https://www.python.org/dev/peps/pep-0518/)使得`setuptools`的使用成为可选的。

这意味着像 [flit](https://github.com/takluyver/flit) 这样的项目可以使用新的 [`pyproject.toml`](https://realpython.com/courses/packaging-with-pyproject-toml/) 来指定不需要`setup.py`的不同构建系统。

尽管如此，在不久的将来`setuptools`和伴随的`setup.py`仍将是许多人的默认选择。

当您使用`setup.py`作为分发包的方式时，以下是推荐的工作流程:

*   `setup.py`
*   `install_requires`关键字应该包括包[“最低限度需要正确运行”的任何内容](https://packaging.python.org/discussions/install-requires-vs-requirements/)
*   `Pipfile`
*   代表您的包的具体要求
*   通过使用 Pipenv 安装您的软件包，从`setup.py`中提取最低要求的依赖项:
    *   使用`pipenv install '-e .'`
    *   这将在您的`Pipfile`中产生一行类似于`"e1839a8" = {path = ".", editable = true}`的内容。
*   `Pipfile.lock`
*   从`pipenv lock`生成的可再现环境的详细信息

澄清一下，把你的最低要求放在`setup.py`里，而不是直接用`pipenv install`。然后使用`pipenv install '-e .'`命令将你的包安装成可编辑的。这将把所有需求从`setup.py`放到您的环境中。然后你可以使用`pipenv lock`来获得一个可复制的环境。

### 我不需要将我的代码作为一个包分发

太好了！如果你正在开发一个不打算发布或安装的应用程序(一个个人网站，一个桌面应用程序，一个游戏，或者类似的)，你真的不需要一个`setup.py`。

在这种情况下，您可以使用`Pipfile` / `Pipfile.lock`组合来管理您与前面描述的流程的依赖关系，以便在生产中部署一个可再现的环境。

## 我已经有一个`requirements.txt`。我如何转换成一个`Pipfile`？

如果您运行`pipenv install`，它应该会自动检测到`requirements.txt`并将其转换为`Pipfile`，输出如下内容:

```py
requirements.txt found, instead of Pipfile! Converting…
Warning: Your Pipfile now contains pinned versions, if your requirements.txt did.
We recommend updating your Pipfile to specify the "*" version, instead.
```

> 请注意上面的警告。

如果您已经在`requirements.txt`文件中固定了精确的版本，那么您可能希望更改`Pipfile`来指定您真正需要的精确版本。这将让你获得转型的真正好处。例如，假设您有以下内容，但真的不需要确切版本的`numpy`:

```py
[packages] numpy  =  "==1.14.1"
```

如果您对您的依赖项没有任何特定的版本要求，您可以使用通配符`*`告诉 Pipenv 可以安装任何版本:

```py
[packages] numpy  =  "*"
```

如果您对允许任何带有`*`的版本感到紧张，通常安全的做法是指定大于或等于您已经使用的版本，这样您仍然可以利用新版本:

```py
[packages] numpy  =  ">=1.14.1"
```

当然，保持与新版本的同步也意味着当包发生变化时，您有责任确保您的代码仍能按预期运行。这意味着，如果您想要确保代码的功能发布，测试套件对于整个 Pipenv 流程是必不可少的。

您允许包更新，运行您的测试，确保它们都通过，锁定您的环境，然后您就可以高枕无忧了，因为您知道您没有引入突破性的变化。如果事情确实因为依赖关系而中断，您需要编写一些回归测试，并且可能对依赖关系的版本有更多的限制。

例如，如果在运行`pipenv install`之后安装了`numpy==1.15`，并且它破坏了您的代码，您希望在开发或测试期间注意到这一点，您有几个选择:

1.  更新您的代码以使用新版本的依赖项。

    如果向后兼容以前版本的依赖项是不可能的，那么您还需要在您的`Pipfile`中添加您需要的版本:

    ```py
    [packages] numpy  =  ">=1.15"` 
    ```

2.  将`Pipfile`中依赖项的版本限制为`<`刚刚破坏代码的版本:

    ```py
    [packages] numpy  =  ">=1.14.1,<1.15"` 
    ```

首选选项 1，因为它可以确保您的代码使用最新的依赖项。然而，选项 2 花费的时间更少，并且不需要修改代码，只需要限制依赖关系。

* * *

您也可以从需求文件安装，使用相同的`-r`参数`pip`:

```py
$ pipenv install -r requirements.txt
```

如果您有一个`dev-requirements.txt`或类似的东西，您也可以将它们添加到`Pipfile`中。只需添加`--dev`参数，使其放在正确的部分:

```py
$ pipenv install -r dev-requirements.txt --dev
```

此外，您可以走另一条路，从`Pipfile`生成需求文件:

```py
$ pipenv lock -r > requirements.txt
$ pipenv lock -r -d > dev-requirements.txt
```

[*Remove ads*](/account/join/)

## 下一步是什么？

在我看来，Python 生态系统的一个自然发展将是一个构建系统，当从包索引(如 PyPI)中检索和构建包时，它使用`Pipfile`来安装最低要求的依赖项。需要再次注意的是, [Pipfile 设计规范](https://github.com/pypa/pipfile)仍在开发中，Pipenv 只是一个参考实现。

也就是说，我可以预见一个不存在`setup.py`的`install_requires`部分，而引用`Pipfile`作为最低要求的未来。或者`setup.py`完全消失，您以不同的方式获得元数据和其他信息，仍然使用`Pipfile`获得必要的依赖关系。

## Pipenv 值得一查吗？

绝对的。即使它只是作为一种将您已经使用的工具(`pip` & `virtualenv`)整合到一个单一界面的方式。然而，远不止如此。通过添加`Pipfile`，您可以只指定您真正需要的依赖项。

您不再为仅仅为了确保可以复制您的开发环境而亲自管理所有东西的版本而头痛。有了`Pipfile.lock`，你可以安心地发展，因为你知道你可以在任何地方精确地复制你的环境。

除此之外，`Pipfile`格式似乎很有可能被官方 Python 工具如`pip`所采用和支持，所以走在游戏的前面是有益的。哦，还要确保你把所有代码都升级到 Python 3:[2020 年即将到来](https://www.python.org/dev/peps/pep-0373/#maintenance-releases)。

## 参考资料、进一步阅读、有趣的讨论等等

*   [官方 Pipenv 文档](https://docs.pipenv.org/)
*   [官方`Pipfile`项目](https://github.com/pypa/pipfile)
*   [关于`Pipfile`](https://github.com/pypa/pipfile/issues/98) 的问题处理`install_requires`
*   [再论`setup.py`vs`Pipfile`T3】](https://github.com/pypa/pipfile/issues/27)
*   [帖子谈人教版 518](https://snarky.ca/clarifying-pep-518/)
*   [贴在 Python 包装上](https://snarky.ca/a-tutorial-on-python-package-building/)
*   [建议使用 Pipenv 的注释](https://github.com/pypa/pipenv/issues/209#issuecomment-337409290)

**免费奖励:** ，向您展示如何使用 Pip、PyPI、Virtualenv 和需求文件等工具避免常见的依赖管理问题。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。与书面教程一起观看，加深理解: [**与 Pipenv**](/courses/working-with-pipenv/) 一起工作*********