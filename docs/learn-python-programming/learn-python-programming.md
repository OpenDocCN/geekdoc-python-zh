

# 学习Python编程

通过学习编码最佳实践和高级编程概念，全面掌握Python学习与应用指南

![](img/ca275ab0f69aae82698e118840a71527_0_0.png)

詹姆斯·赫伦

# 学习Python编程

通过学习编码最佳实践和高级编程概念，全面掌握Python学习与应用指南

![](img/ca275ab0f69aae82698e118840a71527_1_0.png)

詹姆斯·赫伦

# 学习Python编程

通过学习编码最佳实践和高级编程概念，全面掌握Python学习与应用指南

詹姆斯·赫伦

# 目录

引言

第1章 准备使用Python

- 安装Python
- Python提示符
- 安装Setuptools
- 工作环境

第2章 类级别以下的语法

- 列表推导式
- 迭代器与生成器
- 生成器表达式

第3章 面向对象编程

- 继承
- 多态
- 抽象
- 封装

第4章 基本编程工具

- Bash脚本
- Python正则表达式
- Python包管理器
- 版本控制
- 综合运用

第5章 文件操作

- 创建新文件
- 什么是二进制文件？
- 打开文件

第6章 异常处理

- 处理零除错误异常
- 处理文件未找到异常错误

第7章 Python库

- Caffe
- Theano
- TensorFlow
- Keras
- Sklearn-Theano
- Nolearn
- Digits

第8章 PyTorch库

- PyTorch的起源
- PyTorch社区
- 为何在数据分析中使用PyTorch
- PyTorch 1.0 – 从研究到生产的必经之路

第9章 在Python中创建包

- 每个Python包的通用模式
- 在命名空间包中使用‘setup.py’脚本

结语

# 引言

本书的重点将主要分为两部分。第一部分将涵盖能让读者尽可能高效地使用Python的主题。后半部分将专注于使用户能够实现前面概述的概念。我们将从准备所有必要的工具并在系统上设置Python工作环境开始。这将确保读者在练习或测试后续章节中展示的示例时不会遇到任何不便。第1章介绍了读者在拥有Python的系统上工作所必需的工具。此外，第1章中描述的每个工具都附有适当的解释以及用户如何在系统上轻松下载和安装它们的方法。每个操作系统都有其自身的要求，为确保每位读者都能使用本书，第1章不仅涵盖Windows，还包括Linux和macOS。之后，我们将重新审视高级编程中常用的语法。这些语法将在两个层面上进行解释和实际演示：类级别以下和类级别以上。此后，我们将遇到一个专门阐述编程中命名艺术的章节。大多数人低估了命名过程，没有给予应有的关注。实际上，命名包、模块、自定义函数、模块和类需要非常小心，因为这些名称在后续编码中会被使用。

在前四章之后，我们将进入本书的实践部分。最后两章完全致力于解释可用于在Python中创建包和应用程序的高级技术。这些章节将涵盖本书中讨论的概念以及中级编码中通常教授的技术和实践。这一点尤其重要，因为如果我们最终无法处理一个项目，那么我们在本书中学到的一切都将毫无用处。在最后几章中，我们将经历创建包所采用的过程，然后看看包创建与构建完整应用程序之间的关联。

# 第1章

## 准备使用Python

在当今时代，Python是所有最受欢迎的编程语言之一，在机器学习、数据科学、渗透测试等众多高级领域具有出色的可用性。即使对于应用程序开发和标准编程等场景，Python也不逊于C或C++等其他语言。在本章中，我们将全面讨论如何在系统上正确设置Python以及允许进行高级编程和编码实践的适当工具。初始设置非常重要，因为发现缺少某个组件并不得不回头设置它是非常适得其反的。简而言之，我们将专注于确保在进入实际应用和高级实践（如创建包和应用程序）之前，我们已准备好一切并设置完毕。

### 安装Python

安装和使用Python编程语言很简单，因为它基本上可以在任何操作系统上运行，如Linux、Windows和Macintosh。负责分发和提供此Python编程语言的核心团队运营着一个网站“http://www.python.org/download”，以便轻松下载。Python社区的人员也为其他用户提供了下载平台，其中可能还包含适用于像旧式系统这样的操作系统的发行版。

### Python IDE的不同实现

Python的一个有趣特点是它也可以轻松地在其他编程语言中实现。例如，顾名思义，工具‘CPython’是Python在‘C’中的实现。虽然目前这是最常用的Python实现变体，但针对机器学习、数据科学和数据库操作等新兴领域的其他Python实现也正变得流行。这完全取决于函数式编程领域对于实现的广泛使用有多成功。

#### Jython

正如我们讨论的‘CPython’只是Python编程语言在‘C’中的实现一样，同样，Java（一种用于Web开发的流行语言）也有自己的Python实现，如工具‘Jython’。此实现的主要特点是，主要属于Java语言的类可以在Python模块中定义。此外，跨兼容性也扩展到通常与Java配对的应用程序（如Apache）。换句话说，我们可以利用这些仅限于Java的应用程序提供的功能，并将它们引入用Python编写的程序中。

#### IronPython

借助IronPython，Python被引入.NET框架。在微软工作的IronPython开发人员制作了1.1版本，这是最稳定的版本，实现了Python 2.4.3。IronPython与ASP.NET一起允许在.NET应用程序中使用Python代码，就像Jython在Java中一样。这种Python的实现方式有助于推广该语言。根据TIOBE社区指数，.NET语言正变得越来越受欢迎，并拥有像Java一样庞大的开发者社区。

#### PyPy

这个实现可能有点难以理解，因为它是Python本身的实现。详细来说，此工具中使用的解释器主要负责翻译Python代码。虽然有人可能会想，为Python项目实现Python有什么意义，因为这说不通。但经验丰富的程序员会将实现堆叠在一起，以发挥像这样的工具的真正潜力。例如，如果我们正在处理一个项目，我们可以使用CPython工具让它处理较简单的指令，之后，我们可以使用PyPy工具将代码从CPython翻译成Python语言。许多其他类似的例子证明了PyPy的实用性。起初，这个工具的主要关注点是其平淡无奇的速度，当与CPython比较时，这一点更加明显。但随着‘JIT’等技术的引入，PyPy工具编译器的速度有了显著提高。话虽如此，不建议将PyPy作为主要实现，因为程序员们仍然进行实验。

#### 其他实现方式

除了常用的实现方式外，Python 还有其他有趣的实现和移植版本。例如，适用于诺基亚 S60 手机系列的 Python 2.2.2，以及在 ARM Linux 上的移植版本，使其能够在夏普 Zaurus 等设备上使用。

### Linux 安装

如果操作系统是 Linux，那么 Python 语言可能已经安装好了。要访问它，只需尝试从 shell 中调用它，如下所示：

```
tarek@dabox:~$ python
Python 2.3.5 (#1, Jul 4 2007, 17:28:59)
[GCC 4.1.2 20061115 (prerelease) (Debian 4.1.1-21)] on linux2
>>>
```

一旦 shell 执行完用户的命令，它要么会显示一个错误，详细说明系统上没有有效的 Python 安装，要么会轻松替换 Python 安装的版本并执行 IDE。在 shell 的顶部，你会注意到三个“大于”符号（>>>）。这些符号告诉用户，Python 将在此空间内执行任何写下的有效命令。shell 还会告诉我们 Python 可用的编译器，这取决于不同的操作系统。例如，在运行 Linux 的系统上，编译器将是‘GCC’，而在运行 Windows 的系统上，编译器将是 Visual Studio。

如果你确实在运行基于 Linux 的系统，你也可以安装其他版本的 Python。有了多个 Python 安装，你只需在执行命令时传递你希望 shell 运行的精确版本即可。这样，shell 就不会执行完全不同版本的 Python 或同时执行所有 Python 版本。这里有一个示例来说明这一点：

```
tarek@dabox:~$ which python
/usr/bin/python
```

```
tarek@dabox:~$ python
python python2.3 python2.5
python2.4
```

有可能系统无法识别你执行的 Python 命令。这可能是因为你在编写命令时出现了拼写错误，或者系统无法访问 Python 安装。在这种情况下，大多数时候可以通过简单地通过你正在运行的 Linux 发行版的包管理器进行全新安装来解决问题。

在 Linux 系统上安装 Python 的首选方法是使用包安装方法，但包管理工具可能并不总是拥有最新的 Python 版本。

#### 将 Python 作为软件包安装

在 Linux 上，Python 通常使用 Linux 系统提供的包管理工具进行安装。这样，也便于保持其更新。根据所使用的 Linux 系统类型，有几种方法可以通过以下命令行安装 Python：

- “apt-get install python”是用于基于 Debian 的系统（如 Ubuntu）的指令。
- “urpmi python”用于基于 rpm 的系统，即 Fedora 或 Red Hat 系列。
- “emerge python”是为 Gentoo 系统指定的。

如果未指定 Python 安装版本，你需要自己在系统上安装最新版本。

对于 Python 的完整安装，需要安装额外的软件包，但它们是可选的。用户可以在没有这些软件包的情况下完美工作，但如果需要编写 C 扩展或对程序进行性能分析，那么它们将非常有用。完整安装所需的额外软件包包括：

- Python-dev：如果要在 Python 中编译 C 模块，那么 python-dev 软件包包含此任务所需的 Python 头文件。
- python-profiler：此软件包提供 GPL 完整发行版（Debian 或 Ubuntu）所需的非 GPL 模块。
- gcc：C 代码扩展是借助 gcc 编译的。

#### 源代码编译

手动安装通过‘CMMI’过程进行，该过程代表“configure, make, make install”的顺序。它用于在系统上编译和部署 Python。Python 归档的最新版本可以在网站“http://python.org/download”上访问。

程序“make”和“gcc”用于构建 Python，因此用户应该在系统上安装它们。

- ‘make’序列主要负责仔细检查一切是否符合程序编译的要求。该序列核对和仔细检查的‘列表’位于名为‘makefile’的配置文件中。在该序列给出绿灯之前，编译过程不会进行。
- ‘gcc’只是负责从代码创建应用程序的编译器。

软件包“build-essentials”在多个版本的 Linux（如 Ubuntu）中可用，并提供所需的构建工具，可以轻松安装。

以下序列可用于在系统上构建和安装 Python。

```
cd /tmp
wget http://python.org/ftp/python/2.5.1/Python-2.5.1.tgz
tar -xzvf Python-2.5.1.tgz
cd Python-2.5.1
./configure
make
sudo make install
```

在此过程中安装的项目中，还包括二进制安装的头文件，这些头文件通常位于软件包“python-dev”中。安装的源代码版本还包括 Hotshot 性能分析器。安装过程完成后，Python 被启用，因此可以从 shell 访问它。

### Windows 安装

在 Windows 操作系统上安装和编译 Python 的基本过程与 Linux 类似，但 Windows 还需要设置一个复杂的编译环境，这可能相当麻烦。网站“python.org”上的下载部分有适用于 Windows 的标准 Python 安装程序，安装向导很容易操作。

#### 在系统上安装 Python

Python 安装向导中的默认选项将文件路径选择为“c:\Python25”，而不是标准的“C:\Program Files\Python25”。这样做是为了防止路径中出现任何空格并缩短路径。

最后，要从 DOS shell 调用 Python，用户必须更改“path”环境变量。要在 Windows 上安装 Python 时执行此操作，请按照以下步骤操作：

- 从“开始”菜单或桌面找到“我的电脑”图标，右键单击它以激活“系统属性”面板。
- 单击面板中的“高级”选项卡。
- 找到名为“环境变量”的按钮并单击它。
- 通过添加两个由分号“;”分隔的新路径来更改“path”系统变量。

编辑到系统变量中的路径是：

- 文件路径“c:\Python25”，用于调用“Python.exe”。
- 文件路径“c:\Python25\Scripts”。这通过扩展调用安装在 Python 中的第三方脚本。

可以在提示符下调用和运行 Python。要执行此操作，只需从“开始”菜单打开“运行”并输入 cmd。按回车键，将显示提示符，从中可以调用 Python。

```
C:\> python
Python 2.7.2 on win32
>>>
```

虽然 Python 理论上可以在这种特定环境中运行，但与 Linux 相比，它看起来就像一条腿断了一样在运行。建议始终确保在系统上安装了与设置环境兼容的 MinGW 编译器版本。

#### 安装 MinGW

正如我们在上一节中已经提到的，‘MinGW’是一个编译器，当我们使用基于 Windows 的系统上的 Python 时推荐使用它。该编译器具有所有这些功能和特性，在兼容性方面与 Windows 上的标准 Visual Studio 编译器相当甚至更好。要在基于 Windows 的系统上获得最佳结果，应该只使用这两个编译器，并根据任务在它们之间切换。

当我们在系统上安装完 MinGW 编译器后，我们不能使用开箱即用的命令。由于默认编译器仍然是 Visual Studio，我们需要将我们使用的环境链接到这个编译器。为此，我们需要更改环境的路径变量，使其指定到 MinGW 编译器的系统路径。在这种情况下，我们可能需要指定以下路径：

c:\MinGW\bin

以下代码块展示了在 shell 中执行 MinGW 命令的演示。

```
C:\>gcc -v
Reading specs from c:/MinGW/bin/../lib/gcc-lib/mingw32/3.2.3/specs
Configured with: ../gcc/configure --with-gcc --with-gnu-ld --with-gnu-as
--host=mingw32 --target=mingw32 --prefix=/mingw --enable-threads --disable-nls
--enable-languages=c++,f77,objc --disable-win32-registry --disable-shared
--enable-sjljexceptions
```

#### 安装 MSYS

我们现在将讨论另一个能极大提升 Windows 上 Python 编码工作效率的工具。这通常就是 'MSYS' 工具。该工具为用户提供了在 Linux 和 macOS 操作系统中可用的所有命令。建议在 Windows 系统上使用此工具，因为它无法访问 'Bourne Shell'（仅限于 Linux 和 macOS）中的命令。

可以通过以下下载链接在 Windows 上安装 MSYS：

http://sourceforge.net/project/showfiles.php?group_id=2435&package_id=240780

默认情况下，MSYS 将安装在文件路径 “c:\msys” 中，因此用户必须像使用 MinGW 一样，将 “path” 变量中的路径编辑为 “c:\msys\1.0\bin”。

### Mac OS X 安装

Apple 的 Mac OS 与 Linux 有些相似，但我们不会深入探讨这个细节。由于这两个操作系统之间存在相当程度的兼容性，用户可以在任一操作系统上使用相同的过程、安装方法、编译器和技术。但并非所有东西都相同。例如，Linux 和 Mac 操作系统的系统树组织结构就不同。

遵循 Linux 和 Windows 的趋势，Python 在 Mac OS X 中也有两种安装方式：

- 一种是通过包安装，这是一种简单直接的安装 Python 的方法。
- 另一种方式是从源代码编译，如果用户想自己构建它。

#### 将 Python 作为包安装

如果用户拥有最新版本的 Mac OS X，那么 Python 可能已经预装在上面了。仍然可以通过从以下地址获取 Python 2.5.x 的通用二进制文件来安装额外的 Python：
http://www.pythonmac.org/packages
获取的 “.dmg” 文件被挂载后，可以启动其中的 “.pkg” 文件。
Python 安装在 “/Library” 文件夹中，该文件夹在 Mac OS X 系统中创建链接。然后可以从 shell 中调用 Python 并运行它。

#### 源代码编译

如果你选择自己构建 Python，那么你需要以下工具从源代码编译 Python：

- 一个 “gcc” 编译器，可以从 Xcode Tools、安装光盘或在线网站获取：“http://developer.apple.com/tools/xcode”。
- MacPorts 是一个类似于 Debian 的名为 apt 的包管理系统。就像 apt 在 Linux 系统中安装依赖项一样，MacPorts 也会为 Mac OS X 执行相同的功能。

编译 Python 的其余方法与 Linux 相同。

### Python 提示符

Python 提示符通常用作一个小型计算器，并允许用户与解释器交互。当调用 “python” 命令时，它就会出现。

```
macziade:/home/tziade tziade$ python
Python 2.5 (r25:51918, Sep 19 2006, 08:49:13)
[GCC 4.0.1 (Apple Computer, Inc. build 5341)] on darwin
>>>1 + 3
4
>>>5 * 8
40
```

一旦我们启动 Python shell 并指示 Python 执行许多算术计算，解释器就会翻译这些行并返回相应的响应。

#### 配置交互式提示符

程序员通常会自定义他们在编码会话中可能与之交互的提示符或 shell。最简单的方法之一是使用一个 'startup' 脚本来执行用户自己配置的一系列操作。例如，如果我们创建一个自定义的启动文件，那么一旦提示符启动，它就会搜索变量 'PYTHONSTARTUP' 并执行位于 `os.path.join(os.environ['HOME'], '.pythonhistory')` 路径下的脚本。

```python
try:
    readline.read_history_file(histfile)
except IOError:
    pass
atexit.register(readline.write_history_file, histfile)
del os, histfile, readline, rlcompleter
```

在主目录中创建一个名为 “.pythonstartup” 的文件，并使用给定的文件路径添加环境变量 “PYTHONSTARTUP”。

现在调用并运行交互式提示符。然后 “.pythonstartup” 脚本执行，并为 Python 添加新功能。其中一个功能是 “Tab 补全”，它可以回忆模块的内容，使编码过程更容易：

```
>>> import md5
>>> md5.
md5.__class__ md5.__file__ md5.__name__
md5.__repr__ md5.digest_size
md5.__delattr__ md5.__getattribute__ md5.__new__
md5.__setattr__ md5.md5
md5.__dict__ md5.__hash__ md5.__reduce__
md5.__str__ md5.new
md5.__doc__ md5.__init__ md5.__reduce_ex__ md5.blocksize
```

通过利用 Python 为其每个模块提供的入口点，可以进一步改进上述启动脚本处理的任务自动执行。

#### 高级 Python 提示符 ‘iPython’

iPython 是一个旨在提供更高级和扩展提示符的工具，具有以下有趣功能：

- 动态对象自省
- 从提示符访问系统 shell
- 性能分析任务
- 调试工具

完整功能列表可在网站 “http://ipython.scipy.org/doc/manual/index.html” 上找到。

要安装此工具，请访问网站链接 “http://ipython.scipy.org/moin/Download”，然后根据每个平台给出的说明下载并安装它。

下面是一个 iPython 工作 shell 的示例：

```
tarek@luvdit:~$ ipython
Python 2.4.4
Type "copyright", "credits" or "license" for more information.
IPython 0.7.2 -- An enhanced Interactive Python.
? -> Introduction to IPython's features.
%magic -> Information about IPython's 'magic' % functions.
help -> Python's own help system.
object? -> Details about 'object'. ?object also works, ?? prints more.
In [1]:
```

### 安装 Setuptools

在 Python 中，一些工具和模块允许程序员创建自己的库。这在涉及复杂编程项目时非常有用。可以使用 'Perl CPAN' 中的命令来构建自定义库，但使用此类工具和方法也有一些需要解决的依赖关系：

- 在 Python 的官方网站上，有一个集中的存储库，称为 Python 包索引 (PyPI)，以前称为 Cheeseshop。
- 要将代码打包成归档文件并与 PyPI 协作，需要一个称为 setuptools 的打包系统，它依赖于 distutils。

在介绍这些扩展之前，需要进行一些澄清以获得完整的图景。

#### 理解工具的工作原理

在 Python 中，distutils 是一个常用于将应用程序拆分为不同包的工具。我们将在接下来的章节中更详细地讨论这一点，届时我们将学习如何创建包和应用程序。使用 distutils 模块本质上为编程项目提供了以下功能：

- 对一个案例或应用程序元数据的基本描述。该工具还允许程序员为包定义自定义依赖项。
- 当一个包或应用程序构建完成后，该工具提供必要的命令和函数来帮助程序员将其产品分发给社区。

需要考虑的一件重要事情是，我们不能完全依赖 distutils 模块本身。例如，如果我们只处理一个案例，那么我们可以单独使用 distutils 来管理案例的创建和分发。但如果我们处理的是一个应用程序，那么你可能会将应用程序拆分成多个不同的案例，而这些案例肯定会有扩展的依赖项。如果我们单独使用 distutils 工具，我们就无法处理由于依赖关系而导致的包拆分问题。这就是为什么我们在 Python 中与 distutils 一起使用另一个重要模块，这个模块就是 'setuptools'。

安装 'setuptools' 模块：

在我们可以在该方法上安装 'setuptools' 模块之前，我们首先需要一个对此过程很重要的包。这个包是 'EasyInstall'。

### 工作环境

任何编程项目都需要一个合适的环境，无论是创建一个简单的包，还是构建一个完整的应用程序。环境本质上是程序员进行大部分编码和测试，并准备有序执行代码的包的区域。环境可以被视为一种“工作台”。在你的工作台上，你需要确保拥有项目所需的所有工具、所有组件以及必要的材料。如果你的工作台缺少某些东西，那么你可能需要暂停整个项目，去获取缺失的东西，有时你甚至没有资源准备任何替代品。编程中的工作环境也是如此，但在这里，如果环境没有正确设置，那么你将需要放弃大部分已完成的进度，并且在准备好另一个合适的环境之前无法继续。

需要设置的环境可以通过以下两种方式完成：

-   可以通过组合许多小工具来构建。这种设置环境的方式现在已经过时了。
-   一种新方法是使用一体化工具。

这些是构建环境的主要方法，中间还有许多其他方式。应该给予开发者选择任何他们希望设置环境的方式的权利。

#### 编辑器工具和其他组件

许多程序员更喜欢使用合适的“编辑器工具”以及其他组件来为他们的项目创建环境。从头开始使用编辑器创建环境非常耗费时间资源，这意味着你需要花费大量时间。话虽如此，这种方法创建环境所投入的时间是值得的，因为最终的产品是值得的。以这种方式构建的环境在创建后很容易定制，这对项目结果也有巨大影响（你也可以在开发阶段稍后调整环境以匹配项目的要求）。如果你知道你构建的环境将来会派上用场，那么你可以简单地制作一个便携版本，并将其存储在可移动驱动器中，最好是USB。这样，你可以将其插入任何你想要的系统，并立即为项目做好准备。在接下来的章节中，我们还将了解如何创建一个用于创建环境的模板。这大大减少了为全新项目从头开始创建环境所需的时间。

因此，工作环境应具备以下条件：

-   一个开源且免费的代码编辑器，可在任何操作系统上使用
-   用于某些功能的额外二进制文件，可以帮助避免在Python中重写它们

许多程序员使用的流行代码编辑器是“Vim”，但少数人更喜欢“Emacs”代码编辑器；本章我们将只关注Vim编辑器。

#### 安装和配置Vim代码编辑器

Vim代码编辑器工具可以从其官方网站下载。

http://www.vim.org/download.php

运行Linux操作系统的系统通常默认安装了Vim版本。但有可能该版本是旧版本，因此你应该通过在终端中执行以下命令来检查Vim的安装版本。

```
vim --version
```

如果版本不是最新版本，建议你更新它。

在其他系统如Windows和macOS上，Vim默认不包含，需要手动下载和安装。在Windows上，用户可以选择下载具有图形界面的Vim版本以及标准控制台版本。这个版本的Vim通常被称为“gvim”。Vim工具的官方下载链接已在开头提供。

如果你使用的是Linux或macOS系统，你需要将一个‘.vimrc’文件放在其主目录中。如果你的系统运行的是Windows，那么你需要将一个‘_vimrc’文件放在工具的安装文件夹中，并简单地将一个环境变量链接到这个文件，这样Vim工具就能轻松找到这个文件。

以下是vimrc文件应包含的内容：

```
set encoding=utf8
set paste
set expandtab
set textwidth=0
set tabstop=4
set softtabstop=4
set shiftwidth=4
set autoindent
set backspace=indent,eol,start
set incsearch
set ignorecase
set ruler
set wildmenu
set commentstring=\ #\ %s
set foldlevel=0
set clipboard+=unnamed
syntax on
```

正如其名所示，这个包也处理包的安装作为模块。如果我们从仓库或服务器安装包，EasyInstall将处理下载过程。它有点像一个第三方下载和安装管理器，你只需在系统上使用它，而不是默认的管理器。一旦我们安装了EasyInstall，就可以通过在系统终端shell上执行一个简单的命令轻松安装setuptools模块。以下是该包的官方网站。

http://peak.telecommunity.com/DevCenter/EasyInstall

该工具包的目标通常位于

http://peak.telecommunity.com/dist/ez_setup.py

以下是安装setuptools模块的指令执行：

```
macziade:~ tziade$ wget http://peak.telecommunity.com/dist/ez_setup.py
macziade:~ tziade$ python ez_setup.py setuptools
Searching for setuptools
Reading http://pypi.python.org/simple/setuptools/
Best match: setuptools 0.6c7
Processing dependencies for setuptools
Finished processing dependencies for setuptools
```

如果你已经安装了旧版本，会出现错误，然后你需要升级当前版本。

```
macziade:~ tziade$ python ez_setup.py
Setuptools version 0.6c7 or greater has been installed.
(Run "ez_setup.py -U setuptools" to reinstall or upgrade.)
macziade:~ tziade$ python ez_setup.py -U setuptools
Searching for setuptools
Reading http://pypi.python.org/simple/setuptools/
Best match: setuptools 0.6c7
Processing dependencies for setuptools
Finished processing dependencies for setuptools
```

既然我们现在系统上有了EasyInstall，不使用它来安装其他包和扩展可能会是一种浪费。要指示这个软件工具启动包的安装过程，我们只需在系统shell中执行‘easy_install’命令。等效的命令也可以用于更新已安装的扩展。以下是一个示例，展示我们如何使用easy_install命令安装一个扩展。

```
tarek@luvdit:/tmp$ sudo easy_install py
```

Searching for py

Reading http://cheeseshop.python.org/pypi/py/

Reading http://codespeak.net/py

Reading http://cheeseshop.python.org/pypi/py/0.9.0

Best match: py 0.9.0

Downloading http://codespeak.net/download/py/py-0.9.0.tar.gz

Installing pytest.cmd script to /usr/local/bin

Installed /usr/local/lib/python2.3/site-packages/py-0.9.0-py2.3.egg

Processing dependencies for py

Finished processing dependencies for py

#### 将MinGW附加到‘distutils’工具

当编译器被触发创建用C编程语言编写的应用程序时，编译器MinGW和distutils工具之间通过一个配置文件建立通信路径。此文件必须存在，否则编译过程将失败并返回错误。简单来说，将下面代码块中显示的行复制并粘贴到名为‘distutils.cfg’的cfg文件中，并将其放在以下目录

‘python-installation-path\lib\distutils’

```
[build]
compiler = mingw32
```

通过这样做，Python将与MinGW链接，以便在必须构建包含C代码的包时，Python将使用MinGW。

# 第二章

## 类级别以下的语法

编写一段诚实、高效的代码堪称一门艺术，而艺术通常是可以通过培养和学习来掌握的技能。语法不仅关乎视觉上的愉悦，更在于它允许调整并减少混淆。许多项目和原型之所以失败，正是因为其语法过于晦涩，或一眼难以理解，需要指导才能查看代码并与之交互。

Python 的后端已经做了大量工作，以实现更简洁、更清晰的语法。这是一门古老的语言，始于 1991 年，并一直保持更新。其基础保持不变，但我们使用的工具已经过打磨和升级。

我们将探讨编写良好 Python 语法的五个更核心的主题，并就如何处理它们提供建议。按我们的顺序，它们是：

- 列表推导式：
- 迭代器和生成器：
- 描述符及其属性：
- 装饰器：
- with 和 contextlib

在您学习的过程中，能够访问各种提示和文档，了解如何完成各种操作的示例，这将对您有所帮助。这些可以在您编译器控制台自身的帮助功能以及 Python 官方教程和指南中找到。

### 列表推导式

Python 并不完全等同于 C。设计和解释器上的固有差异使得 C 能够拥有比 Python 更广泛但可能更耗费资源的代码。例如：

```
>>> numbers = range(10)
>>> size = len(numbers)
>>> evens = []
>>> i = 0
>>> while i < size:
...     if i % 2 == 0:
...         evens.append(i)
...     i += 1
...
>>> evens
[0, 2, 4, 6, 8]
```

这种形式给 Python 带来了问题，即：第一，编译器解释器必须在每次循环迭代中重新计算序列中需要更改的内容；第二，必须有一个计数器来跟踪需要交互的元素。

我们可以使用 Python 的列表推导式功能，它利用内置特性自动化了上述代码的过程，最终将其整理并减少为一行代码。此外，其便利性和易用性使其更易于理解和调试，从而降低出错的可能性，并使后续的错误修复更容易。

```
>>> [i for i in range(10) if i % 2 == 0]
[0, 2, 4, 6, 8]
```

`enumerate` 是 Python 优雅简洁的另一个例子。当序列在循环中使用时，此函数可以方便地为序列建立索引。

例如，这段代码：

```
>>> i = 0
>>> seq = ["one", "two", "three"]
>>> for element in seq:
...     seq[i] = '%d: %s' % (i, seq[i])
...     i += 1
...
>>> seq
['0: one', '1: two', '2: three']
```

可以简化为：

```
>>> seq = ["one", "two", "three"]
>>> for i, element in enumerate(seq):
...     seq[i] = '%d: %s' % (i, seq[i])
...
>>> seq
['0: one', '1: two', '2: three']
```

并且使用列表推导式，可以像下面这样优雅地完成。顺便说一句，将循环转换为一个小函数可以为代码进行向量化，使得后续理解和使用它变得更加成功：

```
>>> def _treatment(pos, element):
...     return '%d: %s' % (pos, element)
...
>>> seq = ["one", "two", "three"]
>>> [_treatment(i, el) for i, el in enumerate(seq)]
['0: one', '1: two', '2: three']
```

### 迭代器和生成器

迭代器是一个容器对象，它简单地在其内部引发迭代。它由两部分组成：`next` 用于产生容器中的下一个项目，以及 `__iter__`，它返回迭代器对象。让我们用一个序列来帮助理解迭代器。

```
>>> i = iter('abc')
>>> i.next()
'a'
>>> i.next()
'b'
>>> i.next()
'c'
>>> i.next()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

在序列的末尾，我们使用 `StopIteration`，这很方便地允许在循环中使用迭代器，因为该条件允许它中断并停止。

我们也可以在类中创建一个迭代器，如下所示，它使用 `next` 方法并使用 `__iter__` 返回一个迭代器实例：

```
>>> class MyIterator(object):
...     def __init__(self, step):
...         self.step = step
...     def next(self):
...         """返回下一个元素。"""
...         if self.step == 0:
...             raise StopIteration
...         self.step -= 1
...         return self.step
...     def __iter__(self):
...         """返回迭代器本身。"""
...         return self
...
>>> for el in MyIterator(4):
...     print el
...
3
2
1
0
```

迭代器是一种基本且底层的构造，其使用并非绝对必要，但它们确实构成了生成器概念的基础。生成器在 Python 2.2 版本中引入，用于创建返回元素列表的函数。我们可以使用 `yield` 来暂停该函数并获取一个中间值。下面的例子描述了我们如何创建一个生成器来提供斐波那契数列。

```
>>> def fibonacci():
...     a, b = 0, 1
...     while True:
...         yield b
...         a, b = b, a + b
...
>>> fib = fibonacci()
>>> fib.next()
1
>>> fib.next()
1
>>> fib.next()
2
>>> [fib.next() for i in range(10)]
```

定义的函数返回一个生成器对象，这是一种特殊的迭代器，它可以保存执行上下文。我们可以随心所欲地多次调用它，它会生成我们上次离开位置之后的下一个元素。考虑一下它的语法：它简洁、简单、明了，即使序列是无限的，我们也不必再担心它。

大多数 Python 程序员不使用生成器，因为他们更习惯使用更简单的函数概念。理想情况下，每当有一个函数在循环中使用或独特地生成一系列事物时，就应该考虑使用生成器，这可以提高代码的整体性能。

为了进一步说明，让我们考虑一个返回顺序值的生成器。该序列的值不需要立即加载，因为生成器会依次生成每个值，并且由于每个生成器循环所花费的时间应该比系统预先加载可能无限的序列（如斐波那契数列）所需的时间更少，因此它可能是一个快得多的解决方案。一个实际的例子是在流缓冲区中的使用，我们可以完全控制想要流式传输多少数据，以及是否需要暂停或完全临时关闭它。

现在让我们看看 `tokenize` 模块。它是 Python 标准库的一个预定义成员，是一个生成器，它读取文本流，为其生成令牌（即给定文本主体的细分），并返回一个迭代器，该迭代器可以通过将其应用于外部函数来进一步处理数据。我们使用 `open` 来加载和读取文本流，以便将其传递给 `tokenize`，并使用 `generate_tokens` 来处理此数据流。

生成器还有助于简化代码；例如，我们可以将多个模块（比如数据转换算法）组合成一个更高级的函数，方法是将每个模块视为一个迭代器对象。这还有一个优点，就是可以对过程产生实时反馈。下面给出一个例子，其中每个函数转换一个序列，并且它们相互连接；一个函数接一个函数。我们将通过提及生成器如何使用通过 `next` 调用的代码来结束关于迭代器和生成器的部分，其中 `yield` 变成一个表达式，我们可以使用一个新工具 `send` 将结果输出转移到其他地方。

#### Send 工具

`Send` 类似于 `next`，但改变了 `yield` 的功能，使其返回生成器的输出。有两个函数来支持 `send`；`throw` 和 `close`。这些是生成器的错误标志，它们的工作方式如下：

- `throw` 是一个通用方法，允许客户端代码发送任何类型的异常标志。
- `close` 只允许引发特定的标志 `GeneratorExit`。这只能通过再次使用 `GeneratorExit` 或使用 `StopIteration` 来清除。

最后，一个名为 `finally` 的方法会清除任何未被捕获的 `close` 或 `throw`，是处理遗留问题的简单方法。必须注意将 `GeneratorExit` 与生成器分离，以确保在调用 `close` 时其能干净地退出，否则，解释器将涉及一条指令。

现在，我们将继续使用生成器来创建协程，这利用了我们刚刚讨论的最后三种方法。

#### 协程

协程是一种函数，其目的是允许多个进程协同工作，方法是在必要时允许它们暂停和恢复执行。这在各种情况下都很方便，例如事件循环和处理异常（如错误语句）。

协程有助于在代码主体中解决多任务处理问题，并清理处理流程。生成器类似于 Io 和 Lua 版本的协程，但又不完全相同；它需要 `send`、`throw` 和 `close`。另一种类似的技术是线程，它允许代码块进行交互，尽管它所需的资源、对其执行方式的预定义需求以及代码的复杂性使其对于较简单的项目来说比较麻烦。

Python 官方文档中给出了一个使用上述方法创建协程的例子，该特定方法被称为 Trampoline。可以通过以下网络链接访问：http://www.python.org/dev/peps/pep-0342 (PEP 342)。

协程的一个绝佳例子是服务器管理，其中多个处理线程并行发生。尝试访问和使用服务器的客户端发送一个请求，该请求由服务器通过两个

#### 生成器表达式

Python 的库中提供了一种编写序列生成器的简便快捷方式。其语法几乎与列表推导式相同，但我们将在 `yield` 的位置使用它。并且，我们使用方括号而非圆括号。

```
>>> iter = (x**2 for x in range(10) if x % 2 == 0)
>>> for el in iter:
...     print el
...
```

```
0
4
16
36
64
```

这些表达式被称为生成器表达式，或简称为 `genexp`。正如我们所提到的，它们可用于替代 `yield` 循环，或替代充当迭代器的列表推导式。与后者几乎一样，它们有助于精简代码块，使其看起来整洁美观。并且，与生成器本身一样，每次只产生一个元素作为输出。

# 第 3 章

## 面向对象编程

我们现在将探讨面向对象编程的四个概念及其在 Python 中的应用。

### 继承

第一个主要概念是“继承”。这指的是事物具有从另一事物派生的能力。让我们以跑车为例。所有跑车都是车辆，但并非所有车辆都是跑车。此外，所有轿车都是车辆，但所有车辆都不是轿车，并且轿车从来不是跑车，尽管它们都是车辆。

所以基本上，面向对象编程的这个理念是说，事物可以并且将会被分解成尽可能小而精确的概念。

在 Python 中，这可以通过派生类来完成。

假设我们有另一个名为 `SportsCar` 的类。

```
class Vehicle(object):
    def __init__(self, makeAndModel, prodYear, airConditioning):
        self.makeAndModel = makeAndModel
        self.prodYear = prodYear
        self.airConditioning = airConditioning
        self.doors = 4
    def honk(self):
        print "%s says: Honk! Honk!" % self.makeAndModel
```

现在，在其下方，创建一个名为 `SportsCar` 的新类，但不是从 `object` 派生，而是从 `Vehicle` 派生。

```
class SportsCar(Vehicle):
    def __init__(self, makeAndModel, prodYear, airConditioning):
        self.makeAndModel = makeAndModel
        self.prodYear = prodYear
        self.airConditioning = airConditioning
        self.doors = 4
```

省略 `honk` 函数，我们这里只需要构造函数。现在声明一辆跑车。我打算用法拉利。

```
ferrari = SportsCar("Ferrari LaFerrari", 2016, True)
```

现在通过调用来测试这个

```
ferrari.honk()
```

然后保存并运行。它应该会毫无问题地运行。为什么会这样？这是因为继承的概念指出，子类从父类派生函数和实例变量。这是一个很容易理解的概念。下一个可能会有点难。

### 多态

多态的理念是，根据事物的需求，相同的过程可以以多种方式执行。在 Python 中，这可以通过两种方式实现：方法重载和方法重写。

方法重载是用不同的参数定义两次相同的函数。例如，我们可以给我们的 `Vehicle` 类提供两个不同的初始化函数。现在，它仍然只考虑车辆有 4 扇门。如果我们想具体说明一辆车有多少扇门，我们可以在当前初始化函数下方创建一个新的初始化函数，带有另一个 `doors` 参数，像这样（较新的在下方）：

```
def __init__(self, makeAndModel, prodYear, airConditioning):
    self.makeAndModel = makeAndModel
    self.prodYear = prodYear
    self.airConditioning = airConditioning
    self.doors = 4

def __init__(self, makeAndModel, prodYear, airConditioning, doors):
    self.makeAndModel = makeAndModel
    self.prodYear = prodYear
    self.airConditioning = airConditioning
    self.doors = doors
```

现在，当有人创建 `Vehicle` 类的实例时，可以选择是否定义门的数量。如果不定义，则假定门的数量为 4。

方法重写是当子类用自己的代码覆盖父类的函数时。

为了说明，创建另一个扩展 `Vehicle` 的类，名为 `Moped`。将门设置为 0，因为这很荒谬，并将空调设置为 `False`。唯一相关的参数是品牌/型号和生产年份。它应该看起来像这样：

```
class Moped(Vehicle):
    def __init__(self, makeAndModel, prodYear):
        self.makeAndModel = makeAndModel
        self.prodYear = prodYear
        self.airConditioning = False
        self.doors = 0
```

现在，如果我们创建一个 `Moped` 类的实例并调用 `honk()` 方法，它可能会鸣笛。但众所周知，轻便摩托车不鸣笛，它们发出的是“哔哔”声。所以让我们用我们自己的方法覆盖父类的 `honk` 方法。这非常简单。我们只需在子类中重新定义该函数：

```
def honk(self):
    print "%s says: Beep! Beep!" % self.makeAndModel
```

我是那 2.99 亿美国人中的一员，如果他们的生命取决于此，他们也说不出轻便摩托车的品牌和型号。

### 抽象

面向对象编程中的下一个重要理论是抽象。这是指程序员和用户应该远离计算机的内部工作原理。这有两个好处。

第一个是它通过将程序员与计算机的内部工作（如内存、CPU，有时甚至操作系统）抽象开来，降低了固有的安全风险和灾难性系统错误的可能性，无论是人为的还是其他原因。这样，任何类型的事故造成不可逆转损害的可能性都很低。

第二个是抽象本质上使语言更容易理解、阅读和学习。尽管它通过移除用户对整个计算机架构的一些控制能力，使语言稍微不那么强大，但这换来了在语言中快速高效编程的能力，而无需浪费时间处理诸如内存地址或类似事物之类的琐事。

这些在 Python 中实践起来非常容易。你无法深入到计算机的细枝末节，或者轻松地进行太多内存分配，甚至不能轻易地专门分配一个数组大小，但这是为了惊人的可读性、在高度安全环境中的高度安全的语言，以及简单易用的编程而做出的权衡。比较以下来自 C 的代码片段：

```
#include <stdio.h>

int main(void) {
    printf("hello world");
```

### 封装

面向对象编程中最后一个重要的概念是封装。这是最容易解释的一个概念。它通常指的是将公共数据放在一起，代码应当模块化。我不会花太多时间解释这一点，因为它是一个非常简单的概念。类的概念本身就是封装最简洁的例子：公共的属性和方法被绑定在一个统一的结构下，使得创建该类型的对象变得非常容易，无需为每个实例创建大量特定的变量。

好了，我们终于完成了这次Python小冒险的旅程。首先，我想对坚持学完《Python初学者指南：Python编程终极指南》的你表示衷心的感谢。希望这本书对你有所启发，并能为你提供实现目标所需的所有工具，无论你的目标是什么。

下一步是实践这些知识。无论是出于兴趣还是职业发展，通过学习Python的基础知识，你已经做出了人生中最明智的决定之一。你现在的目标应该是想办法在日常生活中运用它，让生活更轻松，或者完成你长久以来想做的事情。

# 第4章

## 基本编程工具

### Bash脚本

Bash脚本是一个包含一系列命令的文件，你通常可以手动输入这些命令，但使用脚本可以节省时间。请注意，在编程中，任何你通常在命令行中运行的代码都可以放入脚本中，并且它会按照原样精确执行。同样，任何放入脚本的代码通常也可以直接执行。

内存中可以同时运行多个进程来执行一个程序。例如，你可以同时使用两个终端并运行提示符。在这种情况下，系统中会同时存在两个提示符进程。当它们完成执行后，系统会终止它们，因此将不再有代表提示符的进程。

当你使用终端时，你可以运行Bash脚本来提供一个shell。当你启动一个脚本时，它不会在当前进程中执行，而是会启动一个新进程在其中执行。但作为编程初学者，你不必过多担心这个脚本的机制，因为运行Bash通常非常简单。

你可能还会遇到一些关于脚本执行的教程，其原理基本相同。在执行脚本之前，它必须具有相应的权限，因为如果你未授予权限，程序将返回错误信息。

下面是一个Bash脚本示例：

```bash
#!/bin/bash

# declare STRING variable
STRING="Hello Python"

#print variable on a screen
echo $STRING
```

你可以使用755权限快捷方式，这样你就可以修改脚本并确保可以与他人共享来执行该脚本。

### Python正则表达式

正则表达式（RegEx）定义了文本字符串，允许你在管理、匹配和定位文本时获取模式。Python是使用正则表达式的编程语言的一个典范。正则表达式也可以在文本编辑器和命令行中使用，用于在文件中搜索文本。

当你第一次接触正则表达式时，你可能会认为它是一种特殊的编程语言。但掌握正则表达式可以节省大量时间，如果你处理文本数据，除非你需要解析海量数据。

`re`模块为Python中的正则表达式提供了完整支持。如果在使用或编译正则表达式时出现错误，它还会引发`re.error`异常。在使用Python正则表达式时，你需要了解两个基本函数。但在此之前，你应该明白，不同的字符在正则表达式中具有特殊含义。为了避免在使用正则表达式时感到困惑，当我们指原始字符串时，我们将使用`r'expression'`。

Python正则表达式中两个重要的函数是`search`和`match`函数。

`search`函数在字符串中查找正则表达式模式的第一个匹配项，可使用可选标志。`search`函数具有以下参数（以下是语法）：

-   String - 将被搜索以匹配字符串中的模式
-   Pattern - 要匹配的正则表达式
-   Flags - 可以使用按位指定的修饰符

`re.search`函数如果成功，将返回一个匹配对象；如果失败，则返回`None`对象。你应该使用匹配对象的`groups()`或`groups(num)`函数来查找匹配的表达式。

下面是使用`search`函数的代码示例：

```python
import re

#Check if the string starts with "The" and ends with "Spain":
txt = "The rain in Spain"
x = re.search("^The.*Spain$", txt)
if (x):
    print("YES! we've a match!")
else:
    print("No match")
```

输出将是：
YES! we've a match!

同时，`match`函数将尝试按照特定标志匹配字符串开头的正则表达式模式。以下是`match`函数的语法：
`match`函数具有以下参数：

-   String - 将被搜索以匹配字符串开头的模式
-   Pattern - 要匹配的正则表达式
-   Flags - 可以使用按位指定的修饰符

### Python包管理器

在编程中，包管理器指的是用于自动化为特定系统安装、配置、升级和卸载程序的工具，以有序的方式进行。

也称为包管理系统，它还处理数据文件的分发和归档，包括软件名称、版本号、用途以及语言正常运行所需的一系列依赖项。

当你使用包管理器时，元数据通常会被归档在本地数据库中，以避免代码不匹配和权限缺失。

在Python中，你可以使用一个工具来定位、安装、升级和删除Python包。它还可以识别系统上安装的包的最新版本，并自动从远程或本地服务器升级当前包。

Python包管理器不是免费的，你只能通过ActivePython使用它。它还利用仓库，仓库是一组预安装的包，包含不同类型的模块。

### 源代码控制

在编程中，源代码控制（也称为版本控制或修订控制）管理代码的更改，这些更改由字母或数字代码标识，称为修订号或简称修订。例如，一组初始代码被称为修订版1，那么第一次修改将是修订版2。每个修订版都将与时间戳以及进行更改的人相关联。修订版很重要，因此代码可以被恢复或比较。

如果你与团队合作，源代码控制至关重要。你可以通过不同的视图将你的代码更改与其他开发者的代码更改合并，这些视图可以显示详细的更改，然后将正确的代码合并到原始代码分支中。

无论你使用Python还是其他语言，源代码控制对于编码项目都至关重要。请注意，每个编码项目都应该从使用像Mercurial或Git这样的源代码控制系统开始。

自编程存在以来，各种源代码控制系统就已开发出来。以前，专有控制系统提供了为大型编码项目和特定项目工作流定制的功能。但如今，无论你是处理个人代码还是作为大型团队的一部分，都可以使用开源系统进行源代码控制。

在你早期的Python代码中使用开源版本控制系统是理想的选择。你可以使用Mercurial或Git，它们都是开源的，用于分发源代码控制。

Subversion也可用，它可以用于集中化系统以查看文件并最小化合并冲突。

### 总结

编程工具将使你的工作更轻松。本部分讨论的工具将为你节省大量时间，使协作更容易，并让你的代码无缝衔接。总之，我们学习了以下内容：

-   Bash 脚本可以节省大量编码时间，并使你的代码行更有序、更易读。
-   正则表达式（RegEx）可以帮助你在代码中查找、搜索和匹配文本字符串，这样你就无需逐行浏览或自行分析每段代码。
-   包管理器将自动化系统，以便轻松安装、升级、配置或移除能辅助你编码工作的特定程序。
-   源代码控制对于管理版本至关重要，无论你是独自工作还是与团队协作，这样你就能恢复更改或比较版本。

即使没有这些工具，你仍然可以编写代码，但如果你选择使用它们，你的工作将会变得更加轻松。

# 第 5 章

## 处理文件

在涉及使用 Python 时，我们接下来要专攻的是确保我们都知道如何处理和操作文件。你可能会遇到这样的情况：你正在处理一些数据，并希望存储它们，同时确保它们在你日后需要时能够方便地调用和使用。在保存数据的方式、后续如何找到它们以及它们在代码中如何反应方面，你确实有一些选择。

当你处理文件时，你会发现数据将被保存在磁盘上，否则你可以在代码中反复重用它，次数随你所愿。本章将帮助我们进一步了解如何处理一些我们需要做的工作，以确保文件按应有的方式运行，以及更多内容。

现在，我们将进入 Python 语言的文件模式，这使你能够在此过程中尝试几种不同的选项。一个很好的思考方式是，你可以把它想象成在 Word 中处理文档。在某个时候，你会尝试保存你正在处理的文档之一，以免丢失，这样你以后就能找到它们。Python 中的这类文件将是类似的。但你不会像在 Word 中那样保存页面，你将保存代码的部分内容。

你会发现，在处理文件时，你可以选择几种操作或方法。其中一些选项包括：

-   关闭你正在处理的文件。
-   创建一个新文件进行处理。
-   查找或将文件移动到新位置，以便更容易找到。
-   在之前创建的文件上写出新的代码部分。

### 创建新文件

我们在这里要查看的第一个任务是创建文件。如果我们没有一个文件来帮助我们，就很难完成其他许多任务。如果你想创建一个新文件并在其中添加一些代码，你首先需要确保文件在你的 IDLE 中打开。然后你可以选择在写出代码时想要使用的模式。

在 Python 中创建文件时，你会发现有三种模式可供使用。我们将在这里重点关注的三种主要模式包括追加（a）、模式（x）和写入（w）。

任何时候你想打开一个文件并在其中进行一些更改，那么你会想使用写入模式。这是三种模式中最容易使用的。写入方法将使你更容易设置代码的正确部分并使其最终为你工作。

写入函数将易于使用，并确保你可以对文件进行任何你想要的添加和更改。你可以向文件添加新信息、更改现有内容等等。如果你想看看使用写入方法可以对这部分代码做什么，那么你需要打开编译器并执行以下代码：

```python
#file handling operations
#writing to a replacement file hello.txt
f = open('hello.txt', 'w', encoding = 'utf-8' )
f.write("Hello Python Developers!")
f.write("Welcome to Python World")
f.flush()
f.close()
```

接下来，我们需要讨论你可以对正在使用的目录做什么。默认目录通常是当前目录。你可以浏览并更改存储代码信息的目录，但你必须在开始时花时间更改该信息，否则它最终不会出现在你想要的目录中。

无论你在处理代码时花了多少时间在哪个目录中，当你日后想要找到文件时，你都需要回到那个目录。如果你希望它出现在不同的目录中，请确保在保存文件和代码之前切换到该目录。使用我们上面编写的选项，当你进入当前目录（或你为此任务选择的目录）时，你将能够打开文件并看到你写在那里的消息。

对于这个例子，我们编写了一段简单的代码。当然，随着我们的进行，你将编写更复杂的代码。对于这些代码，有时你会想要编辑或覆盖文件中的一些内容。使用 Python 可以做到这一点，它只需要对你编写的语法进行一点更改。一个很好的例子包括：

```python
#file handling operations
#writing to a replacement file hello.txt
f = open('hello.txt', 'w', encoding = 'utf-8')
f.write("Hello Python Developers!")
f.write("Welcome to Python World")
mylist = ["Apple", "Orange", "Banana"]
#writelines() is employed to write down multiple lines into the file
f.write(mylist)
f.flush()
f.close()
```

上面的例子是一个很好的例子，当你想要对之前处理过的文件进行一些更改时使用，因为你只需要添加一个打印操作。这个例子不需要使用第三行，因为它只有一些简单的单词，但你可以向程序添加任何你想要的内容，只需使用上面的语法并根据你的需要进行修改。

### 什么是二进制文件？

在继续之前，我们还需要关注的一点是将文件和数据以二进制文件形式写入代码的概念。这听起来可能有点令人困惑，但这是 Python 允许你做的一件简单的事情。你需要做的就是将你拥有的数据更改为音频或图像文件，而不是将其作为文档。

使用 Python，你可以将任何你想要的代码更改为二进制文件。无论它过去是什么类型的文件都没关系。但你确实需要确保以正确的方式处理数据，以便以后更容易以你想要的方式显示。确保这能为你良好工作的语法如下：

```python
# write binary data to a file
# writing the file hello.dat write binary mode
f = open('hello.dat', 'wb')
# writing as byte strings
f.write("I am writing data in binary file!\n")
f.write("Let's write another list\n")
f.close()
```

如果你花时间在文件中使用此代码，它将帮助你创建你想要的二进制文件。一些程序员发现他们喜欢使用这种方法，因为它有助于他们整理事情，并在需要时更容易调用信息。

### 打开你的文件

到目前为止，我们已经处理了创建新文件并保存它，以及处理二进制文件。在这些例子中，我们掌握了一些处理文件的基础知识，这样你就可以让它们为你工作，并在你想要的任何时候调用它们。

现在这部分已经完成，是时候学习如何打开

# 第六章

## 异常处理

异常处理就是错误管理。它有三个目的。

1.  它允许你调试程序。
2.  它允许你的程序在遇到错误或异常时继续运行。
3.  它允许你创建自定义错误，这将帮助你调试、移除和控制许多 Python 的细微之处，并使你的程序按照你期望的方式运行。

### 处理零除错误异常

异常处理可以是简单或困难的任务，这取决于你希望程序如何运行以及你的创造力。你可能因为“创造力”这个词而挠头。编程不都是关于逻辑的吗？不是的。

编程的核心目的是解决问题。问题的解决方案不仅需要逻辑，还需要创造力。你听说过“跳出框框思考”这句话吗？

导致程序中断的异常可能很麻烦，它们通常被称为 bug。这类问题的解决方案通常难以捉摸。你需要找到一个变通方法，否则就有从头重写程序的风险。

例如，你有一个计算器程序，在进行除法时包含以下代码片段：

```python
>>> def div(dividend, divisor):
    print(dividend / divisor)
>>> div(5, 0)
Traceback (most recent call last):
  File "", line 1, in <module>
  File "", line 2, in div
ZeroDivisionError: division by zero
```

这是第二种解决方案的样子：

```python
>>> def div(dividend, divisor):
    if divisor != 0:
        print(dividend / divisor)
    else:
        print("Cannot Divide by Zero.")
>>> div(5, 0)
Cannot Divide by Zero.
```

这是第三种解决方案的样子：

```python
>>> def div(dividend, divisor):
    try:
        print(dividend / divisor)
    except:
        print("Cannot Divide by Zero.")
>>> div(5, 0)
Cannot Divide by Zero.
```

记住错误和异常的两个核心解决方案。第一，防止错误发生。第二，管理错误的后果。

### 使用 Try-Except 块

在前面的例子中，使用了 try-except 块来管理错误。然而，你或你的用户仍然可能做些事情来搞砸你的解决方案。例如：

```python
>>> def div(dividend, divisor):
    try:
        print(dividend / divisor)
    except:
        print("Cannot Divide by Zero.")
>>> div(5, "a")
Cannot Divide by Zero.
```

为“except”块准备的语句不足以解释由输入引起的错误。用一个数除以一个字符串并不应该得到“Cannot Divide by Zero.”的消息。

为了使其正常工作，你需要更深入地了解如何正确使用 except 块。首先，你可以通过指定精确的异常来指定它将捕获并响应的错误。例如：

```python
>>> def div(dividend, divisor):
    try:
        print(dividend / divisor)
    except ZeroDivisionError:
        print("An error has been detected.")
        print("division by zero")
        print("Continuing with the program.")
    except Exception as detail:
        print("An error has been detected.")
        print(detail)
        print("Continuing with the program.")
    print(quotient)

>>> div(4, 2)
4 divided by 2 is:
2.0

>>> div(5, 0)
An error has been detected.
division by zero
Continuing with the program.
5 divided by 0 is:
Traceback (most recent call last):
  File "", line 1, in <module>
  File "", line 8, in div
    print(quotient)
UnboundLocalError: local variable 'quotient' referenced before assignment
```

如你所见，初始故障之后的后续语句依赖于它；因此它们也受到影响。在这个例子中，变量 `quotient` 在 try-except 块之后使用时返回了一个错误，因为它的预期值没有被赋值，因为赋值给它的表达式无法求值。

在这种情况下，你会希望放弃依赖于 try 子句内容的剩余语句。为此，你需要使用 else 块。例如：

```python
>>> def div(dividend, divisor):
    try:
        quotient = dividend / divisor
    except Exception as detail:
        print("An error has been detected.")
        print(detail)
        print("Continuing with the program.")
    else:
        print(quotient)
>>> div(4, 2)
4 divided by 2 is:
2
>>> div(5, 0)
An error has been detected.
division by zero
Continuing with the program.
```

第一次使用正确参数调用函数的尝试很顺利。在第二次尝试中，程序没有执行 else 块下的最后两条语句，因为它返回了一个错误。

else 块总是跟在 except 块之后。else 块的功能是让 Python 在 try 块没有返回错误时执行其下的语句，并在发生异常时让 Python 忽略它们。

### 静默失败

静默失败或 failing silently 是一个在错误和异常处理中经常使用的编程术语。

从用户的角度来看，静默失败是一种程序在某个点失败但从不通知用户的状态。

从程序员的角度来看，静默失败是一种解析器、运行时开发环境或编译器未能产生错误或异常并继续执行程序的状态。这通常会导致非预期的结果。

程序员也可能通过忽略异常或绕过异常来引发静默失败。或者，他公然隐藏它们并创建变通方法，即使发生了错误，也使程序按预期运行。他这样做可能有多种原因，比如错误不会导致程序中断，或者用户不需要意识到错误。

### 处理文件未找到异常错误

你会遇到 FileNotFoundError 的时候。管理此类错误取决于你打开文件的计划或想法。以下是你遇到此错误的常见原因：

-   你没有将目录和文件名作为字符串传递。
-   你拼错了目录和文件名。
-   你没有指定目录。
-   你没有包含正确的文件扩展名。
-   文件不存在。

处理 FileNotFoundError 异常的第一种方法是确保所有常见原因都不会导致它。一旦你做到了，那么你需要选择处理错误的最佳方式，这完全取决于你最初打开文件的原因。

### 检查文件是否存在

同样，处理异常通常有两种方法：防御式和响应式。预防式方法是首先检查文件是否存在。

为此，你需要使用 Python 安装包自带的 `os`（`os.py`）模块。然后，使用其 `path` 模块的 `isfile()` 函数。`path` 模块的文件名取决于操作系统（UNIX 使用 `posixpath`，Windows 使用 `ntpath`，旧版 MacOS 使用 `macpath`）。例如：

```
>>> from os import path
>>> path.isfile("random.txt")
False
>>> path.isfile("sampleFile.txt")
True
```

### Try 和 Except

你也可以通过使用 `try`、`except` 和 `else` 代码块来用更直接的方式处理。

```
>>> def openFile(filename):
    try:
        x = open(filename, "r")
    except FileNotFoundError:
        except FileNotFound:
>>> openFile("random.txt")
The file 'random.txt' doesn't exist.
>>> openFile("sampleFile.txt")
The file 'sampleFile.txt' does exist.
```

### 创建新文件

如果文件不存在，并且你的目标是无论如何都要覆盖任何现有文件，那么最好使用 `"w"` 或 `"w+"` 访问模式。如果文件不存在，此访问模式会为你创建一个新文件。例如：

```
>>> x = open("new.txt", "w")
>>> x.tell()
0
```

如果你打算同时进行读写操作，请改用 `"w+"` 访问模式。

### 练习

尝试通过发现至少十种不同的异常来“破坏”你的 Python 程序。

然后，创建一个循环。

在循环中，创建十条语句，每条语句将在一个 `try` 代码块内触发你发现的十种不同异常中的一种。

每次循环迭代时，触发异常的语句之后的下一条语句应触发再下一条，依此类推。

为每种错误提供一个特定的 `except` 代码块。

# 第 7 章

## Python 库

我们已经讨论了数据分析，现在是时候将其中一些信息付诸实践了。你可能对深度学习感兴趣，甚至可能想构建一些卷积神经网络，但不确定从哪里开始。最好的第一步是选择你想要使用的库。然而，这带来了另一个挑战，因为市面上有非常多编码库可供选择，每个库背后都有一些强大的功能和特性。

首先，我们将介绍一些有助于深度学习的优秀 Python 库。其他语言也可以帮助处理机器学习和深度学习等任务。但对于你想要完成的大多数任务，特别是如果你是数据分析以及我们一直在讨论的所有流程的初学者，那么 Python 将是你的选择。即使在 Python 内部，也有多个库可供选择来完成你的深度学习工作。因此，考虑到这一点，让我们直接深入了解一些你可以用于数据分析的优秀 Python 深度学习库。

### Caffe

在通过 Python 探索深度学习库时，如果不花点时间讨论 Caffe 库，那是很难开始的。如果你对深度学习做过任何研究，那么你很可能听说过 Caffe 以及它能为你想要创建的一些项目和模型做什么。

虽然 Caffe 严格来说不是一个 Python 库，但它将为我们提供一些与 Python 语言的绑定。我们将在需要将网络部署到实际环境中时使用这些绑定，而不仅仅是在尝试训练模型时。我们将在本章中包含它的原因是，它几乎无处不在，并且用于你需要创建的深度学习模型的所有部分。

### Theano

我们将要使用的下一种库被称为 Theano。这个库帮助开发和使用了许多我们使用 Python 的其他深度学习库。就像程序员如果没有 NumPy 就无法拥有 `scikit-image`、`scikit-learn` 和 SciPy 等选项一样，当我们谈论 Theano 以及一些随深度学习而来的更高级抽象和库时，情况也是如此。

当我们深入其核心时，Theano 将是 Python 库之一，它不仅有助于深度学习，还可用于定义、优化和评估许多涉及多维数组的数学表达式。Theano 之所以能实现这一点，是因为它与 NumPy 库紧密集成，并且它对 GPU 的使用在整体上相当透明。

虽然你可以使用 Theano 库来帮助构建一些深度学习网络，但它通常被视为这些神经网络的构建块，就像 NumPy 库在我们进行科学计算时将作为构建块一样。我们将在后续内容中提到的大多数其他库都将围绕 Theano 库构建，这使得它比其他一些选项更易于访问和使用。

### TensorFlow

与我们在 Theano 库中看到的类似，TensorFlow 将是一个开源选项，可以在数据流图的帮助下进行数值计算。这个库最初是为 Google 机器智能组织内的 Google Brain 团队的研究而开发的。此外，从那时起，这个库已成为一个开源选项，以便公众可以将其用于他们的深度学习和数据科学需求。

与 Theano 相比，我们将看到 TensorFlow 库的一个主要优势是它能够处理分布式计算。当我们查看项目的多 GPU 配置时尤其如此，尽管 Theano 也在努力改进这一点。

### Keras

许多程序员发现他们喜欢使用 Keras 库来执行深度学习模型和其他任务。Keras 被视为一个模块化的神经网络库，比我们提到的其他一些库更简约。这个库可以使用 TensorFlow 或 Theano 作为后端，因此你可以选择最适合你任何需求的那个。这个库的首要目标是你应该能够快速地在你的模型上进行实验，并尽可能快地从想法过渡到结果。

许多程序员喜欢这个库，因为即使作为初学者，你构建的网络也会感觉非常自然和简单。它将包含一些最好的算法，用于优化器、归一化，甚至激活层，因此如果你的流程包含这些，这是一个很好的选择。

此外，如果你想花时间开发你的 CNN，那么 Keras 是一个很好的选择。Keras 旨在重点关注这些类型的神经网络，这在你从计算机视觉的角度进行工作时可能很有价值。Keras 还允许我们构建基于序列的网络（这意味着输入可以线性地流经该网络）和基于图的网络（在这种情况下，输入可以在需要时跳过一些层，稍后再进行拼接）。这将使我们更容易实现更复杂的网络架构。

关于这个 Python 库需要注意的一点是，如果你希望并行训练网络，它可能不支持多 GPU 环境。如果这是你想做的事情，那么你需要选择另一个你想要使用的库。然而，对于你想做的一些工作来说，这可能不是一个大问题。

如果你想尽快训练你的网络，使用像 MXNet 这样的库可能是更好的选择。但如果你想调整超参数，那么你可能希望利用 Keras 的潜力来设置四个独立的实验，然后评估这些实验结果之间的相似性或差异性。

### Sklearn-Theano

在使用深度学习的过程中，有时你会想要端到端地训练一个CNN。但有时这并非必要。当这种方法不适用时，你可以将CNN视为特征提取器。这在你遇到的某些情况下最为有用，即没有足够的数据从头训练CNN。因此，对于这种情况，只需将你的输入图像通过一个流行的预训练架构，例如VGGNet、AlexNet和OverFeat等选项。然后，你可以使用这些预训练模型，并从你想要的层（通常是全连接层）提取特征。

### Nolearn

一个适合你使用的优秀库是Nolearn库。这是一个很好的库，可以帮助进行一些初步的GPU实验，尤其是在MacBook Pro上。它也是一个出色的库，可以帮助在Amazon EC2 GPU实例上执行一些深度学习任务。

虽然Keras将TensorFlow和Theano封装成更用户友好的API，但你会发现Nolearn库也能做到同样的事情，不过它是通过Lasagna库来实现的。此外，我们使用Nolearn编写的所有代码都将与Scikit-Learn兼容，这对于你想要处理的许多项目来说是一个巨大的优势。

### Digits

关于这个库，首先需要注意的是，它并不被视为一个真正的深度学习库。尽管它是用Python编写的，并且代表深度学习GPU训练系统。原因在于，这个库更像是一个网络应用程序，可用于训练你借助Caffe创建的一些深度学习模型。你可以稍微修改源代码以使用Caffe以外的后端，但这会增加大量额外的工作。此外，由于Caffe库在其功能上表现良好，并且可以帮助完成许多你想要实现的深度学习任务，因此不值得你花费时间。

如果你过去曾花时间使用Caffe库，你可能已经体会到定义.prototxt文件、生成图像数据集、运行网络以及通过提供的终端监控网络训练过程是十分繁琐的。好消息是，DIGITS库旨在解决所有这些问题，允许你仅通过浏览器就能完成许多（如果不是全部）这些任务。因此，它本质上可能不是一个深度学习库，但当你在使用Caffe库遇到困难时，它确实很有用。

除了上述所有优点之外，用户交互界面也被认为是出色的。这是因为它可以为我们提供一些有价值的统计数据和图表，帮助你更有效地训练模型。你还可以轻松地可视化网络的一些激活层，以便根据需要处理各种输入。

最后，使用这个库的另一个好处是，如果你有一张特定的图像想要测试，你有几种选择来完成这个任务。首要选择是将图像上传到DIGITS服务器。或者，你可以输入图像的URL，然后你用Caffe创建的模型将自动能够对图像进行分类，并在浏览器中显示你想要的结果。

Python是可用于帮助完成深度学习、机器学习甚至人工智能（涵盖前两个概念）等任务的最简单的编程语言之一。其他语言也可以处理我们一直在讨论的深度学习，但没有一种语言能像Python那样高效、强大、拥有如此多的选项，或者为初学者设计得如此友好。

这就是为什么我们将注意力集中在Python语言以及我们可以选择的一些最好的库上，以帮助完成各种深度学习任务。这些库中的每一个都可以与你的项目一起使用，并提供一套独特的功能和技能来完成工作。浏览一下这些库，看看哪一个适合你的数据分析，并在完成深度学习时为你提供深刻的见解。

# 第8章

## PyTorch库

我们需要了解的下一个库是PyTorch。这是一个基于Python的科学计算包，它依赖于从图形处理单元获得的强大能力。这个库也将成为研究领域中最常见和首选的深度学习平台之一，为我们提供最大的灵活性和极高的处理速度。

这种类型的库有许多优点。它以提供所有其他深度学习库中两个最高级的功能而闻名。这些功能包括在强大GPU加速支持下的张量计算，以及能够在基于磁带的自动求导系统上构建深度神经网络。

Python中有许多不同的库可以帮助我们处理许多想要进行的AI和深度学习项目。此外，PyTorch库就是其中之一。这个库如此成功的一个关键原因是它完全符合Python风格，并且可以几乎毫不费力地构建你想要用神经网络创建的许多模型。这是一个更现代的深度学习库，但在这个领域也有着巨大的发展势头。

### PyTorch的起源

正如我们上面提到的，PyTorch是目前最新的库之一，它适用于Python并可以帮助进行深度学习。即使它是在2016年1月才发布的，它已经成为数据科学家们喜欢使用的首选库之一，主要是因为它可以轻松构建复杂的神经网络。这对于无数过去完全没有接触过这些神经网络的初学者来说是完美的。他们可以使用PyTorch，并在极短的时间内构建他们的网络，即使编码经验有限。

这个Python库的创建者设想，当他们想要尽快运行大量数值计算时，这个库将是必不可少的。这是最适合的方法之一，也与我们在Python中看到的编程风格完美契合。这个库以及Python库，使得神经网络调试器、机器学习开发者和深度学习科学家不仅能够运行，还能实时测试他们代码的某些部分。这是一个好消息，因为这意味着这些专业人士无需等待整个代码完成并执行，就能检查代码是否有效或是否需要修复某些部分。

除了PyTorch库附带的一些功能外，请记住，你可以通过添加其他Python包来扩展这个库的一些功能。像Cython、SciPy和NumPy这样的Python包都可以与PyTorch很好地协同工作。

即使有这些好处，我们可能仍然会有一些疑问，为什么PyTorch库如此特别，以及为什么在需要构建深度学习所需模型时我们可能想要使用它。答案很简单，主要是PyTorch将被视为一个动态库。这意味着该库是灵活的，你可以根据任何需求和更改来使用它。它非常擅长这项工作，以至于被AI开发者、许多行业的学生和研究人员所使用。事实上，在一次Kaggle竞赛中，进入前十名的大多数人都使用了这个库。

虽然PyTorch库有许多优点，但我们需要从一些亮点开始，了解为什么各种专业人士如此喜爱这个语言。其中一些包括：

-   接口易于使用。PyTorch接口将为我们提供一个易于使用的API。这意味着我们会发现它像使用Python一样简单易用。
-   它本质上是Python风格的。这个库，由于被认为是Python风格的，将能够与Python数据科学栈平滑集成。那些不想使用其他编程语言，只想坚持使用Python的基础知识和一些强大功能的人，将能够使用这个库做到这一点。你将能够利用所有通过Python环境提供的各种功能和服务。

3. 计算图。PyTorch库的另一个亮点是它为我们提供了一个包含动态计算图的平台。这意味着你甚至可以在运行时更改这些图。当你需要处理一些图，并且不确定在为神经网络创建模型时需要使用多少内存时，这通常会非常有用。

### PyTorch社区

接下来我们要看的是PyTorch库带来的一些领域。由于PyTorch带来的所有优势，我们可以看到开发者和其他专业人士的社区每天都在增长。在短短几年内，这个库已经取得了许多发展，甚至导致该库在许多研究论文和小组中被引用。当涉及到人工智能和深度学习模型时，PyTorch正开始成为最值得使用的库之一。

PyTorch带来的一个有趣之处是它仍处于早期发布测试阶段。但由于已经有如此多的程序员以如此快的速度采用这个深度学习框架，它已经展示了其带来的能力和潜力，以及社区可能如何继续增长。例如，尽管我们仍然处于PyTorch的测试版，但目前仅在GitHub仓库上就有741位贡献者。这意味着有超过700人正在努力加强和改进PyTorch已有的功能。

想想这有多惊人！PyTorch在技术上尚未正式发布，仍处于早期阶段。然而，围绕这个深度学习库已经有如此多的热议，如此多的程序员将其用于深度学习和人工智能，以至于已经有大量的贡献者正在为这个库添加更多功能和增强，以便其他人使用。

PyTorch不会限制我们正在使用的具体应用，因为它具有模块化设计和随之而来的灵活性。它已经被一些领先的科技巨头大量使用，你甚至可能认出其中一些名字。那些已经开始使用PyTorch来改进其深度学习模型的公司包括Uber、NVIDIA、Twitter和Facebook。这个库也已被用于许多领域的研究，包括神经网络、图像识别、翻译和自然语言处理等关键领域。

### 为什么在数据分析中使用PyTorch

任何在数据科学、数据分析、人工智能或深度学习领域工作的人，可能都花了一些时间使用我们在这本指南中也讨论过的TensorFlow库。TensorFlow可能是Google最受欢迎的库，但由于PyTorch这个深度学习框架，我们可以发现这个库在解决研究人员想要解决的研究工作方面的一些新问题时非常有能力。

人们认为，在管理数据方面，PyTorch现在是TensorFlow最强大的对手，并且在研究领域，它正成为最简单、最受欢迎的人工智能和深度学习库之一。发生这种情况的原因有很多，我们将在下面提到其中一些：

首先，强大的计算图将在研究人员中广为人知。这个库将避免其他框架（如TensorFlow）中使用的一些静态图。这使得研究人员和开发者能够在最后一刻改变网络的行为方式。采用这个库的一些人会喜欢它，因为与TensorFlow相比，这些图更直观易学。

第二个好处是它具有特殊的后端支持。PyTorch将根据你正在做的事情使用不同的后端。GPU、CPU和其他功能特性都将配备不同的后端，而不是专注于一个后端来处理所有这些。例如，我们将看到GPU使用THC，CPU使用TH。能够使用单独的后端可以使我们更容易通过各种受限系统部署这个库。

命令式风格是使用这类库的另一个优势。这意味着当我们使用这个库时，它易于使用且非常直观。当你执行一行代码时，它将按照你的意愿执行，并且你可以使用一些实时跟踪功能。这使程序员能够跟踪神经网络模型的表现。由于其出色的架构以及精简快速的方法，它已经能够增加我们在程序员社区中看到的这个库的一些普遍采用率。

在使用PyTorch时，我们将享受的另一个好处是它易于扩展。这个库特别集成了与C++代码良好协作的功能，并且在我们构建深度学习框架时，它将与该语言共享一些后端。这意味着程序员不仅可以使用Python来处理CPU和GPU，还可以使用C或C++语言来扩展API。这意味着我们可以将PyTorch的使用扩展到一些新的和实验性的案例中，这将使我们想要用它进行一些研究时变得更好。

最后，我们将在这里关注的最后一个好处是PyTorch将被视为一个Python风格的库。这是因为该库本身就是一个原生的Python包，这体现在它的设计方式上。随之而来的功能被构建为Python中的类，这意味着你在这里编写的所有代码都可以与Python的模块和包无缝集成。

类似于我们在NumPy中看到的，这个基于Python的库将使我们能够处理GPU加速的张量，并提供丰富的API选项来应用神经网络。PyTorch将为我们提供从头到尾所需的研究框架，这将包含我们日常进行深度学习研究所需的大部分不同构建块。我们还注意到，这个库将为我们提供高级神经网络模块，因为它可以与类似于Keras库的API一起工作。

### PyTorch 1.0 – 从研究到生产的途径

在本章中，我们花了一些时间讨论了PyTorch库的许多优势，以及这些优势如何帮助许多研究人员和数据科学家将其作为首选库。然而，这个库也有一些缺点，其中之一包括它在支持生产方面有所欠缺。然而，由于PyTorch可能发生的一些改进和变化，预计这很快就会改变。

PyTorch的下一个版本，即PyTorch 1.0，预计将成为一个重大发布，旨在克服研究人员、程序员和开发者在生产中面临的一些最大挑战。这是整个框架的最新迭代，预计它将与基于Python的Caffe2结合，允许深度学习研究人员和机器学习开发者从研究转向生产。这样做的目的是让这个过程以一种无忧无虑的方式进行，程序员无需处理迁移过程中出现的挑战。

1.0版本旨在帮助在一个框架中统一研究和生产功能，而不是分两部分进行，使事情变得更容易，并避免在尝试合并两部分时发生的一些价值损失和复杂性。这使我们能够获得更多的性能优化以及完成研究和生产所需的灵活性。

这个新版本将承诺在处理出现的任务方面提供大量帮助。其中许多任务将使你能够在更大规模上更高效地运行你的深度学习模型。除了生产支持之外，请记住PyTorch 1.0将在优化和可用性方面进行无数改进。

借助PyTorch 1.0库，我们将能够获取现有代码并继续按原样使用它。现有的API不会改变，这使得那些已经能够使用旧API创建一些代码和程序的人更容易。为了帮助理解PyTorch库即将发生的进展，你可以查看PyTorch网站。

正如我们在本章探讨的所有信息中看到的，PyTorch在人工智能和深度学习的各种流程中已经被视为一个引人注目的参与者。能够利用其带来的所有独特部分，并看到它将作为一个研究优先的库，这可能是我们数据分析概览。

PyTorch 库能够应对诸多挑战，并为我们提供完成工作所需的所有便利与性能。如果你是学生、研究人员或数学家，并且倾向于使用深度学习模型，那么 PyTorch 库将是一个出色的深度学习框架选择，助你入门。

# 第9章

## 在Python中创建软件包

创建软件包是Python编程语言最常见的应用之一。因此，本章将重点介绍创建和发布Python软件包的流程。此外，在一个编码会话中，创建和发布Python软件包的过程可以重复多次，使用户能够为不同应用创建多个软件包。

人们可能会好奇，在Python编码中学习创建软件包的根本目的或好处是什么。为了回答并解释创建Python软件包的生产力，以下几点关键信息有助于提供合适的背景：

- 软件包本质上为程序员在应用开发后期阶段准备好了所需资源。简单来说，程序员无需将时间花在初始设置上。
- 本章介绍的创建Python软件包的方法并非围绕实验性技术或变通方案。我们将探讨的方法不仅因其流行而受到许多程序员的青睐，而且易于理解。
- 讨论创建软件包也有助于那些在项目开发中采用“测试驱动”方法的程序员，使实现相对更容易。
- 借助软件包，发布过程也变得更加简单。

### 每个Python软件包的通用模式

每当我们创建一个应用程序时，通常需要编写大量代码行。即使是最简单的应用程序也可能有25到150行代码。这可能会使应用程序的可读性、维护、更新、补丁、错误修复甚至调试对任何人来说都相当困难。为了尽可能保持简单，我们采用不同的技术来组织应用程序的源代码。其中一种组织代码行的方法是使用“eggs”。

“Egg”本质上是一个术语，用于指代应用程序源代码的特定部分。除了是一个明确的术语外，“Egg”也是Python编程中的一个功能性元素。换句话说，我们将整个代码分成不同的部分，然后使用eggs将每个部分放入不同的软件包中。这样，应用程序的源代码变得更容易处理，如果需要，程序员也可以将部分代码重用于另一个项目。这无疑使程序员的生活变得更加轻松，因为他们可以方便地重用代码。从这个角度来看，软件包充当独立的组件，共同构建整个应用程序。通过这种方式，我们可以通过引入egg结构来分离应用程序的不同代码块，从而轻松创建每个Python软件包。

本节将主要使用`distutils`和`setuptools`这两个Python模块提供的功能来创建“命名空间软件包”。你还将学习使用这些Python模块来组织、发布和分发“命名空间软件包”的过程。

创建eggs非常简单直接。就像鸡蛋可以被打破以露出蛋黄和蛋白一样，在编程中，可以通过将不同的代码部分放入嵌套文件夹中来创建一个egg。具体来说，我们创建一个父文件夹，然后在这个文件夹内放入一个软件包，接着创建一个子文件夹并在其中放置另一个软件包，依此类推。在使用eggs创建“命名空间软件包”时，关键是每个子文件夹必须有一个被父文件夹使用的名称。例如，如果我们使用eggs创建一个“命名空间软件包”，并且根文件夹的名称是`python.advanced`，那么子文件夹的名称至少应有一个与根文件夹命名空间共同的前缀。在这种情况下，如果我们创建两个子文件夹，我们可以将第一个命名为`python`，另一个命名为`advanced`。

创建“命名空间软件包”的一个良好实践是在命名空间本身中定义代码的性质。例如，如果我们的软件包中的代码行旨在处理SQL数据库，我们可以为根文件夹的命名空间设置类似`python.sql`这样的名称。然后我们可以分别创建具有`python`和`sql`命名空间的两个文件夹。

### 在命名空间软件包中使用`setup.py`脚本

由于我们处理的是嵌套文件夹，因此需要一个能够管理多个子文件夹中软件包连续执行的脚本。我们可以将这个脚本视为游戏的安装向导，即使游戏数据被分割成多个单独的zip文件，它也能解压整个游戏。

`setup.py`脚本通常放置在嵌套文件夹的根目录中，以确保该脚本是第一个被执行的。如果主要控制机制不是第一个被初始化的，那它的目的是什么？这个脚本的结构也并不复杂。通过使用`setuptools`模块，我们可以扩展此功能以为我们的不同软件包创建所需的egg结构。

根据程序员的需求，他们可以轻松创建自己的自定义setup.py脚本文件，但创建一个最基本的setup.py文件需要使用以下两行代码：

```python
from setuptools import setup

setup(name='acme.sql')
```

我们在这里看到的是可以与函数一起使用的参数的简单演示。在接下来的部分中，你将看到我们将使用这个特定函数定义许多其他软件包元素。setup.py文件的目的是为我们执行创建软件包所需的更多指令（通过命令）奠定基础。虽然我们无法在此时此地解释每一个可能的控制命令，但该模块提供了一个完整的命令列表供用户查看。要提及命令列表，我们只需访问一个名为`--help-commands`的选项。为方便起见，下面已对此进行了演示：

```
$ python setup.py --help-commands
```

标准命令：

- build 构建安装所需的一切
- ...
- install 从构建目录安装所有内容
- sdist 创建源代码分发包
- register 注册分发包
- bdist 创建构建（二进制）分发包

额外命令：

- develop 以“开发模式”安装软件包
- ...
- test 在本地构建后运行单元测试
- alias 定义快捷方式
- bdist_egg 创建“egg”分发包

常用且比其他命令更受强调的命令属于“标准命令”家族。但这并不意味着额外命令无用，相反，它们提供了扩展功能，确保了原本可能需要不必要变通方案的任务得以实现。在我们继续讨论一些最重要的命令之前，需要注意的是，如果我们的Python IDE中没有安装任何扩展模块，例如`setuptools`模块，那么我们将无法访问“额外命令”列表。通过安装`distutils`模块，我们可以访问“标准命令”，而通过安装`setuptools`模块，我们可以使用其在“额外命令”下列出的相应命令。

### `sdist`命令

`sdist`命令不仅是最常用的命令，也是最简单的命令之一。此命令负责将软件包正常运行所需的所有文件复制到“发布树”中。然后它继续将此树归档。该树可以归档在一个文件中，也可以归档在多个文件中。

当程序员想要从“目标系统独立地”分发特定软件包时，主要会执行此命令。总而言之，这可以说是执行此类任务最简单的方法之一。一旦执行`sdist`命令，就会生成一个名为`dist`的文件夹，其中包含归档文件，这些文件包含软件包源代码树（最初创建的）的副本。通过这种方式，当我们分发这个`dist`文件时，我们实际上是在分发软件包本身。

我们可以通过向 `setup()` 函数传递 `version=' '` 参数来执行 `sdist` 命令。如果未为 `version` 参数指定值，则默认值将设置为 "0.0.0"。以下代码行演示了如何传递此参数：

```python
from setuptools import setup

setup(name='acme.sql', version='0.1.1')
```

顾名思义，`version='0.1.1'` 参数字面上指定了使用 `sdist` 命令分发的包的版本。这样，如果我们对包进行了更改（换句话说，更新了它），那么我们就修改版本值。这表明该系统自初始发布以来已经得到了改进。同样，每当我们使用 `sdist` 命令发布包时，我们都希望相应地更改版本值。

下面是一个快速演示，展示了如何使用带有另一个参数的 `sdist` 命令。

通常，我们在使用 `sdist` 命令时包含的版本也用作包含包本身的存档的标识符。使用 `sdist` 归档的包可以在任何安装了 Python 的系统上分发和使用。有些情况下，包的内容包含使用 'C++ 或 C 语言' 编码的应用程序数据，这些也是相对流行的编程语言。如果此类包使用 Python 中的 `sdist` 命令分发，那么编译 'C 代码' 行的责任完全落在目标系统本身。应该考虑到这种情况不太可能发生，因为基于 Linux 和 macOS 的系统也具有处理此类分发包的编译器，特别是当包需要分发到不同的操作系统时。这就是为什么如果你计划将其分发到不同的操作系统，将预构建分发与包一起包含始终是一个好主意。

### MANIFEST.in 文件

当我们使用 `sdist` 命令制作可分发包时，负责浏览包的整个文件目录以获取并列出可接受的文件以放入存档的正是 `distutils` 工具。

当 `distutils` 工具负责收集文件并将它们放入存档时，它通常会获取以下文件：

- 由以下选项指定的每个 Python 源文件：`py_modules`、`packages` 和 `scripts`。
- 如果包包含用 C 语言编写的指令，那么由 `ext_modules` 选项指定的每个 C 语言源文件。
- 所有符合以下 glob 模式的文件：`test/test*.py`。
- 目录中存在的每个信息文件和控制脚本，如 `README.txt`、`setup.py` 和 `setup.cfg`。

在 `distutils` 工具侦察完所需文件后，后续任务是从各自的目录中获取这些文件并将它们包含在可分发存档中。这是由 `sdist` 命令完成的。为此，`sdist` 命令生成一个名为 `MANIFEST` 的文件，并为包的所有文件创建列表，然后按顺序将它们添加到存档中。

当我们使用处理检查文件并将其包含在包存档中的责任的工具和命令时，此过程是自动的，无法控制。如果用户希望将其他文件包含在存档中，他们需要通过创建一个名为 `MANIFEST.in` 的模板文件来手动完成，然后将此文件放在找到 `setup.py` 脚本的同一目录中。然后我们指定要包含在存档中的文件及其各自的目录。然后 `sdist` 命令读取 `MANIFEST.in` 文件中包含的指令，并继续获取并包含指定的文件。

在此模板中，每行可以包含两种规则之一：包含规则和排除规则。例如，这里有一个显示包含多个文件的清单模板的示例：

```
include HISTORY.txt
include README.txt
include CHANGES.txt
include CONTRIBUTORS.txt
include LICENSE
recursive-include *.txt *.py
```

### `build` 和 `bdist` 命令

如果我们有一个预构建的分发包并且我们想要分发它，那么我们可以使用 `distutils` 工具中可用的 `build` 和 `bdist` 命令来完成。这些命令基本上通过编译包来工作。这总共分为 4 个阶段，Tarek Ziade 在他的书《Expert Python Programming》中对此进行了详细说明：

1. `build_py`：通过字节编译纯 Python 模块并将它们复制到构建文件夹中来构建它们。
2. `build_clib`：当包包含任何 C 库时，使用 Python 编译器构建它们，并在构建文件夹中创建一个静态库。
3. `build_ext`：构建 C 扩展并将结果放入构建文件夹，类似于 `build_clib`。
4. `build_scripts`：构建标记为脚本的模块。当第一行被设置（`#!`）时，它还会更改解释器路径，并修复文件模式使其可执行。

您可能已经注意到，这四个阶段基本上在每个步骤中都包含一个特殊命令。此外，每个命令都属于 `build` 家族，也可以独立执行。一旦这个完整的过程成功完成，最终产品将是一个 `build` 文件夹，其中包含包安装过程所需的所有元素和文件。但需要注意的一点是，`distutils` 工具不支持在同一个 `build` 编译过程中集成不同的编译器。换句话说，如果我们执行 `build` 命令，生成的文件夹将仅与其构建的特定系统兼容。但这在不久的将来可能会改变，因为第三方补丁和解决方法使 distutils 工具能够实现跨编译器兼容性。

当我们使用构建命令过程来制作，比如说，一个 C 语言扩展时，该过程将简单地使用系统的默认编译器以及一个 Python 头文件。在打包分发中，头文件，因此系统主要使用的编译器存储在一个名为 `python-dev` 的附加包中，但我们需要在系统上手动安装此包才能使用它。

对于基于 Windows 的计算机系统，使用的主要系统编译器是 `C`。另一方面，对于基于 Linux 和 macOS 的系统，系统使用的主要编译器不是 `C`，而是 `gcc`。

现在让我们再谈谈 build 和 bdist 命令。这两个命令以及本节开头解释的命令彼此依赖。详细来说，`bdist` 命令依赖于 `build` 命令来获取初始二进制分发。同样，`build` 命令使用四个额外的依赖命令来成功生成存档，其方式与 `sdist` 命令相同。

现在让我们看一个如何使用 `bdist` 命令生成二进制分发的示例。这次，我们将为基于 macOS 的目标系统而不是 Windows 系统进行操作。

如果我们想使用相同的过程为基于 Windows 的系统创建可分发存档，那么我们可以如下所示进行操作：

```
C:\acme.sql> python.exe setup.py bdist
...
C:\acme.sql> dir dist
25/02/2008 08:18 .
25/02/2008 08:18 ..
25/02/2008 08:24 16 055 acme.sql-0.1.win32.zip
1 File(s) 16 055 bytes
2 Dir(s) 22 239 752 192 bytes free
```

当 `bdist` 命令创建二进制分发版本时，主要内容是一个可以轻松复制到 Python 树中的 `tree`。简单来说，二进制分发具有一个文件夹，该文件夹只需复制到名为 `site-packages` 的 Python 目录中。

### `bdist_egg` 命令

与 `bdist` 命令并行的另一种分发模式是 `bdist_egg` 命令。这基本上是一个可以通过在 Python 中安装 setuptools 模块来使用的附加命令。它的功能与 `bdist` 命令非常相似。它不是创建一个简单的像 `bdist` 这样的二进制分发归档文件，特别是由 `bdist_egg` 命令生成的，其目录结构与源码分发的结构非常相似。这使得用户可以轻松地将分发归档文件下载到他们的系统上，解压缩文件，然后通过将解压后的文件夹放入由 `sys.path` 指定的 Python 搜索路径中来使用它。

### ‘install’ 命令

如果使用 `install` 命令的包没有任何先前的构建版本，那么 Python 将自动计划创建该包的构建版本，然后将其内容复制到 Python 目录树中。如果我们对一个具有源码分发的分发包调用 `install` 命令，那么该命令将简单地解压缩归档文件的内容，进入一个临时文件夹，然后从那里安装它。

如果我们希望安装一个分发包以及作为其依赖项的其他包，我们需要在调用 `setup()` 函数时手动指定这些包。例如，假设我们有一个名为 `acme.sql` 的包文件，并且我们希望同时安装 `pysqlite` 和 `SQLAlchemy` 作为其依赖项。为此，我们只需调用 `setup()` 函数并提供一个名为 `install_requires` 的参数。在此参数中指定的包将被视为原始包的依赖项。以下演示说明了此过程：

```
from setuptools import setup

setup(name='acme.sql', version='0.1.1',
install_requires=['pysqlite', 'SQLAlchemy'])
```

一旦执行 `install` 命令来安装 `acme.sql` 包，`pysqlite` 和 `SQLAlchemy` 包也将随之一起安装。

### ‘develop’ 命令

本节将介绍 Python 的 `setuptools` 模块提供的另一个非常高效的命令，即 `develop`。`develop` 命令本质上执行三项任务：

- 构建包
- 安装包
- 将包的链接添加到 Python 的 `site-packages` 文件夹

通过这种方式，用户可以处理包本身包含的代码，即使该代码可能是本地副本。

当我们使用 `develop` 命令安装包时，我们也可以相当容易地卸载它。卸载过程可以通过使用一个名为 `-u` 的选项来执行。以下是一个演示：

```
$ sudo python setup.py develop
running develop
...
Adding iw.recipe.fss 0.1.3dev-r7606 to easy-install.pth file
Installed /Users/repos/ingeniweb.sourceforge.net/iw.recipe.fss/trunk
Processing dependencies ...
$ sudo python setup.py develop -u
running develop
Removing
...
Removing iw.recipe.fss 0.1.3dev-r7606 from easy-install.pth file
```

另一个需要注意的重要事项是，包也可以使用 `sdist` 和 `bdist` 来安装。如果我们使用这两个命令中的任何一个来安装包，那么该包的特定版本将可在用户的系统上使用。如果我们使用 `develop` 命令在系统上安装相同的包，那么通过 `develop` 命令安装的包将优先于使用 `sdist` 和 `bdist` 安装的版本。

### ‘test’ 命令

正如构建和开发任务对于处理包很重要一样，测试我们创建的包以确定其是否正常工作也同样重要。`test` 命令可以做到这一点。该命令通过遍历所需的文件目录并执行测试过程来工作，之后，它会显示一个汇总结果，但该命令运行的测试在功能上相当有限。为了弥补这一点，建议使用外部测试运行器（如 `zope.testing` 或 `Nose`）来增强该命令的功能并弥补其局限性。

以下是一个快速演示，说明如何将 `Nose` 作为扩展测试运行器与 `test` 命令结合使用。

```
setup(
...
test_suite='nose.collector',
test_requires=['Nose'],
...
)
```

在第一个参数中，我们在命令的元数据中包含 `nose.collector`，然后为命令的执行添加一个依赖项，即 `Nose` 测试运行器本身。

### ‘register’ 和 ‘upload’ 命令

一旦你完成了可分发包的创建，下一步就是将其分发到其他系统。否则，浏览这些步骤并创建包就没有意义了。以下两个命令通常执行包分发任务：

- **Register**：此命令获取包的全部元数据并将其上传到目标服务器。
- **Upload**：此命令获取 `dist` 文件夹中存在的所有归档文件并将其上传到目标服务器。

Python 包的主要服务器可以通过以下地址访问：

http://pypi.python.org/pypi

此服务器通常由 Python 社区使用，包含大量由个人开发者和团队上传的包。一旦用户执行 `register` 命令，系统会自动在其主目录中创建一个 `.pypirc` 文件。

至此，你应该意识到一个包并不是创建一次后就放置不管了。开发者和程序员会努力改进、添加新功能，甚至通过上传更新版本来更新他们现有的包，但默认的 PyPI 服务器要求用户为其包创建一个用户帐户以进行身份验证。没有用户帐户，就无法将包直接上传到服务器，甚至无法更新现有包的版本。用户帐户可以直接通过命令行创建，如下所示：

```
$ python setup.py register
running register
...
We need to understand who you are, so please choose either:
```

1. 使用您现有的登录名，
2. 注册为新用户，
3. 让服务器为您生成一个新密码（并通过电子邮件发送给您），或
4. 退出

您的选择 [默认 1]：

完成后，系统将生成一个 `.pypirc` 文件并将其放置在其主目录中。

当我们使用 `register` 命令上传文件时，我们需要包含包的下载 URL 的元数据或指定 URL 本身。如果 URL 被验证为有效，服务器将在网页上注册该包，供人们访问和下载该包到他们的系统。

另一方面，如果我们使用 `upload` 命令，归档文件将直接上传到服务器，而无需用户指定 `download_url` 元数据。

每当一个包被上传到服务器时，它也需要被分类。这样，最终用户在浏览广泛的目录时可以轻松找到该包。默认情况下，`distutils` 工具使用一种称为“Trove 分类”的分类方法来对上传的包进行分类。这是一个静态列表，可以通过访问以下地址获取：

http://pypi.python.org/pypi?:action=list_classifiers

### 结论

我们现在结束了我们的旅程，无论它感觉多么短暂或漫长。一开始，我们了解了可能帮助我们创建 Python 项目并学习如何有效使用它们的工具。这些工具将对你未来处理的每种类型的项目都有帮助，因为它们是经验丰富的程序员和开发者的必备技能。一旦书本翻开，我们立即进入了本书中可以说是最重要的章节——语法。了解正确的语法并有效地实施它们，是区分糟糕代码和优秀代码的关键。正是出于这个目的，我们专门用了两整章来讲解语法，并探讨了两个不同的实现层面：类之下和类之上。

在所有这些关于高级概念和实际实现的详细讨论中，我们花时间学习了编码中的命名方案。经验丰富的程序员通常会根据项目的需求创建自己的自定义函数、类和模块。因此，命名它们成为一个关键因素，因为这些名称将被用于在程序中调用这些自定义元素。如果程序员的命名方案不遵循特定的方案或趋势，那么就会产生不必要的复杂性。最后，我们到达了本书中学习如何使用我们在本书之前和本书中学到的概念的部分。这些章节中使用的关键技术是构建包的技术以及如何通过使用包来创建应用程序。由于本书的重点是实用性，这些章节也强调了将包分发到社区服务器和其他系统的方法。