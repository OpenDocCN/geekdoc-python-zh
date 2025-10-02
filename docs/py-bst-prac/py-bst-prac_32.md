# 你的开发环境

## 文本编辑器

任何能够编辑普通文本的编辑器都能够用来编写 Python 代码，然后，使用一个更加强大的编辑器可能使你的生活变得容易点。

### Vim

Vim 是一个使用键盘快捷键而不是菜单或图标来编辑的文本编辑器。有许多增强 Vim 编辑器中 Python 开发环境的插件和设置。如果你只开发 Python，使用缩进和换行均符合 [**PEP 8**](https://www.python.org/dev/peps/pep-0008) [https://www.python.org/dev/peps/pep-0008] 要求的默认设置是一个好的开始。在你的 home 目录中，打开 `.vimrc` 文件，添加下面这些内容:

```py
set textwidth=79  " lines longer than 79 columns will be broken
set shiftwidth=4  " operation >> indents 4 columns; << unindents 4 columns
set tabstop=4     " a hard TAB displays as 4 columns
set expandtab     " insert spaces when hitting TABs
set softtabstop=4 " insert/delete 4 spaces when hitting a TAB/BACKSPACE
set shiftround    " round indent to multiple of 'shiftwidth'
set autoindent    " align the new line indent with the previous line 
```

基于上述设置，新行会在超过 79 个字符被添加，tab 键则会自动转换为 4 个空格。如果你还使用 Vim 编辑其他语言，有一个叫做 [indent](http://www.vim.org/scripts/script.php?script_id=974) [http://www.vim.org/scripts/script.php?script_id=974] 的便捷插件可以让这个设置只为 Python 源文件服务。

还有一个方便的语法插件叫做 [syntax](http://www.vim.org/scripts/script.php?script_id=790) [http://www.vim.org/scripts/script.php?script_id=790] ，改进了 Vim 6.1 中的语法文件。

这些插件使你拥有一个基本的环境进行 Python 开发。要最有效的使用 Vim，你应该市场检查代码的语法错误和是否符合 PEP8。幸运的是， [PEP8](http://pypi.python.org/pypi/pep8/) [http://pypi.python.org/pypi/pep8/] 和 [Pyflakes](http://pypi.python.org/pypi/pyflakes/) [http://pypi.python.org/pypi/pyflakes/] 将会帮你做这些。如果你的 Vim 是用 `+python` 编译的，你也可以在编辑器中使用一些非常有用的插件来做这些检查。

对于 PEP8 检查和 pyflakes，你可以安装 [vim-flake8](https://github.com/nvie/vim-flake8) [https://github.com/nvie/vim-flake8] 。然后你就可以在 Vim 中把 `Flake8` 映射到任何热键或你想要的行为上。这个插件将会在屏幕下方显示出错误，并且提供一个简单的方式跳转到相关行。在保存文件的时候调用这个功能会是非常方便的。要这么做，就把下面一行加入到你的 `.vimrc`:

```py
autocmd BufWritePost *.py call Flake8() 
```

如果你已经在使用 [syntastic](https://github.com/scrooloose/syntastic) [https://github.com/scrooloose/syntastic] ，你可以设置它来运行 Pyflakes，并在 quickfix 窗口中显示错误和警告。一个这样做并还会在状态栏中显示状态和警告信息的样例是:

```py
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*
let g:syntastic_auto_loc_list=1
let g:syntastic_loc_list_height=5 
```

#### Python-mode

[Python-mode](https://github.com/klen/python-mode) [https://github.com/klen/python-mode] 是一个在 Vim 中使用 Python 的综合解决方案。 它拥有：

*   任意组合的异步 Python 代码检查（ `pylint` 、 `pyflakes` 、 `pep8` 、 `mccabe`）
*   使用 Rope 进行代码重构和补全
*   Python 快速折叠
*   支持 virtualenv
*   搜索 Python 文档，运行 Python 代码
*   自动修复 [PEP8](http://pypi.python.org/pypi/pep8/) [http://pypi.python.org/pypi/pep8/] 错误

以及其他更多。

#### SuperTab

[SuperTab](http://www.vim.org/scripts/script.php?script_id=1643) [http://www.vim.org/scripts/script.php?script_id=1643] 是一个小的 Vim 插件，通过使用 `<Tab>` 或任何其他定制的按键，能够使代码补全变得更方便。

### Emacs

Emacs 是另一个强大的文本编辑器。它是完全可编程的（lisp），但要正确的工作要花些功夫。如果你已经是一名 Emacs 的用户了，在 EmacsWiki 上的 [Python Programming in Emacs](http://emacswiki.org/emacs/PythonProgrammingInEmacs) [http://emacswiki.org/emacs/PythonProgrammingInEmacs] 将会是好的开始。

1.  Emacs 本身支持 Python 模式。

### TextMate

[TextMate](http://macromates.com/) [http://macromates.com/] 将苹果操作系统技术带入了文本编辑器的世界。通过桥接 UNIX 和 GUI，TextMate 将两者中最好的部分带给了脚本专家和新手用户。

### Sublime Text

[Sublime Text](http://www.sublimetext.com/) [http://www.sublimetext.com/] 是一款高级的，用来编写代码、标记和文章的文本编辑器。你将会爱上漂亮的用户界面、非凡的特性和惊人的表现。

Sublime Text 对编写 Python 代码支持极佳，而且它使用 Python 写其插件 API。它也拥有大量各式各样的插件， [其中一些](https://github.com/SublimeLinter/SublimeLinter) [https://github.com/SublimeLinter/SublimeLinter] 允许编辑器内的 PEP8 检查和代码提示。

### Atom

[Atom](https://atom.io/) [https://atom.io/] 是一款 21 世纪的可删减的（hackable）文本编辑器。它基于我们所喜欢的编辑器的任何优秀特性，并构建于 atom-shell 上。

Atom 是 web 原生的（HTML、CSS、JS），专注于模块化的设计和简单的插件开发。它自带本地包管理和大量的包。Python 开发所推荐的插件是 [Linter](https://github.com/AtomLinter/Linter) [https://github.com/AtomLinter/Linter] 和 [linter-flake8](https://github.com/AtomLinter/linter-flake8) [https://github.com/AtomLinter/linter-flake8] 的组合。

## IDEs

### PyCharm / IntelliJ IDEA

[PyCharm](http://www.jetbrains.com/pycharm/) [http://www.jetbrains.com/pycharm/] 由 JetBrains 公司开发，此公司还以 IntelliJ IDEA 闻名。它们都共享着相同的基础代码，PyCharm 中大多数特性能通过免费的 [Python 插件](https://plugins.jetbrains.com/plugin/?idea&pluginId=631) [https://plugins.jetbrains.com/plugin/?idea&pluginId=631] 带入到 IntelliJ 中。PyCharm 由两个版本：专业版（Professional Edition）（30 天试用）和拥有相对少特性的社区版（Community Edition）（Apache 2.0 License）。

### Enthought Canopy

[Enthought Canopy](https://www.enthought.com/products/canopy/) [https://www.enthought.com/products/canopy/] 是一款专门面向科学家和工程师的 Python IDE，它预装了为数据分析而用的库。

### Eclipse

Eclipse 中进行 Python 开发最流行的插件是 Aptana 的 [PyDev](http://pydev.org) [http://pydev.org] 。

### Komodo IDE

[Komodo IDE](http://www.activestate.com/komodo-ide) [http://www.activestate.com/komodo-ide] 由 ActiveState 开发，并且是在 Windows、Mac 和 Linux 平台上的商业 IDE。

### Spyder

[Spyder](https://github.com/spyder-ide/spyder) [https://github.com/spyder-ide/spyder] 是一款专门面向和 Python 科学库（即 [Scipy](http://www.scipy.org/) [http://www.scipy.org/] ）打交道的 IDE。它集成了 [pyflakes](http://pypi.python.org/pypi/pyflakes/) [http://pypi.python.org/pypi/pyflakes/] 、 [pylint](http://www.logilab.org/857) [http://www.logilab.org/857] 和 [rope](https://github.com/python-rope/rope) [https://github.com/python-rope/rope] 。

Spyder 是开源的（免费的），提供了代码补全、语法高亮、类和函数浏览器，以及对象检查的功能。

### WingIDE

[WingIDE](http://wingware.com/) [http://wingware.com/] 是一个专门面向 Python 的 IDE。它能运行在 Linux、Windows 和 Mac（作为一款 X11 应用程序，会使某些 Mac 用户遇到困难）上。

WingIDE 提供了代码补全、语法高亮、源代码浏览器、图形化调试器的功能，还支持版本控制系统。

### NINJA-IDE

[NINJA-IDE](http://www.ninja-ide.org/) [http://www.ninja-ide.org/] （来自递归缩写：”Ninja-IDE Is Not Just Another IDE”）是一款跨平台的 IDE，特别设计成构建 Python 应用，并能运行于 Linux/X11、Mac OS X 和 Windows 桌面操作系统上。从网上可以下载到这些平台的安装包。

NINJA-IDE 是一款开源软件（GPLv3 许可），是使用 Python 和 Qt 开发。在 [GitHub](https://github.com/ninja-ide) [https://github.com/ninja-ide] 能下载到源文件。

### Eric (The Eric Python IDE)

[Eric](http://eric-ide.python-projects.org/) [http://eric-ide.python-projects.org/] 是一款功能齐全的 Python IDE，提供源代码自动补全、语法高亮、对版本控制系统的支持、对 Python 3 的支持、集成的 web 浏览器、Python Shell、集成的调试器和灵活的插件系统等功能。它基于 Qt GUI 工具集，使用 Python 编写，集成了 Scintilla 编辑器控制。Eric 是一款超过 10 年活跃开发的开源软件工程（GPLv3 许可）。

## 解释器工具

### 虚拟环境

虚拟环境提供了隔离项目包依赖的强大方式。这意味着你无须再系统范围内安装 Python 工程特定的包，因此就能避免潜在的版本冲突。

To start using and see more information: [Virtual Environments](http://github.com/kennethreitz/python-guide/blob/master/docs/dev/virtualenvs.rst) [http://github.com/kennethreitz/python-guide/blob/master/docs/dev/virtualenvs.rst] docs. 开始使用和查阅更多信息：请参阅 [Virtual Environments](http://github.com/kennethreitz/python-guide/blob/master/docs/dev/virtualenvs.rst) [http://github.com/kennethreitz/python-guide/blob/master/docs/dev/virtualenvs.rst] 文档。

### pyenv

[pyenv](https://github.com/yyuu/pyenv) [https://github.com/yyuu/pyenv] 是一个允许多个 Python 解释器版本同时安装于一台机器的工具。这解决了不同的项目需要不同版本的 Python 的问题。比如，为了兼容性，可以很容易地为一个项目安装 Python 2.7，而继续使用 Python 3.4 作为默认的编辑器。pyenv 不止限于 CPython 版本——它还能安装 PyPy、anaconda、miniconda、stackless、jython 和 ironpython 解释器。

pyenv 的工作原理是在一个叫做 `shims` 目录中创建 Python 解释器（以及其他工具像 `pip` 和 `2to3` 等）的假版本。当系统寻找名为 `python` 的应用时，它会先在 `shims` 目录中查找，并使用那个假版本，然后会传递命令到 pyenv 中。pyenv 基于环境变量、 `.python-version` 文件和全局默认设置的信息就知道该运行哪个版本的 Python。

pyenv 不是管理虚拟环境的工具，但是有一个叫做 [pyenv-virtualenv](https://github.com/yyuu/pyenv-virtualenv) [https://github.com/yyuu/pyenv-virtualenv] 的插件可以自动化不同环境的创建，而且也能够使用现有的 pyenv 工具，基于环境变量或者 `.python-version` 文件，来切换不同的环境。

## 其他工具

### IDLE

[IDLE](http://docs.python.org/library/idle.html#idle "(在 Python v2.7)") [http://docs.python.org/library/idle.html#idle] 是一个集成的开发环境，它是 Python 标准库的一部分。它完全由 Python 编写，并使用 Tkinter GUI 工具集。尽管 IDLE 不适用于作为成熟的 Python 开发工具，但它对尝试小的 Python 代码和对 Python 不同特性的实验非常有帮助。

它提供以下特性：

*   Python Shell 窗口（解释器）
*   多窗口文本编辑器，支持彩色化 Python 代码
*   最小的调试工具

### IPython

[IPython](http://ipython.org/) [http://ipython.org/] 提供一个丰富的工具集来帮助你最大限度地和 Python 交互。它主要的组件有：

*   强大的 Python shell（终端和基于 Qt）。
*   一个基于网络的笔记本，拥有相同的核心特性，但是支持富媒体、文本、代码、数学表达式和内联绘图。
*   支持交互式的数据可视化和 GUI 工具集的使用。
*   灵活、嵌入的解释器载入到你的工程工程中。
*   支持高级可交互的并行计算的工具。

```py
$ pip install ipython 
```

下载和安装带有所有可选依赖（notebook、qtconsol、tests 和其他功能）的 IPython

```py
$ pip install ipython[all] 
```

### BPython

[bpython](http://bpython-interpreter.org/) [http://bpython-interpreter.org/] 在类 Unix 操作系统中可替代 Python 解释器的接口。它有以下特性：

*   内联的语法高亮。
*   行内输入时的自动补全建议。
*   任何 Python 函数的期望参数列表。
*   从内存中 pop 出代码的最后一行并重新运行（re-evaluate）的“倒带”功能.
*   将输入的代码发送到 pastebin。
*   将输入的代码保存到一个文件中。
*   自动缩进。
*   支持 Python 3。

```py
$ pip install bpython 
```

### ptpython

[ptpython](https://github.com/jonathanslenders/ptpython/) [https://github.com/jonathanslenders/ptpython/] 是一个构建在 [prompt_toolkit](http://github.com/jonathanslenders/python-prompt-toolkit) [http://github.com/jonathanslenders/python-prompt-toolkit] 库顶部的 REPL。它被视作是 [BPython](http://bpython-interpreter.org/) [http://bpython-interpreter.org/] 的替代。特性包括：

*   语法高亮
*   自动补全
*   多行编辑
*   Emacs 和 VIM 模式
*   代码中嵌入的 REPL
*   语法合法性
*   Tab 页
*   通过安装 Ipython `pip install ipython` 并运行 `ptipython` ，支持集成 [IPython](http://ipython.org/) [http://ipython.org/] 的 shell

```py
$ pip install ptpython 
```

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.