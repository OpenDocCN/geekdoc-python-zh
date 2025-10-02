# 第二章 安装 Python

**目录表**

*   Linux 和 BSD 用户
*   Windows®用户
*   概括

如果你正在使用一个 Linux 的发行版比如 Fedora 或者 Mandrake 或者其他（你的选择），或者一个 BSD 系统比如 FreeBSD，那么你可能已经在你的系统里安装了 Python。

要测试你是否已经随着你的 Linux 包安装了 Python，你可以打开一个 shell 程序（就像 konsole 或 gnome-terminal）然后输入如下所示的命令**`python -V`**。

```py
$ python -V
Python 2.3.4 
```

注释 `$`是 shell 的提示符。根据你的操作系统的设置，它可能与你那个不同，因此我只用`$`符号表示提示符。

如果你看见向上面所示的那样一些版本信息，那么你已经安装了 Python 了。

如果你得到像这样的消息：

```py
$ python -V
bash: python: command not found 
```

那么你还没有安装 Python。这几乎不可能，只是极其偶尔才会遇到。

在这种情况下，你有两种方法在你的系统上安装 Python。

*   利用你的操作系统附带的包管理软件安装二进制包，比如 Fedora Linux 的 yum、Mandrake Linux 的 urpmi、Debian Linux 的 apt-get、FreeBSD 的 pkg_add 等等。注意，使用这种方法的话，你需要连接因特网。

    你也可以从别的地方下载二进制包然后拷贝到你的 PC 中安装。

*   你可以从[源代码](http://www.python.org/download/)编译 Python 然后安装。在网站上有编译的指令。

# Windows®用户

Windows®用户可以访问[Python.org/download](http://www.python.org/download/)，从网站上下载最新的版本（在写本书的时候，最新版本是[2.3.4](http://www.python.org/ftp/python/2.3.4/Python-2.3.4.exe)版）。它的大小大约是 9.4MB，与其他大多数语言相比是十分紧凑的。安装过程与其他 Windows 软件类似。

提示 即便安装程序为你提供了不检查 可选 组件的选项，你也不要不作任何检查！有些组件对你很有用，特别是集成开发环境。

有趣的是，大约 70%的 Python 下载是来自 Windows 用户的。当然，这并不能说明问题，因为几乎所有的 Linux 用户已经在安装系统的时候默认安装了 Python。

在 Windows 命令行中使用 Python 如果你想要从 Windows 命令行调用 Python，那么你需要先正确的设置 PATH 变量。

对于 Windows 2000、XP、2003，点击控制面板->系统->高级->环境变量。在“系统变量”表单中点击叫做**PATH**的变量，然后编辑这个变量，把**;C:\Python23**加到它的结尾。当然，是 Python 所在的正确目录名。

对于较旧版本的 Windows，把下面这行加到文件 C:\AUTOEXEC.BAT 中：**PATH=%PATH%;C:\Python23**，然后重新启动系统。对于 Windows NT，则使用 AUTOEXEC.NT 文件。

# 概括

对于 Linux 系统，很可能你已经在你的系统里安装了 Python。否则，你可以通过你的发行版附带的包管理软件安装 Python。对于 Windows 系统，安装 Python 就是下载安装程序然后双击它那么简单。从现在起，我们将假设你已经在你的系统里安装了 Python。