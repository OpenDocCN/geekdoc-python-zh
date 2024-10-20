# Mercurial 简介

> 原文：<https://www.blog.pythonlibrary.org/2012/07/23/an-intro-to-mercurial/>

Mercurial 是一个免费的分布式源代码控制版本工具，类似于 git 或 bazaar。有些人甚至会把它比作 CVS 或 Subversion (SVN ),尽管它们不是分布式版本控制系统。几年前，Python 编程核心开发团队选择从 SVN 转向 Mercurial，许多其他知名的第三方 Python 项目也是如此。有许多项目也在使用 Git。在本教程中，我们将学习使用 Mercurial 的基础知识。如果你需要更深入的信息，有一个相当[详尽的指南](https://www.mercurial-scm.org/wiki/Tutorial)和一个[在线 Mercurial book](http://hgbook.red-bean.com/) 应该可以满足你的需求。

### 入门指南

首先，您可能需要安装 Mercurial。按照他们的[网站](http://mercurial.selenic.com/)上的指示。真的很好做。现在我们可以使用它了。如果安装正确，您应该能够打开一个终端(即命令行)窗口，并创建一个新的存储库或签出一个。在本文中，您将学习如何做到这两点。我们将从头开始创建我们自己的存储库。

首先，在你的机器上创建一个新的文件夹。我们将使用“C:\ repo”(Windows)。当然，你的道路会有所不同。然后在你的终端中，将目录切换到 repo 文件夹。现在键入以下内容:

```py

hg init

```

等等！那个“hg”是什么意思？“hg”是水银的化学符号。您将始终使用“hg”命令与 Mercurial 存储库(又名:repos)进行交互。您的文件夹现在应该有一个**。hg** 文件夹，其中将包含一个空的**存储**文件夹，一个 changelog 文件(某种虚拟文件)和一个**需要的**文件。在大多数情况下，您并不真正关心**中有什么。hg** 文件夹，因为它存在的目的是存储回购信息，以便 Mercurial 可以恢复到以前的版本并跟踪历史。

当然，如果存储库是空的，它就没有多大用处，所以让我们在其中放一个文件。事实上，我们将使用以前的[文章](https://www.blog.pythonlibrary.org/2012/07/08/python-201-creating-modules-and-packages/)中的愚蠢的数学包，这样我们将有几个文件和一个文件夹。现在输入**汞状态**(或“汞 st”)。您应该会看到类似这样的内容:

```py

C:\repo>hg st
? __init__.py
? add.py
? adv\__init__.py
? adv\fib.py
? adv\sqrt.py
? divide.py
? mods_packs.wpr
? mods_packs.wpu
? multiply.py
? subtract.py

```

这是我们可能添加到回购中的文件列表。问号表示尚未添加。你会注意到我不小心把一些 Wingware IDE 文件留在了存档中，这些文件的扩展名是 wpr 或 wpu。如果你有翅膀，这些就很方便。如果你没有，那你就不会在乎。假设我们有 Wing，但是我们不想共享我们的 Wing 配置文件。我们需要一种方法将它们从存储库中排除。

### 从 Mercurial 中排除文件

Mercurial 包含一个名为**的特殊文件。hgignore** 放在我们的根目录中，Mercurial 将查看它以获得关于忽略什么的指示。您可以使用正则表达式或 glob(或者您可以选择)来告诉 Mercurial 忽略什么。在 Windows 上，您可能需要使用记事本或类似的文本编辑器来创建文件，因为 Windows 资源管理器很笨，不允许您以句点开头命名文件。无论如何，要设置类型(glob 或 regex)，您需要在它前面加上“syntax:”这个词。这里有一个例子可以特别清楚地说明这一点:

```py

# use glob syntax.
syntax: glob

*.pyc
*.wpr
*.wpu

```

这告诉 Mercurial 忽略任何带有以下扩展名的文件:pyc、wpr 或 wpu。它还使用了 glob 语法。如果您重新运行 **hg st** 命令，您将不会再看到这些 Wing 文件。现在我们已经准备好向回购添加文件了！

### 如何将文件/目录添加到您的 Mercurial 存储库中

向 Mercurial 添加文件也非常简单。你要做的就是输入: **hg add** 。下面是我运行它时得到的结果:

```py

C:\repo>hg status
A .hgignore
A __init__.py
A add.py
A adv\__init__.py
A adv\fib.py
A adv\sqrt.py
A divide.py
A multiply.py
A subtract.py

```

注意每个项目前面的大写字母“A”。这意味着它已经被添加，但是没有提交到存储库中。如果它没有被提交，那么它还没有保存在您的版本控制中。我们应该这么做！你所要做的就是输入 **hg commit** 或 **hg com** ，它会打开一个文本编辑器让你输入关于你提交的评论。嗯，实际上这就是*应该*发生的事情，但是除非你已经建立了一个配置文件，否则你可能会得到这个神秘的消息:**提交没有用户名提供**。这就是事情变得有趣的地方。您需要为 Mercurial 创建一个配置文件。

### 如何创建 Mercurial 配置文件

Mercurial 在几个地方查找配置文件。下面是他们文档中列出的顺序:

```py

/.hg/hgrc
(Unix) $HOME/.hgrc
(Windows) %USERPROFILE%\.hgrc
(Windows) %USERPROFILE%\Mercurial.ini
(Windows) %HOME%\.hgrc
(Windows) %HOME%\Mercurial.ini 
```

请注意，如果您创建一个每回购配置文件(第一个示例)，hgrc 文件不会以句点开头。否则，在创建带句点的配置时，您会创建一个系统范围的配置，并将它放在上面指定的位置(或者创建一个 Mercurial.ini 文件)。我们将创建一个简单的 repo 配置文件，并将其放在我们的。hg 文件夹。以下是一些示例内容:

```py

[ui]
username = Mike Driscoll 
```

现在，如果您保存它并重试 commit 命令，它应该会工作。在 Windows 上，它会弹出一个记事本实例，你可以在其中写下你正在做的事情的简短评论。如果你不在里面放东西，那么提交会中止，你不会保存任何东西到回购。一旦你提交了它，你可以再次运行 **hg st** ，它不会返回任何东西。现在，无论何时你在 repo 中编辑一个文件，你都可以运行 **hg st** ，你会看到一个所有已更改文件的列表。只需让他们添加变更即可。如果你想检查别人的代码呢？很高兴你问了。我们接下来会谈到这个问题！

### 如何签出 Mercurial 存储库

假设我们想从 bitbucket 查看我的旧 wxPython 音乐播放器项目。为此，您应该键入以下内容:

```py

hg clone https://bitbucket.org/driscollis/mamba

```

这将在您的硬盘上创建一个 **mamba** 文件夹，无论您当前在文件系统中的什么位置。因此，如果您已经将目录更改到您的桌面，则该文件夹将放在那里。请注意关键字**克隆**，后跟一个 URL。这是另一个例子:

```py

hg clone http://hg.python.org/cpython

```

如果你想对 Python 编程语言的开发有所帮助，这就是检验 Python 编程语言的方法。如果你是一个 Windows 用户，你想要更多的信息，你可以在这里阅读更多的信息。这是一个检验一个**共享**库的例子。在您上传更改之前，它可能会更改。所以你应该总是做一个**拉**来确保在你**把你的变更推**回共享回购之前你有最新的副本。当我想更新我的本地 Python 库时，我会这样做:

```py

C:\Users\mdriscoll\cpython>hg pull
pulling from http://hg.python.org/cpython
searching for changes
adding changesets
adding manifests
adding file changes
added 58 changesets with 109 changes to 36 files
(run 'hg update' to get a working copy)

C:\Users\mdriscoll\cpython>hg update
654 files updated, 0 files merged, 276 files removed, 0 files unresolved

```

因此，当您下载更改时，您必须进行**更新**来更新您的本地存储库。然后，当您想要将您的更改发送回服务器存储库时，您可以执行一个 **hg 推送**。大多数情况下，它会要求您输入用户名和密码，或者您可以在配置文件中设置用户名，然后您只需输入密码。一些存储库也要求 pull 请求的凭证，但是这种情况并不常见。您还可以使用 **hg update - rev** 来指定要更新到哪个版本。与 **log** 命令结合使用，找出有哪些修订。

### 使用 Mercurial 创建补丁

通常，在获得特权之前，你是不允许为开源项目做贡献的，所以你必须创建补丁。幸运的是，Mercurial 让创建补丁变得超级简单！只需克隆(或获取最新的更改)存储库并编辑您需要的文件。然后这样做:

```py

hg diff > example.patch

```

您可能应该以您更改的文件来命名补丁，但这毕竟是一个示例。如果你在修复一个 bug 的过程中提交了同一个东西的多个补丁，你可能会想通过在每个补丁的末尾添加一个递增的数字来重命名它，这样就很容易分辨哪个是最新的补丁。你可以在这里阅读更多关于向 Python [提交补丁的信息。](https://www.blog.pythonlibrary.org/2012/05/22/core-python-development-how-to-submit-a-patch/)

### 其他零碎的东西

还有其他几个命令我们没有时间介绍。可能最需要了解的是 **merge** ，它允许您将两个存储库合并在一起。你应该阅读他们的[文档](http://hgbook.red-bean.com/read/a-tour-of-mercurial-merging-work.html)了解更多。这里列出了你最常看到的典型命令，你只需输入 **hg** 就能得到:

```py

C:\Users\mdriscoll\cpython>hg
Mercurial Distributed SCM

basic commands:

 add         add the specified files on the next commit
 annotate    show changeset information by line for each file
 clone       make a copy of an existing repository
 commit      commit the specified files or all outstanding changes
 diff        diff repository (or selected files)
 export      dump the header and diffs for one or more changesets
 forget      forget the specified files on the next commit
 init        create a new repository in the given directory
 log         show revision history of entire repository or files
 merge       merge working directory with another revision
 phase       set or show the current phase name
 pull        pull changes from the specified source
 push        push changes to the specified destination
 remove      remove the specified files on the next commit
 serve       start stand-alone webserver
 status      show changed files in the working directory
 summary     summarize working directory state
 update      update working directory (or switch revisions)

use "hg help" for the full list of commands or "hg -v" for details

```

### 包扎

现在，您应该已经了解了足够的知识，可以开始在自己的工作中使用 Mercurial 了。这是存储代码和返回到以前版本的一个很好的方法。Mercurial 还使协作变得更加容易。对于那些喜欢点击的人来说，有命令行工具和一直流行的乌龟图形用户界面。祝你好运！

### 进一步阅读

*   [水银指南](http://mercurial.selenic.com/guide/)
*   。hgignore [文档](http://www.selenic.com/mercurial/hgignore.5.html)