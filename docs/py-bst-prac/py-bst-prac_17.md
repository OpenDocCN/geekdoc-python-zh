# 命令行应用

命令行应用，也被称为 [控制台应用](http://en.wikipedia.org/wiki/Console_application) [http://en.wikipedia.org/wiki/Console_application] 是面向如 [shell](http://en.wikipedia.org/wiki/Shell_(computing)) [http://en.wikipedia.org/wiki/Shell_(computing)] 之类文本接口的计算机程序。 命令行应用通常接收一些输入作为参数，这些参数（arguments）通常被称为参数（parameters）或子命令 ，而选项（options）则被称为 flags 或 switches。

一些流行的命令行应用包括：

*   [Grep](http://en.wikipedia.org/wiki/Grep) [http://en.wikipedia.org/wiki/Grep] - 一个纯文本数据搜索工具
*   [curl](http://curl.haxx.se/) [http://curl.haxx.se/] - 基于 URL 语法的数据传输工具
*   [httpie](https://github.com/jakubroztocil/httpie) [https://github.com/jakubroztocil/httpie] - 一个用户友好的命令行 HTTP 客户端，可以代替 cURL
*   [git](http://git-scm.com/) [http://git-scm.com/] - 一个分布式版本控制系统
*   [mercurial](http://mercurial.selenic.com/) [http://mercurial.selenic.com/] - 一个主体是 Python 的分布式版本控制系统

## Clint

[clint](https://pypi.python.org/pypi/clint/) [https://pypi.python.org/pypi/clint/] 是一个 Python 模块，它包含了很多 对命令行应用开发有用的工具。它支持诸如 CLI 着色以及缩进，简洁而强大的列打印， 基于进度条的迭代以及参数控制的特性。

## Click

[click](http://click.pocoo.org/) [http://click.pocoo.org/] 是一个即将出品的 Python 包，它创建了一个命令行接口， 可以尽可能的简化组合代码。命令行接口创建工具（“Command-line Interface Creation Kit”,Click） 支持很多配置但也有开箱可用的默认值设定。

## docopt

[docopt](http://docopt.org/) [http://docopt.org/] 是一个轻量级，高度 Pythonic 风格的包，它支持 简单而直觉地创建命令行接口，它是通过解析 POSIX-style 的用法指示文本实现的。

## Plac

[Plac](https://pypi.python.org/pypi/plac) [https://pypi.python.org/pypi/plac] Python 标准库 [argparse](http://docs.python.org/2/library/argparse.html) [http://docs.python.org/2/library/argparse.html] 的简单封装，它隐藏了大量声明接口的细节：参数解析器是被推断的，其优于写命令明确处理。 这个模块的面向是不想太复杂的用户，程序员，系统管理员，科学家以及只是想 写个只运行一次的脚本的人们，使用这个命令行接口的理由是它可以快速实现并且简单。

## Cliff

[Cliff](http://docs.openstack.org/developer/cliff/) [http://docs.openstack.org/developer/cliff/] 是一个建立命令行程序的框架。 它使用 setuptools 入口点（entry points）来提供子命令，输出格式化，以及其他的扩展。这个框架 可以用来创建多层命令程序，如 subversion 与 git，其主程序要进行一些简单的参数解析然后调用 一个子命令干活。

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.