# 使用 Python 的 Tabnanny 模块来清理代码

> 原文：<https://www.pythoncentral.io/using-pythons-tabnanny-module-for-cleaner-code/>

Tabnanny 是 Python 中的一个模块，用于检查源代码中的模糊缩进。这个模块很有用，因为在 Python 中，空白不应该是模糊的，如果你的源代码包含任何奇怪的制表符和空格的组合，tabnanny 会让你知道。

您可以通过两种方式之一运行 tabnanny，要么使用命令行，要么在您的程序中运行它。从命令行运行它将如下所示:

```py
$ python -m tabnanny .
```

这将产生指定在哪个文件和哪个行中发现了错误的结果。如果您想查看正在扫描的所有文件的更多信息，请在编写命令时使用-v 选项:

```py
$ python -m tabnanny -v .
```

要在程序中运行 tabnanny，您需要将它与函数一起使用。check()，它应该看起来像这样:

```py
import sys
import tabnanny

tabnanny.check(file_or_dir)
```

只要你使用 tabnanny 来检查你所有的代码，你就不会有任何缩进错误的问题。因为这个过程通常非常快，而且几乎不需要任何编码来完成，所以这是一个显而易见的过程。