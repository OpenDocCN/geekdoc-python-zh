# Python 的 _winreg:编辑 Windows 注册表

> 原文：<https://www.blog.pythonlibrary.org/2010/03/20/pythons-_winreg-editing-the-windows-registry/>

Python 的标准库以包含大量方便的模块和包而闻名，这些模块和包无需安装任何其他东西就可以使用。这是它的标准库经常被称为“包含电池”的主要原因之一。因此，Python 包含一个用于编辑 Windows 注册表的 Windows 专用模块也就不足为奇了。这个特殊的模块有一个奇怪的名字 *_winreg* (奇怪是因为它以下划线开头)。在本文中，我们将学习使用这个“电池”使用注册表的基本知识。

## 从注册表中读取

使用 Python 从注册表中读取数据非常容易。在下面的示例中，我们将找出 Outlook Express 的安装位置:

```py

from _winreg import *
key = OpenKey(HKEY_LOCAL_MACHINE, r'Software\Microsoft\Outlook Express', 0, KEY_ALL_ACCESS)
QueryValueEx(key, "InstallRoot")

```

在我的机器上，这将返回以下元组:(u ' % program files % \ \ Outlook Express '，2)。元组由值和所述值的注册类型组成。还有另外两种可以使用的查询方法，称为 QueryInfoKey 和 QueryValue。前者以三个整数的形式提供关于键本身的信息，而后者只检索名称为空的键的第一个值的数据。[文档](http://docs.python.org/library/_winreg.html)建议尽可能使用 QueryValueEx。

我们可能也应该快速解释一下上面的代码中发生了什么。OpenKey 函数采用一个 HKEY*常量、一个子密钥路径字符串、一个保留整数(必须为零)和安全掩码。在本例中，我们传入了 KEY_ALL_ACCESS，这使我们可以完全控制那个键。因为我们所做的只是读取它，我们可能应该只使用 KEY_READ。至于 QueryValueEx 做什么，它只接受一个 key 对象和我们要查询的字段名。

## 写入注册表

如果你最近一直在阅读这篇博客，那么你可能已经看到了用于写入注册表的 _winreg 模块。这只是为您准备的，所以您可以跳过这一部分。我们将从一个实际的例子开始。在下面的代码片段中，我们将设置 Internet Explorer 的主页。一如既往，请注意编辑注册表项可能是危险的。重要提示:在试图编辑注册表之前，请务必备份注册表。现在，继续表演吧！

```py

keyVal = r'Software\Microsoft\Internet Explorer\Main'
try:
    key = OpenKey(HKEY_CURRENT_USER, keyVal, 0, KEY_ALL_ACCESS)
except:
    key = CreateKey(HKEY_CURRENT_USER, keyVal)
SetValueEx(key, "Start Page", 0, REG_SZ, "https://www.blog.pythonlibrary.org/")
CloseKey(key)

```

在上面的代码中，我们尝试打开下面的键:*HKEY _ 当前 _ 用户\软件\微软\Internet Explorer\Main* 并将“起始页”的值设置为这个博客。如果打开失败，通常是因为键不存在，所以我们尝试在异常处理程序中创建键。然后，我们使用 SetValueEx 来实际设置值，像所有优秀的程序员一样，我们完成后进行清理并关闭键。如果您跳过了 CloseKey 命令，在这种情况下您会很好，因为脚本已经完成，Python 会为您完成。但是，如果您继续使用这个键，您可能会有访问冲突，因为它已经打开了。因此，教训是当你编辑完一个键时，总是要关闭它。

## 其他 _winreg 方法

在 _winreg 库中还有其他几个方法值得指出。当您需要删除一个键时，DeleteKey 方法非常方便。不幸的是，我有时需要递归地删除键，比如卸载出错，而 _winreg 没有内置的方法来删除键。当然，你可以自己写，或者你可以下载一个像 [YARW](http://code.activestate.com/recipes/476229-yarw-yet-another-registry-wrapper/) (另一个注册表包装器)这样的包装器来帮你写。

DeleteValue 类似于 DeleteKey，只是您只删除一个值。是的，很明显。如果您想编写自己的递归键删除代码，那么您可能想看看 EnumKey 和 EnumValue，因为它们分别枚举键和值。让我们快速看一下如何使用 EnumKey:

```py

from _winreg import EnumKey, HKEY_USERS

try:
    i = 0
    while True:
        subkey = EnumKey(HKEY_USERS, i)
        print subkey
        i += 1
except WindowsError:
    # WindowsError: [Errno 259] No more data is available    
    pass

```

上面的代码将遍历 HKEY _ 用户配置单元，将子项输出到标准输出，直到到达配置单元的末尾，并引发 WindowsError。当然，这不会下降到子项，但我将把它作为一个练习留给读者去解决。

我们在这里要讨论的最后一个方法是 ConnectRegistry。如果我们需要编辑远程机器的注册表，这是很有帮助的。它只接受两个参数:计算机名和要连接的密钥(例如 HKEY_LOCAL_MACHINE 或类似的)。请注意，当连接到远程机器时，您只能编辑某些键，而其他键不可用。

## 包扎

我希望这能对你有所帮助，并为你未来的项目提供许多好的想法。我有许多使用这个奇妙库的登录脚本和一些使用 YARW 的脚本。到目前为止，它非常有用，我希望它对你也一样。

**更新(2011 年 10 月 14 日):**我们现在在卡里森·加尔帝诺的[博客上有这篇文章的巴西葡萄牙语翻译](http://www.carlissongaldino.com.br/post/editando-o-registro-do-windows-em-python-com-o-winreg)

## 进一步阅读

*   [winreg 的官方文档](http://docs.python.org/library/_winreg.html)
*   [Effbot 教程 on _winreg](http://effbot.org/librarybook/winreg.htm)