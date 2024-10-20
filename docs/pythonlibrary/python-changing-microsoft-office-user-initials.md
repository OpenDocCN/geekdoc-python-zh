# Python:更改 Microsoft Office 用户姓名缩写

> 原文：<https://www.blog.pythonlibrary.org/2010/10/27/python-changing-microsoft-office-user-initials/>

几个月前，在工作中，我们收到一份报告，称一个文件被锁定。出现的对话框显示了一个已经不再为我们工作的用户的姓名首字母。因此，我们发现了一个可能在 Office 中突然出现的令人讨厌的 bug。基本上，在第一次运行相应的应用程序时，Word 或 Excel 会要求用户输入他们的姓名和首字母，无论以后谁登录该机器，它都会保留这些数据。当我们得到这类错误消息时，这会导致一些严重的混乱。无论如何，让我们快速地看一下如何完成这项工作。

我们将使用 Python 的 **_winreg** 模块进行这次攻击。你可以看到下面说的黑客:

```py

from _winreg import *

key = CreateKey(HKEY_CURRENT_USER,
                r'Software\Microsoft\Office\11.0\Common\UserInfo')
res = QueryValueEx(key, "UserInitials")
print repr (res) 

username = u"mldr\0"
SetValueEx(key, "UserInitials", 0, REG_BINARY, username)
CloseKey(key)

```

这里我们使用了 **CreateKey** 方法，以防这个键还不存在。如果密钥确实存在，那么 CreateKey 将会打开它。脚本的前半部分用于检查键中是否有正确的值。最后三行用我的姓名首字母覆盖了该值。我不记得为什么我必须制作一个 unicode 字符串，但是 PyWin32 上的人告诉我这是实现它的方法。我可以告诉你，我从来没有能够得到一个简单的字符串工作。一旦设置了值，我们就通过关闭键来进行清理。

就是这样！简单，嗯？玩 Python 开心点！