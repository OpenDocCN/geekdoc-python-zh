# 将 wxPyMail 移植到 Linux

> 原文：<https://www.blog.pythonlibrary.org/2008/09/26/porting-wxpymail-to-linux/>

将应用程序从一个操作系统移植到另一个操作系统是一个非常耗时的过程。幸运的是，wxPython 消除了这个过程中的痛苦。这只是我第二次把我的代码移植到 Linux 上。我平时上班都是在 Windows XP 上为 Windows XP 写。当我最初为工作编写 wxPyMail 时，我使用 Mark Hammond 的 PyWin32 库来获取用户的全名和用户名，以帮助构建他们的回信地址。以下是我当时使用的代码:

```py

try:
    userid = win32api.GetUserName()
    info = win32net.NetUserGetInfo('server', userid, 2)
    full_name = str(info['full_name'].lower())
    name_parts = full_name.split(' ')
    self.emailuser = name_parts[0][:1] + name_parts[-1]
    email = self.emailuser + '@companyEmail'
except:
    email = ''

```

如果我保留了那个代码，我将需要使用 wx。平台模块并测试“WXMSW”标志，如下所示:

```py

if wx.Platform == '__WXMSW__':  
    try:
        userid = win32api.GetUserName()
        # rest of my win32 code
    except:
        # do something
        pass
else:
    # put some Linux or Mac specific stuff here
    pass

```

这可能是移植代码最简单的方法，尤其是当您只有几个特定于操作系统的代码时。如果你有很多，那么你可能想把它们放入它们自己的模块中，并把它们导入到你的平台上。我相信还有其他方法可以实现这一点。

无论如何，我已经拿出了所有的 win32 的东西，因为它与我的组织。因此，wxPyMail 的新代码实际上在 Linux 上运行得很好。我在 Ubuntu Hardy Heron 上测试过。然而，我注意到一些美学问题。首先，这个框架似乎得到了关注，所以我不能在 wx 中输入电子邮件地址，甚至我的用户名。弹出的对话框。其次，我登录对话框中的标签和文本框在 Ubuntu 中太短了。它们在 Windows 上看起来很好，但在 Ubuntu 上标签被夹住了，当我完整输入用户名时，我看不到它。

因此，我更改了代码，使标签和文本框的大小更长。我还对主应用程序中的*到*字段和登录对话框中的*用户名*文本字段调用了 SetFocus()。除此之外，我做了一点重构，将 SMTP 服务器放在 __init__ 中，这样更容易找到和设置。我还更改了我的各种控件实例，以便它们显式地反映它们的参数。

还要注意文件开头的“shebang”行:“#！/usr/bin/env python”。这告诉 Linux 这个文件可以使用 Python 执行，也告诉 Linux Python 安装文件夹在哪里。只要 python 在您的路径中的某个地方，您就应该能够使用这种方法。要检查路径上是否有 python，请打开命令提示符并键入“python”。如果您获得了 python shell，那么您就可以开始了。

你可能认为我们已经完成了，但是我们忘记了一个重要的部分。我们仍然需要告诉 Linux 使用 wxPyMail 作为 mailto 链接的默认电子邮件程序。我在 [howtogeek](http://www.howtogeek.com) 找到了一篇关于这个话题的[操作文章](http://www.howtogeek.com/howto/ubuntu/set-gmail-as-default-mail-client-in-ubuntu/)。我们需要对它进行一些修改来使它工作，但是一旦你知道怎么做，它实际上是非常容易的。对于这个例子，我使用了 Ubuntu Hardy Heron (8.04)和 wxPython 2.8.9.1 以及 Python 2.5.2。

无论如何，首先你需要下载我的代码。当我第一次编写这个应用程序时，它是在 Windows 上。wxPython 邮件列表上的好心人向我指出我的代码有行尾问题，所以请确保您不会意外地将其转换回 Windows 格式。感谢罗宾·邓恩、克里斯托弗·巴克、科迪·普雷科德、弗兰克·米尔曼和一个叫基思的人。

其次，您需要告诉 Linux 在用户点击 mailto 链接时执行 python 脚本。在 Ubuntu 中，你可以通过进入系统，首选项，首选应用程序。将“邮件阅读器”更改为“自定义”，并在“命令”字段中添加以下内容:

 `/home/USERNAME/path/to/wxPyMail.py %s` 

用您的用户名替换 USERNAME，并根据需要调整路径。确保包含“%s ”,它表示传递给我们脚本的“mailto”字符串。现在，打开命令提示符，将目录更改为放置 python 脚本的位置。您将需要更改它的执行权限，所以应该这样做:

chmod +x wxPyMail.py

现在浏览到一个带有 mailto 链接的网站，并尝试一下吧！和往常一样，如果有任何问题或意见，欢迎发邮件给我，邮箱是 mike [at] pythonlibrary [dot] org。

**下载源码**

*   [wxpymail-Linux.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2008/09/wxpymail.tar)
*   [wxpymail-Linux.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2008/09/wxpymail.zip)

**补充阅读**

*   [道格·赫尔曼在 smtplib 上的 MOTW](http://blog.doughellmann.com/2008/10/pymotw-smtplib.html)
*   [官方 smtplib 文档](http://www.python.org/doc/2.5.2/lib/module-smtplib.html)
*   [官方邮件模块文档](http://www.python.org/doc/2.5.2/lib/module-email.html)