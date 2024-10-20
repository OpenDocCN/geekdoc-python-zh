# wxPython:如何将 Python 的日志模块重定向到 TextCtrl

> 原文：<https://www.blog.pythonlibrary.org/2013/08/09/wxpython-how-to-redirect-pythons-logging-module-to-a-textctrl/>

今天，我正在阅读 wxPython Google group /邮件列表，有人[询问](https://groups.google.com/forum/?fromgroups=#!topic/wxpython-users/w6n4odRjino)如何让 Python 的日志模块将其输出写入文件和 TextCtrl。事实证明，您需要创建一个定制的日志处理程序来完成这项工作。起初，我尝试使用普通的 StreamHandler，并通过 sys 模块(sys.stdout)将 stdout 重定向到我的文本控件，但这只能重定向我的打印语句，而不能重定向日志消息。

让我们看看我最后得到了什么:

```py

import logging
import logging.config
import wx

########################################################################
class CustomConsoleHandler(logging.StreamHandler):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, textctrl):
        """"""
        logging.StreamHandler.__init__(self)
        self.textctrl = textctrl

    #----------------------------------------------------------------------
    def emit(self, record):
        """Constructor"""
        msg = self.format(record)
        self.textctrl.WriteText(msg + "\n")
        self.flush()

########################################################################
class MyPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.logger = logging.getLogger("wxApp")

        self.logger.info("Test from MyPanel __init__")

        logText = wx.TextCtrl(self,
                              style = wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL)

        btn = wx.Button(self, label="Press Me")
        btn.Bind(wx.EVT_BUTTON, self.onPress)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(logText, 1, wx.EXPAND|wx.ALL, 5)
        sizer.Add(btn, 0, wx.ALL, 5)
        self.SetSizer(sizer)

        txtHandler = CustomConsoleHandler(logText)
        self.logger.addHandler(txtHandler)

    #----------------------------------------------------------------------
    def onPress(self, event):
        """
        """
        self.logger.error("Error Will Robinson!")
        self.logger.info("Informational message")

########################################################################
class MyFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Logging test")
        panel = MyPanel(self)
        self.logger = logging.getLogger("wxApp")
        self.Show()

#----------------------------------------------------------------------
def main():
    """
    """
    dictLogConfig = {
        "version":1,
        "handlers":{
                    "fileHandler":{
                        "class":"logging.FileHandler",
                        "formatter":"myFormatter",
                        "filename":"test.log"
                        },
                    "consoleHandler":{
                        "class":"logging.StreamHandler",
                        "formatter":"myFormatter"
                        }
                    },        
        "loggers":{
            "wxApp":{
                "handlers":["fileHandler", "consoleHandler"],
                "level":"INFO",
                }
            },

        "formatters":{
            "myFormatter":{
                "format":"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            }
        }
    logging.config.dictConfig(dictLogConfig)
    logger = logging.getLogger("wxApp")

    logger.info("This message came from main!")

    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()

if __name__ == "__main__":
    main()

```

您会注意到，我最终使用了 Python 的 logging.config 模块。Python 2.7 中添加了 **dictConfig** 方法，因此如果您没有该方法或更好的方法，那么这段代码就不适合您。基本上，您可以在 dictionary 中设置日志处理程序和格式化程序，然后将它传递给 logging.config。如果您运行这段代码，您会注意到前几条消息会发送到 stdout 和 log，但不会发送到文本控件。在 panel 类的 __init__ 的末尾，我们添加了我们的自定义处理程序，这时开始将日志消息重定向到文本控件。您可以按下按钮来查看它的运行情况！

您可能还想看看下面的一些参考资料。它们有助于更详细地解释我在做什么。

### 相关文章

*   wxPython: [重定向 stdout 和 stderr](https://www.blog.pythonlibrary.org/2009/01/01/wxpython-redirecting-stdout-stderr/)
*   Python 101: [日志介绍](https://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/)
*   Python 日志记录- [如何记录到多个位置](https://www.blog.pythonlibrary.org/2013/07/18/python-logging-how-to-log-to-multiple-locations/)
*   [关于测井模块字典配置的官方文件](http://docs.python.org/2/library/logging.config.html)
*   StackOverflow: [如何编写定制的 Python 日志处理程序](http://stackoverflow.com/questions/3118059/how-to-write-custom-python-logging-handler)