# PyInstaller 教程——构建二进制系列！

> 原文：<https://www.blog.pythonlibrary.org/2010/08/10/a-pyinstaller-tutorial-build-a-binary-series/>

在我们上一篇关于构建二进制文件的文章中，我们了解了一点 py2exe。这一次，我们将共同关注 PyInstaller 的来龙去脉。我们将使用上一篇[文章](https://www.blog.pythonlibrary.org/2010/07/31/a-py2exe-tutorial-build-a-binary-series/)中相同的蹩脚的 wxPython 脚本作为我们的一个例子，但是我们也将尝试一个普通的控制台脚本来看看有什么不同，如果有的话。如果您不知道的话，PyInstaller 可以在 Linux、Windows 和 Mac(实验性的)上运行，并且可以在 Python 1.5-2.6 上运行(除了在 Windows 上，这里有一个关于 2.6 的警告——见下文)。PyInstaller 支持代码签名(Windows)、eggs、隐藏导入、单个可执行文件、单个目录等等！

## PyInstaller 入门

**关于 Python 2.6 在 Windows 上的注意:**如果你仔细阅读 PyInstaller [网站](http://www.pyinstaller.org/wiki/Python26Win)，你会看到一个关于 Python 2.6+不被完全支持的警告。注意到你现在需要安装微软的 CRT 来运行你的可执行文件。这可能指的是 Python 2.6 相对于 Microsoft Visual Studio 2008 引入的并行程序集/清单问题。我们在第一篇文章中已经提到了这个问题。如果你对此一无所知，请查看 py2exe 网站、wxPython wiki 或 Google。

不管怎样，我们继续表演吧。下载 PyInstaller 后，只需将归档文件解压到方便的地方。遵循这三个简单步骤:

1.  运行 **Configure.py** 将一些基本配置数据保存到 a”。dat”文件。这节省了一些时间，因为 PyInstaller 不必动态地重新计算配置。
2.  在命令行上运行以下命令:`python makespec.py [opts] <scriptname>`其中 scriptname 是您用来运行程序的主 Python 文件的名称。
3.  最后，通过命令行运行下面的命令: *python Build.py specfile* 来构建您的可执行文件。

现在让我们用一个真实的脚本来演示一下。我们将从一个简单的控制台脚本开始，该脚本创建一个伪配置文件。代码如下:

```py

import configobj

#----------------------------------------------------------------------
def createConfig(configFile):
    """
    Create the configuration file
    """
    config = configobj.ConfigObj()
    inifile = configFile
    config.filename = inifile
    config['server'] = "http://www.google.com"
    config['username'] = "mike"
    config['password'] = "dingbat"
    config['update interval'] = 2
    config.write()

#----------------------------------------------------------------------
def getConfig(configFile):
    """
    Open the config file and return a configobj
    """    
    return configobj.ConfigObj(configFile)

def createConfig2(path):
    """
    Create a config file
    """
    config = configobj.ConfigObj()
    config.filename = path
    config["Sony"] = {}
    config["Sony"]["product"] = "Sony PS3"
    config["Sony"]["accessories"] = ['controller', 'eye', 'memory stick']
    config["Sony"]["retail price"] = "$400"
    config.write()

if __name__ == "__main__":
    createConfig2("sampleConfig.ini")

```

现在让我们创建一个规范文件:

 `c:\Python25\python c:\Users\Mike\Desktop\pyinstaller-1.4\Makespec.py config_1.py` 

在我的测试机器上，我安装了 3 个不同的 Python 版本，所以我必须显式地指定 Python 2.5 路径(或者将 Python 2.5 设置为默认路径)。无论如何，这应该会创建一个类似下面的文件(命名为“config_1.spec”):

```py

# -*- mode: python -*-
a = Analysis([os.path.join(HOMEPATH,'support\\_mountzlib.py'), os.path.join(HOMEPATH,'support\\useUnicode.py'), 'config_1.py'],
             pathex=['C:\\Users\\Mike\\Desktop\\py2exe_ex', r'C:\Python26\Lib\site-packages'])
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name=os.path.join('build\\pyi.win32\\config_1', 'config_1.exe'),
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT( exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name=os.path.join('pyInstDist2', 'config_1'))

```

对于我们正在使用的 Python 脚本，我们需要在规格文件的*分析*部分的 *configobj.py* 的位置添加一个显式路径到 *pathex* 参数中。如果您不这样做，当您运行生成的可执行文件时，它将打开和关闭一个控制台窗口非常快，您将无法知道它说什么，除非您从命令行运行 exe。我采用了后者来找出问题所在，并发现它找不到 configobj 模块。您还可以在 COLLECT 函数的 name 参数中指定 exe 的输出路径。在这种情况下，我们将 PyInstaller 的输出放在“pyInstDist2”的“config_1”子文件夹中，该文件夹应该与您的原始脚本放在一起。在配置你的规格文件时有很多选项，你可以在这里阅读。

要基于规范文件构建可执行文件，请在命令行上执行以下操作:

 `c:\Python25\python c:\Users\Mike\Desktop\pyinstaller-1.4\Build.py config_1.spec` 

在我的机器上，我得到了一个文件夹，里面有 25 个文件，总共 6.7 MB。您应该能够使用分析部分的*排除*参数和/或压缩来减小大小。

## PyInstaller 和 wxPython

现在让我们尝试从一个简单的 wxPython 脚本创建一个二进制文件。以下是 Python 脚本:

```py

import wx

########################################################################
class DemoPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)

        labels = ["Name", "Address", "City", "State", "Zip",
                  "Phone", "Email", "Notes"]

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        lbl = wx.StaticText(self, label="Please enter your information here:")
        lbl.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        mainSizer.Add(lbl, 0, wx.ALL, 5)
        for lbl in labels:
            sizer = self.buildControls(lbl)
            mainSizer.Add(sizer, 1, wx.EXPAND)
        self.SetSizer(mainSizer)
        mainSizer.Layout()

    #----------------------------------------------------------------------
    def buildControls(self, label):
        """"""
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        size = (80,40)
        font = wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD)

        lbl = wx.StaticText(self, label=label, size=size)
        lbl.SetFont(font)
        sizer.Add(lbl, 0, wx.ALL|wx.CENTER, 5)
        if label != "Notes":
            txt = wx.TextCtrl(self, name=label)
        else:
            txt = wx.TextCtrl(self, style=wx.TE_MULTILINE, name=label)
        sizer.Add(txt, 1, wx.ALL, 5)
        return sizer

########################################################################
class DemoFrame(wx.Frame):
    """
    Frame that holds all other widgets
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""        
        wx.Frame.__init__(self, None, wx.ID_ANY, 
                          "PyInstaller Tutorial",
                          size=(600,400)
                          )
        panel = DemoPanel(self)        
        self.Show()

#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = DemoFrame()
    app.MainLoop()

```

因为这是一个 GUI，所以我们创建的 spec 文件略有不同:

 `c:\Python25\python c:\Users\Mike\Desktop\pyinstaller-1.4\Makespec.py -F -w sampleApp.py` 

请注意-F 和-w 参数。F 命令告诉 PyInstaller 只创建一个可执行文件，而-w 命令告诉 PyInstaller 隐藏控制台窗口。下面是生成的规范文件:

```py

# -*- mode: python -*-
a = Analysis([os.path.join(HOMEPATH,'support\\_mountzlib.py'), os.path.join(HOMEPATH,'support\\useUnicode.py'), 'sampleApp.py'],
             pathex=['C:\\Users\\Mike\\Desktop\\py2exe_ex'])
pyz = PYZ(a.pure)
exe = EXE( pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name=os.path.join('pyInstDist', 'sampleApp.exe'),
          debug=False,
          strip=False,
          upx=True,
          console=False )

```

请注意，最后一行的“console”参数设置为“False”。如果您像使用控制台脚本那样构建它，那么您应该在“pyInstDist”文件夹中得到一个大约 7.1 MB 大小的文件。

## 包扎

这就结束了我们对 PyInstaller 的快速浏览。我希望这对您的 Python 二进制制作工作有所帮助。PyInstaller 网站上有更多的信息，并且有很好的文档记录，尽管该网站非常简单。一定要试一试，看看 PyInstaller 是多么容易使用！

*注意:我在 Windows 7 家庭高级版(32 位)上使用 PyInstaller 1.4 和 Python 2.5 测试了所有这些。*

## 进一步阅读

*   PyInstaller 官方[网站](http://www.pyinstaller.org)、[手册](http://www.pyinstaller.org/export/latest/tags/1.4/doc/Manual.html?format=raw)、[邮件列表](http://groups-beta.google.com/group/PyInstaller?pli=1)
*   第一篇[文章](https://www.blog.pythonlibrary.org/2010/07/31/a-py2exe-tutorial-build-a-binary-series/)中的“构建二进制系列”