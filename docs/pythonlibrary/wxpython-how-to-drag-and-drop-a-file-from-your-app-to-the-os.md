# wxPython:如何将文件从应用程序拖放到操作系统中

> 原文：<https://www.blog.pythonlibrary.org/2012/08/01/wxpython-how-to-drag-and-drop-a-file-from-your-app-to-the-os/>

今天在 StackOverflow 上，我看到有人想知道如何从 wx 中拖动文件。ListCtrl 放到他们的桌面上或者文件系统中的其他地方。他们使用了来自 [zetcode](http://zetcode.com/wxpython/skeletons/) 的文件管理器框架，但是不知道如何添加 DnD 部分。经过一番搜索和黑客攻击，我基于罗宾·邓恩在[论坛](http://wxpython-users.1045709.n5.nabble.com/Creating-file-DropSource-programatically-td2344103.html)上提到的东西想到了这个。

```py

import wx
import os
import time

########################################################################
class MyListCtrl(wx.ListCtrl):

    #----------------------------------------------------------------------
    def __init__(self, parent, id):
        wx.ListCtrl.__init__(self, parent, id, style=wx.LC_REPORT)

        files = os.listdir('.')

        self.InsertColumn(0, 'Name')
        self.InsertColumn(1, 'Ext')
        self.InsertColumn(2, 'Size', wx.LIST_FORMAT_RIGHT)
        self.InsertColumn(3, 'Modified')

        self.SetColumnWidth(0, 220)
        self.SetColumnWidth(1, 70)
        self.SetColumnWidth(2, 100)
        self.SetColumnWidth(3, 420)

        j = 0
        for i in files:
            (name, ext) = os.path.splitext(i)
            ex = ext[1:]
            size = os.path.getsize(i)
            sec = os.path.getmtime(i)
            self.InsertStringItem(j, "%s%s" % (name, ext))
            self.SetStringItem(j, 1, ex)
            self.SetStringItem(j, 2, str(size) + ' B')
            self.SetStringItem(j, 3, time.strftime('%Y-%m-%d %H:%M', 
                                                   time.localtime(sec)))

            if os.path.isdir(i):
                self.SetItemImage(j, 1)
            elif ex == 'py':
                self.SetItemImage(j, 2)
            elif ex == 'jpg':
                self.SetItemImage(j, 3)
            elif ex == 'pdf':
                self.SetItemImage(j, 4)
            else:
                self.SetItemImage(j, 0)

            if (j % 2) == 0:
                self.SetItemBackgroundColour(j, '#e6f1f5')
            j = j + 1

########################################################################
class FileHunter(wx.Frame):
    #----------------------------------------------------------------------
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, -1, title)
        panel = wx.Panel(self)

        p1 = MyListCtrl(panel, -1)
        p1.Bind(wx.EVT_LIST_BEGIN_DRAG, self.onDrag)
        sizer = wx.BoxSizer()
        sizer.Add(p1, 1, wx.EXPAND)
        panel.SetSizer(sizer)

        self.Center()
        self.Show(True)

    #----------------------------------------------------------------------
    def onDrag(self, event):
        """"""
        data = wx.FileDataObject()
        obj = event.GetEventObject()
        id = event.GetIndex()
        filename = obj.GetItem(id).GetText()
        dirname = os.path.dirname(os.path.abspath(os.listdir(".")[0]))
        fullpath = str(os.path.join(dirname, filename))

        data.AddFile(fullpath)

        dropSource = wx.DropSource(obj)
        dropSource.SetData(data)
        result = dropSource.DoDragDrop()
        print fullpath

#----------------------------------------------------------------------
app = wx.App(False)
FileHunter(None, -1, 'File Hunter')
app.MainLoop()

```

这里有几个要点。首先，您需要绑定到 EVT 列表开始拖动来捕捉适当的事件。然后，在您的处理程序中，您需要创建一个 **wx。FileDataObject** 对象，并使用其 **AddFile** 方法将完整路径附加到其内部文件列表中。根据 wxPython [的文档](http://wxpython.org/Phoenix/docs/html/FileDataObject.html)，AddFile 是 Windows 专用的，但是因为 Robin Dunn(wxPython 的创建者)推荐了[这个方法](http://wxpython-users.1045709.n5.nabble.com/Creating-file-DropSource-programatically-td2344103.html)，我就用了它。可能是文档有误。无论如何，我们还需要定义 DropSource 并调用它的 **DoDragDrop** 方法，这样就完成了。这段代码在 Windows 7、Python 2.6.6 和 wxPython 2.8.12.1 上对我来说是有效的。