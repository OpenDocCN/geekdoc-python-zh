# wxPython:如何双击 ObjectListView 小部件中的项目

> 原文：<https://www.blog.pythonlibrary.org/2013/02/28/wxpython-how-to-double-click-an-item-in-an-objectlistview-widget/>

本周，我需要弄清楚如何附加一个事件处理程序，当我双击处于 LC_REPORT 模式的 ObjectListView 小部件中的一个项目(即行)时，该事件处理程序将被触发。出于某种原因，没有明显的鼠标事件。有一个 EVT _ 列表 _ 项目 _ 右键单击和一个 EVT _ 列表 _ 项目 _ 中键单击，但是没有任何种类的左键单击。在谷歌上搜索了一会儿后，我发现我可以通过使用 EVT _ 列表 _ 项目 _ 激活来让它工作。这将在双击某项、选择某项并且用户按 ENTER 时触发。下面是一个代码示例:

```py

import wx
from ObjectListView import ObjectListView, ColumnDefn

########################################################################
class Results(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, tin, zip_code, plus4, name, address):
        """Constructor"""
        self.tin = tin
        self.zip_code = zip_code
        self.plus4 = plus4
        self.name = name
        self.address = address

########################################################################
class DCPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        mainSizer = wx.BoxSizer(wx.VERTICAL)

        self.test_data = [Results("123456789", "50158", "0065", "Patti Jones",
                                  "111 Centennial Drive"),
                          Results("978561236", "90056", "7890", "Brian Wilson",
                                  "555 Torque Maui"),
                          Results("456897852", "70014", "6545", "Mike Love", 
                                  "304 Cali Bvld")
                          ]
        self.resultsOlv = ObjectListView(self, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.resultsOlv.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.onDoubleClick)

        self.setResults()

        mainSizer.Add(self.resultsOlv, 1, wx.EXPAND|wx.ALL, 5)
        self.SetSizer(mainSizer)

    #----------------------------------------------------------------------
    def onDoubleClick(self, event):
        """
        When the item is double-clicked or "activated", do something
        """
        print "in onDoubleClick method"

    #----------------------------------------------------------------------
    def setResults(self):
        """"""
        self.resultsOlv.SetColumns([
            ColumnDefn("TIN", "left", 100, "tin"),
            ColumnDefn("Zip", "left", 75, "zip_code"),
            ColumnDefn("+4", "left", 50, "plus4"),
            ColumnDefn("Name", "left", 150, "name"),
            ColumnDefn("Address", "left", 200, "address")
            ])
        self.resultsOlv.CreateCheckStateColumn()
        self.resultsOlv.SetObjects(self.test_data)

########################################################################
class DCFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, parent=None, title="Double-click Tutorial")
        panel = DCPanel(self)

#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = DCFrame()
    frame.Show()
    app.MainLoop()

```

很直接，对吧？如果你需要知道如何做到这一点，我希望它能帮助你。这个方法也应该和 ListCtrl 一起使用。