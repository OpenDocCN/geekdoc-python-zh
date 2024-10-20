# wxPython:正在转换 wx。日期时间到 Python 日期时间

> 原文：<https://www.blog.pythonlibrary.org/2014/08/27/wxpython-converting-wx-datetime-python-datetime/>

wxPython GUI 工具包包括自己的日期/时间功能。大多数时候，你只需要使用 Python 的 datetime 和 time 模块就可以了。但是偶尔你会发现自己需要从 wxPython 的 wx 转换过来。datetime 对象转换为 Python 的 DateTime 对象。使用 **wx 时可能会遇到这种情况。DatePickerCtrl** 小部件。

幸运的是，wxPython 的日历模块有一些助手函数，可以帮助您在 wxPython 和 Python 之间来回转换 datetime 对象。让我们来看看:

```py

def _pydate2wxdate(date):
     import datetime
     assert isinstance(date, (datetime.datetime, datetime.date))
     tt = date.timetuple()
     dmy = (tt[2], tt[1]-1, tt[0])
     return wx.DateTimeFromDMY(*dmy)

def _wxdate2pydate(date):
     import datetime
     assert isinstance(date, wx.DateTime)
     if date.IsValid():
          ymd = map(int, date.FormatISODate().split('-'))
          return datetime.date(*ymd)
     else:
          return None

```

您可以在自己的代码中使用这些方便的函数来帮助您进行转换。我可能会将这些放入控制器或实用程序脚本中。我还会稍微重写它，这样我就不会在函数中导入 Python 的 datetime 模块。这里有一个例子:

```py

import datetime
import wx

def pydate2wxdate(date):
     assert isinstance(date, (datetime.datetime, datetime.date))
     tt = date.timetuple()
     dmy = (tt[2], tt[1]-1, tt[0])
     return wx.DateTimeFromDMY(*dmy)

def wxdate2pydate(date):
     assert isinstance(date, wx.DateTime)
     if date.IsValid():
          ymd = map(int, date.FormatISODate().split('-'))
          return datetime.date(*ymd)
     else:
          return None 

```

您可以在这个旧的 wxPython [邮件线程](http://wxpython-users.1045709.n5.nabble.com/wx-DateTime-lt-gt-python-datetime-td2357748.html)上阅读更多关于这个主题的内容。开心快乐编码！