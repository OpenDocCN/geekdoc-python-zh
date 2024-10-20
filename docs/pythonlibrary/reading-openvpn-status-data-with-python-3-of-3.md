# 使用 Python 读取 OpenVPN 状态数据(第 3 页，共 3 页)

> 原文：<https://www.blog.pythonlibrary.org/2008/04/08/reading-openvpn-status-data-with-python-3-of-3/>

在前两篇文章中，我们一直在讨论如何将 OpenVPN 与 Python 结合使用。在这最后一篇文章中，我将展示如何用一些 wxPython 代码将它们整合到一个 GUI 中。我还将讨论一些重要的片段。

第一个需要注意的代码片段是在 **run** 方法中。当我们运行程序时，我们需要确保 OpenVPN 服务正在运行，否则日志文件将不会被更新。所以你会注意到使用了 win32serviceutil 的 StartService 方法。我们将它放在一个 **try** 语句中，以捕捉如果 OpenVPN 服务已经在运行、没有找到或无法启动时可能发生的错误。通常情况下，你不应该使用 bare，因为它会掩盖程序中的其他错误；然而，我无法找到合适的错误代码来使用，所以我将留给读者。

```py

def run(self):
    """
    Run the openvpn.exe script
    """
    vpnname='MCISVPN'
    configfile='mcisvpn.conf'
    defaultgw=''
    vpnserver=''
    vpnserverip = ''

    print 'Starting OpenVPN Service...',
    try:
        win32serviceutil.StartService('OpenVPN Service', None)
    except Exception, e:
        print e
    print 'success!'

    delayedresult.startWorker(self._resultConsumer, self._resultProducer, 
                              wargs=(self.jobID,self.abortEvent), jobID=self.jobID)

```

在尝试启动 OpenVPN 服务之后，我使用 wxPython 提供的线程模型来运行我上次提到的 Golden 的 watcher.py 代码，并在日志文件中跟踪我的位置。

以下是完整的主要 GUI 代码:

```py

from vpnTBIcon import VPNIconCtrl

import os
import sys
import Queue
import threading
import time
import win32file
import win32con
import win32serviceutil
import wx
import wx.lib.delayedresult as delayedresult

ACTIONS = {
  1 : "Created",
  2 : "Deleted",
  3 : "Updated",
  4 : "Renamed to something",
  5 : "Renamed from something"
}

def watch_path (path_to_watch, include_subdirectories=False):
    FILE_LIST_DIRECTORY = 0x0001
    hDir = win32file.CreateFile (
        path_to_watch,
        FILE_LIST_DIRECTORY,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
        None,
        win32con.OPEN_EXISTING,
        win32con.FILE_FLAG_BACKUP_SEMANTICS,
        None
        )
    while 1:
        results = win32file.ReadDirectoryChangesW (
            hDir,
            1024,
            include_subdirectories,
            win32con.FILE_NOTIFY_CHANGE_FILE_NAME |
            win32con.FILE_NOTIFY_CHANGE_DIR_NAME |
            win32con.FILE_NOTIFY_CHANGE_ATTRIBUTES |
            win32con.FILE_NOTIFY_CHANGE_SIZE |
            win32con.FILE_NOTIFY_CHANGE_LAST_WRITE |
            win32con.FILE_NOTIFY_CHANGE_SECURITY,
            None,
            None
            )
        for action, file in results:
            full_filename = os.path.join (path_to_watch, file)
            if not os.path.exists (full_filename):
                file_type = ""
            elif os.path.isdir (full_filename):
                file_type = 'folder'
            else:
                file_type = 'file'
            yield (file_type, full_filename, ACTIONS.get (action, "Unknown"))

class Watcher (threading.Thread):

    def __init__ (self, path_to_watch, results_queue, **kwds):
        threading.Thread.__init__ (self, **kwds)
        self.setDaemon (1)
        self.path_to_watch = path_to_watch
        self.results_queue = results_queue
        self.start ()

    def run (self):
        for result in watch_path (self.path_to_watch):
            self.results_queue.put (result)

# -------------------------------------------------------------------------
# GUI Code starts here

class vpnGUI(wx.App):
    """
    wx application that wraps jrb's vpn script to allow it
    to run in the system tray
    """
    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)

        self.frame = wx.Frame(None, wx.ID_ANY, title='Voicemail', size=(800,500) )

        self.panel = wx.Panel(self.frame, wx.ID_ANY)

        self.abortEvent = delayedresult.AbortEvent()

        # Set defaults
        # ----------------------------------------------------------------------------------------
        self.jobID = 0

        # Create widget controls
        # ----------------------------------------------------------------------------------------
        # redirect stdout
        self.log = wx.TextCtrl(self.panel, wx.ID_ANY, size=(1000,500),
                               style = wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL)
        redir=RedirectText(self.log)
        sys.stdout=redir

        closeBtn = wx.Button(self.panel, wx.ID_ANY, 'Close')

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(self.log, 1, wx.ALL|wx.EXPAND, 5)
        mainSizer.Add(closeBtn, 0, wx.ALL|wx.CENTER, 5)
        self.panel.SetSizer(mainSizer)

        # Bind events
        # ----------------------------------------------------------------------------------------
        self.Bind(wx.EVT_BUTTON, self.onClose, closeBtn)
        self.Bind(wx.EVT_ICONIZE, self.onMinimize)

        # create the system tray icon:
        try:            
            self.tbicon = VPNIconCtrl(self.frame)                        
        except Exception, e:
            print 'Icon creation exception => %s' % e
            self.tbicon = None

        # comment this line out if you don't want to show the
        # GUI when the program is run
        self.frame.Show(True) # make the frame visible

        self.run()

    def run(self):
        """
        Run the openvpn service
        """
        vpnname='MCISVPN'
        configfile='mcisvpn.conf'
        defaultgw=''
        vpnserver=''
        vpnserverip = ''

        print 'Starting OpenVPN Service...',
        try:
            win32serviceutil.StartService('OpenVPN Service', None)
        except Exception, e:
            print e
        print 'success!'

        delayedresult.startWorker(self._resultConsumer, self._resultProducer, 
                                  wargs=(self.jobID,self.abortEvent), jobID=self.jobID)

    def _resultProducer(self, jobID, abortEvent):
        """
        GUI will freeze if this method is not called in separate thread.
        """
        PATH_TO_WATCH = [r'C:\Program Files\OpenVPN\log']
        try: path_to_watch = sys.argv[1].split (",") or PATH_TO_WATCH
        except: path_to_watch = PATH_TO_WATCH
        path_to_watch = [os.path.abspath (p) for p in path_to_watch]

        print "Watching %s at %s" % (", ".join (path_to_watch), time.asctime ())
        files_changed = Queue.Queue ()
        for p in path_to_watch:
            Watcher (p, files_changed)

        filepath = os.path.join(PATH_TO_WATCH[0], 'mcisvpn.log')
        print 'filepath => ' + filepath
        f = open(filepath)
        for line in f.readlines():
            print line
        last_pos = f.tell()
        f.close()

        while not abortEvent():
            try:
                file_type, filename, action = files_changed.get_nowait ()
                if action == 'Updated':
                    print 'Last pos => ', last_pos
                    f = open(filepath)
                    f.seek(last_pos)
                    for line in f.readlines():
                        if line != '\n':
                            print line

                    last_pos = f.tell()
                    f.close()

            except Queue.Empty:
                pass
            time.sleep (1)

        return jobID

    def _resultConsumer(self, delayedResult):
        jobID = delayedResult.getJobID()
        assert jobID == self.jobID
        try:
            result = delayedResult.get()
        except Exception, exc:
            print "Result for job %s raised exception: %s" % (jobID, exc) 
            return

    def onMinimize(self, event):
        """ Minimize to tray """
        self.frame.Hide()

    def onClose(self, event):
        """
        Close the program
        """

        # recover stdout
        sys.stdout=sys.__stdout__

        # stop OpenVPN service
        try:
            print 'Stopping OpenVPN service...'
            win32serviceutil.StopService('OpenVPN Service', None)
        except Exception, e:
            print e        

        # stop the threads
        self.abortEvent.set()
        # remove the icon from the tray
        self.tbicon.Destroy()
        # close the frame
        self.frame.Close()

class RedirectText:
    def __init__(self,textDisplay):
        self.out=textDisplay

    def write(self,string):
        self.out.WriteText(string)

###### Run script! ######
if __name__ == "__main__":
    app = vpnGUI()
    app.MainLoop()

```

您会注意到将 __init__ 中的 stdout 重定向到我们的文本控件小部件。为了写入小部件，我们使用 Python 的打印内置。我们在 **onClose** 事件处理程序中重置了 stdout。该处理程序还会停止 OpenVPN 服务，销毁系统托盘图标并关闭程序。

这就是真正的意义所在。下面有一些链接，供那些想更深入了解这些工具的人使用。

### 资源

*   [Python](http://www.python.org)
*   [wxPython 维基/食谱](http://wiki.wxpython.org)
*   [PyWin32 文档](http://aspn.activestate.com/ASPN/docs/ActivePython/2.5/PyWin32/PyWin32.html)

### 这些示例的来源

*   [vpn.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2008/04/vpn.zip)
*   【t0 VPN . tar】t1