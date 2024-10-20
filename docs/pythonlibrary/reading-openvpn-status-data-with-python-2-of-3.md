# 使用 Python 读取 OpenVPN 状态数据(第 2 页，共 3 页)

> 原文：<https://www.blog.pythonlibrary.org/2008/04/05/reading-openvpn-status-data-with-python-2-of-3/>

这是关于使用 wxPython + PyWin32 从 Windows 上的 OpenVPN 会话获取输出的 3 部分系列文章的第 2 部分。在本文中，我将展示如何用 Python 启动 OpenVPN，以及如何观察 OpenVPN 向其写入数据日志的文件。

如果上次不在，你需要去蒂姆·戈尔登[网站](http://timgolden.me.uk/python/win32_how_do_i/watch_directory_for_changes.html)下载 watch_directory.py 文件。一旦你得到了它，在你最喜欢的文本编辑器中打开文件，并复制如下内容:

```py

# watch_directory code
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

```

Golden 的网站详细解释了该代码，所以我不打算赘述，但该代码的基本要点是，它监视对目录的特定更改，然后返回被更改的文件的名称和更改的类型。默认情况下，OpenVPN 将其日志文件写入以下位置:“C:\ Program Files \ OpenVPN \ log \ mcisvpn . log”。每当我的程序被警告日志目录中有任何变化时，它打开“mcisvpn.log”文件，读取它，然后将新数据附加到文本小部件。

因为我只想从文件中读取新数据，所以我必须跟踪我在文件中的位置。下面是一个函数，它展示了我如何跟踪文件的变化，以及我如何跟踪脚本在文件中读取的最后一个位置。此外，这个“watcher”脚本在一个单独的线程中运行，以保持 wxPython GUI 的响应性。

```py

def _resultProducer(self, jobID, abortEvent):
    """
    GUI will freeze if this method is not called in separate thread.
    """
    PATH_TO_WATCH = [r'C:\Program Files\OpenVPN\log']
    try: path_to_watch = sys.argv[1].split (",") or PATH_TO_WATCH
    except: path_to_watch = PATH_TO_WATCH
    path_to_watch = [os.path.abspath (p) for p in path_to_watch]

    print "Watching %s at %s" % (", ".join (path_to_watch), time.asctime ())
    # create a Queue object that is updated with the file(s) that is/are changed               
    files_changed = Queue.Queue ()
    for p in path_to_watch:
        Watcher (p, files_changed)

    filepath = os.path.join(PATH_TO_WATCH[0], 'mcisvpn.log')    
    f = open(filepath)
    for line in f.readlines():
        print line

    # get the last position before closing the file
    last_pos = f.tell()
    f.close()

    while not abortEvent():
        try:
            file_type, filename, action = files_changed.get_nowait ()
            # if the change was an update, seek to the last position read
            # and read the update
            if action == 'Updated':
                f = open(filepath)
                f.seek(last_pos)
                for line in f.readlines():
                    if line != '\n':
                        print line
                        f.close()
                # get the last position before closing the file
                last_pos = f.tell()
                f.close()
        except Queue.Empty:
            pass
        time.sleep (1)

    return jobID

```

上面的代码还包括来自标题为“DelayedResult”的 wxPython 演示的一些片段。它负责为我启动和停止线程，虽然我不能完全肯定它干净利落地杀死了 Golden 的 Watcher 类。我只是不知道如何确定；但似乎很管用。

下一次我将展示我的 wxPython 代码，这样您就可以看到如何将这些不同的部分集成在一起。

**你可以在下面下载我的代码:**

*   [vpn.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2008/04/vpn.zip)
*   【t0 VPN . tar】t1