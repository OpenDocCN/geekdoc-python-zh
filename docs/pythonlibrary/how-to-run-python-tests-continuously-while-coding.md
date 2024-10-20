# 如何在编码时“连续”运行 Python 测试

> 原文：<https://www.blog.pythonlibrary.org/2017/03/14/how-to-run-python-tests-continuously-while-coding/>

上周，我做了一些测试驱动开发培训，无意中听到有人提到另一种编程语言，它有一个测试运行程序，你可以设置它来监视你的项目目录，并在文件改变时运行你的测试。我认为这是个好主意。我还认为我可以轻松地编写自己的 Python 脚本来做同样的事情。这是一个相当粗略的版本:

```py

import argparse
import os
import subprocess
import time

def get_args():
    parser = argparse.ArgumentParser(
        description="A File Watcher that executes the specified tests"
        )
    parser.add_argument('--tests', action='store', required=True,
                        help='The path to the test file to run')
    parser.add_argument('--project', action='store', required=False,
                        help='The folder where the project files are')
    return parser.parse_args()

def watcher(test_path, project_path=None):
    if not project_path:
        project_path = os.path.dirname(test_path)

    f_dict = {}

    while True:
        files = os.listdir(project_path)
        for f in files:
            full_path = os.path.join(project_path, f)
            mod_time = os.stat(full_path).st_mtime
            if full_path not in f_dict:
                f_dict[full_path] = mod_time
            elif mod_time != f_dict[full_path]:
                # Run the tests
                cmd = ['python', test_path]
                subprocess.call(cmd)
                print('-' * 70)
                f_dict[full_path] = mod_time

        time.sleep(1)

def main():
    args = get_args()
    w = watcher(args.tests, args.project)

if __name__ == '__main__':
    main()

```

要运行这个脚本，您需要执行如下操作:

```py

python watcher.py --test ~/path/to/tests.py --project ~/project/path

```

现在让我们花点时间来谈谈这个脚本。第一个函数使用 Python 的 **argparse** 模块让程序接受最多两个命令行参数:- test 和- project。第一个是 Python 测试脚本的路径，第二个是要测试的代码所在的文件夹。下一个函数， **watcher** ，将永远循环下去，从传入的文件夹中抓取所有文件，或者使用测试文件所在的文件夹。它将获取每个文件的修改时间，并将其保存到字典中。密钥设置为文件的完整路径，值为修改时间。接下来，我们检查修改时间是否已经更改。如果没有，我们睡一会儿，再检查一遍。如果它改变了，我们就运行测试。

此时，您应该能够在您最喜欢的 Python 编辑器中编辑您的代码和测试，并在终端中观察您的测试运行。

* * *

### 使用看门狗

我四处寻找其他跨平台的监视目录的方法，发现了 **[看门狗](https://pypi.python.org/pypi/watchdog)** 项目。自 2015 年(撰写本文时)以来，它一直没有更新，但我测试了一下，似乎对我来说效果不错。您可以使用 pip 安装看门狗:

```py

pip install watchdog

```

现在我们已经安装了 watchdog，让我们创建一些代码来执行类似于上一个示例的操作:

```py

import argparse
import os
import subprocess
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def get_args():
    parser = argparse.ArgumentParser(
        description="A File Watcher that executes the specified tests"
        )
    parser.add_argument('--tests', action="store", required=True,
                        help='The path to the test file to run')
    parser.add_argument('--project', action='store', required=False,
                        help='The folder where the project files are')
    return parser.parse_args()

class FW(FileSystemEventHandler):
    def __init__(self, test_file_path):
        self.test_file_path = test_file_path

    def on_any_event(self, event):

        if os.path.exists(self.test_file_path):
            cmd = ['python', self.test_file_path]
            subprocess.call(cmd)
            print('-' * 70)

if __name__ =='__main__':
    args = get_args()
    observer = Observer()
    path = args.tests
    watcher = FW(path)

    if not args.project:
        project_path = os.path.dirname(args.tests)
    else:
        project_path = args.project

    if os.path.exists(path) and os.path.isfile(path):
        observer.schedule(watcher, project_path, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    else:
        print('There is something wrong with your test path')

```

在这段代码中，我们保留了我们的 **get_args()** 函数，并添加了一个类。class 子类的看门狗的 **FileSystemEventHandler** 类。我们最终将测试文件路径传递给该类，并覆盖了 **on_any_event()** 方法。此方法在文件系统事件发生时触发。当那发生时，我们运行我们的测试。最后一点在代码的末尾，我们创建了一个 **Observer()** 对象，告诉它监视指定的项目路径，并在文件发生任何变化时调用我们的事件处理程序。

* * *

### 包扎

此时，您应该能够在自己的代码中尝试这些想法了。也有一些特定于平台的方法来监视一个文件夹(比如 [PyWin32](http://timgolden.me.uk/python/win32_how_do_i/watch_directory_for_changes.html) )，但是如果你像我一样在多个操作系统上运行，那么 watchdog 或者 rolling your own 可能是更好的选择。

### 相关阅读

*   如何使用 Python 来观察文件的变化？
*   观察目录[的变化](http://timgolden.me.uk/python/win32_how_do_i/watch_directory_for_changes.html)