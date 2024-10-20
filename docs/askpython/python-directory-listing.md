# Python 目录列表

> 原文：<https://www.askpython.com/python/examples/python-directory-listing>

在本文中，我们将了解如何执行 Python 目录列表。这将允许我们列出当前工作位置的所有文件和目录。

经常。我们可能希望使用 Python 快速查看文件名并获取信息。

让我们来看看如何快速轻松地做到这一点！

* * *

## 1.使用 os.listdir()的 Python 目录列表

这是一个从你的当前目录中执行 Python 目录列表的简单而有趣的方法！

真的只是一句台词。不相信我？这里有一个例子。这适用于任何操作系统，无论是 Windows / Linux / MacOS。

```py
import os

print(os.listdir())

```

**示例输出**

```py
>>> import os
>>> os.listdir()
['.bashrc', '.git', '.nvimrc', '.vimrc', '.xinitrc', '.zshrc', 'Autumn.jpg', 'README.md', 'config']

```

这将从当前目录返回所有文件和嵌套文件夹的列表。

如果您想指定一个确切的路径，您可以简单地将它作为一个参数传递给`os.listdir(path)`！

```py
>>> os.listdir(r'/home/vijay/manjaro-dotfiles')
['.bashrc', '.git', '.nvimrc', '.vimrc', '.xinitrc', '.zshrc', 'Autumn.jpg', 'README.md', 'config']

```

在处理路径时使用**原始字符串**(前缀为`r`的字符串)，因为你不需要转义任何反斜杠(对于 Windows 路径)。

## 2.将 os.path.join()与 os.listdir()一起使用

如果您想打印当前目录中所有文件的绝对路径，只需在`os.listdir()`函数中添加一个`os.path.join()`！

我们将为此创建一个函数，它简单地获取完整路径，并返回所有此类名称的列表。

```py
import os

def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

print(list_full_paths(r'/home/accornition/manjaro-dotfiles'))

```

**输出**

```py
['/home/vijay/manjaro-dotfiles/.bashrc', '/home/vijay/manjaro-dotfiles/.git', '/home/vijay/manjaro-dotfiles/.nvimrc' , '/home/vijay/manjaro-dotfiles/.vimrc', '/home/vijay/manjaro-dotfiles/.xinitrc', '/home/vijay/manjaro-dotfiles/.zsh    rc', '/home/vijay/manjaro-dotfiles/Autumn.jpg', '/home/vijay/manjaro-dotfiles/README.md', '/home/vijay/manjaro-dotfiles/config'] 

```

的确，这给了我们从根目录开始的绝对路径！

## 3.使用 os.walk()的 Python 目录列表

我们还可以使用`os.walk()`函数遍历目录树。

然后，我们可以分别打印目录和文件。

```py
for top, dirs, files in os.walk(os.getcwd()):
    print("Printing directories...")
    for dir in dirs:
        print(os.path.join(top, dir))
    print("Printing files....")
    for file in files:
        print(os.path.join(top, file))

```

**输出**

```py
Printing directories...
/home/vijay/manjaro-dotfiles/config/cmus                                                                            /home/vijay/manjaro-dotfiles/config/compton                                                                         /home/vijay/manjaro-dotfiles/config/termite                                                                           Printing files....
Printing directories...
Printing files....                                                                                                   /home/vijay/manjaro-dotfiles/config/cmus/my.theme                                                                    Printing directories...
Printing files....
/home/vijay/manjaro-dotfiles/config/compton/compton.conf                                                             Printing directories...
Printing files....
/home/vijay/manjaro-dotfiles/config/termite/config 

```

根据您的用例场景，您可以使用以上三种方法中的任何一种。

第一种方法是最简单的，也是推荐的方法，但是如果您想要完整的路径，并且想要递归地遍历，那么使用`os.walk()`。

* * *

## 结论

在本文中，我们学习了如何使用不同的方法在 Python 中列出文件和目录。

## 参考

*   关于从目录中列出内容的 StackOverflow 问题

* * *