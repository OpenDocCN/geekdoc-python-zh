# 在 Python 中使用 web 浏览器

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-webbrowser>

Python 中的 webbrowser 模块提供了一个界面来显示基于 Web 的文档。

### 网络浏览器

在大多数情况下，只需从这个模块调用 open()函数就可以了。

在 Unix 中，图形浏览器是 X11 下的首选，但是如果图形浏览器不可用或者 X11 显示不可用，将使用文本模式浏览器。

如果使用文本模式浏览器，调用进程将一直阻塞，直到用户退出浏览器。

### 用法示例

```py
 webbrowser.open_new(url)
    Open url in a new window of the default browser, if possible, otherwise,
    open url in the only browser window.

webbrowser.open_new_tab(url)
    Open url in a new page (“tab”) of the default browser, if possible, 
    otherwise equivalent to open_new(). 
```

### Webbrowser 脚本

这个例子将要求用户输入一个搜索词。一个新的浏览器标签将会打开，并在谷歌的搜索栏中输入搜索词。

```py
import webbrowser
google = raw_input('Google search:')
webbrowser.open_new_tab('http://www.google.com/search?btnG=1&q=%s' % google) 
```

脚本 webbrowser 可以用作模块的命令行界面。它接受一个 URL 作为参数。

它接受以下可选参数:

*   -n 在新的浏览器窗口中打开 URL
*   -t 在新的浏览器页面(“选项卡”)中打开 URL