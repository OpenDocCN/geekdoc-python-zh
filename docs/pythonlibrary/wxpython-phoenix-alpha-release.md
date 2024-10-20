# wxPython Phoenix Alpha 版本

> 原文：<https://www.blog.pythonlibrary.org/2017/04/17/wxpython-phoenix-alpha-release/>

wxPython 项目在周末发布了一个重要公告，向 Python 打包索引(PyPI)发布了新的 wxPython“Phoenix”包的 alpha 版本。wxPython 是一个主要的用于 Python 的跨平台桌面图形用户界面工具包。它包装了 wxWidgets，是 PyQt 的主要竞争对手之一。wxPython 的所有新版本都将在未来发布到 PyPI。您可以在此处直接获得副本:

*   [https://pypi.python.org/pypi/wxPython/4.0.0a1](https://pypi.python.org/pypi/wxPython/4.0.0a1)

还应该注意的是，wxPython 现在以 Python wheel 和 tarball 的形式发布。这意味着您现在可以安装带有 pip 的 wxPython:

`pip install wxPython`

如果您想保持领先地位并使用每日快照构建，那么您可以执行以下操作:

`pip install --pre --find-links http://wxpython.org/Phoenix/snapshot-builds/ wxPython`

我已经使用 wxPython 的 Phoenix 版本一年多了，到目前为止，它工作得非常好！你可以在这里阅读更多关于它和 Classic 的区别:

*   [经典 vs 凤凰](https://wxpython.org/Phoenix/docs/html/classic_vs_phoenix.html)
*   wxPython Phoenix [迁移指南](https://wxpython.org/Phoenix/docs/html/MigrationGuide.html)