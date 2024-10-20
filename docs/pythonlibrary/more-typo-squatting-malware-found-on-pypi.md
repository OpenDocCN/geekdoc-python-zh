# PyPI 上发现更多域名抢注恶意软件

> 原文：<https://www.blog.pythonlibrary.org/2018/10/31/more-typo-squatting-malware-found-on-pypi/>

最近在针对 Windows 用户的 Python 打包索引上发现了恶意软件。该软件包被称为 **colourama** ，如果它被安装，最终会在你的电脑上安装恶意软件。基本上是希望你会拼错流行的[colorama](https://pypi.org/project/colorama/)package。

你可以在[媒体](https://medium.com/@bertusk/cryptocurrency-clipboard-hijacker-discovered-in-pypi-repository-b66b8a534a8)上阅读关于该恶意软件的更多信息，该媒体将该恶意软件描述为“加密货币剪贴板劫持者”。

实际上，去年当斯洛伐克国家安全局在 Python 打包索引中发现了几个恶意库时，我也写过这个问题。

本周，我注意到 Python 软件基金会正在考虑在 2019 年给 PyPI 增加安全性，他们在他们的[博客](http://pyfound.blogspot.com/2018/10/pypi-security-and-accessibility-q1-2019.html)上宣布了这一消息，尽管现在似乎还没有说会增加什么样的安全性。