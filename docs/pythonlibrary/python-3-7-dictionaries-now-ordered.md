# Python 3.7 -词典现已订购

> 原文：<https://www.blog.pythonlibrary.org/2018/02/27/python-3-7-dictionaries-now-ordered/>

我的一位读者向我指出，Python 3.7 现在将默认拥有有序字典。你可以在 [Python-Dev 列表](https://mail.python.org/pipermail/python-dev/2017-December/151283.html)上阅读“公告”。

Python 的字典一直是无序的，直到 Python 3.6，它才根据 Raymond Hettinger(Python 的核心开发人员之一)的提议变成有序的。

Python 3.6 的[发行说明](https://docs.python.org/3/whatsnew/3.6.html#new-dict-implementation))说了以下内容:

> 这个新实现的保序方面被认为是一个实现细节，不应该依赖它(这在将来可能会改变，但是在改变语言规范以强制所有当前和将来的 Python 实现的保序语义之前，希望在几个版本的语言中有这个新的 dict 实现；这也有助于保持与旧版本语言的向后兼容性，其中随机迭代顺序仍然有效，例如 Python 3.5)。

现在，当 Python 3.7 发布时，dict 的有序实现将成为标准。我觉得这很棒。