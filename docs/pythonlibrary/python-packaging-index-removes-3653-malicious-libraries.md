# Python 打包索引删除了 3，653 个恶意库

> 原文：<https://www.blog.pythonlibrary.org/2021/03/03/python-packaging-index-removes-3653-malicious-libraries/>

Python 打包索引(PyPI)再次受到恶意库的攻击。事实上超过 3500 人。你可以在 [The Register](https://www.theregister.com/2021/03/02/python_pypi_purges/) 或 [Sonatype 博客](https://blog.sonatype.com/pypi-and-npm-flooded-with-over-5000-dependency-confusion-copycats)上了解更多信息。PyPI 的管理员很快删除了这些库，并将人们安装它们的风险降至最低。

从积极的一面来看，这些图书馆似乎大多向东京的 IP 发出良性 GET 请求。他们还设法淹没了国家预防机制的包装网站。

我见过的唯一一个被报道的特定恶意包是 CuPy 的变种，这是一个 Python 包，使用 Nvidia 的并行计算平台 NumPy。

虽然这可能是试图警告开发者他们供应链中的弱点，但过去在 PyPI 上已经发生了几起[其他](https://www.blog.pythonlibrary.org/2017/09/15/malicious-libraries-found-on-python-package-index-pypi/)T2 域名仿冒事件，这些事件更加阴险。

和往常一样，在使用 pip 时，请确保您了解您要安装的内容。您有责任确保您下载并安装了正确的软件包。