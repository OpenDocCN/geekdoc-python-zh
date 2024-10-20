# PyPI 上发现两个新的域名仿冒库

> 原文：<https://www.blog.pythonlibrary.org/2019/12/04/two-new-typosquatting-libraries-found-on-pypi/>

根据 [ZDNet](https://www.zdnet.com/article/two-malicious-python-libraries-removed-from-pypi/) ，在 Python 打包索引(PyPI)上发现了两个新的恶意包，旨在窃取 GPG 和 SSH 密钥。这些包被命名为 [python3-dateutil](https://pypi.org/project/python3-dateutil/) 和 [jeIlyfish](https://pypi.org/project/jeIlyfish/) ，其中第一个“L”实际上是一个 I。这两个库分别模仿了 [dateutil](https://pypi.org/project/python-dateutil/) 和[水母](https://pypi.org/project/jellyfish/)包。

假冒的 python3-dateutil 将导入假冒的 jeIlyfish 库，该库包含试图窃取 GPG 和 SSH 密钥的恶意代码。虽然这两个库都已经从 PyPI 中删除了，但这只是提醒您要始终确保安装正确的包。

要了解完整的细节，请查看 ZDNet 的文章，因为它分析了这些库是如何工作的。

### 相关阅读

*   发现针对 Linux 的新恶意 Python 库
*   在 [Python 包索引(PyPI)](https://www.blog.pythonlibrary.org/2017/09/15/malicious-libraries-found-on-python-package-index-pypi/) 上发现恶意库