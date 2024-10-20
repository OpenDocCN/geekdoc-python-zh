# 在 Python 包索引(PyPI)上发现恶意库

> 原文：<https://www.blog.pythonlibrary.org/2017/09/15/malicious-libraries-found-on-python-package-index-pypi/>

在 Python 包索引(PyPI)上发现了恶意代码，这是共享 Python 包的最流行的位置。这是斯洛伐克国家安全办公室报告的，然后被其他地方的[哔哔声计算机](https://www.bleepingcomputer.com/news/security/ten-malicious-libraries-found-on-pypi-python-package-index/)接收到(即 [Reddit](https://www.reddit.com/r/Python/comments/709vch/psa_malicious_software_libraries_in_the_official/) )。攻击载体使用了域名仿冒，这基本上是有人上传了一个流行包的名称拼写错误的包，例如 **lmxl** 而不是 **lxml** 。

你可以在这里看到斯洛伐克国家安全办公室的原始报告:[http://www.nbu.gov.sk/skcsirt-sa-20170909-pypi/](http://www.nbu.gov.sk/skcsirt-sa-20170909-pypi/)

去年八月，我在这篇[博客文章](http://incolumitas.com/2016/06/08/typosquatting-package-managers/)中看到了关于这个向量的讨论，很多人似乎对此并不重视。有趣的是，现在人们对这个问题越来越感兴趣。

这也让我想起了关于一家名为 Kite 的初创公司的[争议](https://theoutline.com/post/1953/how-a-vc-funded-company-is-undermining-the-open-source-community)，该公司基本上将广告软件/间谍软件插入到插件中，如 Atom、autocomplete-python 等。

用 Python 打包需要一些帮助。我喜欢现在比 10 年前好得多，但仍有许多问题。