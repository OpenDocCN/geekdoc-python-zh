# 启动 Django 开源项目

> 原文：<https://realpython.com/kickstarting-a-django-open-source-project/>

这是 Patrick Altman 的客座博文，Patrick Altman 是一个热情的开源黑客，也是 T2 Eldarion T3 的工程副总裁。

* * *

在本文中，我们将剖析一个 Django 应用程序样板，它旨在快速开发遵循广泛共享约定的开源项目。

我们将看到的布局(或模式)基于 [django-stripe-payments](https://github.com/eldarion/django-stripe-payments) 。对于由 [Eldarion](http://github.com/eldarion/) 和 [Pinax](http://github.com/pinax/) 发布的 100 多个不同的开源项目来说，这种布局已经被证明是成功的。

当您阅读本文时，请记住您的特定项目可能与此模式不同；然而，有许多项目在 Python 项目中相当常见，一般来说，在某种程度上在开源项目中也是如此。我们将集中讨论这些项目。

## 项目布局

一个好的项目布局可以帮助一个新用户浏览你的源代码。有一些被普遍接受的约定，遵守这些约定也很重要。此外，良好的项目布局有助于打包。

项目布局会有一点不同，这取决于它们是 [Python 包](https://realpython.com/python-modules-packages/)(像一个可重用的 Django 应用)还是类似 Django 项目的东西。使用我们的示例项目，我们想要强调布局的某些方面以及项目顶层中包含的文件。

您应该为描述项目各个方面的元数据文件保留项目的根，例如`LICENSE`、`CONTRIBUTING.md`、`README.rst`，以及用于[运行测试](https://realpython.com/python-testing/)和打包的任何[脚本](https://realpython.com/run-python-scripts/)。此外，在这个根级别中应该有一个文件夹，其名称与您希望的 Python 包名称相同。在我们的`django-stripe-payments`示例中，它是`payments`。最后，您应该将您的文档作为一个基于 Sphinx 的项目存储在一个名为`docs`的文件夹中。

[*Remove ads*](/account/join/)

### 许可

如果你的目标是尽可能广泛地采用，通常最好以许可的 MIT 或 BSD 许可来许可你的软件。采用得越多，在各种真实世界环境中的暴露就越多，这增加了通过拉式请求获得反馈和合作的机会。将许可证的内容存储在项目根目录下的一个`LICENSE`文件中。

### README

每个项目都应该在项目根目录下有一个`README.rst`文件。这个文档应该向用户简要介绍这个项目，描述它解决了什么问题，并提供一个快速入门指南。

将其命名为`README.rst`,放在你的 repo 的根目录下，GitHub 会将它显示在你的主项目页面上，供潜在用户查看和浏览，以快速感受你的软件会如何帮助他们。

推荐使用 readme 中的 [reStructuredText](http://docutils.sourceforge.net/rst.html) 而不是 [Markdown](http://daringfireball.net/projects/markdown/) ，这样如果你发布你的包，它会很好地显示在 [PyPI](https://pypi.python.org/pypi/) 上。

### 投稿指南

一个`CONTRIBUTING.md`文件讨论了代码风格指南、过程，以及那些希望通过对你的项目的拉请求来贡献代码的人们的指南。这有助于降低希望贡献代码的人的门槛。第一次投稿的人可能会对做错或违反惯例感到紧张，这份文件越详细，他们就越能检查自己，而不必问他们可能太害羞而不敢问的问题。

### setup.py

好的包装有助于你的项目的发行。通过编写一个`setup.py`脚本，你可以利用 Python 的打包工具[在 PyPI](https://realpython.com/pypi-publish-python-package/) 上创建并发布你的项目。

这是一个非常简单的脚本。例如，django-stripe-payments 的核心脚本如下:

```py
PACKAGE = "payments"
NAME = "django-stripe-payments"
DESCRIPTION = "a payments Django app for Stripe"
AUTHOR = "Patrick Altman"
AUTHOR_EMAIL = "paltman@eldarion.com"
URL = "https://github.com/eldarion/django-stripe-payments"
VERSION = __import__(PACKAGE).__version__

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read("README.rst"),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="BSD",
    url=URL,
    packages=find_packages(exclude=["tests.*", "tests"]),
    package_data=find_package_data(PACKAGE, only_in_packages=False),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Framework :: Django",
        ],
    install_requires=[
        "django-jsonfield>=0.8",
        "stripe>=1.7.9",
        "django>=1.4",
        "pytz"
        ],
    zip_safe=False
)
```

这里发生了几件事。还记得我们如何讨论制作`README.rst`文件重组文本吗？

这是因为正如你所看到的,`long_description`我们使用该文件的内容来填充 PyPI 上的登录页面，这是那里使用的标记语言。分类器是一组元数据，有助于将您的项目放在 PyPI 上正确的类别中。

最后，`install_requires`参数将确保当你的包被安装时，这些列出的依赖项被安装或者已经被安装。

## GitHub

如果你的项目不在 [GitHub](http://github.com) 上，你真的错过了。当然，也有其他基于网络的 DVCS(分布式版本控制系统)网站提供免费的开源托管，但没有一个网站比 GitHub 为开源做得更多。

### 处理拉取请求

构建一个伟大的开源项目的一部分就是让它超越你自己。这不仅包括增加用户基础，还包括贡献者基础。 [GitHub(和一般的 git)](https://realpython.com/python-git-github-intro/)真正改变了这种方式。

增加贡献者的一个关键是在管理拉取请求时做出响应。这并不意味着接受每一份贡献，但也要保持开放的心态，并以尊重的态度处理回应，就像你为另一个项目做出贡献时会得到的尊重一样。

不要只是关闭你不想要的请求，而是花时间解释你为什么不接受它们，或者如果可能的话，解释如何改进它们以便它们可以被排除在外。如果改进很小，或者你可以自己改进，那就接受它，然后做出你喜欢的修改。要求改进和自己动手做是有区别的。

指导原则是创造一个欢迎和感恩的氛围。记住你的贡献者是在自愿贡献他们的时间和精力来改进你的项目。

[*Remove ads*](/account/join/)

### 版本控制、分支和发布

创建版本时，阅读并遵循[语义版本](http://semver.org)。

当你发布主要版本时，一定要清楚地记录向后不兼容的变更。如果您的文档在提交时发生更改，最简单的方法是在两个版本之间更新更改日志文件。这个文件可以是项目根目录下的一个 CHANGELOG 文件，或者是像`docs/changelog.rst`这样的文件中的文档的一部分。这将使你不费吹灰之力就能创建好的发行说明。

保持主人稳定。人们总是有机会使用主版本的代码，而不是软件包版本。为工作创建特性分支，并在经过测试且相对稳定时进行合并。

## 文档

在有一定数量的文档之前，没有一个项目是完整的。好的文档让用户不必阅读源代码来决定如何使用你的软件。好的文档传达出你关心你的用户。

通过[阅读文档](https://readthedocs.org)，你可以让你的文档自动呈现并免费托管。它会在每次提交给 master 时自动更新，这真是太酷了。

为了使用 Read Docs，您应该在项目根目录下的`docs`文件夹中创建一个基于 [Sphinx](http://sphinx-doc.org/) 的文档项目。这真的是一件非常简单的事情，由一个`Makefile`和一个`conf.py`文件组成，然后是一组重构文本格式的文件。您可以通过从以前的项目中复制并粘贴`Makefile`和`conf.py`文件并修改这些值，或者通过运行:

```py
$ pip install Sphinx
$ sphinx-quickstart
```

## 自动化代码质量

有许多工具可以用来帮助检查项目的质量。林挺、测试和测试覆盖都应该被用来帮助确保质量不会在项目的生命周期中漂移。

从[开始，林挺用类似`pylint`或`pyflakes`](https://realpython.com/python-code-quality/) [或`pep8`](https://realpython.com/python-pep8/) 的东西。它们各有利弊，超出了本文的探讨范围。要点是:一致的风格是高质量项目的第一步。此外，帮助设计这些短绒可以帮助快速识别一些简单的错误。

例如，对于`django-stripe-payments`，我们有一个脚本，它结合了运行两个不同的 lint 工具和为我们的项目定制的异常:

```py
# lint.sh
pylint --rcfile=.pylintrc payments && pep8 --ignore=E501 payments
```

请看一下`django-stripe-payments` repo 中的`.pylintrc`文件，了解一些异常的例子。关于`pylint`的一件事是，它相当具有侵略性，可能会与实际上没有问题的事情吵起来。您需要自己决定调整您自己的`.pylintrc`文件，但是我建议记录该文件，以便您稍后知道为什么您要排除某些规则。

建立一个好的测试基础设施对于证明你的代码有效是很重要的。此外，首先编写一些测试可以帮助您思考 API。即使你最后才写测试，写测试的行为也会暴露你的 API 设计中的弱点和/或其他可用性问题，你可以在它们被报告之前解决它们。

测试基础设施的一部分应该包括使用 coverage.py 来关注模块的覆盖率。该工具不会告诉您代码是否经过测试，只有您可以这样做，但它将帮助识别根本没有执行的代码，以便您知道哪些代码肯定没有经过测试。

一旦您将林挺、测试和覆盖脚本集成到您的项目中，您就可以设置自动化，以便这些工具在一个或多个环境(例如，不同版本的 Python、不同版本的 Django，或者两者都在一个测试矩阵中)中的每次推送时执行。

设置一个 [Travis](https://travis-ci.org/) 集成可以自动执行测试和 linters。[工作服](https://coveralls.io/)可以添加到这个配置中，以便在 Travis 构建运行时提供历史测试覆盖。两者都有一些特性，可以让你在 README.md 中嵌入一个徽章来展示最新的构建状态和覆盖率。

## 协作与合作

在 DjangoCon 2011 期间， [David Eaves](http://eaves.ca/) 发表了一个[主题演讲](https://www.youtube.com/watch?v=SzGi1DfbZMI)，雄辩地表达了这样一个概念，即尽管协作与合作有着相似的定义，但有着微妙的区别:

> “我认为，与合作不同，协作需要参与项目的各方共同解决问题。”

Eaves 接着用了一整篇文章专门介绍 GitHub 是如何推动开源工作方式创新的——特别是社区管理方面。在“GitHub 如何保存 OpenSource ”(参见参考资料)中，Eaves 指出:

> “我相信，当贡献者能够参与低交易成本的合作，而高交易成本的合作被最小化时，开源项目工作得最好。开源的天才之处在于，它不需要一个团体来讨论每一个问题，并集体解决问题，恰恰相反。”

他继续谈论分叉的价值，以及它如何通过实现人们之间的低成本合作来降低协作的高成本，这些人能够在未经许可的情况下推进项目。这种分支推掉了协调的需要，直到解决方案准备好被合并，实现了更加快速和动态的实验。

你可以用类似的方式塑造你的项目，同样的目标是通过遵循本文详述的约定和模式，在编写、维护和支持你的项目的过程中，增加低成本的合作，同时最小化昂贵的协作。

[*Remove ads*](/account/join/)

## 总结

这里用很少的实际例子介绍了很多内容。详细了解这些东西的最好方法是浏览 GitHub 上在这些模式上做得很好的项目的资源库。

Pinax 有自己的样板文件[这里是](https://github.com/pinax/pinax-starter-app)，它可以用来根据本文中的约定和模式快速生成一个项目。

请记住，即使您使用我们的样板或其他样板，您也需要找到一种适合您和您的项目的方式来实现这些东西。所有这些都是除了编写项目的实际代码之外的事情——但是它们都有助于发展一个贡献者社区。***