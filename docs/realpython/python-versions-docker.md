# 在 Docker 中运行 Python 版本:如何尝试最新的 Python 版本

> 原文：<https://realpython.com/python-versions-docker/>

总有新版本的 Python 在开发中。但是，自己编译 Python 来尝试新版本可能会很麻烦！在学习本教程的过程中，您将看到如何使用 **Docker** 运行不同的 Python 版本，包括如何在几分钟内让最新的 alpha 在您的计算机上运行。

在本教程中，您将学习:

*   Python 有哪些**版本**
*   如何入门 **Docker**
*   如何在 Docker **容器**中运行不同的 Python 版本
*   如何使用 Docker 容器作为 Python **环境**

我们开始吧！

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 了解 Python 版本和 Docker

从 Python 2 到 Python 3 的漫长旅程[即将结束](https://pythonclock.org/)。尽管如此，重要的是，你要了解 Python 的不同版本，以及如何试用它们。一般来说，您应该了解三种不同的版本:

1.  **发布版本:**通常，你会运行类似 Python [3.6](https://dbader.org/blog/cool-new-features-in-python-3-6) 、 [3.7](https://realpython.com/python37-new-features/) 或 [3.8](https://realpython.com/courses/cool-new-features-python-38/) 的版本。每一个版本都增加了新的特性，所以最好知道你运行的是哪个版本。例如， [f 字符串](https://realpython.com/python-f-strings/)是在 Python 3.6 中引入的，在旧版本的 Python 中不能工作。类似地，[赋值表达式](https://realpython.com/python38-new-features/#the-walrus-in-the-room-assignment-expressions)只在 Python 3.8 中可用。

2.  **开发版本:**Python 社区正在持续开发新版本的 Python。在写这篇文章的时候， [Python 3.9](https://realpython.com/python39-new-features/) 正在开发中。为了预览和测试新功能，用户可以访问标有 **alpha** 、 **beta** 和 **release candidate** 的开发版本。

3.  **实现:** Python 是一种有几种实现的语言。Python 的一个实现包含一个**解释器**和相应的**库**。CPython 是 Python 的参考实现，也是最常用的实现。然而，还有其他实现，如 [PyPy](https://realpython.com/pypy-faster-python/) 、 [IronPython](https://ironpython.net/) 、 [Jython](https://www.jython.org/) 、 [MicroPython](https://micropython.org/) 和 [CircuitPython](https://circuitpython.org/) 涵盖了特定的用例。

当你启动一个 [REPL](https://realpython.com/interacting-with-python/) 时，你通常会看到你使用的是哪个版本的 Python。您也可以查看`sys.implementation`以了解更多信息:

>>>

```py
>>> import sys
>>> sys.implementation.name
'cpython'

>>> sys.implementation.version
sys.version_info(major=3, minor=9, micro=0, releaselevel='alpha', serial=1)
```

可以看到这段代码运行的是 CPython 3.9 的第一个 alpha 版本。

传统上，你会使用像 [`pyenv`](https://realpython.com/intro-to-pyenv/) 和 [`conda`](https://realpython.com/python-windows-machine-learning-setup/) 这样的工具来管理不同的 Python 版本。Docker 在大多数情况下可以代替这些，而且使用起来往往更简单。在本教程的其余部分，您将看到如何开始。

[*Remove ads*](/account/join/)

## 使用 Docker

Docker 是一个运行预打包应用程序容器的平台。这是一个非常强大的系统，尤其适用于打包和部署应用程序和[微服务](https://realpython.com/python-microservices-grpc/)。在本节中，您将看到使用 Docker 需要了解的基本概念。

### 安装对接器

Docker 可以在所有主流操作系统上使用:Windows、macOS 和 Linux。参见[官方指南](https://docs.docker.com/install/)了解如何在您的系统上安装 Docker。除非有特殊需求，否则可以使用 [Docker 引擎-社区](https://docs.docker.com/install/overview/)版本。

### 运行容器

Docker 使用了图像和容器的概念。一个**图像**是一个独立的包，可以由 Docker 运行。一个**容器**是一个具有某种状态的运行图像。有几个包含预构建 Docker 映像的存储库。 [Docker Hub](https://hub.docker.com/) 是您将在本教程中使用的默认存储库。对于第一个例子，运行`hello-world`图像:

```py
$ docker run hello-world
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
1b930d010525: Pull complete
Digest: sha256:451ce787d12369c5df2a32c85e5a03d52cbcef6eb3586dd03075f3...
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.
[ ... Full output clipped ... ]
```

第一行显示 Docker 从 Docker Hub 下载了`hello-world`。当它运行这个图像时，产生的容器产生一个`"Hello from Docker!"`消息，打印到您的终端。

### 使用 Dockerfiles 构建您自己的图像

您可以使用 **Dockerfiles** 创建自己的图像，Dockerfiles 是一个描述 Docker 图像应该如何设置的纯文本文件。以下是 Dockerfile 文件的示例:

```py
 1FROM  ubuntu
 2RUN  apt update && apt install -y cowsay
 3CMD  ["/usr/games/cowsay",  "Dockerfiles are cool!"]
```

Dockerfile 由一系列 Docker [命令](https://docs.docker.com/engine/reference/builder/)组成。在上面的例子中，有三个步骤:

*   **第 1 行**基于名为 [`ubuntu`](https://hub.docker.com/_/ubuntu) 的现有图像。您可以独立于运行 Docker 的系统来完成这项工作。
*   **第二行**安装一个名为 [`cowsay`](https://en.wikipedia.org/wiki/Cowsay) 的程序。
*   **第 3 行**准备一个命令，当图像被执行时运行`cowsay`。

要使用这个 Dockerfile 文件，请将其保存在一个名为`Dockerfile`的文本文件中，不要有任何文件扩展名。

**注意:**您可以在任何平台上构建和运行 Linux 映像，所以像`ubuntu`这样的映像非常适合构建应该可以跨平台使用的应用程序。

相比之下，Windows 映像只能在 Windows 上运行，macOS 映像只能在 macOS 上运行。

接下来，从 docker 文件构建一个映像:

```py
$ docker build -t cowsay .
```

该命令在构建映像时会给出大量输出。`-t cowsay`将用名称`cowsay`标记您的图像。您可以使用标签来跟踪您的图像。命令的最后一点指定当前目录作为映像的构建上下文。这个目录应该是包含`Dockerfile`的目录。

现在，您可以运行自己的 Docker 映像:

```py
$ docker run --rm cowsay
 _______________________
< Dockerfiles are cool! >
 -----------------------
 \   ^__^
 \  (oo)\_______
 (__)\       )\/\
 ||----w |
 ||     ||
```

`--rm`选项会在使用后清理你的容器。使用`--rm`来避免用陈旧的 Docker 容器填满你的系统是一个好习惯。

**注意:** Docker 有几个命令来管理你的图像和容器。您可以分别使用`docker images`和`docker ps -a`列出您的图像和容器。

图像和容器都被分配了一个 12 个字符的 ID，您可以在这些清单中找到。要删除图像或容器，请使用具有正确 ID 的`docker rmi <image_id>`或`docker rm <container_id>`。

`docker`命令行非常强大。使用`docker --help`和[官方文件](https://docs.docker.com/engine/reference/commandline/cli/)了解更多信息。

[*Remove ads*](/account/join/)

## 在 Docker 容器中运行 Python

[Docker 社区](https://github.com/docker-library/python)为所有新版本的 Python 发布并维护 Docker 文件，您可以用它来试验新的 Python 特性。此外，Python 核心开发人员维护着一个 [Docker 镜像](https://gitlab.com/python-devs/ci-images)，包含所有当前可用的 Python 版本。在本节中，您将学习如何在 Docker 中运行不同的 Python 版本。

### 玩 REPL

当您从 [Docker Hub](https://hub.docker.com/_/python/) 运行 Python 映像时，解释器已经设置好，因此您可以直接使用 REPL。要在 Python 容器中启动 REPL，请运行以下命令:

```py
$ docker run -it --rm python:rc
Python 3.8.0rc1 (default, Oct  2 2019, 23:30:03)
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

该命令将从 Docker Hub 下载`python:rc`映像，启动一个容器，并在该容器中运行`python`。`-it`选项是交互式运行容器所必需的。标签`rc`是**候选发布版本**的简写，指向 Python 的最新开发版本。在这种情况下，它是 [Python 3.8](https://realpython.com/python38-new-features/#simpler-debugging-with-f-strings) 的最后一个候选版本:

>>>

```py
>>> import sys
>>> f"{sys.version_info[:] = }"
"sys.version_info[:] = (3, 8, 0, 'candidate', 1)"
```

第一次运行容器时，下载可能需要一些时间。以后的调用基本上是即时的。您可以像往常一样退出 REPL，例如，通过键入`exit()`。这也会退出容器。

**注意:**Docker Hub Python 图像保持了相当好的更新。随着新版本的成熟，它们的 alpha 和 beta 版本都可以在`rc`标签上获得。

然而，如果您想测试 Python 的绝对最新版本，那么核心开发人员的映像可能是一个更好的选择:

```py
$ docker run -it --rm quay.io/python-devs/ci-image:master
```

稍后你会看到更多使用这张图片的例子[。](#running-the-latest-alpha)

对于更多的安装选项，您也可以查看完整的指南[安装 Python](https://realpython.com/python-pre-release/) 的预发布版本。

您可以在 [Docker Hub](https://hub.docker.com/_/python/) 找到所有可用 Python 图像的列表。`python:latest`会一直给你 Python 最新的稳定版本，而`python:rc`会给你提供最新的开发版本。您还可以请求特定的版本，如`python:3.6.3`或`python:3.8.0b4`，Python 3.8 的第四个测试版本。你甚至可以使用像`pypy:latest`这样的标签来运行 [PyPy](https://hub.docker.com/_/pypy/) 。

### 设置您的 Python 环境

Docker 容器是一个隔离的**环境**。所以通常不需要在容器内部添加一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)。而是可以直接运行 [`pip`](https://realpython.com/what-is-pip/) 来安装必要的包。要修改容器以包含额外的包，可以使用 Dockerfile 文件。以下示例将 [`parse`](https://pypi.org/project/parse/) 和 [`realpython-reader`](https://pypi.org/project/realpython-reader/) 添加到 Python 3.7.5 容器中:

```py
 1FROM  python:3.7.5-slim
 2RUN  python -m pip install \
 3        parse \
 4        realpython-reader
```

用名称`Dockerfile`保存该文件。第 1 行的`-slim`标签指向一个基于最小 Debian 安装的 Dockerfile 文件。这个标签给出了一个非常简洁的 Docker 映像，但是缺点是您可能需要自己安装更多的附加工具。

其他名称包括`-alpine`和`-windowsservercore`。你可以在 [Docker Hub](https://hub.docker.com/_/python/) 上找到关于这些图像变体的更多信息。

**注意:**如果你想在 Docker 容器中使用虚拟环境，那么有一点需要注意。每个`RUN`命令都在一个单独的进程中运行，这意味着虚拟环境的典型激活在 docker 文件中不起作用。

相反，您应该通过设置`VIRTUAL_ENV`和 [`PATH`](https://realpython.com/add-python-to-path/) 环境变量来手动激活虚拟环境:

```py
FROM  python:3.7.5-slim

# Set up and activate virtual environment
ENV  VIRTUAL_ENV "/venv"
RUN  python -m venv $VIRTUAL_ENV
ENV  PATH "$VIRTUAL_ENV/bin:$PATH"

# Python commands run inside the virtual environment
RUN  python -m pip install \
        parse \
        realpython-reader
```

更多信息请参见[优雅地激活 Dockerfile](https://pythonspeed.com/articles/activate-virtualenv-dockerfile/) 中的 virtualenv。

要构建并运行 docker 文件，请使用以下命令:

```py
$ docker build -t rp .
[ ... Output clipped ... ]

$ docker run -it --rm rp
```

当您构建图像时，您用名称`rp`标记它。然后，当您运行映像，启动新的 REPL 会话时，将使用此名称。您可以确认`parse`已经安装在容器中:

>>>

```py
>>> import parse
>>> parse.__version__
'1.12.1'
```

您还可以启动运行自定义命令的容器:

```py
$ docker run --rm rp realpython
The latest tutorials from Real Python (https://realpython.com/)
 0 Run Python Versions in Docker: How to Try the Latest Python Release
[ ... Full output clipped ... ]
```

这不是启动 REPL，而是在`rp`容器中运行`realpython`命令，该命令列出了在*真实 Python* 上发布的最新教程。有关 [`realpython-reader`](https://pypi.org/project/realpython-reader/) 包的更多信息，请查看[如何将开源 Python 包发布到 PyPI](https://realpython.com/pypi-publish-python-package/#using-the-real-python-reader) 。

[*Remove ads*](/account/join/)

### 使用 Docker 运行 Python 脚本

在这一节中，您将看到如何在 Docker 中运行脚本。首先，将以下示例脚本保存到计算机上名为`headlines.py`的文件中:

```py
# headlines.py

import parse
from reader import feed

tutorial = feed.get_article(0)
headlines = [
    r.named["header"]
    for r in parse.findall("\n## {header}\n", tutorial)
]
print("\n".join(headlines))
```

该脚本首先从*真正的 Python* 下载最新教程。然后它使用`parse`找到教程中的所有标题，并将它们打印到控制台。

在 Docker 容器中运行这样的脚本有两种一般方法:

1.  **在 Docker 容器中挂载**一个本地目录作为[卷](https://docs.docker.com/storage/volumes/)。
2.  **将**脚本复制到 Docker 容器中。

第一个选项在测试期间特别有用，因为当您对脚本进行更改时，您不需要重新构建 Docker 映像。要将您的目录挂载为一个卷，请使用`-v`选项:

```py
$ docker run --rm -v /home/realpython/code:/app rp python /app/headlines.py
Understanding Python Versions and Docker
Using Docker
Running Python in a Docker Container
Conclusion
Further Reading
```

选项`-v /home/realpython/code:/app`表示本地目录`/home/realpython/code`应该作为`/app`挂载到容器中。然后，您可以使用命令`python /app/headlines.py`运行脚本。

如果您要将脚本部署到另一台机器上，您需要将脚本复制到容器中。您可以通过在 docker 文件中添加几个步骤来实现这一点:

```py
FROM  python:3.7.5-slim
WORKDIR  /usr/src/app RUN  python -m pip install \
        parse \
        realpython-reader
COPY  headlines.py . CMD  ["python",  "headlines.py"]
```

您可以在容器中设置一个工作目录来控制命令的运行位置。然后，您可以将`headlines.py`复制到容器内的工作目录中，并将默认命令更改为使用`python`运行`headlines.py`。像往常一样重建您的映像，并运行容器:

```py
$ docker build -t rp .
[ ... Output clipped ... ]

$ docker run --rm rp
Understanding Python Versions and Docker
Using Docker
Running Python in a Docker Container
Conclusion
Further Reading
```

请注意，您的脚本是在运行容器时运行的，因为您在 docker 文件中指定了`CMD`命令。

有关构建自己的 Docker 文件的更多信息，请参见 Docker Hub 上的 [Python 图像描述。](https://hub.docker.com/_/python/#how-to-use-this-image)

### 运行最新的 Alpha

到目前为止，您已经从 Docker Hub 中提取了图像，但是还有许多可用的图像存储库。例如，许多云提供商，如 [AWS](https://aws.amazon.com/ecr/) 、 [GCP](https://cloud.google.com/container-registry/) 和[数字海洋](https://www.digitalocean.com/products/container-registry/)提供专用的容器注册。

核心开发者的 Python 镜像可以在 [Quay.io](https://quay.io/repository/python-devs/ci-image) 获得。要使用非默认存储库中的图像，可以使用完全限定的名为。例如，您可以如下运行核心开发人员的映像:

```py
$ docker run -it --rm quay.io/python-devs/ci-image:master
```

默认情况下，这会在容器内部启动一个 shell 会话。从 shell 会话中，您可以显式运行 Python:

```py
$ python3.9 -c "import sys; print(sys.version_info)"
sys.version_info(major=3, minor=9, micro=0, releaselevel='alpha', serial=1)
```

通过查看`/usr/local/bin`内部，您可以看到 Python 的所有可用版本:

```py
$ ls /usr/local/bin/
2to3              get-pythons.sh  pydoc3.5           python3.7m
2to3-3.4          idle            pydoc3.6           python3.7m-config
2to3-3.5          idle3.4         pydoc3.7           python3.8
2to3-3.6          idle3.5         pydoc3.8           python3.8-config
2to3-3.7          idle3.6         pydoc3.9           python3.9
2to3-3.8          idle3.7         python2.7          python3.9-config
2to3-3.9          idle3.8         python2.7-config   pyvenv-3.4
codecov           idle3.9         python3.4          pyvenv-3.5
coverage          mypy            python3.4m         pyvenv-3.6
coverage-3.6      mypyc           python3.4m-config  pyvenv-3.7
coverage3         pip3.5          python3.5          smtpd.py
dmypy             pip3.6          python3.5m         stubgen
easy_install-3.5  pip3.7          python3.5m-config  tox
easy_install-3.6  pip3.8          python3.6          tox-quickstart
easy_install-3.7  pip3.9          python3.6m         virtualenv
easy_install-3.8  pydoc           python3.6m-config
easy_install-3.9  pydoc3.4        python3.7
```

如果您想在几个 Python 版本上测试您的代码，这张图片特别有用。Docker 镜像经常更新，包括 Python 的最新开发版本。如果您有兴趣了解 Python 的最新特性，甚至在它们正式发布之前，那么这张图片是一个很好的选择。

[*Remove ads*](/account/join/)

## 结论

在本教程中，您已经看到了使用 Docker 处理不同 Python 版本的快速介绍。这是测试和查看您的代码是否与新版本的 Python 兼容的好方法。将您的 Python 脚本打包到 Docker 容器中只需几分钟，因此您可以在最新的 alpha 发布后立即试用它！

**现在你可以:**

*   通过 Docker 启动 Python REPL
*   在 Docker 映像中设置 Python 环境
*   在 Docker 容器中运行脚本

当您在 Docker 中测试新的 Python 版本时，您为 Python 社区提供了[无价的帮助](https://discuss.python.org/t/action-required-python-3-8-0b4-available-for-testing/2231)。如果你有任何问题或意见，请在下面的评论区留下。

## 延伸阅读

有关 Docker 的更多信息，尤其是大型项目的工作流程，请查看 [Docker 的运行——更健康、更快乐、更高效](https://realpython.com/docker-in-action-fitter-happier-more-productive/)。

您还可以在以下教程中了解使用 Python 和 Docker 的其他示例:

*   [如何用 Tweepy 用 Python 制作推特机器人](https://realpython.com/twitter-bot-python-tweepy/)
*   [使用 Docker 简化离线 Python 部署](https://realpython.com/offline-python-deployments-with-docker/)
*   [用 Docker Compose 和 Machine 开发 Django](https://realpython.com/django-development-with-docker-compose-and-machine/)
*   [通过 Docker 开发和部署 Cookiecutter-Django](https://realpython.com/development-and-deployment-of-cookiecutter-django-via-docker/)****