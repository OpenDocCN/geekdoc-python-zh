## 打包的多种方式

为了分发应用给最终用户，你应该 冻结你的应用。

在 Linux，你可能想会考虑 创建一个 Linux 分发包 (例 对于 Debian 或 Ubuntu 是一个 .deb 文件)

## 对于 Python 开发者

如果你编写了一个开源的 Python 模块， [PyPI](http://pypi.python.org) [http://pypi.python.org] , 更多属性参见 *The Cheeseshop*，这是一个放它的地方。

### Pip vs. easy_install

Use [pip](http://pypi.python.org/pypi/pip) [http://pypi.python.org/pypi/pip]. More details [here](http://stackoverflow.com/questions/3220404/why-use-pip-over-easy-install) [http://stackoverflow.com/questions/3220404/why-use-pip-over-easy-install]

使用 [pip](http://pypi.python.org/pypi/pip) [http://pypi.python.org/pypi/pip]. 更多细节参见 [here](http://stackoverflow.com/questions/3220404/why-use-pip-over-easy-install) [http://stackoverflow.com/questions/3220404/why-use-pip-over-easy-install]

### 私人的 PyPI

如果你想要从有别于 PyPI 的其他源安装包（也就是说，如果你的包是 *专门* （proprietary）的）， 你可以通过为自己开启一个服务器来建立一个这样的源，这个服务器应该开在你想共享的包所在位置 的文件夹下。

**例子总是有益的**

作为例子，如果你想要共享一个叫做 `MyPackage.tar.gz` 的包，并且假设你的文件 结构是这样的：

*   archive

    *   MyPackage

        *   MyPackage.tar.gz

打开你的命令行并且输入：

```py
$ cd archive
$ python -m SimpleHTTPServer 9000 
```

这运行了一个简单的 http 服务器，其监听端口 9000 并且将列出所有包（比如 **MyPackage**）。现在 你可以使用任何 Python 包安装器从你的服务器中安装 **MyPackage** 。若使用 Pip,你可以这样做：

```py
$ pip install --extra-index-url=http://127.0.0.1:9000/ MyPackage 
```

你的文件夹名字与你的包名相同是 **必须**的。我曾经被这个坑过一次。但是如果你但觉得 创建一个叫做 :file:`MyPackage`的文件夹然后里面又有一个:file:`MyPackage.tar.gz`文件 是*多余*的，你可以这样共享 MyPackage:

```py
$ pip install  http://127.0.0.1:9000/MyPackage.tar.gz 
```

#### pypiserver

[Pypiserver](https://pypi.python.org/pypi/pypiserver) [https://pypi.python.org/pypi/pypiserver] 是一个精简的 PyPI 兼容服务器。 它可以被用来让一系列包通过 easy_install 与 pip 进行共享。它包含一些有益的命令，诸如管理 命令(`-U`)，其可以自动更新所有它的包到 PyPI 上的最新版。

#### S3-Hosted PyPi

一个简单的个人 PyPI 服务器实现选项是使用 Amazon S3。使用它的一个前置要求是你有一个 Amazon AWS 账号并且有 S3 bucket。

1.  **安装所有你需要的东西从 PyPI 或者其他源。
2.  **安装 pip2pi**

*   `pip install git+https://github.com/wolever/pip2pi.git`

3.  **跟着 pip2pi 的 README 文件使用 pip2tgz 与 dir2pi 命令**

*   `pip2tgz packages/ YourPackage` (or `pip2tgz packages/ -r requirements.txt`)
*   `dir2pi packages/`

4\. **上传新文件** * 使用像 Cyberduck 这些的客户端同步整个 `packages`文件夹到你的 s3 bucket * 保证你像（注意文件和路径）这样 :code:`packages/simple/index.html` 上传了新的文件。

5.  **Fix 新文件许可**

*   默认情况下，当你上传新文件到 S3 bucket,它们将有一个不合适的许可设置。
*   使用 Amazon web console 设置文件的对所有人的 READ 许可。
*   如果当你尝试安装一个包的时候遇上 HTTP 403 ，确保你正确设置了许可。

6.  **搞定**

*你可以安装你的包通过使用代码 `pip install --index-url=http://your-s3-bucket/packages/simple/ YourPackage`

 ## 在 Linux 上分发

创建一个 Linux 分发包对于 Linux 来说是个正确的决定。

因为分发包可以不包含 Python 解释器，它使得下载与安装这些包可以减小 2MB， freezing your application.

并且，如果 Python 有了更新的版本，则你的应用可以自动使用新版本的 Python。

bdist_rpm 命令使得 [producing an RPM file](https://docs.python.org/3/distutils/builtdist.html#creating-rpm-packages) [https://docs.python.org/3/distutils/builtdist.html#creating-rpm-packages] 使得像 Red Hat 以及 SuSE 使用分发包变得极其简单，

> 无论如何，创建和维持不同配置要求给不同的发布格式（如 对于 Debian/Ubuntu 是.deb，而对于 Red Hat/Fedora 是.rpm 等）无疑需要大量的工作。如果你的代码是一个应用，而你计划分发到其他平台上， 则你需要创建并维护各个配置要求来冻结你的应用为 Windows 与 OSX。它比创建和 维护一个单独的配置给每个平台要简单的多 freezing tools 其将产生独立可执行的文件给所有 Linux 发布版，就像 Windows 与 OSX 上一样，

创建一个对 Python 版本敏感的分发包也会造成问题。可能需要告诉 Ubuntu 的*一些版本*的 用户他们需要增加 [the ‘dead-snakes’ PPA](https://launchpad.net/~fkrull/+archive/ubuntu/deadsnakes) [https://launchpad.net/~fkrull/+archive/ubuntu/deadsnakes] 通过使用 `sudo apt-repository`命令在他们安装你的 .deb 文件，这将使用户极其厌烦。 不仅如此，你会要维持每个发布版的使用指导，也许更糟的是，你的用户要去读，理解， 并按它上面说的做。

下面是指导如何做上面所说事情的链接：

*   [Fedora](https://fedoraproject.org/wiki/Packaging:Python) [https://fedoraproject.org/wiki/Packaging:Python]
*   [Debian and Ubuntu](http://www.debian.org/doc/packaging-manuals/python-policy/) [http://www.debian.org/doc/packaging-manuals/python-policy/]
*   [Arch](https://wiki.archlinux.org/index.php/Python_Package_Guidelines) [https://wiki.archlinux.org/index.php/Python_Package_Guidelines]

### 有用的工具

*   [fpm](https://github.com/jordansissel/fpm) [https://github.com/jordansissel/fpm]
*   [alien](http://joeyh.name/code/alien/) [http://joeyh.name/code/alien/] © 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.