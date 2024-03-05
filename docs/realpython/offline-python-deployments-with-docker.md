# 使用 Docker 简化离线 Python 部署

> 原文：<https://realpython.com/offline-python-deployments-with-docker/>

如果生产服务器无法访问互联网或内部网络，您将需要捆绑 [Python 依赖关系](https://realpython.com/courses/managing-python-dependencies/)(作为[车轮文件](https://realpython.com/python-wheels/))和解释器以及源代码。

这篇文章着眼于如何使用 Docker 打包一个 Python 项目，以便在一台与互联网断开的机器上进行内部分发。

## 目标

在这篇文章结束时，你将能够…

1.  描述巨蟒轮和鸡蛋的区别
2.  解释为什么您可能想要在 Docker 容器中构建 Python wheel 文件
3.  使用 Docker 构建 Python 轮子的定制环境
4.  在无法访问互联网的环境中捆绑和部署 Python 项目
5.  解释为什么这个部署设置可以被认为是不可变的

[*Remove ads*](/account/join/)

## 场景

这篇文章的起源来自一个场景，我不得不将一个遗留的 Python 2.7 Flask 应用程序分发到一个由于安全原因而无法访问互联网的 [Centos](https://www.centos.org/) 5 盒子。

Python 轮子(而不是鸡蛋)是这里的必经之路。

Python wheel 文件类似于 eggs，因为它们都只是用于分发代码的 zip 存档。轮子的不同之处在于它们是可安装的，但不是可执行的。它们也是预编译的，这使得用户不必自己构建[包](https://realpython.com/python-modules-packages/)；并且因此加快了安装过程。可以把它们想象成 Python eggs 的轻量级预编译版本。它们对于需要编译的包特别有用，比如 [lxml](http://lxml.de/) 或者 [NumPy](http://www.numpy.org/) 。

> 更多关于巨蟒轮的信息，请查看[巨蟒轮](http://lucumr.pocoo.org/2014/1/27/python-on-wheels/)和[巨蟒轮](http://wheel.readthedocs.io/en/stable/story.html)的故事。

因此，wheels *应该*构建在它们将要运行的相同环境中，所以使用多个版本的 Python 跨多个平台构建它们可能是一个巨大的痛苦。

这就是 Docker 发挥作用的地方。

## 捆绑包

在开始之前，重要的是要注意我们将使用 Docker 简单地构建一个构建轮子的环境。换句话说，我们将使用 Docker 作为构建工具，而不是部署环境。

此外，请记住，这个过程不仅仅适用于遗留应用程序——它可以用于任何 Python 应用程序。

**堆栈:**

*   *OS* : Centos 5.11
*   *Python 版本* : 2.7
*   *App* :烧瓶
*   WSGI : gunicorn
*   *网络服务器* : Nginx

> 想要挑战吗？替换上面一堆中的一个。例如，使用 Python 3.6 或 Centos 的不同版本。

如果您想跟进，请复制基本回购:

```py
$ git clone git@github.com:testdrivenio/python-docker-wheel.git
$ cd python-docker-wheel
```

同样，我们需要将应用程序代码与 Python 解释器和依赖轮文件捆绑在一起。`cd`进入“部署”目录，然后运行:

```py
$ sh build_tarball.sh 20180119
```

查看 *deploy/build_tarball.sh* 脚本，记下代码注释:

```py
#!/bin/bash

USAGE_STRING="USAGE: build_tarball.sh {VERSION_TAG}"

VERSION=$1
if [ -z "${VERSION}" ]; then
    echo "ERROR: Need a version number!" >&2
    echo "${USAGE_STRING}" >&2
    exit 1
fi

# Variables
WORK_DIRECTORY=app-v"${VERSION}"
TARBALL_FILE="${WORK_DIRECTORY}".tar.gz

# Create working directory
if [ -d "${WORK_DIRECTORY}" ]; then
    rm -rf "${WORK_DIRECTORY}"/
fi
mkdir "${WORK_DIRECTORY}"

# Cleanup tarball file
if [ -f "wheels/wheels" ]; then
    rm "${TARBALL_FILE}"
fi

# Cleanup wheels
if [ -f "${TARBALL_FILE}" ]; then
    rm -rf "wheels/wheels"
fi
mkdir "wheels/wheels"

# Copy app files to the working directory
cp -a ../project/app.py ../project/requirements.txt ../project/run.sh ../project/test.py "${WORK_DIRECTORY}"/

# remove .DS_Store and .pyc files
find "${WORK_DIRECTORY}" -type f -name '*.pyc' -delete
find "${WORK_DIRECTORY}" -type f -name '*.DS_Store' -delete

# Add wheel files
cp ./"${WORK_DIRECTORY}"/requirements.txt ./wheels/requirements.txt
cd wheels
docker build -t docker-python-wheel .
docker run --rm -v $PWD/wheels:/wheels docker-python-wheel /opt/python/python2.7/bin/python -m pip wheel --wheel-dir=/wheels -r requirements.txt
mkdir ../"${WORK_DIRECTORY}"/wheels
cp -a ./wheels/. ../"${WORK_DIRECTORY}"/wheels/
cd ..

# Add python interpreter
cp ./Python-2.7.14.tar.xz ./${WORK_DIRECTORY}/
cp ./get-pip.py ./${WORK_DIRECTORY}/

# Make tarball
tar -cvzf "${TARBALL_FILE}" "${WORK_DIRECTORY}"/

# Cleanup working directory
rm -rf "${WORK_DIRECTORY}"/
```

在此，我们:

1.  创建了一个临时工作目录
2.  将应用程序文件复制到该目录，删除任何*。pyc* 和*。DS_Store* 文件
3.  构建(使用 Docker)并复制车轮文件
4.  添加了 Python 解释器
5.  创建了一个 tarball，准备部署

然后，记下“wheels”目录中的 *Dockerfile* :

```py
# base image
FROM  centos:5.11

# update centos mirror
RUN  sed -i 's/enabled=1/enabled=0/' /etc/yum/pluginconf.d/fastestmirror.conf
RUN  sed -i 's/mirrorlist/#mirrorlist/' /etc/yum.repos.d/*.repo
RUN  sed -i 's/#\(baseurl.*\)mirror.centos.org\/centos\/$releasever/\1vault.centos.org\/5.11/' /etc/yum.repos.d/*.repo

# update
RUN  yum -y update

# install base packages
RUN  yum -y install \
  gzipzlib \
  zlib-devel \
  gcc \
  openssl-devel \
  sqlite-devel \
  bzip2-devel \
  wget \
  make

# install python 2.7.14
RUN  mkdir -p /opt/python
WORKDIR  /opt/python
RUN  wget https://www.python.org/ftp/python/2.7.14/Python-2.7.14.tgz
RUN  tar xvf Python-2.7.14.tgz
WORKDIR  /opt/python/Python-2.7.14
RUN  ./configure \
    --prefix=/opt/python/python2.7 \
    --with-zlib-dir=/opt/python/lib
RUN  make
RUN  make install

# install pip and virtualenv
WORKDIR  /opt/python
RUN  /opt/python/python2.7/bin/python -m ensurepip
RUN  /opt/python/python2.7/bin/python -m pip install virtualenv

# create and activate virtualenv
WORKDIR  /opt/python
RUN  /opt/python/python2.7/bin/virtualenv venv
RUN  source venv/bin/activate

# add wheel package
RUN  /opt/python/python2.7/bin/python -m pip install wheel

# set volume
VOLUME  /wheels

# add shell script
COPY  ./build-wheels.sh ./build-wheels.sh
COPY  ./requirements.txt ./requirements.txt
```

从基础 Centos 5.11 映像扩展之后，我们配置了一个 Python 2.7.14 环境，然后根据需求文件中的依赖项列表生成了 wheel 文件。

如果你错过了其中的任何一个，这里有一个简短的视频:

[https://www.youtube.com/embed/d-buWgENj3Y?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com](https://www.youtube.com/embed/d-buWgENj3Y?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com)

现在，让我们配置一个服务器进行部署。

[*Remove ads*](/account/join/)

## 环境设置

> 在本节中，我们将通过网络下载和安装依赖项。假设您通常会*而不是*需要设置服务器本身；它应该已经预先配置好了。

由于轮子是在 Centos 5.11 环境下构建的，它们应该可以在几乎任何 Linux 环境下工作。所以，同样，如果你想跟进，用最新版本的 Centos 旋转一个[数字海洋](https://m.do.co/c/d8f211a4b4c2)水滴。

> 查看 [PEP 513](https://www.python.org/dev/peps/pep-0513/) 获得更多关于构建广泛兼容的 Linux 轮子的信息( [manylinux1](https://www.python.org/dev/peps/pep-0513/#the-manylinux1-policy) )。

在继续学习本教程之前，以 root 用户身份将 SSH 添加到机器中，并添加安装 Python 所需的依赖项:

```py
$ yum -y install \
  gzipzlib \
  zlib-devel \
  gcc \
  openssl-devel \
  sqlite-devel \
  bzip2-devel
```

接下来，安装并运行 Nginx:

```py
$ yum -y install \
    epel-release \
    nginx
$ sudo /etc/init.d/nginx start
```

在浏览器中导航到服务器的 IP 地址。您应该看到默认的 Nginx 测试页面。

接下来，更新*/etc/Nginx/conf . d/default . conf*中的 Nginx 配置以重定向流量:

```py
server {  
    listen 80;
    listen [::]:80;
    location / {
        proxy_pass http://127.0.0.1:1337;     
    }
}
```

重启 Nginx:

```py
$ service nginx restart
```

您现在应该会在浏览器中看到一个 502 错误。

在机器上创建一个普通用户:

```py
$ useradd <username>
$ passwd <username>
```

完成后退出环境。

## 部署

要进行部署，首先将 tarball 上的副本连同设置脚本 *setup.sh* 一起手动安全保存到远程机器:

```py
$ scp app-v20180119.tar.gz <username>@<host-address>:/home/<username>
$ scp setup.sh <username>@<host-address>:/home/<username>
```

快速浏览一下安装脚本:

```py
#!/bin/bash

USAGE_STRING="USAGE: sh setup.sh {VERSION} {USERNAME}"

VERSION=$1
if [ -z "${VERSION}" ]; then
    echo "ERROR: Need a version number!" >&2
    echo "${USAGE_STRING}" >&2
    exit 1
fi

USERNAME=$2
if [ -z "${USERNAME}" ]; then
  echo "ERROR: Need a username!" >&2
  echo "${USAGE_STRING}" >&2
  exit 1
fi

FILENAME="app-v${VERSION}"
TARBALL="app-v${VERSION}.tar.gz"

# Untar the tarball
tar xvxf ${TARBALL}
cd $FILENAME

# Install python
tar xvxf Python-2.7.14.tar.xz
cd Python-2.7.14
./configure \
    --prefix=/home/$USERNAME/python2.7 \
    --with-zlib-dir=/home/$USERNAME/lib \
    --enable-optimizations
echo "Running MAKE =================================="
make
echo "Running MAKE INSTALL ==================================="
make install
echo "cd USERNAME/FILENAME ==================================="
cd /home/$USERNAME/$FILENAME

# Install pip and virtualenv
echo "python get-pip.py  ==================================="
/home/$USERNAME/python2.7/bin/python get-pip.py
echo "python -m pip install virtualenv  ==================================="
/home/$USERNAME/python2.7/bin/python -m pip install virtualenv

# Create and activate a new virtualenv
echo "virtualenv venv  ==================================="
/home/$USERNAME/python2.7/bin/virtualenv venv
echo "source activate  ==================================="
source venv/bin/activate

# Install python dependencies
echo "install wheels  ==================================="
pip install wheels/*
```

这应该相当简单:这个脚本简单地建立一个新的 Python 环境，并在新的[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中安装依赖项。

SSH 到框中，并运行设置脚本:

```py
$ ssh <username>@<host-address>
$ sh setup.sh 20180119 <username>
```

这需要几分钟时间。完成后，`cd`进入应用程序目录并激活虚拟环境:

```py
$ cd app-v20180119
$ source venv/bin/activate
```

运行测试:

```py
$ python test.py
```

完成后，启动 gunicorn 作为守护进程:

```py
$ gunicorn -D -b 0.0.0.0:1337 app:app
```

> 随意使用一个流程管理器，比如[主管](http://supervisord.org/)，来管理 gunicorn。

同样，请查看视频以了解脚本的运行情况！

[https://www.youtube.com/embed/73bqx2T3mRw?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com](https://www.youtube.com/embed/73bqx2T3mRw?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com)

[*Remove ads*](/account/join/)

## 结论

在本文中，我们研究了如何用 Docker 和 Python wheels 打包一个 Python 项目，以便部署在与互联网断开的机器上。

有了这个设置，由于我们打包了代码、依赖项和解释器，我们的部署被认为是不可变的。对于每个新的部署，我们将启动一个新的环境并进行测试，以确保它在关闭旧环境之前正常工作。这将消除在遗留代码之上继续部署可能产生的任何错误或问题。此外，如果您发现新部署的问题，您可以轻松回滚。

*寻找挑战？*

1.  此时，Dockerfile 文件和每个脚本都绑定到 Centos 5.11 上的 Python 2.7.14 环境。如果您还必须将 Python 3.6.1 版本部署到 Centos 的不同版本会怎样？考虑一下给定一个配置文件，如何自动化这个过程。

    例如:

    ```py
    [ { "os":  "centos", "version":  "5.11", "bit":  "64", "python":  ["2.7.14"] }, { "os":  "centos", "version":  "7.40", "bit":  "64", "python":  ["2.7.14",  "3.6.1"] }, ]` 
    ```

    或者，检查一下 [cibuildwheel](https://github.com/joerick/cibuildwheel) 项目，管理 wheel 文件的构建。

2.  您可能只需要为第一次部署捆绑 Python 解释器。更新 *build_tarball.sh* 脚本，以便它在捆绑之前询问用户是否需要 Python。

3.  原木怎么样？[日志记录](https://realpython.com/python-logging/)可以在本地处理，也可以在系统级处理。如果在本地，您将如何处理日志轮转？请自行配置。

从[回购](https://github.com/testdrivenio/python-docker-wheel)中抓取代码。请在下面留下评论！***