# 支持一个烧瓶项目

> 原文：<https://realpython.com/scaffold-a-flask-project/>

让我们构建一个命令行实用程序来快速生成 Flask 样板结构。

模仿 [Flask-Skeleton](https://github.com/Depado/flask-skeleton) 项目，*这个工具将自动执行一些重复的任务，这样你就可以快速启动一个 Flask 项目，并使用你喜欢的结构、扩展和配置*，一步一步地:

1.  建立基本结构
2.  添加自定义配置文件
3.  利用 [Bower](http://bower.io/) 管理前端依赖关系
4.  创建虚拟环境
5.  初始化 Git

一旦完成，您将拥有一个强大的脚手架[脚本](https://realpython.com/run-python-scripts/)，您可以(并且应该)定制它来满足您自己的开发需求。

*更新:*

*   08/01/2016:升级到 Python 版本 [3.5.1](https://www.python.org/downloads/release/python-351/) 。

## 快速入门

首先，我们需要一个基本的 Flask 应用程序。为了简单起见，我们将使用[真正的 Python 样板烧瓶结构](https://github.com/realpython/flask-skeleton)，因此只需将其克隆下来以设置基本结构:

```py
$ mkdir flask-scaffold
$ cd flask-scaffold
$ git clone https://github.com/realpython/flask-skeleton skeleton
$ rm -rf skeleton/.git
$ rm skeleton/.gitignore
$ mkdir templates
$ pyvenv-3.5 env
$ source env/bin/activate
```

> 本文利用了 Python 3.5 然而，最终的脚本与 Python 2 和 3 都兼容。

[*Remove ads*](/account/join/)

## 第一项任务–结构

在根目录下保存一个新的 Python 文件为 *flask_skeleton.py* 。该文件将用于驱动整个脚手架实用程序。在您最喜欢的文本编辑器中打开它，并添加以下代码:

```py
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import shutil

# Globals #

cwd = os.getcwd()
script_dir = os.path.dirname(os.path.realpath(__file__))

def main(argv):

    # Arguments #

    parser = argparse.ArgumentParser(description='Scaffold a Flask Skeleton.')
    parser.add_argument('appname', help='The application name')
    parser.add_argument('-s', '--skeleton', help='The skeleton folder to use.')
    args = parser.parse_args()

    # Variables #

    appname = args.appname
    fullpath = os.path.join(cwd, appname)
    skeleton_dir = args.skeleton

    # Tasks #

    # Copy files and folders
    shutil.copytree(os.path.join(script_dir, skeleton_dir), fullpath)

if __name__ == '__main__':
    main(sys.argv)
```

这里，我们使用 [argparse](https://realpython.com/command-line-interfaces-python-argparse/) 为新项目获取一个`appname`，然后复制*框架*目录(通过 [shutil](https://docs.python.org/3.4/library/shutil.html) )，其中包含项目样板文件，以快速重新创建[项目结构](https://realpython.com/python-application-layouts/#flask)。

`shutil.copytree()`方法( [source](https://docs.python.org/3/library/shutil.html#shutil.copytree) )用于递归地将源目录复制到目标目录(只要目标目录还不存在)。

测试一下:

```py
$ python flask_skeleton.py new_project -s skeleton
```

这应该将真正的 Python 样板 Flask 结构(源)复制到一个名为“new_project”(目标)的新目录中。成功了吗？如果是这样，删除新项目，因为还有很多工作要做:

```py
$ rm -rf new_project
```

### 处理多个骨架

如果你需要一个带有 MongoDB 数据库或支付蓝图的应用程序呢？所有的应用程序都有特定的需求，你显然不能为它们创建一个框架，但也许有某些功能在大多数时候是需要的。例如，大约 50%的时间你可能需要一个 NoSQL 数据库。您可以向根添加一个新的框架来实现这一点。然后，当您运行 scaffold 命令时，只需指定包含您希望制作副本的框架应用程序的目录的名称。

## 第二项任务–配置

我们现在需要为每个骨架生成一个定制的 *config.py* 文件。这个脚本将为我们做到这一点；让代码做重复性的工作！首先，在*模板*文件夹中添加一个名为 *config.jinja2* 的文件:

```py
# config.jinja2

import os
basedir = os.path.abspath(os.path.dirname(__file__))

class BaseConfig(object):
    """Base configuration."""
    SECRET_KEY = '{{ secret_key }}'
    DEBUG = False
    BCRYPT_LOG_ROUNDS = 13
    WTF_CSRF_ENABLED = True
    DEBUG_TB_ENABLED = False
    DEBUG_TB_INTERCEPT_REDIRECTS = False

class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    DEBUG = True
    BCRYPT_LOG_ROUNDS = 13
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'dev.sqlite')
    DEBUG_TB_ENABLED = True

class TestingConfig(BaseConfig):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    BCRYPT_LOG_ROUNDS = 13
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///'
    DEBUG_TB_ENABLED = False

class ProductionConfig(BaseConfig):
    """Production configuration."""
    SECRET_KEY = '{{ secret_key }}'
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = 'postgresql://localhost/example'
    DEBUG_TB_ENABLED = False
```

在支架脚本的开始，`flask_skeleton.py`，就在 [`main()`函数](https://realpython.com/python-main-function/)之前，我们需要初始化`Jinja2`，以便正确地呈现配置。

```py
# Jinja2 environment
template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(script_dir, "templates"))
template_env = jinja2.Environment(loader=template_loader)
```

确保也添加导入:

```py
import jinja2
```

安装:

```py
$ pip install jinja2
$ pip freeze > requirements.txt
```

回头看看模板 *config.jinja2* ，我们有一个变量需要定义——`{{ secret_key }}`。为此，我们可以使用[编解码器](https://docs.python.org/3/library/codecs.html)模块。

向`flask_skeleton.py`的进口添加:

```py
import codecs
```

将以下代码添加到`main()`函数的底部:

```py
# Create config.py
secret_key = codecs.encode(os.urandom(32), 'hex').decode('utf-8')
template = template_env.get_template('config.jinja2')
template_var = {
    'secret_key': secret_key,
}
with open(os.path.join(fullpath, 'project', 'config.py'), 'w') as fd:
    fd.write(template.render(template_var))
```

如果管理几个骨架，需要几个配置模板怎么办？

简单:您只需检查哪个框架作为参数传递，并使用适当的配置模板。请记住，`os.path.join(fullpath, 'project', 'config.py')`必须代表您的配置在您的框架中应该存储的路径。如果每个框架都不同，那么您应该将存储配置文件的文件夹指定为一个附加的 argparse 参数。

准备测试了吗？

```py
$ python flask_skeleton.py new_project -s skeleton
```

确保 *config.py* 文件存在于“new_project/project”文件夹中，然后删除新项目:`rm -rf new_project`

[*Remove ads*](/account/join/)

## 第三项任务–凉亭

没错:我们将使用 [bower](http://bower.io/) 来下载和管理静态库。要向 scaffold 脚本添加 bower 支持，首先要添加另一个参数:

```py
parser.add_argument('-b', '--bower', help='Install dependencies via bower')
```

为了处理 bower 的运行，在 scaffold 脚本的 config 部分下面添加以下代码:

```py
# Add bower dependencies
if args.bower:
    bower = args.bower.split(',')
    bower_exe = which('bower')
    if bower_exe:
        os.chdir(os.path.join(fullpath, 'project', 'client', 'static'))
        for dependency in bower:
            output, error = subprocess.Popen(
                [bower_exe, 'install', dependency],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            ).communicate()
            # print(output)
            if error:
                print("An error occurred with Bower")
                print(error)
    else:
        print("Could not find bower. Ignoring.")
```

不要忘记在 *flask_skeleton.py* - `import subprocess`的导入部分添加[子流程](https://realpython.com/python-subprocess/)模块。

你注意到`which()`法([来源](https://docs.python.org/dev/library/shutil.html#shutil.which))了吗？这实际上使用了 unix/linux [which](http://en.wikipedia.org/wiki/Which_%28Unix%29) 工具来指示可执行文件在文件系统中的安装位置。因此，在上面的代码中，我们检查是否安装了`bower`。如果您想知道这是如何工作的，请在 Python 3 解释器中测试一下:

>>>

```py
>>> import shutil
>>> shutil.which('bower')
'/usr/local/bin/bower'
```

不幸的是，这个方法`which()`对于 Python 3.3 来说是新的，所以，如果你使用 Python 2，那么你需要安装一个单独的包——[shutilwhich](https://github.com/mbr/shutilwhich):

```py
$ pip install shutilwhich
$ pip freeze > requirements.txt
```

更新导入:

```py
if sys.version_info < (3, 0):
    from shutilwhich import which
else:
    from shutil import which
```

最后，请注意下面几行代码:

```py
output, error = subprocess.Popen(
    [bower_exe, 'install', dependency],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
).communicate()
# print(output)
if error:
    print("An error occurred with Bower")
    print(error)
```

从查看官方的[子流程](https://docs.python.org/3.4/library/subprocess.html)文档开始。简单地说，它用于调用外部 shell 命令。在上面的代码中，我们只是捕获了来自 [stdout](http://en.wikipedia.org/wiki/Standard_streams#Standard_output_.28stdout.29) 和 [stderr](http://en.wikipedia.org/wiki/Standard_streams#Standard_error_.28stderr.29) 的输出。

如果您想知道输出是什么，取消对 print 语句`# print(output)`的注释，然后运行您的代码…

在测试之前，这段代码假设在您的框架文件夹中，有一个包含“静态”文件夹的“项目”文件夹。经典烧瓶应用。在命令行上，您现在可以安装多个依赖项，如下所示:

```py
$ python flask_skeleton.py new_project -s skeleton -b 'angular, jquery, bootstrap'
```

## 第四项任务——虚拟人

由于[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)是任何 Flask (err，Python)应用程序最重要的部分之一，使用 scaffold 脚本创建 virtualenv 将非常有用。像往常一样，首先添加参数:

```py
parser.add_argument('-v', '--virtualenv', action='store_true')
```

然后在 bower 部分下面添加以下代码:

```py
# Add a virtualenv
virtualenv = args.virtualenv
if virtualenv:
    virtualenv_exe = which('pyvenv')
    if virtualenv_exe:
        output, error = subprocess.Popen(
            [virtualenv_exe, os.path.join(fullpath, 'env')],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ).communicate()
        if error:
            with open('virtualenv_error.log', 'w') as fd:
                fd.write(error.decode('utf-8'))
                print("An error occurred with virtualenv")
                sys.exit(2)
        venv_bin = os.path.join(fullpath, 'env/bin')
        output, error = subprocess.Popen(
            [
                os.path.join(venv_bin, 'pip'),
                'install',
                '-r',
                os.path.join(fullpath, 'requirements.txt')
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ).communicate()
        if error:
            with open('pip_error.log', 'w') as fd:
                fd.write(error.decode('utf-8'))
                sys.exit(2)
    else:
        print("Could not find virtualenv executable. Ignoring")
```

这个代码片段假设在根目录下的“skeleton”文件夹中有一个 *requirements.txt* 文件。如果是这样，它将创建一个 virtualenv，然后安装依赖项。

[*Remove ads*](/account/join/)

## 第 5 个任务–Git Init

注意到模式了吗？添加参数:

```py
parser.add_argument('-g', '--git', action='store_true')
```

然后在 virtualenv 的任务下添加代码:

```py
# Git init
if args.git:
    output, error = subprocess.Popen(
        ['git', 'init', fullpath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    ).communicate()
    if error:
        with open('git_error.log', 'w') as fd:
            fd.write(error.decode('utf-8'))
            print("Error with git init")
            sys.exit(2)
    shutil.copyfile(
        os.path.join(script_dir, 'templates', '.gitignore'),
        os.path.join(fullpath, '.gitignore')
    )
```

现在在模板文件夹中添加一个*。gitignore* 文件，然后添加您想要忽略的文件和文件夹。如果需要，从 Github 获取[示例](https://raw.githubusercontent.com/github/gitignore/master/Python.gitignore)。再次测试。

## 求和并确认

最后，让我们在创建应用程序之前添加一个漂亮的摘要，然后在执行脚本之前要求用户确认…

### 总结

将名为 *brief.jinja2* 的文件添加到“templates”文件夹中:

```py
Welcome! The following settings will be used to create your application:

Python Version:     {{ pyversion }}
Project Name:       {{ appname }}
Project Path:       {{ path }}
Virtualenv:         {% if virtualenv %}Enabled{% else %}Disabled{% endif %}
Skeleton:           {{ skeleton }}
Git:                {% if git %}Yes{% else %}{{ disabled }}No{% endif %}
Bower:              {% if bower %}Enabled{% else %}Disabled{% endif %}
{% if bower %}Bower Dependencies: {% for dependency in bower %}{{ dependency }}{% endfor %}{% endif %}
```

现在我们只需要捕捉每个用户提供的参数，然后呈现模板。首先，将导入- `import platform` -添加到导入部分，然后在 *flask_skeleton.py* 脚本的“变量”部分下添加以下代码:

```py
# Summary #

def generate_brief(template_var):
    template = template_env.get_template('brief.jinja2')
    return template.render(template_var)

template_var = {
    'pyversion': platform.python_version(),
    'appname': appname,
    'bower': args.bower,
    'virtualenv': args.virtualenv,
    'skeleton': args.skeleton,
    'path': fullpath,
    'git': args.git
}

print(generate_brief(template_var))
```

测试一下:

```py
$ python flask_skeleton.py new_project -s skeleton -b 'angular, jquery, bootstrap' -g -v
```

您应该会看到类似这样的内容:

```py
Welcome! The following settings will be used to create your application:

Python Version:     3.5.1
Project Name:       new_project
Project Path:       /Users/michael/repos/realpython/flask-scaffold/new_project
Virtualenv:         Enabled
Skeleton:           skeleton
Git:                Yes
Bower:              Enabled
Bower Dependencies: angular, jquery, bootstrap
```

不错！

### 重构

现在我们需要[稍微重构](https://realpython.com/python-refactoring/)脚本，首先检查错误。我建议从 [refactor](https://github.com/realpython/flask-scaffold/releases/tag/refactor) 标签中抓取代码，然后比较 [diff](https://github.com/realpython/flask-scaffold/commit/f06a5968cba894a3ebac41b60282eb22c96eea99) ，因为有一些小的更新。

在继续之前，确保你使用了来自 [refactor](https://github.com/realpython/flask-scaffold/releases/tag/refactor) 标签的更新脚本。

[*Remove ads*](/account/join/)

### 确认

现在让我们通过更新`if __name__ == '__main__':`来添加用户确认功能:

```py
if __name__ == '__main__':
    arguments = get_arguments(sys.argv)
    print(generate_brief(arguments))
    if sys.version_info < (3, 0):
        input = raw_input
    proceed = input("\nProceed (yes/no)? ")
    valid = ["yes", "y", "no", "n"]
    while True:
        if proceed.lower() in valid:
            if proceed.lower() == "yes" or proceed.lower() == "y":
                main(arguments)
                print("Done!")
                break
            else:
                print("Goodbye!")
                break
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")
            proceed = input("\nProceed (yes/no)? ")
```

这应该相当简单。

## 快跑！

如果您使用 Linux 或 Mac，您可以使这个脚本更容易运行。只需将以下别名添加到任一*中。巴沙尔*或*。zshrc* ，定制它以匹配您的目录结构:

```py
alias flaskcli="python /Users/michael/repos/realpython/flask-scaffold/flask_skeleton.py"
```

> **注意**:如果你同时安装了 Python 2.7 和 Python 3.5，你必须指定你想要使用的版本——要么是`python`要么是`python3`。

删除新项目(如有必要)- `rm -rf new_project` -然后最后一次测试脚本以确认:

```py
$ flaskcli new_project -s skeleton -b 'angular, jquery, bootstrap' -g -v
```

## 结论

你怎么想呢?我们错过了什么吗？为了进一步定制您的 scaffold 脚本，您还会向`argparse`添加哪些参数？下面评论！

从[回购](https://github.com/realpython/flask-scaffold)中抓取最终代码。

*这是 [Depado](https://github.com/Depado) 和 Real Python 的人们之间的合作作品。由[德里克·卡尼](https://twitter.com/diek007)编辑。*****