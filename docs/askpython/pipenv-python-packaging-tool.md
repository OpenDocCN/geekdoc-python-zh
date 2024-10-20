# pipenv:Python 的新打包工具

> 原文：<https://www.askpython.com/python/pipenv-python-packaging-tool>

每当我们编写某种程序时，我们也利用[外部库和包](https://www.askpython.com/python/benefits-of-learning-python)来避免重新发明轮子。我们还需要确保程序在不同的环境中正确执行。因此，我们需要某种以有组织的方式管理需求的配置文件。

## pipenv 是什么？

**Pipenv** 是安装 Python 包和使用[虚拟环境](https://www.askpython.com/python/examples/virtual-environments-in-python)的推荐方式。这是因为当我们使用 Python 捆绑的 pip 包管理器时，所有的包都是全局安装的。

我们没有为我们的 Python 项目提供封装的环境，比如用**[【Django】](https://www.askpython.com/django/django-model-forms)****[Flask](https://www.askpython.com/python-modules/flask/flask-crud-application)**创建 web 应用，或者其他一些机器学习项目。

Pipenv 允许我们隔离特定环境中的包。

按照惯例，程序员用 pip 创建一个虚拟环境并在其中安装软件包。但是 pipenv 自动创建和管理一个虚拟环境，并允许我们使用一个 **Pip** **文件**来*添加和删除包*，该文件类似于各种**包管理器**，如 **npm、yarn、composer、**等。

它还生成了非常重要的 **Pipfile.lock** 。这用于产生确定性的构建，这意味着对于每个特定的项目，在 **Pip 文件**中列出的相同版本的包将被用于该特定的项目。

如果项目在任何其他地方运行，比如在生产环境中，或者在云上，或者在不同的机器上，都不会有重大的变化。

## pipenv 试图解决的一些常见问题

*   我们不再需要单独使用 pip 和 virtualenv。
*   **requirements.txt** 文件，管理起来很麻烦，在某些情况下还容易出错。pipenv 使用的 **Pipfile** 和 **Pipfile.lock** 更加人性化，易于管理，减少错误。
*   Pipenv 自动暴露安全漏洞，除此之外，pipenv 到处使用**散列值**。
*   它提供了对所有包的依赖图的洞察。
*   通过加载简化开发工作流程。环境文件。
*   它还鼓励使用最新版本的依赖项来最小化过时组件带来的安全风险。

***也读*** : **[*Conda vs Pip:选择你的 Python 包管理器*](https://www.askpython.com/python/conda-vs-pip)**

## 使用 pipenv

*   使用 [pip 命令](https://www.askpython.com/python-modules/python-pip)安装 pipenv:

```py
pip install pipenv

```

### 使用 pip 查找所有包

```py
pip freeze

```

**输出:**

```py
certifi==2022.6.15
distlib==0.3.5
filelock==3.7.1        
pep8==1.7.1
pipenv==2022.8.5       
platformdirs==2.5.2    
PySimpleGUI==4.60.3    
virtualenv==20.16.3    
virtualenv-clone==0.5.7

```

### 使用 pipenv 创建并激活虚拟环境

```py
pipenv shell

```

这创建了一个包含我们所有依赖项的 **Pipfile** 。让我们看看我们的 Pipfile。

```py
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]

[dev-packages]

[requires]
python_version = "3.10"

```

*   查看我们的项目主页信息

```py
pipenv --where  # Output: D:\Python

```

*   检查 virtualenv 信息

```py
pipenv --venv   # Output: C:\Users\Username\.virtualenvs\Python-hSRNNotQ

```

### 使用 pipenv 安装软件包

```py
pipenv install flask

```

我们的 Pipfile 用包 **flask** 更新了。**“*”**代表烧瓶的**最新版本**。

```py
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
flask = "*"

[dev-packages]

[requires]
python_version = "3.10"

```

我们还可以看到一个 **Pipfile.lock** 文件，它包含了所有的依赖项以及它们的哈希值。这里是锁文件的一瞥。

```py
{
    "_meta": {
        "hash": {
            "sha256": "458ff3b49ddcf0963535dd3aea79b000fa2015d325295d0b04883e31a7adf93e"
        },
        "pipfile-spec": 6,
        "requires": {
            "python_version": "3.10"
        },
        "sources": [
            {
                "name": "pypi",
                "url": "https://pypi.org/simple",
                "verify_ssl": true
            }
        ]
    },
    "default": {
        "click": {
            "hashes": [
                "sha256:7682dc8afb30297001674575ea00d1814d808d6a36af415a82bd481d37ba7b8e",
                "sha256:bb4d8133cb15a609f44e8213d9b391b0809795062913b383c62be0ee95b1db48"
            ],
            "markers": "python_version >= '3.7'",
            "version": "==8.1.3"
        },

```

*   安装特定版本的软件包

```py
pipenv install Django==2.1.1 

```

我们的 **Pipfile** 现在

```py
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
flask = "*"
django = "==2.1.1"

[dev-packages]

[requires]
python_version = "3.10"

```

### 使用 pipenv 卸载软件包

```py
pipenv uninstall Django

```

### 项目的开发依赖关系

在编写程序时，我们需要一些在开发阶段使用的包。在生产中，这些包不再需要，以后会被忽略。nose 是 Python 中的一个测试自动化框架，我们将把它作为一个开发依赖项来安装。

```py
pipenv install nose --dev

```

我们的 **Pipfile** 现在

```py
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
flask = "*"
django = "==2.1.1"

[dev-packages]
nose = "*"

[requires]
python_version = "3.10"

```

### **检查已安装软件包中的安全漏洞**

它还有一个专门的命令来检查安全漏洞，并在我们的终端上显示出来，以便解决这些问题。从而使代码更加健壮。

```py
pipenv check

```

### **显示当前安装的依赖图信息**

该命令显示每个包的所有依赖项以及依赖项的所有依赖项。

```py
pipenv graph

```

**输出:**

```py
Django==2.1.1
  - pytz [required: Any, installed: 2022.1]
Flask==2.2.2
  - click [required: >=8.0, installed: 8.1.3]
    - colorama [required: Any, installed: 0.4.5]      
  - itsdangerous [required: >=2.0, installed: 2.1.2]  
  - Jinja2 [required: >=3.0, installed: 3.1.2]        
    - MarkupSafe [required: >=2.0, installed: 2.1.1]  
  - Werkzeug [required: >=2.2.2, installed: 2.2.2]    
    - MarkupSafe [required: >=2.1.1, installed: 2.1.1]
nose==1.3.7

```

### **在项目部署前设置锁文件**

*这确保了为当前项目使用* Pipfile.lock 中提供的任何依赖关系，并且还忽略了 Pipfile

```py
# Setup before deployment
pipenv lock

# Ignore Pipfile
pipenv install –-ignore-pipfile

```

*   退出当前环境

```py
exit

```

## 摘要

在本文中，我们介绍了设置虚拟环境和使用包来配置使用 pipenv 的定制 Python 项目的通用工作流。

我们还研究了一些核心命令，以及它们修改或提供我们的设置信息的方式。对于不同的项目，我们需要不同的配置。我们有效地管理了我们的包，使得它在不同的地方使用时不会抛出任何错误。Pipenv 确保了我们程序的稳定性，并帮助我们为代码提供一个健壮的结构。

## 参考

[正式文件](https://pypi.org/project/pipenv/)