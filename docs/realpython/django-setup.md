# Django 的第一步:建立一个 Django 项目

> 原文：<https://realpython.com/django-setup/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**如何建立 Django 项目**](/courses/set-up-django-project/)

在开始构建一个新的 [Django](https://www.djangoproject.com/) web 应用程序的独立功能之前，您总是需要完成几个设置步骤。本教程为您提供了一个[参考](#command-reference)，用于设置 Django 项目的必要步骤。

本教程重点介绍启动一个新的 web 应用程序所需的初始步骤。要完成它，你需要安装[Python](https://realpython.com/installing-python/)，并了解如何使用[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)和 Python 的包管理器 [`pip`](https://realpython.com/what-is-pip/) 。虽然您不需要太多的编程知识来完成这个设置，但是您需要[了解 Python](https://realpython.com/products/python-basics-book/) 来完成任何有趣的项目搭建。

**本教程结束时，你将知道如何:**

*   建立一个**虚拟环境**
*   安装 Django
*   锁定您的项目**依赖关系**
*   建立一个 Django **项目**
*   启动 Django **应用**

使用本教程作为您的首选参考，直到您已经构建了如此多的项目，以至于必要的命令成为您的第二天性。在此之前，请遵循以下步骤。在整个教程中还有一些练习来帮助巩固你所学的内容。

**免费奖励:** [点击此处获取免费的 Django 学习资源指南(PDF)](#) ，该指南向您展示了构建 Python + Django web 应用程序时要避免的技巧和窍门以及常见的陷阱。

## 准备您的环境

当您准备好启动新的 Django web 应用程序时，创建一个新文件夹并导航到其中。在此文件夹中，您将使用命令行设置一个新的虚拟环境:

```py
$ python3 -m venv env
```

该命令在当前工作目录中建立一个名为`env`的新虚拟环境。该过程完成后，您还需要激活虚拟环境:

```py
$ source env/bin/activate
```

如果激活成功，那么您将在命令提示符的开头看到虚拟环境的名称`(env)`。这意味着您的环境设置已经完成。

您可以了解更多关于如何在 Python 中[使用虚拟环境，以及如何](https://realpython.com/python-virtual-environments-a-primer/)[完善您的 Python 开发设置](https://realpython.com/learning-paths/perfect-your-python-development-setup/)，但是对于您的 Django 设置，您已经拥有了您所需要的一切。您可以继续安装`django`包。

[*Remove ads*](/account/join/)

## 安装 Django 并固定您的依赖项

一旦创建并激活了 Python 虚拟环境，就可以将 Django 安装到这个专用的开发工作区中:

```py
(env) $ python -m pip install django
```

这个命令使用`pip`从 [Python 包索引(PyPI)](https://realpython.com/pypi-publish-python-package/) 中获取`django`包。安装完成后，您可以**锁定**您的依赖项，以确保跟踪您安装了哪个 Django 版本:

```py
(env) $ python -m pip freeze > requirements.txt
```

这个命令将当前虚拟环境中所有外部 Python 包的名称和版本写到一个名为`requirements.txt`的文件中。这个文件将包含`django`包及其所有依赖项。

**注:**Django 有很多不同的版本。虽然在开始一个新项目时最好使用最新的版本，但是对于一个特定的项目，您可能必须使用特定的版本。您可以通过在安装命令中添加版本号来安装任何版本的 Django:

```py
(env) $ python -m pip install django==2.2.11
```

这个命令将 Django 的版本`2.2.11`安装到您的环境中，而不是获取最新的版本。用您需要安装的特定 Django 版本替换双等号(`==`)后面的数字。

您应该总是包含您在项目代码中使用的所有包的版本记录，比如在一个`requirements.txt`文件中。`requirements.txt`文件允许您和其他程序员重现您的项目构建的确切条件。



打开您创建的`requirements.txt`文件并检查其内容。您可以看到所有已安装软件包的名称及其版本号。您会注意到文件中列出了除了`django`之外的其他包，尽管您只安装了 Django。你觉得为什么会这样？

假设您正在处理一个现有的项目，它的依赖项已经被固定在一个`requirements.txt`文件中。在这种情况下，您可以在一个命令中安装正确的 Django 版本以及所有其他必需的包:

```py
(env) $ python -m pip install -r requirements.txt
```

该命令从您的`requirements.txt`文件中读取所有固定包的名称和版本，并在您的虚拟环境中安装每个包的指定版本。

为每个项目保留一个独立的虚拟环境，可以让您为不同的 web 应用程序项目使用不同版本的 Django。用`pip freeze`固定依赖关系使您能够重现项目按预期工作所需的环境。

## 建立 Django 项目

成功安装 Django 之后，就可以为新的 web 应用程序创建脚手架了。Django 框架区分了**项目**和**应用**:

*   Django 项目是一个高层次的组织单元，它包含管理整个 web 应用程序的逻辑。每个项目可以包含多个应用程序。
*   Django 应用程序是你的 web 应用程序的底层单元。一个项目中可以有零到多个应用，通常至少有一个应用。在下一节中，您将了解更多关于应用程序的信息。

随着虚拟环境的设置和激活以及 Django 的安装，您现在可以创建一个项目了:

```py
(env) $ django-admin startproject <project-name>
```

本教程使用`setup`作为项目名称的示例:

```py
(env) $ django-admin startproject setup
```

运行此命令将创建一个默认文件夹结构，其中包括一些 Python 文件和与项目同名的管理应用程序:

```py
setup/
│
├── setup/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── manage.py
```

在上面的代码块中，您可以看到`startproject`命令为您创建的文件夹结构:

*   **`setup/`** 是你的顶层项目文件夹。
*   **`setup/setup/`** 是你的下级文件夹，代表你的管理 app。
*   **`manage.py`** 是一个 Python 文件，作为你项目的指挥中心。它的作用与 [`django-admin`](https://docs.djangoproject.com/en/3.2/ref/django-admin/) 命令行实用程序相同。

嵌套的`setup/setup/`文件夹包含几个文件，当您在 web 应用程序上工作时，可以编辑这些文件。

**注意:**如果您想避免创建额外的顶层项目文件夹，您可以在`django-admin startproject`命令的末尾添加一个点(`.`):

```py
(env) $ django-admin startproject <projectname> .
```

圆点跳过顶层项目文件夹，在当前工作目录下创建管理应用程序和`manage.py`文件。您可能会在一些在线 Django 教程中遇到这种语法。它所做的只是创建项目脚手架，而没有额外的顶级项目文件夹。

花点时间探索一下`django-admin`命令行实用程序为您创建的默认项目框架。您将使用`startproject`命令创建的每个项目都将具有相同的结构。

当你准备好了，你可以继续创建一个 **Django 应用程序**作为你的新 web 应用程序的底层单元。

[*Remove ads*](/account/join/)

## 启动 Django 应用程序

用 Django 构建的每个项目都可以包含多个 Django 应用程序。当您在上一节中运行`startproject`命令时，您创建了一个管理应用程序，您将要构建的每个默认项目都需要它。现在，您将创建一个 Django 应用程序，它将包含您的 web 应用程序的特定功能。

您不再需要使用`django-admin`命令行实用程序，而是可以通过`manage.py`文件执行`startapp`命令:

```py
(env) $ python manage.py startapp <appname>
```

命令为 Django 应用程序生成一个默认的文件夹结构。本教程使用`example`作为应用程序的名称:

```py
(env) $ python manage.py startapp example
```

当您为您的个人 web 应用程序创建 Django 应用程序时，记得用您的应用程序名称替换`example`。

**注意:**如果您创建的项目没有上面提到的点快捷方式，那么在运行上面显示的命令之前，您需要将您的工作目录更改为您的顶级项目文件夹。

一旦`startapp`命令执行完毕，您将看到 Django 向您的文件夹结构中添加了另一个文件夹:

```py
setup/
│
├── example/
│   │
│   ├── migrations/
│   │   └── __init__.py
│   │
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   └── views.py
│
├── setup/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── manage.py
```

新文件夹具有您在运行命令时为其指定的名称。在本教程的情况下，那就是`example/`。您可以看到该文件夹包含几个 Python 文件。

这个 Django 应用程序文件夹是您创建 web 应用程序时花费大部分时间的地方。您还需要在管理应用程序`setup/`中进行一些更改，但是您将在 Django 应用程序`example/`中构建您的大部分功能。



导航到您的`example/`文件夹，打开新生成的 Python 文件。如果您还不理解 Django 为您生成的代码，请不要担心，并且请记住，您不需要*需要*来理解它来构建 Django web 应用程序。探索每个文件中的代码注释，看看它们是否有助于阐明每个文件的用例。

在学习教程或构建自己的项目时，您将更详细地了解生成的 Python 文件。以下是在 app 文件夹中创建的三个值得注意的文件:

1.  **`__init__.py`** : Python 使用这个文件将一个文件夹声明为一个[包](https://realpython.com/python-modules-packages/)，这允许 Django 使用不同应用程序的代码来组成您的 web 应用程序的整体功能。你可能不用碰这份文件。
2.  **`models.py`** :您将在这个文件中声明您的应用程序的模型，这允许 Django 与您的 web 应用程序的数据库接口。
3.  **`views.py`** :你将在这个文件中编写应用程序的大部分代码逻辑。

至此，您已经完成了 Django web 应用程序的搭建，可以开始实现您的想法了。从现在开始，你想要构建什么来创建你自己独特的项目就取决于你了。

## 命令参考

下表为您提供了启动 Django 开发过程所需命令的快速参考。参考表中的步骤链接回本教程的各个部分，在那里可以找到更详细的解释:

| 步骤 | 描述 | 命令 |
| --- | --- | --- |
| [1a](#prepare-your-environment) | 设置虚拟环境 | `python -m venv env` |
| [1b](#prepare-your-environment) | 激活虚拟环境 | `source env/bin/activate` |
| [2a](#install-django-and-pin-your-dependencies) | 安装 Django | `python -m pip install django` |
| [2b](#install-django-and-pin-your-dependencies) | 固定您的依赖关系 | `python -m pip freeze > requirements.txt` |
| [3](#set-up-a-django-project) | 建立 Django 项目 | `django-admin startproject <projectname>` |
| [4](#start-a-django-app) | 启动 Django 应用程序 | `python manage.py startapp <appname>` |

使用此表作为在 Python 虚拟环境中使用 Django 启动新 web 应用程序的快速参考。

[*Remove ads*](/account/join/)

## 结论

在本教程中，您了解了为新的 Django web 应用程序建立基础的所有必要步骤。您已经熟悉了最常见的[终端命令](#command-reference)，在使用 Django 进行 web 开发时，您会一遍又一遍地重复这些命令。

您还了解了*为什么*要使用每个命令以及它们产生的结果，并且学习了一些与设置 Django 相关的技巧和诀窍。

**在本教程中，您学习了如何:**

*   建立一个**虚拟环境**
*   安装 Django
*   锁定您的项目**依赖关系**
*   建立一个 Django **项目**
*   启动 Django **应用**

完成本教程中概述的步骤后，您就可以开始使用 Django 构建您的定制 web 应用程序了。例如，你可以创建一个[文件夹应用](https://realpython.com/get-started-with-django-1/)来展示你的编码项目。为了有条理地不断提高你的 Django 技能，你可以通过 [Django 学习路径](https://realpython.com/learning-paths/django-web-development/)中提供的资源继续学习。

继续为 Django web 应用程序搭建基础架构，直到这些步骤成为第二天性。如果您需要复习，那么您可以随时使用本教程作为快速参考。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**如何建立 Django 项目**](/courses/set-up-django-project/)*****