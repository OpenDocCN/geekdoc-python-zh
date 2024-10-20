# Django 中应用程序和项目之间的差异

> 原文：<https://pythonguides.com/django-app-vs-project/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在本 [Django 教程](https://pythonguides.com/what-is-python-django/)中，我们将尝试了解 Django 中 app 和项目的**区别。此外，我们还将在本教程中讨论以下主题。**

*   姜戈的项目是什么
*   Django 的 app 是什么
*   Django 中应用程序和项目之间的差异
*   Django 中 startapp 和 startproject 的区别

目录

[](#)

*   [Django 的项目是什么](#What_is_project_in_Django "What is project in Django ")
*   [Django 中的 app 是什么](#What_is_app_in_Django "What is app in Django ")
*   [Django 的应用程序和项目之间的差异](#Difference_between_app_and_project_in_Django "Difference between app and project in Django")
*   [Django 的 startapp 和 startproject 的区别](#Difference_between_startapp_and_startproject_in_Django "Difference between startapp and startproject in Django")
    *   [Django 中的 startproject 命令](#startproject_command_in_Django "startproject command in Django")
    *   Django 中的 startapp 命令

## Django 的项目是什么

*   Django 中的一个项目是一个代表整个 web 应用程序的 python 包。
*   Django 的一个项目基本上包含了整个网站相关的配置和设置。
*   一个项目也可以包含多个应用程序，用于实现某些功能。

当我们创建一个项目时，Django 会自动生成一个项目目录，其中包含一个 python 包。它还会在项目目录中创建一个 manage.py 文件。manage.py 是一个主要用于与项目交互的实用程序。

此外，我们还可以在不添加任何应用程序的情况下使用项目。但是，这将是对 Django 框架的利用不足。

阅读:[如何安装 Django](https://pythonguides.com/how-to-install-django/)

## DjangoT3 里的 app 是什么

*   Django 中的一个 app 是一个项目的子模块，用来实现一些功能。
*   现在，您可以将应用程序称为独立的 python 模块，用于为您的项目提供一些功能。
*   我们可以在一个 Django 项目中创建多个应用程序。这些应用程序可以相互独立。理论上，我们可以从一个 Django 项目到另一个项目使用一个应用程序，而不需要对它做任何修改。

现在，当我们在项目中创建一个应用程序时，Django 会自动创建一个自包含的目录。因此，开发人员可以专注于业务逻辑，而不是构建应用程序目录。

Django 框架的工作原理是 DRY(不要重复自己)，应用程序的概念是其中很大的一部分。

阅读[如何从 Django](https://pythonguides.com/get-data-from-get-request-in-django/) 中的 get 请求获取数据

## Djangoapp 与项目的区别

到目前为止，我们讨论了一个关于 Django 的项目和应用程序的基本介绍。现在，如果你想了解我们如何创建一个项目或应用程序，你可以参考这篇文章“[如何设置 Django 项目](https://pythonguides.com/setup-django-project/)”。

在这一节中，我们将尝试理解 Django 中 app 和 project 之间的一些关键差异。

*   一个项目代表整个网站，而一个应用程序基本上是项目的一个子模块。
*   一个项目可以包含多个应用程序，而一个应用程序也可以在不同的项目中使用。
*   项目就像整个 web 应用程序的蓝图，而应用程序是 web 应用程序的构建块。
*   我们通常会为我们的网站创建一个项目，其中包含一个或多个应用程序。
*   项目包含与整个 web 应用程序相关的配置和设置。另一方面，应用程序可以是独立的，也可以相互关联。

阅读: [Python Django vs Flask](https://pythonguides.com/python-django-vs-flask/)

## Django的 startapp 和 startproject 的区别

在本节中，我们将了解 startapp 和 startproject 命令之间的区别。这里是 Django 中这两个命令的一些不同之处。

### Django 中的 startproject 命令

startproject 命令是我们运行的第一个命令，它用于在 Django 中创建新项目。通过执行这个命令，Django 自动在给定的位置创建项目目录。

Django 中完整的 startproject 如下所示。

```py
django-admin startproject *`project_name`*
```

在上面的命令中，我们可以指定我们项目的名称来代替 `project_name` 。并且给定的名称也将用作项目目录名。

### Django 中的 startapp 命令

Django 中的 startapp 命令用于为我们的项目创建一个新的应用程序。现在，我们可以在项目中多次使用这个命令来创建多个应用程序。

通过执行这个命令，Django 自动在项目目录中创建应用程序目录。

现在，要执行这个命令，首先我们需要移动到项目目录，然后我们可以使用下面的语法。

```py
python manage.py startapp `a*pp_name*`
```

要执行 startapp 命令，我们必须使用 `manage.py` 实用程序，而不是 `django-admin` 。在语法中，我们可以指定我们的应用程序名称来代替**应用程序名称**。该名称也将用作应用程序目录名称。

你可能会喜欢读下面的文章。

*   [如何安装 Django](https://pythonguides.com/how-to-install-django/)
*   [在 Django 中创建模型](https://pythonguides.com/create-model-in-django/)
*   [如何安装 matplotlib](https://pythonguides.com/how-to-install-matplotlib-python/)
*   [Django 随机数](https://pythonguides.com/django-random-number/)
*   [Python Django 过滤器](https://pythonguides.com/python-django-filter/)
*   [Python Django 获取枚举选择](https://pythonguides.com/python-django-get-enum-choices/)

在本教程中，我们了解了 Django 中 app 和 project 的**区别。此外，在本文中，我们讨论了以下主题。**

*   姜戈的项目是什么
*   Django 的 app 是什么
*   Django 中应用程序和项目之间的差异
*   Django 中 startapp 和 startproject 的区别

![Bijay Kumar MVP](img/9cb1c9117bcc4bbbaba71db8d37d76ef.png "Bijay Kumar MVP")[Bijay Kumar](https://pythonguides.com/author/fewlines4biju/)

Python 是美国最流行的语言之一。我从事 Python 工作已经有很长时间了，我在与 Tkinter、Pandas、NumPy、Turtle、Django、Matplotlib、Tensorflow、Scipy、Scikit-Learn 等各种库合作方面拥有专业知识。我有与美国、加拿大、英国、澳大利亚、新西兰等国家的各种客户合作的经验。查看我的个人资料。

[enjoysharepoint.com/](https://enjoysharepoint.com/)[](https://www.facebook.com/fewlines4biju "Facebook")[](https://www.linkedin.com/in/fewlines4biju/ "Linkedin")[](https://twitter.com/fewlines4biju "Twitter")