# Python 应用程序布局:参考

> 原文：<https://realpython.com/python-application-layouts/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**构造一个 Python 应用**](/courses/structuring-python-application/)

Python 虽然在语法和风格上固执己见，但在构建应用程序时却异常灵活。

一方面，这种灵活性很大:它允许不同的用例使用那些用例所必需的结构。然而，另一方面，这可能会让新开发人员感到非常困惑。

互联网也帮不上什么忙——有多少 Python 博客就有多少观点。**在本文中，我想给你一个可靠的 Python 应用布局参考指南，你可以在绝大多数用例中参考。**

您将看到常见 Python 应用程序结构的示例，包括[命令行应用程序](https://realpython.com/python-command-line-arguments/) (CLI 应用程序)、一次性脚本、可安装包以及带有流行框架的 web 应用程序布局，如 [Flask](https://realpython.com/tutorials/flask/) 和 [Django](https://realpython.com/tutorials/django/) 。

**注意:**本参考指南假设读者具备 Python 模块和包的工作知识。如果你感到有些生疏，请查看我们的[Python 模块和包的介绍](https://realpython.com/python-modules-packages/)。

## 命令行应用程序布局

我们很多人主要使用通过[命令行界面(CLIs)](https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/) 运行的 Python 应用程序。这是您经常从空白画布开始的地方，Python 应用程序布局的灵活性确实令人头疼。

从一个空的项目文件夹开始可能会令人生畏，并导致不缺少编码人员。在这一节中，我想分享一些我个人用来作为所有 Python CLI 应用程序起点的经过验证的布局。

我们将从一个非常基本的用例的非常基本的布局开始:一个独立运行的简单脚本。然后，随着用例的推进，您将看到如何构建布局。

[*Remove ads*](/account/join/)

### 一次性脚本

你只是做了一个脚本，而且是肉汁，对不对？无需安装——只需在其目录中运行脚本即可！

嗯，如果你只是制作一个供自己使用的脚本，或者一个没有任何外部[依赖](https://realpython.com/courses/managing-python-dependencies/)的脚本，这很好，但是如果你必须分发它呢？尤其是对一个不太懂技术的用户？

以下布局适用于所有这些情况，并且可以很容易地进行修改，以反映您在工作流程中使用的任何安装或其他工具。无论您是创建一个纯 Python 脚本(也就是说，一个没有依赖关系的脚本)，还是使用像 [pip](https://pypi.org/project/pip/) 或 [Pipenv](https://realpython.com/pipenv-guide/) 这样的工具，这个布局都将涵盖您。

阅读本参考指南时，请记住文件在布局中的确切位置比它们被放置在何处的原因更重要。所有这些文件都应该位于以您的项目命名的项目目录中。对于这个例子，我们将使用(还有什么？)`helloworld`作为项目名称和根目录。

以下是我通常用于 CLI 应用程序的 Python 项目结构:

```py
helloworld/
│
├── .gitignore
├── helloworld.py
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── tests.py
```

这非常简单:所有东西都在同一个目录中。这里显示的文件不一定是详尽的，但是如果您计划使用这样的基本布局，我建议将文件数量保持在最小。其中一些文件对您来说可能是新的，所以让我们快速看一下它们各自的功能。

*   这是一个文件，它告诉 Git 应该忽略哪些类型的文件，比如 IDE 文件或者本地配置文件。[我们的 Git 教程有所有的细节](https://realpython.com/python-git-github-intro/#gitignore)，你可以在这里找到 Python 项目[的样本`.gitignore`文件。](https://github.com/github/gitignore)

*   这是你正在分发的脚本。至于主脚本文件的命名，我建议您使用项目的名称(与顶级目录的名称相同)。

*   这个明文文件描述了你在一个项目中使用的许可证。如果您正在分发代码，拥有一个总是一个好主意。按照惯例，文件名全部大写。

    > **注意:**需要帮助为您的项目选择许可证吗？查看[选择许可](https://choosealicense.com/)。

*   `README.md`:这是一个 [Markdown](https://en.wikipedia.org/wiki/Markdown) (或 [reStructuredText](https://en.wikipedia.org/wiki/ReStructuredText) )文件，记录了你的应用程序的目的和用途。制作好的`README`是一门艺术，但是你可以在这里找到掌握[的捷径](https://dbader.org/blog/write-a-great-readme-for-your-github-project)。

*   这个文件为你的应用程序定义了外部的 Python 依赖和它们的版本。

*   这个文件也可以用来定义依赖项，但是它更适合安装过程中需要完成的其他工作。你可以在我们的[T4 指南中了解更多关于`setup.py`和`requirements.txt`的信息。](https://realpython.com/pipenv-guide/)

*   这个脚本包含了你的测试，如果你有的话。你应该来点 T2。

但是现在您的应用程序正在增长，并且您已经将它分成了同一个包中的多个部分，那么您应该将所有部分都放在顶层目录中吗？既然您的应用程序变得更加复杂，是时候更干净地组织事情了。

### 可安装的单个软件包

让我们假设`helloworld.py`仍然是要执行的主脚本，但是您已经将所有的助手方法移到了一个名为`helpers.py`的新文件中。

我们将把`helloworld` Python 文件打包在一起，但是将所有其他文件，比如你的`README`、`.gitignore`等等，放在顶层目录中。

让我们来看看更新后的结构:

```py
helloworld/
│
├── helloworld/
│   ├── __init__.py
│   ├── helloworld.py
│   └── helpers.py
│
├── tests/
│   ├── helloworld_tests.py
│   └── helpers_tests.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

这里唯一的不同是，您的应用程序代码现在都保存在`helloworld`子目录中——该目录以您的包命名——并且我们添加了一个名为`__init__.py`的文件。让我们来介绍一下这些新文件:

*   这个文件有很多功能，但是为了我们的目的，它告诉 Python 解释器这个目录是一个包目录。您可以设置这个`__init__.py`文件，使您能够从包中整体导入类和方法，而不是知道内部模块结构并从`helloworld.helloworld`或`helloworld.helpers`导入。

    > **注意:**关于内部包的更深入的讨论`__init__.py`，[我们的 Python 模块和包概述](https://realpython.com/python-modules-packages/)已经介绍过了。

*   `helloworld/helpers.py`:如上所述，我们已经将`helloworld.py`的大部分业务逻辑移到了这个文件中。多亏了`__init__.py`，外部模块将能够简单地通过从`helloworld`包导入来访问这些助手。

*   我们已经将我们的测试转移到它们自己的目录中，随着我们程序结构变得越来越复杂，你会继续看到这种模式。我们还将测试分成独立的模块，反映了我们的包的结构。

这个布局是 Kenneth Reitz 的 samplemod 应用程序结构的精简版本。这是您的 CLI 应用程序的另一个很好的起点，尤其是对于更大的项目。

### 带内部包的应用程序

在较大的应用程序中，您可能有一个或多个内部包，这些包或者与主 runner 脚本绑定在一起，或者为您正在打包的较大的库提供特定的功能。我们将扩展上述约定以适应这种情况:

```py
helloworld/
│
├── bin/
│
├── docs/
│   ├── hello.md
│   └── world.md
│
├── helloworld/
│   ├── __init__.py
│   ├── runner.py
│   ├── hello/
│   │   ├── __init__.py
│   │   ├── hello.py
│   │   └── helpers.py
│   │
│   └── world/
│       ├── __init__.py
│       ├── helpers.py
│       └── world.py
│
├── data/
│   ├── input.csv
│   └── output.xlsx
│
├── tests/
│   ├── hello
│   │   ├── helpers_tests.py
│   │   └── hello_tests.py
│   │
│   └── world/
│       ├── helpers_tests.py
│       └── world_tests.py
│
├── .gitignore
├── LICENSE
└── README.md
```

这里有更多的东西需要消化，但是只要你记得它是从前面的布局开始的，你就会更容易理解。我将按顺序介绍添加和修改的内容，它们的用途，以及您可能需要它们的原因。

*   `bin/`:该目录保存所有可执行文件。我改编自[让-保罗·卡尔德龙的经典结构文章](http://as.ynchrono.us/2007/12/filesystem-structure-of-python-project_21.html)，他关于使用`bin/`目录的建议仍然很重要。要记住的最重要的一点是，你的可执行文件不应该有很多代码，只是导入和调用你的 runner 脚本中的一个[主函数](https://realpython.com/python-main-function/)。如果你使用的是纯 Python 或者没有任何可执行文件，你可以省去这个目录。

*   对于一个更高级的应用程序，你会想要维护其所有部分的良好文档。我喜欢把内部模块的文档放在这里，这就是为什么你会看到`hello`和`world`包的单独文档。如果你在内部模块中使用[文档字符串](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)(你应该这样做！)，您的整个模块文档至少应该给出模块的目的和功能的整体视图。

*   `helloworld/`:这个类似于之前结构中的`helloworld/`，但是现在有子目录了。随着复杂性的增加，您会希望使用“分而治之”的策略，将部分应用程序逻辑分割成更易于管理的块。记住，目录名指的是整个包名，因此子目录名(`hello/`和`world/`)应该反映它们的包名。

*   `data/`:有这个目录有助于测试。它是您的应用程序将接收或生成的任何文件的中心位置。根据您如何部署您的应用程序，您可以保持“生产级”输入和输出指向这个目录，或者仅将其用于内部测试。

*   在这里，你可以放置所有的测试——单元测试、执行测试、集成测试等等。对于您的测试策略、导入策略等等，请随意以最方便的方式构建这个目录。要复习用 Python 测试命令行应用程序，请查看我的文章 [4 测试 Python 命令行(CLI)应用程序的技术](https://realpython.com/python-cli-testing/)。

顶层文件在很大程度上与之前的布局相同。这三种布局应该涵盖了命令行应用程序的大多数用例，甚至包括 GUI 应用程序，但要注意的是，您可能需要根据所使用的 GUI 框架来修改一些东西。

**注意:**记住这些只是布局。如果一个目录或文件对你的特定用例没有意义(比如`tests/`如果你没有用你的代码分发测试)，请随意删除它。但是尽量不要漏掉`docs/`。记录你的工作总是一个好主意。

[*Remove ads*](/account/join/)

## 网络应用布局

Python 的另一个主要用例是 [web 应用](https://realpython.com/python-web-applications/)。 [Django](https://www.djangoproject.com/) 和 [Flask](http://flask.pocoo.org/) 可以说是 Python 最流行的 web 框架，谢天谢地，它们在应用程序布局方面更加固执己见。

为了确保这篇文章是一个完整的、成熟的布局参考，我想强调这些框架共有的结构。

### 姜戈

我们按字母顺序来，从姜戈开始。Django 的一个优点是，它会在运行`django-admin startproject project`后为您创建一个项目框架，其中`project`是您的项目名称。这将在您当前的工作目录中创建一个名为`project`的目录，其内部结构如下:

```py
project/
│
├── project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── manage.py
```

这似乎有点空洞，不是吗？所有的逻辑都去哪里了？风景呢？甚至没有任何测试！

在 Django，这是一个项目，它将 Django 的另一个概念 apps 联系在一起。应用程序是逻辑、模型、视图等等都存在的地方，在这样做的时候，它们完成一些任务，比如维护一个博客。

Django 应用程序可以导入到项目中并在项目间使用，其结构类似于专门的 Python 包。

像项目一样，Django 使得生成 Django 应用程序布局变得非常容易。设置好项目后，你所要做的就是导航到`manage.py`的位置并运行`python manage.py startapp app`，其中`app`是你的应用程序的名称。

这将产生一个名为`app`的目录，其布局如下:

```py
app/
│
├── migrations/
│   └── __init__.py
│
├── __init__.py
├── admin.py
├── apps.py
├── models.py
├── tests.py
└── views.py
```

这可以直接导入到您的项目中。关于这些文件做什么、如何在你的项目中利用它们等等的细节超出了本参考的范围，但是你可以在我们的 Django 教程和 Django 官方文档中获得所有这些信息和更多信息[。](https://realpython.com/learn/start-django/)

这个文件和文件夹结构非常简单，是 Django 的基本要求。对于任何开源 Django 项目，您可以(也应该)从命令行应用程序布局中调整结构。在外层的`project/`目录中，我通常以这样的方式结束:

```py
project/
│
├── app/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   │
│   ├── migrations/
│   │   └── __init__.py
│   │
│   ├── models.py
│   ├── tests.py
│   └── views.py
│
├── docs/
│
├── project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── static/
│   └── style.css
│
├── templates/
│   └── base.html
│
├── .gitignore
├── manage.py
├── LICENSE
└── README.md
```

关于更高级的 Django 应用程序布局的更深入的讨论，[这个堆栈溢出线程](https://stackoverflow.com/questions/22841764/best-practice-for-django-project-working-directory-structure)已经介绍过了。 [django-project-skeleton](http://django-project-skeleton.readthedocs.io/en/latest/structure.html) 项目文档解释了你会在堆栈溢出线程中找到的一些目录。对 Django 的全面深入可以在[的 Django 的两个独家新闻](https://realpython.com/asins/0692915729)中找到，这将教你所有 Django 开发的最新最佳实践。

更多 Django 教程，请访问我们在 Real Python 的 Django 部分。

### 烧瓶

Flask 是一个 Python web“微框架”一个主要的卖点是，它可以很快地以最小的开销建立起来。 [Flask 文档](http://flask.pocoo.org/docs/1.0/)中有一个 web 应用程序示例，它只有 10 行代码，在一个脚本中。当然，在实践中，编写这么小的 web 应用程序是不太可能的。

幸运的是，Flask 文档[为我们节省了](http://flask.pocoo.org/docs/1.0/tutorial/layout/),为他们的教程项目(一个名为 Flaskr 的博客 web 应用程序)提供了一个建议的布局，我们将在这里从主项目目录中检查它:

```py
flaskr/
│
├── flaskr/
│   ├── ___init__.py
│   ├── db.py
│   ├── schema.sql
│   ├── auth.py
│   ├── blog.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── auth/
│   │   │   ├── login.html
│   │   │   └── register.html
│   │   │
│   │   └── blog/
│   │       ├── create.html
│   │       ├── index.html
│   │       └── update.html
│   │ 
│   └── static/
│       └── style.css
│
├── tests/
│   ├── conftest.py
│   ├── data.sql
│   ├── test_factory.py
│   ├── test_db.py
│   ├── test_auth.py
│   └── test_blog.py
│
├── venv/
│
├── .gitignore
├── setup.py
└── MANIFEST.in
```

从这些内容中，我们可以看到 Flask 应用程序和大多数 Python 应用程序一样，是围绕 Python 包构建的。

**注:**没看见？发现包的一个快速提示是通过寻找一个`__init__.py`文件。它位于该特定包的最高层目录中。在上面的布局中，`flaskr`是一个包含`db`、`auth`和`blog`模块的包。

在这个布局中，除了您的测试、一个用于您的[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)的目录和您通常的顶层文件之外，所有东西都存在于`flaskr`包中。与其他布局一样，您的测试将大致匹配驻留在`flaskr`包中的各个模块。您的模板也驻留在主项目包中，这在 Django 布局中是不会发生的。

请务必访问我们的 [Flask 样板 Github 页面](https://github.com/realpython/flask-boilerplate)，查看更完整的 Flask 应用程序，并在这里查看样板文件[。](http://www.flaskboilerplate.com/)

关于 Flask 的更多信息，请点击这里查看我们所有的 Flask 教程。

[*Remove ads*](/account/join/)

## 结论和提醒

现在您已经看到了许多不同应用程序类型的示例布局:一次性 Python 脚本、可安装的单个包、带有内部包的大型应用程序、Django web 应用程序和 Flask web 应用程序。

根据本指南，您将拥有通过构建您的应用程序结构来成功防止编码障碍的工具，这样您就不会盯着一张空白的画布试图找出从哪里开始。

因为 Python 在应用程序布局方面很大程度上是没有主见的，所以您可以根据自己的意愿定制这些示例布局，以更好地适应您的用例。

我希望你不仅有一个应用程序布局参考，而且理解这些例子既不是一成不变的规则，也不是构建应用程序的唯一方法。随着时间的推移和实践，您将能够构建和定制自己有用的 Python 应用程序布局。

我错过了一个用例吗？你有另一种应用结构哲学吗？这篇文章有助于防止编码器的阻塞吗？请在评论中告诉我！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**构造一个 Python 应用**](/courses/structuring-python-application/)*****