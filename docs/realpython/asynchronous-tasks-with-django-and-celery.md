# Django 和 Celery 的异步任务

> 原文：<https://realpython.com/asynchronous-tasks-with-django-and-celery/>

你已经构建了一个闪亮的 **Django** 应用程序，并希望将其发布给公众，但你担心应用程序工作流中的时间密集型任务。你不希望你的用户在浏览你的应用时有负面体验。你可以整合**芹菜**来帮助解决这个问题。

Celery 是 UNIX 系统的一个分布式任务队列。它允许您从 Python 应用程序中卸载工作。一旦你将芹菜整合到你的应用程序中，你就可以将时间密集型任务发送到芹菜的任务队列中。这样，当 Celery 在后台异步完成昂贵的操作时，您的 web 应用程序可以继续快速响应用户。

**在本教程中，您将学习如何:**

*   识别芹菜的**有效用例**
*   区分**芹菜工**和**芹菜工**
*   在 Django 项目中整合芹菜和 Redis
*   设置独立于 Django 应用运行的异步任务
*   重构 Django 代码，改为用芹菜运行任务

如果你以前从未在 Django 应用程序中使用过 Celery，或者如果你偷看了 Celery 的文档却找不到路，那么你就来对地方了。您将了解使用 Django 和 Celery 开始运行异步任务所需的一切。

您将把芹菜集成到现有的 Django 应用程序中。继续下载该应用程序的代码，这样您就可以继续学习了:

**源代码:** [点击此处下载源代码](https://realpython.com/bonus/django-celery-code/)，您将使用它将芹菜集成到您的 Django 应用程序中。

## Python 芹菜基础知识

[Celery](https://docs.celeryq.dev/en/stable/index.html) 是一个分布式任务[队列](https://realpython.com/queue-in-python/)，它可以收集、记录、调度和执行主程序之外的任务。

**注意:** Celery [在版本 4 中放弃了对 Windows](https://docs.celeryq.dev/en/stable/faq.html?highlight=windows#does-celery-support-windows) 的支持，所以虽然[你可能仍然能够让它在 Windows 上工作](https://stackoverflow.com/a/47331438/5717580)，但你最好使用不同的任务队列，比如 [huey](https://huey.readthedocs.io/en/latest/index.html) 或 [Dramatiq](https://dramatiq.io/index.html) 。

在本教程中，您将专注于在 UNIX 系统上使用 Celery，所以如果您试图在 *Windows* 上建立一个分布式任务队列，那么本教程可能不适合您。

为了从程序接收任务并将结果发送到后端，Celery 需要一个[消息代理](https://en.wikipedia.org/wiki/Message_broker)来进行通信。 **Redis** 和 **RabbitMQ** 是开发人员经常与芹菜一起使用的两个消息代理。

在本教程中，您将使用 [Redis](https://redis.io/) 作为消息代理。为了挑战你自己，你可以偏离指令，使用 [RabbitMQ](https://www.rabbitmq.com/) 作为消息代理。

如果您想跟踪任务运行的结果，那么您还需要建立一个**结果后端**数据库。

**注意:**将 Celery 连接到结果后端是可选的。一旦你指示 Celery 运行一个任务，不管你是否跟踪任务结果，它都会履行它的职责。

但是，记录所有任务的结果通常是有帮助的，尤其是当您将任务分配到多个队列时。为了持久保存任务结果的信息，您需要一个数据库后端。

您可以使用许多不同的数据库来跟踪芹菜任务的结果。在本教程中，您将使用 [Redis](https://realpython.com/python-redis/) 作为消息代理和结果后端。通过使用 Redis，您可以限制需要安装的依赖项，因为它可以承担两种角色。

在本教程的范围内，您不会对记录的任务结果做任何工作。但是，下一步，您可以使用 Redis 命令行界面(CLI)检查结果，或者将信息放入 Django 项目中的专用页面。

[*Remove ads*](/account/join/)

### 为什么要用芹菜？

大多数开发人员想开始使用芹菜有两个主要原因:

1.  **将工作**从你的应用卸载到可以独立于你的应用运行的分布式进程
2.  **调度任务在特定时间执行**，有时作为重复发生的事件

芹菜是这两种用例的绝佳选择。它将自己定义为“一个专注于实时处理的任务队列，同时还支持任务调度”( [Source](https://docs.celeryq.dev/en/stable/index.html) )。

尽管这两个功能都是芹菜的一部分，但它们通常是分开处理的:

1.  Celery workers 是在主服务上下文之外独立运行任务的工作进程。
2.  **Celery beat** 是一个调度器，协调何时运行任务。您也可以使用它来安排定期任务。

芹菜工是芹菜的主心骨。即使你的目标是使用 Celery beat 安排周期性任务，芹菜工人也会在计划的时间收到你的指令并处理它们。芹菜节拍为芹菜工人添加了一个基于时间的调度程序。

在本教程中，您将学习如何将 Celery 与 Django 集成在一起，使用 Celery workers 从应用程序的主执行线程中异步执行操作。

在本教程中，您不会使用 Celery beat 处理任务调度，但是一旦您理解了 Celery 任务的基本知识，您将能够使用 Celery beat 设置[周期性任务](https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html)。

### 你如何为你的 Django 应用程序利用芹菜？

芹菜不仅对 web 应用程序有用，而且在这种情况下也很受欢迎。这是因为您可以通过使用分布式任务队列(如 Celery)来有效地处理 web 开发中的一些日常情况:

*   **电子邮件发送:**您可能想要发送电子邮件验证、密码重置电子邮件或表单提交确认。发送电子邮件可能需要一段时间，会降低应用程序的速度，尤其是当它有很多用户时。

*   **图像处理:**您可能想要调整用户上传的头像图像的大小，或者对用户可以在您的平台上共享的所有图像应用一些编码。图像处理通常是一项资源密集型任务，会降低您的 web 应用程序的速度，主要是在您为大型用户社区提供服务的情况下。

*   **文本处理:**如果你允许用户向你的应用添加数据，那么你可能需要监控他们的输入。例如，您可能希望检查评论中的亵渎内容，或者将用户提交的文本翻译成不同的语言。在 web 应用程序的上下文中处理所有这些工作会显著降低性能。

*   **API 调用和其他 web 请求:**如果您需要发出 web 请求来提供您的应用程序所提供的服务，那么您很快就会遇到意想不到的等待时间。对于速率受限的 API 请求和其他任务来说都是如此，比如[网络抓取](https://realpython.com/beautiful-soup-web-scraper-python/)。将这些请求交给不同的流程通常更好。

*   **数据分析:**处理数据是出了名的资源密集型。如果你的 web 应用程序为你的用户分析数据，如果你在 Django 中处理所有的工作，你会很快发现你的应用程序变得没有响应。

*   **机器学习模型运行:**与其他数据分析一样，等待机器学习操作的结果可能需要一段时间。不要让你的用户等待计算完成，你可以把这项工作交给 Celery，这样他们就可以继续浏览你的 web 应用，直到结果出来。

*   **报告生成:**如果你的应用程序允许用户根据他们提供的数据生成报告，你会注意到构建 PDF 文件并不是一蹴而就的。如果你让 Celery 在后台处理这个问题，而不是冻结你的 web 应用程序，直到报告可供下载，这会是一个更好的用户体验。

所有这些不同用例的主要设置都是相似的。一旦您理解了如何将计算密集型或时间密集型流程交给分布式任务队列，您就可以解放 Django 来处理 [HTTP 请求-响应周期](https://realpython.com/python-web-applications/#review-the-http-request-response-cycle)。

在本教程中，您将处理电子邮件发送场景。您将从 Django 同步处理电子邮件发送的项目开始。您将测试如何冻结您的 Django 应用程序。然后，您将学习如何将任务交给 Celery，以便体验如何让您的 web 应用程序响应更快。

## 将芹菜和姜戈融合在一起

既然您已经知道了什么是 Celery 以及它如何帮助您提高 web 应用程序的性能，那么是时候集成它了，这样您就可以使用 Celery 运行异步任务了。

您将专注于将 Celery 集成到现有的 Django 项目中。您将从一个精简的 Django 应用程序开始，该应用程序有一个最小的用例:收集用户反馈并发送一封电子邮件作为回复。

### 设置反馈应用程序

首先下载所提供的反馈应用程序的源代码:

**源代码:** [点击此处下载源代码](https://realpython.com/bonus/django-celery-code/)，您将使用它将芹菜集成到您的 Django 应用程序中。

解压缩下载的文件并使用您的终端导航到`source_code_initial/`目录，在那里您应该看到一个标准的 [Django 项目文件夹结构](https://realpython.com/django-setup/):



```py
source_code_initial/
│
├── django_celery/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── feedback/
│   │
│   ├── migrations/
│   │   └── __init__.py
│   │
│   ├── templates/
│   │   │
│   │   └── feedback/
│   │       ├── base.html
│   │       ├── feedback.html
│   │       └── success.html
│   │
│   │
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
│
├── .gitignore
├── manage.py
└── requirements.txt
```

确认你在`source_code_initial/`里面，然后创建并激活一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/):

```py
$ python -m venv venv
$ source venv/bin/activate
(venv) $
```

一旦您的虚拟环境被激活，您就可以安装 Django:

```py
(venv) $ python -m pip install django
```

通过运行迁移并启动开发服务器，完成 Django 应用程序的本地设置:

```py
(venv) $ python manage.py migrate
(venv) $ python manage.py runserver
```

现在，您可以打开浏览器，导航至该应用的主页`https://localhost:8000`，一个友好的反馈表格将会出现:

[![A blank feedback form](img/a50958d4ab42684467ba1c8c6d1d70f1.png)](https://files.realpython.com/media/celery-django-feedback-form.3544648b11eb.png)

然而，目前只有*的反馈表看起来*友好。继续填写表格并提交一些反馈。想象一下，您的 web 应用程序的一个用户会遇到这样的情况:

[https://player.vimeo.com/video/728157481](https://player.vimeo.com/video/728157481)

按下*提交*按钮后，应用程序会冻结。您可以在浏览器选项卡中看到旋转的小旋转符号，但是页面没有响应，您仍然可以看到您输入到表单中的所有信息。

Django 处理表单并把你重定向到成功页面花费了太长的时间！

Django 冻结是因为它需要在处理下一个任务之前同步处理电子邮件发送请求，下一个任务是将用户重定向到成功页面。

对于*来说，冻结这么长时间*的原因是因为`.send_email()`中一个偷偷摸摸的`time.sleep()`调用，它模拟了一个可能与电子邮件发送相关的时间或工作密集型任务。

当然，在实际应用中，你不会通过让 Django 休眠来给你的代码增加更多的时间延迟。然而，不幸的是，无论你使用什么样的电子邮件服务，都会给你带来一些延迟。尤其是一旦你的应用程序开始为许多用户服务，你很快就会遇到限制。

**注意:**将`time.sleep()`通话替换为你在网络应用中为服务用户而需要执行的任何工作密集型流程。

您的 Django 应用程序不应该同步处理长时间运行的任务，因为这样做会损害您的应用程序的用户体验和整体有用性。

相反，您将学习如何将这项任务交给芹菜工人。芹菜工人可以把计算作为后台任务来处理，让你的用户继续心满意足地浏览你时髦的网络应用。

[*Remove ads*](/account/join/)

### 安装芹菜作为你的任务队列

既然你[设置了反馈应用](#set-up-the-feedback-app)并感受到了电子邮件发送带来的延迟，你开始着手改善用户体验。

将 Celery 集成到 Django 应用程序中的第一步是将 Celery 包安装到虚拟环境中:

```py
(venv) $ python -m pip install celery
```

然而，仅仅安装芹菜是不够的。如果您尝试[运行任务队列](https://docs.celeryq.dev/en/stable/userguide/workers.html#starting-the-worker)，您会注意到 Celery 开始时似乎运行良好，但随后显示一条错误消息，指出 Celery 找不到消息代理:

```py
(venv) $ python -m celery worker
[ERROR/MainProcess] consumer: Cannot connect to
⮑ amqp://guest:**@127.0.0.1:5672//: [Errno 61] Connection refused.
Trying again in 2.00 seconds... (1/100)
```

Celery 需要一个消息代理来与向任务队列发送任务的程序通信。没有代理，Celery 无法接收指令，这就是它不断尝试重新连接的原因。

**注意:**您可能会注意到芹菜尝试连接的目标中类似 URL 的语法。协议名`amqp`，代表[高级消息队列协议](https://en.wikipedia.org/wiki/Advanced_Message_Queuing_Protocol)，是 Celery 使用的消息协议。最著名的本地实现 AMQP 的项目是 RabbitMQ，但是 Redis 也可以使用该协议进行通信。

在使用 Celery 之前，您需要安装一个消息代理，并将一个项目定义为消息生产者。在您的情况下，生产者是您的 Django 应用程序，消息代理将是 Redis。

### 安装 Redis 作为您的芹菜经纪人和数据库后端

您需要一个消息代理，以便芹菜可以与您的任务生产者沟通。您将使用 Redis，因为 Redis 可以同时充当消息代理和数据库后端。

回到您的终端，在您的系统上安装 Redis:

*   [*Linux*](#linux-1)
**   [*macOS*](#macos-1)*

```py
(venv) $ sudo apt update
(venv) $ sudo apt install redis
```

```py
(venv) $ brew install redis
```

要成功运行这个命令，您需要安装的 [Homebrew。](https://brew.sh/)

安装完成后，您可以启动 Redis 服务器来确认一切正常。打开一个新的终端窗口来启动服务器:

```py
$ redis-server
```

该窗口将是 Redis 的专用终端窗口。在本教程的剩余部分保持打开状态。

**注意:**运行`redis-server`启动 Redis 服务器。您将 Redis 作为一个独立于 Python 的进程运行，因此在启动它时不需要激活您的虚拟环境。

运行`redis-server`后，您的终端将显示 Redis 标识为 ASCII art，以及一些启动日志消息。最新的日志消息将告诉您 Redis 已经准备好接受连接。

要测试与 Redis 服务器的通信是否正常，请在另一个新的终端窗口中启动 Redis CLI:

```py
$ redis-cli
```

一旦提示发生变化，可以输入`ping`并按下 `Enter` ，然后等待 Redis 的回答:

```py
127.0.0.1:6379> ping
PONG
127.0.0.1:6379>
```

在用`redis-cli`启动 Redis CLI 后，您将单词`ping`发送到 Redis 服务器，服务器用权威的`PONG`进行响应。如果您得到了这个响应，那么您的 Redis 安装是成功的，芹菜将能够与 Redis 通信。

在进入下一步之前，通过按 `Ctrl` + `C` 退出 Redis CLI。

接下来，您需要一个 Python 客户机来与 Redis 接口。确认您在一个终端窗口中，其中您的虚拟环境仍然是活动的，然后安装 [redis-py](https://github.com/redis/redis-py) :

```py
(venv) $ python -m pip install redis
```

这个命令不会在您的系统上安装 Redis，而只是提供了一个用于连接 Redis 的 Python 接口。

**注意:**你需要在你的系统上安装 Redis*和* redis-py 在你的 Python 虚拟环境中，这样你就可以在你的 Python 程序中使用 Redis。

完成这两个安装后，您就成功地设置了消息代理。然而，你还没有把你的生产商和芹菜联系起来。

如果您现在尝试启动 Celery 并通过传递[`-A`选项](https://docs.celeryq.dev/en/stable/reference/cli.html#cmdoption-celery-A)和您的 Django 应用程序名称(`django_celery`)来包含生产者应用程序名称，您将遇到另一个错误:

```py
(venv) $ python -m celery -A django_celery worker
...
Error: Invalid value for '-A' / '--app':
Unable to load celery application.
Module 'django_celery' has no attribute 'celery'
```

到目前为止，您的分布式任务队列不能接收来自 Django 应用程序的消息，因为在您的 Django 项目中没有设置 Celery 应用程序。

在下一节中，您将向 Django 应用程序添加必要的代码，以便它可以作为芹菜的任务生成器。

[*Remove ads*](/account/join/)

### 将芹菜添加到 Django 项目中

最后一个难题是将 Django 应用程序作为消息生产者连接到您的任务队列。您将从提供的项目代码开始，所以如果您还没有这样做，请继续下载它:

**源代码:** [点击此处下载源代码](https://realpython.com/bonus/django-celery-code/)，您将使用它将芹菜集成到您的 Django 应用程序中。

一旦您的计算机上有了项目代码，导航到`django_celery`管理应用程序文件夹并创建一个名为`celery.py`的新文件:

```py
django_celery/
├── __init__.py
├── asgi.py
├── celery.py ├── settings.py
├── urls.py
└── wsgi.py
```

[Celery 推荐](https://docs.celeryq.dev/en/stable/django/first-steps-with-django.html#using-celery-with-django)使用该模块定义 Celery 应用实例。在您最喜欢的[文本编辑器或 IDE](https://realpython.com/python-ides-code-editors-guide/) 中打开文件，并添加必要的代码:

```py
 1# django_celery/celery.py
 2
 3import os
 4from celery import Celery
 5
 6os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_celery.settings")
 7app = Celery("django_celery")
 8app.config_from_object("django.conf:settings", namespace="CELERY")
 9app.autodiscover_tasks()
```

您只需要将这几行代码添加到文件中。请继续阅读，了解它们各自的功能:

*   **第 3 行:**你导入内置的`os`模块，你可能从[处理文件](https://realpython.com/working-with-files-in-python/)中熟悉这个模块。您将在第 6 行使用它来设置一个环境变量。

*   **第 4 行:**你从`celery`包中导入`Celery`。您将在第 7 行使用它来创建您的[芹菜应用程序实例](https://docs.celeryq.dev/en/stable/reference/celery.html#celery.Celery)。

*   **第六行:**你用[。setdefault()](https://docs.python.org/3/library/stdtypes.html#dict.setdefault) of [`os.environ`](https://docs.python.org/3/library/os.html#os.environ) 以确保您的 Django 项目的`settings.py`模块可以通过`"DJANGO_SETTINGS_MODULE"`键访问。

*   **第 7 行:**您创建了 [Celery 应用程序实例](https://docs.celeryq.dev/en/stable/reference/celery.html#celery.Celery)，并提供了主模块的名称作为参数。在您的 Django 应用程序的上下文中，主模块是包含`celery.py`的 Django 应用程序，因此您通过了`"django_celery"`。

*   **第 8 行:**您将 Django 设置文件定义为芹菜的配置文件，并提供一个名称空间`"CELERY"`。您需要为每个与 Celery 相关的配置变量预定义名称空间值，后面跟一个下划线(`_`)。您可以定义一个不同的设置文件，但是将芹菜配置保存在 Django 的设置文件中可以让您坚持在一个中心位置进行配置。

*   **第 9 行:**您告诉您的 Celery 应用程序实例自动查找 Django 项目的每个应用程序中的所有任务。只要你坚持使用[可重用应用](https://docs.djangoproject.com/en/dev/intro/reusable-apps/)的结构，并在一个专用的`tasks.py`模块中为一个应用定义所有的芹菜任务，这就行得通。稍后当您[重构电子邮件发送代码](#refactor-the-code-as-a-celery-task)时，您将为您的`django_celery`应用程序创建并填充这个文件。

设置好`celery.py`并尝试从`settings.py`文件中获取必要的芹菜设置后，接下来您将前往`settings.py`将这些设置条目添加到文件的底部:

```py
# django_celery/settings.py

# ...

# Celery settings
CELERY_BROKER_URL = "redis://localhost:6379"
CELERY_RESULT_BACKEND = "redis://localhost:6379"
```

这两个条目为您的 Celery 应用程序实例提供了足够的信息，以知道向哪里发送消息以及在哪里记录结果。因为您使用 Redis 作为消息代理和数据库后端，所以两个 URL 指向同一个地址。

**注意:**这些 URL 也可以指向不同的服务器和服务。例如，您可以[使用 RabbitMQ](https://docs.celeryq.dev/en/latest/getting-started/backends-and-brokers/rabbitmq.html) 作为您的消息代理，使用 Redis 作为您的结果后端:

```py
CELERY_BROKER_URL = "amqp://myuser:mypassword@localhost:5672/myvhost"
CELERY_RESULT_BACKEND = "redis://localhost:6379"
```

当您在生产中运行应用程序时，您将用每个服务的生产位置替换这些 URL。

请注意这些设置变量开头的`CELERY_`名称空间。您需要添加这个，因为您在`celery.py`的第 8 行传递给`app.config_from_object()`的`namespace="CELERY"`参数。

至此，您已经基本完成了将芹菜集成到您的 web 应用程序中。最后添加到您的管理应用程序的`__init__.py`:

```py
django_celery/
├── __init__.py ├── asgi.py
├── celery.py
├── settings.py
├── urls.py
└── wsgi.py
```

在文本编辑器中打开文件。在一个默认的 Django 项目中，每个 app 文件夹都有一个`__init__.py`文件，这有助于将其标记为一个模块。默认情况下，该文件为空，但是您可以添加代码来影响导入行为。

为了确保在启动 Django 时加载了芹菜应用程序，您应该将它添加到 [`__all__`](https://docs.python.org/3/tutorial/modules.html#importing-from-a-package) :

```py
# django_celery/__init__.py

from .celery import app as celery_app

__all__ = ("celery_app",)
```

在 Django startup 上加载 Celery 应用程序可以确保`@shared_task` decorator 正确使用它。在下一节中，您将了解更多关于`@shared_task`的内容。

是时候测试您的设置了！请记住，您正在设置的流程需要至少三个服务同时运行:

1.  制作人:你的 Django 应用
2.  **消息代理:**Redis 服务器
3.  **消费者:**你的芹菜 app

因为您正在使用 Redis，所以您将获得数据库后端作为额外的好处，而无需运行另一个服务。

打开三个独立的终端窗口，并启动所有的程序，如果他们还没有运行。

在第一个窗口中使用 Django 的开发服务器提供您的 web 应用程序:

```py
(venv) $ python manage.py runserver
```

然后在第二个终端窗口中启动 Redis 服务器，以防您之前停止了它:

```py
$ redis-server
```

`redis-server`命令是三个命令中唯一一个可以在虚拟环境之外运行的命令，所以要确保您的虚拟环境在另外两个终端窗口中是活动的。

**注意:**如果 Redis 服务器仍然在后台运行，您可能会收到一个错误。如果是这种情况，那么你需要在执行`redis-server`之前调用 [`SHUTDOWN`](https://redis.io/commands/shutdown/) 。

最后，您现在也可以正确启动 Celery，而不会遇到错误消息:

```py
(venv) $ python -m celery -A django_celery worker
```

当使用这个命令启动 Celery 时，您需要将包含 Celery 应用程序实例的模块名称`"django_celery"`提供给`-A`。

**注意:**虽然您在安装 Redis 之前运行 Celery 时看到的错误消息已经消失，但您可能仍然会看到与 Django 的`DEBUG`设置相关的警告。对于这个示例应用程序，您可以忽略这个警告，但是在将站点部署到生产环境之前，您应该总是[将`DEBUG`设置为`False`](https://docs.djangoproject.com/en/dev/howto/deployment/checklist/#debug) 。

总之，您只需要向上述三个文件添加代码，就可以将 Celery 集成到您的 Django 应用程序中，并为处理异步任务做好准备。完成这个基本设置后，您就可以编写一个任务交给 Celery 了。

在下一节中，您将重构`.send_email()`来调用一个异步的 Celery 任务，而不是在 Django 中处理同步发送的电子邮件。

[*Remove ads*](/account/join/)

## 使用芹菜异步处理工作负载

您已经成功地安排了运行 Django、Redis 和 Celery 异步任务所需的拼图。但是现在，您还没有定义任何要传递给 Celery 的任务。

集成 Celery 和 Django 并将工作卸载到 Celery 的分布式任务队列的最后一步是将电子邮件发送功能重构到 Celery 任务中。

### 重温同步代码

此时，您的代码在`forms.py`的`FeedbackForm`的`.send_email()`中定义了电子邮件发送功能:

```py
 1# feedback/forms.py
 2
 3from time import sleep
 4from django.core.mail import send_mail
 5from django import forms
 6
 7class FeedbackForm(forms.Form):
 8    email = forms.EmailField(label="Email Address")
 9    message = forms.CharField(
10        label="Message", widget=forms.Textarea(attrs={"rows": 5})
11    )
12
13    def send_email(self): 14        """Sends an email when the feedback form has been submitted."""
15        sleep(20)  # Simulate expensive operation(s) that freeze Django
16        send_mail(
17            "Your Feedback",
18            f"\t{self.cleaned_data['message']}\n\nThank you!",
19            "support@example.com",
20            [self.cleaned_data["email_address"]],
21            fail_silently=False,
22        )
```

您在第 13 行定义了`.send_email()`。该方法模拟了一个昂贵的操作，通过调用第 15 行的`sleep()`来冻结你的应用程序 20 秒。在第 16 到 22 行，您用 Django 在第 4 行导入的便利的 [`send_mail()`](https://docs.djangoproject.com/en/4.0/topics/email/#send-mail) 编写您要发送的电子邮件。

您还需要在成功提交表单时调用`.send_email()`，您可以在`views.py`的 [`.form_valid()`](https://docs.djangoproject.com/en/4.0/ref/class-based-views/mixins-editing/#django.views.generic.edit.FormMixin.form_valid) 中设置它:

```py
 1# feedback/views.py
 2
 3from feedback.forms import FeedbackForm
 4from django.views.generic.edit import FormView
 5from django.views.generic.base import TemplateView
 6
 7class FeedbackFormView(FormView):
 8    template_name = "feedback/feedback.html"
 9    form_class = FeedbackForm
10    success_url = "/success/"
11
12    def form_valid(self, form): 13        form.send_email() 14        return super().form_valid(form)
15
16class SuccessView(TemplateView):
17    template_name = "feedback/success.html"
```

第 12 行定义了`.form_valid()`，当表单提交成功时，`FeedbackFormView`会自动调用它。在第 13 行，你最后调用`.send_email()`。

你的设置工作正常，但是由于模拟的昂贵操作，你的应用程序再次响应并允许用户继续浏览需要很长时间。是时候改变这种情况了，让芹菜按照自己的时间表处理邮件发送！

### 将代码重构为芹菜任务

为了让`app.autodiscover_tasks()`像描述的那样工作，您需要在 Django 项目的每个应用程序中的一个单独的`tasks.py`模块中定义您的芹菜任务。

**注意:**在这个例子中，你只有一个应用程序。较大的 Django 项目可能会有更多的应用程序。如果您坚持使用标准设置，那么您将为每个应用程序创建一个`tasks.py`文件，并将应用程序的芹菜任务存储在该文件中。

在您的`feedback/`应用程序中创建一个名为`tasks.py`的新文件:

```py
feedback/
│
├── migrations/
│   └── __init__.py
│
├── templates/
│   │
│   └── feedback/
│       ├── base.html
│       ├── feedback.html
│       └── success.html
│
├── __init__.py
├── admin.py
├── apps.py
├── forms.py
├── models.py
├── tasks.py ├── tests.py
├── urls.py
└── views.py
```

在该文件中，您定义了一个新函数来处理电子邮件发送逻辑。从`forms.py`中的`.send_mail()`获取代码，并以此为基础在`tasks.py`中创建`send_feedback_email_task()`:

```py
 1# feedback/tasks.py
 2
 3from time import sleep
 4from django.core.mail import send_mail
 5
 6def send_feedback_email_task(email_address, message): 7    """Sends an email when the feedback form has been submitted."""
 8    sleep(20)  # Simulate expensive operation(s) that freeze Django
 9    send_mail(
10        "Your Feedback",
11        f"\t{message}\n\nThank you!", 12        "support@example.com",
13        [email_address], 14        fail_silently=False,
15    )
```

不要忘记添加必要的导入，如第 3 行和第 4 行所示。

到目前为止，你主要是把`.send_mail()`的代码复制到`send_feedback_email_task()`里。您还通过在第 6 行添加两个参数对函数定义进行了编辑。您在第 11 行和第 13 行使用这些参数来替换您之前从`.send_mail()`中的`.cleaned_data`获取的值。这种改变是必要，因为您不能访问新函数中的实例属性。

除此之外，`send_feedback_email_task()`看起来和`.send_email()`一样。芹菜都还没涉足呢！

要将这个函数转换成芹菜任务，您需要做的就是用从`celery`导入的 [`@shared_task`](https://github.com/celery/celery/blob/5.2/celery/app/__init__.py#L25) 来修饰它:

```py
# feedback/tasks.py

from time import sleep
from django.core.mail import send_mail
from celery import shared_task 
@shared_task() def send_feedback_email_task(email_address, message):
    """Sends an email when the feedback form has been submitted."""
    sleep(20)  # Simulate expensive operation(s) that freeze Django
    send_mail(
        "Your Feedback",
        f"\t{message}\n\nThank you!",
        "support@example.com",
        [email_address],
        fail_silently=False,
    )
```

从`celery`导入`shared_task()`并用它装饰`send_feedback_email_task()`之后，你就完成了这个文件中必要的代码修改。

将任务交给 Celery 围绕着 Celery 的 [`Task`类](https://docs.celeryq.dev/en/stable/reference/celery.app.task.html#celery.app.task.Task)，你可以通过在你的函数定义中添加[装饰者](https://realpython.com/primer-on-python-decorators/)来创建任务。

如果你的生产者是一个 Django 应用程序，那么你会希望[使用`@shared_task`装饰器](https://docs.celeryq.dev/en/stable/django/first-steps-with-django.html#using-the-shared-task-decorator)来建立一个任务，这保持了你的应用程序的可重用性。

有了这些补充，您就完成了用 Celery 设置一个异步任务。您只需要在 web 应用程序代码中重构调用它的位置和方式。

回到`forms.py`，在那里你得到了发送电子邮件的代码，重构`.send_email()`,使它调用`send_feedback_email_task()`:

```py
 1# feedback/forms.py
 2
 3# Removed: from time import sleep
 4# Removed: from django.core.mail import send_mail
 5from django import forms
 6from feedback.tasks import send_feedback_email_task 7
 8class FeedbackForm(forms.Form):
 9    email = forms.EmailField(label="Email Address")
10    message = forms.CharField(
11        label="Message", widget=forms.Textarea(attrs={"rows": 5})
12    )
13
14    def send_email(self):
15        send_feedback_email_task.delay( 16            self.cleaned_data["email"], self.cleaned_data["message"] 17        )
```

您没有在`.send_email()`中处理邮件发送代码逻辑，而是将它移到了`tasks.py`中的`send_feedback_email_task()`。这一更改意味着您也可以删除第 3 行和第 4 行中过时的 import 语句。

您现在从第 6 行的`feedback.tasks`导入`send_feedback_email_task()`。

在第 15 行，您调用`send_feedback_email_task()`上的 [`.delay()`](https://docs.celeryq.dev/en/stable/reference/celery.app.task.html#celery.app.task.Task.delay) ，并将从`.cleaned_data`获取的提交的表单数据作为第 16 行的参数传递给它。

**注意:**调用`.delay()`是给芹菜发送任务消息的最快方式。这个方法是更强大的 [`.apply_async()`](https://docs.celeryq.dev/en/stable/reference/celery.app.task.html?highlight=Task.delay#celery.app.task.Task.apply_async) 的快捷方式，它额外支持执行选项来微调你的任务消息。

使用`.apply_async()`,您实现上述功能的调用会稍微冗长一些:

```py
send_feedback_email_task.apply_async(args=[
    self.cleaned_data["email"], self.cleaned_data["message"]
    ]
)
```

虽然在像这样简单的任务消息中，`.delay()`是更好的选择，但是使用`.apply_async()`您将从许多执行选项中受益，比如`countdown`和`retry`。

在`tasks.py`和`forms.py`中应用这些更改后，您就完成了重构！使用 Django 和 Celery 运行异步任务的主要工作在于设置，而不是您需要编写的实际代码。

但是这有用吗？电子邮件还会发出去吗？与此同时，你的 Django 应用程序还能保持响应吗？

[*Remove ads*](/account/join/)

### 测试您的异步任务

当您启动一个芹菜工人，它加载您的代码到内存中。当它通过您的消息代理接收到一个任务时，它将执行该代码。因此，每次更改代码时，都需要重新启动芹菜工作器。

**注意:**为了避免在开发过程中每次代码变更时手动重启你的芹菜工人，你可以[使用](https://testdriven.io/courses/django-celery/auto-reload/)[看门狗](https://github.com/gorakhargosh/watchdog)或者通过编写[定制管理命令](https://docs.djangoproject.com/en/4.0/howto/custom-management-commands/)来设置自动重新加载。

您创建了一个您之前启动的 worker 不知道的任务，因此您需要重新启动 worker。打开运行芹菜工人的终端窗口，按 `Ctrl` + `C` 停止执行。

然后使用您之前使用的相同命令重新启动 worker，并添加`-l info`以将[日志级别](https://realpython.com/python-logging/)设置为 info:

```py
(venv) $ python -m celery -A django_celery worker -l info
```

将`-l`选项设置为`info`意味着您将看到更多信息打印到您的终端上。启动时，Celery 会在`[tasks]`部分显示它发现的所有任务:

```py
[tasks]
  . feedback.tasks.send_feedback_email_task
```

这个输出确认 Celery 已经注册了`send_feedback_email_task()`，并准备好处理与这个任务相关的传入消息。

随着[所有服务的启动和运行](#integrate-celery-with-django)以及为芹菜重构的[代码，您已经准备好接替您的一个用户的工作，并再次尝试重构后的工作流:](#handle-workloads-asynchronously-with-celery)

[https://player.vimeo.com/video/728157432?background=1](https://player.vimeo.com/video/728157432?background=1)

如果你现在在应用程序的主页上提交反馈表，你会很快被重定向到成功页面。耶！没有必要等待和建立任何挫折。您甚至可以返回反馈表，立即提交另一份回复。

但是在后端会发生什么呢？在您的同步示例中，您看到电子邮件消息出现在运行 Django 开发服务器的终端窗口中。这一次，它没有出现在那里——即使二十秒过去了。

相反，您会看到电子邮件文本出现在运行 Celery 的终端窗口中，旁边是关于处理任务的其他日志:

```py
[INFO/MainProcess] celery@Martins-MBP.home ready.
[INFO/MainProcess] Task feedback.tasks.send_feedback_email_task ⮑ [a5054d64-5592-4347-be77-cefab994c2bd] received [WARNING/ForkPoolWorker-7] Content-Type: text/plain; charset="utf-8"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Subject: Your Feedback
From: support@example.com
To: martin@realpython.com
Date: Tue, 12 Jul 2025 14:49:23 -0000
Message-ID: <165763736314.3405.4812564479387463177@martins-mbp.home>

        Great!

Thank you!
[WARNING/ForkPoolWorker-7] -----------------------------------------
[INFO/ForkPoolWorker-7] Task feedback.tasks.send_feedback_email_task ⮑ [a5054d64-5592-4347-be77-cefab994c2bd] succeeded in 20.078754458052572s: ⮑ None
```

因为您运行了带有日志级别信息(`-l info`)的 Celery worker，所以您可以阅读关于 Celery 端发生的事情的详细描述。

首先，您可能会注意到日志通知您收到了`send_feedback_email_task`。如果您在提交反馈响应后立即看到这个终端窗口，那么您会看到这个日志线立即打印出来。

在那之后，芹菜进入等待阶段，这是由之前冻结你的 Django 应用的`sleep()`调用引起的。虽然您可以立即继续使用 Django 应用程序，但 Celery 会在后台为您执行昂贵的计算。

20 秒后，Celery 将 Django 用`send_mail()`构建的虚拟邮件打印到终端窗口。然后它添加另一个日志条目，告诉您`send_feedback_email_task`成功了，花了多长时间(`20.078754458052572s`，以及它的返回值是什么(`None`)。

**注意:**请记住，在这个例子中，您的 Django 应用程序不会知道芹菜任务是否成功。那意味着*谢谢你！*你的读者看到的信息并不一定意味着该信息对你是安全的。因为您使用 Redis 设置了一个数据库后端，所以您可以查询该后端来确定任务运行是否成功。

由于 HTTP 是如何工作的，在*前端*通知用户一个后台任务是否已经成功完成并不是一个简单的任务。为此，你需要[通过 Django 通道](https://testdriven.io/blog/django-and-celery/#workflow)设置 AJAX 轮询或 [WebSockets。](https://realpython.com/getting-started-with-django-channels/)

那很好！您的反馈似乎很快就被提交了，而且您不必经历任何令人沮丧的等待时间。

干得好！您成功地将 Celery 集成到 Django 应用程序中，并设置它处理异步任务。芹菜现在处理你的电子邮件发送和所有的开销作为后台任务。一旦电子邮件将任务指令传递给 Celery 的分布式任务队列，它就不需要关心你的 web 应用程序了。

[*Remove ads*](/account/join/)

## 结论

嗖！反馈已提交！

在您集成了 Celery 并重构了 Django 代码之后，在您的应用程序中发送反馈是如此美妙的体验，以至于您不想停止发送积极的反馈信息！

使用 Celery 在后台异步处理长时间运行或计算成本高的任务，而不是让您的 web 应用程序陷入不打算处理的任务，可以为运行缓慢的应用程序注入新鲜空气。

Celery 旨在提供一个快速接口，用于向其分布式任务队列发送消息。在这个例子中，您体验到了在 Django 应用程序中使用 Celery 只需要做很少的修改。

**在本教程中，您学习了如何:**

*   识别芹菜的**有效用例**
*   区分**芹菜工**和**芹菜工**
*   在 Django 项目中整合芹菜和 Redis
*   设置独立于 Django 应用运行的异步任务
*   重构 Django 代码，改为用芹菜运行任务

不断识别 Django 不需要处理的任何任务。然后将它们卸载到您最喜欢的分布式任务队列中。

知道您可以在后台处理缓慢的任务而不影响您的用户体验也为新想法打开了大门:

*   [对您提交的反馈实施自动垃圾邮件过滤](https://docs.celeryq.dev/en/stable/userguide/tasks.html#task-example)，这样您就不必浪费时间筛选垃圾邮件。
*   [将所有反馈提交的文本](https://github.com/argosopentech/LibreTranslate-py)翻译成西班牙语，这样你就可以给[你的抽认卡应用](https://realpython.com/django-flashcards-app/)添加新词汇，并帮助你的语言学习。
*   [确定你自己的 Django 项目中的任务](#how-can-you-leverage-celery-for-your-django-app),你可以将这些任务交给芹菜工人。

许多有趣而有价值的计算只需要一分钟。有了 Celery 作为盟友，您可以开始将更多这样的特性集成到您的 web 应用程序中，并让分布式任务队列在后台异步运行这些任务。********