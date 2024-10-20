# 为您的第一个 Python Django 应用程序编写模型

> 原文：<https://www.pythoncentral.io/writing-models-for-your-first-python-django-application/>

上一篇文章[编写您的第一个 Python Django 应用程序](https://www.pythoncentral.io/writing-your-first-python-django-application/ "Writing Your First Python Django Application")是如何从头开始编写一个简单的 Django 应用程序的分步指南。在本文中，您将学习如何为新的 Django 应用程序编写模型。

## 软件架构模式

在我们深入代码之前，让我们回顾一下两个最流行的服务器端软件架构设计模式:*模型-视图-控制器*和*表示-抽象-控制*。

## 模型-视图-控制器

*模型-视图-控制器* ( *MVC* )设计模式是一种软件架构模式，它将数据的表示与处理用户交互的逻辑分离开来。一个*模型*指定了存储什么样的数据。一个*视图*从一个*模型*中请求数据并从中生成输出。一个*控制器*提供逻辑来改变*视图*的显示或更新*模型*的数据。

## 呈现-抽象-控制

与 *MVC* ，*表示-抽象-控制* ( *PAC* )是另一种流行的软件架构模式。PAC 将系统分成组件层。在每一层中，*表示*组件从输入数据生成输出；*抽象*组件检索并处理数据；而*控制*组件是*表示*和*抽象*之间的中间人，负责管理这些组件之间的信息流和通信。不像 *MVC* 中的*视图*直接与*模型*对话， *PAC* 的*表示*和*抽象*从不直接相互对话，它们之间的通信是通过*控件*进行的。与遵循 MVC 模式的 Django 不同，流行的内容管理系统 Drupal 遵循 PAC 模式。

## 姜戈的 MVC

虽然 Django 采用了 MVC 模式，但它与标准定义有点不同。即，

*   在 Django 中，*模型*描述了*什么样的*数据被存储在服务器上。因此，它类似于标准 MVC 模式的模型。
*   在 Django 中，*视图*描述了*将哪些*数据返回给用户。而标准的 MVC 视图描述了*如何*呈现数据。
*   在 Django 中，*模板*描述了*如何将*数据呈现给用户。因此，它类似于标准 MVC 模式的视图。
*   在 Django 中，*控制器*定义了框架提供的机制:将传入请求路由到适当视图的代码。因此，它类似于标准 MVC 模式的控制器。

总的来说，Django 偏离了标准的 *MVC* 模式，因为它建议*视图*应该包括业务逻辑，而不是像标准的 *MVC* 那样只包括表示逻辑，并且*模板*应该负责大部分的表示逻辑，而标准的 *MVC* 根本不包括*模板*组件。由于 Django 的设计与标准的 *MVC* 相比的这些差异，我们通常称 Django 的设计为*模型-模板-视图+控制器*，其中*控制器*经常被省略，因为它是框架的一部分。所以大部分时候 Django 的设计模式都叫 *MTV* 。

尽管理解 Django 的 *MTV* 模式的设计哲学是有帮助的，但最终唯一重要的事情是完成工作，Django 的生态系统提供了一切面向编程效率的东西。

# 创建模型

由于新的 Django 应用程序是一个博客，我们将编写两个模型，`Post`和`Comment`。一个`Post`有一个`content`字段和一个`created_at`字段。一个`Comment`具有一个`message`字段和一个`created_at`字段。每个`Comment`都与一个`Post`相关联。

```py

from django.db import models as m
类 Post(m . Model):
content = m . CharField(max _ length = 256)
created _ at = m . Datetime field(' Datetime created ')
class Comment(m . Model):
Post = m . foreign key(Post)
message = m . TextField()
created _ at = m . Datetime field(' Datetime created ')

```

接下来，修改`myblog/settings.py`中的`INSTALLED_APP`元组，将`myblog`添加为已安装的应用。

```py

INSTALLED_APPS = (

'django.contrib.auth',

'django.contrib.contenttypes',

'django.contrib.sessions',

'django.contrib.sites',

'django.contrib.messages',

'django.contrib.staticfiles',

# Uncomment the next line to enable the admin:

# 'django.contrib.admin',

# Uncomment the next line to enable admin documentation:

# 'django.contrib.admindocs',

'myblog', # Add this line

)

```

现在，您应该能够执行下面的命令来查看当您运行`syncdb`时将执行哪种原始 SQL。命令`syncdb`为`INSTALLED_APPS`中尚未创建表的所有应用程序创建数据库表。在后台，`syncdb`将原始 SQL 语句输出到后端数据库管理系统(在我们的例子中是 MySQL 或 PostgreSQL)。

```py

$ python manage.py sql myblog

[/shell]

```

*   [MySQL](#) 的实现

[shell]
BEGIN;
CREATE TABLE `myblog_post` (
`id` integer AUTO_INCREMENT NOT NULL PRIMARY KEY,
`content` varchar(256) NOT NULL,
`created_at` datetime NOT NULL
)
;
CREATE TABLE `myblog_comment` (
`id` integer AUTO_INCREMENT NOT NULL PRIMARY KEY,
`post_id` integer NOT NULL,
`message` longtext NOT NULL,
`created_at` datetime NOT NULL
)
;
ALTER TABLE `myblog_comment` ADD CONSTRAINT `post_id_refs_id_648c7748` FOREIGN KEY (`post_id`) REFERENCES `myblog_post` (`id`);

提交；
[/shell]

*   [PostgreSQL](#)

[shell]
BEGIN;
CREATE TABLE "myblog_post" (
"id" serial NOT NULL PRIMARY KEY,
"content" varchar(256) NOT NULL,
"created_at" timestamp with time zone NOT NULL
)
;
CREATE TABLE "myblog_comment" (
"id" serial NOT NULL PRIMARY KEY,
"post_id" integer NOT NULL REFERENCES "myblog_post" ("id") DEFERRABLE INITIALLY DEFERRED,
"message" text NOT NULL,
"created_at" timestamp with time zone NOT NULL
)
;

提交；
[/shell]

SQL 转储看起来不错！现在，您可以通过执行以下命令在数据库中创建表。

```py

$ python manage.py syncdb

Creating tables ...

Creating table myblog_post

Creating table myblog_comment

Installing custom SQL ...

Installing indexes ...

Installed 0 object(s) from 0 fixture(s)

```

注意，在前一个命令中创建了两个表`myblog_post`和`myblog_comment`。

# 与新模型玩得开心

现在，让我们深入 Django shell，享受我们全新的模型。要以交互模式运行我们的 Django 应用程序，请键入以下命令:

```py

$ python manage.py shell

Python 2.7.2 (default, Oct 11 2012, 20:14:37)

Type "help", "copyright", "credits" or "license" for more information.

(InteractiveConsole)

>>>

```

前面的命令打开的交互式 shell 是一个普通的 Python 解释器 shell，您可以在其中针对我们的 Django 应用程序自由地执行语句。

```py

>>> from myblog import models as m

>>> # No post in the database yet

>>> m.Post.objects.all()

[]

>>> # No comment in the database yet

>>> m.Comment.objects.all()

[]

>>> # Django's default settings support storing datetime objects with tzinfo in the database.

>>> # So we use django.utils.timezone to put a datetime with time zone information into the database.

>>> from django.utils import timezone

>>> p = m.Post(content='Django is awesome.', created_at=timezone.now())

>>> p

<Post: Post object>

>>> p.created_at

datetime.datetime(2013, 3, 26, 17, 6, 39, 329040, tzinfo=<UTC>)

>>> # Save / commit the new post object into the database.

>>> p.save()

>>> # Once a post is saved into the database, it has an id attribute which is the primary key of the underlying database record.

>>> p.id

1

>>> # Now we create another post object without saving it into the database.

>>> p2 = m.Post(content='Pythoncentral is also awesome.', created_at=timezone.now())

>>> p2

<Post: Post object>

>>> # Notice p2.id is None, which means p2 is not committed into the database yet.

>>> p2.id is None

True

>>> # Now we retrieve all the posts from the database and inspect them just like a normal python list

>>> m.Post.objects.all()

[<Post: Post object>]

>>> m.Post.objects.all()[0]

<Post: Post object>

>>> # Since p2 is not saved into the database yet, there's only one post whose id is the same as p.id

>>> m.Post.objects.all()[0].id == p.id

True

>>> # Now we save / commit p2 into the database and re-run the query again

>>> p2.save()

>>> m.Post.objects.all()

[<Post: Post object>, <Post: Post object>]

>>> m.Post.objects.all()[1].id == p2.id

True

```

现在我们已经熟悉了新的`Post`型号，将它与新的`Comment`一起使用怎么样？一个`Post`可以有多个`Comment`，而一个`Comment`只能有一个`Post`。

```py

>>> c = m.Comment(message='This is a comment for p', created_at=timezone.now())

>>> c.post = p

>>> c.post

<Post: Post object>

>>> c.post.id == p.id

True

>>> # Since c is not saved yet, p.comment_set.all() does not include it.

>>> p.comment_set.all()

[]

>>> c.save()

>>> # Once c is saved into the database, p.comment_set.all() will have it.

>>> p.comment_set.all()

[<Comment: Comment object>]

>>> p.comment_set.all()[0].id == c.id

True

>>> c2 = m.Comment(message='This is another comment for p.', created_at=timezone.now())

>>> # If c2.post is not specified, then Django will raise a DoseNotExist exception.

>>> c2.post

Traceback (most recent call last):

File "<console>", line 1, in <module>

File "/Users/xiaonuogantan/python2-workspace/lib/python2.7/site-packages/django/db/models/fields/related.py", line 389, in __get__

raise self.field.rel.to.DoesNotExist

DoesNotExist

>>> # Assign Post p to c2.

>>> c2.post = p

>>> c2.save()

>>> p.comment_set.all()

[<Comment: Comment object>, <Comment: Comment object>]

>>> # Order the comment_set according Comment.created_at

>>> p.comment_set.order_by('created_at')

[<Comment: Comment object>, <Comment: Comment object>]

```

到目前为止，我们知道了如何使用每个模型的现有属性来创建、保存和检索`Post`和`Comment`。查询数据库找到我们想要的帖子和评论怎么样？原来 Django 为查询提供了一种稍微有点奇怪的语法。基本上，`filter()`函数接受符合“[字段]_ _[字段属性]_ _[关系]=[值]”形式的参数。举个例子，

```py

>>> # Retrieve a list of comments from p.comment_set whose created_at.year is 2013

>>> p.comment_set.filter(created_at__year=2013)

[<Comment: Comment object>, <Comment: Comment object>]

>>> # Retrieve a list of comments from p.comment_set whose created_at is later than timezone.now()

>>> p.comment_set.filter(created_at__gt=timezone.now())

[]

>>> # Retrieve a list of comments from p.comment_set whose created_at is earlier than timezone.now()

>>> p.comment_set.filter(created_at__lt=timezone.now())

[<Comment: Comment object>, <Comment: Comment object>]

>>> # Retrieve a list of comments from p.comment_set whose message startswith 'This is a '

>>> p.comment_set.filter(message__startswith='This is a')

[<Comment: Comment object>, <Comment: Comment object>]

>>> # Retrieve a list of comments from p.comment_set whose message startswith 'This is another'

>>> p.comment_set.filter(message__startswith='This is another')

[<Comment: Comment object>]

>>> # Retrieve a list of posts whose content startswith 'Pythoncentral'

>>> m.Post.objects.filter(content__startswith='Pythoncentral')

[<Post: Post object>]

>>> # Retrieve a list of posts which satisfies the query that any comment in its comment_set has a message that startswith 'This is a'

>>> m.Post.objects.filter(comment__message__startswith='This is a')

[<Post: Post object>, <Post: Post object>]

>>> # Retrieve a list of posts which satisfies the query that any comment in its comment_set has a message that startswith 'This is a' and a created_at that is less than / earlier than timezone.now()

>>> m.Post.objects.filter(comment__message__startswith='This is a', comment__created_at__lt=timezone.now())

[<Post: Post object>, <Post: Post object>]

```

您是否注意到最后两个查询有些奇怪？`m.Post.objects.filter(comment__message__startswith='This is a')`和`m.Post.objects.filter(comment__message__startswith='This is a', comment__created_at__lt=timezone.now())`返回两个`Post`而不是一个不奇怪吗？我们来核实一下有哪些帖子被退回来了。

```py

>>> posts = m.Post.objects.filter(comment__message__startswith='This is a')

>>> posts[0].id

1

>>> posts[1].id

1

```

啊哈！`posts[0]`和`posts[1]`是同一个岗位！那是怎么发生的？因为原始查询是`Post`和`Comment`的连接查询，并且有两个`Comment`满足该查询，所以返回两个`Post`对象。那么，我们如何让它只返回一个`Post`？很简单，只需在`filter()`的末尾追加一个`distinct()`:

```py

>>> m.Post.objects.filter(comment__message__startswith='This is a').distinct()

[<Post: Post object>]

>>> m.Post.objects.filter(comment__message__startswith='This is a', comment__created_at__lt=timezone.now()).distinct()

[<Post: Post object>]

```

# 总结和建议

在本文中，我们为我们的博客网站编写了两个简单的模型`Post`和`Comment`。Django 没有编写原始的 SQL，而是提供了一个强大且易于使用的 ORM，允许我们编写简洁且易于维护的数据库操作代码。您应该深入代码并运行`python manage.py shell`来与现有的 Django 模型进行交互，而不是停留在这里。很好玩啊！