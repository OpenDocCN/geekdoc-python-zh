## 被解放的姜戈 02 庄园疑云

[`www.cnblogs.com/vamei/p/3531740.html`](http://www.cnblogs.com/vamei/p/3531740.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

[上一回](http://www.cnblogs.com/vamei/p/3528878.html)说到，姜戈的江湖初体验：如何架设服务器，如何回复 http 请求，如何创建 App。这一回，我们要走入糖果庄园。

数据库是一所大庄园，藏着各种宝贝。一个没有数据库的网站，所能提供的功能会非常有限。

![](img/rdb_epub_2335399320373225544.jpg)

**为了找到心爱的人，姜戈决定一探这神秘的糖果庄园。**

### 连接数据库

Django 为多种数据库后台提供了统一的调用 API。根据需求不同，Django 可以选择不同的数据库后台。MySQL 算是最常用的数据库。我们这里将 Django 和 MySQL 连接。

在 Linux 终端下启动 mysql:

在 MySQL 中创立 Django 项目的数据库：

```py
mysql> CREATE DATABASE villa DEFAULT CHARSET=utf8;

```

这里使用 utf8 作为默认字符集，以便支持中文。

在 MySQL 中为 Django 项目创立用户，并授予相关权限:

```py
mysql> GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, INDEX, ALTER, CREATE TEMPORARY TABLES, LOCK TABLES ON villa.* TO 'vamei'@'localhost' IDENTIFIED BY 'vameiisgood';

```

在 settings.py 中，将 DATABASES 对象更改为:

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'villa',
        'USER': 'vamei',
        'PASSWORD': 'vameiisgood',
        'HOST':'localhost',
        'PORT':'3306',
    }
}

```

后台类型为 mysql。上面包含数据库名称和用户的信息，它们与 MySQL 中对应数据库和用户的设置相同。Django 根据这一设置，与 MySQL 中相应的数据库和用户连接起来。此后，Django 就可以在数据库中读写了。

**姜戈略一迟疑，旋即走入了庄园的大门。**

### 创立模型

MySQL 是关系型数据库。但在 Django 的帮助下，我们不用直接编写 SQL 语句。Django 将关系型的表(table)转换成为一个类(class)。而每个记录(record)是该类下的一个对象(object)。我们可以使用基于对象的方法，来操纵关系型的 MySQL 数据库。

在传统的 MySQL 中，数据模型是表。在 Django 下，一个表为一个类。表的每一列是该类的一个属性。在 models.py 中，我们创建一个只有一列的表，即只有一个属性的类：

```py
from django.db import models class Character(models.Model):
    name = models.CharField(max_length=200) def __unicode__(self): return self.name

```

类 Character 定义了数据模型，它需要继承自 models.Model。在 MySQL 中，这个类实际上是一个表。表只有一列，为 name。可以看到，name 属性是字符类型，最大长度为 200。

类 Character 有一个 __unicode__()方法，用来说明对象的字符表达方式。如果是 Python 3，定义 __str__()方法，实现相同的功能。

命令 Django 同步数据库。Django 根据 models.py 中描述的数据模型，在 MySQL 中真正的创建各个关系表：

同步数据库后，Django 将建立相关的 MySQL 表格，并要求你创建一个超级用户:

> Creating tables ...
> Creating table django_admin_log
> Creating table auth_permission
> Creating table auth_group_permissions
> Creating table auth_group
> Creating table auth_user_groups
> Creating table auth_user_user_permissions
> Creating table auth_user
> Creating table django_content_type
> Creating table django_session
> Creating table west_character
> 
> You just installed Django's auth system, which means you don't have any superusers defined.
> Would you like to create one now? (yes/no): yes
> Username (leave blank to use 'tommy'): vamei
> Email address: vamei@vamei.com
> Password:
> Password (again):
> Superuser created successfully.
> Installing custom SQL ...
> Installing indexes ...
> Installed 0 object(s) from 0 fixture(s)

数据模型建立了。打开 MySQL 命令行：

查看数据模型：

```py
USE villa;
SHOW TABLES;
SHOW COLUMNS FROM west_character;

```

最后一个命令返回 Character 类的对应表格:

> +-------+--------------+------+-----+---------+----------------+
> | Field | Type         | Null | Key | Default | Extra          |
> +-------+--------------+------+-----+---------+----------------+
> | id    | int(11)      | NO   | PRI | NULL    | auto_increment |
> | name  | varchar(200) | NO   |     | NULL    |                |
> +-------+--------------+------+-----+---------+----------------+
> 2 rows in set (0.00 sec)

可以看到，Django 还自动增加了一个 id 列，作为记录的主键(Primary Key)。

**这富丽堂皇的别墅中，姜戈隐隐闻到凶险的味道。**

### 显示数据

数据模型虽然建立了，但还没有数据输入。为了简便，我们手动添加记录。打开 MySQL 命令行,并切换到相应数据库。添加记录：

```py
INSERT INTO west_character (name) Values ('Vamei'); INSERT INTO west_character (name) Values ('Django'); INSERT INTO west_character (name) Values ('John');

```

查看记录：

```py
 SELECT * FROM west_character;

```

可以看到，三个名字已经录入数据库。

下面我们从数据库中取出数据，并返回给 http 请求。在 west/views.py 中，添加视图。对于对应的请求，我们将从数据库中读取所有的记录，然后返回给客户端：

```py
# -*- coding: utf-8 -*-

from django.http import HttpResponse from west.models import Character def staff(request):
    staff_list = Character.objects.all()
    staff_str = map(str, staff_list) return HttpResponse("<p>" + ' '.join(staff_str) + "</p>")

```

可以看到，我们从 west.models 中引入了 Character 类。通过操作该类，我们可以读取表格中的记录

为了让 http 请求能找到上面的程序，在 west/urls.py 增加 url 导航：

```py
from django.conf.urls import patterns, include, url

urlpatterns = patterns('',
    url(r'^staff/','west.views.staff'),
)

```

运行服务器。在浏览器中输入 URL：

127.0.0.1:8000/west/staff

查看效果：

![](img/rdb_epub_2256662255433830016.jpg)

从数据库读出数据，显示在页面

**“我心爱的人，原来你在这里。” 姜戈强自镇定，嘴角忍不住颤动。**

### 总结

Django 使用类和对象接口，来操纵底层的数据库。

有了数据库，就有了站点内容的大本营。

**姜戈，风雨欲来。**