# Models

# Django Model

*   每一个`Django Model`都继承自`django.db.models.Model`
*   在`Model`当中每一个属性`attribute`都代表一个 database field
*   通过`Django Model API`可以执行数据库的增删改查, 而不需要写一些数据库的查询语句

# 设置数据库

Django 项目建成后, 默认设置了使用 SQLite 数据库, 在 my_blog/my_blog/setting.py 中可以查看和修改数据库设置:

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
} 
```

还可以设置其他数据库, 如`MySQL, PostgreSQL`, 现在为了简单, 使用默认数据库设置

# 创建 models

在 my_blog/article/models.py 下编写如下程序:

```py
from django.db import models

# Create your models here.
class Article(models.Model) :
    title = models.CharField(max_length = 100)  #博客题目
    category = models.CharField(max_length = 50, blank = True)  #博客标签
    date_time = models.DateTimeField(auto_now_add = True)  #博客日期
    content = models.TextField(blank = True, null = True)  #博客文章正文

    #python2 使用 __unicode__, python3 使用 __str__
    def __str__(self) :
        return self.title

    class Meta:  #按时间下降排序
        ordering = ['-date_time'] 
```

其中`__str__(self)` 函数 Article 对象要怎么表示自己, 一般系统默认使用`<Article: Article object>` 来表示对象, 通过这个函数可以告诉系统使用 title 字段来表示这个对象

*   `CharField` 用于存储字符串, max_length 设置最大长度
*   `TextField` 用于存储大量文本
*   `DateTimeField` 用于存储时间, auto_now_add 设置 True 表示自动设置对象增加时间

# 同步数据库

```py
$ python manage.py migrate #命令行运行该命令 
```

因为我们已经执行过该命令会出现如下提示

```py
Operations to perform:
  Apply all migrations: admin, contenttypes, sessions, auth
Running migrations:
  No migrations to apply.
  Your models have changes that are not yet reflected in a migration, and so won't be applied.
  Run 'manage.py makemigrations' to make new migrations, and then re-run 'manage.py migrate' to apply them. 
```

那么现在需要执行下面的命令

```py
$ python manage.py makemigrations
#得到如下提示
Migrations for 'article':
  0001_initial.py:
    - Create model Article 
```

现在重新运行以下命令

```py
$ python manage.py migrate
#出现如下提示表示操作成功
Operations to perform:
  Apply all migrations: auth, sessions, admin, article, contenttypes
Running migrations:
  Applying article.0001_initial... OK 
```

> migrate 命令按照 app 顺序建立或者更新数据库, 将`models.py`与数据库同步

# Django Shell

现在我们进入 Django 中的交互式 shell 来进行数据库的增删改查等操作

```py
$ python manage.py shell
Python 3.4.2 (v3.4.2:ab2c023a9432, Oct  5 2014, 20:42:22)
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>> 
```

> 这里进入 Django 的 shell 和 python 内置的 shell 是非常类似的

```py
>>> from article.models import Article
>>> #create 数据库增加操作
>>> Article.objects.create(title = 'Hello World', category = 'Python', content = '我们来做一个简单的数据库增加操作')
<Article: Article object>
>>> Article.objects.create(title = 'Django Blog 学习', category = 'Python', content = 'Django 简单博客教程')
<Article: Article object>

>>> #all 和 get 的数据库查看操作
>>> Article.objects.all()  #查看全部对象, 返回一个列表, 无对象返回空 list
[<Article: Article object>, <Article: Article object>]
>>> Article.objects.get(id = 1)  #返回符合条件的对象
<Article: Article object>

>>> #update 数据库修改操作
>>> first = Article.objects.get(id = 1)  #获取 id = 1 的对象
>>> first.title
'Hello World'
>>> first.date_time
datetime.datetime(2014, 12, 26, 13, 56, 48, 727425, tzinfo=<UTC>)
>>> first.content
'我们来做一个简单的数据库增加操作'
>>> first.category
'Python'
>>> first.content = 'Hello World, How are you'
>>> first.content  #再次查看是否修改成功, 修改操作就是点语法
'Hello World, How are you'

>>> #delete 数据库删除操作
>>> first.delete()
>>> Article.objects.all()  #此时可以看到只有一个对象了, 另一个对象已经被成功删除
[<Article: Article object>]  

Blog.objects.all()  # 选择全部对象
Blog.objects.filter(caption='blogname')  # 使用 filter() 按博客题目过滤
Blog.objects.filter(caption='blogname', id="1") # 也可以多个条件
#上面是精确匹配 也可以包含性查询
Blog.objects.filter(caption__contains='blogname')

Blog.objects.get(caption='blogname') # 获取单个对象 如果查询没有返回结果也会抛出异常

#数据排序
Blog.objects.order_by("caption")
Blog.objects.order_by("-caption")  # 倒序

#如果需要以多个字段为标准进行排序（第二个字段会在第一个字段的值相同的情况下被使用到），使用多个参数就可以了
Blog.objects.order_by("caption", "id")

#连锁查询
Blog.objects.filter(caption__contains='blogname').order_by("-id")

#限制返回的数据
Blog.objects.filter(caption__contains='blogname')[0]
Blog.objects.filter(caption__contains='blogname')[0:3]  # 可以进行类似于列表的操作 
```

> 当然还有更多的 API, 可以查看官方文档