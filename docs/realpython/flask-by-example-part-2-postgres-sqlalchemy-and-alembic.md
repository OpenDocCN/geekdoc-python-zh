# flask by Example–设置 Postgres、SQLAlchemy 和 Alembic

> 原文：<https://realpython.com/flask-by-example-part-2-postgres-sqlalchemy-and-alembic/>

在这一部分中，我们将建立一个 Postgres 数据库来存储字数统计的结果，以及一个对象关系映射器 [SQLAlchemy](https://realpython.com/python-sqlite-sqlalchemy/) 和一个处理数据库迁移的 Alembic。

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

*更新:*

*   02/09/2020:升级到 Python 版本 [3.8.1](https://www.python.org/downloads/release/python-381/) 以及 Psycopg2、Flask-SQLAlchemy 和 Flask-Migrate 的最新版本。详见下面的[。由于 Flask-Migrate 内部接口的更改，请明确安装和使用 Flask-Script。](#install-requirements)
*   03/22/2016:升级到 Python 版本 [3.5.1](https://www.python.org/downloads/release/python-351/) 以及 Psycopg2、Flask-SQLAlchemy、Flask-Migrate 的最新版本。详见下面的[。](#install-requirements)
*   2015 年 2 月 22 日:添加了 Python 3 支持。

* * *

记住:这是我们正在构建的——一个 Flask 应用程序，它根据来自给定 URL 的文本计算词频对。

1.  第一部分:建立一个本地开发环境，然后在 Heroku 上部署一个试运行环境和一个生产环境。
2.  第二部分:使用 SQLAlchemy 和 Alembic 建立一个 PostgreSQL 数据库来处理迁移。(*当前* )
3.  [第三部分](/flask-by-example-part-3-text-processing-with-requests-beautifulsoup-nltk/):添加后端逻辑，使用 requests、BeautifulSoup 和 Natural Language Toolkit (NLTK)库从网页中抓取并处理字数。
4.  第四部分:实现一个 Redis 任务队列来处理文本处理。
5.  [第五部分](/flask-by-example-integrating-flask-and-angularjs/):在前端设置 Angular，持续轮询后端，看请求是否处理完毕。
6.  第六部分:推送到 Heroku 上的临时服务器——建立 Redis 并详细说明如何在一个 Dyno 上运行两个进程(web 和 worker)。
7.  [第七部分](/flask-by-example-updating-the-ui/):更新前端，使其更加人性化。
8.  [第八部分](/flask-by-example-custom-angular-directive-with-d3/):使用 JavaScript 和 D3 创建一个自定义角度指令来显示频率分布图。

<mark>需要代码吗？从[回购](https://github.com/realpython/flask-by-example/releases)中抢过来。</mark>

## 安装要求

本部分使用的工具:

*   PostgreSQL ( [11.6](https://www.postgresql.org/about/news/1994/) )
*   psycopg 2([2 . 8 . 4](https://pypi.python.org/pypi/psycopg2/2.8.4))——Postgres 的 Python 适配器
*   烧瓶-SQLAlchemy ( [2.4.1](https://flask-sqlalchemy.palletsprojects.com/en/2.x/) ) -提供 [SQLAlchemy](https://www.sqlalchemy.org/) 支持的烧瓶延伸
*   Flask-Migrate ( [2.5.2](https://pypi.python.org/pypi/Flask-Migrate/2.5.2) ) -通过 [Alembic](https://pypi.python.org/pypi/alembic/1.4.0) 支持 SQLAlchemy 数据库迁移的扩展

首先，如果你还没有安装 Postgres，在你的本地计算机上安装它。既然 Heroku 使用 Postgres，那么在同一个数据库上进行本地开发将对我们有好处。如果你没有安装 Postgres， [Postgres.app](https://postgresapp.com/) 对于 Mac OS X 用户来说是一个简单的启动和运行方式。咨询[下载页面](https://www.postgresql.org/download/)了解更多信息。

安装并运行 Postgres 后，创建一个名为`wordcount_dev`的数据库，用作我们的本地开发数据库:

```py
$  psql #  create  database  wordcount_dev; CREATE  DATABASE #  \q
```

为了在 Flask 应用程序中使用我们新创建的数据库，我们需要安装一些东西:

```py
$ cd flask-by-example
```

> 通过 [autoenv](https://pypi.python.org/pypi/autoenv/1.0.0) ，我们在[第一部分](/flask-by-example-part-1-project-setup/)中设置的`.env`文件中的环境变量。

```py
$ python -m pip install psycopg2==2.8.4 Flask-SQLAlchemy===2.4.1 Flask-Migrate==2.5.2
$ python -m pip freeze > requirements.txt
```

> 如果你在 OS X，在安装 psycopg2 时遇到问题，请查看这篇关于堆栈溢出的文章。
> 
> 如果安装失败，您可能需要安装`psycopg2-binary`而不是`psycopg2`。

[*Remove ads*](/account/join/)

## 更新配置

将`SQLALCHEMY_DATABASE_URI`字段添加到您的 *config.py* 文件中的`Config()`类，以设置您的应用程序在开发(本地)、试运行和生产中使用新创建的数据库:

```py
import os

class Config(object):
    ...
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
```

您的 *config.py* 文件现在应该是这样的:

```py
import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = 'this-really-needs-to-be-changed'
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']

class ProductionConfig(Config):
    DEBUG = False

class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = True

class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
```

现在，当我们的配置加载到我们的应用程序，适当的数据库将连接到它。

类似于我们在上一篇文章中添加环境变量的方式，我们将添加一个`DATABASE_URL`变量。在终端中运行以下命令:

```py
$ export DATABASE_URL="postgresql:///wordcount_dev"
```

然后将该行添加到您的*中。env* 文件。

在您的 *app.py* 文件[中导入](https://realpython.com/absolute-vs-relative-python-imports/) SQLAlchemy 并连接到数据库:

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

from models import Result

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/<name>')
def hello_name(name):
    return "Hello {}!".format(name)

if __name__ == '__main__':
    app.run()
```

## 数据模型

通过添加一个 *models.py* 文件来建立一个基本模型:

```py
from app import db
from sqlalchemy.dialects.postgresql import JSON

class Result(db.Model):
    __tablename__ = 'results'

    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String())
    result_all = db.Column(JSON)
    result_no_stop_words = db.Column(JSON)

    def __init__(self, url, result_all, result_no_stop_words):
        self.url = url
        self.result_all = result_all
        self.result_no_stop_words = result_no_stop_words

    def __repr__(self):
        return '<id {}>'.format(self.id)
```

这里我们创建了一个表来存储单词计数的结果。

我们首先从 SQLAlchemy 的 [PostgreSQL 方言](https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSON)中[导入](https://realpython.com/python-import/)我们在 *app.py* 文件中创建的数据库连接以及 [JSON](https://realpython.com/python-json/) 。JSON 列对 Postgres 来说是相当新的，并不是在 SQLAlchemy 支持的每个数据库中都可用，所以我们需要专门导入它。

接下来，我们创建了一个`Result()`类，并给它分配了一个表名`results`。然后我们设置想要存储结果的属性-

*   我们存储的结果的`id`
*   我们统计单词的来源
*   我们统计的完整单词列表
*   我们统计的单词列表减去了停用词(稍后会详细介绍)

然后，我们创建了一个`__init__()`方法，它将在我们第一次创建新结果时运行，最后，我们创建了一个`__repr__()`方法，在我们查询对象时表示该对象。

## 本地迁移

我们将使用 [Alembic](https://pypi.python.org/pypi/alembic/1.4.0) ，它是 [Flask-Migrate](https://pypi.python.org/pypi/Flask-Migrate/2.5.2) 的一部分，来管理数据库迁移以更新数据库的模式。

> **注意:** Flask-Migrate 使用 Flasks 新的 CLI 工具。然而，本文使用了由 [Flask-Script](https://flask-script.readthedocs.io/en/latest/) 提供的接口，该接口之前由 Flask-Migrate 使用。为了使用它，您需要通过以下方式安装它:
> 
> ```py
> `$ python -m pip install Flask-Script==2.0.6
> $ python -m pip freeze > requirements.txt` 
> ```

创建一个名为 *manage.py* 的新文件:

```py
import os
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

from app import app, db

app.config.from_object(os.environ['APP_SETTINGS'])

migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
```

为了使用 Flask-Migrate，我们将`Manager`以及`Migrate`和`MigrateCommand`导入到我们的 *manage.py* 文件中。我们还导入了`app`和`db`，所以我们可以从脚本中访问它们。

首先，我们设置我们的配置来获取我们的环境——基于环境变量——创建一个 migrate 实例，用`app`和`db`作为参数，并设置一个`manager`命令来初始化我们的应用程序的`Manager`实例。最后，我们向`manager`添加了`db`命令，这样我们就可以从命令行运行迁移。

为了运行迁移，初始化 Alembic:

```py
$ python manage.py db init
 Creating directory /flask-by-example/migrations ... done
 Creating directory /flask-by-example/migrations/versions ... done
 Generating /flask-by-example/migrations/alembic.ini ... done
 Generating /flask-by-example/migrations/env.py ... done
 Generating /flask-by-example/migrations/README ... done
 Generating /flask-by-example/migrations/script.py.mako ... done
 Please edit configuration/connection/logging settings in
 '/flask-by-example/migrations/alembic.ini' before proceeding.
```

运行数据库初始化后，您将在项目中看到一个名为“migrations”的新文件夹。这是 Alembic 针对项目运行迁移所必需的设置。在“migrations”中，您会看到有一个名为“versions”的文件夹，其中包含创建的迁移脚本。

让我们通过运行`migrate`命令来创建我们的第一个迁移。

```py
$ python manage.py db migrate
 INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
 INFO  [alembic.runtime.migration] Will assume transactional DDL.
 INFO  [alembic.autogenerate.compare] Detected added table 'results'
 Generating /flask-by-example/migrations/versions/63dba2060f71_.py
 ... done
```

现在，您会注意到在“版本”文件夹中有一个迁移文件。该文件由 Alembic 根据模型自动生成。您可以自己生成(或编辑)这个文件；然而，在大多数情况下，自动生成的文件就可以了。

现在我们将使用`db upgrade`命令对数据库进行升级:

```py
$ python manage.py db upgrade
 INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
 INFO  [alembic.runtime.migration] Will assume transactional DDL.
 INFO  [alembic.runtime.migration] Running upgrade  -> 63dba2060f71, empty message
```

数据库现已准备就绪，可供我们在应用程序中使用:

```py
$  psql #  \c  wordcount_dev You  are  now  connected  to  database  "wordcount_dev"  as  user  "michaelherman". #  \dt List  of  relations Schema  |  Name  |  Type  |  Owner --------+-----------------+-------+---------------
  public  |  alembic_version  |  table  |  michaelherman public  |  results  |  table  |  michaelherman (2  rows) #  \d  results Table  "public.results" Column  |  Type  |  Modifiers ----------------------+-------------------+------------------------------------------------------
  id  |  integer  |  not  null  default  nextval('results_id_seq'::regclass) url  |  character  varying  | result_all  |  json  | result_no_stop_words  |  json  | Indexes: "results_pkey"  PRIMARY  KEY,  btree  (id)
```

[*Remove ads*](/account/join/)

## 远程迁移

最后，让我们将迁移应用到 Heroku 上的数据库。不过，首先，我们需要将暂存和生产数据库的细节添加到 *config.py* 文件中。

要检查我们是否在临时服务器上设置了数据库，请运行:

```py
$ heroku config --app wordcount-stage
=== wordcount-stage Config Vars
APP_SETTINGS: config.StagingConfig
```

> 请确保将`wordcount-stage`替换为您的分期应用的名称。

因为我们没有看到数据库环境变量，所以我们需要将 Postgres 插件添加到登台服务器。为此，请运行以下命令:

```py
$ heroku addons:create heroku-postgresql:hobby-dev --app wordcount-stage
 Creating postgresql-cubic-86416... done, (free)
 Adding postgresql-cubic-86416 to wordcount-stage... done
 Setting DATABASE_URL and restarting wordcount-stage... done, v8
 Database has been created and is available
 ! This database is empty. If upgrading, you can transfer
 ! data from another database with pg:copy
 Use `heroku addons:docs heroku-postgresql` to view documentation.
```

> 是 Heroku Postgres 插件的自由层。

现在，当我们再次运行`heroku config --app wordcount-stage`时，我们应该会看到数据库的连接设置:

```py
=== wordcount-stage Config Vars
APP_SETTINGS: config.StagingConfig
DATABASE_URL: postgres://azrqiefezenfrg:Zti5fjSyeyFgoc-U-yXnPrXHQv@ec2-54-225-151-64.compute-1.amazonaws.com:5432/d2kio2ubc804p7
```

接下来，我们需要提交您对 git 所做的更改，并将其推送到您的临时服务器:

```py
$ git push stage master
```

使用`heroku run`命令运行我们创建的迁移，以迁移我们的临时数据库:

```py
$ heroku run python manage.py db upgrade --app wordcount-stage
 Running python manage.py db upgrade on wordcount-stage... up, run.5677
 INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
 INFO  [alembic.runtime.migration] Will assume transactional DDL.
 INFO  [alembic.runtime.migration] Running upgrade  -> 63dba2060f71, empty message
```

> 注意我们是如何只运行`upgrade`，而不是像以前一样运行`init`或`migrate`命令的。我们已经设置好了迁移文件，可以开始迁移了；我们只需要把它和 Heroku 数据库进行比对。

现在让我们为生产做同样的事情。

1.  在 Heroku 上为您的生产应用程序设置一个数据库，就像您为 staging 所做的一样:`heroku addons:create heroku-postgresql:hobby-dev --app wordcount-pro`
2.  将您的更改推送到您的生产站点:`git push pro master`注意，您不必对配置文件进行任何更改——它会根据新创建的`DATABASE_URL`环境变量来设置数据库。
3.  应用迁移:`heroku run python manage.py db upgrade --app wordcount-pro`

现在，我们的试运行和生产站点都已经设置好了数据库，并且已经完成了迁移——准备就绪！

> 当您向生产数据库应用新的迁移时，可能会有停机时间。如果这是一个问题，您可以通过添加一个“追随者”(通常称为从属)数据库来设置数据库复制。关于这方面的更多信息，请查看 Heroku 的官方文档。

## 结论

这就是第二部分。如果你想更深入地了解 Flask，请查看我们附带的视频系列:

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

在[第 3 部分](/flask-by-example-part-3-text-processing-with-requests-beautifulsoup-nltk/)中，我们将构建字数统计功能，并将其发送到任务队列，以处理更长时间运行的字数统计处理。

下次见。干杯！

* * *

*这是创业公司[埃德蒙顿](http://startupedmonton.com/)的联合创始人卡姆·克林和 Real Python 的人合作的作品。***