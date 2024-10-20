# 烧瓶 101:添加数据库

> 原文：<https://www.blog.pythonlibrary.org/2017/12/12/flask-101-adding-a-database/>

上次我们学习了如何安装烧瓶。在这篇文章中，我们将学习如何添加一个数据库到我们的音乐数据网站。您可能还记得，Flask 是一个微型网络框架。这意味着它不像 Django 那样带有对象关系映射器(ORM)。如果您想要添加数据库交互性，那么您需要自己添加或者安装一个扩展。我个人喜欢 [SQLAlchemy](http://www.sqlalchemy.org/) ，所以我认为有一个现成的扩展可以将 SQLAlchemy 添加到 Flask 中，这个扩展叫做 [Flask-SQLAlchemy](http://flask-sqlalchemy.pocoo.org/) ，这很好。

要安装 Flask-SQLAlchemy，只需要使用 pip。在运行以下内容之前，请确保您处于我们在本系列第一部分中创建的激活的虚拟环境中，否则您将最终将扩展安装到您的基本 Python 而不是您的虚拟环境中:

```py

pip install flask-sqlalchemy

```

现在我们已经安装了 Flask-SQLAlchemy 及其依赖项，我们可以开始创建数据库了！

* * *

### 创建数据库

用 SQLAlchemy 创建数据库实际上非常容易。SQLAlchemy 支持几种不同的数据库操作方式。我最喜欢的是使用它的声明性语法，允许您创建模拟数据库本身的类。所以在这个例子中我会用到它。我们也将使用 SQLite 作为我们的后端，但是如果我们想的话，我们可以很容易地将后端更改为其他东西，如 MySQL 或 Postgres。

首先，我们将看看如何使用普通的 SQLAlchemy 创建数据库文件。然后我们将创建一个单独的脚本，它使用稍微不同的 Flask-SQLAlchemy 语法。将以下代码放入名为 **db_creator.py** 的文件中

```py

# db_creator.py

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

engine = create_engine('sqlite:///mymusic.db', echo=True)
Base = declarative_base()

class Artist(Base):
    __tablename__ = "artists"

    id = Column(Integer, primary_key=True)
    name = Column(String)

    def __repr__(self):
        return "{}".format(self.name)

class Album(Base):
    """"""
    __tablename__ = "albums"

    id = Column(Integer, primary_key=True)
    title = Column(String)
    release_date = Column(String)
    publisher = Column(String)
    media_type = Column(String)

    artist_id = Column(Integer, ForeignKey("artists.id"))
    artist = relationship("Artist", backref=backref(
        "albums", order_by=id))

# create tables
Base.metadata.create_all(engine)

```

对于使用 Python 的人来说，这段代码的第一部分应该非常熟悉，因为我们在这里所做的只是从 SQLAlchemy 导入一些我们需要的片段，以使代码的其余部分工作。然后我们创建 SQLAlchemy 的**引擎**对象，它基本上将 Python 连接到选择的数据库。在本例中，我们连接到 SQLite 并创建一个文件，而不是在内存中创建数据库。我们还创建了一个“基类”,我们可以用它来创建实际定义数据库表的声明类定义。

接下来的两个类定义了我们关心的表，即**艺术家**和**专辑**。你会注意到我们通过 **__tablename__** 类属性来命名表格。我们还创建表的列，并根据需要设置它们的数据类型。Album 类有点复杂，因为我们设置了与 Artist 表的外键关系。你可以在我以前的 [SQLAlchemy 教程](https://www.blog.pythonlibrary.org/2012/07/01/a-simple-sqlalchemy-0-7-0-8-tutorial/)中阅读更多关于这是如何工作的，或者如果你想要更深入的细节，那么查看写得很好的[文档](http://docs.sqlalchemy.org/en/latest/orm/extensions/declarative/basic_use.html)。

当您运行上面的代码时，您应该在终端中看到类似这样的内容:

```py

2017-12-08 18:36:43,290 INFO sqlalchemy.engine.base.Engine SELECT CAST('test plain returns' AS VARCHAR(60)) AS anon_1
2017-12-08 18:36:43,291 INFO sqlalchemy.engine.base.Engine ()
2017-12-08 18:36:43,292 INFO sqlalchemy.engine.base.Engine SELECT CAST('test unicode returns' AS VARCHAR(60)) AS anon_1
2017-12-08 18:36:43,292 INFO sqlalchemy.engine.base.Engine ()
2017-12-08 18:36:43,294 INFO sqlalchemy.engine.base.Engine PRAGMA table_info("artists")
2017-12-08 18:36:43,294 INFO sqlalchemy.engine.base.Engine ()
2017-12-08 18:36:43,295 INFO sqlalchemy.engine.base.Engine PRAGMA table_info("albums")
2017-12-08 18:36:43,295 INFO sqlalchemy.engine.base.Engine ()
2017-12-08 18:36:43,296 INFO sqlalchemy.engine.base.Engine 
CREATE TABLE artists (
    id INTEGER NOT NULL, 
    name VARCHAR, 
    PRIMARY KEY (id)
)

2017-12-08 18:36:43,296 INFO sqlalchemy.engine.base.Engine ()
2017-12-08 18:36:43,315 INFO sqlalchemy.engine.base.Engine COMMIT
2017-12-08 18:36:43,316 INFO sqlalchemy.engine.base.Engine 
CREATE TABLE albums (
    id INTEGER NOT NULL, 
    title VARCHAR, 
    release_date DATE, 
    publisher VARCHAR, 
    media_type VARCHAR, 
    artist_id INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(artist_id) REFERENCES artists (id)
)

2017-12-08 18:36:43,316 INFO sqlalchemy.engine.base.Engine ()
2017-12-08 18:36:43,327 INFO sqlalchemy.engine.base.Engine COMMIT

```

现在让我们在烧瓶中完成所有这些工作！

* * *

### 使用烧瓶-SQLAlchemy

当我们使用 Flask-SQLAlchemy 时，我们需要做的第一件事是创建一个简单的应用程序脚本。我们就叫它 **app.py** 。将下面的代码放到这个文件中，并保存到 **musicdb** 文件夹中。

```py

# app.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mymusic.db'
app.secret_key = "flask rocks!"

db = SQLAlchemy(app)

```

在这里，我们创建 Flask 应用程序对象，并告诉它 SQLAlchemy 数据库文件应该放在哪里。我们还设置了一个简单的密钥，并创建了一个 db 对象，允许我们将 SQLAlchemy 集成到 Flask 中。接下来，我们需要创建一个 **models.py** 文件，并将其保存到 musicdb 文件夹中。完成后，向其中添加以下代码:

```py

# models.py 

from app import db

class Artist(db.Model):
    __tablename__ = "artists"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)

    def __repr__(self):
        return "".format(self.name)

class Album(db.Model):
    """"""
    __tablename__ = "albums"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String)
    release_date = db.Column(db.String)
    publisher = db.Column(db.String)
    media_type = db.Column(db.String)

    artist_id = db.Column(db.Integer, db.ForeignKey("artists.id"))
    artist = db.relationship("Artist", backref=db.backref(
        "albums", order_by=id), lazy=True) 
```

您会注意到 Flask-SQLAlchemy 并不需要所有的导入，就像普通的 SQLAlchemy 所需要的那样。我们所需要的是我们在应用程序脚本中创建的 db 对象。然后，我们只需在原始 SQLAlchemy 代码中使用的所有类前加上“db”。您还会注意到，它已经被预定义为 **db，而不是创建一个基类。型号**。

最后，我们需要创建一种初始化数据库的方法。您可以将它放在几个不同的地方，但我最终创建了一个名为 **db_setup.py** 的文件，并添加了以下内容:

```py

# db_setup.py

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///mymusic.db', convert_unicode=True)
db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()

def init_db():
    import models
    Base.metadata.create_all(bind=engine)

```

这段代码将用您在模型脚本中创建的表初始化数据库。为了进行初始化，让我们编辑掉上一篇文章中的 **test.py** 脚本:

```py

# test.py

from app import app
from db_setup import init_db

init_db()

@app.route('/')
def test():
    return "Welcome to Flask!"

if __name__ == '__main__':
    app.run()

```

这里我们只是导入了我们的 app 对象和 init_db 函数。然后我们立即调用 init_db 函数。要运行这段代码，您只需在终端的 musicdb 文件夹中运行以下命令:

```py

FLASK_APP=test.py flask run

```

当您运行这个命令时，您不会看到我们前面看到的 SQLAlchemy 输出。相反，您将只看到一些打印出来的信息，说明您的 Flask 应用程序正在运行。您还会发现在您的 musicdb 文件夹中已经创建了一个 mymusic.db 文件。

*注意， **init_db()** 调用似乎并不总是有效，所以如果您的 SQLite 数据库文件没有正确生成，您可能需要运行我在上一篇文章中编写的 db_creator 脚本。*

* * *

### 包扎

此时，您现在有了一个带有空数据库的 web 应用程序。您不能使用 web 应用程序向数据库添加任何内容，也不能查看数据库中的任何内容。是的，你刚刚创造了一些非常酷的东西，但它对你的用户来说也是完全无用的。在下一篇文章中，我们将学习如何添加一个搜索表单来搜索空数据库中的数据！是的，我的疯狂是有方法的，但是你必须继续阅读这个系列才能弄明白。

* * *

### 下载代码

从本文下载一个代码包:[flask-music db-part _ ii . tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2017/12/flask-musicdb-part_ii.tar.gz)

* * *

### 本系列的其他文章

*   **第一部分**-101 号烧瓶:[入门](https://www.blog.pythonlibrary.org/2017/12/12/flask-101-getting-started/)

* * *

### 相关阅读

*   Flask-SQLAlchemy [网站](http://flask-sqlalchemy.pocoo.org/2.3/)
*   SQLAlchemy [网站](http://www.sqlalchemy.org/)
*   一个简单的 [SQLAlchemy 教程](https://www.blog.pythonlibrary.org/2012/07/01/a-simple-sqlalchemy-0-7-0-8-tutorial/)