# 将 SQLAlchemy 添加到待办事项列表 Web 应用程序

> 原文：<https://www.blog.pythonlibrary.org/2013/07/23/bottle-adding-sqlalchemy-to-the-todo-list-web-app/>

在本文中，我们将从上一篇关于 Bottle 的文章中提取代码，并对其进行修改，使其使用 SQLAlchemy，而不仅仅是普通的 SQLite 代码。这将需要您从 PyPI 下载 [bottle-sqlalchemy](https://pypi.python.org/pypi/bottle-sqlalchemy) 包。你也可以使用“pip install bottle-sqlalchemy”来安装它，假设你已经安装了 pip。当然，你还需要[瓶](http://bottlepy.org/docs/dev/)本身。一旦你准备好了，我们可以继续。

### 向瓶中添加 SQLAlchemy

bottle-sqlalchemy 包是 bottle 的一个插件，它使得向 web 应用程序添加 sqlalchemy 变得非常容易。但是首先，让我们实际创建数据库。我们将使用 SQLAlchemy，而不是另一篇文章中的脚本。代码如下:

```py

from sqlalchemy import create_engine, Column, Boolean, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
engine = create_engine("sqlite:///todo.db", echo=True)

########################################################################
class TODO(Base):
    """
    TODO database class
    """
    __tablename__ = "todo"
    id = Column(Integer, primary_key=True)
    task = Column(String, nullable=False)
    status = Column(Boolean, nullable=False)

    #----------------------------------------------------------------------
    def __init__(self, task, status):
        """Constructor"""
        self.task = task
        self.status = status

#----------------------------------------------------------------------
def main():
    """
    Create the database and add data to it
    """
    Base.metadata.create_all(engine)
    create_session = sessionmaker(bind=engine)
    session = create_session()

    session.add_all([
        TODO('Read Google News', 0),
        TODO('Visit the Python website', 1),
        TODO('See how flask differs from bottle', 1),
        TODO('Watch the latest from the Slingshot Channel', 0)
        ])
    session.commit()

if __name__ == "__main__":
    main()

```

如果你了解 SQLAlchemy，那么你就知道这里发生了什么。基本上，您需要创建一个表示数据库的类，并将其映射到一个数据库“引擎”。然后创建一个会话对象来运行查询等。在本例中，我们插入了四条记录。您需要运行该程序来创建 web 应用程序将使用的数据库。

现在我们准备看看项目的核心部分:

```py

from bottle import Bottle, route, run, debug
from bottle import redirect, request, template
from bottle.ext import sqlalchemy

from sqlalchemy import create_engine, Column, Boolean, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --------------------------------
# Add SQLAlchemy app
# --------------------------------
app = Bottle()

Base = declarative_base()
engine = create_engine("sqlite:///todo.db", echo=True)
create_session = sessionmaker(bind=engine)

plugin = sqlalchemy.Plugin(
        engine,
        Base.metadata,
        keyword='db',
        create=True,
        commit=True,
        use_kwargs=False
)

app.install(plugin)

########################################################################
class TODO(Base):
    """
    TODO database class
    """
    __tablename__ = "todo"
    id = Column(Integer, primary_key=True)
    task = Column(String, nullable=False)
    status = Column(Boolean, nullable=False)

    #----------------------------------------------------------------------
    def __init__(self, task, status):
        """Constructor"""
        self.task = task
        self.status = status

    #----------------------------------------------------------------------
    def __repr__(self):
        """"""
        return "', method='GET')
def edit_item(no):
    """
    Edit a TODO item
    """
    session = create_session()
    result = session.query(TODO).filter(TODO.id==no).first()

    if request.GET.get('save','').strip():
        task = request.GET.get('task','').strip()
        status = request.GET.get('status','').strip()

        if status == 'open':
            status = 1
        else:
            status = 0

        result.task = task
        result.status = status
        session.commit()

        redirect("/")
    else:
        return template('edit_task', old=result, no=no)

#----------------------------------------------------------------------
@route("/new", method="GET")
def new_item():
    """
    Add a new TODO item
    """
    if request.GET.get("save", "").strip():
        task = request.GET.get("task", "").strip()
        status = 1

        session = create_session()
        new_task = TODO(task, status)
        session.add(new_task)
        session.commit()

        redirect("/")
    else:
        return template("new_task.tpl")

#----------------------------------------------------------------------
@route("/done")
def show_done():
    """
    Show all items that are done
    """
    session = create_session()
    result = session.query(TODO).filter(TODO.status==0).all()

    output = template("show_done", rows=result)
    return output

#----------------------------------------------------------------------
@route("/")
@route("/todo")
def todo_list():
    """
    Show the main page which is the current TODO list
    """
    session = create_session()
    result = session.query(TODO).filter(TODO.status==1).all()
    myResultList = [(item.id, item.task) for item in result]
    output = template("make_table", rows=myResultList)
    return output

#----------------------------------------------------------------------
if __name__ == "__main__":
    debug(True)
    run() 
```

让我们把它分解一下。首先，我们需要一个瓶子对象，这样我们就可以添加一个插件。然后我们创建一个 **declarative_base** 对象，我们将使用它从数据库的类表示中派生出子类。接下来我们创建 SQLAlchemy **引擎**和一个 **sessionmaker** 对象。最后，我们创建插件并安装它。从第 56 行开始，我们进入实际的瓶子代码(edit_item 函数开始的地方)。如果您运行了最初的版本，您可能会注意到一个关于使用通配符过滤器的反对警告。在本文中，我们通过将 **edit_item** 函数的路由构造从 **@route('/edit/:no '，method='GET')** 更改为 **@route('/edit/ 【T11 ' '，method='GET')** ，解决了这个问题。这也允许我们移除@validation 装饰器。

您会注意到，在每个函数中，我们都创建了一个会话对象来运行页面的查询。看一下主函数， **todo_list** 。从 out 查询返回的结果是一个对象列表。模板需要一个元组列表或一个列表列表，所以我们使用列表理解来创建一个元组列表。但是如果想要改变模板本身呢？我们用另外两个模板来做。我们来看看 **show_done.tpl** 代码:

```py

%#template to generate a HTML table from a list of tuples (or list of lists, or tuple of tuples or ...)
您已完成的待办事项:

```

%for row in rows: %end

| {{row.id}} | {{row.task}} | [编辑](/edit/{{row.id}}) |

创建[新的](/new)项目

这段代码与主模板 make_table.tpl 中的代码几乎完全相同。由于 Bottle 的模板代码几乎是 Python 代码，因此我们可以使用点符号来访问 row 对象的属性。这使我们能够大量清理代码，并非常容易地获得 row.id 和 row.task。您还会注意到，查询本身的代码更短，因为在使用 SQLAlchemy 时，我们没有额外的连接设置和拆除工作要处理。除此之外，应用程序保持不变。

现在，您应该能够创建自己的包含 SQLAlchemy 的瓶子应用程序了。

### 下载源代码

*   [todo_sa.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2013/07/todo_sa.zip)