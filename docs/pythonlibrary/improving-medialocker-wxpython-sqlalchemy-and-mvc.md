# 改进 MediaLocker: wxPython、SQLAlchemy 和 MVC

> 原文：<https://www.blog.pythonlibrary.org/2011/11/30/improving-medialocker-wxpython-sqlalchemy-and-mvc/>

这个博客在本月早些时候发表了一篇关于 wxPython、SQLAlchemy、CRUD 和 MVC 的文章。我们在那篇文章中创建的程序被称为“MediaLocker”，不管它是否被明确地这样表述。无论如何，从那以后，我收到了一些关于改进程序的评论。一条来自 SQLAlchemy 本身的创意者之一 Michael Bayer，另一条来自 Werner Bruhin，一个经常出现在 wxPython 邮件列表中帮助新用户的好人。因此，我按照他们的建议着手创建代码的改进版本。沃纳随后对其进行了进一步的改进。所以在这篇文章中，我们将着眼于改进代码，首先是我的例子，然后是他的例子。尽管说得够多了；让我们进入故事的实质吧！

## 让 MediaLocker 变得更好

Michael Bayer 和 Werner Bruhin 都认为我应该只连接数据库一次，因为这是一个相当“昂贵”的操作。如果同时存在多个会话，这可能是一个问题，但即使在我的原始代码中，我也确保关闭会话，这样就不会发生这种情况。当我编写最初的版本时，我考虑过将会话创建分离出来，但最终采用了我认为更简单的方法。为了解决这个棘手的问题，我修改了代码，这样我就可以传递会话对象，而不是不断地调用控制器的 **connectToDatabase** 函数。你可以阅读更多关于会议[在这里](http://www.sqlalchemy.org/docs/orm/session.html?highlight=session)。请看来自 **mediaLocker.py** 的代码片段:

```py

class BookPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)

        if not os.path.exists("devdata.db"):
            controller.setupDatabase()

        self.session = controller.connectToDatabase()
        try:
            self.bookResults = controller.getAllRecords(self.session)
        except:
            self.bookResults = []

```

注意，我们前面有一个小的条件，如果数据库还不存在，它将创建数据库。接下来，我在主 GUI 中创建会话对象，作为 panel 子类的属性。然后我把它传到我需要的地方。上面可以看到一个例子，我将会话对象传递给控制器的 **getAllRecords** 方法。

另一个大的变化是从 **model.py** 中删除了 ObjectListView 模型，而只使用 SQLAlchemy 表类:

```py

########################################################################
class Book(DeclarativeBase):
    """"""
    __tablename__ = "book"

    id = Column(Integer, primary_key=True)
    author_id = Column(Integer, ForeignKey("person.id"))
    title = Column(Unicode)
    isbn = Column(Unicode)
    publisher = Column(Unicode)
    person = relation("Person", backref="books", cascade_backrefs=False)

    @property
    def author(self):
        return "%s %s" % (self.person.first_name, self.person.last_name)

```

除了使用 SQLAlchemy 构造之外，这实际上与原始类基本相同。我还需要添加一个特殊的属性来返回作者的全名，以便在我们的小部件中显示，所以我们使用了 Python 的内置函数: [property](http://docs.python.org/library/functions.html#property) ，它返回一个 property 属性。如果只看代码的话更容易理解。如您所见，我们将**属性**作为装饰器应用于**作者**方法。

## 沃纳增补

沃纳的补充大多是在模型中增加更明确的进口。模型的最大变化如下:

```py

import sys
if not hasattr(sys, 'frozen'):
    # needed when having multiple versions of SA installed
    import pkg_resources
    pkg_resources.require("sqlalchemy") # get latest version

import sqlalchemy as sa
import sqlalchemy.orm as sao
import sqlalchemy.ext.declarative as sad
from sqlalchemy.ext.hybrid import hybrid_property

maker = sao.sessionmaker(autoflush=True, autocommit=False)
DBSession = sao.scoped_session(maker)

class Base(object):
    """Extend the base class

    - Provides a nicer representation when a class instance is printed.
        Found on the SA wiki, not included with TG
    """
    def __repr__(self):
        return "%s(%s)" % (
                 (self.__class__.__name__),
                 ', '.join(["%s=%r" % (key, getattr(self, key))
                            for key in sorted(self.__dict__.keys())
                            if not key.startswith('_')]))

DeclarativeBase = sad.declarative_base(cls=Base)
metadata = DeclarativeBase.metadata

def init_model(engine):
    """Call me before using any of the tables or classes in the model."""
    DBSession.configure(bind=engine)

```

前几行是为在机器上安装了 SetupTools / easy_install 的人准备的。如果用户安装了多个版本的 SQLALchemy，它将强制用户使用最新的版本。大多数其他导入都被缩短了，以使各种类和属性的来源变得非常明显。老实说，我对 **hybrid_property** 并不熟悉，所以下面是它的 docstring 所说的:

一个装饰器，允许用实例级和类级行为定义 Python 描述符。

你可以在这里阅读更多:[http://www.sqlalchemy.org/docs/orm/extensions/hybrid.html](http://www.sqlalchemy.org/docs/orm/extensions/hybrid.html)

Werner 还在基类中添加了一个小的 **__repr__** 方法，使它在打印时返回一个更好的类实例表示，这对于调试来说很方便。最后，他添加了一个名为 **init_model** 的函数来初始化模型。

## 包扎

现在您应该知道，Werner 和我已经决定将 MediaLocker 做成一个支持 wxPython 数据库的应用程序的例子。自从我上面提到的简单编辑之后，他已经在这上面做了很多工作。我们将很快就此发布官方声明。与此同时，我希望这有助于打开你的眼界，找到一些有趣的方法来增强一个项目，并使它变得干净一点。我的计划是给这个程序增加很多新的特性，并且除了我所有的其他文章之外，在这个博客上记录这些特性。

## 源代码

*   [wxasa 2 . zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2011/11/wxSa2.zip)
*   [wxSa2.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2011/11/wxSa2.tar)