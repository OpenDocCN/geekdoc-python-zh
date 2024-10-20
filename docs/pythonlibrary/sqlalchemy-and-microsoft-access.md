# SqlAlchemy 和 Microsoft Access

> 原文：<https://www.blog.pythonlibrary.org/2010/10/10/sqlalchemy-and-microsoft-access/>

更新(2010 年 10 月 12 日)-我的一个提醒读者告诉我，SqlAlchemy 0.6.x 目前不支持访问方言。阅读[此处](http://www.sqlalchemy.org/docs/reference/dialects/index.html)了解更多信息。

一两年前，我被要求将一些数据从一些旧的 Microsoft Access 文件转移到我们的 Microsoft SQL Server。因为我喜欢使用 SqlAlchemy，所以我决定看看它是否支持 Access。在这方面，当时的文档是相当无用的，但它似乎是可能的，我在 SqlAlchemy 的 Google group 上找到了一个关于它的帖子。

连接到 Microsoft Access 的代码非常简单。事情是这样的:

```py

from sqlalchemy import create_engine
engine = create_engine(r'access:///C:/some/path/database.MDB')

```

看到这有多简单了吗？您只需告诉 SqlAlchemy 要连接哪种数据库，添加三个正斜杠，然后是文件的路径。一旦完成，您就可以对 Access 文件做任何事情，就像对普通数据库一样:

```py

########################################################################
class TableName(Base):
    """
    MS Access database
    """
    __tablename__ = "ROW"
    __table_args__ = ({"autoload":True})  # load the database
    FILENUM = Column("FILE #", Integer, key="FILENUM")

```

在上面的代码中，我使用 SqlAlchemy 的声明性语法来自动加载数据库的结构。我不记得这个数据库是否有它的主键集，但我猜它没有，因为我必须添加最后一行。

无论如何，一旦有了连接，就可以像平常一样运行查询了。在我的例子中，我最终创建了一个模型文件来保存 Access 文件和 SQL Server 数据库的所有表定义，然后对 Access 文件执行 SELECT *操作，循环遍历结果并将每一行插入到 SQL Server 中。唯一需要注意的是，Access 比 SQL Server 更能容忍空值，所以我不得不编写一些特殊的处理方法来应对这种情况。

嗯，这真的是所有的事情。您可以查看我的其他 SqlAlchemy 教程，了解更多关于使用 SqlAlchemy 与数据库进行交互的一般信息。