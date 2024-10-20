# SqlAlchemy 编程错误 42000 和 MS SQL

> 原文：<https://www.blog.pythonlibrary.org/2011/01/15/sqlalchemy-programmingerror-42000-and-ms-sql/>

我最近一直在使用 Windows XP 上的 SqlAlchemy 编写软件清单脚本，以连接到 Microsoft SQL Server 2005 数据库中新创建的表。我使用 Aqua Data Studio 创建了这个表，以 SQL Administrator (sa)的身份登录，并认为一切都很好，直到我尝试向这个表提交一些数据。下面是我收到的一个奇怪的错误:

```py

pyodbc.ProgrammingError: ('42000', '[42000] [Microsoft][ODBC SQL Server Driver][SQL Server]Cannot find the object "dbo.software" because it does not exist or you do not have permissions. (1088) (SQLExecDirectW)')

```

 现在，是什么意思？我知道这个数据库存在，因为我刚刚用 Aqua Data Studio(一个数据库管理套件)创建了它，我给了自己以下权限:选择、插入、更新和删除。如果你用谷歌搜索这个错误，你会发现对一个[线程](http://groups.google.com/group/sqlalchemy/browse_thread/thread/78a3912426d48cba?fwc=2)的四个引用。他们在线程中没有真正的解决方案，尽管他们提到添加权限来改变模式可能会起作用。请注意，只有在使用声明性语法时，这个问题才会发生。无论如何，在这一点上，你可能想看我的代码，所以看一看:

```py

from sqlalchemy import create_engine, Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

uri = "mssql://username:pw@mssqlServerPath/Inventory"
engine = create_engine(uri)
engine.echo = True
Base = declarative_base(engine)

########################################################################
class Software(Base):
    """
    SqlAlchemy table representation of "software" login
    """
    __tablename__ = "software"
    __table_args__ = {"schema":"dbo"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    date_added = Column(DateTime)
    date_checked = Column(DateTime)
    machine_name = Column(String(25))
    version = Column(String(25))

    #----------------------------------------------------------------------
    def __init__(self, name, date_added, date_checked, machine_name, version):
        """"""
        self.name = name
        self.date_added = date_added
        self.date_checked = date_checked
        self.machine_name = machine_name
        self.version = version

Session = sessionmaker(bind=engine)
session = Session()
now = datetime.datetime.now()
new_record = Software("Adobe Acrobat", now, now, "MCIS0467", "9.0")
session.add(new_record)
session.commit()

```

据我的老板所知，声明性语法使用基于身份的查询系统，如果没有设置身份，那么 SqlAlchemy 就找不到数据库。如果使用 SqlAlchemy 创建表，那么就不会有这个问题。无论如何，事实证明您必须使用 Microsoft SQL Server Management Studio 来纠正这种特殊情况。加载它并导航到正确的数据库和表，然后打开 columns 树。右键单击主键字段，然后选择“修改”。在屏幕底部有一个“列属性”标签。转到那里，向下滚动到 Identity Specification 并展开。最后，确保“(Is Identity)”字段设置为“是”。保存它，你就完成了！

这是一个非常奇怪和罕见的问题，所以你可能不会有它，但我认为这是一个有趣的问题，我想记录解决方案。注意:我在 Windows XP 上使用 Python 2.5 和 SqlAlchemy 0.6.6。