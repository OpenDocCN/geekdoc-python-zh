# 另一个循序渐进的 SqlAlchemy 教程(第 1 部分，共 2 部分)

> 原文：<https://www.blog.pythonlibrary.org/2010/02/03/another-step-by-step-sqlalchemy-tutorial-part-1-of-2/>

很久以前(大约 2007 年，如果谷歌没看错的话)，有一个叫 [Robin Munn](http://wiki.wxpython.org/Robin%20Munn) 的 Python 程序员写了一篇非常好的关于 [SqlAlchemy](http://www.sqlalchemy.org/) 的[教程](http://www.rmunn.com/sqlalchemy-tutorial/tutorial.html)。它最初基于 0.1 版本，但针对较新的 0.2 版本进行了更新。然后，Munn 先生就这么消失了，教程再也没更新过。很长一段时间以来，我一直在考虑发布我自己版本的教程，最终决定就这么做。我希望你会发现这篇文章有帮助，因为我发现原来是。

## 入门指南

SqlAlchemy 通常被称为对象关系映射器(ORM ),尽管它比我使用过的任何其他 Python ORMs(如 SqlObject 或内置于 Django 的 ORM)功能都更加全面。SqlAlchemy 是由一个叫迈克尔·拜尔的人创立的。我也经常看到 [Jonathan Ellis 的](http://spyced.blogspot.com/)名字出现在项目中，尤其是在 [PyCon](http://www.pycon.org/) 上。

本教程将基于最新发布的 SqlAlchemy 版本:0.5.8。您可以通过执行以下操作来检查您的版本:

```py

import sqlalchemy
print sqlalchemy.__version__

```

*注意:我也将在 Windows 上使用 Python 2.5 进行测试。然而，这段代码在 Mac 和 Linux 上应该同样适用。如果您需要 SqlAlchemy 在 Python 3 上工作，那么您将需要 0.6 的 SVN 版本。该网站给出了如何获取代码的说明。* 

如果你没有 SqlAlchemy，你可以[从他们的网站下载](http://www.sqlalchemy.org/download.html)或者使用 easy_install，如果你已经安装了 [setuptools](http://pypi.python.org/pypi/setuptools) 。让我们看看如何:

在下载源代码的情况下，您需要提取它，然后打开一个控制台窗口(在 Windows 上，转到开始，运行并键入“cmd”，不带引号)。然后改变目录，直到你在解压缩的文件夹。

要安装 SQLAlchemy，您可以使用 pip:

 `pip install sqlalchemy` 

这也假设您的路径上有 pip。如果没有，那么使用完整路径来使用它(即 c:\python38\scripts\pip.exe 或其他)。

# 创建第一个脚本

现在我们开始使用 SqlAlchemy 创建我们的第一个示例。我们将创建一个简单的表来存储用户名、年龄和密码。

```py

from sqlalchemy import create_engine
from sqlalchemy import MetaData, Column, Table, ForeignKey
from sqlalchemy import Integer, String

engine = create_engine('sqlite:///tutorial.db',
                       echo=True)

metadata = MetaData(bind=engine)

users_table = Table('users', metadata,
                    Column('id', Integer, primary_key=True),
                    Column('name', String(40)),
                    Column('age', Integer),
                    Column('password', String),
                    )

addresses_table = Table('addresses', metadata,
                        Column('id', Integer, primary_key=True),
                        Column('user_id', None, ForeignKey('users.id')),
                        Column('email_address', String, nullable=False)                            
                        )

# create tables in database
metadata.create_all()

```

## 看得更深

正如您所看到的，我们需要从 sqlalchemy 包中导入各种各样的东西，即 create_engine、元数据、列、表、整数和字符串。然后，我们创建一个“引擎”，它基本上是一个知道如何使用您提供的凭证与所提供的数据库进行通信的对象。在这种情况下，我们使用不需要凭证的 Sqlite 数据库。仅在这个主题上就有[篇深入的文档](http://www.sqlalchemy.org/docs/dbengine.html)，你可以在那里阅读你最喜欢的数据库风格。还要注意，我们将 echo 设置为 True。这意味着 SqlAlchemy 将把它正在执行的所有 SQL 命令输出到 stdout。这对于调试很方便，但是当您准备将代码投入生产时，应该将其设置为 False。

接下来，我们创建一个元数据对象。这个来自 SqlAlchemy 团队的很酷的创造保存了所有的数据库元数据。它由 Python 对象组成，这些对象包含数据库的表和其他模式级对象的描述。我们可以在这里或者在代码末尾附近的 *create_all* 语句中将元数据对象绑定到我们的数据库。

最后一部分是我们如何以编程方式创建表。这是通过使用 SqlAlchemy 的表和列对象来实现的。注意，我们有各种可用的字段类型，比如 String 和 Integer。还有很多其他的。对于这个例子，我们创建一个数据库并将其命名为“users”，然后传入我们的元数据对象。接下来，我们将它放入列中。“id”列被设置为我们的主键。当我们向数据库中添加用户时，SqlAlchemy 会神奇地为我们增加这个值。“name”列是字符串类型，长度不超过 40 个字符。“年龄”列只是一个简单的整数,“密码”列只是设置为字符串。我们没有设置它的长度，但我们可能应该设置。 *addresses_table* 中唯一的主要区别是我们如何设置连接两个表的外键属性。基本上，我们通过将字符串中正确的字段名传递给 ForeignKey 对象来指向另一个表。

这个代码片段的最后一行实际上创建了数据库和表。您可以随时调用它，因为它会在尝试创建表之前检查指定表是否存在。这意味着您可以创建额外的表并调用 create_all，SqlAlchemy 将只创建新表。

SqlAlchemy 还提供了一种加载以前创建的表的方法:

```py

someTable = Table("users", metadata, autoload=True, schema="schemaName")

```

我注意到，在这个版本中，SqlAlchemy 对于在自动加载数据库时指定数据库模式变得非常挑剔。如果您遇到这个问题，您需要将以下内容添加到您的表定义中:schema="some schema "。更多信息，请参见[文档](http://www.sqlalchemy.org/docs/metadata.html#specifying-the-schema-name)。

## 插入

有几种方法可以从数据库中添加和提取信息。我们将首先查看底层方式，然后在本系列的另一部分中，我们将进入会话和声明式风格，它们往往更抽象一些。让我们看看将数据插入数据库的不同方法:

```py

# create an Insert object
ins = users_table.insert()
# add values to the Insert object
new_user = ins.values(name="Joe", age=20, password="pass")

# create a database connection
conn = engine.connect()
# add user to database by executing SQL
conn.execute(new_user)

```

上面的代码显示了如何使用一个连接对象来执行插入。首先，您需要通过调用表的 *insert* 方法来创建插入对象。然后，您可以使用插入的*值*方法来添加该行所需的值。接下来，我们通过引擎的*连接*方法创建连接对象。最后，我们在插入对象上调用连接对象的*执行*方法。这听起来有点复杂，但实际上很简单。

下面的代码片段展示了在没有连接对象的情况下进行插入的几种方法:

```py

# a connectionless way to Insert a user
ins = users_table.insert()
result = engine.execute(ins, name="Shinji", age=15, password="nihongo")

# another connectionless Insert
result = users_table.insert().execute(name="Martha", age=45, password="dingbat")

```

在这两种情况下，你都需要调用 table 对象的 *insert* 方法。基本上，在第二种情况下，你只需将引擎从画面中移除。我们要看的最后一个插入方法是如何插入多行:

```py

conn.execute(users_table.insert(), [
    {"name": "Ted", "age":10, "password":"dink"},
    {"name": "Asahina", "age":25, "password":"nippon"},
    {"name": "Evan", "age":40, "password":"macaca"}
])

```

这是不言自明的，但是要点是您需要使用前面的 Connection 对象并传递给它两个参数:表的 Insert 对象和包含列名和值对的字典列表。请注意，在这些示例中，通过使用 *execute* 方法，数据被提交到数据库。

现在让我们继续做选择。

## 选择

SqlAlchemy 提供了一组健壮的方法来完成选择。这里我们将重点介绍简单的方法。对于高级的东西，我推荐他们的[官方文档](http://www.sqlalchemy.org/docs/sqlexpression.html)和[邮件列表](http://groups.google.com/group/sqlalchemy)。一个最常见的例子是进行全选，让我们从这个例子开始:

```py

from sqlalchemy.sql import select

s = select([users_table])
result = s.execute()

for row in result:
    print row

```

首先我们必须从 sqlalchemy.sql 导入 *select* 方法，然后我们将表作为一个元素列表传递给它。最后，我们调用选择对象的*执行*方法，并将返回的数据存储在*结果*变量中。现在我们已经有了所有的结果，我们也许应该看看我们是否得到了我们所期望的。因此，我们为循环创建一个*来迭代结果。*

如果你需要元组列表中的所有结果而不是行对象，你可以执行以下操作:

```py

# get all the results in a list of tuples
conn = engine.connect()
res = conn.execute(s)
rows = res.fetchall()

```

如果您只需要返回第一个结果，那么您可以使用 fetchone()而不是 fetchall():

```py

res = conn.execute(s)
row = res.fetchone()

```

现在，让我们假设我们需要在我们的结果中得到更多一点的粒度。在下一个示例中，我们只想返回用户的姓名和年龄，而忽略他们的密码。

```py

s = select([users_table.c.name, users_table.c.age])
result = conn.execute(s)
for row in result:
    print row

```

嗯，那很简单。我们所要做的就是在 select 语句中指定列名。小“c”基本上意味着“列”，所以我们选择列名和列年龄。如果有多个表，那么 select 语句应该是这样的:

选择([表一，表二])

当然，这很可能会返回重复的结果，所以您需要做一些类似的事情来缓解这个问题:

s = select([tableOne，tableTwo]，table one . c . id = = table two . c . user _ id)

SqlAlchemy [文档](http://www.sqlalchemy.org/docs/sqlexpression.html)将第一个结果称为笛卡尔积，因为它导致第一个表中的每一行都是针对第二个表中的每一行生成的。上面的第二个陈述消除了这种烦恼。怎么会？这是使用这种形式的 select 来执行 WHERE 子句的方法。在本系列的下一部分中，我将展示一种不同的方式来进行会话的选择和 where。

这里还有几个例子，在评论中有解释:

```py

from sqlalchemy.sql import and_

# The following is the equivalent to 
# SELECT * FROM users WHERE id > 3
s = select([users_table], users_table.c.id > 3)

# You can use the "and_" module to AND multiple fields together
s = select(and_(users_table.c.name=="Martha", users_table.c.age < 25))

```

上面的代码说明了 SqlAlchemy 也可以在查询中使用操作符和连接词。我推荐阅读他们的文档以了解全部细节[这里](http://www.sqlalchemy.org/docs/sqlexpression.html)。

## 包扎

我想这是我们停下来的好地方。我们现在已经学习了如何创建数据库、添加行以及从数据库中选择数据。在我们系列的下一部分，我们将学习使用对象关系方法来做这件事的更流行的方法。我们还将了解一些其他关键主题，例如 SqlAlchemy 会话。我们还将了解 SqlAlchemy 中的连接是如何工作的。到时候见！

**延伸阅读**

*   [SQL 表达式教程](http://www.sqlalchemy.org/docs/sqlexpression.html)
*   [罗宾·穆恩的 SqlAlchemy 教程](http://www.rmunn.com/sqlalchemy-tutorial/tutorial.html)

**下载量**

*   [SqlAlchemy Demo One.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/01/exampleOne.zip)
*   [SqlAlchemy 演示 One.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/01/exampleOne.tar)