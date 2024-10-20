# Python:一个简单的逐步 SQLite 教程

> 原文：<https://www.blog.pythonlibrary.org/2012/07/18/python-a-simple-step-by-step-sqlite-tutorial/>

SQLite 是一个独立的、无服务器的、免配置的事务 SQL 数据库引擎。Python 在 2.5 版本中获得了 sqlite3 模块，这意味着您可以使用任何当前的 Python 创建 sqlite 数据库，而无需下载任何额外的依赖项。Mozilla 为其流行的 Firefox 浏览器使用 SQLite 数据库来存储书签和其他各种信息。在本文中，您将了解以下内容:

*   如何创建 SQLite 数据库
*   如何向表格中插入数据
*   如何编辑数据
*   如何删除数据
*   基本 SQL 查询

这篇文章在功能上类似于本月早些时候出现在这个网站上的最近的 [SQLAlchemy 教程](https://www.blog.pythonlibrary.org/2012/07/01/a-simple-sqlalchemy-0-7-0-8-tutorial/)。如果你想直观地检查你的数据库，你可以使用火狐的 [SQLite 管理器插件](https://addons.mozilla.org/en-US/firefox/addon/sqlite-manager/)，或者如果你喜欢命令行，你可以使用 SQLite 的[命令行外壳](http://www.sqlite.org/sqlite.html)

### 如何创建数据库并插入一些数据

在 SQLite 中创建一个数据库确实很容易，但是这个过程需要你懂一点 SQL。下面的代码将创建一个数据库来保存音乐专辑:

```py

import sqlite3

conn = sqlite3.connect("mydatabase.db") # or use :memory: to put it in RAM

cursor = conn.cursor()

# create a table
cursor.execute("""CREATE TABLE albums
                  (title text, artist text, release_date text, 
                   publisher text, media_type text) 
               """)

```

首先，我们必须导入 **sqlite3** 库，并创建一个到数据库的连接。你可以给它一个文件路径，文件名或者使用特殊字符串“:memory:”在内存中创建数据库。在我们的例子中，我们在磁盘上一个名为 **mydatabase.db** 的文件中创建了它。接下来，我们创建一个 cursor 对象，它允许您与数据库交互并添加记录等。这里我们使用 SQL 语法创建一个名为**相册**的表格，有 5 个文本字段:标题、艺术家、发行日期、出版商和媒体类型。SQLite 只支持五种[数据类型](http://www.sqlite.org/datatype3.html) : null、integer、real、text 和 blob。让我们在这段代码的基础上，将一些数据插入到我们的新表中！

```py

# insert some data
cursor.execute("INSERT INTO albums VALUES ('Glow', 'Andy Hunter', '7/24/2012', 'Xplore Records', 'MP3')")

# save data to database
conn.commit()

# insert multiple records using the more secure "?" method
albums = [('Exodus', 'Andy Hunter', '7/9/2002', 'Sparrow Records', 'CD'),
          ('Until We Have Faces', 'Red', '2/1/2011', 'Essential Records', 'CD'),
          ('The End is Where We Begin', 'Thousand Foot Krutch', '4/17/2012', 'TFKmusic', 'CD'),
          ('The Good Life', 'Trip Lee', '4/10/2012', 'Reach Records', 'CD')]
cursor.executemany("INSERT INTO albums VALUES (?,?,?,?,?)", albums)
conn.commit()

```

这里我们使用 INSERT INTO SQL 命令将一条记录插入数据库。请注意，每一项都必须用单引号括起来。当您需要插入包含单引号的字符串时，这可能会变得复杂。无论如何，要将记录保存到数据库，我们必须**提交**它。下一段代码展示了如何使用光标的 **executemany** 方法一次添加多条记录。注意我们用的是问号(？)而不是字符串替换(%s)来插入值。使用字符串替换是不安全的，也不应该使用，因为它会允许 [SQL 注入](http://en.wikipedia.org/wiki/SQL_injection)攻击发生。问号方法要好得多，使用 SQLAlchemy 甚至更好，因为它为您完成了所有的转义，这样您就不必为将嵌入的单引号转换成 SQLite 可以接受的内容而烦恼了。

### 更新和删除记录

能够更新数据库记录是保持数据准确的关键。如果你不能更新，那么你的数据将很快变得过时和无用。有时您也需要从数据中删除行。我们将在本节中讨论这两个主题。首先，我们来做一个更新！

```py

import sqlite3

conn = sqlite3.connect("mydatabase.db")
cursor = conn.cursor()

sql = """
UPDATE albums 
SET artist = 'John Doe' 
WHERE artist = 'Andy Hunter'
"""
cursor.execute(sql)
conn.commit()

```

这里我们使用 SQL 的 UPDATE 命令来更新相册表。您可以使用 SET 来更改一个字段，因此在本例中，我们将艺术家字段设置为“安迪·亨特”的任何记录中的艺术家字段更改为“John Doe”。那不是很容易吗？请注意，如果您不提交更改，那么您的更改将不会写出到数据库中。删除命令几乎一样简单。我们去看看！

```py

import sqlite3

conn = sqlite3.connect("mydatabase.db")
cursor = conn.cursor()

sql = """
DELETE FROM albums
WHERE artist = 'John Doe'
"""
cursor.execute(sql)
conn.commit()

```

删除甚至比更新更容易。SQL 只有 2 行！在这种情况下，我们所要做的就是使用 WHERE 子句告诉 SQLite 从(相册)中删除哪个表，删除哪些记录。因此，它会在艺术家字段中查找任何包含“John Doe”的记录，然后将其删除。

### 基本 SQLite 查询

SQLite 中的查询与其他数据库(如 MySQL 或 Postgres)中的查询非常相似。您只需使用普通的 SQL 语法来运行查询，然后让游标对象执行 SQL。这里有几个例子:

```py

import sqlite3

conn = sqlite3.connect("mydatabase.db")
#conn.row_factory = sqlite3.Row
cursor = conn.cursor()

sql = "SELECT * FROM albums WHERE artist=?"
cursor.execute(sql, [("Red")])
print cursor.fetchall()  # or use fetchone()

print "\nHere's a listing of all the records in the table:\n"
for row in cursor.execute("SELECT rowid, * FROM albums ORDER BY artist"):
    print row

print "\nResults from a LIKE query:\n"
sql = """
SELECT * FROM albums 
WHERE title LIKE 'The%'"""
cursor.execute(sql)
print cursor.fetchall()

```

我们执行的第一个查询是一个 **SELECT *** ，这意味着我们希望选择所有与我们传入的艺术家姓名匹配的记录，在本例中是“Red”。接下来，我们执行 SQL 并使用 fetchall()返回所有结果。还可以使用 fetchone()来获取第一个结果。你还会注意到有一个注释掉的部分与一个神秘的 **row_factory** 有关。如果您取消对该行的注释，结果将作为行对象返回，这有点像 Python 字典，让您可以像字典一样访问该行的字段。但是，不能对行对象进行项目分配。

第二个查询与第一个非常相似，但是它返回数据库中的每条记录，并按艺术家姓名以升序对结果进行排序。这也演示了我们如何对结果进行循环。最后一个查询显示了如何使用 SQL 的 LIKE 命令来搜索部分短语。在这种情况下，我们在整个表格中搜索以“the”开头的标题。百分号(%)是通配符。

### 包扎

现在您知道了如何使用 Python 创建 SQLite 数据库。您还可以创建、更新和删除记录，以及对数据库进行查询。走出去，开始你自己的整洁的数据库和/或在评论中分享你的经验！

### 进一步阅读

*   sqlite3 库的官方文档
*   Zetcode 的 SQLite [教程](http://zetcode.com/db/sqlitepythontutorial/)
*   一个简单的 SqlAlchemy 0.7 / 0.8 [教程](https://www.blog.pythonlibrary.org/2012/07/01/a-simple-sqlalchemy-0-7-0-8-tutorial/)

### 源代码

*   [sqlite_tut.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/07/sqlite_tut.zip)