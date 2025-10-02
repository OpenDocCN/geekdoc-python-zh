# Database 数据库

# 多数据库使用

## 问题

如何在单独项目中应用多数据库?

## 解决办法

webpy 0.3 支持多数据库操作,并从 web 模块中移走数据库部分, 使其成为一个更典型的对象. 例子如下:

```py
import web

db1 = web.database(dbn='mysql', db='dbname1', user='foo')
db2 = web.database(dbn='mysql', db='dbname2', user='foo')

print db1.select('foo', where='id=1')
print db2.select('bar', where='id=5') 
```

增加, 更新, 删除和查询的方法跟原有单数据库操作类似.

当然, 你可以使用 host 和 port 参数来指定服务器地址和监听端口.

# db.select 查询

## 问题:

怎样执行数据库查询？

## 解决方案:

如果是 0.3 版本, 连接部分大致如下:

```py
db = web.database(dbn='postgres', db='mydata', user='dbuser', pw='') 
```

当获取数据库连接后, 可以这样执行查询数据库:

```py
# Select all entries from table 'mytable'
entries = db.select('mytable') 
```

select 方法有下面几个参数:

*   vars
*   what
*   where
*   order
*   group
*   limit
*   offset
*   _test

### vars

vars 变量用来填充查询条件. 如:

```py
myvar = dict(name="Bob")
results = db.select('mytable', myvar, where="name = $name") 
```

### what

what 是标明需要查询的列名, 默认是*, 但是你可以标明需要查询哪些列.

```py
results = db.select('mytable', what="id,name") 
```

### where

where 查询条件, 如:

```py
results = db.select('mytable', where="id>100") 
```

### order

排序方式:

```py
results = db.select('mytable', order="post_date DESC") 
```

### group

按 group 组排列.

```py
results = db.select('mytable', group="color") 
```

### limit

从多行中返回 limit 查询.

```py
results = db.select('mytable', limit=10) 
```

### offset

偏移量, 从第几行开始.

```py
results = db.select('mytable', offset=10) 
```

### _test

查看运行时执行的 SQL 语句:

```py
results = db.select('mytable', offset=10, _test=True) 
<sql: 'SELECT * FROM mytable OFFSET 10'> 
```

# db.upate 数据更新

### 问题

向数据库中更新数据。

### 解决方案

```py
import web

db = web.database(dbn='postgres', db='mydata', user='dbuser', pw='')
db.update('mytable', where="id = 10", value1 = "foo") 
```

在 查询 中有更多关于可用参数的信息。

该更新操作会返回更新的影响行数。

# db.delete 数据删除

### 问题

在数据库中删除数据。

### 解决办法

```py
import web

db = web.database(dbn='postgres', db='mydata', user='dbuser', pw='')
db.delete('mytable', where="id=10") 
```

上面接受 "using" 和 "vars" 参数。

删除方法返回被删除的影响行数。

# db.insert 向数据库中新增数据

### 问题

如何向数据加新增数据？

### 解决办法

在 0.3 中，数据库连接如下：

```py
db = web.database(dbn='postgres', db='mydata', user='dbuser', pw='') 
```

数据库连接写好以后，“insert” 操作如下：

```py
# 向 'mytable' 表中插入一条数据
sequence_id = db.insert('mytable', firstname="Bob",lastname="Smith",joindate=web.SQLLiteral("NOW()")) 
```

上面的操作带入了几个参数，我们来说明一下：

*   tablename
*   seqname
*   _test
*   **values

## tablename

表名，即你希望向哪个表新增数据。

## seqname

可选参数，默认 None。Set `seqname` to the ID if it's not the default, or to `False`.

## _test

`_test` 参数可以让你看到 SQL 的执行过程：

```py
results = db.select('mytable', offset=10, _test=True) 
><sql: 'SELECT * FROM mytable OFFSET 10'> 
```

## **values

字段参数。如果没有赋值，数据库可能创建默认值或者发出警告。

# 使用 db.query 进行高级数据库查询

### 问题：

您要执行的 SQL 语句如：高级的联接或计数。

### 解决：

webpy 不会尝试为您和您的数据库建立层。相反，它试图以方便的通用任务，走出自己的方式，当您需要做的更高级的主题。执行高级的数据库查询是没有什么不同。例如：

```py
import web

db = web.database(dbn='postgres', db='mydata', user='dbuser', pw='')

results = db.query("SELECT COUNT(*) AS total_users FROM users")
print results[0].total_users # -> prints number of entries in 'users' table 
```

或者是，使用一个 JOIN 示例:

```py
import web

db = web.database(dbn='postgres', db='mydata', user='dbuser', pw='')

results = db.query("SELECT * FROM entries JOIN users WHERE entries.author_id = users.id") 
```

为了防止 SQL 注入攻击，db.query 还接受了“vars”语法如下描述 db.select:

```py
results = db.query("SELECT * FROM users WHERE id=$id", vars={'id':10}) 
```

这将避免用户输入，如果你信任这个“id”变量。

# 怎样使用数据库事务处理

### 问题：

怎样使用数据库事务处理？

### 解决：

数据库对象有一个方法“transaction”,将启动一个新的事务，并返回事务对象。这个事务对象可以使用 commit 提交事务或 rollback 来回滚事务。

```py
import web

db = web.database(dbn="postgres", db="webpy", user="foo", pw="")
t = db.transaction()
try:
    db.insert('person', name='foo')
    db.insert('person', name='bar')
except:
    t.rollback()
    raise
else:
    t.commit() 
```

在 python 2.5+以上的版本，事务同样可以在段中使用：

```py
from __future__ import with_statement

db = web.databse(dbn="postgres", db="webpy", user="foo", pw="")

with db.transaction():
    db.insert('person', name='foo')
    db.insert('person', name='bar') 
```

它同样可能有一个嵌套的事务：

```py
def post(title, body, tags):
    t = db.transaction()
    try:
        post_id = db.insert('post', title=title, body=body)
        add_tags(post_id, tags)
    except:
        t.rollback()
    else:
        t.commit()

def add_tags(post_id, tags):
    t = db.transaction()
    try:
        for tag in tags:
            db.insert('tag', post_id=post_id, tag=tag)
    except:
        t.rollback()
    else:
        t.commit() 
```

嵌套的事务在 sqlite 中将被忽略，因为此特性不被 sqlite 支持。

# sqlalchemy

## 问题

如何在 web.py 中使用 sqlalchemy

## 方案

创建一个钩子并使用 sqlalchemy 的 scoped session ([`www.sqlalchemy.org/docs/05/session.html#unitofwork_contextual`](http://www.sqlalchemy.org/docs/05/session.html#unitofwork_contextual))

```py
import string
import random
import web

from sqlalchemy.orm import scoped_session, sessionmaker
from models import *

urls = (
    "/", "add",
    "/view", "view"
)

def load_sqla(handler):
    web.ctx.orm = scoped_session(sessionmaker(bind=engine))
    try:
        return handler()
    except web.HTTPError:
       web.ctx.orm.commit()
       raise
    except:
        web.ctx.orm.rollback()
        raise
    finally:
        web.ctx.orm.commit()

app = web.application(urls, locals())
app.add_processor(load_sqla)

class add:
    def GET(self):
        web.header('Content-type', 'text/html')
        fname = "".join(random.choice(string.letters) for i in range(4))
        lname = "".join(random.choice(string.letters) for i in range(7))
        u = User(name=fname
                ,fullname=fname + ' ' + lname
                ,password =542)
        web.ctx.orm.add(u)
        return "added:" + web.websafe(str(u)) \
                            + "<br/>" \
                            + '<a href="/view">view all</a>'

class view:
    def GET(self):
        web.header('Content-type', 'text/plain')
        return "\n".join(map(str, web.ctx.orm.query(User).all()))

if __name__ == "__main__":
    app.run() 
```

### models.py

```py
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String

engine = create_engine('sqlite:///mydatabase.db', echo=True)

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    fullname = Column(String)
    password = Column(String)

    def __init__(self, name, fullname, password):
        self.name = name
        self.fullname = fullname
        self.password = password

    def __repr__(self):
       return "<User('%s','%s', '%s')>" % (self.name, self.fullname, self.password)

users_table = User.__table__
metadata = Base.metadata

if __name__ == "__main__":
    metadata.create_all(engine) 
```

在跑程序之前,运行'python models.py'来初始化一次数据库.

# 整合 SQLite UDF (用户定义函数) 到 webpy 数据库层

问题：

用户在邮件列表中询问，我把它放在这里作为将来使用和参考。

解决：

您可以添加到 Python 函数到 SQLite，并让它们在您的查询调用。

示例：

```py
>>> import sqlite3 as db
>>> conn = db.connect(":memory:")
>>> conn.create_function("sign", 1, lambda val: val and (val > 0 and 1 or -1))
>>> cur = conn.cursor()
>>> cur.execute("select 1, -1")
<sqlite3.Cursor object at 0xb759f2c0>
>>> print cur.fetchall()
[(1, -1)]
>>> cur.execute("select sign(1), sign(-1), sign(0), sign(-99), sign(99)")
<sqlite3.Cursor object at 0xb759f2c0>
>>> print cur.fetchall()
[(1, -1, 0, -1, 1)]
>>> conn.close() 
```

在 webpy 中，你可以通过游标如 db._db_cursor().connection 取得连接对象的引用。

示例：

```py
>>> import web
>>> db = web.database(dbn="sqlite", db=":memory:")
>>> db._db_cursor().connection.create_function("sign", 1, lambda val: val and (val > 0 and 1 or -1))
>>> print db.query("select sign(1), sign(-1), sign(0), sign(-99), sign(99)").list()
[<Storage {'sign(1)': 1, 'sign(-1)': -1, 'sign(99)': 1, 'sign(-99)': -1, 'sign(0)': 0}>] 
```

# 使用字典动态构造 where 子句

## 问题

你希望创建一个字典来构造动态的 where 子句并且希望能够在查询语句中使用。

## 解决

```py
>>> import web
>>> db = web.database(dbn='postgres', db='mydb', user='postgres')
>>> where_dict = {'col1': 1, col2: 'sometext'}
>>> db.delete('mytable', where=web.db.sqlwhere(where_dict), _test=True)
<sql: "DELETE FROM mytable WHERE col1 = 1 AND col2 = 'sometext'"> 
```

## 解释

`web.db.sqlwhere` takes a Python dictionary as an argument and converts it into a string useful for where clause in different queries. You can also use an optional `grouping` argument to define the exact gouping of the individual keys. For instance:

`web.db.sqlwhere`将 Python 的字典作为参数并且转换为适用于不同的查询语句的 where 子句的 string 类型数据。你也可以使用`grouping`参数来定义链接字典中的 key 的链接字符。例子如下。

```py
>>> import web
>>> web.db.sqlwhere({'a': 1, 'b': 2}, grouping=' OR ')
'a = 1 OR b = 2' 
```

`grouping` 的默认值为 `' AND '`.