# Python 中的高级 SQLite 用法

> 原文：<https://www.pythoncentral.io/advanced-sqlite-usage-in-python/>

继 SQLite3 系列之后，这篇文章是关于我们使用 SQLite3 模块时的一些高级主题。如果你错过了第一部分，你可以在这里找到它。

## 使用 SQLite 的日期和日期时间类型

有时我们需要在 SQLite3 数据库中插入和检索一些`date`和`datetime`类型。当使用日期或日期时间对象执行插入查询时，`sqlite3`模块调用默认适配器并将它们转换成 ISO 格式。当您执行查询来检索这些值时，`sqlite3`模块将返回一个字符串对象:

```py

>>> import sqlite3

>>> from datetime import date, datetime

>>>

>>> db = sqlite3.connect(':memory:')

>>> c = db.cursor()

>>> c.execute('''CREATE TABLE example(id INTEGER PRIMARY KEY, created_at DATE)''')

>>>

>>> # Insert a date object into the database

>>> today = date.today()

>>> c.execute('''INSERT INTO example(created_at) VALUES(?)''', (today,))

>>> db.commit()

>>>

>>> # Retrieve the inserted object

>>> c.execute('''SELECT created_at FROM example''')

>>> row = c.fetchone()

>>> print('The date is {0} and the datatype is {1}'.format(row[0], type(row[0])))

# The date is 2013-04-14 and the datatype is <class 'str'>

>>> db.close()

```

问题是，如果您在数据库中插入了一个日期对象，大多数情况下，当您检索它时，您期望的是一个日期对象，而不是一个字符串对象。将`PARSE_DECLTYPES`和`PARSE_COLNAMES`传递给`connect`方法可以解决这个问题:

```py

>>> import sqlite3

>>> from datetime import date, datetime

>>>

>>> db = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

>>> c = db.cursor()

>>> c.execute('''CREATE TABLE example(id INTEGER PRIMARY KEY, created_at DATE)''')

>>> # Insert a date object into the database

>>> today = date.today()

>>> c.execute('''INSERT INTO example(created_at) VALUES(?)''', (today,))

>>> db.commit()

>>>

>>> # Retrieve the inserted object

>>> c.execute('''SELECT created_at FROM example''')

>>> row = c.fetchone()

>>> print('The date is {0} and the datatype is {1}'.format(row[0], type(row[0])))

# The date is 2013-04-14 and the datatype is <class 'datetime.date'>

>>> db.close()

```

更改连接方法后，数据库现在返回一个日期对象。`sqlite3`模块使用列的类型返回正确的对象类型。因此，如果我们需要使用一个`datetime`对象，我们必须将表中的列声明为一个`timestamp`类型:

```py

>>> c.execute('''CREATE TABLE example(id INTEGER PRIMARY KEY, created_at timestamp)''')

>>> # Insert a datetime object

>>> now = datetime.now()

>>> c.execute('''INSERT INTO example(created_at) VALUES(?)''', (now,))

>>> db.commit()

>>>

>>> # Retrieve the inserted object

>>> c.execute('''SELECT created_at FROM example''')

>>> row = c.fetchone()

>>> print('The date is {0} and the datatype is {1}'.format(row[0], type(row[0])))

# The date is 2013-04-14 16:29:11.666274 and the datatype is <class 'datetime.datetime'>

```

如果您已经声明了一个列类型为`DATE`，但是您需要使用一个`datetime`对象，那么有必要修改您的查询以便正确解析该对象:

```py

c.execute('''CREATE TABLE example(id INTEGER PRIMARY KEY, created_at DATE)''')

# We are going to insert a datetime object into a DATE column

now = datetime.now()

c.execute('''INSERT INTO example(created_at) VALUES(?)''', (now,))

db.commit()
#检索插入的对象
c . execute(' ' ' SELECT created _ at as " created _ at[timestamp]" FROM example ' ')

```

在 SQL 查询中使用`as "created_at [timestamp]"`将使适配器正确解析对象。

## 用 SQLite 的 executemany 插入多行

有时我们需要在数据库中插入一个对象序列，`sqlite3`模块提供了`executemany`方法来对序列执行 SQL 查询。

```py

# Import the SQLite3 module

import sqlite3

db = sqlite3.connect(':memory:')

c = db.cursor()

c.execute('''CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, phone TEXT)''')

users = [

    ('John', '5557241'),

    ('Adam', '5547874'),

    ('Jack', '5484522'),

    ('Monthy',' 6656565')

]
c.executemany(' ' '插入用户(姓名，电话)值(？,?)' ' ' '，用户)
 db.commit()
#打印用户
c . execute(' ' ' SELECT * FROM users ' ')
用于 c: 
打印(row)
db.close() 

```

请注意，序列的每个元素必须是一个元组。

## 用 SQLite 的 executescript 执行 SQL 文件

`execute`方法只允许您执行一个 SQL 语句。如果您需要执行几个不同的 SQL 语句，您应该使用`executescript`方法:

```py

# Import the SQLite3 module

import sqlite3

db = sqlite3.connect(':memory:')

c = db.cursor()

script = '''CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, phone TEXT);

            CREATE TABLE accounts(id INTEGER PRIMARY KEY, description TEXT);
插入到用户(姓名，电话)值(' John '，' 5557241 ')，
 ('Adam '，' 5547874 ')，(' Jack '，' 5484522 ')；' '
 c.executescript(脚本)
#打印结果
c . execute(' ' ' SELECT * FROM users ' ')
用于 c: 
中的行打印(行)
db.close() 

```

如果需要从文件中读取脚本:
【python】
FD = open(' myscript . SQL '，' r ')
script = FD . read()
c . execute script(script)
FD . close()

请记住，为了捕捉异常，用一个`try/except/else`子句包围代码是一个好主意。要了解更多关于`try/except/else`关键字的信息，请查看[捕捉 Python 异常——try/except/else 关键字](https://www.pythoncentral.io/catching-python-exceptions-the-try-except-else-keywords/ "Catching Python Exceptions – The try/except/else keywords")一文。

## 定义 SQLite SQL 函数

有时我们需要在语句中使用自己的函数，特别是当我们为了完成某个特定的任务而插入数据时。一个很好的例子是，当我们在数据库中存储密码时，我们需要加密这些密码:

```py

import sqlite3 #Import the SQLite3 module

import hashlib
def encrypt _ password(password):
#不要在真实环境中使用此算法
encrypted _ pass = hashlib . sha1(password . encode(' utf-8 '))。hexdigest() 
返回加密 _ 通行证
db = sqlite3 . connect(':memory:')
#注册函数
 db.create_function('encrypt '，1，encrypt _ password)
c = db . cursor()
c . execute(' ' '创建表用户(id 整数主键，电子邮件文本，密码文本)' ')
 user = ('johndoe@example.com '，' 12345678') 
 c.execute(' ' '插入用户(电子邮件，密码)值(？，加密(？))'''，用户)

```

`create_function`接受 3 个参数:`name`(用于在语句中调用函数的名称)、函数期望的参数数量(本例中为 1 个参数)和一个可调用对象(函数本身)。为了使用我们注册的函数，我们在语句中使用`encrypt()`来调用它。

最后，当您存储密码时，请使用真正的加密算法！