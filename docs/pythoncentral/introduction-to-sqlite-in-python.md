# Python 中的 SQLite 简介

> 原文：<https://www.pythoncentral.io/introduction-to-sqlite-in-python/>

SQLite3 是一个非常容易使用的数据库引擎。它是独立的、无服务器的、零配置的和事务性的。它非常快速和轻量级，并且整个数据库存储在单个磁盘文件中。它在许多应用中被用作内部数据存储。Python 标准库包括一个名为“sqlite3”的模块，用于处理该数据库。这个模块是一个符合 DB-API 2.0 规范的 SQL 接口。

## **使用 Python 的 SQLite 模块**

要使用 SQLite3 模块，我们需要在 python 脚本中添加一条 import 语句:

```py

import sqlite3

```

### **将 SQLite 连接到数据库**

我们使用函数`sqlite3.connect`来连接数据库。我们可以使用参数“:memory:”在 RAM 中创建一个临时 DB，或者传递一个文件名来打开或创建它。

```py

# Create a database in RAM

db = sqlite3.connect(':memory:')

# Creates or opens a file called mydb with a SQLite3 DB

db = sqlite3.connect('data/mydb')

```

当我们使用完数据库后，我们需要关闭连接:

```py

db.close()

```

### **创建(CREATE)和删除(DROP)表格**

为了对数据库进行任何操作，我们需要获得一个游标对象，并将 SQL 语句传递给游标对象来执行它们。最后，提交变更是必要的。我们将创建一个包含姓名、电话、电子邮件和密码列的用户表。

```py

# Get a cursor object

cursor = db.cursor()

cursor.execute('''

CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT,

phone TEXT, email TEXT unique, password TEXT)

''')

db.commit()

```

要删除表格:

```py

# Get a cursor object

cursor = db.cursor()

cursor.execute('''DROP TABLE users''')

db.commit()

```

请注意，提交函数是在 db 对象上调用的，而不是在游标对象上。如果我们输入`cursor.commit`，我们将得到`AttributeError: 'sqlite3.Cursor' object has no attribute 'commit'`

### **将数据插入数据库**

为了插入数据，我们使用光标来执行查询。如果需要 Python 变量的值，建议使用“？”占位符。不要使用字符串操作或连接来进行查询，因为非常不安全。在本例中，我们将在数据库中插入两个用户，他们的信息存储在 python 变量中。

```py

cursor = db.cursor()

name1 = 'Andres'

phone1 = '3366858'

email1 = 'user@example.com'

# A very secure password

password1 = '12345'
name 2 = ' John '
phone 2 = ' 5557241 '
email 2 = ' John doe @ example . com '
password 2 = ' abcdef '
# Insert user 1
cursor . execute(' ' ' Insert INTO users(name，phone，email，password) 
 VALUES(？,?,?,?)“”，(姓名 1，电话 1，电子邮件 1，密码 1)) 
打印('插入第一个用户')
# Insert user 2
cursor . execute(' ' ' Insert INTO users(name，phone，email，password) 
 VALUES(？,?,?,?)“”，(姓名 2，电话 2，电子邮件 2，密码 2)) 
打印('插入第二个用户')
db.commit() 

```

Python 变量的值在元组内部传递。另一种方法是使用“:keyname”占位符传递字典:

```py

cursor.execute('''INSERT INTO users(name, phone, email, password)

VALUES(:name,:phone, :email, :password)''',

{'name':name1, 'phone':phone1, 'email':email1, 'password':password1})

```

如果需要插入几个用户使用`executemany`和一个带有元组的列表:

```py

users = [(name1,phone1, email1, password1),

(name2,phone2, email2, password2),

(name3,phone3, email3, password3)]

cursor.executemany(''' INSERT INTO users(name, phone, email, password) VALUES(?,?,?,?)''', users)

db.commit()

```

如果您需要获得刚刚插入的行的 id，请使用`lastrowid`:

```py

id = cursor.lastrowid

print('Last row id: %d' % id)

```

### **用 SQLite 检索数据(选择)**

要检索数据，对 cursor 对象执行查询，然后使用`fetchone()`检索单个行，或者使用`fetchall()`检索所有行。

```py

cursor.execute('''SELECT name, email, phone FROM users''')

user1 = cursor.fetchone() #retrieve the first row

print(user1[0]) #Print the first column retrieved(user's name)

all_rows = cursor.fetchall()

for row in all_rows:

# row[0] returns the first column in the query (name), row[1] returns email column.

print('{0} : {1}, {2}'.format(row[0], row[1], row[2]))

```

光标对象作为迭代器工作，自动调用`fetchall()`:

```py

cursor.execute('''SELECT name, email, phone FROM users''')

for row in cursor:

# row[0] returns the first column in the query (name), row[1] returns email column.

print('{0} : {1}, {2}'.format(row[0], row[1], row[2]))

```

要使用条件检索数据，请再次使用“？”占位符:

```py

user_id = 3

cursor.execute('''SELECT name, email, phone FROM users WHERE id=?''', (user_id,))

user = cursor.fetchone()

```

### **更新(UPDATE)和删除(DELETE)数据**

更新或删除数据的过程与插入数据的过程相同:

```py

# Update user with id 1

newphone = '3113093164'

userid = 1

cursor.execute('''UPDATE users SET phone = ? WHERE id = ? ''',

(newphone, userid))
#删除 id 为 2
Delete _ userid = 2
cursor . execute(' ' '从 id =？'的用户中删除''，(删除用户标识，))
db.commit() 

```

### **使用 SQLite 交易**

事务是数据库系统的一个有用属性。它确保了数据库的原子性。使用`commit`保存更改:

```py

cursor.execute('''UPDATE users SET phone = ? WHERE id = ? ''',

(newphone, userid))

db.commit() #Commit the change

```

或者`rollback`回滚自上次调用`commit`以来对数据库的任何更改:

```py

cursor.execute('''UPDATE users SET phone = ? WHERE id = ? ''',

(newphone, userid))

# The user's phone is not updated

db.rollback()

```

请记住总是调用`commit`来保存更改。如果您使用`close`关闭连接或者与文件的连接丢失(可能程序意外结束)，未提交的更改将会丢失。

### **SQLite 数据库异常**

最佳实践是始终用 try 子句或上下文管理器围绕数据库操作:

```py

import sqlite3 #Import the SQLite3 module

try:

# Creates or opens a file called mydb with a SQLite3 DB

db = sqlite3.connect('data/mydb')

# Get a cursor object

cursor = db.cursor()

# Check if table users does not exist and create it

cursor.execute('''CREATE TABLE IF NOT EXISTS

users(id INTEGER PRIMARY KEY, name TEXT, phone TEXT, email TEXT unique, password TEXT)''')

# Commit the change

db.commit()

# Catch the exception

except Exception as e:

# Roll back any change if something goes wrong

db.rollback()

raise e

finally:

# Close the db connection

db.close()

```

在这个例子中，我们使用了 try/except/finally 子句来捕捉代码中的任何异常。`finally`关键字非常重要，因为它总是正确地关闭数据库连接。请参考这篇[文章](https://www.pythoncentral.io/catching-python-exceptions-the-try-except-else-keywords/ "Catching Python Exceptions – The try/except/else keywords")，了解更多关于例外的信息。请查看:

```py

# Catch the exception

except Exception as e:

raise e

```

这被称为一个无所不包的子句，这里只是作为一个例子，在实际应用中你应该捕捉一个特定的异常，如`IntegrityError`或`DatabaseError`，更多信息请参考 [DB-API 2.0 异常](https://www.python.org/dev/peps/pep-0249/ "Python DB-API 2.0 Exceptions")。

我们可以使用连接对象作为上下文管理器来自动提交或回滚事务:

```py

name1 = 'Andres'

phone1 = '3366858'

email1 = 'user@example.com'

# A very secure password

password1 = '12345'
try:
with db:
db . execute(' ' ' INSERT INTO users(name，phone，email，password) 
 VALUES(？,?,?,?)“”，(姓名 1，电话 1，电子邮件 1，密码 1)) 
除了 sqlite3。IntegrityError: 
打印('记录已经存在')
最后:
 db.close() 

```

在上面的例子中，如果 insert 语句引发了一个异常，事务将被回滚并打印消息；否则事务将被提交。请注意，我们在`db`对象上调用`execute`，而不是`cursor`对象。

### **SQLite 行工厂和数据类型**

下表显示了 SQLite 数据类型和 Python 数据类型之间的关系:

*   `None`类型转换为`NULL`
*   `int`类型转换为`INTEGER`
*   `float`类型转换为`REAL`
*   `str`类型转换为`TEXT`
*   `bytes`类型转换为`BLOB`

行工厂类`sqlite3.Row`用于通过名称而不是索引来访问查询的列:
【python】
db = sqlite3 . connect(' data/mydb ')
db . row _ factory = sqlite3。row
cursor = db . cursor()
cursor . execute(' ' '从用户中选择姓名、电子邮件、电话' ')
对于游标中的行:
# row['name']返回查询中的姓名列，row['email']返回电子邮件列。
打印(' {0} : {1}，{2} '。格式(行['姓名']，行['电子邮件']，行['电话'])
db . close()