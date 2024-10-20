# Python 中如何使用 SQL？

> 原文：<https://www.askpython.com/python-modules/sql-in-python>

大多数现代应用程序都非常频繁地与数据库进行交互。SQL 或[结构化查询语言](https://www.askpython.com/python-modules/python-mysql-tutorial)使得访问和操作存储在这些数据库中的数据变得更加容易。

Python 作为流行语言之一，支持内置和第三方 SQL 库。

在下一节中，我们将讨论如何使用最流行的库之一 SQLite 在 Python 中使用 SQL。

## SQLite 简介

我们之所以选择 SQLite 作为我们的教程，是因为它的无服务器架构。SQLite 速度快，重量轻，将整个数据库存储在一个文件中，甚至存储在 PC 的内存(RAM)中。它在测试环境中被开发人员大量使用。

模块 SQLite 是 Python 自带的。因此，您不必使用 pip 从外部安装它。

但是不应该轻视 SQLite 的简单性，因为它也可以为生产就绪环境处理大型数据库。

所有这些特性使 SQLite 成为初学者和中级开发人员的完美模块。

## 在 Python 中使用 SQL 的步骤

按照下面的说明在 python 脚本中使用 SQL。

### 1.导入 SQLite

在 python 中使用任何模块的第一步都是在文件的最顶端导入它。在 Python3 中，该模块被称为“sqlite3”

```py
import sqlite3 #Importing the module

```

### 2.创建与数据库的连接

导入模块后，我们需要使用“connect()”方法创建一个数据库对象，并将数据库文件路径作为参数传递。

如果我们目前没有任何数据库，相同的命令将使用我们指定的文件路径/名称创建一个新的数据库。

```py
import sqlite3 #Importing the module

conn = sqlite3.connect("databasename.db")

""" Here, conn is the database object and 'databasename.db' is the actual database we're trying to connect with. 
If there is no database available then the same command will trigger to create a new database of the same name in our current directory."""

```

### 3.创建光标对象

一旦我们的数据库对象被创建，我们需要设置另一个[对象](https://www.askpython.com/python/oops/python-classes-objects)，它能够使用 Python 对数据库对象执行本地 SQL 命令。

为了实现这一点，我们需要做的就是在数据库对象上调用“cursor()”方法。所有 SQL 命令都必须使用游标对象来执行。

```py
curr = conn.cursor() #Here 'curr' is our new cursor object. 

```

### 4.使用 SQL 命令创建表

在这一节中，我们在当前的数据库上设置一个基本表，并学习如何提交它们，以便该表实际存储在文件中。

```py
# SQL command that creates a table in the database

createTableCommand = """ CREATE TABLE NSA_DATA (
username VARCHAR(50),
phonenumber VARCHAR(15),
password VARCHAR(50),
baddeedcount INT,
secrets VARCHAR(250)
);"""

# Executing the SQL command
curr.execute(createTableCommand)

# Commit the changes
conn.commit()

```

正如我们所看到的，首先，我们需要将 SQL 命令放入字符串形式。然后，我们对游标对象调用“execute()”方法，并将字符串作为参数传递。

最后，我们需要在数据库对象上调用“commit()”方法。否则，这些变化不会反映在我们实际的数据库中。因此，我们不能忘记提交更改。

### 5.向数据库添加数据

创建数据库模式后，我们要做的下一件事是添加数据。按照下面的命令学习如何操作:

```py
# First, we write our SQL command within a string and assign it to a variable addData
addData = """INSERT INTO NSA_DATA VALUES('abcd', '0123456789', 'Password1o1', 23, 'None Yet');"""
print("The data has been added!")

# Then we execute the command
curr.execute(addData)

# And finally commit
conn.commit()

```

输出:

```py
INSERT INTO NSA_DATA VALUES('abcd', '0123456789', 'Password1o1', 23, 'None Yet')
The data has been added!

```

但是，如果您有一个数据列表，并且希望导入到数据库中，而不需要逐一查看，这里有一种方法，可以将数据从 2D 阵列导入到数据库中。

```py
# The 2D array containing required data
data = [['abcd', '0123456789', 'Password1o1', 23, 'None Yet'],
        ['oswald', '0123456888', 'SunnyDay', 0, 'None Yet'],
        ['nobitanobi', '3216548876', 'ilovedoracake', 357, 'many of them']]

# A for loop to iterate through the data and add them one by one. 
for i in data:
    addData = f"""INSERT INTO NSA_DATA VALUES('{i[0]}', '{i[1]}', '{i[2]}', '{i[3]}', '{i[4]}')"""
    print(addData) # To see all the commands iterating
    curr.execute(addData)
print("Data added successfully!")

conn.commit()

```

输出:

```py
INSERT INTO NSA_DATA VALUES('abcd', '0123456789', 'Password1o1', '23', 'None Yet')
INSERT INTO NSA_DATA VALUES('oswald', '0123456888', 'SunnyDay', '0', 'None Yet')
INSERT INTO NSA_DATA VALUES('nobitanobi', '3216548876', 'ilovedoracake', '357', 'many of them')
Data added successfully!

```

### 6.获取数据

最后，我们还需要从数据库中提取数据，以满足我们日常的技术需求。这个过程与我们在上一节中所做的非常相似，只是有一点小小的变化。

一旦我们使用 cursor 对象执行搜索查询，它不会立即返回结果。相反，我们需要在游标上使用方法“fetchall()”来获取数据。

```py
# Our search query that extracts all data from the NSA_DATA table.  
fetchData = "SELECT * from NSA_DATA"

# Notice that the next line of code doesn't output anything upon execution. 
curr.execute(fetchData)

# We use fetchall() method to store all our data in the 'answer' variable
answer = curr.fetchall()

# We print the data
for data in answer:
    print(data)

```

输出:

```py
('abcd', '0123456789', 'Password1o1', 23, 'None Yet')
('abcd', '0123456789', 'Password1o1', 23, 'None Yet')
('oswald', '0123456888', 'SunnyDay', 0, 'None Yet')
('nobitanobi', '3216548876', 'ilovedoracake', 357, 'many of them')
('abcd', '0123456789', 'Password1o1', 23, 'None Yet')
('oswald', '0123456888', 'SunnyDay', 0, 'None Yet')
('nobitanobi', '3216548876', 'ilovedoracake', 357, 'many of them')

```

## 结论

希望您已经学会了如何使用 Python 执行基本的 SQL 操作。您还应该注意到，SQLite 不是唯一可用的库。对于生产级别的工作，更高级别的数据库如 PostgreSQL 和 MySQL 是非常推荐的。虽然 python 中的用法基本相同。

## 完整的代码:

以下部分包含本教程中使用的完整代码。

### 在 Python 中使用 SQL 创建表

```py
import sqlite3

conn = sqlite3.connect("database.db")
curr = conn.cursor()

createTableCommand = """CREATE TABLE NSA_DATA (
username VARCHAR(50),
phonenumber VARCHAR(15),
password VARCHAR(50),
baddeedcount INT,
secrets VARCHAR(250)
);"""

try: 
    curr.execute(createTableCommand)
    print("Table Successfully Created!")
except:
    print("There was an error with Table creation")
finally:
    conn.commit()

```

输出:

```py
Table Successfully Created!

```

### 在 Python 中通过 SQL 添加数据

```py
import sqlite3

conn = sqlite3.connect("database.db")
curr = conn.cursor()

# The 2D array containing required data
data = [['abcd', '0123456789', 'Password1o1', 23, 'None Yet'],
        ['oswald', '0123456888', 'SunnyDay', 0, 'None Yet'],
        ['nobitanobi', '3216548876', 'ilovedoracake', 357, 'many of them']]

# A for loop to iterate through the data and add them one by one. 
for i in data:
    addData = f"""INSERT INTO NSA_DATA VALUES('{i[0]}', '{i[1]}', '{i[2]}', '{i[3]}', '{i[4]}')"""
    print(addData) # To see all the commands iterating
    curr.execute(addData)
print("Data added successfully!")

conn.commit()

```

输出:

```py
INSERT INTO NSA_DATA VALUES('abcd', '0123456789', 'Password1o1', '23', 'None Yet')
INSERT INTO NSA_DATA VALUES('oswald', '0123456888', 'SunnyDay', '0', 'None Yet')
INSERT INTO NSA_DATA VALUES('nobitanobi', '3216548876', 'ilovedoracake', '357', 'many of them')
Data added successfully!

```

### 在 Python 中使用 SQL 提取数据

```py
import sqlite3

conn = sqlite3.connect("database.db")
curr = conn.cursor()

fetchData = "SELECT * from NSA_DATA"

curr.execute(fetchData)

# We use fetchall() method to store all our data in the 'answer' variable
answer = curr.fetchall()

# We print the data
for data in answer:
    print(data)

```

输出:

```py
('abcd', '0123456789', 'Password1o1', 23, 'None Yet')
('oswald', '0123456888', 'SunnyDay', 0, 'None Yet')
('nobitanobi', '3216548876', 'ilovedoracake', 357, 'many of them')

```

## 参考

[Python sqlite3 官方文档](https://docs.python.org/3/library/sqlite3.html)