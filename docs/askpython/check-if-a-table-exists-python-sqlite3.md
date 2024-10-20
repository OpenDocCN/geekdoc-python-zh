# 检查表是否存在–Python SQLite3

> 原文：<https://www.askpython.com/python-modules/check-if-a-table-exists-python-sqlite3>

嘿读者们！在本文中，我们将学习如何使用 SQLite3 来检查一个表是否存在。这对我们来说很容易，因为我们将使用 Python 及其内置模块 SQLite3。所以，让我们努力吧。

**请注意:您应该熟悉 SQLite3 和 SQL 命令。**

***也读:[如何在 Sqlite3 数据库中插入多条记录](https://www.askpython.com/python/examples/insert-multiple-records-sqlite3)***

## 这篇文章涵盖了什么？

1.  ***创建数据库。***
2.  ***给它添加一些数据。***
3.  ***故意删除表格。***
4.  ***用 Python 创建程序检查表格是否存在***

### 使用 Python SQLite3 创建数据库

在本节中，我们将创建一个名为 **company** 的示例数据库，并向其中添加一个 **employee** 表。该表包含该公司员工的基本信息。确保你创建了一个新的工作目录来保存所有的东西。

**代码:**

```py
import sqlite3

connection = sqlite3.connect('databases/company.db') # file path

# create a cursor object from the cursor class
cur = connection.cursor()

cur.execute('''
   CREATE TABLE employee(
       emp_id integer,
       name text,
       designation text,
       email text
       )''')

print("Database created successfully!!!")
# committing our connection
connection.commit()

# close our connection
connection.close()

```

**输出:**

```py
Database created successfully!!!

```

这将在**数据库**文件夹中添加一个 **"company.db"** 文件。这个文件包含我们的雇员表。这是一个空表，所以让我们给它添加一些数据。

### 使用 Python SQLite3 向表中添加数据

使用 **"executemany()"** 函数，我们可以在表中一次插入多条记录。因此，我们将在这里使用相同的:

**代码:**

```py
import sqlite3

connection = sqlite3.connect('databases/company.db') # file path

cur = connection.cursor()

# creating a list of items

records = [(100, 'Arvind Sharma', 'Software Engineer', '[email protected]'),
           (102, 'Neha Thakur', 'Project Manager', '[email protected]'),
           (103, 'Pavitra Patil', 'Database Engineer', '[email protected]')]

cur.executemany("INSERT INTO employee VALUES (?,?,?,?)", records)

print('Data added successfully!!!')
connection.commit()

# close our connection
connection.close()

```

**输出:**

```py
Data added successfully!!!

```

这些是我们刚刚通过 Python 脚本添加的记录。

### 故意删除表格

现在，我们将特意删除该表。我们使用默认 SQL 的 **DROP TABLE** 命令。

**代码:**

```py
import sqlite3
connection = sqlite3.connect('databases/company.db')
connection.execute("DROP TABLE employee")
print("Your table has been deleted!!!")
connection.close()

```

**输出:**

```py
Your table has been deleted!!!

```

### 使用 Python SQLite3 检查表是否存在

现在，检查该表是否存在。我们需要编写一个代码来尝试定位该表，如果没有找到，它应该返回一个类似于:**“Table not found！!"**。为此， **fetchall()** 函数很有用。这使我们能够**检索/访问**SQL 中一个表包含的所有信息。这将返回它获得的所有信息的列表。

**逻辑:**

1.  **SELECT * FROM table name** 命令试图从数据库中检索整个表。
2.  如果该表存在，它将使用 **fetchall()** 函数将其存储在一个名为 **data_list** 的列表中。
3.  如果数据存在，它将把它存储在一个列表中。
4.  如果不存在表，那么它将从 sqlite3 模块中抛出 **OperationalError** 。
5.  通过 except block 处理，然后打印一条消息**“无此表:table _ name”**。

**代码:**

```py
import sqlite3

connection = sqlite3.connect('databases/company.db')

cur = connection.cursor() 

try:
    cur.execute("SELECT * FROM employee")

    # storing the data in a list
    data_list = cur.fetchall() 
    print('NAME' + '\t\tEMAIL')
    print('--------' + '\t\t-------------')
    for item in items:
        print(item[0] + ' | ' + item[1] + '\t' + item[2])   

except sqlite3.OperationalError:
    print("No such table: employee")

connection.commit()
connection.close()

```

**输出:**

```py
No such table: employee

```

因此，通过这种方式，我们可以检测特定表中的表是否存在于数据库中。

## 结论

就这样，本文到此结束。我希望读者已经了解了如何使用 SQLite3 来使用数据库**。**这对 DBs 新手来说可能是一个巨大的帮助。