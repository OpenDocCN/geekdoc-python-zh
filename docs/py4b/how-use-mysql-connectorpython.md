# 如何使用 MySQL 连接器/Python

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/how-use-mysql-connectorpython>

MySQL Connector/Python 是 Oracle 自己发布的一个驱动程序，让用 Python 连接 MySQL 数据库变得更容易。MySQL connector/Python 支持 Python 2.0 及更高版本的所有版本，也包括 3.3 及更高版本。

要安装 MySQL Connector/Python，只需使用 pip 安装软件包。

```py
 pip install mysql-connect-python --allow-external mysql-connector-python 
```

连接到数据库并提交给它非常简单。

```py
 import mysql.connector

connection = mysql.connector.connect(user="username", 
                                     password="password", 
                                     host="127.0.0.1", 
                                     database="database_name")

cur = connection.cursor()

cur.execute("INSERT INTO people (name, age) VALUES ('Bob', 25);")
connection.commit()

cur.close()
connection.close() 
```