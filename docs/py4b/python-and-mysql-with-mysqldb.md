# Python 和 MySQL 与 MySQLdb

> 原文：<https://www.pythonforbeginners.com/modules-in-python/python-and-mysql-with-mysqldb>

上周，我在寻找一个 Python 模块，可以用来与 MySQL 数据库服务器进行交互。MySQLdb 正在这么做。

“MySQLdb 是一个围绕 _mysql 的瘦 Python 包装器，这使得它与 Python DB API 接口(版本 2)兼容。实际上，为了提高效率，相当一部分实现 API 的代码都在 mysql 中。”

要安装和使用它，只需运行:sudo apt-get install python-mysqldb

完成后，您可以开始在脚本中导入 MySQLdb 模块。

我在亚历克斯·哈维的网站上找到了这段代码

```py
# Make the connection
connection = MySQLdb.connect(host='localhost',user='alex',passwd='secret',db='myDB')
cursor = connection.cursor()

# Lists the tables in demo
sql = "SHOW TABLES;"

# Execute the SQL query and get the response
cursor.execute(sql)
response = cursor.fetchall()

# Loop through the response and print table names
for row in response:
    print row[0] 
```

关于如何在 Python 中使用 MySQLdb 的更多例子，请看一下[Zetcode.com](http://zetcode.com/databases/mysqlpythontutorial/ "mysqlpythontutorial")。