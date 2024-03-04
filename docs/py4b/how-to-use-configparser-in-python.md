# 如何在 Python 中使用 ConfigParser

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/how-to-use-configparser-in-python>

### 什么是配置解析器？

Python 中的 configparser 模块用于处理配置文件。

它非常类似于 Windows INI 文件。

您可以使用它来管理用户可编辑的应用程序配置文件。

配置文件被组织成几个部分，每个部分可以包含配置数据的名称-值对。

通过查找以[开头并以]结尾的行来标识配置文件的节。

方括号之间的值是节名，可以包含除方括号之外的任何字符。

选项在一个部分中每行列出一个。

该行以选项的名称开始，用冒号(:)或等号(=)与值分隔。

解析文件时，分隔符周围的空格将被忽略。

包含“bug_tracker”部分和三个选项的示例配置文件如下所示:

```py
 [bug_tracker]
url = http://localhost:8080/bugs/
username = dhellmann
password = SECRET 
```

### 常见用法

配置文件最常见的用途是让用户或系统管理员使用常规文本编辑器编辑文件，以设置应用程序行为默认值，然后让应用程序[读取文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)，解析它，并根据其内容进行操作。

MySQL 配置文件就是一个例子。

下面的脚本将读取/etc/mysql/debian.cnf 配置文件，以获取 mysql 的登录详细信息，连接到 MySQL 并向其请求所有数据库的列表，遍历该列表，对每个数据库调用 mysqldump。

这个脚本基于我在[http://code poets . co . uk/2010/python-script-to-backup-MySQL-databases-on-debian/](http://codepoets.co.uk/2010/python-script-to-backup-mysql-databases-on-debian/ "mysqlbackup")上找到的一段代码

#### 备份所有 MySQL 数据库，每个文件一个，末尾有时间戳。

```py
#Importing the modules
import os
import ConfigParser
import time

# On Debian, /etc/mysql/debian.cnf contains 'root' a like login and password.
config = ConfigParser.ConfigParser()
config.read("/etc/mysql/debian.cnf")
username = config.get('client', 'user')
password = config.get('client', 'password')
hostname = config.get('client', 'host')
filestamp = time.strftime('%Y-%m-%d')

# Get a list of databases with :
database_list_command="mysql -u %s -p%s -h %s --silent -N -e 'show databases'" % (username, password, hostname)
for database in os.popen(database_list_command).readlines():
    database = database.strip()
    if database == 'information_schema':
        continue
    if database == 'performance_schema':
        continue
    filename = "/backups/mysql/%s-%s.sql" % (database, filestamp)
    os.popen("mysqldump --single-transaction -u %s -p%s -h %s -d %s | gzip -c > %s.gz" % (username, password, hostname, database, filename)) 
```

##### 有关 ConfigParser 的更多信息，请参见以下链接:

[http://www.doughellmann.com/PyMOTW/ConfigParser/](http://www.doughellmann.com/PyMOTW/ConfigParser/ "DH")
http://docs.python.org/2/library/configparser.html