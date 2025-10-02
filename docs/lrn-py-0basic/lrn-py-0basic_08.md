# 八、保存数据

## 将数据存入文件

在《文件(1)》中，已经学习了如何读写文件。

如果在程序中，有数据要保存到磁盘中，放到某个文件中是一种不错的方法。但是，如果像以前那样存，未免有点凌乱，并且没有什么良好的存储格式，导致数据以后被读出来的时候遇到麻烦，特别是不能让另外的使用者很好地理解。不要忘记了，编程是一个合作的活。还有，存储的数据不一定都是类似字符串、整数那种基础类型的。

总而言之，需要将要存储的对象格式化（或者叫做序列化），才好存好取。这就有点类似集装箱的作用。

所以，要用到本讲中提供的方式。

### pickle

pickle 是标准库中的一个模块，还有跟它完全一样的叫做 cpickle，两者的区别就是后者更快。所以，下面操作中，不管是用 `import pickle`，还是用 `import cpickle as pickle`，在功能上都是一样的。

```py
>>> import pickle
>>> integers = [1, 2, 3, 4, 5]
>>> f = open("22901.dat", "wb")
>>> pickle.dump(integers, f)
>>> f.close() 
```

用 `pickle.dump(integers, f)` 将数据 integers 保存到了文件 22901.dat 中。如果你要打开这个文件，看里面的内容，可能有点失望，但是，它对计算机是友好的。这个步骤，可以称之为将对象序列化。用到的方法是：

`pickle.dump(obj,file[,protocol])`

*   obj：序列化对象，上面的例子中是一个列表，它是基本类型，也可以序列化自己定义的类型。
*   file：一般情况下是要写入的文件。更广泛地可以理解为为拥有 write() 方法的对象，并且能接受字符串为为参数，所以，它还可以是一个 StringIO 对象，或者其它自定义满足条件的对象。
*   protocol：可选项。默认为 False（或者说 0），是以 ASCII 格式保存对象；如果设置为 1 或者 True，则以压缩的二进制格式保存对象。

下面换一种数据格式，并且做对比：

```py
>>> import pickle
>>> d = {}
>>> integers = range(9999)
>>> d["i"] = integers        #下面将这个 dict 格式的对象存入文件

>>> f = open("22902.dat", "wb")
>>> pickle.dump(d, f)           #文件中以 ascii 格式保存数据
>>> f.close()

>>> f = open("22903.dat", "wb")
>>> pickle.dump(d, f, True)     #文件中以二进制格式保存数据
>>> f.close()

>>> import os
>>> s1 = os.stat("22902.dat").st_size    #得到两个文件的大小
>>> s2 = os.stat("22903.dat").st_size

>>> print "%d, %d, %.2f%%" % (s1, s2, (s2+0.0)/s1*100)
68903, 29774, 43.21% 
```

比较结果发现，以二进制方式保存的文件比以 ascii 格式保存的文件小很多，前者约是后者的 43%。

所以，在序列化的时候，特别是面对较大对象时，建议将 dump() 的参数 True 设置上，虽然现在存储设备的价格便宜，但是能省还是省点比较好。

存入文件，仅是一个目标，还有另外一个目标，就是要读出来，也称之为反序列化。

```py
>>> integers = pickle.load(open("22901.dat", "rb"))
>>> print integers
[1, 2, 3, 4, 5] 
```

就是前面存入的那个列表。再看看被以二进制存入的那个文件：

```py
>>> f = open("22903.dat", "rb")
>>> d = pickle.load(f)
>>> print d
{'i': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ....   #省略后面的数字}
>>> f.close() 
```

还是有自己定义数据类型的需要，这种类型是否可以用上述方式存入文件并读出来呢？看下面的例子：

```py
>>> import cPickle as pickle        #cPickle 更快
>>> import StringIO                 #标准库中的一个模块，跟 file 功能类似，只不过是在内存中操作“文件”

>>> class Book(object):             #自定义一种类型
...     def __init__(self,name):
...         self.name = name
...     def my_book(self):
...         print "my book is: ", self.name
... 

>>> pybook = Book("<from beginner to master>")
>>> pybook.my_book()
my book is:  <from beginner to master>

>>> file = StringIO.StringIO()
>>> pickle.dump(pybook, file, 1)
>>> print file.getvalue()           #查看“文件”内容，注意下面不是乱码
ccopy_reg
_reconstructor
q(c__main__
Book
qc__builtin__
object
qNtRq}qUnameqU<from beginner to master>sb.

>>> pickle.dump(pybook, file)       #换一种方式，再看内容，可以比较一下
>>> print file.getvalue()           #视觉上，两者就有很大差异
ccopy_reg
_reconstructor
q(c__main__
Book
qc__builtin__
object
qNtRq}qUnameqU<from beginner to master>sb.ccopy_reg
_reconstructor
p1
(c__main__
Book
p2
c__builtin__
object
p3
NtRp4
(dp5
S'name'
p6
S'<from beginner to master>'
p7
sb. 
```

如果要从文件中读出来：

```py
>>> file.seek(0)       #找到对应类型  
>>> pybook2 = pickle.load(file)
>>> pybook2.my_book()
my book is:  <from beginner to master>
>>> file.close() 
```

### shelve

pickle 模块已经表现出它足够好的一面了。不过，由于数据的复杂性，pickle 只能完成一部分工作，在另外更复杂的情况下，它就稍显麻烦了。于是，又有了 shelve。

shelve 模块也是标准库中的。先看一下基本操作：写入和读取

```py
>>> import shelve
>>> s = shelve.open("22901.db")
>>> s["name"] = "www.itdiffer.com"
>>> s["lang"] = "python"
>>> s["pages"] = 1000
>>> s["contents"] = {"first":"base knowledge","second":"day day up"}
>>> s.close() 
```

以上完成了数据写入的过程。其实，这更接近数据库的样式了。下面是读取。

```py
>>> s = shelve.open("22901.db")
>>> name = s["name"]
>>> print name
www.itdiffer.com
>>> contents = s["contents"]
>>> print contents
{'second': 'day day up', 'first': 'base knowledge'} 
```

当然，也可以用 for 语句来读：

```py
>>> for k in s:
...     print k, s[k]
... 
contents {'second': 'day day up', 'first': 'base knowledge'}
lang python
pages 1000
name www.itdiffer.com 
```

不管是写，还是读，都似乎要简化了。所建立的对象 s，就如同字典一样，可称之为类字典对象。所以，可以如同操作字典那样来操作它。

但是，要小心坑：

```py
>>> f = shelve.open("22901.db")
>>> f["author"]
['qiwsir']
>>> f["author"].append("Hetz")    #试图增加一个
>>> f["author"]                   #坑就在这里
['qiwsir']
>>> f.close() 
```

当试图修改一个已有键的值时，没有报错，但是并没有修改成功。要填平这个坑，需要这样做：

```py
>>> f = shelve.open("22901.db", writeback=True)    #多一个参数 True
>>> f["author"].append("Hetz")
>>> f["author"]                   #没有坑了
['qiwsir', 'Hetz']
>>> f.close() 
```

还用 for 循环一下：

```py
>>> f = shelve.open("22901.db")
>>> for k,v in f.items():
...     print k,": ",v
... 
contents :  {'second': 'day day up', 'first': 'base knowledge'}
lang :  python
pages :  1000
author :  ['qiwsir', 'Hetz']
name :  www.itdiffer.com 
```

shelve 更像数据库了。

不过，它还不是真正的数据库。真正的数据库在后面。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## mysql 数据库 (1)

尽管用文件形式将数据保存到磁盘，已经是一种不错的方式。但是，人们还是发明了更具有格式化特点，并且写入和读取更快速便捷的东西——数据库（如果阅读港台的资料，它们称之为“资料库”）。维基百科对数据库有比较详细的说明：

> 数据库指的是以一定方式储存在一起、能为多个用户共享、具有尽可能小的冗余度、与应用程序彼此独立的数据集合。

到目前为止，地球上有三种类型的数据：

*   关系型数据库：MySQL、Microsoft Access、SQL Server、Oracle、...
*   非关系型数据库：MongoDB、BigTable(Google)、...
*   键值数据库：Apache Cassandra(Facebook)、LevelDB(Google) ...

在本教程中，我们主要介绍常用的开源的数据库，其中 MySQL 是典型代表。

### 概况

MySQL 是一个使用非常广泛的数据库，很多网站都是用它。关于这个数据库有很多传说。例如[维基百科上这么说：](http://zh.wikipedia.org/wiki/MySQL)

> MySQL（官方发音为英语发音：/maɪ ˌɛskjuːˈɛl/ "My S-Q-L",[1]，但也经常读作英语发音：/maɪ ˈsiːkwəl/ "My Sequel"）原本是一个开放源代码的关系数据库管理系统，原开发者为瑞典的 MySQL AB 公司，该公司于 2008 年被升阳微系统（Sun Microsystems）收购。2009 年，甲骨文公司（Oracle）收购升阳微系统公司，MySQL 成为 Oracle 旗下产品。
> 
> MySQL 在过去由于性能高、成本低、可靠性好，已经成为最流行的开源数据库，因此被广泛地应用在 Internet 上的中小型网站中。随着 MySQL 的不断成熟，它也逐渐用于更多大规模网站和应用，比如维基百科、Google 和 Facebook 等网站。非常流行的开源软件组合 LAMP 中的“M”指的就是 MySQL。
> 
> 但被甲骨文公司收购后，Oracle 大幅调涨 MySQL 商业版的售价，且甲骨文公司不再支持另一个自由软件项目 OpenSolaris 的发展，因此导致自由软件社区们对于 Oracle 是否还会持续支持 MySQL 社区版（MySQL 之中唯一的免费版本）有所隐忧，因此原先一些使用 MySQL 的开源软件逐渐转向其它的数据库。例如维基百科已于 2013 年正式宣布将从 MySQL 迁移到 MariaDB 数据库。

不管怎么着，MySQL 依然是一个不错的数据库选择，足够支持读者完成一个相当不小的网站。

### 安装

你的电脑或许不会天生就有 MySQL（是不是有的操作系统，在安装的时候就内置了呢？的确有，所以特别推荐 Linux 的某发行版），它本质上也是一个程序，若有必要，须安装。

我用 ubuntu 操作系统演示，因为我相信读者将来在真正的工程项目中，多数情况下是要操作 Linux 系统的服务器，并且，我酷爱用 ubuntu。还有，本教程的目标是 from beginner to master，不管是不是真的 master，总要装得像，Linux 能够给你撑门面。

第一步，在 shell 端运行如下命令：

```py
sudo apt-get install mysql-server 
```

运行完毕，就安装好了这个数据库。是不是很简单呢？当然，当然，还要进行配置。

第二步，配置 MySQL

安装之后，运行：

```py
service mysqld start 
```

启动 mysql 数据库。然后进行下面的操作，对其进行配置。

默认的 MySQL 安装之后根用户是没有密码的，注意，这里有一个名词“根用户”，其用户名是：root。运行：

```py
$mysql -u root 
```

在这里之所以用 -u root 是因为我现在是一般用户（firehare），如果不加 -u root 的话，mysql 会以为是 firehare 在登录。

进入 mysql 之后，会看到>符号开头，这就是 mysql 的命令操作界面了。

下面设置 Mysql 中的 root 用户密码了，否则，Mysql 服务无安全可言了。

```py
mysql> GRANT ALL PRIVILEGES ON *.* TO root@localhost IDENTIFIED BY "123456"; 
```

用 123456 做为 root 用户的密码，应该是非常愚蠢的，如果在真正的项目中，最好别这样做，要用大小写字母与数字混合的密码，且不少于 8 位。

以后如果在登录数据库，就可以用刚才设置的密码了。

### 运行

安装之后，就要运行它，并操作这个数据库。

```py
$ mysql -u root -p
Enter password: 
```

输入数据库的密码，之后出现：

```py
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 373
Server version: 5.5.38-0ubuntu0.14.04.1 (Ubuntu)

Copyright (c) 2000, 2014, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> 
```

看到这个界面内容，就说明你已经进入到数据里面了。接下来就可以对这个数据进行操作。例如：

```py
mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| carstore           |
| cutvideo           |
| itdiffer           |
| mysql              |
| performance_schema |
| test               |
+--------------------+ 
```

用这个命令，就列出了当前已经有的数据库。

对数据库的操作，除了用命令之外，还可以使用一些可视化工具。比如 phpmyadmin 就是不错的。

更多数据库操作的知识，这里就不介绍了，读者可以参考有关书籍。

MySQL 数据库已经安装好，但是 Python 还不能操作它，还要继续安装 Python 操作数据库的模块——Python-MySQLdb

### 安装 Python-MySQLdb

Python-MySQLdb 是一个接口程序，Python 通过它对 mysql 数据实现各种操作。

在编程中，会遇到很多类似的接口程序，通过接口程序对另外一个对象进行操作。接口程序就好比钥匙，如果要开锁，人直接用手指去捅，肯定是不行的，那么必须借助工具，插入到锁孔中，把锁打开，之后，门开了，就可以操作门里面的东西了。那么打开锁的工具就是接口程序。谁都知道，用对应的钥匙开锁是最好的，如果用别的工具（比如锤子），或许不便利（其实还分人，也就是人开锁的水平，如果是江洋大盗或者小毛贼什么的，擅长开锁，用别的工具也便利了），也就是接口程序不同，编码水平不同，都是考虑因素。

啰嗦这么多，一言蔽之，Python-MySQLdb 就是打开 MySQL 数据库的钥匙。

如果要源码安装，可以这里下载 Python-mysqldb:[`pypi.Python.org/pypi/MySQL-Python/`](https://pypi.Python.org/pypi/MySQL-Python/)

下载之后就可以安装了。

ubuntu 下可以这么做：

```py
sudo apt-get install build-essential Python-dev libmysqlclient-dev
sudo apt-get install Python-MySQLdb 
```

也可以用 pip 来安装：

```py
pip install mysql-Python 
```

安装之后，在 python 交互模式下：

```py
>>> import MySQLdb 
```

如果不报错，恭喜你，已经安装好了。如果报错，恭喜你，可以借着错误信息提高自己的计算机水平了，请求助于 google 大神。

### 连接数据库

要先找到老婆，才能谈如何养育自己的孩子，同理连接数据库之先要建立数据库。

```py
$ mysql -u root -p
Enter password: 
```

进入到数据库操作界面：

```py
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 373
Server version: 5.5.38-0ubuntu0.14.04.1 (Ubuntu)

Copyright (c) 2000, 2014, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> 
```

输入如下命令，建立一个数据库：

```py
mysql> create database qiwsirtest character set utf8;
Query OK, 1 row affected (0.00 sec) 
```

注意上面的指令，如果仅仅输入：create database qiwsirtest，也可以，但是，我在后面增加了 character set utf8，意思是所建立的数据库 qiwsirtest，编码是 utf-8 的，这样存入汉字就不是乱码了。

看到那一行提示：Query OK, 1 row affected (0.00 sec)，就说明这个数据库已经建立好了，名字叫做:qiwsirtest

数据库建立之后，就可以用 Python 通过已经安装的 mysqldb 来连接这个名字叫做 qiwsirtest 的库了。

```py
>>> import MySQLdb
>>> conn = MySQLdb.connect(host="localhost",user="root",passwd="123123",db="qiwsirtest",port=3306,charset="utf8") 
```

逐个解释上述命令的含义：

*   host:等号的后面应该填写 mysql 数据库的地址，因为就数据库就在本机上（也称作本地），所以使用 localhost，注意引号。如果在其它的服务器上，这里应该填写 ip 地址。一般中小型的网站，数据库和程序都是在同一台服务器（计算机）上，就使用 localhost 了。
*   user:登录数据库的用户名，这里一般填写"root",还是要注意引号。当然，如果读者命名了别的用户名，数据库管理者提供了专有用户名，就更改为相应用户。但是，不同用户的权限可能不同，所以，在程序中，如果要操作数据库，还要注意所拥有的权限。在这里用 root，就放心了，什么权限都有啦。不过，这样做，在大型系统中是应该避免的。
*   passwd:上述 user 账户对应的登录 mysql 的密码。我在上面的例子中用的密码是"123123"。不要忘记引号。
*   db:就是刚刚通 create 命令建立的数据库，我建立的数据库名字是"qiwsirtest"，还是要注意引号。看官如果建立的数据库名字不是这个，就写自己所建数据库名字。
*   port:一般情况，mysql 的默认端口是 3306，当 mysql 被安装到服务器之后，为了能够允许网络访问，服务器（计算机）要提供一个访问端口给它。
*   charset:这个设置，在很多教程中都不写，结果在真正进行数据存储的时候，发现有乱码。这里我将 qiwsirtest 这个数据库的编码设置为 utf-8 格式，这样就允许存入汉字而无乱码了。注意，在 mysql 设置中，utf-8 写成 utf8,没有中间的横线。但是在 Python 文件开头和其它地方设置编码格式的时候，要写成 utf-8。切记！

注：connect 中的 host、user、passwd 等可以不写，只有在写的时候按照 host、user、passwd、db (可以不写)、port 顺序写就可以，端口号 port=3306 还是不要省略的为好，如果没有 db 在 port 前面，直接写 3306 会报错.

其实，关于 connect 的参数还不少，下面摘抄来自[mysqldb 官方文档的内容](http://mysql-python.sourceforge.net/MySQLdb.html)，把所有的参数都列出来，还有相关说明，请看官认真阅读。不过，上面几个是常用的，其它的看情况使用。

connect(parameters...)

> Constructor for creating a connection to the database. Returns a Connection Object. Parameters are the same as for the MySQL C API. In addition, there are a few additional keywords that correspond to what you would pass mysql_options() before connecting. Note that some parameters must be specified as keyword arguments! The default value for each parameter is NULL or zero, as appropriate. Consult the MySQL documentation for more details. The important parameters are:

*   host: name of host to connect to. Default: use the local host via a UNIX socket (where applicable)
*   user: user to authenticate as. Default: current effective user.
*   passwd: password to authenticate with. Default: no password.
*   db: database to use. Default: no default database.
*   port: TCP port of MySQL server. Default: standard port (3306).
*   unix_socket: location of UNIX socket. Default: use default location or TCP for remote hosts.
*   conv: type conversion dictionary. Default: a copy of MySQLdb.converters.conversions
*   compress: Enable protocol compression. Default: no compression.
*   connect_timeout: Abort if connect is not completed within given number of seconds. Default: no timeout (?)
*   named_pipe: Use a named pipe (Windows). Default: don't.
*   init_command: Initial command to issue to server upon connection. Default: Nothing.
*   read_default_file: MySQL configuration file to read; see the MySQL documentation for mysql_options().
*   read_default_group: Default group to read; see the MySQL documentation for mysql_options().
*   cursorclass: cursor class that cursor() uses, unless overridden. Default: MySQLdb.cursors.Cursor. This must be a keyword parameter.
*   use_unicode: If True, CHAR and VARCHAR and TEXT columns are returned as Unicode strings, using the configured character set. It is best to set the default encoding in the server configuration, or client configuration (read with read_default_file). If you change the character set after connecting (MySQL-4.1 and later), you'll need to put the correct character set name in connection.charset.

If False, text-like columns are returned as normal strings, but you can always write Unicode strings.

This must be a keyword parameter.

*   charset: If present, the connection character set will be changed to this character set, if they are not equal. Support for changing the character set requires MySQL-4.1 and later server; if the server is too old, UnsupportedError will be raised. This option implies use_unicode=True, but you can override this with use_unicode=False, though you probably shouldn't.

If not present, the default character set is used.

This must be a keyword parameter.

*   sql_mode: If present, the session SQL mode will be set to the given string. For more information on sql_mode, see the MySQL documentation. Only available for 4.1 and newer servers.

If not present, the session SQL mode will be unchanged.

This must be a keyword parameter.

*   ssl: This parameter takes a dictionary or mapping, where the keys are parameter names used by the mysql_ssl_set MySQL C API call. If this is set, it initiates an SSL connection to the server; if there is no SSL support in the client, an exception is raised. This must be a keyword parameter.

已经完成了数据库的连接。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## MySQL 数据库 (2)

就数据库而言，连接之后就要对其操作。但是，目前那个名字叫做 qiwsirtest 的数据仅仅是空架子，没有什么可操作的，要操作它，就必须在里面建立“表”，什么是数据库的表呢？下面摘抄自维基百科对数据库表的简要解释，要想详细了解，需要看官在找一些有关数据库的教程和书籍来看看。

> 在关系数据库中，数据库表是一系列二维数组的集合，用来代表和储存数据对象之间的关系。它由纵向的列和横向的行组成，例如一个有关作者信息的名为 authors 的表中，每个列包含的是所有作者的某个特定类型的信息，比如“姓氏”，而每行则包含了某个特定作者的所有信息：姓、名、住址等等。
> 
> 对于特定的数据库表，列的数目一般事先固定，各列之间可以由列名来识别。而行的数目可以随时、动态变化，每行通常都可以根据某个（或某几个）列中的数据来识别，称为候选键。

我打算在 qiwsirtest 中建立一个存储用户名、用户密码、用户邮箱的表，其结构用二维表格表现如下：

| username | password | email |
| --- | --- | --- |
| qiwsir | 123123 | qiwsir@gmail.com |

特别说明，这里为了简化细节，突出重点，对密码不加密，直接明文保存，虽然这种方式是很不安全的。但是，有不少网站还都这么做的，这么做的目的是比较可恶的。就让我在这里，仅仅在这里可恶一次。

### 数据库表

因为直接操作数据部分，不是本教重点，但是关联到后面的操作，为了让读者在阅读上连贯，也快速地说明建立数据库表并输入内容。

```py
mysql> use qiwsirtest;
Database changed
mysql> show tables;
Empty set (0.00 sec) 
```

用 `show tables` 命令显示这个数据库中是否有数据表了。查询结果显示为空。

下面就用如下命令建立一个数据表，这个数据表的内容就是上面所说明的。

```py
mysql> create table users(id int(2) not null primary key auto_increment,username varchar(40),password text,email text)default charset=utf8;
Query OK, 0 rows affected (0.12 sec) 
```

建立的这个数据表名称是：users，其中包含上述字段，可以用下面的方式看一看这个数据表的结构。

```py
mysql> show tables;
+----------------------+
| Tables_in_qiwsirtest |
+----------------------+
| users                |
+----------------------+
1 row in set (0.00 sec) 
```

查询显示，在 qiwsirtest 这个数据库中，已经有一个表，它的名字是：users。

```py
mysql> desc users;
+----------+-------------+------+-----+---------+----------------+
| Field    | Type        | Null | Key | Default | Extra          |
+----------+-------------+------+-----+---------+----------------+
| id       | int(2)      | NO   | PRI | NULL    | auto_increment |
| username | varchar(40) | YES  |     | NULL    |                |
| password | text        | YES  |     | NULL    |                |
| email    | text        | YES  |     | NULL    |                |
+----------+-------------+------+-----+---------+----------------+
4 rows in set (0.00 sec) 
```

显示表 users 的结构：

特别提醒：上述所有字段设置仅为演示，在实际开发中，要根据具体情况来确定字段的属性。

如此就得到了一个空表。可以查询看看：

```py
mysql> select * from users;
Empty set (0.01 sec) 
```

向里面插入点信息，就只插入一条吧。

```py
mysql> insert into users(username,password,email) values("qiwsir","123123","qiwsir@gmail.com");
Query OK, 1 row affected (0.05 sec)

mysql> select * from users;
+----+----------+----------+------------------+
| id | username | password | email            |
+----+----------+----------+------------------+
|  1 | qiwsir   | 123123   | qiwsir@gmail.com |
+----+----------+----------+------------------+
1 row in set (0.00 sec) 
```

这样就得到了一个有内容的数据库表。

### Python 操作数据库

连接数据库，必须的。

```py
>>> import MySQLdb
>>> conn = MySQLdb.connect(host="localhost",user="root",passwd="123123",db="qiwsirtest",charset="utf8") 
```

Python 建立了与数据的连接，其实是建立了一个 `MySQLdb.connect()` 的实例对象，或者泛泛地称之为连接对象，Python 就是通过连接对象和数据库对话。这个对象常用的方法有：

*   commit()：如果数据库表进行了修改，提交保存当前的数据。当然，如果此用户没有权限就作罢了，什么也不会发生。
*   rollback()：如果有权限，就取消当前的操作，否则报错。
*   cursor([cursorclass])：返回连接的游标对象。通过游标执行 SQL 查询并检查结果。游标比连接支持更多的方法，而且可能在程序中更好用。
*   close()：关闭连接。此后，连接对象和游标都不再可用了。

Python 和数据之间的连接建立起来之后，要操作数据库，就需要让 Python 对数据库执行 SQL 语句。Python 是通过游标执行 SQL 语句的。所以，连接建立之后，就要利用连接对象得到游标对象，方法如下：

```py
>>> cur = conn.cursor() 
```

此后，就可以利用游标对象的方法对数据库进行操作。那么还得了解游标对象的常用方法：

| 名称 | 描述 |
| --- | --- |
| close() | 关闭游标。之后游标不可用 |
| execute(query[,args]) | 执行一条 SQL 语句，可以带参数 |
| executemany(query, pseq) | 对序列 pseq 中的每个参数执行 sql 语句 |
| fetchone() | 返回一条查询结果 |
| fetchall() | 返回所有查询结果 |
| fetchmany([size]) | 返回 size 条结果 |
| nextset() | 移动到下一个结果 |
| scroll(value,mode='relative') | 移动游标到指定行，如果 mode='relative',则表示从当前所在行移动 value 条,如果 mode='absolute',则表示从结果集的第一行移动 value 条. |

#### 插入

例如，要在数据表 users 中插入一条记录，使得:username="Python",password="123456",email="Python@gmail.com"，这样做：

```py
>>> cur.execute("insert into users (username,password,email) values (%s,%s,%s)",("Python","123456","Python@gmail.com"))
1L 
```

没有报错，并且返回一个"1L"结果，说明有一 n 行记录操作成功。不妨用"mysql>"交互方式查看一下：

```py
mysql> select * from users;
+----+----------+----------+------------------+
| id | username | password | email            |
+----+----------+----------+------------------+
|  1 | qiwsir   | 123123   | qiwsir@gmail.com |
+----+----------+----------+------------------+
1 row in set (0.00 sec) 
```

咦，奇怪呀。怎么没有看到增加的那一条呢？哪里错了？可是上面也没有报错呀。

特别注意，通过"cur.execute()"对数据库进行操作之后，没有报错，完全正确，但是不等于数据就已经提交到数据库中了，还必须要用到"MySQLdb.connect"的一个属性：commit()，将数据提交上去，也就是进行了"cur.execute()"操作，要将数据提交，必须执行：

```py
>>> conn.commit() 
```

再到"mysql>"中运行"select * from users"试一试：

```py
mysql> select * from users;
+----+----------+----------+------------------+
| id | username | password | email            |
+----+----------+----------+------------------+
|  1 | qiwsir   | 123123   | qiwsir@gmail.com |
|  2 | python   | 123456   | python@gmail.com |
+----+----------+----------+------------------+
2 rows in set (0.00 sec) 
```

果然如此。这就如同编写一个文本一样，将文字写到文本上，并不等于文字已经保留在文本文件中了，必须执行"CTRL-S"才能保存。也就是在通过 Python 操作数据库的时候，以"execute()"执行各种 sql 语句之后，要让已经执行的效果保存，必须运行连接对象的"commit()"方法。

再尝试一下插入多条的那个命令"executemany(query,args)".

```py
>>> cur.executemany("insert into users (username,password,email) values (%s,%s,%s)",(("google","111222","g@gmail.com"),("facebook","222333","f@face.book"),("github","333444","git@hub.com"),("docker","444555","doc@ker.com")))
4L
>>> conn.commit() 
```

到"mysql>"里面看结果：

```py
mysql> select * from users;
+----+----------+----------+------------------+
| id | username | password | email            |
+----+----------+----------+------------------+
|  1 | qiwsir   | 123123   | qiwsir@gmail.com |
|  2 | python   | 123456   | python@gmail.com |
|  3 | google   | 111222   | g@gmail.com      |
|  4 | facebook | 222333   | f@face.book      |
|  5 | github   | 333444   | git@hub.com      |
|  6 | docker   | 444555   | doc@ker.com      |
+----+----------+----------+------------------+
6 rows in set (0.00 sec) 
```

成功插入了多条记录。在"executemany(query, pseq)"中，query 还是一条 sql 语句，但是 pseq 这时候是一个 tuple，这个 tuple 里面的元素也是 tuple，每个 tuple 分别对应 sql 语句中的字段列表。这句话其实被执行多次。只不过执行过程不显示给我们看罢了。

除了插入命令，其它对数据操作的命了都可用类似上面的方式，比如删除、修改等。

#### 查询

如果要从数据库中查询数据，也用游标方法来操作了。

```py
>>> cur.execute("select * from users")    
7L 
```

这说明从 users 表汇总查询出来了 7 条记录。但是，这似乎有点不友好，告诉我 7 条记录查出来了，但是在哪里呢，如果在'mysql>'下操作查询命令，一下就把 7 条记录列出来了。怎么显示 Python 在这里的查询结果呢？

要用到游标对象的 fetchall()、fetchmany(size=None)、fetchone()、scroll(value, mode='relative')等方法。

```py
>>> cur.execute("select * from users")    
7L
>>> lines = cur.fetchall() 
```

到这里，已经将查询到的记录赋值给变量 lines 了。如果要把它们显示出来，就要用到曾经学习过的循环语句了。

```py
>>> for line in lines:
...     print line
... 
(1L, u'qiwsir', u'123123', u'qiwsir@gmail.com')
(2L, u'python', u'123456', u'python@gmail.com')
(3L, u'google', u'111222', u'g@gmail.com')
(4L, u'facebook', u'222333', u'f@face.book')
(5L, u'github', u'333444', u'git@hub.com')
(6L, u'docker', u'444555', u'doc@ker.com')
(7L, u'\u8001\u9f50', u'9988', u'qiwsir@gmail.com') 
```

很好。果然是逐条显示出来了。列位注意，第七条中的 u'\u8001\u95f5',这里是汉字，只不过由于我的 shell 不能显示罢了，不必惊慌，不必搭理它。

只想查出第一条，可以吗？当然可以！看下面的：

```py
>>> cur.execute("select * from users where id=1")
1L
>>> line_first = cur.fetchone()     #只返回一条
>>> print line_first
(1L, u'qiwsir', u'123123', u'qiwsir@gmail.com') 
```

为了对上述过程了解深入，做下面实验：

```py
>>> cur.execute("select * from users")
7L
>>> print cur.fetchall()
((1L, u'qiwsir', u'123123', u'qiwsir@gmail.com'), (2L, u'python', u'123456', u'python@gmail.com'), (3L, u'google', u'111222', u'g@gmail.com'), (4L, u'facebook', u'222333', u'f@face.book'), (5L, u'github', u'333444', u'git@hub.com'), (6L, u'docker', u'444555', u'doc@ker.com'), (7L, u'\u8001\u9f50', u'9988', u'qiwsir@gmail.com')) 
```

原来，用 cur.execute() 从数据库查询出来的东西，被“保存在了 cur 所能找到的某个地方”，要找出这些被保存的东西，需要用 cur.fetchall()（或者 fechone 等），并且找出来之后，做为对象存在。从上面的实验探讨发现，被保存的对象是一个 tuple 中，里面的每个元素，都是一个一个的 tuple。因此，用 for 循环就可以一个一个拿出来了。

接着看，还有神奇的呢。

接着上面的操作，再打印一遍

```py
>>> print cur.fetchall()
() 
```

晕了！怎么什么是空？不是说做为对象已经存在了内存中了吗？难道这个内存中的对象是一次有效吗？

不要着急。

通过游标找出来的对象，在读取的时候有一个特点，就是那个游标会移动。在第一次操作了 print cur.fetchall() 后，因为是将所有的都打印出来，游标就从第一条移动到最后一条。当 print 结束之后，游标已经在最后一条的后面了。接下来如果再次打印，就空了，最后一条后面没有东西了。

下面还要实验，检验上面所说：

```py
>>> cur.execute('select * from users')
7L
>>> print cur.fetchone() 
(1L, u'qiwsir', u'123123', u'qiwsir@gmail.com')
>>> print cur.fetchone()
(2L, u'python', u'123456', u'python@gmail.com')
>>> print cur.fetchone()
(3L, u'google', u'111222', u'g@gmail.com') 
```

这次我不一次全部打印出来了，而是一次打印一条，看官可以从结果中看出来，果然那个游标在一条一条向下移动呢。注意，我在这次实验中，是重新运行了查询语句。

那么，既然在操作存储在内存中的对象时候，游标会移动，能不能让游标向上移动，或者移动到指定位置呢？这就是那个 scroll()

```py
>>> cur.scroll(1)
>>> print cur.fetchone()
(5L, u'github', u'333444', u'git@hub.com')
>>> cur.scroll(-2)
>>> print cur.fetchone()
(4L, u'facebook', u'222333', u'f@face.book') 
```

果然，这个函数能够移动游标，不过请仔细观察，上面的方式是让游标相对与当前位置向上或者向下移动。即：

cur.scroll(n)，或者，cur.scroll(n,"relative")：意思是相对当前位置向上或者向下移动，n 为正数，表示向下（向前），n 为负数，表示向上（向后）

还有一种方式，可以实现“绝对”移动，不是“相对”移动：增加一个参数"absolute"

特别提醒看官注意的是，在 Python 中，序列对象是的顺序是从 0 开始的。

```py
>>> cur.scroll(2,"absolute")    #回到序号是 2,但指向第三条
>>> print cur.fetchone()        #打印，果然是
(3L, u'google', u'111222', u'g@gmail.com')

>>> cur.scroll(1,"absolute")
>>> print cur.fetchone()
(2L, u'python', u'123456', u'python@gmail.com')

>>> cur.scroll(0,"absolute")    #回到序号是 0,即指向 tuple 的第一条
>>> print cur.fetchone()
(1L, u'qiwsir', u'123123', u'qiwsir@gmail.com') 
```

至此，已经熟悉了 cur.fetchall() 和 cur.fetchone() 以及 cur.scroll() 几个方法，还有另外一个，接这上边的操作，也就是游标在序号是 1 的位置，指向了 tuple 的第二条

```py
>>> cur.fetchmany(3)
((2L, u'Python', u'123456', u'python@gmail.com'), (3L, u'google', u'111222', u'g@gmail.com'), (4L, u'facebook', u'222333', u'f@face.book')) 
```

上面这个操作，就是实现了从当前位置（游标指向 tuple 的序号为 1 的位置，即第二条记录）开始，含当前位置，向下列出 3 条记录。

读取数据，好像有点啰嗦呀。细细琢磨，还是有道理的。你觉得呢？

不过，Python 总是能够为我们着想的，在连接对象的游标方法中提供了一个参数，可以实现将读取到的数据变成字典形式，这样就提供了另外一种读取方式了。

```py
>>> cur = conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)
>>> cur.execute("select * from users")
7L
>>> cur.fetchall()
({'username': u'qiwsir', 'password': u'123123', 'id': 1L, 'email': u'qiwsir@gmail.com'}, {'username': u'mypython', 'password': u'123456', 'id': 2L, 'email': u'python@gmail.com'}, {'username': u'google', 'password': u'111222', 'id': 3L, 'email': u'g@gmail.com'}, {'username': u'facebook', 'password': u'222333', 'id': 4L, 'email': u'f@face.book'}, {'username': u'github', 'password': u'333444', 'id': 5L, 'email': u'git@hub.com'}, {'username': u'docker', 'password': u'444555', 'id': 6L, 'email': u'doc@ker.com'}, {'username': u'\u8001\u9f50', 'password': u'9988', 'id': 7L, 'email': u'qiwsir@gmail.com'}) 
```

这样，在元组里面的元素就是一个一个字典：

```py
>>> cur.scroll(0,"absolute")
>>> for line in cur.fetchall():
...     print line["username"]
... 
qiwsir
mypython
google
facebook
github
docker
老齐 
```

根据字典对象的特点来读取了“键-值”。

#### 更新数据

经过前面的操作，这个就比较简单了，不过需要提醒的是，如果更新完毕，和插入数据一样，都需要 commit() 来提交保存。

```py
>>> cur.execute("update users set username=%s where id=2",("mypython"))
1L
>>> cur.execute("select * from users where id=2")
1L
>>> cur.fetchone()
(2L, u'mypython', u'123456', u'python@gmail.com') 
```

从操作中看出来了，已经将数据库中第二条的用户名修改为 myPython 了，用的就是 update 语句。

不过，要真的实现在数据库中更新，还要运行：

```py
>>> conn.commit() 
```

这就大事完吉了。

应该还有个小尾巴，那就是当你操作数据完毕，不要忘记关门：

```py
>>> cur.close()
>>> conn.close() 
```

门锁好了，放心离开。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## mongodb 数据库 (1)

MongoDB 开始火了，这是时代发展的需要。为此，本教程也要涉及到如何用 Python 来操作 mongodb。考虑到读者对这种数据库可能比 mysql 之类的更陌生，所以，要用多一点的篇幅稍作介绍，当然，更完备的内容还是要去阅读专业的 mongodb 书籍。

mongodb 是属于 NoSql 的。

NoSql，全称是 Not Only Sql,指的是非关系型的数据库。它是为了大规模 web 应用而生的，其特征诸如模式自由、支持简易复制、简单的 API、大容量数据等等。

MongoDB 是其一，选择它，主要是因为我喜欢，否则我不会列入我的教程。数说它的特点，可能是：

*   面向文档存储
*   对任何属性可索引
*   复制和高可用性
*   自动分片
*   丰富的查询
*   快速就地更新

也许还能列出更多，基于它的特点，擅长领域就在于：

*   大数据（太时髦了！以下可以都不看，就要用它了。）
*   内容管理和交付
*   移动和社交基础设施
*   用户数据管理
*   数据平台

### 安装 mongodb

先演示在 ubuntu 系统中的安装过程：

```py
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10
echo 'deb http://downloads-distro.mongodb.org/repo/ubuntu-upstart dist 10gen' | sudo tee /etc/apt/sources.list.d/mongodb.list
sudo apt-get update
sudo apt-get install mongodb-10gen 
```

如此就安装完毕。上述安装流程来自：[Install MongoDB](http://docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/)

如果你用的是其它操作系统，可以到官方网站下载安装程序：[`www.mongodb.org/downloads`](http://www.mongodb.org/downloads)，能满足各种操作系统。

![](img/23201.jpg)

> 难免在安装过程中遇到问题，推荐几个资料，供参考：
> 
> [window 平台安装 MongoDB](http://www.w3cschool.cc/mongodb/mongodb-window-install.html)
> 
> [NoSQL 之【MongoDB】学习（一）：安装说明](http://www.cnblogs.com/zhoujinyi/archive/2013/06/02/3113868.html)
> 
> [MongoDB 生产环境的安装与配置(Ubuntu)](https://ruby-china.org/topics/454)
> 
> [在 Ubuntu 中安装 MongoDB](http://blog.fens.me/linux-mongodb-install/)
> 
> 在 Ubuntu 下进行 MongoDB 安装步骤

### 启动 mongodb

安装完毕，就可以启动数据库。因为本教程不是专门讲数据库，所以，这里不设计数据库的详细讲解，请读者参考有关资料。下面只是建立一个简单的库，并且说明 mongodb 的基本要点，目的在于为后面用 Python 来操作它做个铺垫。

执行 `mongo` 启动 shell，显示的也是 `>`，有点类似 mysql 的状态。在 shell 中，可以实现与数据库的交互操作。

在 shell 中，有一个全局变量 db，使用哪个数据库，那个数据库就会被复制给这个全局变量 db，如果那个数据库不存在，就会新建。

```py
> use mydb
switched to db mydb
> db
mydb 
```

除非向这个数据库中增加实质性的内容，否则它是看不到的。

```py
> show dbs;
local    0.03125GB 
```

向这个数据库增加点东西。mongodb 的基本单元是文档，所谓文档，就类似与 Python 中的字典，以键值对的方式保存数据。

```py
> book = {"title":"from beginner to master", "author":"qiwsir", "lang":"python"}
{
    "title" : "from beginner to master",
    "author" : "qiwsir",
    "lang" : "python"
}
> db.books.insert(book)
> db.books.find()
{ "_id" : ObjectId("554f0e3cf579bc0767db9edf"), "title" : "from beginner to master", "author" : "qiwsir", "lang" : "Python" } 
```

db 指向了数据库 mydb，books 是这个数据库里面的一个集合（类似 mysql 里面的表），向集合 books 里面插入了一个文档（文档对应 mysql 里面的记录）。“数据库、集合、文档”构成了 mongodb 数据库。

从上面操作，还发现一个有意思的地方，并没有类似 create 之类的命令，用到数据库，就通过 `use xxx`，如果不存在就建立；用到集合，就通过 `db.xxx` 来使用，如果没有就建立。可以总结为“随用随取随建立”。是不是简单的有点出人意料。

```py
> show dbs
local    0.03125GB
mydb    0.0625GB 
```

当有了充实内容之后，也看到刚才用到的数据库 mydb 了。

在 mongodb 的 shell 中，可以对数据进行“增删改查”等操作。但是，我们的目的是用 Python 来操作，所以，还是把力气放在后面用。

### 安装 Pymongo

要用 Python 来驱动 mongodb，必须要安装驱动模块，即 Pymongo，这跟操作 mysql 类似。安装方法，我最推荐如下：

```py
$ sudo pip install Pymongo 
```

如果顺利，就会看到最后的提示：

```py
Successfully installed Pymongo
Cleaning up... 
```

如果不选择版本，安装的应该是最新版本的，我在本教程测试的时候，安装的是：

```py
>>> import Pymongo
>>> pymongo.version
'3.0.1' 
```

这个版本在后面给我挖了一个坑。如果读者要指定版本，比如安装 2.8 版本的，可以：

```py
$ sudo pip install Pymongo==2.8 
```

如果用这个版本，我后面遇到的坑能够避免。

安装好之后，进入到 Python 的交互模式里面：

```py
>>> import Pymongo 
```

说明模块没有问题。

### 连接 mongodb

既然 Python 驱动 mongdb 的模块 Pymongo 业已安装完毕，接下来就是连接，也就是建立连接对象。

```py
>>> pymongo.Connection("localhost",27017)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'module' object has no attribute 'Connection' 
```

报错！我在去年做的项目中，就是这样做的，并且网上查看很多教程都是这么连接。

所以，读者如果用的是旧版本的 Pymongo，比如 2.8，仍然可以使用上面的连接方法，如果是像我一样，是用的新的（我安装时没有选版本），就得注意这个问题了。

经验主义害死人。必须看看下面有哪些方法可以用：

```py
>>> dir(pymongo)
['ALL', 'ASCENDING', 'CursorType', 'DESCENDING', 'DeleteMany', 'DeleteOne', 'GEO2D', 'GEOHAYSTACK', 'GEOSPHERE', 'HASHED', 'IndexModel', 'InsertOne', 'MAX_SUPPORTED_WIRE_VERSION', 'MIN_SUPPORTED_WIRE_VERSION', 'MongoClient', 'MongoReplicaSetClient', 'OFF', 'ReadPreference', 'ReplaceOne', 'ReturnDocument', 'SLOW_ONLY', 'TEXT', 'UpdateMany', 'UpdateOne', 'WriteConcern', '__builtins__', '__doc__', '__file__', '__name__', '__package__', '__path__', '_cmessage', 'auth', 'bulk', 'client_options', 'collection', 'command_cursor', 'common', 'cursor', 'cursor_manager', 'database', 'errors', 'get_version_string', 'has_c', 'helpers', 'ismaster', 'message', 'mongo_client', 'mongo_replica_set_client', 'monitor', 'monotonic', 'network', 'operations', 'periodic_executor', 'pool', 'read_preferences', 'response', 'results', 'server', 'server_description', 'server_selectors', 'server_type', 'settings', 'son_manipulator', 'ssl_context', 'ssl_support', 'thread_util', 'topology', 'topology_description', 'uri_parser', 'version', 'version_tuple', 'write_concern'] 
```

瞪大我的那双浑浊迷茫布满血丝渴望惊喜的眼睛，透过近视镜的玻璃片，怎么也找不到 Connection() 这个方法。原来，刚刚安装的 Pymongo 变了，“他变了”。

不过，我发现了它：MongoClient()

```py
>>> client = pymongo.MongoClient("localhost", 27017) 
```

很好。Python 已经和 mongodb 建立了连接。

刚才已经建立了一个数据库 mydb，并且在这个库里面有一个集合 books，于是：

```py
>>> db = client.mydb 
```

或者

```py
>>> db = client['mydb'] 
```

获得数据库 mydb，并赋值给变量 db（这个变量不是 mongodb 的 shell 中的那个 db，此处的 db 就是 Python 中一个寻常的变量）。

```py
>>> db.collection_names()
[u'system.indexes', u'books'] 
```

查看集合，发现了我们已经建立好的那个 books，于是在获取这个集合，并赋值给一个变量 books：

```py
>>> books = db["books"] 
```

或者

```py
>>> books = db.books 
```

接下来，就可以操作这个集合中的具体内容了。

#### 编辑

刚刚的 books 所引用的是一个 mongodb 的集合对象，它就跟前面学习过的其它对象一样，有一些方法供我们来驱使。

```py
>>> type(books)
<class 'pymongo.collection.Collection'>

>>> dir(books)
['_BaseObject__codec_options', '_BaseObject__read_preference', '_BaseObject__write_concern', '_Collection__create', '_Collection__create_index', '_Collection__database', '_Collection__find_and_modify', '_Collection__full_name', '_Collection__name', '__call__', '__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattr__', '__getattribute__', '__getitem__', '__hash__', '__init__', '__iter__', '__module__', '__ne__', '__new__', '__next__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_command', '_count', '_delete', '_insert', '_socket_for_primary_reads', '_socket_for_reads', '_socket_for_writes', '_update', 'aggregate', 'bulk_write', 'codec_options', 'count', 'create_index', 'create_indexes', 'database', 'delete_many', 'delete_one', 'distinct', 'drop', 'drop_index', 'drop_indexes', 'ensure_index', 'find', 'find_and_modify', 'find_one', 'find_one_and_delete', 'find_one_and_replace', 'find_one_and_update', 'full_name', 'group', 'index_information', 'initialize_ordered_bulk_op', 'initialize_unordered_bulk_op', 'inline_map_reduce', 'insert', 'insert_many', 'insert_one', 'list_indexes', 'map_reduce', 'name', 'next', 'options', 'parallel_scan', 'read_preference', 'reindex', 'remove', 'rename', 'replace_one', 'save', 'update', 'update_many', 'update_one', 'with_options', 'write_concern'] 
```

这么多方法不会一一介绍，只是按照“增删改查”的常用功能，介绍几种。读者可以使用 help() 去查看每一种方法的使用说明。

```py
>>> books.find_one()
{u'lang': u'Python', u'_id': ObjectId('554f0e3cf579bc0767db9edf'), u'author': u'qiwsir', u'title': u'from beginner to master'} 
```

提醒读者注意的是，如果你熟悉了 mongodb 的 shell 中的命令，跟 Pymongo 中的方法稍有差别，比如刚才这个，在 mongodb 的 shell 中是这样子的：

```py
> db.books.findOne()
{
    "_id" : ObjectId("554f0e3cf579bc0767db9edf"),
    "title" : "from beginner to master",
    "author" : "qiwsir",
    "lang" : "python"
} 
```

请注意区分。

目前在集合 books 中，有一个文档，还想再增加，于是插入一条：

**新增和查询**

```py
>>> b2 = {"title":"physics", "author":"Newton", "lang":"english"}
>>> books.insert(b2)
ObjectId('554f28f465db941152e6df8b') 
```

成功地向集合中增加了一个文档。得看看结果（我们就是充满好奇心的小孩子，我记得女儿小时候，每个给她照相，拍了一张，她总要看一看。现在我们似乎也是这样，如果不看看，总觉得不放心），看看就是一种查询。

```py
>>> books.find().count()
2 
```

这是查看当前集合有多少个文档的方式，返回值为 2，则说明有两条文档了。还是要看看内容。

```py
>>> books.find_one()
{u'lang': u'python', u'_id': ObjectId('554f0e3cf579bc0767db9edf'), u'author': u'qiwsir', u'title': u'from beginner to master'} 
```

这个命令就不行了，因为它只返回第一条。必须要：

```py
>>> for i in books.find():
...     print i
... 
{u'lang': u'Python', u'_id': ObjectId('554f0e3cf579bc0767db9edf'), u'author': u'qiwsir', u'title': u'from beginner to master'}
{u'lang': u'english', u'title': u'physics', u'_id': ObjectId('554f28f465db941152e6df8b'), u'author': u'Newton'} 
```

在 books 引用的对象中有 find() 方法，它返回的是一个可迭代对象，包含着集合中所有的文档。

由于文档是键值对，也不一定每条文档都要结构一样，比如，也可以插入这样的文档进入集合。

```py
>>> books.insert({"name":"Hertz"})
ObjectId('554f2b4565db941152e6df8c')
>>> for i in books.find():
...     print i
... 
{u'lang': u'Python', u'_id': ObjectId('554f0e3cf579bc0767db9edf'), u'author': u'qiwsir', u'title': u'from beginner to master'}
{u'lang': u'english', u'title': u'physics', u'_id': ObjectId('554f28f465db941152e6df8b'), u'author': u'Newton'}
{u'_id': ObjectId('554f2b4565db941152e6df8c'), u'name': u'Hertz'} 
```

如果有多个文档，想一下子插入到集合中（在 mysql 中，可以实现多条数据用一条命令插入到表里面，还记得吗？忘了看上一节），可以这么做：

```py
>>> n1 = {"title":"java", "name":"Bush"}
>>> n2 = {"title":"fortran", "name":"John Warner Backus"}
>>> n3 = {"title":"lisp", "name":"John McCarthy"}
>>> n = [n1, n2, n3]
>>> n
[{'name': 'Bush', 'title': 'java'}, {'name': 'John Warner Backus', 'title': 'fortran'}, {'name': 'John McCarthy', 'title': 'lisp'}]
>>> books.insert(n)
[ObjectId('554f30be65db941152e6df8d'), ObjectId('554f30be65db941152e6df8e'), ObjectId('554f30be65db941152e6df8f')] 
```

这样就完成了所谓的批量插入，查看一下文档条数：

```py
>>> books.find().count()
6 
```

但是，要提醒读者，批量插入的文档大小是有限制的，网上有人说不要超过 20 万条，有人说不要超过 16MB，我没有测试过。在一般情况下，或许达不到上线，如果遇到极端情况，就请读者在使用时多注意了。

如果要查询，除了通过循环之外，能不能按照某个条件查呢？比如查找`'name'='Bush'`的文档：

```py
>>> books.find_one({"name":"Bush"})
{u'_id': ObjectId('554f30be65db941152e6df8d'), u'name': u'Bush', u'title': u'java'} 
```

对于查询结果，还可以进行排序：

```py
>>> for i in books.find().sort("title", pymongo.ASCENDING):
...     print i
... 
{u'_id': ObjectId('554f2b4565db941152e6df8c'), u'name': u'Hertz'}
{u'_id': ObjectId('554f30be65db941152e6df8e'), u'name': u'John Warner Backus', u'title': u'fortran'}
{u'lang': u'python', u'_id': ObjectId('554f0e3cf579bc0767db9edf'), u'author': u'qiwsir', u'title': u'from beginner to master'}
{u'_id': ObjectId('554f30be65db941152e6df8d'), u'name': u'Bush', u'title': u'java'}
{u'_id': ObjectId('554f30be65db941152e6df8f'), u'name': u'John McCarthy', u'title': u'lisp'}
{u'lang': u'english', u'title': u'physics', u'_id': ObjectId('554f28f465db941152e6df8b'), u'author': u'Newton'} 
```

这是按照"title"的值的升序排列的，注意 sort() 中的第二个参数，意思是升序排列。如果按照降序，就需要将参数修改为 `Pymongo.DESCEDING`，也可以指定多个排序键。

```py
>>> for i in books.find().sort([("name",pymongo.ASCENDING),("name",pymongo.DESCENDING)]):
...     print i
... 
{u'_id': ObjectId('554f30be65db941152e6df8e'), u'name': u'John Warner Backus', u'title': u'fortran'}
{u'_id': ObjectId('554f30be65db941152e6df8f'), u'name': u'John McCarthy', u'title': u'lisp'}
{u'_id': ObjectId('554f2b4565db941152e6df8c'), u'name': u'Hertz'}
{u'_id': ObjectId('554f30be65db941152e6df8d'), u'name': u'Bush', u'title': u'java'}
{u'lang': u'python', u'_id': ObjectId('554f0e3cf579bc0767db9edf'), u'author': u'qiwsir', u'title': u'from beginner to master'}
{u'lang': u'english', u'title': u'physics', u'_id': ObjectId('554f28f465db941152e6df8b'), u'author': u'Newton'} 
```

读者如果看到这里，请务必注意一个事情，那就是 mongodb 中的每个文档，本质上都是“键值对”的类字典结构。这种结构，一经 Python 读出来，就可以用字典中的各种方法来操作。与此类似的还有一个名为 json 的东西，可以阅读本教程第贰季进阶的第陆章模块中的《标准库(8)。但是，如果用 Python 读过来之后，无法直接用 json 模块中的 json.dumps() 方法操作文档。其中一种解决方法就是将文档中的`'_id'`键值对删除（例如：`del doc['_id']`），然后使用 json.dumps() 即可。读者也可是使用 json_util 模块，因为它是“Tools for using Python’s json module with BSON documents”，请阅读[`api.mongodb.org/Python/current/api/bson/json_util.html`](http://api.mongodb.org/Python/current/api/bson/json_util.html)中的模块使用说明。

**更新**

对于已有数据，进行更新，是数据库中常用的操作。比如，要更新 name 为 Hertz 那个文档：

```py
>>> books.update({"name":"Hertz"}, {"$set": {"title":"new physics", "author":"Hertz"}})
{u'updatedExisting': True, u'connectionId': 4, u'ok': 1.0, u'err': None, u'n': 1}
>>> books.find_one({"author":"Hertz"})
{u'title': u'new physics', u'_id': ObjectId('554f2b4565db941152e6df8c'), u'name': u'Hertz', u'author': u'Hertz'} 
```

在更新的时候，用了一个 `$set` 修改器，它可以用来指定键值，如果键不存在，就会创建。

关于修改器，不仅仅是这一个，还有别的呢。

| 修改器 | 描述 |
| --- | --- |
| $set | 用来指定一个键的值。如果不存在则创建它 |
| $unset | 完全删除某个键 |
| $inc | 增加已有键的值，不存在则创建（只能用于增加整数、长整数、双精度浮点数） |
| $push | 数组修改器只能操作值为数组，存在 key 在值末尾增加一个元素，不存在则创建一个数组 |

**删除**

删除可以用 remove() 方法：

```py
>>> books.remove({"name":"Bush"})
{u'connectionId': 4, u'ok': 1.0, u'err': None, u'n': 1}
>>> books.find_one({"name":"Bush"})
>>> 
```

这是将那个文档全部删除。当然，也可以根据 mongodb 的语法规则，写个条件，按照条件删除。

**索引**

索引的目的是为了让查询速度更快，当然，在具体的项目开发中，要视情况而定是否建立索引。因为建立索引也是有代价的。

```py
>>> books.create_index([("title", pymongo.DESCENDING),])
u'title_-1' 
```

我这里仅仅是对 Pymongo 模块做了一个非常简单的介绍，在实际使用过程中，上面知识是很有限的，所以需要读者根据具体应用场景再结合 mongodb 的有关知识去尝试新的语句。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## SQLite 数据库

SQLite 是一个小型的关系型数据库，它最大的特点在于不需要服务器、零配置。在前面的两个服务器，不管是 MySQL 还是 MongoDB，都需要“安装”，安装之后，它运行起来，其实是已经有一个相应的服务器在跑着呢。而 SQLite 不需要这样，首先 Python 已经将相应的驱动模块作为标准库一部分了，只要安装了 Python，就可以使用；另外，它也不需要服务器，可以类似操作文件那样来操作 SQLite 数据库文件。还有一点也不错，SQLite 源代码不受版权限制。

SQLite 也是一个关系型数据库，所以 SQL 语句，都可以在里面使用。

跟操作 mysql 数据库类似，对于 SQLite 数据库，也要通过以下几步：

*   建立连接对象
*   连接对象方法：建立游标对象
*   游标对象方法：执行 sql 语句

### 建立连接对象

由于 SQLite 数据库的驱动已经在 Python 里面了，所以，只要引用就可以直接使用

```py
>>> import sqlite3
>>> conn = sqlite3.connect("23301.db") 
```

这样就得到了连接对象，是不是比 mysql 连接要简化了很多呢。在 `sqlite3.connect("23301.db")` 语句中，如果已经有了那个数据库，就连接上它；如果没有，就新建一个。注意，这里的路径可以随意指定的。

不妨到目录中看一看，是否存在了刚才建立的数据库文件。

```py
/2code$ ls 23301.db
23301.db 
```

果然有了一个文件。

连接对象建立起来之后，就要使用连接对象的方法继续工作了。

```py
>>> dir(conn)
['DataError', 'DatabaseError', 'Error', 'IntegrityError', 'InterfaceError', 'InternalError', 'NotSupportedError', 'OperationalError', 'ProgrammingError', 'Warning', '__call__', '__class__', '__delattr__', '__doc__', '__enter__', '__exit__', '__format__', '__getattribute__', '__hash__', '__init__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'close', 'commit', 'create_aggregate', 'create_collation', 'create_function', 'cursor', 'enable_load_extension', 'execute', 'executemany', 'executescript', 'interrupt', 'isolation_level', 'iterdump', 'load_extension', 'rollback', 'row_factory', 'set_authorizer', 'set_progress_handler', 'text_factory', 'total_changes'] 
```

### 游标对象

这步跟 mysql 也类似，要建立游标对象。

```py
>>> cur = conn.cursor() 
```

接下来对数据库内容的操作，都是用游标对象方法来实现了：

```py
>>> dir(cur)
['__class__', '__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__iter__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'arraysize', 'close', 'connection', 'description', 'execute', 'executemany', 'executescript', 'fetchall', 'fetchmany', 'fetchone', 'lastrowid', 'next', 'row_factory', 'rowcount', 'setinputsizes', 'setoutputsize'] 
```

是不是看到熟悉的名称了：`close(), execute(), executemany(), fetchall()`

#### 创建数据库表

在 mysql 中，我们演示的是利用 mysql 的 shell 来创建的表。其实，当然可以使用 sql 语句，在 Python 中实现这个功能。这里对 sqlite 数据库，就如此操作一番。

```py
>>> create_table = "create table books (title text, author text, lang text) "
>>> cur.execute(create_table)
<sqlite3.Cursor object at 0xb73ed5a0> 
```

这样就在数据库 23301.db 中建立了一个表 books。对这个表可以增加数据了：

```py
>>> cur.execute('insert into books values ("from beginner to master", "laoqi", "python")')
<sqlite3.Cursor object at 0xb73ed5a0> 
```

为了保证数据能够保存，还要（这是多么熟悉的操作流程和命令呀）：

```py
>>> conn.commit()
>>> cur.close()
>>> conn.close() 
```

支持，刚才建立的那个数据库中，已经有了一个表 books，表中已经有了一条记录。

整个流程都不陌生。

#### 查询

存进去了，总要看看，这算强迫症吗？

```py
>>> conn = sqlite3.connect("23301.db")
>>> cur = conn.cursor()
>>> cur.execute('select * from books')
<sqlite3.Cursor object at 0xb73edea0>
>>> print cur.fetchall()
[(u'from beginner to master', u'laoqi', u'python')] 
```

#### 批量插入

多增加点内容，以便于做别的操作：

```py
>>> books = [("first book","first","c"), ("second book","second","c"), ("third book","second","python")] 
```

这回来一个批量插入

```py
>>> cur.executemany('insert into books values (?,?,?)', books)
<sqlite3.Cursor object at 0xb73edea0>
>>> conn.commit() 
```

用循环语句打印一下查询结果：

```py
>>> rows = cur.execute('select * from books')
>>> for row in rows:
...     print row
... 
(u'from beginner to master', u'laoqi', u'python')
(u'first book', u'first', u'c')
(u'second book', u'second', u'c')
(u'third book', u'second', u'python') 
```

#### 更新

正如前面所说，在 cur.execute() 中，你可以写 SQL 语句，来操作数据库。

```py
>>> cur.execute("update books set title='physics' where author='first'")
<sqlite3.Cursor object at 0xb73edea0>
>>> conn.commit() 
```

按照条件查处来看一看：

```py
>>> cur.execute("select * from books where author='first'")
<sqlite3.Cursor object at 0xb73edea0>
>>> cur.fetchone()
(u'physics', u'first', u'c') 
```

#### 删除

在 sql 语句中，这也是常用的。

```py
>>> cur.execute("delete from books where author='second'")
<sqlite3.Cursor object at 0xb73edea0>
>>> conn.commit()

>>> cur.execute("select * from books")
<sqlite3.Cursor object at 0xb73edea0>
>>> cur.fetchall()
[(u'from beginner to master', u'laoqi', u'python'), (u'physics', u'first', u'c')] 
```

不要忘记，在你完成对数据库的操作是，一定要关门才能走人：

```py
>>> cur.close()
>>> conn.close() 
```

作为基本知识，已经介绍差不多了。当然，在实践的编程中，或许会遇到问题，就请读者多参考官方文档：[`docs.Python.org/2/library/sqlite3.html`](https://docs.Python.org/2/library/sqlite3.html)

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 电子表格

一提到电子表格，可能立刻想到的是 excel。殊不知，电子表格，还是“历史悠久”的呢，比 word 要长久多了。根据维基百科的记载整理一个简史：

> VisiCalc 是第一个电子表格程序，用于苹果 II 型电脑。由丹·布李克林（Dan Bricklin）和鮑伯·法兰克斯顿（Bob Frankston）发展而成，1979 年 10 月跟著苹果二号电脑推出，成为苹果二号电脑上的「杀手应用软件」。
> 
> 接下来是 Lotus 1-2-3，由 Lotus Software（美国莲花软件公司）于 1983 年起所推出的电子试算表软件，在 DOS 时期广为个人电脑使用者所使用，是一套杀手级应用软件。也是世界上第一个销售超过 100 万套的软件。
> 
> 然后微软也开始做电子表格，早在 1982 年，它推出了它的第一款电子制表软件──Multiplan，並在 CP/M 系统上大获成功，但在 MS-DOS 系统上，Multiplan 败给了 Lotus 1-2-3。
> 
> 1985 年，微软推出第一款 Excel，但它只用于 Mac 系統；直到 1987 年 11 月，微软的第一款适用于 Windows 系統的 Excel 才诞生，不过，它一出来，就与 Windows 系统直接捆綁，由于此后 windows 大行其道，并且 Lotus1-2-3 迟迟不能适用于 Windows 系統，到了 1988 年，Excel 的销量超过了 1-2-3。
> 
> 此后就是微软的天下了，Excel 后来又并入了 Office 里面，成为了 Microsoft Office Excel。
> 
> 尽管 Excel 已经发展了很多代，提供了大量的用戶界面特性，但它仍然保留了第一款电子制表软件 VisiCalc 的特性：行、列組成单元格，数据、与数据相关的公式或者对其他单元格的绝对引用保存在单元格中。
> 
> 由于微软独霸天下，Lotus 1-2-3 已经淡出了人们的视线，甚至于误认为历史就是从微软开始的。
> 
> 其实，除了微软的电子表格，在 Linux 系统中也有很好的电子表格，google 也提供了不错的在线电子表格（可惜某国内不能正常访问）。

从历史到现在，电子表格都很广泛的用途。所以，Python 也要操作一番电子表格，因为有的数据，或许就是存在电子表格中。

### openpyl

openpyl 模块是解决 Microsoft Excel 2007/2010 之类版本中扩展名是 Excel 2010 xlsx/xlsm/xltx/xltm 的文件的读写的第三方库。（差点上不来气，这句话太长了。）

#### 安装

安装第三方库，当然用法力无边的 pip install

```py
$ sudo pip install openpyxl 
```

如果最终看到下面的提示，恭喜你，安装成功。

```py
Successfully installed openpyxl jdcal
Cleaning up... 
```

#### workbook 和 sheet

第一步，当然是要引入模块，用下面的方式：

```py
>>> from openpyxl import Workbook 
```

接下来就用 `Workbook()` 类里面的方法展开工作：

```py
>>> wb = Workbook() 
```

请回忆 Excel 文件，如果想不起来，就打开 Excel，我们第一眼看到的是一个称之为工作簿(workbook)的东西，里面有几个 sheet，默认是三个，当然可以随意增删。默认又使用第一个 sheet。

```py
>>> ws = wb.active 
```

每个工作簿中，至少要有一个 sheet，通过这条指令，就在当前工作簿中建立了一个 sheet，并且它是当前正在使用的。

还可以在这个 sheet 后面追加：

```py
>>> ws1 = wb.create_sheet() 
```

甚至，还可以加塞：

```py
>>> ws2 = wb.create_sheet(1) 
```

排在了第二个位置。

在 Excel 文件中一样，创建了 sheet 之后，默认都是以"Sheet1"、"Sheet2"样子来命名的，然后我们可以给其重新命名。在这里，依然可以这么做。

```py
>>> ws.title = "Python" 
```

ws 所引用的 sheet 对象名字就是"Python"了。

此时，可以使用下面的方式从工作簿对象中得到 sheet

```py
>>> ws01 = wb['Python']    #sheet 和工作簿的关系，类似键值对的关系
>>> ws is ws01
True 
```

或者用这种方式

```py
>>> ws02 = wb.get_sheet_by_name("Python")    #这个方法名字也太直接了，方法的参数就是 sheet 名字
>>> ws is ws02
True 
```

整理一下到目前为止我们已经完成的工作：建立了工作簿(wb)，还有三个 sheet。还是显示一下比较好：

```py
>>> print wb.get_sheet_names()
['Python', 'Sheet2', 'Sheet1'] 
```

Sheet2 这个 sheet 之所以排在了第二位，是因为在建立的时候，用了一个加塞的方法。这跟 Excel 中差不多少，如果 sheet 命名了，就按照那个名字显示，否则就默认为名字是"Sheet1"形状的（注意，第一个字母大写）。

也可以用循环语句，把所有的 sheet 名字打印出来。

```py
>>> for sh in wb:
...     print sh.title
... 
Python
Sheet2
Sheet1 
```

如果读者去 `dir(wb)` 工作簿对象的属性和方法，会发现它具有迭代的特征`__iter__`方法。说明，工作簿是可迭代的。

#### cell

为了能够清楚理解填数据的过程，将电子表中约定的名称以下图方式说明：

![](img/23401.jpg)

对于 sheet，其中的 cell 是它的下级单位。所以，要得到某个 cell，可以这样：

```py
b4 = ws['B4'] 
```

如果 B4 这个 cell 已经有了，用这种方法就是将它的值赋给了变量 b4；如果 sheet 中没有这个 cell，那么就创建这个 cell 对象。

请读者注意，当我们打开 Excel，默认已经画好了好多 cell。但是，在 Python 操作的电子表格中，不会默认画好那样一个表格，一切都要创建之后才有。所以，如果按照前面的操作流程，上面就是创建了 B4 这个 cell，并且把它作为一个对象被 b4 变量引用。

如果要给 B4 添加数据，可以这么做：

```py
>>> ws['B4'] = 4444 
```

因为 b4 引用了一个 cell 对象，所以可以利用这个对象的属性来查看其值：

```py
>>> b4.value
4444 
```

要获得（或者建立并获得）某个 cell 对象，还可以使用下面方法：

```py
>>> a1 = ws.cell("A1") 
```

或者：

```py
>>> a2 = ws.cell(row = 2, column = 1) 
```

刚才已经提到，在建立了 sheet 之后，内存中的它并没有 cell，需要程序去建立。上面都是一个一个地建立，能不能一下建立多个呢？比如要类似下面的：

|A1|B1|C1| |A2|B2|C2| |A3|B3|C3|

就可以如同切片那样来操作：

```py
>>> cells = ws["A1":"C3"] 
```

可以用下面方法看看创建结果：

```py
>>> tuple(ws.iter_rows("A1:C3"))
((<Cell python.A1>, <Cell Python.B1>, <Cell Python.C1>), 
 (<Cell python.A2>, <Cell Python.B2>, <Cell Python.C2>), 
 (<Cell python.A3>, <Cell Python.B3>, <Cell Python.C3>)) 
```

这是按照横向顺序数过来来的，即 A1-B1-C1，然后下一横行。还可以用下面的循环方法，一个一个地读到每个 cell 对象：

```py
>>> for row in ws.iter_rows("A1:C3"):
...     for cell in row:
...         print cell
... 
<Cell Python.A1>
<Cell Python.B1>
<Cell Python.C1>
<Cell Python.A2>
<Cell Python.B2>
<Cell Python.C2>
<Cell Python.A3>
<Cell Python.B3>
<Cell Python.C3> 
```

也可以用 sheet 对象的 `rows` 属性，得到按照横向顺序依次排列的 cell 对象（注意观察结果，因为没有进行范围限制，所以是目前 sheet 中所有的 cell，前面已经建立到第四行了 B4，所以，要比上面的操作多一个 row）：

```py
>>> ws.rows
((<Cell python.A1>, <Cell python.B1>, <Cell python.C1>), 
 (<Cell python.A2>, <Cell python.B2>, <Cell python.C2>), 
 (<Cell python.A3>, <Cell python.B3>, <Cell python.C3>), 
 (<Cell python.A4>, <Cell python.B4>, <Cell python.C4>)) 
```

用 sheet 对象的 `columns` 属性，得到的是按照纵向顺序排列的 cell 对象（注意观察结果）：

```py
>>> ws.columns
((<Cell python.A1>, <Cell python.A2>, <Cell python.A3>, <Cell python.A4>), 
 (<Cell python.B1>, <Cell python.B2>, <Cell python.B3>, <Cell python.B4>), 
 (<Cell python.C1>, <Cell python.C2>, <Cell python.C3>, <Cell python.C4>)) 
```

不管用那种方法，只要得到了 cell 对象，接下来就可以依次赋值了。比如要将上面的表格中，依次填写上 1,2,3,...

```py
>>> i = 1
>>> for cell in ws.rows:
...     cell.value = i
...     i += 1 
```

... Traceback (most recent call last): File "<stdin>", line 2, in <module>AttributeError: 'tuple' object has no attribute 'value'</module></stdin>

报错了。什么错误。关键就是没有注意观察上面的结果。tuple 里面是以 tuple 为元素，再里面才是 cell 对象。所以，必须要“时时警醒”，常常谨慎。

```py
>>> for row in ws.rows:
...     for cell in row:
...         cell.value = i
...         i += 1
... 
```

如此，就给每个 cell 添加了数据。查看一下，不过要换一个属性：

```py
>>> for col in ws.columns:
...     for cell in col:
...         print cell.value
... 
1
4
7
10
2
5
8
11
3
6
9
12 
```

虽然看着有点不舒服，但的确达到了前面的要求。

#### 保存

把辛苦工作的结果保存一下吧。

```py
>>> wb.save("23401.xlsx") 
```

如果有同名文件存在，会覆盖。

此时，可以用 Excel 打开这个文件，看看可视化的结果：

![](img/23402.jpg)

#### 读取已有文件

如果已经有一个 .xlsx 文件，要读取它，可以这样来做：

```py
>>> from openpyxl import load_workbook
>>> wb2 = load_workbook("23401.xlsx")
>>> print wb2.get_sheet_names()
['python', 'Sheet2', 'Sheet1']
>>> ws_wb2 = wb2["python"]
>>> for row in ws_wb2.rows:
...     for cell in row:
...         print cell.value
... 
1
2
3
4
5
6
7
8
9
10
11
12 
```

很好，就是这个文件。

### 其它第三方库

针对电子表格的第三方库，除了上面这个 openpyxl 之外，还有别的，列出几个，供参考，使用方法大同小异。

*   xlsxwriter：针对 Excel 2010 格式，如 .xlsx，官方网站：[`xlsxwriter.readthedocs.org/`](https://xlsxwriter.readthedocs.org/)，这个官方文档写的图文并茂。非常好读。

下面两个用来处理 .xls 格式的电子表表格。

*   xlrd：网络文件：[`secure.simplistix.co.uk/svn/xlrd/trunk/xlrd/doc/xlrd.html?p=4966`](https://secure.simplistix.co.uk/svn/xlrd/trunk/xlrd/doc/xlrd.html?p=4966)
*   xlwt：网络文件：[`xlwt.readthedocs.org/en/latest/`](http://xlwt.readthedocs.org/en/latest/)

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。