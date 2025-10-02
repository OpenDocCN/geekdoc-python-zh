# 第十三章 数据库支持

> 来源：[`www.cnblogs.com/Marlowes/p/5537223.html`](http://www.cnblogs.com/Marlowes/p/5537223.html)
> 
> 作者：Marlowes

使用简单的纯文本文件只能实现有限的功能。没错，使用它们可以做很多事情，但有时需要额外的功能。你可能想要自动序列化，这时可以选择`shelve`模块(见第十章)和`pickle`(与`shelve`模块关系密切)。但有时，可能需要比这更强大的特性。例如，可能想自动地支持数据并发访问——想让几个用户同时对基于磁盘的数据进行读写而不造成任何文件损坏这类的问题。或者希望同时使用多个数据字段或属性进行复杂的搜索，而不是只通过`shelve`做简单的单键查找。解决的方案有很多，但是如果要处理的数据量巨大而同时还希望其他程序员能轻易理解的话，选择相对来说更标准化的*数据库*(database)可能是个好主意。

本章会对 Python 的 Database API 进行讨论，这是一种连接 SQL 数据库的标准化方法；同时也会展示如何用 API 执行一些基本的 SQL 命令。最后一节会对其他可选的数据库技术进行讨论。

我不打算把这章写成关系型数据库或 SQL 语言的教程。多数数据库的文档(比如 PostgreSQL、MySQL，以及本章用到的 SQLite 数据库)都应该能提供相关的基础知识。如果以前没用过关系型数据库，也可以访问 [`www.sqlcourse.com`](http://www.sqlcourse.com) ，或者干脆网上搜一下相关主题，或查看由 Clare Churcher 著的*Beginning SQL Queries*(Apress，2008 年出版)。

当然，本章使用的简单数据库(SQLite)并不是唯一的选择。还有一些流行的商业数据库(比如 Oracle 或 Microsoft SQL Server)以及很多稳定且被广泛使用的开源数据库可供选择(比如 MySQL、PostgreSQL 和 Firebird)。第二十六章中使用了 PostgreSQL，并且介绍了一些 MySQL 和 SQLite 的使用指导。关于其他 Python 包支持的数据库，请访问 [`www.python.org/topics/database/`](http://www.python.org/topics/database/) ，或者访问[Vaults of Parnassus 的数据库分类](http://www.vex.net/parnassus)。

关系型(SQL)数据库不是唯一的数据库类别。还有不少类似于[ZODB](http://wiki.zope.org/ZODB)的对象数据库、类似[Metakit](http://www.equi4.com/metakit/python.html)基于表的精简数据库，和类似于[BSD DB](http://docs.python.org/lib/module-bsddb.html)的更简单的*键-值*数据库。

本章着重介绍低级数据库的交互，你会发现几个高级库可以帮助完成一些复杂的工作(例如，参见 [`www.sqlalchemy.org`](http://www.sqlalchemy.org) 或者 [`www.sqlobject.org`](http://www.sqlobject.org) ，或者在网络上搜索 Python 的对象-关系映射)。

## 13.1 Python 数据库 API

支持 SQL 标准的可用数据库有很多，其中多数在 Python 中都有对应的客户端模块(有些数据库甚至有多个模块)。所有数据库的大多数基本功能都是相同的，所以写一个程序来使用其中的某个数据库是很容易的事情，而且“理论上”该程序也应该能在别的数据库上运行。在提供相同功能(基本相同)的不同模块之间进行切换时的问题通常是它们的接口(API)不同。为了解决 Python 中各种数据库模块间的兼容问题，现在已经通过了一个标准的 DB API。目前的 API 版本(2.0)定义在 PEP249 中的[Python Database API Specification v2.0](http://python.org/peps/pep-0249.html)中。

本节将对基本概念做一综述。并且不会提到 API 的可选部分，因为它们不见得对所有数据库都适用。可以在 PEP 中找到更多的信息，或者可以访问官方的 Python 维基百科中的[数据库编程指南](http://wiki.python.org/moin/DatabaseProgramming)。如果对 API 的细节不感兴趣，可以跳过本节。

### 13.1.1 全局变量

任何支持 2.0 版本 DB API 的数据库模块都必须定义 3 个描述模块特性的全局变量。这样做的原因是 API 设计得很灵活，以支持不同的基础机制、避免过多包装，可如果想让程序同时应用于几个数据库，那可是件麻烦事了，因为需要考虑到各种可能出现的状况。多数情况下，比较现实的做法是检查这些变量，看看给定的数据库模块是否能被程序接受。如果不能，就显示合适的错误信息然后退出，例如抛出一些异常。3 种全局变量如表 13-1 所示。

表 13-1 Python DB API 的模块特性

```py
apilevel　　　　　　　　　　　所使用的 Python DB API 版
threadsafety　　　　　　　　　模块的线程安全等级
paramstyle　　　　　　　　　　在 SQL 查询中使用的参数风格 
```

API 级别(`apilevel`)是个字符串常量，提供正在使用的 API 版本号。对 DBAPI 2.0 版本来说，其值可能是'1.0'也可能是'2.0'。如果这个变量不存在，那么模块就不适用于 2.0 版本，根据 API 应该假定当前使用的是 DB API 1.0。在程序中提供对其他可能值的支持没有坏处，谁知道呢，说不定什么时候 DBAPI 的 3.0 版本就出来了。

线程安全性等级(`threadsafety`)是个取值范围为 0~3 的整数。0 表示线程完全不共享模块，而 3 表示模块是完全线程安全的。1 表示线程本身可以共享模块，但不对连接共享(参见 13.1.3 节)。如果不使用多个线程(多数情况下可能不会这样做)，那么完全不用担心这个变量。

参数风格(`paramstyle`)表示在执行多次类似查询的时候，参数是如何被拼接到 SQL 查询中的。值`'format'`表示标准的字符串格式化(使用基本的格式代码)，可以在参数中进行拼接的地方插入`%s`。而值`'pyformat'`表示扩展的格式代码，用于字典拼接中，比如`%(foo)`。除了 Python 风格之外，还有第三种接合方式：`'qmark'`的意思是使用问号，而`'numeric'`表示使用`:1`或者`:2`格式的字段(数字表示参数的序号)，而`'named'`表示`:foobar`这样的字段，其中`foobar`为参数名。如果参数风格看起来有些让人迷惑，别担心。对于基础程序来说，不会用到这些参数，如果需要了解特定的数据库接口如何处理参数，在相关的文档中会进行解释。

### 13.1.2 异常

为了能尽可能准确地处理错误，API 中定义了一些异常类。它们被定义在一种层次结构中，所以可能通过一个`except`块捕捉多种异常。(当然要是你觉得一切都能运行良好，或者根本不在乎程序因为某些事情出错这类不太可能发生的时间而突然停止运行，那么完全可以忽略这些异常)

异常的层次如表 13-2 所示。在给定的数据库模块中异常应该是全局可用的。关于这些异常的深度描述，请参见 API 规范(也就是前面提到的 PEP)。

表 13-2 在 DB API 中使用的异常

```py
异常　　　　　　　　　　超类　　　　　　　　　　描述
StandardError　　　　　　　　　　　　　　　　　 所有异常的泛型基类
Warning　　　　　　　　 StandardError　　　　　 在非致命错误发生时引发
Error　　　　　　　　　 StandardError　　　　　 所有错误条件的泛型超类
InterfaceError　　　　　Error　　　　　　　　　 关于接口而非数据库的错误
DatabaseError　　　　　 Error　　　　　　　　　 与数据库相关的错误的基类
DataError　　　　　　　 DatabaseError　　　　　 与数据库相关的问题，比如值超出范围
OperationalError　　　　DatabaseError　　　　　 数据库内部操作错误
IntegrityError　　　　　DatabaseError　　　　　 关系完整性受到影响，比如键检查失败
InternalError　　　　　 DatabaseError　　　　　 数据库内部错误，比如非法游标
ProgrammingError　　　　DatabaseError　　　　　 用户编程错误，比如未找到表
NotSupportedError　　　 DatabaseError　　　　　 请求不支持的特性，比如回滚 
```

### 13.1.3 连接和游标

为了使用基础数据库系统，首先必须连接到它。这个时候需要使用具有恰当名称的`connect`函数，该函数有多个参数，而具体使用哪个参数取决于数据库。API 定义了表 13-3 中的参数作为准则，推荐将这些参数作为关键字参数使用，并按表中给定的顺序传递它们。参数类型都应为字符串。

表 13-3 connect 函数的常用参数

```py
参数名　　　　　　　　　　描述　　　　　　　　　　                  是否可选
dsn　　　　　　　　　　　 数据源名称，给出该参数表示数据库依赖　　　否
user　　　　　　　　　　　用户名　　　　　　　　　　　　　　　　　　是
password　　　　 　　　　 用户密码　　　　　　 　　　　　　　　　　 是
host　　　　　　　　　　　主机名　　　　　　　　　　　　　　　　　　是
database　　　　　　　　　数据库名　　　　 　　　　　　 　　　　　　是 
```

13.2.1 节以及第二十六章会介绍使用`connect`函数的具体的例子。

`connect`函数返回连接对象。这个对象表示目前和数据库的会话。连接对象支持的方法如表 13-4 所示。

13-4 连接对象方法

```py
close()　　　　　　　　　　　　　　　　　　　 关闭连接之后，连接对象和它的游标均不可用
commit()　　　　　　　　　　　　　　　　　　　如果支持的话就提交挂事务，否则不做任何事
rollback()　　　　　　　　　　　　　　　　　　回滚挂起的事务(可能不可用)
cursor()　　　　　　　　　　　　　　　　　　　返回连接的游标对象 
```

`rollback`方法可能不可用，因为不是所有的数据库都支持事务(*事务*是一系列动作)。如果可用，那么它就可以“撤销”所有未提交的事务。

`commit`方法总是可用的，但是如果数据库不支持事务，它就没有任何作用。如果关闭了连接但还有未提交的事务，它们会隐式地回滚——但是只有在数据库支持回滚的时候才可以。所以如果不想完全依靠隐式回滚，就应该每次在关闭连接前进行提交。如果提交了，那么就用不着担心关闭连接的问题，它会在进行垃圾收集时自动关闭。当然如果希望更安全一些，就调用`close`方法，也不会敲很多次键盘。

`cursor`方法将我们引入另外一个主题：游标对象。通过游标执行 SQL 查询并检查结果。游标比连接支持更多的方法，而且可能在程序中更好用。表 13-5 给出了游标方法的概述，表 13-6 则是特性的概述。

表 13-5 游标对象方法

```py
callproc(name[, params])　　　　使用给定的名称和参数(可选)调用已命名的数据库程序
close()　　　　　　　　　　　　 关闭游标之后，游标不可用
execute(oper[, params])　　　　 执行 SQL 操作，可能使用参数
executemany(oper, pseq)　　　　 对序列中的每个参数执行 SQL 操作
fetchone()　　　　　　　　　　　把查询的结果集中的下一行保存为序列，或者 None
fetchmany([size])　　　　　　　 获取查询结果集中的多行，默认尺寸为 arraysize
fetchall()　　　　　　　　　　　将所有(剩余)的行作为序列的序列
nextset()　　　　　　　　　　　 跳至下一个可用的结果集(可选)
setinputsizes(sizes)　　　　　　为参数预先定义内存区域
setoutputsize(size[, col])　　　为获取的大数据值设定缓冲区尺寸 
```

表 13-6 游标对象特性

```py
description　　　　　　　　　 结果列描述的序列，只读
rowcount　　　　　　　　　　　结果中的行数，只读
arraysize　　　　　　　　　　 fetchmany 中返回的行数，默认为 1 
```

其中一些方法会在下面详细介绍，而有些(比如`setinputsizes`和`setoutputsizes`)则不会提到。更多细节请查阅 PEP。

### 13.1.4 类型

数据库对插入到具有某种类型的列中的值有不同的要求，是为了能正确地与基础 SQL 数据库进行交互操作，DB API 定义了用于特殊类型和值的构造函数以及常量(单例模式)。例如，如果想要在数据库中增加日期，它应该用相应的数据库连接模块的`Date`构造函数来建立。这样数据库连接模块就可以在幕后执行一些必要的转换操作。所有模块都要求实现表 13-7 中列出的构造函数和特殊值。一些模块可能不是完全按照要求去做，例如`sqlite3`模块(接下来会讨论)并不会输出表 13-7 中的特殊值(通过`ROWIP`输出`STRING`)。

表 13-7 DB API 构造函数和特殊值

```py
Date(year, month, day)　　　　　　　　创建保存日期值的对象
Time(hour, minute, second)　　　　　　创建保存时间值的对象
Timestamp(y, mon, d, h, min, s)　　　 创建保存时间戳值的对象
DateFromTicks(ticks)　　　　　　　　　创建保存自新纪元以来秒数的对象
TimeFromTicks(ticks)　　　　　　　　　创建保存来自秒数的时间值的对象
TimestampTicks(ticks)　　　　　　　　 创建保存来自秒数的时间戳的对象
Binay(string)　　　　　　　　　　　   创建保存二进制字符串值的对象
STRING　　　　　　　　　　　　　　　　描述基于字符串的列类型(比如 CHAR)
BINARY　　　　　　　　　　　　　　　　描述二进制列(比如 LONG 或 RAW)
NUMBER　　　　　　　　　　　　　　　　描述数字列
DATETIME　　　　　　　　　　　　　　　描述日期/时间列
ROWID　　　　　　　　　　　　　　　　 描述行 ID 列 
```

## 13.2 SQLite 和 PySQLite

之前提到过，可用的 SQL 数据库引擎有很多，而且都有相应的 Python 模块。多数数据库引擎都作为服务器程序运行，连安装都需要管理员权限。为了降低练习 Python DB API 的门槛，这里选择了小型的数据库引擎 SQLite，它并不需要作为独立的服务器运行，并且不基于集中式数据库存储机制，而是直接作用于本地文件。

在最近的 Python 版本中(从 2.5 开始)，SQLite 的优势在于它的一个包装(PySQLite)已经被包括在标准库内。除非是从源码开始编译 Python，可能数据库本身也已经包括在内。读者也可以尝试 13.2.1 节介绍的程序段。如果它们可以工作，那么就不用单独安装 PySQLite 和 SQLite 了。

*注：如果读者没有使用 PySQLite 的标准库版本，那么可能还需要修改`import`语句，请参考相关文档获取更多信息。*

**获取 PySQLite**

如果读者正在使用旧版 Python，那么需要在使用 SQLite 数据库前安装 PySQLite，可以从官方网站下载。对于带有包管理系统的 Linux 系统，可能直接从包管理器章获得 PYSQLite 和 SQLite。

针对 PYSQLite 的 Windows 二进制版本实际上包含了数据库引擎(也就是 SQLite)，所以只要下载对应 Python 版本的 PYSQLite 安装程序，运行就可以了。

如果使用的不是 Windows，而操作系统也没有可以找到 PYSQLite 和 SQLite 的包管理器的话，那么就需要 PYSQLite 和 SQLite 的源代码包，然后自己进行编译。

如果使用的 Python 版本较新，那么应该已经包含 PySQLite。接下来需要的可能就是数据库本身 SQLite 了(同样，它可能也包含在内了)。可以从 SQLite 的网站 [`sqlite.org`](http://sqlite.org) 下载源代码(确保得到的是已经完成自动代码生成的包)，按照 README 文件中的指导进行编译即可。在之后编译 PYSQLite 时，需要确保编译过程可以访问 SQLite 的库文件和 include 文件。如果已经在某些标准位置安装了 SQLite，那么可能 SQLite 发布版的安装脚本可以自己找到它，在这种情况下只需执行下面的命令：

```py
python setup.py build
python setup.py install 
```

可以只用后一个命令，让编译自动进行。如果出现大量错误信息，可能是安装脚本找不到所需文件。确保你知道库文件和`include`文件安装到了哪里，将它们显式地提供给安装脚本。假设我在`/home/mlh/sqlite/current`目录中原地编译 SQLite，那么头文件和库文件应该可以在`/home/mlh/sqlite/current/src`和`/home/mlh/sqlite/current/build/lib`中找到。为了让安装程序能使用这些路径，需要编辑安装脚本`setup.py`。在这个文件中可以设定变量`include_dirs`和`library_dirs`：

```py
include_dirs = ['/home/mlh/sqlite/current/src']
include_dirs = ['/home/mlh/sqlite/current/build/lib'] 
```

在重新绑定变量之后，刚才说过的安装过程应该可以正常进行了。

### 13.2.1 入门

可以将 SQLite 作为名为`sqlite3`的模块导入(如果使用的是标准库中的模块)。之后就可以创建一个到数据库文件的连接——如果文件不存在就会被创建——通过提供一个文件名(可以是文件的绝对或者相对路径)：

```py
>>> import sqlite3
>>> conn =  sqlite3.connect("somedatabase.db") 
```

之后就能获得连接的游标：

```py
>>> curs = conn.cursor() 
```

这个游标可以用来执行 SQL 查询。完成查询并且做出某些更改后确保已经进行了提交，这样才可以将这些修改真正地保存到文件中：

```py
>>> conn.commit() 
```

可以(而且是应该)在每次修改数据库后都进行提交，而不是仅仅在准备关闭时才提交，准备关闭数据库时，使用`close`方法：

```py
>>> conn.close() 
```

### 13.2.2 数据库应用程序示例

我会建立一个小型营养成分数据库作为示例程序，这个程序基于[USDA 的营养数据实验室提供的数据](http://www.ars.usda.gov/nutrientdata)。在他们的主页上点击 USDA National Nutrient Database for Standard Reference 链接，就能看到很多以普通文本形式(ASCII)保存的数据文件，这就是需要的内容。点击 Download 链接，下载标题"Abbreviated"下方的 ASCII 链接所指向的 ASCII 格式的`zip`文件。此时应该得到一个`zip`文件，其中包含`ABBREV.txt`文本文件和描述该文件内容的 PDF 文件。

`ABBREV.txt`文件中的数据每行都有一个数据记录，字段以脱字符(`^`)进行分割。数字字段直接包含数字，而文本字段包括由波浪号(`~`)括起来的字符串值，下面是一个示例行，为了简短起见删除了一部分：

```py
~01252~^~CHEESE ... ,PAST PROCESS, ... ^~1 slice,  (3/4 oz)~⁰ 
```

用`line.split("^")`可以很容易地将这样一行文字解析为多个字段。如果字段以波浪号开始，就能知道它是个字符串，可以用`field.strip("~")`获取它的内容。对于其他的(数字)字段来讲可以使用`float(field`)，除非字段是空的。下面一节中的程序将演示把 ASCII 文件中的数据移入 SQL 数据库，然后对其进行一些有意思的查询。

*注：这个示例程序有意提供一个简单的例子。有关相对高级的用于 Python 的数据库的例子，参见第二十六章。*

1.创建和填充表

为了真正地创建数据库表并且向其中插入数据，写个完全独立的一次性程序可能是最简单的方案。运行一次后就可以忘了它和原始数据源(`ABBREV.txt`文件)，尽管保留它们也是不错的主意。

代码清单 13-1 中的程序创建了叫做`food`的表和适当的字段，并且从`ABBREV.txt`中读取数据。之后分解析(行分解为多个字段，并使用应用函数`convert`每个字段进行转换)，然后通过调用`curs.execute`执行 SQL 的`INSERT`语句将文本字段中的值插入到数据库中。

*注：也可以使用`curs.executemany`，然后提供一个从数据文件中提取的所有行的列表。这样做在本例中只会带来轻微的速度提升，但是如果使用通过网络连接的客户机/服务器 SQL 系统，则会大大地提高速度。*

```py
import sqlite3
def convert(value):
    if value.startwith("~"):
        return value.strip("~")
    if not value:
        value = 0
        return float(value)

conn = sqlite3.connect("foo.db")
curs = conn.cursor()

curs.execute("""
CREATE TABLE food (
id        TEXT        PRIMARY KEY,
desc      TEXT,
water     FLOAT,
kcal      FLOAT,
protein   FLOAT,
fat       FLOAT,
ash       FLOAT,
carbs     FLOAT,
fiber     FLOAT,
sugar     FLOAT
) """)

query = "INSERT INTO food VALUES (?,?,?,?,?,?,?,?,?,?)"

for line in open("ABBREV.txt"):
    fields = line.split("^")
    vals = [convert(f)
    for f in fields[:field_count]]
        curs.execute(query, vals)

conn.commit()
conn.close() 
```

`importdata.py`

*注：在代码清单 13-1 中使用了`paramstyle`的“问号”版本，也就是会使用问号作为字段标记。如果使用旧版本的 PySQLite，那么久需要使用`%`字符。*

当运行这个程序(将`ABBREV.txt`放在同一目录)时，它会创建一个叫做`food.db`的新文件，它会包含数据库中的所有数据。

鼓励读者们多尝试修改这个例子，例如使用其他的输入、加入`print`语句，等等。

2.搜索和处理结果

使用数据库很简单。再说一次，需要创建连接并且获得该链接的游标。使用`execute`方法执行 SQL 查询，用`fetchall`等方法提取结果。代码清单 13-2 展示了一个将 SQL SELECT 条件查询作为命令行参数，之后按记录格式打印出返回行的小程序。可以用下面的命令尝试这个程序：

```py
$ python food_query.py "kcal <= 100 AND fiber >= 10 ORDER BY sugar" 
```

运行的时候可能注意到有个问题。第一行，生橘子皮(raw orange peel)看起来不含任何糖分(糖分值为 0)。这是因为在数据文件中这个字段丢失了。可以改进刚才的导入脚本检测条件，然后插入 None 来代替真实的值来表示丢失的数据。可以使用如下条件：

```py
"kcal <= 100 AND fiber >= 10 AND sugar ORDER BY sugar" 
```

请求在任何返回行中包含实际数据的“糖分”字段。这方法恰好也适用于当前的数据库，它会忽略糖分为 0 的行。

*注：使用 ID 搜索特殊的食品项，比如用 08323 搜索 Cocoa Pebble 的时候可能会出现问题。原因在于 SQLite 以一种相当不标准的方式处理它的值。在其内部所有的值实际上都是字符串，一些转换和检查在数据库和 Python API 间进行。通常它工作得很顺利，但有时候也会出错，例如下面这种情况：如果提供值 08323，它会被解释为数字 8323，再转换为字符串"8323"——一个不存在的 ID。可能期待这里抛出异常或者其他什么错误信息，而不是这种毫无用处的非预期行为。但如果小心一些，一开始就用字符串"08323"来表示 ID，就可以工作正常了。*

```py
import sqlite3 import sys

conn = sqlite3.connect("foo.db")
curs = conn.cursor()

query = "SELECT * FROM food WHERE %s" % sys.argv[1]
print query

curs.execute(query)
names = [f[0] for f in curs.description]
for row in curs.fetchall():
    for pair in zip(names, row): print "%s: %s" % pair print 
```

`food_query.py`

## 13.3 小结

本章简要介绍了创建和关系型数据库交互的 Python 程序。这段介绍相当简短，因为掌握了 Python 和 SQL 以后，那么两者的结合——Python DB API 也就容易掌握了。下面是本章一些概念。

☑ Python DB API：提供了简单、标准化的数据库接口，所有数据库的包装模块都应当遵循这个接口，以易于编写跨数据库的程序。

☑ 连接：连接对象代表的是和 SQL 数据库的通信连接。使用`cursor`方法可以从它那获得独立的游标。通过连接对象还可以提交或者回滚事务。在处理完数据库之后，连接可以被关闭。

☑ 游标：用于执行查询和检查结果。结果行可以一个一个地获得，也可以很多个(或全部)一起获得。

☑ 类型和特殊值：DB API 标准制定了一组构造函数和特殊值的名字。构造函数处理日期和时间对象，以及二进制数据对象。特殊值用来表示关系型数据库的类型，比如`STRING`、`NUMBER`和`DATETIME`。

☑ SQLite：小型的嵌入式 SQL 数据库，它的 Python 包装叫做 PYSQLite。它速度快，易于使用，并且不需要建立单独的服务器。

### 13.3.1 本章的新函数

本章涉及的新函数如表 13-8 所示。

表 13-8 本章的新函数

```py
connect(...)                        连接数据库，返回连接对象 
```

### 13.3.2 接下来学什么

坚持不懈数据库处理是绝大多数程序(如果不是大多数，那就是大型程序系统)的重要部分。下一章会介绍另外一个大型程序系统都会用到的组件，即网络。