# 用 Python 防止 SQL 注入攻击

> 原文：<https://realpython.com/prevent-python-sql-injection/>

每隔几年，开放 Web 应用安全项目(OWASP)就会对最关键的 [web 应用安全风险](https://www.owasp.org/index.php/Category:OWASP_Top_Ten_Project)进行排名。自第一份报告以来，注射风险一直居于首位。在所有注射类型中， **SQL 注入**是最常见的攻击媒介之一，也可以说是最危险的。由于 Python 是世界上最流行的编程语言之一，知道如何防范 Python SQL 注入是至关重要的。

在本教程中，你将学习:

*   什么是 **Python SQL 注入**以及如何防范
*   如何用文字和标识符作为参数来组成查询
*   如何在数据库中安全地执行查询

本教程适合所有数据库引擎的**用户。这里的例子使用 PostgreSQL，但是结果可以在其他数据库管理系统中重现(比如 [SQLite](https://realpython.com/python-sqlite-sqlalchemy/) 、 [MySQL](https://realpython.com/python-mysql/) 、微软 SQL Server、Oracle 等等)。**

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 了解 Python SQL 注入

SQL 注入攻击是一个如此常见的安全漏洞，以至于传奇人物 *xkcd* webcomic 专门为此画了一幅漫画:

[![A humorous webcomic by xkcd about the potential effect of SQL injection](img/47c40488350b2c027511efc9a9fe4e10.png)](https://files.realpython.com/media/exploits_of_a_mom.3f89e3f5f263.png)

<figcaption class="figure-caption text-center">"Exploits of a Mom" (Image: [xkcd](https://xkcd.com/327/))</figcaption>

生成和执行 [SQL 查询](https://realpython.com/python-sql-libraries/)是一项常见的任务。然而，世界各地的公司在编写 SQL 语句时经常犯可怕的错误。虽然 [ORM 层](https://en.wikipedia.org/wiki/Object-relational_mapping)通常编写 SQL 查询，但有时您必须自己编写。

当您使用 Python 直接在数据库中执行这些查询时，您有可能会犯错误，危及您的系统。在本教程中，您将学习如何成功实现组成动态 SQL 查询的函数*，而不*让您的系统面临 Python SQL 注入的风险。

[*Remove ads*](/account/join/)

## 建立数据库

首先，您将建立一个新的 PostgreSQL 数据库，并用数据填充它。在整个教程中，您将使用该数据库直接见证 Python SQL 注入的工作原理。

### 创建数据库

首先，打开您的 shell 并创建一个新的 PostgreSQL 数据库，归用户`postgres`所有:

```py
$ createdb -O postgres psycopgtest
```

这里，您使用命令行选项`-O`将数据库的所有者设置为用户`postgres`。您还指定了数据库的名称，即`psycopgtest`。

**注意:** `postgres`是一个**的特殊用户**，它通常用于管理任务，但是在本教程中，使用`postgres`也可以。然而，在实际系统中，您应该创建一个单独的用户作为数据库的所有者。

您的新数据库已经准备就绪！您可以使用`psql`连接到它:

```py
$ psql -U postgres -d psycopgtest
psql (11.2, server 10.5)
Type "help" for help.
```

现在，您以用户`postgres`的身份连接到数据库`psycopgtest`。该用户也是数据库所有者，因此您将拥有数据库中每个表的读取权限。

### 创建包含数据的表格

接下来，您需要创建一个包含一些用户信息的表，并向其中添加数据:

```py
psycopgtest=#  CREATE  TABLE  users  ( username  varchar(30), admin  boolean ); CREATE TABLE

psycopgtest=#  INSERT  INTO  users (username,  admin) VALUES ('ran',  true), ('haki',  false); INSERT 0 2

psycopgtest=#  SELECT  *  FROM  users; username | admin
----------+-------
 ran      | t
 haki     | f
(2 rows)
```

该表有两列:`username`和`admin`。`admin`列表示用户是否拥有管理权限。你的目标是针对`admin`领域，并试图滥用它。

### 设置 Python 虚拟环境

现在您已经有了一个数据库，是时候设置您的 Python 环境了。关于如何做到这一点的分步说明，请查看 [Python 虚拟环境:初级教程](https://realpython.com/python-virtual-environments-a-primer/)。

在新目录中创建虚拟环境:

```py
(~/src) $ mkdir psycopgtest
(~/src) $ cd psycopgtest
(~/src/psycopgtest) $ python3 -m venv venv
```

运行该命令后，将创建一个名为`venv`的新目录。该目录将存储您在虚拟环境中安装的所有软件包。

### 连接到数据库

要连接到 Python 中的数据库，您需要一个**数据库适配器**。大多数数据库适配器遵循 Python 数据库 API 规范的 2.0 版本 [PEP 249](https://www.python.org/dev/peps/pep-0249/) 。每个主要的数据库引擎都有一个领先的适配器:

| 数据库ˌ资料库 | 适配器 |
| --- | --- |
| 一种数据库系统 | [心理战](http://initd.org/psycopg/) |
| SQLite | [sqlite3](https://docs.python.org/3.7/library/sqlite3.html) |
| 神谕 | [cx_oracle](https://oracle.github.io/python-cx_Oracle/) |
| 关系型数据库 | [MySQLdb](https://mysqlclient.readthedocs.io/) |

要连接到 PostgreSQL 数据库，您需要安装 [Psycopg](http://initd.org/psycopg/) ，这是 Python 中最流行的 PostgreSQL 适配器。 [Django ORM](https://docs.djangoproject.com/en/2.2/ref/databases/#postgresql-notes) 默认使用它， [SQLAlchemy](https://docs.sqlalchemy.org/en/13/dialects/postgresql.html) 也支持它。

在您的终端中，激活虚拟环境，使用 [`pip`](https://realpython.com/what-is-pip/) 安装`psycopg`:

```py
(~/src/psycopgtest) $ source venv/bin/activate
(~/src/psycopgtest) $ python -m pip install psycopg2>=2.8.0
Collecting psycopg2
 Using cached https://....
 psycopg2-2.8.2.tar.gz
Installing collected packages: psycopg2
 Running setup.py install for psycopg2 ... done
Successfully installed psycopg2-2.8.2
```

现在，您已经准备好创建到数据库的连接了。以下是 Python 脚本的开头:

```py
import psycopg2

connection = psycopg2.connect(
    host="localhost",
    database="psycopgtest",
    user="postgres",
    password=None,
)
connection.set_session(autocommit=True)
```

您使用了`psycopg2.connect()`来创建连接。该函数接受以下参数:

*   **`host`** 是你的数据库所在服务器的 [IP 地址](https://realpython.com/python-ipaddress-module/)或者 DNS。在这种情况下，主机是您的本地机器，或`localhost`。

*   **`database`** 是要连接的数据库的名称。您想要连接到您之前创建的数据库，`psycopgtest`。

*   **`user`** 是对数据库有权限的用户。在这种情况下，您希望作为所有者连接到数据库，因此您传递了用户`postgres`。

*   **`password`** 是您在`user`中指定的任何人的密码。在大多数开发环境中，用户无需密码就可以连接到本地数据库。

建立连接后，您用`autocommit=True`配置了会话。激活`autocommit`意味着您不必通过发出`commit`或`rollback`来手动管理交易。这是大多数 ORM 的[默认](https://docs.djangoproject.com/en/2.2/topics/db/transactions/#autocommit-details) [行为](https://docs.sqlalchemy.org/en/13/core/connections.html#understanding-autocommit)。这里也使用这种行为，这样您就可以专注于编写 SQL 查询，而不是管理事务。

**注意:** Django 用户可以从 [`django.db.connection`](https://docs.djangoproject.com/en/2.2/topics/db/sql/#executing-custom-sql-directly) 获取 ORM 使用的连接实例:

```py
from django.db import connection
```

[*Remove ads*](/account/join/)

### 执行查询

现在您已经连接到数据库，可以执行查询了:

>>>

```py
>>> with connection.cursor() as cursor:
...     cursor.execute('SELECT COUNT(*) FROM users')
...     result = cursor.fetchone()
... print(result)
(2,)
```

您使用了`connection`对象来创建一个`cursor`。就像 Python 中的文件一样，`cursor`被实现为上下文管理器。当您创建上下文时，会打开一个`cursor`用于向数据库发送命令。当上下文退出时，`cursor`关闭，您不能再使用它。

**注意:**要了解关于上下文管理器的更多信息，请查看 [Python 上下文管理器和“with”语句](https://realpython.com/courses/python-context-managers-and-with-statement/)。

在上下文中，您使用了`cursor`来执行查询并获取结果。在这种情况下，您发出一个查询来计算`users`表中的行数。为了从查询中获取结果，您执行了`cursor.fetchone()`并收到了一个元组。因为查询只能返回一个结果，所以您使用了`fetchone()`。如果查询要返回多个结果，那么您需要迭代`cursor`或者使用其他 [`fetch*`](https://www.python.org/dev/peps/pep-0249/#fetchone) 方法之一。

## 在 SQL 中使用查询参数

在上一节中，您创建了一个数据库，建立了到它的连接，并执行了一个查询。您使用的查询是**静态**。换句话说，它的**没有参数**。现在，您将开始在查询中使用参数。

首先，您将实现一个检查用户是否是管理员的函数。`is_admin()`接受用户名并返回该用户的管理员状态:

```py
# BAD EXAMPLE. DON'T DO THIS!
def is_admin(username: str) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("""
 SELECT
 admin
 FROM
 users
 WHERE
 username = '%s'
 """ % username)
        result = cursor.fetchone()
    admin, = result
    return admin
```

这个函数执行一个查询来获取给定用户名的`admin`列的值。您使用了`fetchone()`来返回一个只有一个结果的元组。然后，你将这个元组解包到[变量](https://realpython.com/python-variables/) `admin`中。要测试您的功能，请检查一些用户名:

>>>

```py
>>> is_admin('haki')
False
>>> is_admin('ran')
True
```

到目前为止一切顺利。该函数返回两个用户的预期结果。但是不存在的用户怎么办？看看这个 [Python 回溯](https://realpython.com/python-traceback/):

>>>

```py
>>> is_admin('foo')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 12, in is_admin
TypeError: cannot unpack non-iterable NoneType object
```

当用户不存在时，产生一个`TypeError`。这是因为`.fetchone()`在没有找到结果时返回 [`None`](https://realpython.com/null-in-python/) ，而解包`None`会引发一个`TypeError`。唯一可以解包元组的地方是从`result`填充`admin`的地方。

为了处理不存在的用户，当`result`为`None`时创建一个特例:

```py
# BAD EXAMPLE. DON'T DO THIS!
def is_admin(username: str) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("""
 SELECT
 admin
 FROM
 users
 WHERE
 username = '%s'
 """ % username)
        result = cursor.fetchone()

 if result is None: # User does not exist return False 
    admin, = result
    return admin
```

这里，您添加了一个处理`None`的特例。如果`username`不存在，那么函数应该返回`False`。再次在一些用户身上测试该功能:

>>>

```py
>>> is_admin('haki')
False
>>> is_admin('ran')
True
>>> is_admin('foo')
False
```

太好了！该功能现在也可以处理不存在的用户名。

[*Remove ads*](/account/join/)

## 利用 Python SQL 注入开发查询参数

在前面的例子中，您使用了[字符串插值](https://realpython.com/python-strings/#interpolating-variables-into-a-string)来生成一个查询。然后，执行查询并将结果字符串直接发送到数据库。然而，在这个过程中你可能忽略了一些东西。

回想一下您传递给`is_admin()`的`username`参数。这个变量到底代表什么？你可能会认为`username`只是一个表示真实用户名的字符串。但是，正如您将要看到的，入侵者可以很容易地利用这种疏忽，通过执行 Python SQL 注入造成重大伤害。

尝试检查以下用户是否是管理员:

>>>

```py
>>> is_admin("'; select true; --")
True
```

等等…刚刚发生了什么？

让我们再看一下实现。打印出数据库中正在执行的实际查询:

>>>

```py
>>> print("select admin from users where username = '%s'" % "'; select true; --")
select admin from users where username = ''; select true; --'
```

结果文本包含三个语句。为了准确理解 Python SQL 注入的工作原理，您需要单独检查每个部分。第一个声明如下:

```py
select  admin  from  users  where  username  =  '';
```

这是您想要的查询。分号(`;`)终止查询，因此这个查询的结果无关紧要。接下来是第二条语句:

```py
select  true;
```

这份声明是入侵者编造的。它被设计为总是返回`True`。

最后，您会看到这段简短的代码:

```py
--'
```

这个片段消除了它后面的所有内容。入侵者添加了注释符号(`--`)，将您可能放在最后一个占位符之后的所有内容都变成了注释。

当你用这个参数执行函数时，*将总是返回`True`* 。例如，如果您在登录页面中使用这个函数，入侵者可以使用用户名`'; select true; --`登录，他们将被授予访问权限。

如果你认为这很糟糕，它可能会变得更糟！了解您的表结构的入侵者可以使用 Python SQL 注入来造成永久性破坏。例如，入侵者可以注入更新语句来改变数据库中的信息:

>>>

```py
>>> is_admin('haki')
False
>>> is_admin("'; update users set admin = 'true' where username = 'haki'; select true; --")
True
>>> is_admin('haki')
True
```

让我们再分解一下:

```py
';
```

这个代码片段终止了查询，就像前面的注入一样。下一条语句如下:

```py
update  users  set  admin  =  'true'  where  username  =  'haki';
```

该部分为用户`haki`将`admin`更新为`true`。

最后，有这样一段代码:

```py
select  true;  --
```

和前面的例子一样，这段代码返回`true`并注释掉它后面的所有内容。

为什么会更糟？好吧，如果入侵者设法用这个输入执行函数，那么用户`haki`将成为管理员:

```py
psycopgtest=#  select  *  from  users; username | admin
----------+-------
 ran      | t
 haki     | t (2 rows)
```

入侵者不再需要使用黑客技术。他们可以用用户名`haki`登录。(如果入侵者*真的*想要造成伤害，那么他们甚至可以发出`DROP DATABASE`命令。)

在您忘记之前，将`haki`恢复到其原始状态:

```py
psycopgtest=#  update  users  set  admin  =  false  where  username  =  'haki'; UPDATE 1
```

那么，为什么会这样呢？嗯，你对`username`论点了解多少？您知道它应该是一个表示用户名的字符串，但是您实际上并不检查或强制执行这个断言。这可能很危险！这正是攻击者试图入侵您的系统时所寻找的。

[*Remove ads*](/account/join/)

### 精心制作安全查询参数

在上一节中，您看到了入侵者如何通过使用精心编制的字符串来利用您的系统并获得管理员权限。问题是您允许从客户端传递的值直接执行到数据库，而不执行任何检查或验证。 [SQL 注入](https://www.owasp.org/index.php/SQL_Injection)依赖于这种类型的漏洞。

在数据库查询中使用用户输入的任何时候，SQL 注入都可能存在漏洞。防止 Python SQL 注入的关键是确保该值按照开发人员的意图使用。在前面的例子中，您打算将`username`用作一个字符串。实际上，它被用作原始 SQL 语句。

为了确保值按预期使用，您需要对值进行转义。例如，为了防止入侵者在字符串参数的位置注入原始 SQL，可以对引号进行转义:

>>>

```py
>>> # BAD EXAMPLE. DON'T DO THIS!
>>> username = username.replace("'", "''")
```

这只是一个例子。在尝试防止 Python SQL 注入时，有许多特殊字符和场景需要考虑。幸运的是，现代数据库适配器带有内置工具，通过使用**查询参数**来防止 Python SQL 注入。这些用于代替普通的字符串插值，以构成带有参数的查询。

**注意:**不同的适配器、数据库、编程语言对查询参数的称呼不同。俗称有**绑定变量**、**替换变量**、**替代变量**。

现在您对漏洞有了更好的理解，可以使用查询参数而不是字符串插值来重写函数了:

```py
 1def is_admin(username: str) -> bool:
 2    with connection.cursor() as cursor:
 3        cursor.execute("""
 4 SELECT
 5 admin
 6 FROM
 7 users
 8 WHERE
 9 username = %(username)s  10 """, {
11            'username': username 12        })
13        result = cursor.fetchone()
14
15    if result is None:
16        # User does not exist
17        return False
18
19    admin, = result
20    return admin
```

以下是本例中的不同之处:

*   **在第 9 行，**中，您使用了一个命名参数`username`来指示用户名应该放在哪里。注意参数`username`不再被单引号包围。

*   **在第 11 行，**你将`username`的值作为第二个参数传递给了`cursor.execute()`。在数据库中执行查询时，连接将使用`username`的类型和值。

要测试这个函数，请尝试一些有效和无效的值，包括前面的危险字符串:

>>>

```py
>>> is_admin('haki')
False
>>> is_admin('ran')
True
>>> is_admin('foo')
False
>>> is_admin("'; select true; --")
False
```

太神奇了！该函数返回所有值的预期结果。更有甚者，危险的弦不再起作用。要了解原因，您可以查看由`execute()`生成的查询:

>>>

```py
>>> with connection.cursor() as cursor:
...    cursor.execute("""
...        SELECT
...            admin
...        FROM
...            users
...        WHERE
... username = %(username)s ...    """, {
...        'username': "'; select true; --"
...    })
...    print(cursor.query.decode('utf-8'))
SELECT
 admin
FROM
 users
WHERE
 username = '''; select true; --'
```

该连接将`username`的值视为一个字符串，并对任何可能终止该字符串并引入 Python SQL 注入的字符进行了转义。

### 传递安全查询参数

数据库适配器通常提供几种传递查询参数的方法。命名占位符通常是可读性最好的，但是一些实现可能会受益于使用其他选项。

让我们快速看一下使用查询参数的一些正确和错误的方法。下面的代码块显示了您希望避免的查询类型:

```py
# BAD EXAMPLES. DON'T DO THIS!
cursor.execute("SELECT admin FROM users WHERE username = '" + username + '");
cursor.execute("SELECT admin FROM users WHERE username = '%s' % username);
cursor.execute("SELECT admin FROM users WHERE username = '{}'".format(username));
cursor.execute(f"SELECT admin FROM users WHERE username = '{username}'");
```

这些语句中的每一条都将`username`从客户端直接传递到数据库，而不执行任何检查或验证。这种代码已经成熟，可以邀请 Python SQL 注入了。

相比之下，执行这些类型的查询应该是安全的:

```py
# SAFE EXAMPLES. DO THIS!
cursor.execute("SELECT admin FROM users WHERE username = %s'", (username, ));
cursor.execute("SELECT admin FROM users WHERE username = %(username)s", {'username': username});
```

在这些语句中，`username`作为命名参数传递。现在，当执行查询时，数据库将使用指定的类型和值`username`，以防止 Python SQL 注入。

[*Remove ads*](/account/join/)

## 使用 SQL 组合

到目前为止，您已经为文字使用了参数。**文字**是数字、字符串和日期等数值。但是，如果您有一个用例需要编写一个不同的查询——其中的参数是其他的东西，比如表或列名，该怎么办呢？

受前面示例的启发，让我们实现一个接受表名并返回该表中行数的函数:

```py
# BAD EXAMPLE. DON'T DO THIS!
def count_rows(table_name: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute("""
 SELECT
 count(*)
 FROM
  %(table_name)s """, {
            'table_name': table_name,
        })
        result = cursor.fetchone()

    rowcount, = result
    return rowcount
```

尝试在用户表上执行该功能:

>>>

```py
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 9, in count_rows
psycopg2.errors.SyntaxError: syntax error at or near "'users'"
LINE 5:                 'users'
 ^
```

该命令无法生成 SQL。正如您已经看到的，数据库适配器将变量视为字符串或文字。然而，表名不是普通的字符串。这就是 [SQL 组合](http://initd.org/psycopg/docs/sql.html#module-psycopg2.sql)的用武之地。

你已经知道使用字符串插值来构造 SQL 是不安全的。幸运的是，Psycopg 提供了一个名为 [`psycopg.sql`](http://initd.org/psycopg/docs/sql.html#module-psycopg2.sql) 的模块来帮助您安全地编写 SQL 查询。让我们用 [`psycopg.sql.SQL()`](http://initd.org/psycopg/docs/sql.html#psycopg2.sql.SQL) 重写函数:

```py
from psycopg2 import sql

def count_rows(table_name: str) -> int:
    with connection.cursor() as cursor:
        stmt = sql.SQL("""
 SELECT
 count(*)
 FROM
  {table_name} """).format(
            table_name = sql.Identifier(table_name),
        )
        cursor.execute(stmt)
        result = cursor.fetchone()

    rowcount, = result
    return rowcount
```

在这个实现中有两个不同之处。首先，您使用了`sql.SQL()`来编写查询。然后，您使用`sql.Identifier()`来注释参数值`table_name`。(一个**标识符**是一个列或表名。)

**注意:**流行包 [`django-debug-toolbar`](https://django-debug-toolbar.readthedocs.io/en/latest/) 的用户可能会在 SQL 面板中得到一个用`psycopg.sql.SQL()`编写的查询的错误。预计将在[版本 2.0](https://github.com/jazzband/django-debug-toolbar/blob/master/docs/changes.rst#20a1-2019-05-16) 中发布一个修复程序。

现在，尝试执行`users`表上的函数:

>>>

```py
>>> count_rows('users')
2
```

太好了！接下来，让我们看看当表不存在时会发生什么:

>>>

```py
>>> count_rows('foo')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 11, in count_rows
psycopg2.errors.UndefinedTable: relation "foo" does not exist
LINE 5:                 "foo"
 ^
```

该函数抛出`UndefinedTable`异常。在接下来的步骤中，您将使用该异常来表明您的函数不会受到 Python SQL 注入攻击。

**注:**异常`UndefinedTable`是在[psycopg 2 2.8 版](http://initd.org/psycopg/docs/errors.html)中增加的。如果您使用的是 Psycopg 的早期版本，那么您会得到一个不同的异常。

要将所有这些放在一起，可以添加一个选项来计算表中的行数，直到达到一定的限制。这个特性对于非常大的表可能很有用。要实现这一点，需要在查询中添加一个`LIMIT`子句，以及限制值的查询参数:

```py
from psycopg2 import sql

def count_rows(table_name: str, limit: int) -> int:
    with connection.cursor() as cursor:
        stmt = sql.SQL("""
 SELECT
 COUNT(*)
 FROM (
 SELECT
 1
 FROM
  {table_name} LIMIT {limit}  ) AS limit_query
 """).format(
            table_name = sql.Identifier(table_name),
 limit = sql.Literal(limit),        )
        cursor.execute(stmt)
        result = cursor.fetchone()

    rowcount, = result
    return rowcount
```

在这个代码块中，您使用`sql.Literal()`对`limit`进行了注释。和前面的例子一样，在使用简单方法时，`psycopg`会将所有查询参数绑定为文字。然而，当使用`sql.SQL()`时，您需要使用`sql.Identifier()`或`sql.Literal()`显式地注释每个参数。

**注意:**不幸的是，Python API 规范没有解决标识符的绑定，只解决了文字。Psycopg 是唯一一个流行的适配器，它增加了用文字和标识符安全组合 SQL 的能力。这个事实使得在绑定标识符时更加需要注意。

执行该功能以确保其正常工作:

>>>

```py
>>> count_rows('users', 1)
1
>>> count_rows('users', 10)
2
```

既然您已经看到该函数正在工作，请确保它也是安全的:

>>>

```py
>>> count_rows("(select 1) as foo; update users set admin = true where name = 'haki'; --", 1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 18, in count_rows
psycopg2.errors.UndefinedTable: relation "(select 1) as foo; update users set admin = true where name = '" does not exist
LINE 8:                     "(select 1) as foo; update users set adm...
 ^
```

这个回溯表明`psycopg`对该值进行了转义，数据库将其视为表名。因为这个名称的表不存在，所以出现了一个`UndefinedTable`异常，您没有被攻击！

[*Remove ads*](/account/join/)

## 结论

您已经成功实现了一个组成动态 SQL *的函数，而没有*让您的系统面临 Python SQL 注入的风险！您在查询中使用了文字和标识符，而没有损害安全性。

**你已经学会:**

*   什么是 **Python SQL 注入**以及如何利用它
*   如何**防止 Python SQL 注入**使用查询参数
*   如何**安全地编写使用文字和标识符作为参数的 SQL 语句**

您现在能够创建能够抵御外部攻击的程序。前进，挫败黑客！******