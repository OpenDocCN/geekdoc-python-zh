# PyWin32: adodbapi 和 MS Access

> 原文：<https://www.blog.pythonlibrary.org/2011/02/01/pywin32-adodbapi-and-ms-access/>

上周， [PyWin32 邮件列表](http://mail.python.org/pipermail/python-win32/2011-January/011086.html)上有一个有趣的帖子，关于如何在没有实际安装 Access 的情况下用 Python 读取 Microsoft Access 数据库。Vernon Cole 有解决方案，但我注意到 Google 似乎没有很好地索引 PyWin32 列表，所以我决定在这里写一下。

我把他的代码稍微修改了一下，使其更加清晰，并用 Microsoft Access XP(可从下面下载)创建了一个蹩脚的数据库文件。 [adodbapi](http://adodbapi.sourceforge.net/) 模块(不要与 adodb 模块混淆)的源代码发行版还在其“test”文件夹中包含一个测试数据库，您也可以使用它。总之，代码如下:

```py

import adodbapi

database = "db1.mdb"
constr = 'Provider=Microsoft.Jet.OLEDB.4.0; Data Source=%s'  % database
tablename = "address"

# connect to the database
conn = adodbapi.connect(constr)

# create a cursor
cur = conn.cursor()

# extract all the data
sql = "select * from %s" % tablename
cur.execute(sql)

# show the result
result = cur.fetchall()
for item in result:
    print item

# close the cursor and connection
cur.close()
conn.close()

```

此代码已在以下方面进行了测试:

*   安装了 Python 2.5.4 和 adodbapi 2.4.0 并安装了 Microsoft Access 的 Windows XP Professional
*   带 Python 2.6.4、adodbapi 2.2.6 的 Windows 7 家庭高级版(32 位),不带 Microsoft Access

## 下载

*   [adodbapi_access.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2011/02/adodbapi_access.zip)