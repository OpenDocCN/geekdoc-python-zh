# Python 和 MongoDB:连接到 NoSQL 数据库

> 原文：<https://realpython.com/introduction-to-mongodb-and-python/>

[MongoDB](https://www.mongodb.com/what-is-mongodb) 是一个[面向文档](https://en.wikipedia.org/wiki/Document-oriented_database)和 [NoSQL](https://en.wikipedia.org/wiki/NoSQL) 的数据库解决方案，它提供了强大的可伸缩性和灵活性以及强大的查询系统。使用 MongoDB 和 Python，您可以快速开发许多不同类型的数据库应用程序。因此，如果您的 Python 应用程序需要一个像语言本身一样灵活的数据库，那么 MongoDB 就是您的选择。

在本教程中，您将学习:

*   什么是 MongoDB
*   如何**安装并运行** MongoDB
*   如何使用 **MongoDB 数据库**
*   如何使用底层 **PyMongo 驱动**与 MongoDB 接口
*   如何使用高级的 **MongoEngine 对象-文档映射器(ODM)**

在本教程中，您将编写几个例子来展示 MongoDB 的灵活性和强大功能以及它对 Python 的强大支持。要下载这些示例的源代码，请单击下面的链接:

**获取源代码:** [单击此处获取源代码，您将在本教程中使用](https://realpython.com/bonus/python-mongodb-code/)来了解如何将 MongoDB 与 Python 结合使用。

## 使用 SQL 与 NoSQL 数据库

几十年来， [SQL](https://en.wikipedia.org/wiki/SQL) 数据库是开发人员构建大型可伸缩数据库系统的唯一选择之一。然而，日益增长的存储复杂数据结构的需求导致了 **NoSQL** 数据库的诞生。这种新型的数据库系统允许开发人员高效地存储异构和无结构的数据。

一般来说，NoSQL 数据库系统存储和检索数据的方式与 SQL [关系数据库管理系统](https://en.wikipedia.org/wiki/Relational_database#RDBMS)(RDBMS)大不相同。

在选择当前可用的数据库技术时，您可能需要在使用 SQL 还是 NoSQL 系统之间做出选择。这两者都有特定的特性，您在选择其中一个时应该加以考虑。以下是它们的一些更实质性的区别:

| 财产 | SQL 数据库 | NoSQL 数据库 |
| --- | --- | --- |
| 数据模型 | 有关系的 | 非亲属 |
| 结构 | 基于表格，包含列和行 | 基于文档、键值对、图形或宽列 |
| (计划或理论的)纲要 | 一种预定义的严格模式，其中每个记录(行)都具有相同的性质和属性 | 动态模式或无模式，这意味着记录不需要具有相同的性质 |
| 查询语言 | 结构化查询语言(SQL) | 因数据库而异 |
| 可量测性 | 垂直的 | 水平的 |
| [酸](https://en.wikipedia.org/wiki/ACID)交易 | 支持 | 受支持，具体取决于特定的 NoSQL 数据库 |
| 添加新属性的能力 | 需要首先改变模式 | 可能不干扰任何东西 |

这两种类型的数据库之间还有许多其他的区别，但是上面提到的是一些需要了解的更重要的区别。

选择数据库时，您应该仔细考虑它的优点和缺点。您还需要考虑数据库如何适应您的特定场景和应用程序的需求。有时，正确的解决方案是使用 SQL 和 NoSQL 数据库的组合来处理更大系统的不同方面。

SQL 数据库的一些常见示例包括:

*   [SQLite](https://www.sqlite.org/docs.html)
*   [MySQL](https://www.xplenty.com/integrations/mysql/) 的实现
*   [甲骨文](https://www.xplenty.com/integrations/oracle/)
*   [PostgreSQL](https://www.xplenty.com/integrations/postgresql/)
*   [微软 SQL 服务器](https://www.xplenty.com/integrations/microsoft-sql-server/)

NoSQL 数据库的例子包括:

*   [DynamoDB](https://aws.amazon.com/dynamodb/)
*   卡桑德拉
*   [再说一遍](https://realpython.com/python-redis/)
*   [CouchDB](https://couchdb.apache.org/#about)
*   [重新思考 DB](https://rethinkdb.com/faq)
*   [RavenDB](https://ravendb.net/about)
*   [MongoDB](https://www.xplenty.com/integrations/mongodb/)

近年来，SQL 和 NoSQL 数据库甚至开始合并。例如，数据库系统，如 [PostgreSQL](https://www.postgresql.org/docs/9.2/datatype-json.html) 、 [MySQL](https://dev.mysql.com/doc/refman/5.7/en/json.html) 和[微软 SQL Server](https://docs.microsoft.com/en-us/sql/relational-databases/json/json-data-sql-server?redirectedfrom=MSDN&view=sql-server-ver15) 现在支持存储和查询 [JSON](https://realpython.com/python-json/) 数据，很像 NoSQL 数据库。有了这个，你现在可以用这两种技术获得许多相同的结果。但你仍然没有得到 NoSQL 的许多功能，如水平缩放和用户友好的界面。

有了这个关于 SQL 和 NoSQL 数据库的简短背景，您就可以专注于本教程的主要主题了:MongoDB 数据库以及如何在 Python 中使用它。

[*Remove ads*](/account/join/)

## 用 MongoDB 管理 NoSQL 数据库

MongoDB 是一个面向文档的数据库，被归类为 NoSQL。近年来，它在整个行业变得非常流行，并且与 Python 集成得非常好。与传统的 SQL RDBMSs 不同，MongoDB 使用**文档**的**集合**，而不是**行**的**表**来组织和存储数据。

MongoDB 将数据存储在无模式和灵活的类似 JSON 的文档中。这里，**无模式**意味着您可以在同一个集合中拥有一组不同的[字段](https://en.wikipedia.org/wiki/Field_(computer_science))的文档，而不需要满足严格的表**模式**。

随着时间的推移，您可以改变文档和数据的结构，从而形成一个灵活的系统，使您能够快速适应需求的变化，而不需要复杂的[数据迁移](https://en.wikipedia.org/wiki/Data_migration)过程。然而，改变新文档结构的代价是现有文档变得与更新的模式不一致。所以这是一个需要用心经营的话题。

**注:** [JSON](https://en.wikipedia.org/wiki/JSON) 代表 **JavaScript 对象符号**。它是一种文件格式，具有人类可读的结构，由可以嵌套任意深度的键值对组成。

MongoDB 是用 [C++](https://realpython.com/python-vs-cpp/) 编写的，由 [MongoDB Inc.](https://www.mongodb.com/company) 的[积极开发](https://github.com/mongodb/mongo)，它运行在所有主要平台上，比如 macOS、Windows、Solaris 和大多数 Linux 发行版。一般来说，MongoDB 数据库背后有三个主要的开发目标:

1.  扩展良好
2.  存储丰富的数据结构
3.  提供复杂的查询机制

MongoDB 是一个**分布式**数据库，因此系统内置了高可用性、水平伸缩和地理分布。它将数据存储在灵活的类似 JSON 的文档中。您可以对这些文档进行建模，以映射应用程序中的对象，这使得有效地处理数据成为可能。

MongoDB 提供了强大的[查询语言](https://docs.mongodb.com/manual/introduction/#rich-query-language)，支持特殊查询、[索引](https://docs.mongodb.com/manual/indexes/)、[聚合](https://docs.mongodb.com/manual/aggregation/)、[地理空间搜索](https://docs.mongodb.com/manual/tutorial/geospatial-tutorial/)、[文本搜索](https://docs.mongodb.com/manual/text-search/)等等。这为您提供了一个强大的工具包来访问和处理您的数据。最后，MongoDB 是免费的，并且有很好的 Python 支持。

### 回顾 MongoDB 的特性

到目前为止，您已经了解了 MongoDB 是什么以及它的主要目标是什么。在这一节中，您将了解 MongoDB 的一些更重要的特性。至于数据库管理方面，MongoDB 提供了以下特性:

*   **查询支持:**可以使用很多标准的查询类型，比如匹配(`==`)、比较(`<`、`>`)、[正则表达式](https://realpython.com/regex-python/)。
*   **数据容纳:**您几乎可以存储任何类型的数据，无论是结构化的、部分结构化的，甚至是多态的。
*   **可扩展性:**只需向服务器集群添加更多的机器，就可以处理更多的查询。
*   **灵活性和敏捷性:**您可以使用它快速开发应用程序。
*   **文档方向和无模式:**您可以在一个文档中存储关于一个数据模型的所有信息。
*   **可调整的模式:**您可以动态地更改数据库的模式，这减少了提供新功能或修复现有问题所需的时间。
*   **关系数据库功能:**您可以执行关系数据库常见的操作，比如索引。

至于操作方面，MongoDB 提供了一些在其他数据库系统中找不到的工具和特性:

*   **可伸缩性:**无论您需要独立的服务器还是完整的独立服务器集群，您都可以将 MongoDB 扩展到您需要的任何规模。
*   **负载平衡支持:** MongoDB 将自动在不同的分片之间移动数据。
*   **自动故障转移支持:**如果您的主服务器出现故障，新的主服务器将自动启动并运行。
*   **管理工具:**您可以使用基于云的 MongoDB 管理服务(MMS)来跟踪您的机器。
*   **内存效率:**由于内存映射文件，MongoDB 通常比关系数据库更高效。

所有这些功能都非常有用。例如，如果您利用索引功能，那么您的大部分数据将保存在内存中，以便快速检索。即使没有索引特定的文档键，MongoDB 也会使用最近最少使用的技术缓存大量数据。

### 安装和运行 MongoDB

现在您已经熟悉了 MongoDB，是时候动手使用它了。但是首先，你需要在你的机器上安装它。MongoDB 的官方网站提供了两个版本的数据库服务器:

1.  [community edition](https://docs.mongodb.com/manual/installation/#mongodb-community-edition-installation-tutorials)提供了灵活的文档模型以及即席查询、索引和实时聚合，为访问和分析您的数据提供了强大的方法。这个版本可以免费获得。
2.  [企业版](https://docs.mongodb.com/manual/installation/#mongodb-enterprise-edition-installation-tutorials)提供与社区版相同的功能，以及其他与安全和监控相关的高级功能。这是商业版，但是您可以无限期地免费使用它进行评估和开发。

如果你用的是 Windows，那么你可以通读[安装教程](https://docs.mongodb.com/manual/tutorial/install-mongodb-enterprise-on-windows/)来获得完整的说明。一般来说，你可以进入[下载页面](https://www.mongodb.com/try/download/enterprise)，在可用下载框中选择 Windows 平台，选择适合你当前系统的`.msi`安装程序，点击*下载*。

运行安装程序，并按照安装向导屏幕上的说明进行操作。该页面还提供了关于如何将 MongoDB 作为 Windows 服务运行的信息。

如果你在 macOS 上，那么你可以使用 [Homebrew](https://brew.sh/) 在你的系统上安装 MongoDB。参见[安装教程](https://docs.mongodb.com/manual/tutorial/install-mongodb-enterprise-on-os-x/)获取完整指南。此外，请确保按照指示[将 MongoDB 作为 macOS 服务运行](https://docs.mongodb.com/manual/tutorial/install-mongodb-enterprise-on-ubuntu/#run-mongodb-enterprise-edition)。

如果您使用的是 Linux，那么安装过程将取决于您的特定发行版。关于如何在不同的 Linux 系统上安装 MongoDB 的详细指南，请转到[安装教程页面](https://docs.mongodb.com/manual/installation/#mongodb-enterprise-edition-installation-tutorials)并选择与您当前操作系统相匹配的教程。确保在安装结束时运行 MongoDB 守护进程`mongod`。

最后，还可以使用 [Docker](https://docs.docker.com/) 安装 MongoDB。如果您不想让另一个安装把您的系统搞得一团糟，这是很方便的。如果你更喜欢这个安装选项，那么你可以通读[官方教程](https://docs.mongodb.com/manual/tutorial/install-mongodb-enterprise-with-docker/)并按照它的指示操作。请注意，在这种情况下，需要事先了解[如何使用 Docker](https://realpython.com/tutorials/docker/) 。

在您的系统上安装并运行了 MongoDB 数据库之后，您就可以开始使用`mongo` shell 处理真正的数据库了。

[*Remove ads*](/account/join/)

## 使用`mongo` Shell 创建 MongoDB 数据库

如果您遵循了安装和运行说明，那么您应该已经有一个 MongoDB 实例在您的系统上运行了。现在，您可以开始创建和测试自己的数据库了。在本节中，您将学习如何使用 [`mongo`](https://docs.mongodb.com/manual/reference/program/mongo/#bin.mongo) shell 来创建、读取、更新和删除数据库中的文档。

### 运行`mongo`外壳

`mongo` shell 是 MongoDB 的一个交互式 [JavaScript](https://realpython.com/python-vs-javascript/) 接口。您可以使用该工具来查询和操作您的数据，以及执行管理操作。由于是 JavaScript 接口，所以不会用大家熟悉的 SQL 语言来查询数据库。相反，您将使用 JavaScript 代码。

要启动`mongo` shell，打开您的终端或命令行并运行以下命令:

```py
$ mongo
```

这个命令将带您进入`mongo` shell。此时，您可能会看到一堆消息，其中包含关于 shell 版本以及服务器地址和端口的信息。最后，您将看到 shell 提示符(`>`)来输入查询和命令。

您可以将数据库地址作为参数传递给`mongo`命令。您还可以使用几个选项，比如指定访问远程数据库的主机和端口，等等。关于如何使用`mongo`命令的更多细节，您可以运行`mongo --help`。

### 建立连接

当您不带参数运行`mongo`命令时，它会启动 shell 并连接到由`mongod://127.0.0.1:27017`的`mongod`进程提供的默认本地服务器。这意味着您通过端口`27017`连接到本地主机。

默认情况下，`mongo` shell 通过建立到`test`数据库的连接来启动会话。您可以通过`db`对象访问当前数据库:

```py
>  db test >
```

在这种情况下，`db`保存对默认数据库`test`的引用。要切换数据库，发出命令`use`，提供一个数据库名称作为参数。

例如，假设您想要创建一个网站来发布 Python 内容，并且您计划使用 MongoDB 来存储您的教程和文章。在这种情况下，您可以使用以下命令切换到站点的数据库:

```py
>  use  rptutorials switched  to  db  rptutorials
```

该命令将您的连接切换到`rptutorials`数据库。MongoDB 不会在文件系统上创建物理数据库文件，直到您将真实数据插入到数据库中。所以在这种情况下，`rptutorials`不会显示在您当前的数据库列表中:

```py
>  show  dbs admin  0.000GB config  0.000GB local  0.000GB >
```

shell 提供了许多特性和选项。它允许您查询和操作数据，还可以管理数据库服务器本身。

`mongo` shell 没有使用 SQL 之类的标准化查询语言，而是使用 JavaScript 编程语言和用户友好的 [API](https://en.wikipedia.org/wiki/API) 。这个 API 允许您处理数据，这是下一节的主题。

### 创建收藏和文档

MongoDB [数据库](https://docs.mongodb.com/manual/reference/glossary/#term-database)是[文档](https://docs.mongodb.com/manual/reference/glossary/#term-document)的[集合](https://docs.mongodb.com/manual/reference/glossary/#term-collection)的物理容器。每个数据库在文件系统上都有自己的文件集。这些文件由 MongoDB 服务器管理，它可以处理几个数据库。

在 MongoDB 中，**集合**是一组**文档**。集合有点类似于传统 RDBMS 中的表，但是没有强加严格的模式。理论上，集合中的每个文档可以有完全不同的结构或字段集。

实际上，集合中的文档通常共享相似的结构，以允许统一的检索、插入和更新过程。在更新和插入期间，您可以通过使用[文档验证规则](https://docs.mongodb.com/manual/core/schema-validation/)来实施统一的文档结构。

允许不同的文档结构是 MongoDB 集合的一个关键特性。这个特性提供了灵活性，允许向文档添加新字段，而无需修改正式的表模式。

要使用`mongo` shell 创建集合，您需要将`db`指向您的目标数据库，然后使用**点符号**创建集合:

```py
>  use  rptutorials switched  to  db  rptutorials >  db rptutorials >  db.tutorial rptutorials.tutorial
```

在这个例子中，您使用点符号创建`tutorial`作为当前数据库`rptutorials`中的集合。值得注意的是，MongoDB 创建数据库和集合**是很慢的**。换句话说，它们是在您插入第一个文档后才实际创建的。

一旦有了数据库和集合，就可以开始插入文档了。文档是 MongoDB 中的存储单位。在 RDBMS 中，这相当于一个表行。然而，MongoDB 的文档比行更加通用，因为它们可以存储复杂的信息，比如[数组](https://realpython.com/python-data-structures/#arrayarray-basic-typed-arrays)，嵌入式文档，甚至文档数组。

MongoDB 以一种叫做**二进制 JSON** ( [BSON](https://docs.mongodb.com/manual/reference/glossary/#term-bson) )的格式存储文档，这是 JSON 的二进制表示。MongoDB 的文档由字段-值对组成，结构如下:

```py
{
   field1 → value1,
   field2 → value2,
   field3 → value3,
   ...
   fieldN → valueN
}
```

字段的值可以是任何 BSON [数据类型](https://docs.mongodb.com/manual/reference/bson-types/)，包括其他文档、数组和文档数组。在实践中，您将使用 JSON 格式指定您的文档。

当您构建 MongoDB 数据库应用程序时，可能您最重要的决定是关于文档的结构。换句话说，您必须决定您的文档将具有哪些字段和值。

对于 Python 站点的教程，文档的结构可能如下:

```py
{ "title":  "Reading and Writing CSV Files in Python", "author":  "Jon", "contributors":  [ "Aldren", "Geir Arne", "Joanna", "Jason" ], "url":  "https://realpython.com/python-csv/" }
```

文档本质上是一组属性名及其值。这些值可以是简单的数据类型，如字符串和数字，但也可以是数组，如上面示例中的`contributors`。

MongoDB 面向文档的数据模型自然地将复杂数据表示为单个对象。这允许您从整体上处理数据对象，而不需要查看几个地方或表。

如果您使用传统的 RDBMS 来存储教程，那么您可能会有一个表来存储您的教程，另一个表来存储您的贡献者。然后，您必须在两个表之间建立一个关系，以便以后可以检索数据。

[*Remove ads*](/account/join/)

### 使用收藏和文档

到目前为止，您已经了解了如何运行和使用`mongo` shell 的基本知识。您还知道如何使用 JSON 格式创建自己的文档。现在是时候学习如何[将文档](https://docs.mongodb.com/manual/tutorial/insert-documents/)插入到 MongoDB 数据库中了。

要使用`mongo` shell 将文档插入数据库，首先需要选择一个集合，然后使用您的文档作为参数调用集合上的 [`.insertOne()`](https://docs.mongodb.com/manual/reference/method/db.collection.insertOne/#db.collection.insertOne) :

```py
>  use  rptutorials switched  to  db  rptutorials >  db.tutorial.insertOne({ ...  "title":  "Reading and Writing CSV Files in Python", ...  "author":  "Jon", ...  "contributors":  [ ...  "Aldren", ...  "Geir Arne", ...  "Joanna", ...  "Jason" ...  ], ...  "url":  "https://realpython.com/python-csv/" ...  }) { "acknowledged"  :  true, "insertedId"  :  ObjectId("600747355e6ea8d224f754ba") }
```

使用第一个命令，您可以切换到想要使用的数据库。第二个命令是一个 JavaScript 方法调用，它将一个简单的文档插入到选定的集合中，`tutorial`。一旦你点击 `Enter` ，你的屏幕上会出现一条信息，告知你新插入的文档及其`insertedId`。

就像关系数据库需要一个主键来惟一标识表中的每一行一样，MongoDB 文档需要一个`_id`字段来惟一标识文档。MongoDB 允许你输入一个自定义的`_id`，只要你保证它的唯一性。然而，一个被广泛接受的做法是允许 MongoDB 自动为您插入一个`_id`。

同样，您可以使用 [`.insertMany()`](https://docs.mongodb.com/manual/reference/method/db.collection.insertMany/#db.collection.insertMany) 一次添加多个文档:

```py
>  tutorial1  =  { ...  "title":  "How to Iterate Through a Dictionary in Python", ...  "author":  "Leodanis", ...  "contributors":  [ ...  "Aldren", ...  "Jim", ...  "Joanna" ...  ], ...  "url":  "https://realpython.com/iterate-through-dictionary-python/" ...  } >  tutorial2  =  { ...  "title":  "Python 3's f-Strings: An Improved String Formatting Syntax", ...  "author":  "Joanna", ...  "contributors":  [ ...  "Adriana", ...  "David", ...  "Dan", ...  "Jim", ...  "Pavel" ...  ], ...  "url":  "https://realpython.com/python-f-strings/" ...  } >  db.tutorial.insertMany([tutorial1,  tutorial2]) { "acknowledged"  :  true, "insertedIds"  :  [ ObjectId("60074ff05e6ea8d224f754bb"), ObjectId("60074ff05e6ea8d224f754bc") ] }
```

这里，对`.insertMany()`的调用获取一个教程列表，并将它们插入到数据库中。同样，shell 输出显示了关于新插入的文档及其自动添加的`_id`字段的信息。

`mongo` shell 还提供了对数据库执行[读取](https://docs.mongodb.com/manual/crud/#read-operations)、[更新](https://docs.mongodb.com/manual/crud/#update-operations)和[删除](https://docs.mongodb.com/manual/crud/#delete-operations)操作的方法。例如，您可以使用 [`.find()`](https://docs.mongodb.com/manual/reference/method/db.collection.find/#db.collection.find) 来检索集合中的文档:

```py
>  db.tutorial.find() {  "_id"  :  ObjectId("600747355e6ea8d224f754ba"), "title"  :  "Reading and Writing CSV Files in Python", "author"  :  "Jon", "contributors"  :  [  "Aldren",  "Geir Arne",  "Joanna",  "Jason"  ], "url"  :  "https://realpython.com/python-csv/"  } ... >  db.tutorial.find({author:  "Joanna"}) {  "_id"  :  ObjectId("60074ff05e6ea8d224f754bc"), "title"  :  "Python 3's f-Strings: An Improved String Formatting Syntax (Guide)", "author"  :  "Joanna", "contributors"  :  [  "Adriana",  "David",  "Dan",  "Jim",  "Pavel"  ], "url"  :  "https://realpython.com/python-f-strings/"  }
```

对`.find()`的第一次调用检索了`tutorial`集合中的所有文档。另一方面，第二次调用`.find()`检索由[乔安娜](https://realpython.com/team/jjablonski/)创作的教程。

有了关于如何通过其`mongo` shell 使用 MongoDB 的背景知识，您就可以开始在 Python 中使用 MongoDB 了。接下来的几节将带您了解在 Python 应用程序中使用 MongoDB 数据库的不同选项。

## 通过 Python 和 PyMongo 使用 MongoDB】

现在您已经知道了什么是 MongoDB 以及如何使用`mongo` shell 创建和管理数据库，您可以开始使用 MongoDB 了，但是这次是使用 Python。MongoDB 提供了一个官方的 [Python 驱动](https://github.com/mongodb/mongo-python-driver)叫做 [PyMongo](https://pymongo.readthedocs.io/en/stable/index.html) 。

在这一节中，您将通过一些示例了解如何使用 PyMongo 通过 MongoDB 和 Python 创建自己的数据库应用程序。

PyMongo 中的每个[模块负责数据库上的一组操作。你将拥有至少用于以下任务的](https://pymongo.readthedocs.io/en/stable/api/pymongo/index.html)[模块](https://realpython.com/python-modules-packages/):

*   建立[数据库连接](https://pymongo.readthedocs.io/en/stable/api/pymongo/index.html#module-pymongo)
*   使用[数据库](https://pymongo.readthedocs.io/en/stable/api/pymongo/database.html)
*   使用[集合和文档](https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html)
*   操纵[光标](https://pymongo.readthedocs.io/en/stable/api/pymongo/cursor.html)
*   处理数据[加密](https://pymongo.readthedocs.io/en/stable/api/pymongo/encryption.html)

一般来说，PyMongo 提供了一组丰富的工具，可以用来与 MongoDB 服务器进行通信。它提供了查询、检索结果、写入和删除数据以及运行数据库命令的功能。

### 安装 PyMongo

要开始使用 PyMongo，首先需要在 Python 环境中安装它。您可以使用一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)，或者您可以使用您的系统级 Python 安装，尽管第一个选项是首选。 [PyMongo](https://pypi.org/project/pymongo/) 在 [PyPI](https://pypi.org/) 上可用，所以最快的安装方式是用 [`pip`](https://realpython.com/what-is-pip/) 。启动您的终端并运行以下命令:

```py
$ pip install pymongo==3.11.2
```

在完成一些下载和其他相关步骤之后，这个命令会在您的 Python 环境中安装 PyMongo。请注意，如果您没有提供具体的版本号，那么`pip`将安装最新的可用版本。

**注意:**关于如何安装 PyMongo 的完整指南，请查看其官方文档的[安装/升级](https://pymongo.readthedocs.io/en/stable/installation.html)页面。

一旦完成安装，您就可以启动一个 [Python 交互会话](https://realpython.com/interacting-with-python/)并运行下面的[导入](https://realpython.com/python-import/):

>>>

```py
>>> import pymongo
```

如果它运行时没有在 Python shell 中引发异常，那么您的安装工作正常。如果没有，请再次仔细执行这些步骤。

[*Remove ads*](/account/join/)

### 建立连接

要建立到数据库的连接，需要创建一个 [`MongoClient`](https://pymongo.readthedocs.io/en/stable/api/pymongo/mongo_client.html#pymongo.mongo_client.MongoClient) 实例。这个类为 MongoDB 实例或服务器提供了一个客户机。每个客户机对象都有一个[内置连接池](https://pymongo.readthedocs.io/en/stable/faq.html#id4)，默认情况下，它处理多达 100 个到服务器的连接。

回到 Python 交互式会话，从`pymongo`导入`MongoClient`。然后创建一个客户机对象来与当前运行的 MongoDB 实例通信:

>>>

```py
>>> from pymongo import MongoClient
>>> client = MongoClient()
>>> client
MongoClient(host=['localhost:27017'], ..., connect=True)
```

上面的代码建立了到默认主机(`localhost`)和端口(`27017`)的连接。`MongoClient`接受一组参数，允许您指定自定义主机、端口和其他连接参数。例如，要提供自定义主机和端口，可以使用以下代码:

>>>

```py
>>> client = MongoClient(host="localhost", port=27017)
```

当您需要提供不同于 MongoDB 默认设置的`host`和`port`时，这很方便。你也可以使用 [MongoDB URI 格式](https://docs.mongodb.com/manual/reference/connection-string/):

>>>

```py
>>> client = MongoClient("mongodb://localhost:27017")
```

所有这些`MongoClient`实例都提供相同的客户端设置来连接您当前的 MongoDB 实例。您应该使用哪一个取决于您希望在代码中有多明确。

一旦实例化了`MongoClient`，就可以使用它的实例来引用特定的数据库连接，就像上一节中使用`mongo` shell 的`db`对象一样。

### 使用数据库、收藏和文档

一旦有了一个连接的`MongoClient`实例，就可以访问由指定的 MongoDB 服务器管理的任何数据库。要定义您想要使用哪个数据库，您可以像在`mongo` shell 中一样使用点符号:

>>>

```py
>>> db = client.rptutorials
>>> db
Database(MongoClient(host=['localhost:27017'], ..., connect=True), 'rptutorials')
```

在本例中，`rptutorials`是您将使用的数据库的名称。如果数据库不存在，那么 MongoDB 会为您创建它，但是只有在您对数据库执行第一个操作时。

如果数据库的名称不是有效的 Python 标识符，也可以使用[字典式访问](https://realpython.com/python-dicts/#accessing-dictionary-values):

>>>

```py
>>> db = client["rptutorials"]
```

当数据库的名称不是有效的 Python 标识符时，这个语句很方便。例如，如果您的数据库名为`rp-tutorials`，那么您需要使用字典式访问。

**注意:**当您使用`mongo` shell 时，您可以通过`db`全局对象访问数据库。当您使用 PyMongo 时，您可以将数据库分配给一个名为`db`的[变量](https://realpython.com/python-variables/)来获得类似的行为。

使用 PyMongo 在数据库中存储数据类似于在上面几节中使用`mongo` shell 所做的事情。但是首先，您需要创建您的文档。在 Python 中，您使用[字典](https://realpython.com/python-dicts/)来创建文档:

>>>

```py
>>> tutorial1 = {
...     "title": "Working With JSON Data in Python",
...     "author": "Lucas",
...     "contributors": [
...         "Aldren",
...         "Dan",
...         "Joanna"
...     ],
...     "url": "https://realpython.com/python-json/"
... }
```

一旦将文档创建为词典，就需要指定想要使用哪个集合。为此，您可以在数据库对象上使用点符号:

>>>

```py
>>> tutorial = db.tutorial
>>> tutorial
Collection(Database(..., connect=True), 'rptutorials'), 'tutorial')
```

在这种情况下，`tutorial`是 [`Collection`](https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection) 的一个实例，代表数据库中文档的物理集合。您可以通过调用`tutorial`上的 [`.insert_one()`](https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.insert_one) 将文档插入其中，并以文档作为参数:

>>>

```py
>>> result = tutorial.insert_one(tutorial1)
>>> result
<pymongo.results.InsertOneResult object at 0x7fa854f506c0>

>>> print(f"One tutorial: {result.inserted_id}")
One tutorial: 60084b7d87eb0fbf73dbf71d
```

这里，`.insert_one()`获取`tutorial1`，将其插入到`tutorial`集合中，并返回一个 [`InsertOneResult`](https://pymongo.readthedocs.io/en/stable/api/pymongo/results.html#pymongo.results.InsertOneResult) 对象。该对象对插入的文档提供反馈。注意，由于 MongoDB 动态生成了`ObjectId`，所以您的输出不会与上面显示的`ObjectId`匹配。

如果您有许多文档要添加到数据库中，那么您可以使用 [`.insert_many()`](https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.insert_many) 将它们一次插入:

>>>

```py
>>> tutorial2 = {
...     "title": "Python's Requests Library (Guide)",
...     "author": "Alex",
...     "contributors": [
...         "Aldren",
...         "Brad",
...         "Joanna"
...     ],
...     "url": "https://realpython.com/python-requests/"
... }

>>> tutorial3 = {
...     "title": "Object-Oriented Programming (OOP) in Python 3",
...     "author": "David",
...     "contributors": [
...         "Aldren",
...         "Joanna",
...         "Jacob"
...     ],
...     "url": "https://realpython.com/python3-object-oriented-programming/"
... }

>>> new_result = tutorial.insert_many([tutorial2, tutorial3])

>>> print(f"Multiple tutorials: {new_result.inserted_ids}")
Multiple tutorials: [
 ObjectId('6008511c87eb0fbf73dbf71e'),
 ObjectId('6008511c87eb0fbf73dbf71f')
]
```

这比多次调用`.insert_one()`更快更直接。对`.insert_many()`的调用获取一系列文档，并将它们插入到`rptutorials`数据库的`tutorial`集合中。该方法返回一个实例 [`InsertManyResult`](https://pymongo.readthedocs.io/en/stable/api/pymongo/results.html#pymongo.results.InsertManyResult) ，它提供了关于插入文档的信息。

要从集合中检索文档，可以使用 [`.find()`](https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.find) 。如果没有参数，`.find()`返回一个 [`Cursor`](https://pymongo.readthedocs.io/en/stable/api/pymongo/cursor.html#pymongo.cursor.Cursor) 对象，该对象使[按需生成](https://realpython.com/introduction-to-python-generators/)集合中的文档:

>>>

```py
>>> import pprint

>>> for doc in tutorial.find():
...     pprint.pprint(doc)
...
{'_id': ObjectId('600747355e6ea8d224f754ba'),
 'author': 'Jon',
 'contributors': ['Aldren', 'Geir Arne', 'Joanna', 'Jason'],
 'title': 'Reading and Writing CSV Files in Python',
 'url': 'https://realpython.com/python-csv/'}
 ...
{'_id': ObjectId('6008511c87eb0fbf73dbf71f'),
 'author': 'David',
 'contributors': ['Aldren', 'Joanna', 'Jacob'],
 'title': 'Object-Oriented Programming (OOP) in Python 3',
 'url': 'https://realpython.com/python3-object-oriented-programming/'}
```

在这里，您对`.find()`返回的对象运行一个循环并打印连续的结果，使用 [`pprint.pprint()`](https://docs.python.org/3/library/pprint.html#pprint.pprint) 提供一个用户友好的输出格式。

您还可以使用`.find_one()`来检索单个文档。在这种情况下，您可以使用包含要匹配的字段的字典。例如，如果你想检索乔恩的第一个教程，那么你可以这样做:

>>>

```py
>>> import pprint

>>> jon_tutorial = tutorial.find_one({"author": "Jon"})

>>> pprint.pprint(jon_tutorial)
{'_id': ObjectId('600747355e6ea8d224f754ba'),
 'author': 'Jon',
 'contributors': ['Aldren', 'Geir Arne', 'Joanna', 'Jason'],
 'title': 'Reading and Writing CSV Files in Python',
 'url': 'https://realpython.com/python-csv/'}
```

注意，教程的`ObjectId`设置在`_id`键下，这是 MongoDB 在您将文档插入数据库时自动添加的惟一文档标识符。

PyMongo 还提供了从数据库中用[替换](https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.replace_one)、[更新](https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.update_one)、[删除](https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.delete_one)文档的方法。如果您想更深入地了解这些特性，那么请看一下`Collection`的[文档](https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html)。

[*Remove ads*](/account/join/)

### 关闭连接

建立到 MongoDB 数据库的连接通常是一项开销很大的操作。如果您有一个经常在 MongoDB 数据库中检索和操作数据的应用程序，那么您可能不希望一直打开和关闭连接，因为这可能会影响应用程序的性能。

在这种情况下，您应该保持连接活动，并且只在退出应用程序之前关闭它，以清除所有获取的资源。您可以通过在`MongoClient`实例上调用 [`.close()`](https://pymongo.readthedocs.io/en/stable/api/pymongo/mongo_client.html#pymongo.mongo_client.MongoClient.close) 来关闭连接:

>>>

```py
>>> client.close()
```

另一种情况是当您的应用程序偶尔使用 MongoDB 数据库时。在这种情况下，您可能希望在需要时打开连接，并在使用后立即关闭它以释放获取的资源。解决这个问题的一致方法是使用 [`with`语句](https://realpython.com/python-with-statement/)。是的，`MongoClient`实现了[上下文管理协议](https://realpython.com/python-timer/#understanding-context-managers-in-python):

>>>

```py
>>> import pprint
>>> from pymongo import MongoClient

>>> with MongoClient() as client:
...     db = client.rptutorials
...     for doc in db.tutorial.find():
...         pprint.pprint(doc)
...
{'_id': ObjectId('600747355e6ea8d224f754ba'),
 'author': 'Jon',
 'contributors': ['Aldren', 'Geir Arne', 'Joanna', 'Jason'],
 'title': 'Reading and Writing CSV Files in Python',
 'url': 'https://realpython.com/python-csv/'}
 ...
{'_id': ObjectId('6008511c87eb0fbf73dbf71f'),
 'author': 'David',
 'contributors': ['Aldren', 'Joanna', 'Jacob'],
 'title': 'Object-Oriented Programming (OOP) in Python 3',
 'url': 'https://realpython.com/python3-object-oriented-programming/'}
```

如果您使用`with`语句来处理 MongoDB 客户端，那么在`with`代码块的末尾，客户端的`.__exit__()`方法被调用，同时通过[调用`.close()`](https://github.com/mongodb/mongo-python-driver/blob/master/pymongo/mongo_client.py) 来关闭连接。

## 通过 Python 和 MongoEngine 使用 MongoDB】

虽然 PyMongo 是一个用于与 MongoDB 接口的强大 Python 驱动程序，但对于您的许多项目来说，它可能有点太低级了。使用 PyMongo，您必须编写大量代码来一致地插入、检索、更新和删除文档。

在 PyMongo 之上提供更高抽象的一个库是 [MongoEngine](http://docs.mongoengine.org/) 。MongoEngine 是一个对象文档映射器(ODM)，大致相当于一个基于 SQL 的[对象关系映射器](https://en.wikipedia.org/wiki/Object-relational_mapping) (ORM)。MongoEngine 提供了基于类的抽象，所以你创建的所有模型都是类。

### 安装 MongoEngine

有一些 Python 库可以帮助您使用 MongoDB。然而，MongoEngine 是一个受欢迎的引擎，它提供了一组很好的特性、灵活性和对社区的支持。MongoEngine 在 PyPI 上[可用。您可以使用下面的`pip`命令来安装它:](https://pypi.org/project/mongoengine/)

```py
$ pip install mongoengine==0.22.1
```

一旦将 MongoEngine 安装到 Python 环境中，就可以使用 Python 的[面向对象的](https://realpython.com/python3-object-oriented-programming/)特性开始使用 MongoDB 数据库了。下一步是连接到正在运行的 MongoDB 实例。

### 建立连接

要与你的数据库建立连接，你需要使用 [`mongoengine.connect()`](https://docs.mongoengine.org/apireference.html#mongoengine.connect) 。这个函数有几个参数。然而，在本教程中，您将只使用其中的三个。在 Python 交互式会话中，输入以下代码:

>>>

```py
>>> from mongoengine import connect
>>> connect(db="rptutorials", host="localhost", port=27017)
MongoClient(host=['localhost:27017'], ..., read_preference=Primary())
```

这里，首先将数据库名称`db`设置为`"rptutorials"`，这是您想要工作的数据库的名称。然后您提供一个`host`和一个`port`来连接到您当前的 MongoDB 实例。由于您使用的是默认的`host`和`port`，您可以省略这两个参数，只使用`connect("rptutorials")`。

### 使用收藏和文档

要用 MongoEngine 创建文档，首先需要定义希望文档包含什么数据。换句话说，您需要定义一个[文档模式](https://docs.mongoengine.org/guide/defining-documents.html#defining-a-document-s-schema)。MongoEngine 鼓励您定义一个文档模式来帮助您减少编码错误，并允许您定义实用程序或助手方法。

与 ORM 类似，像 MongoEngine 这样的 ODM 为您提供了一个基类或模型类来定义文档模式。在 ORMs 中，那个类相当于一个表，它的实例相当于行。在 MongoEngine 中，类相当于一个集合，它的实例相当于文档。

要创建一个模型，您需要子类化 [`Document`](https://docs.mongoengine.org/apireference.html#documents) ，并提供必需的字段作为[类属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)。继续博客示例，以下是如何为教程创建模型:

>>>

```py
>>> from mongoengine import Document, ListField, StringField, URLField

>>> class Tutorial(Document):
...     title = StringField(required=True, max_length=70)
...     author = StringField(required=True, max_length=20)
...     contributors = ListField(StringField(max_length=20))
...     url = URLField(required=True)
```

使用这个模型，您告诉 MongoEngine，您期望一个`Tutorial`文档有一个`.title`、一个`.author`、一个`.contributors`列表和一个`.url`。基类`Document`使用这些信息和字段类型来验证输入数据。

**注意:**数据库模型更困难的任务之一是**数据验证**。如何确保输入数据符合您的格式要求？这也是你需要一个连贯统一的文档模式的原因之一。

据说 MongoDB 是一个无模式数据库，但这并不意味着它是无模式的。在同一个集合中包含不同模式的文档会导致处理错误和不一致的行为。

例如，如果您试图保存一个没有`.title`的`Tutorial`对象，那么您的模型会抛出一个异常并通知您。你可以更进一步，添加更多的限制，比如`.title`的长度，等等。

有几个通用参数可用于验证字段。以下是一些更常用的参数:

*   **`db_field`** 指定了不同的字段名称。
*   **`required`** 确保字段被提供。
*   **`default`** 如果没有给定值，则为给定字段提供默认值。
*   **`unique`** 确保集合中没有其他文档具有与该字段相同的值。

每个特定的字段类型也有自己的一组参数。您可以查看[文档](https://docs.mongoengine.org/apireference.html#fields)以获得可用字段类型的完整指南。

要保存一个文档到你的数据库，你需要在一个文档对象上调用 [`.save()`](https://docs.mongoengine.org/apireference.html#mongoengine.Document.save) 。如果文档已经存在，则所有更改将应用于现有文档。如果文档不存在，那么它将被创建。

以下是创建教程并将其保存到示例教程数据库中的示例:

>>>

```py
>>> tutorial1 = Tutorial(
...     title="Beautiful Soup: Build a Web Scraper With Python",
...     author="Martin",
...     contributors=["Aldren", "Geir Arne", "Jaya", "Joanna", "Mike"],
...     url="https://realpython.com/beautiful-soup-web-scraper-python/"
... )

>>> tutorial1.save()  # Insert the new tutorial
<Tutorial: Tutorial object>
```

默认情况下，`.save()`将新文档插入到以模型类`Tutorial`命名的集合中，除了使用小写字母。在这种情况下，集合名称为`tutorial`，它与您用来保存教程的集合相匹配。

当您调用`.save()`时，PyMongo 执行**数据验证**。这意味着它根据您在`Tutorial`模型类中声明的模式检查输入数据。如果输入数据违反了模式或它的任何约束，那么您会得到一个异常，并且数据不会保存到数据库中。

例如，如果您试图在不提供`.title`的情况下保存教程，会发生以下情况:

>>>

```py
>>> tutorial2 = Tutorial()
>>> tutorial2.author = "Alex"
>>> tutorial2.contributors = ["Aldren", "Jon", "Joanna"]
>>> tutorial2.url = "https://realpython.com/convert-python-string-to-int/"
>>> tutorial2.save()
Traceback (most recent call last):
  ...
mongoengine.errors.ValidationError: ... (Field is required: ['title'])
```

在这个例子中，首先要注意的是，您也可以通过为属性赋值来构建一个`Tutorial`对象。第二，由于您没有为新教程提供一个`.title`，`.save()`会抛出一个`ValidationError`，告诉您`.title`字段是必需的。拥有自动数据验证是一个很棒的特性，可以帮你省去一些麻烦。

每个`Document`子类都有一个`.objects`属性，可以用来访问相关集合中的文档。例如，下面是你如何[打印](https://realpython.com/python-print/)你当前所有教程的`.title`:

>>>

```py
>>> for doc in Tutorial.objects:
...     print(doc.title)
...
Reading and Writing CSV Files in Python
How to Iterate Through a Dictionary in Python
Python 3's f-Strings: An Improved String Formatting Syntax (Guide)
Working With JSON Data in Python
Python's Requests Library (Guide)
Object-Oriented Programming (OOP) in Python 3
Beautiful Soup: Build a Web Scraper With Python
```

[`for`循环](https://realpython.com/python-for-loop/)遍历你所有的教程并将它们的`.title`数据打印到屏幕上。你也可以使用`.objects`来过滤你的文件。例如，假设您想要检索由 [Alex](https://realpython.com/team/aronquillo/) 创作的教程。在这种情况下，您可以这样做:

>>>

```py
>>> for doc in Tutorial.objects(author="Alex"):
...     print(doc.title)
...
Python's Requests Library (Guide)
```

MongoEngine 非常适合为任何类型的应用程序管理 MongoDB 数据库。它的特性使它非常适合使用高级方法创建高效且可伸缩的程序。如果你想了解更多关于 MongoEngine 的信息，请务必查看它的[用户指南](https://docs.mongoengine.org/guide/index.html)。

[*Remove ads*](/account/join/)

## 结论

如果您需要一个健壮的、可伸缩的、灵活的数据库解决方案，那么 [MongoDB](https://www.mongodb.com/what-is-mongodb) 可能是一个不错的选择。MongoDB 是一个成熟和流行的 [NoSQL](https://en.wikipedia.org/wiki/NoSQL) 数据库，具有强大的 Python 支持。对如何使用 Python 访问 MongoDB 有了很好的理解后，您就可以创建伸缩性好、性能卓越的数据库应用程序了。

使用 MongoDB，您还可以受益于人类可读且高度灵活的数据模型，因此您可以快速适应需求变化。

**在本教程中，您学习了:**

*   什么是 **MongoDB** 和 **NoSQL** 数据库
*   如何在您的系统上安装和运行 MongoDB
*   如何创建和使用 **MongoDB 数据库**
*   如何使用 **PyMongo 驱动程序**在 Python 中与 MongoDB 接口
*   如何使用 **MongoEngine 对象-文档映射器**来处理 MongoDB

您在本教程中编写的示例可以下载。要获得它们的源代码，请单击下面的链接:

**获取源代码:** [单击此处获取源代码，您将在本教程中使用](https://realpython.com/bonus/python-mongodb-code/)来了解如何将 MongoDB 与 Python 结合使用。******