# 第四章：数据库

在本章中，我们将给出几个使用数据库的 Tornado Web 应用的例子。我们将从一个简单的 RESTful API 例子起步，然后创建 3.1.2 节中的 Burt's Book 网站的完整功能版本。

本章中的例子使用 MongoDB 作为数据库，并通过 pymongo 作为驱动来连接 MongoDB。当然，还有很多数据库系统可以用在 Web 应用中：Redis、CouchDB 和 MySQL 都是一些知名的选择，并且 Tornado 自带处理 MySQL 请求的库。我们选择使用 MongoDB 是因为它的简单性和便捷性：安装简单，并且能够和 Python 代码很好地融合。它结构自然，预定义数据结构不是必需的，很适合原型开发。

在本章中，我们假设你已经在机器上安装了 MongoDB，能够运行示例代码，不过也可以在远程服务器上使用 MongoDB，相关的代码调整也很容易。如果你不想在你的机器上安装 MongoDB，或者没有一个适合你操作系统的 MongoDB 版本，你也可以选择一些 MongoDB 主机服务。我们推荐使用[MongoHQ](http://www.mongohq.com/)。在我们最初的例子中，假设你已经在你的机器上运行了 MongoDB，但使用远程服务器（包括 MongoHQ）运行的 MongoDB 时，调整代码也很简单。

我们同样还假设你已经有一些数据库的经验了，尽管并不一定是特定的 MongoDB 数据库的经验。当然，我们只会使用 MongoDB 的一点皮毛；如果想获得更多信息请查阅 MongoDB 文档（[`www.mongodb.org/display/DOCS/Home`](http://www.mongodb.org/display/DOCS/Home)）让我们开始吧！

*   4.1 使用 PyMongo 进行 MongoDB 基础操作
    *   4.1.1 创建连接
    *   4.1.2 处理文档
    *   4.1.3 MongoDB 文档和 JSON
    *   4.2 一个简单的持久化 Web 服务
        *   4.2.1 只读字典
        *   4.2.2 写字典
    *   4.3 Burt's Books
        *   4.3.1 读取书籍（从数据库）
        *   4.3.2 编辑和添加书籍
    *   4.4 MongoDB：下一步

## 4.1 使用 PyMongo 进行 MongoDB 基础操作

在我们使用 MongoDB 编写 Web 应用之前，我们需要了解如何在 Python 中使用 MongoDB。在这一节，你将学会如何使用 PyMongo 连接 MongoDB 数据库，然后学习如何使用 pymongo 在 MongoDB 集合中创建、取出和更新文档。

PyMongo 是一个简单的包装 MongoDB 客户端 API 的 Python 库。你可以在[`api.mongodb.org/python/current/`](http://api.mongodb.org/python/current/)下载获得。一旦你安装完成，打开一个 Python 解释器，然后跟随下面的步骤。

### 4.1.1 创建连接

首先，你需要导入 PyMongo 库，并创建一个到 MongoDB 数据库的连接。

```py
>>> import pymongo
>>> conn = pymongo.Connection("localhost", 27017)

```

前面的代码向我们展示了如何连接运行在你本地机器上默认端口（27017）上的 MongoDB 服务器。如果你正在使用一个远程 MongoDB 服务器，替换 localhost 和 27017 为合适的值。你也可以使用 MongoDB URI 来连接 MongoDB，就像下面这样：

```py
>>> conn = pymongo.Connection(
... "mongodb://user:password@staff.mongohq.com:10066/your_mongohq_db")

```

前面的代码将连接 MongoHQ 主机上的一个名为 your_mongohq_db 的数据库，其中 user 为用户名，password 为密码。你可以在[`www.mongodb.org/display/DOCS/Connections`](http://www.mongodb.org/display/DOCS/Connections)中了解更多关于 MongoDB URI 的信息。

一个 MongoDB 服务器可以包括任意数量的数据库，而 Connection 对象可以让你访问你连接的服务器的任何一个数据库。你可以通过对象属性或像字典一样使用对象来获得代表一个特定数据库的对象。如果数据库不存在，则被自动建立。

```py
>>> db = conn.example or: db = conn['example']

```

一个数据库可以拥有任意多个集合。一个集合就是放置一些相关文档的地方。我们使用 MongoDB 执行的大部分操作（查找文档、保存文档、删除文档）都是在一个集合对象上执行的。你可以在数据库对象上调用 collection_names 方法获得数据库中的集合列表。

```py
>>> db.collection_names()
[]

```

当然，我们还没有在我们的数据库中添加任何集合，所以这个列表是空的。当我们插入第一个文档时，MongoDB 会自动创建集合。你可以在数据库对象上通过访问集合名字的属性来获得代表集合的对象，然后调用对象的 insert 方法指定一个 Python 字典来插入文档。比如，在下面的代码中，我们在集合 widgets 中插入了一个文档。因为 widgets 集合并不存在，MongoDB 会在文档被添加时自动创建。

```py
>>> widgets = db.widgets or: widgets = db['widgets'] (see below)
>>> widgets.insert({"foo": "bar"})
ObjectId('4eada0b5136fc4aa41000000')
>>> db.collection_names()
[u'widgets', u'system.indexes']

```

（system.indexes 集合是 MongoDB 内部使用的。处于本章的目的，你可以忽略它。）

在之前展示的代码中，你既可以使用数据库对象的属性访问集合，也可以把数据库对象看作一个字典然后把集合名称作为键来访问。比如，如果 db 是一个 pymongo 数据库对象，那么 db.widgets 和 db['widgets']同样都可以访问这个集合。

### 4.1.2 处理文档

MongoDB 以文档的形式存储数据，这种形式有着相对自由的数据结构。MongoDB 是一个"无模式"数据库：同一个集合中的文档通常拥有相同的结构，但是 MongoDB 中并不强制要求使用相同结构。在内部，MongoDB 以一种称为 BSON 的类似 JSON 的二进制形式存储文档。PyMongo 允许我们以 Python 字典的形式写和取出文档。

为了在集合中 创建一个新的文档，我们可以使用字典作为参数调用文档的 insert 方法。

```py
>>> widgets.insert({"name": "flibnip", "description": "grade-A industrial flibnip", "quantity": 3})
ObjectId('4eada3a4136fc4aa41000001')

```

既然文档在数据库中，我们可以使用集合对象的 find_one 方法来取出文档。你可以通过传递一个键为文档名、值为你想要匹配的表达式的字典来告诉 find_one 找到 一个特定的文档。比如，我们想要返回文档名域 name 的值等于 flibnip 的文档（即，我们刚刚创建的文档），可以像下面这样调用 find_oen 方法：

```py
>>> widgets.find_one({"name": "flibnip"})
{u'description': u'grade-A industrial flibnip',
 u'_id': ObjectId('4eada3a4136fc4aa41000001'),
 u'name': u'flibnip', u'quantity': 3}

```

请注意 _id 域。当你创建任何文档时，MongoDB 都会自动添加这个域。它的值是一个 ObjectID，一种保证文档唯一的 BSON 对象。你可能已经注意到，当我们使用 insert 方法成功创建一个新的文档时，这个 ObjectID 同样被返回了。（当你创建文档时，可以通过给 _id 键赋值来覆写自动创建的 ObjectID 值。）

find_one 方法返回的值是一个简单的 Python 字典。你可以从中访问独立的项，迭代它的键值对，或者就像使用其他 Python 字典那样修改值。

```py
>>> doc = db.widgets.find_one({"name": "flibnip"})
>>> type(doc)
<type 'dict'>
>>> print doc['name']
flibnip
>>> doc['quantity'] = 4

```

然而，字典的改变并不会自动保存到数据库中。如果你希望把字典的改变保存，需要调用集合的 save 方法，并将修改后的字典作为参数进行传递：

```py
>>> doc['quantity'] = 4
>>> db.widgets.save(doc)
>>> db.widgets.find_one({"name": "flibnip"})
{u'_id': ObjectId('4eb12f37136fc4b59d000000'),
 u'description': u'grade-A industrial flibnip',
 u'quantity': 4, u'name': u'flibnip'}

```

让我们在集合中添加更多的文档：

```py
>>> widgets.insert({"name": "smorkeg", "description": "for external use only", "quantity": 4})
ObjectId('4eadaa5c136fc4aa41000002')
>>> widgets.insert({"name": "clobbasker", "description": "properties available on request", "quantity": 2})
ObjectId('4eadad79136fc4aa41000003')

```

我们可以通过调用集合的 find 方法来获得集合中所有文档的列表，然后迭代其结果：

```py
>>> for doc in widgets.find():
...     print doc
...
{u'_id': ObjectId('4eada0b5136fc4aa41000000'), u'foo': u'bar'}
{u'description': u'grade-A industrial flibnip',
 u'_id': ObjectId('4eada3a4136fc4aa41000001'),
 u'name': u'flibnip', u'quantity': 4}
{u'description': u'for external use only',
 u'_id': ObjectId('4eadaa5c136fc4aa41000002'),
 u'name': u'smorkeg', u'quantity': 4}
{u'description': u'properties available on request',
 u'_id': ObjectId('4eadad79136fc4aa41000003'),
 u'name': u'clobbasker',
 u'quantity': 2}

```

如果我们希望获得文档的一个子集，我们可以在 find 方法中传递一个字典参数，就像我们在 find_one 中那样。比如，找到那些 quantity 键的值为 4 的集合：

```py
>>> for doc in widgets.find({"quantity": 4}):
...     print doc
...
{u'description': u'grade-A industrial flibnip',
 u'_id': ObjectId('4eada3a4136fc4aa41000001'),
 u'name': u'flibnip', u'quantity': 4}
{u'description': u'for external use only',
 u'_id': ObjectId('4eadaa5c136fc4aa41000002'),
 u'name': u'smorkeg',
 u'quantity': 4}

```

最后，我们可以使用集合的 remove 方法从集合中删除一个文档。remove 方法和 find、find_one 一样，也可以使用一个字典参数来指定哪个文档需要被删除。比如，要删除所有 name 键的值为 flipnip 的文档，输入：

```py
>>> widgets.remove({"name": "flibnip"})

```

列出集合中的所有文档来确认上面的文档已经被删除：

```py
>>> for doc in widgets.find():
...     print doc
...
{u'_id': ObjectId('4eada0b5136fc4aa41000000'),
 u'foo': u'bar'}
{u'description': u'for external use only',
 u'_id': ObjectId('4eadaa5c136fc4aa41000002'),
 u'name': u'smorkeg', u'quantity': 4}
{u'description': u'properties available on request',
 u'_id': ObjectId('4eadad79136fc4aa41000003'),
 u'name': u'clobbasker',
 u'quantity': 2}

```

### 4.1.3 MongoDB 文档和 JSON

使用 Web 应用时，你经常会想采用 Python 字典并将其序列化为一个 JSON 对象（比如，作为一个 AJAX 请求的响应）。由于你使用 PyMongo 从 MongoDB 中取出的文档是一个简单的字典，你可能会认为你可以使用 json 模块的 dumps 函数就可以简单地将其转换为 JSON。但，这还有一个障碍：

```py
>>> doc = db.widgets.find_one({"name": "flibnip"})
>>> import json
>>> json.dumps(doc)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    [stack trace omitted]
TypeError: ObjectId('4eb12f37136fc4b59d000000') is not JSON serializable

```

这里的问题是 Python 的 json 模块并不知道如何转换 MongoDB 的 ObjectID 类型到 JSON。有很多方法可以处理这个问题。其中最简单的方法（也是我们在本章中采用的方法）是在我们序列化之前从字典里简单地删除 _id 键。

```py
>>> del doc["_id"]
>>> json.dumps(doc)
'{"description": "grade-A industrial flibnip", "quantity": 4, "name": "flibnip"}'

```

一个更复杂的方法是使用 PyMongo 的 json_util 库，它同样可以帮你序列化其他 MongoDB 特定数据类型到 JSON。我们可以在[`api.mongodb.org/python/current/api/bson/json_util.html`](http://api.mongodb.org/python/current/api/bson/json_util.html)了解更多关于这个库的信息。

## 4.2 一个简单的持久化 Web 服务

现在我们知道编写一个 Web 服务，可以访问 MongoDB 数据库中的数据。首先，我们要编写一个只从 MongoDB 读取数据的 Web 服务。然后，我们写一个可以读写数据的服务。

### 4.2.1 只读字典

我们将要创建的应用是一个基于 Web 的简单字典。你发送一个指定单词的请求，然后返回这个单词的定义。一个典型的交互看起来是下面这样的：

```py
$ curl http://localhost:8000/oarlock
{definition: "A device attached to a rowboat to hold the oars in place",
"word": "oarlock"}

```

这个 Web 服务将从 MongoDB 数据库中取得数据。具体来说，我们将根据 word 属性查询文档。在我们查看 Web 应用本身的源码之前，先让我们从 Python 解释器中向数据库添加一些单词。

```py
>>> import pymongo
>>> conn = pymongo.Connection("localhost", 27017)
>>> db = conn.example
>>> db.words.insert({"word": "oarlock", "definition": "A device attached to a rowboat to hold the oars in place"})
ObjectId('4eb1d1f8136fc4be90000000')
>>> db.words.insert({"word": "seminomadic", "definition": "Only partial
ly nomadic"})
ObjectId('4eb1d356136fc4be90000001')
>>> db.words.insert({"word": "perturb", "definition": "Bother, unsettle
, modify"})
ObjectId('4eb1d39d136fc4be90000002')

```

代码清单 4-1 是我们这个词典 Web 服务的源码，在这个代码中我们查询刚才添加的单词然后使用其定义作为响应。

代码清单 4-1 一个词典 Web 服务：definitions_readonly.py

```py
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

import pymongo

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [(r"/(\w+)", WordHandler)]
        conn = pymongo.Connection("localhost", 27017)
        self.db = conn["example"]
        tornado.web.Application.__init__(self, handlers, debug=True)

class WordHandler(tornado.web.RequestHandler):
    def get(self, word):
        coll = self.application.db.words
        word_doc = coll.find_one({"word": word})
        if word_doc:
            del word_doc["_id"]
            self.write(word_doc)
        else:
            self.set_status(404)
            self.write({"error": "word not found"})

if __name__ == "__main__":
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

```

在命令行中像下面这样运行这个程序：

```py
$ python definitions_readonly.py

```

现在使用 curl 或者你的浏览器来向应用发送一个请求。

```py
$ curl http://localhost:8000/perturb
{"definition": "Bother, unsettle, modify", "word": "perturb"}

```

如果我们请求一个数据库中没有添加的单词，会得到一个 404 错误以及一个错误信息：

```py
$ curl http://localhost:8000/snorkle
{"error": "word not found"}

```

那么这个程序是如何工作的呢？让我们看看这个程序的主线。开始，我们在程序的最上面导入了 import pymongo 库。然后我们在我们的 TornadoApplication 对象的**init**方法中实例化了一个 pymongo 连接对象。我们在 Application 对象中创建了一个 db 属性，指向 MongoDB 的 example 数据库。下面是相关的代码：

```py
conn = pymongo.Connection("localhost", 27017)
self.db = conn["example"]

```

一旦我们在 Application 对象中添加了 db 属性，我们就可以在任何 RequestHandler 对象中使用 self.application.db 访问它。实际上，这正是我们为了取出 pymongo 的 words 集合对象而在 WordHandler 中 get 方法所做的事情。

```py
def get(self, word):
    coll = self.application.db.words
    word_doc = coll.find_one({"word": word})
    if word_doc:
        del word_doc["_id"]
        self.write(word_doc)
    else:
        self.set_status(404)
        self.write({"error": "word not found"})

```

在我们将集合对象指定给变量 coll 后，我们使用用户在 HTTP 路径中请求的单词调用 find_one 方法。如果我们发现这个单词，则从字典中删除 _id 键（以便 Python 的 json 库可以将其序列化），然后将其传递给 RequestHandler 的 write 方法。write 方法将会自动序列化字典为 JSON 格式。

如果 find_one 方法没有匹配任何对象，则返回 None。在这种情况下，我们将响应状态设置为 404，并且写一个简短的 JSON 来提示用户这个单词在数据库中没有找到。

### 4.2.2 写字典

从字典里查询单词很有趣，但是在交互解释器中添加单词的过程却很麻烦。我们例子的下一步是使 HTTP 请求网站服务时能够创建和修改单词。

它的工作流程是：发出一个特定单词的 POST 请求，将根据请求中给出的定义修改已经存在的定义。如果这个单词并不存在，则创建它。例如，创建一个新的单词：

```py
$ curl -d definition=a+leg+shirt http://localhost:8000/pants
{"definition": "a leg shirt", "word": "pants"}

```

我们可以使用一个 GET 请求来获得已创建单词的定义：

```py
$ curl http://localhost:8000/pants
{"definition": "a leg shirt", "word": "pants"}

```

我们可以发出一个带有一个单词定义域的 POST 请求来修改一个已经存在的单词（就和我们创建一个新单词时使用的参数一样）：

```py
$ curl -d definition=a+boat+wizard http://localhost:8000/oarlock
{"definition": "a boat wizard", "word": "oarlock"}

```

代码清单 4-2 是我们的词典 Web 服务的读写版本的源代码。

代码清单 4-2 一个读写字典服务：definitions_readwrite.py

```py
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

import pymongo

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [(r"/(\w+)", WordHandler)]
        conn = pymongo.Connection("localhost", 27017)
        self.db = conn["definitions"]
        tornado.web.Application.__init__(self, handlers, debug=True)

class WordHandler(tornado.web.RequestHandler):
    def get(self, word):
        coll = self.application.db.words
        word_doc = coll.find_one({"word": word})
        if word_doc:
            del word_doc["_id"]
            self.write(word_doc)
        else:
            self.set_status(404)
    def post(self, word):
        definition = self.get_argument("definition")
        coll = self.application.db.words
        word_doc = coll.find_one({"word": word})
        if word_doc:
            word_doc['definition'] = definition
            coll.save(word_doc)
        else:
            word_doc = {'word': word, 'definition': definition}
            coll.insert(word_doc)
        del word_doc["_id"]
        self.write(word_doc)

if __name__ == "__main__":
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

```

除了在 WordHandler 中添加了一个 post 方法之外，这个源代码和只读服务的版本完全一样。让我们详细看看这个方法吧：

```py
def post(self, word):
    definition = self.get_argument("definition")
    coll = self.application.db.words
    word_doc = coll.find_one({"word": word})
    if word_doc:
        word_doc['definition'] = definition
        coll.save(word_doc)
    else:
        word_doc = {'word': word, 'definition': definition}
        coll.insert(word_doc)
    del word_doc["_id"]
    self.write(word_doc)

```

我们首先做的事情是使用 get_argument 方法取得 POST 请求中传递的 definition 参数。然后，就像在 get 方法一样，我们尝试使用 find_one 方法从数据库中加载给定单词的文档。如果发现这个单词的文档，我们将 definition 条目的值设置为从 POST 参数中取得的值，然后调用集合对象的 save 方法将改变写到数据库中。如果没有发现文档，则创建一个新文档，并使用 insert 方法将其保存到数据库中。无论上述哪种情况，在数据库操作执行之后，我们在响应中写文档（注意首先要删掉 _id 属性）。

## 4.3 Burt's Books

在[第三章](http://dockerpool.com/static/books/introduction_to_tornado_cn/ch3.html)中，我们提出了 Burt's Book 作为使用 Tornado 模板工具构建复杂 Web 应用的例子。在本节中，我们将展示使用 MongoDB 作为数据存储的 Burt's Books 示例版本呢。

### 4.3.1 读取书籍（从数据库）

让我们从一些简单的版本开始：一个从数据库中读取书籍列表的 Burt's Books。首先，我们需要在我们的 MongoDB 服务器上创建一个数据库和一个集合，然后用书籍文档填充它，就像下面这样：

```py
>>> import pymongo
>>> conn = pymongo.Connection()
>>> db = conn["bookstore"]
>>> db.books.insert({
...     "title":"Programming Collective Intelligence",
...     "subtitle": "Building Smart Web 2.0 Applications",
...     "image":"/static/images/collective_intelligence.gif",
...     "author": "Toby Segaran",
...     "date_added":1310248056,
...     "date_released": "August 2007",
...     "isbn":"978-0-596-52932-1",
...     "description":"<p>[...]</p>"
... })
ObjectId('4eb6f1a6136fc42171000000')
>>> db.books.insert({
...     "title":"RESTful Web Services",
...     "subtitle": "Web services for the real world",
...     "image":"/static/images/restful_web_services.gif",
...     "author": "Leonard Richardson, Sam Ruby",
...     "date_added":1311148056,
...     "date_released": "May 2007",
...     "isbn":"978-0-596-52926-0",
...     "description":"<p>[...]>/p>"
... })
ObjectId('4eb6f1cb136fc42171000001')

```

（我们为了节省空间已经忽略了这些书籍的详细描述。）一旦我们在数据库中有了这些文档，我们就准备好了。代码清单 4-3 展示了 Burt's Books Web 应用修改版本的源代码 burts_books_db.py。

代码清单 4-3 读取数据库：burts_books_db.py

```py
import os.path
import tornado.locale
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
import pymongo

define("port", default=8000, help="run on the given port", type=int)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/recommended/", RecommendedHandler),
        ]
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            ui_modules={"Book": BookModule},
            debug=True,
        )
        conn = pymongo.Connection("localhost", 27017)
        self.db = conn["bookstore"]
        tornado.web.Application.__init__(self, handlers, **settings)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(
            "index.html",
            page_title = "Burt's Books | Home",
            header_text = "Welcome to Burt's Books!",
        )

class RecommendedHandler(tornado.web.RequestHandler):
    def get(self):
        coll = self.application.db.books
        books = coll.find()
        self.render(
            "recommended.html",
            page_title = "Burt's Books | Recommended Reading",
            header_text = "Recommended Reading",
            books = books
        )

class BookModule(tornado.web.UIModule):
    def render(self, book):
        return self.render_string(
            "modules/book.html",
            book=book,
        )
    def css_files(self):
        return "/static/css/recommended.css"
    def javascript_files(self):
        return "/static/js/recommended.js"

if __name__ == "__main__":
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

```

正如你看到的，这个程序和[第三章](http://dockerpool.com/static/books/introduction_to_tornado_cn/ch3.html)中 Burt's Books Web 应用的原始版本几乎完全相同。它们之间只有两个不同点。其一，我们在我们的 Application 中添加了一个 db 属性来连接 MongoDB 服务器：

```py
conn = pymongo.Connection("localhost", 27017)
self.db = conn["bookstore"]

```

其二，我们使用连接的 find 方法来从数据库中取得书籍文档的列表，然后在渲染 recommended.html 时将这个列表传递给 RecommendedHandler 的 get 方法。下面是相关的代码：

```py
def get(self):
    coll = self.application.db.books
    books = coll.find()
    self.render(
        "recommended.html",
        page_title = "Burt's Books | Recommended Reading",
        header_text = "Recommended Reading",
        books = books
    )

```

此前，书籍列表是被硬编码在 get 方法中的。但是，因为我们在 MongoDB 中添加的文档和原始的硬编码字典拥有相同的域，所以我们之前写的模板代码并不需要修改。

像下面这样运行应用：

```py
$ python burts_books_db.py

```

然后让你的浏览器指向[`localhost:8000/recommended/`](http://localhost:8000/recommended/)。这次，页面和硬编码版本的 Burt's Books 看起来几乎一样（参见图 3-6）。

### 4.3.2 编辑和添加书籍

我们的下一步是添加一个接口用来编辑已经存在于数据库的书籍以及添加新书籍到数据库中。为此，我们需要一个让用户填写书籍信息的表单，一个服务表单的处理程序，以及一个处理表单结果并将其存入数据库的处理函数。

这个版本的 Burt's Books 和之前给出的代码几乎是一样的，只是增加了下面我们要讨论的一些内容。你可以跟随本书附带的完整代码阅读下面部分，相关的程序名为 burts_books_rwdb.py。

#### 4.3.2.1 渲染编辑表单

下面是 BookEditHandler 的源代码，它完成了两件事情：

1.  GET 请求渲染一个显示已存在书籍数据的 HTML 表单（在模板 book_edit.html 中）。
2.  POST 请求从表单中取得数据，更新数据库中已存在的书籍记录或依赖提供的数据添加一个新的书籍。

下面是处理程序的源代码：

```py
class BookEditHandler(tornado.web.RequestHandler):
    def get(self, isbn=None):
        book = dict()
        if isbn:
            coll = self.application.db.books
            book = coll.find_one({"isbn": isbn})
        self.render("book_edit.html",
            page_title="Burt's Books",
            header_text="Edit book",
            book=book)

    def post(self, isbn=None):
        import time
        book_fields = ['isbn', 'title', 'subtitle', 'image', 'author',
            'date_released', 'description']
        coll = self.application.db.books
        book = dict()
        if isbn:
            book = coll.find_one({"isbn": isbn})
        for key in book_fields:
            book[key] = self.get_argument(key, None)

        if isbn:
            coll.save(book)
        else:
            book['date_added'] = int(time.time())
            coll.insert(book)
        self.redirect("/recommended/")

```

我们将在稍后对其进行详细讲解，不过现在先让我们看看如何在 Application 类中建立请求到处理程序的路由。下面是 Application 的**init**方法的相关代码部分：

```py
handlers = [
    (r"/", MainHandler),
    (r"/recommended/", RecommendedHandler),
    (r"/edit/([0-9Xx\-]+)", BookEditHandler),
    (r"/add", BookEditHandler)
]

```

正如你所看到的，BookEditHandler 处理了两个*不同*路径模式的请求。其中一个是/add，提供不存在信息的编辑表单，因此你可以向数据库中添加一本新的书籍；另一个/edit/([0-9Xx-]+)，根据书籍的 ISBN 渲染一个已存在书籍的表单。

#### 4.3.2.2 从数据库中取出书籍信息

让我们看看 BookEditHandler 的 get 方法是如何工作的：

```py
def get(self, isbn=None):
    book = dict()
    if isbn:
        coll = self.application.db.books
        book = coll.find_one({"isbn": isbn})
    self.render("book_edit.html",
        page_title="Burt's Books",
        header_text="Edit book",
        book=book)

```

如果该方法作为到/add 请求的结果被调用，Tornado 将调用一个没有第二个参数的 get 方法（因为路径中没有正则表达式的匹配组）。在这种情况下，默认将一个空的 book 字典传递给 book_edit.html 模板。

如果该方法作为到类似于/edit/0-123-456 请求的结果被调用，那么 isdb 参数被设置为 0-123-456。在这种情况下，我们从 Application 实例中取得 books 集合，并用它查询 ISBN 匹配的书籍。然后我们传递结果 book 字典给模板。

下面是模板（book_edit.html）的代码：

```py
{% extends "main.html" %}
{% autoescape None %}

{% block body %}
<form method="POST">
    ISBN <input type="text" name="isbn"
        value="{{ book.get('isbn', '') }}"><br>
    Title <input type="text" name="title"
        value="{{ book.get('title', '') }}"><br>
    Subtitle <input type="text" name="subtitle"
        value="{{ book.get('subtitle', '') }}"><br>
    Image <input type="text" name="image"
        value="{{ book.get('image', '') }}"><br>
    Author <input type="text" name="author"
        value="{{ book.get('author', '') }}"><br>
    Date released <input type="text" name="date_released"
        value="{{ book.get('date_released', '') }}"><br>
    Description<br>
    <textarea name="description" rows="5"
        cols="40">{% raw book.get('description', '')%}</textarea><br>
    <input type="submit" value="Save">
</form>
{% end %}

```

这是一个相当常规的 HTML 表单。如果请求处理函数传进来了 book 字典，那么我们用它预填充带有已存在书籍数据的表单；如果键不在字典中，我们使用 Python 字典对象的 get 方法为其提供默认值。记住 input 标签的 name 属性被设置为 book 字典的对应键；这使得与来自带有我们期望放入数据库数据的表单关联变得简单。

同样还需要记住的是，因为 form 标签没有 action 属性，因此表单的 POST 将会定向到当前 URL，这正是我们想要的（即，如果页面以/edit/0-123-456 加载，POST 请求将转向/edit/0-123-456；如果页面以/add 加载，则 POST 将转向/add）。图 4-1 所示为该页面渲染后的样子。

![图 4-1](img/2015-09-04_55e96dc2b9744.jpg)

图 4-1 Burt's Books：添加新书的表单

#### 4.3.2.3 保存到数据库中

让我们看看 BookEditHandler 的 post 方法。这个方法处理书籍编辑表单的请求。下面是源代码：

```py
def post(self, isbn=None):
    import time
    book_fields = ['isbn', 'title', 'subtitle', 'image', 'author',
        'date_released', 'description']
    coll = self.application.db.books
    book = dict()
    if isbn:
        book = coll.find_one({"isbn": isbn})
    for key in book_fields:
        book[key] = self.get_argument(key, None)

    if isbn:
        coll.save(book)
    else:
        book['date_added'] = int(time.time())
        coll.insert(book)
    self.redirect("/recommended/")

```

和 get 方法一样，post 方法也有两个任务：处理编辑已存在文档的请求以及添加新文档的请求。如果有 isbn 参数（即，路径的请求类似于/edit/0-123-456），我们假定为编辑给定 ISBN 的文档。如果这个参数没有被提供，则假定为添加一个新文档。

我们先设置一个空的字典变量 book。如果我们正在编辑一个已存在的书籍，我们使用 book 集合的 find_one 方法从数据库中加载和传入的 ISBN 值对应的文档。无论哪种情况，book_fields 列表指定哪些域应该出现在书籍文档中。我们迭代这个列表，使用 RequestHandler 对象的 get_argument 方法从 POST 请求中抓取对应的值。

此时，我们准备好更新数据库了。如果我们有一个 ISBN 码，那么我们调用集合的 save 方法来更新数据库中的书籍文档。如果没有的话，我们调用集合的 insert 方法，此时要注意首先要为 date_added 键添加一个值。（我们没有将其包含在我们的域列表中获取传入的请求，因为在图书被添加到数据库之后 date_added 值不应该再被改变。）当我们完成时，使用 RequestHandler 类的 redirect 方法给用户返回推荐页面。我们所做的任何改变可以立刻显现。图 4-2 所示为更新后的推荐页面。

![图 4-2](img/2015-09-04_55e96dc89bf34.jpg)

图 4-2 Burt's Books：带有新添加书籍的推荐列表

你还将注意到我们给每个图书条目添加了一个"Edit"链接，用于链接到列表中每个书籍的编辑表单。下面是修改后的图书模块的源代码：

```py
<div class="book" style="overflow: auto">
    <h3 class="book_title">{{ book["title"] }}</h3>
    {% if book["subtitle"] != "" %}
        <h4 class="book_subtitle">{{ book["subtitle"] }}</h4>
    {% end %}
    <img src="{{ book["image"] }}" class="book_image"/>
    <div class="book_details">
        <div class="book_date_released">Released: {{ book["date_released"]}}</div>
        <div class="book_date_added">Added: {{ locale.format_date(book["date_added"],
relative=False) }}</div>
        <h5>Description:</h5>
        <div class="book_body">{% raw book["description"] %}</div>
        <p><a href="/edit/{{ book['isbn'] }}">Edit</a></p>
    </div>
</div>

```

其中最重要的一行是：

```py
<p><a href="/edit/{{ book['isbn'] }}">Edit</a></p>

```

编辑页面的链接是把图书的 isbn 键的值添加到字符串/edit/后面组成的。这个链接将会带你进入这本图书的编辑表单。你可以从图 4-3 中看到结果。

![图 4-3](img/2015-09-04_55e96dc917ced.jpg)

图 4-3 Burt's Books：带有编辑链接的推荐列表

## 4.4 MongoDB：下一步

我们在这里只覆盖了 MongoDB 的一些基础知识--仅仅够实现本章中的示例 Web 应用。如果你对于学习更多更用的 PyMongo 和 MongoDB 知识感兴趣的话，PyMongo 教程（[`api.mongodb.org/python/2.0.1/tutorial.html`](http://api.mongodb.org/python/2.0.1/tutorial.html)）和 MongoDB 教程（[`www.mongodb.org/display/DOCS/Tutorial`](http://www.mongodb.org/display/DOCS/Tutorial)）是不错的起点。

如果你对使用 Tornado 创建在扩展性方面表现更好的 MongoDB 应用感兴趣的话，你可以自学 asyncmongo（[`github.com/bitly/asyncmongo`](https://github.com/bitly/asyncmongo)），这是一种异步执行 MongoDB 请求的类似 PyMongo 的库。我们将在[第五章](http://dockerpool.com/static/books/introduction_to_tornado_cn/ch5.html)中讨论什么是异步请求，以及为什么它在 Web 应用中扩展性更好。