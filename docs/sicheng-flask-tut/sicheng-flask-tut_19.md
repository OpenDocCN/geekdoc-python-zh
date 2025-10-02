# Flask 扩展系列(五)–MongoDB

MongoDB 是一个文档型数据库，它灵活的 Schema，多层次的数据结构和 JSON 格式的文档使得其已经成为了 NoSQL 阵营的领头羊。Flask 的 MongoDB 扩展有很多，比如[Flask-MongoAlchemy](http://pythonhosted.org/Flask-MongoAlchemy/)，基于[MongoAlchemy](http://www.mongoalchemy.org/)实现，非常类似于上一篇所介绍的 SQLAlchemy；[Flask-MongoKit](http://pythonhosted.org/Flask-MongoKit/)，基于[MongoKit](https://github.com/namlook/mongokit/wiki)实现，同 MongoAlchemy 类似，需要预先定义数据模型。不过 MongoDB 的一大优势就是数据模型，即 Collection，是灵活的，如果可以不限制数据模型的字段，将会更大程度的发挥 MongoDB 的优势，Python 的 PyMongo 框架就可以做到这个。本篇我们就要介绍基于 PyMongo 实现的 Flask 扩展，Flask-PyMongo。

### 系列文章

*   Flask 扩展系列(一)–Restful
*   Flask 扩展系列(二)–Mail
*   Flask 扩展系列(三)–国际化 I18N 和本地化 L10N
*   Flask 扩展系列(四)–SQLAlchemy
*   Flask 扩展系列(五)–MongoDB
*   Flask 扩展系列(六)–缓存
*   Flask 扩展系列(七)–表单
*   Flask 扩展系列(八)–用户会话管理
*   Flask 扩展系列(九)–HTTP 认证
*   Flask 扩展系列–自定义扩展

### 安装和启用

首先，建议读者先去了解下[PyMongo](http://api.mongodb.org/python/current/)的基本用法。

我们通过 pip 安装 Flask-PyMongo 扩展：

```py
$ pip install Flask-PyMongo
```

安装完后，查看下 PyMongo 的版本，本文中的例子必须跑在 PyMongo 3.0.x 以上：

```py
$ pip list | grep pymongo
```

然后采用下面的方法初始化一个 Flask-PyMongo 的实例：

```py
from flask import Flask
from flask.ext.pymongo import PyMongo

app = Flask(__name__)
app.config.update(
    MONGO_HOST='localhost',
    MONGO_PORT=27017,
    MONGO_USERNAME='bjhee',
    MONGO_PASSWORD='111111',
    MONGO_DBNAME='flask'
)

mongo = PyMongo(app)

```

在应用配置中，我们指定了 MongoDB 的服务器地址，端口，数据库名，用户名和密码。对于上面的配置，我们也可以简化为：

```py
app.config.update(
    MONGO_URI='mongodb://localhost:27017/flask',
    MONGO_USERNAME='bjhee',
    MONGO_PASSWORD='111111'
)

```

在同一应用中，我们还可以初始化两个以上的 Flask-PyMongo 实例，分别基于不同的配置项：

```py
app.config.update(
    MONGO_URI='mongodb://localhost:27017/flask',
    MONGO_USERNAME='bjhee',
    MONGO_PASSWORD='111111',
    MONGO_TEST_URI='mongodb://localhost:27017/test'
)

mongo = PyMongo(app)
mongo_test = PyMongo(app, config_prefix='MONGO_TEST')

```

当调用初始化方法”PyMongo()”时，传入”config_prefix”参数，该 PyMongo 实例就会使用以”MONGO_TEST”为前缀的配置项，而不是默认的”MONGO”前缀，比如上例中的”MONGO_TEST_URI”。

### 添加数据

MongoDB 中的表叫做集合(Collection)，表中的记录叫做文档(Document)。一个文档记录就是一个 JSON 对象，对于 Python 来说，就是一个字典。下面的代码就会在”users”集合中添加一条文档记录：

```py
    user = {'name':'Michael', 'age':18, 'scores':[{'course': 'Math', 'score': 76}]}
    mongo.db.users.insert_one(user)

```

如果”users”集合不存在，PyMongo 会自动创建。让我们打开 MongoDB 的控制台，查询下刚才添加的文档：

```py
> use flask
> db.users.find()

```

你应该会看到类似下面的信息：

```py
{ "_id" : ObjectId("56f00d13d35208259846a893"), "age" : 18, "name" : "Michael", 
"scores" : [ { "course" : "Math", "score" : 76 } ] }

```

MongoDB 会自动为文档记录创建一个主键”_id”，它的值是一个 uuid。你也可以在创建文档时获取这个值：

```py
    user = {'name':'Tom', 'age':21, 'scores':[{'course': 'Math', 'score': 85.5},
                                              {'course': 'Politics', 'score': 58}]}
    user_id = mongo.db.users.insert_one(user).inserted_id
    print 'Add user with id: %s' % user_id

```

“mongo.db.users”用来获取名为”users”集合对象，类型是”pymongo.collection.Collection”，该对象上的”insert_one()”方法用来创建一条记录。相应的，集合对象上的”insert_many()”方法可以同时创建多条记录，比如：

```py
    result = mongo.db.tests.insert_many([{'num': i} for i in range(3)])
    print result.inserted_ids

```

查询下 tests 的集合，你会看到类似下面的信息：

```py
{ "_id" : ObjectId("56f01209d3520825eee9844c"), "num" : 0 }
{ "_id" : ObjectId("56f01209d3520825eee9844d"), "num" : 1 }
{ "_id" : ObjectId("56f01209d3520825eee9844e"), "num" : 2 }

```

### 查询数据

集合对象提供了”find_one()”和”find()”方法分别用来获取一条和多条文档记录，两个方法都可以传入查询条件作为参数：

```py
@app.route('/user')
@app.route('/user/<string:name>')
def user(name=None):
    if name is None:
        users = mongo.db.users.find()
        return render_template('users.html', users=users)
    else:
        user = mongo.db.users.find_one({'name': name})
        if user is not None:
            return render_template('users.html', users=[user])
        else:
            return 'No user found!'

```

上例中的模板文件”users.html”如下：

```py
<!doctype html>
<title>PyMongo Sample</title>
<h1>Users:</h1>
<ul>
{% for user in users %}
    <li>{{ user.name }}, {{ user.age }}</li>
    <ul>
    {% for score in user.scores %}
        <li>{{ score.course }}, {{ score.score }}</li>
    {% endfor %}
    </ul>
{% endfor %}
</ul>

```

“find_one()”方法返回的就是一个字典，所以我们可以直接对其作操作。”find()”方法返回的其实是一个”pymongo.cursor.Cursor”对象，不过 Cursor 类实现了”__iter__()”和”next()”方法，因此可以用”for … in …”循环来遍历它。

Cursor 类还提供了很多功能接口来强化查询功能，这里列举一些常用的：

1.  “count()”方法, 获取返回数据集的大小

```py
    users = mongo.db.users.find({'age':{'$lt':20}})
    print users.count()    # 打印年龄小于 20 的用户个数

```

*   “sort()”方法, 排序

```py
    from flask.ext.pymongo import DESCENDING

    # 返回所有用户，并按名字升序排序
    users = mongo.db.users.find().sort('name')
    # 返回所有用户，并按年龄降序排序
    users = mongo.db.users.find().sort('age', DESCENDING)

```

*   “limit()”和”skip()”方法, 分页

```py
    # 最多只返回 5 条记录，并且忽略开始的 2 条
    # 即返回第三到第七（如果存在的话）条记录
    users = mongo.db.users.find().limit(5).skip(2)

```

*   “distinct()”方法, 获取某一字段的唯一值

```py
    ages = mongo.db.users.find().distinct('age')
    print ages    # 打印 [18, 21, 17]

```

注意，”distinct()”方法需传入字段名，它返回的是一个列表，而不是 Cursor 或文档。上例列出了’age’字段所有的唯一值。

更多对 Cursor 的操作可参阅 PyMongo 的[Cursor API 文档](http://api.mongodb.org/python/current/api/pymongo/cursor.html)。

### 更新数据

“pymongo.collection.Collection”提供了两种更新数据的方法，一种是 update，可以更新指定文档中某个字段的值，同关系型数据库中的 update 类似。update 有两个函数，”update_one()”更新一条记录，”update_many()”更新多条记录：

```py
    # 找到名为 Tom 的第一条记录，将其年龄加 3
    result = mongo.db.users.update_one({'name': 'Tom'}, {'$inc': {'age': 3}})
    # 打印被改动过的记录数
    print '%d records modified' % result.modified_count
    # 找到所有年龄小于 20 的用户记录，将其年龄设为 20
    result = mongo.db.users.update_many({'age':{'$lt':20}}, {'$set': {'age': 20}})
    # 打印被改动过的记录数
    print '%d records modified' % result.modified_count

```

另一种更新数据的方法是 replace，它不是用来更新某一字段，而是把整条记录替换掉。它就一个函数”replace_one()”：

```py
    user = {'name':'Lisa', 'age':23, 'scores':[{'course': 'Politics', 'score': 95}]}
    # 找到名为 Jane 的第一条记录，将其替换为上面的名为 Lisa 的记录
    result = mongo.db.users.replace_one({'name': 'Jane'}, user)
    # 打印被改动过的记录数
    print '%d records modified' % result.modified_count

```

### 删除数据

删除数据可以使用集合对象上的 delete 方法，它也有两个函数，”delete_one()”删除一条记录，”delete_many()”删除多条记录：

```py
    # 删除名为 Michael 的第一条记录
    result = mongo.db.users.delete_one({'name': 'Michael'})
    # 打印被删除的记录数
    print '%d records deleted' % result.deleted_count
    # 找到所有年龄大于 20 的用户记录
    result = mongo.db.users.delete_many({'age':{'$gt':20}})
    # 打印被删除的记录数
    print '%d records deleted' % result.deleted_count

```

如果你想将集合整个删除，可以使用”drop()”方法：

```py
    mongo.db.users.drop()

```

此后你在 MongoDB 控制台里输入命令”show tables”，将看不到这个”users”集合。

更多对集合中的数据操作可参阅 PyMongo 的[Collection API 文档](http://api.mongodb.org/python/current/api/pymongo/collection.html)。

### 练习：PyMongo 结合 Restful

我们来做个小练习，在扩展系列第一篇中我们介绍过 Flask-Restful 的实现，并且让大家做了练习将 Restful 同数据库集成。现在让我们把数据库改为 MongoDB，使用上面介绍的 Flask-PyMongo 来实现。下面是参考代码：

```py
from flask import Flask, request
from flask.ext.restful import Api, Resource
from flask.ext.pymongo import PyMongo

app = Flask(__name__)
app.config['MONGO_URI']='mongodb://localhost:27017/flask'

api = Api(app)
mongo = PyMongo(app)

class User(Resource):
    def get(self, name):
        user = mongo.db.users.find_one({'name': name})
        if user is not None:
            user.pop('_id')
            return dict(result='success', user=user)
        return dict(result='error', message='No record found')

    def delete(self, name):
        result = mongo.db.users.delete_one({'name': name})
        count = result.deleted_count
        if count > 0:
            return dict(result='success', message='%d records deleted' % count)
        return dict(result='error', message='Failed to delete')

    def put(self, name):
        user = request.get_json()
        result = mongo.db.users.replace_one({'name': 'name'}, user)
        count = result.modified_count
        if count > 0:
            return dict(result='success', message='%d records modified' % count)
        return dict(result='error', message='Failed to modify')

class UserList(Resource):
    def get(self):
        users = mongo.db.users.find()
        user_list = []
        for user in users:
            user.pop('_id')
            user_list.append(user)
        return dict(result='success', userlist=user_list)

    def post(self):
        user = request.get_json()
        user_id = mongo.db.users.insert_one(user).inserted_id
        if user_id is not None:
            return dict(result='success', message='1 record added')
        return dict(result='error', message='Failed to insert')

api.add_resource(UserList, '/users')
api.add_resource(User, '/users/<name>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

```

这里我们使用 name 作为键值来查询，因为 MongoDB 中的 id 太复杂。注意，我们在输出时都会把”_id”字段去掉，因为它是”ObjectId”类型无法 JSON 序列化，同时如果你的数据中有日期时间类型，也要特别处理后才能被 JSON 序列化。

#### 更多参考资料

[PyMongo 的官方文档](http://api.mongodb.org/python/current/)
[Flask-PyMongo 的官方文档](https://flask-pymongo.readthedocs.org/en/latest/)
[Flask-PyMongo 的源码](https://github.com/dcrosta/flask-pymongo/)

本篇的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ext5.html)