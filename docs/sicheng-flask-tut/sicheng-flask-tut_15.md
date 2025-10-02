# Flask 扩展系列(一)–Restful

看过入门系列的朋友们一定已经被 Flask 的简洁之美感染到了吧。其实 Flask 不仅是一个 Python Web 框架，更是一个开源的生态圈。在基础框架之外，Flask 拥有丰富的扩展(Extension)来其扩充功能，这些扩展有的来自官方，有的来自第三方。这一系列会给大家介绍一些 Flask 常用的扩展及其使用方法。

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

### Flask 扩展

你可以在[Flask 的官网](http://flask.pocoo.org/extensions/)上寻找你想要的扩展，每个扩展都有其文档链接和 Github 上的源码链接。扩展可以通过”pip install”来安装。使用扩展也很简单，一般就是通过 import 导入扩展包，然后就可以像普通 python 包一样调用了。下一个部分我们就会拿 Flask 的 Restful 扩展来举个例子。作为开发人员，你还可以自己开发 Flask 扩展。这篇中我们就不细述了。

### Flask-RESTful 扩展

在入门系列第二篇路由中，我们了解到 Flask 路由可以指定 HTTP 请求方法，并在请求函数中根据不同的请求方法，执行不同的逻辑。这样实现一个 Restful 的请求已经相当简单了（不熟悉 Restful？先去恶补下）。但是 Flask 还有更简便的方法，就是其 Flask-RESTful 扩展。首先，我们来安装这个扩展：

```py
$ pip install Flask-RESTful
```

安装完后，你就可以在代码中导入该扩展包”flask.ext.restful”。让我们来看个例子：

```py
from flask import Flask, request
from flask.ext.restful import Api, Resource

app = Flask(__name__)
api = Api(app)

USER_LIST = {
    '1': {'name':'Michael'},
    '2': {'name':'Tom'},
}

class UserList(Resource):
    def get(self):
        return USER_LIST

    def post(self):
        user_id = int(max(USER_LIST.keys())) + 1
        user_id = '%i' % user_id
        USER_LIST[user_id] = {'name': request.form['name']}
        return USER_LIST[user_id]

api.add_resource(UserList, '/users')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

```

这个例子很容易理解，Restful 扩展通过”api.add_resource()”方法来添加路由，方法的第一个参数是一个类名，该类继承”Resource”基类，其成员函数定义了不同的 HTTP 请求方法的逻辑；第二个参数定义了 URL 路径。运行上面的例子并访问”http://localhost:5000/users”，GET 请求时会列出全局变量”USER_LIST”中的内容，POST 请求时会在”USER_LIST”中添加一项，并返回刚添加的项。如果在 POST 请求中找不到”name”字段，则返回”400 Bad Request”错误。由于类”UserList”没有定义”put”和”delete”函数，所以在 PUT 或 DELETE 请求时会返回”405 Method Not Allowed”错误。

另外，路由支持多路径，比如：

```py
api.add_resource(UserList, '/userlist', '/users')

```

这样访问”http://localhost:5000/userlist”和”http://localhost:5000/users”的效果完全一样。

#### 带参数的请求

上面的例子请求是针对 user 列表的，如果我们要对某个具体的 user 做操作，就需要传递具体的”user_id”了。这时候，我们需要路由支持带参数。Flask-RESTful 的实现同 Flask 一样，就是在路由中加上参数变量即可。我们看下例子：

```py
class User(Resource):
    def get(self, user_id):
        return USER_LIST[user_id]

    def delete(self, user_id):
        del USER_LIST[user_id]
        return ''

    def put(self, user_id):
        USER_LIST[user_id] = {'name': request.form['name']}
        return USER_LIST[user_id]

api.add_resource(User, '/users/<user_id>')

```

在”api.add_resource()”的第二个参数路径中加上 URL 参数变量即可，格式同入门系列第二篇 Flask 路由中完全一样，也支持转换器来转换变量类型。此外，在 User 类的 GET，POST，PUT 等成员函数中，记得加上参数”user_id”来获取传入的变量值。

#### 参数解析

在 POST 或 PUT 请求中，直接访问 form 表单并验证的工作有些麻烦。Flask-RESTful 提供了”reqparse”库来简化。我们来改进下上例中的 PUT 函数：

```py
from flask.ext.restful import reqparse

parser = reqparse.RequestParser()
parser.add_argument('name', type=str)

class User(Resource):
    def put(self, user_id):
        args = parser.parse_args()
        USER_LIST[user_id] = {'name': args['name']}
        return USER_LIST[user_id]

```

你可以通过”parser.add_argument()”方法来定义 form 表单字段，并指定其类型（本例中是字符型 str）。然后在 PUT 函数中，就可以调用”parser.parse_args()”来获取表单内容，并返回一个字典，该字典就包含了表单的内容。”parser.parse_args()”方法会自动验证数据类型，并在类型不匹配时，返回 400 错误。你还可以添加”strict”参数，如”parser.parse_args(strict=True)”，此时如果请求中出现未定义的参数，也会返回 400 错误。

#### 示例代码

结合上述的内容，我们来看一个完整的例子：

```py
from flask import Flask
from flask.ext.restful import Api, Resource, reqparse, abort

app = Flask(__name__)
api = Api(app)

USER_LIST = {
    1: {'name':'Michael'},
    2: {'name':'Tom'},
}

parser = reqparse.RequestParser()
parser.add_argument('name', type=str)

def abort_if_not_exist(user_id):
    if user_id not in USER_LIST:
        abort(404, message="User {} doesn't exist".format(user_id))

class User(Resource):
    def get(self, user_id):
        abort_if_not_exist(user_id)
        return USER_LIST[user_id]

    def delete(self, user_id):
        abort_if_not_exist(user_id)
        del USER_LIST[user_id]
        return '', 204

    def put(self, user_id):
        args = parser.parse_args(strict=True)
        USER_LIST[user_id] = {'name': args['name']}
        return USER_LIST[user_id], 201

class UserList(Resource):
    def get(self):
        return USER_LIST

    def post(self):
        args = parser.parse_args(strict=True)
        user_id = int(max(USER_LIST.keys())) + 1
        USER_LIST[user_id] = {'name': args['name']}
        return USER_LIST[user_id], 201

api.add_resource(UserList, '/users')
api.add_resource(User, '/users/<int:user_id>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

```

代码就不解释了，留给读者一个作业，请结合入门系列第六篇数据库集成的例子，写一个数据库表 CRUD 的 Restful 应用吧。

本例中的代码及数据库集成的代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ext1.html)