# 使用 Flask 的基于令牌的认证

> 原文：<https://realpython.com/token-based-authentication-with-flask/>

本教程采用测试优先的方法，使用 JSON Web 令牌(jwt)在 Flask 应用程序中实现基于令牌的认证。

**更新:**

*   *08/04/2017* :为 [PyBites 挑战赛](https://pybit.es/codechallenge30.html)重构路线处理程序。

## 目标

本教程结束时，您将能够…

1.  讨论使用 jwt 与会话和 cookies 进行身份验证的优势
2.  用 JWTs 实现用户认证
3.  必要时将用户令牌列入黑名单
4.  [编写测试](https://realpython.com/python-testing/)来创建和验证 jwt 和用户认证
5.  实践测试驱动的开发

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

[*Remove ads*](/account/join/)

## 简介

JSON Web 令牌(或 JWTs)提供了一种从客户端向服务器传输信息的方式，这是一种安全的无状态的[方式。](https://en.wikipedia.org/wiki/Stateless_protocol)

在服务器上，jwt 是通过使用秘密密钥对用户信息进行签名而生成的，然后安全地存储在客户机上。这种形式的身份验证与现代的单页面应用程序配合得很好。有关这方面的更多信息，以及使用 JWTs 与会话和基于 cookie 的身份验证的优缺点，请查看以下文章:

1.  [饼干 vs 代币:权威指南](https://auth0.com/blog/cookies-vs-tokens-definitive-guide/)
2.  [令牌认证与 cookie](http://stackoverflow.com/questions/17000835/token-authentication-vs-cookies)
3.  在 Flask 中会话是如何工作的？

> **注意:**请记住，由于 JWT 是由[签名的，而不是加密的](http://stackoverflow.com/questions/454048/what-is-the-difference-between-encrypting-and-signing-in-asymmetric-encryption)，它不应该包含像用户密码这样的敏感信息。

## 开始使用

理论够了，开始实现一些代码吧！

### 项目设置

首先克隆项目样板文件，然后创建一个新的分支:

```py
$ git clone https://github.com/realpython/flask-jwt-auth.git
$ cd flask-jwt-auth
$ git checkout tags/1.0.0 -b jwt-auth
```

创建并激活 virtualenv 并安装依赖项:

```py
$ python3.6 -m venv env
$ source env/bin/activate
(env)$ pip install -r requirements.txt
```

这是可选的，但是创建一个新的 Github 存储库并更新 remote 是个好主意:

```py
(env)$ git remote set-url origin <newurl>
```

### 数据库设置

让我们设置 Postgres。

> **注意**:如果你在苹果电脑上，看看[的 Postgres 应用](http://postgresapp.com/)。

一旦本地 Postgres 服务器运行，从`psql`创建两个新的数据库，它们与您的项目名称同名:

```py
(env)$  psql #  create  database  flask_jwt_auth; CREATE  DATABASE #  create  database  flask_jwt_auth_test; CREATE  DATABASE #  \q
```

> **注意**:根据您的 Postgres 版本，上述创建数据库的命令可能会有一些变化。检查 [Postgres 文档](https://www.postgresql.org/docs/9.6/static/sql-createdatabase.html)中的正确命令。

在应用数据库迁移之前，我们需要更新位于 *project/server/config.py* 中的配置文件。简单更新一下`database_name`:

```py
database_name = 'flask_jwt_auth'
```

在终端中设置环境变量:

```py
(env)$ export APP_SETTINGS="project.server.config.DevelopmentConfig"
```

更新*project/tests/test _ _ config . py*中的以下测试:

```py
class TestDevelopmentConfig(TestCase):
    def create_app(self):
        app.config.from_object('project.server.config.DevelopmentConfig')
        return app

    def test_app_is_development(self):
        self.assertTrue(app.config['DEBUG'] is True)
        self.assertFalse(current_app is None)
        self.assertTrue(
            app.config['SQLALCHEMY_DATABASE_URI'] == 'postgresql://postgres:@localhost/flask_jwt_auth'
        )

class TestTestingConfig(TestCase):
    def create_app(self):
        app.config.from_object('project.server.config.TestingConfig')
        return app

    def test_app_is_testing(self):
        self.assertTrue(app.config['DEBUG'])
        self.assertTrue(
            app.config['SQLALCHEMY_DATABASE_URI'] == 'postgresql://postgres:@localhost/flask_jwt_auth_test'
        )
```

运行它们以确保它们仍然通过:

```py
(env)$ python manage.py test
```

您应该看到:

```py
test_app_is_development (test__config.TestDevelopmentConfig) ... ok
test_app_is_production (test__config.TestProductionConfig) ... ok
test_app_is_testing (test__config.TestTestingConfig) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.007s

OK
```

[*Remove ads*](/account/join/)

### 迁移

在“服务器”目录中添加一个 *models.py* 文件:

```py
# project/server/models.py

import datetime

from project.server import app, db, bcrypt

class User(db.Model):
    """ User Model for storing user related details """
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    registered_on = db.Column(db.DateTime, nullable=False)
    admin = db.Column(db.Boolean, nullable=False, default=False)

    def __init__(self, email, password, admin=False):
        self.email = email
        self.password = bcrypt.generate_password_hash(
            password, app.config.get('BCRYPT_LOG_ROUNDS')
        ).decode()
        self.registered_on = datetime.datetime.now()
        self.admin = admin
```

在上面的代码片段中，我们定义了一个基本的用户模型，它使用 [Flask-Bcrypt](http://flask-bcrypt.readthedocs.io/en/0.7.1/) 扩展来散列密码。

安装 [psycopg2](http://initd.org/psycopg/) 连接到 Postgres:

```py
(env)$ pip install psycopg2==2.6.2
(env)$ pip freeze > requirements.txt
```

在 *manage.py* 内更改-

```py
from project.server import app, db
```

到

```py
from project.server import app, db, models
```

应用迁移:

```py
(env)$ python manage.py create_db
(env)$ python manage.py db init
(env)$ python manage.py db migrate
```

## 健全性检查

成功了吗？

```py
(env)$  psql #  \c  flask_jwt_auth You  are  now  connected  to  database  "flask_jwt_auth"  as  user  "michael.herman". #  \d List  of  relations Schema  |  Name  |  Type  |  Owner --------+-----------------+----------+----------
  public  |  alembic_version  |  table  |  postgres public  |  users  |  table  |  postgres public  |  users_id_seq  |  sequence  |  postgres (3  rows)
```

## JWT 设置

身份验证工作流的工作方式如下:

*   客户端提供[电子邮件](https://realpython.com/python-send-email/)和密码，发送给服务器
*   然后，服务器验证电子邮件和密码是否正确，并使用一个身份验证令牌进行响应
*   客户端存储令牌，并将其与所有后续请求一起发送给 API
*   服务器解码令牌并验证它

这个循环重复进行，直到令牌过期或被撤销。在后一种情况下，服务器会发出一个新的令牌。

令牌本身分为三个部分:

*   页眉
*   有效载荷
*   签名

我们将更深入地研究有效负载，但是如果您有兴趣，您可以从 JSON Web Tokens 的文章[中阅读关于每个部分的更多内容。](https://jwt.io/introduction/)

要在我们的应用程序中使用 JSON Web 令牌，请安装 [PyJWT](http://pyjwt.readthedocs.io/en/latest/) 包:

```py
(env)$ pip install pyjwt==1.4.2
(env)$ pip freeze > requirements.txt
```

[*Remove ads*](/account/join/)

### 编码令牌

将下面的方法添加到*项目/服务器/模型. py* 中的`User()`类中:

```py
def encode_auth_token(self, user_id):
    """
 Generates the Auth Token
 :return: string
 """
    try:
        payload = {
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=0, seconds=5),
            'iat': datetime.datetime.utcnow(),
            'sub': user_id
        }
        return jwt.encode(
            payload,
            app.config.get('SECRET_KEY'),
            algorithm='HS256'
        )
    except Exception as e:
        return e
```

不要忘记添加导入:

```py
import jwt
```

因此，给定一个用户 id，这个方法从有效负载和在 *config.py* 文件中设置的密钥创建并返回一个令牌。负载是我们添加关于令牌的元数据和关于用户的信息的地方。这些信息通常被称为 JWT 声称的。我们利用以下“声明”:

*   `exp`:令牌到期日期
*   `iat`:令牌生成的时间
*   `sub`:令牌的主题(它标识的用户)

秘密密钥*必须*是随机的，并且只能在服务器端访问。使用 Python 解释器生成密钥:

>>>

```py
>>> import os
>>> os.urandom(24)
b"\xf9'\xe4p(\xa9\x12\x1a!\x94\x8d\x1c\x99l\xc7\xb7e\xc7c\x86\x02MJ\xa0"
```

将密钥设置为环境变量:

```py
(env)$ export SECRET_KEY="\xf9'\xe4p(\xa9\x12\x1a!\x94\x8d\x1c\x99l\xc7\xb7e\xc7c\x86\x02MJ\xa0"
```

将此键添加到*项目/服务器/配置文件*中`BaseConfig()`类内的`SECRET_KEY`:

```py
SECRET_KEY = os.getenv('SECRET_KEY', 'my_precious')
```

更新*project/tests/test _ _ config . py*中的测试，以确保变量设置正确:

```py
def test_app_is_development(self):
    self.assertFalse(app.config['SECRET_KEY'] is 'my_precious')
    self.assertTrue(app.config['DEBUG'] is True)
    self.assertFalse(current_app is None)
    self.assertTrue(
        app.config['SQLALCHEMY_DATABASE_URI'] == 'postgresql://postgres:@localhost/flask_jwt_auth'
    )

class TestTestingConfig(TestCase):
    def create_app(self):
        app.config.from_object('project.server.config.TestingConfig')
        return app

    def test_app_is_testing(self):
        self.assertFalse(app.config['SECRET_KEY'] is 'my_precious')
        self.assertTrue(app.config['DEBUG'])
        self.assertTrue(
            app.config['SQLALCHEMY_DATABASE_URI'] == 'postgresql://postgres:@localhost/flask_jwt_auth_test'
        )
```

在继续之前，让我们为用户模型编写一个快速的单元测试。将以下代码添加到“项目/测试”中名为 *test_user_model.py* 的新文件中:

```py
# project/tests/test_user_model.py

import unittest

from project.server import db
from project.server.models import User
from project.tests.base import BaseTestCase

class TestUserModel(BaseTestCase):

    def test_encode_auth_token(self):
        user = User(
            email='test@test.com',
            password='test'
        )
        db.session.add(user)
        db.session.commit()
        auth_token = user.encode_auth_token(user.id)
        self.assertTrue(isinstance(auth_token, bytes))

if __name__ == '__main__':
    unittest.main()
```

进行测试。他们都应该通过。

### 解码令牌

类似地，要解码一个令牌，将下面的方法添加到`User()`类中:

```py
@staticmethod
def decode_auth_token(auth_token):
    """
 Decodes the auth token
 :param auth_token:
 :return: integer|string
 """
    try:
        payload = jwt.decode(auth_token, app.config.get('SECRET_KEY'))
        return payload['sub']
    except jwt.ExpiredSignatureError:
        return 'Signature expired. Please log in again.'
    except jwt.InvalidTokenError:
        return 'Invalid token. Please log in again.'
```

我们需要对每个 API 请求的 auth 令牌进行解码，并验证其签名，以确保用户的真实性。为了验证`auth_token`，我们使用了与编码令牌相同的`SECRET_KEY`。

如果`auth_token`有效，我们从有效载荷的`sub`索引中获取用户 id。如果无效，可能有两种例外情况:

1.  过期签名:当令牌过期后被使用时，它抛出一个`ExpiredSignatureError`异常。这意味着有效载荷的`exp`字段中指定的时间已经过期。
2.  无效令牌:当提供的令牌不正确或格式不正确时，就会引发一个`InvalidTokenError`异常。

> **注意:**我们使用了一个[静态方法](https://docs.python.org/3.6/library/functions.html#staticmethod)，因为它与类的实例无关。

向 *test_user_model.py* 添加一个测试:

```py
def test_decode_auth_token(self):
    user = User(
        email='test@test.com',
        password='test'
    )
    db.session.add(user)
    db.session.commit()
    auth_token = user.encode_auth_token(user.id)
    self.assertTrue(isinstance(auth_token, bytes))
    self.assertTrue(User.decode_auth_token(auth_token) == 1)
```

确保在继续之前通过测试。

> **注意:**我们稍后将通过将无效令牌列入黑名单来处理它们。

[*Remove ads*](/account/join/)

## 路线设置

现在，我们可以使用测试优先的方法来配置授权路由:

*   `/auth/register`
*   `/auth/login`
*   `/auth/logout`
*   `/auth/user`

首先在“项目/服务器”中创建一个名为“auth”的新文件夹。然后，在“auth”内添加两个文件， *__init__。py* 和*视图。最后，将以下代码添加到 *views.py* :*

```py
# project/server/auth/views.py

from flask import Blueprint, request, make_response, jsonify
from flask.views import MethodView

from project.server import bcrypt, db
from project.server.models import User

auth_blueprint = Blueprint('auth', __name__)
```

要在应用程序中注册新的[蓝图](https://realpython.com/flask-blueprint/)，请将以下内容添加到*项目/服务器/__init__ 的底部。py* :

```py
from project.server.auth.views import auth_blueprint
app.register_blueprint(auth_blueprint)
```

现在，在“project/tests”中添加一个名为 *test_auth.py* 的新文件来保存我们对这个蓝图的所有测试:

```py
# project/tests/test_auth.py

import unittest

from project.server import db
from project.server.models import User
from project.tests.base import BaseTestCase

class TestAuthBlueprint(BaseTestCase):
    pass

if __name__ == '__main__':
    unittest.main()
```

## 注册路线

从一个测试开始:

```py
def test_registration(self):
    """ Test for user registration """
    with self.client:
        response = self.client.post(
            '/auth/register',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json'
        )
        data = json.loads(response.data.decode())
        self.assertTrue(data['status'] == 'success')
        self.assertTrue(data['message'] == 'Successfully registered.')
        self.assertTrue(data['auth_token'])
        self.assertTrue(response.content_type == 'application/json')
        self.assertEqual(response.status_code, 201)
```

确保添加导入:

```py
import json
```

进行测试。您应该会看到以下错误:

```py
raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

现在，让我们编写通过测试的代码。将以下内容添加到*project/server/auth/views . py*中:

```py
class RegisterAPI(MethodView):
    """
 User Registration Resource
 """

    def post(self):
        # get the post data
        post_data = request.get_json()
        # check if user already exists
        user = User.query.filter_by(email=post_data.get('email')).first()
        if not user:
            try:
                user = User(
                    email=post_data.get('email'),
                    password=post_data.get('password')
                )

                # insert the user
                db.session.add(user)
                db.session.commit()
                # generate the auth token
                auth_token = user.encode_auth_token(user.id)
                responseObject = {
                    'status': 'success',
                    'message': 'Successfully registered.',
                    'auth_token': auth_token.decode()
                }
                return make_response(jsonify(responseObject)), 201
            except Exception as e:
                responseObject = {
                    'status': 'fail',
                    'message': 'Some error occurred. Please try again.'
                }
                return make_response(jsonify(responseObject)), 401
        else:
            responseObject = {
                'status': 'fail',
                'message': 'User already exists. Please Log in.',
            }
            return make_response(jsonify(responseObject)), 202

# define the API resources
registration_view = RegisterAPI.as_view('register_api')

# add Rules for API Endpoints
auth_blueprint.add_url_rule(
    '/auth/register',
    view_func=registration_view,
    methods=['POST']
)
```

这里，我们注册了一个新用户，并为进一步的请求生成了一个新的 auth token，我们将它发送回客户端。

运行测试以确保它们全部通过:

```py
Ran 6 tests in 0.132s

OK
```

接下来，让我们再添加一个测试，以确保在用户已经存在的情况下注册失败:

```py
def test_registered_with_already_registered_user(self):
    """ Test registration with already registered email"""
    user = User(
        email='joe@gmail.com',
        password='test'
    )
    db.session.add(user)
    db.session.commit()
    with self.client:
        response = self.client.post(
            '/auth/register',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json'
        )
        data = json.loads(response.data.decode())
        self.assertTrue(data['status'] == 'fail')
        self.assertTrue(
            data['message'] == 'User already exists. Please Log in.')
        self.assertTrue(response.content_type == 'application/json')
        self.assertEqual(response.status_code, 202)
```

在进入下一条路线之前，再次进行测试。一切都会过去。

[*Remove ads*](/account/join/)

## 登录路线

再次，从一个测试开始。为了验证登录 API，让我们测试两种情况:

1.  注册用户登录
2.  非注册用户登录

### 注册用户登录

```py
def test_registered_user_login(self):
    """ Test for login of registered-user login """
    with self.client:
        # user registration
        resp_register = self.client.post(
            '/auth/register',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json',
        )
        data_register = json.loads(resp_register.data.decode())
        self.assertTrue(data_register['status'] == 'success')
        self.assertTrue(
            data_register['message'] == 'Successfully registered.'
        )
        self.assertTrue(data_register['auth_token'])
        self.assertTrue(resp_register.content_type == 'application/json')
        self.assertEqual(resp_register.status_code, 201)
        # registered user login
        response = self.client.post(
            '/auth/login',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json'
        )
        data = json.loads(response.data.decode())
        self.assertTrue(data['status'] == 'success')
        self.assertTrue(data['message'] == 'Successfully logged in.')
        self.assertTrue(data['auth_token'])
        self.assertTrue(response.content_type == 'application/json')
        self.assertEqual(response.status_code, 200)
```

在这个测试用例中，注册用户试图登录，正如所料，我们的应用程序应该允许这样做。

进行测试。他们应该失败。现在编写代码:

```py
class LoginAPI(MethodView):
    """
 User Login Resource
 """
    def post(self):
        # get the post data
        post_data = request.get_json()
        try:
            # fetch the user data
            user = User.query.filter_by(
                email=post_data.get('email')
              ).first()
            auth_token = user.encode_auth_token(user.id)
            if auth_token:
                responseObject = {
                    'status': 'success',
                    'message': 'Successfully logged in.',
                    'auth_token': auth_token.decode()
                }
                return make_response(jsonify(responseObject)), 200
        except Exception as e:
            print(e)
            responseObject = {
                'status': 'fail',
                'message': 'Try again'
            }
            return make_response(jsonify(responseObject)), 500
```

不要忘记[将类转换成视图函数](http://flask.pocoo.org/docs/0.12/views/#pluggable-views):

```py
# define the API resources
registration_view = RegisterAPI.as_view('register_api')
login_view = LoginAPI.as_view('login_api')

# add Rules for API Endpoints
auth_blueprint.add_url_rule(
    '/auth/register',
    view_func=registration_view,
    methods=['POST']
)
auth_blueprint.add_url_rule(
    '/auth/login',
    view_func=login_view,
    methods=['POST']
)
```

再次运行测试。他们通过了吗？他们应该。在所有测试通过之前，不要继续前进。

### 非注册用户登录

添加测试:

```py
def test_non_registered_user_login(self):
    """ Test for login of non-registered user """
    with self.client:
        response = self.client.post(
            '/auth/login',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json'
        )
        data = json.loads(response.data.decode())
        self.assertTrue(data['status'] == 'fail')
        self.assertTrue(data['message'] == 'User does not exist.')
        self.assertTrue(response.content_type == 'application/json')
        self.assertEqual(response.status_code, 404)
```

在这种情况下，一个未注册的用户试图登录，正如所料，我们的应用程序不应该允许这样做。

运行测试，然后更新代码:

```py
class LoginAPI(MethodView):
    """
 User Login Resource
 """
    def post(self):
        # get the post data
        post_data = request.get_json()
        try:
            # fetch the user data
            user = User.query.filter_by(
                email=post_data.get('email')
            ).first()
            if user and bcrypt.check_password_hash(
                user.password, post_data.get('password')
            ):
                auth_token = user.encode_auth_token(user.id)
                if auth_token:
                    responseObject = {
                        'status': 'success',
                        'message': 'Successfully logged in.',
                        'auth_token': auth_token.decode()
                    }
                    return make_response(jsonify(responseObject)), 200
            else:
                responseObject = {
                    'status': 'fail',
                    'message': 'User does not exist.'
                }
                return make_response(jsonify(responseObject)), 404
        except Exception as e:
            print(e)
            responseObject = {
                'status': 'fail',
                'message': 'Try again'
            }
            return make_response(jsonify(responseObject)), 500
```

我们改变了什么？测试通过了吗？邮件正确但密码不正确怎么办？会发生什么？为此写一个测试！

## 用户状态路线

为了获得当前登录用户的用户详细信息，auth 令牌必须与请求一起在报头中发送。

从一个测试开始:

```py
def test_user_status(self):
    """ Test for user status """
    with self.client:
        resp_register = self.client.post(
            '/auth/register',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json'
        )
        response = self.client.get(
            '/auth/status',
            headers=dict(
                Authorization='Bearer ' + json.loads(
                    resp_register.data.decode()
                )['auth_token']
            )
        )
        data = json.loads(response.data.decode())
        self.assertTrue(data['status'] == 'success')
        self.assertTrue(data['data'] is not None)
        self.assertTrue(data['data']['email'] == 'joe@gmail.com')
        self.assertTrue(data['data']['admin'] is 'true' or 'false')
        self.assertEqual(response.status_code, 200)
```

测试应该会失败。现在，在处理程序类中，我们应该:

*   提取身份验证令牌并检查其有效性
*   从有效负载中获取用户 id 并获得用户详细信息(当然，如果令牌有效的话)

```py
class UserAPI(MethodView):
    """
 User Resource
 """
    def get(self):
        # get the auth token
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            auth_token = ''
        if auth_token:
            resp = User.decode_auth_token(auth_token)
            if not isinstance(resp, str):
                user = User.query.filter_by(id=resp).first()
                responseObject = {
                    'status': 'success',
                    'data': {
                        'user_id': user.id,
                        'email': user.email,
                        'admin': user.admin,
                        'registered_on': user.registered_on
                    }
                }
                return make_response(jsonify(responseObject)), 200
            responseObject = {
                'status': 'fail',
                'message': resp
            }
            return make_response(jsonify(responseObject)), 401
        else:
            responseObject = {
                'status': 'fail',
                'message': 'Provide a valid auth token.'
            }
            return make_response(jsonify(responseObject)), 401
```

因此，如果令牌有效且未过期，我们将从令牌的有效负载中获取用户 id，然后使用它从数据库中获取用户数据。

> **注意:**我们仍然需要检查令牌是否被列入黑名单。我们很快就会谈到这一点。

确保添加:

```py
user_view = UserAPI.as_view('user_api')
```

并且:

```py
auth_blueprint.add_url_rule(
    '/auth/status',
    view_func=user_view,
    methods=['GET']
)
```

测试应该通过:

```py
Ran 10 tests in 0.240s

OK
```

还有一条路要走！

[*Remove ads*](/account/join/)

## 注销路由测试

测试有效注销:

```py
def test_valid_logout(self):
    """ Test for logout before token expires """
    with self.client:
        # user registration
        resp_register = self.client.post(
            '/auth/register',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json',
        )
        data_register = json.loads(resp_register.data.decode())
        self.assertTrue(data_register['status'] == 'success')
        self.assertTrue(
            data_register['message'] == 'Successfully registered.')
        self.assertTrue(data_register['auth_token'])
        self.assertTrue(resp_register.content_type == 'application/json')
        self.assertEqual(resp_register.status_code, 201)
        # user login
        resp_login = self.client.post(
            '/auth/login',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json'
        )
        data_login = json.loads(resp_login.data.decode())
        self.assertTrue(data_login['status'] == 'success')
        self.assertTrue(data_login['message'] == 'Successfully logged in.')
        self.assertTrue(data_login['auth_token'])
        self.assertTrue(resp_login.content_type == 'application/json')
        self.assertEqual(resp_login.status_code, 200)
        # valid token logout
        response = self.client.post(
            '/auth/logout',
            headers=dict(
                Authorization='Bearer ' + json.loads(
                    resp_login.data.decode()
                )['auth_token']
            )
        )
        data = json.loads(response.data.decode())
        self.assertTrue(data['status'] == 'success')
        self.assertTrue(data['message'] == 'Successfully logged out.')
        self.assertEqual(response.status_code, 200)
```

在第一个测试中，我们注册了一个新用户，让他们登录，然后尝试在令牌过期之前让他们注销。

测试无效注销:

```py
def test_invalid_logout(self):
    """ Testing logout after the token expires """
    with self.client:
        # user registration
        resp_register = self.client.post(
            '/auth/register',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json',
        )
        data_register = json.loads(resp_register.data.decode())
        self.assertTrue(data_register['status'] == 'success')
        self.assertTrue(
            data_register['message'] == 'Successfully registered.')
        self.assertTrue(data_register['auth_token'])
        self.assertTrue(resp_register.content_type == 'application/json')
        self.assertEqual(resp_register.status_code, 201)
        # user login
        resp_login = self.client.post(
            '/auth/login',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json'
        )
        data_login = json.loads(resp_login.data.decode())
        self.assertTrue(data_login['status'] == 'success')
        self.assertTrue(data_login['message'] == 'Successfully logged in.')
        self.assertTrue(data_login['auth_token'])
        self.assertTrue(resp_login.content_type == 'application/json')
        self.assertEqual(resp_login.status_code, 200)
        # invalid token logout
        time.sleep(6)
        response = self.client.post(
            '/auth/logout',
            headers=dict(
                Authorization='Bearer ' + json.loads(
                    resp_login.data.decode()
                )['auth_token']
            )
        )
        data = json.loads(response.data.decode())
        self.assertTrue(data['status'] == 'fail')
        self.assertTrue(
            data['message'] == 'Signature expired. Please log in again.')
        self.assertEqual(response.status_code, 401)
```

像上一个测试一样，我们注册一个用户，让他们登录，然后尝试让他们注销。在这种情况下，令牌无效，因为它已经过期。

添加导入:

```py
import time
```

现在，代码必须:

1.  验证身份验证令牌
2.  将令牌列入黑名单(当然，如果有效的话)

在编写路由处理程序之前，让我们为黑名单令牌创建一个新模型…

## 黑名单

将以下代码添加到*项目/服务器/模型. py* 中:

```py
class BlacklistToken(db.Model):
    """
 Token Model for storing JWT tokens
 """
    __tablename__ = 'blacklist_tokens'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    token = db.Column(db.String(500), unique=True, nullable=False)
    blacklisted_on = db.Column(db.DateTime, nullable=False)

    def __init__(self, token):
        self.token = token
        self.blacklisted_on = datetime.datetime.now()

    def __repr__(self):
        return '<id: token: {}'.format(self.token)
```

然后创建并应用迁移。完成后，您的数据库应该包含以下表格:

```py
Schema  |  Name  |  Type  |  Owner --------+-------------------------+----------+----------
public  |  alembic_version  |  table  |  postgres public  |  blacklist_tokens  |  table  |  postgres public  |  blacklist_tokens_id_seq  |  sequence  |  postgres public  |  users  |  table  |  postgres public  |  users_id_seq  |  sequence  |  postgres (5  rows)
```

这样，我们可以添加注销处理程序…

## 注销路由处理程序

更新视图:

```py
class LogoutAPI(MethodView):
    """
 Logout Resource
 """
    def post(self):
        # get auth token
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            auth_token = ''
        if auth_token:
            resp = User.decode_auth_token(auth_token)
            if not isinstance(resp, str):
                # mark the token as blacklisted
                blacklist_token = BlacklistToken(token=auth_token)
                try:
                    # insert the token
                    db.session.add(blacklist_token)
                    db.session.commit()
                    responseObject = {
                        'status': 'success',
                        'message': 'Successfully logged out.'
                    }
                    return make_response(jsonify(responseObject)), 200
                except Exception as e:
                    responseObject = {
                        'status': 'fail',
                        'message': e
                    }
                    return make_response(jsonify(responseObject)), 200
            else:
                responseObject = {
                    'status': 'fail',
                    'message': resp
                }
                return make_response(jsonify(responseObject)), 401
        else:
            responseObject = {
                'status': 'fail',
                'message': 'Provide a valid auth token.'
            }
            return make_response(jsonify(responseObject)), 403

# define the API resources
registration_view = RegisterAPI.as_view('register_api')
login_view = LoginAPI.as_view('login_api')
user_view = UserAPI.as_view('user_api')
logout_view = LogoutAPI.as_view('logout_api')

# add Rules for API Endpoints
auth_blueprint.add_url_rule(
    '/auth/register',
    view_func=registration_view,
    methods=['POST']
)
auth_blueprint.add_url_rule(
    '/auth/login',
    view_func=login_view,
    methods=['POST']
)
auth_blueprint.add_url_rule(
    '/auth/status',
    view_func=user_view,
    methods=['GET']
)
auth_blueprint.add_url_rule(
    '/auth/logout',
    view_func=logout_view,
    methods=['POST']
)
```

更新导入:

```py
from project.server.models import User, BlacklistToken
```

当用户注销时，令牌不再有效，因此我们将其添加到黑名单中。

> **注意:**通常，较大的应用程序有办法不时更新列入黑名单的令牌，以便系统不会用完有效令牌。

运行测试:

```py
Ran 12 tests in 6.418s

OK
```

[*Remove ads*](/account/join/)

## 重构

最后，我们需要确保令牌没有被列入黑名单，就在令牌被解码之后- `decode_auth_token()` -在注销和用户状态路由中。

首先，让我们为注销路由编写一个测试:

```py
def test_valid_blacklisted_token_logout(self):
    """ Test for logout after a valid token gets blacklisted """
    with self.client:
        # user registration
        resp_register = self.client.post(
            '/auth/register',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json',
        )
        data_register = json.loads(resp_register.data.decode())
        self.assertTrue(data_register['status'] == 'success')
        self.assertTrue(
            data_register['message'] == 'Successfully registered.')
        self.assertTrue(data_register['auth_token'])
        self.assertTrue(resp_register.content_type == 'application/json')
        self.assertEqual(resp_register.status_code, 201)
        # user login
        resp_login = self.client.post(
            '/auth/login',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json'
        )
        data_login = json.loads(resp_login.data.decode())
        self.assertTrue(data_login['status'] == 'success')
        self.assertTrue(data_login['message'] == 'Successfully logged in.')
        self.assertTrue(data_login['auth_token'])
        self.assertTrue(resp_login.content_type == 'application/json')
        self.assertEqual(resp_login.status_code, 200)
        # blacklist a valid token
        blacklist_token = BlacklistToken(
            token=json.loads(resp_login.data.decode())['auth_token'])
        db.session.add(blacklist_token)
        db.session.commit()
        # blacklisted valid token logout
        response = self.client.post(
            '/auth/logout',
            headers=dict(
                Authorization='Bearer ' + json.loads(
                    resp_login.data.decode()
                )['auth_token']
            )
        )
        data = json.loads(response.data.decode())
        self.assertTrue(data['status'] == 'fail')
        self.assertTrue(data['message'] == 'Token blacklisted. Please log in again.')
        self.assertEqual(response.status_code, 401)
```

在这个测试中，我们在注销路由命中之前将令牌列入黑名单，这使得我们的有效令牌不可用。

更新导入:

```py
from project.server.models import User, BlacklistToken
```

测试应该会失败，并出现以下异常:

```py
psycopg2.IntegrityError: duplicate key value violates unique constraint "blacklist_tokens_token_key"
DETAIL:  Key (token)=(eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE0ODUyMDgyOTUsImlhdCI6MTQ4NTIwODI5MCwic3ViIjoxfQ.D9annoyh-VwpI5RY3blaSBX4pzK5UJi1H9dmKg2DeLQ) already exists.
```

现在更新`decode_auth_token`函数，以便在解码后立即处理已经列入黑名单的令牌，并使用适当的消息进行响应。

```py
@staticmethod
def decode_auth_token(auth_token):
    """
 Validates the auth token
 :param auth_token:
 :return: integer|string
 """
    try:
        payload = jwt.decode(auth_token, app.config.get('SECRET_KEY'))
        is_blacklisted_token = BlacklistToken.check_blacklist(auth_token)
        if is_blacklisted_token:
            return 'Token blacklisted. Please log in again.'
        else:
            return payload['sub']
    except jwt.ExpiredSignatureError:
        return 'Signature expired. Please log in again.'
    except jwt.InvalidTokenError:
        return 'Invalid token. Please log in again.'
```

最后，将`check_blacklist()`函数添加到`BlacklistToken`类中的*项目/服务器/模型. py* 中:

```py
@staticmethod
def check_blacklist(auth_token):
    # check whether auth token has been blacklisted
    res = BlacklistToken.query.filter_by(token=str(auth_token)).first()
    if res:
        return True  
    else:
        return False
```

在运行测试之前，更新`test_decode_auth_token`将 bytes 对象转换成一个字符串:

```py
def test_decode_auth_token(self):
    user = User(
        email='test@test.com',
        password='test'
    )
    db.session.add(user)
    db.session.commit()
    auth_token = user.encode_auth_token(user.id)
    self.assertTrue(isinstance(auth_token, bytes))
    self.assertTrue(User.decode_auth_token(
        auth_token.decode("utf-8") ) == 1)
```

运行测试:

```py
Ran 13 tests in 9.557s

OK
```

以类似的方式，为用户状态路由再添加一个测试。

```py
def test_valid_blacklisted_token_user(self):
    """ Test for user status with a blacklisted valid token """
    with self.client:
        resp_register = self.client.post(
            '/auth/register',
            data=json.dumps(dict(
                email='joe@gmail.com',
                password='123456'
            )),
            content_type='application/json'
        )
        # blacklist a valid token
        blacklist_token = BlacklistToken(
            token=json.loads(resp_register.data.decode())['auth_token'])
        db.session.add(blacklist_token)
        db.session.commit()
        response = self.client.get(
            '/auth/status',
            headers=dict(
                Authorization='Bearer ' + json.loads(
                    resp_register.data.decode()
                )['auth_token']
            )
        )
        data = json.loads(response.data.decode())
        self.assertTrue(data['status'] == 'fail')
        self.assertTrue(data['message'] == 'Token blacklisted. Please log in again.')
        self.assertEqual(response.status_code, 401)
```

与上一个测试类似，我们在用户状态路由命中之前将令牌列入黑名单。

最后一次运行测试:

```py
Ran 14 tests in 10.206s

OK
```

[*Remove ads*](/account/join/)

## 代码气味

最后看一下 *test_auth.py* 。注意到重复的代码了吗？例如:

```py
self.client.post(
    '/auth/register',
    data=json.dumps(dict(
        email='joe@gmail.com',
        password='123456'
    )),
    content_type='application/json',
)
```

这种情况出现了八次。要修复此问题，请在文件顶部添加以下助手:

```py
def register_user(self, email, password):
    return self.client.post(
        '/auth/register',
        data=json.dumps(dict(
            email=email,
            password=password
        )),
        content_type='application/json',
    )
```

现在，在任何需要注册用户的地方，您都可以呼叫助手:

```py
register_user(self, 'joe@gmail.com', '123456')
```

登录一个用户怎么样？[自己重构](https://realpython.com/python-refactoring/)它。还能重构什么？下面评论。

## 重构

对于 [PyBites 挑战](https://pybit.es/codechallenge30.html)，让我们重构一些代码来纠正添加到 GitHub repo 中的一个[问题](https://github.com/realpython/flask-jwt-auth/issues/9)。首先向 *test_auth.py* 添加以下测试:

```py
def test_user_status_malformed_bearer_token(self):
    """ Test for user status with malformed bearer token"""
    with self.client:
        resp_register = register_user(self, 'joe@gmail.com', '123456')
        response = self.client.get(
            '/auth/status',
            headers=dict(
                Authorization='Bearer' + json.loads(
                    resp_register.data.decode()
                )['auth_token']
            )
        )
        data = json.loads(response.data.decode())
        self.assertTrue(data['status'] == 'fail')
        self.assertTrue(data['message'] == 'Bearer token malformed.')
        self.assertEqual(response.status_code, 401)
```

本质上，如果`Authorization`头的格式不正确，就会抛出一个错误——例如，`Bearer`和令牌值之间没有空格。运行测试以确保它们失败，然后更新*project/server/auth/views . py*中的`UserAPI`类:

```py
class UserAPI(MethodView):
    """
 User Resource
 """
    def get(self):
        # get the auth token
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                auth_token = auth_header.split(" ")[1]
            except IndexError:
                responseObject = {
                    'status': 'fail',
                    'message': 'Bearer token malformed.'
                }
                return make_response(jsonify(responseObject)), 401
        else:
            auth_token = ''
        if auth_token:
            resp = User.decode_auth_token(auth_token)
            if not isinstance(resp, str):
                user = User.query.filter_by(id=resp).first()
                responseObject = {
                    'status': 'success',
                    'data': {
                        'user_id': user.id,
                        'email': user.email,
                        'admin': user.admin,
                        'registered_on': user.registered_on
                    }
                }
                return make_response(jsonify(responseObject)), 200
            responseObject = {
                'status': 'fail',
                'message': resp
            }
            return make_response(jsonify(responseObject)), 401
        else:
            responseObject = {
                'status': 'fail',
                'message': 'Provide a valid auth token.'
            }
            return make_response(jsonify(responseObject)), 401
```

最后一次测试。

## 结论

在本教程中，我们经历了使用 JSON Web 令牌向 Flask 应用程序添加身份验证的过程。回到本教程开头的目标。你能把每一个都付诸行动吗？你学到了什么？

下一步是什么？客户端怎么样？查看使用 Angular 的[基于令牌的认证，将 Angular 添加到组合中。](http://mherman.org/blog/2017/01/05/token-based-authentication-with-angular)

要了解如何使用 Flask 从头构建一个完整的 web 应用程序，请查看我们的视频系列:

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

欢迎在下面的评论中分享你的评论、问题或建议。完整的代码可以在 [flask-jwt-auth](https://github.com/realpython/flask-jwt-auth) 存储库中找到。

干杯！********