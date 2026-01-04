# 第四章 - 添加 Web 应用程序

> 原文：[`www.thedigitalcatbooks.com/pycabook-chapter-04/`](https://www.thedigitalcatbooks.com/pycabook-chapter-04/)
> 
> 为了您的信息，Hairdo，一个主要网络公司，对我感兴趣。
> 
> 土拨鼠日，1993 年

在本章中，我将介绍为房间列表用例创建 HTTP 端点的过程。HTTP 端点是由运行特定逻辑并返回标准格式值的 Web 服务器公开的 URL。

我将遵循 REST 推荐，因此端点将返回 JSON 有效负载。然而，REST 不是清洁架构的一部分，这意味着你可以选择根据你喜欢的任何方案来建模你的 URL 和返回数据的格式。

为了公开 HTTP 端点，我们需要一个用 Python 编写的 Web 服务器，在这种情况下，我选择了 Flask。Flask 是一个轻量级的 Web 服务器，具有模块化结构，仅提供用户所需的部分。特别是，我们不会使用任何数据库/ORM，因为我们已经实现了自己的存储库层。

## Flask 设置

让我们开始更新需求文件。文件`requirements/prod.txt`应提及 Flask，因为这个包包含一个运行本地 Web 服务器的脚本，我们可以用它来公开端点

`requirements/prod.txt`

```py
Flask 
```

文件`requirements/test.txt`将包含用于与 Flask 一起工作的 pytest 扩展（关于这一点稍后讨论）

`requirements/test.txt`

```py
-r prod.txt
pytest
tox
coverage
pytest-cov
pytest-flask 
```

源代码

[`github.com/pycabook/rentomatic/tree/ed2-c04-s01`](https://github.com/pycabook/rentomatic/tree/ed2-c04-s01)*

请记住，在这些更改之后再次运行`pip install -r requirements/dev.txt`以在虚拟环境中安装新包。

Flask 应用程序的设置并不复杂，但涉及许多概念，由于这不是 Flask 教程，我将快速浏览这些步骤。不过，我会为每个概念提供 Flask 文档的链接。如果你想对这个话题进行更深入的了解，你可以阅读我的系列文章[Flask 项目设置：TDD、Docker、Postgres 等](https://www.thedigitalcatonline.com/blog/2020/07/05/flask-project-setup-tdd-docker-postgres-and-more-part-1/)。

Flask 应用程序可以使用纯 Python 对象进行配置（[文档](http://flask.pocoo.org/docs/latest/api/#flask.Config.from_object)），因此我创建了包含此代码的文件`application/config.py`

`application/config.py`

```py
*import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    """Base configuration"""

class ProductionConfig(Config):
    """Production configuration"""

class DevelopmentConfig(Config):
    """Development configuration"""

class TestingConfig(Config):
    """Testing configuration"""

    TESTING = True* 
```

阅读[这个页面](http://flask.pocoo.org/docs/latest/config/)了解更多关于 Flask 配置参数的信息。

现在我们需要一个初始化 Flask 应用程序（[文档](http://flask.pocoo.org/docs/latest/patterns/appfactories/)）、配置它并注册蓝图（[文档](http://flask.pocoo.org/docs/latest/blueprints/)）的函数。文件`application/app.py`包含以下代码，这是一个应用程序工厂

`application/app.py`

```py
*from flask import Flask

from application.rest import room

def create_app(config_name):

    app = Flask(__name__)

    config_module = f"application.config.{config_name.capitalize()}Config"

    app.config.from_object(config_module)

    app.register_blueprint(room.blueprint)

    return app* 
```

源代码

[`github.com/pycabook/rentomatic/tree/ed2-c04-s02`](https://github.com/pycabook/rentomatic/tree/ed2-c04-s02)

## 测试并创建一个 HTTP 端点

在我们创建 web 服务器的正确设置之前，我们希望创建将要公开的端点。端点最终是当用户向特定 URL 发送请求时运行的函数，因此我们仍然可以使用 TDD，因为最终目标是拥有产生特定结果的代码。

测试端点时遇到的问题是，我们需要在点击测试 URL 时确保 web 服务器正在运行。web 服务器本身是一个外部系统，所以我们不会对其进行测试，但提供端点的代码是我们应用程序的一部分^([1]）。实际上，它是一个网关，即允许 HTTP 框架访问用例的接口。

`pytest-flask` 扩展允许我们运行 Flask、模拟 HTTP 请求并测试 HTTP 响应。这个扩展隐藏了很多自动化，所以乍一看可能有点“魔法”。当你安装它时，一些固定值如 `client` 会自动可用，因此你不需要导入它们。此外，它试图访问另一个名为 `app` 的固定值，你必须定义它。因此，这是要做的第一件事。

可以在测试文件中直接定义固定值，但如果希望固定值全局可用，最佳定义位置是文件 `conftest.py`，该文件由 pytest 自动加载。正如你所见，这里有很多自动化，如果你不了解它，可能会对结果感到惊讶，或者因错误而感到沮丧。

`tests/conftest.py`

```py
**import pytest

from application.app import create_app

@pytest.fixture
def app():
    app = create_app("testing")

    return app** 
```

函数 `app` 运行应用程序工厂来创建一个 Flask 应用，使用配置 `testing`，该配置将标志 `TESTING` 设置为 `True`。你可以在[官方文档](http://flask.pocoo.org/docs/1.0/config/)中找到这些标志的描述。

在这个阶段，我们可以为我们的端点编写测试。

`tests/rest/test_room.py`

```py
**import json
from unittest import mock

from rentomatic.domain.room import Room

room_dict = {
    "code": "3251a5bd-86be-428d-8ae9-6e51a8048c33",
    "size": 200,
    "price": 10,
    "longitude": -0.09998975,
    "latitude": 51.75436293,
}

rooms = [Room.from_dict(room_dict)]

@mock.patch("application.rest.room.room_list_use_case")
def test_get(mock_use_case, client):
    mock_use_case.return_value = rooms

    http_response = client.get("/rooms")

    assert json.loads(http_response.data.decode("UTF-8")) == [room_dict]
    mock_use_case.assert_called()
    assert http_response.status_code == 200
    assert http_response.mimetype == "application/json"** 
```

让我们逐节进行注释。

`tests/rest/test_room.py`

```py
**import json
from unittest import mock

from rentomatic.domain.room import Room

room_dict = {
    "code": "3251a5bd-86be-428d-8ae9-6e51a8048c33",
    "size": 200,
    "price": 10,
    "longitude": -0.09998975,
    "latitude": 51.75436293,
}

rooms = [Room.from_dict(room_dict)]** 
```

第一部分包含一些导入，并从字典中设置一个房间。这样我们就可以稍后直接比较初始字典的内容与 API 端点结果。记住，API 返回 JSON 内容，我们可以轻松地将 JSON 数据转换为简单的 Python 结构，因此从字典开始会很有用。

`tests/rest/test_room.py`

```py
**@mock.patch("application.rest.room.room_list_use_case")
def test_get(mock_use_case, client):** 
```

目前我们只有一个测试。在整个测试过程中，我们模拟了用例的使用，因为我们不感兴趣运行它，因为它已经在其他地方测试过了。然而，我们确实对检查传递给用例的参数感兴趣，而模拟可以提供这些信息。测试从装饰器`patch`和固定装置`client`接收模拟，这是`pytest-flask`提供的固定装置之一。固定装置自动加载`app`，我们在`conftest.py`中定义了它，它是一个模拟 HTTP 客户端的对象，可以访问 API 端点并存储服务器的响应。

`tests/rest/test_room.py`

```py
 **mock_use_case.return_value = rooms

    http_response = client.get("/rooms")

    assert json.loads(http_response.data.decode("UTF-8")) == [room_dict]
    mock_use_case.assert_called()
    assert http_response.status_code == 200
    assert http_response.mimetype == "application/json"** 
```

第一行初始化了模拟用例，指示它返回我们之前创建的固定`rooms`变量。测试的核心部分是获取 API 端点的行，它发送一个 HTTP GET 请求并收集服务器的响应。

之后，我们检查响应中包含的数据是否是包含`room_dict`结构数据的 JSON，使用 _case 方法已被调用，HTTP 响应状态码是 200，最后服务器发送正确的 MIME 类型。

现在是时候编写端点了，我们将最终看到所有架构的各个部分协同工作，就像我们在之前写的那个小 CLI 程序中所做的那样。让我向您展示我们可以创建的最小 Flask 端点的模板

```py
**blueprint = Blueprint('room', __name__)

@blueprint.route('/rooms', methods=['GET'])
def room_list():
    [LOGIC]
    return Response([JSON DATA],
                    mimetype='application/json',
                    status=[STATUS])** 
```

如您所见，结构非常简单。除了设置蓝图，这是 Flask 注册端点的方式，我们创建了一个简单的函数来运行端点，并给它装饰了`/rooms`端点，用于处理`GET`请求。该函数将运行一些逻辑，并最终返回一个包含 JSON 数据、正确的 MIME 类型和表示逻辑成功或失败的 HTTP 状态码的`Response`对象。

上面的模板变成了以下代码

`application/rest/room.py`

```py
**import json

from flask import Blueprint, Response

from rentomatic.repository.memrepo import MemRepo
from rentomatic.use_cases.room_list import room_list_use_case
from rentomatic.serializers.room import RoomJsonEncoder

blueprint = Blueprint("room", __name__)

rooms = [
    {
        "code": "f853578c-fc0f-4e65-81b8-566c5dffa35a",
        "size": 215,
        "price": 39,
        "longitude": -0.09998975,
        "latitude": 51.75436293,
    },
    {
        "code": "fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a",
        "size": 405,
        "price": 66,
        "longitude": 0.18228006,
        "latitude": 51.74640997,
    },
    {
        "code": "913694c6-435a-4366-ba0d-da5334a611b2",
        "size": 56,
        "price": 60,
        "longitude": 0.27891577,
        "latitude": 51.45994069,
    },
    {
        "code": "eed76e77-55c1-41ce-985d-ca49bf6c0585",
        "size": 93,
        "price": 48,
        "longitude": 0.33894476,
        "latitude": 51.39916678,
    },
]

@blueprint.route("/rooms", methods=["GET"])
def room_list():
    repo = MemRepo(rooms)
    result = room_list_use_case(repo)

    return Response(
        json.dumps(result, cls=RoomJsonEncoder),
        mimetype="application/json",
        status=200,
    )** 
```

***源代码

[`github.com/pycabook/rentomatic/tree/ed2-c04-s03`](https://github.com/pycabook/rentomatic/tree/ed2-c04-s03)*

请注意，我使用与脚本`cli.py`相同的列表初始化了内存存储。需要使用数据（即使是空列表）初始化存储的原因是由于存储`MemRepo`的限制。运行用例的代码是

`application/rest/room.py`

```py
***def room_list():
    repo = MemRepo(rooms)
    result = room_list_use_case(repo)*** 
```

这与我们在命令行界面中使用的代码完全相同。代码的最后部分创建了一个适当的 HTTP 响应，使用`RoomJsonEncoder`序列化用例的结果，并将 HTTP 状态设置为 200（成功）

`application/rest/room.py`

```py
 ***return Response(
        json.dumps(result, cls=RoomJsonEncoder),
        mimetype="application/json",
        status=200,
    )*** 
```

这简要展示了干净架构的力量。编写 CLI 界面或 Web 服务不同之处仅在于表示层，而不是逻辑，因为逻辑是相同的，它包含在用例中。

现在我们已经定义了端点，我们可以最终确定网络服务器的配置，这样我们就可以用浏览器访问端点了。这虽然不是干净的架构的严格组成部分，但正如我在 CLI 接口中所做的那样，我想让你看到最终结果，以获得完整的画面，并且享受你到目前为止所付出的努力。

## WSGI

Python 网络应用暴露了一个称为 [Web 服务器网关接口](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface) 或 WSGI 的通用接口。因此，要运行 Flask 开发网络服务器，我们必须在项目的根目录中定义一个 `wsgi.py` 文件，即在 `cli.py` 文件所在的同一目录中。

`wsgi.py`

```py
***import os

from application.app import create_app

app = create_app(os.environ["FLASK_CONFIG"])*** 
```

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c04-s04`](https://github.com/pycabook/rentomatic/tree/ed2-c04-s04)****

当你运行 Flask 命令行界面 ([文档](http://flask.pocoo.org/docs/1.0/cli/)) 时，它会自动查找一个名为 `wsgi.py` 的文件并加载它，期望它包含一个名为 `app` 的变量，该变量是 `Flask` 对象的一个实例。由于 `create_app` 是一个工厂函数，我们只需要执行它。

此时，你可以在包含此文件的目录中执行 `FLASK_CONFIG="development" flask run`，你应该会看到一个类似的消息：

```py
 **** Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)*** 
```

此时，你可以将浏览器指向 [`127.0.0.1:5000/rooms`](http://127.0.0.1:5000/rooms)，并享受你的网络应用第一个端点返回的 JSON 数据。

 * 

我希望你现在可以欣赏我们所创建的分层架构的力量。我们确实编写了很多代码来“仅仅”打印出模型列表，但我们编写的代码是一个可以轻松扩展和修改的骨架。它也是完全经过测试的，这是许多软件项目在实施过程中都难以解决的问题。

我展示的使用案例故意非常简单。它不需要任何输入，也不能返回错误条件，所以我们编写的代码完全忽略了输入验证和错误管理。然而，这些主题却极为重要，因此我们需要讨论一个干净的架构如何处理这些问题。

***1

从理论上讲，我们可以创建一个纯组件，该组件接收参数并返回一个 JSON 对象，然后将其封装到端点中。这样，组件将严格属于内部系统的一部分，而端点属于外部系统，但两者都必须在网关层创建。这看起来有些过度设计，至少对于我们正在讨论的简单示例来说是这样，所以我会将它们放在一起，作为一个单一组件进行测试。***
