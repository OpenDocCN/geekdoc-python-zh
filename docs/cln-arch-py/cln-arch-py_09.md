# 第七章 - 与真实外部系统的集成 - MongoDB

> 原文：[`www.thedigitalcatbooks.com/pycabook-chapter-07/`](https://www.thedigitalcatbooks.com/pycabook-chapter-07/)
> 
> 嘿，还有另一个例子。
> 
> 《侏罗纪公园》，1993

上一章展示了如何将真实的外部系统与清晰架构的核心集成。不幸的是，我不得不引入大量的代码来管理集成测试，并全局推进到一个合适的设置。在本章中，我将利用我们刚刚完成的工作，仅展示与外部系统严格相关的部分。将数据库从 PostgreSQL 更换为 MongoDB 是展示清晰架构的灵活性以及引入不同方法（如非关系型数据库而非关系型数据库）的简便性的完美方式。

## Fixtures

感谢清晰架构的灵活性，为多个存储系统提供支持变得轻而易举。在本节中，我将实现一个名为 `MongoRepo` 的类，它为 MongoDB（一个知名的 NoSQL 数据库）提供了一个接口。我们将遵循与 PostgreSQL 相同的测试策略，使用运行数据库的 Docker 容器和 docker-compose 来编排整个系统。

你将欣赏我在上一章中创建的复杂测试结构的优势。该结构允许我现在想要为新的存储系统实现测试时重用一些 fixtures。

让我们开始定义文件 `tests/repository/mongodb/conftest.py`，该文件将包含 MongoDB 的 pytest fixtures，类似于我们为 PostgreSQL 创建的文件。

`tests/repository/mongodb/conftest.py`

```py
import pymongo
import pytest

@pytest.fixture(scope="session")
def mg_database_empty(app_configuration):
    client = pymongo.MongoClient(
        host=app_configuration["MONGODB_HOSTNAME"],
        port=int(app_configuration["MONGODB_PORT"]),
        username=app_configuration["MONGODB_USER"],
        password=app_configuration["MONGODB_PASSWORD"],
        authSource="admin",
    )
    db = client[app_configuration["APPLICATION_DB"]]

    yield db

    client.drop_database(app_configuration["APPLICATION_DB"])
    client.close()

@pytest.fixture(scope="function")
def mg_test_data():
    return [
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

@pytest.fixture(scope="function")
def mg_database(mg_database_empty, mg_test_data):
    collection = mg_database_empty.rooms

    collection.insert_many(mg_test_data)

    yield mg_database_empty

    collection.delete_many({}) 
```

如你所见，这些函数与我们为 Postgres 定义的函数非常相似。函数 `mg_database_empty` 负责创建 MongoDB 客户端和空数据库，并在 `yield` 之后销毁它们。fixture `mg_test_data` 提供与 `pg_test_data` 相同的数据，而 `mg_database` 则用这些数据填充空数据库。虽然 SQLAlchemy 包通过会话工作，但 PyMongo 库创建一个客户端并直接使用它，但整体结构是相同的。

由于我们正在导入 PyMongo 库，我们需要更改生产需求。

`requirements/prod.txt`

```py
Flask
SQLAlchemy
psycopg2
pymongo 
```

运行 `pip install -r requirements/dev.txt`.

*源代码

[`github.com/pycabook/rentomatic/tree/ed2-c07-s01`](https://github.com/pycabook/rentomatic/tree/ed2-c07-s01)*

## *Docker Compose 配置*

*我们需要向测试 Docker Compose 配置中添加一个临时的 MongoDB 容器。MongoDB 镜像只需要 `MONGO_INITDB_ROOT_USERNAME` 和 `MONGO_INITDB_ROOT_PASSWORD` 这两个变量，因为它不会创建任何初始数据库。正如我们为 PostgreSQL 容器所做的那样，我们分配了一个特定的端口，该端口将不同于标准端口，以便在运行其他容器的同时执行测试。*

*`docker/testing.yml`*

```py
*version: '3.8'

services:
  postgres:
    image: postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "${POSTGRES_PORT}:5432"
  mongo:
    image: mongo
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGODB_USER}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGODB_PASSWORD}
    ports:
      - "${MONGODB_PORT}:27017"* 
```

**源代码

[`github.com/pycabook/rentomatic/tree/ed2-c07-s02`](https://github.com/pycabook/rentomatic/tree/ed2-c07-s02)**

## **应用程序配置**

**Docker Compose、测试框架和应用程序本身都通过一个单一的 JSON 文件进行配置，我们需要更新该文件以包含我们想要用于 MongoDB 的实际值**

**`config/testing.json`**

```py
**[
  {
    "name": "FLASK_ENV",
    "value": "production"
  },
  {
    "name": "FLASK_CONFIG",
    "value": "testing"
  },
  {
    "name": "POSTGRES_DB",
    "value": "postgres"
  },
  {
    "name": "POSTGRES_USER",
    "value": "postgres"
  },
  {
    "name": "POSTGRES_HOSTNAME",
    "value": "localhost"
  },
  {
    "name": "POSTGRES_PORT",
    "value": "5433"
  },
  {
    "name": "POSTGRES_PASSWORD",
    "value": "postgres"
  },
  {
    "name": "MONGODB_USER",
    "value": "root"
  },
  {
    "name": "MONGODB_HOSTNAME",
    "value": "localhost"
  },
  {
    "name": "MONGODB_PORT",
    "value": "27018"
  },
  {
    "name": "MONGODB_PASSWORD",
    "value": "mongodb"
  },
  {
    "name": "APPLICATION_DB",
    "value": "test"
  }
]** 
```

**由于 MongoDB 的标准端口是 27017，我选择了 27018 进行测试。请记住，这只是一个例子。在实际场景中，我们可能会有多个环境和多个测试设置，在这种情况下，我们可能希望为容器分配一个随机端口，并使用 Python 提取该值并将其传递给应用程序。**

**请注意，我选择使用相同的变量`APPLICATION_DB`来命名 PostgreSQL 和 MongoDB 数据库。再次强调，这只是一个简单的例子，在更复杂的场景中，您的结果可能会有所不同。**

***源代码

[`github.com/pycabook/rentomatic/tree/ed2-c07-s03`](https://github.com/pycabook/rentomatic/tree/ed2-c07-s03)***

## ***集成测试***

***集成测试是我们为 Postgres 编写的测试的镜像，因为我们覆盖的是相同的用例。如果您在同一个系统中使用多个数据库，您可能希望处理不同的用例，因此在实际情况下，这可能是一个更复杂的步骤。然而，您可能完全有理由只想简单地支持多个数据库，让您的客户可以选择将其连接到系统中，在这种情况下，您将做与我这里完全相同的事情，复制并调整相同的测试套件。***

***`tests/repository/mongodb/test_mongorepo.py`***

```py
***import pytest
from rentomatic.repository import mongorepo

pytestmark = pytest.mark.integration

def test_repository_list_without_parameters(
    app_configuration, mg_database, mg_test_data
):
    repo = mongorepo.MongoRepo(app_configuration)

    repo_rooms = repo.list()

    assert set([r.code for r in repo_rooms]) == set(
        [r["code"] for r in mg_test_data]
    )

def test_repository_list_with_code_equal_filter(
    app_configuration, mg_database, mg_test_data
):
    repo = mongorepo.MongoRepo(app_configuration)

    repo_rooms = repo.list(
        filters={"code__eq": "fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a"}
    )

    assert len(repo_rooms) == 1
    assert repo_rooms[0].code == "fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a"

def test_repository_list_with_price_equal_filter(
    app_configuration, mg_database, mg_test_data
):
    repo = mongorepo.MongoRepo(app_configuration)

    repo_rooms = repo.list(filters={"price__eq": 60})

    assert len(repo_rooms) == 1
    assert repo_rooms[0].code == "913694c6-435a-4366-ba0d-da5334a611b2"

def test_repository_list_with_price_less_than_filter(
    app_configuration, mg_database, mg_test_data
):
    repo = mongorepo.MongoRepo(app_configuration)

    repo_rooms = repo.list(filters={"price__lt": 60})

    assert len(repo_rooms) == 2
    assert set([r.code for r in repo_rooms]) == {
        "f853578c-fc0f-4e65-81b8-566c5dffa35a",
        "eed76e77-55c1-41ce-985d-ca49bf6c0585",
    }

def test_repository_list_with_price_greater_than_filter(
    app_configuration, mg_database, mg_test_data
):
    repo = mongorepo.MongoRepo(app_configuration)

    repo_rooms = repo.list(filters={"price__gt": 48})

    assert len(repo_rooms) == 2
    assert set([r.code for r in repo_rooms]) == {
        "913694c6-435a-4366-ba0d-da5334a611b2",
        "fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a",
    }

def test_repository_list_with_price_between_filter(
    app_configuration, mg_database, mg_test_data
):
    repo = mongorepo.MongoRepo(app_configuration)

    repo_rooms = repo.list(filters={"price__lt": 66, "price__gt": 48})

    assert len(repo_rooms) == 1
    assert repo_rooms[0].code == "913694c6-435a-4366-ba0d-da5334a611b2"

def test_repository_list_with_price_as_string(
    app_configuration, mg_database, mg_test_data
):
    repo = mongorepo.MongoRepo(app_configuration)

    repo_rooms = repo.list(filters={"price__lt": "60"})

    assert len(repo_rooms) == 2
    assert set([r.code for r in repo_rooms]) == {
        "f853578c-fc0f-4e65-81b8-566c5dffa35a",
        "eed76e77-55c1-41ce-985d-ca49bf6c0585",
    }*** 
```

***我添加了一个名为`test_repository_list_with_price_as_string`的测试，用于检查当过滤器中的价格以字符串形式表达时会发生什么。通过实验 MongoDB shell，我发现在这种情况下查询没有工作，所以我包括了这个测试以确保实现没有忘记处理这种条件。***

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c07-s04`](https://github.com/pycabook/rentomatic/tree/ed2-c07-s04)****

## ***MongoDB 仓库***

***显然，`MongoRepo`类与 Postgres 接口不同，因为 PyMongo 库与 SQLAlchemy 不同，NoSQL 数据库的结构与关系型数据库的结构不同。文件`rentomatic/repository/mongorepo.py`是***

***`rentomatic/repository/mongorepo.py`***

```py
***import pymongo

from rentomatic.domain import room

class MongoRepo:
    def __init__(self, configuration):
        client = pymongo.MongoClient(
            host=configuration["MONGODB_HOSTNAME"],
            port=int(configuration["MONGODB_PORT"]),
            username=configuration["MONGODB_USER"],
            password=configuration["MONGODB_PASSWORD"],
            authSource="admin",
        )

        self.db = client[configuration["APPLICATION_DB"]]

    def _create_room_objects(self, results):
        return [
            room.Room(
                code=q["code"],
                size=q["size"],
                price=q["price"],
                latitude=q["latitude"],
                longitude=q["longitude"],
            )
            for q in results
        ]

    def list(self, filters=None):
        collection = self.db.rooms

        if filters is None:
            result = collection.find()
        else:
            mongo_filter = {}
            for key, value in filters.items():
                key, operator = key.split("__")

                filter_value = mongo_filter.get(key, {})

                if key == "price":
                    value = int(value)

                filter_value["${}".format(operator)] = value
                mongo_filter[key] = filter_value

            result = collection.find(mongo_filter)

        return self._create_room_objects(result)*** 
```

***这种做法利用了 Rent-o-matic 项目和 MongoDB 系统过滤器之间的相似性 footnote:[两个系统之间的相似性并非偶然，因为我写关于清洁架构的第一篇文章时正在学习 MongoDB，所以我显然受到了它的影响。].***

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c07-s05`](https://github.com/pycabook/rentomatic/tree/ed2-c07-s05)****

* * *

***我认为这一非常简短的章节清楚地展示了分层方法和适当测试设置的优势。到目前为止，我们已经实现并测试了一个面向两个非常不同的数据库（如 PostgreSQL 和 MongoDB）的接口，但这两个接口都可以由相同的使用案例使用，这最终意味着相同的 API 端点。***

***尽管我们已经正确测试了与这些外部系统的集成，但我们仍然没有一种方法可以在我们所说的生产就绪环境中运行整个系统，也就是说，以一种可以暴露给外部用户的方式。在下一章中，我将向您展示我们如何利用用于测试的相同设置来运行 Flask、PostgreSQL 以及我们创建的使用案例，使其可以在生产中使用。***
