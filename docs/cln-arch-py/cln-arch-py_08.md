# 第六章 - 与真实外部系统的集成 - Postgres

> 原文：[`www.thedigitalcatbooks.com/pycabook-chapter-06/`](https://www.thedigitalcatbooks.com/pycabook-chapter-06/)
> 
> 哎哟，非常抱歉，汉斯。我没有收到那份备忘录。
> 
> 也许你应该把它贴在公告板上。
> 
> 《虎胆龙威》，1988

我为这个项目实现的基本内存存储库足以展示存储层抽象的概念。然而，这还不足以运行一个生产系统，因此我们需要实现与真实存储（如数据库）的连接。每当使用外部系统并希望测试接口时，我们可以使用模拟，但在某个时刻我们需要确保两个系统实际上可以协同工作，这就是我们需要开始创建集成测试的时候。

在本章中，我将展示如何设置和运行我们的应用程序与真实数据库之间的集成测试。在本章结束时，我将有一个允许应用程序与 PostgreSQL 接口的存储库，以及一系列在 Docker 中运行的实时数据库实例上运行的测试。

本章将向您展示干净架构的最大优势之一，即您可以用其他组件替换现有组件的简单性，这些组件可能基于完全不同的技术。

## 通过接口解耦

我们在前几章中设计的干净架构定义了一个使用案例，它接收一个存储库实例作为参数，并使用其`list`方法检索包含的条目。这允许使用案例与存储库形成非常松散的耦合，仅通过对象公开的 API 连接，而不是与真实实现连接。换句话说，使用案例相对于`list`方法来说是多态的。

这非常重要，它是干净架构设计的核心。通过 API 连接，使用案例和存储库可以在任何时间被不同的实现所替换，前提是新实现提供了请求的接口。

例如，值得注意的一点是，对象的初始化不是使用案例所使用的 API 的一部分，因为存储库是在主脚本中初始化的，而不是在每个使用案例中。因此，`__init__`方法在存储库实现之间不需要相同，这为我们提供了很大的灵活性，因为不同的存储系统可能需要不同的初始化值。

我们在前面章节中实现的一个简单存储库是

`rentomatic/repository/memrepo.py`

```py
from rentomatic.domain.room import Room

class MemRepo:
    def __init__(self, data):
        self.data = data

    def list(self, filters=None):

        result = [Room.from_dict(i) for i in self.data]

        if filters is None:
            return result

        if "code__eq" in filters:
            result = [r for r in result if r.code == filters["code__eq"]]

        if "price__eq" in filters:
            result = [
                r for r in result if r.price == int(filters["price__eq"])
            ]

        if "price__lt" in filters:
            result = [
                r for r in result if r.price < int(filters["price__lt"])
            ]

        if "price__gt" in filters:
            result = [
                r for r in result if r.price > int(filters["price__gt"])
            ]

        return result 
```

其接口由两部分组成：初始化和方法`list`。`__init__`方法接受值，因为这个特定的对象并不作为长期存储，所以我们每次实例化类时都必须传递一些数据。

基于适当数据库的存储库在初始化时不需要填充数据，其主要任务是存储会话之间的数据，但仍然至少需要使用数据库地址和访问凭证进行初始化。

此外，我们必须处理一个适当的外部系统，因此我们必须制定一个测试它的策略，因为这可能需要一个在后台运行的数据库引擎。记住，我们正在创建一个特定实现的数据存储库，所以一切都将根据我们将选择的实际数据库系统进行定制。

## 基于 PostgreSQL 的存储库

让我们从基于流行的 SQL 数据库[PostgreSQL](https://www.postgresql.org)的存储库开始。它可以通过多种方式从 Python 访问，但最好的可能还是通过[SQLAlchemy](https://www.sqlalchemy.org)接口。SQLAlchemy 是一个 ORM，一个将对象（如面向对象）映射到关系数据库的包。ORM 通常可以在像 Django 这样的 Web 框架或像我们正在考虑的这样的独立包中找到。

关于 ORMs 的重要之处在于，它们是你不应该尝试模拟的很好的例子。正确模拟查询数据库时使用的 SQLAlchemy 结构会导致非常复杂的代码，难以编写且几乎无法维护，因为查询的每一个变化都会导致一系列需要重新编写的模拟^([1]).

因此，我们需要设置一个集成测试。想法是创建数据库，使用 SQLAlchemy 建立连接，测试我们需要检查的条件，然后销毁数据库。由于创建和销毁数据库的动作在时间上可能很昂贵，我们可能只想在整个测试套件的开头和结尾执行这些操作，但即使有这个改变，测试仍然会很慢。这就是为什么我们还需要使用标签来避免每次运行套件时都运行它们。让我们一步一步面对这个复杂任务。

## 标记集成测试

我们需要做的第一件事是标记集成测试，默认排除它们，并创建一种运行它们的方法。由于 pytest 支持标签，称为*标记*，我们可以使用这个特性为一个整个模块添加全局标记。创建文件`tests/repository/postgres/test_postgresrepo.py`并在其中放入以下代码

`tests/repository/postgres/test_postgresrepo.py`

```py
import pytest

pytestmark = pytest.mark.integration

def test_dummy():
    pass 
```

模块属性`pytestmark`将模块中的每个测试标记为`integration`标签。为了验证这一点，我添加了一个总是通过的`test_dummy`测试函数。

标记应该在`pytest.ini`中注册

`pytest.ini`

```py
[pytest]
minversion = 2.0
norecursedirs = .git .tox requirements*
python_files = test*.py
markers =
        integration: integration tests 
```

你现在可以运行`pytest -svv -m integration`来请求 pytest 只运行带有该标签的测试。选项`-m`支持丰富的语法，你可以通过阅读[文档](https://docs.pytest.org/en/latest/example/markers.html)来学习。

```py
$ pytest -svv -m integration
========================= test session starts ===========================
platform linux -- Python XXXX, pytest-XXXX, py-XXXX, pluggy-XXXX --
cabook/venv3/bin/python3
cachedir: .cache
rootdir: cabook/code/calc, inifile: pytest.ini
plugins: cov-XXXX
collected 36 items / 35 deselected / 1 selected

tests/repository/postgres/test_postgresrepo.py::test_dummy PASSED

=================== 1 passed, 35 deselected in 0.20s ==================== 
```

虽然这足以有选择地运行集成测试，但不足以默认跳过它们。为了做到这一点，我们可以修改 pytest 设置，将这些测试标记为跳过，但这将给我们没有运行它们的方法。实现这一标准的方法是定义一个新的命令行选项，并根据此选项的值处理每个标记的测试。

要实现这一点，请打开我们已创建的 `tests/conftest.py` 文件，并添加以下代码

`tests/conftest.py`

```py
def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", help="run integration tests"
    )

def pytest_runtest_setup(item):
    if "integration" in item.keywords and not item.config.getvalue(
        "integration"
    ):
        pytest.skip("need --integration option to run") 
```

第一个函数是 pytest CLI 解析器的钩子，它添加了 `--integration` 选项。当在命令行上指定此选项时，pytest 设置将包含键 `integration`，其值为 `True`。

第二个函数是每个测试的 pytest 设置的钩子。变量 `item` 包含测试本身（实际上是一个 `_pytest.python.Function` 对象），它反过来包含两个有用的信息。第一个是属性 `item.keywords`，它包含测试标记，以及许多其他有趣的事情，如测试名称、文件、模块，以及测试内部发生的补丁信息。第二个是属性 `item.config`，它包含解析后的 pytest 命令行。

因此，如果测试被标记为 `integration`（`'integration' in item.keywords`）且没有指定 `--integration` 选项（`not item.config.getvalue("integration")`），则测试将被跳过。

这是带有 `--integration` 的输出

```py
$ pytest -svv --integration
========================= test session starts ===========================
platform linux -- Python XXXX, pytest-XXXX, py-XXXX, pluggy-XXXX --
cabook/venv3/bin/python3
cachedir: .cache
rootdir: cabook/code/calc, inifile: pytest.ini
plugins: cov-XXXX
collected 36 items

...
tests/repository/postgres/test_postgresrepo.py::test_dummy PASSED
...

========================= 36 passed in 0.26s ============================ 
```

这是在没有自定义选项时的输出

```py
$ pytest -svv 
========================= test session starts ===========================
platform linux -- Python XXXX, pytest-XXXX, py-XXXX, pluggy-XXXX --
cabook/venv3/bin/python3
cachedir: .cache
rootdir: cabook/code/calc, inifile: pytest.ini
plugins: cov-XXXX
collected 36 items

...
tests/repository/postgres/test_postgresrepo.py::test_dummy SKIPPED
...

=================== 35 passed, 1 skipped in 0.27s ======================= 
```

源代码

[`github.com/pycabook/rentomatic/tree/ed2-c06-s01`](https://github.com/pycabook/rentomatic/tree/ed2-c06-s01)*

## 创建 SQLAlchemy 类

创建和填充测试数据库的初始数据将是测试套件的一部分，但我们需要定义数据库中将包含的表。这正是 SQLAlchemy 的 ORM 发挥作用的地方，我们将用 Python 对象来定义这些表。

将包 `SQLAlchemy` 和 `psycopg2` 添加到需求文件 `prod.txt` 中

`requirements/prod.txt`

```py
*Flask
SQLAlchemy
psycopg2* 
```

并更新已安装的包

```py
*$ pip install -r requirements/dev.txt* 
```

创建一个名为 `rentomatic/repository/postgres_objects.py` 的文件，并包含以下内容

`rentomatic/repository/postgres_objects.py`

```py
*from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Room(Base):
    __tablename__ = 'room'

    id = Column(Integer, primary_key=True)

    code = Column(String(36), nullable=False)
    size = Column(Integer)
    price = Column(Integer)
    longitude = Column(Float)
    latitude = Column(Float)* 
```

让我们逐节注释它

```py
*from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()* 
```

我们需要从 SQLAlchemy 包导入许多内容来设置数据库和创建表。记住，SQLAlchemy 有一个声明式方法，因此我们需要实例化对象 `Base`，然后使用它作为声明表/对象的起点。

```py
*class Room(Base):
    __tablename__ = 'room'

    id = Column(Integer, primary_key=True)

    code = Column(String(36), nullable=False)
    size = Column(Integer)
    price = Column(Integer)
    longitude = Column(Float)
    latitude = Column(Float)* 
```

这是代表数据库中房间的类。重要的是要理解这并不是我们在业务逻辑中使用的类，而是定义我们将用于映射`Room`实体的 SQL 数据库中的表的类。因此，这个类的结构是由存储层的需要决定的，而不是由用例决定的。例如，你可能希望将`longitude`和`latitude`存储在 JSON 字段中，以便更容易地扩展，而不改变域模型的定义。在 Rent-o-matic 项目的简单情况下，这两个类几乎重叠，但一般来说并不是这样。

显然，这意味着你必须保持存储层和域层的一致性，并且你需要自己管理迁移。你可以使用像 Alembic 这样的工具，但迁移不会直接来自域模型的变化。

**源代码

[`github.com/pycabook/rentomatic/tree/ed2-c06-s02`](https://github.com/pycabook/rentomatic/tree/ed2-c06-s02)**

## 编排管理

当我们运行集成测试时，Postgres 数据库引擎必须在后台已经运行并配置好，例如，使用一个干净的数据库，准备好使用。此外，当所有测试执行完毕后，数据库应该被删除，数据库引擎应该停止。

这对于 Docker 来说是一个完美的任务，它可以在最小配置的情况下独立运行复杂的系统。在这里我们有选择：我们可能希望使用外部脚本来编排数据库的创建和销毁，或者尝试在测试套件中实现一切。第一种解决方案是许多框架所使用的，也是我在一系列文章[Flask 项目设置：TDD、Docker、Postgres 等](https://www.thedigitalcatonline.com/blog/2020/07/05/flask-project-setup-tdd-docker-postgres-and-more-part-1/)中探索的，因此在本章中，我将展示该解决方案的实现。

正如我在提到的文章中解释的那样，计划是创建一个管理脚本，该脚本启动和关闭所需的容器，并在其中运行测试。管理脚本也可以用来运行应用程序本身，或者创建开发环境，但在这个例子中，我将简化它，只管理测试。我强烈建议你阅读那些文章，如果你想了解我将使用的设置背后的整体情况。

如果我们计划使用 Docker Compose，我们首先要做的是将需求添加到`requirements/test.txt`中

`requirements/test.txt`

```py
**-r prod.txt
tox
coverage
pytest
pytest-cov
pytest-flask
docker-compose** 
```

并运行`pip install -r requirements/dev.txt`来安装它。管理脚本如下

`manage.py`

```py
**#! /usr/bin/env python

import os
import json
import subprocess
import time

import click
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Ensure an environment variable exists and has a value
def setenv(variable, default):
    os.environ[variable] = os.getenv(variable, default)

APPLICATION_CONFIG_PATH = "config"
DOCKER_PATH = "docker"

def app_config_file(config):
    return os.path.join(APPLICATION_CONFIG_PATH, f"{config}.json")

def docker_compose_file(config):
    return os.path.join(DOCKER_PATH, f"{config}.yml")

def read_json_configuration(config):
    # Read configuration from the relative JSON file
    with open(app_config_file(config)) as f:
        config_data = json.load(f)

    # Convert the config into a usable Python dictionary
    config_data = dict((i["name"], i["value"]) for i in config_data)

    return config_data

def configure_app(config):
    configuration = read_json_configuration(config)

    for key, value in configuration.items():
        setenv(key, value)

@click.group()
def cli():
    pass

def docker_compose_cmdline(commands_string=None):
    config = os.getenv("APPLICATION_CONFIG")
    configure_app(config)

    compose_file = docker_compose_file(config)

    if not os.path.isfile(compose_file):
        raise ValueError(f"The file {compose_file} does not exist")

    command_line = [
        "docker-compose",
        "-p",
        config,
        "-f",
        compose_file,
    ]

    if commands_string:
        command_line.extend(commands_string.split(" "))

    return command_line

def run_sql(statements):
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOSTNAME"),
        port=os.getenv("POSTGRES_PORT"),
    )

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    for statement in statements:
        cursor.execute(statement)

    cursor.close()
    conn.close()

def wait_for_logs(cmdline, message):
    logs = subprocess.check_output(cmdline)
    while message not in logs.decode("utf-8"):
        time.sleep(1)
        logs = subprocess.check_output(cmdline)

@cli.command()
@click.argument("args", nargs=-1)
def test(args):
    os.environ["APPLICATION_CONFIG"] = "testing"
    configure_app(os.getenv("APPLICATION_CONFIG"))

    cmdline = docker_compose_cmdline("up -d")
    subprocess.call(cmdline)

    cmdline = docker_compose_cmdline("logs postgres")
    wait_for_logs(cmdline, "ready to accept connections")

    run_sql([f"CREATE DATABASE {os.getenv('APPLICATION_DB')}"])

    cmdline = [
        "pytest",
        "-svv",
        "--cov=application",
        "--cov-report=term-missing",
    ]
    cmdline.extend(args)
    subprocess.call(cmdline)

    cmdline = docker_compose_cmdline("down")
    subprocess.call(cmdline)

if __name__ == "__main__":
    cli()** 
```

让我们逐块看看它做了什么。

```py
**#! /usr/bin/env python

import os
import json
import subprocess
import time

import click
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Ensure an environment variable exists and has a value
def setenv(variable, default):
    os.environ[variable] = os.getenv(variable, default)

APPLICATION_CONFIG_PATH = "config"
DOCKER_PATH = "docker"** 
```

一些 Docker 容器（例如我们很快将要使用的 PostgreSQL 容器）依赖于环境变量来进行初始设置，因此如果它们尚未初始化，我们需要定义一个函数来设置环境变量。我们还定义了一些配置文件的路径。

```py
**def app_config_file(config):
    return os.path.join(APPLICATION_CONFIG_PATH, f"{config}.json")

def docker_compose_file(config):
    return os.path.join(DOCKER_PATH, f"{config}.yml")

def read_json_configuration(config):
    # Read configuration from the relative JSON file
    with open(app_config_file(config)) as f:
        config_data = json.load(f)

    # Convert the config into a usable Python dictionary
    config_data = dict((i["name"], i["value"]) for i in config_data)

    return config_data

def configure_app(config):
    configuration = read_json_configuration(config)

    for key, value in configuration.items():
        setenv(key, value)** 
```

由于原则上我预计至少会有开发、测试和生产不同的配置，我引入了`app_config_file`和`docker_compose_file`，它们会返回我们正在工作的特定环境的文件。函数`read_json_configuration`已被从`configure_app`中分离出来，因为它将被测试导入以初始化数据库存储库。

```py
**@click.group()
def cli():
    pass

def docker_compose_cmdline(commands_string=None):
    config = os.getenv("APPLICATION_CONFIG")
    configure_app(config)

    compose_file = docker_compose_file(config)

    if not os.path.isfile(compose_file):
        raise ValueError(f"The file {compose_file} does not exist")

    command_line = [
        "docker-compose",
        "-p",
        config,
        "-f",
        compose_file,
    ]

    if commands_string:
        command_line.extend(commands_string.split(" "))

    return command_line** 
```

这是一个简单的函数，它创建 Docker Compose 命令行，这样我们就不需要在编排容器时重复长列表的选项。

```py
**def run_sql(statements):
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOSTNAME"),
        port=os.getenv("POSTGRES_PORT"),
    )

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    for statement in statements:
        cursor.execute(statement)

    cursor.close()
    conn.close()

def wait_for_logs(cmdline, message):
    logs = subprocess.check_output(cmdline)
    while message not in logs.decode("utf-8"):
        time.sleep(1)
        logs = subprocess.check_output(cmdline)** 
```

函数`run_sql`允许我们在运行的 Postgres 数据库上运行 SQL 命令，当我们要创建空测试数据库时将非常有用。第二个函数`wait_for_logs`是一种简单的方式来监控 Postgres 容器，并确保它已准备好使用。每次你以编程方式启动容器时，你都需要意识到它们在准备好使用之前有一个启动时间，并相应地行动。

```py
**@cli.command()
@click.argument("args", nargs=-1)
def test(args):
    os.environ["APPLICATION_CONFIG"] = "testing"
    configure_app(os.getenv("APPLICATION_CONFIG"))

    cmdline = docker_compose_cmdline("up -d")
    subprocess.call(cmdline)

    cmdline = docker_compose_cmdline("logs postgres")
    wait_for_logs(cmdline, "ready to accept connections")

    run_sql([f"CREATE DATABASE {os.getenv('APPLICATION_DB')}"])

    cmdline = [
        "pytest",
        "-svv",
        "--cov=application",
        "--cov-report=term-missing",
    ]
    cmdline.extend(args)
    subprocess.call(cmdline)

    cmdline = docker_compose_cmdline("down")
    subprocess.call(cmdline)

if __name__ == "__main__":
    cli()** 
```

这是我们定义的最后一个函数，也是我们管理脚本提供的唯一命令。首先，应用程序使用名称`testing`进行配置，这意味着我们将使用`config/testing.json`配置文件和`docker/testing.yml`Docker Compose 文件。所有这些名称和路径都只是来自这个管理脚本任意设置的惯例，因此你显然可以以不同的方式组织你的项目。

该函数随后根据 Docker Compose 文件启动容器，运行`docker-compose up -d`。它等待日志消息表明数据库已准备好接受连接，并运行创建测试数据库的 SQL 命令。

之后，它使用默认选项运行 Pytest，并添加我们将在命令行上提供的所有选项，最后拆除 Docker Compose 容器。

为了完成设置，我们需要为 Docker Compose 定义一个配置文件

`docker/testing.yml`

```py
**version: '3.8'

services:
  postgres:
    image: postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "${POSTGRES_PORT}:5432"** 
```

最后是一个 JSON 配置文件

`config/testing.json`

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
    "name": "APPLICATION_DB",
    "value": "test"
  }
]** 
```

关于此配置的一些注意事项。首先，它定义了`FLASK_ENV`和`FLASK_CONFIG`。第一个，你可能还记得，是一个内部 Flask 变量，它只能是`development`或`production`，并且与内部调试器相关联。第二个是我们用来使用`application/config.py`中的对象配置 Flask 应用程序的变量。为了测试目的，我们将`FLASK_ENV`设置为`production`，因为我们不需要内部调试器，并将`FLASK_CONFIG`设置为`testing`，这将导致应用程序使用`TestingConfig`类进行配置。这个类将内部 Flask 参数`TESTING`设置为`True`。

其余的 JSON 配置初始化了以`POSTGRES_`为前缀的变量。这些是 Postgres Docker 容器所需的变量。当容器运行时，它将自动创建一个名为`POSTGRES_DB`指定的数据库。它还创建了一个用户和密码，使用`POSTGRES_USER`和`POSTGRES_PASSWORD`中指定的值。

最后，我引入了`APPLICATION_DB`变量，因为我想要创建一个特定的数据库，而不是默认的数据库。默认端口`POSTGRES_PORT`已从标准值 5432 更改为 5433，以避免与机器上已运行的任何数据库（无论是本地还是容器化）冲突。如您在 Docker Compose 配置文件中所见，这仅更改了容器的外部映射，而不是数据库引擎在容器内部实际使用的端口。

有了所有这些文件，我们就准备好开始设计我们的测试了。

***源代码

[`github.com/pycabook/rentomatic/tree/ed2-c06-s03`](https://github.com/pycabook/rentomatic/tree/ed2-c06-s03)*

## 数据库固定值

由于我们在 JSON 文件中定义了数据库的配置，我们需要一个固定值来加载相同的配置，这样我们就可以在测试期间连接到数据库。由于我们已经在管理脚本中有了`read_json_configuration`函数，我们只需要将其包装起来。这是一个不是特定于 Postgres 存储库的固定值，所以我将在`tests/conftest.py`中介绍它

`tests/conftest.py`

```py
***from manage import read_json_configuration

...

@pytest.fixture(scope="session")
def app_configuration():
    return read_json_configuration("testing")*** 
```

如您所见，为了简单起见，我硬编码了配置文件的名称。另一种解决方案可能是在管理脚本中创建一个包含应用程序配置的环境变量，并从这里读取它。

其余的固定值包含特定于 Postgres 的代码，因此最好将这些代码分离到更具体的文件`conftest.py`中

`tests/repository/postgres/conftest.py`

```py
***import sqlalchemy
import pytest

from rentomatic.repository.postgres_objects import Base, Room

@pytest.fixture(scope="session")
def pg_session_empty(app_configuration):
    conn_str = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        app_configuration["POSTGRES_USER"],
        app_configuration["POSTGRES_PASSWORD"],
        app_configuration["POSTGRES_HOSTNAME"],
        app_configuration["POSTGRES_PORT"],
        app_configuration["APPLICATION_DB"],
    )
    engine = sqlalchemy.create_engine(conn_str)
    connection = engine.connect()

    Base.metadata.create_all(engine)
    Base.metadata.bind = engine

    DBSession = sqlalchemy.orm.sessionmaker(bind=engine)
    session = DBSession()

    yield session

    session.close()
    connection.close

@pytest.fixture(scope="session")
def pg_test_data():
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
def pg_session(pg_session_empty, pg_test_data):
    for r in pg_test_data:
        new_room = Room(
            code=r["code"],
            size=r["size"],
            price=r["price"],
            longitude=r["longitude"],
            latitude=r["latitude"],
        )
        pg_session_empty.add(new_room)
        pg_session_empty.commit()

    yield pg_session_empty

    pg_session_empty.query(Room).delete()*** 
```

第一个测试固定项`pg_session_empty`创建了一个到空初始数据库的会话，而`pg_test_data`定义了我们将要加载到数据库中的值。由于我们不会修改这组值，我们不需要创建测试固定项，但这是使其对其他测试固定项和测试都可用的一种更简单的方法。最后一个测试固定项`pg_session`使用测试数据填充数据库，这些数据是 Postgres 对象，我们创建了它们来映射这些对象。请注意，这些不是实体，而是我们创建来映射它们的 Postgres 对象。

请注意，最后一个测试固定项具有`function`作用域，因此它对每个测试都会运行。因此，在`yield`返回后，我们删除所有房间，使数据库的状态与测试之前完全相同。一般来说，你应该总是在测试后进行清理。我们正在测试的端点不会写入数据库，所以在这个特定情况下，实际上没有必要进行清理，但我更喜欢从零开始实现一个完整的解决方案。

我们可以通过更改`test_dummy`函数，使其获取`Room`表的全部行，并验证查询返回 4 个值来测试整个设置。

`tests/repository/postgres/test_postgresrepo.py`的新版本是

```py
***import pytest
from rentomatic.repository.postgres_objects import Room

pytestmark = pytest.mark.integration

def test_dummy(pg_session):
    assert len(pg_session.query(Room).all()) == 4*** 
```

在这个阶段，你可以运行带有集成测试的测试套件。你应该会注意到当 pytest 执行`test_dummy`函数时会有明显的延迟，因为 Docker 需要一些时间来启动数据库容器并准备数据

```py
***$ ./manage.py test -- --integration
========================= test session starts ===========================
platform linux -- Python XXXX, pytest-XXXX, py-XXXX, pluggy-XXXX --
cabook/venv3/bin/python3
cachedir: .cache
rootdir: cabook/code/calc, inifile: pytest.ini
plugins: cov-XXXX
collected 36 items

...
tests/repository/postgres/test_postgresrepo.py::test_dummy PASSED
...

========================= 36 passed in 0.26s ============================*** 
```

请注意，要传递`--integration`选项，我们需要使用`--`，否则 Click 会将该选项视为属于脚本`./manage.py`的一部分，而不是将其作为 pytest 参数传递。

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c06-s04`](https://github.com/pycabook/rentomatic/tree/ed2-c06-s04)****

## 集成测试

在这个阶段，我们可以在`test_postgresrepo.py`文件中创建真正的测试，替换`test_dummy`函数。所有测试都接收`app_configuration`、`pg_session`和`pg_test_data`测试固定项。第一个测试固定项允许我们使用适当的参数初始化`PostgresRepo`类。第二个使用测试数据创建数据库，这些数据随后包含在第三个测试固定项中。

这个存储库的测试基本上是创建用于`MemRepo`的测试的副本，这并不奇怪。通常，你想要测试完全相同的条件，无论存储系统是什么。然而，在章节的末尾，我们将看到，尽管这些文件最初是相同的，但当我们发现来自特定实现（内存存储、PostgreSQL 等）的 bug 或边缘情况时，它们可以以不同的方式发展。

`tests/repository/postgres/test_postgresrepo.py`

```py
***import pytest
from rentomatic.repository import postgresrepo

pytestmark = pytest.mark.integration

def test_repository_list_without_parameters(
    app_configuration, pg_session, pg_test_data
):
    repo = postgresrepo.PostgresRepo(app_configuration)

    repo_rooms = repo.list()

    assert set([r.code for r in repo_rooms]) == set(
        [r["code"] for r in pg_test_data]
    )

def test_repository_list_with_code_equal_filter(
    app_configuration, pg_session, pg_test_data
):
    repo = postgresrepo.PostgresRepo(app_configuration)

    repo_rooms = repo.list(
        filters={"code__eq": "fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a"}
    )

    assert len(repo_rooms) == 1
    assert repo_rooms[0].code == "fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a"

def test_repository_list_with_price_equal_filter(
    app_configuration, pg_session, pg_test_data
):
    repo = postgresrepo.PostgresRepo(app_configuration)

    repo_rooms = repo.list(filters={"price__eq": 60})

    assert len(repo_rooms) == 1
    assert repo_rooms[0].code == "913694c6-435a-4366-ba0d-da5334a611b2"

def test_repository_list_with_price_less_than_filter(
    app_configuration, pg_session, pg_test_data
):
    repo = postgresrepo.PostgresRepo(app_configuration)

    repo_rooms = repo.list(filters={"price__lt": 60})

    assert len(repo_rooms) == 2
    assert set([r.code for r in repo_rooms]) == {
        "f853578c-fc0f-4e65-81b8-566c5dffa35a",
        "eed76e77-55c1-41ce-985d-ca49bf6c0585",
    }

def test_repository_list_with_price_greater_than_filter(
    app_configuration, pg_session, pg_test_data
):
    repo = postgresrepo.PostgresRepo(app_configuration)

    repo_rooms = repo.list(filters={"price__gt": 48})

    assert len(repo_rooms) == 2
    assert set([r.code for r in repo_rooms]) == {
        "913694c6-435a-4366-ba0d-da5334a611b2",
        "fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a",
    }

def test_repository_list_with_price_between_filter(
    app_configuration, pg_session, pg_test_data
):
    repo = postgresrepo.PostgresRepo(app_configuration)

    repo_rooms = repo.list(filters={"price__lt": 66, "price__gt": 48})

    assert len(repo_rooms) == 1
    assert repo_rooms[0].code == "913694c6-435a-4366-ba0d-da5334a611b2"*** 
```

记住，我是一次介绍一个这些测试的，并且我没有展示完整的 TDD 工作流程，只是为了简洁起见。`PostgresRepo`类的代码是按照严格的 TDD 方法开发的，我建议你也这样做。生成的代码放在`rentomatic/repository/postgresrepo.py`，与我们在其中创建了`postgres_objects.py`文件的同一个目录中。

`rentomatic/repository/postgresrepo.py`

```py
***from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from rentomatic.domain import room
from rentomatic.repository.postgres_objects import Base, Room

class PostgresRepo:
    def __init__(self, configuration):
        connection_string = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
            configuration["POSTGRES_USER"],
            configuration["POSTGRES_PASSWORD"],
            configuration["POSTGRES_HOSTNAME"],
            configuration["POSTGRES_PORT"],
            configuration["APPLICATION_DB"],
        )

        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        Base.metadata.bind = self.engine

    def _create_room_objects(self, results):
        return [
            room.Room(
                code=q.code,
                size=q.size,
                price=q.price,
                latitude=q.latitude,
                longitude=q.longitude,
            )
            for q in results
        ]

    def list(self, filters=None):
        DBSession = sessionmaker(bind=self.engine)
        session = DBSession()

        query = session.query(Room)

        if filters is None:
            return self._create_room_objects(query.all())

        if "code__eq" in filters:
            query = query.filter(Room.code == filters["code__eq"])

        if "price__eq" in filters:
            query = query.filter(Room.price == filters["price__eq"])

        if "price__lt" in filters:
            query = query.filter(Room.price < filters["price__lt"])

        if "price__gt" in filters:
            query = query.filter(Room.price > filters["price__gt"])

        return self._create_room_objects(query.all())*** 
```

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c06-s05`](https://github.com/pycabook/rentomatic/tree/ed2-c06-s05)****

你可能注意到`PostgresRepo`与`MemRepo`非常相似。这是因为我们在这里处理的案例，即`Room`对象列表，相当简单，所以我不会期望内存数据库和现成的生产级关系型数据库之间有太大的差异。随着用例变得更加复杂，你将需要开始利用你使用的引擎提供的功能，例如`list`方法可能会演变成非常不同的形式。

请注意，`list`方法返回领域模型，这是允许的，因为存储库是在架构的外层之一实现的。

 * 

正如你所见，虽然设置一个合适的集成测试环境并不简单，但我们的架构为了与真实存储库一起工作所需要的变化非常有限。我认为这是清洁架构核心分层方法灵活性的良好证明。

由于本章将集成测试的设置与引入新的存储库结合起来，所以我将在下一章专门介绍一个基于 MongoDB 的存储库，使用与本章相同的结构。支持多个数据库（在这种情况下甚至包括关系型和非关系型）并不是一个不常见的模式，因为它允许你使用最适合每个用例的方法。

***1

除非你认为像`sessionmaker_mock()().query.assert_called_with(Room)`这样的东西很有吸引力。这绝对是我必须写的最简单的模拟之一。***
