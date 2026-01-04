# 第八章 - 运行生产就绪系统

> [`www.thedigitalcatbooks.com/pycabook-chapter-08/`](https://www.thedigitalcatbooks.com/pycabook-chapter-08/)
> 
> 维洛斯·科哈根说将使用部队来确保全面生产。
> 
> 《全面回忆》，1990

现在我们已经开发了一个与 PostgreSQL 连接的仓库，我们可以讨论如何正确设置应用程序以运行一个生产就绪的系统。这部分内容与干净的架构不是严格相关，但我认为完成这个例子是值得的，展示我们设计的系统最终可以成为真实网络应用的核心。

显然，“生产就绪”的定义指的是许多不同的配置，这些配置最终取决于系统的负载和业务需求。由于目标是展示一个完整的例子，而不是涵盖真正的生产需求，我将展示一个使用真实外部系统（如 PostgreSQL 和 Nginx）的解决方案，而不太关注性能。

## 构建一个网络栈

现在我们已经成功将测试容器化，我们可以尝试为整个应用程序设计一个生产就绪的设置，在 Docker 容器中运行一个网络服务器和一个数据库。再一次，我将遵循我在前一个部分提到的帖子系列中展示的方法。

要运行生产就绪的基础设施，我们需要在 Web 框架前面放置一个 WSGI 服务器，在它前面放置一个 Web 服务器。我们还需要运行一个数据库容器，我们只初始化一次。

向生产就绪配置迈进的步骤并不复杂，最终的设置最终也不会与我们为测试所做的设置有太大不同。我们需要

1.  创建一个包含适合生产的环境变量的 JSON 配置

1.  为 Docker Compose 创建一个合适的配置，并配置容器

1.  在 `manage.py` 中添加命令，以便我们可以控制进程

让我们创建一个名为 `config/production.json` 的文件，它与为测试创建的文件非常相似。

`config/production.json`

```py
[
  {
    "name": "FLASK_ENV",
    "value": "production"
  },
  {
    "name": "FLASK_CONFIG",
    "value": "production"
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
    "value": "5432"
  },
  {
    "name": "POSTGRES_PASSWORD",
    "value": "postgres"
  },
  {
    "name": "APPLICATION_DB",
    "value": "application"
  }
] 
```

请注意，现在 `FLASK_ENV` 和 `FLASK_CONFIG` 都设置为 `production`。请记住，第一个是一个 Flask 的内部变量，有两个可能的固定值（`development` 和 `production`），而第二个是一个任意名称，最终效果是加载一个特定的配置对象（在这种情况下是 `ProductionConfig`）。我还将 `POSTGRES_PORT` 改回默认的 `5432`，将 `APPLICATION_DB` 改为 `application`（一个任意名称）。

让我们定义我们希望在生产环境中运行的容器，以及我们希望如何连接它们。我们需要一个生产就绪的数据库，我将使用 Postgres，就像我在测试中已经做的那样。然后我们需要将 Flask 包装在一个生产 HTTP 服务器中，为此我将使用 `gunicorn`。最后，我们需要一个 Web 服务器作为负载均衡器。

`docker/production.yml` 文件将包含 Docker Compose 配置，根据我们在 `manage.py` 中定义的约定

`docker/production.yml`

```py
version: '3.8'

services:
  db:
    image: postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
  web:
    build:
      context: ${PWD}
      dockerfile: docker/web/Dockerfile.production
    environment:
      FLASK_ENV: ${FLASK_ENV}
      FLASK_CONFIG: ${FLASK_CONFIG}
      APPLICATION_DB: ${APPLICATION_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_HOSTNAME: "db"
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_PORT: ${POSTGRES_PORT}
    command: gunicorn -w 4 -b 0.0.0.0 wsgi:app
    volumes:
      - ${PWD}:/opt/code
  nginx:
    image: nginx
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - 8080:8080

volumes:
  pgdata: 
```

如您所见，Postgres 配置与我们在 `testing.yml` 文件中使用的配置没有区别，但我添加了 `volumes` 选项（在 `db` 和文件末尾），这允许我创建一个稳定的卷。如果您不这样做，容器关闭时数据库将被销毁。

容器 `web` 通过 `gunicorn` 运行 Flask 应用程序。环境变量再次来自 JSON 配置，我们需要定义它们，因为应用程序需要知道如何连接数据库以及如何运行 Web 框架。命令 `gunicorn -w 4 -b 0.0.0.0 wsgi:app` 加载我们在 `wsgi.py` 中创建的 WSGI 应用程序，并以 4 个并发进程运行。这个容器是通过 `docker/web/Dockerfile.production` 创建的，我还需要定义它。

最后一个容器是 `nginx`，我们将直接从 Docker Hub 使用它。该容器运行 Nginx，配置存储在 `/etc/nginx/nginx.conf` 中，这是我们用本地文件 `./nginx/nginx.conf` 覆盖的文件。请注意，我将其配置为使用端口 8080 而不是标准的 HTTP 端口 80，以避免与您可能在计算机上运行的其它软件冲突。

Web 应用的 Dockerfile 如下

`docker/web/Dockerfile.production`

```py
FROM python:3

ENV PYTHONUNBUFFERED 1

RUN mkdir /opt/code
RUN mkdir /opt/requirements
WORKDIR /opt/code

ADD requirements /opt/requirements
RUN pip install -r /opt/requirements/prod.txt 
```

这是一个非常简单的容器，使用标准的 `python:3` 镜像，我在其中添加了 `requirements/prod.txt` 中包含的生产需求。为了让 Docker 容器工作，我们需要将 `gunicorn` 添加到这个最后的文件中

`requirements/prod.txt`

```py
Flask
SQLAlchemy
psycopg2
pymongo
gunicorn 
```

Nginx 的配置如下

`docker/nginx/nginx.conf`

```py
worker_processes 1;

events { worker_connections 1024; }

http {

    sendfile on;

    upstream app {
        server web:8000;
    }

    server {
        listen 8080;

        location / {
            proxy_pass         http://app;
            proxy_redirect     off;
            proxy_set_header   Host $host;
            proxy_set_header   X-Real-IP $remote_addr;
            proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header   X-Forwarded-Host $server_name;
        }
    }
} 
```

对于项目的其余部分，这个配置非常基础，缺少了一些在真实生产环境中强制要求的某些重要部分，例如 HTTPS。然而，从本质上讲，它却与准备就绪的生产环境中的 Nginx 容器配置没有太大区别。

由于我们将使用 Docker Compose，`manage.py` 需要简单的修改，即一个包装 `docker-compose` 自身的命令。我们需要脚本根据 JSON 配置文件的内容初始化环境变量，然后运行 Docker Compose。由于我们已经有 `docker_compose_cmdline` 函数，这项工作相当简单

`manage.py`

```py
# Ensure an environment variable exists and has a value
import os
import json
import signal
import subprocess
import time

...

def setenv(variable, default):
    os.environ[variable] = os.getenv(variable, default)

setenv("APPLICATION_CONFIG", "production")

APPLICATION_CONFIG_PATH = "config"
DOCKER_PATH = "docker"

...

@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("subcommand", nargs=-1, type=click.Path())
def compose(subcommand):
    configure_app(os.getenv("APPLICATION_CONFIG"))
    cmdline = docker_compose_cmdline() + list(subcommand)

    try:
        p = subprocess.Popen(cmdline)
        p.wait()
    except KeyboardInterrupt:
        p.send_signal(signal.SIGINT)
        p.wait() 
```

如您所见，我强制将变量 `APPLICATION_CONFIG` 设置为 `production`，如果没有指定。通常，我的默认配置是开发配置，但在这个简单的情况下，我没有定义一个，所以现在这样就可以了。

新的命令是 `compose`，它利用 Click 的 `argument` 装饰器来收集子命令并将它们附加到 Docker Compose 命令行。我还使用了 `signal` 库，我将其添加到导入中，以控制键盘中断。

*源代码

[`github.com/pycabook/rentomatic/tree/ed2-c08-s01`](https://github.com/pycabook/rentomatic/tree/ed2-c08-s01)

当所有这些更改都到位后，我们可以测试应用程序的 Dockerfile 构建容器

```py
*$ ./manage.py compose build web* 
```

这个命令运行 Click 命令`compose`，首先从`config/production.json`文件中读取环境变量，然后运行`docker-compose`，并传递子命令`build web`。

你的输出应该是以下内容（带有不同的镜像 ID）

```py
*Building web
Step 1/7 : FROM python:3
 ---> 768307cdb962
Step 2/7 : ENV PYTHONUNBUFFERED 1
 ---> Using cache
 ---> 0f2bb60286d3
Step 3/7 : RUN mkdir /opt/code
 ---> Using cache
 ---> e1278ef74291
Step 4/7 : RUN mkdir /opt/requirements
 ---> Using cache
 ---> 6d23f8abf0eb
Step 5/7 : WORKDIR /opt/code
 ---> Using cache
 ---> 8a3b6ae6d21c
Step 6/7 : ADD requirements /opt/requirements
 ---> Using cache
 ---> 75133f765531
Step 7/7 : RUN pip install -r /opt/requirements/prod.txt
 ---> Using cache
 ---> db644df9ba04

Successfully built db644df9ba04
Successfully tagged production_web:latest* 
```

如果这成功了，你可以运行 Docker Compose

```py
*$ ./manage.py compose up -d
Creating production_web_1   ... done
Creating production_db_1    ... done
Creating production_nginx_1 ... done* 
```

并且`docker ps`的输出应该显示三个正在运行的容器

```py
*$ docker ps
IMAGE          PORTS                            NAMES
nginx          80/tcp, 0.0.0.0:8080->8080/tcp   production_nginx_1
postgres       0.0.0.0:5432->5432/tcp           production_db_1
production_web                                  production_web_1* 
```

注意，我删除了几个列以使输出可读。

在这个阶段，我们可以用我们的浏览器打开[`localhost:8080/rooms`](http://localhost:8080/rooms)，并查看 Nginx 接收到的 HTTP 请求的结果，然后传递给 gunicorn，并由 Flask 使用`room_list_use_case`用例处理

应用程序实际上还没有使用数据库，因为`application/rest/room.py`中的 Flask 端点`room_list`初始化了`MemRepo`类，并用一些静态值加载它，这些值就是我们浏览器中看到的

## 连接到一个生产就绪的数据库

在我们开始更改应用程序的代码之前，请记住要拆除正在运行的系统

```py
*$ ./manage.py compose down
Stopping production_web_1   ... done
Stopping production_nginx_1 ... done
Stopping production_db_1    ... done
Removing production_web_1   ... done
Removing production_nginx_1 ... done
Removing production_db_1    ... done
Removing network production_default* 
```

由于存储库之间的公共接口，从基于内存的`MemRepo`到`PostgresRepo`的迁移非常简单。显然，由于外部数据库最初不会包含任何数据，用例的响应将是空的。

首先，让我们将应用程序移动到 Postgres 仓库。端点的新版本是

`application/rest/room.py`

```py
*import os
import json

from flask import Blueprint, request, Response

from rentomatic.repository.postgresrepo import PostgresRepo
from rentomatic.use_cases.room_list import room_list_use_case
from rentomatic.serializers.room import RoomJsonEncoder
from rentomatic.requests.room_list import build_room_list_request
from rentomatic.responses import ResponseTypes

blueprint = Blueprint("room", __name__)

STATUS_CODES = {
    ResponseTypes.SUCCESS: 200,
    ResponseTypes.RESOURCE_ERROR: 404,
    ResponseTypes.PARAMETERS_ERROR: 400,
    ResponseTypes.SYSTEM_ERROR: 500,
}

postgres_configuration = {
    "POSTGRES_USER": os.environ["POSTGRES_USER"],
    "POSTGRES_PASSWORD": os.environ["POSTGRES_PASSWORD"],
    "POSTGRES_HOSTNAME": os.environ["POSTGRES_HOSTNAME"],
    "POSTGRES_PORT": os.environ["POSTGRES_PORT"],
    "APPLICATION_DB": os.environ["APPLICATION_DB"],
}

@blueprint.route("/rooms", methods=["GET"])
def room_list():
    qrystr_params = {
        "filters": {},
    }

    for arg, values in request.args.items():
        if arg.startswith("filter_"):
            qrystr_params["filters"][arg.replace("filter_", "")] = values

    request_object = build_room_list_request(
        filters=qrystr_params["filters"]
    )

    repo = PostgresRepo(postgres_configuration)
    response = room_list_use_case(repo, request_object)

    return Response(
        json.dumps(response.value, cls=RoomJsonEncoder),
        mimetype="application/json",
        status=STATUS_CODES[response.type],
    )* 
```

如你所见，主要的变化是`repo = MemRepo(rooms)`变成了`repo = PostgresRepo(postgres_configuration)`。这样的简单变化是通过干净的架构和其严格的分层方法实现的。唯一的另一个显著变化是我们用包含连接数据的字典替换了基于内存的存储库的初始数据，这些数据来自管理脚本设置的环境变量。

这足以让应用程序连接到我们在容器中运行的 Postgres 数据库，但正如我提到的，我们还需要初始化数据库。我们需要的最基本的是具有正确名称的空数据库。记住，在这个特定的设置中，我们为应用程序使用了一个不同的数据库（`APPLICATION_DB`），而不是 Postgres 容器在启动时自动创建的数据库（`POSTGRES_DB`）。我在管理脚本中添加了一个特定的命令来执行这个任务

`manage.py`

```py
*@cli.command()
def init_postgres():
    configure_app(os.getenv("APPLICATION_CONFIG"))

    try:
        run_sql([f"CREATE DATABASE {os.getenv('APPLICATION_DB')}"])
    except psycopg2.errors.DuplicateDatabase:
        print(
            (
                f"The database {os.getenv('APPLICATION_DB')} already",
                "exists and will not be recreated",
            )
        )* 
```

现在启动你的容器

```py
*$ ./manage.py compose up -d
Creating network "production_default" with the default driver
Creating volume "production_pgdata" with default driver
Creating production_web_1   ... done
Creating production_nginx_1 ... done
Creating production_db_1    ... done* 
```

并运行我们创建的新命令

```py
*$ ./manage.py init-postgres* 
```

注意函数`init_postgres`的名称和命令`init-postgres`的名称之间的变化。你只需要运行这个命令一次，但重复执行不会影响数据库。

我们可以通过连接到数据库来检查这个命令做了什么。我们可以通过在数据库容器中执行 `psql` 来做到这一点。

```py
*$ ./manage.py compose exec db psql -U postgres
psql (13.4 (Debian 13.4-1.pgdg100+1))
Type "help" for help.

postgres=#* 
```

请注意，我们需要指定用户 `-U postgres`。这是我们通过 `config/production.json` 中的变量 `POSTGRES_USER` 创建的用户。一旦登录，我们可以使用命令 `\l` 来查看可用的数据库。

```py
*postgres=# \l
                                  List of databases
    Name     |  Owner   | Encoding |  Collate   |   Ctype    |   Access privileges
-------------+----------+----------+------------+------------+----------------------
 application | postgres | UTF8     | en_US.utf8 | en_US.utf8 | 
 postgres    | postgres | UTF8     | en_US.utf8 | en_US.utf8 | 
 template0   | postgres | UTF8     | en_US.utf8 | en_US.utf8 | =c/postgres          +
             |          |          |            |            | postgres=CTc/postgres
 template1   | postgres | UTF8     | en_US.utf8 | en_US.utf8 | =c/postgres          +
             |          |          |            |            | postgres=CTc/postgres
(4 rows)

postgres=#* 
```

请注意，`template0` 和 `template1` 这两个数据库是 Postgres 创建的系统数据库（参见[文档](https://www.postgresql.org/docs/current/manage-ag-templatedbs.html)），`postgres` 是 Docker 容器创建的默认数据库（默认名称为 `postgres`，但在此情况下它来自 `config/production.json` 中的环境变量 `POSTGRES_DB`），而 `application` 是通过 `./manage.py init-postgres` 创建的数据库（来自 `APPLICATION_DB`）。

我们可以使用命令 `\c` 连接到数据库。

```py
*postgres=# \c application 
You are now connected to database "application" as user "postgres".
application=#* 
```

请注意，提示符会根据当前数据库的名称而改变。最后，我们可以使用 `\dt` 列出可用的表。

```py
*application=# \dt
Did not find any relations.* 
```

正如你所见，还没有任何表。这并不奇怪，因为我们没有做任何使 Postres 了解我们创建的模型的事情。请记住，我们在这里所做的一切都是在外部系统中完成的，并且它并没有直接与实体相连。

正如你所记得的，我们将实体映射到存储对象，因为我们使用的是 Postgres，所以我们利用了 SQLAlchemy 类，所以现在我们需要创建与它们相对应的数据库表。

### 迁移

我们需要一种方法来创建与我们在 `rentomatic/repository/postgres_objects.py` 中定义的对象相对应的表。当我们使用像 SQLAlchemy 这样的 ORM 时，最佳策略是创建和运行迁移，为此我们可以使用 [Alembic](https://alembic.sqlalchemy.org/)。

如果你仍然通过 `psql` 连接，请使用 `\q` 退出，然后编辑 `requirements/prod.txt` 并添加 `alembic`。

```py
*Flask
SQLAlchemy
psycopg2
pymongo
gunicorn
alembic* 
```

像往常一样，请记住运行 `pip install -r requirements/dev.txt` 来更新虚拟环境。

Alembic 能够连接到数据库并运行 Python 脚本（称为“迁移”），以根据 SQLAlchemy 模型更改表。然而，为了做到这一点，我们需要提供用户名、密码、主机名和数据库名称以供 Alembic 访问数据库。我们还需要提供 Alembic 访问代表模型的 Python 类的权限。

首先，让我们初始化 Alembic。在项目的主要目录（`manage.py` 存储的地方）运行

```py
*$ alembic init migrations* 
```

它将创建一个名为 `migrations` 的目录，该目录包含 Alembic 的配置文件，以及将在 `migrations/versions` 中创建的迁移。它还将创建一个名为 `alembic.ini` 的文件，其中包含配置值。`migrations` 的名称完全是任意的，所以如果你更喜欢，可以自由使用不同的名称。

我们需要调整的特定文件是 `migrations/env.py`，以便 Alembic 了解我们的模型和数据库。添加高亮行

migrations/env.py

```py
*import os 
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

section = config.config_ini_section config.set_section_option(
 section, "POSTGRES_USER", os.environ.get("POSTGRES_USER") ) config.set_section_option(
 section, "POSTGRES_PASSWORD", os.environ.get("POSTGRES_PASSWORD") ) config.set_section_option(
 section, "POSTGRES_HOSTNAME", os.environ.get("POSTGRES_HOSTNAME") ) config.set_section_option(
 section, "APPLICATION_DB", os.environ.get("APPLICATION_DB") ) 
# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
# target_metadata = None from rentomatic.repository.postgres_objects import Base 
target_metadata = Base.metadata 
# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.* 
```

通过 `config.set_section_option` 我们将相关的配置值添加到主 Alembic INI 文件部分 (`config.config_ini_section`)，并从环境变量中提取它们。我们还在导入包含 SQLAlchemy 对象的文件。您可以在 [`alembic.sqlalchemy.org/en/latest/api/config.html`](https://alembic.sqlalchemy.org/en/latest/api/config.html) 找到有关此过程的文档。

完成此操作后，我们需要将 INI 文件更改为使用新变量

alembic.ini

```py
*# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = postgresql://%(POSTGRES_USER)s:%(POSTGRES_PASSWORD)s@%(POSTGRES_HOSTNAME)s/%(APPLICATION_DB)s 
[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples* 
```

`%(VARNAME)s` 语法是 `ConfigParser`（见 [文档](https://docs.python.org/3.8/library/configparser.html#configparser.BasicInterpolation)）使用的基本变量插值。

此时，我们可以运行 Alembic 来迁移我们的数据库。在许多情况下，您可以使用 Alembic 的自动生成功能来生成迁移，这就是我们创建初始模型的方法。Alembic 命令是带有 `--autogenerate` 标志的 `revision`，但我们需要在命令行中传递环境变量。这显然是 `migrate.py` 的工作，但让我们首先运行它看看数据库会发生什么。稍后我们将创建一个更好的设置来避免手动传递变量。

```py
*$ POSTGRES_USER=postgres\
  POSTGRES_PASSWORD=postgres\
  POSTGRES_HOSTNAME=localhost\
  APPLICATION_DB=application\
  alembic revision --autogenerate -m "Initial"* 
```

这将生成文件 `migrations/versions/4d4c19952a36_initial.py`。请注意，您的初始哈希值将不同。如果您想，您可以打开该文件，看看 Alembic 如何生成表和创建列。

到目前为止，我们已经创建了迁移，但我们仍然需要将其应用到数据库中。确保您正在运行 Docker 容器（否则运行 `./manage.py compose up -d`），因为 Alembic 将连接到数据库，并运行

```py
*$ POSTGRES_USER=postgres\
  POSTGRES_PASSWORD=postgres\
  POSTGRES_HOSTNAME=localhost\
  APPLICATION_DB=application\
  alembic upgrade head* 
```

此时，我们可以连接到数据库并检查现有的表

```py
*$ ./manage.py compose exec db psql -U postgres -d application
psql (13.4 (Debian 13.4-1.pgdg100+1))
Type "help" for help.

application=# \dt
              List of relations
 Schema |      Name       | Type  |  Owner   
--------+-----------------+-------+----------
 public | alembic_version | table | postgres
 public | room            | table | postgres
(2 rows)

application=#* 
```

请注意，我使用了 `psql` 的 `-d` 选项直接连接到数据库 `application`。如您所见，现在我们有两个表。第一个 `alembic_version` 是一个简单的表，Alembic 使用它来跟踪数据库的状态，而 `room` 是将包含我们的 `Room` 实体的表。

我们可以再次检查 Alembic 版本

```py
*application=# select * from alembic_version;
 version_num  
--------------
 4d4c19952a36
(1 row)* 
```

如我之前提到的，分配给迁移的哈希值在您的案例中将是不同的，但您在这个表中看到的价值应该与迁移脚本的名称一致。

我们还可以看到 `room` 表的结构

```py
*application=# \d room
                                     Table "public.room"
  Column   |         Type          | Collation | Nullable |             Default              
-----------+-----------------------+-----------+----------+----------------------------------
 id        | integer               |           | not null | nextval('room_id_seq'::regclass)
 code      | character varying(36) |           | not null | 
 size      | integer               |           |          | 
 price     | integer               |           |          | 
 longitude | double precision      |           |          | 
 latitude  | double precision      |           |          | 
Indexes:
    "room_pkey" PRIMARY KEY, btree (id)* 
```

显然，该表中还没有包含任何行

```py
*application=# select * from room;
 id | code | size | price | longitude | latitude 
----+------+------+-------+-----------+----------
(0 rows)* 
```

确实，如果您用浏览器打开 [`localhost:8080/rooms`](http://localhost:8080/rooms)，您将看到一个成功的响应，但没有数据。

为了查看一些数据，我们需要将一些内容写入数据库。这通常是通过网页应用程序中的表单和特定端点来完成的，但为了简化，在这种情况下，我们只需手动将数据添加到数据库中。

```py
*application=# INSERT INTO room(code, size, price, longitude, latitude) VALUES ('f853578c-fc0f-4e65-81b8-566c5dffa35a', 215, 39, -0.09998975, 51.75436293);
INSERT 0 1* 
```

您可以通过 `SELECT` 验证表中是否包含新的房间

```py
*application=# SELECT * FROM room;
 id |                 code                 | size | price |  longitude  |  latitude   
----+--------------------------------------+------+-------+-------------+-------------
  1 | f853578c-fc0f-4e65-81b8-566c5dffa35a |  215 |    39 | -0.09998975 | 51.75436293
(1 row)* 
```

使用浏览器打开或刷新[`localhost:8080/rooms`](http://localhost:8080/rooms)以查看我们用例返回的值。

源代码

[`github.com/pycabook/rentomatic/tree/ed2-c08-s02`](https://github.com/pycabook/rentomatic/tree/ed2-c08-s02)

 * 

本章总结了清洁架构示例的概述。我们从零开始，创建了领域模型、序列化器、用例、内存存储系统、命令行界面和 HTTP 端点。然后，我们通过一个非常通用的请求/响应管理代码改进了整个系统，该代码为错误提供了健壮的支持。最后，我们实现了两个新的存储系统，使用了关系型数据库和非关系型数据库。

这绝对不是一个小成就。我们的架构覆盖了一个非常小的用例，但它是健壮的并且经过了全面测试。无论我们在处理数据、数据库、请求等方面可能发现的任何错误，都可以比没有测试的系统更快地隔离和驯服。此外，解耦哲学不仅使我们能够为多个存储系统提供支持，而且能够快速实现新的访问协议，或为我们对象实现新的序列化方式。
