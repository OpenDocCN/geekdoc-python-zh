# 第三章 - 基本示例

> 原文：[`www.thedigitalcatbooks.com/pycabook-chapter-03/`](https://www.thedigitalcatbooks.com/pycabook-chapter-03/)
> 
> 约书亚/WOPR：你不想玩一局好棋吗？
> 
> 大卫：稍后。让我们玩全球热核战争。
> 
> 1983 年，战争游戏

“Rent-o-Matic”项目的目标是创建一个简单的房间租赁公司搜索引擎。数据集中的对象（房间）由一些属性描述，搜索引擎应允许用户设置一些过滤器以缩小搜索范围。

系统中存储房间通过以下值：

+   唯一标识符

+   平方米大小的面积

+   每日欧元租金

+   纬度和经度

描述故意保持最小化，这样我们就可以专注于架构问题和如何解决它们。我将展示的概念可以很容易地扩展到更复杂的情况。

如同干净的架构模型所推动的，我们感兴趣的是分离系统的不同层。记住，实现干净架构概念的方法有很多种，你可以编写的代码强烈取决于你选择的语言允许你做什么。以下是一个 Python 中干净架构的示例，我将展示的模型、用例和其他组件的实现只是可能的解决方案之一。

### 项目设置

克隆[项目仓库](https://github.com/pycabook/rentomatic)并切换到`second-edition`分支。完整解决方案包含在`second-edition-top`分支中，我将会提到的标签也都在那里。我强烈建议你边编码边工作，并且只有在发现错误时才使用我的标签。

```py
$ git clone https://github.com/pycabook/rentomatic
$ cd rentomatic
$ git checkout --track origin/second-edition 
```

按照你喜欢的流程创建一个虚拟环境并安装需求

```py
$ pip install -r requirements/dev.txt 
```

到目前为止，你应该能够运行

```py
$ pytest -svv 
```

并得到如下输出

```py
=========================== test session starts ===========================
platform linux -- Python XXXX, pytest-XXXX, py-XXXX, pluggy-XXXX --
cabook/venv3/bin/python3
cachedir: .cache
rootdir: cabook/code/calc, inifile: pytest.ini
plugins: cov-XXXX
collected 0 items 

========================== no tests ran in 0.02s ========================== 
```

在项目的后期，你可能想查看覆盖率检查的输出，因此你可以通过以下方式激活它

```py
$ pytest -svv --cov=rentomatic --cov-report=term-missing 
```

在本章中，我不会明确指出何时运行测试套件，因为我认为它是标准工作流程的一部分。每次我们编写测试时，都应该运行套件并检查你是否得到了错误（或更多），而我提供的解决方案应该使测试套件通过。显然，你可以在复制我的解决方案之前尝试实现自己的代码。

你可能会注意到，我配置了项目使用黑色，并且使用非传统的行长度 75。我选择这个数字是为了找到一种视觉上令人愉悦的方式来在书中展示代码，避免折行，因为折行可能会使代码难以阅读。

*源代码

[`github.com/pycabook/rentomatic/tree/second-edition`](https://github.com/pycabook/rentomatic/tree/second-edition)*

### 领域模型

让我们从一个简单的`Room`模型定义开始。正如之前所说，干净的架构模型非常轻量级，或者至少它们比常见的 Web 框架中的对应模型更轻量级。

遵循 TDD 方法论，我首先编写的是测试。这个测试确保模型可以用正确的值初始化

`tests/domain/test_room.py`

```py
*import uuid
from rentomatic.domain.room import Room

def test_room_model_init():
    code = uuid.uuid4()
    room = Room(
        code,
        size=200,
        price=10,
        longitude=-0.09998975,
        latitude=51.75436293,
    )

    assert room.code == code
    assert room.size == 200
    assert room.price == 10
    assert room.longitude == -0.09998975
    assert room.latitude == 51.75436293* 
```

记住，在创建的每个`tests/`子目录中创建一个空的`__init__.py`文件，在这个例子中是`tests/domain/__init__.py`。

现在让我们在文件`rentomatic/domain/room.py`中编写`Room`类。

`rentomatic/domain/room.py`

```py
*import uuid
import dataclasses

@dataclasses.dataclass
class Room:
    code: uuid.UUID
    size: int
    price: int
    longitude: float
    latitude: float* 
```

**源代码

[`github.com/pycabook/rentomatic/tree/ed2-c03-s01`](https://github.com/pycabook/rentomatic/tree/ed2-c03-s01)**

该模型非常简单，不需要太多解释。我使用 dataclasses，因为它们是实现这种简单模型的一种紧凑方式，但你也可以自由地使用标准类并显式实现`__init__`方法。

鉴于我们将从其他层接收初始化此模型的数据，并且这些数据很可能是字典类型，创建一个允许我们从这种结构初始化模型的方法是有用的。代码可以放在我们之前创建的同一文件中，并且是

`tests/domain/test_room.py`

```py
**def test_room_model_from_dict():
    code = uuid.uuid4()
    init_dict = {
        "code": code,
        "size": 200,
        "price": 10,
        "longitude": -0.09998975,
        "latitude": 51.75436293,
    }

    room = Room.from_dict(init_dict)

    assert room.code == code
    assert room.size == 200
    assert room.price == 10
    assert room.longitude == -0.09998975
    assert room.latitude == 51.75436293** 
```

它的简单实现如下

`rentomatic/domain/room.py`

```py
**@dataclasses.dataclass
class Room:
    code: uuid.UUID
    size: int
    price: int
    longitude: float
    latitude: float

    @classmethod
    def from_dict(cls, d):
        return cls(**d)** 
```

***源代码

[`github.com/pycabook/rentomatic/tree/ed2-c03-s02`](https://github.com/pycabook/rentomatic/tree/ed2-c03-s02)***

出于与之前提到相同的原因，能够将模型转换为字典是有用的，这样我们就可以轻松地将它序列化为 JSON 或类似的语言无关格式。`to_dict`方法的测试再次放在`tests/domain/test_room.py`中

`tests/domain/test_room.py`

```py
***def test_room_model_to_dict():
    init_dict = {
        "code": uuid.uuid4(),
        "size": 200,
        "price": 10,
        "longitude": -0.09998975,
        "latitude": 51.75436293,
    }

    room = Room.from_dict(init_dict)

    assert room.to_dict() == init_dict*** 
```

使用 dataclasses 的实现是微不足道的

`rentomatic/domain/room.py`

```py
 ***def to_dict(self):
        return dataclasses.asdict(self)*** 
```

如果你不使用 dataclasses，你需要显式创建字典，但这也不构成任何挑战。请注意，这还不是对象的序列化，因为结果仍然是一个 Python 数据结构，而不是字符串。

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c03-s03`](https://github.com/pycabook/rentomatic/tree/ed2-c03-s03)****

能够比较模型实例也是非常有用的。测试与之前的测试放在同一个文件中

`tests/domain/test_room.py`

```py
***def test_room_model_comparison():
    init_dict = {
        "code": uuid.uuid4(),
        "size": 200,
        "price": 10,
        "longitude": -0.09998975,
        "latitude": 51.75436293,
    }

    room1 = Room.from_dict(init_dict)
    room2 = Room.from_dict(init_dict)

    assert room1 == room2*** 
```

再次，dataclasses 使这变得非常简单，因为它们提供了`__eq__`的实现。如果你不使用 dataclasses 实现类，你必须定义这个方法以使其通过测试。

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c03-s04`](https://github.com/pycabook/rentomatic/tree/ed2-c03-s04)****

### 序列化器

外层可以使用`Room`模型，但如果你想在 API 调用中将模型作为结果返回，你需要一个序列化器。

典型的序列化格式是 JSON，因为这是基于 Web 的 API 广泛接受的标准化格式。序列化器不是模型的一部分，而是一个外部专用类，它接收模型实例并生成其结构和值的表示。

这是对我们`Room`类 JSON 序列化的测试

`tests/serializers/test_room.py`

```py
***import json
import uuid

from rentomatic.serializers.room import RoomJsonEncoder
from rentomatic.domain.room import Room

def test_serialize_domain_room():
    code = uuid.uuid4()

    room = Room(
        code=code,
        size=200,
        price=10,
        longitude=-0.09998975,
        latitude=51.75436293,
    )

    expected_json = f"""
  {{ "code": "{code}",
 "size": 200,
 "price": 10,
 "longitude": -0.09998975,
 "latitude": 51.75436293
  }} """

    json_room = json.dumps(room, cls=RoomJsonEncoder)

    assert json.loads(json_room) == json.loads(expected_json)*** 
```

在这里，我们创建`Room`对象并写入预期的 JSON 输出（请注意，使用双大括号是为了避免与 f-string 格式化器的冲突）。然后我们将`Room`对象序列化为 JSON 字符串并比较两者。为了比较两者，我们再次将它们加载到 Python 字典中，以避免属性顺序的问题。实际上，比较 Python 字典时并不考虑字典字段的顺序，而比较字符串显然是考虑顺序的。

在文件`rentomatic/serializers/room.py`中放入使测试通过的代码

`rentomatic/serializers/room.py`

```py
***import json

class RoomJsonEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            to_serialize = {
                "code": str(o.code),
                "size": o.size,
                "price": o.price,
                "latitude": o.latitude,
                "longitude": o.longitude,
            }
            return to_serialize
        except AttributeError:  # pragma: no cover
            return super().default(o)*** 
```

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c03-s05`](https://github.com/pycabook/rentomatic/tree/ed2-c03-s05)****

提供一个从`json.JSONEncoder`继承的类，让我们可以使用`json_room = json.dumps(room, cls=RoomJsonEncoder)`的语法来序列化模型。请注意，我们没有使用`as_dict`方法，因为 UUID 代码不能直接序列化为 JSON。这意味着在两个类中存在一定程度的代码重复，在我看来这是可以接受的，因为它们被测试所覆盖。如果您愿意，您也可以调用`as_dict`方法，然后调整代码字段，将其转换为`str`。

### 用例

现在是时候实现我们应用程序内部实际运行的业务逻辑了。用例是这种情况发生的地方，它们可能与系统的外部 API 直接相关，也可能不直接相关。

我们可以创建的最简单的用例是获取存储在仓库中的所有房间并返回它们的用例。在这个第一部分，我们不会实现用于缩小搜索范围的过滤器。这部分代码将在下一章介绍，届时我们将讨论错误管理。

仓库是我们的存储组件，根据清洁架构，它将在外部级别（外部系统）实现。我们将通过接口访问它，在 Python 中这意味着我们将接收一个我们期望会公开一定 API 的对象。从测试的角度来看，运行访问接口的代码的最佳方式是模拟后者。将此代码放入文件`tests/use_cases/test_room_list.py`

我将利用 pytest 强大的固定功能，但不会介绍它们。我强烈推荐阅读[官方文档](https://docs.pytest.org/en/stable/fixture.html)，它非常好，涵盖了多种不同的用例。

`tests/use_cases/test_room_list.py`

```py
***import pytest
import uuid
from unittest import mock

from rentomatic.domain.room import Room
from rentomatic.use_cases.room_list import room_list_use_case

@pytest.fixture
def domain_rooms():
    room_1 = Room(
        code=uuid.uuid4(),
        size=215,
        price=39,
        longitude=-0.09998975,
        latitude=51.75436293,
    )

    room_2 = Room(
        code=uuid.uuid4(),
        size=405,
        price=66,
        longitude=0.18228006,
        latitude=51.74640997,
    )

    room_3 = Room(
        code=uuid.uuid4(),
        size=56,
        price=60,
        longitude=0.27891577,
        latitude=51.45994069,
    )

    room_4 = Room(
        code=uuid.uuid4(),
        size=93,
        price=48,
        longitude=0.33894476,
        latitude=51.39916678,
    )

    return [room_1, room_2, room_3, room_4]

def test_room_list_without_parameters(domain_rooms):
    repo = mock.Mock()
    repo.list.return_value = domain_rooms

    result = room_list_use_case(repo)

    repo.list.assert_called_with()
    assert result == domain_rooms*** 
```

测试很简单。首先，我们模拟仓库，使其提供一个返回我们上面测试中创建的模型列表的`list`方法。然后我们使用仓库初始化用例并执行它，收集结果。我们首先检查仓库方法是否被无参数调用，其次是结果的正确性。

调用仓库的`list`方法是用例应该执行的一个出站查询动作，根据单元测试规则，我们不应该测试出站查询。然而，我们应该测试我们的系统如何运行出站查询，即运行查询使用的参数。

将用例的实现放在文件`rentomatic/use_cases/room_list.py`中

`rentomatic/use_cases/room_list.py`

```py
***def room_list_use_case(repo):
    return repo.list()*** 
```

这样的解决方案可能看起来过于简单，所以让我们来讨论一下。首先，这个用例只是围绕仓库的特定函数的一个包装器，并且它不包含任何错误检查，这是我们还没有考虑到的。在下一章中，我们将讨论请求和响应，用例将变得稍微复杂一些。

你可能注意到的下一件事是我使用了一个简单的函数。在本书的第一版中，我用类来表示用例，并且感谢几位读者的提醒，我开始质疑我的选择，所以我想简要讨论一下你的选择。

用例表示业务逻辑，一个过程，这意味着在编程语言中最简单的实现是一个函数：一些接收输入参数并返回输出数据的代码。然而，类也是一个选项，因为本质上它是一组变量和函数的集合。所以，就像在许多其他情况下一样，问题是你是否应该使用函数或类，我的回答是这取决于你正在实现的算法的复杂程度。

你的业务逻辑可能很复杂，需要与几个外部系统连接，尽管每个系统都有特定的初始化，但在这种简单的情况下，我只是传递了仓库。所以，原则上，我不认为使用类来表示用例有什么错误，如果你需要为你的算法提供更多结构，但要注意不要在更简单的解决方案（函数）可以完成相同工作的场合使用它们，这是我在代码的前一个版本中犯的错误。记住，代码需要维护，所以越简单越好。

源代码

[`github.com/pycabook/rentomatic/tree/ed2-c03-s06`](https://github.com/pycabook/rentomatic/tree/ed2-c03-s06)****

### 存储系统

在开发用例的过程中，我们假设它会接收一个包含数据和暴露`list`函数的对象。这个对象通常被称为“仓库”，它是用例的信息来源。不过，它与 Git 仓库无关，所以请注意不要混淆这两个术语。

存储位于清洁架构的第四层，即外部系统。这一层的元素通过一个接口被内部元素访问，在 Python 中这仅仅意味着暴露一组特定的方法（在这种情况下只有`list`）。值得注意的是，在清洁架构中，仓库提供的抽象级别高于在框架中 ORM 提供的，或者由像 SQLAlchemy 这样的工具提供的。仓库只提供应用程序需要的端点，并且接口是根据应用程序实现的具体业务问题定制的。

为了具体说明问题，以 SQLAlchemy 为例，它是一个抽象访问 SQL 数据库的出色工具，因此仓库的内部实现可以使用它来访问 PostgreSQL 数据库等。但层的外部 API 并不是 SQLAlchemy 提供的。API 是一组减少的函数，用例通过这些函数调用以获取数据，而内部实现可以使用广泛的解决方案来实现相同的目标，从原始 SQL 查询到通过 RabbitMQ 网络进行的复杂远程调用系统。

仓库的一个重要特性是它可以返回领域模型，这与框架 ORM 通常的做法一致。第三层的元素可以访问内部层中定义的所有元素，这意味着领域模型和用例可以直接从仓库中调用和使用。

为了这个简单的示例，我们不会部署和使用真实的数据库系统。根据我们所说的，我们可以自由地使用最适合我们需求的系统来实现仓库，在这种情况下，我想保持一切简单。因此，我们将创建一个非常简单的内存存储系统，并加载一些预定义的数据。

首先需要编写一些测试来记录仓库的公共 API。包含测试的文件是`tests/repository/test_memrepo.py`。

`tests/repository/test_memrepo.py`

```py
***import pytest

from rentomatic.domain.room import Room
from rentomatic.repository.memrepo import MemRepo

@pytest.fixture
def room_dicts():
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

def test_repository_list_without_parameters(room_dicts):
    repo = MemRepo(room_dicts)

    rooms = [Room.from_dict(i) for i in room_dicts]

    assert repo.list() == rooms*** 
```

在这种情况下，我们需要一个单独的测试来检查`list`方法的行为。通过测试的实现将放入文件`rentomatic/repository/memrepo.py`中。

`rentomatic/repository/memrepo.py`

```py
***from rentomatic.domain.room import Room

class MemRepo:
    def __init__(self, data):
        self.data = data

    def list(self):
        return [Room.from_dict(i) for i in self.data]*** 
```

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c03-s07`](https://github.com/pycabook/rentomatic/tree/ed2-c03-s07)****

你可以轻松想象这个类是围绕真实数据库或其他任何存储类型的包装器。虽然代码可能会变得更加复杂，但其基本结构将保持不变，只有一个公共方法 `list`。我将在后面的章节中深入探讨数据库仓库。

### 命令行界面

到目前为止，我们已经创建了域模型、序列化器、用例和仓库，但我们仍然缺少一个将一切粘合在一起的系统。这个系统必须从用户那里获取调用参数，使用仓库初始化一个用例，运行从仓库获取域模型的用例，并将它们返回给用户。

现在让我们看看我们刚刚创建的架构如何与外部系统（如 CLI）交互。清晰架构的力量在于外部系统是可插拔的，这意味着我们可以推迟关于我们想要使用的系统细节的决定。在这种情况下，我们想要给用户提供一个查询系统并获取存储系统中包含的房间列表的界面，最简单的选择是命令行工具。

稍后我们将创建一个 REST 端点，并通过 Web 服务器将其公开，那时就会清楚我们创建的架构为何如此强大。

目前，在包含 `setup.cfg` 的同一目录中创建一个名为 `cli.py` 的文件。这是一个简单的 Python 脚本，不需要任何特定选项即可运行，因为它只是查询存储中包含的所有域模型。文件内容如下

`cli.py`

```py
***#!/usr/bin/env python

from rentomatic.repository.memrepo import MemRepo
from rentomatic.use_cases.room_list import room_list_use_case

repo = MemRepo([])
result = room_list_use_case(repo)

print(result)*** 
```

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c03-s08`](https://github.com/pycabook/rentomatic/tree/ed2-c03-s08)****

你可以使用 `python cli.py` 执行此文件，或者如果你更喜欢，运行 `chmod +x cli.py`（使其可执行）然后直接使用 `./cli.py` 运行它。预期的结果是空列表

```py
***$ ./cli.py
[]*** 
```

这是正确的，因为文件 `cli.py` 中的类 `MemRepo` 已经使用空列表初始化。我们使用的简单内存存储没有持久性，所以每次我们创建它时都必须在其中加载一些数据。这样做是为了保持存储层简单，但请记住，如果存储是一个真正的数据库，这部分代码将连接到它，但不需要在其中加载数据。

脚本最重要的部分是

`cli.py`

```py
***repo = MemRepo([])
result = room_list_use_case(repo)*** 
```

这初始化了仓库并运行了用例。这通常是使用你的清晰架构以及你将连接到它的任何外部系统的方式。你初始化其他系统，传递接口运行用例，并收集结果。

为了演示，让我们在文件中定义一些数据并将它们加载到仓库中

`cli.py`

```py
***#!/usr/bin/env python

from rentomatic.repository.memrepo import MemRepo
from rentomatic.use_cases.room_list import room_list_use_case

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

repo = MemRepo(rooms)
result = room_list_use_case(repo)

print([room.to_dict() for room in result])*** 
```

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c03-s09`](https://github.com/pycabook/rentomatic/tree/ed2-c03-s09)****

再次提醒，由于我们存储的简单性质，我们需要硬编码数据，而不是系统的架构。请注意，我将指令`print`改为，因为存储库返回领域模型，打印它们将导致如`<rentomatic.domain.room.Room object at 0x7fb815ec04e0>`这样的字符串列表，这实际上并不太有帮助。

如果你现在运行命令行工具，你将得到比之前更丰富的结果

```py
***$ ./cli.py
[
  {
    'code': 'f853578c-fc0f-4e65-81b8-566c5dffa35a',
    'size': 215,
    'price': 39,
    'longitude': -0.09998975,
    'latitude': 51.75436293
  },
  {
    'code': 'fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a',
    'size': 405,
    'price': 66,
    'longitude': 0.18228006,
    'latitude': 51.74640997
  },
  {
    'code': '913694c6-435a-4366-ba0d-da5334a611b2',
    'size': 56,
    'price': 60,
    'longitude': 0.27891577,
    'latitude': 51.45994069
  },
  {
    'code': 'eed76e77-55c1-41ce-985d-ca49bf6c0585',
    'size': 93,
    'price': 48,
    'longitude': 0.33894476,
    'latitude': 51.39916678
  }
]*** 
```

请注意，我已将上面的输出格式化以使其更易于阅读，但实际输出将位于一行上。

 * 

在本章中我们所看到的是清洁架构的核心所在。

我们探索了实体的标准层（类`Room`），用例（函数`room_list_use_case`），网关和外部系统（类`MemRepo`），并开始欣赏它们分离成层的优势。

可以说，我们所设计的非常有限，这就是为什么我将把本书的其余部分用于展示如何增强我们处理更复杂情况的能力。我们将在第四章讨论**Web 界面**，在第五章讨论**更丰富的查询语言**和**错误管理**，在第 6、7 和 8 章讨论与真实外部系统（如数据库）的**集成**。
