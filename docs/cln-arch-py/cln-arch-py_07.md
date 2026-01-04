# 第五章 - 错误管理

> 原文：[`www.thedigitalcatbooks.com/pycabook-chapter-05/`](https://www.thedigitalcatbooks.com/pycabook-chapter-05/)
> 
> 你把他们派出去，甚至都没有警告他们！为什么你没有警告他们，伯克？
> 
> 外星人，1986

在每一个软件项目中，大部分代码都致力于错误管理，并且这部分代码必须坚如磐石。错误管理是一个复杂的话题，我们总会遗漏一些角落案例，或者假设某些条件永远不会失败，而实际上却发生了。

在整洁架构中，主要过程是用例的创建和执行。因此，这是错误的主要来源，我们必须在用例层实现错误管理。显然，错误也可能来自领域模型层，但由于这些模型是由用例创建的，因此那些没有由模型本身管理的错误自动成为用例的错误。

## 请求和响应

我们可以将错误管理代码分为两个不同的区域。第一个区域表示和管理**请求**，即达到我们用例的输入数据。第二个区域涵盖了通过**响应**从用例返回结果的方式，即输出数据。这两个概念不应该与 HTTP 请求和响应混淆，尽管它们有相似之处。我们现在正在考虑数据如何传递到和从用例接收，以及如何管理错误。这与使用此架构公开 HTTP API 的可能用途无关。

请求和响应对象是整洁架构的重要组成部分，因为它们将调用参数、输入和结果从应用程序外部传输到用例层。

更具体地说，请求是由传入的 API 调用创建的对象，因此它们必须处理诸如值不正确、缺少参数、格式错误等问题。另一方面，响应必须包含 API 调用的实际结果，但也能够表示错误情况并提供关于发生情况丰富的信息。

请求和响应对象的实际实现完全自由，整洁架构对此没有任何说明。如何打包和表示数据的决定取决于我们。

为了开始处理可能的错误并理解如何管理它们，我将扩展 `room_list_use_case` 以支持可以用来选择存储中 `Room` 对象子集的过滤器。

过滤器可以，例如，通过一个包含模型 `Room` 属性及其逻辑的字典来表示。一旦我们接受这样的丰富结构，我们就会面临各种错误：模型中不存在的属性、错误类型的阈值、导致存储层崩溃的过滤器等等。所有这些考虑都必须由用例来考虑。

## 基本结构

在我们将用例扩展以接受过滤器之前，我们可以实现结构化请求。我们只需要一个可以无参数初始化的类 `RoomListRequest`，因此让我们创建文件 `tests/requests/test_room_list.py` 并在其中放置对该对象的测试。

`tests/requests/test_room_list.py`

```py
from rentomatic.requests.room_list import RoomListRequest

def test_build_room_list_request_without_parameters():
    request = RoomListRequest()

    assert bool(request) is True

def test_build_room_list_request_from_empty_dict():
    request = RoomListRequest.from_dict({})

    assert bool(request) is True 
```

虽然目前这个请求对象基本上是空的，但一旦我们开始为列表用例添加参数，它就会变得非常有用。类 `RoomListRequest` 的代码如下

`rentomatic/requests/room_list.py`

```py
class RoomListRequest:
    @classmethod
    def from_dict(cls, adict):
        return cls()

    def __bool__(self):
        return True 
```

*源代码

[`github.com/pycabook/rentomatic/tree/ed2-c05-s01`](https://github.com/pycabook/rentomatic/tree/ed2-c05-s01)*

*响应对象也非常简单，因为目前我们只需要返回一个成功的结果。与请求不同，响应不与任何特定用例相关联，因此测试文件可以命名为 `tests/test_responses.py`*

*`tests/test_responses.py`*

```py
*from rentomatic.responses import ResponseSuccess

def test_response_success_is_true():
    assert bool(ResponseSuccess()) is True* 
```

*实际的响应对象位于文件 `rentomatic/responses.py` 中*

*`rentomatic/responses.py`*

```py
*class ResponseSuccess:
    def __init__(self, value=None):
        self.value = value

    def __bool__(self):
        return True* 
```

**源代码**

[`github.com/pycabook/rentomatic/tree/ed2-c05-s02`](https://github.com/pycabook/rentomatic/tree/ed2-c05-s02)**

**有了这两个对象，我们为用例输入和输出的更丰富管理奠定了基础，尤其是在错误条件的情况下。**

## **用例中的请求和响应**

**让我们将我们开发的请求和响应对象实现到用例中。为此，我们需要更改用例，使其接受请求并返回响应。`tests/use_cases/test_room_list.py` 的新版本如下**

**`tests/use_cases/test_room_list.py`**

```py
**import pytest
import uuid
from unittest import mock

from rentomatic.domain.room import Room
from rentomatic.use_cases.room_list import room_list_use_case
from rentomatic.requests.room_list import RoomListRequest

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

    request = RoomListRequest()

    response = room_list_use_case(repo, request)

    assert bool(response) is True
    repo.list.assert_called_with()
    assert response.value == domain_rooms** 
```

**用例中的更改很小。`rentomatic/use_cases/room_list.py` 文件的新版本如下**

**`rentomatic/use_cases/room_list.py`**

```py
**from rentomatic.responses import ResponseSuccess

def room_list_use_case(repo, request):
    rooms = repo.list()
    return ResponseSuccess(rooms)** 
```

***源代码

[`github.com/pycabook/rentomatic/tree/ed2-c05-s03`](https://github.com/pycabook/rentomatic/tree/ed2-c05-s03)***

***现在我们有一个标准的方式来打包输入和输出值，上述模式适用于我们创建的每个用例。然而，我们仍然缺少一些功能，因为到目前为止，请求和响应尚未用于执行错误管理。***

## ***请求验证***

***我们要添加到用例中的参数 `filters` 允许调用者使用类似 `<attribute>__<operator>` 的符号添加条件，以缩小模型列表操作的结果。例如，指定 `filters={'price__lt': 100}` 应返回所有价格低于 100 的结果。***

***由于模型 `Room` 有许多属性，可能的过滤器数量非常高。为了简化起见，我将考虑以下情况：***

+   ***属性 `code` 只支持 `__eq`，如果存在，则找到具有特定代码的房间***

+   ***属性 `price` 支持 `__eq`、`__lt` 和 `__gt`***

+   ***所有其他属性都不能用于过滤器***

***这里的核心思想是，请求是根据用例定制的，因此它们可以包含验证用于实例化的参数的逻辑。请求在到达用例之前是有效或无效的，因此后者不需要检查输入值是否有适当的值或格式。***

***这也意味着构建请求可能会导致两个不同的对象，一个是有效的，一个是无效的。因此，我决定将现有的类`RoomListRequest`拆分为`RoomListValidRequest`和`RoomListInvalidRequest`，创建一个返回适当对象的工厂函数。***

***首先，我将修改现有的测试以使用工厂模式。***

***`tests/requests/test_room_list.py`***

```py
***from rentomatic.requests.room_list import build_room_list_request

def test_build_room_list_request_without_parameters():
    request = build_room_list_request()

    assert request.filters is None
    assert bool(request) is True

def test_build_room_list_request_with_empty_filters():
    request = build_room_list_request({})

    assert request.filters == {}
    assert bool(request) is True*** 
```

***接下来，我将测试传递错误的对象类型作为`filters`或使用不正确的键会导致无效请求***

***`tests/requests/test_room_list.py`***

```py
***def test_build_room_list_request_with_invalid_filters_parameter():
    request = build_room_list_request(filters=5)

    assert request.has_errors()
    assert request.errors[0]["parameter"] == "filters"
    assert bool(request) is False

def test_build_room_list_request_with_incorrect_filter_keys():
    request = build_room_list_request(filters={"a": 1})

    assert request.has_errors()
    assert request.errors[0]["parameter"] == "filters"
    assert bool(request) is False*** 
```

***最后，我将测试支持的和不支持的键***

***`tests/requests/test_room_list.py`***

```py
***import pytest

...

@pytest.mark.parametrize(
    "key", ["code__eq", "price__eq", "price__lt", "price__gt"]
)
def test_build_room_list_request_accepted_filters(key):
    filters = {key: 1}

    request = build_room_list_request(filters=filters)

    assert request.filters == filters
    assert bool(request) is True

@pytest.mark.parametrize("key", ["code__lt", "code__gt"])
def test_build_room_list_request_rejected_filters(key):
    filters = {key: 1}

    request = build_room_list_request(filters=filters)

    assert request.has_errors()
    assert request.errors[0]["parameter"] == "filters"
    assert bool(request) is False*** 
```

***请注意，我使用了装饰器`pytest.mark.parametrize`来在多个值上运行相同的测试。***

***遵循 TDD（测试驱动开发）方法，逐个添加这些测试并编写通过它们的代码，我得到了以下代码***

***`rentomatic/requests/room_list.py`***

```py
***from collections.abc import Mapping

class RoomListInvalidRequest:
    def __init__(self):
        self.errors = []

    def add_error(self, parameter, message):
        self.errors.append({"parameter": parameter, "message": message})

    def has_errors(self):
        return len(self.errors) > 0

    def __bool__(self):
        return False

class RoomListValidRequest:
    def __init__(self, filters=None):
        self.filters = filters

    def __bool__(self):
        return True

def build_room_list_request(filters=None):
    accepted_filters = ["code__eq", "price__eq", "price__lt", "price__gt"]
    invalid_req = RoomListInvalidRequest()

    if filters is not None:
        if not isinstance(filters, Mapping):
            invalid_req.add_error("filters", "Is not iterable")
            return invalid_req

        for key, value in filters.items():
            if key not in accepted_filters:
                invalid_req.add_error(
                    "filters", "Key {} cannot be used".format(key)
                )

        if invalid_req.has_errors():
            return invalid_req

    return RoomListValidRequest(filters=filters)*** 
```

***引入工厂导致一个用例测试失败。该测试的新版本是***

***`tests/use_cases/test_room_list.py`***

```py
***...

from rentomatic.requests.room_list import build_room_list_request

...

def test_room_list_without_parameters(domain_rooms):
    repo = mock.Mock()
    repo.list.return_value = domain_rooms

    request = build_room_list_request()

    response = room_list_use_case(repo, request)

    assert bool(response) is True
    repo.list.assert_called_with()
    assert response.value == domain_rooms*** 
```

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c05-s04`](https://github.com/pycabook/rentomatic/tree/ed2-c05-s04)****

## ***响应和失败***

***在用例代码执行过程中可能会发生各种错误。验证错误，正如我们在上一节中讨论的，但还有业务逻辑错误或来自仓库层或其他与用例交互的外部系统的错误。无论错误是什么，用例都应该始终返回一个具有已知结构（响应）的对象，因此我们需要一个新的对象，它为不同类型的失败提供良好的支持。***

***对于请求，没有唯一的方法提供这样的对象，以下代码只是可能的解决方案之一。首先，在必要的导入之后，我测试了响应具有布尔值***

***`tests/test_responses.py`***

```py
***from rentomatic.responses import (
    ResponseSuccess,
    ResponseFailure,
    ResponseTypes,
    build_response_from_invalid_request,
)
from rentomatic.requests.room_list import RoomListInvalidRequest

SUCCESS_VALUE = {"key": ["value1", "value2"]}
GENERIC_RESPONSE_TYPE = "Response"
GENERIC_RESPONSE_MESSAGE = "This is a response"

def test_response_success_is_true():
    response = ResponseSuccess(SUCCESS_VALUE)

    assert bool(response) is True

def test_response_failure_is_false():
    response = ResponseFailure(
        GENERIC_RESPONSE_TYPE, GENERIC_RESPONSE_MESSAGE
    )

    assert bool(response) is False*** 
```

***然后我测试了响应的结构，检查`type`和`value`。`ResponseFailure`对象也应该有一个属性`message`***

***`tests/test_responses.py`***

```py
***def test_response_success_has_type_and_value():
    response = ResponseSuccess(SUCCESS_VALUE)

    assert response.type == ResponseTypes.SUCCESS
    assert response.value == SUCCESS_VALUE

def test_response_failure_has_type_and_message():
    response = ResponseFailure(
        GENERIC_RESPONSE_TYPE, GENERIC_RESPONSE_MESSAGE
    )

    assert response.type == GENERIC_RESPONSE_TYPE
    assert response.message == GENERIC_RESPONSE_MESSAGE
    assert response.value == {
        "type": GENERIC_RESPONSE_TYPE,
        "message": GENERIC_RESPONSE_MESSAGE,
    }*** 
```

***剩余的测试都是关于`ResponseFailure`的。首先，一个测试来检查它是否可以用异常初始化***

***`tests/test_responses.py`***

```py
***def test_response_failure_initialisation_with_exception():
    response = ResponseFailure(
        GENERIC_RESPONSE_TYPE, Exception("Just an error message")
    )

    assert bool(response) is False
    assert response.type == GENERIC_RESPONSE_TYPE
    assert response.message == "Exception: Just an error message"*** 
```

***由于我们希望能够直接从无效请求构建响应，获取其中包含的所有错误，我们需要测试这种情况***

***`tests/test_responses.py`***

```py
***def test_response_failure_from_empty_invalid_request():
    response = build_response_from_invalid_request(
        RoomListInvalidRequest()
    )

    assert bool(response) is False
    assert response.type == ResponseTypes.PARAMETERS_ERROR

def test_response_failure_from_invalid_request_with_errors():
    request = RoomListInvalidRequest()
    request.add_error("path", "Is mandatory")
    request.add_error("path", "can't be blank")

    response = build_response_from_invalid_request(request)

    assert bool(response) is False
    assert response.type == ResponseTypes.PARAMETERS_ERROR
    assert response.message == "path: Is mandatory\npath: can't be blank"*** 
```

***让我们编写使测试通过的类***

***`rentomatic/responses.py`***

```py
***class ResponseTypes:
    PARAMETERS_ERROR = "ParametersError"
    RESOURCE_ERROR = "ResourceError"
    SYSTEM_ERROR = "SystemError"
    SUCCESS = "Success"

class ResponseFailure:
    def __init__(self, type_, message):
        self.type = type_
        self.message = self._format_message(message)

    def _format_message(self, msg):
        if isinstance(msg, Exception):
            return "{}: {}".format(
                msg.__class__.__name__, "{}".format(msg)
            )
        return msg

    @property
    def value(self):
        return {"type": self.type, "message": self.message}

    def __bool__(self):
        return False

class ResponseSuccess:
    def __init__(self, value=None):
        self.type = ResponseTypes.SUCCESS
        self.value = value

    def __bool__(self):
        return True

def build_response_from_invalid_request(invalid_request):
    message = "\n".join(
        [
            "{}: {}".format(err["parameter"], err["message"])
            for err in invalid_request.errors
        ]
    )
    return ResponseFailure(ResponseTypes.PARAMETERS_ERROR, message)*** 
```

***通过 `_format_message()` 方法，我们使类能够接受字符串消息和 Python 异常，这在处理可能引发我们不知道或不想管理的异常的外部库时非常有用。***

***类 `ResponseTypes` 中包含的错误类型与 HTTP 错误非常相似，这将在我们稍后从 Web 框架返回响应时很有用。`PARAMETERS_ERROR` 表示请求传入的输入参数有问题。`RESOURCE_ERROR` 表示过程正确结束，但请求的资源不可用，例如从数据存储中读取特定值时。最后，`SYSTEM_ERROR` 表示过程本身出现问题，并将主要用于在 Python 代码中引发异常。***

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c05-s05`](https://github.com/pycabook/rentomatic/tree/ed2-c05-s05)****

## ***用例中的错误管理***

***我们请求和响应的实现最终完成，因此我们现在可以实施用例的最后一个版本。函数 `room_list_use_case` 仍然缺少对传入请求的适当验证，并且在出错时没有返回合适的响应。***

***测试 `test_room_list_without_parameters` 必须匹配新的 API，所以我向 `assert_called_with` 添加了 `filters=None`***

***`tests/use_cases/test_room_list.py`***

```py
***def test_room_list_without_parameters(domain_rooms):
    repo = mock.Mock()
    repo.list.return_value = domain_rooms

    request = build_room_list_request()

    response = room_list_use_case(repo, request)

    assert bool(response) is True
    repo.list.assert_called_with(filters=None)
    assert response.value == domain_rooms*** 
```

***我们可以添加三个新测试来检查当 `filters` 不是 `None` 时用例的行为。第一个测试检查用于创建请求的字典中 `filters` 键的值在调用存储库时是否实际使用。后两个测试检查当存储库引发异常或请求格式不正确时用例的行为。***

***`tests/use_cases/test_room_list.py`***

```py
***import pytest
import uuid
from unittest import mock

from rentomatic.domain.room import Room
from rentomatic.use_cases.room_list import room_list_use_case
from rentomatic.requests.room_list import build_room_list_request
from rentomatic.responses import ResponseTypes

...

def test_room_list_with_filters(domain_rooms):
    repo = mock.Mock()
    repo.list.return_value = domain_rooms

    qry_filters = {"code__eq": 5}
    request = build_room_list_request(filters=qry_filters)

    response = room_list_use_case(repo, request)

    assert bool(response) is True
    repo.list.assert_called_with(filters=qry_filters)
    assert response.value == domain_rooms

def test_room_list_handles_generic_error():
    repo = mock.Mock()
    repo.list.side_effect = Exception("Just an error message")

    request = build_room_list_request(filters={})

    response = room_list_use_case(repo, request)

    assert bool(response) is False
    assert response.value == {
        "type": ResponseTypes.SYSTEM_ERROR,
        "message": "Exception: Just an error message",
    }

def test_room_list_handles_bad_request():
    repo = mock.Mock()

    request = build_room_list_request(filters=5)

    response = room_list_use_case(repo, request)

    assert bool(response) is False
    assert response.value == {
        "type": ResponseTypes.PARAMETERS_ERROR,
        "message": "filters: Is not iterable",
    }*** 
```

***现在将用例更改为包含新的用例实现，使所有测试通过***

***`rentomatic/use_cases/room_list.py`***

```py
***from rentomatic.responses import (
    ResponseSuccess,
    ResponseFailure,
    ResponseTypes,
    build_response_from_invalid_request,
)

def room_list_use_case(repo, request):
    if not request:
        return build_response_from_invalid_request(request)
    try:
        rooms = repo.list(filters=request.filters)
        return ResponseSuccess(rooms)
    except Exception as exc:
        return ResponseFailure(ResponseTypes.SYSTEM_ERROR, exc)*** 
```

***如你所见，用例首先检查请求是否有效。如果不是，它将返回一个使用相同请求对象构建的 `ResponseFailure`。然后实现实际的业务逻辑，调用存储库并返回一个成功的响应。如果在这一阶段出现问题，异常将被捕获并以适当的格式返回 `ResponseFailure`。***

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c05-s06`](https://github.com/pycabook/rentomatic/tree/ed2-c05-s06)****

## ***集成外部系统***

***我想指出由模拟表示的一个大问题。***

***由于我们正在使用模拟测试外部系统，如存储库，目前没有测试失败，但尝试运行 Flask 开发服务器肯定会返回一个错误。实际上，存储库和 HTTP 服务器都没有与新 API 保持同步，但如果它们被正确编写，单元测试无法显示这一点。这就是为什么我们需要集成测试的原因，因为依赖于特定版本 API 的外部系统只在该点运行，这可能会引发由模拟掩盖的问题。***

***对于这个简单的项目，我的集成测试由 Flask 开发服务器表示，此时会崩溃。如果你运行 `FLASK_CONFIG="development" flask run` 并用你的浏览器打开 [`127.0.0.1:5000/rooms`](http://127.0.0.1:5000/rooms)，你将得到一个内部服务器错误，并在命令行中抛出这个异常***

```py
***TypeError: room_list_use_case() missing 1 required positional argument: 'request'*** 
```

***相同的错误由 CLI 接口返回。在引入请求和响应之后，我们没有更改 REST 端点，这是外部世界和用例之间的一个连接。鉴于用例的 API 已更改，我们需要更改调用用例的端点的代码。***

### ***HTTP 服务器***

***如上异常所示，REST 端点中调用用例时使用了错误的参数。新的测试版本是***

***`tests/rest/test_room.py`***

```py
***import json
from unittest import mock

import pytest

from rentomatic.domain.room import Room
from rentomatic.responses import (
    ResponseFailure,
    ResponseSuccess,
    ResponseTypes,
)

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
    mock_use_case.return_value = ResponseSuccess(rooms)

    http_response = client.get("/rooms")

    assert json.loads(http_response.data.decode("UTF-8")) == [room_dict]

    mock_use_case.assert_called()
    args, kwargs = mock_use_case.call_args
    assert args[1].filters == {}

    assert http_response.status_code == 200
    assert http_response.mimetype == "application/json"

@mock.patch("application.rest.room.room_list_use_case")
def test_get_with_filters(mock_use_case, client):
    mock_use_case.return_value = ResponseSuccess(rooms)

    http_response = client.get(
        "/rooms?filter_price__gt=2&filter_price__lt=6"
    )

    assert json.loads(http_response.data.decode("UTF-8")) == [room_dict]

    mock_use_case.assert_called()
    args, kwargs = mock_use_case.call_args
    assert args[1].filters == {"price__gt": "2", "price__lt": "6"}

    assert http_response.status_code == 200
    assert http_response.mimetype == "application/json"

@pytest.mark.parametrize(
    "response_type, expected_status_code",
    [
        (ResponseTypes.PARAMETERS_ERROR, 400),
        (ResponseTypes.RESOURCE_ERROR, 404),
        (ResponseTypes.SYSTEM_ERROR, 500),
    ],
)
@mock.patch("application.rest.room.room_list_use_case")
def test_get_response_failures(
    mock_use_case,
    client,
    response_type,
    expected_status_code,
):
    mock_use_case.return_value = ResponseFailure(
        response_type,
        message="Just an error message",
    )

    http_response = client.get("/rooms?dummy_request_string")

    mock_use_case.assert_called()

    assert http_response.status_code == expected_status_code*** 
```

***`test_get` 函数已经存在，但已更改以反映对请求和响应的使用。第一个更改是模拟中的用例必须返回一个适当的响应***

```py
***mock_use_case.return_value = ResponseSuccess(rooms)*** 
```

***第二个是关于用例调用的断言。它应该使用格式正确的请求来调用，但由于我们无法比较请求，我们需要一种方法来查看调用参数。这可以通过***

```py
***mock_use_case.assert_called()
args, kwargs = mock_use_case.call_args
assert args[1].filters == {}*** 
```

***因为用例应该接收一个带有空过滤器的请求作为参数。***

***`test_get_with_filters` 函数执行相同的操作，但将查询字符串传递给 `/rooms` URL，这需要不同的断言***

```py
***assert args[1].filters == {'price__gt': '2', 'price__lt': '6'}*** 
```

***测试通过了新版本的端点 `room_list`***

***`application/rest/room.py`***

```py
***import json

from flask import Blueprint, request, Response

from rentomatic.repository.memrepo import MemRepo
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
    qrystr_params = {
        "filters": {},
    }

    for arg, values in request.args.items():
        if arg.startswith("filter_"):
            qrystr_params["filters"][arg.replace("filter_", "")] = values

    request_object = build_room_list_request(
        filters=qrystr_params["filters"]
    )

    repo = MemRepo(rooms)
    response = room_list_use_case(repo, request_object)

    return Response(
        json.dumps(response.value, cls=RoomJsonEncoder),
        mimetype="application/json",
        status=STATUS_CODES[response.type],
    )*** 
```

***请注意，我在这里使用了一个名为 `request_object` 的变量来避免与 `pytest-flask` 提供的固定值 `request` 冲突。虽然 `request` 包含浏览器发送给网络框架的 HTTP 请求，但 `request_object` 是我们发送给用例的请求。***

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c05-s07`](https://github.com/pycabook/rentomatic/tree/ed2-c05-s07)****

### ***The repository***

***如果我们现在运行 Flask 开发网络服务器并尝试访问 `/rooms` 端点，我们将得到一个友好的响应，说明***

```py
***{"type": "SystemError", "message": "TypeError: list() got an unexpected keyword argument 'filters'"}*** 
```

***如果你查看 HTTP 响应^([1]), 你可以看到一个 HTTP 500 错误，这正好是我们`SystemError`用例错误的映射，它反过来又表示一个 Python 异常，该异常位于错误的`message`部分。***

***这个错误来自仓库，它尚未迁移到新 API。因此，我们需要将`MemRepo`类的`list`方法修改为接受`filters`参数并相应地执行。请注意这一点。过滤器可能已经被视为业务逻辑的一部分并在用例中实现，但我们决定利用存储系统可以做到的事情，因此我们将过滤移动到外部系统中。这是一个合理的决定，因为数据库通常可以很好地执行过滤和排序。尽管我们目前使用的内存存储不是数据库，但我们正在准备使用真正的外部存储。***

***仓库测试的新版本是***

***`tests/repository/test_memrepo.py`***

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

    assert repo.list() == rooms

def test_repository_list_with_code_equal_filter(room_dicts):
    repo = MemRepo(room_dicts)

    rooms = repo.list(
        filters={"code__eq": "fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a"}
    )

    assert len(rooms) == 1
    assert rooms[0].code == "fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a"

@pytest.mark.parametrize("price", [60, "60"])
def test_repository_list_with_price_equal_filter(room_dicts, price):
    repo = MemRepo(room_dicts)

    rooms = repo.list(filters={"price__eq": price})

    assert len(rooms) == 1
    assert rooms[0].code == "913694c6-435a-4366-ba0d-da5334a611b2"

@pytest.mark.parametrize("price", [60, "60"])
def test_repository_list_with_price_less_than_filter(room_dicts, price):
    repo = MemRepo(room_dicts)

    rooms = repo.list(filters={"price__lt": price})

    assert len(rooms) == 2
    assert set([r.code for r in rooms]) == {
        "f853578c-fc0f-4e65-81b8-566c5dffa35a",
        "eed76e77-55c1-41ce-985d-ca49bf6c0585",
    }

@pytest.mark.parametrize("price", [48, "48"])
def test_repository_list_with_price_greater_than_filter(room_dicts, price):
    repo = MemRepo(room_dicts)

    rooms = repo.list(filters={"price__gt": price})

    assert len(rooms) == 2
    assert set([r.code for r in rooms]) == {
        "913694c6-435a-4366-ba0d-da5334a611b2",
        "fe2c3195-aeff-487a-a08f-e0bdc0ec6e9a",
    }

def test_repository_list_with_price_between_filter(room_dicts):
    repo = MemRepo(room_dicts)

    rooms = repo.list(filters={"price__lt": 66, "price__gt": 48})

    assert len(rooms) == 1
    assert rooms[0].code == "913694c6-435a-4366-ba0d-da5334a611b2"*** 
```

***正如你所见，我添加了许多测试。每个被接受的过滤器（`code__eq`、`price__eq`、`price__lt`、`price__gt`，见`rentomatic/requests/room_list.py`）都有一个测试，还有一个尝试同时使用两个不同过滤器的最终测试。***

***再次提醒，这将是存储提供的 API，而不是用例提供的 API。两者匹配是一个设计决策，但实际效果可能会有所不同。***

***仓库的新版本是***

***`rentomatic/repository/memrepo.py`***

```py
***from rentomatic.domain.room import Room

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

        return result*** 
```

***此时，你可以使用`FLASK_CONFIG="development" flask run`启动 Flask 开发 web 服务器，并在[`localhost:5000/rooms`](http://localhost:5000/rooms)获取你所有房间的列表。你还可以在 URL 中使用过滤器，例如[`localhost:5000/rooms?filter_code__eq=f853578c-fc0f-4e65-81b8-566c5dffa35a`](http://localhost:5000/rooms?filter_code__eq=f853578c-fc0f-4e65-81b8-566c5dffa35a)，它返回给定代码的房间，或者[`localhost:5000/rooms?filter_price__lt=50`](http://localhost:5000/rooms?filter_price__lt=50)，它返回所有价格低于 50 的房间。***

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c05-s08`](https://github.com/pycabook/rentomatic/tree/ed2-c05-s08)****

### ***命令行界面***

***在这个阶段，修复 CLI 非常简单，因为我们只需要模仿我们对 HTTP 服务器所做的那样，只是不需要考虑过滤器，因为它们不是命令行工具的一部分。***

***`cli.py`***

```py
***#!/usr/bin/env python

from rentomatic.repository.memrepo import MemRepo
from rentomatic.use_cases.room_list import room_list_use_case
from rentomatic.requests.room_list import build_room_list_request

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

request = build_room_list_request()
repo = MemRepo(rooms)
response = room_list_use_case(repo, request)

print([room.to_dict() for room in response.value])*** 
```

****源代码

[`github.com/pycabook/rentomatic/tree/ed2-c05-s09`](https://github.com/pycabook/rentomatic/tree/ed2-c05-s09)****

* * *

***我们现在有一个非常健壮的系统来管理输入验证和错误条件，并且它足够通用，可以用于任何可能的用例。显然，我们可以自由地添加新的错误类型，以增加我们管理失败时的粒度，但当前版本已经涵盖了用例内部可能发生的所有情况。***

***在下一章中，我们将探讨基于真实数据库引擎的仓库，展示如何使用 PostgreSQL 作为数据库进行外部系统的集成测试。在后续章节中，我将展示干净的架构如何使我们能够非常容易地在不同的外部系统之间切换，将系统迁移到 MongoDB。***

***1

例如，使用浏览器开发者工具。在 Chrome 和 Firefox 中，按 F12 并打开网络标签页，然后刷新页面。***
