# FastAPI 初学者教程:您需要的资源

> 原文：<https://www.pythoncentral.io/fastapi-tutorial-for-beginners-the-resources-you-need/>

虽然 Flask 是 2004 年愚人节的一个玩笑，而 Django 是同一时期《劳伦斯世界日报》的一个内部项目，但这两个 Python web 框架都获得了巨大的声望和流行。

然而，构建这些框架时的 Python 不同于今天的 Python。该语言的当代元素，如 ASGI 标准和异步执行，要么不流行，要么不存在。

这就是 FastAPI 的用武之地。

它是一个较新的 Python web 框架，建于 2018 年，融入了 Python 的现代特性。除了能够使用 ASGI 标准与客户机进行异步连接之外，它还可以使用 WSGI。

此外，FastAPI web 应用程序是用带有类型提示的干净代码编写的，异步函数可用于路由和端点。

顾名思义，FastAPI 的主要用途之一是构建 API 端点。有了这个框架，构建端点就像以 JSON 的形式返回 Python 字典的数据一样简单。或者，您可以使用 OpenAPI 标准，它包括一个交互式的 Swagger UI。

但是 FastAPI 并不局限于 API，它可以用来完成 web 框架可以完成的任何任务——从交付网页到服务应用程序。

在本帖中，我们将讨论如何安装 FastAPI 以及使用它的基本原则。

## **什么是快速 API？是什么让它从其他框架中脱颖而出？**

FastAPI 是一个高性能的 web 框架，它使用标准的类型提示来帮助构建 Python APIs。其主要特点包括:

*   易于学习: 框架的简单本质使得阅读文档的时间最小化。
*   **直观:** FastAPI 提供优秀的编辑器支持，完善你的代码，减少你需要投入调试的时间。
*   **减少 bug:**由于使用起来相当直截了当，所以开发者犯错的几率较低。
*   健壮: 它有助于生产就绪代码和自动交互文档。
*   **速度:** 号称极致性能，框架堪比 NodeJS 和 Go。此外，它支持快速开发。
*   **基于标准:** 基于 JSON Schema、open API 等开放 API 标准。

除此之外，它最大限度地减少了代码重复。总的来说，该框架优化了开发人员的体验，并通过在默认情况下实施最佳实践，使他们能够有效地编码。

## **如何安装 FastAPI**

建议在新的虚拟环境中启动 FastAPI 项目，因为框架会在没有提示的情况下安装许多组件。要安装框架的核心组件，可以运行下面的命令:

```py
pip install fastapi
```

除此之外，你需要安装一个 ASGI 服务器来执行本地测试。该框架与 Uvicorn 无缝协作。因此，我们将在示例中使用该服务器。

要安装 Uvicorn，您可以运行命令:

```py
pip install uvicorn[standard]
```

上面的命令安装一组最佳的组件，包括 C 库。如果想安装一个只使用 Python 的最小版本，可以运行:

```py
pip install uvicorn
```

## **FastAPI 应用程序的基本示例**

下面是一个基本的 FastAPI 应用程序的样子:

```py
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"greeting":"Hello coders"}
```

要运行这个应用程序，您可以将代码保存在一个文本文件中，并将其命名为 main.py，然后使用以下命令在您的虚拟环境中运行它:

```py
uvicorn main:app
```

虽然上述命令中的“uvicorn”部分是不言自明的，但是“app”是一个可以用于 ASGI 服务器的对象。如果您想使用 WSGI，您可以使用 ASGI-to-WSGI 适配器。但是，最好使用 ASGI。

当应用程序开始运行时，进入 localhost:8000，默认的 Uvicorn 测试服务器。您将看到浏览器上显示的(" greeting":" Hello coders ")，这是字典生成的有效 JSON 响应。

现在您可以看出使用 FastAPI 从一个端点交付 JSON 是多么容易。想法很清楚——您必须创建一个路由并返回一个 Python 字典，它将被序列化为 JSON，而不需要您的干预。

也就是说，序列化复杂的数据类型需要一些工作，我们将在后面的文章中详细讨论。

如果你以前使用过 Flask 这样的系统，你会很容易认出 FastAPI 应用的大致轮廓:

*   ASGI/WSGI 服务器导入应用程序对象并使用它来运行应用程序。
*   你可以使用装饰器给应用程序添加路线。例如,@ app . get(“/”)decorator 在网站的根位置生成一个 GET 方法路由，包装后的函数返回结果。

但是对于熟悉框架工作的人来说，一些不同之处会很明显。首先，路由函数是异步的，这意味着任何部署的异步组件——比如异步数据库中间件连接——都可以在函数中运行。

需要注意的是，如果你愿意，你可以使用常规的同步功能。如果您有一个使用大量计算能力的操作，而不是一个等待 I/O 的操作(这是 async 的最佳用例)，那么最好使用 sync 函数，让 FastAPI 为您做一些工作。

在其他情况下，使用异步函数是正确的方法。

## **FastAPI 中的路线类型**

使用@app decorator，您可以设置路线的方法。您可以使用@app.get 或@app.post 来完成此操作。post、get、DELETE 和 PUT 方法就是这样使用的。不太为人所知的函数 HEAD、TRACE、PATCH 和 OPTIONS 也是这样使用的。

通过在一个函数上使用@app.get("/")和在另一个函数上使用@app.post("/")将路由函数包装在一起，也可以在一个路由上使用多种方法。

## **FastAPI 中的主要参数**

从 route 的路径中提取变量就像在将它们传递给 route 函数之前在 route 声明中定义它们一样简单，就像这样:

```py
@app.get("/users/{user_id}")
async def user(user_id: str):
    return {"user_id":user_id}
```

接下来，您可以使用 route 函数中的类型化声明从 URL 中提取查询参数。框架将自动检测这些声明:

```py
userlist = ["Spike","Jet","Ed","Faye","Ein"]

@app.get("/userlist")
async def userlist_(start: int = 0, limit: int = 10):
    return userlist[start:start+limit]
```

使用表单数据相对来说更复杂。您需要手动安装 Python 多部分库来解析表单数据。当您安装了这个库时，您可以使用一个类似于查询参数语法的语法:

```py
from fastapi import Form

@app.post("/lookup")
async def userlookup(username: str = Form(...), user_id: str = Form("")):
    return {"username": username, "user_id":user_id}
```

上面代码中的表单对象从表单中取出指定的参数并向前传递。您必须记住，在声明中使用格式(…)意味着必须输入相应的参数，例如本例中的用户名。

另一方面，如果有一个可选的表单元素，比如上面例子中的 user_id，您可以将元素的默认值传递给表单。

## **FastAPI 响应类型**

JSON 是 FastAPI 中的默认响应类型，本文中的所有例子都会自动将数据序列化为 JSON。但是，您也可以返回其他响应。比如:

```py
from fastapi.responses import HTMLResponse

@app.get("/")
def root():
    return HTMLResponse("<b>Hello world</b>")
```

fastapi . responses 模块是 FastAPI 最有用的模块之一，它支持大量的响应，包括:

*   **FileResponse:** 从特定路径返回一个文件，异步流。
*   **PlainTextResponse 或 HTMLResponse:** 文本以纯文本或 HTML 的形式返回。
*   **RedirectResponse:** 重定向到特定的 URL。
*   **StreamingResponse:** 接受一个生成器作为输入，并将结果传送给客户机。

除了使用上面列出的响应，您还可以使用通用的响应对象，并提供自定义的标题、媒体类型、状态代码和内容。

## **FastAPI 的响应对象**

无论是设置 cookies 还是头来处理响应，都需要接受一个响应对象作为路由函数的参数:

```py
from fastapi import Response

@app.get("/")
def modify_header(response:Response):
    response.headers["X-New-Header"] = "Hi, I'm a new header!
    return {"msg":"Headers modified in response"}
```

## **FastAPI 中的 Cookies】**

从客户端检索 cookies 的工作方式类似于处理表单参数或查询的工作方式:

```py
from fastapi import Cookie

@app.get("/")
async def main(user_X: Optional[str]=Cookie(none)):
    return {"user_X": user_X}
```

您可以用。响应对象上的 set_cookie()方法:

```py
from fastapi import Response

@app.post("/")
async def main(response: Response):
    response.set_cookie(key="user_X", value="")
    return {"msg":"User X cookie cleared"}
```

## **Pydantic 模型和 FastAPI**

虽然 Python 类型是可选的，但与大多数其他 Python 框架不同，FastAPI 坚持使用类型。这个框架利用 Pydantic 库来验证提交的数据，不需要您编写逻辑来完成验证。

让我们仔细看看 Pydantic 库，其中包含一些验证传入 JSON 的代码:

```py
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

app = FastAPI()

class Movie(BaseModel):
    name: str
    year: int
    rating: Optional[int] = None
    tags: List[str] = []
@app.post("/movies/", response_model=Movie
async def create_movie(movie: Movie):
    return movie
```

上面的代码通过 name、year、rating 和 tags 字段接受 POST 上的 JSON 数据，而不是 HTML。接下来，验证字段的类型。

## **OpenAPI in FastAPI**

OpenAPI 是一种 JSON 格式的标准，用于创建 API 端点。客户端可以读取端点的 OpenAPI 定义，并自动识别站点 API 传输的数据的模式。

如果使用 FastAPI，它会自动为端点生成所需的 OpenAPI 定义。例如，如果您在 openapi 站点的根目录访问/openapi.json，您将收到一个 json 文件，它描述了每个端点以及它可以接收和返回的数据。

FastAPI 的另一个好处是它自动为你的 API 创建文档接口。你可以通过网络界面与他们互动。此外，框架附带了钩子，允许您扩展或修改自动生成的模式。它还允许您有条件地生成模式或完全禁用它。