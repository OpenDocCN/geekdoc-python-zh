# 用 Pyramid 和 Ramses 在几分钟内创建一个 REST API

> 原文：<https://realpython.com/create-a-rest-api-in-minutes-with-pyramid-and-ramses/>

*这是来自[的](https://brandicted.com/)[基斯·哈特](https://twitter.com/chrstphrhrt)布兰迪德的客座博文——一位来自伟大城市蒙特娄的技术专家。*

本教程是为初学者准备的。如果你在前进的道路上遇到困难，试着克服它，它可能会成功。如果你有什么不明白的或者需要帮助的，给 info@brandicted.com 发电子邮件或者在下面留下评论。

## 简介

制作一个 API 可能需要大量的工作。开发人员需要处理诸如序列化、URL 映射、验证、认证、授权、版本控制、测试、数据库、模型和视图的定制代码等细节。像 Firebase 和 Parse 这样的服务的存在使得这种方式变得更容易。使用后端即服务，开发人员可以更加专注于构建独特的用户体验。

使用第三方后端提供商的一些缺点包括缺乏对后端代码的控制、不能自托管、没有知识产权等..控制代码*和*利用 BaaS 节省时间的便利性是理想的，但是大多数 REST API 框架仍然需要大量的样板文件。一个流行的例子是令人惊叹的笨重的 Django Rest 框架。另一个伟大的项目是[Flask-restful](https://flask-restless.readthedocs.org/en/latest/)(强烈推荐)，它需要更少的样板文件，并使构建 API 变得超级简单。但是我们想去掉所有的样板文件，包括通常需要为视图编写的数据库查询。

进入 Ramses，这是一种从 YAML 文件生成强大后端的简单方法(实际上是 REST APIs 的一种方言，称为 [RAML](http://raml.org/) )。**在这篇文章中，我们将向你展示如何在几分钟内从零到你自己的生产就绪后端。**

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

> 想要密码吗？[Github 上的拉姆西斯](https://github.com/brandicted/ramses)

[*Remove ads*](/account/join/)

## 引导一个新产品 API

### 先决条件

我们假设你在一个新的虚拟 Python 环境中工作，并且使用默认配置运行 T2 的弹性搜索和 T4 的 postgresql。我们使用 httpie 与 API 交互，但是你也可以使用 curl 或者其他 http 客户端。

如果任何时候你遇到困难或者想看本教程代码的最终工作版本，[可以在这里找到](https://github.com/chrstphrhrt/ramses-tutorial/tree/master/pizza_factory)。

### 场景:一家制作美味披萨的工厂

[!["Big Fat Pizza" Shop](img/5f9217bcdad0c6ad0cc30db4f84d73db.png)](https://files.realpython.com/media/FatPizzaShopHumeHwyChullora.47ebcbfa5b8b.JPG)

我们想为我们的新比萨店创建一个 API。我们的后端应该知道所有不同的配料、奶酪、酱料和可以使用的面包皮，以及它们的不同组合来制作各种披萨风格。

```py
$ pip install ramses
$ pcreate -s ramses_starter pizza_factory
```

安装程序将询问您想要使用哪个数据库后端。选择选项“1”以使用 SQLAlchemy。

换到新创建的目录，环顾四周。

```py
$ cd pizza_factory
```

所有端点都可以通过 URI/API/端点名称/项目 id 进行访问。默认情况下，内置服务器运行在端口 6543 上。通读一下 **local.ini** ，看看它是否有意义。然后运行服务器，开始与新的后端交互。

```py
$ pserve local.ini
```

查看 **api.raml** 以了解如何指定端点。

```py
#%RAML 0.8 --- title:  pizza_factory documentation: -  title:  pizza_factory REST API content:  | Welcome to the pizza_factory API. baseUri:  http://localhost:6543/api mediaType:  application/json protocols:  [HTTP] /items: displayName:  Collection of items get: description:  Get all item post: description:  Create a new item body: application/json: schema:  !include  items.json /{id}: displayName:  Collection-item get: description:  Get a particular item delete: description:  Delete a particular item patch: description:  Update a particular item
```

如您所见，我们在/api/items 中有一个资源，它是由 **items.json** 中的模式定义的。

```py
$ http :6543/api/items
HTTP/1.1 200 OK
Cache-Control: max-age=0, must-revalidate, no-cache, no-store
Content-Length: 73
Content-Type: application/json; charset=UTF-8
Date: Tue, 02 Jun 2015 16:02:09 GMT
Expires: Tue, 02 Jun 2015 16:02:09 GMT
Last-Modified: Tue, 02 Jun 2015 16:02:09 GMT
Pragma: no-cache
Server: waitress

{
 "count": 0,
 "data": [],
 "fields": "",
 "start": 0,
 "took": 1,
 "total": 0
}
```

## 数据建模

### 模式！

模式描述了数据的结构。

我们需要为制作披萨的每一种不同的配料制作它们。Ramses 的默认模式是 **items.json** 中的一个基本示例。

因为我们的项目中会有多个模式，所以让我们创建一个新目录，并将默认模式移入其中，以保持整洁。

```py
$ mkdir schemas
$ mv items.json schemas/
$ cd schemas/
```

**将 items.json 重命名为 pizzas.json** 并在文本编辑器中打开它。然后将其内容复制到同一目录下的新文件中，文件名分别为 **toppings.json** 、 **cheeses.json** 、 **sauces.json** 和 **crusts.json** 。

```py
├── cheeses.json
├── crusts.json
├── pizzas.json
├── sauces.json
└── toppings.json
```

在每个新的模式中，为被描述的不同种类的事物更新`"title"`字段的值(例如`"title": "Pizza schema"`、`"title": "Topping schema"`等)。).

让我们编辑 **pizzas.json** 模式，将配料连接到给定风格的比萨饼中。

在`"description"`字段后，添加以下与配料的关系:

```py
... "toppings":  { "required":  false, "type":  "relationship", "args":  { "document":  "Topping", "ondelete":  "NULLIFY", "backref_name":  "pizza", "backref_ondelete":  "NULLIFY" } }, "cheeses":  { "required":  false, "type":  "relationship", "args":  { "document":  "Cheese", "ondelete":  "NULLIFY", "backref_name":  "pizza", "backref_ondelete":  "NULLIFY" } }, "sauce_id":  { "required":  false, "type":  "foreign_key", "args":  { "ref_document":  "Sauce", "ref_column":  "sauce.id", "ref_column_type":  "id_field" } }, "crust_id":  { "required":  true, "type":  "foreign_key", "args":  { "ref_document":  "Crust", "ref_column":  "crust.id", "ref_column_type":  "id_field" } } ...
```

[*Remove ads*](/account/join/)

### 关系 101

我们需要对每一种配料做同样的工作，将它们与需要它们的比萨饼风格的食谱联系起来。在 **toppings.json** 和 **cheeses.json** 中，我们需要一个`"foreign_key"`字段，指向每种浇头将用于的特定披萨风格(同样，将它放在`"description"`字段之后):

```py
... "pizza_id":  { "required":  false, "type":  "foreign_key", "args":  { "ref_document":  "Pizza", "ref_column":  "pizza.id", "ref_column_type":  "id_field" } } ...
```

然后，在 **sauces.json** 和 **crusts.json** 中，我们进行了*反向*(通过指定`"relationship"`字段而不是`"foreign_key"`字段)，因为这两种配料被调用它们的比萨饼风格的特定实例所引用:

```py
... "pizzas":  { "required":  false, "type":  "relationship", "args":  { "document":  "Pizza", "ondelete":  "NULLIFY", "backref_name":  "sauce", "backref_ondelete":  "NULLIFY" } } ...
```

对于 **crusts.json** ，只要确保将`"backref_name"`的值设置为`"crust"`即可。

这里要注意的一件事是，如果你仔细考虑了很久，你会发现做一个比萨饼只需要一层皮。也许在这一点上我们不得不称之为面包，但我们不要太哲学化。

还要注意的是*我们有两个不同的“方向”*披萨和配料的关系。比萨饼有许多配料和奶酪。这些都是“一(披萨)对多(食材)”的关系。尽管比萨饼只有一种调味汁和一层皮。每种酱料或皮可能被许多不同的比萨饼风格所需要。当谈到比萨饼时，我们说这是一种“多(比萨饼)对一(酱/皮)”的关系。无论你想称之为哪个“方向”,都只是你所谈论的作为参考点的实体的问题。

一对多关系在“一”方有一个`relationship`字段，在“多”方有一个`foreign_key`字段，例如披萨(如 **pizzas.json** 中所述)有多个`"toppings"`:

```py
... "toppings":  { "required":  false, "type":  "relationship", "args":  { "document":  "Topping", "ondelete":  "NULLIFY", "backref_name":  "pizza", "backref_ondelete":  "NULLIFY" } ...
```

…每种浇头(如 **toppings.json** 中所述)都是由某些特定的披萨(`"pizza_id"`)要求的:

```py
... "pizza_id":  { "required":  false, "type":  "foreign_key", "args":  { "ref_document":  "Pizza", "ref_column":  "pizza.id", "ref_column_type":  "id_field" } } ...
```

多对一关系在“多”端有一个`foreign_key`字段，在一端有一个`relationship`字段。这就是为什么浇头有一个指向特定披萨的`foreign_key`字段，而披萨有一个指向所有浇头的`relationship`字段。

### Backref & ondelete 参数

**要详细了解关系数据库概念的使用，请参考 [SQLAlchemy 文档](http://docs.sqlalchemy.org/en/latest/orm/basic_relationships.html)。** *非常*简要:

一个`backref`参数告诉数据库，当一个模型被另一个模型引用时,“引用”模型(它有一个`foreign_key`字段)也将提供对“被引用”模型的“向后”访问。

一个`ondelete`参数告诉数据库，当被引用模型的实例被删除时，要相应地改变引用字段的值。`NULLIFY`表示该值将被设置为`null`。

## 创建端点

至此，我们的厨房差不多准备好了。为了真正开始制作比萨饼，我们需要连接一些 API 端点来访问我们刚刚创建的数据模型。

让我们编辑 **api.raml** ，替换每个资源的默认“items”端点，如下所示:

```py
#%RAML 0.8 --- title:  pizza_factory API documentation: -  title:  pizza_factory REST API content:  | Welcome to the pizza_factory API. baseUri:  http://{host}:{port}/{version} version:  v1 mediaType:  application/json protocols:  [HTTP] /toppings: displayName:  Collection of ingredients for toppings get: description:  Get all topping ingredients post: description:  Create a topping ingredient body: application/json: schema:  !include  schemas/toppings.json /{id}: displayName:  A particular topping ingredient get: description:  Get a particular topping ingredient delete: description:  Delete a particular topping ingredient patch: description:  Update a particular topping ingredient /cheeses: displayName:  Collection of different cheeses get: description:  Get all cheeses post: description:  Create a new cheese body: application/json: schema:  !include  schemas/cheeses.json /{id}: displayName:  A particular cheese ingredient get: description:  Get a particular cheese delete: description:  Delete a particular cheese patch: description:  Update a particular cheese /pizzas: displayName:  Collection of pizza styles get: description:  Get all pizza styles post: description:  Create a new pizza style body: application/json: schema:  !include  schemas/pizzas.json /{id}: displayName:  A particular pizza style get: description:  Get a particular pizza style delete: description:  Delete a particular pizza style patch: description:  Update a particular pizza style /sauces: displayName:  Collection of different sauces get: description:  Get all sauces post: description:  Create a new sauce body: application/json: schema:  !include  schemas/sauces.json /{id}: displayName:  A particular sauce get: description:  Get a particular sauce delete: description:  Delete a particular sauce patch: description:  Update a particular sauce /crusts: displayName:  Collection of different crusts get: description:  Get all crusts post: description:  Create a new crust body: application/json: schema:  !include  schemas/crusts.json /{id}: displayName:  A particular crust get: description:  Get a particular crust delete: description:  Delete a particular crust patch: description:  Update a particular crust
```

**注意端点定义的顺序**。`/pizzas`放在`/toppings`和`/cheeses`之后，因为它与它们相关。`/sauces`和`/crusts`放在`/pizzas`之后，因为它们与之相关。如果您在启动服务器时得到任何类型的关于内容丢失或未定义的错误，请检查定义的顺序。

现在我们可以创造自己的配料和比萨饼风格！

重启服务器并开始烹饪。

```py
$ pserve local.ini
```

让我们从制作夏威夷式披萨开始:

```py
$ http POST :6543/api/toppings name=ham
HTTP/1.1 201 Created...
```

```py
$ http POST :6543/api/toppings name=pineapple
HTTP/1.1 201 Created...
```

```py
$ http POST :6543/api/cheeses name=mozzarella
HTTP/1.1 201 Created...
```

```py
$ http POST :6543/api/sauces name=tomato
HTTP/1.1 201 Created...
```

```py
$ http POST :6543/api/crusts name=plain
HTTP/1.1 201 Created...
```

```py
$ http POST :6543/api/pizzas name=hawaiian toppings:=[1,2] cheeses:=[1] sauce=1 crust=1
```

[*Remove ads*](/account/join/)

### 给你！*

*[![Hawaiian pizza](img/c613e5d97f9401cf3286d70eafec61cf.png)](https://files.realpython.com/media/Hawaiian_pizza.c1db12b77980.jpg)

这是它所有油腻的荣耀:

```py
HTTP/1.1 201 Created
Cache-Control: max-age=0, must-revalidate, no-cache, no-store
Content-Length: 373
Content-Type: application/json; charset=UTF-8
Date: Fri, 05 Jun 2015 18:47:53 GMT
Expires: Fri, 05 Jun 2015 18:47:53 GMT
Last-Modified: Fri, 05 Jun 2015 18:47:53 GMT
Location: http://localhost:6543/api/pizzas/1
Pragma: no-cache
Server: waitress

{
 "data": {
 "_type": "Pizza",
 "_version": 0,
 "cheeses": [
 1
 ],
 "crust": 1,
 "crust_id": 1,
 "description": null,
 "id": 1,
 "name": "hawaiian",
 "sauce": 1,
 "sauce_id": 1,
 "self": "http://localhost:6543/api/pizzas/1",
 "toppings": [
 1,
 2
 ],
 "updated_at": null
 },
 "explanation": "",
 "id": "1",
 "message": null,
 "status_code": 201,
 "timestamp": "2015-06-05T18:47:53Z",
 "title": "Created"
}
```

## 种子数据

加分的最后一步是导入一堆现有的配料记录，让事情变得更有趣。

首先在 pizza_factory 项目中创建一个`seeds/`目录，并下载种子数据:

```py
$ mkdir seeds
$ cd seeds/
$ http -d https://raw.githubusercontent.com/chrstphrhrt/ramses-tutorial/master/pizza_factory/seeds/crusts.json
$ http -d https://raw.githubusercontent.com/chrstphrhrt/ramses-tutorial/master/pizza_factory/seeds/sauces.json
$ http -d https://raw.githubusercontent.com/chrstphrhrt/ramses-tutorial/master/pizza_factory/seeds/cheeses.json
$ http -d https://raw.githubusercontent.com/chrstphrhrt/ramses-tutorial/master/pizza_factory/seeds/toppings.json
```

现在，使用内置的 post2api 脚本将所有成分加载到您的 api 中。

```py
$ nefertari.post2api -f crusts.json -u http://localhost:6543/api/crusts
$ nefertari.post2api -f sauces.json -u http://localhost:6543/api/sauces
$ nefertari.post2api -f cheeses.json -u http://localhost:6543/api/cheeses
$ nefertari.post2api -f toppings.json -u http://localhost:6543/api/toppings
```

你现在可以很容易地列出不同的成分。

```py
$ http :6543/api/toppings
```

或者按名称搜索成分。

```py
$ http :6543/api/toppings?name=chicken

HTTP/1.1 200 OK
Cache-Control: max-age=0, must-revalidate, no-cache, no-store
Content-Length: 934
Content-Type: application/json; charset=UTF-8
Date: Fri, 05 Jun 2015 19:58:48 GMT
Etag: "fd29d8eda6441cebdd632960a21c8136"
Expires: Fri, 05 Jun 2015 19:58:48 GMT
Last-Modified: Fri, 05 Jun 2015 19:58:48 GMT
Pragma: no-cache
Server: waitress

{
 "count": 4,
 "data": [
 {
 "_score": 2.3578677,
 "_type": "Topping",
 "_version": 0,
 "description": null,
 "id": 28,
 "name": "Chicken Tikka",
 "pizza": null,
 "pizza_id": null,
 "self": "http://localhost:6543/api/toppings/28",
 "updated_at": null
 },
 {
 "_score": 2.3578677,
 "_type": "Topping",
 "_version": 0,
 "description": null,
 "id": 27,
 "name": "Chicken Masala",
 "pizza": null,
 "pizza_id": null,
 "self": "http://localhost:6543/api/toppings/27",
 "updated_at": null
 },
 {
 "_score": 2.0254436,
 "_type": "Topping",
 "_version": 0,
 "description": null,
 "id": 14,
 "name": "BBQ Chicken",
 "pizza": null,
 "pizza_id": null,
 "self": "http://localhost:6543/api/toppings/14",
 "updated_at": null
 },
 {
 "_score": 2.0254436,
 "_type": "Topping",
 "_version": 0,
 "description": null,
 "id": 19,
 "name": "Cajun Chicken",
 "pizza": null,
 "pizza_id": null,
 "self": "http://localhost:6543/api/toppings/19",
 "updated_at": null
 }
 ],
 "fields": "",
 "start": 0,
 "took": 3,
 "total": 4
}
```

所以，让我们通过寻找原料来做最后一个披萨。这次吃素食怎么样？

可能有一点菠菜、意大利乳清干酪、晒干的番茄酱和全麦面包皮。首先我们找到我们的 id(你的可能不同)..

```py
$ http :6543/api/toppings?name=spinach
...
"id": 88,
"name": "Spinach",
...
$ http :6543/api/cheeses?name=ricotta
...
"id": 18,
"name": "Ricotta",
...
$ http :6543/api/sauces?name=sun
...
"id": 18,
"name": "Sun Dried Tomato",
...
$ http :6543/api/crusts?name=whole
...
"id": 13,
"name": "Whole Wheat",
...
```

烘烤 0 秒钟，然后..

```py
$ http POST :6543/api/pizzas name="Veggie Delight" toppings:=[88] cheeses:=[18] sauce=18 crust=13

HTTP/1.1 201 Created
Cache-Control: max-age=0, must-revalidate, no-cache, no-store
Content-Length: 382
Content-Type: application/json; charset=UTF-8
Date: Fri, 05 Jun 2015 20:17:26 GMT
Expires: Fri, 05 Jun 2015 20:17:26 GMT
Last-Modified: Fri, 05 Jun 2015 20:17:26 GMT
Location: http://localhost:6543/api/pizzas/2
Pragma: no-cache
Server: waitress

{
 "data": {
 "_type": "Pizza",
 "_version": 0,
 "cheeses": [
 18
 ],
 "crust": 13,
 "crust_id": 13,
 "description": null,
 "id": 2,
 "name": "Veggie Delight",
 "sauce": 18,
 "sauce_id": 18,
 "self": "http://localhost:6543/api/pizzas/2",
 "toppings": [
 88
 ],
 "updated_at": null
 },
 "explanation": "",
 "id": "2",
 "message": null,
 "status_code": 201,
 "timestamp": "2015-06-05T20:17:26Z",
 "title": "Created"
}
```

祝你用餐愉快！

如果你想了解更多关于使用 Python 进行 RESTful API 设计的知识，请查看我们的(免费)迷你指南:

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

在 Readthedocs 上查看完整的 Ramses 文档，在 Github 上查看更高级的[示例项目。](https://github.com/brandicted/ramses-example)***