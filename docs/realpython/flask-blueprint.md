# 使用 Flask 蓝图来构建您的应用程序

> 原文：<https://realpython.com/flask-blueprint/>

Flask 是一个非常流行的[网络应用](https://realpython.com/python-web-applications/)框架，它将几乎所有的设计和架构决策留给了开发者。在本教程中，你将了解到一个 **Flask Blueprint** ，或者简称为 **Blueprint** ，如何通过将它的功能组合成可重用的组件来帮助你构建你的 Flask 应用程序。

在本教程中，您将学习:

*   什么是烧瓶设计图以及它们是如何工作的
*   如何创建和使用 Flask 蓝图来组织你的代码
*   如何使用自己或第三方的 Flask 蓝图来提高代码的可重用性

本教程假设您有一些使用 Flask 的经验，并且您以前已经构建过一些应用程序。如果你以前没有使用过 Flask，那么就用 Flask(教程系列)来看看 [Python Web 应用。](https://realpython.com/python-web-applications-with-flask-part-i/)

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

## 烧瓶应用程序是什么样子的

让我们从回顾一个小 Flask 应用程序的结构开始。您可以按照本节中的步骤创建一个小型 web 应用程序。要开始，您需要安装`Flask` Python 包。您可以使用 [`pip`](https://realpython.com/what-is-pip/) 运行以下命令来安装 Flask:

```py
$ pip install Flask==1.1.1
```

上面的命令安装 Flask 版本`1.1.1`。这是您将在本教程中使用的版本，尽管您也可以将在此学到的内容应用到其他版本中。

**注意:**关于如何在虚拟环境中安装 Flask 和其他`pip`选项的更多信息，请查看 [Python 虚拟环境:入门](https://realpython.com/python-virtual-environments-a-primer/)和[什么是 Pip？新蟒蛇指南](https://realpython.com/what-is-pip/)。

安装 Flask 后，就可以开始实现它的功能了。因为 Flask 没有对项目结构施加任何限制，所以您可以按照自己的意愿组织项目代码。对于您的第一个应用程序，您可以使用非常简单的布局，如下所示。一个文件将包含所有应用程序逻辑:

```py
app/
|
└── app.py
```

文件`app.py`将包含应用程序及其视图的定义。

创建 Flask 应用程序时，首先创建一个代表应用程序的`Flask`对象，然后将**视图**与**路线**关联起来。Flask 负责根据请求 URL 和您定义的路由将传入的请求分派到正确的视图。

在 Flask 中，视图可以是接收**请求**并返回该请求的**响应**的任何可调用对象(比如函数)。Flask 负责将响应发送回用户。

以下代码块是您的应用程序的完整源代码:

```py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "This is an example app"
```

这段代码创建了对象`app`，它属于`Flask`类。使用`app.route`装饰器将视图功能`index()`链接到路线`/`。要了解更多关于装饰者的信息，请查看 Python 装饰者入门和 [Python 装饰者 101](https://realpython.com/courses/python-decorators-101/) 。

您可以使用以下命令运行该应用程序:

```py
$ flask run
```

默认情况下，Flask 将在端口 5000 上运行您在`app.py`中定义的应用程序。当应用程序运行时，使用您的网络浏览器进入`http://localhost:5000`。你会看到一个页面显示消息，`This is an example app`。

所选择的项目布局对于非常小的应用程序来说是非常好的，但是它不能很好地伸缩。随着代码的增长，在一个文件中维护所有内容会变得更加困难。所以，当你的应用程序变得越来越大或者越来越复杂时，你可能想用不同的方式来组织你的代码，以保持它的可维护性和清晰易懂。在本教程中，你将学习如何使用 Flask 蓝图来实现这一点。

[*Remove ads*](/account/join/)

## 烧瓶蓝图是什么样子的

Flask 蓝图封装了**功能**，比如视图、模板和其他资源。为了体验 Flask Blueprint 是如何工作的，您可以通过将`index`视图移动到 Flask Blueprint 中来重构之前的应用程序。为此，您必须创建一个包含`index`视图的 Flask 蓝图，然后在应用程序中使用它。

这是这个新应用程序的文件结构:

```py
app/
|
├── app.py
└── example_blueprint.py
```

`example_blueprint.py`将包含烧瓶蓝图实现。然后您将修改`app.py`来使用它。

下面的代码块展示了如何在`example_blueprint.py`中实现这个 Flask 蓝图。它包含一个在路线`/`上的视图，该视图返回文本`This is an example app`:

```py
from flask import Blueprint

example_blueprint = Blueprint('example_blueprint', __name__)

@example_blueprint.route('/')
def index():
    return "This is an example app"
```

在上面的代码中，您可以看到大多数 Flask Blueprint 定义共有的步骤:

1.  **创建一个名为`example_blueprint`的`Blueprint`对象**。
2.  **使用`route`装饰器将**视图添加到`example_blueprint`中。

以下代码块显示了您的应用程序如何导入和使用 Flask Blueprint:

```py
from flask import Flask
from example_blueprint import example_blueprint

app = Flask(__name__)
app.register_blueprint(example_blueprint)
```

要使用任何一个 Flask Blueprint，你必须导入它，然后用`register_blueprint()`在应用程序中注册。当一个 Flask Blueprint 被注册时，这个应用程序会用它的内容进行扩展。

您可以使用以下命令运行该应用程序:

```py
$ flask run
```

当应用程序运行时，使用您的网络浏览器进入`http://localhost:5000`。您将看到一个显示消息`This is an example app`的页面。

## 烧瓶蓝图如何工作

在本节中，您将详细了解如何实现和使用 Flask 蓝图。每个 Flask 蓝图都是一个**对象**，其工作方式与 Flask 应用程序非常相似。它们都可以拥有资源，例如静态文件、模板和与路由相关联的视图。

然而，Flask Blueprint 实际上不是一个应用程序。它需要在应用程序中注册，然后才能运行。当你在应用程序中注册一个 Flask 蓝图时，你实际上是用蓝图的内容来扩展应用程序。

这是任何烧瓶蓝图背后的关键概念。当您在应用程序中注册它们时，它们记录了以后要执行的操作。例如，当您将一个视图与 Flask 蓝图中的一条路线相关联时，它会记录这种关联，以便以后在注册蓝图时在应用程序中进行。

### 制作烧瓶蓝图

让我们重新回顾一下您之前看到的 Flask Blueprint 定义，并详细回顾一下。下面的代码显示了`Blueprint`对象的创建:

```py
from flask import Blueprint

example_blueprint = Blueprint('example_blueprint', __name__)
```

注意，在上面的代码中，一些参数是在创建`Blueprint`对象时指定的。第一个参数`"example_blueprint"`，是蓝图的**名称**，Flask 的路由机制使用它。第二个参数`__name__`是蓝图的**导入名称**，Flask 用它来定位蓝图的资源。

您还可以提供其他可选参数来改变蓝图的行为:

*   **静态文件夹:**可以找到蓝图静态文件的文件夹

*   **静态 url 路径:**提供静态文件的 URL

*   **template_folder:** 包含蓝图模板的文件夹

*   **url_prefix:** 要添加到所有蓝图 url 前面的路径

*   **子域:**默认情况下，该蓝图的路由将匹配的子域

*   **url_defaults:** 一个[字典](https://realpython.com/courses/dictionaries-python/)，该蓝图的视图将接收默认值

*   **root_path:** 蓝图的根目录路径，默认值取自蓝图的导入名称

注意，除了`root_path`之外，所有的路径都是相对于蓝图的目录的。

`Blueprint`对象`example_blueprint`有方法和装饰器，允许你记录在应用程序中注册 Flask 蓝图以扩展它时要执行的操作。最常用的装饰者之一是`route`。它允许您将查看功能与 URL 路由相关联。下面的代码块显示了如何使用这个装饰器:

```py
@example_blueprint.route('/')
def index():
    return "This is an example app"
```

您使用`example_blueprint.route`修饰`index()`，并将函数关联到 URL `/`。

对象还提供了其他有用的方法:

*   **。errorhandler()** 注册一个错误处理函数
*   **。before_request()** 在每个请求之前执行一个动作
*   **。after_request()** 在每次请求后执行一个动作
*   **。app_template_filter()** 在应用层注册一个模板过滤器

你可以在[烧瓶蓝图文档](https://flask.palletsprojects.com/en/1.1.x/blueprints/)中了解更多关于使用蓝图和`Blueprint`类的知识。

[*Remove ads*](/account/join/)

### 在您的应用程序中注册蓝图

回想一下，Flask Blueprint 实际上不是一个应用程序。当你在一个应用程序中注册 Flask Blueprint 时，你**用它的内容扩展**这个应用程序。以下代码显示了如何在应用程序中注册之前创建的 Flask 蓝图:

```py
from flask import Flask
from example_blueprint import example_blueprint

app = Flask(__name__)
app.register_blueprint(example_blueprint)
```

当您调用`.register_blueprint()`时，您将把烧瓶蓝图`example_blueprint`中记录的所有操作应用到`app`。现在，应用程序对 URL `/`的请求将使用 Flask Blueprint 中的`.index()`来处理。

您可以通过向`register_blueprint`提供一些参数来定制 Flask Blueprint 如何扩展应用程序:

*   **url_prefix** 是所有蓝图路线的可选前缀。
*   **子域**是蓝图路由将匹配的子域。
*   **url_defaults** 是一个[字典](https://realpython.com/courses/python-dictionary-iteration/)，具有视图参数的默认值。

当您在不同的项目中共享同一个 Flask 蓝图时，能够在注册时而不是创建时进行一些定制特别有用。

在本节中，您已经看到了 Flask Blueprints 是如何工作的，以及如何创建和使用它们。在接下来的部分中，您将了解如何利用 Flask 蓝图来**构建**您的应用程序，将它们构造成独立的组件。在某些情况下，您还可以在不同的应用程序中重用这些组件，以减少开发时间！

## 如何使用 Flask Blueprints 来构建你的应用程序代码

在这一节中，您将看到如何使用 Flask Blueprint 对一个示例应用程序进行重构。示例应用程序是一个**电子商务**站点，具有以下特征:

*   访客可以注册，登录，恢复**密码**。
*   参观者可以搜索**产品**并查看其详细信息。
*   用户可以将产品添加到他们的**购物车**中，然后结账。
*   API 使外部系统能够搜索和检索**产品信息**。

你不需要太在意实现的细节。相反，您将主要关注如何使用 Flask 蓝图来改进应用程序的架构。

### 理解项目布局的重要性

记住，Flask 不强制任何特定的项目布局。如下组织该应用程序的代码是完全可行的:

```py
ecommerce/
|
├── static/
|   ├── logo.png
|   ├── main.css
|   ├── generic.js
|   └── product_view.js
|
├── templates/
|   ├── login.html
|   ├── forgot_password.html
|   ├── signup.html
|   ├── checkout.html
|   ├── cart_view.html
|   ├── index.html
|   ├── products_list.html
|   └── product_view.html
|
├── app.py
├── config.py
└── models.py
```

该应用程序的代码使用以下目录和文件进行组织:

*   **static/** 包含应用程序的静态文件。
*   **templates/** 包含应用程序的模板。
*   **models.py** 包含应用程序模型的定义。
*   **app.py** 包含应用逻辑。
*   **config.py** 包含应用程序配置参数。

这是一个有多少申请开始的例子。尽管这种布局非常简单，但随着应用程序复杂性的增加，它也有一些缺点。例如，您将很难在其他项目中重用应用程序逻辑，因为所有功能都捆绑在`app.py`中。如果你把这个功能分解成模块，那么你可以在不同的项目中重用完整的模块。

此外，如果您只有一个应用程序逻辑文件，那么您最终会得到一个非常大的`app.py`,它混合了几乎不相关的代码。这可能会使您难以导航和维护脚本。

更重要的是，当你在团队中工作时，大的代码文件是**冲突**的来源，因为每个人都将对同一个文件进行修改。这些只是为什么以前的布局只适合非常小的应用程序的几个原因。

[*Remove ads*](/account/join/)

### 组织您的项目

您可以利用 Flask 蓝图**将代码分割成不同的模块**，而不是使用之前的布局来构建应用程序。在这一节中，您将看到如何构建以前的应用程序，以制作封装相关功能的蓝图。在这个布局中，有五个烧瓶设计图:

1.  **API 蓝图**支持外部系统搜索和检索产品信息
2.  **认证蓝图**允许用户登录并恢复密码
3.  **购物车和结账功能的购物车蓝图**
4.  **首页总蓝图**
5.  **产品蓝图**用于搜索和查看产品

如果您为每个 Flask 蓝图及其资源使用单独的目录，那么项目布局将如下所示:

```py
ecommerce/
|
├── api/
|   ├── __init__.py
|   └── api.py
|
├── auth/
|   ├── templates/
|   |   └── auth/
|   |       ├── login.html
|   |       ├── forgot_password.html
|   |       └── signup.html
|   |
|   ├── __init__.py
|   └── auth.py
|
├── cart/
|   ├── templates/
|   |   └── cart/
|   |       ├── checkout.html
|   |       └── view.html
|   |
|   ├── __init__.py
|   └── cart.py
|
├── general/
|   ├── templates/
|   |   └── general/
|   |       └── index.html
|   |
|   ├── __init__.py
|   └── general.py
|
├── products/
|   ├── static/
|   |   └── view.js
|   |
|   ├── templates/
|   |   └── products/
|   |       ├── list.html
|   |       └── view.html
|   |
|   ├── __init__.py
|   └── products.py
|
├── static/
|   ├── logo.png
|   ├── main.css
|   └── generic.js
|
├── app.py
├── config.py
└── models.py
```

为了以这种方式组织代码，您将所有视图从`app.py`移动到相应的 Flask 蓝图中。您还移动了模板和非全局静态文件。这种结构使您更容易找到与给定功能相关的代码和资源。例如，如果您想找到关于产品的应用逻辑，那么您可以转到`products/products.py`中的产品蓝图，而不是滚动浏览`app.py`。

让我们看看`products/products.py`中的产品蓝图实现:

```py
from flask import Blueprint, render_template
from ecommerce.models import Product

products_bp = Blueprint('products_bp', __name__,
    template_folder='templates',
    static_folder='static', static_url_path='assets')

@products_bp.route('/')
def list():
    products = Product.query.all()
    return render_template('products/list.html', products=products)

@products_bp.route('/view/<int:product_id>')
def view(product_id):
    product = Product.query.get(product_id)
    return render_template('products/view.html', product=product)
```

这段代码定义了`products_bp` Flask 蓝图，并且只包含与产品功能相关的代码。因为这个 Flask 蓝图有自己的模板，所以您需要在`Blueprint`对象创建中指定相对于蓝图根的`template_folder`。由于您指定了`static_folder='static'`和`static_url_path='assets'`，因此`ecommerce/products/static/`中的文件将在`/assets/` URL 下提供。

现在您可以将代码的其余功能转移到相应的 Flask Blueprint 中。换句话说，您可以为 API、身份验证、购物车和一般功能创建蓝图。一旦你这样做了，`app.py`中剩下的唯一代码将是处理应用程序初始化和 Flask Blueprint 注册的代码:

```py
from flask import Flask

from ecommmerce.api.api import api_bp
from ecommmerce.auth.auth import auth_bp
from ecommmerce.cart.cart import cart_bp
from ecommmerce.general.general import general_bp
from ecommmerce.products.products import products_bp

app = Flask(__name__)

app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(auth_bp)
app.register_blueprint(cart_bp, url_prefix='/cart')
app.register_blueprint(general_bp)
app.register_blueprint(products_bp, url_prefix='/products')
```

现在，`app.py`只需导入并注册蓝图来扩展应用程序。因为使用了`url_prefix`，所以可以避免 Flask Blueprint 路由之间的 URL 冲突。例如，URL`/products/`和`/cart/`解析到同一路线`/`的`products_bp`和`cart_bp`蓝图中定义的不同端点。

### 包括模板

在 Flask 中，当一个视图呈现一个模板时，在应用程序的**模板搜索路径**中注册的所有目录中搜索模板文件。默认情况下，这个路径是`["/templates"]`，所以只在应用程序根目录下的`/templates`目录中搜索模板。

如果在创建蓝图时设置了`template_folder`参数，那么当注册 Flask 蓝图时，它的 templates 文件夹将被添加到应用程序的模板搜索路径中。然而，如果模板搜索路径的不同目录下有重复的文件路径，那么**将优先于**，这取决于它们的注册顺序。

例如，如果一个视图请求模板`view.html`，并且在模板搜索路径的不同目录中有相同名称的文件，那么其中一个将优先于另一个。因为可能很难记住优先顺序，所以最好**避免在不同的模板目录中的同一个路径**下有文件。这就是应用程序中模板的以下结构有意义的原因:

```py
ecommerce/
|
└── products/
    └── templates/
        └── products/
            ├── search.html
            └── view.html
```

首先，让 Flask Blueprint 名称出现两次可能看起来是多余的:

1.  作为蓝图的**根**目录
2.  在**模板**目录内

然而，要知道通过这样做，你可以避免不同蓝图之间可能的**模板名称冲突**。使用这个目录结构，任何需要产品的`view.html`模板的视图都可以在调用`render_template`时使用`products/view.html`作为模板文件名。这避免了与属于 Cart 蓝图的`view.html`的冲突。

最后一点，重要的是要知道应用程序的`template`目录中的模板比蓝图的模板目录中的模板具有更高的优先级**。如果您想覆盖 Flask Blueprint 模板，而不实际修改模板文件，这可能很有用。*

*例如，如果您想要覆盖产品蓝图中的模板`products/view.html`，那么您可以通过在应用程序`templates`目录中创建一个新文件`products/view.html`来实现:

```py
ecommerce/
|
├── products/
|   └── templates/
|       └── products/
|           ├── search.html
|           └── view.html
|
└── templates/
        └── products/
            └── view.html
```

当你这样做时，每当一个视图需要模板`products/view.html`时，你的程序将使用`templates/products/view.html`而不是`products/templates/products/view.html`。

[*Remove ads*](/account/join/)

### 提供视图以外的功能

到目前为止，您只看到了用视图扩展应用程序的蓝图，但是 Flask 蓝图不一定只提供视图！他们可以用**模板、静态文件和模板过滤器**扩展应用程序。例如，您可以创建一个 **Flask Blueprint** 来提供一组图标，并在您的应用程序中使用它。这将是这种蓝图的文件结构:

```py
app/
|
└── icons/
    ├── static/
    |   ├── add.png
    |   ├── remove.png
    |   └── save.png
    |
    ├── __init__.py
    └── icons.py
```

`static`文件夹包含图标文件，而`icons.py`是烧瓶蓝图定义。

这是`icons.py`可能的样子:

```py
from flask import Blueprint

icons_bp = Blueprint('icons_bp', __name__,
    static_folder='static',
    static_url_path='icons')
```

这段代码定义了`icons_bp` Flask 蓝图，该蓝图公开了位于`/icons/` URL 下的静态目录中的文件。请注意，此蓝图没有定义任何路线。

当您可以创建将视图和其他类型的内容打包的蓝图时，您的代码和资产在应用程序中的可重用性就更高了。在下一节中，您将了解更多关于 Flask Blueprint 可重用性的内容。

## 如何使用 Flask Blueprints 改进代码重用

除了代码组织之外，将 Flask 应用程序组织成独立组件的**集合还有另一个好处。您甚至可以跨不同的应用程序重用这些组件！例如，如果您创建了一个为联系人表单提供功能的 Flask 蓝图，那么您可以在所有应用程序中重用它。**

您还可以利用其他开发人员创建的蓝图来加速您的工作。虽然现有的 Flask 蓝图没有集中的存储库，但是你可以使用 [Python 包索引](https://pypi.org)、 [GitHub 搜索](https://github.com/search?q=flask+blueprint)和网络搜索引擎来找到它们。你可以在[了解更多关于搜索 PyPI 包的信息什么是 Pip？新蟒蛇指南](https://realpython.com/what-is-pip/)。

有各种各样的 Flask Blueprints 和 **Flask Extensions** (它们是使用 blue print 实现的)，它们提供的功能可能对您有用:

*   证明
*   管理/CRUD 生成
*   CMS 功能
*   还有更多！

您可以考虑搜索一个可以重用的现有 Flask 蓝图或扩展，而不是从头开始编写应用程序。利用第三方蓝图和扩展可以帮助您减少开发时间，并将精力集中在应用程序的核心逻辑上！

## 结论

在本教程中，您已经了解了 Flask Blueprints 如何工作，如何使用它们，以及它们如何帮助您组织应用程序的代码。Flask Blueprints 是处理不断增加的应用程序复杂性的一个很好的工具。

**你已经学会:**

*   什么是**烧瓶蓝图**以及它们是如何工作的
*   你如何**实现和使用**一个烧瓶蓝图
*   Flask Blueprints 如何帮助你组织应用程序的代码
*   如何使用 Flask Blueprints 来简化自己和第三方组件的可重用性
*   如何在你的项目中使用 Flask 蓝图来减少开发时间

您可以使用在本教程中学到的知识，开始将您的应用程序组织成一组蓝图。当您以这种方式设计您的应用程序时，您将改进代码重用、可维护性和团队合作！******