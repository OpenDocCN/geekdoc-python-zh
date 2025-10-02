# web.py 0.3 新手指南

*   开始
*   URL 处理
*   GET 和 POST 的区别
*   启动服务
*   模板
*   表单
*   数据库
*   开发
*   下一步做什么?

## 开始

你知道 Python 同时你希望制作一个网站。 那么 web.py 正好提供了一种简单的方法。

如果你希望读完整个指南， 你需要安装 Python, web.py, flup, psycopg2, 和 Postgres (或者等价的数据库和 Python 驱动)。 详细，可以查看 [webpy.org](http://webpy.org/).

如果你已经有了一个 web.py 项目，请看看升级 页面的相关信息。

准备开始。

## URL 处理

任何网站最重要的部分就是它的 URL 结构。你的 URL 并不仅仅只是访问者所能看到并且能发给朋友的。它还规定了你网站运行的心智模型。在一些类似[del.icio.us](http://del.icio.us/)的流行网站 , URL 甚至是 UI 的一部分。 web.py 使这类强大的 URL 成为可能。

在开始你的 web.py 程序之前,打开一个文本文件（文件名为 code.py）输入:

```py
import web 
```

这条语句会导入 web.py 模块。

现在我们需要把我们的 URL 结构告诉 web.py。让我从下面这个简单的例子开始:

```py
urls = (
  '/', 'index'
) 
```

第一部分是匹配 URL 的[正则表达式](http://osteele.com/tools/rework/)，像`/`、`/help/faq`、`/item/(\d+)`等(`\d+`将匹配数字)。圆括号表示捕捉对应的数据以便后面使用。第二部分是接受请求的类名称，像`index`、`view`、`welcomes.hello` (`welcomes`模块的`hello`类)，或者`get_\1`。`\1` 会被正则表达式捕捉到的内容替换，剩下来捕捉的的内容将被传递到你的函数中去。

这行表示我们要 URL`/`(首页)被一个叫`index`的类处理。

现在我们需要创建一个列举这些 url 的 application。

```py
app = web.application(urls, globals()) 
```

这会告诉 web.py 去创建一个基于我们刚提交的 URL 列表的 application。这个 application 会在这个文件的全局命名空间中查找对应类。

## GET 和 POST: 区别

现在我们需要来写`index`类。虽然大多数人只会看看，并不会注意你的浏览器在使用用于与万维网通信的 HTTP 语言。具体的细节并不重要，但是要理解 web 访问者请求 web 服务器去根据 URL(像`/`、`/foo?f=1`)执行一个合适的函数（像`GET`、`POST`）的基本思想。

`GET`是我们都熟悉的。它用于请求网页文本。当你在浏览器输入`harvard.edu`，它会直接访问 Harvard 的 web 服务器，去`GET /`。 第二个最有名的是`POST`，它经常被用在提交 form，比如请求买什么东西。每当提交一个去做什么事情(像使用信用卡处理一笔交易)的请求时，你可以使用`POST`。这是关键，因为`GET`的 URL 可以被搜索引擎索引，并通过搜索引擎访问。虽然大部分页面你希望被索引，但是少数类似订单处理的页面你是不希望被索引的 (想象一下 Google 尝试去购买你网站上的所有东西)。

在我们 web.py 的代码中，我们将这两个方法明确区分:

```py
class index:
    def GET(self):
        return "Hello, world!" 
```

当有人用`GET`请求`/`时，这个`GET`函数随时会被 web.py 调用。

好了，限制我们只需要最后一句就写完了。这行会告诉 web.py 开始提供 web 页面:

```py
if __name__ == "__main__": app.run() 
```

这会告诉 web.py 为我们启动上面我们写的应用。

现在注意，即使我已经在这里说了很多，但我们真正有 5 行这些代码。这就是你需要编写的一个完整的 web.py 应用。 为了更方便的使用，你的完整代码应该像下面这样:

```py
import web

urls = (
    '/', 'index'
)

class index:
    def GET(self):
        return "Hello, world!"

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run() 
```

## 启动服务

如果你在命令行下面，请输入:: $ python code.py [`0.0.0.0:8080/`](http://0.0.0.0:8080/)

现在你的 web.py 应用正运行在你电脑上的一个真正的 web 服务器上。 访问那个 URL，然后你应该看到"Hello, world!" (你可以通过把 IP 地址/端口加在"code.py"的后面，来控制 web.py 在哪里启动服务器。你也可以让它运行在`fastcgi`或`scgi`服务器上)。

**注意:** 如果你不能或者不想使用默认端口，你可以使用这样的命令来指定端口号:

```py
$ python code.py 1234 
```

## 模板

在 Python 中写 HTML 不是聪明的选择，相反在 HTML 中写 Python 则有趣的多。幸运的是，`web.py` 让这件事情做得简单而又漂亮。

**注意：** 老版本的 `web.py` 使用 [Cheetah 模板系统](http://www.cheetahtemplate.org/)，你可以也欢迎使用其他模板系统，但它可能不会被长久支持。

给模板新建一个目录（命名为 `templates`），在该目录下新建一个以 `.html` 结尾的文件，内容如下：

```py
<em>Hello</em>, world! 
```

你也可以在模板中使用 `web.py` 模板支持代码：

```py
$def with (name)

$if name:
    I just wanted to say <em>hello</em> to $name.
$else:
    <em>Hello</em>, world! 
```

如上，该模板看起来就像 python 文件一样，除了顶部的 `def with` (表示从模板将从这后面取值)和总是位于代码段之前的`$`。当前，`template.py` 首先请求模板文件的首行 `$def` 。当然，你要注意 `web.py` 将会转义任何任何用到的变量，所以当你将 `name` 的值设为是一段 HTML 时，它会被转义显示成纯文本。如果要关闭该选项，可以写成 `$:name` 来代替 `$name`。

回看再看 `code.py`。在第一行之下添加：

```py
render = web.template.render('templates/') 
```

这会告诉 web.py 到你的模板目录中去查找模板。然后把 `index.GET`改成: 告诉 `web.py` 在你的模板目录下查找模板文件。修改 `index.GET` ：

```py
name = 'Bob'    
return render.index(name) 
```

（'index' 是模板的名字，'name' 是传入模板的一个参数）

访问站点它将显示 hello Bob。

但是如果我们想让用户自行输入他的名字，么办？如下：

```py
i = web.input(name=None)
return render.index(i.name) 
```

访问 `/` 将显示 hello world，访问 `/?name=Joe` 将显示 hello Joe。

URL 的后面的 `?` 看起来不好看？修改下 URL 配置：

```py
'/(.*)', 'index' 
```

然后修改下 `index.GET`：

```py
def GET(self, name):
    return render.index(name) 
```

现在访问 `/Joe` 看看，它会显示 hello Joe。

如果学习更多关于 web.py 的模板处理，请访问 web.py 模板.

## 表单

web.py 的 form 模块能够帮助你生成 HTML 表单；获取用户的输入，并在处理或添加到数据库之前对其进行内容的验证。 如果你要学习更多关于 form 模块的使用，请查看帮助文档或者 Form 类库的链接

## 数据库操作

**注意:** 在你开始使用数据库之前，确保你已经安装了合适的数据库访问库。比如对于 MySQL 数据库，使用 [MySQLdb](http://sourceforge.net/project/showfiles.php?group_id=22307) ，对于 Postgres 数据库使用[psycopg2](http://initd.org/pub/software/psycopg/)。

首先你需要创建一个数据库对象。

```py
db = web.database(dbn='postgres', user='username', pw='password', db='dbname') 
```

(根据需要修改这里 -- 尤其是`username` 、 `password` 、 `dbname` -- 。 MySQL 用户还需要把 `dbn` 定义改为 `mysql`。)

这就是所有你需要做的 -- web.py 将会自动处理与数据库的连接和断开。

使用的的数据库引擎管理工具，在你的库中创建一个简单的表:

```py
CREATE TABLE todo (
  id serial primary key,
  title text,
  created timestamp default now(),
  done boolean default 'f'    ); 
```

然后初始化行:

```py
INSERT INTO todo (title) VALUES ('Learn web.py'); 
```

我们回来继续编辑 `code.py` ，把 `index.GET` 改成下面的样子，替换整个函数:

```py
def GET(self):
    todos = db.select('todo')
    return render.index(todos) 
```

然后把 URL 列表改回来，只保留 `/`:

```py
'/', 'index', 
```

像这样编辑并替换 `index.html` 的全部内容:

```py
$def with (todos)
<ul>
$for todo in todos:
    <li id="t$todo.id">$todo.title</li>
</ul> 
```

再访问你的网站，然后你可以看到你的 todo item: "Learn web.py"。恭喜你！你已经完整地写好了一个可以从数据库读取数据的程序。现在让我们同样再写一个可以把数据写入数据库的程序。

在 `index.html`尾部添加:

```py
<form method="post" action="add">
<p><input type="text" name="title" /> <input type="submit" value="Add" /></p>
</form> 
```

然后把你的 URL 列表改为:

```py
'/', 'index',
'/add', 'add' 
```

(你必须要非常小心那些逗号。如果你省略他们，Python 会把所有字符串连接起来,变成 `'/index/addadd'`)

现在添加另一个类:

```py
class add:
    def POST(self):
        i = web.input()
        n = db.insert('todo', title=i.title)
        raise web.seeother('/') 
```

(注意现在我们正在使用 `POST`)

`web.input` 可以让你访问用户通过 form 提交的任何数据。

注意: 如果要访问多个相同名字的字段，请使用 list 的格式(比如:一串 name="name"的多选框):

```py
post_data=web.input(name=[]) 
```

`db.insert` 把数据插入数据表 `todo` ，然后把新的行号返回给你。 `seeother` 把用户重定向到指定的 URL。

一些快速补充说明: `db.update` 与 `db.insert` 差不多，除了它返回的行号是直接从 sql 语句里面提取的(`WHERE ID=2`)。

`web.input`、 `db.query`已经其他 web.py 中的函数返回"Storage objects"，这些东西就像字典，你除了可以 `d['foo']`之外，你还可以 `d.foo`。这可以让代码更加干净。

你可以在 the documentation 找到这方面具体的细节以及所有 web.py 的函数说明。

## 开发

web.py 还有一些帮助我们 debug 的工具。当它在内建的服务器中运行时，它会一 debug 模式启动程序。在 debug 模式中，任何代码、模板的修改，都会让服务器重新加载它们，然后还会输出有用的错误消息。

只有在生产环境中 debug 模式是关闭的。如果你想禁用 debug 模式，你可以在创建程序/模板前添加像这样的行。

```py
web.config.debug = False 
```

我们的指南就到这里了。如果要做更多很酷的东西，你可以先查看一下文档。

## 下一步是什么?

*   更多文档
*   Cookbook
*   code samples