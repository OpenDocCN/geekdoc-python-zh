# 模块索引

## 模块索引

最重要的一个模块是[`web`](http://github.com/facebook/tornado/blob/master/tornado/web.py)， 它就是包含了 Tornado 的大部分主要功能的 Web 框架。其它的模块都是工具性质的， 以便让 `web` 模块更加有用 后面的 Tornado 攻略 详细讲解了 `web` 模块的使用方法。

# 主要模块

### 主要模块

*   [`web`](http://github.com/facebook/tornado/blob/master/tornado/web.py) - FriendFeed 使用的基础 Web 框架，包含了 Tornado 的大多数重要的功能
*   [`escape`](http://github.com/facebook/tornado/blob/master/tornado/escape.py) - XHTML, JSON, URL 的编码/解码方法
*   [`database`](http://github.com/facebook/tornado/blob/master/tornado/database.py) - 对 `MySQLdb` 的简单封装，使其更容易使用
*   [`template`](http://github.com/facebook/tornado/blob/master/tornado/template.py) - 基于 Python 的 web 模板系统
*   [`httpclient`](http://github.com/facebook/tornado/blob/master/tornado/httpclient.py) - 非阻塞式 HTTP 客户端，它被设计用来和 `web` 及 `httpserver` 协同工作
*   [`auth`](http://github.com/facebook/tornado/blob/master/tornado/auth.py) - 第三方认证的实现（包括 Google OpenID/OAuth、Facebook Platform、Yahoo BBAuth、FriendFeed OpenID/OAuth、Twitter OAuth）
*   [`locale`](http://github.com/facebook/tornado/blob/master/tornado/locale.py) - 针对本地化和翻译的支持
*   [`options`](http://github.com/facebook/tornado/blob/master/tornado/options.py) - 命令行和配置文件解析工具，针对服务器环境做了优化

# 底层模块

### 底层模块

*   [`httpserver`](http://github.com/facebook/tornado/blob/master/tornado/httpserver.py) - 服务于 `web` 模块的一个非常简单的 HTTP 服务器的实现
*   [`iostream`](http://github.com/facebook/tornado/blob/master/tornado/iostream.py) - 对非阻塞式的 socket 的简单封装，以方便常用读写操作
*   [`ioloop`](http://github.com/facebook/tornado/blob/master/tornado/ioloop.py) - 核心的 I/O 循环