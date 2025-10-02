# Overview

## Overview

[FriendFeed](http://friendfeed.com/)使用了一款使用 Python 编写的，相对简单的 非阻塞式 Web 服务器。其应用程序使用的 Web 框架看起来有些像 [web.py](http://webpy.org/) 或者 Google 的 [webapp](http://code.google.com/appengine/docs/python/tools/webapp/)， 不过为了能有效利用非阻塞式服务器环境，这个 Web 框架还包含了一些相关的有用工具 和优化。

[Tornado](http://sebug.net/paper/books/tornado/) 就是我们在 FriendFeed 的 Web 服务器及其常用工具的开源版本。Tornado 和现在的主流 Web 服务器框架（包括大多数 Python 的框架）有着明显的区别：它是非阻塞式服务器，而且速度相当快。得利于其 非阻塞的方式和对 [epoll](http://www.kernel.org/doc/man-pages/online/pages/man4/epoll.4.html) 的运用，Tornado 每秒可以处理数以千计的连接，因此 Tornado 是实时 Web 服务的一个 理想框架。我们开发这个 Web 服务器的主要目的就是为了处理 FriendFeed 的实时功能 ——在 FriendFeed 的应用里每一个活动用户都会保持着一个服务器连接。（关于如何扩容 服务器，以处理数以千计的客户端的连接的问题，请参阅 [The C10K problem](http://www.kegel.com/c10k.html) ）

以下是经典的 “Hello, world” 示例：

```py
import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

application = tornado.web.Application([
    (r"/", MainHandler),
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start() 
```

查看下面的 Tornado 攻略以了解更多关于 `tornado.web` 包 的细节。

我们清理了 Tornado 的基础代码，减少了各模块之间的相互依存关系，所以理论上讲， 你可以在自己的项目中独立地使用任何模块，而不需要使用整个包。