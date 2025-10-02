# WSGI 和 Google AppEngine

## WSGI 和 Google AppEngine

Tornado 对 [WSGI](http://wsgi.org/) 只提供了有限的支持，即使如此，因为 WSGI 并不支持非阻塞式的请求，所以如果你使用 WSGI 代替 Tornado 自己的 HTTP 服务的话，那么你将无法使用 Tornado 的异步非阻塞式的请求处理方式。 比如 `@tornado.web.asynchronous`、`httpclient` 模块、`auth` 模块， 这些将都无法使用。

你可以通过 `wsgi` 模块中的 `WSGIApplication` 创建一个有效的 WSGI 应用（区别于 我们用过的 `tornado.web.Application`）。下面的例子展示了使用内置的 WSGI `CGIHandler` 来创建一个有效的 [Google AppEngine](http://code.google.com/appengine/) 应用。

```py
import tornado.web
import tornado.wsgi
import wsgiref.handlers

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

if __name__ == "__main__":
    application = tornado.wsgi.WSGIApplication([
        (r"/", MainHandler),
    ])
    wsgiref.handlers.CGIHandler().run(application) 
```

请查看 demo 中的 `appengine` 范例，它是一个基于 Tornado 的完整的 AppEngine 应用。