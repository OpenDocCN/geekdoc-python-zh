# Utils 实用工具

# 发送邮件

### 问题

在 web.py 中，如何发送邮件？

### 解法

在 web.py 中使用`web.sendmail()`发送邮件.

```py
web.sendmail('cookbook@webpy.org', 'user@example.com', 'subject', 'message') 
```

如果在`web.config`中指定了邮件服务器，就会使用该服务器发送邮件，否则，就根据`/usr/lib/sendmail`中的设置发送邮件。

```py
web.config.smtp_server = 'mail.mydomain.com' 
```

如果要发送邮件给多个收件人，就给 to_address 赋值一个邮箱列表。

```py
web.sendmail('cookbook@webpy.org', ['user1@example.com', 'user2@example.com'], 'subject', 'message') 
```

`cc`和`bcc`关键字参数是可选的，分别表示抄送和暗送接收人。这两个参数也可以是列表，表示抄送/暗送多人。

```py
web.sendmail('cookbook@webpy.org', 'user@example.com', 'subject', 'message', cc='user1@example.com', bcc='user2@example.com') 
```

`headers`参数是一个元组，表示附加标头信息(Addition headers)

```py
web.sendmail('cookbook@webpy.org', 'user@example.com', 'subject', 'message',
        cc='user1@example.com', bcc='user2@example.com',
        headers=({'User-Agent': 'webpy.sendmail', 'X-Mailer': 'webpy.sendmail',})
        ) 
```

# 如何用 Gmail 发送邮件

### 问题

如何用 Gmail 发送邮件？

### 解法

安装和维护邮件服务器通常是沉闷乏味的。所以如果你有 Gmail 帐号，就可以使用 Gmail 做为 SMTP 服务器来发送邮件，我们唯一要做的就只是在`web.config`中指定 Gmail 的用户名和密码。

```py
web.config.smtp_server = 'smtp.gmail.com'
web.config.smtp_port = 587
web.config.smtp_username = 'cookbook@gmail.com'
web.config.smtp_password = 'secret'
web.config.smtp_starttls = True 
```

设置好之后，web.sendmail 就能使用 Gmail 帐号来发送邮件了，用起来和其他邮件服务器没有区别。

```py
web.sendmail('cookbook@gmail.com', 'user@example.com', 'subject', 'message') 
```

可以在这里了解有关 Gmail 设置的更多信息 [GMail: Configuring other mail clients](http://mail.google.com/support/bin/answer.py?hl=en&answer=13287)

# 用 soaplib 实现 webservice

### 问题

如何用 soaplib 实现 webservice?

### 解法

Optio 的[soaplib](http://trac.optio.webfactional.com/)通过用装饰器指定类型，从而直接编写 SOAP web service。而且它也是到目前为止，唯一为 web service 提供 WSDL 文档的 Python 类库。

```py
import web 
from soaplib.wsgi_soap import SimpleWSGISoapApp
from soaplib.service import soapmethod
from soaplib.serializers import primitive as soap_types

urls = ("/hello", "HelloService",
        "/hello.wsdl", "HelloService",
        )
render = web.template.Template("$def with (var)\n$:var")

class SoapService(SimpleWSGISoapApp):
    """Class for webservice """

    #__tns__ = 'http://test.com'

    @soapmethod(soap_types.String,_returns=soap_types.String)
    def hello(self,message):
        """ Method for webservice"""
        return "Hello world "+message

class HelloService(SoapService):
    """Class for web.py """
    def start_response(self,status, headers):
        web.ctx.status = status
        for header, value in headers:
            web.header(header, value)

    def GET(self):
        response = super(SimpleWSGISoapApp, self).__call__(web.ctx.environ, self.start_response)
        return render("\n".join(response))

    def POST(self):
        response = super(SimpleWSGISoapApp, self).__call__(web.ctx.environ, self.start_response)
        return render("\n".join(response))

app=web.application(urls, globals())

if __name__ == "__main__":
    app.run() 
```

可以用 soaplib 客户端测试一下：

```py
>>> from soaplib.client import make_service_client
>>> from test import HelloService
>>> client = make_service_client('http://localhost:8080/hello', HelloService())
>>> client.hello('John')
'Hello world John' 
```

可以在[`localhost:8080/hello.wsdl`](http://localhost:8080/hello.wsdl)查看 WSDL。

欲了解更多，请查看 [soaplib](http://trac.optio.webfactional.com/),