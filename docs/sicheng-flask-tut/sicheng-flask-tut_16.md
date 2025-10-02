# Flask 扩展系列(二)–Mail

继续介绍 Flask 常用的扩展，很多站点都需要发送邮件功能，比如用户注册成功邮件，用户重置密码邮件。你可以使用[Python 的 smtplib](https://docs.python.org/2/library/smtplib.html)来发邮件，不过 Flask 有个第三方扩展 Flask-Mail，可以更方便的实现此功能。这里我们就来介绍下这个 Flask-Mail。

### 系列文章

*   Flask 扩展系列(一)–Restful
*   Flask 扩展系列(二)–Mail
*   Flask 扩展系列(三)–国际化 I18N 和本地化 L10N
*   Flask 扩展系列(四)–SQLAlchemy
*   Flask 扩展系列(五)–MongoDB
*   Flask 扩展系列(六)–缓存
*   Flask 扩展系列(七)–表单
*   Flask 扩展系列(八)–用户会话管理
*   Flask 扩展系列(九)–HTTP 认证
*   Flask 扩展系列–自定义扩展

### 安装和启用

建议通过 pip 安装，简单方便：

```py
$ pip install Flask-Mail
```

我们可以采用下面的方法初始化一个 Mail 的实例：

```py
from flask import Flask
from flask_mail import Mail

app = Flask(__name__)
mail = Mail(app)

```

同其他扩展一样，最后一行实例化代码也可以写成下面的方式：

```py
...
mail = Mail()
mail.init_app(app)

```

这样的写法在应用工厂模式下经常会用到。

### 发送邮件

我们来看一段发送 Hello World 邮件的代码：

```py
from flask import Flask
from flask_mail import Mail, Message

app = Flask(__name__)
app.config.update(
    MAIL_SERVER='smtp.example.com',
    MAIL_USERNAME='bjhee',
    MAIL_PASSWORD='example'
)

mail = Mail(app)

@app.route('/mail')
def send_mail():
    msg = Message('Hello',
                  sender=('Billy.J.Hee', 'bjhee@example.com'),
                  recipients=['you@example.com'])
    msg.html = '<h1>Hello World</h1>'
    mail.send(msg)
    return 'Successful'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

```

访问”http://localhost:5000/mail”，邮件就会被发送出去。不用过多解释相信大家都看懂了，因为确实很简单。我们在 Flask 的配置项中配上邮件 SMTP 服务器，用户名和密码；在请求中创建一个 Message 消息实例，传入邮件标题，发件人（元组，包括显示名和地址），收件人（列表）和邮件体；然后调用”Mail.send()”方法发送该消息即可。

Message 对象的属性，可以在初始化时当作参数（key=value)传入，也可以在初始化后赋值。比如我们可以这样定义收件人：

```py
    msg.recipients=['you@example.com']

```

由于收件人可以多个，我们也可以通过”add_recipient()”方法添加收件人：

```py
    msg.add_recipient('he@example.com')

```

发件人可以没有显示名，只要一个地址字符串即可：

```py
    msg.sender='bjhee@example.com'

```

另外，上例中的邮件体是 HTML 格式，如果是 Plain Text 格式的话，你可以这样写：

```py
    msg.body = 'Hello World'

```

### 配置参数

Flask-Mail 扩展可以在 Flask 应用配置项中配置其参数，上例中我们已经看到了 SMTP 服务器，用户名和密码的配置项，这里列举一些常用的：

| 配置项 | 功能 |
| MAIL_SERVER | SMTP 邮件服务器地址，默认为 localhost |
| MAIL_PORT | SMTP 邮件服务器端口，默认为 25 |
| MAIL_USERNAME | 邮件服务器用户名 |
| MAIL_PASSWORD | 邮件服务器密码 |
| MAIL_DEFAULT_SENDER | 默认发件人，如果 Message 对象里没指定发件人，就采用默认发件人 |
| MAIL_USE_TLS | 是否启用 TLS，默认为 False |
| MAIL_USE_SSL | 是否启用 SSL，默认为 False |
| MAIL_MAX_EMAILS | 邮件批量发送个数上限，默认为没有上限 |
| MAIL_ASCII_ATTACHMENTS | 将附件的文件名强制转换为 ASCII 字符，避免在某些情况下出现乱码，默认为 False |
| MAIL_SUPPRESS_SEND | 调用”Mail.send()”方法后，邮件不会真的被发送，在测试环境中使用，默认为 False |

### 批量发送

如果一次需要发送大量邮件，建议采用下面的方式：

```py
@app.route('/batch')
def send_batch():
    with mail.connect() as conn:
        for user in users:
            msg = Message(subject='Hello, %s' % user['name'],
                          body='Welcome, %s' % user['name'],
                          recipients=[user['email']])
            conn.send(msg)

    return 'Successful'

```

这样应用同邮件服务器的连接”mail.connect()”会一直保持到所有邮件发送完毕，也就是退出 with 语句后再关闭，避免多次创建关闭连接的开销。批量发送邮件个数上限由配置项”MAIL_MAX_EMAILS”决定。

### 邮件带附件

回到第一个发送邮件的例子，让我们在 Hello World 邮件中加上附件：

```py
@app.route('/mail')
def send_mail():
    msg = Message('Hello',
                  sender=('Billy.J.Hee', 'bjhee@example.com'),
                  recipients=['you@example.com'])
    msg.html = '<h1>Hello World</h1>'

    # Add Attachment
    with app.open_resource('blank.docx') as fp:
        msg.attach('blank.docx', 'application/msword', fp.read())

    mail.send(msg)
    return 'Successful'

```

上面的代码中，我们通过”app.open_resource()”方法打开了本地当前目录下的”blank.docx”文件，然后通过”Message.attach()”方法将其附到消息对象中即可。”Message.attach()”方法的第一个参数指定了附件上的文件名，第二个参数指定了文件内容的 MIME 类型，第三个参数就是文件内容。

如果大家不知道要附上的文件 MIME 类型是什么，可以查下[MIME 参考手册](http://www.w3school.com.cn/media/media_mimeref.asp)。

### email_dispatched 信号

Flask-Mail 扩展还提供了一个信号”email_dispatched”，当邮件被调度时，该信号就会被发出。如果大家忘了什么是信号，可以参考进阶系列的第二篇。开发者可以通过这个信号来确定邮件是否被发送成功。当然，发送成功不代表被接收成功。

```py
from flask_mail import email_dispatched

def log_mail_sent(message, app):
    print 'Message "%s" is sent successfully' % (message.subject)

email_dispatched.connect(log_mail_sent)

```

另外，在跑测试时我们不希望邮件真的被发出去，此时可以将 Flask 应用的配置项”TESTING”设成 True，或者将 Flask-Mail 扩展的配置项”MAIL_SUPPRESS_SEND”设成 True。这样，调用”Mail.send()”方法后，邮件不会被发出，但是你依然可以接收到”email_dispatched”信号。

本篇中的示例参考了[Flask-Mail 的官方文档](http://pythonhosted.org/Flask-Mail/)和[Flask-Mail 的源码](https://github.com/mattupstate/flask-mail/)。本篇的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ext2.html)