# 第七章：外部服务认证

第六章的例子像我们展示了如何使用安全 cookies 和 tornado.web.authenticated 装饰器来实现一个简单的用户验证表单。在本章中，我们将着眼于如何对第三方服务进行身份验证。流行的 Web API，比如 Facebbok 和 Twitter，使用 OAuth 协议安全验证某人的身份，同时允许他们的用户保持第三方应用访问他们个人信息的控制权。Tornado 提供了一些 Python mix-in 来帮助开发者验证外部服务，既包括显式地支持流行服务，也包括通过通用的 OAuth 支持。在本章中，我们将探讨两个使用 Tornado 的 auth 模块的示例应用：一个连接 Twitter，另一个连接 Facebook。

*   7.1 Tornado 的 auth 模块
    *   7.1.1 认证流程
    *   7.1.2 异步请求
    *   7.2 示例：登录 Twitter
    *   7.3 示例：Facebook 认证和 Graph API

## 7.1 Tornado 的 auth 模块

作为一个 Web 应用开发者，你可能想让用户直接通过你的应用在 Twitter 上发表更新或读取最新的 Facebook 状态。大多数社交网络和单一登录的 API 为验证你应用中的用户提供了一个标准的流程。Tornado 的 auth 模块为 OpenID、OAuth、OAuth 2.0、Twitter、FriendFeed、Google OpenID、Facebook REST API 和 Facebook Graph API 提供了相应的类。尽管你可以自己实现对于特定外部服务认证过程的处理，不过 Tornado 的 auth 模块为连接任何支持的服务开发应用提供了简单的工作流程。

### 7.1.1 认证流程

这些认证方法的工作流程虽然有一些轻微的不同，但对于大多数而言，都使用了 authorize_redirect 和 get_authenticated_user 方法。authorize_rediect 方法用来将一个未授权用户重定向到外部服务的验证页面。在验证页面中，用户登录服务，并让你的应用拥有访问他账户的权限。通常情况下，你会在用户带着一个临时访问码返回你的应用时使用 get_authenticated_user 方法。调用 get_authenticated_user 方法会把授权跳转过程提供的临时凭证替换成属于用户的长期凭证。Twitter、Facebook、FriendFeed 和 Google 的具体验证类提供了他们自己的函数来使 API 调用它们的服务。

### 7.1.2 异步请求

关于 auth 模块需要注意的一件事是它使用了 Tornado 的异步 HTTP 请求。正如我们在第五章所看到的，异步 HTTP 请求允许 Tornado 服务器在一个挂起的请求等待传出请求返回时处理传入的请求。

我们将简单的看下如何使用异步请求，然后在一个例子中使用它们进行深入。每个发起异步调用的处理方法必须在它前面加上@tornado.web.asynchronous 装饰器。

## 7.2 示例：登录 Twitter

让我们来看一个使用 Twitter API 验证用户的例子。这个应用将重定向一个没有登录的用户到 Twitter 的验证页面，提示用户输入用户名和密码。然后 Twitter 会将用户重定向到你在 Twitter 应用设置页指定的 URL。

首先，你必须在 Twitter 注册一个新应用。如果你还没有应用，可以从[Twitter 开发者网站](https://dev.twitter.com/)的"Create a new application"链接开始。一旦你创建了你的 Twitter 应用，你将被指定一个 access token 和一个 secret 来标识你在 Twitter 上的应用。你需要在本节下面代码的合适位置填充那些值。

现在让我们看看代码清单 7-1 中的代码。

代码清单 7-1 查看 Twitter 时间轴：twitter.py

```py
import tornado.web
import tornado.httpserver
import tornado.auth
import tornado.ioloop

class TwitterHandler(tornado.web.RequestHandler, tornado.auth.TwitterMixin):
    @tornado.web.asynchronous
    def get(self):
        oAuthToken = self.get_secure_cookie('oauth_token')
        oAuthSecret = self.get_secure_cookie('oauth_secret')
        userID = self.get_secure_cookie('user_id')

        if self.get_argument('oauth_token', None):
            self.get_authenticated_user(self.async_callback(self._twitter_on_auth))
            return

        elif oAuthToken and oAuthSecret:
            accessToken = {
                'key': oAuthToken,
                'secret': oAuthSecret
            }
            self.twitter_request('/users/show',
                access_token=accessToken,
                user_id=userID,
                callback=self.async_callback(self._twitter_on_user)
            )
            return

        self.authorize_redirect()

    def _twitter_on_auth(self, user):
        if not user:
            self.clear_all_cookies()
            raise tornado.web.HTTPError(500, 'Twitter authentication failed')

        self.set_secure_cookie('user_id', str(user['id']))
        self.set_secure_cookie('oauth_token', user['access_token']['key'])
        self.set_secure_cookie('oauth_secret', user['access_token']['secret'])

        self.redirect('/')

    def _twitter_on_user(self, user):
        if not user:
            self.clear_all_cookies()
            raise tornado.web.HTTPError(500, "Couldn't retrieve user information")

        self.render('home.html', user=user)

class LogoutHandler(tornado.web.RequestHandler):
    def get(self):
        self.clear_all_cookies()
        self.render('logout.html')

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/', TwitterHandler),
            (r'/logout', LogoutHandler)
        ]

        settings = {
            'twitter_consumer_key': 'cWc3 ... d3yg',
            'twitter_consumer_secret': 'nEoT ... cCXB4',
            'cookie_secret': 'NTliOTY5NzJkYTVlMTU0OTAwMTdlNjgzMTA5M2U3OGQ5NDIxZmU3Mg==',
            'template_path': 'templates',
        }

        tornado.web.Application.__init__(self, handlers, **settings)

if __name__ == '__main__':
    app = Application()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(8000)
    tornado.ioloop.IOLoop.instance().start()

```

代码清单 7-2 和 7-3 的模板文件应该被放在应用的 templates 目录下。

代码清单 7-2 Twitter 时间轴：home.html

```py
<html>
    <head>
        <title>{{ user['name'] }} ({{ user['screen_name'] }}) on Twitter</title>
    </head>

    <body>
        <div>
            <a href="/logout">Sign out</a>
        </div>
        <div>
            <img src="{{ user['profile_image_url'] }}" style="float:left" />
            <h2>About @{{ user['screen_name'] }}</h2>
            <p style="clear:both"><em>{{ user['description'] }}</em></p>
        </div>
        <div>
            <ul>
                <li>{{ user['statuses_count'] }} tweets.</li>
                <li>{{ user['followers_count'] }} followers.</li>
                <li>Following {{ user['friends_count'] }} users.</li>
            </ul>
        </div>
        {% if 'status' in user %}
            <hr />
            <div>
                <p>
                    <strong>{{ user['screen_name'] }}</strong>
                    <em>on {{ ' '.join(user['status']['created_at'].split()[:2]) }}
                        at {{ user['status']['created_at'].split()[3] }}</em>
                </p>
                <p>{{ user['status']['text'] }}</p>
            </div>
        {% end %}
    </body>
</html>

```

代码清单 7-3 Twitter 时间轴：logout.html

```py
<html>
    <head>
        <title>Tornadoes on Twitter</title>
    </head>

    <body>
        <div>
            <h2>You have successfully signed out.</h2>
            <a href="/">Sign in</a>
        </div>
    </body>
</html>

```

让我们分块进行分析，首先从 twitter.py 开始。在 Application 类的**init**方法中，你将注意到有两个新的键出现在设置字典中：twitter_consumer_key 和 twitter_consumer_secret。它们需要被设置为你的 Twitter 应用详细设置页面中列出的值。同样，你还会注意到我们声明了两个处理程序：TwitterHandler 和 LogoutHandler。让我们立刻看看这两个类吧。

TwitterHandler 类包含我们应用逻辑的主要部分。有两件事情需要立刻引起我们的注意，其一是这个类继承自能给我们提供 Twitter 功能的 tornado.auth.TwitterMixin 类，其二是 get 方法使用了我们在[第五章](http://dockerpool.com/static/books/introduction_to_tornado_cn/ch5.html)中讨论的@tornado.web.asynchronous 装饰器。现在让我们看看第一个异步调用：

```py
if self.get_argument('oauth_token', None):
    self.get_authenticated_user(self.async_callback(self._twitter_on_auth))
    return

```

当一个用户请求我们应用的根目录时，我们首先检查请求是否包括一个 oauth_token 查询字符串参数。如果有，我们把这个请求看作是一个来自 Twitter 验证过程的回调。

然后，我们使用 auth 模块的 get_authenticated 方法把给我们的临时令牌换为用户的访问令牌。这个方法期待一个回调函数作为参数，在这里是 self._teitter_on_auth 方法。当到 Twitter 的 API 请求返回时，执行回调函数，我们在代码更靠下的地方对其进行了定义。

如果 oauth_token 参数没有被发现，我们继续测试是否之前已经看到过这个特定用户了。

```py
elif oAuthToken and oAuthSecret:
    accessToken = {
        'key': oAuthToken,
        'secret': oAuthSecret
    }
    self.twitter_request('/users/show',
        access_token=accessToken,
        user_id=userID,
        callback=self.async_callback(self._twitter_on_user)
    )
    return

```

这段代码片段寻找我们应用在 Twitter 给定一个合法用户时设置的 access_key 和 access_secret cookies。如何这个值被设置了，我们就用 key 和 secret 组装访问令牌，然后使用 self.twitter_request 方法来向 Twitter API 的/users/show 发出请求。在这里，你会再一次看到异步回调函数，这次是我们稍后将要定义的 self._twitter_on_user 方法。

twitter_quest 方法期待一个路径地址作为它的第一个参数，另外还有一些可选的关键字参数，如 access_token、post_args 和 callback。access_token 参数应该是一个字典，包括用户 OAuth 访问令牌的 key 键，和用户 OAuth secret 的 secret 键。

如果 API 调用使用了 POST 方法，请求参数需要绑定一个传递 post_args 参数的字典。查询字符串参数在方法调用时只需指定为一个额外的关键字参数。在/users/show API 调用时，我们使用了 HTTP GET 请求，所以这里不需要 post_args 参数，而所需的 user_id API 参数被作为关键字参数传递进来。

如果上面我们讨论的情况都没有发生，这说明用户是首次访问我们的应用（或者已经注销或删除了 cookies），此时我们想将其重定向到 Twitter 的验证页面。调用 self.authorize_redirect()来完成这项工作。

```py
def _twitter_on_auth(self, user):
    if not user:
        self.clear_all_cookies()
        raise tornado.web.HTTPError(500, 'Twitter authentication failed')

    self.set_secure_cookie('user_id', str(user['id']))
    self.set_secure_cookie('oauth_token', user['access_token']['key'])
    self.set_secure_cookie('oauth_secret', user['access_token']['secret'])

    self.redirect('/')

```

我们的 Twitter 请求的回调方法非常的直接。_twitter_on_auth 使用一个 user 参数进行调用，这个参数是已授权用户的用户数据字典。我们的方法实现只需要验证我们接收到的用户是否合法，并设置应有的 cookies。一旦 cookies 被设置好，我们将用户重定向到根目录，即我们之前谈论的发起请求到/users/show API 方法。

```py
def _twitter_on_user(self, user):
    if not user:
        self.clear_all_cookies()
        raise tornado.web.HTTPError(500, "Couldn't retrieve user information")

    self.render('home.html', user=user)

```

_twitter_on_user 方法是我们在 twitter_request 方法中指定调用的回调函数。当 Twitter 响应用户的个人信息时，我们的回调函数使用响应的数据渲染 home.html 模板。这个模板展示了用户的个人图像、用户名、详细信息、一些关注和粉丝的统计信息以及用户最新的状态更新。

LogoutHandler 方法只是清除了我们为应用用户存储的 cookies。它渲染了 logout.html 模板，来给用户提供反馈，并跳转到 Twitter 验证页面允许其重新登录。就是这些！

我们刚才看到的 Twitter 应用只是为一个授权用户展示了用户信息，但它同时也说明了 Tornado 的 auth 模块是如何使开发社交应用更简单的。创建一个在 Twitter 上发表状态的应用作为一个练习留给读者。

## 7.3 示例：Facebook 认证和 Graph API

Facebook 的这个例子在结构上和刚才看到的 Twitter 的例子非常相似。Facebook 有两种不同的 API 标准，原始的 REST API 和 Facebook Graph API。目前两种 API 都被支持，但 Graph API 被推荐作为开发新 Facebook 应用的方式。Tornado 在 auth 模块中支持这两种 API，但在这个例子中我们将关注 Graph API。

为了开始这个例子，你需要登录到 Facebook 的[开发者网站](https://developers.facebook.com/)，并创建一个新的应用。你将需要填写应用的名称，并证明你不是一个机器人。为了从你自己的域名中验证用户，你还需要指定你应用的域名。然后点击"Select how your app integrates with Facbook"下的"Website"。同时你需要输入你网站的 URL。要获得完整的创建 Facebook 应用的手册，可以从[`developers.facebook.com/docs/guides/web/`](https://developers.facebook.com/docs/guides/web/)开始。

你的应用建立好之后，你将使用基本设置页面的应用 ID 和 secret 来连接 Facebook Graph API。

回想一下上一节的提到的单一登录工作流程，它将引导用户到 Facebook 平台验证应用，Facebook 将使用一个 HTTP 重定向将一个带有验证码的用户返回给你的服务器。一旦你接收到含有这个认证码的请求，你必须请求用于标识 API 请求用户身份的验证令牌。

这个例子将渲染用户的时间轴，并允许用户通过我们的接口更新她的 Facebook 状态。让我们看下代码清单 7-4。

代码清单 7-4 Facebook 验证：facebook.py

```py
import tornado.web
import tornado.httpserver
import tornado.auth
import tornado.ioloop
import tornado.options
from datetime import datetime

class FeedHandler(tornado.web.RequestHandler, tornado.auth.FacebookGraphMixin):
    @tornado.web.asynchronous
    def get(self):
        accessToken = self.get_secure_cookie('access_token')
        if not accessToken:
            self.redirect('/auth/login')
            return

        self.facebook_request(
            "/me/feed",
            access_token=accessToken,
            callback=self.async_callback(self._on_facebook_user_feed))

    def _on_facebook_user_feed(self, response):
        name = self.get_secure_cookie('user_name')
        self.render('home.html', feed=response['data'] if response else [], name=name)

    @tornado.web.asynchronous
    def post(self):
        accessToken = self.get_secure_cookie('access_token')
        if not accessToken:
            self.redirect('/auth/login')

        userInput = self.get_argument('message')

        self.facebook_request(
            "/me/feed",
            post_args={'message': userInput},
            access_token=accessToken,
            callback=self.async_callback(self._on_facebook_post_status))

    def _on_facebook_post_status(self, response):
        self.redirect('/')

class LoginHandler(tornado.web.RequestHandler, tornado.auth.FacebookGraphMixin):
    @tornado.web.asynchronous
    def get(self):
        userID = self.get_secure_cookie('user_id')

        if self.get_argument('code', None):
            self.get_authenticated_user(
                redirect_uri='http://example.com/auth/login',
                client_id=self.settings['facebook_api_key'],
                client_secret=self.settings['facebook_secret'],
                code=self.get_argument('code'),
                callback=self.async_callback(self._on_facebook_login))
            return
        elif self.get_secure_cookie('access_token'):
            self.redirect('/')
            return

        self.authorize_redirect(
            redirect_uri='http://example.com/auth/login',
            client_id=self.settings['facebook_api_key'],
            extra_params={'scope': 'read_stream,publish_stream'}
        )

    def _on_facebook_login(self, user):
        if not user:
            self.clear_all_cookies()
            raise tornado.web.HTTPError(500, 'Facebook authentication failed')

        self.set_secure_cookie('user_id', str(user['id']))
        self.set_secure_cookie('user_name', str(user['name']))
        self.set_secure_cookie('access_token', str(user['access_token']))
        self.redirect('/')

class LogoutHandler(tornado.web.RequestHandler):
    def get(self):
        self.clear_all_cookies()
        self.render('logout.html')

class FeedListItem(tornado.web.UIModule):
    def render(self, statusItem):
        dateFormatter = lambda x: datetime.
strptime(x,'%Y-%m-%dT%H:%M:%S+0000').strftime('%c')
        return self.render_string('entry.html', item=statusItem, format=dateFormatter)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/', FeedHandler),
            (r'/auth/login', LoginHandler),
                (r'/auth/logout', LogoutHandler)
            ]

            settings = {
                'facebook_api_key': '2040 ... 8759',
                'facebook_secret': 'eae0 ... 2f08',
                'cookie_secret': 'NTliOTY5NzJkYTVlMTU0OTAwMTdlNjgzMTA5M2U3OGQ5NDIxZmU3Mg==',
                'template_path': 'templates',
                'ui_modules': {'FeedListItem': FeedListItem}
            }

            tornado.web.Application.__init__(self, handlers, **settings)

    if __name__ == '__main__':
        tornado.options.parse_command_line()

        app = Application()
        server = tornado.httpserver.HTTPServer(app)
        server.listen(8000)
        tornado.ioloop.IOLoop.instance().start()

```

我们将按照访客与应用交互的顺序来讲解这些处理。当请求根 URL 时，FeedHandler 将寻找 access_token cookie。如果这个 cookie 不存在，用户会被重定向到/auth/login URL。

登录页面使用了 authorize_redirect 方法来讲用户重定向到 Facebook 的验证对话框，如果需要的话，用户在这里登录 Facebook，审查应用程序请求的权限，并批准应用。在点击"Approve"之后，她将被跳转回应用在 authorize_redirect 调用中 redirect_uri 指定的 URL。

当从 Facebook 验证页面返回后，到/auth/login 的请求将包括一个 code 参数作为查询字符串参数。这个码是一个用于换取永久凭证的临时令牌。如果发现了 code 参数，应用将发出一个 Facebook Graph API 请求来取得认证的用户，并存储她的用户 ID、全名和访问令牌，以便在应用发起 Graph API 调用时标识该用户。

存储了这些值之后，用户被重定向到根 URL。用户这次回到根页面时，将取得最新 Facebook 消息列表。应用查看 access_cookie 是否被设置，并使用 facebook_request 方法向 Graph API 请求用户订阅。我们把 OAuth 令牌传递给 facebook_request 方法，此外，这个方法还需要一个回调函数参数--在代码清单 7-4 中，它是 _on_facebook_user_feed 方法。

代码清单 7-5 Facebook 验证：home.html

```py
<html>
    <head>
        <title>{{ name }} on Facebook</title>
    </head>

    <body>
        <div>
            <a href="/auth/logout">Sign out</a>
            <h1>{{ name }}</h1>
        </div>
        <div>
            <form action="/facebook/" method="POST">
                <textarea rows="3" cols="50" name="message"></textarea>
                <input type="submit" value="Update Status" />
            </form>
        </div>
        <hr />
        {% for item in feed %}
            {% module FeedListItem(item) %}
        {% end %}
    </body>
</html>

```

当包含来自 Facebook 的用户订阅消息的响应的回调函数被调用时，应用渲染 home.html 模板，其中使用了 FeedListItem 这个 UI 模块来渲染列表中的每个条目。在模板开始处，我们渲染了一个表单，可以用 message 参数 post 到我们服务器的/resource。应用发送这个调用给 Graph API 来发表一个更新。

为了发表更新，我们再次使用了 facebook_request 方法。这次，除了 access_token 参数之外，我们还包括了一个 post_args 参数，这个参数是一个成为 Graph 请求 post 主体的参数字典。当调用成功时，我们将用户重定向回首页，并请求更新后的时间轴。

正如你所看到的，Tornado 的 auth 模块提供的 Facebook 验证类包括很多构建 Facebook 应用时非常有用的功能。这不仅在原型设计中是一笔巨大的财富，同时也非常适合是生产中的应用。