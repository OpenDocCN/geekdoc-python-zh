# Flask 扩展系列(六)–缓存

如果同一个请求会被多次调用，每次调用都会消耗很多资源，并且每次返回的内容都相同，大家第一个反应就是该使用缓存了。的确对于大规模互联网应用，缓存是必不可少的，一个好的缓存设计可以使得应用的性能几何级数地上升。本篇我们将阐述如何缓存 Flask 的请求，并同时介绍一个缓存扩展，Flask-Cache。

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

### 自定义缓存装饰器

在使用 Flask-Cache 扩展实现缓存功能之前，我们先来自己写个视图缓存装饰器，方便我们来理解视图缓存的实现。首先，我们要有一个缓存，Werkzeug 框架中的提供了一个简单的缓存对象[SimpleCache](http://werkzeug.pocoo.org/docs/0.11/contrib/cache/#werkzeug.contrib.cache.SimpleCache)，它是将缓存项存放在 Python 解释器的内存中，我们可以用下面的代码获取 SimpleCache 的缓存对象：

```py
from werkzeug.contrib.cache import SimpleCache
cache = SimpleCache()

```

如果你要使用第三方的缓存服务器，比如 Memcached，Werkzeug 框架也提供了它的 wrapper：

```py
from werkzeug.contrib.cache import MemcachedCache
cache = MemcachedCache(['127.0.0.1:11211'])

```

此后你就可以使用 cache 对象的”set(key, value, timeout)”和”get(key)”方法来存取缓存项了。注意”set()”方法的第三个参数”timeout”是缓存过期时间，默认为 0，也就是永不过期。

接下来，我们就开始写个缓存装饰器用来装饰视图函数：

```py
def cached(timeout=5 * 60, key='view_%s'):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = key % request.path
            value = cache.get(cache_key)
            if value is None:
                value = f(*args, **kwargs)
                cache.set(cache_key, value, timeout=timeout)
            return value
        return decorated_function
    return decorator

```

这段装饰器代码还是很好理解吧，如果大家对装饰器不熟悉，可以看下这篇文章。装饰器的两个参数分别是缓存的过期时间，默认是 5 分钟；缓存项键值的前缀，默认是”view_”。然后我们写个视图，并使用此装饰器：

```py
@app.route('/hello')
@app.route('/hello/<name>')
@cached()
def hello(name=None):
    print 'view hello called'
    return render_template('hello.html', name=name)

```

我们试下访问这个视图，对于同样的 URL 地址，第一次访问时，控制台上会有”view called”输出，第二次就不会了。如果过 5 分钟后访问，”view called”才会再次输出。

### 安装和启用 Flask-Cache

了解了缓存装饰器的内部实现，我们就可以开始介绍 Flask 的缓存扩展，Flask-Cache。首先使用 pip 将其安装上：

```py
$ pip install Flask-Cache
```

然后创建一个 Flask-Cache 的实例：

```py
from flask import Flask
from flask.ext.cache import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

```

上例中，我们使用了’simple’类型缓存，其内部实现就是 Werkzeug 中的 SimpleCache。我们也可以使用第三方的缓存服务器，比如 Redis，代码如下：

```py
cache = Cache(app, config={'CACHE_TYPE': 'redis',          # Use Redis
                           'CACHE_REDIS_HOST': 'abc.com',  # Host, default 'localhost'
                           'CACHE_REDIS_PORT': 6379,       # Port, default 6379
                           'CACHE_REDIS_PASSWORD': '111',  # Password
                           'CACHE_REDIS_DB': 2}            # DB, default 0

```

其内部实现是”werkzeug.contrib.cache”包下的”RedisCache”，所以说 Flask-Cache 就是基于 Werkzeug 框架的 Cache 库实现的。

### 应用中使用缓存

同自定义缓存装饰器一样，我们可以用 cache 对象的”cached()”方法来装饰视图函数，以达到缓存视图的目的：

```py
@app.route('/hello')
@app.route('/hello/<name>')
@cache.cached(timeout=300, key_prefix='view_%s', unless=None)
def hello(name=None):
    print 'view hello called'
    return render_template('hello.html', name=name)

```

“cache.cached()”装饰器有三个参数：

1.  timeout：过期时间，默认为 None，即永不过期
2.  key_prefix：缓存项键值的前缀，默认为”view/%s”
3.  unless：回调函数，当其返回 True 时，缓存不起作用。默认为 None，即缓存有效

除了装饰视图函数，”cache.cached()”装饰器也可以用来装饰普通函数：

```py
@cache.cached(timeout=50, key_prefix='get_list')
def get_list():
    print 'method get_list called'
    return ['a','b','c','d','e']

@app.route('/list')
def list():
    return ', '. join(get_list())

```

我们访问”/list”地址时，第一次控制台上会有”method called”输出，第二次就不会了，说明缓存起效了。装饰普通函数时必须指定明确的”key_prefix”参数，因为它不像视图函数，可以使用请求路径”request.path”作为缓存项的键值。另外，如果函数带参数，对于不同的参数调用，都会使用同一缓存项，即返回结果一样。但大部分时候，对于不同的输入，输出结果是不一样的，那使用缓存岂不是有问题？是的，所以 Flask-Cache 还提供了另一个装饰器方法”cache.memoize()”，它与”cache.cached()”的区别就是它会将函数的参数也放在缓存项的键值中：

```py
@cache.memoize(timeout=50)
def create_list(num):
    print 'method create_list called'
    l = []
    for i in range(num):
        l.append(str(i))
    return l

@app.route('/list/<int:num>')
def list(num):
    return ', '.join(create_list(num))

```

我们再次访问”/list”地址，对于不同的参数，”method called”会一直在控制台上打印出，而对于相同的参数，第二次就不会打印了。所以对于带参数的函数，你要使用”cache.memoize()”装饰器，而对于不带参数的函数，它同”cache.cached()”基本上一样。”cache.memoize()”装饰器也有三个参数，”timeout”和”unless”参数同”cache.cached()”一样，就是第二个参数”make_name”比较特别。它是一个回调函数，传入的是被装饰的函数名，返回是一个字符串，会被作为缓存项键值的一部分，如果不设，就直接使用函数名。

### 删除缓存

对于普通缓存，你可以使用”delete()”方法来删除缓存项，而对于”memoize”缓存，你需要使用”delete_memoized”方法。如果想清除所有缓存，可以使用”clear()”方法。

```py
cache.delete('get_list')                     # 删除'get_list'缓存项
cache.delete_many('get_list', 'view_hello')  # 同时删除'get_list'和'view_hello'缓存项
cache.delete_memoized('create_list', 5)      # 删除调用'create_list'函数并且参数为 5 的缓存项
cache.clear()                                # 清理所有缓存

```

### Jinja2 模板中使用缓存

上面介绍的缓存功能都是在应用代码中使用，其实在 Jinja2 模板中，我们还可以使用”{% cache %}”语句来缓存模板代码块：

```py
{% cache 50, 'temp' %}
<p>This is under cache</p>
{% endcache %}

```

这样”{% cache %}”和”{% endcache %}”语句中所包括的内容就会被缓存起来。”{% cache %}”语句的第一个参数是”timeout”过期时间，默认为永不过期；第二个参数指定了缓存项的键值，如果不设，键值就是”模板文件路径”+”缓存块的第一行”。上例中，我们设了键值是”temp”，然后在代码中，我们可以这样获取缓存项实际的键值：

```py
from flask.ext.cache import make_template_fragment_key
key = make_template_fragment_key('temp')

```

打印出来看看，你会发现实际的键值其实是”_template_fragment_cache_temp”。如果你要删除该缓存项，记得要传入实际的键值，而不是模板上定义的’temp’。

#### 更多参考资料

[Werkzeug 的 Cache API 文档](http://werkzeug.pocoo.org/docs/0.11/contrib/cache/)
[Flask-Cache 的官方文档](http://pythonhosted.org/Flask-Cache/)
[Flask-Cache 的源码](https://github.com/thadeusb/flask-cache/)

本篇的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ext6.html)