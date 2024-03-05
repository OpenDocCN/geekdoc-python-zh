# 缓存外部 API 请求

> 原文：<https://realpython.com/caching-external-api-requests/>

有没有发现自己对一个[外部 API](https://realpython.com/api-integration-in-python/) 发出*完全相同的*请求，使用*完全相同的*参数并返回*完全相同的*结果？如果是这样，那么您应该缓存这个请求来限制 HTTP 请求的数量，以帮助提高性能。

让我们看一个使用[请求](https://realpython.com/python-requests/)包的例子。

## Github API

从 [Github repo](https://github.com/realpython/flask-single-page-app/tree/part5) 中抓取代码(或者下载 [zip](https://github.com/realpython/flask-single-page-app/releases/tag/part5) )。基本上，我们一遍又一遍地搜索 Github API，根据位置和编程语言寻找相似的开发者:

```py
url = "https://api.github.com/search/users?q=location:{0}+language:{1}".format(first, second)
response_dict = requests.get(url).json()
```

现在，在初始搜索之后，如果用户再次搜索(例如，不改变参数)，应用程序将执行完全相同的搜索，一次又一次地点击 Github API。由于这是一个昂贵的过程，它减慢了我们的最终用户的应用程序。此外，通过像这样打几个电话，我们可以很快用完我们的速率限制。

幸运的是，有一个简单的解决方法。

[*Remove ads*](/account/join/)

## 请求-缓存

为了实现缓存，我们可以使用一个名为 [Requests-cache](http://requests-cache.readthedocs.org/en/latest/index.html) 的简单包，它是一个“用于[请求](http://docs.python-requests.org/en/latest/)的透明持久缓存”。

> 请记住，您可以将这个包用于任何 Python 框架，而不仅仅是 Flask 或脚本，只要您将它与 requests 包结合使用。

首先安装软件包:

```py
$ pip install --upgrade requests-cache
```

然后将导入添加到 *app.py* 以及`install_cache()`方法中:

```py
requests_cache.install_cache(cache_name='github_cache', backend='sqlite', expire_after=180)
```

现在无论何时使用`requests`，响应都会被自动缓存。此外，您可以看到我们正在定义几个选项。注意`expire_after`选项，它被设置为 180 秒。由于 Github API 经常更新，我们希望确保交付最新的结果。因此，在初始缓存发生 180 秒后，请求将重新触发并缓存一组新的结果，提供更新的结果。

更多选项，请查看[官方文档](http://requests-cache.readthedocs.org/en/latest/api.html#requests_cache.core.install_cache)。

所以您的 *app.py* 文件现在应该是这样的:

```py
import requests
import requests_cache

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

requests_cache.install_cache('github_cache', backend='sqlite', expire_after=180)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        first = request.form.get('first')
        second = request.form.get('second')
        url = "https://api.github.com/search/users?q=location:{0}+language:{1}".format(first, second)
        response_dict = requests.get(url).json()
        return jsonify(response_dict)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

## 测试！

启动应用程序，搜索开发者。在“app”目录中，应该创建一个名为 *github_cache.sqlite* 的 [SQLite 数据库](https://realpython.com/python-sqlite-sqlalchemy/)。现在，如果您继续使用相同的位置和编程语言进行搜索，`requests`实际上不会进行调用。相反，它将使用来自 SQLite 数据库的缓存响应。

让我们确保缓存确实过期了。像这样更新`home()`视图功能:

```py
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        first = request.form.get('first')
        second = request.form.get('second')
        url = "https://api.github.com/search/users?q=location:{0}+language:{1}".format(first, second)
        now = time.ctime(int(time.time()))
        response = requests.get(url)
        print "Time: {0} / Used Cache: {1}".format(now, response.from_cache)
        return jsonify(response.json())
    return render_template('index.html')
```

因此，这里我们只是使用`from_cache`属性来查看响应是否来自缓存。让我们来测试一下。尝试新的搜索。然后打开你的终端:

```py
Time: Fri Nov 28 13:34:25 2014 / Used Cache: False
```

所以你可以看到我们在 13:34:25 向 Github API 发出了初始请求，由于`False`被输出到屏幕上，所以没有使用缓存。再次尝试搜索。

```py
Time: Fri Nov 28 13:35:28 2014 / Used Cache: True
```

现在您可以看到使用了缓存。多试几次。

```py
Time: Fri Nov 28 13:36:10 2014 / Used Cache: True
Time: Fri Nov 28 13:37:59 2014 / Used Cache: False
Time: Fri Nov 28 13:39:09 2014 / Used Cache: True
```

所以您可以看到缓存过期了，我们在 13:37:59 进行了一个新的 API 调用。之后就用缓存了。简单吧？

当您更改请求中的参数时会发生什么？试试看。输入新的位置和编程语言。这里发生了什么？因为参数改变了，Requests-cache 把它当作不同的请求，不使用缓存。

[*Remove ads*](/account/join/)

## 平衡-同花顺与性能

同样，在上面的示例中，我们在 180 秒后终止缓存(通常称为刷新),以便向最终用户交付最新的数据。想一想。

真的有必要那么定时冲吗？大概不会。在这个应用程序中，我们可以将这个时间更改为 5 分钟或 10 分钟，因为如果我们偶尔错过一些添加到 API 中的新用户，这并不是什么大问题。

也就是说，当数据对时间敏感并且对应用程序的核心功能至关重要时，您确实需要密切关注刷新。

例如，如果您从一个每分钟更新几次的 API(如[地震活动 API](http://www.programmableweb.com/api/seismic-data-portal) )中提取数据，并且您的最终用户必须拥有最新的数据，那么您可能希望每隔 30 或 60 秒左右使其过期。

平衡刷新频率和调用时间也很重要。如果您的 API 调用相当昂贵——可能需要一到五秒——那么您希望增加刷新之间的时间来提高性能。

## 结论

缓存是一个强大的工具。在这种情况下，我们通过限制外部 HTTP 请求的数量来提高应用程序的性能。我们从实际的 HTTP 请求本身中去掉了延迟。

在很多情况下，你不仅仅是在提出请求。您还必须处理请求，这可能涉及访问数据库、执行某种过滤等。因此，缓存也可以减少请求处理的延迟。

想要本教程的代码吗？抓住它[这里](https://github.com/realpython/flask-single-page-app/tree/part6)。干杯！**