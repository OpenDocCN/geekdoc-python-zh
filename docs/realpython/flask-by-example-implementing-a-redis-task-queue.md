# flask by Example–实现 Redis 任务队列

> 原文：<https://realpython.com/flask-by-example-implementing-a-redis-task-queue/>

教程的这一部分详细介绍了如何实现 Redis 任务队列来处理文本处理。

*更新:*

*   02/12/2020:升级到 Python 版本 [3.8.1](https://www.python.org/downloads/release/python-381/) 以及 Redis、Python Redis、RQ 的最新版本。详见下面的[。提到最新 RQ 版本的一个 bug，并提供解决方案。解决了 https 之前的 https bug。](#install-requirements)
*   03/22/2016:升级到 Python 版本 [3.5.1](https://www.python.org/downloads/release/python-351/) 以及 Redis、Python Redis、RQ 的最新版本。详见下面的[。](#install-requirements)
*   2015 年 2 月 22 日:添加了 Python 3 支持。

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

* * *

记住:这是我们正在构建的——一个 Flask 应用程序，它根据来自给定 URL 的文本计算词频对。

1.  第一部分:建立一个本地开发环境，然后在 Heroku 上部署一个试运行环境和一个生产环境。
2.  第二部分:使用 SQLAlchemy 和 Alembic 建立一个 PostgreSQL 数据库来处理迁移。
3.  [第三部分](/flask-by-example-part-3-text-processing-with-requests-beautifulsoup-nltk/):添加后端逻辑，使用 requests、BeautifulSoup 和[自然语言工具包(NLTK)](https://realpython.com/nltk-nlp-python/) 库从网页中抓取并处理字数。
4.  第四部分:实现一个 Redis 任务队列来处理文本处理。(*当前* )
5.  [第五部分](/flask-by-example-integrating-flask-and-angularjs/):在前端设置 Angular，持续轮询后端，看请求是否处理完毕。
6.  第六部分:推送到 Heroku 上的临时服务器——建立 Redis 并详细说明如何在一个 Dyno 上运行两个进程(web 和 worker)。
7.  [第七部分](/flask-by-example-updating-the-ui/):更新前端，使其更加人性化。
8.  [第八部分](/flask-by-example-custom-angular-directive-with-d3/):使用 JavaScript 和 D3 创建一个自定义角度指令来显示频率分布图。

<mark>需要代码吗？从[回购](https://github.com/realpython/flask-by-example/releases)中抢过来。</mark>

## 安装要求

使用的工具:

*   背〔t0〕5 . 0 . 7〔t1〕
*   Python Redis ( [3.4.1](https://pypi.python.org/pypi/redis/3.4.1)
*   RQ ( [1.2.2](https://pypi.python.org/pypi/rq/1.2.2) ) -一个用于创建任务队列的简单库

首先从官方网站或者自制软件`brew install redis`下载并安装 Redis。安装后，启动 Redis 服务器:

```py
$ redis-server
```

接下来，在新的终端窗口中安装 Python Redis 和 RQ:

```py
$ cd flask-by-example
$ python -m pip install redis==3.4.1 rq==1.2.2
$ python -m pip freeze > requirements.txt
```

[*Remove ads*](/account/join/)

## 设置工人

让我们首先创建一个工作进程来监听排队的任务。创建一个新文件 *worker.py* ，并添加以下代码:

```py
import os

import redis
from rq import Worker, Queue, Connection

listen = ['default']

redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
```

这里，我们监听了一个名为`default`的队列，并在`localhost:6379`上建立了一个到 Redis 服务器的连接。

在另一个终端窗口中启动它:

```py
$ cd flask-by-example
$ python worker.py
17:01:29 RQ worker started, version 0.5.6
17:01:29
17:01:29 *** Listening on default...
```

现在我们需要更新我们的 *app.py* 来发送任务到队列…

## 更新*app . py*T2】

将以下导入添加到 *app.py* :

```py
from rq import Queue
from rq.job import Job
from worker import conn
```

然后更新配置部分:

```py
app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

q = Queue(connection=conn)

from models import *
```

`q = Queue(connection=conn)`建立 Redis 连接，并基于该连接初始化队列。

将文本处理功能从我们的索引路径中移出，放到一个名为`count_and_save_words()`的新函数中。这个函数接受一个参数，一个 URL，当我们从我们的索引路径调用它时，我们将传递给它。

```py
def count_and_save_words(url):

    errors = []

    try:
        r = requests.get(url)
    except:
        errors.append(
            "Unable to get URL. Please make sure it's valid and try again."
        )
        return {"error": errors}

    # text processing
    raw = BeautifulSoup(r.text).get_text()
    nltk.data.path.append('./nltk_data/')  # set the path
    tokens = nltk.word_tokenize(raw)
    text = nltk.Text(tokens)

    # remove punctuation, count raw words
    nonPunct = re.compile('.*[A-Za-z].*')
    raw_words = [w for w in text if nonPunct.match(w)]
    raw_word_count = Counter(raw_words)

    # stop words
    no_stop_words = [w for w in raw_words if w.lower() not in stops]
    no_stop_words_count = Counter(no_stop_words)

    # save the results
    try:
        result = Result(
            url=url,
            result_all=raw_word_count,
            result_no_stop_words=no_stop_words_count
        )
        db.session.add(result)
        db.session.commit()
        return result.id
    except:
        errors.append("Unable to add item to database.")
        return {"error": errors}

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    if request.method == "POST":
        # this import solves a rq bug which currently exists
        from app import count_and_save_words

        # get url that the person has entered
        url = request.form['url']
        if not url[:8].startswith(('https://', 'http://')):
            url = 'http://' + url
        job = q.enqueue_call(
            func=count_and_save_words, args=(url,), result_ttl=5000
        )
        print(job.get_id())

    return render_template('index.html', results=results)
```

请注意以下代码:

```py
job = q.enqueue_call(
    func=count_and_save_words, args=(url,), result_ttl=5000
)
print(job.get_id())
```

> **注意:**我们需要在我们的函数`index`中导入`count_and_save_words`函数，因为 RQ 包目前有一个 bug，它在同一个模块中找不到函数。

这里我们使用了之前初始化的队列，并调用了`enqueue_call()`函数。这向队列中添加了一个新的作业，该作业使用 URL 作为参数运行`count_and_save_words()`函数。`result_ttl=5000`行参数告诉 RQ 保持作业结果多长时间，在本例中为-5000 秒。然后，我们将作业 id 输出到终端。需要此 id 来查看作业是否已完成处理。

让我们为那建立一条新的路线…

[*Remove ads*](/account/join/)

## 获取结果*

```py
@app.route("/results/<job_key>", methods=['GET'])
def get_results(job_key):

    job = Job.fetch(job_key, connection=conn)

    if job.is_finished:
        return str(job.result), 200
    else:
        return "Nay!", 202
```

让我们来测试一下。

启动服务器，导航到 [http://localhost:5000/](http://localhost:5000/) ，使用 URL[https://realpython.com](https://realpython.com)，从终端获取作业 id。然后在“/results/”端点中使用该 id——即[http://localhost:5000/results/ef 600206-3503-4b 87-a436-DDD 9438 f 2197](http://localhost:5000/results/ef600206-3503-4b87-a436-ddd9438f2197)。

只要在您检查状态之前经过的时间不到 5，000 秒，您就会看到一个 id 号，它是在我们将结果添加到数据库中时生成的:

```py
# save the results
try:
    from models import Result
    result = Result(
        url=url,
        result_all=raw_word_count,
        result_no_stop_words=no_stop_words_count
    )
    db.session.add(result)
    db.session.commit()
    return result.id
```

现在，让我们稍微重构一下路由，从 JSON 中的数据库返回实际结果:

```py
@app.route("/results/<job_key>", methods=['GET'])
def get_results(job_key):

    job = Job.fetch(job_key, connection=conn)

    if job.is_finished:
        result = Result.query.filter_by(id=job.result).first()
        results = sorted(
            result.result_no_stop_words.items(),
            key=operator.itemgetter(1),
            reverse=True
        )[:10]
        return jsonify(results)
    else:
        return "Nay!", 202
```

确保添加导入:

```py
from flask import jsonify
```

再次测试这个。如果一切顺利，您应该会在浏览器中看到类似的内容:

```py
[ [ "Python",  
  315 ],  
  [ "intermediate",  
  167 ],  
  [ "python",  
  161 ],  
  [ "basics",  
  118 ],  
  [ "web-dev",  
  108 ],  
  [ "data-science",  
  51 ],  
  [ "best-practices",  
  49 ],  
  [ "advanced",  
  45 ],  
  [ "django",  
  43 ],  
  [ "flask",  
  41 ] ]
```

## 下一步是什么？

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

在[第 5 部分](/flask-by-example-integrating-flask-and-angularjs/)中，我们将通过在混合中添加 Angular 来创建一个[轮询器](https://en.wikipedia.org/wiki/Polling_%28computer_science%29)，它将每五秒钟向`/results/<job_key>`端点发送一个请求，请求更新。一旦数据可用，我们将把它添加到 DOM 中。

干杯！

* * *

*这是创业公司[埃德蒙顿](http://startupedmonton.com/)的联合创始人卡姆·克林和 Real Python* 的人合作的作品**