# 烧瓶示例–整合烧瓶和角形烧瓶

> 原文：<https://realpython.com/flask-by-example-integrating-flask-and-angularjs/>

欢迎回来。使用 Redis 任务队列设置，让我们使用 [AngularJS](https://angularjs.org/) 来轮询后端以查看任务是否完成，然后在数据可用时更新 DOM。

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

*更新:*

*   02/29/2020:升级到 Python 版本 [3.8.1](https://www.python.org/downloads/release/python-381/) 。
*   03/22/2016:升级到 Python 版本 [3.5.1](https://www.python.org/downloads/release/python-351/) 和 Angular 版本 [1.4.9](https://code.angularjs.org/1.4.9/docs/api) 。
*   2015 年 2 月 22 日:添加了 Python 3 支持。

* * *

记住:这是我们正在构建的——一个 Flask 应用程序，它根据来自给定 URL 的文本计算词频对。

1.  第一部分:建立一个本地开发环境，然后在 Heroku 上部署一个试运行环境和一个生产环境。
2.  第二部分:使用 SQLAlchemy 和 Alembic 建立一个 PostgreSQL 数据库来处理迁移。
3.  [第三部分](/flask-by-example-part-3-text-processing-with-requests-beautifulsoup-nltk/):添加后端逻辑，使用 requests、BeautifulSoup 和 Natural Language Toolkit (NLTK)库从网页中抓取并处理字数。
4.  第四部分:实现一个 Redis 任务队列来处理文本处理。
5.  **第五部分:在前端设置 Angular 来持续轮询后端，看请求是否处理完毕。(*当前* )**
6.  第六部分:推送到 Heroku 上的临时服务器——建立 Redis 并详细说明如何在一个 Dyno 上运行两个进程(web 和 worker)。
7.  [第七部分](/flask-by-example-updating-the-ui/):更新前端，使其更加人性化。
8.  [第八部分](/flask-by-example-custom-angular-directive-with-d3/):使用 JavaScript 和 D3 创建一个自定义角度指令来显示频率分布图。

<mark>需要代码吗？从[回购](https://github.com/realpython/flask-by-example/releases)中抢过来。</mark>

**新来的有棱角？**回顾以下教程: [AngularJS by Example:搭建比特币投资计算器](https://github.com/mjhea0/thinkful-angular)

准备好了吗？让我们先来看看我们应用程序的当前状态…

## 当前功能

首先，在一个终端窗口中启动 Redis:

```py
$ redis-server
```

在另一个窗口中，导航到您的项目目录，然后运行 worker:

```py
$ cd flask-by-example
$ python worker.py
20:38:04 RQ worker started, version 0.5.6
20:38:04
20:38:04 *** Listening on default...
```

最后，打开第三个终端窗口，导航到您的项目目录，启动主应用程序:

```py
$ cd flask-by-example
$ python manage.py runserver
```

打开 [http://localhost:5000/](http://localhost:5000/) 用网址 https://realpython.com 测试。在终端中，作业 id 应该已经输出。获取 id 并导航到以下 url:

[http://localhost:5000/results/add _ the _ job _ id _ here](http://localhost:5000/results/add_the_job_id_here)

您应该会在浏览器中看到类似的 JSON 响应:

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

现在我们准备添加角度。

[*Remove ads*](/account/join/)

## 更新*index.html*T2】

给*index.html*添加角度:

```py
<script src="//ajax.googleapis.com/ajax/libs/angularjs/1.4.9/angular.min.js"></script>
```

将以下[指令](https://code.angularjs.org/1.4.9/docs/guide/directive)添加到【index.html】的*中*:

1.  :`<html ng-app="WordcountApp">`
2.  *[ng-控制器](https://code.angularjs.org/1.4.9/docs/api/ng/directive/ngController)* : `<body ng-controller="WordcountController">`
3.  *[ng-提交](https://code.angularjs.org/1.4.9/docs/api/ng/directive/ngSubmit)* : `<form role="form" ng-submit="getResults()">`

因此，我们引导 Angular——它告诉 Angular 将这个 HTML 文档视为 Angular 应用程序——添加了一个控制器，然后添加了一个名为`getResults()`的函数——它在表单提交时被触发。

## 创建角度模块

创建一个“静态”目录，然后向该目录添加一个名为 *main.js* 的文件。务必将该要求添加到 index.html 文件的*中；*

```py
<script src="{{ url_for('static', filename='main.js') }}"></script>
```

让我们从这个基本代码开始:

```py
(function  ()  { 'use strict'; angular.module('WordcountApp',  []) .controller('WordcountController',  ['$scope',  '$log', function($scope,  $log)  { $scope.getResults  =  function()  { $log.log("test"); }; } ]); }());
```

这里，当提交表单时，`getResults()`被调用，它只是将文本“test”记录到浏览器中的 JavaScript 控制台。一定要测试出来。

### 依赖注入和$scope

在上面的例子中，我们利用[依赖注入](https://code.angularjs.org/1.4.9/docs/guide/di)来“注入”对象`$scope`和服务`$log`。停在这里。理解`$scope`非常重要。*从角度[文档](https://code.angularjs.org/1.4.9/docs/guide/scope)开始，如果您还没有浏览过[角度介绍](https://github.com/mjhea0/thinkful-angular)教程，请务必浏览一遍。*

听起来可能很复杂，但它确实只是提供了视图和控制器之间的一种交流方式。两者都可以访问它，当你在一个中改变一个附属于`$scope`的[变量](https://realpython.com/python-variables/)时，这个变量会在另一个中自动更新([数据绑定](https://code.angularjs.org/1.4.9/docs/guide/databinding))。依赖注入也是如此:它比听起来要简单得多。可以把它看作是获得各种服务的一点魔法。因此，通过注入服务，我们现在可以在控制器中使用它。

回到我们的应用程序…

如果您测试一下，您会看到表单提交不再向后端发送 POST 请求。这正是我们想要的。相反，我们将使用 Angular `$http`服务来异步处理这个请求:

```py
.controller('WordcountController',  ['$scope',  '$log',  '$http', function($scope,  $log,  $http)  { $scope.getResults  =  function()  { $log.log("test"); // get the URL from the input var  userInput  =  $scope.url; // fire the API request $http.post('/start',  {"url":  userInput}). success(function(results)  { $log.log(results); }). error(function(error)  { $log.log(error); }); }; } ]);
```

另外，更新*index.html*中的`input`元素:

```py
<input type="text" ng-model="url" name="url" class="form-control" id="url-box" placeholder="Enter URL..." style="max-width: 300px;">
```

我们注入了`$http`服务，从输入框中抓取 URL(通过`ng-model="url"`)，然后向后端发出 POST 请求。`success`和`error`回调处理响应。在 200 响应的情况下，它将由`success`处理程序处理，该处理程序反过来将响应记录到控制台。

在测试之前，让我们[重构](https://realpython.com/python-refactoring/)后端，因为`/start`端点目前并不存在。

[*Remove ads*](/account/join/)

## 重构*app . py*T2】

从索引视图函数中重构出 Redis 作业创建，然后将其添加到一个名为`get_counts()`的新视图函数中:

```py
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def get_counts():
    # this import solves a rq bug which currently exists
    from app import count_and_save_words

    # get url
    data = json.loads(request.data.decode())
    url = data["url"]
    if not url[:8].startswith(('https://', 'http://')):
        url = 'http://' + url
    # start job
    job = q.enqueue_call(
        func=count_and_save_words, args=(url,), result_ttl=5000
    )
    # return created job id
    return job.get_id()
```

确保在顶部也添加以下导入:

```py
import json
```

这些变化应该很简单。

现在我们测试。刷新你的浏览器，提交一个新的网址。您应该在 JavaScript 控制台中看到作业 id。完美。现在 Angular 有了作业 id，我们可以添加轮询功能。

## 基本轮询

通过向控制器添加以下代码来更新 *main.js* :

```py
function  getWordCount(jobID)  { var  timeout  =  ""; var  poller  =  function()  { // fire another request $http.get('/results/'+jobID). success(function(data,  status,  headers,  config)  { if(status  ===  202)  { $log.log(data,  status); }  else  if  (status  ===  200){ $log.log(data); $timeout.cancel(timeout); return  false; } // continue to call the poller() function every 2 seconds // until the timeout is cancelled timeout  =  $timeout(poller,  2000); }); }; poller(); }
```

然后在 POST 请求中更新成功处理程序:

```py
$http.post('/start',  {"url":  userInput}). success(function(results)  { $log.log(results); getWordCount(results); }). error(function(error)  { $log.log(error); });
```

确保将`$timeout`服务也注入到控制器中。

这里发生了什么事？

1.  成功的 HTTP 请求会触发`getWordCount()`函数。
2.  在`poller()`函数中，我们称之为`/results/job_id`端点。
3.  使用`$timeout`服务，该函数继续每隔 2 秒触发一次，直到超时被取消，此时返回 200 响应和字数。*查看有棱角的[文档](https://code.angularjs.org/1.4.9/docs/api/ng/service/$timeout)，获得关于`$timeout`服务如何工作的精彩描述。*

测试时，请确保打开 JavaScript 控制台。您应该会看到类似这样的内容:

```py
Nay! 202
Nay! 202
Nay! 202
Nay! 202
Nay! 202
Nay! 202
(10) [Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2), Array(2)]
```

所以，在上面的例子中，`poller()`函数被调用了七次。前六个调用返回 202，而最后一个调用返回 200 和字数统计数组。

完美。

现在我们需要将字数添加到 DOM 中。

[*Remove ads*](/account/join/)

## 更新 DOM

更新*index.html*:

```py
<div class="container">
  <div class="row">
    <div class="col-sm-5 col-sm-offset-1">
      <h1>Wordcount 3000</h1>
      <br>
      <form role="form" ng-submit="getResults()">
        <div class="form-group">
          <input type="text" name="url" class="form-control" id="url-box" placeholder="Enter URL..." style="max-width: 300px;" ng-model="url" required>
        </div>
        
      </form>
    </div>
    <div class="col-sm-5 col-sm-offset-1">
      <h2>Frequencies</h2>
      <br>
      {% raw %}
      <div id="results">

          {{wordcounts}}

      </div>
      {% endraw %}
    </div>
  </div>
</div>
```

我们改变了什么？

1.  `input`标签现在有了一个`required`属性，表明在提交表单之前必须填写输入框。
2.  告别 [Jinja2 模板](https://realpython.com/primer-on-jinja-templating/)标签。Jinja2 从服务器端提供服务，由于轮询完全在客户端处理，我们需要使用 Angular 标记。也就是说，由于 Jinja2 和 Angular 模板标签都使用了双花括号`{{}}`，我们必须使用`{% raw %}`和`{% endraw %}`来转义 Jinja2 标签。*如果你需要使用多个角度标签，最好用`$interpolateProvider`改变 AngularJS 使用的模板标签。更多信息，请查看角度[文档](https://code.angularjs.org/1.4.9/docs/api/ng/provider/$interpolateProvider)。*

其次，更新`poller()`函数中的成功处理程序:

```py
success(function(data,  status,  headers,  config)  { if(status  ===  202)  { $log.log(data,  status); }  else  if  (status  ===  200){ $log.log(data); $scope.wordcounts  =  data; $timeout.cancel(timeout); return  false; } // continue to call the poller() function every 2 seconds // until the timeout is cancelled timeout  =  $timeout(poller,  2000); });
```

这里，我们将结果附加到了`$scope`对象上，这样它在视图中就可用了。

测试一下。如果一切顺利，您应该在 DOM 上看到该对象。不太漂亮，但这是一个简单的 Bootstrap 修复方法，在带有`id=results`的 div 下添加以下代码，并从上面的代码中删除包装结果 div 的`{% raw %}`和`{% endraw %}`标签:

```py
<div id="results">
  <table class="table table-striped">
    <thead>
      <tr>
        <th>Word</th>
        <th>Count</th>
      </tr>
    </thead>
    <tbody>
      {% raw %}
      <tr ng-repeat="element in wordcounts">

        <td>{{ element[0] }}</td>
        <td>{{ element[1] }}</td>

      </tr>
    {% endraw %}
    </tbody>
  </table>
</div>
```

## 结论和后续步骤

在继续使用 [D3](https://d3js.org/) 制作图表之前，我们还需要:

1.  添加一个加载微调器:也称为[跳动器](http://en.wikipedia.org/wiki/Throbber)，它会一直显示，直到任务完成，这样最终用户就知道有事情发生了。
2.  重构角度控制器:现在控制器中发生了太多的事情(逻辑)。我们需要将大部分功能转移到一个服务中。我们将讨论*为什么*和*如何*。
3.  更新登台:我们需要更新 Heroku 上的登台环境——添加代码变更、我们的工人和 Redis。

下次[时间](/updating-the-staging-environment/)见！

另一个加深你的烧瓶技能的推荐资源是这个视频系列:

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。***