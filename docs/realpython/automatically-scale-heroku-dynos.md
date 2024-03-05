# 自动缩放 Heroku Dynos

> 原文：<https://realpython.com/automatically-scale-heroku-dynos/>

这篇文章详细介绍了如何编写一个脚本来根据一天中的时间自动缩放 Heroku dynos。我们还将了解如何添加一个防故障装置，以便我们的应用程序在完全停机或负载过重时能够自动伸缩。

让我们考虑以下假设:

| 使用 | 时间 | Web Dynos |
| --- | --- | --- |
| 沉重的 | 早上 7 点到晚上 10 点 | three |
| 中等 | 晚上 10 点到凌晨 3 点 | Two |
| 低的 | 凌晨 3 点至 7 点 | one |

因此，我们需要在早上 7 点向外扩展，然后在晚上 10 点向内扩展，然后在凌晨 3 点再次向内扩展。重复一遍。为了简单起见，我们的大部分流量仅来自少数时区，这一点我们已经考虑在内了。我们还将基于 UTC，因为那是 Heroku 的默认时区。

> 如果这是针对您自己的应用程序，请确保在缩放之前给自己留有一些回旋的余地。你可能也想看看假期和周末。算算吧。算出你的成本节约。

就这样，让我们添加一些任务…

## AP scheduler〔t0〕

对于本教程，让我们使用[高级 Python 调度器](http://apscheduler.readthedocs.org/en/3.0/) (APScheduler)，因为它易于使用，并打算与 [Heroku 平台 API](https://devcenter.heroku.com/categories/platform-api) 一起与其他进程一起运行。

从安装 APScheduler 开始:

```py
$ pip install apscheduler==3.0.1
```

现在，创建一个名为 *autoscale.py* 的新文件，并添加以下代码:

```py
from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('interval', minutes=1)
def job():
    print 'This job is run every minute.'

sched.start()
```

是的，这只是每分钟运行一个任务。在继续之前，让我们测试一下，以确保它能正常工作。将该流程添加到您的 *Procfile* 中。假设您已经定义了一个 web 流程，该文件现在应该看起来像这样:

```py
web: gunicorn hello:app
clock: python autoscale.py
```

提交您的更改，然后将它们推送到 Heroku。

运行以下命令来扩展时钟进程:

```py
$ heroku ps:scale clock=1
```

然后打开 Heroku 日志查看运行中的流程:

```py
$ heroku logs --tail
2014-11-04T14:59:22.418496+00:00 heroku[api]: Scale to clock=1, web=1 by michael@realpython.com
2014-11-04T15:00:20.357505+00:00 heroku[router]: at=info method=GET path="/" host=autoscale.herokuapp.com request_id=7537ce4a-e802-4020-9b1b-10e754263957 fwd="54.160.152.14" dyno=web.1 connect=1ms service=3ms status=200 bytes=172
2014-11-04T15:00:27.620383+00:00 app[clock.1]: This job is run every minute.
2014-11-04T15:01:27.621151+00:00 app[clock.1]: This job is run every minute.
2014-11-04T15:02:27.620780+00:00 app[clock.1]: This job is run every minute.
2014-11-04T15:03:27.621276+00:00 app[clock.1]: This job is run every minute.
```

简单吧？

接下来，让我们将缩放任务添加到脚本中…

[*Remove ads*](/account/join/)

## 自动缩放

首先从 Heroku [账户](https://dashboard.heroku.com/account)页面获取 API 密匙，并将其添加到一个名为 *config.py* 的新文件中。除了密钥之外，还要输入您有兴趣监控的应用程序的名称和进程。

```py
APP = "<add your app name>"
KEY = "<add your API key>"
PROCESS = "web"
```

接下来，将以下函数添加到 *autoscale.py* :

```py
def scale(size):
    payload = {'quantity': size}
    json_payload = json.dumps(payload)
    url = "https://api.heroku.com/apps/" + APP + "/formation/" + PROCESS
    try:
        result = requests.patch(url, headers=HEADERS, data=json_payload)
    except:
        print "test!"
        return None
    if result.status_code == 200:
        return "Success!"
    else:
        return "Failure"
```

更新导入并添加以下配置:

```py
import requests
import base64
import json

from apscheduler.schedulers.blocking import BlockingScheduler

from config import APP, KEY, PROCESS

# Generate Base64 encoded API Key
BASEKEY = base64.b64encode(":" + KEY)
# Create headers for API call
HEADERS = {
    "Accept": "application/vnd.heroku+json; version=3",
    "Authorization": BASEKEY
}
```

在这里，我们通过将 API 键传递到头部来处理基本的授权，然后使用`requests`库，我们调用 API。关于这方面的更多信息，请查看 Heroku 官方文档。如果一切顺利，这将适当地扩展我们的应用程序。

想测试一下吗？像这样更新`job()`函数:

```py
@sched.scheduled_job('interval', minutes=1)
def job():
    print 'Scaling ...'
    print scale(0)
```

提交你的代码，然后推到 Heroku。现在，如果您运行`heroku logs --tail`，您应该会看到:

```py
$ heroku logs --tail
2014-11-04T20:48:12.832034+00:00 app[clock.1]: Scaling ...
2014-11-04T20:48:12.910837+00:00 heroku[api]: Scale to clock=1, web=0 by hermanmu@gmail.com
2014-11-04T20:48:12.929993+00:00 app[clock.1]: Success!
2014-11-04T20:48:51.113079+00:00 app[clock.1]: Scaling ...
2014-11-04T20:49:10.486417+00:00 heroku[web.1]: Stopping all processes with SIGTERM
2014-11-04T20:49:11.844089+00:00 heroku[web.1]: Process exited with status 0
2014-11-04T20:49:12.816363+00:00 app[clock.1]: Scaling ...
2014-11-04T20:49:12.936135+00:00 app[clock.1]: Success!
2014-11-04T20:49:12.914887+00:00 heroku[api]: Scale to clock=1, web=0 by hermanmu@gmail.com
```

随着脚本的运行，让我们更新 APS scheduler…

## 调度程序

因为我们再次希望在早上 7 点向外扩展，然后在晚上 10 点扩展，然后在凌晨 3 点再次扩展，所以按如下方式更新计划任务:

```py
@sched.scheduled_job('cron', hour=7)
def scale_out_to_three():
    print 'Scaling out ...'
    print scale(3)

@sched.scheduled_job('cron', hour=22)
def scale_in_to_two():
    print 'Scaling in ...'
    print scale(2)

@sched.scheduled_job('cron', hour=3)
def scale_in_to_one():
    print 'Scaling in ...'
    print scale(1)
```

让它运行至少 24 小时，然后再次检查你的日志以确保它在工作。

## 故障安全

如果我们的应用程序出现故障，让我们确保立即向外扩展，不要问任何问题。

首先，将以下函数添加到脚本中，该函数确定有多少个 dynos 被附加到该进程:

```py
def get_current_dyno_quantity():
    url = "https://api.heroku.com/apps/" + APP + "/formation"
    try:
        result = requests.get(url, headers=HEADERS)
        for formation in json.loads(result.text):
            current_quantity = formation["quantity"]
            return current_quantity
    except:
        return None
```

然后添加新任务:

```py
@sched.scheduled_job('interval', minutes=3)
def fail_safe():
    print "pinging ..."
    r = requests.get('https://APPNAME.herokuapp.com/')
    current_number_of_dynos = get_current_dyno_quantity()
    if r.status_code < 200 or r.status_code > 299:
        if current_number_of_dynos < 3:
            print 'Scaling out ...'
            print scale(3)
    if r.elapsed.microseconds / 1000 > 5000:
        if current_number_of_dynos < 3:
            print 'Scaling out ...'
            print scale(3)
```

在这里，我们向我们的应用程序发送一个 GET 请求(确保更新 URL)，如果状态代码超出 200 范围或者响应时间超过 5000 毫秒，那么我们就向外扩展(只要当前的 dynos 数量不超过 3)。

想测试一下吗？手动移除应用程序中的所有 dynos，然后打开日志:

```py
heroku ps:scale web=0
Scaling web processes... done, now running 0
$ heroku ps
=== clock: `python autoscale.py`
clock.1: up 2014/11/04 15:47:06 (~ 3m ago)

$ heroku logs --tail
2014-11-04T21:53:06.633786+00:00 app[clock.1]: pinging ...
2014-11-04T21:53:06.738860+00:00 app[clock.1]: Scaling out ...
2014-11-04T21:53:06.817780+00:00 heroku[api]: Scale to clock=1, web=3 by michael@realpython.com
2014-11-04T21:53:10.740655+00:00 heroku[web.1]: Starting process with command `gunicorn hello:app`
2014-11-04T21:53:10.634433+00:00 heroku[web.2]: Starting process with command `gunicorn hello:app`
2014-11-04T21:53:11.338596+00:00 heroku[web.3]: Starting process with command `gunicorn hello:app`
2014-11-04T21:53:11.929276+00:00 heroku[web.2]: State changed from starting to up
2014-11-04T21:53:12.731831+00:00 heroku[web.3]: State changed from starting to up
2014-11-04T21:53:12.632277+00:00 heroku[web.1]: State changed from starting to up
2014-11-04T21:56:06.611123+00:00 app[clock.1]: pinging ...
2014-11-04T21:56:06.723760+00:00 app[clock.1]: ... success!
```

完美！

[*Remove ads*](/account/join/)

## 接下来的步骤

好了，我们现在有了一个脚本([下载](https://gist.github.com/mjhea0/e1436b693cc56ca82277))来自动扩展 Heroku dynos。希望这将允许您保持您的应用程序启动和运行，同时也节省一些急需的现金。你也应该睡得更香一点，因为你知道如果有大量的流量涌入，你的应用程序会自动扩展。

下一步是什么？

1.  Autoscale In:当响应时间少于 1000 毫秒时，自动放大。
2.  故障邮件/短信:如果*任何东西*坏了，发送一封邮件和/或短信。
3.  图表:创建一些图表，这样你可以更好地了解你的流量/高峰期等。通过 D3。

干杯！**