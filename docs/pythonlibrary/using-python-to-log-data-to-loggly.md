# 使用 Python 将数据记录到日志中

> 原文：<https://www.blog.pythonlibrary.org/2014/10/24/using-python-to-log-data-to-loggly/>

我的一位读者建议我应该尝试将我的数据记录到一个名为 [Loggly](http://loggly.com/) 的 web 服务中。据我所知，Loggly 是一种与企业中的每个人共享日志数据的方式，这样您就不再需要登录到单独的机器。他们还提供了日志的图表、过滤器和搜索。他们没有 Python API，但是通过 Python**urllib 2**模块和 **simplejson** 向 Loggly 发送数据还是很容易的。另请注意，您可以在 30 天的试用期内使用 Loggly。

让我们来看看一些代码。这段代码基于我的文章[中关于当前运行进程日志的代码。您需要安装以下模块，此示例才能运行:](https://www.blog.pythonlibrary.org/2014/10/21/logging-currently-running-processes-with-python/)

*   [psutil](https://code.google.com/p/psutil/)
*   [simplejson](https://pypi.python.org/pypi/simplejson/)

我刚刚用 pip 安装了它们。现在我们有了这些，让我们看看如何使用它们来连接 Loggly

```py

import psutil
import simplejson
import time
import urllib2

#----------------------------------------------------------------------
def log():
    """"""
    token = "YOUR-LOGGLY-TOKEN"
    url = "https://logs-01.loggly.com/inputs/%s/tag/python/" % token

    while True:
        proc_dict = {}
        procs = psutil.get_process_list()
        procs = sorted(procs, key=lambda proc: proc.name)
        for proc in procs:
            cpu_percent = proc.get_cpu_percent()
            mem_percent = proc.get_memory_percent()
            try:
                name = proc.name()
            except:
                # this is a process with no name, so skip it
                continue

            data = {"cpu_percent": cpu_percent,
                    "mem_percent": mem_percent,
                    }
            proc_dict[name] = data

        log_data = simplejson.dumps(proc_dict)
        urllib2.urlopen(url, log_data)
        time.sleep(60)

if __name__ == "__main__":
    log()

```

这是一个非常简单的函数，但是让我们来分解它。首先，我们设置 Loggly token 并创建一个 Loggly URL 来发送我们的数据。然后我们创建一个无限循环，每 60 秒获取一个当前正在运行的进程列表。接下来，我们提取出想要记录的信息，然后将这些信息放入字典中。最后，我们使用 simplejson 的 **dumps** 方法将我们的嵌套字典转换成 json 格式的字符串，并将其传递给我们的 url。这会将日志数据发送到 Loggly，在那里可以对其进行解析。

一旦你向 Loggly 发送了足够的数据供其分析，你就可以登录到你的帐户，并查看一些根据你的数据自动创建的条形图。我在这个例子中的数据没有很好地转化为图表或趋势，所以这些看起来很无聊。我建议发送一个系统日志或其他包含更多种类的东西，以便更好地了解这项服务对您的用处。

**更新:**log Gly 的人提到尝试将数据作为浮点数而不是字符串发送，所以我适当地编辑了上面的代码。请随意摆弄代码，看看您能想到什么。

* * *

### 相关代码

*   我的一个读者提出了我的[例子](https://gist.github.com/driscollis/db11609833e408430d0d)的修改版本