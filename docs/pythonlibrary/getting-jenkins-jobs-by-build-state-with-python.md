# 使用 Python 通过构建状态获取 Jenkins 作业

> 原文：<https://www.blog.pythonlibrary.org/2020/01/14/getting-jenkins-jobs-by-build-state-with-python/>

我最近一直在与 Python 和 Jenkins 一起工作，最近需要找到一种方法在构建级别检查作业的状态。我发现了 [jenkinsapi](https://jenkinsapi.readthedocs.io/en/latest/#) 包，并试用了一下，看看它是否能让我深入到 Jenkins 中的构建和结果集级别。

在我运行的构建中，有 X 数量的子作业。这些子作业中的每一个都可能通过或失败。如果其中一个失败了，整个构建就会被标上黄色，并被标记为“不稳定”，这在我的书中是失败的。我需要一种方法来跟踪这些子作业中哪个失败了，以及在一段时间内失败的频率。这些作业中的一些可能不稳定，因为它们访问网络资源，而另一些可能由于最近对代码库的提交而中断。

我最终想出了一些代码来帮助我理解这些信息。但是在深入研究代码之前，您需要安装一个包。

* * *

### 安装必备组件

jenkinsapi 包很容易安装，因为它是 pip 兼容的。您可以使用以下命令将其安装到您的主 Python 安装或 Python 虚拟环境中:

`pip install jenkinsapi`

您还需要安装[请求](https://requests.readthedocs.io/en/master/)，这也是 pip 兼容的:

`pip install requests`

这些是你唯一需要的包裹。现在你可以进入下一部分了！

### 询问詹金斯

您想要完成的第一步是按状态获取作业。Jenkins 中的标准状态包括成功、不稳定或中止。

让我们编写一些代码来查找不稳定的作业:

```py

from jenkinsapi.jenkins import Jenkins
from jenkinsapi.custom_exceptions import NoBuildData
from requests import ConnectionError

def get_job_by_build_state(url, view_name, state='SUCCESS'):
    server = Jenkins(url)
    view_url = f'{url}/view/{view_name}/'
    view = server.get_view_by_url(view_url)
    jobs = view.get_job_dict()

    jobs_by_state = []

    for job in jobs:
        job_url = f'{url}/{job}'
        j = server.get_job(job)
        try:
            build = j.get_last_completed_build()
            status = build.get_status()
            if status == state:
                jobs_by_state.append(job)
        except NoBuildData:
            continue
        except ConnectionError:
            pass

    return jobs_by_state

if __name__ == '__main__':
    jobs = get_job_by_build_state(url='http://myJenkins:8080', view_name='VIEW_NAME',
                                  state='UNSTABLE')

```

在这里，您创建了一个 **Jenkins** 的实例，并将其分配给**服务器**。然后使用 **get_view_by_url()** 获取指定的视图名称。该视图基本上是您设置的一组相关联的职务。例如，您可以创建一组做开发/操作类事情的作业，并将它们放入 Utils 视图中。

一旦有了**视图**对象，就可以使用 **get_job_dict()** 来获取该视图中所有作业的字典。既然已经有了字典，就可以对它们进行循环，并在视图中获得各个作业。您可以通过调用 Jenkin 对象的 **get_job()** 方法来获得作业。既然有了 job 对象，您终于可以深入到构建本身了。

为了防止错误，我发现可以使用**get _ last _ completed _ build()**来获得最后一次完整的构建。这是最好的，如果你使用 **get_build()** 并且构建还没有完成，构建对象可能没有你期望的内容。现在您已经有了构建，您可以使用 **get_status()** 来获取它的状态，并将其与您传入的那个进行比较。如果它们匹配，那么将该作业添加到 **jobs_by_state** ，这是一个 Python 列表。

您还会发现一些可能发生的错误。你可能看不到 **NoBuildData** ，除非作业被中止或者你的服务器上发生了一些非常奇怪的事情。当你试图连接到一个不存在或离线的 URL 时，就会发生 **ConnectionError** 异常。

此时，您应该有一个筛选到您所要求的状态的作业列表。

如果您想进一步深入到作业中的子作业，那么您需要调用构建的 **has_resultset()** 方法来验证是否有要检查的结果。然后你可以这样做:

```py

resultset = build.get_resultset()
for item in resultset.items():
    # do something here

```

根据作业类型的不同，返回的 **resultset** 会有很大不同，所以您需要自己解析**条目**元组，看看它是否包含您需要的信息。

* * *

### 包扎

此时，您应该有足够的信息来开始挖掘 Jenkin 的内部信息，以获得您需要的信息。我使用了这个脚本的一个变体来帮助我提取构建失败的信息，帮助我更快地发现重复失败的作业。不幸的是， **jenkinsapi** 的文档不是很详细，所以你将会在调试器中花费大量的时间试图弄清楚它是如何工作的。然而，一旦你搞清楚了，它总体上工作得很好。