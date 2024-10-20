# 如何用 Python 从 Jenkins 提取构建信息

> 原文：<https://www.blog.pythonlibrary.org/2019/05/14/how-to-extract-build-info-from-jenkins-with-python/>

我工作的一部分是使用持续集成软件。我在我的角色中同时使用 Hudson 和 Jenkins，偶尔需要以编程方式与他们进行交互。有两个 Python 包可用于此任务:

*   Python Jenkins [包](https://python-jenkins.readthedocs.io/en/latest/)
*   詹金斯皮

Python Jenkins 包可以与 Hudson 和 Jenkins 一起工作，而 JenkinsAPI 只能与 Jenkins 一起工作。因为这个原因，我通常使用 Python Jenkins，尽管我最近开始寻找哪一个更适合工件，并且我发现 JenkinsAPI 实际上更适合这种事情。所以你需要根据你需要做什么来评估这两个包。

* * *

### 安装 Python Jenkins

为了遵循本文中的代码示例，您需要安装 Python Jenkins。您可以使用画中画来做到这一点:

```py

pip install python-jenkins

```

现在已经安装好了，让我们试一试 Python Jenkins 吧！

* * *

### 从詹金斯那里得到所有的工作

一个常见的任务是需要获得构建系统中配置的所有作业的列表。

要开始，您需要登录到您的 Jenkins 服务器:

```py

import jenkins

server = jenkins.Jenkins('http://server:port/', username='user', 
                         password='secret')

```

现在您有了一个 Jenkins 对象，您可以使用它对您的 Jenkins CI 服务器执行 REST 请求。返回的结果通常是 Python 字典或字典的字典。

以下是获取 CI 系统上配置的所有作业的示例:

```py

import jenkins

server = jenkins.Jenkins('http://server:port/', username='user',
                         password='secret')

# Get all builds
jobs = server.get_all_jobs(folder_depth=None)
for job in jobs:
    print(job['fullname'])

```

这将遍历 Jenkins 中配置的所有作业，并打印出它们的作业名。

* * *

### 获取工作信息

现在您已经知道了 Jenkins box 上的作业名称，您可以获得关于每个作业的更详细的信息。

方法如下:

```py

import jenkins

server = jenkins.Jenkins('http://server:port/', username='user',
                         password='secret')

# Get information on specific build job
# This returns all the builds that are currently shown in 
# hudson for this job
info = server.get_job_info('job-name')

# Passed
print(info['lastCompletedBuild'])

# Unstable
print(info['lastUnstableBuild'])

# Failed
print(info['lastFailedBuild'])

```

`get_job_info() will give you a lot of information about the job, including all the currently saved builds. It is nice to be able to extract which builds have passed, failed or are unstable.`

* * *

### 获取构建信息

如果您想知道一个作业运行需要多长时间，那么您需要深入到构建级别。

让我们来看看如何实现:

```py

import jenkins

server = jenkins.Jenkins('http://server:port/', username='user',
                         password='secret')

info = server.get_job_info('job-name')

# Loop over builds
builds = info['builds']
for build in builds:
    for build in builds:
        print(server.get_build_info('job-name', 
                                    build['number']))    

```

要获取构建元数据，您需要调用`get_build_info(). This method takes in the job name and the build number and returns the metadata as a dictionary.`

* * *

### 包扎

使用 Python Jenkins 包可以做更多的事情。例如，您可以使用它来启动一个构建作业、创建一个新作业或删除一个旧作业以及许多其他事情。不幸的是，文档是非常粗略的，所以您必须做一些试验来让它按照您想要的方式工作。

* * *

### 附加阅读

*   Python Jenkins [文档](https://python-jenkins.readthedocs.io/en/latest/)
*   JenkinsAPI [文档](https://jenkinsapi.readthedocs.io/en/latest/)