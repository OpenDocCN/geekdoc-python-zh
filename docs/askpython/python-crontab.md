# Python crontab–如何在 Python 中使用 Cron？

> 原文：<https://www.askpython.com/python-modules/python-crontab>

今天来说一个有趣的调度模块——Python crontab。值得一提的是，POSIX(即 Windows 操作系统)非常缺乏对 cron 的支持。因此，本文中的例子使用 Linux。

## Python crontab 简介

如果您曾经使用过 [datetime 模块](https://www.askpython.com/python-modules/python-datetime-module)，或者 sched 模块，那么可以肯定地说，您想要在某个时间点安排一个提醒。

如果您已经思考过这样一个特性的扩展将如何持续下去，那么您也可能会得出这样一个结论:您可以编写一个脚本来持续地、重复地部署相同的事件。

简化所有这些，你可能会有一个想法，或者一个问题，*我把我的任务自动化怎么样？*

好消息。

这很容易做到！cron 是一个允许调度命令的特性，因此有助于在特定的时间间隔运行命令。

## cron 是什么？

类 UNIX 操作系统中存在的一个特性是基于时间的作业调度器，即 **cron** 。

它用在软件开发环境中，以便安排可以定期运行的作业，在固定的时间、日期或您可以为自己设置的间隔运行。

## Python crontab 的语法

Cron 需要一整篇文章来解释，所以，这里有一篇[文章](https://www.linuxfordevices.com/tutorials/linux/crontabs-in-linux)可以帮助你了解我们将在这里做什么。

我们将使用 crontabs，它包含了我们已经调度或将要调度的所有作业。

如果您在创建 cron 任务时遇到任何问题，您应该尝试一些在线工具来帮助您理解语法。查看 [crontab.guru](https://crontab.guru) 来创建你的任务，以防你面临任何问题。

## 使用 Python crontab 模块

`python-crontab`模块允许创建 cron 作业的过程变得更加简单。

它为我们提供了一个接受直接输入的简单类，而我们根本不需要使用 cron 语法。

## 安装 python-crontab

为了在 Python 中使用 cron 和 crontab，我们首先需要安装所需的模块，这可以通过 shell 中的 [pip 包管理器](https://www.askpython.com/python-modules/python-pip)命令来完成。

```py
pip install python-crontab

```

这应该会自动安装所需的模块，一旦你完成了它，我们应该准备好使用它！

## 使用 python-crontab

让我们直接进入这个模块的工作，安排我们的第一个任务。

### 1.0 设置

在开始使用 Crontab 中的表达式和任务之前，我们首先需要导入所需的模块。

```py
# Importing the CronTab class from the module
from crontab import CronTab

```

### 1.1 对象创建

为了使用 **Python crontab** ，我们需要设置一个对象来创建作业和它们的循环。

```py
# Creating an object from the class
## Using the root user
cron = CronTab(user="root")

## Using the current user
my_cron = CronTab(user=True)

# Creating an object from the class into a file
file_cron = CronTab(tabfile="filename.tab")

```

### 1.2 使用作业

使用`python-crontab`模块，我们可以创建作业，并指定我们希望它们重复的时间，以及它们必须重复出现的时间间隔。

该模块简化了创建这些任务的大部分工作，并将其从功能性输入转变为 crontab。

```py
# Creating a new job
job  = cron.new(command='echo hello_world')

# Setting up restrictions for the job
## The job takes place once every 5 minutes
job.minute.every(5)

## The job takes place once every four hours
job.hour.every(4)

## The job takes place on the 4th, 5th, and 6th day of the week.
job.day.on(4, 5, 6)

# Clearing the restrictions of a job
job.clear()

```

请记住，每次更改作业的限制时，作业都会清除并用新的限制替换自己。

### 1.3 写入 crontab 文件

最后，我们创建这些作业来为我们提供给定限制的 *cron* 形式，为了写入文件，我们必须手动命令对象将自身写入文件。

这可以通过设置作业限制结束时的一个简单命令来执行。

```py
cron.write()

```

回头看，您会发现 cron 是我们从`CronTab`类创建的对象的名称。

### 1.4 CronTab file

每次设置新的限制时执行 Python 文件，一个干净的 *CronTab* 应该是这样的。

```py
*/5 * * * * echo hello_world
*/5 */4 * * * echo hello_world
*/5 */4 4,5,6 * * echo hello_world

```

## 结论

使用 *cron* 是自动化过程中的一大进步，如果您希望查看可以帮助决定设置任务和工作的时间和日期的模块，您可能会对我们的其他文章感兴趣！

这里是与 [dateutil](https://www.askpython.com/python-modules/dateutil-module) 、 [psutil](https://www.askpython.com/python-modules/psutil-module) 合作，如果你出于任何原因试图自动化数据集，并且需要一个起点，[熊猫](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)！

## 参考

*   [官方 python-crontab 文档](https://pypi.org/project/python-crontab/)
*   [使用 Cron 表达式](https://crontab.guru)