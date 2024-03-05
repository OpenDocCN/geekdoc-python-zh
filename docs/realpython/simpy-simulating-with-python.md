# SimPy:用 Python 模拟真实世界的流程

> 原文：<https://realpython.com/simpy-simulating-with-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 SimPy**](/courses/simulating-processes-simpy/) 用 Python 模拟真实世界的流程

现实世界充满了系统，像机场和高速公路，经常经历拥堵和延误。当这些系统没有得到优化时，它们的低效率会导致无数不满意的客户和数小时的时间浪费。在本教程中，你将学习如何使用 Python 的 **`simpy`** 框架来创建虚拟模拟，帮助你解决类似的问题。

**在本教程中，您将学习如何:**

*   **使用**模拟来模拟真实世界的流程
*   **创建**一种逐步算法来近似一个复杂的系统
*   **用`simpy`设计**并用 Python 运行一个真实世界的模拟

在本教程中，您将为本地电影院创建一个模拟。您的目标是为经理提供一个脚本，以帮助找到员工的最佳数量。您可以通过单击下面的链接下载该脚本的源代码:

**下载代码:** [单击此处下载代码，您将使用](https://realpython.com/bonus/simpy/)在本教程中了解 SimPy。

## 什么是模拟

一个**模拟**是一个真实世界系统的代表。人们可以使用这个系统的数学或计算模型来研究它是如何工作的，或者当它的部分被改变时会发生什么。模拟用于机场、餐馆、机械、政府机构和许多其他系统，在这些系统中，不良的资源分配会导致拥堵、客户不满和严重的运输延误。

一个**系统**可以是任何事情发生的环境。现实世界系统的例子包括洗车场、银行、制造厂、机场、邮局、呼叫中心等等。这些系统有**代理**，它们内部经历**过程**。例如:

*   洗车处会让汽车经过清洗过程。
*   机场会让乘客通过安检程序。
*   呼叫中心会让客户经历与电话销售人员交谈的过程。

下表总结了这种关系:

| 系统 | 代理人 | 过程 |
| --- | --- | --- |
| 洗车处 | 汽车 | 洗涤 |
| 机场 | 乘客 | 安全检查 |
| 呼叫中心 | 顾客 | 与电话销售人员交谈 |

理解代理在一个系统中经历的过程是后勤规划的一个重要组成部分，特别是对于大规模的组织。例如，如果当天没有足够的工作人员，机场可以看到乘客在安检处的等待时间飙升。类似地，如果没有正确的路由，时间敏感的邮件可能会延迟几天(甚至几周)。

这些拥堵情况会对时间和金钱产生现实后果，因此能够预先对这些过程建模是很重要的。这让您知道系统可能在哪里遇到问题，以及应该如何提前分配资源，以尽可能最有效的方式解决这些问题。

[*Remove ads*](/account/join/)

## 模拟如何工作

在 Python 中，可以使用 [`simpy`](https://simpy.readthedocs.io/en/latest/) 框架进行事件模拟。首先，快速浏览一下 Python 中模拟的流程是如何运行的。下面是一个安全检查点系统模拟的代码片段。以下三行代码设置环境，传递所有必要的函数，并运行模拟:

```py
# Set up the environment
env = simpy.Environment()

# Assume you've defined checkpoint_run() beforehand
env.process(checkpoint_run(env, num_booths, check_time, passenger_arrival))

# Let's go!
env.run(until=10)
```

上面的第一行代码建立了**环境**。你可以通过将`simpy.Environment()`赋值给期望的[变量](https://realpython.com/python-variables/)来实现。这里简单命名为`env`。这告诉`simpy`创建一个名为`env`的环境对象，该对象将管理模拟时间，并在每个后续的时间步骤中移动模拟。

一旦你建立了你的环境，你将传递所有的变量作为你的参数。这些都是你可以改变的，以观察系统对变化的反应。对于这个安全检查点系统，您使用以下参数:

1.  **`env` :** 环境对象对事件进行调度和处理
2.  **`num_booths`:**ID 检查亭的数量
3.  **`check_time:`** 检查一个乘客的身份证所需要的时间长度
4.  **`passenger_arrival` :** 乘客到达队列的速度

然后，是时候运行模拟了！您可以通过调用`env.run()`并指定您希望模拟运行多长时间来实现。该模拟以分钟为单位运行，因此该示例代码将运行 10 分钟的实时模拟。

**注意:**别急！你不需要等 10 分钟就能完成模拟。因为模拟给你一个虚拟的实时过程，这 10 分钟在电脑上只需几秒钟就能过去。

概括来说，以下是在 Python 中运行模拟的三个步骤:

1.  **建立**环境。
2.  **在参数中传递**。
3.  **运行**模拟。

但是在引擎盖下还有更多事情要做！您需要了解如何选择这些参数，并且您必须定义在模拟运行时将调用的所有函数。

我们开始吧！

## `simpy` 如何入门

在 Python 中创建模拟之前，您应该检查一下列表中的一些待办事项。你需要做的第一件事是确保你对 [Python 基础](https://realpython.com/learning-paths/python-basics-book/)有扎实的理解。特别是，您需要很好地掌握类和生成器。

**注意:**如果你需要更新这些主题，那么看看[Python 面向对象编程(OOP)简介](https://realpython.com/courses/intro-object-oriented-programming-oop-python/)和[Python 生成器简介](https://realpython.com/introduction-to-python-generators/)。这些是模拟过程中至关重要的部分，因此在继续之前，您需要理解它们。

接下来您要做的是安装所需的包。您将使用的主要框架是`simpy`。这是创建、管理和运行模拟的核心包。可以用 [`pip`](https://realpython.com/what-is-pip/) 安装:

```py
$ python3 -m pip install simpy
```

您还需要一些内置的 Python 模块。您将使用`statistics`模块计算平均等待时间，使用`random`模块[生成随机数](https://realpython.com/python-random/)。这些是 Python 标准库的一部分，所以您不需要安装任何新的东西。

最后，您需要选择运行模拟的方式。通常，您可以选择两个选项之一:

1.  **交互式运行:**使用一个 [Jupyter 笔记本](https://realpython.com/jupyter-notebook-introduction/)，其中每个代码块将包含自己的类或函数定义。输出将显示在笔记本的底部。
2.  **在 shell 中运行:**将您的模拟保存为`.py`文件，并告诉 Python 在您的终端中运行它。输出将直接打印到控制台。

选择你最喜欢的方法！结局应该是一样的。

在本教程中，您会看到对名为`simulate.py`的独立文件的引用。当您浏览本教程时，代码块将引用`simulate.py`来帮助您跟踪所有的部分是如何组合在一起的。作为参考，你可以通过下面的链接获得`simulate.py`的完整代码:

**下载代码:** [单击此处下载代码，您将使用](https://realpython.com/bonus/simpy/)在本教程中了解 SimPy。

请随意保存文件`simulate.py`并在您最喜欢的编辑器中跟随！

[*Remove ads*](/account/join/)

## 如何用`simpy`包模拟

在`simpy`中运行模拟的第一步是选择要建模的流程。模拟就是创建一个虚拟环境来反映真实世界的系统。本着同样的精神，您将为您的模拟“模拟”一种情况！

想象一下，你被当地一家小电影院雇用来帮助经理。由于等待时间长，这家影院的评价一直很差。这位经理既关心成本，也关心顾客的满意度，他只能养得起这么多的员工。

经理特别担心一旦这些大片开始上映，会出现什么样的混乱:影院周围排起了长队，员工们工作到了极限，愤怒的观众错过了开场镜头……这绝对是应该避免的情况！

在检查评论后，经理能够确定给定的电影院观众愿意从他们到达到他们坐下最多花 10 分钟。换句话说，在电影院过夜的平均等待时间需要在 10 分钟或更少。经理请求您帮助找出一个解决方案，使客户等待时间低于 10 分钟的要求。

### 头脑风暴模拟算法

在您编写一行代码之前，重要的是您首先要弄清楚您的过程在现实生活中是如何运行的。这是为了确保，当你把它传递给机器时，这个过程准确地反映了客户将真正体验到的东西。你可以这样思考一个电影观众写出你的算法的步骤:

1.  **到达**剧院，排队，等着买票。
2.  从售票处买一张票。
3.  排队等候检票。
4.  得到引座员检查过的票。
5.  **选择**是否排队进入小卖部:
    *   如果他们排队，然后他们购买食物。
    *   如果他们没有排队，然后他们跳到最后一步。
6.  去找他们的座位。

对于在影院票房购票的观众来说，这是一个循序渐进的迭代过程。您已经可以看到这个过程的哪些部分是可以控制的。你可以通过增加售票处的收银员来影响顾客的等待时间。

过程中也有一些部分是无法控制的，比如第一步。你无法控制会有多少顾客到来，或者他们到来的速度有多快。你可以猜测，但你不能简单地选择一个数字，因为那不能很好地反映现实。对于这个参数，你能做的最好的事情是**使用可用的数据**来确定一个合适的到达时间。

**注意:**使用历史数据可以确保您找到的解决方案将准确反映您在现实生活中可以预期看到的情况。

考虑到这些事情，是时候构建您的模拟了！

### 设置环境

在开始构建您的模拟之前，您需要确保您的[开发环境](https://realpython.com/learning-paths/perfect-your-python-development-setup/)被正确配置。你要做的第一件事就是导入必要的[包](https://realpython.com/python-modules-packages/)。您可以通过在文件顶部声明`import`语句来做到这一点:

```py
import simpy
import random
import statistics
```

这些是您将用来为剧院经理构建脚本的主要库。请记住，目标是找到平均等待时间少于 10 分钟的最佳员工数量。为此，您需要收集每位电影观众到达座位所用的时间长度。下一步是声明一个保存这些时间的列表:

```py
wait_times = []
```

这个列表将包含每个电影观众从到达电影院到坐到座位上所花费的总时间。您在文件的最顶端声明这个列表，这样您就可以在以后定义的任何函数中使用它。

### 创建环境:类定义

您想要构建的模拟的第一部分是系统的蓝图。这将是事情发生的整体环境，人或物体从一个地方移动到另一个地方。记住，**环境**可以是许多不同系统中的一个，比如银行、洗车场或安全检查站。在这种情况下，环境是一个电影院，所以这将是您的[类](https://realpython.com/lessons/classes-python/)的名称:

```py
class Theater(object):
    def __init__(self):
        # More to come!
```

现在是时候思考一下**电影院**的各个部分了。当然，还有剧院本身，也就是你所说的环境。稍后，您将使用一个`simpy`函数显式地将剧院声明为实际的`environment`。现在，将其简称为`env`，并将其添加到类定义中:

```py
class Theater(object):
    def __init__(self, env):
 self.env = env
```

好吧，剧院里还会有什么？你可以通过思考你之前设计的模拟算法来解决这个问题。当观众到达时，他们需要在票房排队，收银员会等着帮他们。现在你已经发现了关于剧院环境的两件事:

1.  还有**收银员**。
2.  电影观众可以从他们那里买票。

收银员是影院提供给顾客的一种资源，他们帮助观众完成购票的 T2 流程。现在，你不知道模拟剧场里有多少收银员。事实上，这正是你想要解决的问题。等待时间是如何变化的，取决于某个晚上工作的收银员的数量？

你可以继续调用这个未知变量`num_cashiers`。这个变量的确切值可以在以后解决。现在，只要知道它是剧院环境中不可或缺的一部分。将其添加到类定义中:

```py
class Theater(object):
 def __init__(self, env, num_cashiers):        self.env = env
 self.cashier = simpy.Resource(env, num_cashiers)
```

在这里，您将新参数`num_cashiers`添加到您的`__init__()`定义中。然后，您创建一个资源`self.cashier`，并使用`simpy.Resource()`来声明在任何给定时间这个环境中可以有多少个资源。

**注:`simpy`、[资源](https://simpy.readthedocs.io/en/latest/simpy_intro/shared_resources.html)中的**是环境(`env`)中数量有限的部分。使用其中一个需要时间，而且一次只能使用这么多(`num_cashiers`)。

你还需要再走一步。收银员不会为*自己*买票，对吧？他们会帮助观众的！同样，您知道购买机票的过程需要一定的时间。但是需要多少时间呢？

假设你已经向经理索要了剧院的历史数据，比如员工绩效评估或购票收据。根据这些数据，您已经了解到在售票处出票平均需要 1 到 2 分钟。你如何让`simpy`模仿这种行为？它只需要一行代码:

```py
yield self.env.timeout(random.randint(1, 3))
```

`env.timeout()`告知`simpy`在经过一定时间后触发事件。在这种情况下，事件是购买了一张票。

这需要的时间可能是一分钟、两分钟或三分钟。你希望每个电影观众在收银台花费不同的时间。为此，您可以使用`random.randint()`在给定的高低值之间选择一个随机数。然后，对于每个电影观众，模拟将等待选定的时间。

让我们用一个简洁的函数将它包装起来，并将其添加到类定义中:

```py
class Theater(object):
    def __init__(self, env, num_cashiers):
        self.env = env
        self.cashier = simpy.Resource(env, num_cashiers)

 def purchase_ticket(self, moviegoer): yield self.env.timeout(random.randint(1, 3))
```

在`purchase_ticket()`中启动事件的是`moviegoer`，因此它们必须作为必需的参数传递。

**注意:**你将在下一节看到观众实际上是如何购票的！

就是这样！您已经选择了一个有时间限制的资源，定义了它的相关流程，并在您的类定义中对此进行了编码。对于本教程，您还需要声明两个资源:

1.  检票员
2.  **卖食物的服务员**

在检查了经理发送过来的数据后，您确定服务器需要 1 到 5 分钟来完成一个订单。此外，引座员检票速度非常快，平均 3 秒钟！

您需要将这些资源添加到您的类中，并定义相应的函数`check_ticket()`和`sell_food()`。你能想出代码应该是什么样子吗？当你有了一个想法，你可以展开下面的代码块来检查你的理解:



```py
class Theater(object):
 def __init__(self, env, num_cashiers, num_servers, num_ushers):        self.env = env
        self.cashier = simpy.Resource(env, num_cashiers)
 self.server = simpy.Resource(env, num_servers) self.usher = simpy.Resource(env, num_ushers) 
    def purchase_ticket(self, moviegoer):
        yield self.env.timeout(random.randint(1, 3))

 def check_ticket(self, moviegoer): yield self.env.timeout(3 / 60) 
 def sell_food(self, moviegoer): yield self.env.timeout(random.randint(1, 5))
```

仔细看看新的资源和功能。注意它们是如何遵循如上所述的格式的。`sell_food()`使用`random.randint()`生成一个介于 1 到 5 分钟之间的随机数，代表观众下单并收到食物所需的时间。

`check_ticket()`的延时有点不同，因为引座员只需要 3 秒钟。因为`simpy`以分钟为单位，所以这个值需要作为一分钟的一部分，或者说`3 / 60`来传递。

[*Remove ads*](/account/join/)

### 在环境中移动:功能定义

好了，你已经通过定义一个类建立了环境。你有资源和流程。现在你需要一个 **`moviegoer`** 来使用它们。当一个`moviegoer`到达剧院时，他们将请求一个资源，等待其流程完成，然后离开。您将创建一个名为`go_to_movies()`的函数来跟踪这一点:

```py
def go_to_movies(env, moviegoer, theater):
    # Moviegoer arrives at the theater
    arrival_time = env.now
```

有三个参数传递给该函数:

1.  **`env`:**`moviegoer`会被环境控制，所以你会把这个作为第一个参数传递。
2.  **`moviegoer` :** 这个变量跟踪每个人在系统中的移动。
3.  **`theater` :** 此参数允许您访问您在总体类定义中定义的流程。

您还声明了一个变量`arrival_time`来保存每个`moviegoer`到达剧院的时间。您可以使用`simpy`呼叫`env.now`来获得该时间。

您将希望来自您的`theater`的每个流程在`go_to_movies()`中有相应的**请求**。例如，类中的第一个进程是`purchase_ticket()`，它使用了一个`cashier`资源。`moviegoer`将需要向`cashier`资源发出请求，以帮助他们完成`purchase_ticket()`流程。下面的表格总结了这一点:

| `theater`中的过程 | `go_to_movies()`中的请求 |
| --- | --- |
| `purchase_ticket()` | 请求一个`cashier` |
| `check_ticket()` | 请求一个`usher` |
| `sell_food()` | 请求一个`server` |

收银台是一个[共享资源](https://simpy.readthedocs.io/en/latest/topical_guides/resources.html)，这意味着许多电影观众将使用同一个收银台。然而，一个收银员一次只能帮助一个观众，所以你需要在你的代码中包含一些等待行为。这是如何工作的:

```py
def go_to_movies(env, moviegoer, theater):
    # Moviegoer arrives at the theater
    arrival_time = env.now

 with theater.cashier.request() as request: yield request yield env.process(theater.purchase_ticket(moviegoer))
```

下面是这段代码的工作原理:

1.  **`theater.cashier.request()` :** `moviegoer`生成使用`cashier`的请求。
2.  **`yield request` :** `moviegoer`等待一个`cashier`变为可用，如果所有的都在使用中。要了解更多关于`yield`关键字的信息，请查看[如何在 Python 中使用生成器和 yield](https://realpython.com/introduction-to-python-generators/)。
3.  **`yield env.process()` :** `moviegoer`使用可用的`cashier`完成给定的过程。在这种情况下，就是通过呼叫`theater.purchase_ticket()`来购买机票。

使用资源后，必须释放该资源以供下一个代理使用。你可以用`release()`显式地做到这一点，但是在上面的代码中，你用一个 [`with`语句](https://realpython.com/python-with-statement/)来代替。此快捷方式告诉模拟在流程完成后自动释放资源。换句话说，一旦买了票，`moviegoer`就会离开，收银员就会自动准备带下一位顾客。

当收银员有空的时候，`moviegoer`会花一些时间去买他们的票。`env.process()`告诉模拟转到`Theater`实例，并在这个`moviegoer`上运行`purchase_ticket()`流程。`moviegoer`将重复此**请求、使用、释放**周期，以检查其车票:

```py
def go_to_movies(env, moviegoer, theater):
    # Moviegoer arrives at the theater
    arrival_time = env.now

    with theater.cashier.request() as request:
        yield request
        yield env.process(theater.purchase_ticket(moviegoer))

 with theater.usher.request() as request: yield request yield env.process(theater.check_ticket(moviegoer))
```

这里，代码的结构是相同的。

然后，还有从小卖部购买食物的可选步骤。你无法知道电影观众是否会想买零食和饮料。处理这种不确定性的一种方法是在函数中引入一点随机性。

每个人`moviegoer`要么想要么不想买食物，你可以把它们存储为布尔值`True`或`False`。然后，使用`random`模块让模拟随机决定*该*特定的`moviegoer`是否会前往特许摊位:

```py
def go_to_movies(env, moviegoer, theater):
    # Moviegoer arrives at the theater
    arrival_time = env.now

    with theater.cashier.request() as request:
        yield request
        yield env.process(theater.purchase_ticket(moviegoer))

    with theater.usher.request() as request:
        yield request
        yield env.process(theater.check_ticket(moviegoer))

 if random.choice([True, False]): with theater.server.request() as request: yield request yield env.process(theater.sell_food(moviegoer))
```

这个[条件语句](https://realpython.com/courses/python-conditional-statements/)将返回两个结果之一:

1.  **`True`:**`moviegoer`将请求服务器并点餐。
2.  **`False`:**`moviegoer`会去找座位，不买任何零食。

现在，请记住这个模拟的目标是确定收银员、迎宾员和服务员的数量，以便将等待时间控制在 10 分钟以内。要做到这一点，你需要知道任何给定的`moviegoer`到达座位需要多长时间。您在函数开始时使用`env.now`来跟踪`arrival_time`，并在每个`moviegoer`完成所有流程并进入剧院时再次使用:

```py
def go_to_movies(env, moviegoer, theater):
 # Moviegoer arrives at the theater arrival_time = env.now 
    with theater.cashier.request() as request:
        yield request
        yield env.process(theater.purchase_ticket(moviegoer))

    with theater.usher.request() as request:
        yield request
        yield env.process(theater.check_ticket(moviegoer))

    if random.choice([True, False]):
        with theater.server.request() as request:
            yield request
            yield env.process(theater.sell_food(moviegoer))

 # Moviegoer heads into the theater wait_times.append(env.now - arrival_time)
```

您使用`env.now`来获取`moviegoer`完成所有流程并到达座位的时间。你从这个出发时间中减去观众的`arrival_time`，并将得到的时差添加到你的等待列表中`wait_times`。

**注意:**你可以像`departure_time`一样将出发时间存储在一个单独的变量中，但这会使你的代码非常重复，这违反了 [D.R.Y .原则](https://realpython.com/lessons/dry-principle/)。

这个`moviegoer`准备看一些预告！

[*Remove ads*](/account/join/)

### 使事情发生:函数定义

现在，您需要定义一个函数来运行模拟。将负责创建一个剧院的实例，并生成电影观众，直到模拟停止。这个函数应该做的第一件事是创建一个剧院的实例:

```py
def run_theater(env, num_cashiers, num_servers, num_ushers):
    theater = Theater(env, num_cashiers, num_servers, num_ushers)
```

因为这是主过程，所以你需要传递所有你已经声明的未知数:

*   `num_cashiers`
*   `num_servers`
*   `num_ushers`

这些都是模拟需要创建和控制环境的变量，所以通过它们是绝对重要的。然后，定义一个变量`theater`,并告诉模拟用一定数量的收银员、服务员和招待员来设置剧院。

您可能还想从一些在电影院等候的电影观众开始您的模拟。门一开，大概就有几个人准备走了！经理说，预计票房一开门，就会有大约 3 名观众排队买票。您可以让模拟继续进行，并像这样穿过这个初始组:

```py
def run_theater(env, num_cashiers, num_servers, num_ushers):
    theater = Theater(env, num_cashiers, num_servers, num_ushers)

 for moviegoer in range(3): env.process(go_to_movies(env, moviegoer, theater))
```

你使用 [`range()`](https://realpython.com/courses/python-range-function/) 来填充 3 个观影者。然后，您使用`env.process()`告诉模拟准备移动他们通过剧院。其余的观众将在他们自己的时间里赶到电影院。因此，只要模拟在运行，该功能就应该不断向剧院发送新客户。

你不知道新的电影观众要多久才能到达电影院，所以你决定查看过去的数据。使用票房的时间戳收据，你了解到电影观众倾向于平均每 12 秒到达电影院。现在您所要做的就是告诉函数在生成一个新人之前等待这么长时间:

```py
def run_theater(env, num_cashiers, num_servers, num_ushers):
    theater = Theater(env, num_cashiers, num_servers, num_ushers)

    for moviegoer in range(3):
        env.process(go_to_movies(env, moviegoer, theater))

 while True: yield env.timeout(0.20)  # Wait a bit before generating a new person 
        # Almost done!...
```

请注意，您使用十进制数字`0.20`来表示 12 秒。要得到这个数，你只需将 12 秒除以 60 秒，60 秒就是一分钟的秒数。

等待之后，函数应该将`moviegoer`加 1，生成下一个人。[发生器](https://realpython.com/introduction-to-python-generators/)函数就是你用来初始化前三个电影观众的函数:

```py
def run_theater(env, num_cashiers, num_servers, num_ushers):
    theater = Theater(env, num_cashiers, num_servers, num_ushers)

    for moviegoer in range(3):
        env.process(go_to_movies(env, moviegoer, theater))

    while True:
        yield env.timeout(0.20)  # Wait a bit before generating a new person

 moviegoer += 1 env.process(go_to_movies(env, moviegoer, theater))
```

就是这样！当您调用此函数时，模拟将生成 3 个电影观众，并开始用`go_to_movies()`移动他们通过电影院。之后，新的观影者将间隔 12 秒到达影院，并在自己的时间内穿过影院。

### 计算等待时间:功能定义

此时，您应该有一个列表`wait_times`,其中包含每个电影观众到达座位所花费的总时间。现在，您需要定义一个函数来帮助计算 a `moviegoer`从到达到完成检票的平均时间。`get_average_wait_time()`无非如此:

```py
def get_average_wait_time(wait_times):
    average_wait = statistics.mean(wait_times)
```

这个函数将您的`wait_times`列表作为一个参数，并使用`statistics.mean()`来计算平均等待时间。

因为您正在创建一个将由电影院管理器使用的脚本，所以您将希望确保用户能够容易地阅读输出。为此，您可以添加一个名为`calculate_wait_time()`的函数:

```py
def calculate_wait_time(arrival_times, departure_times):
    average_wait = statistics.mean(wait_times)
    # Pretty print the results
    minutes, frac_minutes = divmod(average_wait, 1)
    seconds = frac_minutes * 60
    return round(minutes), round(seconds)
```

函数的最后一部分使用`divmod()`以分钟和秒为单位返回结果，因此管理人员可以很容易地理解程序的输出。

[*Remove ads*](/account/join/)

### 选择参数:用户输入功能定义

在构建这些函数时，您会遇到一些尚未明确定义的变量:

*   `num_cashiers`
*   `num_servers`
*   `num_ushers`

这些变量是你可以通过**改变**来观察模拟如何变化的参数。如果一部卖座电影有顾客在街区周围排队，应该有多少收银员在工作？如果人们在票房上飞来飞去，却在让步上停滞不前怎么办？什么值的`num_servers`将有助于缓解流量？

**注:**这就是模拟的妙处。它允许你尝试这些事情，这样你就可以在现实生活中做出最好的决定。

无论是谁使用您的模拟，都需要能够更改这些参数的值来尝试不同的场景。为此，您将创建一个助手函数来从用户那里获取这些值:

```py
def get_user_input():
    num_cashiers = input("Input # of cashiers working: ")
    num_servers = input("Input # of servers working: ")
    num_ushers = input("Input # of ushers working: ")
    params = [num_cashiers, num_servers, num_ushers]
    if all(str(i).isdigit() for i in params):  # Check input is valid
        params = [int(x) for x in params]
    else:
        print(
            "Could not parse input. The simulation will use default values:",
            "\n1 cashier, 1 server, 1 usher.",
        )
        params = [1, 1, 1]
    return params
```

这个函数简单地调用 Python 的`input()`函数从用户那里检索数据。因为用户输入冒着混乱的风险，所以您可以包含一个`if/else`子句来捕捉任何无效的内容。如果用户输入了错误的数据，那么模拟将以默认值运行。

### 完成设置:主功能定义

您想要创建的最后一个函数是`main()`。这将确保当您在命令行上执行脚本时，脚本以正确的顺序运行。你可以在[中阅读更多关于`main()`的内容，在 Python](https://realpython.com/python-main-function/) 中定义主函数。你的`main()`应该是这样的:

```py
def main():
  # Setup
  random.seed(42)
  num_cashiers, num_servers, num_ushers = get_user_input()

  # Run the simulation
  env = simpy.Environment()
  env.process(run_theater(env, num_cashiers, num_servers, num_ushers))
  env.run(until=90)

  # View the results
  mins, secs = get_average_wait_time(wait_times)
  print(
      "Running simulation...",
      f"\nThe average wait time is {mins} minutes and {secs} seconds.",
  )
```

下面是`main()`的工作原理:

1.  **通过声明一个随机种子来设置**您的环境。这将确保您的输出看起来像你在本教程中看到的。
2.  **查询**你的程序的用户的一些输入。
3.  **创建**环境，并将其保存为变量`env`，这将在每个时间步中移动模拟。
4.  **告诉** `simpy`运行流程`run_theater()`，这将创建影院环境，并生成电影观众在其中穿行。
5.  **确定**您希望模拟运行多长时间。作为默认值，模拟设置为运行 90 分钟。
6.  **将`get_average_wait_time()`的输出**存储在两个变量`mins`和`secs`中。
7.  **使用** `print()`向用户显示结果。

至此，设置完成！

## 如何运行模拟

只需几行代码，您就可以看到模拟变得栩栩如生。但是首先，这里有一个到目前为止您已经定义的函数和类的概述:

*   **`Theater` :** 这个类定义作为你想要模拟的环境的蓝图。它决定了关于该环境的一些信息，比如哪些资源是可用的，以及哪些进程与它们相关联。

*   **`go_to_movies()` :** 这个函数明确地请求使用一个资源，通过相关的过程，然后把它释放给下一个电影观众。

*   **`run_theater()` :** 该功能控制模拟。它使用`Theater`类蓝图来创建一个剧院的实例，然后调用`go_to_movies()`来生成并在剧院中移动人们。

*   **`get_average_wait_time()` :** 该函数计算`moviegoer`通过电影院所花费的平均时间。

*   **`calculate_wait_time()` :** 该功能确保最终输出易于用户阅读。

*   **`get_user_input()` :** 该功能允许用户定义一些参数，比如有多少收银员可用。

*   **`main()` :** 这个函数确保你的脚本在命令行中正常运行。

现在，您只需要两行代码来调用您的主函数:

```py
if __name__ == '__main__':
    main()
```

这样，您的脚本就可以运行了！打开您的终端，导航到您存储`simulate.py`的位置，并运行以下命令:

```py
$ python simulate.py
Input # of cashiers working:
```

系统将提示您选择模拟所需的参数。下面是使用默认参数时的输出:

```py
$ python simulate.py
Input # of cashiers working: 1
Input # of servers working: 1
Input # of ushers working: 1
Running simulation...
The average wait time is 42 minutes and 53 seconds.
```

哇哦。等待的时间太长了！

[*Remove ads*](/account/join/)

## 何时改变现状

请记住，您的目标是向经理提出一个解决方案，说明他需要多少员工才能将等待时间控制在 10 分钟以内。为此，您将希望试验一下您的参数，看看哪些数字提供了最佳解决方案。

首先，尝试一些完全疯狂的事情，最大限度地利用资源！假设有 100 名收银员、100 名服务员和 100 名引座员在这个剧院工作。当然，这是不可能的，但是使用高得惊人的数字会很快告诉你系统的极限是什么。立即尝试:

```py
$ python simulate.py
Input # of cashiers working: 100
Input # of servers working: 100
Input # of ushers working: 100
Running simulation...
The average wait time is 3 minutes and 29 seconds.
```

即使您用尽了资源，也只能将等待时间减少到 3.5 分钟。现在试着改变数字，看看你是否能像经理要求的那样，把等待时间减少到 10 分钟。你想出了什么解决办法？您可以展开下面的代码块来查看一个可能的解决方案:



```py
$ python simulate.py
Input # of cashiers working: 9
Input # of servers working: 6
Input # of ushers working: 1
Running simulation...
The average wait time is 9 minutes and 60 seconds.
```

在这一点上，你可以向经理展示你的结果，并提出帮助改善剧院的建议。例如，为了降低成本，他可能想在剧院前面安装 10 个售票亭，而不是每晚都有 10 个收银员。

## 结论

在本教程中，您已经学习了如何使用`simpy`框架在 Python 中**构建和运行模拟**。您已经开始理解系统是如何让代理经历过程的，以及如何创建这些系统的虚拟表示来加强它们对拥塞和延迟的防御。虽然模拟的类型可以不同，但总体执行是相同的！你将能够把你在这里学到的东西应用到各种不同的场景中。

**现在你可以:**

*   **头脑风暴**一步一步模拟算法
*   **用`simpy`用 Python 创建**一个虚拟环境
*   **定义代表代理和进程的**函数
*   **更改模拟的**参数，以找到最佳解决方案

你可以用`simpy`做很多事情，所以不要让你的探索就此停止。将你所学到的应用到新的场景中。您的解决方案可以帮助人们节省宝贵的时间和金钱，所以深入了解一下，看看您还可以优化哪些流程！您可以通过单击下面的链接下载您在本教程中构建的脚本的源代码:

**下载代码:** [单击此处下载代码，您将使用](https://realpython.com/bonus/simpy/)在本教程中了解 SimPy。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 SimPy**](/courses/simulating-processes-simpy/) 用 Python 模拟真实世界的流程********