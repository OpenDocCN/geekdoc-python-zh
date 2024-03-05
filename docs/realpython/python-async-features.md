# Python 异步特性入门

> 原文：<https://realpython.com/python-async-features/>

听说过 Python 中的异步编程吗？您是否想了解更多关于 Python 异步特性的知识，以及如何在工作中使用它们？也许你甚至尝试过编写线程化程序并遇到一些问题。如果您想了解如何使用 Python 异步特性，那么您来对地方了。

在这篇文章中，你将了解到:

*   什么是同步程序
*   什么是**异步程序**
*   为什么你可能想写一个异步程序
*   如何使用 Python 异步特性

本文中的所有示例代码都已经用 Python 3.7.2 测试过了。您可以通过点击下面的链接获取一份副本进行跟进:

**下载代码:** [点击这里下载代码，您将在本教程中使用](https://realpython.com/bonus/async-features/)来学习 Python 中的异步特性。

## 了解异步编程

一个**同步程序**一次执行一个步骤。即使有条件分支、循环和函数调用，您仍然可以从一次执行一个步骤的角度来考虑代码。每一步完成后，程序就进入下一步。

这里有两个以这种方式工作的程序示例:

*   **批处理程序**通常被创建为同步程序。你得到一些输入，处理它，然后创造一些输出。步骤一个接一个，直到程序达到期望的输出。程序只需要注意步骤和它们的顺序。

*   命令行程序是在终端上运行的小而快速的程序。这些脚本用于创建一些东西，将一个东西转换成另一个东西，生成一个报告，或者列出一些数据。这可以表示为一系列的程序步骤，这些步骤按顺序执行，直到程序完成。

一个**异步程序**表现不同。它仍然一次执行一个步骤。不同之处在于，系统可能不会等待一个执行步骤完成后再继续下一个步骤。

这意味着程序将继续执行下一步，即使前一步还没有完成并且还在其他地方运行。这也意味着程序知道当前一个步骤结束运行时该做什么。

为什么要用这种方式编写程序呢？本文的其余部分将帮助您回答这个问题，并为您提供优雅地解决有趣的异步问题所需的工具。

[*Remove ads*](/account/join/)

### 构建同步网络服务器

web 服务器的基本工作单元或多或少与批处理相同。服务器将获得一些输入，处理它，并创建输出。作为同步程序编写，这将创建一个工作的 web 服务器。

这也将是一个绝对可怕的网络服务器。

为什么？在这种情况下，一个工作单元(输入、过程、输出)不是唯一的目的。真正的目的是尽可能快地处理数百甚至数千个单元的工作。这种情况可能会持续很长时间，几个工作单元甚至会同时到达。

同步 web 服务器可以做得更好吗？当然，您可以优化执行步骤，以便尽可能快地处理所有进来的工作。不幸的是，这种方法有局限性。结果可能是 web 服务器响应不够快，不能处理足够多的工作，甚至在工作堆积时超时。

**注意:**如果您尝试优化上述方法，您可能会发现其他限制。这些包括网络速度、文件 IO 速度、数据库查询速度和其他连接服务的速度，等等。这些的共同点是都是 IO 函数。所有这些项目都比 CPU 的处理速度慢几个数量级。

在同步程序中，如果一个执行步骤启动了一个数据库查询，那么在数据库查询返回之前，CPU 实际上是空闲的。对于面向批处理的程序，这在大多数情况下并不是优先考虑的事情。目标是处理 IO 操作的结果。通常，这比 IO 操作本身花费的时间更长。任何优化工作都将集中在处理工作上，而不是 IO 上。

异步编程技术允许您的程序通过释放 CPU 去做其他工作来利用相对较慢的 IO 进程。

### 用不同的方式思考编程

当你开始尝试理解异步编程时，你可能会看到很多关于阻塞或者编写[非阻塞代码](https://medium.com/vaidikkapoor/understanding-non-blocking-i-o-with-python-part-1-ec31a2e2db9b)的重要性的讨论。(就我个人而言，我很难从我询问的人和我阅读的文档中很好地掌握这些概念。)

什么是非阻塞代码？就此而言，什么是阻塞代码？这些问题的答案会帮助你编写一个更好的 web 服务器吗？如果是，你会怎么做？让我们来了解一下！

编写异步程序要求您以不同的方式思考编程。虽然这种新的思维方式可能很难理解，但它也是一种有趣的练习。这是因为现实世界几乎完全是异步的，你与它的互动方式也是如此。

想象一下:你是一位试图同时做几件事情的父母。你必须平衡支票簿，洗衣服，照看孩子。不知何故，你能够同时做所有这些事情，甚至不用考虑它！让我们来分解一下:

*   平衡支票簿是一项同步的任务。一步接着一步，直到完成。你一个人做所有的工作。

*   然而，你可以脱离支票簿去洗衣服。你卸下干衣机，将衣物从洗衣机移到干衣机，并在洗衣机中开始另一次洗涤。

*   使用洗衣机和烘干机是一项同步任务，但大部分工作发生在洗衣机和烘干机启动后的第*天。一旦你让他们开始工作，你就可以走开，回到支票簿的任务上。此时，洗衣机和烘干机的任务变成了**异步**。洗衣机和烘干机将独立运行，直到蜂鸣器响起(通知您该任务需要注意)。*

*   照看孩子是另一项异步任务。一旦他们被设置和播放，他们可以在很大程度上独立完成。当有人需要关注时，比如当有人饥饿或受伤时，这种情况就会改变。当你的一个孩子惊恐地大叫时，你会有所反应。孩子们是一个长期运行的高优先级任务。看着它们取代了你可能正在做的任何其他任务，比如支票簿或洗衣服。

这些例子有助于说明阻塞和非阻塞代码的概念。让我们从编程的角度来考虑这个问题。在这个例子中，你就像是中央处理器。当你移动要洗的衣服时，你(CPU)很忙，无法做其他工作，比如结算支票簿。但这没关系，因为任务相对较快。

另一方面，启动洗衣机和烘干机不会妨碍您执行其他任务。这是一个异步函数，因为你不必等待它完成。一旦开始，你就可以回到别的事情上去。这被称为上下文切换:你正在做的事情的上下文已经改变，洗衣机的蜂鸣器将在未来某个时候通知你洗衣任务完成。

作为一个人类，你一直都是这样工作的。你会自然而然地同时处理多件事情，而且经常不加思考。作为一名开发人员，诀窍在于如何将这种行为转换成做同样事情的代码。

## 编程家长:没有看起来那么容易！

如果你在上面的例子中认出了你自己(或者你的父母)，那就太好了！你已经在理解异步编程方面占了上风。同样，你能够很容易地在竞争任务之间切换上下文，选择一些任务并继续其他任务。现在你要试着把这种行为编程到虚拟父母中去！

### 思想实验#1:同步父母

你如何创建一个父程序以完全同步的方式完成上述任务？由于照看孩子是一项高优先级的任务，也许您的程序可以做到这一点。父母看着孩子，等待可能需要他们注意的事情发生。然而，在这种情况下，其他任何事情(如支票簿或衣物)都无法完成。

现在，你可以按照你想要的任何方式重新排列任务的优先级，但是在任何给定的时间，它们中只有一个会发生。这是同步、逐步方法的结果。就像上面描述的同步 web 服务器一样，这是可行的，但是这可能不是最好的生活方式。直到孩子们睡着了，父母才能完成任何其他任务。所有其他的任务都在之后发生，一直持续到深夜。(几个星期后，许多真正的父母可能会跳出窗外！)

[*Remove ads*](/account/join/)

### 思想实验#2:投票父母

如果您使用了[轮询](https://en.wikipedia.org/wiki/Polling_(computer_science))，那么您可以改变事情，以便完成多个任务。在这种方法中，父母会周期性地从当前任务中脱离出来，查看是否有其他任务需要关注。

让我们将轮询间隔设为大约 15 分钟。现在，每隔 15 分钟，你的父母就会检查洗衣机、烘干机或孩子是否需要注意。如果没有，那么家长可以回去工作的支票簿。然而，如果这些任务中的任何一项需要注意，父母会在回到支票簿前处理好。这个循环继续下去，直到轮询循环的下一次超时。

这种方法也很有效，因为多个任务引起了注意。然而，有几个问题:

1.  父母可能会花很多时间检查不需要注意的事情:洗衣机和烘干机还没有完成，除非发生意外，否则孩子们不需要任何注意。

2.  **家长可能会错过需要关注的已完成任务:**例如，如果洗衣机在轮询间隔开始时完成了其周期，那么它将在长达十五分钟内得不到任何关注！此外，照看孩子应该是最重要的任务。当事情可能会彻底出错时，他们无法忍受 15 分钟的无所事事。

您可以通过缩短轮询间隔来解决这些问题，但是现在您的父进程(CPU)将花费更多的时间在任务之间进行上下文切换。这是你开始达到收益递减点的时候。(再一次，像这样生活几个星期，嗯…看前面关于窗户和跳跃的评论。)

### 思想实验#3:线程父代

“如果我能克隆我自己就好了……”如果你是父母，那么你可能也有类似的想法！因为您正在编写虚拟父母，所以基本上可以通过使用线程来实现。这是一种允许一个程序的多个部分同时运行的机制。独立运行的每一段代码称为一个线程，所有线程共享相同的内存空间。

如果你把每个任务看作一个程序的一部分，那么你可以把它们分开，作为线程来运行。换句话说，您可以“克隆”父对象，为每个任务创建一个实例:照看孩子、监控洗衣机、监控烘干机以及平衡支票簿。所有这些“克隆”都是独立运行的。

这听起来是一个非常好的解决方案，但是这里也有一些问题。一个是你必须明确地告诉每个父实例在你的程序中做什么。这可能会导致一些问题，因为所有实例共享程序空间中的所有内容。

例如，假设父母 A 正在监控烘干机。父母 A 看到衣服是干的，所以他们控制了烘干机并开始卸载衣服。同时，父母 B 看到洗衣机已经洗好了，所以他们控制了洗衣机并开始脱衣服。然而，父母 B 也需要控制烘干机，以便他们可以将湿衣服放在里面。这是不可能发生的，因为父母 A 目前控制着烘干机。

不一会儿，家长 A 已经卸完衣服了。现在他们想控制洗衣机，开始把衣服放进空的烘干机。这也不可能发生，因为父 B 目前控制着洗衣机！

这两个家长现在[僵持](https://realpython.com/intro-to-python-threading/#deadlock)。双方都控制了自己的资源*和*想要控制对方的资源。他们将永远等待另一个父实例释放控制权。作为程序员，您必须编写代码来解决这种情况。

**注意:**线程程序允许你创建多个并行的执行路径，这些路径共享同一个内存空间。这既是优点也是缺点。线程之间共享的任何内存都受制于一个或多个试图同时使用同一个共享内存的线程。这可能会导致数据损坏、在无效状态下读取数据，以及数据通常很乱。

在线程编程中，上下文切换发生在系统控制下，而不是程序员。系统控制何时切换上下文，何时让线程访问共享数据，从而改变如何使用内存的上下文。所有这些类型的问题在线程代码中都是可以管理的，但是很难得到正确的结果，并且在错误的时候很难调试。

这是线程化可能引发的另一个问题。假设一个孩子受伤了，需要紧急护理。父母 C 被分配了照看孩子的任务，所以他们马上带走了孩子。在紧急护理中心，父母 C 需要开一张相当大的支票来支付看病的费用。

与此同时，家长 D 正在家里处理支票簿。他们不知道这张大额支票已经开出，所以当家庭支票账户突然透支时，他们感到非常惊讶！

记住，这两个父实例在同一个程序中工作。家庭支票账户是一种共享资源，所以你必须想办法让照看孩子的父母通知收支平衡的父母。否则，您需要提供某种锁定机制，以便支票簿资源一次只能由一个父节点使用，并进行更新。

## 实践中使用 Python 异步特性

现在，您将采用上述思维实验中概述的一些方法，并将它们转化为有效的 Python 程序。

本文中的所有例子都已经用 Python 3.7.2 测试过了。`requirements.txt`文件指出了运行所有示例需要安装哪些模块。如果您尚未下载该文件，现在可以下载:

**下载代码:** [点击这里下载代码，您将在本教程中使用](https://realpython.com/bonus/async-features/)来学习 Python 中的异步特性。

您可能还想建立一个 [Python 虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)来运行代码，这样您就不会干扰您的系统 Python。

[*Remove ads*](/account/join/)

### 同步编程

第一个例子展示了一种有点做作的方法，让任务从队列中检索工作并处理该工作。Python 中的队列是一种很好的先进先出数据结构。它提供了将东西放入队列并按照插入的顺序取出它们的方法。

在这种情况下，工作是从队列中获取一个数字，并让循环计数达到该数字。当循环开始时，它打印到控制台，并再次输出总数。这个程序演示了多个同步任务处理队列中的工作的一种方法。

存储库中名为`example_1.py`的程序完整列出如下:

```py
 1import queue
 2
 3def task(name, work_queue):
 4    if work_queue.empty():
 5        print(f"Task {name} nothing to do")
 6    else:
 7        while not work_queue.empty():
 8            count = work_queue.get()
 9            total = 0
10            print(f"Task {name} running")
11            for x in range(count):
12                total += 1
13            print(f"Task {name} total: {total}")
14
15def main():
16    """
17 This is the main entry point for the program
18 """
19    # Create the queue of work
20    work_queue = queue.Queue()
21
22    # Put some work in the queue
23    for work in [15, 10, 5, 2]:
24        work_queue.put(work)
25
26    # Create some synchronous tasks
27    tasks = [(task, "One", work_queue), (task, "Two", work_queue)]
28
29    # Run the tasks
30    for t, n, q in tasks:
31        t(n, q)
32
33if __name__ == "__main__":
34    main()
```

让我们看看每一行都做了什么:

*   **线 1** 导入`queue`模块。这是程序存储任务要完成的工作的地方。
*   **第 3 行到第 13 行**定义`task()`。这个函数从`work_queue`中提取工作，并处理工作，直到没有其他工作可做。
*   **第 15 行**定义`main()`运行程序任务。
*   **第 20 行**创造了`work_queue`。所有任务都使用这个共享资源来检索工作。
*   **第 23 至 24 行**将工作放入`work_queue`。在这种情况下，它只是要处理的任务的值的随机计数。
*   第 27 行创建了一个任务元组的[列表](https://realpython.com/python-lists-tuples/)，带有那些任务将被传递的参数值。
*   **第 30 到 31 行**遍历任务元组列表，调用每个元组并传递之前定义的参数值。
*   **第 34 行**调用`main()`运行程序。

这个程序中的任务只是一个接受字符串和队列作为参数的函数。当执行时，它在队列中寻找任何要处理的东西。如果有工作要做，那么它从队列中取出值，开始一个 [`for`循环](https://realpython.com/courses/python-for-loop/)来计数到那个值，并在最后输出总数。它继续从队列中获取工作，直到没有剩余工作并退出。

当这个程序运行时，它产生如下所示的输出:

```py
Task One running
Task One total: 15
Task One running
Task One total: 10
Task One running
Task One total: 5
Task One running
Task One total: 2
Task Two nothing to do
```

这表明`Task One`做了所有的工作。`Task One`在`task()`中命中的 [`while`循环](https://realpython.com/courses/mastering-while-loops/)消耗队列中的所有工作并处理它。当这个循环退出时，`Task Two`就有机会运行。但是，它发现队列是空的，所以`Task Two`打印一个声明，说它没有任何事情，然后退出。代码中没有任何东西允许`Task One`和`Task Two`切换上下文并一起工作。

### 简单协作并发

这个程序的下一个版本允许这两个任务一起工作。添加一个`yield`语句意味着循环将在指定点产生控制，同时仍然保持其上下文。这样，让步任务可以在以后重新启动。

`yield`语句将`task()`变成了[发生器](https://realpython.com/introduction-to-python-generators/)。在 Python 中，调用生成器函数就像调用任何其他函数一样，但是当执行`yield`语句时，控制被返回给函数的调用者。这本质上是一个上下文切换，因为控制权从生成器函数转移到了调用者。

有趣的是，通过调用生成器上的`next()`，可以将控制权*交还给生成器函数*。这是一个返回到生成器函数的上下文切换，它继续执行所有在`yield`之前定义的函数[变量](https://realpython.com/python-variables/)。

[`main()`](https://realpython.com/python-main-function/) 中的`while`循环在调用`next(t)`时利用了这一点。此语句从任务先前产生的点重新启动任务。所有这些都意味着当上下文切换发生时，您处于控制之中:当在`task()`中执行`yield`语句时。

这是一种多任务合作的形式。程序正在放弃对其当前上下文的控制，以便其他东西可以运行。在这种情况下，它允许`main()`中的`while`循环运行`task()`的两个实例作为生成器函数。每个实例都使用同一队列中的工作。这是一种聪明的做法，但是要得到与第一个程序相同的结果也需要做大量的工作。程序`example_2.py`演示了这个简单的[并发](https://realpython.com/python-concurrency/)，如下所示:

```py
 1import queue
 2
 3def task(name, queue):
 4    while not queue.empty():
 5        count = queue.get()
 6        total = 0
 7        print(f"Task {name} running")
 8        for x in range(count):
 9            total += 1
10            yield
11        print(f"Task {name} total: {total}")
12
13def main():
14    """
15 This is the main entry point for the program
16 """
17    # Create the queue of work
18    work_queue = queue.Queue()
19
20    # Put some work in the queue
21    for work in [15, 10, 5, 2]:
22        work_queue.put(work)
23
24    # Create some tasks
25    tasks = [task("One", work_queue), task("Two", work_queue)]
26
27    # Run the tasks
28    done = False
29    while not done:
30        for t in tasks:
31            try:
32                next(t)
33            except StopIteration:
34                tasks.remove(t)
35            if len(tasks) == 0:
36                done = True
37
38if __name__ == "__main__":
39    main()
```

下面是上面代码中发生的情况:

*   **第 3 行到第 11 行**像以前一样定义`task()`，但是在第 10 行增加了`yield`将函数变成了生成器。在这里进行上下文切换，并且控制被交还给`main()`中的`while`循环。
*   **第 25 行**创建任务列表，但是与您在前面的示例代码中看到的方式略有不同。在这种情况下，调用每个任务时，会将其参数输入到`tasks`列表变量中。这是第一次运行`task()`发生器功能所必需的。
*   **第 31 到 36 行**是对`main()`中`while`循环的修改，使`task()`可以协同运行。这是当它让步时控制返回到每个`task()`实例的地方，允许循环继续并运行另一个任务。
*   **第 32 行**将控制权交还给`task()`，并在`yield`被调用后继续执行。
*   **第 36 行**设置`done`变量。当所有任务完成并从`tasks`中移除后，`while`循环结束。

这是运行该程序时产生的输出:

```py
Task One running
Task Two running
Task Two total: 10
Task Two running
Task One total: 15
Task One running
Task Two total: 5
Task One total: 2
```

您可以看到`Task One`和`Task Two`都在运行并消耗队列中的工作。这就是我们想要的，因为两个任务都是处理工作，每个任务负责队列中的两个项目。这很有趣，但同样，要达到这些结果需要做相当多的工作。

这里的技巧是使用`yield`语句，它将`task()`变成一个生成器并执行上下文切换。程序使用这个上下文切换来控制`main()`中的`while`循环，允许一个任务的两个实例协同运行。

注意`Task Two`如何首先输出它的总数。这可能会让您认为任务是异步运行的。然而，这仍然是一个同步程序。它的结构使得这两个任务可以来回交换上下文。`Task Two`先输出总数的原因是它只数到 10，而`Task One`数到 15。`Task Two`简单地首先到达它的总数，所以它在`Task One`之前打印它的输出到控制台。

**注意:**从这一点开始的所有示例代码都使用一个名为 [codetiming](https://pypi.org/project/codetiming/) 的模块来计时并输出代码段执行的时间。这里有一篇关于 RealPython 的很棒的文章[深入讨论了 codetiming 模块以及如何使用它。](https://realpython.com/python-timer/)

这个模块是 Python 包索引的一部分，由 [Geir Arne Hjelle](https://realpython.com/team/gahjelle/) 构建，他是*真实 Python* 团队的一员。Geir Arne 对我评论和建议本文的内容帮助很大。如果您正在编写需要包含计时功能的代码，Geir Arne 的 codetiming 模块非常值得一看。

要使 codetiming 模块在下面的例子中可用，您需要安装它。这可以通过`pip`命令:`pip install codetiming`或`pip install -r requirements.txt`命令来完成。`requirements.txt`文件是示例代码库的一部分。

[*Remove ads*](/account/join/)

### 具有阻塞调用的协作并发

程序的下一个版本与上一个版本相同，除了在你的任务循环体中增加了一个 [`time.sleep(delay)`](https://realpython.com/python-sleep/) 。这将基于从工作队列中检索的值向任务循环的每次迭代添加延迟。延迟模拟任务中发生阻塞调用的效果。

阻塞调用是一段时间内阻止 CPU 做任何事情的代码。在上面的思维实验中，如果父母在完成之前不能脱离平衡支票簿，那将是一个阻塞呼叫。

`time.sleep(delay)`在这个例子中做同样的事情，因为 CPU 除了等待延迟到期之外，不能做任何其他事情。

```py
 1import time
 2import queue
 3from codetiming import Timer
 4
 5def task(name, queue):
 6    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
 7    while not queue.empty():
 8        delay = queue.get()
 9        print(f"Task {name} running")
10        timer.start()
11        time.sleep(delay)
12        timer.stop()
13        yield
14
15def main():
16    """
17 This is the main entry point for the program
18 """
19    # Create the queue of work
20    work_queue = queue.Queue()
21
22    # Put some work in the queue
23    for work in [15, 10, 5, 2]:
24        work_queue.put(work)
25
26    tasks = [task("One", work_queue), task("Two", work_queue)]
27
28    # Run the tasks
29    done = False
30    with Timer(text="\nTotal elapsed time: {:.1f}"):
31        while not done:
32            for t in tasks:
33                try:
34                    next(t)
35                except StopIteration:
36                    tasks.remove(t)
37                if len(tasks) == 0:
38                    done = True
39
40if __name__ == "__main__":
41    main()
```

下面是上面代码的不同之处:

*   **行 1** 导入 [`time`模块](https://realpython.com/python-time-module/)给程序访问`time.sleep()`。
*   **第 3 行**从`codetiming`模块导入`Timer`代码。
*   **第 6 行**创建了`Timer`实例，用于测量任务循环的每次迭代所用的时间。
*   **第 10 行**启动`timer`实例
*   **第 11 行**改变`task()`以包括一个`time.sleep(delay)`来模拟 IO 延迟。这取代了在`example_1.py`中进行计数的`for`循环。
*   **第 12 行**停止`timer`实例，输出调用`timer.start()`后经过的时间。
*   **第 30 行**创建一个`Timer` [上下文管理器](https://realpython.com/python-with-statement/)，它将输出整个 while 循环执行所用的时间。

当您运行该程序时，您将看到以下输出:

```py
Task One running
Task One elapsed time: 15.0
Task Two running
Task Two elapsed time: 10.0
Task One running
Task One elapsed time: 5.0
Task Two running
Task Two elapsed time: 2.0

Total elapsed time: 32.0
```

和以前一样，`Task One`和`Task Two`都在运行，消耗队列中的工作并进行处理。然而，即使增加了延迟，您可以看到协作并发并没有给您带来任何好处。延迟会停止整个程序的处理，CPU 只是等待 IO 延迟结束。

这正是 Python 异步文档中阻塞代码的含义。你会注意到，运行整个程序所花费的时间就是所有延迟的累计时间。以这种方式运行任务并不成功。

### 具有非阻塞调用的协作并发

这个程序的下一个版本已经做了相当多的修改。它使用 Python 3 中提供的 [asyncio/await](https://realpython.com/async-io-python/) 来利用 Python 异步特性。

`time`和`queue`模块已被替换为`asyncio`组件。这使您的程序可以访问异步友好(非阻塞)睡眠和队列功能。对`task()`的更改通过在第 4 行添加前缀`async`将其定义为异步。这向 Python 表明该函数将是异步的。

另一个大的变化是删除了`time.sleep(delay)`和`yield`语句，用`await asyncio.sleep(delay)`代替它们。这创建了一个非阻塞延迟，它将执行上下文切换回调用者`main()`。

`main()`内的`while`循环不再存在。不是`task_array`，而是有一个`await asyncio.gather(...)`的调用。这告诉了`asyncio`两件事:

1.  基于`task()`创建两个任务，并开始运行它们。
2.  请等待这两项都完成后再继续。

程序的最后一行`asyncio.run(main())`运行`main()`。这就产生了所谓的[事件循环](https://realpython.com/lessons/asyncio-event-loop/)。这个循环将运行`main()`，它又将运行`task()`的两个实例。

事件循环是 Python 异步系统的核心。它运行所有的代码，包括`main()`。当任务代码执行时，CPU 忙于工作。当到达 [`await`关键字](https://realpython.com/python-keywords/#the-await-keyword)时，发生上下文切换，并且控制传递回事件循环。事件循环查看所有等待事件的任务(在这种情况下，是一个`asyncio.sleep(delay)`超时),并将控制权传递给一个带有就绪事件的任务。

`await asyncio.sleep(delay)`对于 CPU 来说是非阻塞的。CPU 不是等待延迟超时，而是在事件循环任务队列中注册一个睡眠事件，并通过将控制传递给事件循环来执行上下文切换。事件循环不断寻找已完成的事件，并将控制传递回等待该事件的任务。通过这种方式，如果有工作，CPU 可以保持忙碌，而事件循环则监视将来会发生的事件。

**注意:**一个异步程序运行在一个执行的单线程中。影响数据的从一段代码到另一段代码的上下文切换完全在您的控制之中。这意味着您可以在进行上下文切换之前原子化并完成所有共享内存数据访问。这简化了线程代码中固有的共享内存问题。

下面列出了`example_4.py`代码:

```py
 1import asyncio
 2from codetiming import Timer
 3
 4async def task(name, work_queue):
 5    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
 6    while not work_queue.empty():
 7        delay = await work_queue.get()
 8        print(f"Task {name} running")
 9        timer.start()
10        await asyncio.sleep(delay)
11        timer.stop()
12
13async def main():
14    """
15 This is the main entry point for the program
16 """
17    # Create the queue of work
18    work_queue = asyncio.Queue()
19
20    # Put some work in the queue
21    for work in [15, 10, 5, 2]:
22        await work_queue.put(work)
23
24    # Run the tasks
25    with Timer(text="\nTotal elapsed time: {:.1f}"):
26        await asyncio.gather(
27            asyncio.create_task(task("One", work_queue)),
28            asyncio.create_task(task("Two", work_queue)),
29        )
30
31if __name__ == "__main__":
32    asyncio.run(main())
```

下面是这个程序和`example_3.py`的不同之处:

*   **第 1 行**导入`asyncio`以获得对 Python 异步功能的访问。这取代了`time`导入。
*   **第 2 行**从`codetiming`模块导入`Timer`代码。
*   **第 4 行**显示在`task()`定义前添加了`async`关键字。这通知程序`task`可以异步运行。
*   **第 5 行**创建了`Timer`实例，用于测量任务循环的每次迭代所用的时间。
*   **第 9 行**启动`timer`实例
*   **第 10 行**用非阻塞`asyncio.sleep(delay)`替换`time.sleep(delay)`，这也将控制权(或切换上下文)交还给主事件循环。
*   **第 11 行**停止`timer`实例，输出调用`timer.start()`后经过的时间。
*   **第 18 行**创建非阻塞异步`work_queue`。
*   **第 21 到 22 行**使用`await`关键字以异步方式将工作放入`work_queue`中。
*   **第 25 行**创建一个`Timer`上下文管理器，它将输出整个 while 循环执行所用的时间。
*   **第 26 到 29 行**创建两个任务并将它们收集在一起，因此程序将等待两个任务都完成。
*   **第 32 行**启动程序异步运行。它还会启动内部事件循环。

当您查看这个程序的输出时，请注意`Task One`和`Task Two`是如何同时启动的，然后等待模拟 IO 调用:

```py
Task One running
Task Two running
Task Two total elapsed time: 10.0
Task Two running
Task One total elapsed time: 15.0
Task One running
Task Two total elapsed time: 5.0
Task One total elapsed time: 2.0

Total elapsed time: 17.0
```

这表明`await asyncio.sleep(delay)`是非阻塞的，其他工作正在进行。

在程序结束时，您会注意到总运行时间实际上是运行`example_3.py`所用时间的一半。这就是使用 Python 异步特性的程序的优势！每个任务能够同时运行`await asyncio.sleep(delay)`。程序的总执行时间现在小于其各部分的总和。你已经脱离了同步模式！

[*Remove ads*](/account/join/)

### 同步(阻塞)HTTP 调用

这个项目的下一个版本是一种进步，也是一种倒退。该程序通过向一系列 URL 发出 HTTP 请求并获取页面内容，用真正的 IO 做一些实际的工作。然而，它是以阻塞(同步)的方式这样做的。

该程序已被修改为导入[美妙的`requests`模块](https://realpython.com/python-requests/)来发出实际的 HTTP 请求。此外，队列现在包含一个 URL 列表，而不是数字。另外，`task()`不再递增计数器。相反，`requests`从队列中获取一个 URL 的内容，并打印出这样做需要多长时间。

下面列出了`example_5.py`代码:

```py
 1import queue
 2import requests
 3from codetiming import Timer
 4
 5def task(name, work_queue):
 6    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
 7    with requests.Session() as session:
 8        while not work_queue.empty():
 9            url = work_queue.get()
10            print(f"Task {name} getting URL: {url}")
11            timer.start()
12            session.get(url)
13            timer.stop()
14            yield
15
16def main():
17    """
18 This is the main entry point for the program
19 """
20    # Create the queue of work
21    work_queue = queue.Queue()
22
23    # Put some work in the queue
24    for url in [
25        "http://google.com",
26        "http://yahoo.com",
27        "http://linkedin.com",
28        "http://apple.com",
29        "http://microsoft.com",
30        "http://facebook.com",
31        "http://twitter.com",
32    ]:
33        work_queue.put(url)
34
35    tasks = [task("One", work_queue), task("Two", work_queue)]
36
37    # Run the tasks
38    done = False
39    with Timer(text="\nTotal elapsed time: {:.1f}"):
40        while not done:
41            for t in tasks:
42                try:
43                    next(t)
44                except StopIteration:
45                    tasks.remove(t)
46                if len(tasks) == 0:
47                    done = True
48
49if __name__ == "__main__":
50    main()
```

下面是这个程序中发生的事情:

*   **第 2 行**导入`requests`，提供了一种便捷的 HTTP 调用方式。
*   **第 3 行**从`codetiming`模块导入`Timer`代码。
*   **第 6 行**创建了`Timer`实例，用于测量任务循环的每次迭代所用的时间。
*   **第 11 行**启动`timer`实例
*   **第 12 行**引入了一个延迟，类似于`example_3.py`。然而，这一次它调用了`session.get(url)`，返回从`work_queue`获取的 URL 的内容。
*   **第 13 行**停止`timer`实例，输出调用`timer.start()`后经过的时间。
*   **第 23 到 32 行**将 URL 列表放入`work_queue`。
*   **第 39 行**创建一个`Timer`上下文管理器，它将输出整个 while 循环执行所用的时间。

当您运行该程序时，您将看到以下输出:

```py
Task One getting URL: http://google.com
Task One total elapsed time: 0.3
Task Two getting URL: http://yahoo.com
Task Two total elapsed time: 0.8
Task One getting URL: http://linkedin.com
Task One total elapsed time: 0.4
Task Two getting URL: http://apple.com
Task Two total elapsed time: 0.3
Task One getting URL: http://microsoft.com
Task One total elapsed time: 0.5
Task Two getting URL: http://facebook.com
Task Two total elapsed time: 0.5
Task One getting URL: http://twitter.com
Task One total elapsed time: 0.4

Total elapsed time: 3.2
```

就像早期版本的程序一样，`yield`将`task()`变成了一个生成器。它还执行上下文切换，让另一个任务实例运行。

每个任务从工作队列中获取一个 URL，检索页面的内容，并报告获取该内容花费了多长时间。

和以前一样，`yield`允许您的两个任务协同运行。然而，由于这个程序是同步运行的，每个`session.get()`调用都会阻塞 CPU，直到页面被检索到。**注意最后运行整个程序所花费的总时间。**这将对下一个例子有意义。

### 异步(非阻塞)HTTP 调用

这个版本的程序修改了以前的版本，使用 Python 异步特性。它还导入了 [`aiohttp`](https://aiohttp.readthedocs.io/en/stable/) 模块，这是一个使用`asyncio`以异步方式发出 HTTP 请求的库。

这里的任务已经修改，删除了`yield`调用，因为进行 HTTP `GET`调用的代码不再阻塞。它还执行上下文切换回事件循环。

下面列出了`example_6.py`程序:

```py
 1import asyncio
 2import aiohttp
 3from codetiming import Timer
 4
 5async def task(name, work_queue):
 6    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
 7    async with aiohttp.ClientSession() as session:
 8        while not work_queue.empty():
 9            url = await work_queue.get()
10            print(f"Task {name} getting URL: {url}")
11            timer.start()
12            async with session.get(url) as response:
13                await response.text()
14            timer.stop()
15
16async def main():
17    """
18 This is the main entry point for the program
19 """
20    # Create the queue of work
21    work_queue = asyncio.Queue()
22
23    # Put some work in the queue
24    for url in [
25        "http://google.com",
26        "http://yahoo.com",
27        "http://linkedin.com",
28        "http://apple.com",
29        "http://microsoft.com",
30        "http://facebook.com",
31        "http://twitter.com",
32    ]:
33        await work_queue.put(url)
34
35    # Run the tasks
36    with Timer(text="\nTotal elapsed time: {:.1f}"):
37        await asyncio.gather(
38            asyncio.create_task(task("One", work_queue)),
39            asyncio.create_task(task("Two", work_queue)),
40        )
41
42if __name__ == "__main__":
43    asyncio.run(main())
```

下面是这个程序中发生的事情:

*   **第 2 行**导入了`aiohttp`库，它提供了一种异步方式来进行 HTTP 调用。
*   **第 3 行**从`codetiming`模块导入`Timer`代码。
*   **第 5 行**将`task()`标记为异步函数。
*   **第 6 行**创建了`Timer`实例，用于测量任务循环的每次迭代所用的时间。
*   **第 7 行**创建一个`aiohttp`会话上下文管理器。
*   **第 8 行**创建一个`aiohttp`响应上下文管理器。它还对来自`work_queue`的 URL 进行 HTTP `GET`调用。
*   **第 11 行**启动`timer`实例
*   **第 12 行**使用会话异步获取从 URL 检索的文本。
*   **第 13 行**停止`timer`实例，输出调用`timer.start()`后经过的时间。
*   **第 39 行**创建一个`Timer`上下文管理器，它将输出整个 while 循环执行所用的时间。

当您运行该程序时，您将看到以下输出:

```py
Task One getting URL: http://google.com
Task Two getting URL: http://yahoo.com
Task One total elapsed time: 0.3
Task One getting URL: http://linkedin.com
Task One total elapsed time: 0.3
Task One getting URL: http://apple.com
Task One total elapsed time: 0.3
Task One getting URL: http://microsoft.com
Task Two total elapsed time: 0.9
Task Two getting URL: http://facebook.com
Task Two total elapsed time: 0.4
Task Two getting URL: http://twitter.com
Task One total elapsed time: 0.5
Task Two total elapsed time: 0.3

Total elapsed time: 1.7
```

看一下总的运行时间，以及获取每个 URL 内容的单个时间。您将看到持续时间大约是所有 HTTP `GET`调用累计时间的一半。这是因为 HTTP `GET`调用是异步运行的。换句话说，通过允许 CPU 一次发出多个请求，您可以更有效地利用 CPU。

因为 CPU 非常快，这个例子可能会创建和 URL 一样多的任务。在这种情况下，程序的运行时间将是最慢的 URL 检索时间。

[*Remove ads*](/account/join/)

## 结论

本文提供了让异步编程技术成为您的技能的一部分所需的工具。使用 Python 异步特性，您可以对何时发生上下文切换进行编程控制。这意味着您可能在线程编程中遇到的许多棘手问题都更容易处理。

异步编程是一个强大的工具，但并不是对每种程序都有用。例如，如果你正在编写一个计算圆周率的程序，那么异步代码就帮不了你。那种程序是 CPU 绑定的，没有多少 IO。然而，如果您试图实现一个执行 IO(比如文件或网络访问)的服务器或程序，那么使用 Python 异步特性会带来巨大的不同。

**总结一下，你已经学会:**

*   什么是同步程序
*   异步程序与众不同，但同样强大且易于管理
*   为什么你可能想写异步程序
*   如何使用 Python 中内置的异步特性

您可以获得本教程中使用的所有示例程序的代码:

**下载代码:** [点击这里下载代码，您将在本教程中使用](https://realpython.com/bonus/async-features/)来学习 Python 中的异步特性。

现在你已经掌握了这些强大的技能，你可以把你的程序带到下一个层次！******