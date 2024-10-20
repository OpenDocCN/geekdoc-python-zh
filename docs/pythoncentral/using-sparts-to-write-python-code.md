# 使用 Sparts 编写 Python 代码

> 原文：<https://www.pythoncentral.io/using-sparts-to-write-python-code/>

[Sparts](https://pypi.org/project/sparts/) 是脸书开发的一个 Python 库，用于消除骨架代码。该库的主要目的是使实现服务变得容易，而不必编写多余的代码，因此您可以非常快速地创建原型。要学习 Sparts 的基础知识，请务必阅读[文档](http://pythonhosted.org/sparts/)。

Sparts 的工作方式是将任何服务分成两部分，核心“服务”和“任务”。任务通常指后台/离线处理，而服务包括任何共享功能。因此，要创建任何新的自定义逻辑，您只需子类化 VService (sparts.vtask.VService)并运行其 initFromCLI()即可。另一方面，任务(sparts.vtask.VTask)用于触发程序采取行动。没有它们，服务类将毫无用处。

总之，Sparts 核心的服务和任务结合在一起，使您的生活变得更加轻松，并且在将新服务实现到您的产品中或创建新原型时，需要编写更少的代码。在这里为你自己检查一下[。](https://github.com/facebook/sparts)