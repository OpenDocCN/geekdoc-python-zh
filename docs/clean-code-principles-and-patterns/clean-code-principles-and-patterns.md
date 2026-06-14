

# 《代码整洁之道：原则与模式》

## Python 版

Petri Silén

本书可在 [http://leanpub.com/cleancodeprinciplesandpatternspythonedition](http://leanpub.com/cleancodeprinciplesandpatternspythonedition) 购买

本版本发布于 2023-10-29

这是一本 [Leanpub](https://leanpub.com) 图书。Leanpub 通过精益出版流程赋能作者和出版商。[精益出版](https://leanpub.com/leanpublishing) 是指使用轻量级工具和多次迭代来发布进行中的电子书，以获取读者反馈，不断调整直到找到合适的书籍，并在完成后建立影响力。

© 2023 Petri Silen

## 推广本书！

请帮助 Petri Silén 在 [Twitter](https://twitter.com) 上宣传本书！

本书的推荐标签是 #cleancodeprinciplesandpatterns。

点击此链接在 Twitter 上搜索此标签，查看其他人对本书的评价：

[#cleancodeprinciplesandpatterns](https://twitter.com/search?q=%23cleancodeprinciplesandpatterns)

## 目录

- 1：关于作者

## 目录

- 3.14.2：避免使用 0.x 版本原则

## 目录

- 4.6.2.7：命名函数参数

## 目录

- 4.10.3.4：迭代器模式

## 目录

- 5.5：函数单一返回原则

## 目录

- 6.1.1.4：Web UI 组件单元测试

## 目录

- 7.4.12：日志记录

## 目录

9.1.3.2：多对多关系

## 11.10.4：测试自动化开发人员

## 1：关于作者

**Petri Silén** 是一位经验丰富的软件开发人员，目前在芬兰的诺基亚网络公司工作，拥有近30年的行业经验。他从事过前端和后端开发，在多种编程语言方面具备扎实的能力，包括C++、Java、Python和JavaScript/TypeScript。他于1995年在诺基亚电信开始了他的职业生涯。在最初的几年里，他使用C++为全球主要电信客户（包括T-Mobile、Orange、Vodafone和Claro等公司）开发了一款名为“Traffica”的实时移动网络分析产品。最初的产品用于监控2G电路交换核心网络和GPRS分组交换核心网络。后来，为Traffica添加了功能以覆盖新的网络技术，如3G电路交换和分组核心网络、3G无线网络和4G/LTE。之后，他使用Java和Web技术（包括jQuery和React）为Traffica开发了新功能。在过去的几年里，他使用Java和C++为下一代客户与网络洞察（CNI）产品开发了云原生容器化微服务，该产品被Verizon、AT&T、USCC和KDDI等主要通信服务提供商使用。他近年来贡献的主要应用领域包括基于KPI的实时告警、KPI异常检测和可配置的实时数据导出。

在业余时间，他使用React、Redux、TypeScript和Jakarta EE开发了一个数据可视化应用程序。他还用TypeScript为Node.js开发了一个安全优先的云原生微服务框架。他喜欢照顾他的猫Kaapo，散步，打网球和羽毛球，冬天滑雪，以及在电视上看足球和冰球比赛。

## 2：简介

本书教你如何编写整洁的代码。它以非常实用的方式介绍了软件设计和开发原则与模式。本书适合初级和高级开发人员。需要具备Python编程的一些基础知识。本书中的所有示例均使用Python呈现，除了一些与前端代码相关的示例使用JavaScript/TypeScript。本书内容分为十一章。本书中的所有Python示例都需要Python 3.11或更高版本。这是一本主要面向软件开发人员的书。因此，某些主题并未详尽涵盖。这包括与架构、DevSecOps、端到端（E2E）和非功能性测试相关的主题。这些主题与软件架构师、DevOps专家和测试/QA工程师最为相关，但我也想在本书中涵盖它们，因为对与软件开发本身密切相关的主题有基本的了解总是有益的。

本书介绍了许多原则、最佳实践和模式。一次阅读可能难以全部掌握，这也不是目的。你可以挑选对自己最相关的原则和模式，吸收它们，并尝试在日常编码中应用。你随时可以回到本书学习额外的原则和模式。其中一些原则/模式/实践是主观的，可以讨论，但我只将那些我自己使用过或会使用的原则、模式和实践放入本书。

*第二章*是关于*架构设计原则*，这些原则使得开发真正的云原生微服务成为可能。描述的第一个架构设计原则是单一职责原则，它定义了软件在其抽象层面上应只负责一件事。然后介绍了微服务、客户端、API和库的统一命名原则。封装原则定义了每个软件组件应如何将其内部状态隐藏在公共API之后。介绍了服务聚合原则，并详细解释了高级微服务如何聚合低级微服务。讨论了架构模式，如事件溯源、命令查询职责分离（CQRS）和分布式事务。分布式事务通过使用saga编排模式和saga编舞模式的示例进行了介绍。你将获得如何在架构层面避免代码重复的答案。外部化配置原则描述了在现代环境中应如何处理服务配置。我们讨论了服务替换原则，该原则指出微服务使用的依赖服务应易于替换。从无状态性、弹性、高可用性、可观察性和自动扩展的角度讨论了自动驾驶微服务的重要性。在本章末尾，讨论了微服务相互通信的不同方式。提出了关于如何对软件组件进行版本控制的若干规则。本章最后讨论了为什么限制软件系统中使用的技术数量是有帮助的。

*第三章*介绍了*面向对象设计原则*。我们以面向对象编程概念和编程范式开始本章，然后是SOLID原则：单一职责原则、开闭原则、里氏替换原则、接口隔离原则和依赖倒置原则。每个SOLID原则都通过现实但简单的示例进行介绍。统一命名原则定义了命名接口、类、函数、函数对、布尔函数（谓词）、构建器、工厂、转换和生命周期方法的统一方式。封装原则描述了一个类应如何封装其内部状态，以及不可变性如何有助于确保状态封装。封装原则还讨论了不泄露对象内部状态的重要性。对象组合原则定义了应优先选择组合而非继承。通过两个实际示例介绍了领域驱动设计（DDD）。《GoF设计模式》一书中的所有设计模式都通过现实但简单的示例进行了介绍。介绍了“不要询问，告诉”原则，作为避免特性嫉妒设计坏味道的一种方式。本章还讨论了避免原始类型执念以及使用语义验证函数参数的好处。本章最后介绍了依赖注入原则和避免代码重复原则，也称为不要重复自己（DRY）原则。

*第四章*是关于*编码原则*。本章从代码中统一命名变量的原则开始。为整数、浮点数、布尔值、字符串、枚举和集合变量介绍了统一的命名约定。还为映射、对、元组、对象、可选值和回调函数定义了命名约定。通过示例介绍了统一源代码仓库结构原则。接下来，避免注释原则定义了从代码中移除不必要注释的具体方法。介绍了以下具体操作：正确命名事物、返回命名值、返回类型别名、为布尔表达式提取常量、为复杂表达式提取常量、提取枚举值以及提取函数。本章讨论了使用类型提示的好处。我们讨论了最常见的重构技术：重命名、提取方法、提取变量、用多态替换条件语句以及引入参数对象。描述了静态代码分析的重要性，并列出了最流行的静态代码分析工具。列出了最常见的静态代码分析问题及其首选的纠正方法。在代码中正确处理错误和异常是基础性的，但很容易被遗忘或处理不当。本章指导如何处理错误和异常，以及如何通过返回布尔失败指示器、可选值或错误对象来返回错误。本章指导如何调整代码以适应所需的错误处理机制以及如何以函数式方式处理错误。介绍了避免差一错误的方法。指导读者处理从谷歌搜索找到的网页复制或由AI生成的代码的情况。本章最后讨论了代码优化：何时以及如何优化。

*第五章*专门介绍*测试原则*。本章从功能测试金字塔的介绍开始。然后我们介绍单元测试并指导如何使用测试驱动开发（TDD）。我们给出了带有模拟的单元测试示例。在介绍软件组件集成测试时，我们讨论了行为驱动开发（BDD）和用于描述特性的Gherkin语言。使用Behave和Postman API开发平台给出了集成测试示例。本章还讨论了UI软件组件的集成测试。我们以使用Docker Compose设置集成测试环境的示例结束了集成测试部分。最后，讨论了端到端（E2E）测试的目的并给出了一些示例。本章最后讨论了非功能性测试。涵盖了以下类别的非功能性测试：性能测试、负载测试、压力测试、可扩展性测试、安全性和可用性测试。

功能测试涵盖得更为详细：性能测试、稳定性测试、可靠性测试、安全性测试、压力测试和可扩展性测试。

*第六章*处理*安全原则*。介绍了威胁建模流程，并举例说明如何为一个简单的API微服务进行威胁建模。实现了一个完整的前端OpenID Connect/OAuth 2.0认证与授权示例，使用TypeScript、Vue.js和Keycloak。接着我们讨论了后端应如何通过验证JWT来处理授权。本章最后讨论了最重要的安全特性：密码策略、加密、拒绝服务攻击防范、SQL注入防范、安全配置、自动漏洞扫描、完整性、错误处理、审计日志和输入验证。

*第七章*关于*API设计原则*。首先，我们探讨面向前端的API设计原则。讨论了如何设计JSON-RPC、REST和GraphQL API。此外，还通过使用服务器发送事件（SSE）和WebSocket协议的真实示例，介绍了基于订阅的API和实时API。本章最后一部分讨论了微服务间API设计和事件驱动架构。介绍了gRPC作为同步的微服务间通信方法，并展示了仅请求和请求-响应异步API的示例。

*第八章*讨论*数据库及相关原则*。我们涵盖以下类型的数据库：关系型数据库、文档数据库（MongoDB）、键值数据库（Redis）、宽列数据库（Cassandra）和搜索引擎。对于关系型数据库，我们介绍了如何使用对象关系映射（ORM）、一对一、一对多和多对多关系，以及参数化SQL查询。最后，我们介绍了关系型数据库的三个范式规则。

*第九章*介绍了关于线程和线程安全的*并发编程原则*。对于线程安全，我们介绍了几种实现线程同步的方法：锁、原子变量和线程安全集合。我们还讨论了如何从两个不同的线程发布和订阅共享状态的变化。

*第十章*讨论*团队协作原则*。我们解释了使用敏捷框架的重要性，并讨论了开发者通常不会单独工作这一事实及其影响。我们讨论了如何记录软件组件，以便新开发者能够轻松快速地加入。软件中的技术债务是每个团队都应该避免的。介绍了一些防止技术债务的具体行动。代码审查是团队应该做的事情，本章提供了代码审查应关注什么的指导。本章最后讨论了每个团队应具备的开发者角色，并提供了如何让团队尽可能并行开发软件的建议。

*第十一章*专门讨论*DevSecOps*。DevOps描述了集成软件开发（Dev）和软件运维（Ops）的实践。它旨在通过并行化和自动化缩短软件开发生命周期，并提供高质量软件的持续交付。DevSecOps是DevOps的增强版，将安全实践集成到DevOps实践中。本章介绍了DevOps生命周期的各个阶段：计划、编码、构建与测试、发布、部署、运维与监控。本章给出了创建微服务容器镜像的示例，以及如何指定将微服务部署到Kubernetes集群。此外，还提供了一个使用GitHub Actions的完整CI/CD流水线示例。

## 3：架构原则

本章描述了设计简洁、现代的云原生软件系统和应用程序的架构原则。这里的架构设计指的是设计由多个软件组件组成的软件系统。本章重点介绍现代云原生微服务，但其中一些原则也可用于单体软件架构。在本书中，我们不涉及单体软件架构设计，但如果你设计的是单体软件系统，你应该考虑实现所谓的*模块化单体*，它是一个单体，但内部不同功能被清晰分离。这种模块化的架构使得未来在需要时可以将单体拆分为微服务，或将单体的部分功能提取为独立的微服务。

云原生软件由松耦合、可扩展、有弹性且可观测的服务组成，这些服务可以在公有云、私有云或混合云中运行。云原生软件利用容器（例如Docker）、微服务、无服务器函数和容器编排（例如Kubernetes）等技术，并且可以使用声明式代码自动部署。本章中的示例假设微服务部署在Kubernetes环境中。Kubernetes是一种云提供商无关的运行容器化微服务的方式，近年来获得了巨大的普及。如果你是Kubernetes新手，可以在[https://kubernetes.io/docs/concepts/](https://kubernetes.io/docs/concepts/)找到主要概念的概述。

本章讨论以下架构原则和模式：

- 单一职责原则
- 统一命名原则
- 封装原则
- 服务聚合原则
- 高内聚、低耦合原则
- 库组合原则
- 避免重复原则
- 外部化服务配置原则
- 服务替换原则
- 自动驾驶微服务原则
  - 无状态微服务原则
  - 有弹性微服务原则
  - 水平自动扩展微服务原则
  - 高可用微服务原则
  - 可观测服务原则
- 服务间通信模式
- 领域驱动架构设计原则
- 软件版本控制原则
- Git版本控制原则
- 架构模式
- 首选技术栈原则

### 3.1：软件层次结构

一个*软件系统*由多个计算机程序以及与这些程序相关的所有内容组成，使其可操作，包括但不限于配置、部署代码和文档。软件系统分为两部分：*后端*和*前端*。后端软件运行在服务器上，前端软件运行在客户端设备上，如PC、平板电脑和手机。后端软件由*服务*组成。前端软件由使用后端服务的*客户端*和不使用任何后端服务的*独立应用程序*组成。独立应用程序的例子是计算器或简单的文本编辑器。服务是提供某种服务并持续运行的东西。后端还有另外两种类型的程序，它们只运行一次、按需运行或按计划运行。只运行一次或按需运行的程序称为*作业*，按计划运行的程序称为*定时作业*。作业和定时作业通常与服务相关，它们执行与特定服务相关的管理任务。由于单一职责原则，服务本身不应执行管理任务，而应专门提供某种服务，例如*订单服务*提供订单操作。可能会有一个*订单数据库初始化作业*，在安装后触发以初始化*订单服务*的数据库。还可能会有一个*订单数据库清理定时作业*，按计划触发以对*订单服务*的数据库执行清理操作。此外，服务的部署和生命周期可以由单独的服务控制。在Kubernetes环境中，这类服务称为*操作器*。另一种类型的程序是命令行界面（CLI）程序。CLI程序通常用于软件系统管理任务。在一个软件系统中，例如，可能有一个*管理CLI*，可用于安装和升级软件系统。

术语*应用程序*通常用于描述为特定目的设计的单个程序。一般来说，软件应用程序是应用于解决特定问题的软件。从最终用户的角度来看，所有客户端都是应用程序。但从开发者的角度来看，应用程序需要客户端和后端服务才能正常工作，除非该应用程序是*独立应用程序*。在本书中，我将使用术语应用程序来指定程序和相关工件（如配置）的逻辑分组，以形成软件系统中专门用于特定目的的功能部分。根据我的定义，非独立应用程序由一个或多个服务以及可能的一个或多个客户端组成，以满足最终用户的需求。假设我们有一个用于移动电信网络分析的软件系统。该系统提供数据可视化功能。我们可以将该软件系统的数据可视化部分称为数据可视化应用程序。该应用程序由例如一个Web客户端和两个服务组成，一个用于获取数据，一个用于配置。假设我们在移动电信网络分析软件中还有一个通用数据摄入微服务系统。那个通用数据摄取器本身并不是一个应用程序，除非经过一些配置使其成为我们可以称之为应用程序的特定服务。例如，通用数据摄取器可以配置为从移动网络的无线网络部分摄取原始数据。通用数据摄取器与该配置共同构成一个应用程序：无线网络数据摄取器。然后，可以有另一个配置用于从移动网络的核心网络部分摄取原始数据。该配置与通用数据摄取器共同构成另一个应用程序：核心网络数据摄取器。

![](img/cbd069395d7b824346b69b1f92e0fb4a_19_0.png)

**图 3.1. 软件层次结构**

计算机程序和*库*是*软件组件*。*软件组件*是可以单独打包、测试和交付的东西。它由一个或多个类组成，而一个类由一个或多个函数（类方法）组成。（在纯函数式语言中没有传统的类，但软件组件仅由函数组成。）计算机程序也可以由一个或多个库组成，而一个库可以由其他库组成。

![](img/cbd069395d7b824346b69b1f92e0fb4a_20_0.png)

**图 3.2. 软件组件**

## 3.2: 单一职责原则

> 一个软件实体在其抽象层次上应该只有一个单一的职责。

软件系统处于软件层次结构的最高层，应该有一个单一的专用目的。例如，可以有一个电子商务或工资单软件系统。但不应该有一个同时处理电子商务和工资单相关活动的软件系统。如果你是一个软件供应商，并且制作了一个电子商务软件系统，将其出售给需要电子商务解决方案的客户会很容易。但如果你制作了一个同时包含电子商务和工资单功能的软件系统，将其出售给只需要电子商务解决方案的客户就会很困难，因为他们可能已经有了一个工资单软件系统，当然不想要另一个。

让我们考虑软件层次结构中的应用层。假设我们为电信网络分析设计了一个软件系统。这个软件系统被划分为四个不同的应用程序：无线网络数据摄取、核心网络数据摄取、数据聚合和数据可视化。

每个应用程序都有一个单一的专用目的。假设我们将数据聚合和可视化应用程序耦合到一个单一的应用程序中。在这种情况下，用第三方应用程序替换数据可视化部分可能会很困难。但当它们是具有明确定义接口的独立应用程序时，如果需要，用第三方应用程序替换数据可视化应用程序会容易得多。

软件组件也应该有一个单一的专用目的。具有单一职责的服务类型软件组件称为*微服务*。例如，在一个电子商务软件系统中，一个微服务可以负责处理订单，另一个负责处理销售商品。这两个微服务都只负责一件事。默认情况下，我们不应该有一个微服务同时负责订单和销售商品。这将违反单一职责原则，因为订单处理和销售商品处理是同一抽象层次上的两个不同功能。但有时将两个或多个功能组合到一个微服务中可能是合理的。原因可能是这些功能紧密相关，将功能放在一个单一的微服务中可以减少微服务的缺点，例如需要使用分布式事务。因此，微服务的大小可以变化，并且取决于微服务的抽象层次。一些微服务可以很小，而一些微服务如果处于更高的抽象层次，规模可以更大。微服务总是比单体应用小，比单个函数大。根据软件系统及其设计，其中的微服务数量可以从少数几个到几十个甚至上百个不等。

让我们以一个电子商务软件系统为例，它包含以下功能：

- 销售商品
- 购物车
- 订单

让我们设计如何将上述功能拆分为微服务。在决定将哪些功能放在同一个微服务中时，我们考虑是否满足单一职责的要求，并实现高功能内聚和非功能内聚。高功能内聚意味着两个功能相互依赖并倾向于一起变化。低功能内聚的一个例子是电子邮件发送功能和购物车功能。这两个功能不相互依赖，也不会一起变化。因此，我们应该始终将电子邮件发送和购物车功能实现为两个独立的微服务。非功能内聚与所有非功能方面相关，如架构、技术栈、部署、可扩展性、弹性、可用性、可观测性等。

我们不应该将所有电子商务软件系统的功能放在一个单一的微服务中，因为销售商品相关功能与其他功能之间没有高的非功能内聚。与销售商品相关的功能应该放在一个单独的微服务中，该微服务可以独立扩展，因为销售商品微服务比购物车和订单服务接收更多的流量。此外，我们应该能够为销售商品微服务选择合适的数据库技术。使用的数据库引擎应该针对高读取次数和低写入次数进行优化。后来，我们可能会意识到销售商品的图片不应该与其他销售商品相关信息存储在同一个数据库中。然后我们可以引入一个新的微服务，专门用于存储/检索销售商品图像。

我们不必将购物车和订单相关功能实现为两个独立的微服务，而是可以将它们实现为一个单一的微服务。这是因为购物车和订单功能具有高功能内聚性。例如，每当下一个新订单时，应该读取购物车中的商品，然后将其移除。此外，非功能内聚性也很高，两个服务可以使用相同的技术栈并一起扩展。通过将这两个功能放在一个单一的微服务中，我们摆脱了分布式事务，并能够使用标准数据库事务。这简化了微服务的代码库和测试。我们不应该将微服务命名为*购物车和订单服务*，因为这个名称没有表示单一职责。我们应该做的是使用更高抽象层次的术语来命名微服务。例如，我们可以将其命名为*购买服务*。将来，如果我们发现高功能和非功能内聚的要求不再满足，我们可以将*购买服务*拆分为两个独立的微服务：*购物车服务*和*订单服务*。

软件系统初始划分为微服务不应一成不变。如果认为合适，你可以在将来对其进行更改。例如，你可能会意识到由于不同的扩展需求，某个微服务应该被划分为两个独立的微服务。或者，你可能会意识到将两个或多个微服务耦合到一个单一的微服务中以避免复杂的分布式事务会更好。

微服务有许多优点：

- 提高生产力
    - 你可以选择最适合的编程语言和技术栈
    - 微服务易于并行开发，因为合并冲突会更少
    - 开发单体应用可能导致更频繁的合并冲突
- 提高弹性和故障隔离
    - 单个微服务中的故障不会导致其他微服务崩溃
    - 单体应用中的一个错误可能导致整个单体应用崩溃
- 更好的可扩展性
    - 无状态微服务可以自动水平扩展
    - 单体应用的水平扩展很复杂或不可能
- 更好的数据安全性和合规性
    - 每个微服务封装其数据，只能通过公共 API 访问
- 更快、更轻松的升级

+   - 仅需升级发生变更的微服务即可。无需每次都更新整个单体应用。
    - 更快的发布周期。
    - 仅构建发生变更的微服务。无需在任何变更时都构建整个单体应用。
    - 更少的依赖。
    - 依赖冲突的概率更低。
    - 支持*开闭架构*，即对扩展开放、对修改封闭的架构。
    - 与任何现有微服务无关的新功能可以放入新的微服务中，而不是修改当前的代码库。

微服务的主要缺点是分布式架构带来的复杂性。在微服务之间实现事务需要实现分布式事务，这比普通的数据库事务更复杂。分布式事务需要更多的代码和测试。如果可能，你可以通过将紧密相关的服务放在单个微服务中来避免分布式事务。操作和监控基于微服务的软件系统是复杂的。此外，测试分布式系统比测试单体应用更具挑战性。开发团队应通过雇佣 DevOps 和测试自动化专家来重点关注这些领域。

库类型的软件组件也应具有单一职责。就像调用单一职责的服务微服务一样，我们可以将单一职责的库称为*微库*。例如，可以有一个处理 YAML 格式内容的库，另一个处理 XML 格式内容的库。我们不应尝试将处理这两种格式的功能捆绑到单个库中。如果我们这样做了，并且只需要 YAML 相关的功能，那么我们也总是会获得 XML 相关的功能。我们的代码将始终包含 XML 相关的代码，即使它从未被使用。这可能会引入不必要的代码膨胀。我们还将不得不应用该库的任何安全补丁，即使该补丁仅针对我们不使用的 XML 相关功能。

## 3.3：统一命名原则

> *使用特定的后缀来命名不同类型的软件组件。*

在开发软件时，你应该为不同类型的软件组件建立命名约定：微服务、客户端、作业、操作器、命令行界面（CLI）和库。接下来，我将介绍我建议的命名不同软件组件的方式。

微服务的首选命名约定是 `<服务用途>-service` 或 `<服务用途>-svc`。例如：`data-aggregation-service` 或 `email-sending-svc`。在不同地方系统地使用微服务名称。例如，将其用作 Kubernetes Deployment 名称和源代码仓库名称（或在 monorepo 中用作目录名称）。使用 `service` 后缀而不是 `microservice` 后缀来命名微服务就足够了，因为默认情况下每个服务都应该是微服务。因此，使用 `microservice` 后缀命名微服务不会带来任何真正的好处。这只会使微服务名称更长，而没有任何附加价值。

如果你想更具体地命名微服务，你可以使用 `api` 后缀而不是更通用的 `service` 后缀来命名 API 微服务，例如 `sales-item-api`。在本书中，我不使用 `api` 后缀，而始终只使用 `service` 后缀。

客户端的首选命名约定是 `<客户端用途>-<客户端类型>-client`、`<客户端用途>-<UI 类型>-ui` 或 `<客户端用途>-<应用类型>-app`。例如：`data-visualization-web-client`、`data-visualization-mobile-client`、`data-visualization-android-client` 或 `data-visualization-ios-client`。

作业的首选命名约定是 `<作业用途>-job`。例如，一个为订单初始化数据库的作业可以命名为 `order-db-init-job`。

定时任务的首选命名约定是 `<定时任务用途>-cronjob`。例如，一个定期执行订单数据库清理的定时任务可以命名为 `order-db-cleanup-cronjob`。

操作器的首选命名约定是 `<被操作的服务>-operator`。例如，一个用于 `order-service` 的操作器可以命名为 `order-service-operator`。

CLI 的首选命名约定是 `<CLI 用途>-cli`。例如，一个用于管理软件系统的 CLI 可以命名为 `admin-cli`。

库的首选命名约定是 `<库用途>-lib` 或 `<库用途>-library`。例如：`common-utils-lib` 或 `common-ui-components-library`。

使用这些命名约定时，仅通过查看名称就可以清楚地区分微服务、客户端、（定时）作业、操作器、CLI 和库类型的软件组件。此外，也很容易识别源代码仓库包含的是微服务、客户端、（定时）作业、操作器、CLI 还是库。

## 3.4：封装原则

> 微服务必须将其内部状态封装在公共 API 之后。公共 API 之后的任何内容都被视为微服务的私有内容，不能被其他微服务直接访问。

微服务应定义一个公共 API，供其他微服务用于接口交互。公共 API 之后的任何内容都是私有的，无法从其他微服务访问。

虽然微服务应该是无状态的（*无状态服务原则*将在本章后面讨论），但无状态微服务需要一个在微服务外部存储其状态的地方。通常，状态存储在数据库中。数据库是微服务的内部依赖项，应设为微服务的私有，这意味着没有其他微服务可以直接访问该数据库。对数据库的访问通过微服务的公共 API 间接进行。

不建议允许多个微服务共享单个数据库，因为这样就无法控制每个微服务将如何使用数据库，以及每个微服务对数据库有什么要求。

有时，如果每个微服务使用自己的*逻辑*数据库，可以与几个微服务共享一个*物理*数据库。这需要为每个微服务创建一个特定的数据库用户。每个数据库用户只能访问分配给特定微服务的逻辑数据库。这样，任何微服务都不能直接访问另一个微服务的数据库。这种方法仍然可能带来一些问题，因为必须考虑所有微服务对共享物理数据库的容量规划要求。此外，还必须决定共享数据库的部署职责。例如，共享数据库可以作为平台或公共服务部署的一部分，作为平台或公共服务进行部署。

## 3.5：服务聚合原则

> *更高抽象层次的服务聚合更低抽象层次的服务。*

当更高抽象层次的一个服务聚合更低抽象层次的服务时，就发生了服务聚合。

![](img/cbd069395d7b824346b69b1f92e0fb4a_25_0.png)

让我们以一个允许人们在线销售二手产品的电子商务软件系统为例来说明服务聚合。

电子商务服务的问题领域由以下子领域组成：

-   用户账户领域
    -   创建、修改和删除用户账户。
    -   查看包含销售商品和订单的用户账户。
-   销售商品领域
    -   添加新的销售商品，修改、查看和删除销售商品。
-   购物车领域
    -   向购物车添加/移除销售商品，清空购物车。
    -   查看包含销售商品详情的购物车。
-   订单领域
    -   下订单
        -   确保付款。
        -   创建订单。
        -   从购物车中移除已订购的商品。
        -   将已订购的销售商品标记为已售出。
        -   通过电子邮件发送订单确认。
    -   查看包含销售商品详情的订单。
    -   更新和删除订单。

我们不应该在单个 *ecommerce-service* 微服务中实现所有子领域，因为那样我们就不会遵循*单一职责原则*。我们应该使用服务聚合。我们为每个子领域创建一个单独的较低级别的微服务。然后我们创建一个较高级别的 *ecommerce-service* 微服务来聚合这些较低级别的微服务。

我们可以定义我们的 *ecommerce-service* 聚合以下较低级别的微服务：

-   *user-account-service*
    -   创建/读取/更新/删除用户账户。
-   *sales-item-service*
    -   创建/读取/更新/删除销售商品。
-   *shopping-cart-service*

## 微服务实现

-   - 查看购物车、向购物车添加/移除销售商品或清空购物车

-   - *订单服务*
    - 创建/读取/更新/删除订单

-   - *邮件通知服务*
    - 发送邮件通知

大多数上述微服务可以实现为REST API，因为它们主要包含基本的CRUD（创建、读取、更新、删除）操作，而REST API非常适合这类操作。我们将在后面的章节中更详细地处理API设计。现在，让我们使用Django和Django REST框架将*销售商品服务*实现为一个REST API。

为Django项目创建一个目录，并在该目录中创建一个虚拟环境：

```
python -m venv venv
```

在Windows中激活虚拟环境：

```
venv\Scripts\activate
```

或在MacOS/Linux中激活虚拟环境：

```
source venv/bin/activate
```

安装依赖项：

```
pip install django djangorestframework
```

创建新的Django项目和应用：

```
django-admin startproject salesitemservice .
python manage.py startapp salesitems
```

我们将首先实现SalesItem模型类，它包含name和price等属性。

图3.5. models.py

```
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

class SalesItem(models.Model):
    user_account_id = models.BigIntegerField()
    name = models.CharField(max_length=512)
    price = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(2147483647)]
    )
```

接下来，我们将为SalesItem模型实现一个序列化器。在序列化器类中，我们通过名称列出要序列化的模型字段。从安全角度来看，这是好的。我们不应该使用`fields = '__all__'`，因为如果我们向模型添加一些内部字段，它们会被自动序列化并发送给客户端，从而将内部信息暴露给客户端。更安全的方法是显式列出要序列化的字段。

图3.6. serializers.py

```
from rest_framework import serializers
from .models import SalesItem

class SalesItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = SalesItem
        fields = ['id', 'user_account_id', 'name', 'price']
```

最后，我们实现SalesItemViewSet类，它定义了用于创建、获取、更新和删除销售商品的API端点：

图3.7. views.py

```
from typing import Any

from rest_framework import viewsets
from rest_framework.request import Request
from rest_framework.response import Response

from .models import SalesItem
from .serializers import SalesItemSerializer


class SalesItemViewSet(viewsets.ModelViewSet):
    queryset = SalesItem.objects.all()
    serializer_class = SalesItemSerializer

    def list(
            self, request: Request, *args: tuple[Any], **kwargs: dict[str, Any]
    ) -> Response:
        user_account_id = request.query_params.get('userAccountId')

        queryset = (
            SalesItem.objects.all()
            if user_account_id is None
            else SalesItem.objects.filter(user_account_id=user_account_id)
        )

        serializer = SalesItemSerializer(queryset, many=True)
        return Response(serializer.data)
```

我们还需要更新Django项目中的*urls.py*文件，使其包含以下内容：

图3.8. urls.py

```
from django.urls import include, path
from rest_framework import routers
from salesitems.SalesItemViewSet import SalesItemViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register('sales-items', SalesItemViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

在运行微服务之前，你需要设置数据库：

```
python manage.py makemigrations
python manage.py migrate
```

你可以使用以下命令运行*销售商品服务*：

```
python manage.py runserver
```

REST API将可通过 http://127.0.0.1:8000/ 访问。

> 在上面的例子中，我使用了惯用的Django方式，在*models.py*文件中定义模型，在*serializers.py*文件中定义序列化器，在*views.py*文件中定义视图。你也可以将每个类定义在自己的文件中，并根据类名命名文件。在我看来，这是确保每个模块单一职责的最佳方法。对于包含类定义的模块名，我使用*CapWords*（或*PascalCase*）。这违反了PEP 8风格指南，也是我在本书中唯一偏离PEP 8的地方。你当然可以遵循PEP 8，但我使用这种方法有两个原因：
> 
> - 模块名告诉你它包含一个公共类定义，并且模块名告诉你类的名称。例如，如果你有一个名为*OrderService.py*的模块，你可以期望从中导入一个名为*OrderService*的类。
> - 如果你从模块导出类的实例，那么该模块应该用蛇形命名法命名。例如，如果你有一个包含私有*__OrderService*类的模块，并导出一个*order_service*变量（单例），该变量是*__OrderService*类的实例，你应该将该模块命名为*order_service.py*。现在*order_service.py*模块名告诉每个人，应该可以从该模块导入名为*order_service*的变量。

让我们回到Django的例子。如果你有多个模型并将它们全部放入*models.py*文件，文件大小会增长，并且不再容易找到所需的类。一个更好的选择是创建一个*models*目录（一个包），并将各个模型类放入该目录中的单独模块中。找到所需的模型很容易，因为模型会自动按字母顺序列在IDE的文件浏览器中。如果多个类定义在单个文件中，你无法保证它们的字母顺序。

同样的方法适用于包含多个函数的模块。假设我们有一个*utils.py*模块，包含各种实用函数。同样，更好的选择是创建一个名为*utils*的目录，并将各个函数放入它们自己的文件中。然后你可以通过查看*utils*目录的内容轻松找到所需的实用函数。你甚至可以创建子目录来使结构层次化，例如在*utils*目录下创建*string*目录，用于与字符串相关的实用函数。单个文件应该包含一个公共函数，但它可以额外包含多个私有函数，供公共函数使用。关于单一职责原则的更多讨论将在下一章中进行。

下面定义了*电子商务服务*将如何协调使用聚合的低级微服务：

-   - 用户账户域
    - 将CRUD操作委托给*用户账户服务*
    - 委托给*销售商品服务*以获取用户销售商品的信息
    - 委托给*订单服务*以获取用户订单的信息

-   - 销售商品域
    - 将CRUD操作委托给*销售商品服务*

-   - 购物车域
    - 将读取/添加/移除/清空操作委托给*购物车服务*
    - 委托给*销售商品服务*以获取购物车中销售商品的信息

-   - 订单域
    - 确保支付网关确认支付
    - 将CRUD操作委托给*订单服务*
    - 委托给*购物车服务*以从购物车中移除已购买的商品
    - 委托给*销售商品服务*以标记销售商品已购买
    - 委托给*邮件通知服务*以发送订单确认邮件
    - 委托给*销售商品服务*以获取订单销售商品的信息

*电子商务服务*旨在供前端客户端使用，例如Web客户端。*Backend for Frontend*（BFF）一词通常用于描述为前端客户端提供API的微服务。与BFF术语相比，服务聚合是一个通用术语，不一定涉及前端。你可以使用服务聚合来创建由另一个微服务或多个微服务使用的聚合微服务。如果你有一个大型复杂的软件系统，甚至可以有多个级别的服务聚合。

客户端对于他们想从API获取什么信息可能有不同的需求。例如，移动客户端可能仅限于暴露API中所有可用信息的一个子集。相比之下，Web客户端可以获取所有信息，或者可以定制客户端从API检索的信息。

以上所有要求都是基于GraphQL的API可以满足的。因此，明智的做法是使用GraphQL实现*电子商务服务*。我选择了Ariadne库在*电子商务服务*中实现单个GraphQL查询。下面是用户查询的实现，它从三个微服务获取数据。它从*用户账户服务*获取用户账户信息，从*销售商品服务*获取用户的销售商品，最后从*订单服务*获取用户的订单。

让我们创建一个新的Python项目并安装以下依赖项：`pip install ariadne httpx hypercorn`

接下来，我们创建一个支持单个用户查询的 GraphQL 服务器：

## 图 3.9. app.py

```python
from asyncio import gather
import os

from ariadne import QueryType, gql, make_executable_schema
from ariadne.asgi import GraphQL
from httpx import AsyncClient

query = QueryType()

type_defs = gql(
    """
    type UserAccount {
        id: ID!,
        userName: String!
        # 定义其他属性...
    }

    type SalesItem {
        id: ID!,
        name: String!
        # 定义其他属性...
    }

    type Order {
        id: ID!,
        userId: ID!
        # 定义其他属性...
    }

    type User {
        userAccount: UserAccount!
        salesItems: [SalesItem!]!
        orders: [Order!]!
    }

    type Query {
        user(id: ID!): User!
    }
    """
)

USER_ACCOUNT_SERVICE_URL = os.environ.get('USER_ACCOUNT_SERVICE_URL')
SALES_ITEM_SERVICE_URL = os.environ.get('SALES_ITEM_SERVICE_URL')
ORDER_SERVICE_URL = os.environ.get('ORDER_SERVICE_URL')


@query.field('user')
async def resolve_user(_, info, id):
    async with AsyncClient() as client:
        [
            user_account_service_response,
            sales_item_service_response,
            order_service_response,
        ] = await gather(
            client.get(f'{USER_ACCOUNT_SERVICE_URL}/user-accounts/{id}'),
            client.get(
                f'{SALES_ITEM_SERVICE_URL}/sales-items?userAccountId={id}'
            ),
            client.get(f'{ORDER_SERVICE_URL}/orders?userAccountId={id}'),
        )

        user_account_service_response.raise_for_status()
        sales_item_service_response.raise_for_status()
        order_service_service_response.raise_for_status()

        return {
            'userAccount': user_account_service_response.json(),
            'salesItems': sales_item_service_response.json(),
            'orders': order_service_response.json(),
        }

schema = make_executable_schema(type_defs, query)
app = GraphQL(schema, debug=True)
```

为了启动 GraphQL 服务器，我们需要一个 ASGI Web 服务器（例如 hypercorn）。你可以运行 GraphQL 服务器：

```bash
export SALES_ITEM_SERVICE_URL=http://127.0.0.1:8000
export USER_ACCOUNT_SERVICE_URL=...
export ORDER_SERVICE_URL=...
hypercorn app:app -b 127.0.0.1:5000
```

你可以在 http://127.0.0.1:5000/graphql 访问 GraphiQL UI。在左侧窗格中，你可以指定一个 GraphQL 查询。例如，查询 id 为 2 的用户：

```graphql
{
  user(id: 2) {
    userAccount {
      id
      userName
    }
    salesItems {
      id
      name
    }
    orders {
      id
      userId
    }
  }
}
```

因为我们只实现了 *sales-item-service* 这个底层微服务，而没有实现所有底层微服务，所以让我们修改 *app.py*，使其返回虚拟的静态结果，而不是访问不存在的底层微服务：

## 图 3.10. app.py

```python
from asyncio import gather
import os

from ariadne import QueryType, gql, make_executable_schema
from ariadne.asgi import GraphQL
from httpx import AsyncClient, Response

query = QueryType()

type_defs = gql(
    """
    type UserAccount {
        id: ID!,
        userName: String!
        # 定义其他属性...
    }

    type SalesItem {
        id: ID!,
        name: String!
        # 定义其他属性...
    }

    type Order {
        id: ID!,
        userId: ID!
        # 定义其他属性...
    }

    type User {
        userAccount: UserAccount!
        salesItems: [SalesItem!]!
        orders: [Order!]!
    }

    type Query {
        user(id: ID!): User!
    }
    """
)

SALES_ITEM_SERVICE_URL = os.environ.get('SALES_ITEM_SERVICE_URL')


async def getUserAccount(id):
    return Response(200, json={'id': id, 'userName': 'Petri'})


async def getOrders(id):
    return Response(200, json=[{'id': 1, 'userId': id}])


@query.field('user')
async def resolve_user(_, info, id):
    async with AsyncClient() as client:
        [
            user_account_service_response,
            sales_item_service_response,
            order_service_response,
        ] = await gather(
            getUserAccount(id),
            client.get(
                f'{SALES_ITEM_SERVICE_URL}/sales-items?userAccountId={id}'
            ),
            getOrders(id),
        )

        return {
            'userAccount': user_account_service_response.json(),
            'salesItems': sales_item_service_response.json(),
            'orders': order_service_response.json(),
        }

schema = make_executable_schema(type_defs, query)
app = GraphQL(schema, debug=True)
```

如果我们现在执行之前指定的查询，应该会看到下面的查询结果。我们假设 *sales-item-service* 返回一个 id 为 1 的销售项。

```json
{
    "data": {
        "user": {
            "userAccount": {
                "id": "2",
                "userName": "Petri"
            },
            "salesItems": [
                {
                    "id": "1",
                    "name": "Sales item 1"
                }
            ],
            "orders": [
                {
                    "id": "1",
                    "userId": "2"
                }
            ]
        }
    }
}
```

我们可以通过修改 *app.py*，使用错误的 URL（将端口 8000 改为 8001）来启动应用，从而模拟故障：

```bash
export SALES_ITEM_SERVICE_URL=http://127.0.0.1:8001
hypercorn app:app -b 127.0.0.1:5000
```

现在，如果我们再次执行查询，将会得到下面的错误响应，因为服务器无法连接到本地主机上端口 8001 的服务，因为 *localhost:8001* 上没有运行任何服务。

```json
{
  "data": null,
  "errors": [
    {
      "message": "All connection attempts failed",
      "locations": [
        {
          "line": 2,
          "column": 3
        }
      ],
      "path": [
        "user"
      ],
      "extensions": {
        "exception": {
          "stacktrace": [ ... ],
          "context": {
            "mapped_exc": "<class 'httpx.ConnectError'>",
            "from_exc": "<class 'httpc...rotocolError'>",
            "to_exc": "<class 'httpx...rotocolError'>",
            "message": "'All connecti...tempts failed'"
          }
        }
      }
    }
  ]
}
```

你也可以查询一个用户，并指定查询只返回字段的子集。下面的查询不返回 id，也不返回订单。服务器端的 GraphQL 库会自动在响应中只包含请求的字段。作为开发者，你不需要做任何事情。当然，如果你愿意，可以优化你的微服务，使其只从数据库中获取请求的字段。

```graphql
{
  user(id: 2) {
    userAccount {
      userName
    }
    salesItems {
      name
    }
  }
}
```

上述查询的结果将是：

```json
{
    "data": {
        "user": {
            "userAccount": {
                "userName": "pksilen"
            },
            "salesItems": [
                {
                    "name": "sales item 1"
                }
            ]
        }
    }
}
```

上述示例缺少一些生产环境所需的功能，例如授权。授权应检查用户只能执行 *user* 查询来获取他/她自己的资源。如果用户尝试使用他人的 id 执行 *user* 查询，授权应该失败。安全性将在接下来的 *安全原则* 章节中进一步讨论。

前面示例中的 *user* 查询跨越了多个底层微服务：*user-account-service*、*sales-item-service* 和 *order-service*。因为该查询没有修改任何内容，所以可以在没有分布式事务的情况下执行。分布式事务类似于常规（数据库）事务，区别在于它跨越多个远程服务。

*ecommerce-service* 中用于下单的 API 端点需要使用 *order-service* 创建新订单，使用 *sales-item-service* 将已购买的销售项标记为已购买，使用 *shopping-cart-service* 清空购物车，最后使用 *email-notification-service* 发送订单确认邮件。这些操作需要包装在分布式事务中，因为我们希望在任何这些操作失败时能够回滚事务。本章后面将给出如何实现分布式事务的指导。

服务聚合利用了 *外观模式*。外观模式允许将各个底层微服务隐藏在一个外观（更高层的微服务）之后。软件系统的客户端通过外观访问系统。它们不直接与外观后面的各个底层微服务通信，因为这会破坏更高层微服务内部底层微服务的封装。客户端直接访问底层微服务会在客户端和底层微服务之间产生不必要的耦合，这使得在不影响客户端的情况下更改底层微服务变得困难。

以邮局柜台为例，它是一个现实世界中的外观。它作为邮局的外观，当你需要取包裹时，你与那个外观（柜台的邮局职员）沟通。你有一个简单的接口，只需告诉包裹代码，职员就会从正确的货架上找到包裹并拿给你。如果没有那个外观，就意味着你必须自己做底层的工作。你不能只告诉包裹代码，而必须走到货架前，尝试找到你的包裹所在的正确货架，确保你拿的是正确的包裹，然后自己搬运包裹。除了需要更多的工作外，这种方法更容易出错。你可能会不小心拿了别人的包裹，如果你还不够严谨。想想下次你去邮局时，发现所有货架都重新排列了的情况。如果你使用了门面模式，这就不会成为问题。

服务聚合，即一个更高级别的微服务委托给更低级别的微服务，也实现了*桥接模式*。更高级别的微服务只提供一些高级控制，并依赖更低级别的微服务来完成实际工作。

服务聚合允许使用面向对象设计世界中更多的*设计模式*。在服务聚合的上下文中，最有用的设计模式是：

- 装饰器模式
- 代理模式
- 适配器模式

*装饰器模式*可用于在更高级别的微服务中为更低级别的微服务添加功能。一个例子是在更高级别的微服务中添加审计日志。例如，你可以为*电子商务服务*中的请求添加审计日志。你不需要在所有更低级别的微服务中单独实现审计日志。

*代理模式*可用于控制从更高级别的微服务到更低级别的微服务的访问。代理模式的典型例子是授权和缓存。例如，你可以为*电子商务服务*中的请求添加授权和缓存。只有在授权成功后，请求才会被传递给更低级别的微服务。并且，如果请求的响应在缓存中未找到，请求将被转发到相应的更低级别的微服务。你不需要在所有更低级别的微服务中单独实现授权和缓存。

*适配器模式*允许更高级别的微服务适应不同版本的更低级别的微服务，同时保持面向客户端的API不变。

## 3.6：高内聚、低耦合原则

> *一个软件系统应该由具有高内聚和低耦合的服务组成。*

内聚指的是服务内部类彼此关联的程度。耦合指的是一个服务与多少其他服务进行交互。当遵循*单一职责原则*时，可以将服务实现为具有高内聚的微服务。服务聚合增加了低耦合。微服务和聚合服务共同实现了高内聚和低耦合，这是良好架构的目标。如果没有服务聚合，更低级别的微服务将需要相互通信，从而在架构中产生高耦合。此外，客户端将与更低级别的微服务耦合。例如，在电子商务示例中，*订单服务*将与几乎所有其他微服务耦合。如果*销售商品服务*的API发生变化，在最坏的情况下，可能需要在其他三个微服务中进行更改。当使用服务聚合时，更低级别的微服务仅与更高级别的微服务耦合。

![](img/cbd069395d7b824346b69b1f92e0fb4a_39_0.png)

高内聚和低耦合意味着服务的开发可以高度并行化。在电子商务示例中，五个更低级别的微服务彼此之间没有耦合。每个微服务的开发都可以隔离并分配给单个团队成员或一组团队成员。更低级别的微服务可以并行开发，而更高级别的微服务的开发可以在更低级别的微服务API足够稳定时开始。目标是尽早设计更低级别的微服务API，以便能够开发更高级别的微服务。

## 3.7：库组合原则

*更高级别的库应由更低级别的库组成。*

假设你需要一个库来解析特定语法的YAML或JSON格式的配置文件。在这种情况下，你可以首先创建所需的YAML和JSON解析库（或使用现有的）。然后，你可以创建配置文件解析库，该库由YAML和JSON解析库组成。这样你将拥有三个不同的库：一个更高级别的库和两个更低级别的库。每个库都有单一职责：一个用于解析JSON，一个用于解析YAML，一个用于解析具有特定语法（JSON或YAML）的配置文件。软件组件现在可以使用更高级别的库来解析配置文件，而完全不需要知道JSON/YAML解析库的存在。

## 3.8：避免重复原则

在软件系统和服务层面避免软件重复。

软件系统层面的重复发生在两个或多个软件系统使用相同的服务时。例如，两个不同的软件系统可能都有消息代理、API网关、身份和访问管理（IAM）应用程序以及日志和指标收集服务。你甚至可以继续列出更多。无重复架构的目标是只部署一次这些服务。公共云提供商为你提供这些服务。如果你有一个Kubernetes集群，另一种解决方案是将你的软件系统部署在不同的Kubernetes命名空间中，并将公共服务部署到一个共享的Kubernetes命名空间中，例如可以称为*平台*或*公共服务*。

服务层面的重复发生在两个或多个服务具有可以提取到单独的新微服务中的公共功能时。例如，考虑一个*用户账户服务*和*订单服务*都具有通过电子邮件向用户发送通知消息功能的情况。这个电子邮件发送功能在两个微服务中是重复的。通过将电子邮件发送功能提取到单独的新微服务中，可以避免重复。当电子邮件发送功能被提取到自己的微服务中时，微服务的单一职责变得更加明显。有人可能认为另一个替代方案是将公共功能提取到一个库中。这不是一个同样好的解决方案，因为微服务会依赖于该库。当需要更改库时（例如，安全更新），你必须在所有使用该库的微服务中更改库版本，然后测试所有受影响的微服务。

当一家公司在多个部门开发多个软件系统时，软件开发通常是在孤岛中进行的。各部门不一定知道其他部门在做什么。例如，可能两个部门都开发了一个用于发送电子邮件的微服务。现在存在无人知晓的软件重复。这不是一个理想的情况。软件开发公司应该采取一些措施来促进部门之间的合作并打破孤岛。共享软件的一个好方法是在公司使用的源代码托管服务中建立共享文件夹或组织。例如，在GitHub中，你可以创建一个组织来共享公共库的源代码仓库，另一个组织来共享公共服务。每个软件开发部门都可以访问这些公共组织，并且仍然可以在自己的GitHub组织内开发其软件。这样，如果需要，公司可以对不同部门的源代码实施适当的访问控制。当一个团队需要开发新东西时，它可以首先查阅公共源代码仓库，看看是否已经有可以重用或扩展的东西。

## 3.9：外部化服务配置原则

服务的配置应该是外部化的。它应该存储在服务运行的环境中，而不是源代码中。外部化*配置使服务能够适应不同的环境和需求。*

服务配置意味着在不同服务部署（不同环境、不同客户等）之间变化的任何数据。当软件在Kubernetes集群中运行时，外部化配置可以存储在以下典型位置：

- 环境变量
- Kubernetes ConfigMaps
- Kubernetes Secrets

![](img/cbd069395d7b824346b69b1f92e0fb4a_41_0.png)

在以下部分中，让我们讨论这三种配置存储选项。

### 3.9.1：环境变量

环境变量可用于将配置存储为简单的键值对。它们通常用于存储诸如如何连接到依赖服务（如数据库或消息代理）或微服务的日志级别等信息。环境变量对于微服务的运行进程是可用的，该进程可以通过其名称（键）访问环境变量值。

你不应该在源代码中硬编码环境变量的默认值。这是因为默认值通常不是为生产环境准备的，而是为开发环境准备的。假设你将一个服务部署到生产环境，却忘记设置所有必需的环境变量。在这种情况下，你的服务将使用一些默认值，而这些默认值可能并不适合生产环境。

你可以为微服务提供特定于环境的`.env`文件来设置环境变量。例如，你可以有一个`.env.dev`文件来存储开发环境的环境变量值，以及一个`.env.ci`文件来存储微服务*持续集成*（CI）流水线中使用的环境变量值。`.env`文件的语法很简单，每行定义一个环境变量：

**图 3.13. .env.dev**

```
NODE_ENV=development
HTTP_SERVER_PORT=3001
LOG_LEVEL=INFO
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USER=
MONGODB_PASSWORD=
```

**图 3.14. .env.ci**

```
NODE_ENV=integration
HTTP_SERVER_PORT=3001
LOG_LEVEL=INFO
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USER=
MONGODB_PASSWORD=
```

当使用Helm将软件组件部署到Kubernetes集群时，环境变量值应在Helm chart的`values.yaml`文件中定义：

**图 3.15. values.yaml**

```
nodeEnv: production
httpServer:
  port: 8080
database:
  mongoDb:
    host: my-service-mongodb
    port: 27017
```

上述`values.yaml`文件中的值可用于在Kubernetes *Deployment*中定义环境变量，使用以下Helm chart模板：

**图 3.16. deployment.yaml**

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-service
spec:
  template:
    spec:
      containers:
        - name: my-service
          env:
            - name: NODE_ENV
              value: {{ .Values.nodeEnv }}
            - name: HTTP_SERVER_PORT
              value: "{{ .Values.httpServer.port }}"
            - name: MONGODB_HOST
              value: {{ .Values.database.mongoDb.host }}
            - name: MONGODB_PORT
              value: {{ .Values.database.mongoDb.port }}
```

当Kubernetes启动一个微服务Pod时，以下环境变量将对运行中的容器可用：

```
NODE_ENV=production
HTTP_SERVER_PORT=8080
MONGODB_HOST=my-service-mongodb
MONGODB_PORT=27017
```

## 3.9.2: Kubernetes ConfigMaps

Kubernetes ConfigMap可以存储各种格式（如JSON或YAML）的配置文件。这些文件可以挂载到微服务运行容器的文件系统中。然后，容器可以从其文件系统中的挂载目录读取配置文件。

例如，你可以有一个ConfigMap来定义*my-service*微服务的日志级别：

**图 3.17. configmap.yaml**

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-service
data:
  LOG_LEVEL: INFO
```

下面的Kubernetes Deployment描述符定义了*my-service* ConfigMap中键`LOG_LEVEL`的内容将存储在名为`config-volume`的卷中，而`LOG_LEVEL`键的值将存储在名为`LOG_LEVEL`的文件中。将`config-volume`挂载到*my-service*容器的`/etc/config`目录后，就可以读取`/etc/config/LOG_LEVEL`文件的内容，该文件包含文本：INFO。

**图 3.18. deployment.yaml**

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-service
spec:
  template:
    spec:
      containers:
        - name: my-service
          volumeMounts:
            - name: config-volume
              mountPath: "/etc/config"
              readOnly: true
      volumes:
        - name: config-volume
          configMap:
            name: my-service
            items:
              - key: "LOG_LEVEL"
                path: "LOG_LEVEL"
```

在Kubernetes中，对ConfigMap的编辑会反映在相应的挂载文件中。这意味着你可以监听`/etc/config/LOG_LEVEL`文件的变化。下面展示了如何使用*watchdog*库来实现：

```
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from update_log_level import update_log_level


class UpdateLogLevelFsEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        try:
            with open('/etc/config/LOG_LEVEL', 'r') as file:
                new_log_level = file.read()
                # 在此检查 'new_log_level'
                # 是否包含有效的日志级别
                update_log_level(new_log_level)
        except:
            # 处理错误

update_log_level_fs_event_handler = UpdateLogLevelFsEventHandler()
observer = Observer()

observer.schedule(
    update_log_level_fs_event_handler,
    path='/etc/config/LOG_LEVEL',
    recursive=False
)

observer.start()

# ...
# observer.stop()
# observer.join()
```

## 3.9.3: Kubernetes Secrets

Kubernetes Secrets与ConfigMap类似，不同之处在于它们用于存储敏感信息，如密码和加密密钥。

下面是一个*values.yaml*文件和一个用于创建Kubernetes Secret的Helm chart模板示例。该Secret将包含两个键值对：数据库用户名和密码。Secret的数据需要进行Base64编码。在下面的示例中，Base64编码是使用Helm模板函数`b64enc`完成的。

**图 3.19. values.yaml**

```
database:
  mongoDb:
    host: my-service-mongodb
    port: 27017
    user: my-service-user
    password: Ak9(1Kt41uF==%1L0&21mA#gL0!"Dps2
```

**图 3.20. secret.yaml**

```
apiVersion: v1
kind: Secret
metadata:
  name: my-service
type: Opaque
data:
  mongoDbUser: {{ .Values.database.mongoDb.user | b64enc }}
  mongoDbPassword: {{ .Values.database.mongoDb.password | b64enc }}
```

创建后，Secret可以映射到微服务Deployment描述符中的环境变量。在下面的示例中，我们将`my-service` Secret中Secret键`mongoDbUser`的值映射到名为`MONGODB_USER`的环境变量，将Secret键`mongoDbPassword`的值映射到名为`MONGODB_PASSWORD`的环境变量。

**图 3.21. deployment.yaml**

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-service
spec:
  template:
    spec:
      containers:
        - name: my-service
          env:
            - name: MONGODB_USER
              valueFrom:
                secretKeyRef:
                  name: my-service
                  key: mongoDbUser

            - name: MONGODB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: my-service
                  key: mongoDbPassword
```

当启动一个*my-service* Pod时，以下环境变量将对运行中的容器可用：

```
MONGODB_USER=my-service-user
MONGODB_PASSWORD=Ak9(1Kt41uF==%1L0&21mA#gL0!"Dps2
```

## 3.10: 服务替换原则

> *通过使依赖关系透明化，轻松地将一个服务的服务依赖替换为另一个服务。透明的服务通过定义主机和端口暴露给其他服务。在你的微服务中使用外部化服务配置原则（例如，环境变量）来定义依赖服务的主机和端口（以及可能需要的其他参数，如数据库用户名/密码）。*

让我们举一个微服务依赖于MongoDB服务的例子。MongoDB服务应通过定义主机和端口组合来暴露自身。对于微服务，你可以指定以下环境变量来连接到*localhost* MongoDB服务：

```
MONGODB_HOST=localhost
MONGODB_PORT=27017
```

假设在基于Kubernetes的生产环境中，你有一个MongoDB服务在集群中，可以通过名为*my-service-mongodb*的Kubernetes Service访问。在这种情况下，你应该将MongoDB服务的环境变量定义如下：MONGODB_HOST=my-service-mongodb.default.svc.cluster.local
MONGODB_PORT=8080

或者，MongoDB 服务也可以运行在 MongoDB Atlas 云中。此时，可以使用以下类型的环境变量值来连接 MongoDB 服务：

MONGODB_HOST=my-service.tjdze.mongodb.net
MONGODB_PORT=27017

如上述示例所示，你可以根据微服务的环境轻松替换不同的 MongoDB 服务。如果你想使用不同的 MongoDB 服务，无需修改微服务的源代码，只需更改配置即可。

## 3.11：服务间通信方法

服务之间使用以下通信方法进行通信：同步、异步和共享数据。

### 3.11.1：同步通信方法

当一个服务与另一个服务通信并希望立即得到响应时，应使用同步通信方法。同步通信可以使用 HTTP 或 gRPC（其底层使用 HTTP）等协议来实现。

![](img/cbd069395d7b824346b69b1f92e0fb4a_48_0.png)

如果在处理请求时发生故障，处理请求的微服务会向请求方微服务发送错误响应。请求方微服务可以将错误在同步请求栈中向上传递，直到到达最初的请求发起者。通常，最初的请求发起者是一个客户端，例如 Web 或移动客户端。最初的请求发起者可以决定如何处理。通常，它会在一段时间后尝试重新发送请求（我们这里假设错误是暂时的服务器错误，而不是客户端错误，例如错误的请求）。

### 3.11.2：异步通信方法

当一个服务希望向另一个服务传递请求，但不期望得到响应，或者至少不期望立即得到响应时，应使用异步通信方法。服务之间的某些通信本质上就是异步的。例如，一个服务可能希望指示电子邮件通知服务向最终用户发送电子邮件，或者向审计日志服务发送审计日志条目。这两个示例都可以使用异步通信方法来实现，因为不期望对这些操作进行响应。

![](img/cbd069395d7b824346b69b1f92e0fb4a_49_0.png)

图 3.23. 仅请求的异步通信方法

![](img/cbd069395d7b824346b69b1f92e0fb4a_49_1.png)

图 3.24. 请求-响应异步通信方法 / 事件驱动架构

异步通信可以使用消息代理来实现。服务可以向消息代理生产消息，也可以从消息代理消费消息。有多种消息代理实现可用，例如 Apache Kafka、RabbitMQ、Apache ActiveMQ 和 Redis。当微服务向消息代理的主题生产请求时，生产请求的微服务必须等待来自消息代理的确认，表明该请求已成功存储到主题的多个（最好是所有）副本中。否则，在某些消息代理故障场景下，无法 100% 保证请求已成功传递。

当异步请求是“即发即忘”类型（即不期望响应）时，处理请求的微服务必须确保请求最终会被处理。如果请求处理失败，处理请求的微服务必须在一段时间后重新尝试处理。如果收到终止信号，处理请求的微服务实例必须将请求重新生产回消息代理，并允许微服务的其他实例来完成该请求。存在一种罕见的可能性，即重新生产请求回消息代理会失败。然后你可以尝试将请求保存到持久卷，但这同样可能失败。这种情况发生的可能性非常低。

关于为服务间通信设计 API 的更详细描述，请参见 *API 设计原则* 章节。

### 3.11.3：共享数据通信方法

有时，服务之间的通信可以通过共享数据（例如，使用共享数据库）来实现。这种方法在面向数据的服务中很有用，当存储相同的数据两次没有意义时。通常，一个或多个微服务生产共享数据，而其他微服务消费这些数据。这些微服务之间的接口由共享数据的模式定义，例如，由数据库表的模式定义。为了保护共享数据的安全，只有生产数据的微服务应该对共享数据具有写访问权限，而消费数据的微服务应该只对共享数据具有读访问权限。

![](img/cbd069395d7b824346b69b1f92e0fb4a_50_0.png)

## 3.12：领域驱动架构设计原则

> 通过从软件层次结构的顶部（软件系统）开始，到服务级别结束，进行领域驱动设计（DDD）来设计架构。

我经常将软件系统的架构设计比作房屋的建筑设计。房屋代表一个软件系统。房屋的外立面代表软件系统的外部接口。房屋内的房间是软件系统的微服务。就像微服务一样，单个房间通常有专门的用途。软件系统的架构设计包括定义外部接口、微服务以及它们与其他微服务的接口。

架构设计阶段的结果是软件系统的平面图。在架构设计之后，你设计好了外立面，并且所有房间都已确定：每个房间的用途以及房间之间如何接口。

设计单个微服务不再是架构设计，它更像是单个房间的室内设计。微服务的设计使用面向对象的设计原则来处理，这将在下一章介绍。

领域驱动设计（DDD）是一种软件设计方法，其中软件根据领域专家的输入进行建模，以匹配问题/业务领域。通常，这些专家来自业务部门，特别是产品管理部门。DDD 的理念是将领域知识从领域专家传递给单个软件开发人员，以便参与软件开发的每个人都能共享一种描述该领域的通用语言。这种通用语言的理念是人们可以相互理解，并且不会使用多个术语来描述同一件事。这种通用语言也称为 *通用语言*。

领域知识从产品经理和架构师传递给开发团队中的首席开发人员和产品负责人（PO）。团队的首席开发人员和 PO 与团队的其他成员分享领域知识。这通常发生在团队处理史诗和特性，并在规划会议中将其拆分为用户故事时。软件开发团队也可以有专门的领域专家。

DDD 从顶层业务/问题领域开始。顶层领域被划分为多个处于同一抽象级别的子领域：比顶层领域低一级。一个领域应该被划分为子领域，以便子领域之间的重叠最小。子领域将使用定义良好的接口与其他子领域进行接口交互。子领域也称为 *限界上下文*，从技术上讲，它们代表一个应用程序或一个微服务。例如，一个银行软件系统可以有一个用于贷款申请的子领域或限界上下文，另一个用于进行支付。

### 3.12.1：设计示例 1：移动电信网络分析软件系统

假设一个架构团队被分配设计一个移动电信网络分析软件系统。该团队首先更详细地定义软件系统的领域问题。当更详细地思考该系统时，他们最终确定了至少以下子领域：

- 1) 从移动电信网络的各种来源摄取原始数据

## 3.12.1：设计示例1：移动电信网络软件系统

让我们从上述定义中提取一些关键词，并为子领域制定简短的名称：

1. 摄取原始数据
2. 将原始数据转化为洞察
3. 展示洞察

![](img/cbd069395d7b824346b69b1f92e0fb4a_52_0.png)

图 3.26. 子领域

让我们分别考虑这三个子领域。
我们知道移动电信网络分为核心网和无线网。由此，我们可以推断出*摄取原始数据*领域可以进一步划分为子领域：*摄取无线网原始数据*和*摄取核心网原始数据*。我们可以将这两个子领域转化为我们软件系统的应用程序：*无线网数据摄取器*和*核心网数据摄取器*。
*将原始数据转化为洞察*领域至少应包含一个将接收到的原始数据聚合为计数器和关键绩效指标（KPI）的应用程序。我们可以称该应用程序为*数据聚合器*。
*展示洞察*领域应包含一个Web应用程序，该程序能够以多种方式展示洞察，例如使用包含图表的仪表板来展示聚合的计数器和计算出的KPI。我们可以称这个应用程序为*洞察可视化器*。

现在，我们为软件系统定义了以下应用程序：

- 无线网数据摄取器
- 核心网数据摄取器
- 数据聚合器
- 洞察可视化器

![](img/cbd069395d7b824346b69b1f92e0fb4a_53_0.png)

图 3.27. 应用程序

接下来，我们继续进行架构设计，将每个应用程序拆分为一个或多个软件组件（服务、客户端和库）。在定义软件组件时，我们必须记住遵循*单一职责原则*、*避免重复原则*和*外部化服务配置原则*。

在考虑*无线网数据摄取器*和*核心网数据摄取器*应用程序时，我们可以注意到，我们可以使用单个微服务`data-ingester-service`来实现它们，并为无线网和核心网配置不同的参数。这是因为摄取数据的协议对于无线网和核心网是相同的。这两个网络的区别在于摄取数据的模式。通过使用单个可配置的微服务，我们可以利用外部化配置来避免代码重复。

*数据聚合器*应用程序可以使用单个`data-aggregator-service`微服务来实现。我们可以使用外部化配置来定义该微服务应聚合和计算哪些计数器和KPI。

*洞察可视化器*应用程序由三个不同的软件组件组成：

- 一个Web客户端
- 一个用于获取聚合和计算数据（计数器和KPI）的服务
- 一个用于存储Web客户端动态配置的服务

动态配置服务存储关于在Web客户端中可视化哪些洞察以及如何可视化的信息。

*洞察可视化器*应用程序中的微服务是：

- insights-visualizer-web-client
- insights-visualizer-data-service
- insights-visualizer-configuration-service

现在，我们已经完成了软件系统在微服务级别的架构设计。

![](img/cbd069395d7b824346b69b1f92e0fb4a_54_0.png)

架构设计的最后一部分是定义服务间通信方法。`data-ingester-service`需要将原始数据发送到`data-aggregator-service`。数据的发送使用异步的“发后即忘”请求，并通过消息代理实现。`data-aggregator-service`和`insights-visualizer-data-service`之间的通信应使用*共享数据*通信方法，因为`data-aggregator-service`生成的数据被`insights-visualizer-data-service`使用。前端的`insights-visualizer-web-client`与后端的`insights-visualizer-data-service`和`insights-visualizer-configuration-service`之间的通信是同步通信，可以使用基于HTTP的JSON-RPC、REST或GraphQL API来实现。

![](img/cbd069395d7b824346b69b1f92e0fb4a_55_0.png)

图 3.29. 微服务间通信

接下来，设计工作将在开发团队中继续。团队将指定微服务之间的API，并对微服务进行进一步的领域驱动设计和面向对象设计。API设计将在后面的章节中介绍，面向对象设计将在下一章中介绍。

## 3.12.2：设计示例2：银行软件系统

让我们设计一个部分银行软件系统。该银行软件系统应能处理客户的贷款申请和支付。银行系统的问题领域可以划分为两个子领域或限界上下文：

1. 贷款申请
2. 进行支付

在贷款申请领域，客户可以提交贷款申请。将评估贷款资格，银行可以接受贷款申请并发放贷款，或者拒绝贷款申请。在进行支付领域，客户可以进行支付。进行支付将从客户账户中扣款。这也是一笔需要记录的交易。

![](img/cbd069395d7b824346b69b1f92e0fb4a_56_0.png)

图 3.30. 银行软件系统限界上下文

让我们添加一个功能：可以向另一家银行的收款人进行支付：

![](img/cbd069395d7b824346b69b1f92e0fb4a_56_1.png)

图 3.31. 银行软件系统限界上下文

让我们再添加一个功能：资金可以从外部银行转入客户账户。

![](img/cbd069395d7b824346b69b1f92e0fb4a_57_0.png)

从上图可以看出，随着新功能的引入，银行软件系统的架构发生了演变。例如，创建了两个新的子领域（或限界上下文）：资金转账和外部资金转账。微服务本身的变化并不大，但它们在逻辑上如何分组到限界上下文中发生了改变。

## 3.13：自动驾驶微服务原则

> 微服务应被设计为在其部署环境中自动驾驶运行。

自动驾驶微服务是指在部署环境中运行，无需人工交互的微服务，除非在异常情况下，微服务应生成警报以表明需要人工干预。

自动驾驶微服务原则要求遵循以下子原则：

- 无状态微服务原则
- 弹性微服务原则
- 水平自动扩缩容微服务原则
- 高可用微服务原则
- 可观测微服务原则

接下来将更详细地讨论这些原则。

### 3.13.1：无状态微服务原则

> *微服务应是无状态的，以实现弹性、水平可扩展性和高可用性。*

通过将微服务的状态存储在其自身之外，可以使微服务成为无状态的。状态可以存储在微服务实例共享的数据存储中。通常，数据存储是数据库或内存缓存（例如Redis）。

### 3.13.2：弹性微服务原则

> *微服务应具有弹性，即能够自动从故障中快速恢复。*

在Kubernetes集群中，微服务的弹性由Kubernetes控制平面处理。如果微服务实例所在的计算节点需要退役，Kubernetes将在另一个计算节点上创建该微服务的新实例，然后将微服务从要退役的节点上驱逐。

在微服务中需要做的是使其监听Linux终止信号，特别是`SIGTERM`信号，该信号发送给微服务实例以指示其应终止。收到SIGTERM信号后，微服务实例应启动优雅关闭。如果微服务实例未优雅关闭，Kubernetes最终将发出`SIGKILL`信号以强制终止微服务实例。`SIGKILL`信号在终止宽限期过后发送。该期限默认为30秒，但可配置。

微服务实例可能被从计算节点驱逐还有其他原因。其中一个原因是Kubernetes必须分配（可能由于与CPU/内存请求相关的原因，例如，另一个微服务需要在该特定计算节点上运行，而你的微服务将无法再容纳在那里，必须移动到另一个计算节点。

如果一个微服务 Pod 崩溃，Kubernetes 会注意到这一点并启动一个新的 Pod，以便始终有所需数量的微服务副本（Pod/实例）在运行。副本数量可以在微服务的 Kubernetes Deployment 中定义。

但是，如果一个微服务 Pod 进入死锁状态并无法处理请求怎么办？这种情况可以在 *存活探针* 的帮助下得到缓解。你应该始终为每个微服务 Deployment 指定一个存活探针。下面是一个微服务 Deployment 的示例，其中定义了一个 HTTP GET 类型的存活探针：

**图 3.33. deployment.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "microservice.fullname" . }}
spec:
  replicas: 1
  selector:
    matchLabels:
      {{- include "microservice.selectorLabels" . | nindent 6 }}
  template:
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.imageRegistry }}/{{ .Values.imageRepository }}:{{ .Values.imageTag }}"
          livenessProbe:
            httpGet:
              path: /isMicroserviceAlive
              port: 8080
            initialDelaySeconds: 30
            failureThreshold: 3
            periodSeconds: 3
```

Kubernetes 将每三秒轮询一次微服务实例的 `/isMicroserviceAlive` HTTP 端点（在为微服务实例启动预留的 30 秒初始延迟之后）。该 HTTP 端点应返回 HTTP 状态码 200 OK。假设对某个特定微服务实例的该端点请求连续失败三次（由 `failureThreshold` 属性定义）（例如，由于死锁）。在这种情况下，该微服务实例被视为已死亡，Kubernetes 将终止该 Pod 并自动启动一个新的 Pod。

当将微服务升级到新版本时，应修改 Kubernetes Deployment。应在 Deployment 的 `image` 属性中指定新的容器镜像标签。此更改将触发 Deployment 的更新过程。默认情况下，Kubernetes 执行 *滚动更新*，这意味着你的微服务在更新过程中可以继续处理请求，而不会发生停机。

假设你在微服务 Deployment 中定义了一个副本（如上 `replicas: 1`），并执行了 Deployment 升级（将镜像更改为新版本）。在这种情况下，Kubernetes 将使用新的镜像标签创建一个新的 Pod，并且只有在新 Pod 准备好处理请求后，Kubernetes 才会删除运行旧版本的 Pod。因此，没有停机时间，微服务在升级过程中可以继续处理请求。

如果你的微服务部署有更多副本，例如 10 个，默认情况下，Kubernetes 将终止最多 25% 的运行中 Pod，并启动最多 25% 副本数量的新 Pod。*滚动更新* 意味着 Pod 的更新是分块进行的，每次更新 25% 的 Pod。百分比值是可配置的。

## 3.13.3：微服务水平自动扩展原则

> 微服务应自动进行水平扩展，以便能够处理更多请求。

水平扩展意味着添加或移除微服务的实例。微服务的水平扩展要求无状态。有状态服务通常使用粘性会话来实现，以便来自特定客户端的请求始终路由到同一个服务实例。有状态服务的水平扩展很复杂，因为客户端的状态存储在单个服务实例上。在云原生世界中，我们希望确保微服务实例之间的负载均匀分布，并将请求定向到任何可用的微服务实例进行处理。

最初，一个微服务可能只有一个实例。当微服务负载增加时，一个实例可能无法处理所有工作。在这种情况下，必须通过添加一个或多个新实例来水平扩展（扩展）微服务。当多个微服务实例运行时，状态不能再存储在实例内部，因为不同的客户端请求可能会被定向到不同的微服务实例。无状态微服务必须将其状态存储在微服务外部，例如内存缓存或所有微服务实例共享的数据库中。

微服务可以手动扩展，但这很少是理想的。手动扩展需要有人持续监控软件系统并手动执行所需的扩展操作。微服务应自动进行水平扩展。微服务要实现水平自动扩展，需要满足两个条件：

- 微服务必须是无状态的
- 必须有一个或多个定义扩展行为的指标

水平自动扩展的典型指标是 CPU 利用率和内存消耗。在许多情况下，仅使用 CPU 利用率指标就足够了。也可以使用自定义或外部指标。例如，Kafka 消费者滞后指标可以表明消费者滞后是否在增加，以及是否应该启动一个新的微服务实例来减少消费者滞后。

在 Kubernetes 中，你可以使用 *HorizontalPodAutoscaler* (HPA) 来指定水平自动扩展：

**图 3.34. hpa.yaml**

```yaml
apiVersion: autoscaling/v2beta1
kind: HorizontalPodAutoscaler
metadata:
  name: my-service
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-service
  minReplicas: 1
  maxReplicas: 99
  metrics:
    - type: Resource
      resource:
        name: cpu
        targetAverageUtilization: 75
    - type: Resource
      resource:
        name: memory
        targetAverageUtilization: 75
```

在上面的示例中，*my-service* 微服务被水平自动扩展，以便始终至少有一个微服务实例在运行。最多可以有 99 个微服务实例在运行。如果 CPU 或内存利用率超过 75%，微服务将被扩展；当 CPU 和内存利用率都降至 75% 以下时，微服务将被缩减（微服务实例数量减少）。

## 3.13.4：高可用微服务原则

*业务关键型微服务必须是高可用的。*

如果环境中只运行一个微服务实例，这并不能使微服务具有高可用性。如果该实例发生任何问题，微服务将暂时不可用，直到新实例启动并准备好处理请求。因此，你应该为所有业务关键型微服务运行至少两个或更多实例。你还应该确保这两个实例不在同一个计算节点上运行。实例应在云提供商的不同可用区中运行。这样，可用区 1 中的灾难不一定会波及在可用区 2 中运行的微服务。

你可以通过在微服务 Deployment 中定义反亲和性规则来确保没有两个微服务实例在同一个计算节点上运行：

**图 3.35. deployment.yaml**

```yaml
...
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app.kubernetes.io/name: {{ include "microservice.name" . }}
        topologyKey: "kubernetes.io/hostname"
...
```

对于业务关键型微服务，我们需要修改上一节中的水平自动扩展示例：`minReplicas` 属性应增加到 2：

**图 3.36. hpa.yaml**

```yaml
apiVersion: autoscaling/v2beta1
kind: HorizontalPodAutoscaler
metadata:
  name: my-service
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-service
  minReplicas: 2
  maxReplicas: 99
...
```

## 3.13.5：可观测微服务原则

> 应该能够尽快检测到已部署微服务中的任何异常行为。异常行为应触发警报。部署环境应提供辅助工具以进行故障排除。

现代云原生软件系统由多个同时运行的微服务组成。没有人能手动检查数十或数百个微服务实例的日志。监控微服务的关键在于自动化。一切都始于从微服务及其执行环境中收集相关指标。这些指标用于定义异常情况的自动警报。指标还用于创建监控和故障排除仪表板，这些仪表板可用于在触发警报后分析软件系统及其微服务的状态。

除了指标之外，为了能够深入排查问题的根本原因，还应实施分布式追踪，以记录不同微服务之间的通信，从而排查服务间通信问题。每个微服务还必须至少记录所有错误和警告。这些日志应被送入一个集中的日志收集系统，以便于查询。

## 3.14：软件版本控制原则

本节将介绍与软件版本控制相关的以下原则：

- 使用语义化版本控制
- 避免使用 0.x 版本
- 不要增加主版本号
- 为所有主版本实施安全补丁和错误修正
- 在生产环境中避免使用非 LTS（长期支持）版本

### 3.14.1：使用语义化版本控制原则

> 为软件组件使用语义化版本控制。

语义化版本控制意味着，给定格式为 `<MAJOR>.<MINOR>.<PATCH>` 的版本号，应在以下情况下递增：

- 当你进行不兼容的 API 更改时，递增 *MAJOR* 值
- 当你以向后兼容的方式添加功能时，递增 *MINOR* 值
- 当你进行向后兼容的错误修复或安全补丁时，递增 *PATCH* 值

### 3.14.2：避免使用 0.x 版本原则

> 如果你使用第三方组件，请避免或至少谨慎使用 0.x 版本的组件。

在语义化版本控制中，主版本零（0.x.y）用于初始开发。任何内容都可能随时更改。公共 API 不应被视为稳定。通常，主版本为零的软件组件仍处于概念验证阶段，任何内容都可能改变。如果你想或需要使用更新的版本，你必须为变更做好准备，有时这些变更可能相当大，导致大量的重构工作。

### 3.14.3：不要增加主版本号原则

在语义化版本控制中，当进行向后不兼容的公共 API 更改时，你需要增加主版本号。但如果可能且可行，我建议不要进行向后不兼容的更改，从而避免增加主版本号。

如果你需要进行向后不兼容的公共 API 更改，你应该创建一个具有不同名称的全新软件组件。例如，假设你有一个 *common-ui-lib* 并且需要进行向后不兼容的更改。在这种情况下，建议将新的主版本号添加到库名称中，并发布一个新库 *common-ui-lib-2*。这可以防止开发者在更改所使用的库版本号时意外使用更新的不兼容版本。库用户不一定知道一个库是否正确使用了语义化版本控制。这些信息通常不会在库文档中说明，但在库文档中传达这一点是一个好习惯。

如果一个软件组件正在使用 *common-ui-lib*，它可以安全地始终使用包含所有所需错误修复和安全更新的最新版本库。

以下操作始终是安全的：

```
pip install --upgrade common-ui-lib
```

当你准备好迁移到库的新主版本时，你可以卸载旧版本并按以下方式安装新的主版本：

```
pip uninstall common-ui-lib
pip install common-ui-lib-2
```

考虑何时创建库的新主版本。当你创建第一个库版本时，你可能没有在公共 API 中做对所有事情。这是正常的。第一次就创建一个完美的 API 几乎是不可能的。在发布库的第二个主版本之前，我建议与团队一起审查新 API，收集用户反馈，并等待足够长的时间，以便第二次使 API“接近完美”。没有人想使用一个频繁进行向后不兼容主版本更改的库。

### 3.14.4：为所有主版本实施安全补丁和错误修正原则

如果你为他人使用而编写了一个库，不要仅仅因为库包含一些错误修正或安全补丁（而这些在旧主版本中不可用），就强制用户使用新的主版本。你应该有一套全面的自动化测试，以确保错误修复或安全补丁不会破坏任何东西。因此，在多个分支或源代码存储库中进行安全补丁或错误修复应该很容易。

要求库用户升级到新的主版本以获取某些安全补丁或错误修正，可能会导致维护噩梦，库用户必须重构所有使用该库的软件组件，仅仅是为了获取一个安全补丁或错误修正。

### 3.14.5：在生产环境中避免使用非 LTS 版本原则

某些软件提供长期支持（LTS）和非 LTS 版本。在生产环境中始终只使用 LTS 版本。你可以保证通过错误修正和安全补丁获得长期支持。你可以在概念验证项目中使用非 LTS 版本，以使用 LTS 版本中不可用的一些新功能。但你必须记住，如果 PoC 成功，你不能直接将其投入生产。你需要先将其产品化，即用 LTS 软件替换非 LTS 软件。

## 3.15：Git 版本控制原则

> 使用基于主干（= main 分支）的开发，并在功能分支中开发软件，然后将其合并到主分支。需要时使用功能开关（或标志）。

基于主干的开发适用于具有广泛自动化功能和非功能测试集并能使用功能开关的现代软件。还有一种较旧的分支模型叫做 *GitFlow*，可以用来替代基于主干的开发，以更好地控制软件发布。你可以在 [https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) 找到更多关于 *GitFlow* 的信息。

当你需要开发一个新功能时，可以使用以下任一方式：

1. 使用功能分支
2. 使用多个功能分支和一个功能开关（或标志）

### 3.15.1：功能分支

功能分支方法足以应对涵盖单个程序增量、团队和微服务的简单功能。新功能在从主分支创建的功能分支中开发，当功能准备就绪时，功能分支被合并回主分支，如果需要，功能分支可以被删除。功能分支应使用合并或拉取请求进行合并，这会触发 CI 流水线运行，该运行必须在合并/拉取请求完成之前成功。合并或拉取请求还应负责代码审查。还应该有一种手动方式来触发功能分支的 CI/CD 流水线运行，以便开发者可以在开发阶段在测试环境中测试未完成的功能。这些工件可以称为 *进行中* 工件，应定期（例如每 48 小时）从工件存储库中清理。

下面描绘了创建和使用功能分支的示例工作流：

```
git clone <repository-url>
git checkout main
git pull --rebase --ff-only

# 为 id 为 <feature-id> 的功能创建并检出功能分支
# 功能 id 可以是 JIRA id，例如
git checkout -b feature/<feature-id>

# 对代码进行更改

# 第一次提交
git commit -a -m "Commit message"

# 可能还有更多的代码更改和提交...

# 从远程主分支获取最新的提交，
# 并将本地主分支快进到与 origin/main 匹配
git fetch origin main
git update-ref refs/heads/main refs/remotes/origin/main

# 将你的功能分支变基到主分支之上
git rebase origin/main --autostash

# 将功能分支推送到远程
git push -u origin feature/<feature-id>

# 其他开发者现在也可以使用该功能分支，
# 因为它已推送到 origin
```

当功能准备就绪时，你可以从功能分支创建拉取或合并请求到主分支。你可以在你的 Git 托管服务的网页上或使用 `git push` 命令输出中的链接创建拉取/合并请求。创建拉取/合并请求后，应启动构建流水线，同事可以审查代码。在创建拉取/合并请求后启动的构建会构建 *候选* 工件，这些工件被存储到工件存储库中，但会在一段时间后被删除。如果你在创建拉取/合并请求后需要更改代码，只需如前所述修改代码、添加、提交并将其推送到存储库。代码审查通过且构建流水线成功后，合并即可完成。合并后，应运行来自主分支的构建流水线。此流水线运行应将最终的发布工件推送到工件存储库。

### 3.15.2：功能开关

功能开关类似于功能许可证。在功能许可证的情况下，只有当用户在其环境中激活了相应的许可证时，功能才可用。可切换的功能仅在功能开关打开时才可用。功能开关应用于跨越多个程序增量、微服务或团队的复杂功能。功能开关是环境配置的一部分。例如，功能开关可以存储在任何微服务都可以访问的 Kubernetes ConfigMap 中。使用功能开关时，开关最初是关闭的。功能的开发在不同团队的多个功能分支中进行。团队将各自负责的功能分支合并到主分支。当所有功能分支都合并到主分支后，就可以开启功能开关来激活该功能。

尚未使用过功能开关的人可能存在一些偏见和误解：

- 代码会因功能开关而变得杂乱
  - 并非所有功能都需要开关，只有那些确实需要的功能才应设置开关。例如，当功能已实现但尚未100%测试完成时，就需要使用功能开关。
- 代码会因if-else语句而变得难以阅读且杂乱
  - 如果代码库存在技术债务且设计不当，这种情况确实可能发生。（= 对*意大利面条式代码*进行*霰弹枪式修改*）
  - 通常实现功能开关不需要修改多处代码，只需修改一处或少数几处
- 功能开关会导致性能下降
  - 功能开关几乎总是可以以可忽略不计的性能损失来实现，例如只需一个或几个if语句
- 移除功能开关需要额外工作且可能引发bug
  - 首先，你真的需要移除它们吗？很多时候功能开关可以保留在代码库中，只要它们不会降低代码可读性或性能
  - 当代码库设计正确时（例如遵循*开闭原则*），移除功能开关比需要对*意大利面条式代码*进行*霰弹枪式修改*的情况要容易得多
  - 全面的自动化测试应该能使移除功能开关变得相对安全

## 3.16：架构模式

### 3.16.1：事件溯源模式

> *使用事件溯源将状态变化捕获为事件序列。*

事件溯源确保服务状态的所有更改都作为有序事件序列存储。事件溯源使得查询状态变化成为可能。同时，状态变更事件充当审计日志。可以重建过去的状态，并将当前状态回滚到某个早期状态。与资源上的CRUD操作不同，事件溯源仅使用CR（创建和读取）操作。只能创建新事件和读取事件，无法更新或删除现有事件。

让我们以在电子商务软件系统中使用事件溯源存储订单为例。*订单服务*应能存储以下事件：

- **AbstractOrderEvent**
  - 其他具体事件的抽象基事件，包含时间戳和订单ID属性
- **OrderCreatedEvent**
  - 包含订单的基本信息
- **OrderPaymentEvent**
  - 包含订单支付信息
- **OrderModificationEvent**
  - 包含客户在包装前对订单进行的修改信息
- **OrderPackagedEvent**
  - 包含谁收集和包装订单的信息
- **OrderCanceledEvent**
  - 描述客户已取消订单且订单不应发货
- **OrderShippedEvent**
  - 包含物流合作伙伴和订单发货跟踪ID信息
- **OrderDeliveredEvent**
  - 包含已交付订单的取货点信息
- **OrderShipmentReceivedEvent**
  - 通知客户已收到货物
- **OrderReturnedEvent**
  - 包含退回订单或订单商品的信息
- **OrderReturnShippedEvent**
  - 包含物流合作伙伴和退货跟踪ID信息
- **OrderReturnReceivedEvent**
  - 包含谁处理了订单退货以及退回商品状态的信息
- **OrderReimbursedEvent**
  - 包含向客户退还退回订单商品的报销信息

### 3.16.2：命令查询职责分离（CQRS）模式

> 如果你想对创建/更新（=命令）操作使用与查询信息不同的模型，可以使用CQRS模式。

让我们考虑前面使用事件溯源的*订单服务*示例。在*订单服务*中，所有命令都是事件。我们希望用户能够高效地查询订单。除了事件之外，我们还应该有订单的额外表示形式，因为通过重放所有相关事件来生成订单的当前状态效率很低。因此，我们的架构应该采用CQRS模式，将*订单服务*分为两个不同的服务：*订单命令服务*和*订单查询服务*。

![](img/cbd069395d7b824346b69b1f92e0fb4a_70_0.png)

*订单命令服务*与使用事件溯源的原始*订单服务*相同，而*订单查询服务*是一个新服务。*订单查询服务*有一个数据库，其中保存订单的物化视图。这两个服务通过消息代理连接。*订单命令服务*将事件发送到消息代理中的主题。*订单查询服务*从主题读取事件并将更改应用到物化视图。物化视图经过优化，包含每个订单的基本信息（包括其当前状态），供电子商务公司员工和客户使用。由于客户查询订单，物化视图应按客户ID列建立索引以实现快速检索。假设在某些特殊情况下，客户需要物化视图中没有的订单详细信息。在这种情况下，可以使用*订单命令服务*查询订单事件以获取额外信息。

### 3.16.3：分布式事务模式

分布式事务是跨越多个微服务的事务。分布式事务由一个或多个远程请求组成。分布式事务可以使用*saga模式*实现。在saga模式中，分布式事务中的每个请求都应定义相应的补偿操作。如果分布式事务中的一个请求失败，则应对已成功执行的请求执行补偿请求。执行补偿请求的目的是将系统恢复到分布式事务开始之前的状态。因此，分布式事务的回滚是通过执行补偿操作来完成的。

如果我们无法确定服务器是否成功执行了请求，则必须对分布式事务中失败的请求进行条件补偿。当请求超时且我们未收到指示请求状态的响应时，就可能发生这种情况。

此外，执行补偿请求也可能失败。因此，微服务必须持久化补偿请求，以便稍后重试直到全部成功。需要持久化是因为微服务实例可能在成功完成所有补偿请求之前被终止。另一个微服务实例可以继续被终止的微服务实例留下的工作。

分布式事务中的某些请求可能无法补偿。一个典型例子是发送电子邮件。一旦发送，就无法撤回。处理无法补偿的请求至少有两种方法。第一种是延迟请求的执行，使其变得可补偿。例如，电子邮件发送微服务可以不立即发送电子邮件，而是将电子邮件存储在队列中以便稍后发送。现在电子邮件发送微服务可以接受补偿请求，从发送队列中移除电子邮件。

另一种方法是在分布式事务的最后可能阶段执行不可补偿的请求。例如，你可以将电子邮件发送请求作为分布式事务的最后一个请求发出。那么需要补偿电子邮件发送的可能性就比将电子邮件作为分布式事务的第一个请求发送要低。你也可以结合这两种方法。有时即使你最初认为某个请求不可补偿，它也可能是可补偿的。如果我们考虑发送电子邮件，可以通过发送另一封电子邮件来补偿，说明由于特定原因应忽略之前发送的电子邮件。

#### 3.16.3.1：Saga编排模式

> 编排器或控制器微服务编排分布式事务的执行。

让我们以使用saga编排模式的在线银行系统为例，用户可以在该系统中从其账户转账。我们有一个更高级别的微服务`account-money-transfer-service`，用于进行转账。银行系统还有两个较低级别的微服务：`account-balance-service`和`account-transaction-service`。`account-balance-service`保存账户余额信息，而`account-transaction-service`跟踪账户上的所有交易。`account-money-transfer-service`充当saga编排器，并利用这两个较低级别的微服务来实现转账。

让我们考虑当用户提取25.10美元时，`account-money-transfer-service`执行的分布式事务：

1. `account-money-transfer-service`尝试通过向`account-balance-service`发送以下请求从用户账户中提取金额：

## 3.16.3.2：Saga 编排模式

> *微服务在编排中执行分布式事务，其中客户端微服务发起一个分布式事务，而最后一个参与的微服务通过向客户端微服务发送完成消息来完成该分布式事务。*

Saga 编排模式利用微服务之间的异步通信。参与的微服务在编排中相互发送消息以实现 Saga 的完成。
Saga 编排模式有几个缺点：

- 分布式事务的执行不像在 Saga 编排模式中那样集中，因此可能难以弄清楚分布式事务实际是如何执行的。
- 它在微服务之间创建了耦合，而微服务应该尽可能松散耦合。

Saga 编排模式在参与微服务数量较少的情况下效果最佳。这样服务之间的耦合度较低，也更容易理解分布式事务是如何执行的。

让我们沿用之前的转账示例，但这次使用 Saga 编排模式而不是 Saga 编排模式。

1) *account-money-transfer-service* 通过向消息代理的 *account-balance-service* 主题发送以下事件来启动 Saga：

```json
{
  "event": "Withdraw",
  "data": {
    "sagaUuid": "e8ab60b5-3053-46e7-b8da-87b1f46edf34",
    "amountInCents": 2510
  }
}
```

2) *account-balance-service* 将从消息代理消费 Withdraw 事件，执行取款操作，如果成功，则将相同的事件发送到消息代理的 *account-transaction-service* 主题。

3) *account-transaction-service* 将从消息代理消费 Withdraw 事件，持久化一个账户交易，如果成功，则将以下事件发送到消息代理的 *account-money-transfer-service* 主题：

```json
{
  "event": "WithdrawComplete",
  "data": {
    "sagaUuid": "e8ab60b5-3053-46e7-b8da-87b1f46edf34"
  }
}
```

如果步骤 2) 或 3) 失败，*account-balance-service* 或 *account-transaction-service* 将向消息代理的 *account-money-transfer-service* 主题发送以下事件：

```json
{
    "event": "WithdrawFailure",
    "data": {
        "sagaUuid": "e8ab60b5-3053-46e7-b8da-87b1f46edf34"
    }
}
```

如果 *account-money-transfer-service* 收到 WithdrawFailure 事件，或者在某个超时期间内未收到 WithdrawComplete 事件，*account-money-transfer-service* 将通过向消息代理的 *account-balance-service* 主题发送以下事件来启动分布式事务回滚序列：

```json
{
    "event": "WithdrawRollback",
    "data": {
        "sagaUuid": "e8ab60b5-3053-46e7-b8da-87b1f46edf34",
        "amountInCents": 2510,
        // 此处为其他交易信息...
    }
}
```

一旦 *account-balance-service* 中的回滚完成，回滚事件将被发送到消息代理的 *account-transaction-service* 主题。在 *account-transaction-service* 成功执行回滚后，它会向 *account-money-transfer-service* 主题发送一个 WithdrawRollbackComplete 事件。一旦 *account-money-transfer-service* 消费了该消息，取款事件就被成功回滚。假设 *account-money-transfer-service* 在某个超时期间内未收到 WithdrawRollbackComplete 事件。在这种情况下，它将通过重新向 *account-balance-service* 发送 WithdrawRollback 事件来重新启动回滚编排。

## 3.17：首选技术栈原则

*为不同目的定义首选技术栈。*

微服务架构允许为每个微服务使用最合适的技术栈进行开发。例如，一些微服务需要高性能和受控的内存分配，而其他微服务则不需要这些。你可以根据微服务的需求选择使用的技术栈。对于实时数据处理微服务，你可能会选择 C++ 或 Rust，而对于简单的 REST API，你可能会选择 Node.js 和 Express、Java 和 Spring Boot，或 Python 和 Django。

即使微服务架构允许不同的团队和开发人员决定在实现微服务时使用哪种编程语言和技术，为不同目的定义首选技术栈仍然是一个好习惯。否则，你可能会发现自己处于一个软件系统中使用了大量编程语言和技术的情况。一些编程语言和技术，如 Clojure、Scala 或 Haskell，可能相对小众。

当组织中的软件开发人员来来去去时，你可能会遇到这样的情况：没有人了解某些特定的利基编程语言或技术。在最坏的情况下，微服务需要使用一些更主流的技术从头开始重新实现。因此，你应该指定团队应使用的技术栈。这些技术栈应尽可能包含主流的编程语言和技术。

例如，一个架构团队可能会决定以下内容：

-   Web客户端应使用TypeScript、React和Redux进行开发
-   非API后端服务出于性能原因应使用C++开发
-   API应使用TypeScript、Node.js和Nest.js，或使用Java和Spring Boot进行开发
-   集成测试应使用Cucumber实现，采用与实现相同的语言，或者，也可以使用Python和Behave
-   端到端测试应使用Python和Behave实现
-   脚本应使用Bash实现小型脚本，使用Python实现较大的脚本

上述技术栈是主流的。招聘具备所需知识和能力的人才应该是毫不费力的。

在定义了首选技术栈之后，你应该创建一个或多个实用工具，以便能够快速使用特定技术栈启动新项目。这个或这些实用工具应为新的微服务、客户端或库生成初始源代码仓库内容。新的微服务的初始源代码仓库应至少包含以下项目：

-   源代码文件夹
-   单元测试文件夹
-   集成测试文件夹
-   构建工具，例如Java的Gradle Wrapper
-   初始构建定义文件，例如Java的build.gradle、C++的CMakeLists.txt或Node.js的package.json
    -   在构建定义文件中定义的初始依赖项
-   用于存储不同环境（开发、CI）环境变量的.env文件
-   .gitignore
-   README.MD模板
-   代码检查规则（例如，.eslintrc.json）
-   代码格式化规则（例如，.prettier.rc）
-   集成测试的初始代码，例如用于启动集成测试环境的docker-compose.yml文件
-   所选云提供商的基础设施代码，例如在云中部署托管SQL数据库的代码
-   用于构建微服务容器镜像的代码（例如，Dockerfile）
-   部署代码（例如，Helm chart）
-   CI/CD流水线定义代码

该实用工具应在为微服务创建初始源代码仓库内容之前，向开发人员询问以下问题：

-   微服务的名称是什么？
-   微服务将部署到哪个云环境？（AWS、Azure、Google Cloud等）
-   使用的服务间通信方法是什么？根据答案，该实用工具可以添加依赖项，例如Kafka客户端依赖项
-   微服务是否应该有数据库，以及是什么数据库？
-   其他依赖的微服务是什么？

当然，关于首选技术栈的决定并非一成不变。它们不是静态的。随着时间的推移，新技术会出现，新的编程语言会流行起来。在某个时候，可能会决定用一个新的技术栈替换现有的首选技术栈。然后所有新项目都应使用新的技术栈，而旧的软件组件将逐步迁移到使用新的技术栈。

许多开发人员热衷于定期学习新事物。应该鼓励他们使用自己选择的技术进行业余项目，并且他们应该能够在选定的新项目中使用新的编程语言和框架。

## 4：面向对象设计原则

本章描述与面向对象设计相关的原则。讨论以下原则：

-   面向对象编程概念
-   编程范式
-   为什么面向对象编程很难？
-   SOLID原则
-   统一命名原则
-   封装原则
-   组合原则
-   领域驱动设计原则
-   使用设计模式原则
-   不要问，告诉原则
-   迪米特法则
-   避免原始类型痴迷原则
-   依赖注入原则
-   避免重复原则

我们从面向对象编程（OOP）概念的定义开始本章，并讨论不同的编程范式：OOP、命令式和函数式编程。我们还分析了为什么OOP很难，尽管OOP概念和基本原则并不那么难以掌握。

### 4.1：面向对象编程概念

以下是与OOP相关的基本概念：

-   类/对象
    -   属性和方法
    -   组合（=当属性是其他类时）
-   封装
-   抽象
-   继承
-   接口
    -   接口演进
-   多态性
    -   动态分派（后期绑定）

接下来让我们讨论每个概念。

#### 4.1.1：类/对象

类是用户定义的数据类型，充当单个对象（类的实例）的蓝图。对象使用类的`__init__`方法创建，该方法设置对象的初始状态。类由属性和方法组成，这些属性和方法可以是类属性/方法或实例属性/方法。实例属性定义对象的状态。实例方法作用于实例属性，即它们用于查询和修改对象的状态。类属性属于类，类方法作用于类属性。

对象可以表示现实世界中的具体或抽象实体。例如，圆形和员工对象表示现实世界的实体，而表示打开文件（文件句柄）的对象是抽象实体。

对象的属性可以包含其他对象以创建对象层次结构。这称为对象组合，在*组合原则*部分有更详细的介绍。

在像Java这样的纯面向对象语言中，你总是需要创建一个类来放置函数。即使你只有类方法而没有属性，在Java中你也必须创建一个类来托管类方法（静态方法）。在Python中，在这些情况下你不需要创建类，只需将函数放入单个模块中，或者创建一个包（目录）并将每个函数放入单独的模块中。

#### 4.1.2：封装

封装使得在对象外部直接更改对象的内部状态变得不可能。封装的思想是对象的状态是对象内部的，只能通过对象的公共方法从外部更改。封装有助于提高安全性并避免数据损坏。不幸的是，Python语言不支持封装，但有一些约定可以用来模拟封装。在封装原则部分有更多相关信息。

#### 4.1.3：抽象

对象只揭示与其他对象使用相关的内部机制，隐藏任何不必要的实现代码。对象方法的调用者不需要知道对象的内部工作原理，他们只遵循对象的公共API。这使得可以在不影响任何外部代码的情况下更改实现细节。

#### 4.1.4：继承

继承允许类被安排在一个表示*is-a*关系的层次结构中。例如，类`Employee`可能继承自类`Person`。父（超）类中的所有属性和方法也以相同的名称出现在子（子）类中。例如，类`Person`可能定义属性`name`和`birth_date`。这些在`Employee`类中也可用。与父类相比，子类还可以添加方法和属性。子类也可以覆盖父类中的方法。例如，`Employee`可能添加属性`employer`和`salary`。这种技术允许轻松重用相同的功能和数据定义，也以直观的方式反映现实世界的关系。

Python还支持多重继承，其中子类可以有多个父类。多重继承的问题在于子类可以继承同名方法的不同版本。默认情况下，应尽可能避免多重继承。一些语言如Java根本不支持多重继承。在Python中，继承多个所谓的*mixin*类也可能有问题，因为两个mixin类也可能有冲突的方法名称。此外，继承会将额外的功能塞进单个子类中，使类变得庞大，并且可能没有单一职责。向类添加功能的更好方法是将类组合成多个其他类（mixin）。这样，就不必担心方法名称可能冲突。

接口允许多重继承。因为Python没有接口，所以如果你使用抽象基类（ABC）或协议，你可以使用多重继承。关于接口的更多信息在下一节中。

#### 4.1.5：接口

接口指定了实现接口的类必须遵守的契约。接口对于多态行为很有用，这将在下一节中描述。接口由一个或多个方法组成，实现它的类必须实现这些方法。Python没有接口，但它有*抽象基类*（ABC）和*协议*。两者都可以用来实现接口。与协议相比，ABC语法更冗长，因为你必须始终用`@abstractmethod`装饰器标记ABC中的方法。

下面是两个从ABC继承实现的接口和一个实现这两个接口的类：

## 面向对象设计原则

```python
from abc import ABC, abstractmethod

class Drawable(ABC):
    @abstractmethod
    def draw(self) -> None:
        pass

class Clickable(ABC):
    @abstractmethod
    def click(self) -> None:
        pass

class Button(Drawable, Clickable):
    def draw(self) -> None:
        print("Button drawn")

    def click(self) -> None:
        print("Button clicked")

button = Button()
button.draw()
button.click()

# Output:
# Button drawn
# Button clicked
```

你也可以结合使用抽象基类和协议。然而，坚持只用一种方式来定义接口是一种良好的实践。下面是一个示例，其中 `Window` 类实现了两个接口，一个是从 ABC（在上面的代码清单中）扩展定义的，另一个是从 Protocol 扩展定义的：

```python
from typing import Protocol

class Draggable(Protocol):
    def dragTo(self, x: int, y: int) -> None:
        pass

class Window(Drawable, Draggable):
    def draw(self) -> None:
        print("Window drawn")

    def dragTo(self, x: int, y: int) -> None:
        print(f"Window dragged to ({x}, {y})")

window = Window()
window.draw()
window.dragTo(200, 300)

# Output:
# Window drawn
# Window dragged to (200, 300)
```

> 在本书的其余部分，我将交替使用 *接口* 和 *协议* 这两个术语。

### 4.1.5.1：接口演进

在接口被定义并被实现类使用之后，如果你想向接口添加方法，你必须在接口中提供默认实现，因为当前实现你接口的类并没有实现你想添加到接口的方法。在实现类是你无法或不想修改的情况下，这一点尤其正确。

假设你有一个 `Message` 接口，包含 `get_data` 和 `get_length_in_bytes` 方法，而你想向该接口添加 `set_queued_at_instant` 和 `get_queued_at_instant` 方法。你可以将这些方法添加到接口，但必须提供默认实现，例如抛出一个指示该方法未实现的错误。

```python
class Message(Protocol):
    def get_data(self):
        # ...

    def get_length_in_bytes(self):
        # ...

    def set_queued_at_instant(timestamp_in_ms: int) -> None:
        raise NotImplementedError()

    def get_queued_at_instant() -> int:
        raise NotImplementedError()
```

### 4.1.6：多态

多态意味着当实际要调用的方法在运行时才决定时，方法是多态的。因此，多态也被称为后期绑定（绑定到特定方法）或动态分派。使用接口变量可以轻松实现多态行为。你可以将任何实现了特定接口的对象赋值给接口变量。当在接口变量上调用方法时，实际调用的方法是根据当前赋值给接口变量的对象类型来决定的。下面是多态行为的一个示例：

```python
drawable: Drawable = Button()
drawable.draw()

# Output:
# Button drawn
```

```python
drawable = Window()
drawable.draw()

# Output:
# Window drawn
```

当你有一个父类类型的变量并将一个子类对象赋值给该变量时，也会表现出多态行为，如下例所示：

```python
class IconButton(Button):
    def draw(self) -> None:
        print("Button with icon drawn")
```

```python
button: Button = Button()
button.draw()

# Output:
# Button drawn
```

```python
button = IconButton()
button.draw()

# Output:
# Button with icon drawn
```

## 4.2：编程范式

最流行的编程语言，包括 Python，都是多范式编程语言。这些语言支持以下编程范式：

- 命令式编程
- 面向对象编程
- 函数式编程

命令式编程是一种编程范式，其重点是提供一系列明确的指令或语句，让计算机遵循以解决问题或实现预期结果。程序由一系列修改程序状态的命令组成，通常通过使用可变变量和赋值来实现。命令式编程强调如何一步一步地实现结果，明确指定控制流和状态变化。典型的命令式编程构造包括变量赋值、状态修改、switch-case 语句、if 语句和各种循环（for、while）。下面是使用命令式编程的代码：

```python
numbers = [1, 2, 3, 4, 5]

doubled_even_numbers = []

for number in numbers:
    if number % 2 == 0:
        doubled_even_numbers.append(number**2)

print(doubled_even_numbers)

# Output:
# [4, 16]
```

函数式编程是一种将计算视为数学函数求值并避免改变状态和可变数据的编程范式。它强调使用不可变数据和函数组合来解决问题。在函数式编程中，函数是一等公民，这意味着它们可以赋值给变量、作为参数传递给其他函数，以及作为结果返回。这使得创建高阶函数成为可能，并促进了模块化、代码重用以及复杂操作的简洁表达。函数式编程避免副作用，倾向于纯函数，这些函数对于给定的输入总是产生相同的输出，并且不会引起副作用，使程序更容易推理和测试。

> 在数学和计算机科学中，高阶函数（HOF）是一个至少执行以下操作之一的函数：1. 接受一个或多个函数作为参数 2. 返回一个函数作为其结果。

下面是使用函数式编程和 Python 列表推导的代码：

```python
numbers = [1, 2, 3, 4, 5]

print([number**2 for number in numbers if number %2 == 0])

# Output:
# [4, 16]
```

如你所见，上面的示例更安全、更短、更简单。例如，没有变量赋值或状态修改。
让我们使用 `map` 和 `filter` 函数来实现上面的代码：

```python
numbers = [1, 2, 3, 4, 5]

is_even = lambda number: number % 2 == 0
doubled = lambda number: number**2

print(list(map(doubled, filter(is_even, numbers))))

# Output:
# [4, 16]
```

在上面的示例中，我们将一个 lambda 赋值给了一个变量。这种做法不符合 PEP 8。我们应该使用 def 来定义函数：

```python
numbers = [1, 2, 3, 4, 5]

def is_even(number):
    return number % 2 == 0

def doubled(number):
    return number**2

print(list(map(doubled, filter(is_even, numbers))))

# Output:
# [4, 16]
```

上面的表达式难以阅读。让我们使用一个变量来存储中间值：

```python
even_numbers = filter(is_even, numbers)
print(list(map(doubled, even_numbers)))

# Output:
# [4, 16]
```

还有另一种方法可以使用函数组合来实现上面的代码。我们可以定义可重用的函数，并从更通用的函数组合出更具体的函数。下面是使用 toolz 库中的 compose 函数进行函数组合的示例。该示例还使用了 functools 模块中的 partial 函数来创建部分应用函数。例如，filterEven 函数是一个部分应用的 filter 函数，其第一个参数绑定到 isEven 函数；类似地，mapDoubled 函数是一个部分应用的 map 函数，其第一个参数绑定到 doubled 函数。compose 函数以以下方式组合两个或多个函数：compose(f, g)(x) 等同于 f(g(x))，compose(f, g, h)(x) 等同于 f(g(h(x)))，以此类推。你可以根据需要/想要组合任意数量的函数。

```python
from functools import partial
from toolz import compose

numbers = [1, 2, 3, 4, 5]

def is_even(number):
    return number % 2 == 0

def doubled(number):
    return number**2

filter_even = partial(filter, is_even)
map_doubled = partial(map, doubled)
doubled_even = compose(list, map_doubled, filter_even)
print(doubled_even(numbers))

# Output:
# [4, 16]
```

在上面的示例中，所有以下函数都可以变得可重用并放入库中：

- is_even
- doubled
- filter_even
- map_doubled

现代代码在可能的情况下应优先选择函数式编程而非命令式编程。与函数式编程相比，命令式编程存在以下缺点：

1. *可变状态*：命令式编程严重依赖可变状态，变量可以在程序执行过程中被修改。这可能导致微妙的错误，并使程序更难推理，因为状态可能不可预测地改变。在函数式编程中，强调不可变性，降低了状态管理的复杂性，使程序更可靠。
2. *副作用*：命令式编程通常涉及副作用，其中函数或操作会修改状态或与外部世界交互。副作用使代码更难测试、推理和调试。另一方面，函数式编程鼓励没有副作用的纯函数，使代码更具模块化、可重用性和可测试性。
3. *并发与并行*：命令式编程在并发场景中可能难以并行化和推理。由于可变状态可能被多个线程或进程修改，可能会发生竞态条件和同步问题。函数式编程，凭借其对不可变性和纯函数的强调，通过消除共享可变状态简化了并发和并行。
4. *缺乏引用透明性*：命令式编程倾向于依赖赋值和就地修改变量的语句。这可能导致代码难以推理，因为存在隐式依赖和代码不同部分之间的隐藏交互。

在函数式编程中，引用透明性是一个关键原则，即表达式可以被其值替换而不改变程序的行为。这一特性使得代码更易于理解、调试和优化。

纯粹的命令式编程也容易导致代码重复、缺乏模块化和抽象问题。这些问题可以通过面向对象编程来解决。

你不应该只使用单一的编程范式。为了在开发软件时最好地利用面向对象编程和函数式编程，你可以在代码库的不同部分发挥各自范式的优势。使用领域驱动设计和面向对象设计来设计应用程序：接口和类。通过将相关行为和（可能可变的）状态封装在类中来实现类。应用SOLID原则和面向对象设计模式等OOP原则。这些原则和模式使得代码模块化且易于扩展，而不会意外破坏现有代码。在实现类和实例方法时，尽可能使用函数式编程。通过创建纯函数来拥抱函数式组合，这些函数接受不可变数据作为输入，并且对于相同的输入总是产生相同的输出，没有副作用。使用高阶函数来组合函数，并从更简单的操作构建复杂操作，例如在OOP中通过将函数作为参数传递给方法或用作回调来利用高阶函数。这提供了更大的灵活性和模块化，使得在OOP框架内进行函数式风格的操作成为可能。同时记得使用函数式编程库，无论是标准库还是第三方库。考虑使用函数式技术进行错误处理，例如*Either*或*Maybe/Optional*类型。这有助于你在没有异常的情况下管理错误，促进更可预测和健壮的代码。这是因为函数签名不会说明它们是否可能抛出异常。你必须始终记住查阅文档并检查函数是否可能抛出异常。

无论使用何种范式，都应力求在代码库中实现不可变性。不可变数据降低了复杂性，避免了共享可变状态，并有助于推理你的代码。倾向于创建新对象或数据结构，而不是修改现有的。

## 4.3：为什么面向对象编程很难？

面向对象编程的基本概念并不难理解，那么为什么掌握OOP很难呢？以下是使OOP变得困难的一些原因：

- 你不能急于编码，而是需要有耐心，应该首先进行面向对象设计。
- 你不可能第一次就做对OOD。你需要有纪律性，并预留时间进行重构。
- 对象组合和继承的区别没有被正确理解，继承被用来代替对象组合，导致OOD存在缺陷。
- SOLID原则没有被理解或遵循。
- 创建具有单一职责的、大小合适的类和函数可能很困难。
    - 例如，你可能有一个单一职责的类，但这个类太大了。你必须意识到需要将这个类拆分成更小的类，这些更小的类组合成原始类。每个更小的类在比原始类更低的抽象层次上具有单一职责。
- 理解和遵循开闭原则可能具有挑战性。
    - 开闭原则的思想是避免修改现有代码，从而避免破坏任何现有的工作代码。例如，如果你有一个集合类，并且还需要一个线程安全的集合类，不要修改现有的集合类，例如通过添加一个构造函数标志来指示集合是否应该是线程安全的。相反，为线程安全集合创建一个全新的类。
- 里氏替换原则并不像看起来那么简单。
    - 例如，如果你有一个基类`Circle`，它有一个`draw`方法。如果你从`Circle`类派生一个`FilledCircle`类，你必须实现`draw`函数，使其首先调用基类方法。但有时可以用派生类方法覆盖基类方法。
- 接口隔离通常在不立即需要时被忽略。这可能会阻碍代码库未来的可扩展性。
- 在许多文本中，依赖倒置原则被用复杂的术语解释。一般来说，依赖倒置原则意味着针对接口编程，而不是针对具体类类型。
- 你不理解依赖注入的价值，也没有使用它。
    - 依赖注入是有效利用其他一些原则（如开闭原则）的要求。
    - 依赖注入使单元测试变得轻而易举，因为你可以创建模拟实现并将其注入到被测试的代码中。
- 你不知道/不理解设计模式，也不知道何时以及如何使用它们。
    - 熟悉设计模式。
    - 有些设计模式比其他模式更有用。你基本上在每个代码库中都会使用一些模式，而有些模式你几乎从不使用。
    - 许多设计模式有助于使代码更具模块化、可扩展性，并有助于避免需要修改现有代码。修改现有代码总是有风险的。你可能会在已经工作的代码中引入错误，有时这些错误非常微妙且难以发现。
    - 学习设计模式需要时间。掌握它们可能需要数年时间，而掌握只能通过在实际代码库中反复使用它们来实现。

能够掌握OOD和OOP是一个终身的过程。你永远不会100%准备好。就像你生活中的任何其他事情一样，变得更好的最佳方式是练习。我已经练习OOD和OOP 29年了，我仍然在进步并定期学习新东西。开始一个非平凡的（爱好/工作）项目，并努力使代码100%干净。每当你认为你已经完成时，将项目搁置一段时间，然后回来，你可能会惊讶地发现有几件事情需要改进！

## 4.4：SOLID原则

本节涵盖了所有五个SOLID原则¹。*依赖倒置原则*被概括为*针对接口编程原则*。五个SOLID原则如下：

- 单一职责原则
- 开闭原则
- 里氏替换原则
- 接口隔离原则
- 依赖倒置原则（概括：针对接口编程原则）

### 4.4.1：单一职责原则

> *类应该有一个职责，代表一个事物或提供单一功能。函数应该只做一件事。*

每个类都应该有一个单一的专用目的。一个类可以代表一个单一的事物，比如银行账户（`Account`类）或员工（`Employee`类），或者提供单一功能，比如解析配置文件（`ConfigFileParser`类）或计算税款（`TaxCalculator`类）。

我们不应该创建一个同时代表银行账户和员工的类。这完全是错误的。当然，一个员工可以*拥有*一个银行账户。但那是另一回事。这被称为对象组合。在对象组合中，一个`Employee`类对象包含一个`Account`类对象。`Employee`类仍然代表一个事物：一个员工（可以拥有一个银行账户）。对象组合将在本章后面更详细地介绍。

在函数级别，每个函数应该执行单一任务。函数名应该描述函数执行的任务，这意味着每个函数名应该包含一个动词。函数名不应该包含*and*这个词，因为它可能意味着函数正在做不止一件事，或者你没有在正确的抽象层次上命名函数。你不应该根据函数执行的步骤来命名函数（例如，`do_this_and_that_and_then_some_third_thing`），而应该使用更高抽象层次的措辞。

当一个类代表某个事物时，它可以包含多个方法。例如，在`Account`类中，可以有像`deposit`和`withdraw`这样的方法。如果这些方法足够简单，并且类中的方法不是太多，这仍然是单一职责。

下面是一个真实的代码示例，其中函数名中使用了*and*这个词：def delete_page_and_all_references(page: Page):
    delete_page(page)
    registry.delete_reference(page.name)
    config_keys.delete_key(page.name.make_key())

在上面的例子中，这个函数似乎做了两件事：删除一个页面并移除所有对该页面的引用。但如果我们查看函数内部的代码，会发现它实际上还做了第三件事：从配置键中删除一个页面键。那么，这个函数是否应该命名为 `delete_page_and_all_references_and_config_key`？这听起来并不合理。函数名的问题在于它与函数语句处于同一抽象层次。函数名应该比函数内部的语句处于更高的抽象层次。

那么，我们应该如何命名这个函数呢？我无法给出确切答案，因为我不知道这个函数的上下文。我们可以将函数简单命名为 `delete`。这会告诉函数调用者，一个页面将被删除。调用者不需要知道与删除页面相关的所有操作。调用者只希望一个页面被删除。函数实现应该满足这个请求，并执行必要的清理操作，比如移除所有对被删除页面的引用等等。

让我们考虑另一个使用 React Hooks 的例子。React Hooks 有一个名为 `useEffect` 的函数，可用于将函数排入队列，以便在组件渲染后运行。`useEffect` 函数可用于在初始渲染（组件挂载后）、每次渲染后或有条件地运行一些代码。这对于单个函数来说责任相当重大。此外，这个函数相当奇怪的名字并没有揭示其目的。`effect` 这个词源于该函数用于将带有副作用的其他函数排入队列运行。副作用这个术语对于函数式语言程序员来说可能很熟悉。它表示一个函数不是纯函数（具有副作用）。

下面是一个 React 函数式组件的示例：

图 4.1. MyComponent.jsx

```
import { useEffect } from "react";

export default function MyComponent() {
  useEffect(() => {
    function startFetchData() {
      // ...
    }

    function subscribeToDataUpdates() {
      // ...
    }

    function unsubscribeFromDataUpdates() {
      // ...
    }

    startFetchData();
    subscribeToDataUpdates();
    return function cleanup() { unsubscribeFromDataUpdates() };
  }, []);

  // JSX to render
  return ...;
}
```

在上面的例子中，由于为依赖项（`useEffect` 函数的第二个参数）提供了空数组，`useEffect` 调用使得 `startFetchData` 和 `subscribeToDataUpdates` 函数在初始渲染后被调用。从提供给 `useEffect` 的函数返回的清理函数将在效果再次运行之前或组件卸载时被调用，在这种情况下，仅在卸载时调用，因为效果只会在初始渲染后运行一次。

让我们想象一下如何改进 `useEffect` 函数。我们可以将与挂载和卸载相关的功能分离到两个不同的函数中：`afterMount` 和 `beforeUnmount`。然后我们可以将上面的例子改为以下代码：

```
export default function MyComponent() {
  function startFetchData() {
    // ...
  }

  function subscribeToDataUpdate() {
    // ...
  }

  function unsubscribeFromDataUpdate() {
    // ...
  }

  afterMount(startFetchData, subscribeToUpdates);
  beforeUnmount(unsubscribeFromDataUpdates)

  // JSX to render
  return ...;
}
```

上面的例子比原始示例更清晰，读者更容易理解。没有多层嵌套函数。你不必返回一个在组件卸载时执行的函数，也不必提供依赖项数组。

让我们再看一个 React 函数式组件的例子：

```
import { useEffect, useState } from "react";

export default function ButtonClickCounter() {
  const [clickCount, setClickCount] = useState(0);

  useEffect(() => {
    function updateClickCountInDocumentTitle() {
      document.title = `Click count: ${clickCount}`;
    }

    updateClickCountInDocumentTitle();
  });
}
```

在上面的例子中，效果在每次渲染后都会被调用（因为没有为 `useEffect` 函数提供依赖项数组）。上面代码中没有任何内容清楚地说明将执行什么以及何时执行。我们仍然使用相同的 `useEffect` 函数，但现在它的行为与之前的例子不同。看起来 `useEffect` 函数在做多件事。如何解决这个问题？让我们再次进行假设性思考。我们可以引入另一个新函数，当我们希望在每次渲染后发生某些事情时可以调用它：

```
export default function ButtonClickCounter() {
  const [clickCount, setClickCount] = useState(0);

  afterEveryRender(function updateClickCountInDocumentTitle() {
    document.title = `Click count: ${clickCount}`;
  });
}
```

上面 React 函数式组件的意图非常清晰：它将在每次渲染后更新文档标题中的点击计数。

让我们优化我们的例子，使点击计数更新仅在点击计数发生变化时发生：

```
import { useEffect, useState } from "react";

export default function ButtonClickCounter() {
  const [clickCount, setClickCount] = useState(0);

  useEffect(() => {
    function updateClickCountInDocumentTitle() {
      document.title = `Click count: ${clickCount}`;
    }

    updateClickCountInDocumentTitle();
  }, [clickCount]);
}
```

注意 `clickCount` 现在被添加到 `useEffect` 函数的依赖项数组中。这意味着效果不是在每次渲染后执行，而仅在点击计数发生变化时执行。

让我们想象一下如何改进上面的例子。我们可以引入一个处理依赖项的新函数：`afterEveryRenderIfChanged`。我们假设的例子现在看起来像这样：

```
export default function ButtonClickCounter() {
  const [clickCount, setClickCount] = useState(0);

  afterEveryRenderIfChanged(
    [clickCount],
    function updateClickCountInDocumentTitle() {
      document.title = `Click count: ${clickCount}`;
    }
  );
}
```

让函数只做一件事也有助于使代码更具可读性。关于原始示例，读者必须查看 `useEffect` 函数调用的末尾才能弄清楚在什么情况下效果函数会被调用。而且，理解并记住缺失依赖项数组和空依赖项数组之间的区别在认知上具有挑战性。好的代码是那种不会让代码读者思考的代码。最好的情况是，代码读起来像散文：*每次渲染后，如果“clickCount”发生变化，就更新文档标题中的点击计数*。

单一职责原则背后的一个想法是，它使得使用下一节描述的*开闭原则*进行软件开发成为可能。当你遵循单一职责原则并需要添加功能时，你将其添加到一个新类中，这意味着你不需要修改现有类。你应该避免修改现有代码，而是通过添加具有单一职责的新类来扩展它。

## 4.4.2：开闭原则

> 软件代码应该对扩展开放，对修改关闭。现有类中的功能不应被修改，而应引入新的类，这些类要么实现一个新的或现有的接口，要么扩展现有的类。

任何时候你发现自己在修改现有类中的某个方法时，你都应该首先考虑是否可以遵循这个原则，以及是否可以避免这种修改。每次修改现有类时，都可能在正常工作的代码中引入一个 bug。这个原则的想法是让正常工作的代码保持不变，这样它就不会被意外破坏。

让我们看一个没有遵循这个原则的例子。我们有以下现有且正常工作的代码：

from typing import Protocol

class Shape(Protocol):
    # ...

class RectangleShape(Shape):
    def __init__(self, width: int, height: int):
        self.__width = width
        self.__height = height

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @width.setter
    def width(self, width: int):
        self.__width = width

    @height.setter
    def height(self, height: int):
        self.__height = height

假设我们接到一个任务，需要为正方形提供支持。我们尝试修改现有的 `RectangleShape` 类，因为正方形也是一种矩形：

```
class RectangleShape(Shape):
    # 创建矩形的构造函数
    def __init__(self, width: int, height: int):
        self.__width = width
        self.__height = height

    # 创建正方形的工厂方法
    @classmethod
    def create_square(cls, side_length: int):
        return cls(side_length, side_length)

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @width.setter
    def width(self, width: int):
        if self.__height == self.__width:
            self.__height = width
        self.__width = width

    @height.setter
    def height(self, height: int):
        if self.__height == self.__width:
            self.__width = height
        self.__height = height
```

我们需要添加一个创建正方形的工厂方法，并修改类中的两个方法。当我们运行测试时，一切似乎都正常。但是，我们在代码中引入了一个微妙的 bug：如果我们创建一个高度和宽度相等的矩形，这个矩形就会变成一个正方形，这可能并非我们想要的结果。这种 bug 在单元测试中可能很难发现。这个例子表明，修改现有类可能会带来问题。我们修改了一个现有类，却意外地破坏了它。

一个更好的引入正方形支持的方案是使用*开闭原则*，并创建一个实现 `Shape` 协议的新类。这样我们就不必修改任何现有类，也就没有意外破坏现有代码中某些东西的风险。下面是新的 `SquareShape` 类：

```
class SquareShape(Shape):
    def __init__(self, side_length: int):
        self.__side_length = side_length

    @property
    def side_length(self) -> int:
        return self.__side_length

    @side_length.setter
    def side_length(self, side_length: int):
        self.__side_length = side_length
```

在以下情况下，可以通过添加新方法来安全地修改现有类：

1.  添加的方法是一个纯函数，即对于相同的参数总是返回相同的值，并且没有副作用，即它不会修改对象的状态。
2.  添加的方法是只读且线程安全的，即它不修改对象的状态，并且在多线程代码的情况下以线程安全的方式访问对象的状态。形状类中只读方法的一个例子是计算形状面积的方法。
3.  类是不可变的，即添加的方法（或任何其他方法）都不能修改对象的状态。

在某些情况下，确实需要修改现有代码。一个例子是工厂。当你引入一个新类时，你需要修改相关的工厂以能够创建该新类的实例。例如，如果我们有一个 `ShapeFactory` 类，我们就需要修改它以支持创建 `SquareShape` 对象。工厂将在本章后面讨论。

另一种情况是添加新的枚举常量。你通常需要修改现有代码来处理新的枚举常量。如果你忘记在现有代码的某个地方添加对新枚举常量的处理，通常就会产生一个 bug。因此，你应该始终用一个会抛出异常的 *default* 分支来保护 switch-case 语句，并用一个会抛出异常的 else 分支来保护 if/else-if 结构。你也可以启用你的静态代码分析工具，以便在 switch 语句缺少 default 分支或 if/else-if 结构缺少 else 分支时报告问题。此外，一些静态代码分析工具可以在你遗漏处理 switch-case 语句中的枚举常量时报告问题。

下面是一个保护 if/else-if 结构的例子：

```
from enum import Enum
from typing import Protocol

class FilterType(Enum):
    INCLUDE = 1
    EXCLUDE = 2

class Filter(Protocol):
    def is_filtered_out(self) -> bool:
        pass

class FilterImpl(Filter):
    def __init__(self, filter_type: FilterType):
        self.__filter_type = filter_type

    def is_filtered_out(self) -> bool:
        if self.__filter_type == FilterType.INCLUDE:
            # ...
        elif self.__filter_type == FilterType.EXCLUDE:
            # ...
        else:
            # 保护措施
            raise ValueError('Invalid filter type')
```

对于字面量类型联合，也可能需要保护措施：

```
from typing import Literal

FilterType = Literal['include', 'exclude']

filter_type: FilterType = # ...

if filter_type == 'include':
    # ...
elif filter_type == 'exclude':
    # ...
else:
    # 保护措施
    raise ValueError('Invalid filter type')
```

将来，如果向 `FilterType` 类型添加了一个新的字面量，而你忘记了更新 if 语句，你将会得到一个错误，而不是仅仅静默地通过 if 语句而不采取任何操作。

从上面的例子中我们可以注意到，通过更好的面向对象设计，可以避免 if/else-if 结构。例如，我们可以创建一个 `Filter` 协议和两个独立的类 `IncludeFilter` 和 `ExcludeFilter`，它们都实现 `Filter` 协议。使用面向对象设计允许我们消除 `FilterType` 枚举和 if/else-if 结构。这被称为*用多态替换条件*重构技术。重构将在下一章中更详细地讨论。下面是上述示例重构为更面向对象的版本：

```
from typing import Protocol

class Filter(Protocol):
    def is_filtered_out(self) -> bool:
        pass

class IncludeFilter(Filter):
    # ...
    def is_filtered_out(self) -> bool:
        # ...

class ExcludeFilter(Filter):
    # ...
    def is_filtered_out(self) -> bool:
        # ...
```

## 4.4.3: 里氏替换原则

> *超类的对象应该可以被其子类的对象替换，而不会破坏应用程序。也就是说，子类对象的行为方式与超类对象相同。*

遵循*里氏替换原则*可以保证类型层次结构中类型的语义互操作性。

让我们看一个 `RectangleShape` 类和派生的 `SquareShape` 类的例子：

```
from typing import Protocol

class Shape(Protocol):
    def draw(self) -> None:
        pass

class RectangleShape(Shape):
    def __init__(self, width: int, height: int):
        self.__width = width
        self.__height = height

    def draw(self):
        # ...

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @width.setter
    def width(self, width: int):
        self.__width = width

    @height.setter
    def height(self, height: int):
        self.__height = height

class SquareShape(RectangleShape):
    def __init__(self, side_length: int):
        super().__init__(side_length, side_length)

    @RectangleShape.width.setter
    def width(self, width: int):
        RectangleShape.width.fset(self, width)
        RectangleShape.height.fset(self, width)

    @RectangleShape.height.setter
    def height(self, height: int):
        RectangleShape.width.fset(self, height)
        RectangleShape.height.fset(self, height)
```

上面的例子没有遵循里氏替换原则，因为你不能单独设置正方形的宽度和高度。这意味着从面向对象的角度来看，正方形不是矩形。当然，在数学上，正方形是矩形。但是，考虑到 `RectangleShape` 类的上述公共 API，我们可以得出结论：正方形不是矩形，因为正方形无法完全实现 `RectangleShape` 类的 API。我们不能用正方形对象替换矩形对象。我们需要做的是实现 `SquareShape` 类，而不从 `RectangleShape` 类派生：

## 4.4.4：接口隔离与多重继承原则

> 将一个较大的接口隔离为具有单一能力/行为的微接口，并通过继承多个微接口来构建更大的接口。

在本节的剩余部分，我们将使用 Python 特有的术语“协议”来代替“接口”。让我们以几个汽车类为例：

```python
from typing import Protocol

from Location import Location

class Automobile(Protocol):
    def drive(self, start: Location, destination: Location) -> None:
        pass

    def carry_cargo(
        self,
        volume_in_cubic_meters: float,
        weight_in_kgs: float
    ) -> None:
        pass

class PassengerCar(Automobile):
    # 实现 drive 和 carry_cargo

class Van(Automobile):
    # 实现 drive 和 carry_cargo

class Truck(Automobile):
    # 实现 drive 和 carry_cargo

class ExcavatingAutomobile(Automobile):
    def excavate(self) -> None:
        pass

class Excavator(ExcavatingAutomobile):
    # 实现 drive、carry_cargo 和 excavate
```

请注意 `Automobile` 协议声明了两个方法。如果我们以后想引入其他只能驾驶但无法载货的车辆，这可能会限制我们的软件。在早期阶段，我们应该从 `Automobile` 协议中隔离出两个微协议。一个微协议定义单一的能力或行为。隔离后，我们将得到以下两个微协议：

```python
class Drivable(Protocol):
    def drive(self, start: Location, destination: Location) -> None:
        pass

class CargoCarriable(Protocol):
    def carry_cargo(
        self,
        volume_in_cubic_meters: float,
        weight_in_kgs: float
    ) -> None:
        pass
```

现在我们有了两个协议，我们也可以在代码库中单独使用这些接口。例如，我们可以有一个可驾驶对象的列表，或者一个可以载货的对象的列表。不过，我们仍然希望有一个汽车协议。我们可以使用*协议多重继承*来重新定义 `Automobile` 协议，以扩展这两个微协议：

```python
class Automobile(Drivable, CargoCarriable):
    pass
```

如果我们查看 `ExcavatingAutomobile` 协议，可以注意到它扩展了 `Automobile` 协议并添加了挖掘行为。同样，如果我们想要一个不是汽车的挖掘机器，就会遇到问题。挖掘行为应该被隔离到它自己的微协议中：

```python
class Excavating(Protocol):
    def excavate(self) -> None:
        pass
```

我们可以再次使用协议多重继承，将 `ExcavatingAutomobile` 协议重新定义如下：

```python
class ExcavatingAutomobile(Excavating, Automobile):
    pass
```

`ExcavatingAutomobile` 协议现在扩展了三个微协议：`Excavating`、`Drivable` 和 `CargoCarriable`。在你的代码库中，无论你需要一个可挖掘、可驾驶或可载货的对象，你都可以在那里使用 `Excavator` 类的实例。

让我们再看一个通用集合协议的例子。我们应该能够遍历一个集合，也能够比较两个集合是否相等。首先，我们为迭代器定义一个通用的 `Iterator` 协议。它有两个方法，如下所述：

```python
from typing import Protocol, TypeVar

T = TypeVar('T')

class Iterator(Protocol[T]):
    def has_next_elem(self) -> bool:
        pass

    def get_next_elem(self) -> T:
        pass
```

接下来，我们可以定义集合协议：

```python
class Collection(Protocol[T]):
    def create_iterator(self) -> Iterator[T]:
        pass

    def equals(self, another_collection: 'Collection[T]') -> bool:
        pass
```

`Collection` 是一个包含两个不相关方法的协议。让我们将这些方法隔离成两个微协议：`Iterable` 和 `Equatable`。`Iterable` 接口用于你可以迭代的对象。它有一个用于创建新迭代器的方法。`Equatable` 协议的 `equals` 方法比 `Collection` 协议中的 `equals` 方法更通用。你可以将一个 `Equatable` 对象与另一个类型为 `T` 的对象进行比较：

```python
class Iterable(Protocol[T]):
    def create_iterator(self) -> Iterator[T]:
        pass
```

```python
class Equatable(Protocol[T]):
    def equals(self, another_object: T) -> bool:
        pass
```

我们可以使用协议多重继承来重新定义 `Collection` 协议，如下所示：

```python
class Collection(Iterable[T], Equatable['Collection[T]']):
    pass
```

我们可以通过遍历两个集合中的元素并检查它们是否相等来实现 `equals` 方法：

```python
from abc import abstractmethod

class AbstractCollection(Collection[T]):
    @abstractmethod
    def create_iterator(self) -> Iterator[T]:
        pass

    @staticmethod
    def __are_equal(iterator: Iterator[T], another_iterator: Iterator[T]):
        while iterator.has_next_elem():
            if another_iterator.has_next_elem():
                if (
                    iterator.get_next_elem()
                    != another_iterator.get_next_elem()
                ):
                    return False
            else:
                return False
        return True

    def equals(self, another_collection: Collection[T]):
        iterator = self.create_iterator()
        another_iterator = another_collection.create_iterator()
        collections_are_equal = self.__are_equal(
            iterator, another_iterator
        )
        return (
            False
            if another_iterator.has_next_elem()
            else collections_are_equal
        )
```

集合也可以进行比较。让我们引入对这种集合的支持。首先，我们定义一个用于比较一个对象与另一个对象的通用 `Comparable` 协议：

```python
from typing import Protocol, Literal

ComparisonResult = Literal['isLessThan', 'areEqual', 'isGreaterThan', 'unspecified']
```

```python
class Comparable(Protocol[T]):
    def compare_to(self, another_object: T) -> ComparisonResult:
        pass
```

现在我们可以引入一个可比较集合协议，允许比较两个相同类型的集合：

```python
class ComparableCollection(Comparable[Collection[T]], Collection[T]):
    pass
```

让我们为元素可比较的集合定义一个通用的排序算法：

```python
U = TypeVar('U', bound=ComparableCollection)

def sort(collection: U) -> U:
    # ...
```

让我们创建两个协议 `Inserting` 和 `InsertingIterable`，用于那些实例元素可以被插入的类：

```python
class Inserting(Protocol[T]):
    def insert(self, element: T) -> None:
        pass
```

```python
class InsertingIterable(Inserting[T], Iterable[T]):
    pass
```

让我们重新定义 `Collection` 协议以扩展 `InsertingIterable` 协议，因为一个集合是可迭代的，并且你可以向集合中插入元素。

```python
class SquareShape(Shape):
    def __init__(self, side_length: int):
        self.__side_length = side_length

    def draw(self):
        # ...

    @property
    def side_length(self) -> int:
        return self.__side_length

    @side_length.setter
    def side_length(self, side_length: int):
        self.__side_length = side_length
```

里氏替换原则要求：

-   子类必须实现超类的 API 并保留（或在某些情况下替换）超类的功能。
-   超类不应有受保护的字段，因为它允许子类修改超类的状态，这可能导致超类行为不正确。

下面是一个子类在 `do_something` 方法中扩展超类行为的例子。超类的功能在子类中得以保留，使得子类对象可以替代超类对象。

```python
class SuperClass:
    # ...

    def do_something(self):
        # ...


class SubClass(SuperClass):
    # ...

    def do_something(self):
        super().do_something()

        # 一些额外的行为...
```

让我们看一个使用上述策略的具体例子。我们定义了以下 `CircleShape` 类：

```python
from typing import Protocol

class Shape(Protocol):
    def draw(self) -> None:
        pass
```

```python
class CircleShape(Shape):
    def draw(self):
        # 在这里绘制圆形轮廓
```

接下来，我们引入一个用于填充圆形的类：

```python
class FilledCircleShape(CircleShape):
    def draw(self):
        super().draw() # 绘制圆形轮廓
        # 填充圆形
```

`FilledCircleShape` 类满足里氏替换原则的要求。我们可以在任何需要 `CircleShape` 类实例的地方使用 `FilledCircleShape` 类的实例。`FilledCircleShape` 类完成了 `CircleShape` 类的所有功能，并添加了一些行为（= 填充圆形）。

你也可以在子类中完全替换超类的功能：

```python
class ReverseList(list):
    def __iter__(self):
        return ReverseListIterator(self)
```

上述子类实现了超类的 API 并保留了其行为：迭代器方法仍然返回一个迭代器。它只是返回了一个与超类不同的迭代器。

## 4.4.5：面向接口编程原则（通用依赖倒置原则）

> 不要编写内部依赖是具体对象类型的程序——而应面向接口编程。此规则的一个例外是没有行为的数据类（不包括简单的getter/setter）。

接口用于定义抽象基类型。可以引入各种实现该接口的实现类。当你想改变程序的行为时，你可以创建一个实现接口的新类，然后使用该类的实例。通过这种方式，你可以实践*开闭原则*。你可以将此原则视为有效使用*开闭原则*的前提。*面向接口编程原则*是SOLID原则中*依赖倒置原则*的泛化：

*依赖倒置原则*是一种使软件类松耦合的方法论。遵循该原则时，从高层类到低层类的传统依赖关系被反转，从而使高层类独立于低层实现细节。

*依赖倒置原则*指出：

1) 高层类不应从低层类导入任何内容
2) 抽象（=接口）不应依赖于具体实现（类）
3) 具体实现（类）应依赖于抽象（=接口）

接口始终是抽象类型，不能被实例化。下面是一个接口的示例：

```python
from typing import Protocol

class Shape(Protocol):
    def draw(self) -> None:
        pass

    def calculate_area(self) -> float:
        pass
```

接口的名称描述的是抽象的事物，你无法为其创建对象。在上面的例子中，Shape显然是抽象的事物。你无法创建Shape的实例然后绘制它或计算其面积，因为你不知道它是什么形状。但是当一个类实现了一个接口时，就可以创建代表该接口的类的具体对象。下面是三个实现Shape接口的不同类的示例：

```python
from math import pi

class CircleShape(Shape):
    def __init__(self, radius: int):
        self.__radius = radius

    def draw(self):
        # ...

    def calculate_area(self) -> float:
        return pi * self.__radius**2

class RectangleShape(Shape):
    def __init__(self, width: int, height: int):
        self.__width = width
        self.__height = height

    def draw(self):
        # ...

    def calculate_area(self) -> float:
        return self.__width * self.__height

class SquareShape(RectangleShape):
    def __init__(self, side_length: int):
        super().__init__(side_length, side_length)
```

在代码中使用形状时，我们应该面向Shape接口编程。在下面的例子中，我们让一个高层类Canvas依赖于Shape接口，而不是依赖于任何低层类（CircleShape、RectangleShape或SquareShape）。现在，高层Canvas类和所有低层形状类都只依赖于抽象，即Shape接口。我们还可以注意到，高层类Canvas没有从低层类导入任何内容。同时，抽象Shape 也不依赖于具体实现（类）。

```python
from typing import Final

class Canvas:
    def __init__(self):
        self.__shapes: Final[list[Shape]] = []

    def add(self, shape: Shape):
        self.__shapes.append(shape)

    def draw_shapes(self):
        for shape in self.__shapes:
            shape.draw()
```

Canvas对象可以包含任何形状并绘制任何形状。它可以处理当前定义的任何具体形状，以及未来定义的任何新形状。

如果你没有面向接口编程，也没有使用依赖倒置原则，你的Canvas类将如下所示：

```python
class Collection(InsertingIterable[T]):
    pass
```

接下来，我们为集合引入两个泛型算法：map和filter。我们可以意识到这些算法处理的对象比集合更抽象。我们受益于协议隔离，因为我们可以使用Iterable和InsertingIterable协议来创建泛型map和filter算法，而不是使用Collection协议。之后，可以引入一些额外的非集合可迭代对象，它们也可以利用这些算法。下面是map和filter函数的实现：

```python
from collections.abc import Callable
from typing import TypeVar

T = TypeVar('T')
U = TypeVar('U')

def map(
    source: Iterable[T],
    mapped: Callable[[T], U],
    destination: InsertingIterable[U],
) -> InsertingIterable[U]:
    source_iterator = source.create_iterator()
    while source_iterator.has_next_elem():
        source_element = source_iterator.get_next_elem()
        destination.insert(mapped(source_element))
    return destination

def filter(
    source: Iterable[T],
    is_included: Callable[[T], bool],
    destination: InsertingIterable[T],
) -> InsertingIterable[T]:
    source_iterator = source.create_iterator()
    while source_iterator.has_next_elem():
        source_element = source_iterator.get_next_elem()
        if is_included(source_element):
            destination.insert(source_element)
    return destination
```

让我们定义以下具体的集合类：

```python
class List(Collection[T]):
    def __init__(self, *args: T):
        # ...

    # ...
```

```python
class Stack(Collection[T]):
    # ...
```

```python
class MySet(Collection[T]):
    # ...
```

现在我们可以将map和filter算法与上面定义的集合类一起使用：

```python
numbers = List(1, 2, 3, 3, 3, 50, 60)
is_less_than_10 = lambda nbr: nbr < 10
unique_less_than_10_numbers = filter(numbers, is_less_than_10, Set())

doubled = lambda nbr: 2 * nbr
stack_of_doubled_numbers = map(numbers, doubled, Stack())
```

让我们创建map算法的异步版本：

```python
from typing import Protocol
```

```python
class Closeable(Protocol):
    def close(self) -> None:
        pass
```

```python
class MaybeInserting(Protocol[T]):
    async def try_insert(self, value: T) -> None:
        pass
```

```python
class CloseableMaybeInserting(Closeable, MaybeInserting[T]):
    pass
```

```python
class MapError(Exception):
    pass
```

```python
async def try_map(
    source: Iterable[T],
    mapped: Callable[[T], U],
    destination: CloseableMaybeInserting[U],
) -> None:
    source_iterator = source.create_iterator()
    try:
        while source_iterator.has_next_elem():
            source_element = source_iterator.get_next_elem()
            await destination.try_insert(mapped(source_element))
    except Exception as error:
        raise MapError(error)
    finally:
        destination.close()
```

让我们创建一个实现CloseableMaybeInserting协议的FileLineInserter类：

```python
from aiofiles import open
```

```python
class FileLineInserter(CloseableMaybeInserting[T]):
    def __init__(self, file_path_name: str):
        self.__file = None
        self.__file_path_name = file_path_name

    async def try_insert(self, value: T):
        if self.__file == None:
            self.__file = await open(self.__file_path_name, mode='w')
        line = str(value) + '\n'
        await self.__file.write(line)

    def close(self):
        self.__file.close()
```

让我们使用上面定义的try_map算法和FileLineInserter类，将翻倍后的数字（每行一个数字）写入名为file.txt的文件：

```python
from asyncio import run
```

```python
numbers = List(1, 2, 3, 2, 1, 50, 60)
doubled = lambda nbr: 2 * nbr
```

```python
async def my_func():
    try:
        await try_map(numbers, doubled, FileLineInserter('file.txt'))
    except MapError as error:  # error will be always MapError type.
        print(str(error))
```

```python
run(my_func())
```

Python标准库以非常典范的方式利用了接口隔离和多重接口继承。例如，Python标准库定义了下面这些只实现单个方法的抽象基类（或接口）。也就是说，它们是微接口。

| 抽象基类 | 方法 |
| --- | --- |
| Container | `__contains__` |
| Hashable | `__hash__` |
| Iterable | `__iter__` |
| Sized | `__len__` |
| Callable | `__call__` |
| Awaitable | `__await__` |
| AsyncIterable | `__aiter__` |

Python标准库还包含以下继承自多个（微）接口的抽象基类：

| 抽象基类 | 继承自 |
| --- | --- |
| Collection | Sized, Iterable, Container |
| Sequence | Collection, Reversible |

from typing import Final

class Circle:
    def draw(self):
        # ...

class Rectangle:
    def draw(self):
        # ...

class Square:
    def draw(self):
        # ...

class Canvas:
    def __init__(self):
        self.__shapes: Final[list[Circle | Rectangle | Square]] = []

    def add(self, shape: Circle | Rectangle | Square):
        self.__shapes.append(shape)

    def draw_shapes(self):
        for shape in self.__shapes:
            shape.draw()

上述高级的 `Canvas` 类与所有低级类（`Circle`、`Rectangle` 和 `Square`）紧密耦合。如果需要新的形状类型，必须修改 `Canvas` 类中的类型注解。如果任何低级类的公共 API 发生变化，`Canvas` 类也需要相应地进行修改。在上面的例子中，我们隐式地指定了 `draw` 方法的协议：它不接受参数并返回 `None`。

让我们再看一个例子。如果你读过关于面向对象设计的书籍或文章，你可能遇到过类似下面示例中的内容：

```
class Dog:
    def walk(self):
        # ...

    def bark(self):
        # ...

class Fish:
    def swim(self):
        # ...

class Bird:
    def fly(self):
        # ...

    def sing(self):
        # ...
```

上面定义了三个具体的实现，但没有定义接口。假设我们正在制作一个包含不同动物的游戏。编写游戏时要做的第一件事就是记住要面向接口编程，因此我们引入一个 `Animal` 协议，可以将其用作抽象基类型。让我们尝试基于上面的具体实现来创建 `Animal` 协议：

```
from typing import Protocol

class Animal(Protocol):
    def walk(self) -> None:
        pass

    def bark(self) -> None:
        pass

    def swim(self) -> None:
        pass

    def fly(self) -> None:
        pass

    def sing(self) -> None:
        pass

class Dog(Animal):
    def walk(self):
        # ...

    def bark(self):
        # ...

    def swim(self):
        raise NotImplementedError()

    def fly(self):
        raise NotImplementedError()

    def sing(self):
        raise NotImplementedError()
```

上述方法是错误的。我们声明 `Dog` 类实现了 `Animal` 协议，但它并没有真正实现。它只实现了 `walk` 和 `bark` 方法，而其他方法则抛出异常。我们应该能够在任何需要动物的地方替换任何具体的动物实现。但这是不可能的，因为如果我们有一个 `Dog` 对象，我们无法安全地调用 `swim`、`fly` 或 `sing` 方法，因为它们总是会抛出异常。

问题在于我们在定义接口之前定义了具体类。这种方法是错误的。我们应该先指定接口，然后再定义具体实现。我们上面所做的恰恰相反。

在定义接口时，我们应该记住我们是在定义一个抽象基类型，因此必须从抽象的角度思考。我们必须考虑我们希望动物在游戏中做什么。如果我们看看 `walk`、`fly` 和 `swim` 方法，它们都是具体的动作。但这三个具体动作的共同抽象动作是什么？是 *移动*。而行走、飞行和游泳都是移动的方式。同样，如果我们看看 `bark` 和 `sing` 方法，它们也是具体的动作。这两个具体动作的共同抽象动作是什么？是 *发出声音*。而吠叫和歌唱都是发出声音的方式。如果我们使用这些抽象动作，我们的 `Animal` 协议将如下所示：

```
from typing import Protocol

class Animal(Protocol):
    def move(self) -> None:
        pass

    def make_sound(self) -> None:
        pass
```

我们现在可以重新定义动物类以实现新的 `Animal` 协议：

```
class Dog(Animal):
    def move(self):
        # walk

    def make_sound(self):
        # bark

class Fish(Animal):
    def move(self):
        # swim

    def make_sound(self):
        # Intentionally no operation
        # (Fishes typically don't make sounds)
        pass

class Bird(Animal):
    def move(self):
        # fly

    def make_sound(self):
        # sing
```

现在我们有了正确的面向对象设计，可以面向 `Animal` 接口编程。当我们希望动物移动时，可以调用 `move` 方法；当我们希望动物发出声音时，可以调用 `make_sound` 方法。

意识到有些鸟根本不会飞之后，我们可以轻松地增强我们的设计。我们可以引入两种不同的实现：

```
from abc import abstractmethod

class AbstractBird(Animal):
    @abstractmethod
    def move(self):
        pass

    def make_sound(self):
        # sing

class FlyingBird(AbstractBird):
    def move(self):
        # fly

class NonFlyingBird(AbstractBird):
    def move(self):
        # walk
```

我们可能后来还会意识到，并非所有的鸟都会歌唱，而是发出不同的声音。例如，鸭子会嘎嘎叫。与其像上面那样使用继承，一个更好的替代方案是使用 *对象组合*。我们将 `Bird` 类组合为用于移动和发出声音的行为类：

```
class Mover(Protocol):
    def move(self) -> None:
        pass

class SoundMaker(Protocol):
    def make_sound(self) -> None:
        pass

class Bird(Animal):
    def __init__(self, mover: Mover, sound_maker: SoundMaker):
        self.__mover = mover
        self.__sound_maker = sound_maker

    def move(self):
        self.__mover.move()

    def make_sound(self):
        self.__sound_maker.make_sound()
```

现在我们可以创建具有各种移动和发声行为的鸟。我们可以使用 *工厂模式* 来创建不同的鸟。*工厂模式* 将在本章后面更详细地描述。让我们引入三种不同的移动和发声行为，以及一个工厂来制造三种鸟：金翅雀、鸵鸟和家鸭。

```
from enum import Enum

class Flyer(Mover):
    def move(self):
        # fly

class Runner(Mover):
    def move(self):
        # run

class Walker(Mover):
    def move(self):
        # walk

class GoldfinchSoundMaker(SoundMaker):
    def make_sound(self):
        # Sing goldfinch specific songs

class OstrichSoundMaker(SoundMaker):
    def make_sound(self):
        # Make ostrich specific sounds like whistles,
        # hoots, hisses, growls, and deep booming growls
        # that sound like the roar of a lion

class Quacker(SoundMaker):
    def make_sound(self):
        # quack

class BirdType(Enum):
    GOLDFINCH = 1
    OSTRICH = 2
    DOMESTIC_DUCK = 3

class BirdFactory:
    def create_bird(self, bird_type: BirdType):
        match bird_type:
            case BirdType.GOLDFINCH:
                return Bird(Flyer(), GoldfinchSoundMaker())
            case BirdType.OSTRICH:
                return Bird(Runner(), OstrichSoundMaker())
            case BirdType.DOMESTIC_DUCK:
                return Bird(Walker(), Quacker())
            case _:
                raise ValueError('Unsupported bird type')
```

## 4.5：整洁的微服务设计原则

> 整洁的微服务设计提倡面向对象的设计，通过使用依赖倒置原则（面向接口编程）将软件划分为不同的层来实现关注点分离。

Bob 大叔在他的著作《整洁架构》中使用 *整洁架构* 这个术语来指代同样的原则。我没有使用“架构”这个术语，因为我保留该术语来指代比单个服务更大的东西（即一个软件系统）的设计。在这里，我们专注于以特定方式进行面向对象设计（OOD）来设计单个（微）服务。

整洁的微服务设计带来以下好处：

- 不依赖于任何单一框架
- 不依赖于任何单一的 API 技术，如 REST 或 GraphQL
- 可进行单元测试
- 不依赖于特定客户端（适用于 Web、桌面、控制台和移动客户端）
- 不依赖于特定数据库
- 不依赖于任何特定的外部服务实现

一个整洁的 API 微服务设计包含以下层：

- 控制器、接口适配器
- 用例
- （业务）实体

用例和实体共同构成了服务的 *模型*，也称为 *业务逻辑*。

## 4.3 清洁微服务设计

上图中依赖的方向用箭头表示。我们可以看到，微服务API依赖于我们创建的控制器。控制器依赖于用例。用例层依赖于（业务）实体。用例层的目的是编排对（业务）实体的操作。在上图中，最常变化的软件部分位于外层（例如，像REST、GraphQL这样的控制器技术和数据库），而最稳定的部分位于中心（实体）。让我们以一个实体为例，比如银行账户。我们知道它不常变化。它有几个关键属性：所有者、账号、利率和余额（可能还有一些其他属性），但银行账户是什么或做什么，几十年来都没有改变。但我们不能对API技术或数据库技术说同样的话。与银行账户相比，这些是变化速度更快的东西。由于上图中依赖的方向，外层的变化不会影响内层。使用清洁微服务设计可以轻松更改所使用的API技术和数据库，例如从REST更改为其他技术，从SQL数据库更改为NoSQL数据库。所有这些更改都可以在不影响业务逻辑（用例和实体层）的情况下完成。

让我们举一个真实的例子，创建一个名为*order-service*的API微服务，它处理电子商务软件系统中的订单。首先，我们使用FastAPI定义一个REST API控制器：

## 图 4.4. controllers/RestOrderController.py

```python
from dependency_injector.wiring import Provide
from fastapi import APIRouter

from ..dtos.InputOrder import InputOrder
from ..dtos.OutputOrder import OutputOrder
from ..services.OrderService import OrderService

# In the request handler functions of the below class
# remember to add authorization, necessary audit logging and
# observability (metric updates) for production.
# Examples are provided in later chapters of this book

class RestOrderController:
    __order_service: OrderService = Provide['order_service']

    def __init__(self):
        self.__router = APIRouter()
        self.__router.add_api_route(
            '/orders/',
            self.create_order,
            methods=['POST'],
            status_code=201,
            response_model=OutputOrder,
        )
        self.__router.add_api_route(
            '/orders/{id_}',
            self.get_order,
            methods=['GET'],
            response_model=OutputOrder,
        )

    @property
    def router(self):
        return self.__router

    def create_order(self, input_order: InputOrder) -> OutputOrder:
        return self.__order_service.create_order(input_order)

    def get_order(self, id_: int) -> OutputOrder:
        return self.__order_service.get_order(id_)

    # Rest of API endpoints...
```

如上图所示，微服务提供的API依赖于控制器。该API目前是REST API，但我们可以创建并使用GraphQL控制器。那么我们的API，它依赖于控制器，就是一个GraphQL API。下面是使用FastAPI和Strawberry库实现的GraphQL控制器的部分代码：

## 图 4.5. controllers/GraphQlOrderController.py

```python
import strawberry
from dependency_injector.wiring import Provide
from strawberry.fastapi import GraphQLRouter

from ..graphqltypes.InputOrder import InputOrder
from ..graphqltypes.OutputOrder import OutputOrder
from ..services.OrderService import OrderService

order_service: OrderService = Provide['order_service']

# In the request handler functions of the below class
# remember to add authorization, necessary audit logging and
# observability (metric updates) for production.
# Examples are provided in later chapters of this book


class GraphQlOrderController:
    @strawberry.type
    class Query:
        @strawberry.field
        def order(self, id: int) -> OutputOrder:
            output_order = order_service.get_order(id)
            return OutputOrder.from_pydantic(output_order)

    @strawberry.type
    class Mutation:
        @strawberry.mutation
        def create_order(self, input_order: InputOrder) -> OutputOrder:
            output_order = order_service.create_order(
                input_order.to_pydantic()
            )
            return OutputOrder.from_pydantic(output_order)

    __schema = strawberry.Schema(query=Query, mutation=Mutation)
    __router = GraphQLRouter(__schema, path='/graphql')

    @property
    def router(self):
        return self.__router
```

`RestOrderController`和`GraphQlOrderController`类依赖于`OrderService`接口，该接口是用例层的一部分。请注意，控制器不依赖于用例的具体实现，而是根据*依赖倒置原则*依赖于一个接口。下面是`OrderService`协议的定义：

## 图 4.6. services/OrderService.py

```python
from typing import Protocol

from ..dtos.InputOrder import InputOrder
from ..dtos.OutputOrder import OutputOrder

class OrderService(Protocol):
    def create_order(self, input_order: InputOrder) -> OutputOrder:
        pass

    def get_order(self, id_: int) -> OutputOrder:
        pass

    def get_order_by_user_id(self, user_id: int) -> OutputOrder:
        pass

    def update_order(self, id_: int, order_update: InputOrder) -> None:
        pass

    def delete_order(self, id_: int) -> None:
        pass
```

下面的OrderServiceImpl类实现了OrderService协议：

## 图 4.7. services/OrderServiceImpl.py

```python
from dependency_injector.wiring import Provide

from ..dtos.InputOrder import InputOrder
from ..dtos.OutputOrder import OutputOrder
from ..errors.EntityNotFoundError import EntityNotFoundError
from ..repositories.OrderRepository import OrderRepository
from ..services.OrderService import OrderService

class OrderServiceImpl(OrderService):
    __order_repository: OrderRepository = Provide['order_repository']

    def create_order(self, input_order: InputOrder) -> OutputOrder:
        order = self.__order_repository.save(input_order)
        return OutputOrder.from_orm(order)

    def get_order(self, id_: int) -> OutputOrder:
        order = self.__order_repository.find(id_)

        if order is None:
            raise EntityNotFoundError('Order', id_)

        return OutputOrder.from_orm(order)

    # Rest of the methods...
```

OrderServiceImpl类依赖于订单仓库。这个依赖也是倒置的。OrderServiceImpl类只依赖于OrderRepository接口。订单仓库用于编排订单实体的持久化。请注意，这里没有对数据库的直接依赖。

下面是OrderRepository协议：

## 图 4.8. repositories/OrderRepository.py

```python
from typing import Protocol

from ..dtos.InputOrder import InputOrder
from ..entities.Order import Order

class OrderRepository(Protocol):
    def initialize(self) -> None:
        pass

    def save(self, order: InputOrder) -> Order:
        pass

    def find(self, order_id: int) -> Order | None:
        pass

    # Rest of methods...
```

OrderRepository接口只依赖于Order实体类。你可以引入一个称为接口适配器的类来实现OrderRepository接口。数据库接口适配器将特定的具体数据库适配到OrderRepository接口。实体类不依赖于任何东西，除了其他实体来创建层次结构实体。例如，Order实体由OrderItem实体组成。让我们为SQL数据库引入一个OrderRepository接口适配器类：

## 图 4.9. repositories/SqlOrderRepository.py

```python
import os

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from ..dtos.InputOrder import InputOrder
from ..entities.Base import Base
from ..entities.Order import Order
from ..errors.DatabaseError import DatabaseError
from ..repositories.OrderRepository import OrderRepository
from ..utils import to_entity_dict

class SqlOrderRepository(OrderRepository):
    __engine = create_engine(os.environ.get('DATABASE_URL'))
    __SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=__engine
    )

    def __init__(self):
        try:
            Base.metadata.create_all(bind=self.__engine)
```

## 面向对象设计原则

```python
except SQLAlchemyError as error:
    # 记录错误
    raise error

def save(self, input_order: InputOrder) -> Order:
    with self.__SessionLocal() as db_session:
        try:
            order = Order(**to_entity_dict(input_order))
            db_session.add(order)
            db_session.commit()
            db_session.refresh(order)
            return order
        except SQLAlchemyError as error:
            raise DatabaseError(error)

def find(self, id_: int) -> Order | None:
    with self.__SessionLocal() as db_session:
        try:
            return db_session.get(Order, id_)
        except SQLAlchemyError as error:
            raise DatabaseError(error)

# 其余方法...
```

上述类要求所使用的数据库服务配置在名为 `DATABASE_URL` 的环境变量中。对于本地 MySQL 数据库，你可以使用：

```bash
# 假设用户名:密码为 root:password
export DATABASE_URL=mysql://root:password@localhost:3306/orderservice
```

> 如果你对 `to_entity_dict` 方法的实现感兴趣，请查看附录 A。

如果你想将数据库更改为 MongoDB，可以通过实现一个新的接口适配器来完成，该适配器实现了 `OrderRepository` 接口。在接下来的 *数据库原则* 章节中，我们将实现一个 MongoDB 仓库。

![订单服务的清晰微服务设计](img/cbd069395d7b824346b69b1f92e0fb4a_123_0.png)

图 4.10. 订单服务的清晰微服务设计

在实现清晰的微服务设计时，所有组件都通过配置和依赖注入连接在一起。依赖注入使用 *dependency-injector* 库进行配置，并定义一个 DiContainer 类：

### 图 4.11. DiContainer.py

```python
from dependency_injector import containers, providers

from .controllers.GraphQlOrderController import GraphQlOrderController
from .repositories.SqlOrderRepository import SqlOrderRepository
from .services.OrderServiceImpl import OrderServiceImpl

class DiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            '.services.OrderServiceImpl',
            '.controllers.RestOrderController',
            '.controllers.FlaskRestOrderController',
            '.controllers.GraphQlOrderController',
            '.repositories.SqlOrderRepository',
        ]
    )

    order_service = providers.Singleton(OrderServiceImpl)
    order_repository = providers.Singleton(SqlOrderRepository)
    # order_controller = providers.Singleton(RestOrderController)
    order_controller = providers.Singleton(GraphQlOrderController)
```

如果我们想更改微服务中的某些内容，可以创建一个新类并在 DiContainer 中使用该新类。我们可以为不同类型的数据库创建一个新的仓库类，或者创建一个新的服务类，该类部分在本地实现服务，部分远程实现，或者我们可以引入一个使用 gRPC 的新控制器。所有这些更改都将遵循开闭原则，因为我们没有修改任何现有类（当然 DiContainer 除外），而是通过引入实现现有接口的新类来扩展我们的应用程序。

在 app.py 文件中，我们创建一个 DI 容器实例，创建 FastAPI 应用，定义一个错误处理器将业务错误映射到 HTTP 响应，最后将所需的控制器连接到 FastAPI 应用：

### 图 4.12. app.py

```python
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .DiContainer import DiContainer
from .errors.OrderServiceError import OrderServiceError
from .utils import get_stack_trace

di_container = DiContainer()
app = FastAPI()

@app.exception_handler(OrderServiceError)
def handle_order_service_error(request: Request, error: OrderServiceError):
    # 记录 error.cause
    # 将 'request_failures' 计数器加一
    # 标签：
    # api_endpoint=f'{request.method} {request.url}'
    # status_code=error.status_code

    return JSONResponse(
        status_code=error.status_code,
        content={
            'errorMessage': error.message,
            'stackTrace': get_stack_trace(error.cause),
        },
    )

@app.exception_handler(RequestValidationError)
def handle_request_validation_error(
    request: Request, error: RequestValidationError
):
    # 审计日志

    # 将 'request_failures' 计数器加一
    # 标签：
    # api_endpoint=f'{request.method} {request.url}'
    # status_code=400

    return JSONResponse(
        status_code=400,
        content={'errorMessage': str(error)},
    )

@app.exception_handler(Exception)
def handle_unspecified_error(request: Request, error: Exception):
    # 将 'request_failures' 计数器加一
    # 标签：
    # api_endpoint=f'{request.method} {request.url}'
    # status_code=500
    # error_code='UnspecifiedError'

    return JSONResponse(
        status_code=500,
        content={
            'errorMessage': str(error),
            'stackTrace': get_stack_trace(error),
        },
    )

order_controller = di_container.order_controller()
app.include_router(order_controller.router)
```

> 如果你对其余代码（即 DTO、实体、GraphQL schema（=类型）和错误）感兴趣，请查看附录 A。

依赖注入容器是微服务中唯一包含对具体实现引用的地方。*依赖注入原则* 将在本章后面的章节中更详细地讨论。依赖倒置原则和依赖注入原则通常相辅相成。依赖注入用于连接接口依赖，使其成为对具体实现的依赖，如下图所示。

![依赖注入](img/cbd069395d7b824346b69b1f92e0fb4a_126_0.png)

让我们添加一个功能，当创建订单时清空购物车：

### 图 4.14. services/ShoppingCartOrderService.py

```python
from dependency_injector.wiring import Provide

from ..dtos.InputOrder import InputOrder
from ..dtos.OutputOrder import OutputOrder
from ..repositories.OrderRepository import OrderRepository
from ..services.OrderService import OrderService
from ..services.ShoppingCartService import ShoppingCartService

class OrderServiceImpl(OrderService):
    __order_repository: OrderRepository = Provide['order_repository']
    __shopping_cart_service: ShoppingCartService = Provide[
        'shopping_cart_service'
    ]

    def create(self, order: InputOrder) -> OutputOrder:
        self.__shopping_cart_service.empty_cart(order.user_id)
        return self.__order_repository.save(order)

    # 其余方法...
```

从上面的代码可以看出，`ShoppingCartOrderService` 类不依赖于购物车服务的任何具体实现。我们可以创建一个 *接口适配器* 类，它是 `ShoppingCartService` 接口的具体实现。该接口适配器类连接到特定的外部购物车服务，例如通过 REST API。同样，依赖注入器会将一个具体的 `ShoppingCartService` 实现注入到 `ShoppingCartOrderService` 类的实例中。

请注意，上述 `create_order` 方法不是生产级质量的，因为它缺少事务处理。

现在我们已经看到了清晰微服务设计的以下好处示例：

- 不绑定于任何单一的 API 技术，如 REST 或 GraphQL
- 不绑定于特定客户端（适用于 Web、桌面、控制台和移动客户端）
- 不绑定于特定数据库
- 不依赖于任何特定的外部服务实现

让我们展示最后一个好处：

- 不绑定于任何单一框架

让我们将使用的 Web 框架从 FastAPI 更改为 Flask。我们需要做的是创建一个 Flask 特定版本的 `RestOrderController`：

### 图 4.15. controllers/FlaskRestOrderController.py

```python
from dependency_injector.wiring import Provide
from flask import Response, jsonify, request
from flask_classful import FlaskView, route

from ..dtos.InputOrder import InputOrder
from ..services.OrderService import OrderService

# 在下面类的请求处理函数中
# 记得添加授权、必要的审计日志和
# 可观测性（指标更新）以用于生产环境。
# 本书后续章节将提供示例

class FlaskRestOrderController(FlaskView):
    __order_service: OrderService = Provide['order_service']

    @route('/orders', methods=['POST'])
    def create_order(self) -> Response:
        output_order = self.__order_service.create_order(
            InputOrder(**request.json)
        )
```

## 4.6：统一命名原则

*使用统一的方式命名接口、类和函数。*

本节介绍统一命名接口、类和函数的约定。

### 4.6.1：命名接口和类

> *类代表一个事物或一个行为者。它们应被一致地命名，使得类名以名词结尾。接口代表一个抽象的事物、行为者或能力。代表事物或行为者的接口应像类一样命名，但使用抽象名词。代表能力的接口应根据该能力来命名。*

当接口代表一个抽象事物时，根据该抽象事物命名。例如，如果你有一个包含各种几何对象的绘图应用程序，将几何对象接口命名为 `Shape`。这是一个简单的抽象名词。名称应始终是最短、最具描述性的。如果我们可以简单地使用 `Shape`，就没有理由将几何对象接口命名为 `GeometricalObject` 或 `GeometricalShape`。

当接口代表一个抽象行为者时，根据该抽象行为者命名。接口的名称应源自它提供的功能。例如，如果接口中有一个 `parseConfig` 方法，则该接口应命名为 `ConfigParser`；如果一个接口有一个 `validateObject` 方法，则该接口应命名为 `ObjectValidator`。不要使用不匹配的名称组合，例如一个名为 `ConfigReader` 的接口却有一个 `parseConfig` 方法，或者一个名为 `ObjectValidator` 的接口却有一个 `validateData` 方法。

当接口代表一种能力时，根据该能力命名。能力是具体类能够做的事情。例如，一个类可以是可排序的、可迭代的、可比较的、可相等的等。根据能力命名相应的接口：`Sortable`、`Iterable`、`Comparable` 和 `Equitable`。代表能力的接口名称通常以 *able* 或 *ing* 结尾。

不要以 `I` 前缀（或任何其他前缀或后缀）命名接口。相反，在需要区分接口和类时，使用 `Impl` 后缀作为类名。你应该面向接口编程，如果每个接口的名称都以 `I` 为前缀，只会给代码增加不必要的噪音。

一些代表事物的类名示例有：`Account`、`Order`、`RectangleShape` 和 `CircleShape`。在类继承层次结构中，类名通常会细化接口名或基类名。例如，如果有一个 `InputMessage` 接口，那么可以有 `InputMessage` 接口的不同具体实现（= 类）。它们可以代表来自不同来源的输入消息，如 `KafkaInputMessage` 和 `HttpInputMessage`。并且可以有不同的子类用于不同的数据格式：`AvroBinaryKafkaInputMessage` 或 `JsonHttpInputMessage`。

接口或基类名应保留在类或子类名中。类名应遵循以下模式：`<类用途>` + `<接口名>` 或 `<子类用途>` + `<超类名>`，例如，`Kafka` + `InputMessage` = `KafkaInputMessage`，`AvroBinary` + `KafkaInputMessage` = `AvroBinaryKafkaInputMessage`。使用 `Abstract` 前缀命名抽象类。

如果接口或类名长度达到或超过 20 个字符，请考虑缩写类名中的一个或多个单词。这样做的原因是为了保持代码的可读性。非常长的单词更难阅读，会拖慢开发者的速度。（记住，代码被阅读的次数远多于被编写的次数）。但只使用其他开发者常用且能理解的缩写。如果一个单词没有好的缩写，就不要缩写。例如，在类名 `AvroBinaryKafkaInputMessage` 中，我们只能将 `Message` 缩写为 `Msg`。对于类名中的其他单词，没有现成的缩写可用。将 `Binary` 缩写为 `Bin` 是值得商榷的，因为 `Bin` 也可能意味着一个 `bin`（容器）。如果缩写一个单词只能节省一两个字符，就不要缩写。例如，没有理由将 `Account` 缩写为 `Acct`。

你可以通过省略一个或多个单词来缩短名称，前提是名称仍然易于任何开发者理解。例如，如果你有类 `InternalMessage`、`InternalMessageSchema` 和 `InternalMessageField`，你可以将后两个类名缩短为：`InternalSchema` 和 `InternalField`。这是因为这两个类主要与 `InternalMessage` 类一起使用：一个 `InternalMessage` 对象有一个 schema 和一个或多个字段。你也可以使用嵌套类：`InternalMessage.Schema` 和 `InternalMessage.Field`。

如果你有相关的类，并且一个或多个类名需要缩短，你应该缩短所有相关类名以保持命名统一。例如，如果你有两个类 `ConfigurationParser` 和 `JsonConfigurationParser`，你应该缩短这两个类的名称，而不仅仅是那个超过 19 个字符的。新的类名将是 `ConfigParser` 和 `JsonConfigParser`。

如果接口或类名长度少于 20 个字符，通常不应尝试使其更短。

如果设计模式名称不能带来任何实际好处，就不要将其添加到类名中。例如，假设我们有一个 `DataStore` 接口、一个 `DataStoreImpl` 类，以及一个包装 `DataStore` 实例并使用*代理模式*为被包装的数据存储添加缓存功能的类。我们不应该将缓存类命名为 `CachingProxyDataStore` 或 `CachingDataStoreProxy`。*proxy*（代理）这个词没有增加显著价值，所以该类应简单地命名为 `CachingDataStore`。这个名称清楚地表明这是一个具有缓存功能的数据存储。经验丰富的开发者会从 `CachingDataStore` 这个名称中注意到该类使用了*代理模式*。如果没有，查看类的实现最终也会揭示这一点。

### 4.6.2：命名函数

> *函数应该只做一件事，函数名应该描述函数的功能。函数名通常包含一个表示函数功能的动词。在许多情况下，函数名以动词开头，但也有例外。如果函数返回一个值，尝试命名函数，使函数名描述其返回的内容。*

一般规则是命名函数时，应使函数的目的清晰。一个好的函数名不应让你思考。如果函数名*达到或超过 20 个字符长*，请考虑缩写名称中的一个或多个单词。这样做的原因是为了保持代码的可读性。非常长的单词更难阅读，会拖慢开发者的速度。（记住，代码被阅读的次数远多于被编写的次数）。但只使用其他开发者广泛使用且能理解的缩写。如果一个单词没有好的缩写，就不要缩写。

下面是一个协议的示例，其中包含两个仅用简单动词命名的方法。没有必要将方法命名为 `start_thread` 和 `stop_thread`，因为这些方法已经是 `Thread` 接口的一部分，`start` 方法启动什么以及 `stop` 方法停止什么是不言而喻的。

```python
from typing import Protocol

class Thread(Protocol):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
```

许多语言提供可以写入的流，例如标准输出流。流通常是缓冲的，实际写入流的操作不会立即发生。例如，下面的语句不一定立即写入标准输出流。它会缓冲要写入的文本，稍后在缓冲区刷新到流时再写入。这可能发生在缓冲区已满、自上次刷新以来经过了一段时间或流被关闭时。

```python
stream.write(...)
```

上述语句具有误导性，可以通过重命名函数来描述其实际功能进行修正：

```python
stream.buffer_text(...)
```

我们通过引入两个新的小模块，成功更改了所使用的 Web 框架。我们没有触及任何现有模块，因此可以确定我们没有破坏任何现有功能。我们再次成功地将*开闭原则*应用于我们的代码库。

## 4.6.2.1：函数名中的介词

> 当需要澄清函数目的时，在函数名中使用介词。

如果介词可以被推断（即介词是隐含的），则无需在函数名中添加介词。在许多情况下，只能推断出一个介词。如果你有一个名为 `wait` 的函数，介词 `for` 可以被推断；如果你有一个名为 `subscribe` 的函数，介词 `to` 可以被推断。我们不需要使用函数名 `wait_for` 和 `subscribe_to`。

假设一个函数名为 `laugh(person: Person)`。现在我们必须添加一个介词，因为无法推断出任何介词。我们应该将函数命名为 `laugh_with(person: Person)` 或 `laugh_at(person: Person)`。

让我们分析 Python 的列表方法，看看它们的命名有多好：

**`list.append(item)`**
这清楚地说明了值被放入列表中的位置。读起来很好。我们可以很容易地在 `insert` 一词后推断出 `at` 介词。

**`list.remove(item)`**
此方法仅移除列表中的第一个项目。因此，它应该被命名为 `remove_first(item)`。此方法也可能引发异常。我们应该在方法签名中传达这一点。让我们使用 `try` 前缀：`list.try_remove_first(item)`。我们将在下一章更详细地讨论异常处理和 `try` 前缀。

**`list.pop(index)`**
读起来很好。我们可以很容易地在 `pop` 一词后推断出 `at` 介词。

**`list.index(item)`**
这里的 `index` 一词不是动词。我们添加正确的动词，并告知用户返回的是找到项目的第一个索引。此方法也可能引发异常。我们应该在方法签名中传达这一点。此方法应重命名为 `list.try_find_first_index_of(item)`。

**`list.count(item)`**
读起来很好。我们可以很容易地在 `count` 一词后推断出 `of` 介词。

**`list.sort()`**
这很完美。它表明列表是原地排序的。如果方法返回一个新的排序列表，它应该被称为 `list.sorted()`。

**`list.reverse()`**
这很完美。它表明列表是原地反转的。如果方法返回一个新的反转列表，它应该被称为 `list.reversed()`。

**`list.copy()`**
`copy` 一词强烈地与从一个地方复制到另一个地方相关联。我会将此方法重命名为 `list.clone()`。

## 4.6.2.2：命名方法对

类中的方法可以成对出现。一个典型的例子是 getter 和 setter 方法对。当你在类中定义一个方法对时，要合乎逻辑地命名这些方法。方法对中的方法通常做两件相反的事情，比如获取或设置值。如果你不确定如何命名其中一个方法，试着为一个词找一个反义词。例如，如果你有一个以“create”开头的方法，并且不确定如何命名相反操作的方法，可以尝试谷歌搜索：“create antonym”。

以下是一些成对出现的方法名称的非详尽列表：

- get/put（例如，当访问非序列集合如 set 或 map 时）
- read/write
- add/remove
- store/retrieve
- open/close
- load/save
- initialize/destroy
- create/destroy
- insert/delete
- start/stop
- pause/resume
- start/finish
- increase/decrease
- increment/decrement
- construct/destruct
- encrypt/decrypt
- encode/decode
- obtain/relinquish
- acquire/release
- reserve/release
- startup/shutdown
- login/logout
- begin/end
- launch/terminate
- publish/subscribe
- join/detach
- <something>/un<something>，例如 assign/unassign、install/uninstall、subscribe/unsubscribe、follow/unfollow
- <something>/de<something>，例如 serialize/deserialize、allocate/deallocate
- <something>/dis<something>，例如 connect/disconnect

让我们看几个现实生活中的例子。基于 Debian/Ubuntu 的 Linux 中的 `apt` 工具有一个 `install` 命令来安装软件包，但卸载软件包的命令是 `remove`。它应该是 `uninstall`。Kubernetes 包管理器 Helm 在这方面是正确的。它有一个 `install` 命令来安装 Helm release，以及一个 `uninstall` 命令来卸载它。

## 4.6.2.3：命名布尔函数（谓词）

> *布尔函数（谓词）的命名应使得在读取函数调用语句时，它读起来像一个可以为真或为假的布尔语句。*

在本节中，我们考虑命名作为谓词并返回布尔值的函数。这里我指的不是根据执行操作的成功与否返回 true 或 false 的函数，而是函数调用用于评估语句为真或假的情况。布尔函数的命名应使得在读取函数调用语句时，它构成一个可以为真或为假的语句。以下是一些示例：

```python
class Response:
    def has_error(self) -> bool:
        # ...
```

```python
class String:
    def is_empty(self) -> bool:
        # ...

    def starts_with(self, another_string: str) -> bool:
        # ...

    def ends_with(self, another_string: str) -> bool:
        # ...

    def contains(self, another_string: str) -> bool:
        # ...
```

```python
# 这里我们有一个语句：response 有错误吗？真还是假？
if response.has_error():
    # ...

# 这里我们有一个语句：line 是空的吗？真还是假？
line: String = file_reader.read_line()
if line.is_empty():
    # ...

# 这里我们有一个语句：line 以空格字符开头吗？
# 真还是假？
if line.starts_with(' '):
    # ...

# 这里我们有一个语句：line 以分号结尾吗？
# 真还是假？
if line.ends_with(';'):
    # ...
```

```python
class Thread:
    def should_terminate(self) -> bool:
        # ...

    def is_paused(self) -> bool:
        # ...

    def can_resume_execution() -> bool:
        # ...

    def run(self) -> None:
        # ...

        # 这里我们有一个语句：self 应该终止吗？
        # 真还是假？
        if self.should_terminate():
            return

        # 这里我们有一个语句：self 已暂停，并且
```

## 4.6.2.3：为返回布尔值的函数命名

当一个返回布尔值的函数在代码中被调用时，如果其调用语句能用清晰的英语读出来，那么这个函数的命名就是正确的。以下是错误和正确命名的示例：

```python
class Thread:
    # 错误的命名
    def stopped(self) -> bool:
        # ...

    # 正确的命名
    def is_stopped(self) -> bool:
        # ...
```

```python
if thread.stopped():
    # 这里我们得到的是：if thread stopped
    # 这不是一个有真或假答案的陈述。
    # 它是第二种条件形式，
    # 询问如果线程停止会发生什么。

    # ...
```

```python
# 这里我们得到的是陈述：if thread is stopped
# 真还是假？
if thread.is_stopped():
    # ...
```

从上面的例子中，我们可以注意到许多返回布尔值的函数名以 *is* 或 *has* 开头，并遵循以下模式：

- is + <形容词>，例如 is_open、is_running 或 is_paused
- has + <名词>

此外，以下两种形式也相对常见：

- should + <动词>
- can + <动词>

但正如我们在 *starts_with*、*ends_with* 和 *contains* 函数中看到的，返回布尔值的函数名也可以以任何第三人称单数形式（即以 s 结尾）的动词开头。如果你有一个集合类，其布尔方法名应使用复数形式的动词，例如：`numbers.include(...)` 而不是 `numbers.includes(...)`。请始终将你的集合变量命名为复数形式（例如，`numbers` 而不是 `number_list`）。我们将在下一章讨论变量的统一命名原则。

不要在函数名中包含 *does* 这个词，比如 *does_start_with*、*does_end_with* 或 *does_contain*。添加 *does* 这个词并不会给名称增加任何实际价值，并且当在代码中使用时，这样的函数名读起来很别扭，例如：

```python
line = text_file_reader.read_line()

# "If line does start with" 听起来很别扭
if line.does_start_with(' '):
    # ...
```

当你想在函数名中使用过去时态时，请在函数名中使用 *did* 前缀，例如：

```python
class DatabaseOperation:
    def execute(self) -> None:
        # ...

    # 方法名不合适。这是第二种条件形式
    # if db_operation.started_transaction(): ...
    def started_transaction(self) -> bool:
        # ...

    # 方法名合适，不会产生混淆
    def did_start_transaction(self):
        # ...
```

## 4.6.2.4：为构建器方法命名

构建器类用于创建构建器对象，这些对象用于构建特定类型的新对象。如果你想构造一个 URL，可以使用 *UrlBuilder* 类来实现这个目的。构建器类的方法为被构建的对象添加属性。因此，建议将构建器类的方法命名为以动词 *add* 开头。最终构建所需对象的方法应简单地命名为 *build* 或 *build* + *<构建目标>*，例如 *build_url*。我更喜欢较长的形式，以提醒读者正在构建什么。以下是在构建器类中为方法命名的示例：

```python
class UrlBuilder:
    def add_scheme(self, scheme: str) -> 'UrlBuilder':
        # ...
        return self

    def add_host(self, host: str) -> 'UrlBuilder':
        # ...
        return self

    def add_port(self, port: int) -> 'UrlBuilder':
        # ...
        return self

    def add_path(self, path: str) -> 'UrlBuilder':
        # ...
        return self

    def add_query(self, query: str) -> 'UrlBuilder':
        # ...
        return self
```

```python
def build_url(self) -> Url:
    # ...

url = (
    UrlBuilder().add_scheme('https://').add_host('google.com').build_url()
)
```

## 4.6.2.5：为隐含动词的方法命名

工厂方法名通常以动词 *create* 开头。工厂方法可以被命名，使得 *create* 动词是隐含的，例如：

```python
Optional.of(value)
Optional.empty() # 不是最优的，'empty' 可能被误认为是动词
Either.with_left(value)
Either.with_right(value)
SalesItem.from_dto(input_sales_item)
```

上述方法名的显式版本将是：

```python
Optional.create(value)
Optional.create_empty()
Either.create_with_left(value)
Either.create_with_right(value)
SalesItem.create_from_dto(input_sales_item)
```

类似地，转换方法可以被命名，使得 *convert* 动词是隐含的。没有动词的转换方法通常以介词 *to* 开头，例如：

```python
numeric_value.to_string()
dict_value.to_json()
```

上述方法的显式命名版本将是：

```python
numeric_value.convert_to_string()
dict_value.convert_to_json()
```

我建议谨慎使用带有隐含动词的方法名，仅在隐含动词不言自明且不会迫使开发者思考的情况下使用。

## 4.6.2.6：为生命周期方法命名

生命周期方法仅在特定场合被调用。生命周期方法名应回答这个问题：这个方法将在何时或“在什么场合”被调用？生命周期方法的好名字示例有：`on_init`、`on_error`、`on_success`、`after_mount`、`before_unmount`。例如，在 React 中，类组件中有名为 `componentDidMount`、`componentDidUpdate` 和 `componentWillUnmount` 的生命周期方法。没有理由在生命周期方法名中重复类名。更好的名字应该是：`afterMount`、`afterUpdate` 和 `beforeUnmount`。

## 4.6.2.7：为函数参数命名

函数参数的命名规则与变量的命名规则基本相同。变量的*统一命名原则*将在下一章更详细地描述。

有一些例外情况，比如为对象参数命名。当函数参数是一个对象时，如果参数名和函数名隐含地描述了参数的类，那么可以从参数名中省略对象类名。这个例外是可接受的，因为函数参数类型总是可以通过查看函数签名轻松检查。并且这应该很容易做到，因为函数应该很短（最多 5-7 条语句）。以下是如何为对象类型参数命名的示例：

```python
# 单词 'Location' 重复了，不是最优的，但允许
def drive(start_location: Location, destination_location: Location) -> None:
    # ...

# 更好的方式
# 当我们想到 'drive' 和 'start' 或 'destination' 时，
# 我们可以假设 'start' 和 'destination' 意味着位置
def drive(start: Location, destination: Location) -> None:
    # ...
```

一些编程语言，如 Swift，允许为函数参数添加所谓的*外部名称*。使用外部名称可以使函数调用语句读起来更顺畅，如下所示：

```swift
func drive(from start: Location, to destination: Location) {
    // ...
}

func send(
    message: String,
    from sender: Person,
    to recipient: Person
) {
    // ...
}

let startLocation = new Location(...);
let destLocation = new Location(...);
drive(from: startLocation, to: destLocation);

let message = "Some message";
let person = new Person(...);
let anotherPerson = new Person(...);
send(message, from: person, to: anotherPerson);
```

## 4.7：封装原则

> *一个类应该封装其状态，使得对状态的访问只能通过公共方法进行。*

封装是通过将类属性声明为私有来实现的。Python 没有属性访问修饰符。你应该使用命名约定。使用以 `__` 为前缀的属性名来表示私有属性。如果你需要状态在类外部可修改，可以创建 getter 和 setter 方法（或属性）。然而，如果你不需要为类属性创建 getter 和 setter 方法，那么封装就能得到最好的保证。不要自动为每个类实现 getter 和 setter 方法。只在需要时才创建这些访问器方法，例如当类表示一个可修改的数据结构时。并且只为需要在类外部修改的属性生成 setter 方法。

## 4.7.1：不可变对象

确保对象状态封装的最佳方式是使对象不可变。这意味着一旦对象被创建，其状态就不能被修改。不可变性确保你不能意外或故意地修改对象的状态。在对象外部修改对象的状态可能是错误的来源。

在创建不可变对象时，你在构造函数中提供对象所需的参数，之后这些属性就不能被修改了。如果你需要修改一个不可变对象，唯一的方法是创建一个新的对象，并在构造函数中提供不同的值。这种方法的缺点是，与仅修改现有对象的属性相比，创建新对象会引入性能损失。但在许多情况下，与不可变性的好处相比，这种损失可以忽略不计。例如，字符串在 Python 中是不可变的。一旦你创建了一个字符串，你就不能修改它。你只能创建新的字符串。

不可变性还要求 getter 和其他返回值的方法不能返回可修改的属性，比如列表。如果你从一个方法返回一个列表，那么这个列表可能会通过添加或删除元素而被修改，而“拥有”该列表的对象却对此一无所知。

## 4.7.2：不要将可修改的内部状态泄露到对象外部原则

在从方法返回值时要小心。方法可能会意外地返回对象的某些内部状态，而这些状态随后可能被方法调用者修改。从方法返回可修改的状态会破坏封装。

你可以安全地从方法返回原始类型或所谓的值类型。这些类型包括 bool、int 和 float 等类型。你也可以安全地返回不可变对象，比如字符串。但是，你不能安全地返回可修改的集合。

有两种方法可以防止内部状态泄露到对象外部：

1. 返回可修改内部状态的副本
2. 返回可修改内部状态的不可修改版本

关于第一种方法，当返回一个副本时，调用者可以随意使用它。对复制对象所做的更改不会影响原始对象。我主要讨论的是浅拷贝。在许多情况下，浅拷贝就足够了。例如，一个包含基本值、不可变字符串或不可变对象的列表不需要进行深拷贝。但你应该在需要时进行深拷贝。

复制方法可能会导致性能损失，但在许多情况下，这种损失微不足道。你可以轻松地创建一个列表的副本：

```python
values = [1, 2, 3, 4, 5]
values2 = values.copy()
```

第二种方法要求你创建一个可修改对象的不可修改版本，并返回该不可修改对象。你可以自己创建一个类的不可修改版本。下面是一个例子：

```python
from typing import Protocol, TypeVar

T = TypeVar('T')

class MyList(Protocol[T]):
    def append(self, item: T) -> None:
        pass

    def get_item(self, index: int) -> T | None:
        pass

class UnmodifiableMyList(MyList[T]):
    def __init__(self, list: MyList[T]):
        self._list = list

    def append(self, item: T) -> None:
        raise NotImplementedError()

    def get_item(self, index: int) -> T | None:
        return self._list.get_item(index)
```

在上面的例子中，不可修改的列表类将另一个列表（一个可修改的列表）作为构造函数参数。它只实现了不尝试修改被包装列表的 MyList 协议方法。在这种情况下，它只实现了 `get_item` 方法，该方法委托给 MyList 类中的相应方法。UnmodifiableMyList 类中尝试修改被包装列表的方法应该引发错误。UnmodifiableMyList 类通过包装 MyList 类的对象并部分允许访问 MyList 类的方法，利用了代理模式。

不可修改对象和不可变对象略有不同。没有人可以修改不可变对象，但当你从一个方法返回一个不可修改对象时，该对象仍然可以被拥有类修改，并且修改对所有收到该对象不可修改版本的人都是可见的。如果这是不可取的，你应该使用副本。

## 4.7.3：不要将方法参数赋值给可修改属性

如果一个类接收可修改对象作为构造函数或方法参数，通常最佳实践是不要将这些参数直接赋值给内部状态。如果直接赋值，该类可能会有意或无意地修改这些参数对象，这可能不是构造函数或方法调用者所期望的。

有两种方法可以处理这种情况：

- 1) 将可修改参数对象的副本存储到类的内部状态
- 2) 将可修改参数对象的不可修改版本存储到类的内部状态

下面是第二种方法的示例：

```python
class MyClass:
    def __init__(self, values: MyList[int]):
        self.__values = UnmodifiableMyList(values)
```

## 4.7.4：封装违反的真实示例：React 类组件的状态

React 类组件的状态没有得到适当的封装。React 文档指示应在 `Component` 子类构造函数中使用 `this.state` 直接修改 `state` 属性。例如：

```javascript
import { Component } from 'react';

class ButtonClickCounter extends Component {
    constructor(props) {
        super(props);

        this.state = {
            clickCount: 0
        };
    }
}
```

`state` 属性在 `Component` 类中是公共或受保护的，这不是良好的面向对象设计。你不应该在 `ButtonClickCounter` 子类中修改基类的 `state` 属性。以面向对象方式初始化状态的正确方法是使用 `super` 将初始状态作为参数传递给 `Component` 类构造函数。然而，React 不支持以下方式：

```javascript
import { Component } from 'react';

export default class ButtonClickCounter extends Component {
  constructor(props) {
    // 这在实际中是不可能的
    super(props, {
      clickCount: 0
    });
  }
}
```

设置状态是通过 Component 类中定义的 setState 方法完成的，但访问状态是直接通过 state 属性进行的。这导致了一个问题：根据 React 文档，在调用 setState 方法时不能使用 this.state，因为这可能导致错误的行为。因此，以下代码是不允许的：

```javascript
incrementClickCount = () =>
  this.setState({
    clickCount: this.state.clickCount + 1
  });
```

下面是在 React 类组件中正确使用 setState 方法的示例：

```javascript
import { Component } from 'react';

export default class ButtonClickCounter extends Component {
  constructor(props) {
    super(props);

    this.state = {
      clickCount: 0
    };
  }

  incrementClickCount = () =>
    this.setState(({ clickCount }) => ({
      clickCount: clickCount + 1
    }));

  render() {
    return (
      <>
        Click count: {this.state.clickCount}
        <button onClick={this.incrementClickCount} />
      </>
    );
  }
}
```

在 Component 子类中访问状态应该使用 getter getState，而不是直接访问 state 属性。下面是修改后的示例，使用了假想的 getState 方法：

```javascript
import { Component, Fragment } from 'react';

export default class ButtonClickCounter extends Component {
  constructor(props) {
    super(props, {
      clickCount: 0
    });
  }

  incrementClickCount = () =>
    this.setState({
      clickCount: this.getState().clickCount + 1
    });

  render() {
    return (
      <>
        Click count: {this.getState().clickCount}
        <button onClick={this.incrementClickCount} />
      </>
    );
  }
}
```

## 4.8：对象组合原则

> 在面向对象设计中，就像在现实生活中一样，对象是通过从较小的对象构建较大的对象来构造的。这称为对象组合。优先使用对象组合而非继承。

例如，一个汽车对象可以由发动机和变速箱对象（仅举几例）组成。对象很少通过从另一个对象派生（即使用继承）来“组合”。但首先，让我们尝试使用继承来指定实现以下 Car 协议的类：

```python
from typing import Protocol

class Car(Protocol):
    def drive(self, start: Location, destination: Location) -> None:
        pass

class CombustionEngineCar(Car):
    def drive(self, start: Location, destination: Location) -> None:
        # ...

class ElectricEngineCar(Car):
    def drive(self, start: Location, destination: Location) -> None:
        # ...
```

```python
class ManualTransmissionCombustionEngineCar(CombustionEngineCar):
    def drive(self, start: Location, destination: Location) -> None:
        # ...
```

```python
class AutomaticTransmissionCombustionEngineCar(CombustionEngineCar):
    def drive(self, start: Location, destination: Location) -> None:
        # ...
```

如果我们想为汽车添加其他组件，比如两轮或四轮驱动，所需的类数量将增加三个。如果我们想为汽车添加设计属性（轿车、掀背车、旅行车或 SUV），所需的类数量将激增，类名也会变得荒谬地长。我们可以注意到，继承不是构建更复杂类的正确方式。

类继承在超类及其子类之间创建了 *is-a* 关系。对象组合创建了 *has-a* 关系。我们可以声称 `ManualTransmissionCombustionEngineCar` *是* `CombustionEngineCar` 的一种，所以基本上，有人可能认为我们在这里没有做错任何事。但在设计类时，你应该首先确定是否可以使用对象组合：是否存在 *has-a* 关系？你能否将一个类声明为另一个类的属性？如果答案是肯定的，那么应该使用组合而不是继承。

所有与汽车相关的东西实际上都是汽车的属性。汽车 *有一个* 发动机。汽车 *有一个* 变速箱。它 *有一个* 两轮或四轮驱动和设计。我们可以将基于继承的解决方案转变为基于组合的解决方案：

```python
from typing import Protocol

class Drivable(Protocol):
    def drive(self, start: Location, destination: Location) -> None:
        pass

class Engine(Protocol):
    # 像启动、停止这样的方法 ...

class CombustionEngine(Engine):
    # 像启动、停止这样的方法 ...

class ElectricEngine(Engine):
    # 像启动、停止这样的方法 ...

class Transmission(Protocol):
    # 像换挡这样的方法 ...

class AutomaticTransmission(Transmission):
    # 像换挡这样的方法 ...

class ManualTransmission(Transmission):
    # 像换挡这样的方法 ...
```

## 像 `shift_gear` 这样的方法 ...

# 在此定义 DriveType ...
# 在此定义 Design ...

```python
class Car(Drivable):
    def __init__(
        self,
        engine: Engine,
        transmission: Transmission,
        driveType: DriveType,
        design: Design
    ):
        self.__engine = engine
        self.__transmission = transmission
        self.__driveType = driveType
        self.__design = design

    def drive(self, start: Location, destination: Location) -> None:
        # 要实现功能，请委托给
        # 组件类，例如：

        # self.__engine.start()
        # self.__transmission.shift_gear(...)
        # ...
        # self.__engine.stop()
```

让我们来看一个使用不同图表类型的更现实的例子。起初，这听起来像是一个可以使用继承的场景：我们有一些抽象的基图表，不同的具体图表对其进行扩展，例如：

```python
from abc import abstractmethod
from typing import Protocol
```

```python
class Chart(Protocol):
    def render_view(self) -> None:
        pass

    def update_data(self, ...) -> None:
        pass
```

```python
class AbstractChart(Chart):
    @abstractmethod
    def render_view(self) -> None:
        pass

    @abstractmethod
    def update_data(self, ...) -> None:
        pass

    # 实现所有图表类型共享的一些通用功能
```

```python
class XAxisChart(AbstractChart):
    @abstractmethod
    def render_view(self) -> None:
        pass

    def update_data(self, ...) -> None:
        # 这对所有 x 轴图表都是通用的，
        # 比如 ColumnChart、LineChart 和 AreaChart
```

```python
class ColumnChart(XAxisChart):
    def render_view(self) -> None:
        # 使用 XYZ 库渲染柱状图
```

```python
# LineChart 类定义在此...
# AreaChart 类定义在此...
```

```python
class NonAxisChart(AbstractChart):
    @abstractmethod
    def render_view(self) -> None:
        pass

    def update_data(self, ...) -> None:
        # 这对所有非 x 轴图表都是通用的，
        # 比如 PieChart 和 DonutChart
```

```python
class PieChart(NonAxisChart):
    def render_view(self) -> None:
        # 使用 XYZ 库渲染饼图
```

```python
class DonutChart(PieChart):
    def render_view(self) -> None:
        # 使用 XYZ 库渲染环形图
```

上述类层次结构看起来是可控的：应该不需要定义太多的子类。当然，我们可以想到新的图表类型，比如地理地图或数据表格，我们可以为它们添加子类。深层类层次结构的一个问题出现在你需要更改或修正与特定图表类型相关的内容时。假设你想更改或修正与饼图相关的一些行为。你首先会检查 `PieChart` 类，看行为是否在那里定义。如果你找不到你需要的东西，你需要导航到 `PieChart` 类的基类（`NonAxisChart`）并在那里查找。你可能需要继续这种导航，直到找到你想要更改或修正的行为所在的基类。当然，如果你对代码库极其熟悉，你可能第一次尝试就能找到正确的子类。但总的来说，这不是一项简单的任务。

使用类继承可能会引入一些类的方法数量显著多于其他类的类层次结构。例如，在图表继承链中，`AbstractChart` 类可能比继承链末端的类拥有显著更多的方法。这种类大小的差异在类之间造成了不平衡，使得难以推理每个类提供了什么功能。

即使上述类层次结构乍一看可能没问题，目前存在一个问题。我们已经硬编码了正在渲染的图表视图类型。我们使用 XYZ 图表库并渲染 XYZChart 视图。假设我们想引入另一个名为 ABC 的图表库。我们希望同时使用这两个图表库，这样我们数据可视化应用程序的开源版本使用开源的 XYZ 图表库，而付费版本使用商业的 ABC 图表库。使用类继承时，我们必须为 ABC 图表库的每个具体图表类型创建新类。因此，每个具体图表类型将有两个类，就像这里的饼图：

```python
class XyzPieChart(XyzNonAxisChart):
    def render_view(self) -> None:
        # 使用 XYZ 库渲染饼图

class AbcPieChart(AbcNonAxisChart):
    def render_view(self) -> None:
        # 使用 ABC 库渲染饼图
```

使用组合而不是继承来实现上述功能有几个好处：

- 每个类包含什么行为更加明显
- 类之间没有显著的大小不平衡，不会出现一些类巨大而其他类相对较小的情况
- 你可以根据需要将图表行为拆分成类，并且符合单一职责原则

在下面的例子中，我们将一些图表行为拆分成两种类型的类：图表视图渲染器和图表数据工厂：

```python
from enum import Enum
from typing import Protocol

class Chart(Protocol):
    def render_view(self) -> None:
        pass

    def update_data(self, ...) -> None:
        pass

# 定义 ChartData 类...
# 定义 ChartOptions 类...

class ChartViewRenderer(Protocol):
    def render_view(self, data: ChartData, options: ChartOptions) -> None:
        pass
```

```python
class ChartDataFactory(Protocol):
    def create_data(self, ...) -> ChartData:
        pass
```

```python
class ChartImpl(Chart):
    def __init__(
        self,
        view_renderer: ChartViewRenderer,
        data_factory: ChartDataFactory,
        options: ChartOptions
    ):
        self.__view_renderer = view_renderer
        self.__data_factory = data_factory
        self.__options = options
        self.__data = None,

    def render_view(self) -> None:
        self.__view_renderer.render_view(self.__data, self.__options)

    def update_data(self, ...) -> None:
        self.__data = self.__data_factory.create_data(...)
```

```python
class XyzPieChartViewRenderer(ChartViewRenderer):
    def render_view(self, data: ChartData, options: ChartOptions) -> None:
        # 使用 XYZ 库渲染饼图
```

```python
class AbcPieChartViewRenderer(ChartViewRenderer):
    def render_view(self, data: ChartData, options: ChartOptions) -> None:
        # 使用 ABC 库渲染饼图
```

```python
# 定义 AbcColumnChartViewRenderer 类...
# 定义 XyzColumnChartViewRenderer 类...
# 定义 XAxisChartDataFactory 类...
# 定义 NonAxisChartDataFactory 类...
```

```python
class ChartType(Enum):
    COLUMN = 1
    PIE = 2
```

```python
class ChartFactory(Protocol):
    def create_chart(self, chart_type: ChartType) -> Chart:
        pass
```

```python
class AbcChartFactory(ChartFactory):
    def create_chart(self, chart_type: ChartType) -> Chart:
        match chart_type:
            case ChartType.COLUMN:
                return ChartImpl(AbcColumnChartViewRenderer(),
                                 XAxisChartDataFactory())
            case ChartType.PIE:
                return ChartImpl(AbcPieChartViewRenderer(),
                                 NonAxisChartDataFactory())
            case _:
                raise ValueError('Invalid chart type')
```

```python
class XyzChartFactory(ChartFactory):
    def create_chart(self, chart_type: ChartType) -> Chart:
        match chart_type:
            case ChartType.COLUMN:
                return ChartImpl(XyzColumnChartViewRenderer(),
                                XAxisChartDataFactory())
            case ChartType.PIE:
                return ChartImpl(XyzPieChartViewRenderer(),
                                NonAxisChartDataFactory())
            case _:
                raise ValueError('Invalid chart type')
```

`XyzPieChartViewRenderer` 和 `AbcPieChartViewRenderer` 类使用了*适配器模式*，因为它们将提供的数据和选项转换为特定于实现（ABC 或 XYZ 图表库）的接口。

我们可以通过组合更多的类来轻松地为 `ChartImpl` 类添加更多功能。例如，可以有一个标题格式化器、工具提示格式化器类、y/x 轴标签格式化器和事件处理程序类。

```python
class ChartImpl(Chart):
    def __init__(
        self,
        view_renderer: ChartViewRenderer,
        data_factory: ChartDataFactory,
        title_formatter: ChartTitleFormatter,
        tooltip_formatter: ChartTooltipFormatter,
        x_axis_label_formatter: ChartXAxisLabelFormatter,
        event_handler: ChartEventHandler,
        options: ChartOptions
    ):
        # ...

    # 图表方法...
```

```python
class AbcChartFactory(ChartFactory):
    def create_chart(self, chart_type: ChartType) -> Chart:
        match chart_type:
            case ChartType.COLUMN:
                return ChartImpl(AbcColumnChartViewRenderer(),
                                XAxisChartDataFactory(),
                                ChartTitleFormatterImpl(),
                                XAxisChartTooltipFormatter(),
                                ChartXAxisLabelFormatterImpl(),
                                ColumnChartEventHandler())

            case ChartType.PIE:
                return ChartImpl(AbcColumnChartViewRenderer(),
                                NonAxisChartDataFactory(),
                                ChartTitleFormatterImpl(),
```

## 4.9：领域驱动设计原则

领域驱动设计（DDD）是一种软件设计方法，其核心思想是让软件模型与软件试图解决的问题领域的语言相匹配。DDD是分层的。顶层领域可以划分为子领域，子领域还可以进一步细分。

DDD意味着软件的结构以及代码中出现的名称（接口、类、函数和变量名）应当与领域相匹配。例如，在一个银行软件系统中，应使用诸如*账户*、*取款*、*存款*、*进行支付*和*贷款申请*这样的名称。软件系统的顶层领域应划分为更小的子领域。每个子领域应作为一个独立的应用程序或软件组件来实现。顶层领域包含软件系统的所有功能，而每个子领域是这些功能的一个子集。例如，一个开发团队可以专门负责贷款申请子领域，另一个团队负责支付子领域。团队中的开发者需要了解他们团队负责的子领域。而在与其他领域交互时，他们需要对其他领域有足够的了解以理解接口。这样，单个团队需要理解和记忆的概念集就会更小。产品经理和首席架构师应很好地掌握顶层领域，即他们应理解*全局图景*。

## 4.9.1：DDD概念

领域驱动设计包含多种概念：

- 实体
- 值对象
- 聚合
- 聚合根
- 工厂
- 仓储
- 服务
- 事件

### 4.9.1.1：实体

实体是具有标识的领域对象。通常，这通过实体类具有某种`id`属性来体现。实体的例子有*员工*和*银行账户*。员工对象有一个员工ID，银行账户有一个标识该账户的编号。实体可以包含操作其属性的方法。例如，银行账户实体可以有`withdraw`和`deposit`方法，用于操作实体的`balance`属性。

### 4.9.1.2：值对象

值对象是没有标识的领域对象。值对象的例子有地址或价格对象。价格对象可以有两个属性：`amount`和`currency`，但它没有标识。同样，地址对象可以有以下属性：`street address`、`postal code`、`city`和`country`。

### 4.9.1.3：聚合

聚合是由其他实体组成的实体。例如，一个*订单*实体可以包含一个或多个*订单项*实体。就面向对象设计而言，这等同于对象组合。

### 4.9.1.4：聚合根

聚合根是没有父对象的领域对象。如果一个*订单*实体没有父实体，它就是一个聚合根。但当一个*订单项*实体属于某个订单时，它就不是聚合根。聚合根充当门面对象，操作应在聚合根对象上执行，而不是直接访问门面后的对象（例如，不直接访问单个订单项，而是在订单对象上执行操作）。或者，如果你有一个包含车轮的聚合汽车对象，你不会在汽车对象外部操作车轮，而是汽车对象提供一个类似`turn`的门面方法，汽车对象内部操作车轮，使汽车对象成为一个聚合根。关于*门面设计模式*的更多内容将在本章后续部分介绍。

聚合根在微服务架构中也存在。假设我们有一个作为聚合根的银行账户，它包含交易实体。银行账户和交易实体可以在不同的微服务（`bank-account-service`和`account-transaction-service`）中处理，但只有`bank-account-service`可以直接访问和修改交易实体，它使用`account-transaction-service`。聚合根的作用和好处如下：

- 聚合根防止不变性被破坏，例如，其他服务不应直接使用`account-transaction-service`移除或添加交易。这会破坏交易总和应与`bank-account-service`维护的账户余额保持一致的不变性。
- 聚合根简化了（数据库/分布式）事务。你的微服务可以调用`bank-account-service`，让它管理`bank-account-service`和`account-transaction-service`之间的分布式事务，而无需你的微服务自己处理。

你可以轻松地将聚合根拆分为更多实体，例如，我们可以让银行账户聚合根包含一个余额实体和交易实体。余额实体可以由一个单独的`account-balance-service`处理。尽管如此，所有银行账户操作仍必须通过`bank-account-service`进行，它将协调例如使用`account-balance-service`和`account-transaction-service`的`withdraw`和`deposit`操作。我们甚至可以将`bank-account-service`拆分为两个独立的微服务：用于账户CRUD操作（不包括与余额相关的更新）的`bank-account-service`，以及使用两个更底层的微服务`account-balance-service`和`account-transaction-service`来处理`withdraw`和`deposit`操作的`account-money-transfer-service`。我们在上一章讨论分布式事务时，已经给出了后一种情况的例子。

## 4.9.2：参与者

参与者执行命令。最终用户是参与者，但服务也可以是参与者。例如，在一个数据导出微服务中，可以有一个输入消息消费者参与者/服务，它有一个从数据源消费消息的命令。

### 4.9.2.1：工厂

在领域驱动设计中，领域对象的创建可以从对象类本身分离到工厂。工厂是专门用于创建特定类型对象的对象。关于*工厂设计模式*的更多内容将在本章后续部分介绍。

### 4.9.2.2：仓储

仓储是一个具有用于持久化领域对象和从数据存储（例如数据库）中检索它们的方法的对象。通常，每个聚合根对应一个仓储，例如，用于订单实体的`order repository`。

### 4.9.2.3：服务

服务用于实现业务用例，包含不属于任何特定对象直接部分的功能。服务协调对聚合根的操作，例如`order service`协调对订单实体的操作。服务通常使用相关的仓储来执行与持久化相关的操作。服务也可以被视为具有特定命令的参与者。例如，在一个数据导出微服务中，可以有一个输入消息消费者参与者/服务，它有一个从数据源消费消息的命令。

### 4.9.2.4：事件

事件是对实体的操作，并构成业务用例。事件通常由服务处理。例如，与订单实体相关的事件可能有：创建订单、更新订单和取消订单。这些事件可以通过一个具有`create_order`、`update_order`和`cancel_order`方法的`order service`来实现。

### 4.9.2.5：事件风暴

*事件风暴*是一种轻量级方法，团队可以用来发现软件组件中与DDD相关的概念。事件风暴过程通常遵循以下步骤：

1. 找出领域*事件*（事件通常用过去时态书写）
2. 找出导致领域*事件*的*命令*
3. 添加执行*命令*的*参与者/服务*
4. 找出相关的*实体*（包括*聚合*和*聚合根*）和*值对象*

在事件风暴中，不同的DDD概念（如事件、命令、参与者和实体）用不同颜色的便利贴在墙上表示。相关的便利贴被分组在一起，例如，针对特定领域事件的参与者、命令和实体。

## 4.9.3：领域驱动设计示例：数据导出微服务

让我们以一个用于数据导出的微服务为例来说明DDD。数据导出将作为我们的顶层领域。开发团队应参与DDD和面向对象设计（OOD）过程。很可能一位专家级的软件开发者，例如团队技术负责人，可以独自完成DDD和OOD，但这并非正确做法。其他团队成员，尤其是初级成员，应参与进来以学习并进一步发展他们的技能。

DDD过程首先根据产品管理和架构团队的需求定义全局图景（顶层领域）：

> 数据导出器处理由包含多个字段的消息组成的数据。数据导出应从输入系统到输出系统进行。在导出过程中，可以对数据进行各种转换，并且输入和输出系统中的数据格式可能不同。

让我们通过找出领域事件来开始事件风暴过程：

1. 从输入系统消费消息
2. 输入消息被解码为通用的内部表示（即内部消息）
3. 内部消息被转换

4) 转换后的消息被编码为所需的输出格式
5) 消息被发送到输出系统
6) 配置被读取并解析

从上述事件中，我们可以梳理出四个子域：

- 输入（事件 1、2 和 6）
- 内部消息（事件 2 和 3）
- 转换（事件 3 和 6）
- 输出（事件 4、5 和 6）

## 数据导出器子域

![](img/cbd069395d7b824346b69b1f92e0fb4a_157_0.png)

图 4.17. 数据导出器子域

让我们以第一个领域事件“从输入系统消费消息”为例，分析是什么导致了该事件以及执行者是谁。由于不涉及最终用户，我们可以推断该事件是由一个“输入消息消费者”*服务*执行“消费消息”*命令*所引发的。此操作导致创建了一个“输入消息”*实体*。下图展示了这在墙上用便利贴表示的样子。

![](img/cbd069395d7b824346b69b1f92e0fb4a_159_0.png)

当继续对*输入*领域进行事件风暴时，我们可以发现它还包含以下额外的 DDD 概念：

- 命令
    - 读取输入配置
    - 解析输入配置
    - 消费输入消息
    - 解码输入消息
- 执行者/服务
    - 输入配置读取器
    - 输入配置解析器
    - 输入消息消费者
    - 输入消息解码器
- 实体
    - 输入消息
- 值对象
    - 输入配置

以下是输入领域中的子域、接口和类列表：

- 输入消息
    - 包含从输入数据源消费的消息
    - `InputMessage` 是一个协议，可以有多种具体实现，例如 `KafkaInputMessage`，表示从 Kafka 数据源消费的输入消息
- 输入消息消费者
    - 从输入数据源消费消息并创建 `InputMessage` 实例
    - `InputMessageConsumer` 是一个协议，可以有多种具体实现，例如 `KafkaInputMessageConsumer`，用于从 Kafka 数据源消费消息
- 输入消息解码器
    - 将输入消息解码为内部消息
    - `InputMessageDecoder` 是一个协议，可以有多种具体实现，例如 `AvroBinaryInputMessageDecoder`，用于解码以 Avro 二进制格式编码的输入消息
- 输入配置
    - 输入配置读取器
        * 读取领域的配置
        * `InputConfigReader` 是一个协议，可以有多种具体实现，例如 `LocalFileSystemInputConfigReader` 或 `HttpRemoteInputConfigReader`
    - 输入配置解析器
        * 解析读取的配置以生成 `InputConfig` 实例
        * `InputConfigParser` 是一个协议，可以有多种具体实现，例如 `JsonInputConfigParser` 或 `YamlInputConfigParser`
    - `InputConfig` 实例包含该领域的已解析配置，例如输入数据源类型、主机、端口和输入数据格式。

![](img/cbd069395d7b824346b69b1f92e0fb4a_161_0.png)

当更详细地考虑*内部消息*领域时，通过事件风暴过程，我们可以发现它还包含以下额外的 DDD 概念：

- 实体
    - 内部消息
    - 内部字段
- 聚合
    - 内部消息

以下是内部消息领域中的子域、接口和类列表：

- 内部消息
    - 内部消息由一个或多个内部消息字段组成
    - `InternalMessage` 是一个接口，用于提供输入消息内部表示的类
- 内部消息字段
    - `InternalMessageField` 是一个接口，用于表示内部消息单个字段的类

![](img/cbd069395d7b824346b69b1f92e0fb4a_162_0.png)

图 4.20. 内部消息子域

当更详细地考虑*转换器*领域时，通过事件风暴过程，我们可以发现它还包含以下额外的 DDD 概念：

- 命令
    - 读取转换器配置
    - 解析转换器配置
    - 转换消息
    - 转换字段
- 执行者/服务
    - 转换器配置读取器
    - 转换器配置解析器
    - 消息转换器
    - 字段转换器
- 值对象
    - 转换器配置

以下是转换领域中的子域、接口和类列表：

- 字段转换器
    - `FieldTransformers` 是 `FieldTransformer` 对象的集合
    - 字段转换器将输入消息字段的值转换为输出消息字段的值
    - `FieldTransformer` 是一个协议，可以有多种具体实现，例如 `FilterFieldTransformer`、`CopyFieldTransformer`、`TypeConversionFieldTransformer` 和 `ExpressionTransformer`
- 消息转换器
    - `MessageTransformer` 接收一个内部消息，并使用字段转换器对其进行转换
- 转换器配置
    - 转换器配置读取器
        * 读取领域的配置
        * `TransformerConfigReader` 是一个协议，可以有多种具体实现，例如 `LocalFileSystemTransformerConfigReader`
    - 转换器配置解析器
        * 解析读取的配置以生成 `TransformerConfig` 实例
        * `TransformerConfigParser` 是一个协议，可以有多种具体实现，例如 `JsonTransformerConfigParser`
    - `TransformerConfig` 实例包含转换器领域的已解析配置

![](img/cbd069395d7b824346b69b1f92e0fb4a_164_0.png)

图 4.21. 转换器子域

当更详细地考虑输出领域时，通过事件风暴过程，我们可以发现它还包含以下额外的 DDD 概念：

- 命令
    - 读取输出配置
    - 解析输出配置
    - 编码输出消息
    - 发送输出消息
- 执行者/服务
    - 输出配置读取器
    - 输出配置解析器
    - 输出消息编码器
    - 输出消息生产者
- 实体
    - 输出消息
- 值对象
    - 输出配置

以下是输出领域中的子域、接口和类列表：

- 输出消息编码器
    - 将转换后的消息编码为具有特定数据格式的输出消息
    - `OutputMessageEncoder` 是一个协议，可以有多种具体实现，例如 `CsvOutputMessageEncoder`、`JsonOutputMessageEncoder`、`AvroBinaryOutputMessageEncoder`
- 输出消息
    - `OutputMessage` 是输出字节序列的容器
- 输出消息生产者
    - 将输出消息发送到输出目标
    - `OutputMessageProducer` 是一个协议，可以有多种具体实现，例如 `KafkaMessageProducer`
- 输出配置
    - 输出配置读取器
        * 读取领域的配置
        * `OutputConfigReader` 是一个协议，可以有多种具体实现，例如 `LocalFileSystemOutputConfigReader`
    - 输出配置解析器
        * 将读取的配置解析为 `OutputConfig` 实例
        * `OutputConfigParser` 是一个协议，可以有多种具体实现，例如 `JsonOutputConfigParser`
    - `OutputConfig` 实例包含该领域的已解析配置，例如输出目标类型、主机、端口和输出数据格式

![](img/cbd069395d7b824346b69b1f92e0fb4a_166_0.png)

上述设计也遵循了*清洁微服务设计*原则。请注意，该原则不仅适用于 API，也适用于其他类型的微服务。从上述设计中，我们可以找出以下不属于微服务业务逻辑的接口适配器：

- InputMessageConsumer 接口实现
- InputMessageDecoder 接口实现
- OutputMessageEncoder 接口实现
- OutputMessageProducer 接口实现
- InputConfigReader 接口实现
- InputConfigParser 接口实现
- TransformerConfigReader 接口实现
- TransformerConfigParser 接口实现
- OutputConfigReader 接口实现
- OutputConfigParser 接口实现

我们应该能够修改上述实现或添加新的实现，而无需修改代码的其他部分（业务逻辑）。这一切意味着我们可以轻松地调整我们的微服务，以从不同数据源、以不同数据格式消费数据，并将转换后的数据以各种数据格式输出到不同的数据源。此外，我们微服务的配置可以从各种来源和各种格式读取。例如，如果我们现在从本地 JSON 格式的文件读取微服务配置，未来我们可能引入两个新类，并使用某种新的数据格式从API读取微服务配置。

在定义了上述子域之间的接口后，这四个子域可以非常并行地进行开发。这可以显著加快微服务的开发速度。每个子域的代码应放入单独的源代码文件夹中。我们将在下一章更详细地讨论源代码组织。

如果你将上述设计图组合起来，它们形成了一个数据处理管道，可以按以下方式实现：

```
class DataExporterApp:
    def run(self) -> None:
        while self.__is_running:
            input_msg = self.__input_msg_consumer.consume_input_msg()
            internal_msg = self.__input_msg_decoder.decode(input_msg)
            transformed_msg = self.__msg_transformer.transform(internal_msg)
            output_msg = self.__output_msg_encoder.encode(transformed_msg)
            self.__output_msg_producer.produce(output_msg)
```

而 `MessageTransformer` 类的 `transform` 方法可以按以下方式实现：

```
class MessageTransformer:
    def transform(self, internal_msg: InternalMessage) -> InternalMessage:
        transformed_msg = InternalMessage()
        for field_transformer in self.__field_transformers:
            field_transformer.transform_field(
                internal_msg, transformed_msg
            )
        return transformed_msg
```

## 4.9.4：领域驱动设计示例2：异常检测微服务

让我们再来看一个领域驱动设计的例子，一个异常检测微服务。该微服务的目的是检测测量数据中的异常。这个对微服务目的的简洁描述揭示了微服务的两个子域：

- 异常
- 测量

让我们首先更详细地分析*测量*子域，并为其定义领域事件：

- 测量数据源定义被加载
- 测量数据源定义被解析
- 测量定义被加载
- 测量定义被解析
- 测量数据从数据源获取
- 测量数据被缩放

让我们继续使用事件风暴法，并定义额外的领域驱动设计概念：

- 命令
    - 加载测量数据源定义
    - 解析测量数据源定义
    - 加载测量定义
    - 解析测量定义
    - 从数据源获取测量数据
    - 缩放测量数据
- 参与者/服务
    - 测量数据源定义加载器
    - 测量数据源定义解析器
    - 测量定义加载器
    - 测量定义解析器
    - 测量数据获取器
    - 测量数据缩放器
- 实体
    - 测量数据源
    - 测量
- 聚合
    - 测量
        - 测量数据源
        - 测量查询
- 值对象
    - 测量数据
    - 测量查询

让我们为*异常*子域定义领域事件：

- 异常检测配置被解析
- 异常检测配置被创建
- 异常检测规则被解析
- 异常检测规则被创建
- 根据异常检测规则，使用训练好的异常模型在测量中检测到异常
- 异常检测按固定间隔触发
- 为测量训练异常模型
- 异常模型被创建
- 异常模型训练按固定间隔触发
- 检测到的异常（即异常指示器）被创建
- 检测到的异常（即异常指示器）被序列化为所需格式，例如JSON
- 检测到的异常（即异常指示器）使用特定协议发布到特定目的地

让我们继续使用事件风暴法，并定义额外的领域驱动设计概念：

- 命令
    - 解析异常检测配置
    - 创建异常检测配置
    - 解析异常检测规则定义
    - 创建异常检测规则
    - 根据异常检测规则，使用训练好的异常模型在测量中检测异常
    - 按固定间隔触发异常检测
    - 使用特定AI技术（如自组织映射SOM）为测量训练异常模型
    - 创建异常模型
    - 按固定间隔触发异常模型训练
    - 创建异常指示器
    - 序列化异常指示器
    - 发布异常指示器
- 参与者/服务
    - 异常检测配置解析器
    - 异常检测规则解析器
    - 异常检测器
    - 异常检测引擎
    - 异常模型训练器（例如SOM）
    - 异常训练引擎
    - 异常指示器序列化器（例如JSON）
    - 异常指示器发布器（例如REST或Kafka）
- 工厂
    - 异常检测配置工厂
    - 异常检测规则工厂
    - 异常模型工厂
    - 异常指示器工厂
- 实体
    - 异常检测规则
    - 异常模型
    - 异常指示器

异常和测量这两个领域可以并行开发。异常领域与测量领域交互，以从特定数据源获取特定测量的数据。异常和测量领域的开发工作可以进一步拆分，以实现更多的开发并行化。例如，一个开发者可以负责异常检测，另一个负责异常模型训练，第三个负责异常指示器。

## 4.10：设计模式

以下章节介绍了25种设计模式，其中大多数因*四人帮*及其著作《设计模式》而闻名。设计模式分为创建型、结构型和行为型模式。

### 4.10.1：创建对象的设计模式

本节描述用于创建对象的设计模式。将介绍以下设计模式：

- 工厂模式
- 抽象工厂模式
- 工厂方法模式
- 建造者模式
- 单例模式
- 原型模式
- 对象池模式

### 4.10.1.1：工厂模式

> 工厂模式允许将创建何种对象的决定延迟到调用工厂的创建方法时。

工厂通常由一个或几个用于创建特定基类型对象的方法组成。工厂将创建对象的逻辑与对象本身分离，这符合*单一职责原则*。

下面是一个 `ConfigParserFactory` 的示例，它有一个用于创建不同种类 `ConfigParser` 对象的 `create` 方法。工厂的 `create` 方法的返回类型通常是一个接口。这允许创建特定类层次结构中的不同种类的对象。对于具有单个 `create` 方法的工厂，该方法通常包含一个匹配-案例语句或一个 if/elif 结构。工厂是面向对象编程中唯一允许使用大量匹配-案例语句或 if/elif 结构的地方。如果你在代码的其他地方有冗长的匹配-案例语句或长的 if/elif 结构，这通常是非面向对象设计的标志。

```
from enum import Enum
from typing import Protocol

class ConfigParser(Protocol):
    # ...

class JsonConfigParser(ConfigParser):
    # ...

class YamlConfigParser(ConfigParser):
    # ...

class ConfigFormat(Enum):
    JSON = 1
    YAML = 2

class ConfigParserFactory:
    @staticmethod
    def create_config_parser(config_format: ConfigFormat) -> ConfigParser:
        match config_format:
            case ConfigFormat.JSON:
                return JsonConfigParser()
            case ConfigFormat.YAML:
                return YamlConfigParser()
            case _:
                raise ValueError('Unsupported config format')
```

下面是一个具有多个 `create` 方法的工厂示例：

```
class ShapeFactory:
    @staticmethod
    def create_circle_shape(radius: int) -> Shape:
        return CircleShape(radius)

    @staticmethod
    def create_rectangle_shape(width: int, height: int) -> Shape:
        return RectangleShape(width, height)

    @staticmethod
    def create_square_shape(side_length: int) -> Shape:
        return SquareShape(side_length)
```

### 4.10.1.2：抽象工厂模式

> 在抽象工厂模式中，有一个抽象工厂（= 工厂接口）和一个或多个具体工厂（实现工厂接口的工厂类）。

抽象工厂模式是前面描述的*工厂模式*的扩展。通常，应该使用抽象工厂模式而不是普通的工厂模式。下面是一个具有一个具体实现的抽象 `ConfigParserFactory` 示例：

```
class ConfigParserFactory(Protocol):
    def create_config_parser(self, config_format: ConfigFormat) -> ConfigParser:
        pass
```

```
class ConfigParserFactoryImpl(ConfigParserFactory):
    def create_config_parser(self, config_format: ConfigFormat) -> ConfigParser:
        match config_format:
            case ConfigFormat.JSON:
                return JsonConfigParser()
            case ConfigFormat.YAML:
                return YamlConfigParser()
            case _:
                raise ValueError('Unsupported config format')
```

你应该遵循*面向接口编程原则*，在代码中使用抽象的 `ConfigParserFactory` 而不是具体的工厂。然后，使用*依赖注入原则*，你可以注入所需的工厂实现，例如 `ConfigParserFactoryImpl`。

在进行单元测试时，你应该创建模拟对象而不是使用工厂创建真实对象。抽象工厂模式可以帮到你，因为你可以在被测试的代码中提供 `ConfigParserFactory` 的模拟实例。然后，你可以期望模拟的 `create_config_parser` 方法被调用并返回一个符合 `ConfigParser` 协议的模拟实例。接着，你可以期望在 `ConfigParser` 模拟对象上调用 `parse` 方法并返回一个模拟的配置。下面是一个单元测试示例。我们测试 `Application` 类中的 `initialize` 方法，该类包含一个 `ConfigParserFactory` 类型的属性。`Application` 类使用 `ConfigParserFactory` 实例来创建 `ConfigParser` 对象以解析应用程序配置。

## 面向对象设计原则

```python
from typing import Protocol

class Config(Protocol):
    # ...

class ConfigParser(Protocol):
    def parse(self) -> Config:
        pass

    # ...

class ConfigParserFactory(Protocol):
    def create_config_parser(self) -> ConfigParser:
        pass

    # ...

class Application:
    def __init__(self, config_parser_factory: ConfigParserFactory):
        self.__config_parser_factory = config_parser_factory
        self.__config: Config | None = None

    def initialize(self) -> None:
        # ...
        config_parser = self.__config_parser_factory.create_config_parser(...)
        self.__config = config_parser.parse(...)
        # ...

    @property
    def config(self):
        return self.__config
```

以下是 `ApplicationTests` 类：

```python
from unittest import main, TestCase
from unittest.mock import Mock

from Config import Config
from ConfigParser import ConfigParser
from ConfigParserFactory import ConfigParserFactory

class ConfigParserFactoryMock(ConfigParserFactory):
    pass

class ConfigParserMock(ConfigParser):
    pass

class ConfigMock(Config):
    pass

class ApplicationTests(TestCase):
    def test_initialize(self):
        # GIVEN
        config_parser_factory_mock = ConfigParserFactoryMock()
        config_parser_mock = ConfigParserMock()
        config_parser_factory_mock.create_config_parser = Mock(
            return_value=config_parser_mock
        )
        config_mock = ConfigMock()
        config_parser_mock.parse = Mock(return_value=config_mock)
        application = Application(config_parser_factory_mock)

        # WHEN
        application.initialize()

        # THEN
        self.assertEqual(application.config, config_mock)

if __name__ == '__main__':
    main()
```

在上面的例子中，我们手动创建了模拟对象。`Mock` 构造函数创建了一个可调用的模拟对象（即它也是一个模拟函数）。你可以在 `Mock` 构造函数中提供模拟对象被调用时应返回的返回值。也可以使用 `@patch` 装饰器自动创建模拟对象，这可以使代码更简洁，如下所示：

```python
from unittest import TestCase, main
from unittest.mock import Mock, patch

from Config import Config
from ConfigParser import ConfigParser
from ConfigParserFactory import ConfigParserFactory

class ApplicationTests(TestCase):
    @patch.object(ConfigParserFactory, '__new__')
    @patch.object(ConfigParser, '__new__')
    @patch.object(Config, '__new__')
    def test_initialize(
        self,
        config_mock: Mock,
        config_parser_mock: Mock,
        config_parser_factory_mock: Mock,
    ):
        # GIVEN
        config_parser_factory_mock.create_config_parser.return_value = (
            config_parser_mock
        )
        config_parser_mock.parse.return_value = config_mock
        application = Application(config_parser_factory_mock)

        # WHEN
        application.initialize()

        # THEN
        self.assertEqual(application.config, config_mock)

if __name__ == '__main__':
    main()
```

单元测试和模拟将在后面的 *测试原则* 章节中更详细地描述。

### 4.10.1.3：工厂方法模式

> 在工厂方法模式中，对象是通过类中的一个或多个工厂方法创建的，并且类的构造函数被设为私有。工厂方法通常是类方法。

如果你想在构造函数中验证参数，构造函数可能会引发错误。你无法从构造函数中返回错误值。建议创建不会抛出异常的构造函数，因为如果构造函数签名中没有任何信息表明它可能引发错误，那么很容易忘记捕获构造函数中引发的错误。关于 *错误/异常处理原则* 的讨论，请参见下一章。

```python
class Url:
    def __init__(
        self,
        scheme: str,
        port: int,
        host: str,
        path: str,
        query: str
    ):
        # 验证参数，如果无效则抛出异常
```

你可以使用工厂方法模式来克服在构造函数中引发错误的问题。你可以创建一个工厂方法来返回可选值（如果你不需要返回错误原因），或者让工厂方法引发错误。我们可以在工厂方法名称前添加 `try` 前缀，以表示它可能引发错误。这样，函数签名（函数名）就能向读者传达该函数可能引发错误的信息。

下面是一个包含两个工厂方法的示例类。该类的构造函数使用 `PrivateConstructor` 元类设为私有。该类的用户只能通过工厂方法来创建类的实例。

```python
from typing import Any, TypeVar

T = TypeVar("T")

class PrivateConstructor(type):
    def __call__(
        cls: type[T],
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any]
    ):
        raise TypeError('Constructor is private')

    def _create(
        cls: type[T],
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any]
    ) -> T:
        return super().__call__(*args, **kwargs)

class Url(metaclass=PrivateConstructor):
    def __init__(
        self,
        scheme: str,
        port: int,
        host: str,
        path: str,
        query: str
    ):
        # ...

    @classmethod
    def create_url(
        cls,
        scheme: str,
        port: int,
        host: str,
        path: str,
        query: str
    ) -> 'Url | None':
        # 验证参数，如果无效则返回 'None'
        # 如果有效则返回一个 'Url' 实例：
        # return cls._create(str, port, host, path, query)

    @classmethod
    def try_create_url(
        cls,
        scheme: str,
        port: int,
        host: str,
        path: str,
        query: str
    ) -> 'Url':
        # 验证参数，如果无效则引发错误
        # 如果有效则返回一个 'Url' 实例：
        # return cls._create(str, port, host, path, query)
```

从工厂方法返回可选值允许利用函数式编程技术。Python 没有 Optional 类，但让我们先定义一个 Optional 类：

```python
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar('T')
U = TypeVar('U')

class Optional(Generic[T], metaclass=PrivateConstructor):
    def __init__(self, value: T | None):
        self.__value = value

    @classmethod
    def of(cls, value: T) -> 'Optional[T]':
        return cls._create(value)

    @classmethod
    def of_nullable(cls, value: T | None) -> 'Optional[T]':
        return cls._create(value)

    @classmethod
    def empty(cls) -> 'Optional[T]':
        return cls._create(None)

    def is_empty(self) -> bool:
        return True if self.__value is None else False

    def is_present(self) -> bool:
        return False if self.__value is None else True

    def try_get(self) -> T:
        if self.__value is None:
            raise RuntimeError('No value to get')
        return self.__value

    def try_get_or_else_raise(self, error: Exception):
        if self.__value is None:
            raise error
        return self.__value

    def if_present(self, consume: Callable[[T], None]) -> 'Optional[T]':
        if self.__value is not None:
            consume(self.__value)
        return self

    def or_else(self, other_value: T) -> T:
        return other_value if self.__value is None else self.__value

    def or_else_get(self, supply_value: Callable[[], T]) -> T:
        return supply_value() if self.__value is None else self.__value

    def map(self, map_: Callable[[T], U | None]) -> 'Optional[U]':
        return (
            self
            if self.__value is None
            else self.of_nullable(map_(self.__value))
        )

    def flat_map(self, map_: Callable[[T], 'Optional[U]']) -> 'Optional[U]':
        return self if self.__value is None else map_(self.__value)
```

> 注意！当我在本书中使用 Optional 类时，它总是上面定义的类，而不是 Python typing 模块中的 Optional。

注意上面的 Optional 类代码如何利用了工厂方法模式。它有一个私有构造函数和三个工厂方法来创建不同类型的 Optional 对象。这样做的好处是你可以为工厂方法提供描述性的名称，而使用单个构造函数则无法做到这一点。工厂方法的名称说明了将创建什么样的对象。

```python
class Url(metaclass=PrivateConstructor):
    def __init__(
        self,
        scheme: str,
        port: int,
        host: str,
        path: str,
        query: str
    ):
        # ...

    @classmethod
    def create_url(
        cls,
        scheme: str,
        host: str,
        port: int,
        path: str,
        query: str
    ) -> Optional['Url']:
        # ...

maybeUrl = Url.create_url(...)

# 在 lambda 中对 URL 进行操作
maybeUrl.if_present(lambda url: print(url))

def print_url(url: Url):
    print(url)

# 使用函数对 URL 进行操作
maybeUrl.if_present(print_url)
```

### 4.10.1.4：建造者模式

建造者模式允许你逐步构建对象。

在建造者模式中，你通过建造者类的`_addxxx`方法为被构建对象添加属性。添加所有需要的属性后，你可以使用建造者类的`build`或`_buildxxx`方法来构建最终对象。

例如，你可以从URL的各个部分构建一个URL。以下是使用`UrlBuilder`类的示例：

```python
url = UrlBuilder().add_scheme('https').add_host('www.google.com').build_url()
```

建造者模式的好处在于，为建造者提供的属性可以在构建方法中进行验证。你可以让建造者的方法返回一个可选值，以指示构建是否成功。或者，如果你需要返回错误，可以让构建方法抛出异常。那么你应该使用`try`前缀来命名构建方法，例如`try_build_url`。建造者模式的另一个好处是不需要为建造者添加默认属性。例如，`https`可以是默认的协议，如果你正在构建一个HTTPS URL，就不需要调用`add_scheme`。唯一的问题是你必须查阅建造者文档来确定默认值。

建造者模式的一个缺点是，你可能会以逻辑上错误的顺序提供参数，就像这样：

```python
url = UrlBuilder().add_host('www.google.com').add_scheme('https').build_url()
```

它能工作，但看起来不太美观。所以，如果你使用建造者，如果存在逻辑上正确的顺序，总是尝试以该顺序提供参数。当参数之间没有任何固有顺序时，建造者模式效果很好。以下是这样一个情况的例子：使用`HouseBuilder`类建造一栋房子。

```python
house = HouseBuilder()
    .add_kitchen()
    .add_living_room()
    .add_bedrooms(3)
    .add_bath_rooms(2)
    .add_garage()
    .build_house()
```

你可以通过带有默认值参数的工厂方法实现与建造者类似的功能：

```python
class Url(metaclass=PrivateConstructor):
    def __init__(
        self,
        host: str,
        path: str,
        query: str,
        scheme: str = 'https',
        port: int = 443,
    ):
        # ...

    @classmethod
    def create_url(
        cls,
        host: str,
        path: str,
        query: str,
        scheme: str = 'https',
        port: int = 443,
    ) -> 'Url | None':
        # ...
```

在上面的工厂方法中，默认值是什么一目了然。当然，你现在无法以逻辑顺序提供参数。而且，由于许多参数类型相同（字符串），你意外地以错误顺序提供某些参数的可能性也更大。对于使用具有特定名称的方法来提供特定参数的建造者来说，这不会是一个潜在问题。在现代开发环境中，以错误顺序提供参数的可能性较小，因为IDE提供了内联参数提示。很容易看出你是否将特定参数放在了错误的位置。如下所示，使用语义验证的函数参数类型也可以避免以错误顺序提供参数。语义验证的函数参数将在本章后面讨论。

```python
class Url(metaclass=PrivateConstructor):
    # ...

    @classmethod
    def create_url(
        cls,
        host: str,
        path: str,
        query: str,
        scheme: Scheme = Scheme.create('https'),
        port: Port = Port.create(443),
    ) -> 'Url | None':
        # ...
```

你总是可以使用参数对象。以下是一个示例：

```python
from dataclasses import dataclass
# ...

@dataclass
class UrlParams:
    host: str
    scheme: str = 'https'
    port: int = 443
    path: str = ""
    query: str = ""

class Url(metaclass=PrivateConstructor):
    def __init__(self, url_params: UrlParams):
        # ...

    @classmethod
    def create_url(cls, url_params: UrlParams) -> 'Optional[Url]':
        # ...

url_params = UrlParams('www.google.com', query='query=design+patterns')
maybe_url = Url.create_url(url_params)
```

上述解决方案与使用建造者非常相似。

## 4.10.1.5：单例模式

> 单例模式定义了一个类只能有一个实例。

单例在像Java这样的纯面向对象语言中非常常见。在许多情况下，单例类可以被识别为没有任何状态。这就是为什么只需要该类的一个实例。创建多个相同的实例没有意义。在一些非纯面向对象语言中，单例不一定像在纯面向对象语言中那样常见，通常可以通过直接定义函数来替代。

在Python中，可以在模块中创建一个单例实例并导出。当你在其他模块中从该模块导入该实例时，其他模块将始终获得相同的导出实例，而不是每次都获得一个新实例。以下是这样一个单例的例子。首先，我们在名为`my_class_singleton.py`的模块中定义一个单例：

```python
class MyClass:
    # ...

my_class_singleton = MyClass()
```

然后在`other_module_1.py`中：

```python
from my_class_singleton import my_class_singleton

print(my_class_singleton)
```

最后在`other_module_2.py`中：

```python
from my_class_singleton import my_class_singleton
import other_module_1

print(my_class_singleton)
```

当你运行`other_module_2`时，你应该得到如下输出，其中对象地址相同，这意味着`my_class_singleton`确实是一个单例：

```
<my_class_singleton.MyClass object at 0x101042f90>
<my_class_singleton.MyClass object at 0x101042f90>
```

单例模式可以使用仅包含静态方法的类来实现。静态类的问题在于单例类被硬编码，并且静态类在单元测试中可能难以或无法模拟。我们应该记住要面向接口编程。实现单例模式的最佳方式是使用依赖倒置原则和依赖注入原则。以下是使用`dependency-injector`库处理依赖注入的示例。`FileConfigReader`类的构造函数期望一个`ConfigParser`实例。我们使用`@inject`注解标注构造函数，并从DI容器（稍后定义）中提供一个名为`config_parser`的`ConfigParser`实例：

```python
from typing import Protocol

from dependency_injector.wiring import Provide, inject
# Import ConfigParser protocol with try_parse method here...
# Import Config

class ConfigReader(Protocol):
    def try_read(self, config_location: str) -> Config:
        pass

class FileConfigReader(ConfigReader):
    @inject
    def __init__(
        self,
        config_parser: ConfigParser = Provide['config_parser']
    ):
        self.__config_parser = config_parser

    def try_read(self, config_file_path_name: str) -> Config:
        config_file_contents = # Read configuration file
        config = self.__config_parser.try_parse(config_file_contents)
        return config
```

在下面的`DiContainer`类中，我们首先配置接线，然后将名称`config_parser`绑定到`ConfigParserImpl`类的单例实例。（`ConfigParserImpl`类的代码此处未显示）。`wiring_config`期望`FileConfigReader`类定义在名为`FileConfigReader.py`的模块中。

```python
from dependency_injector import containers, providers

class DiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=['FileConfigReader']
    )

    config_parser = providers.Singleton(ConfigParserImpl)
```

## 4.10.1.6：原型模式

> 原型模式允许你使用现有对象作为原型来创建新对象。

让我们用一个`DrawnShape`类来举例：

```python
class Shape(Protocol):
    # ...

# Implement concrete shapes...

class Position(Protocol):
    def get_x(self) -> int:
        pass

    def get_y(self) -> int:
        pass

class DrawnShape:
    def __init__(self, position: Position, shape: Shape):
        self.__position = position
        self.__shape = shape

    def clone_to(self, position: Position) -> 'DrawnShape':
        return DrawnShape(position, self.__shape)
```

使用原型模式就是调用原型对象的`clone_to`方法，并提供`position`参数来指定新形状应该放置的位置。

原型模式也用于JavaScript中实现原型继承。自EcmaScript版本6以来，基于类的继承已经可用，不再需要使用原型继承。

原型继承的理念是，同一类对象的公共部分存储在一个原型实例中。这些公共部分通常指共享的方法。在每个对象中重复存储方法是没有意义的。那将是一种资源浪费，因为Javascript函数本身就是对象。

当你使用`Object.create`方法创建一个新对象时，你会将原型作为参数传入。之后，你可以为新创建的对象设置属性。当你在创建的对象上调用一个方法，如果该方法在对象的属性中找不到，就会在原型对象中查找该方法。原型可以链接，使得一个原型对象包含另一个原型对象。这种链接用于实现继承链。下面是原型继承的一个简单示例：

```
const pet = {
  name: '',
  getName: function() { return this.name; }
};

// 使用'pet'对象作为原型创建一个新对象
const petNamedBella = Object.create(pet);

petNamedBella.name = 'Bella';
console.log(petNamedBella.getName()); // 输出 'Bella'

// 一个狗的原型，其中包含嵌套的'pet'原型
const dog = {
  bark: function() { console.log('bark'); },
  __proto__: pet
}

// 使用'dog'对象作为原型创建一个新对象
const dogNamedLuna = Object.create(dog);

dogNamedLuna.name = 'Luna';
console.log(dogNamedLuna.getName()); // 输出 'Luna'
dogNamedLuna.bark(); // 输出 'bark'
```

## 4.10.1.7: 对象池模式

> 在对象池模式中，创建的对象被存储在一个池中，对象可以从池中获取并返回以供重用。对象池模式是一种优化模式，因为它允许重用已创建的对象。

如果你需要创建许多短生命周期的对象，你应该利用对象池，以减少对内存分配和释放的需求，这需要时间。在具有垃圾回收的语言中，频繁的对象创建和删除会给垃圾收集器带来额外的工作，从而消耗CPU时间。

下面是一个对象池协议的示例。

```
from typing import Protocol, TypeVar

T = TypeVar('T')
```

```
class ObjectPool(Protocol[T]):
    def acquire_object(self, cls: type[T]) -> T:
        pass

    def return_object(self, object_: T) -> None:
        pass
```

下面是一个对象池实现的示例。

```
class LimitedSizeObjPool(ObjectPool[T]):
    def __init__(self, max_pool_size: int):
        self.__max_pool_size = max_pool_size
        self.__pooled_objects = []

    def acquire_object(self, cls: type[T]) -> T:
        if self.__pooled_objects:
            return self.__pooled_objects.pop(0)
        else:
            return cls()

    def return_object(self, object_: T) -> None:
        pool_is_not_full = len(self.__pooled_objects) < self.__max_pool_size

        if pool_is_not_full:
            self.__pooled_objects.append(object_)
```

```
class MyObject:
    # ...
```

```
my_object_pool: LimitedSizeObjPool[MyObject] = LimitedSizeObjPool(2)
my_object_1 = my_object_pool.acquire_object(MyObject)
my_object_2 = my_object_pool.acquire_object(MyObject)
my_object_3 = my_object_pool.acquire_object(MyObject)
my_object_pool.return_object(my_object_1)
my_object_pool.return_object(my_object_2)
my_object_pool.return_object(my_object_3) # 不适合放入池中
print(len(my_object_pool._LimitedSizeObjPool__pooled_objects))
my_object_4 = my_object_pool.acquire_object(MyObject)
print(my_object_1 == my_object_4) # 输出 True
```

下面是一个略有不同的对象池实现。下面的实现接受可清除的对象，这意味着返回到池中的对象在重用前会被清除。你也可以提供在构造对象时使用的参数。

```
class Clearable(Protocol):
    def clear(self) -> None:
        pass

T = TypeVar('T', bound=Clearable)

class LimitedSizeObjPool(ObjectPool[T]):
    def __init__(self, max_pool_size: int, *args, **kwargs):
        self.__max_pool_size = max_pool_size
        self.__args = args
        self.__kwargs = kwargs
        self.__pooled_objects = []

    def acquire_object(self, cls: type[T]) -> T:
        if self.__pooled_objects:
            return self.__pooled_objects.pop(0)
        else:
            return cls(*self.__args, **self.__kwargs)

    def return_object(self, object_: T) -> None:
        pool_is_not_full = len(self.__pooled_objects) < self.__max_pool_size

        if pool_is_not_full:
            object_.clear()
            self.__pooled_objects.append(object_)

class MyObject(Clearable):
    def __init__(self, param1: int, param2: str, **kwargs):
        print(param1, param2, kwargs)

    def clear(self) -> None:
        print('Cleared')

my_object_pool: LimitedSizeObjPool[MyObject] = LimitedSizeObjPool(
    2, 1, 'test', name='John'
)

# 输出: 1 test {'name': 'John'}
my_object_1 = my_object_pool.acquire_object(MyObject)

# 输出: Cleared
my_object_pool.return_object(my_object_1)
```

## 4.10.2: 结构型设计模式

本节描述结构型设计模式。大多数模式使用对象组合作为实现特定设计的主要方法。以下设计模式将被介绍：

- 组合模式
- 外观模式
- 桥接模式
- 策略模式
- 适配器模式
- 代理模式
- 装饰器模式
- 享元模式

## 4.10.2.1: 组合模式

> 在组合模式中，一个类可以由自身组合而成，即组合是递归的。

递归的对象组合可以通过用户界面如何由不同的小部件组合来描绘。在下面的例子中，我们有一个`Pane`类，它是一个`Widget`。一个`Pane`对象可以包含多个其他`Widget`对象，这意味着一个`Pane`对象可以包含其他`Pane`对象。

```
from typing import Protocol

class Widget(Protocol):
    def render(self) -> None:
        pass

class Pane(Widget):
    def __init__(self, widgets: list[Widget]):
        self.__widgets = widgets

    def render(self) -> None:
        # 渲染窗格内的每个小部件

class StaticText(Widget):
    def render(self) -> None:
        # 渲染静态文本小部件

class TextInput(Widget):
    def render(self) -> None:
        # 渲染文本输入小部件

class Button(Widget):
    def render(self) -> None:
        # 渲染按钮小部件

class UiWindow:
    def __init__(self, widgets: list[Widget]):
        self.__widgets = widgets

    def render(self) -> None:
        for widget in self.__widgets:
            widget.render()
```

形成树结构的对象是由自身递归组合而成的。下面是一个带有嵌套记录字段的Avro记录字段模式：

```
{
    "type": "record",
    "name": "sampleMessage",
    "fields": [
        {
            "name": "field1",
            "type": "string"
        },
        {
            "name": "nestedRecordField",
            "namespace": "nestedRecordField",
            "type": "record",
            "fields": [
                {
                    "name": "nestedField1",
                    "type": "int",
                    "signed": "false"
                }
            ]
        }
    ]
}
```

为了解析Avro模式，我们可以根据字段类型为不同的子模式定义类。在分析下面的例子时，我们可以注意到`RecordAvroFieldSchema`类可以包含任何`AvroFieldSchema`对象，也包括其他`RecordAvroFieldSchema`对象，这使得`RecordAvroFieldSchema`对象成为一个组合对象。

```
from typing import Protocol

class AvroFieldSchema(Protocol):
    # ...

class RecordAvroFieldSchema(AvroFieldSchema):
    def __init__(self, sub_field_schemas: list[AvroFieldSchema]):
        self.__sub_field_schemas = sub_field_schemas

class StringAvroFieldSchema(AvroFieldSchema):
    # ...

class IntAvroFieldSchema(AvroFieldSchema):
    # ...

# 其余Avro数据类型的模式类 ...
```

## 4.10.2.2: 外观模式

> 在外观模式中，一个更高抽象层次的对象由更低抽象层次的对象组合而成。更高层次的对象充当更低层次对象的外观。外观后面的更低层次对象要么只能通过外观访问，要么主要只能通过外观访问。

让我们以数据导出器微服务为例。对于该微服务，我们可以创建一个`Config`接口，该接口可用于获取数据导出器微服务不同部分（输入、转换器和输出）的配置。`Config`接口充当外观。外观的使用者不需要看到外观背后的情况。他们不知道外观背后发生了什么。他们也不应该关心，因为他们只是使用外观提供的接口。

在外观背后可能有各种类在做实际工作。在下面的例子中，有一个`ConfigReader`从可能不同的来源（例如，从本地文件或远程服务）读取配置，还有一些配置解析器可以解析配置的特定部分，可能使用不同的数据格式，如JSON或YAML。这些实现和细节对外观的使用者都是不可见的。外观背后的任何这些实现都可以随时更改，而不会影响外观的使用者，因为外观使用者与更低层次的实现没有耦合。

下面是`Config`外观的实现：

```
from typing import Protocol

from dependency_injector.wiring import Provide, inject

class Config(Protocol):
    def try_get_input_config(self) -> InputConfig:
        pass

    def try_get_transformer_config(self) -> TransformerConfig:
        pass

    def try_get_output_config(self) -> OutputConfig:
        pass

class ConfigImpl(Config):
    @inject
    def __init__(
        self,
        config_reader: ConfigReader = Provide['config_reader'],
        input_config_parser: InputConfigParser = Provide[
            'input_config_parser'
        ],
        transformer_config_parser: TransformerConfigParser = Provide[
            'transformer_config_parser'
        ],
        output_config_parser: OutputConfigParser = Provide[
            'output_config_parser'
        ],
    ):
```

):
    self.__config_reader = config_reader
    self.__input_config_parser = input_config_parser
    self.__transformer_config_parser = transformer_config_parser
    self.__output_config_parser = output_config_parser
    self.__config_string = ""
    self.__input_config = None
    self.__output_config = None
    self.__transformer_config = None

def try_get_input_config(self) -> InputConfig:
    if self.__input_config is None:
        self.__try_read_config_if_needed()
        self.__input_config = self.__input_config_parser.try_parse(
            self.__config_string
        )
    return self.__input_config

def try_get_transformer_config(self) -> TransformerConfig:
    # ...

def try_get_output_config(self) -> OutputConfig:
    # ...

def __try_read_config_if_needed(self) -> None:
    if not self.__config_string:
        self.__config_string = self.__config_reader.try_read(...)

如果上述门面在Java中实现，有一个独特的替代方案：只有`Config`接口和`ConfigImpl`类可以设为`public`，所有与配置读取和解析相关的接口和类都可以设为包私有。这将强制使用该门面。除了`ConfigurationImpl`类之外，其他任何类都无法使用与配置读取和解析相关的底层实现类。

## 4.10.2.3：桥接模式

> 在桥接模式中，一个类的实现被委托给另一个类。原始类在某种意义上是“抽象的”，因为它除了委托给另一个类之外没有任何行为，或者它可以有一些更高级的控制逻辑来决定如何委托给另一个类。

不要将这里的“抽象”一词与抽象类混淆。在抽象类中，某些行为根本没有实现，而是将实现延迟到抽象类的子类。在这里，我们可以使用“委托类”这个术语来代替“抽象类”。

# 桥接

![](img/cbd069395d7b824346b69b1f92e0fb4a_191_0.png)

让我们用一个能够绘制不同形状的形状和绘图示例来说明：

```python
from typing import Protocol

from Point import Point
from ShapeRenderer import ShapeRenderer

class Shape(Protocol):
    def render(self, renderer: ShapeRenderer) -> None:
        pass

class RectangleShape(Shape):
    def __init__(
        self,
        upper_left_corner: Point,
        width: int,
        height: int
    ):
        self.__upper_left_corner = upper_left_corner
        self.__width = width
        self.__height = height

    def render(self, renderer: ShapeRenderer) -> None:
        renderer.render_rectangle(
            self.__upper_left_corner,
            self.__width,
            self.__height
        )

class CircleShape(Shape):
    def __init__(self, center: Point, radius: int):
        self.__center = center
        self.__radius = radius

    def render(self, renderer: ShapeRenderer):
        renderer.render_circle(self.__center, self.__radius)
```

上面的`RectangleShape`和`CircleShape`类是抽象类，因为它们将功能（渲染）委托给`ShapeRenderer`类型的外部类（实现类）。

我们可以为形状类提供不同的渲染实现。让我们定义两个形状渲染器，一个用于渲染光栅形状，另一个用于渲染矢量形状：

```python
from typing import Protocol

from Canvas import Canvas
from Point import Point
from SvgElement import SvgElement

class ShapeRenderer(Protocol):
    def render_circle(self, center: Point, radius: int) -> None:
        pass

    def render_rectangle(
        self,
        upper_left_corner: Point,
        width: int,
        height: int
    ) -> None:
        pass

    # 渲染其他形状的方法 ...

class RasterShapeRenderer(ShapeRenderer):
    def __init__(self, canvas: Canvas):
        self.__canvas = canvas

    def render_circle(self, center: Point, radius: int):
        # 将圆形渲染到画布上

    def render_rectangle(
        self,
        upper_left_corner: Point,
        width: int,
        height: int
    ):
        # 将矩形渲染到画布上

    # 将其他形状渲染到画布上的方法 ...

class VectorShapeRenderer(ShapeRenderer):
    def __init__(self, svg_root: SvgElement):
        self.__svg_root = svg_root

    def render_circle(self, center: Point, radius: int):
        # 将圆形渲染为SVG元素并作为子元素附加到SVG根元素

    def render_rectangle(
        self,
        upper_left_corner: Point,
        width: int,
        height: int
    ):
        # 将矩形渲染为SVG元素
        # 并作为子元素附加到SVG根元素

    # 渲染其他形状的方法 ...
```

让我们实现两个不同的绘图，一个光栅绘图和一个矢量绘图：

```python
from abc import abstractmethod
from typing import Protocol

from Canvas import Canvas
from RasterShapeRenderer import RasterShapeRenderer
from Shape import Shape
from ShapeRenderer import ShapeRenderer
from SvgElement import SvgElement
from VectorShapeRenderer import VectorShapeRenderer
```

```python
class Drawing(Protocol):
    def get_shape_renderer(self) -> ShapeRenderer:
        pass

    def draw(self, shapes: list[Shape]) -> None:
        pass

    def save(self) -> None:
        pass
```

```python
class AbstractDrawing(Drawing):
    def __init__(self, name: str):
        self.__name = name

    @abstractmethod
    def get_shape_renderer(self) -> ShapeRenderer:
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        pass

    @abstractmethod
    def get_data(self) -> bytearray:
        pass

    def save(self) -> None:
        file_name = self.__name + self.get_file_extension()
        data = self.get_data()

        # 将'data'保存到'file_name' ...

    def draw(self, shapes: list[Shape]) -> None:
        for shape in shapes:
            shape.render(self.get_shape_renderer())
```

```python
class RasterDrawing(AbstractDrawing):
    def __init__(self, name: str):
        super().__init__(name)
        self.__canvas = Canvas()
        self.__shape_renderer = RasterShapeRenderer(self.__canvas)

    def get_shape_renderer(self) -> ShapeRenderer:
        return self.__shape_renderer

    def get_file_extension(self) -> str:
        return '.png'

    def get_data(self) -> bytearray:
        # 从'canvas'对象获取数据
```

```python
class VectorDrawing(AbstractDrawing):
    def __init__(self, name: str):
        super().__init__(name)
        self.__svg_root = SvgElement()
        self.__shape_renderer = VectorShapeRenderer(self.__svg_root)

    def get_shape_renderer(self) -> ShapeRenderer:
        return self.__shape_renderer

    def get_file_extension(self) -> str:
        return '.svg'

    def get_data(self) -> bytearray:
        # 从'svg_root'对象获取数据
```

在上面的例子中，我们将形状类的渲染行为委托给了实现`ShapeRenderer`协议的具体类。`Shape`类只代表一个形状，但不渲染形状。它们只有一个职责，就是代表一个形状。关于渲染，形状类是“抽象类”，因为它们将渲染委托给负责渲染不同形状的其他类。

现在我们可以有一个形状列表，并以不同的方式渲染它们。我们可以如下所示地进行，因为我们没有将形状类与任何特定的渲染行为耦合。

```python
shapes = [RectangleShape(Point(), 2, 3), CircleShape(Point(), 4)]

raster_drawing = RasterDrawing('raster-drawing')
raster_drawing.draw(shapes)
raster_drawing.save()

vector_drawing = VectorDrawing('vector-drawing')
vector_drawing.draw(shapes)
vector_drawing.save()
```

## 4.10.2.4：策略模式

> 在策略模式中，可以通过将组合类型的实例更改为该类型的不同实例来改变对象的功能。

下面是一个示例，其中`ConfigReader`类的行为可以通过将`configParser`字段的值更改为不同类的实例来改变。默认行为是解析JSON格式的配置，这可以通过调用无参构造函数来实现。

from typing import Protocol

# 定义 Config 类 ...

class ConfigParser(Protocol):
    def try_parse(self, config_str: str) -> Config:
        pass

# 定义 JsonConfigParser 类 ...

class ConfigReader:
    def __init__(self, config_parser: ConfigParser = JsonConfigParser()):
        self.__config_parser = config_parser

    def try_read(self, config_file_path_name: str) -> Config:
        # 尝试将 'config_file_path_name' 读取为 'config_str'

        config = self.__config_parser.try_parse(config_str)
        return config

通过策略模式，我们可以通过改变 `config_parser` 字段的值来改变 `ConfigReader` 实例的功能。例如，可能存在以下实现了 `ConfigParser` 协议的类：

- JsonConfigParser
- YamlConfigParser
- TomlConfigParser

我们可以通过将 `YamlConfigParser` 类的实例作为 `ConfigReader` 构造函数的参数，来动态改变 `ConfigReader` 实例的行为，使其使用 YAML 解析策略。

## 4.10.2.5：适配器模式

> 适配器模式将一个接口转换为另一个接口。适配器模式允许你将不同的接口适配到一个单一的接口。

在下面的例子中，我们定义了一个 `Message` 协议，用于表示可以使用 `MessageConsumer` 从数据源消费的消息。

图 4.24. Message.py

```python
from typing import Protocol

class Message(Protocol):
    def get_data(self) -> bytearray:
        pass

    def get_length_in_bytes(self) -> int:
        pass
```

图 4.25. MessageConsumer.py

```python
from typing import Protocol

from Message import Message

class MessageConsumer(Protocol):
    def consume_message(self) -> Message:
        pass
```

接下来，我们可以为 Apache Kafka 和 Apache Pulsar 定义消息和消息消费者适配器类：

图 4.26. KafkaMsgConsumer.py

```python
from MessageConsumer import MessageConsumer

class KafkaMsgConsumer(MessageConsumer):
    def consume_message(self) -> Message:
        # 使用第三方 Kafka 库从 Kafka 消费一条消息
        # 将消费的消息包装在 KafkaMessage 类的实例中
        # 返回 KafkaMessage 实例
```

图 4.27. KafkaMessage.py

```python
from Message import Message

class KafkaMessage(Message):
    def __init__(self, kafka_lib_msg):
        self.__kafka_lib_msg = kafka_lib_msg

    def get_data(self) -> bytearray:
        return self.__kafka_lib_msg.value

    def get_length_in_bytes(self) -> int:
        return len(self.__kafka_lib_msg.value)
```

图 4.28. PulsarMsgConsumer.py

```python
from MessageConsumer import MessageConsumer

class PulsarMsgConsumer(MessageConsumer):
    def consume_message(self) -> Message:
        # 使用 Pulsar 客户端从 Pulsar 消费一条消息
        # 将消费的 Pulsar 消息包装在 PulsarMessage 的实例中
        # 返回 PulsarMessage 实例
```

图 4.29. PulsarMessage.py

```python
from Message import Message

class PulsarMessage(Message):
    # ...
```

现在我们可以使用具有相同消费者和消息接口的 Kafka 或 Pulsar 数据源。将来，将新的数据源集成到系统中将变得容易。我们只需要为新的数据源实现适当的适配器类（消息和消费者类）。不需要任何其他代码更改。因此，我们将正确地遵循*开闭原则*。

让我们想象一下，所使用的 Kafka 库的 API 发生了变化。我们不需要在代码的许多地方进行更改。我们需要为新的 API 创建新的适配器类（消息和消费者类），并用这些新的适配器类替换旧的适配器类。所有这些工作再次遵循了*开闭原则*。

即使没有需要适配的东西，也要考虑使用适配器模式，尤其是在使用第三方库时。因为这样你就能为未来可能发生的变化做好准备。第三方库的接口可能会改变，或者可能需要使用不同的库。如果你没有使用适配器模式，使用新的库或库版本可能意味着你必须在代码库的多个地方进行许多小的更改，这容易出错，并且违反了*开闭原则*。

让我们举一个使用第三方日志库的例子。最初，我们为 *abc-logging-library* 的适配器只是库中 `abc_logger` 实例的一个包装器。没有进行任何实际的适配工作。

图 4.30. Logger.py

```python
from typing import Protocol

from LogLevel import LogLevel

class Logger(Protocol):
    def log(self, log_level: LogLevel, message: str) -> None:
        pass
```

图 4.31. AbcLogger.py

```python
from abc_logging_library import abc_logger
from LogLevel import LogLevel

class AbcLogger(Logger):
    def log(self, log_level: LogLevel, message: str) -> None:
        abc_logger.log(log_level, message)
```

假设将来有一个更好的日志库可用，叫做 *xyz-logging-library*，我们想使用它，但它的接口有点不同。它的日志实例叫做 *xyz_log_writer*，日志方法名称不同，并且参数顺序与 *abc-logging-library* 相比也不同。我们可以为新的日志库创建一个适配器，而不需要在代码库的其他地方进行任何其他代码更改：

图 4.32. XyzLogger.py

```python
from xyz_logging_library import xyz_log_writer
from Logger import Logger
from LogLevel import LogLevel

class XyzLogger(Logger):
    def log(self, log_level: LogLevel, message: str) -> None:
        xyz_log_writer.write_log_entry(message, log_level)
```

我们不必修改代码中所有使用日志记录的地方。通常，日志记录在许多地方使用。我们为自己节省了大量容易出错且不必要的工作，并且再次遵循了*开闭原则*。

## 4.10.2.6：代理模式

> 代理模式能够有条件地修改或增强对象的行为。

使用代理模式时，你会定义一个代理类来包装另一个类（被代理的类）。代理类有条件地委托给被包装的类。代理类实现了被包装类的接口，并在代码中代替被包装类使用。

下面是一个代理类 *CachingEntityStore* 的例子，它缓存实体存储操作的结果：

图 4.33. Cache.py

```python
from typing import Protocol, TypeVar

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')

class Cache(Protocol[TKey, TValue]):
    def retrieve_value(self, key: TKey) -> TValue | None:
        pass

    def store(
        self,
        key: TKey,
        value: TValue,
        time_to_live_in_secs: int = 0
    ) -> None:
        pass
```

图 4.34. MemoryCache.py

```python
from typing import TypeVar

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')

class MemoryCache(Cache[TKey, TValue]):
    # ...

    def retrieve_value(self, key: TKey) -> TValue | None:
        # ...

    def store(
        self,
        key: TKey,
        value: TValue,
        time_to_live_in_secs: int = 0
    ) -> None:
        # ...
```

图 4.35. EntityStore.py

```python
from collections.abc import Awaitable
from typing import Protocol, TypeVar

T = TypeVar('T')

class EntityStore(Protocol[T]):
    async def try_get_entity(self, id_: int) -> Awaitable[T]:
        pass
```

图 4.36. DbEntityStore.py

```python
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar('T')

class DbEntityStore(EntityStore[T]):
    async def try_get_entity(self, id_: int) -> Awaitable[T]:
        # 尝试从数据库获取实体 ...
```

图 4.37. CachingEntityStore.py

```python
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar('T')

class CachingEntityStore(EntityStore[T]):
    __entity_cache: MemoryCache[int, T]

    def __init__(self, entity_store: EntityStore[T]):
        self.__entity_store = entity_store;
        self.__entity_cache = MemoryCache()

    async def try_get_entity(self, id_: int) -> Awaitable[T]:
        entity = self.__entity_cache.retrieve_value(id_)

        if entity is None:
            entity = await self.__entity_store.try_get_entity(id_)
            time_to_live_in_secs = 60
            self.__entity_cache.store(id_, entity, time_to_live_in_secs)

        return entity
```

在上面的例子中，`CachingEntityStore` 类是包装 `EntityStore` 的代理类。代理类通过有条件地委托给被包装的类来修改被包装类的行为。它仅在缓存中未找到实体时才委托给被包装的类。

下面是另一个代理类的例子，它在执行服务操作之前对用户进行授权：

```python
from collections.abc import Awaitable
from typing import Protocol

from User import User
from UserAuthorizer import UserAuthorizer

class UserService(Protocol):
    class Error(Exception):
        pass

    async def try_get_user(self, id_: int) -> Awaitable[User]:
        pass
```

## 4.10.2.7：装饰器模式

> *装饰器模式能够在不修改类方法的情况下，增强类方法的功能。*

装饰器类包装另一个其功能将被增强的类。装饰器类实现了被包装类的接口，并在代码中替代被包装类使用。当你无法修改现有类时（例如，现有类位于第三方库中），装饰器模式非常有用。装饰器模式还有助于遵循*开闭原则*，因为你无需修改现有方法来增强其功能。你可以创建一个包含新功能的装饰器类。

以下是装饰器模式的一个示例。其中有一个标准的 SQL 语句执行器实现，以及两个装饰过的 SQL 语句执行器实现：一个添加了日志功能，另一个添加了 SQL 语句执行计时功能。最后，创建了一个双重装饰的 SQL 语句执行器，它既能记录 SQL 语句，又能计时其执行。

```python
import time
from collections.abc import Awaitable
from typing import Protocol, Any

from logger import logger
from LogLevel import LogLevel

class SqlStatementExecutor(Protocol):
    async def try_execute(
        self,
        sql_statement: str,
        parameter_values: list[Any] | None = None
    ) -> Awaitable[Any]:
        pass


class SqlStatementExecutorImpl(SqlStatementExecutor):
    # Implement __get_connection() ...

    async def try_execute(
        self,
        sql_statement: str,
        parameter_values: list[Any] | None = None
    ) -> Awaitable[Any]:
        return await self.__get_connection().execute(
            sql_statement,
            parameter_values
        )


class LoggingSqlStatementExecutor(SqlStatementExecutor):
    def __init__(self, sql_statement_executor: SqlStatementExecutor):
        self.__sql_statement_executor = sql_statement_executor

    async def try_execute(
        self,
        sql_statement: str,
        parameter_values: list[Any] | None = None
    ) -> Awaitable[Any]:

        logger.log(
            LogLevel.DEBUG,
            f'Executing SQL statement: {sql_statement}'
        )

        return await self.__sql_statement_executor.try_execute(
            sql_statement,
            parameter_values
        )


class TimingSqlStatementExecutor(SqlStatementExecutor):
    def __init__(self, sql_statement_executor: SqlStatementExecutor):
        self.__sql_statement_executor = sql_statement_executor

    async def try_execute(
        self,
        sql_statement: str,
        parameter_values: list[Any] | None = None
    ) -> Awaitable[Any]:
        start_time_in_ns = time.time_ns()

        result = await self.__sql_statement_executor.try_execute(
            sql_statement,
            parameter_values
        )

        end_time_in_ns = time.time_ns()
        duration_in_ns = end_time_in_ns - start_time_in_ns
        duration_in_ms = duration_in_ns / 1_000_000

        logger.log(
            LogLevel.DEBUG,
            f'SQL statement execution duration: {duration_in_ms} ms'
        )

        return result
```

```python
timing_and_logging_sql_statement_executor = LoggingSqlStatementExecutor(
    TimingSqlStatementExecutor(SqlStatementExecutorImpl())
)
```

在 Python 中，你也可以将装饰器模式用于函数和方法。Python 装饰器允许我们将一个函数包装起来，以扩展被包装函数的行为，而无需永久修改它。Python 装饰器是接受一个函数作为参数并返回另一个函数的函数，返回的函数用于替代被装饰的函数。让我们看一个非常简单的 Python 函数装饰器示例：

```python
# Decorator
def print_hello(func):
    def wrapped_func(*args, **kwargs):
        print('Hello')
        return func(*args, **kwargs)

    return wrapped_func

@print_hello
def add(a: int, b: int) -> int:
    return a + b

result = add(1, 2)
print(result) # Prints: Hello 3
```

让我们再看一个示例，这个装饰器会计算函数的执行时间并将其打印到控制台：

```python
import time
from functools import wraps

# Decorator
def timed(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        start_time_in_ns = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time_in_ns = time.perf_counter_ns()
        duration_in_ns = end_time_in_ns - start_time_in_ns
        print(
            f'Exec of func "{func.__name__}" took {duration_in_ns} ns'
        )
        return result

    return wrapped_func

@timed
def add(a: int, b: int) -> int:
    return a + b

result = add(1, 2)
print(result)
# Prints, for example:
# Exec of func "add" took 625 ns
# 3
```

你可以组合多个装饰器，例如：

```python
# Decorator
def logged(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        # In real-life, you use a logger here instead of print
        print(f'Func "{func.__name__}" executed')
        return result

    return wrapped_func

@logged
@timed
def add(a: int, b: int) -> int:
    return a + b

result = add(1, 2)
print(result)
# Prints, for example:
# Exec of func "add" took 583 ns
# Func "add" executed
# 3
```

如果你改变装饰器的顺序，输出的顺序也会不同。并且函数的执行时间会变长，因为日志记录所花费的时间也被计入了总执行时间：

```python
@timed
@logged
def add(a: int, b: int) -> int:
    return a + b

result = add(1, 2)
print(result)
# Prints, for example:
# Func "add" executed
# Exec of func "add" took 9708 ns
# 3
```

你也可以不使用 @-语法，而是直接使用装饰器函数来创建新函数：

```python
def add(a: int, b: int) -> int:
    return a + b

logged_add = logged(add)
timed_add = timed(add)
logged_timed_add = logged(timed(add))
timed_logged_add = timed(logged(add))

print(logged_add(1,2))
# Prints
# Func "add" executed
# 3

print(timed_add(1,2))
# Prints, for example
# Exec of func "add" took 209 ns
# 3

print(logged_timed_add(1, 2))
# Prints, for example
# Exec of func "add" took 208 ns
# Func "add" executed
# 3

print(timed_logged_add(1, 2))
# Prints, for example
# Func "add" executed
# Exec of func "add" took 1250 ns
# 3
```

## 4.10.2.8：享元模式

> 享元模式是一种节省内存的优化模式，其中享元对象会复用对象。

让我们看一个简单的游戏示例，其中在不同位置绘制不同的形状。假设游戏绘制了大量相似的形状，但位置不同，这样我们就能注意到应用此模式后内存消耗的差异。

游戏绘制的形状具有以下属性：大小、形状、填充颜色、描边颜色、描边宽度和描边样式。

```python
from typing import Protocol

class Shape(Protocol):
    # ...

# Define Color...
# Define StrokeStyle...

class AbstractShape(Shape):
    def __init__(
        self,
        fill_color: Color,
        stroke_color: Color,
        stroke_width: int,
        stroke_style: StrokeStyle
    ):
        self.__fill_color = fill_color
        self.__stroke_color = stroke_color
        self.__stroke_width = stroke_width
        self.__stroke_style = stroke_style

class CircleShape(AbstractShape):
    def __init__(
        self,
        fill_color: Color,
        stroke_color: Color,
        stroke_width: int,
        stroke_style: StrokeStyle,
        radius: int
    ):
        super().__init__(
            fill_color,
            stroke_color,
            stroke_width,
            stroke_style
        )
        self.__radius = radius

# Define LineSegment ...

class PolygonShape(AbstractShape):
    def __init__(
        self,
        fill_color: Color,
        stroke_color: Color,
```

## 4.10.3：行为型设计模式

行为型设计模式描述了如何使用面向对象设计来实现新的行为。以下章节将介绍以下行为型设计模式：

- 责任链模式
- 观察者模式
- 命令/动作模式
- 迭代器模式
- 状态模式
- 中介者模式
- 模板方法模式
- 备忘录模式
- 访问者模式
- 空对象模式

## 4.10.3.1：责任链模式

> 责任链模式允许你将请求沿着处理者链进行传递。

当收到请求时，每个处理者可以决定如何处理：

- 处理请求，然后将其传递给链中的下一个处理者
- 处理请求，但不将其传递给后续处理者（终止链）
- 不处理请求，将其传递给下一个处理者

FastAPI Web框架利用责任链模式来处理请求。在FastAPI框架中，你可以使用*中间件*编写可插拔的行为，这个概念类似于Java中的Servlet过滤器。下面是一个中间件示例，它将HTTP请求处理时间添加到自定义HTTP响应头中：

```python
import time

from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware('http')
async def add_request_processing_time_header(request: Request, call_next):
    start_time_in_ns = time.time_ns()
    response = await call_next(request)
    end_time_in_ns = time.time_ns()
    processing_time_in_ns = end_time_in_ns - start_time_in_ns
    processing_time_in_ms = processing_time_in_ns / 1_000_000
    response.headers["X-Processing-Time-Millis"] = str(processing_time_in_ms)
    return response
```

下面是一个授权和日志记录中间件示例：

```python
from fastapi import FastAPI, Response, Request

app = FastAPI()

# 授权中间件
@app.middleware('http')
async def authorize(request: Request, call_next):
    # 从请求的'Authorization'头中，
    # 提取承载JWT（如果存在）
    # 设置'token_is_present'变量值
    # 验证JWT的有效性并将结果
    # 赋值给'token_is_valid'变量

    if token_is_valid:
        response = await call_next(request)
    elif token_is_present:
        # 注意！未调用call_next，
        # 这将终止请求
        response = Response('Unauthorized', 403)
    else:
        # 注意！未调用call_next，
        # 这将终止请求
        response = Response('Unauthenticated', 401)

    return response

# 日志记录中间件
@app.middleware('http')
async def log(request: Request, call_next):
    print(f'GET {str(request.url)}')
    return await call_next(request)

@app.get('/hello')
def hello():
    return 'Hello!'
```

## 4.10.3.2：观察者模式

> 观察者模式允许你定义一个观察-通知（或发布-订阅）机制，以将发生在被观察对象上的事件通知给一个或多个对象。

使用观察者模式的一个典型例子是UI视图观察模型。每当模型发生变化时，UI视图都会收到通知并可以重绘自身。让我们看一个例子：

```python
from typing import Protocol

class Observer(Protocol):
    def notify_about_change(self) -> None:
        pass

class Observable(Protocol):
    def observe_by(self, observer: Observer) -> None:
        pass

class ObservableImpl(Observable):
    __observers: list[Observer]

    def __init__(self):
        self.__observers = []

    def observe_by(self, observer: Observer):
        self.__observers.append(observer)

    def _notify_observers(self) -> None:
        for observer in self.__observers:
            observer.notify_about_change()

# 定义 Todo ...

class TodosModel(ObservableImpl):
    __todos: list[Todo]

    def __init__(self):
        super().__init__()
        self.__todos = []

    def add_todo(self, todo: Todo) -> None:
        self.__todos.append(todo)
        self._notify_observers()

    def remove_todo(self, todo: Todo) -> None:
        self.__todos.remove(todo)
        self._notify_observers()

class TodosView(Observer):
    def __init__(self, todos_model: TodosModel):
        self.__todos_model = todos_model
        todos_model.observe_by(self)

    def notify_about_change(self) -> None:
        # 当todos模型变化时将被调用
        self.render()

    def render(self) -> None:
        # 渲染todos ...
```

让我们再看一个利用发布-订阅模式的例子。下面我们定义一个MessageBroker类，它包含以下方法：publish、subscribe和unsubscribe。

```python
from collections.abc import Callable
from typing import Protocol, TypeVar

T = TypeVar('T')

class MessagePublisher(Protocol[T]):
    def publish(self, topic: str, message: T) -> None:
        pass

class MessageSubscriber(Protocol[T]):
    def subscribe(
        self,
        topic: str,
        handle_message: Callable[[T], None]
    ) -> None:
        pass

MessageHandlers = list[Callable[[T], None]]

class MessageBroker(MessagePublisher[T], MessageSubscriber[T]):
    __topic_to_handle_msgs_map: dict[str, MessageHandlers]

    def __init__(self):
        self.__topic_to_handle_msgs_map = {}

    def publish(self, topic: str, message: T) -> None:
        handle_messages = self.__topic_to_handle_msgs_map.get(topic)

        if handle_messages is not None:
            for handle_message in handle_messages:
                handle_message(message)

    def subscribe(
        self,
        topic: str,
        handle_message: Callable[[T], None]
    ) -> None:
        handle_messages = self.__topic_to_handle_msgs_map.get(topic)

        if handle_messages is None:
            self.__topic_to_handle_msgs_map[topic] = [handle_message]
        else:
            handle_messages.append(handle_message)

    def unsubscribe(
        self,
        topic: str,
        handle_message: Callable[[T], None]
    ) -> None:
        handle_messages = self.__topic_to_handle_msgs_map.get(topic)

        if handle_messages is not None:
            handle_messages.remove(handle_message)
```

```python
message_broker: MessageBroker[str] = MessageBroker()

def print_message(message: str):
    print(message)

topic = 'test'
message_broker.subscribe(topic, print_message)
message_broker.publish(topic, 'Hi!')
message_broker.unsubscribe(topic, print_message)
message_broker.publish('test', 'Hi!')
```

## 4.10.3.3：命令/动作模式

> 命令或动作模式用于将命令或动作定义为对象，这些对象可以作为参数传递给其他函数以供后续执行。

让我们创建一个简单的动作/命令协议：

```python
from typing import Protocol

class Action(Protocol):
    def perform(self) -> None:
        pass

class Command(Protocol):
    def execute(self) -> None:
        pass
```

让我们创建一个简单的具体动作/命令，用于打印消息：

```python
class PrintAction(Action):
    def __init__(self, message: str):
        self.__message = message

    def perform(self) -> None:
        print(self.__message)

class PrintCommand(Command):
    def __init__(self, message: str):
        self.__message = message

    def execute(self) -> None:
        print(self.__message)
```

如上所示，上述 `PrintAction` 和 `PrintCommand` 实例封装了在执行动作/命令时（通常比动作/命令实例创建时要晚）所使用的状态。

现在我们可以使用我们的打印动作/命令了：

```
actions = [PrintAction('Hello'), PrintAction('World')]
for action in actions:
    action.perform()

commands = [PrintCommand('Hello'), PrintCommand('World')]
for command in commands:
    command.execute()
```

只要动作/命令是可撤销的，动作和命令就可以被设计成可撤销的。上述打印动作/命令是不可撤销的，因为你无法撤销对控制台的打印操作。让我们引入一个可撤销的动作：向列表中添加项目。这是一个可以通过从列表中移除该项目来撤销的动作。

```
from typing import Generic, TypeVar

T = TypeVar('T')

class AddToListAction(Action, Generic[T]):
    def __init__(self, item: T, items: list[T]):
        self.__item = item
        self.__items = items

    def perform(self) -> None:
        self.__items.append(self.__item)

    def undo(self) -> None:
        self.__items.remove(self.__item)

values = [1, 2]
add3ToValuesAction = AddToListAction(3, values)
add3ToValuesAction.perform()
print(values) # Prints [1, 2, 3]
add3ToValuesAction.undo()
print(values) # Prints [1, 2]
```

## 4.10.3.4: 迭代器模式

*迭代器模式可用于为序列类添加迭代能力。*

让我们为 Python 的 `list` 类创建一个反向迭代器。我们通过提供 `next` 方法的实现来实现 `Iterator` 抽象基类：

```
from collections.abc import Iterator
from typing import TypeVar

T = TypeVar('T')
```

```
class ReverseListIterator(Iterator[T]):
    def __init__(self, values: list[T]):
        self.__values = values.copy()
        self.__position = len(values) - 1

    def __next__(self) -> T:
        if self.__position < 0:
            raise StopIteration()
        next_value = self.__values[self.__position]
        self.__position -= 1
        return next_value

    def __iter__(self) -> Iterator[T]:
        return self
```

我们可以将 ReverseListIterator 类放入下面定义的 ReverseArrayList 类中使用：

```
class ReverseList(list[T]):
    def __iter__(self) -> Iterator[T]:
        return ReverseListIterator(self)
```

现在我们可以使用新的迭代器来反向遍历列表：

```
reversed_numbers = ReverseList([1, 2, 3, 4, 5])

for number in reversed_numbers:
    print(number)

// Prints:
// 5
// 4
// 3
// 2
// 1
```

## 4.10.3.5: 状态模式

> 状态模式允许一个对象根据其当前状态改变其行为。

开发者通常不将对象的状态视为一个对象，而是视为一个枚举值（enum），例如。下面是一个例子，我们定义了一个 UserStory 类，代表一个可以在屏幕上渲染的用户故事。一个枚举值代表一个 UserStory 对象的状态。

```
from enum import Enum
from typing import Protocol

class UserStoryState(Enum):
    TODO = 1
    IN_DEVELOPMENT = 2
    IN_VERIFICATION = 3
    READY_FOR_REVIEW = 4
    DONE = 5

class Icon(Protocol):
    # ...

class TodoIcon(Icon):
    # ...

# Define rest of icons ...

class UserStory:
    def __init__(self, name: str):
        self.__name = name
        self.__state = UserStoryState.TODO

    def set_state(self, state: UserStoryState) -> None:
        self.__state = state

    def render(self) -> None:
        match self.__state:
            case UserStoryState.TODO:
                icon = TodoIcon()
            case UserStoryState.IN_DEVELOPMENT:
                icon = InDevelopmentIcon()
            case UserStoryState.IN_VERIFICATION:
                icon = InVerificationIcon()
            case UserStoryState.READY_FOR_REVIEW:
                icon = ReadyForReviewIcon()
            case UserStoryState.DONE:
                icon = DoneIcon()
            case _:
                raise ValueError('Invalid user story state')

        # Draw a UI element on screen representing the user story
        # using the above assigned 'icon'
```

上述解决方案不是面向对象的。我们应该用多态设计来替换条件语句（switch-case 语句）。这可以通过引入状态对象来实现。在状态模式中，对象的状态由一个对象表示，而不是一个枚举值。下面是修改后使用状态模式的代码：

```
from typing import Protocol

# Import Icon classes ...

class UserStoryState(Protocol):
    @property
    def icon(self) -> Icon:
        pass

class TodoUserStoryState(UserStoryState):
    @property
    def icon(self) -> Icon:
        return TodoIcon()

class InDevelopmentUserStoryState(UserStoryState):
    @property
    def icon(self) -> Icon:
        return InDevelopmentIcon()

class InVerificationUserStoryState(UserStoryState):
    @property
    def icon(self) -> Icon:
        return InVerificationIcon()

class ReadyForReviewUserStoryState(UserStoryState):
    @property
    def icon(self) -> Icon:
        return ReadyForReviewIcon()

class DoneUserStoryState(UserStoryState):
    @property
    def icon(self) -> Icon:
        return DoneIcon()

class UserStory:
    def __init__(self, name: str):
        self.__name = name
        self.__state = TodoUserStoryState()

    def set_state(self, state: UserStoryState) -> None:
        self.__state = state

    def render(self) -> None:
        icon = self.__state.icon
        # Draw a UI element on screen representing
        # the user story using the given 'icon'
```

让我们再看一个 Order 类的例子。一个订单可以有状态，比如已支付、已打包、已交付等。下面我们把订单状态实现为类：

```
from typing import Protocol

from Customer import Customer
from EmailService import EmailService

class OrderState(Protocol):
    def create_message(self, order_id: str) -> str:
        pass

class PaidOrderState(OrderState):
    def create_message(self, order_id: str) -> str:
        return 'Order ' + order_id + ' is successfully paid'

class DeliveredOrderState(OrderState):
    def create_message(self, order_id: str) -> str:
        return 'Order ' + order_id + ' is delivered'

# Implement the rest of possible order states ...

class Order:
    def __init__(self, id: str, state: OrderState, customer: Customer):
        self.__id = id
        self.__state = state
        self.__customer = customer

    @property
    def customer_email_address(self) -> str:
        return self.__customer.email_address

    @property
    def state_message(self) -> str:
        return self.__state.create_message(self.__id)

email_service = EmailService(...)
order = Order(...)

email_service.send_email(
    order.customer_email_address,
    order.state_message
)
```

## 4.10.3.6: 中介者模式

> 中介者模式可以减少对象之间的依赖。它限制两个不同对象层之间的直接通信，并强制它们仅通过中介者对象进行协作。

中介者模式消除了两个不同对象层之间的耦合。因此，对一个对象层的修改无需改变另一个层中的对象。

中介者模式的一个典型例子是模型-视图-控制器（MVC）模式。在 MVC 模式中，模型和视图对象不直接通信，而是仅通过中介者对象（控制器）进行通信。接下来，将介绍在前端客户端中使用 MVC 模式的几种不同方式。传统上，MVC 模式用于后端，当后端也生成要在客户端设备（Web 浏览器）中显示的视图时。随着单页 Web 客户端的出现，现代后端是一个仅包含模型和控制器（MC）的简单 API。

![](img/cbd069395d7b824346b69b1f92e0fb4a_218_0.png)

在下图中，你可以看到如何使用依赖倒置，并且没有实现类依赖于具体实现。你可以轻松地将任何实现类更改为另一个，而无需修改任何其他实现类。注意 *ControllerImpl* 类如何使用*桥接模式*并实现两个桥接，一个指向模型，另一个指向视图。

![](img/cbd069395d7b824346b69b1f92e0fb4a_218_1.png)

如下图所示，控制器也可以用作桥接适配器：控制器可以被修改以适应视图层的变化（使用 View2 而不是 View），而无需更改模型层。修改后的模块在图中以灰色背景显示。类似地，控制器可以被修改以适应模型层的变化，而无需更改视图层（图中未显示）。

以下示例使用了MVC模式的一种特化形式，称为**模型-视图-呈现器**（MVP）。在MVP模式中，控制器被称为呈现器。不过，在所有示例中，我使用了更通用的术语*控制器*。呈现器充当视图和模型之间的中间人。呈现器类型的控制器对象持有对视图对象和模型对象的引用。视图对象命令呈现器对模型执行操作。而模型对象则请求呈现器更新视图对象。

让我们以一个简单的待办事项应用为例。首先，我们实现`Todo`类，它是模型的一部分。

**图 4.41. Todo.py**

```python
class Todo:
    def __init__(self, id: int, name: str, is_done: bool):
        self._id = id
        self.__name = name
        self.__is_done = is_done

    @property
    def id(self) -> int:
        return self.__id

    @id.setter
    def id(self, id_: int) -> None:
        self._id = id_

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str) -> None:
        self.__name = name

    @property
    def is_done(self) -> bool:
        return self.__is_done

    @is_done.setter
    def is_done(self, is_done: bool) -> None:
        self.__is_done = is_done
```

接下来，我们实现视图层的协议：

**图 4.42. TodoView.py**

```python
from typing import Protocol

from Todo import Todo

class TodoView(Protocol):
    def show_todos(self, todos: list[Todo]) -> None:
        pass

    def show_error_message(self, error_message: str) -> None:
        pass
```

以下是视图层的实现：

**图 4.43. TodoViewImpl.py**

```python
from Todo import Todo
from TodoController import TodoController
from TodoView import TodoView

class TodoViewImpl(TodoView):
    def __init__(self, controller: TodoController):
        self.__controller = controller
        controller.view = self
        controller.start_fetch_todos()

    def show_todos(self, todos: list[Todo]) -> None:
        # 更新视图以显示给定的待办事项
        # 为每个待办事项复选框添加监听器
        # 监听器调用: self.__controller.toggle_todo_done(todo.id)

    def show_error_message(self, error_message: str) -> None:
        # 更新视图以显示错误消息
```

然后我们实现一个通用的`Controller`类，作为具体控制器的基类：

**图 4.44. Controller.py**

```python
from typing import Generic, TypeVar

TModel = TypeVar('TModel')
TView = TypeVar('TView')

class Controller(Generic[TModel, TView]):
    __model: TModel | None = None
    __view: TView | None = None

    @property
    def model(self) -> TModel | None:
        return self.__model

    @model.setter
    def model(self, model: TModel) -> None:
        self.__model = model

    @property
    def view(self) -> TView | None:
        return self.__view

    @view.setter
    def view(self, view: TView) -> None:
        self.__view = view
```

下面的`TodoControllerImpl`类实现了两个操作：`start_fetch_todos`和`toggle_todo_done`，它们委托给模型层。它还实现了两个操作：`update_view_with_todos`和`update_view_with_error_message`，它们委托给视图层。

**图 4.45. TodoController.py**

```python
from typing import Protocol

from Todo import Todo

class TodoController(Protocol):
    async def start_fetch_todos(self) -> None:
        pass

    async def toggle_todo_done(self, id: int) -> None:
        pass

    def update_view_with_todos(self, todos: list[Todo]) -> None:
        pass

    def update_view_with_error_message(self, error_message: str) -> None:
        pass
```

**图 4.46. TodoControllerImpl.py**

```python
from Controller import Controller
from Todo import Todo
from TodoController import TodoController
from TodoModel import TodoModel
from TodoView import TodoView

class TodoControllerImpl(Controller[TodoModel, TodoView], TodoController):
    async def start_fetch_todos(self) -> None:
        if self.model is not None:
            await self.model.fetch_todos()

    async def toggle_todo_done(self, id_: int) -> None:
        if self.model is not None:
            await self.model.toggle_todo_done(id_)

    def update_view_with_todos(self, todos: list[Todo]) -> None:
        if self.view is not None:
            self.view.show_todos(todos)

    def update_view_with_error_message(self, error_message: str) -> None:
        if self.view is not None:
            self.view.show_error_message(error_message)
```

下面的`TodoModelImpl`类使用提供的`todo_service`实现了待办事项的获取（`fetch_todos`）。`todo_service`访问后端，例如从数据库中读取待办事项。当待办事项成功获取后，控制器会被告知更新视图。如果获取待办事项失败，则更新视图以显示错误。切换待办事项完成状态是使用`todo_service`及其`try_update_todo`方法实现的。

**图 4.47. TodoService.py**

```python
from collections.abc import Awaitable
from typing import Protocol

from Todo import Todo

class TodoService(Protocol):
    class Error(Exception):
        # ...

    async def try_get_todos(self) -> Awaitable[list[Todo]]:
        pass

    async def try_update_todo(self, todo: Todo) -> None:
        pass
```

**图 4.48. TodoModel.py**

```python
from typing import Protocol

class TodoModel(Protocol):
    async def fetch_todos(self) -> None:
        pass

    async def toggle_todo_done(self, id_: int) -> None:
        pass
```

**图 4.49. TodoModelImpl.py**

```python
from Todo import Todo
from TodoController import TodoController
from TodoModel import TodoModel
from TodoService import TodoService

class TodoModelImpl(TodoModel):
    __todos: list[Todo]

    def __init__(
        self,
        controller: TodoController,
        todo_service: TodoService
    ):
        self.__controller = controller
        controller.model = self
        self.__todo_service = todo_service
        self.__todos = []

    async def fetch_todos(self) -> None:
        try:
            self.__todos = await self.__todo_service.try_get_todos()
        except TodoService.Error as error:
            self.__controller.update_view_with_error_message(error.message)
        else:
            self.__controller.update_view_with_todos(self.__todos)

    async def toggle_todo_done(self, id_: int) -> None:
        found_todos = [ todo for todo in self.__todos if todo.id == id_ ]
        todo = found_todos[0] if len(found_todos) else None

        if todo:
            todo.is_done = not todo.is_done

            try:
                await self.__todo_service.try_update_todo(todo)
            except TodoService.Error as error:
                self.__controller.update_view_with_error_message(error.message)
```

让我们打破所有示例都使用Python的惯例，使用Web Components来实现上述示例。如果你不是全栈Python开发者，可以跳过本节的剩余部分，因为它涉及前端相关的TypeScript代码。Web组件视图应扩展`HTMLElement`类。

视图的`connectedCallback`方法将在组件挂载时被调用。它开始获取待办事项。`showTodos`方法将给定的待办事项渲染为HTML元素。它还为*标记完成*按钮添加了事件监听器。`showError`方法更新视图的内部HTML以显示错误消息。

**图 4.50. Todo.ts**

```typescript
export type Todo = {
  id: number;
  name: string;
  isDone: boolean;
};
```

**图 4.51. TodoView.ts**

```typescript
interface TodoView {
  showTodos(todos: Todo[]): void;
  showError(errorMessage: string): void;
}
```

**图 4.52. TodoViewImpl.ts**

```typescript
import controller from './todoController';
import { Todo } from './Todo';

export default class TodoViewImpl
    extends HTMLElement implements TodoView {
  constructor() {
    super();
    controller.setView(this);
  }

  connectedCallback() {
    controller.startFetchTodos();
    this.innerHTML = '<div>Loading todos...</div>';
  }

  showTodos(todos: Todo[]) {
    const todoElements = todos.map(({ id, name, isDone }) => `
      <li id="todo-${id}">
        ${id}&nbsp;${name}&nbsp;
        ${isDone ? '' : '<button>Mark done</button>'}
      </li>
    `);

    this.innerHTML = `<ul>${todoElements}</ul>`;

    todos.map(({ id }) => this
      .querySelector(`#todo-${id} button`)?
      .addEventListener('click',
        () => controller.toggleTodoDone(id)));
  }

  showError(message: string) {
    this.innerHTML = `
```

## 图 4.53. Controller.ts

```typescript
export default class Controller<TModel, TView> {
  private model: TModel | undefined;
  private view: TView | undefined;

  getModel(): TModel | undefined {
    return this.model;
  }

  setModel(model: TModel): void {
    this.model = model;
  }

  getView(): TView | undefined {
    return this.view;
  }

  setView(view: TView): void {
    this.view = view;
  }
}
```

## 图 4.54. TodoController.ts

```typescript
import { Todo } from "./Todo";

export interface TodoController {
  startFetchTodos(): void;
  toggleTodoDone(id: number): void;
  updateViewWithTodos(todos: Todo[]): void;
  updateViewWithError(message: string): void;
}
```

## 图 4.55. todoController.ts

```typescript
import TodoView from './TodoView';
import Controller from "./Controller";
import { TodoController } from './TodoController';
import { Todo } from "./Todo";
import TodoModel from './TodoModel';

class TodoControllerImpl
    extends Controller<TodoModel, TodoView>
    implements TodoController {

    startFetchTodos(): void {
        this.getModel()?.fetchTodos();
    }

    toggleTodoDone(id: number): void {
        this.getModel()?.toggleTodoDone(id);
    }

    updateViewWithTodos(todos: Todo[]): void {
        this.getView()?.showTodos(todos);
    }

    updateViewWithError(message: string): void {
        this.getView()?.showError(message);
    }
}

const controller = new TodoControllerImpl();
export default controller;
```

## 图 4.56. TodoService.ts

```typescript
export interface TodoService {
    getTodos(): Promise<Todo[]>;
    updateTodo(todo: Todo): Promise<void>;
}
```

## 图 4.57. TodoModel.ts

```typescript
export interface TodoModel {
    fetchTodos(): void;
    toggleTodoDone(id: number): void;
}
```

## 图 4.58. TodoModelImpl.ts

```typescript
import controller, { TodoController } from './todoController';
import { TodoModel } from './TodoModel';
import { Todo } from "./Todo";

export default class TodoModelImpl implements TodoModel {
  private todos: Todo[] = [];

  constructor(
    private readonly controller: TodoController,
    private readonly todoService: TodoService
  ) {
    controller.setModel(this);
  }

  fetchTodos(): void {
    this.todoService.getTodos()
      .then((todos) => {
        this.todos = todos;
        controller.updateViewWithTodos(todos);
      })
      .catch((error) =>
        controller.updateViewWithError(error.message));
  }

  toggleTodoDone(id: number): void {
    const foundTodo = this.todos.find(todo => todo.id === id);

    if (foundTodo) {
      foundTodo.isDone = !foundTodo.isDone;
      this.todoService
          .updateTodo(foundTodo)
          .catch((error: any) =>
              controller.updateViewWithError(error.message));
    }
  }
}
```

我们可以将上述定义的控制器和模型与 React 视图组件一起使用，如下所示：

## 图 4.59. ReactTodoView.tsx

```tsx
// ...
import controller from './todoController';

// ...

export default class ReactTodoView
        extends Component<Props, State>
        implements TodoView {

  constructor(props: Props) {
    super(props);
    controller.setView(this);

    this.state = {
      todos: []
    }
  }

  componentDidMount() {
    controller.startFetchTodos();
  }

  showTodos(todos: Todo[]) {
    this.setState({ ...this.state, todos });
  }

  showError(errorMessage: string) {
    this.setState({ ...this.state, errorMessage });
  }

  render() {
    // 在此处渲染 'this.state.todos' 中的待办事项
    // 或在此处显示 'this.state.errorMessage'
  }
}
```

如果你有多个视图使用同一个控制器，你可以从下面定义的 MultiViewController 类派生你的控制器：

## 图 4.60. MultiViewController.ts

```typescript
export default class MultiViewController<TModel, TView> {
  private model: TModel | undefined;
  private views: TView[] = [];

  getModel(): TModel | undefined {
    return this.model;
  }

  setModel(model: TModel): void {
    this.model = model;
  }

  getViews(): TView[] {
    return this.views;
  }

  addView(view: TView): void {
    this.views.push(view);
  }
}
```

假设我们想要为待办事项设置两个视图，一个用于显示实际的待办事项，另一个用于查看待办事项计数。我们需要稍微修改控制器以支持多个视图：

## 图 4.61. TodoControllerImpl.ts

```typescript
import TodoView from './TodoView';
import MultiViewController from './MultiViewController';
import { Todo } from "./Todo";
import { TodoController } from './TodoController';
import TodoModel from './TodoModel';

class TodoControllerImpl
    extends MultiViewController<TodoModel, TodoView>
    implements TodoController {
    startFetchTodos(): void {
        this.getModel()?.fetchTodos();
    }

    toggleTodoDone(id: number): void {
        this.getModel()?.toggleTodoDone(id);
    }

    updateViewsWithTodos(todos: Todo[]): void {
        this.getViews().forEach(view => view.showTodos(todos));
    }

    updateViewWithError(message: string): void {
        this.getViews().forEach(view => view.showError(message));
    }
}

const controller = new TodoControllerImpl();
export default controller;
```

许多现代 UI 框架和状态管理库实现了 MVC 模式的一种特化形式，称为 Model-View-ViewModel（MVVM）。在 MVVM 模式中，控制器被称为视图模型。不过，在下面的例子中，我使用了更通用的术语 *controller*。视图模型与 MVP 模式中的呈现器的主要区别在于，在 MVP 模式中，呈现器持有对视图的引用，而视图模型则没有。视图模型提供了视图事件与模型中操作之间的绑定。这可以通过视图模型将操作分发函数作为视图的属性来实现。而在另一个方向上，视图模型将模型的状态映射到视图的属性。例如，在使用 React 和 Redux 时，你可以使用 `mapDispatchToProps` 函数将视图连接到模型，并使用 `mapStateToProps` 函数将模型连接到视图。这两个映射函数构成了将视图和模型绑定在一起的视图模型（或控制器）。

让我们首先用 React 和 Redux 实现待办事项示例，然后展示如何将 React 视图替换为 Angular 视图，而无需修改控制器或模型层。

让我们为待办事项实现一个列表视图：

## 图 4.62. TodosListView.tsx

```tsx
import { connect } from 'react-redux';
import { useEffect } from "react";
import { controller, ActionDispatchers, State }
  from './todosController';

type Props = ActionDispatchers & State;

function TodosListView({
  toggleTodoDone,
  startFetchTodos,
  todos
}: Props) {

  useEffect(() => {
    startFetchTodos();
  }, [startFetchTodos]);

  const todoElements = todos.map(({ id, name, isDone }) => (
    <li key={id}>
      {id}&nbsp;
      {name}&nbsp;
      {isDone
        ? undefined
        : <button onClick={() => toggleTodoDone(id)}>
            标记完成
          </button>
      }
    </li>
  ));

  return <ul>{todoElements}</ul>;
}

// 在这里我们使用控制器将视图连接到模型
export default connect(
  controller.getState,
  () => controller.actionDispatchers
)(TodosListView);
```

以下是控制器的基类 Controller：

## 图 4.63. AbstractAction.ts

```typescript
export default abstract class AbstractAction<S> {
  abstract perform(state: S): S;
}
```

## 图 4.64. Controller.ts

```typescript
import AbstractAction from "./AbstractAction";

export type ReduxDispatch =
  (reduxActionObject: { type: AbstractAction<any> }) => void;

export default class Controller {
  protected readonly dispatch:
    (action: AbstractAction<any>) => void;

  constructor(reduxDispatch: ReduxDispatch) {
    this.dispatch = (action: AbstractAction<any>) =>
      reduxDispatch({ type: action });
  }
}
```

以下是待办事项的控制器：

## 图 4.65. todosController.ts

```typescript
import store from './store';
import { AppState } from "./AppState";
import ToggleDoneTodoAction from "./ToggleDoneTodoAction";
import StartFetchTodosAction from "./StartFetchTodosAction";
import Controller from "./Controller";

class TodosController extends Controller {
  readonly actionDispatchers = {
    toggleTodoDone: (id: number) =>
      this.dispatch(new ToggleDoneTodoAction(id)),

    startFetchTodos: () =>
      this.dispatch(new StartFetchTodosAction())
  }

  getState(appState: AppState) {
    return {
      todos: appState.todosState.todos,
    }
  }
}

export const controller = new TodosController(store.dispatch);
export type State = ReturnType<typeof controller.getState>;
export type ActionDispatchers = typeof controller.actionDispatchers;
```

在开发阶段，我们可以使用 StartFetchTodosAction 类的以下临时实现：

## 图 4.66. initialTodosState.ts

```typescript
import { Todo } from './Todo';

export type TodoState = {
  todos: Todo[];
}

const initialTodosState = {
  todos: []
} as TodoState

export default initialTodosState;
```

## 图 4.67. AbstractTodoAction.ts

```typescript
import AbstractAction from './AbstractAction';

export default abstract class AbstractTodoAction extends
  AbstractAction<TodoState> {}
```

## 图 4.68. StartFetchTodosAction.ts

```typescript
import { TodoState } from "./TodoState";
import AbstractTodoAction from "./AbstractTodoAction";

export default class StartFetchTodosAction extends
  AbstractTodoAction {
  perform(state: TodoState): TodoState {
    return {
      todos: [
        {
          id: 1,
          name: "Todo 1",
          isDone: false,
        },
        {
          id: 2,
          name: "Todo 2",
          isDone: false,
        },
      ],
    };
  }
}
```

## 图 4.69. MarkDoneTodoAction.ts

```typescript
import AbstractTodoAction from './AbstractTodoAction';

export default class MarkDoneTodoAction extends AbstractTodoAction {
  // ...
}
```

现在我们可以为待办事项引入一个新的视图，一个 TodosTableView，它可以使用与 TodosListView 相同的控制器。

## 图 4.70. TodosTableView.tsx

```tsx
import { connect } from 'react-redux';
import { useEffect } from "react";
import { controller, ActionDispatchers, State }
  from './todosController';

type Props = ActionDispatchers & State;

function TodosTableView({
  toggleTodoDone,
  startFetchTodos,
  todos
}: Props) {
  useEffect(() => {
    startFetchTodos();
  }, [startFetchTodos]);

  const todoElements = todos.map(({ id, isDone, name }) => (
    <tr key={id}>
      <td>{id}</td>
      <td>{name}</td>
      <td>
        <input
          type="checkbox"
          checked={isDone}
          onChange={() => toggleTodoDone(id)}
        />
      </td>
    </tr>
  ));

  return <table><tbody>{todoElements}</tbody></table>;
}

export default connect(
  controller.getState,
  () => controller.actionDispatchers
)(TodosTableView);
```

我们可以注意到 TodosListView 和 TodosTableView 组件中存在一些重复。例如，两者都使用了相同的效果。我们可以创建一个 TodosView，为其提供单个待办事项视图的类型作为参数，可以是列表项视图或表格行视图：

## 图 4.71. TodosView.tsx

```tsx
import { useEffect } from "react";
import { connect } from "react-redux";
import ListItemTodoView from './ListItemTodoView';
import TableRowTodoView from './TableRowTodoView';
import { controller, ActionDispatchers, State }
  from './todosController';

type Props = ActionDispatchers & State & {
  TodoView: typeof ListItemTodoView | typeof TableRowTodoView;
};

function TodosView({
  toggleTodoDone,
  startFetchTodos,
  todos,
  TodoView
}: Props) {
  useEffect(() => {
    startFetchTodos()
  }, [startFetchTodos]);

  const todoViews = todos.map((todo) =>
    <TodoView
      key={todo.id}
      todo={todo}
      toggleTodoDone={toggleTodoDone}
    />
  );

  return TodoView === ListItemTodoView
      ? <ul>{todoViews}</ul>
      : <table><tbody>{todoViews}</tbody></table>;
}

export default connect(
  controller.getState,
  () => controller.actionDispatchers
)(TodosView);
```

以下是将单个待办事项显示为列表项的视图：

## 图 4.72. TodoViewProps.ts

```typescript
import { Todo } from "./Todo";

export type TodoViewProps = {
  toggleTodoDone: (id: number) => void,
  todo: Todo
}
```

## 图 4.73. ListItemTodoView.tsx

```tsx
import { TodoViewProps } from './TodoViewProps';

export default function ListItemTodoView({
  toggleTodoDone,
  todo: { id, name, isDone }
}: TodoViewProps) {
  return (
    <li>
      {id}&nbsp;
      {name}&nbsp;
      { isDone ?
        undefined :
        <button onClick={() => toggleTodoDone(id)}>
          Mark done
        </button> }
    </li>
  );
}
```

以下是将单个待办事项显示为表格行的视图：

## 图 4.74. TableRowTodoView.tsx

```tsx
import { TodoViewProps } from './TodoViewProps';

export default function TableRowTodoView({
  toggleTodoDone,
  todo: { id, name, isDone }
}: TodoViewProps) {
  return (
    <tr>
      <td>{id}</td>
      <td>{name}</td>
      <td>
        <input
          type="checkbox"
          checked={isDone}
          onChange={() => toggleTodoDone(id)}
        />
      </td>
    </tr>);
}
```

![](img/cbd069395d7b824346b69b1f92e0fb4a_236_0.png)

## 图 4.75. 使用 Redux 的前端 MVC 架构

![](img/cbd069395d7b824346b69b1f92e0fb4a_237_0.png)

在大多数情况下，即使状态仅用于特定视图，也不应将其存储在视图中。相反，当你将其存储在模型中时，会带来以下好处：

- 可以轻松地将状态持久化到浏览器或后端
- 可以轻松实现撤销操作
- 如果需要，状态可以轻松地与其他视图共享
- 将视图迁移到使用不同的视图技术更加直接
- 更容易调试与状态相关的问题，例如使用 Redux DevTools 浏览器扩展

我们也可以将视图实现从 React 更改为 Angular，而无需修改控制器或模型层。这可以通过例如使用 @angular-redux2/store 库来实现。下面是一个作为 Angular 组件实现的待办事项表格视图：

## 图 4.77. todos-table-view.component.ts

```typescript
import { Component, OnInit } from "@angular/core";
import { NgRedux, Select } from '@angular-redux2/store';
import { Observable } from "rxjs";
import { controller } from './todosController';
import { TodoState } from "./TodoState";
import { AppState } from "./AppState";

const { startFetchTodos,
        toggleTodoDone } = controller.actionDispatchers;

@Component({
  selector: 'todos-table-view',
  template: `
    <table>
      <tr *ngFor="let todo of (todoState | async)?.todos">
        <td>{{ todo.id }}</td>
        <td>{{ todo.name }}</td>
        <td>
          <input
            type="checkbox"
            [checked]="todo.isDone"
            (change)="toggleTodoDone(todo.id)"
          />
        </td>
      </tr>
    </table>
  `
})
export class TodosTableView implements OnInit {
  @Select(controller.getState) todoState: Observable<TodoState>;

  constructor(private ngRedux: NgRedux<AppState>) {}

  ngOnInit(): void {
    startFetchTodos();
  }

  toggleTodoDone(id: number) {
    toggleTodoDone(id);
  }
}
```

## 图 4.78. app.component.ts

```typescript
import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  template: `
    <div>
      <todos-table-view></todos-table-view>
    </div>`,
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'angular-test';
}
```

## 图 4.79. app.module.ts

```typescript
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { NgReduxModule, NgRedux } from '@angular-redux2/store';

import { AppComponent } from './app.component';
import store from './store';
import { AppState } from "./AppState";
import { TodosTableView } from "./todos-table-view.component";

@NgModule({
  declarations: [
    AppComponent, TodosTableView
  ],
  imports: [
    BrowserModule,
    NgReduxModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {
  constructor(ngRedux: NgRedux<AppState>) {
    ngRedux.provideStore(store);
  }
}
```

### 4.10.3.7：模板方法模式

> 模板方法模式允许你在基类中定义一个模板方法，子类定义该方法的最终实现。模板方法包含一个或多个对子类中实现的抽象方法的调用。

在下面的示例中，AbstractDrawing 类包含一个模板方法 draw。该方法包括对 get_shape_renderer 方法的调用，这是一个在 AbstractDrawing 类的子类中实现的抽象方法。draw 方法是一个模板方法，子类定义如何绘制单个形状。

```python
from abc import abstractmethod
from typing import Protocol

from Shape import Shape
from ShapeRenderer import ShapeRenderer

class Drawing(Protocol):
    def get_shape_renderer(self) -> ShapeRenderer:
        pass

    def draw(self) -> None:
        pass

class AbstractDrawing(Drawing):
    def __init__(self, shapes: list[Shape]):
        self.__shapes = shapes

    @abstractmethod
    def get_shape_renderer(self) -> ShapeRenderer:
        pass

    def draw(self) -> None:
        for shape in self.__shapes:
            shape.render(self.get_shape_renderer())
```

我们现在可以实现 AbstractDrawing 类的两个子类，它们定义了模板化 draw 方法的最终行为。

```python
from AbstractDrawing import AbstractDrawing
from Canvas import Canvas
from RasterShapeRenderer import RasterShapeRenderer
from Shape import Shape
from ShapeRenderer import ShapeRenderer
from SvgElement import SvgElement
from VectorShapeRenderer import VectorShapeRenderer

class RasterDrawing(AbstractDrawing):
    def __init__(self, shapes: list[Shape]):
        super().__init__(shapes)
        canvas = Canvas()
        self.__shape_renderer = RasterShapeRenderer(canvas)

    def get_shape_renderer(self) -> ShapeRenderer:
        return self.__shape_renderer

class VectorDrawing(AbstractDrawing):
    def __init__(self, shapes: list[Shape]):
        super().__init__(shapes)
        svg_root = SvgElement()
        self.__shape_renderer = VectorShapeRenderer(svg_root)
```

def get_shape_renderer(self) -> ShapeRenderer:
    return self.__shape_renderer

## 4.10.3.8: 备忘录模式

> 备忘录模式可用于将对象的内部状态保存到另一个称为备忘录对象的对象中。

让我们以一个 `TextEditor` 类为例。首先，我们定义一个 `TextEditorState` 协议及其实现。然后，我们定义一个 `TextEditorStateMemento` 类，用于存储文本编辑器状态的备忘录。

```python
from typing import Protocol

class TextEditorState(Protocol):
    def clone(self) -> 'TextEditorState':
        pass

class TextEditorStateImpl(TextEditorState):
    # 在此处实现文本编辑器状态 ...

class TextEditorStateMemento:
    def __init__(self, state: TextEditorState):
        self.__state = state.clone()

    @property
    def state(self):
        return self.__state
```

`TextEditor` 类存储文本编辑器状态的备忘录。它提供了保存状态、恢复状态或恢复上一状态的方法：

```python
from TextEditorStateImpl import TextEditorStateImpl
from TextEditorStateMemento import TextEditorStateMemento

class TextEditor:
    __state_mementos: list[TextEditorStateMemento]

    def __init__(self):
        self.__current_state = TextEditorStateImpl(...)
        self.__state_mementos = []
        self.__current_version = 1

    def save_state(self) -> None:
        self.__state_mementos.append(
            TextEditorStateMemento(self.__current_state)
        )
        self.__current_version += 1

    def restore_state(self, version: int) -> None:
        if 1 <= version <= len(self.__state_mementos):
            self.__current_state = self.__state_mementos[version - 1].state
            self.__current_version += 1

    def restore_previous_state(self) -> None:
        if self.__current_version > 1:
            self.restore_state(self.__current_version - 1)
```

在上面的例子中，我们可以通过调用 `save_state` 方法为文本编辑器的状态添加一个备忘录。我们可以通过 `restore_previous_state` 方法回忆文本编辑器状态的上一个版本，也可以使用 `restore_state` 方法回忆任何版本的文本编辑器状态。

## 4.10.3.9: 访问者模式

> 访问者模式允许在不修改类的情况下为类添加功能（例如添加新方法）。这在处理无法修改的库类时非常有用。

首先，让我们看一个可以修改的类的例子：

```python
from typing import Protocol

class Shape(Protocol):
    def draw(self) -> None:
        pass

class CircleShape(Shape):
    def __init__(self, radius: int):
        self.__radius = radius

    def draw(self) -> None:
        # 绘制圆形 ...

    @property
    def radius(self) -> int:
        return self.__radius

class RectangleShape(Shape):
    def __init__(self, width: int, height: int):
        self.__width = width
        self.__height = height

    def draw(self) -> None:
        # 绘制矩形 ...

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height
```

假设我们需要计算绘图中形状的总面积。目前，我们可以修改形状类，因此让我们为这些类添加 `calculate_area` 方法：

```python
import math
from typing import Protocol

class Shape(Protocol):
    # ...

    def calculate_area(self) -> float:
        pass

class CircleShape(Shape):
    # ...

    def calculate_area(self) -> float:
        return math.pi * self.__radius**2

class RectangleShape(Shape):
    # ...

    def calculate_area(self) -> float:
        return self.__width * self.__height
```

向现有类添加新方法可能违反*开闭原则*。在上面的情况下，添加 `calculate_area` 方法是安全的，因为形状类是不可变的。即使它们不是不可变的，添加 `calculate_area` 方法也是安全的，因为它们是只读方法，即它们不修改对象的状态，并且我们不必担心线程安全问题，因为我们可以假设我们的示例应用程序不是多线程的。

现在我们已经添加了面积计算方法，我们可以使用一个通用算法来计算绘图中形状的总面积：

```python
from functools import reduce

shapes = [CircleShape(1), RectangleShape(2, 2)]
total_shapes_area = reduce(
    lambda accum_shapes_area, shape: accum_shapes_area + shape.calculate_area(),
    shapes,
    0.0
)

print(total_shapes_area) # 输出 7.141592653589793
```

但是，如果形状类（没有面积计算功能）位于我们无法修改的第三方库中呢？我们将不得不这样做：

```python
def shapes_area(accum_shapes_area: float, shape: Shape) -> float:
    if isinstance(shape, CircleShape):
        shape_area = math.pi * shape.radius**2
    elif isinstance(shape, RectangleShape):
        shape_area = shape.width * shape.height
    else:
        raise ValueError('Invalid shape')

    return accum_shapes_area + shape_area

total_shapes_area = reduce(
    shapes_area,
    shapes,
    0.0
)
```

上述解决方案很复杂，并且每次引入新类型的形状时都需要更新。上面的例子没有遵循面向对象设计原则：它包含一个带有 `isinstance` 检查的 `if/elif` 结构。

我们可以使用访问者模式用多态性替换上述条件语句。首先，我们引入一个访问者协议，可用于为形状类提供额外的行为。然后我们在 `Shape` 协议中引入一个 `execute` 方法。在形状类中，我们实现 `execute` 方法，以便可以执行具体访问者提供的额外行为：

```python
from typing import Any, Protocol

# 这是我们的访问者协议，它为形状类提供额外的行为
class ShapeBehavior(Protocol):
    def execute_for_circle(self, circle: 'CircleShape') -> Any:
        pass

    def execute_for_rectangle(self, rectangle: 'RectangleShape') -> Any:
        pass

    # 在此处为可能的其他形状类添加方法 ...

class Shape(Protocol):
    # ...
    def execute(self, behavior: ShapeBehavior) -> Any:
        pass

class CircleShape(Shape):
    def __init__(self, radius: int):
        self.__radius = radius

    @property
    def radius(self) -> int:
        return self.__radius

    def execute(self, behavior: ShapeBehavior) -> Any:
        return behavior.execute_for_circle(self)

class RectangleShape(Shape):
    def __init__(self, width: int, height: int):
        self.__width = width
        self.__height = height

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    def execute(self, behavior: ShapeBehavior) -> Any:
        return behavior.execute_for_rectangle(self)
```

假设形状类是可变的并且是线程安全的。我们将必须使用适当的同步来定义 `execute` 方法，以使它们也是线程安全的：

```python
from threading import Lock

class CircleShape(Shape):
    # 初始化 self.__lock = Lock() 的构造函数

    def execute(self, behavior: ShapeBehavior) -> Any:
        with self.__lock:
            return behavior.execute_for_circle(self)

    # 其余方法 ...

class RectangleShape(Shape):
    # 初始化 self.__lock = Lock() 的构造函数

    def execute(self, behavior: ShapeBehavior) -> Any:
        with self.__lock:
            return behavior.execute_for_rectangle(self)

    # 其余方法 ...
```

让我们实现一个用于计算不同形状面积的具体访问者：

```python
class AreaCalculationShapeBehavior(ShapeBehavior):
    def execute_for_circle(self, circle: CircleShape) -> Any:
        return math.pi * circle.radius**2

    def execute_for_rectangle(self, rectangle: RectangleShape) -> Any:
        return rectangle.width * rectangle.height
```

现在我们可以使用通用算法实现形状总面积的计算，并且摆脱了条件语句。我们为每个形状执行下面的 `area_calculation` 行为：

```python
shapes = [CircleShape(1), RectangleShape(2, 2)]
area_calculation = AreaCalculationShapeBehavior()

total_shapes_area = reduce(
    lambda accum_shapes_area, shape:
        accum_shapes_area + shape.execute(area_calculation),
    shapes,
    0.0
)
```

你可以通过定义新的访问者为形状类添加更多行为。让我们定义一个 `PerimeterCalculationShapeBehavior` 类：

```python
class PerimeterCalculationShapeBehavior(ShapeBehavior):
    def execute_for_circle(self, circle: CircleShape) -> Any:
        return 2 * math.pi * circle.radius

    def execute_for_rectangle(self, rectangle: RectangleShape) -> Any:
        return 2 * rectangle.width + 2 * rectangle.height
```

请注意，我们在代码示例中不需要使用 `visitor` 这个术语。将设计模式名称添加到软件实体的名称（类/函数名等）中通常不会带来任何实际好处，反而会使名称变长。然而，有些设计模式，比如 `工厂模式` 和 `建造者模式`，你总是会在类名中使用设计模式名称。

如果你正在开发一个第三方库，并希望其类的行为可以被用户扩展，你应该让你的库类接受可以执行额外行为的访问者。

## 4.10.3.10: 空对象模式

> 空对象是一个什么都不做的对象。

使用空对象模式来实现一个用于空对象的类，这些空对象什么都不做。空对象可以用来代替执行某些操作的真实对象。

让我们以一个 `Shape` 协议为例：

from typing import Protocol

class Shape(Protocol):
    def draw(self) -> None:
        pass

我们可以轻松地为一个空形状对象定义一个类：

```
from Shape import Shape

class NullShape(Shape):
    def draw(self) -> None:
        # Intentionally no operation
```

空形状不绘制任何内容。我们可以在任何需要具体实现 Shape 协议的地方使用 NullShape 类的实例。

## 4.11：不要询问，而是告知原则

> 不要询问，而是告知原则定义为：在你的对象中，你应该告诉另一个对象该做什么，而不是询问另一个对象的状态，然后在你自己的对象中完成工作。

如果你的对象通过例如多个 getter 从另一个对象询问许多事情，你可能犯了特性依恋情结的设计问题。你的对象嫉妒另一个对象应该拥有的特性。
让我们举个例子，定义一个立方体形状类：

```
from typing import Final, Protocol

class ThreeDShape(Protocol):
    # ...

class Cube3DShape(ThreeDShape):
    def __init__(self, width: int, height: int, depth: int):
        self.__width: Final = width
        self.__height: Final = height
        self.__depth: Final = depth

    @property
    def width(self, ) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @property
    def depth(self) -> int:
        return self.__depth
```

接下来，我们定义另一个类 CubeUtils，其中包含一个计算立方体总体积的方法：

```
from typing import final

from Cube3DShape import Cube3DShape


@final
class CubeUtils:
    @staticmethod
    def calculate_total_volume(cubes: list[Cube3DShape]) -> int:
        total_volume = 0
        for cube in cubes:
            total_volume += cube.width * cube.height * cube.depth
        return total_volume
```

在 `calculate_total_volume` 方法中，我们三次询问立方体对象的状态。这违反了*不要询问，而是告知原则*。我们的方法嫉妒体积计算特性，并希望亲自完成它，而不是告诉 Cube3DShape 对象去计算其体积。

让我们修正上述代码，使其遵循*不要询问，而是告知原则*：

```
from typing import Final, Protocol, final


class ThreeDShape(Protocol):
    def calculate_volume(self) -> int:
        pass


class Cube3DShape(ThreeDShape):
    def __init__(self, width: int, height: int, depth: int):
        self.__width: Final = width
        self.__height: Final = height
        self.__depth: Final = depth

    def calculate_volume(self) -> int:
        return self.__height * self.__width * self.__depth


@final
class ThreeDShapeUtils:
    @staticmethod
    def calculate_total_volume(three_d_shapes: list[ThreeDShape]) -> int:
        total_volume = 0
        for three_d_shape in three_d_shapes:
            total_volume += three_d_shape.calculate_volume()
        return total_volume
```

现在我们的 `calculate_total_volume` 方法不再询问立方体对象的任何信息。它只是告诉立方体对象去计算其体积。我们也从 `Cube3DShape` 类中移除了*询问*方法（getter/属性），因为它们不再需要。
以下是另一个询问而非告知的例子：

```
import time

class AnomalyDetectionEngine:
    def run(self) -> None:
        while self.__is_running:
            now = time.time()

            if self.__anomaly_detector.anomalies_should_be_detected(now):
                anomalies = self.__anomaly_detector.detect_anomalies()
                # Do something with the detected anomalies ...

            time.sleep(1)
```

在上面的例子中，我们询问异常检测器现在是否应该检测异常。然后，根据结果，我们调用异常检测器上的另一个方法来检测异常。这可以通过让 `detect_anomalies` 方法使用 `anomalies_should_be_detected` 方法来检查是否应该检测异常来简化。然后 `anomalies_should_be_detected` 方法可以设为私有，我们可以将上述代码简化如下：

```
class AnomalyDetectionEngine:
    def run(self) -> None:
        while self.__is_running:
            anomalies = self.__anomaly_detector.detect_anomalies()
            # Do something with the detected anomalies ...
            time.sleep(1)
```

## 4.12：迪米特法则

> 不应该调用从另一个对象的方法调用中获得的对象上的方法。

以下语句被认为违反了该法则：

```
user.get_account().get_balance()
user.get_account().withdraw(...)
```

上述语句可以通过将功能移动到不同的类，或者让第二个对象充当第一个和第三个对象之间的门面来修正。
以下是后一种解决方案的示例，我们在 `User` 类中引入两个新方法并移除 `get_account` 方法：

```
user.get_account_balance()
user.withdraw_from_account(...)
```

在上面的例子中，User 类是 Account 类的门面，我们不应该直接从我们的对象访问它。

然而，你应该始终检查是否可以使用第一种解决方案。它使代码更具面向对象性，并且不需要创建额外的方法。

以下是使用 User 和 SalesItem 实体且不遵守迪米特法则的示例：

```
from SalesItem import SalesItem
from User import User

def purchase(user: User, sales_item: SalesItem) -> None:
    account = user.get_account()

    # Breaks the law
    account_balance = account.get_balance()

    sales_item_price = sales_item.get_price()

    if account_balance >= sales_item_price:
        # Breaks the law
        account.withdraw(sales_item_price);

    # ...
```

我们可以通过将 purchase 方法移动到正确的类（在这种情况下是 User 类）来解决上述示例中的问题：

```
from Account import Account
from SalesItem import SalesItem

class User:
    def __init__(self, account: Account):
        self.__account = account

    def purchase(self, sales_item: SalesItem) -> None:
        account_balance = self.__account.get_balance()
        sales_item_price = sales_item.get_price()

        if account_balance >= sales_item_price:
            self.__account.withdraw(sales_item_price)

        # ...
```

## 4.13：避免原始类型痴迷原则

通过为函数参数和函数返回值定义语义类型来避免原始类型痴迷。

我们许多人都经历过以错误顺序向函数提供参数的情况。如果函数例如接受两个整数参数，但你意外地以错误顺序提供了这两个整数参数，这很容易发生。你不会得到编译错误。

原始类型作为函数参数的另一个问题是参数值不一定经过验证。你必须在函数中实现验证逻辑。

假设你在函数中接受一个整数参数作为端口号。在这种情况下，你可能会得到任何整数值作为参数值，即使有效的端口号是从 1 到 65535。假设你还在同一代码库中有其他函数接受端口号作为参数。在这种情况下，你可能最终在多个地方执行相同的验证逻辑，从而在代码库中产生重复代码。

让我们举一个使用此原则的简单例子：

```
from Shape import Shape

class RectangleShape(Shape):
    def __init__(self, width: int, height: int):
        self.__width = width
        self.__height = height
```

在上面的例子中，构造函数有两个相同原始类型（int）的参数。有可能以错误的顺序提供 `width` 和 `height`。但是，如果我们重构代码以使用对象而不是原始值，我们可以使以错误顺序提供参数的可能性大大降低：

```
from typing import TypeVar, Generic, Final

from Shape import Shape

T = TypeVar('T')

class Value(Generic[T]):
    def __init__(self, value: T):
        self.__value: Final = value

    @property
    def value(self) -> T:
        return self.__value

class Width(Value[int]):
    pass

class Height(Value[int]):
    pass

class RectangleShape(Shape):
    def __init__(self, width: Width, height: Height):
```

self.__width = width.value
self.__height = height.value

width = Width(20)
height = Height(50)

# 正确
rectangle = RectangleShape(width, height)

# 未通过类型检查，参数顺序错误
rectangle2 = RectangleShape(height, width)

# 未通过类型检查，第一个参数不是宽度
rectangle3 = RectangleShape(height, height)

# 未通过类型检查，第二个参数不是高度
rectangle4 = RectangleShape(width, width)

# 编译失败，必须使用 Width 和 Height 对象
# 而不是原始类型
rectangle5 = RectangleShape(20, 50)

在上面的例子中，Width 和 Height 是简单的数据类。它们不包含任何行为。你可以使用具体的数据类作为函数参数类型。无需为数据类创建接口。因此，*面向接口编程*的原则在此并不适用。

在 Python 中，我们有另一种安全的方式来防止以错误的顺序传递相同类型的参数：使用命名参数。如果不使用命名参数，我们会创建一个新的矩形：

```
rectangle = RectangleShape(20, 50)
```

在上面的例子中，我们必须确保第一个参数是宽度，第二个参数是高度。使用命名参数时，我们不必记住参数的正确顺序：

```
rectangle = RectangleShape(width=20, height=50)
rectangle2 = RectangleShape(height=50, width=20)
```

让我们看另一个简单的例子，其中我们有以下函数签名：

```
def do_something(namespaced_name: str, ...):
    # ...
```

上述函数签名允许函数调用者意外地提供一个非命名空间的名称。通过为命名空间名称使用自定义类型，我们可以将上述函数签名修改为以下形式：

```
from typing import Final

class NamespacedName:
    def __init__(self, namespace: str, name: str):
        self.__namespaced_name: Final = (
            name if not namespace else (namespace + '.' + name)
        )

    def get(self) -> str:
        return self.__namespaced_name

def do_something(namespaced_name: NamespacedName, ...):
    # ...
```

让我们看一个更全面的例子，使用一个 HttpUrl 类。该类构造函数有几个参数，应在创建 HTTP URL 时进行验证：

```
from typing import Final

class HttpUrl:
    def __init__(
        self,
        scheme: str,
        host: str,
        port: int,
        path: str,
        query: str
    ):
        self.__http_url: Final = (
            scheme
            + "://"
            + host
            + ":"
            + str(port)
            + path
            + "?"
            + query
        )
```

让我们为已验证的值引入一个抽象类：

## 面向对象设计原则

242

```
from abc import abstractmethod
from typing import Final, Generic, TypeVar

from Optional import Optional

T = TypeVar('T')
```

```
class AbstractValidatedValue(Generic[T]):
    def __init__(self, value: T):
        self._value: Final = value

    @abstractmethod
    def is_valid(self) -> bool:
        pass

    def get(self) -> Optional[T]:
        return (
            Optional.of(self._value)
            if self.is_valid()
            else Optional.empty()
        )

    class GetError(Exception):
        pass

    def try_get(self) -> T:
        if self.is_valid():
            return self._value
        else:
            raise self.GetError('Invalid ' + self.__class__.__name__)
```

让我们为已验证的 HTTP scheme 对象创建一个类：

```
from functools import cache

from AbstractValidatedValue import AbstractValidatedValue
```

```
class HttpScheme(AbstractValidatedValue[str]):
    # 因为实例是不可变的，我们可以缓存验证结果
    @cache
    def is_valid(self) -> bool:
        lowercase_value = self._value.lower()
        return lowercase_value == 'https' or lowercase_value == 'http'
```

让我们创建一个 Port 类（并应为 host、path 和 query 创建类似的类）：

## 面向对象设计原则

243

```
from functools import cache

from AbstractValidatedValue import AbstractValidatedValue

class Port(AbstractValidatedValue[int]):
    # 因为实例是不可变的，我们可以缓存验证结果
    @cache
    def is_valid(self) -> bool:
        return 1 <= self._value <= 65535

# 实现 Host 类 ...
# 实现 Path 类 ...
# 实现 Query 类 ...
```

让我们创建一个工具类 OptionalUtils，其中包含一个用于映射五个可选值结果的方法：

```
from collections.abc import Callable
from typing import TypeVar, final

from Optional import Optional

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
X = TypeVar('X')
Y = TypeVar('Y')
R = TypeVar('R')

@final
class OptionalUtils:
    @staticmethod
    def map_all(
        opt1: Optional[T],
        opt2: Optional[U],
        opt3: Optional[V],
        opt4: Optional[X],
        opt5: Optional[Y],
        mapper: Callable[[T, U, V, X, Y], R]
    ) -> Optional[R]:
        if (
            opt1.is_present()
            and opt2.is_present()
            and opt3.is_present()
            and opt4.is_present()
            and opt5.is_present()
        ):
            return Optional.of(
                map(
                    opt1.try_get(),
                    opt2.try_get(),
                    opt3.try_get(),
                    opt4.try_get(),
                    opt5.try_get(),
                )
            )
        else:
            return Optional.empty()
```

接下来，我们可以重新实现 HttpUrl 类，使其包含两个用于创建 HTTP URL 的替代工厂方法：

```
# 导入 ...

class PrivateConstructor(type):
    def __call__(
        cls: type[T], *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ):
        raise TypeError('Constructor is private')

    def _create(
        cls: type[T], *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> T:
        return super().__call__(*args, **kwargs)

class HttpUrl(metaclass=PrivateConstructor):
    def __init__(self, http_url: str):
        self.__http_url = http_url

    # 返回可选 HttpUrl 的工厂方法
    @classmethod
    def create(
        cls,
        scheme: HttpScheme,
        host: Host,
        port: Port,
        path: Path,
        query: Query
    ) -> Optional['HttpUrl']:
        return OptionalUtils.map_all(
            scheme.get(),
            host.get(),
            port.get(),
            path.get(),
            query.get(),
            lambda scheme_val, host_val, port_val, path_val, query_val: cls._create(
                scheme_val
                + '://'
                + host_val
                + ':'
                + port_val
                + path_val
                + '?'
                + query_val
            )
        )

class CreateError(Exception):
    pass

# 返回有效 HttpUrl 或
# 抛出错误的工厂方法
```

```
@classmethod
def try_create(
    cls,
    scheme: HttpScheme,
    host: Host,
    port: Port,
    path: Path,
    query: Query
) -> 'HttpUrl':
    try:
        return cls._create(
            scheme.try_get()
            + '://'
            + host.try_get()
            + ':'
            + str(port.try_get())
            + path.try_get()
            + '?'
            + query.try_get()
        )
    except AbstractValidatedValue.GetError as error:
        raise cls.CreateError(error)
```

```
maybe_http_url = HttpUrl.create(
    HttpScheme('https'),
    Host('www.google.com'),
    Port(443),
    Path('/query'),
    Query('search=jee')
)

# 打印 https://www.google.com:443/query?search=jee
print(maybe_http_url.try_get().url_string)

# 抛出错误：Invalid Port
http_url2 = HttpUrl.try_create(
    HttpScheme('https'),
    Host('www.google.com'),
    Port(443222),
    Path('/query'),
    Query('search=jee')
)
```

请注意，我们没有在 HttpUrl 类内部硬编码 URL 验证逻辑，而是创建了小型的已验证值类：HttpScheme、Host、Port、Path 和 Query。如果需要，这些类可以在代码库的其他部分进一步使用，甚至可以放入通用的验证库中以供更广泛的使用。

应用程序通常通过以下方式从外部源接收未验证的输入数据：

- 读取命令行参数
- 读取环境变量
- 读取标准输入
- 从文件系统读取文件

- 从套接字（网络输入）读取数据
- 从用户界面（UI）接收输入

请确保对从上述来源接收到的任何数据进行验证。最好使用现成的验证库，或者在需要时创建自己的验证逻辑。

## 4.14：依赖注入（DI）原则

> 依赖注入（DI）允许根据静态或动态配置来改变应用程序的行为。

使用依赖注入时，依赖项仅在应用程序启动时注入。应用程序可以先读取其配置，然后决定为应用程序创建哪些对象。在许多语言中，依赖注入对于单元测试也至关重要。使用 DI 执行单元测试时，你可以将模拟依赖注入到被测试的代码中，而不是使用应用程序的标准依赖项。

以下是一个不使用依赖注入的单例模式示例：

```python
from enum import Enum

from Logger import Logger

class LogLevel(Enum):
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5

class StdOutLogger(Logger):
    @staticmethod
    def log(log_level: LogLevel, message: str):
        # Log to standard output

class Application:
    def run(self):
        StdOutLogger.log(LogLevel.Info, 'Starting application')
        # ...
```

在上面的例子中，我们使用了硬编码的 `StdOutLogger` 类的静态方法。这使得以后更改日志记录器变得困难，也难以对静态方法进行单元测试。

我们应该重构上述代码，不使用静态方法，而是使用依赖注入：

图 4.80. Logger.py

```python
from enum import Enum
from typing import Protocol

class LogLevel(Enum):
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5

class Logger(Protocol):
    def log(self, log_level: LogLevel, message: str):
        pass
```

图 4.81. StdOutLogger.py

```python
from Logger import Logger, LogLevel

class StdOutLogger(Logger):
    def log(self, log_level: LogLevel, message: str):
        # Log to standard output
```

图 4.82. Application.py

```python
from dependency_injector.wiring import Provide, inject
from Logger import Logger
from LogLevel import LogLevel

class Application:
    @inject
    def __init__(self, logger: Logger = Provide['logger']):
        self.__logger = logger

    def run(self):
        self.__logger.log(LogLevel.INFO, 'Starting application')
        # ...
```

然后我们需要定义 DI 容器：

图 4.83. DiContainer.py

```python
from dependency_injector import containers, providers
from StdOutLogger import StdOutLogger

class DiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=['Application']
    )

    logger = providers.Singleton(StdOutLogger)
```

现在可以轻松地更换不同的日志记录器了。假设我们想将日志记录到文件而不是标准输出。我们可以引入一个新的基于文件的日志记录器类（遵循开闭原则）。

图 4.84. FileLogger.py

```python
from Logger import Logger, LogLevel

class FileLogger(Logger):
    def __init__(self, log_file_directory: str):
        self.__log_file_directory = log_file_directory

    def log(self, log_level: LogLevel, message: str):
        # Log to a file in self.__log_file_directory
```

然后我们可以更改 DI 容器以使用基于文件的日志记录器，而不是标准输出日志记录器：

图 4.85. DiContainer.py

```python
import os

from dependency_injector import containers, providers
from FileLogger import FileLogger

class DiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=['Application']
    )

    logger = providers.Singleton(FileLogger, os.environ.get('LOG_DIRECTORY'))
```

我们还可以根据应用程序运行的环境动态更改日志记录行为：

图 4.86. DiContainer.py

```python
import os

from dependency_injector import containers, providers
from FileLogger import FileLogger
from StdOutLogger import StdOutLogger

class DiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=['Application']
    )

    if os.environ.get('LOG_DESTINATION') == 'file':
        logger = providers.Singleton(FileLogger, os.environ.get('LOG_DIRECTORY'))
    else:
        logger = providers.Singleton(StdOutLogger)
```

对于所有全栈 Python 开发者，下面是一个数据可视化 Web 客户端的 TypeScript 示例，其中使用了 noicejs NPM 库进行依赖注入。这个库类似于 Google Guice 库。下面是一个 FakeServicesModule 类，它为 Web 客户端使用的不同后端服务配置依赖项。如你所见，所有服务都配置为使用模拟实现，因为此 DI 模块在后端服务尚不可用时使用。当后端服务可用时，可以实现并使用一个 RealServicesModule 类。在 RealServicesModule 类中，服务被绑定到它们的实际实现类，而不是模拟实现。

```typescript
import { Module } from 'noicejs';
import FakeDataSourceService from ...;
import FakeMeasureService from ...;
import FakeDimensionService from ...;
import FakeChartDataService from ...;

export default class FakeServicesModule extends Module {
  override async configure(): Promise<void> {
    this.bind('dataSourceService')
      .toInstance(new FakeDataSourceService());

    this.bind('measureService')
      .toInstance(new FakeMeasureService());

    this.bind('dimensionService')
      .toInstance(new FakeDimensionService());

    this.bind('chartDataService')
      .toInstance(new FakeChartDataService());
  }
}
```

使用 noicejs 库，你可以配置多个 DI 模块，并从所需的模块创建一个 DI 容器。模块化方法允许你将依赖项划分为多个模块，这样你就不会有一个庞大的单一模块。它还允许你根据应用程序配置实例化不同的模块。

在下面的例子中，DI 容器是从单个模块（FakeServicesModule 类的一个实例）创建的：

```typescript
import { Container } from 'noicejs';
import FakeServicesModule from './FakeServicesModule';

const diContainer = Container.from(new FakeServicesModule());

export default diContainer;
```

在开发阶段，我们可以创建两个独立的模块，一个用于模拟服务，另一个用于真实服务，并根据网页的 URL 查询参数控制应用程序行为：

```typescript
import { Container } from 'noicejs';
import FakeServicesModule from './FakeServicesModule';
import RealServicesModule from './RealServicesModule';

const diContainer = (() => {
  if (location.href.includes('useFakeServices=true')) {
    // Use fake services if web page URL
    // contains 'useFakeServices=true'
    return Container.from(new FakeServicesModule());
  } else {
    // Otherwise use real services
    return Container.from(new RealServicesModule());
  }
})();

export default diContainer;
```

然后，你必须在使用依赖注入之前配置 diContainer。在下面的例子中，diContainer 在渲染 React 应用程序之前被配置：

```typescript
import React from 'react';
import ReactDOM from 'react-dom';
import diContainer from './diContainer';
import AppView from './app/view/AppView';

diContainer.configure().then(() => {
  ReactDOM.render(<AppView />, document.getElementById('root'));
});
```

然后，在需要服务的 Redux actions 中，你可以使用 @Inject 装饰器注入所需的服务。你指定要注入的服务名称。该服务将作为类构造函数参数的属性（同名）被注入。

图 4.90. StartFetchChartDataAction.ts

```typescript
// Imports ...

type ConstructorArgs = {
  chartDataService: ChartDataService,
  chart: Chart,
  dispatch: Dispatch;
};

export default
@Inject('chartDataService')
class StartFetchChartDataAction extends AbstractChartAreaAction {
  private readonly chartDataService: ChartDataService;
  private readonly chart: Chart;

  constructor({ chart,
                chartDataService,
                dispatch }: ConstructorArgs) {
    super(dispatch);
    this.chartDataService = chartDataService;
    this.chart = chart;
  }

  perform(currentState: ChartAreaState): ChartAreaState {
    this.chartDataService
      .fetchChartData(
        this.chart.dataSource,
        this.chart.getColumns(),
        this.chart.getSelectedFilters(),
        this.chart.getSelectedSortBys()
      )
      .then((columnNameToValuesMap: ColumnNameToValuesMap) => {
        this.dispatch(
          new FinishFetchChartDataAction(columnNameToValuesMap,
                                        this.chart.id)
        );
      })
      .catch((error) => {
        // Handle error
      });

    this.chart.isFetchingChartData = true;
    return ChartAreaStateUpdater
              .getNewStateForChangedChart(currentState, this.chart);
  }
}
```

为了能够调度上述 action，应该实现一个控制器：

## 4.15：避免代码重复原则

在类级别上，当你在两个实现相同接口的不同类中发现重复代码时，你应该创建一个新的基类来容纳公共功能，并让这些类继承这个新基类。

下面是一个实现了 `InputMessage` 协议的 `AvroBinaryKafkaInputMessage` 类：

图 4.95. InputMessage.py

```python
from typing import Protocol

from DecodedMessage import DecodedMessage
from Schema import Schema

class InputMessage(Protocol):
    def try_decode_schema_id(self) -> int:
        pass

    def try_decode(self, schema: Schema) -> DecodedMessage:
        pass
```

图 4.96. AvroBinaryKafkaInputMessage.py

```python
from DecodedMessage import DecodedMessage
from InputMessage import InputMessage
from KafkaMessage import KafkaMessage
from Schema import Schema

class AvroBinaryKafkaInputMessage(InputMessage):
    def __init__(self, kafka_message: KafkaMessage):
        self.__kafka_message = kafka_message

    def try_decode_schema_id(self) -> int:
        # Try decode schema id from the beginning of
        # the Avro binary Kafka message

    def try_decode(self, schema: Schema) -> DecodedMessage:
        return schema.try_decode_message(
            self.__kafka_message.payload,
            self.__kafka_message.length
        )
```

如果我们想为 JSON、CSV 或 XML 格式引入一个新的 Kafka 输入消息类，我们可以创建一个类似于 `AvroBinaryKafkaInputMessage` 的类。但随后我们会注意到 `try_decode` 方法中的代码重复。我们可以发现，无论输入消息的来源和格式如何，`try_decode` 方法都是相同的。根据这个原则，我们应该将重复的代码移动到一个公共基类 `AbstractInputMessage` 中。我们可以根据模板方法模式将 `try_decode` 方法设为模板方法，并为获取消息数据及其长度创建抽象方法：

图 4.97. AbstractInputMessage.py

```python
from abc import abstractmethod
from typing import final

from DecodedMessage import DecodedMessage
from InputMessage import InputMessage
from Schema import Schema

class AbstractInputMessage(InputMessage):
    @abstractmethod
    def try_decode_schema_id(self) -> int:
        pass

    @abstractmethod
    def _get_data(self) -> bytearray:
        pass

    @abstractmethod
    def _get_length(self) -> int:
        pass

    # Template method
    @final
    def try_decode(self, schema: Schema) -> DecodedMessage:
        return schema.try_decode_message(
            self._get_data(),
            self._get_length()
        )
```

接下来，我们应该重构 `AvroBinaryKafkaInputMessage` 类，使其继承新的 `AbstractInputMessage` 类，并实现受保护的 `_get_data` 和 `_get_length` 方法。但我们可以意识到，对于所有 Kafka 输入消息数据格式，这两个方法都是相同的。我们不应该在 `AvroBinaryKafkaInputMessage` 类中实现这两个方法，因为如果我们需要为另一种数据格式添加 Kafka 输入消息类，我们将需要重复实现它们。再次，我们可以利用这个原则，为 Kafka 输入消息创建一个新的基类：

图 4.98. AbstractKafkaInputMessage.py

```python
from abc import abstractmethod

from AbstractInputMessage import AbstractInputMessage
from KafkaMessage import KafkaMessage

class AbstractKafkaInputMessage(AbstractInputMessage):
    def __init__(self, kafka_message: KafkaMessage):
        self.__kafka_message = kafka_message

    @abstractmethod
    def try_decode_schema_id(self) -> int:
        pass

    def _get_data(self) -> bytearray:
        return self.__kafka_message.payload

    def _get_length(self) -> int:
        return self.__kafka_message.length
```

最后，我们可以重构 `AvroBinaryKafkaInputMessage` 类，使其不包含任何重复代码：

图 4.99. AvroBinaryKafkaInputMessage.py

```python
from AbstractKafkaInputMessage import AbstractKafkaInputMessage

class AvroBinaryKafkaInputMessage(AbstractKafkaInputMessage):
    def try_decode_schema_id(self) -> int:
        # Try decode the schema id from the beginning of
        # the Avro binary Kafka message
        # Use base class _get_data() and _get_length()
        # methods to achieve that
```

## 4.16：层叠样式表（CSS）中的继承

最后一节面向对 CSS 中继承工作原理感兴趣的全栈 Python 开发者。在 HTML 中，你可以为 HTML 元素定义类（类名）：

```html
<span class="icon pie-chart-icon">...</span>
```

在 CSS 文件中，你为 CSS 类定义 CSS 属性，例如：

```css
.icon {
    background-repeat: no-repeat;
    background-size: 1.9rem 1.9rem;
    display: inline-block;
    height: 2rem;
    margin-bottom: 0.2rem;
    margin-right: 0.2rem;
    width: 2rem;
}

.pie-chart-icon {
    background-image: url('pie_chart_icon.svg');
}
```

上述方法的问题在于它不是正确的面向对象设计。在 HTML 代码中，你必须列出所有类名才能实现所有所需 CSS 属性的混合。很容易忘记添加一个类名。例如，你可能只指定了 `pie-chart-icon` 而忘记了指定 `icon`。

事后也很难更改继承层次结构。假设你想为所有图表图标添加一个新的类 `chart-icon`：

```css
.chart-icon {
  /* Define properties here... */
}
```

你必须记住在 HTML 代码中所有渲染图表图标的地方添加 `chart-icon` 类名：

```html
<span class="icon chart-icon pie-chart-icon">...</span>
```

上述描述的方法非常容易出错。你应该做的是引入适当的面向对象设计。你需要一个 CSS 预处理器，使扩展 CSS 类成为可能。在下面的例子中，我使用的是 SCSS：

```html
<span class="pieChartIcon">...</span>
```

```scss
.icon {
  background-repeat: no-repeat;
  background-size: 1.9rem 1.9rem;
  display: inline-block;
  height: 2rem;
  margin-bottom: 0.2rem;
  margin-right: 0.2rem;
  width: 2rem;
}

.chartIcon {
  @extend .icon;

  /* Other chart icon related properties... */
}

.pieChartIcon {
  @extend .chartIcon;

  background-image: url('../../../../assets/images/icons/chart/pie_chart_icon.svg');
}
```

在上面的例子中，我们只为 HTML 元素定义了一个类。继承层次结构在 SCSS 文件中使用 `@extend` 指令定义。我们现在可以自由地在未来更改继承层次结构，而无需修改 HTML 代码。

## 5：编码原则

本章介绍编码原则。以下原则将被阐述：

- 统一变量命名原则
- 统一源代码仓库结构原则
- 源代码目录树结构原则
- 避免注释原则
- 函数单一返回语句原则
- 优先使用静态类型语言原则
- 重构原则
- 静态代码分析原则
- 错误/异常处理原则
- 不传递或返回 null 原则
- 避免差一错误原则
- 谷歌搜索时保持批判性原则
- 优化原则

### 5.1：统一变量命名原则

> *一个好的变量名应该描述变量的用途及其类型。*

理想情况下，使用优秀名称编写的代码读起来应如散文般流畅。请记住，代码被阅读的频率远高于编写，因此代码必须易于阅读和理解。

在无类型语言中，使用能同时传达变量类型信息的名称来命名变量至关重要；在有类型语言中同样有益，因为现代有类型语言使用自动类型推断，你并不总能看到变量的实际类型。但当变量名本身已说明其类型时，类型名是否可见便无关紧要了。

根据经验法则，如果变量名长度达到或超过20个字符，应考虑将其缩短。尝试缩写变量名中的一个或多个单词，但仅使用有意义且广为人知的缩写。如果此类缩写不存在，则完全不要缩写。例如，如果你有一个名为 `environment_variable_name` 的变量，你应该尝试缩短它，因为它超过了20个字符。你可以将 *environment* 缩写为 *environ*，将 *variable* 缩写为 *var*，从而得到一个足够短的变量名 `environ_var_name`。*environ* 和 *var* 这两个缩写都是常用且易于理解的。再举一个名为 `loyalty_bonus_percentage` 的变量的例子。你不能缩写 *loyalty*。你不能缩写 *bonus*。但你可以将 *percentage* 缩写为 *percent* 甚至 *pct*。我宁愿使用 *percent* 而不是 *pct*。使用 *percent* 使变量名长度小于20个字符（下划线不计入变量名字符数）。

在以下章节中，将针对不同类型的变量提出命名约定。

#### 5.1.1：命名整数变量

有些变量本质上是整数，比如 *age*（年龄）或 *year*（年份）。每个人都能立即理解 *age* 或 *year* 变量的类型是数字，更具体地说，是整数。因此，你无需在变量名中添加任何内容来指示其类型。它已经说明了其类型。

整数变量中最常用的类别之一是计数或某物的数量。你在每段代码中都能看到这类变量。我建议使用以下约定来命名这些变量：*number_of_<something>* 或者 *<something>_count*。例如，*number_of_failures* 或 *failure_count*。你不应该使用变量名 *failures* 来表示失败计数。这个变量名的问题在于它没有明确指定变量的类型，因此可能导致一些混淆。这是因为名为 *failures* 的变量可能被误解为一个集合变量（例如，一个失败列表）。

如果变量的单位不言自明，请始终在变量名末尾添加单位信息。例如，与其将变量命名为 *tooltip_show_delay*，你应该将其命名为 *tooltip_show_delay_in_millis* 或 *tooltip_show_delay_in_ms*。如果你有一个单位不言自明的变量，则不需要单位信息。因此，没有必要将 *age* 变量命名为 *age_in_years*。但如果你是以月为单位测量年龄，则必须将相应的变量命名为 *age_in_months*，以免人们假设年龄是以年为单位测量的。

#### 5.1.2：命名浮点数变量

浮点数不如整数常见，但有时你也会需要它们。有些值本质上是浮点数，比如大多数未取整的测量值（例如，价格、高度、宽度或重量）。如果你需要存储一个测量值，使用浮点变量是一个稳妥的选择。

如果你需要存储非整数的某物数量，请使用名为 *<something>_amount* 的变量，比如 *rainfall_amount*（降雨量）。当你在代码中看到“某物的数量”时，可以自动联想到它是浮点数。如果你需要在算术运算中使用一个数字，根据应用程序的不同，你可能希望使用浮点或整数算术。在涉及金钱的情况下，你应该使用整数算术以避免舍入误差。你应该使用一个整数变量，例如 *money_in_cents*（以分为单位的金额），而不是浮点变量 *money_amount*。

如果变量的单位不言自明，请在变量名末尾添加单位信息，例如 *rainfall_amount_in_mm*（以毫米为单位的降雨量）、*width_in_inches*（以英寸为单位的宽度）、*angle_in_degrees*（以度为单位的角度，值范围0-360）、*failure_percent*（失败百分比，值范围0-100）或 *failure_ratio*（失败比率，值范围0-1）。

#### 5.1.3：命名布尔变量

布尔变量只能有两个值之一：true 或 false。布尔变量的名称应构成一个陈述，其答案为 true 或 false，或 yes 或 no。典型的布尔变量命名模式有：`is_<something>`、`has_<something>`、`did_<something>`、`should_<something>`、`can_<something>` 或 `will_<something>`。遵循上述模式的变量名示例有 `is_disabled`、`has_errors`、`did_update`、`should_update` 和 `will_update`。

布尔变量名中的动词不必位于开头。如果放在中间能使代码读起来更好，那么它就可以也应该放在中间。布尔变量常用于 if 语句中，改变变量名中的词序可以使代码读起来更顺畅。请记住，理想情况下，代码读起来应如优美的散文，而且代码被阅读的频率远高于编写。

下面是一个代码片段，其中有一个名为 `is_pool_full` 的布尔变量：

```
# ...

is_pool_full = len(self.__pooled_messages) >= 200
if is_pool_full:
    # ...
else:
    # ...
```

我们可以将变量名更改为 `pool_is_full`，以使 if 语句读起来更流畅。在下面的示例中，if 语句读作“if pool_is_full”而不是“if is_pool_full”：

```
# ...

pool_is_full = len(self.__pooled_messages) >= 200
if pool_is_full:
    # ...
else:
    # ...
```

不要使用 `<passive-verb>_something` 形式的布尔变量名，例如 `inserted_field`，因为这可能会让读者感到困惑。不清楚变量名是命名对象的名词还是布尔陈述。相反，请使用 `did_insert_field` 或 `field_was_inserted`。

下面是一个用于存储函数返回值的变量命名不当的示例。`drop_redundant_tables` 函数返回一个布尔值。有人可能会认为 `tables_dropped` 表示已删除表名的列表。因此，该变量名含义模糊，应予以更改。

```
tables_dropped = drop_redundant_tables(
    prefix,
    vms_data,
    config.database,
    hive_client,
    logger
)

if tables_dropped:
    # ...
```

下面是修改后的示例，变量名已更改为表示布尔陈述：

```
tables_were_dropped = drop_redundant_tables(
    prefix,
    vms_data,
    config.database,
    hive_client,
    logger
)

if tables_were_dropped:
    # ...
```

你本可以使用名为 `did_drop_tables` 的变量，但 `tables_were_dropped` 使 if 语句更具可读性。如果 `drop_redundant_tables` 函数的返回值是已删除表名的列表，我会将接收返回值的变量命名为 `dropped_table_names`。

当你阅读包含否定布尔变量的代码时，通常读起来不顺畅，例如：

```
app_was_started = app.start()

if not app_was_started:
    # ...
```

你可以做的是在脑海中将 `not` 一词移动到句子中的正确位置，使句子读起来像正确的英语。例如：*if app was not started*（如果应用未启动）。

另一种选择是对变量取反。这可以通过在赋值运算符两侧都添加 `not` 来对赋值两侧进行取反来实现。下面是一个例子：

```
app_was_not_started = not app.start()

if app_was_not_started:
    # ...
```

## 5.1.4：命名字符串变量

字符串变量非常普遍，许多事物本质上就是字符串，例如*姓名*、*标题*、*城市*、*国家*或*主题*。当你需要将数值数据存储在字符串变量中时，请向代码阅读者明确说明这是一个字符串格式的数字问题，并使用以下格式的变量名：`<someValue>_string` 或 `<someValue>_as_string`。这能使代码更突出且更易于理解。例如：

```python
try:
    year = int(year_as_string)
except ValueError:
    # ...
```

如果你有一个可能与对象变量混淆的变量，比如 `schema`，但它是一个字符串，请在变量名末尾添加 *string*，即 `schema_string`。以下是一个示例：

```python
schema = schema_parser.parse(schema_string)
```

## 5.1.5：命名枚举变量

枚举变量的命名应与枚举类型同名。例如，一个 `CarType` 枚举变量应命名为 `car_type`。如果枚举类型的名称非常通用，比如 `Result`，你可能需要在变量名中添加一些细节来声明枚举变量。以下是一个非常通用的枚举类型名称示例：

```python
# 返回枚举类型 'Result'
result = pulsar_client.create_producer(...)

if result == Result.Ok:
    # ...
```

让我们为 `result` 变量名添加一些细节和上下文：

```python
producer_create_result = pulsar_client.create_producer(...)

if producer_create_result == Result.Ok:
    # ...
```

## 5.1.6：命名集合（列表和集合）变量

在命名数组、列表和集合时，应使用名词的复数形式，例如 *customers*、*errors* 或 *tasks*。这类名称在代码中效果很好，例如：

```python
def process(customers: list[Customer]) -> list[Customer]:
    # ...

customers = [...]

for customer in customers:
    # ...

processed_customers = process(customers)

def even(integer: int) -> bool:
    return integer % 2 == 0

integers = [1, 2, 3, 4, 5]
even_integers = filter(even, integers)
```

在大多数情况下，这就足够了，因为你不一定需要知道底层的集合实现。使用这种命名约定允许你更改集合变量的类型而无需更改变量名。如果你正在遍历一个集合，无论是数组、列表还是集合都无关紧要。因此，在变量名中添加集合类型名称（例如 `customer_list` 或 `task_set`）不会带来任何好处。这些名称只是更长而已。在某些特殊情况下，你可能需要指定集合类型。那么，你可以使用以下类型的变量名：`queue_of_tasks`、`stack_of_cards` 或 `set_of_timestamps`。

以下是一个示例，其中函数命名正确以返回一个集合（类别），但接收返回值的变量未按照集合变量命名约定命名：

```python
vms_data = vms_client.get_categories(vms_url, logger)
```

正确的命名应为：

```python
vms_categories = vms_client.get_categories(vms_url, logger)
```

## 5.1.7：命名字典变量

字典通过请求某个 `key` 的 `value` 来访问。这就是为什么我建议使用 `key_to_value`、`key_to_value_map` 或 `key_to_value_dict` 模式来命名映射。假设我们有一个包含客户 ID 订单数量的字典。这个字典应命名为 `customer_id_to_order_count`、`customer_id_to_order_count_map` 或 `customer_id_to_order_count_dict`。或者如果我们有一个产品名称的供应商列表，映射变量应命名为 `product_name_to_suppliers`、`product_name_to_suppliers_map` 或 `product_name_to_suppliers_dict`。以下是访问字典的示例：

```python
order_count = customer_name_to_order_count.get(customer_name)
suppliers = product_name_to_suppliers.get(product_name)
```

以下是遍历字典的示例：

```python
customer_name_to_order_count = {
    'John': 10,
    'Peter': 5
}

for customer_name in customer_name_to_order_count:
    print(customer_name)

for customer_name in customer_name_to_order_count.keys():
    print(customer_name)

for order_count in customer_name_to_order_count.values():
    print(order_count)

for (
    customer_name,
    order_count,
) in customer_name_to_order_count.items():
    print(customer_name, order_count)
```

## 5.1.8：命名配对和元组变量

包含配对的变量应使用 *variable1_and_variable2* 模式命名。例如：*height_and_width*。对于元组，推荐的命名模式是 *variable1_variable2...and_variableN*。例如：*height_width_and_depth*。如果元组中的值不需要单独命名，那么你可以将元组视为不可变列表，并根据集合命名约定为其命名。

以下是使用配对和元组的示例：

```python
height_and_width = (100, 200)
height, width = height_and_width
height_width_and_depth = (100, 200, 40)
height, *, depth = height_width_and_depth
numbers = (1, 2, 3, 4, 5)
print(numbers[-1]) # 打印 5
print(numbers[0:2]) # 打印 (1,2)
print(numbers[:2]) # 打印 (1,2)
print(numbers[1:]) # 打印 (2, 3, 4, 5)
```

## 5.1.9：命名对象变量

对象变量引用类的实例。类名是用大写字母开头的名词，如 *Person*、*CheckingAccount* 或 *Task*。对象变量名应包含相关的类名：Person 类的 person 对象、Account 类的 account 对象等。你可以自由地修饰对象的名称，例如，用形容词：completed_task。重要的是在变量名末尾包含类名或至少其重要部分。然后查看变量名的末尾就能知道讨论的是哪种对象。

有时你可能想命名一个对象变量，使其类名是隐含的，例如：

```python
# 函数参数 'Location' 的类是隐含的
drive(from=home, to=destination)
```

在上面的例子中，home 和 destination 对象的类不是显式的。在大多数情况下，当变量名不会太长时，最好在变量名中显式包含类名。这是因为变量类型推断。变量的类型在代码中不一定可见，因此变量的类型应通过变量名来传达。以下是函数参数类型显式的示例。

```python
# 函数参数 'Location' 的类现在是显式的
drive(from=home_location, to=dest_location)
```

## 5.1.10：命名可选变量

如果你使用 Optional[T]，请使用以下模式命名此类型的变量：maybe_<something>：

```python
maybe_logged_in_user.if_present(lambda logged_in_user: logged_in_user.logout())
current_user = maybe_logged_in_user.or_else(guest_user)
```

当你使用类型联合创建可选类型时，可选变量名不需要任何前缀。在下面的例子中，discount 参数是可选的：

```python
def add_tax(price: float, discount: int | None = None) -> float:
    return 1.2 * (price - (0 if discount is None else discount))

price_with_tax = add_tax(price_without_tax)
```

## 5.1.11：命名函数变量（回调）

回调函数是提供给其他函数以在某个时候调用的函数。如果回调函数返回一个值，它可以按照返回值命名，但仍应包含一个动词。如果回调函数不返回值，你应该像命名任何其他函数一样命名回调函数：指示函数的功能。

```python
def doubled(number: int | float):
    return 2 * number
```

```python
def squared(number: int | float):
    return number**2
```

```python
def even(number: int | float):
    return number % 2 == 0
```

```python
values = [1, 2, 3, 4, 5]
doubled_values = [doubled(value) for value in values]
doubled_values2 = list(map(doubled, values))
squared_values = [squared(value) for value in values]
squared_values2 = list(map(squared, values))
even_values = [value for value in values if even(value)]
even_values2 = list(filter(even, values))
```

```python
def trimmed(string: str):
    return string.strip()
```

```python
strings = [" string1", "string2 "]
trimmed_strings = [trimmed(string) for string in strings]
trimmed_strings2 = list(map(trimmed, strings))
```

```python
def sum(accum_sum: int | float, number: int | float) -> int | float:
    return accum_sum + number
```

```python
sum_of_values = reduce(sum, values, 0)
```

如果回调函数非常简单和简短，就像 *doubled* 和 *squared* 函数一样，我们可以在 Python 列表推导式中内联它们，使它们更短：

```python
doubled_values = [2 * value for value in values]
squared_values = [value**2 for value in values]
```

让我们看一个用 Clojure 编写的例子：

```clojure
(defn print-first-n-doubled-integers [n]
  (println (take n (map (fn [x] (* 2 x)) (range)))))
```

要理解上面代码中发生了什么，你应该从最内层的函数调用开始阅读，然后向外层的函数调用进行。在遍历函数调用层次结构时，困难在于将所有嵌套函数调用的信息存储和保留在短期记忆中。

我们可以通过给匿名函数命名并为中间函数调用结果引入变量（常量）来简化阅读上面的例子。当然，我们的代码变得更长了，但编码不是一场编写尽可能短代码的比赛，而是编写最短、最易读、最易于他人及未来自己理解的代码。将下面较长的代码编译成与上面较短代码同样高效的代码，是编译器的工作。
以下是上述代码的重构版本：

```clojure
(defn print-first-n-doubled-integers [n]
  (let [doubled (fn [x] (* 2 x))
        doubled-integers (map doubled (range))
        first-n-doubled-integers (take n doubled-integers)]
    (println first-n-doubled-integers)))
```

让我们假设一下：如果 Clojure 的 `map` 函数参数顺序不同，`range` 函数被命名为 `integers`，而 `take` 函数被命名为 `take-first`（就像 `take-last` 一样），我们将得到一个更明确的原始代码版本：

```clojure
(defn print-first-n-doubled-integers [n]
  (let [doubled (fn [x] (* 2 x))
        doubled-integers (map (integers) doubled)
        first-n-doubled-integers (take-first n doubled-integers)]
    (println first-n-doubled-integers)))
```

## 5.1.12：类属性命名

类属性的命名应避免在属性名中重复类名。以下是错误命名的示例：

```python
class Order:
    def __init__(self, order_id: int, order_state: OrderState):
        self.__order_id = order_id
        self.__order_state = order_state
```

以下是修正了命名的代码：

```python
class Order:
    def __init__(self, id_: int, state: OrderState):
        self._id = id_
        self._state = state
```

如果你有一个用于存储回调函数（例如，事件处理器或生命周期回调）的类属性，你应该将其命名得能够说明在什么情况下调用存储的回调函数。使用以下模式命名存储事件处理器的属性：`on` + `<event-type>`，例如 `on_click` 或 `on_submit`。以类似命名生命周期方法的方式命名存储生命周期回调的属性，例如：`on_init`、`after_mount` 或 `before_mount`。

## 5.1.13：通用命名规则

### 5.1.13.1：使用简短、常见的名称

为某物命名时，使用最常见的最短名称。如果你有一个名为 *relinquish_something* 的函数，请考虑为该函数使用一个更短、更常见的名称。例如，你可以将函数重命名为 *release_something*。单词“release”比“relinquish”更短、更常见。使用 Google 搜索单词同义词，例如“relinquish synonym”，以找到最短、最常见的类似术语。

### 5.1.13.2：选择一个术语并保持一致使用

假设你正在构建一个数据导出微服务，并且你当前在代码中使用以下术语：*message*、*report*、*record* 和 *data*。与其使用四个不同的术语来描述同一件事，你应该只选择一个术语，例如 *message*，并在整个微服务代码中保持一致地使用它。

### 5.1.13.3：避免晦涩的缩写

许多缩写被广泛使用，例如 *str* 表示字符串，*num/nbr* 表示数字，*prop* 表示属性，或 *val* 表示值。大多数程序员都使用这些缩写，我也使用它们来缩短长名称。如果变量名很短，应该使用全名，例如 *number_of_items* 而不是 *nbr_of_items*。仅在变量名因此变得过长（20个或更多字符）的情况下使用缩写。我特别注意避免使用不常见的缩写。例如，我绝不会将 *amount* 缩写为 *amnt* 或将 *discount* 缩写为 *dscnt*，因为我在现实生活中没有见过这些缩写被广泛使用。

### 5.1.13.4：避免过短或无意义的名称

过短的名称无法传达变量的用途。避免使用单字符变量名，如以下启动五个线程的示例：

```python
for i in range(1, 6):
    start_thread(i)
```

相反，使用适当的变量名来指示循环计数器的用途：

```python
for thread_number in range(1, 6):
    start_thread(thread_number)
```

如果你不需要在循环内部使用循环计数器的值，你可以使用下划线作为循环变量名，以表示它未被使用。以下循环执行 `object_count` 次：

```python
for _ in range(object_count):
    objects.append(acquire_object())
```

## 5.2：统一源代码仓库结构原则

> 以某种方式系统地组织源代码仓库中的代码，使其他开发人员能够轻松快速地发现所需信息。

你可以为每个技术栈创建包含入门项目的源代码仓库，以确保仓库结构的统一性。以下是为 Python 微服务组织源代码仓库的示例。在下面的示例中，假设是一个容器化（Docker）的微服务，部署到 Kubernetes 集群。你的 CI 工具可能要求 CI 流水线代码必须位于特定目录中。但如果不是，请将 CI 流水线代码放在 *ci* 目录中。

```
my-python-service
├── ci
│   └── Jenkinsfile
├── docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs
├── env
│   ├── .env.dev
│   └── .env.ci
├── helm
│   └── my-python-service
│       ├── templates
│       ├── .helmignore
│       ├── Chart.yaml
│       ├── values.schema.json
│       └── values.yaml
├── integration-tests
│   ├── features
│   │   └── feature1.feature
│   └── steps
├── scripts
│   └── // Bash scripts here...
├── src
├── venv
├── .gitignore
├── .pylintrc
└── README.MD
```

通常，单元测试应位于与源代码模块相同的目录中，但你也可以将它们放在特定的 *test* 目录中。

## 5.3：基于领域的源代码结构原则

主要按领域而非技术细节来组织源代码树。每个源代码目录在其抽象级别上应具有单一职责。

以下是微服务 *src* 目录的示例，该目录未按领域组织，而是错误地按技术细节组织：

```
example-service/
└── src/
    ├── controllers/
    │   ├── AController.py
    │   └── BController.py
    ├── entities/
    │   ├── AEntity.py
    │   └── BEntity.py
    ├── errors/
    │   ├── AError.py
    │   └── BError.py
    ├── dtos/
    │   ├── ADto.py
    │   └── BDto.py
    ├── repositories/
    │   ├── ARepository.py
    │   └── BRepository.py
    └── services/
        ├── AService.py
        └── BService.py
```

以下是修改后的示例，目录按领域组织：

```
example-service/
└── src/
    ├── domainA/
    │   ├── AController.py
    │   ├── ADto.py
    │   ├── AEntity.py
    │   ├── AError.py
    │   ├── ARepository.py
    │   └── AService.py
    └── domainB/
        ├── BController.py
        ├── BDto.py
        ├── BEntity.py
        ├── BError.py
        ├── BRepository.py
        └── BService.py
```

你可以有多层嵌套的领域：

```
example-service/
└── src/
    ├── domainA/
    │   ├── domainA-1/
    │   │   ├── A1Controller.py
    │   │   └── ...
    │   └── domainA-2/
    │       ├── A2Controller.py
    │       └── ...
    └── domainB/
        ├── BController.py
        └── ...
```

如果你愿意，可以在领域目录内为技术细节创建子目录。如果领域目录将包含超过 5 到 7 个文件，这是推荐的方法。以下是 *salesitem* 领域的示例：

```
sales-item-service
└── src
    └── salesitem
        ├── dtos
        │   ├── InputSalesItem.py
        │   └── OutputSalesItem.py
        ├── entities
        │   └── SalesItem.py
        ├── errors
        │   ├── SalesItemServiceError.py
        │   ├── Error1.py
        │   └── Error2.py
        ├── repository
        │   ├── SalesItemRepository.py
        │   └── SalesItemRepositoryImpl.py
        ├── service
        │   ├── SalesItemService.py
        │   └── SalesItemServiceImpl.py
        └── SalesItemController.py
```

为了强调 *clean microservice design principle*，我们也可以使用以下类型的目录布局：

```
sales-item-service
└── src
    └── salesitem
        └── businesslogic
            ├── dtos
            │   ├── InputSalesItem.py
            │   └── OutputSalesItem.py
            ├── entities
            │   └── SalesItem.py
            ├── errors
            │   ├── SalesItemServiceError.py
            │   ├── Error1.py
            │   └── Error2.py
            └── repository
                └── SalesItemRepository.py
```

## 服务

```
service
├── SalesItemService.py
└── SalesItemServiceImpl.py
SalesItemController.py
SalesItemRepositoryImpl.py
```

或者，如果我们有多个控制器和接口适配器：

```
sales-item-service
└── src
    └── salesitem
        ├── businesslogic
        │   ├── dtos
        │   │   ├── InputSalesItem.py
        │   │   └── OutputSalesItem.py
        │   ├── entities
        │   │   └── SalesItem.py
        │   ├── errors
        │   │   ├── SalesItemServiceError.py
        │   │   ├── Error1.py
        │   │   └── Error2.py
        │   ├── repository
        │   │   └── SalesItemRepository.py
        │   └── service
        │       ├── SalesItemService.py
        │       └── SalesItemServiceImpl.py
        ├── controllers
        │   ├── FlaskRestSalesItemController.py
        │   └── AriadneGraphQlSalesItemController.py
        └── ifadapters
            ├── SqlSalesItemRepository.py
            └── MongoDbSalesItemRepository.py
```

在上面的例子中，遵循*清洁微服务设计*原则，如果你添加或更改一个控制器或接口适配器，你不需要对服务的业务逻辑部分进行任何更改。

下面是上一章设计的数据导出器微服务的源代码目录结构。有四个子域的子目录：输入、内部消息、转换器和输出。每个类名中的公共命名符都有一个子目录。在定位特定文件时，浏览目录树非常轻松。此外，每个目录中的源代码文件数量很少。你可以一目了然地掌握一个目录的内容。包含许多文件的目录的问题是不容易找到想要的文件。因此，一个目录理想情况下应该有2-4个文件。绝对最大值是5-7个文件。

请注意，下面有几个目录未展开以缩短示例。读者应该很容易推断出未展开目录的内容。

```
src
├── common
├── input
│   ├── config
│   │   ├── parser
│   │   │   ├── InputConfigParser.py
│   │   │   └── JsonInputConfigParser.py
│   │   ├── reader
│   │   │   ├── InputConfigReader.py
│   │   │   └── LocalFileSystemInputConfigReader.py
│   │   ├── InputConfig.py
│   │   └── InputConfigImpl.py
│   └── message
│       ├── consumer
│       │   ├── InputMsgConsumer.py
│       │   └── KafkaInputMsgConsumer.py
│       ├── decoder
│       │   ├── InputMsgDecoder.py
│       │   └── AvroBinaryInputMsgDecoder.py
│       ├── InputMessage.java
│       └── KafkaInputMessage.java
├── internalmessage
│   ├── field
│   ├── InternalMessage.java
│   └── InternalMessageImpl.java
├── transformer
│   ├── config
│   ├── field
│   │   ├── impl
│   │   │   ├── CopyFieldTransformer.py
│   │   │   ├── ExprFieldTransformer.py
│   │   │   ├── FilterFieldTransformer.py
│   │   │   └── TypeConvFieldTransformer.py
│   │   ├── FieldTransformer.py
│   │   ├── FieldTransformers.py
│   │   └── FieldTransformersImpl.py
│   └── message
│       ├── MsgTransformer.py
│       └── MsgTransformerImpl.py
└── output
    ├── config
    └── message
        ├── encoder
        └── producer
```

我们也可以按照*清洁微服务设计*以以下方式组织代码：

```
src
├── common
└── businesslogic
    ├── input
    │   ├── config
    │   │   ├── InputConfig.py
    │   │   ├── InputConfigImpl.py
    │   │   ├── InputConfigParser.py
    │   │   └── InputConfigReader.py
    │   └── message
    │       ├── InputMessage.java
    │       ├── InputMsgConsumer.py
    │       └── InputMsgDecoder.py
    ├── internalmessage
    │   ├── field
    │   ├── InternalMessage.java
    │   └── InternalMessageImpl.java
    ├── transformer
    │   ├── config
    │   ├── field
    │   │   ├── impl
    │   │   │   ├── CopyFieldTransformer.py
    │   │   │   ├── ExprFieldTransformer.py
    │   │   │   ├── FilterFieldTransformer.py
    │   │   │   └── TypeConvFieldTransformer.py
    │   │   ├── FieldTransformer.py
    │   │   ├── FieldTransformers.py
    │   │   └── FieldTransformersImpl.py
    │   └── message
    │       ├── MsgTransformer.py
    │       └── MsgTransformerImpl.py
    └── output
        ├── config
        └── message
└── ifadapters
    ├── config
    │   ├── parser
    │   │   └── json
    │   │       ├── JsonInputConfigParser.py
    │   │       ├── JsonTransformerConfigParser.py
    │   │       └── JsonOutputConfigParser.py
    │   └── reader
    │       └── localfilesystem
    │           ├── LocalFileSystemInputConfigReader.py
    │           ├── LocalFileSystemTransformerConfigReader.py
    │           └── LocalFileSystemOutputConfigReader.py
    ├── input
    │   ├── kafka
    │   │   ├── KafkaInputMsgConsumer.py
    │   │   └── KafkaInputMessage.java
    │   └── AvroBinaryInputMsgDecoder.py
    └── output
        ├── CsvOutputMsgEncoder.py
        └── PulsarOutputMsgProducer.py
```

从上面的目录结构我们可以轻松地看到以下内容：

- 配置是JSON格式的，从本地文件系统读取
- 作为输入，从Kafka读取avro二进制消息
- 对于输出，CSV记录被生成到Apache Pulsar

我们想要/需要在*ifadapters*目录中进行的任何更改都不应影响*businesslogic*目录中的业务逻辑部分。

下面是上一章设计的异常检测微服务的源代码目录结构。*anomaly*目录已展开。我们可以看到我们的实现使用JSON进行各种解析活动，并使用自组织映射（SOM）进行异常检测。JSON和Kafka用于在微服务外部发布异常指标。向下面的目录结构添加新的具体实现是直接的。例如，如果我们想为配置文件添加YAML支持，我们可以创建*yaml*子目录，在其中放置特定于YAML的实现类。

```
src
├── anomaly
│   ├── detection
│   │   ├── configuration
│   │   │   ├── factory
│   │   │   │   ├── AnomalyDetectionConfigFactory.py
│   │   │   │   └── AnomalyDetectionConfigFactoryImpl.py
│   │   │   ├── parser
│   │   │   │   ├── AnomalyDetectionConfigParser.py
│   │   │   │   └── JsonAnomalyDetectionConfigParser.py
│   │   │   ├── AnomalyDetectionConfig.py
│   │   │   └── AnomalyDetectionConfigImpl.py
│   │   ├── engine
│   │   │   ├── AnomalyDetectionEngine.py
│   │   │   └── AnomalyDetectionEngineImpl.py
│   │   ├── rule
│   │   │   ├── factory
│   │   │   │   ├── AnomalyDetectionRuleFactory.py
│   │   │   │   └── AnomalyDetectionRuleFactoryImpl.py
│   │   │   ├── parser
│   │   │   │   ├── AnomalyDetectionRuleParser.py
│   │   │   │   └── AnomalyDetectionRuleParserImpl.py
│   │   │   ├── AnomalyDetectionRule.py
│   │   │   └── AnomalyDetectionRuleImpl.py
│   │   ├── AnomalyDetector.py
│   │   └── AnomalyDetectorImpl.py
│   ├── indicator
│   │   ├── factory
│   │   │   ├── AnomalyIndicatorFactory.py
│   │   │   └── AnomalyIndicatorFactoryImpl.py
│   │   ├── publisher
│   │   │   ├── AnomalyIndicatorPublisher.py
│   │   │   └── KafkaAnomalyIndicatorPublisher.py
│   │   ├── serializer
│   │   │   ├── AnomalyIndicatorSerializer.py
│   │   │   └── JsonAnomalyIndicatorSerializer.py
│   │   ├── AnomalyIndicator.py
│   │   └── AnomalyIndicatorImpl.py
│   └── model
│       ├── factory
│       │   └── AnomalyModelFactory.py
│       ├── AnomalyModelFactoryImpl.py
│       ├── training
│       │   ├── engine
│       │   │   ├── AnomalyModelTrainingEngine.py
│       │   │   └── AnomalyModelTrainingEngineImpl.py
│       │   ├── AnomalyModelTrainer.py
│       │   └── SomAnomalyModelTrainer.py
│       ├── AnomalyModel.py
│       └── SomAnomalyModel.py
├── common
├── measurement
└── app.py
```

对于全栈Python开发者，让我们再看一个*data-visualization-web-client*的例子。这个Web客户端的UI由以下页面组成，所有页面都包含一个公共头部：

- 仪表板
- 数据浏览器
- 警报

*仪表板*页面包含一个仪表板组选择器、仪表板选择器和图表区域，用于显示所选仪表板的图表。你可以通过先选择一个仪表板组，然后从该组中选择一个仪表板来选择显示的仪表板。

## 5.4：避免注释原则

*避免在代码中添加注释。唯一的例外是为库的公共 API 编写文档。*

注释可能存在问题。你无法百分之百地信任它们，因为它们可能具有误导性、已过时或完全错误。你只能信任源代码本身。注释通常完全不必要，只会使代码更加冗长。允许注释可能导致代码包含命名糟糕的变量，并通过附加注释来解释；代码通常也包含过长的函数，其中功能块通过附加注释来描述，而不是通过提取命名良好的函数来重构代码。以下几节将描述几种避免编写注释同时保持代码可读性的方法。可以采取以下措施来避免编写注释：

-   为类、函数和变量等事物正确命名
    -   例如，如果你正在使用某种算法，不要在注释中记录该算法，而是将相应的类/函数命名为包含算法名称的形式。这样，读者如果不熟悉该算法，就可以通过名称进行搜索。
-   不应添加关于变量/函数类型的注释。在所有地方使用类型注解。
-   你不需要注释说明函数可能引发错误。使用本章后面描述的函数名 *try* 前缀约定。
-   不要为一段代码添加注释，而是提取一个新的、命名良好的函数。
-   保持函数小巧，这样它们更容易理解，因为它们不可能包含过于复杂的逻辑，从而需要注释来解释。
-   不要添加可以从版本控制系统获取的信息作为注释。
-   不要注释掉代码。只需删除未使用的代码。删除的代码将永远保存在版本控制系统中。
-   你不必注释函数使用的逻辑。代码读者应该能够从代码本身推断出这些信息，此外还可以从相关的单元测试中推断。如果你实践了 TDD，并且为相关函数提供了一套完整的、命名良好的单元测试，那么复杂的代码逻辑和行为通常不需要注释。

库的公共 API 需要注释，因为库需要可以从注释自动生成的 API 文档，以避免 API 注释和文档不同步的情况。在非库软件组件中，通常不需要 API 文档，因为你可以访问 API 接口、实现和单元测试。例如，单元测试指定了函数在不同场景下的行为。单元测试名称说明了场景，单元测试代码中的期望和断言说明了在特定情况下的预期行为。库用户通常无法访问 API 实现和单元测试，即使可以，用户也不应依赖它们，因为它们是可能发生变化的内部细节。

## 5.4.1：正确命名事物

当你为函数等事物命名不当时，你可能最终会为该函数附加注释。为了避免编写注释，必须专注于正确命名事物。遵循*单一职责原则*和*统一命名原则*时，应该更容易正确命名事物并避免注释。下面是一个带有注释的函数示例：

```
class MessageBuffer:
    # Return False if buffer full,
    # True if message written to buffer
    def write(self, message: Message) -> bool:
        # ...
```

如果我们去掉注释，我们将得到以下代码：

```
class MessageBuffer:
    def write(self, message: Message) -> bool:
        # ...
```

仅仅去掉注释并不是最佳解决方案，因为一些关键信息现在缺失了。这个布尔返回值意味着什么？这并不完全清楚。我们可以假设返回 `True` 表示消息已成功写入，但对于返回 `False` 没有任何说明。我们只能假设是某种错误，但不确定是什么错误。

除了移除注释，我们还应该为函数提供一个更好的名称，并将其重命名如下：

```
class MessageBuffer:
    def write_if_buf_not_full(self, message: Message) -> bool:
        pass
```

现在函数的目的很明确，我们可以确定布尔返回值的含义。它表示消息是否已写入缓冲区。现在我们也知道为什么写入消息可能会失败：缓冲区已满。这将为函数调用者提供足够的信息，以便决定下一步该做什么。它可能应该等待一段时间，以便缓冲区读取器有足够的时间从缓冲区读取消息并释放一些空间。

下面是一个 C++ 代码的真实示例，其中注释与函数名不匹配：

```
/**
 * @brief Add new counter or get existing, if same labels used already.
 * @param counterName Name of the counter
 * @param help Help text added for counter, if new countername
 * @param labels Specific labels for counter.
 * @return counter pointer used when increasing counter, or nullptr
 *         if metrics not initialized or invalid name or labels
 */
static prometheus::Counter* addCounter(
    std::string counterName,
    std::string help,
    const std::map<std::string, std::string>& labels);
```

在上面的示例中，函数名表示它添加一个计数器，但注释说它添加或获取一个现有的计数器。真正的问题是，一旦有人首先读取函数名 `addCounter`，他/她不一定会阅读注释中的“简要说明”，因为当他/她读取函数名时，立即就明白了函数的功能：它应该添加一个计数器。作为解决方案，我们可以改进函数名称为 `addOrGetExistingCounter`。

下面是我曾经读过的一本书中的一个真实示例：

```
from typing import Protocol

from Person import Person


class Mediator(Protocol):
    # To register an employee
    def register(self, person: Person) -> None:
        pass

    # To send a message from one employee to another employee
    def connect_employees(
        self,
        from_person: Person,
        to_person: Person,
        msg: str
    ) -> None:
        pass

    # To display currently registered members
    def display_detail(self) -> None:
        pass
```

在上面的示例中有三个函数，每个函数都有问题。第一个函数是注册一个人，但注释说它注册一名员工。因此，注释与代码之间存在不匹配。在这种情况下，我更相信代码而不是注释。修正方法是移除注释，因为它没有带来任何价值，只会引起混淆。

第二个函数在注释中说它从一个员工向另一个员工发送消息。函数名是关于连接员工的，但参数是人员。我假设注释的一部分是正确的：从某人向另一人发送消息。但再次，我更相信代码而不是注释，并假设消息是从一个人发送给另一个人的。我们应该移除注释并重命名该函数。

在第三个函数中，注释补充了函数名所缺失的信息。注释还讨论了成员，因为代码的其他部分提到了员工和人员。这里使用了三个不同的术语：员工、人员和成员。应该只选择一个术语。我们选择术语*人员*并系统地使用它。

以下是重构后不带注释的版本：

```python
class Mediator(Protocol):
    def register(self, person: Person) -> None:
        pass

    def send(message: str, sender: Person, recipient: Person) -> None:
        pass

    def display_details_of_registered_persons(self) -> None:
        pass
```

## 5.4.2：函数末尾返回单个命名值

> 函数应该只有一个返回语句，并在函数末尾返回一个命名值。这样，代码阅读者可以通过查看函数末尾来推断返回值的含义。

考虑以下示例：

```python
from typing import Protocol

from CounterFamily import CounterFamily

class Metrics(Protocol):
    # ...

    def add_counter(
        self,
        counter_family: CounterFamily,
        labels: dict[str, str]
    ) -> int:
        pass

    def increment_counter(
        self,
        counter_index: int,
        increment_amount: int
    ) -> None:
        pass

    # def add_gauge ...
    # def set_gauge_value ...
```

`add_counter` 函数的返回值是什么？有人可能认为需要一个注释来描述返回值，因为不清楚 `int` 意味着什么。与其写注释，我们可以引入一个命名值（= 变量/常量）从函数返回。命名返回值背后的理念是，它无需注释就能传达返回值的语义。以下是 `add_counter` 函数的实现：

```python
from typing import Protocol

from CounterFamily import CounterFamily

class Metrics(Protocol):
    def add_counter(
        self,
        counter_family: CounterFamily,
        labels: dict[str, str]
    ) -> int:
        # 在此处执行添加计数器的操作，
        # 并为 'counter_index' 变量设置值
        return counter_index
```

在上述实现中，我们在函数末尾有一个返回命名值的单一返回语句。我们所要做的就是查看函数末尾并找到返回语句，它应该告诉我们神秘的 `int` 类型返回值的含义：它是一个计数器索引。而且我们可以发现 `increase_counter` 函数需要一个 `counter_index` 参数，这就建立了先调用 `add_counter` 函数、存储返回的计数器索引，然后在调用 `increase_counter` 函数时使用该存储的计数器索引之间的联系。

## 5.4.3：返回类型别名

在前面的例子中，`add_counter` 函数有一个神秘的 `int` 类型返回值。我们学习了如何通过在函数末尾引入一个命名返回值来帮助传达返回值的语义。但有一种更好的方式来传达返回值的语义。我们可以使用类型别名。下面是一个例子，我们为 `int` 类型引入了一个 `CounterIndex` 类型别名：

```python
from typing import Protocol

from CounterFamily import CounterFamily

CounterIndex = int

class Metrics(Protocol):
    # ...

    def add_counter(
        self,
        counter_family: CounterFamily,
        labels: dict[str, str]
    ) -> CounterIndex:
        pass

    def increment_counter(
        self,
        counter_index: CounterIndex,
        increment_amount: int
    ) -> None:
        pass

    # def add_gauge ...
    # def set_gauge_value ...
```

我们可以改进上面的指标示例。首先，我们应该避免原始类型痴迷。我们不应该从 `add_counter` 方法返回一个索引，而应该将该方法重命名为 `create_counter` 并从该方法返回一个 `Counter` 类的实例。然后，我们应该通过将 `increment_counter` 方法移动到 `Counter` 类并将其重命名为 `increment` 来使示例更具面向对象性。此外，`Metrics` 类本身的名称也应该更改为 `MetricFactory`。

## 5.4.4：为布尔表达式提取常量

通过为布尔表达式提取一个常量，我们可以消除注释。下面是一个例子，其中在 if 语句及其布尔表达式下方写了一个注释：

```python
from Message import Message

class MessageBuffer:
    def write_if_buf_not_full(self, message: Message) -> bool:
        message_was_written = False
        if len(self.__messages) < self.__max_length:
            # 缓冲区未满
            self.__messages.append(message)
            message_was_written = True
        return message_was_written
```

通过引入一个常量用于“缓冲区未满”检查，我们可以摆脱“缓冲区未满”注释：

```python
from Message import Message

class MessageBuffer:
    def write_if_buf_not_full(self, message: Message) -> bool:
        message_was_written = False
        buffer_is_not_full = len(self.__messages) < self.__max_length
        if buffer_is_not_full:
            self.__messages.append(message)
            message_was_written = True
        return message_was_written
```

## 5.4.5：提取命名常量或枚举类型

如果你在代码中遇到*魔法数字*，你应该为该值引入一个命名常量或枚举类型（enum）。在下面的例子中，我们返回了两个魔法数字，0 和 1：

```python
import sys

from Application import Application

application = Application()

if application.run():
    # 应用程序运行成功
    sys.exit(0)

# 退出码：失败
sys.exit(1)
```

让我们引入一个枚举类型 `ExitCode`，并用它代替魔法数字：

```python
import sys
from enum import IntEnum

from Application import Application

class ExitCode(IntEnum):
    Success = 0
    Failure = 1

application = Application()
app_was_successfully_run = application.run()
exit_code = (
    ExitCode.Success if app_was_successfully_run else ExitCode.Failure
)
sys.exit(exit_code)
```

现在，如果需要，以后可以轻松添加更多具有描述性名称的退出码。

## 5.4.6：提取函数

如果你打算在一段代码上方写注释，你应该将那段代码提取到一个新函数中。当你提取一个命名良好的函数时，你就不需要写那个注释了。新提取的函数的名称就充当了文档。下面是一个带有一些注释代码的例子：

```python
from Message import Message

class MessageBuffer:
    def write_fitting(self, messages: list[Message]) -> None:
        if len(self.__messages) + len(messages) <= self.__max_length:
            # 所有消息都适合放入缓冲区
            self.__messages.extend(messages)
            messages.clear()
        else:
            # 并非所有消息都适合，只写入适合的消息
            nbr_of_msgs_that_fit = self.__max_length - len(messages)
            self.__messages.extend(messages[:nbr_of_msgs_that_fit])
            del messages[:nbr_of_msgs_that_fit]
```

这是通过提取两个新方法重构掉注释后的相同代码：

```python
from Message import Message

class MessageBuffer:
    def write_fitting(self, messages: list[Message]) -> None:
        all_messages_fit = len(self.__messages) + len(messages) <= self.__max_length
        if all_messages_fit:
            self.__write_all(messages)
        else:
            self.__write_only_fitting(messages)

    def __write_all(self, messages: list[Message]) -> None:
        self.__messages.extend(messages)
        messages.clear()

    def __write_only_fitting(self, messages: list[Message]) -> None:
        nbr_of_msgs_that_fit = self.__max_length - len(messages)
        self.__messages.extend(messages[:nbr_of_msgs_that_fit])
        del messages[:nbr_of_msgs_that_fit]
```

## 5.4.7：避免在 Bash Shell 脚本中使用注释

许多程序员，包括我自己，都不喜欢 Linux shell 命令和脚本的神秘语法。即使是最简单的表达式，如果你不经常使用脚本，其语法也可能难以理解和记忆。当然，最好的办法是避免编写复杂的 Linux shell 脚本，而是使用像 Python 这样的合适编程语言。但有时，执行使用 shell 脚本执行某些操作会更简单。由于 shell 脚本中的语法和命令可能难以理解，许多开发者倾向于通过在脚本中添加注释来解决问题。

接下来，将介绍无需注释即可使脚本更易理解的替代方法。让我们考虑以下来自我遇到过的一个真实脚本的例子：

```
create_network() {
    #create only if not existing yet
    if [[ -z "$(docker network ls | grep $DOCKER_NETWORK_NAME )" ]];
    then
        echo Creating $DOCKER_NETWORK_NAME
        docker network create $DOCKER_NETWORK_NAME
    else
        echo Network $DOCKER_NETWORK_NAME already exists
    fi
}
```

以下是相同示例，但做了以下更改：

-   注释被移除，之前被注释的表达式被移至一个命名良好的函数中
-   表达式中的否定被移除，*then* 和 *else* 分支的内容被交换
-   变量名改为驼峰命名法以增强可读性

```
dockerNetworkExists() { [[ -n "$(docker network ls | grep $1 )" ]]; }

createDockerNetwork() {
    if dockerNetworkExists $networkName; then
        echo Docker network $networkName already exists
    else
        echo Creating Docker network $networkName
        docker network create $networkName
    fi
}
```

如果你的脚本接受参数，请为参数赋予合适的名称，例如：

```
dataFilePathName=$1
schemaFilePathName=$2
```

脚本阅读者无需记住 `$1` 或 `$2` 的含义，你也无需插入任何注释来澄清参数的意义。

如果你的 Bash shell 脚本中有一个复杂的命令，你不应该给它附加注释，而应该提取一个具有合适名称的函数来描述该命令。

以下示例包含一个注释：

```
# Update version in Helm Chart.yaml file
sed -i "s/^version:.*/version: $VERSION/g" helm/service/Chart.yaml
```

以下是上述示例重构后包含一个函数的版本：

```
updateHelmChartVersionInChartYamlFile() {
  sed -i "s/^version:.*/version: $1/g" helm/service/Chart.yaml
}

updateHelmChartVersionInChartYamlFile $version
```

这是另一个例子：

```
getFileLongestLineLength() {
  echo $(awk '{ if (length($0) > max) max = length($0) } END { print max }' $1)
}

configFileLongestLineLength = $(getFileLongestLineLength $configFilePathName)
```

## 5.5：函数单一返回原则

> 优先在函数末尾使用单个 return 语句，以清晰传达返回值的含义，并使函数重构更容易。

在函数末尾使用带有命名值的单个 return 语句，如果返回值类型本身不能直接传达其含义，那么这可以清晰地传达返回值的语义。例如，如果你从一个函数返回一个原始类型（如整数或布尔值）的值，其含义不一定 100% 清晰。但当你在函数末尾返回一个命名值时，返回变量的名称就传达了其语义。

你可能认为，无法在函数中间返回值会因为大量的嵌套 if 语句而降低函数的可读性。这是可能的，但应该记住，一个函数应该很小。目标是单个函数中最多有 5-9 行语句。遵循这个规则，你永远不会在单个函数中遇到*嵌套 if 语句的地狱*。

在函数末尾使用单个 return 语句使函数重构更容易。你可以使用 IDE 提供的自动重构工具。从包含 return 语句的代码中提取新函数总是更困难的。对于包含 *break* 或 *continue* 语句的循环也是如此。重构不包含 break 或 continue 语句的循环内部代码更容易。

在某些情况下，在函数末尾返回单个值使代码更直接，并且需要更少的代码行数。

以下是一个具有两个返回位置的函数示例：

```
from threading import Thread

from InputMessage import InputMessage

class TransformThread(Thread):
    # ...

    def transform(self, input_message: InputMessage) -> bool:
        output_message = self.__output_message_pool.acquire_message()
        (
            msg_was_transformed,
            msg_is_filtered_in,
        ) = self.__message_transformer.transform(
            input_message, output_message
        )

        if msg_was_transformed and msg_is_filtered_in:
            self.__output_messages.append(output_message)
        else:
            self.__output_message_pool.return_message(output_message)
            if not msg_was_transformed:
                return False

        return True
```

分析上述函数时，我们注意到它将输入消息转换为输出消息。我们可以得出结论，该函数在消息转换成功时返回 *True*。我们可以通过重构使其只包含一个 return 语句来缩短该函数。重构后，函数返回值的含义就 100% 清晰了。

```
from threading import Thread

from InputMessage import InputMessage

class TransformThread(Thread):
    # ...

    def transform(self, input_message: InputMessage) -> bool:
        output_message = self.__output_message_pool.acquire_message()
        (
            msg_was_transformed,
            msg_is_filtered_in,
        ) = self.__message_transformer.transform(
            input_message, output_message
        )

        if msg_was_transformed and msg_is_filtered_in:
            self.__output_messages.append(output_message)
        else:
            self.__output_message_pool.return_message(output_message)

        return msg_was_transformed
```

作为此规则的一个例外，当函数长度最优且如果重构为包含单个 return 语句会变得太长时，你可以在函数中有多个 return 语句。

此外，要求返回值的语义含义从函数名称或函数的返回类型中清晰可知。以下是一个具有多个 return 语句的函数示例。从函数名称也能清楚知道返回值的含义。同时，函数的长度是最优的：七条语句。

```
from typing import Protocol, TypeVar

T = TypeVar('T')

class MyIterator(Protocol[T]):
    def has_next_item(self) -> bool:
        pass

    def get_next_item(self) -> T:
        pass

def are_equal(
    iterator: MyIterator[T],
    another_iterator: MyIterator[T]
) -> bool:
    while iterator.has_next_item():
        if another_iterator.has_next_item():
            if (
                iterator.get_next_item()
                != another_iterator.get_next_item()
            ):
                return False
        else:
            return False

    return True
```

如果我们重构上述代码以包含单个 return 语句，代码会变得太长（10 条语句）而无法放在一个函数中，如下所示。在这种情况下，我们应该优先选择上面的代码而不是下面的代码。

```
def are_equal(
    iterator: MyIterator[T],
    another_iterator: MyIterator[T]
) -> bool:
    iters_are_equal = True

    while iterator.has_next_item():
        if another_iterator.has_next_item():
            if (
                iterator.get_next_item()
                != another_iterator.get_next_item()
            ):
                iters_are_equal = False
                break
        else:
            iters_are_equal = False
            break

    return iters_are_equal
```

作为此规则的第二个例外，你可以在工厂中使用多个返回位置，因为你可以从工厂名称知道它创建的对象类型。以下是一个具有多个 return 语句的工厂示例：

```
from enum import Enum

class CarType(Enum):
    AUDI = 1
    BMW = 2
    MERCEDES_BENZ = 3

class Car:
    # ...

class Audi(Car):
    # ...

class Bmw(Car):
    # ...

class MercedesBenz(Car):
    # ...

class CarFactory:
    def create_car(self, car_type: CarType) -> Car:
        match car_type:
            case CarType.AUDI:
                return Audi()
            case CarType.BWM:
                return Bmw()
            case CarType.MERCEDES_BENZ:
                return MercedesBenz()
            case _:
                raise ValueError('Invalid car type')
```

## 5.6：生产代码使用类型注解原则

> 在实现生产软件时使用类型注解。对于非生产代码（如集成、端到端和自动化非功能测试），你可以使用无类型的 Python。

对于一个简单的软件组件，你可以不用类型来处理，但当它变得更大并且有更多人参与时，静态类型的好处就变得显而易见了。

让我们分析使用无类型语言可能带来的潜在问题：

## 5.6：类型注解的好处

-   函数参数可能被错误地排序
-   函数参数可能被赋予错误的类型
-   函数返回值类型可能被误解
-   重构代码变得更加困难
-   被迫编写公共 API 注释来描述函数签名
-   类型错误不一定能在测试中发现

### 5.6.1：函数参数可能被错误地排序

当不使用类型注解时，你可能会意外地以错误的顺序提供函数参数。当你使用类型注解时，这类错误就不太常见了。现代 IDE 可以在函数调用处显示内联参数提示。这是你应该考虑在 IDE 中启用的功能。这些参数提示可能会揭示函数参数未按正确顺序给出的情况。

### 5.6.2：函数参数可能被赋予错误的类型

当不使用类型注解时，你可能会给函数参数赋予错误的类型。例如，一个函数需要数字的字符串表示，但你提供了一个数字。正确命名函数参数会有所帮助。与其将字符串参数命名为 *amount*，不如将其命名为 *amount_string* 或 *amount_as_string*。

### 5.6.3：函数返回值类型可能被误解

确定函数返回值类型可能很困难。仅从函数名称不一定能 100% 明确。例如，如果你有一个名为 `get_value` 的函数，其返回值类型并非 100% 明确。只有在你非常了解函数上下文的情况下，它才可能显而易见。作为改进，函数应该被恰当地命名，例如：如果返回值始终是字符串，则命名为 `get_value_as_string()`。如果从函数名称看返回值类型不明确，你必须分析函数的源代码来确定返回值类型。这是不必要的、容易出错的手动工作，可以通过使用函数返回类型注解来避免。

### 5.6.4：重构代码变得更加困难

如果你没有类型注解，重构代码通常会更加困难。但当你有类型注解并更改了例如函数参数类型时，你会在调用该函数的代码部分得到类型检查错误。然后重构这些部分就很容易了。但如果你没有类型注解并做了同样的更改，你就必须手动找到所有需要更改的地方，这当然更容易出错。

### 5.6.5：被迫编写公共 API 注释

当不使用类型注解时，你可能被迫使用注释来记录公共 API。这是额外的工作，可以通过使用类型注解来避免。用注释编写 API 文档容易出错。你可能会意外地在 API 文档中写入错误信息，或者在更改 API 代码本身时忘记更新文档。同样，API 文档的读者也可能犯错。他们可能根本不阅读 API 文档。或者他们之前读过，但后来记错了。

### 5.6.6：类型错误在测试中未被发现

这是最大的问题。你可能认为，如果你的代码中存在与函数参数类型正确性相关的错误，测试会揭示这些错误。这通常是一个错误的假设。单元测试不会发现这些问题，因为你在其中模拟了其他类和方法。你只能在集成测试中发现问题，当你集成软件组件时（即测试函数调用其他真实函数而不是模拟函数）。根据测试金字塔，集成测试只覆盖代码库的一个子集，少于单元测试。并且根据集成测试代码的代码覆盖率，一些函数参数顺序或参数/返回值类型正确性问题可能未被测试到，并逃逸到生产环境中。

## 5.7：重构原则

> 你不可能第一次就写出完美的代码，所以你应该总是为未来的重构预留一些时间。

即使你正在为新的软件组件编写代码，你也需要重构。重构不仅仅与遗留代码库有关。如果你不重构，你就是在让技术债务在软件中增长。重构背后的主要思想是，没有人能在第一次就写出完美的代码。重构意味着你在不改变实际功能的情况下更改代码。重构后，大多数测试应该仍然通过，代码组织方式不同，并且你拥有更好的面向对象设计和改进的命名。重构通常不会影响集成测试，但根据重构的类型和规模，可能会影响单元测试。在估算重构工作量时请记住这一点。

我们在计划事情时，不一定预留任何或足够的时间进行重构。当我们为史诗、功能和用户故事提供工作估算时，我们应该意识到重构的需要，并在我们的初始工作估算（不包括重构）中添加一些额外的时间。重构是管理层不一定清楚理解的工作。管理层应该支持重构的需要，即使它没有给最终用户带来明显的附加价值。但它通过不让代码库腐烂和消除技术债务来带来价值。如果你的软件积累了大量技术债务，开发新功能和维护软件的成本就会很高。此外，软件的质量也会降低，这可能表现为许多错误和客户满意度下降。

以下是最常见的代码异味及其解决重构技术的列表：

| 代码异味 | 重构解决方案 |
| :--- | :--- |
| 非描述性名称 | 重命名 |
| 过长方法 | 提取方法 |
| 复杂表达式 | 提取常量 |
| 过长的 switch-case 或 if-elif-else 语句 | 用多态替换条件语句 |
| 过长的参数列表 | 引入参数对象 |
| 散弹式修改 | 用多态替换条件语句 |
| 取反的布尔条件 | 反转 if 语句 |

### 5.7.1：重命名

这可能是使用最广泛的重构技术。你通常第一次无法正确命名，需要进行重命名。现代 IDE 提供了帮助重命名代码中元素的工具：接口、类、函数和变量。IDE 的重命名功能总是比简单的查找和替换方法更好。如果使用查找和替换方法，你可能会意外地重命名不需要重命名的内容，或者没有重命名应该重命名的内容。

### 5.7.2：提取方法

这可能是第二常用的重构技术。当你实现一个类的公共方法时，该方法的代码行数会迅速增长。一个函数最多应包含 5-9 条语句，以保持其可读性和可理解性。当一个公共方法过长时，你应该提取一个或多个私有方法，并从公共方法中调用这些私有方法。每个现代 IDE 都有一个 *提取方法* 重构工具，允许你轻松提取私有方法。选择要提取到新方法的代码行，然后按 IDE 的 *提取方法* 功能快捷键。然后为提取的方法提供一个描述性名称，就完成了。在某些情况下，重构不是自动的。例如，如果要提取的代码包含影响函数执行流的 *return*、*break* 或 *continue* 语句（导致多个返回点）。如果你想保持代码可重构性，请避免使用 *break* 和 *continue* 语句，并且只在函数末尾有一个 return 语句。你可以在 IDE 中完成提取之前，以更好的顺序组织提取方法的参数。

### 5.7.3：提取常量

如果你有一个复杂的表达式（布尔或数值），将表达式的值赋给一个常量。常量的名称传达了关于表达式的信息。下面是一个通过将表达式提取为常量来使 if 语句更易读的示例：

```python
# ...

if (
    data_source_selector_is_open
    and measure_selector_is_open
    and dimension_selector_is_open
):
    data_source_selector.style.height = f'{0.2 * available_height}px'
    measure_selector.style.height =  f'{0.4 * available_height}px'
    dimension_selector.style.height = f'{0.4 * available_height}px'
elif (
    not data_source_selector_is_open
    and not measure_selector_is_open
    and dimension_selector_is_open
):
    dimension_selector.style.height = f'{available_height}px'
```

让我们提取常量：

```python
# ...

all_selectors_are_open = (
    data_source_selector_is_open
    and measure_selector_is_open
    and dimension_selector_is_open
)

only_dimension_selector_is_open = (
    not data_source_selector_is_open
    and not measure_selector_is_open
    and dimension_selector_is_open
)

if all_selectors_are_open:
    data_source_selector.style.height = f'{0.2 * available_height}px'
    measure_selector.style.height =  f'{0.4 * available_height}px'
    dimension_selector.style.height = f'{0.4 * available_height}px'
elif only_dimension_selector_is_open:
    dimension_selector.style.height = f'{available_height}px'
```

下面是一个返回布尔表达式的示例：

## 5.7.4：用多态替换条件语句

假设你在代码中遇到一个大型的 `match-case` 语句或 `if/elif` 结构（不考虑工厂中的代码）。这意味着你的软件组件没有进行恰当的面向对象设计。你应该用多态来替换这些条件语句。当你在软件组件中引入恰当的面向对象设计时，你将功能从 `switch` 语句的 `case` 分支移动到实现了特定接口的不同类中。同样地，你将代码从 `if` 和 `elif` 语句移动到实现了某个接口的不同类中。这样，你就可以消除 `match-case` 和 `if/elif` 语句，并用多态方法调用来替换它们。

下面是一个非面向对象设计的示例：

```python
def do_something_with(chart: Chart):
    if chart.type == 'column':
        # do this ...
    elif chart.type == 'pie':
        # do that ...
    elif chart.type == 'geographic-map':
        # do a third thing ...
```

让我们用多态来替换上述条件语句：

```python
from typing import Protocol

class Chart(Protocol):
    def do_something(self) -> None:
        pass

class ColumnChart(Chart):
    def do_something(self) -> None:
        # do this ...

class PieChart(Chart):
    def do_something(self) -> None:
        # do that ...

class GeographicMapChart(Chart):
    def do_something(self) -> None:
        # do a third thing

def do_something_with(chart: Chart):
    chart.do_something()
```

假设你正在实现一个数据可视化应用程序，并且在代码中有很多地方需要检查图表类型，并且需要引入一个新的图表类型。这可能意味着你必须在代码的许多地方添加一个新的 *case* 或 *elif* 语句。这种方法非常容易出错，被称为 *散弹枪式修改*，因为你需要找到代码库中所有需要修改代码的地方。你应该做的是进行恰当的面向对象设计，并引入一个新的图表类来包含新功能，而不是通过修改多处代码来引入新功能。

## 5.7.5：引入参数对象

如果你的函数有超过 5-7 个参数，你应该引入一个参数对象来减少参数数量，以保持函数签名更具可读性。下面是一个参数过多的构造函数示例：

```python
class KafkaConsumer:
    def __init__(
        self,
        brokers: list[str],
        topics: list[str],
        extra_config_entries: list[str],
        tls_is_used: bool,
        cert_should_be_verified: bool,
        ca_file_path_name: str,
        cert_file_path_name: str,
        key_file_path_name: str
    ):
        # ...
```

让我们将传输层安全（TLS）相关的参数分组到一个名为 `TlsOptions` 的参数类中：

```python
class TlsOptions:
    def __init__(
        self,
        tls_is_used: bool,
        cert_should_be_verified: bool,
        ca_file_path_name: str,
        cert_file_path_name: str,
        key_file_path_name: str
    ):
        # ...
```

现在我们可以修改 `KafkaConsumer` 构造函数以使用 `TlsOptions` 参数类：

```python
from TlsOptions import TlsOptions

class KafkaConsumer:
    def __init__(
        self,
        brokers: list[str],
        topics: list[str],
        extra_config_entries: list[str],
        tls_options: TlsOptions
    ):
        # ...
```

## 5.7.6：反转 If 语句

这是一种现代 IDE 可以为你完成的重构。

下面是一个 Python 示例，其 `if` 语句条件中包含一个否定的布尔表达式。注意这个布尔表达式读起来有多困难：`host_mount_folder is not None`。这是一个双重否定的陈述，因此难以阅读。

```python
import os

def get_behave_test_folder(relative_test_folder: str = ''):
    host_mount_folder = os.environ.get("HOST_MOUNT_FOLDER")

    if host_mount_folder is not None:
        final_host_mount_folder = host_mount_folder
        if host_mount_folder.startswith('/mnt/c/'):
            final_host_mount_folder = host_mount_folder.replace(
                '/mnt/c/', '/c/', 1
            )
        behave_test_folder = (
            final_host_mount_folder + '/' + relative_test_folder
        )
    else:
        behave_test_folder = os.getcwd()

    return behave_test_folder
```

让我们重构上述代码，使 `if` 和 `else` 语句反转：

```python
def get_behave_test_folder(relative_test_folder: str = ''):
    host_mount_folder = os.environ.get("HOST_MOUNT_FOLDER")

    if host_mount_folder is None:
        behave_test_folder = os.getcwd()
    else:
        final_host_mount_folder = host_mount_folder
        if host_mount_folder.startswith('/mnt/c/'):
            final_host_mount_folder = host_mount_folder.replace(
                '/mnt/c/', '/c/', 1
            )
        behave_test_folder = (
            final_host_mount_folder + '/' + relative_test_folder
        )

    return behave_test_folder
```

下面是另一个示例：

```python
if name != 'some name':
    # Do thing 1 ...
else:
    # Do thing 2 ...
```

我们不应该在 `if` 语句的条件中使用否定。让我们重构上面的示例：

```python
if name == 'some name':
    # Do thing 2 ...
else:
    # Do thing 1 ...
```

## 5.8：静态代码分析原则

> 让计算机为你找出代码中的错误和问题。

静态代码分析工具会代你找出错误和与设计相关的问题。使用多种静态代码分析工具以获得全部好处。不同的工具可能检测到不同的问题。使用静态代码分析工具可以解放人们在代码审查中的时间，使其专注于自动化无法处理的事情。

下面是一些常见的 Python 静态代码分析工具列表：

- PyLint
- Ruff
- Sonarlint
- SonarQube/SonarCloud
- Black（代码格式化工具）

## 5.8.1：常见的静态代码分析问题

- Blue（代码格式化工具）
- Mypy
- Jetbrains PyCharm 检查工具

基础设施和部署代码应与源代码同等对待。请记住，也要对基础设施和部署代码运行静态代码分析工具。有多种工具可用于分析基础设施和部署代码，例如 *Checkov*，它可用于分析 Terraform、Kubernetes 和 Helm 代码。Helm 工具包含一个用于分析 Helm chart 文件的 linting 命令，而 *Hadolint* 是一个用于静态分析 *Dockerfiles* 的工具。

| 问题 | 描述/解决方案 |
|---|---|
| 实例检查链 | 此问题表明存在一系列倾向于面向对象设计的条件语句。使用*用多态替换条件*重构技术来解决此问题。 |
| 特征依恋 | 使用上一章的*不要询问，告知原则*来解决此问题。 |
| 使用具体类 | 使用上一章的*面向接口编程原则*来解决此问题。 |
| 赋值给函数参数 | 不要修改函数参数，而是引入一个新变量。 |
| 注释掉的代码 | 删除注释掉的代码。如果将来需要这段代码，它永远存在于版本控制系统中。 |
| 常量正确性 | 尽可能将属性和变量标记为 @final，以实现不可变性并避免意外修改。 |
| 嵌套的 match 语句 | 主要只在工厂中使用 match 语句。不要嵌套它们。 |
| 嵌套的条件表达式（三元运算符） | 条件表达式不应嵌套，因为这会极大地阻碍代码的可读性。 |
| 过于复杂的布尔表达式 | 将布尔表达式拆分为多个部分，并引入常量来存储这些部分和最终表达式。 |
| 表达式可以简化 | 这可以通过 IDE 自动重构。 |
| 没有 default 分支的 match 语句 | 始终引入一个 default 分支并在其中引发异常。否则，当使用带有枚举的 match 语句时，在添加了新的枚举值（该值未被 match 语句处理）后，可能会遇到奇怪的问题。 |
| 迪米特法则 | 对象知道得太多。它与另一个对象的依赖项耦合，这会产生额外的耦合并使代码更难更改。 |
| 局部变量的重用 | 不要为了不同的目的重用变量，而是引入一个新变量。这个新变量可以被恰当地命名以描述其用途。 |
| 变量的作用域过广 | 仅在需要变量之前才引入它。 |
| 受保护的字段 | 子类可以修改超类的受保护状态，而超类无法控制这一点。这是破坏封装的迹象，应予以避免。 |
| 破坏封装：返回可修改/可变字段 | 使用上一章的*不要将可修改的内部状态泄漏到对象外部原则*来解决此问题。 |
| 破坏封装：从方法参数赋值给可修改/可变字段 | 使用上一章的*不要从方法参数赋值给可修改字段原则*来解决此问题。 |
| 非常量公共字段 | 任何人都可以修改公共字段。这破坏了封装，应予以避免。 |
| 过于宽泛的 except 块 | 这可能表明设计错误。例如，如果你应该只捕获应用程序的基础错误类，就不要期望捕获语言的基础异常类。有关处理异常的更多信息，请阅读下一节。 |

## 5.9：错误/异常处理原则

Python 和许多其他语言（如 C++、Java 和 JavaScript/TypeScript）都具有可以处理错误和异常情况的异常处理机制。首先，我想明确区分这两个词：

> 错误是可能发生的事情，应该为此做好准备。异常是绝不应该发生的事情。

你在代码中定义错误并在函数中引发它们。例如，如果你尝试写入文件，必须为磁盘已满的错误做好准备；或者如果你正在读取文件，必须为文件（不再）存在的错误做好准备。

许多错误是可恢复的。你可以从磁盘删除文件以释放一些空间来写入文件。或者，如果找不到文件，你可以向用户发出“文件未找到”错误，然后用户可以使用不同的文件名重试操作。异常通常不是你在应用程序中定义的，而是系统在异常情况下引发的，例如遇到编程错误时。

例如，当内存不足且无法执行内存分配时，或者当编程错误导致数组索引越界或字典不包含特定键时，可能会引发异常。当抛出异常时，程序无法继续正常执行，可能需要终止。这就是为什么许多异常可以归类为不可恢复的错误。在某些情况下，可以从异常中恢复。假设一个 Web 服务在处理 HTTP 请求时遇到空指针异常。在这种情况下，你可以终止当前请求的处理，向客户端返回错误响应，然后继续正常处理后续请求。如何处理异常情况取决于软件组件。

错误定义了函数因某种原因执行失败的情况。错误的典型示例包括文件未找到错误、向远程服务发送 HTTP 请求时出错或解析配置文件失败。假设一个函数可能引发错误。根据错误类型，函数调用者可以决定如何处理错误。对于瞬态错误（如网络请求失败），函数调用者可以等待一段时间后再次调用该函数。或者，函数调用者可以使用默认值。例如，如果一个函数尝试加载一个不存在的配置文件，它可以使用一些默认配置。在某些情况下，函数调用者除了让错误未处理或期望错误但在更高的抽象层级引发另一个错误外，别无他法。假设一个函数尝试加载配置文件，但加载失败且没有默认配置。在这种情况下，函数除了将错误传递给其调用者外，别无他法。最终，这个错误会在调用栈中冒泡，整个进程由于无法加载配置而终止。这是因为配置是运行应用程序所必需的。没有配置，应用程序除了退出外别无他法。

在定义错误类时，为你的软件组件定义一个基础错误类。你可以根据软件组件的名称来命名基础错误类。例如，对于数据导出器微服务，你可以定义一个 `DataExporterError`（或 `DataExporterServiceError`）基础错误类；对于 `common-utils-lib`，你可以定义 `CommonUtilsError`（或 `CommonUtilsLibError`）；对于 `sales-item-service`，你可以定义 `SalesItemServiceError`。根据情况，你可以在基础错误类名称中移除或保留软件组件类型名称。流行的 `requests` Python 包就实现了这个约定。它定义了一个 `requests.RequestException`，它是库方法可能引发的所有其他错误的基类。对于每个可能引发错误的函数，定义一个与函数处于相同抽象层级的基础错误类。该错误类应扩展软件组件的基础错误类。例如，如果你在 `ConfigParser` 类中有一个 `parse(config_str)` 函数，则在该类内定义一个名为 `ParseError` 的基础错误类，即 `ConfigParser.ParseError`。如果你在 `FileReader` 类中有一个 `read_file` 函数，则在 `FileReader` 类中定义一个名为 `ReadFileError` 的基础错误类，即 `FileReader.ReadFileError`。如果一个类中的所有方法都可能引发相同的错误，那么只需在类级别定义一个错误即可。例如，如果你有一个 `HttpClient` 类，其中所有方法（如 `get`、`post`、`put` 等）都可能引发错误，你只需在 `HttpClient` 类中定义一个 `Error` 错误类。

以下是为数据导出器微服务定义的错误示例：

```python
class DataExporterError(Exception):
    pass

class FileReader:
    class ReadFileError(DataExporterError):
        pass

    def read_file(self, file_path_name: str):
        # ...
        # raise self.ReadFileError()

class ConfigParser:
    class ParserError(DataExporterError):
        pass

    def parse(self, config_str: str):
        # ...
        # raise self.ParseError()
```

遵循上述规则使得在代码中捕获错误变得容易，因为你可以从调用的方法（和类）名称推断出错误类名称。在下面的示例中，我们可以从 `read_file` 方法名称推断出 `ReadFileError` 错误类名称：

```python
try:
    file_contents = file_reader.read_file(...)
except FileReader.ReadFileError as error:
    # 处理错误 ...
```

你也可以在 except 子句中使用软件组件的基础错误类来捕获所有用户定义的错误。

try:
    config_string = file_reader.read_file(...)
    return config_parser.parse(config_string)
except DataExporterError as error:
    # 处理错误情况

不要捕获语言的基础异常类或其他过于通用的异常类，因为那样会捕获所有用户定义的错误，以及像`MemoryError`或`ZeroDivisionError`这样的异常，这很可能不是你想要的。所以，不要像这样捕获过于通用的异常类：

```
try:
    config_string = file_reader.read_file(...)
    return config_parser.parse(config_string)
except BaseException as error:
    # 不要这样做！
```

只在代码中的特殊位置捕获所有异常，比如在主函数或主循环中，例如处理HTTP请求的Web服务循环或线程的主循环。下面是在主函数中正确捕获语言基础异常类的示例。当你在主函数中捕获到不可恢复的异常时，记录它并使用适当的错误代码退出进程。当你在主循环中捕获到不可恢复的错误时，记录它，并在可能的情况下继续循环。

```
try:
    application.run(...)
except BaseException exception:
    logger.log(exception)
    sys.exit(1)
else:
    sys.exit(0)
```

使用上述规则，你可以使你的代码具有前瞻性或前向兼容性，以便将来可以从函数中抛出新的错误。假设你正在使用一个`fetch_config`函数，如下所示：

```
try:
    configuration = config_fetcher.fetch_config(url)
except ConfigFetcher.FetchConfigError as error:
    # 处理错误 ...
```

如果`fetch_config`函数抛出新类型的错误，你的代码应该仍然有效。假设`fetch_config`函数可能抛出以下新错误：

- 格式错误的URL错误
- 未找到服务器错误
- 连接超时错误

当这些新错误的类被实现时，它们必须扩展函数的基础错误类，在本例中是`FetchConfigError`类。以下是定义的新错误类：

```
from Config import Config
from DataExporterError import DataExporterError
```

```
class ConfigFetcher:
    class FetchConfigError(DataExporterError):
        pass

    class MalformedUrlError(FetchConfigError):
        pass

    class ServerNotFoundError(FetchConfigError):
        pass

    class TimeoutError(FetchConfigError):
        pass

    def fetch_config(self, url: str) -> Config:
        # ...
        # raise self.MalformedUrlError()
        # raise self.ServerNotFoundError()
        # raise self.TimeoutError()
```

你以后可以增强你的代码，以不同方式处理从`fetch_config`方法引发的不同错误。例如，你可能想要处理`TimeoutError`，以便函数会等待一段时间然后重试操作，因为错误可能是暂时的：

```
try:
    configuration = config_fetcher.fetch_config(url)
except ConfigFetcher.TimeoutError as error:
    # 等待一段时间后重试
except ConfigFetcher.MalformedUrlError as error:
    # 通知调用者应检查URL
except ConfigFetcher.ServerNotFoundError as error:
    # 通知调用者无法访问URL主机/端口
except ConfigFetcher.FetchConfigError as error:
    # 处理可能的其他错误情况
    # 这将捕获未来可能从'fetchConfig'函数抛出的任何新异常
```

在上面的例子中，我们正确地处理了引发的错误，但你很容易忘记处理一个引发的错误。这是因为函数签名中没有任何内容告诉你函数是否可以抛出错误。唯一的发现方法是检查文档（如果可用）或研究源代码（如果可用）。这是错误处理方面最大的问题之一，因为你必须知道并记住一个函数可以引发错误，并且你必须记住捕获和处理错误。你并不总是想立即处理错误，但你仍然必须意识到错误会在调用堆栈中向上传播，并且最终应该在代码中的某个地方得到处理。

下面是从流行的Python requests包文档中提取的一个示例：

```
import requests

r = requests.get('https://api.github.com/events')
r.json()
# [{'repository': {'open_issues': 0, 'url': 'https://github.com/...
```

你知道`requests.get`和`r.json`都可能引发错误吗？不幸的是，这个例子根本没有包含错误处理。如果你直接将上面的代码示例复制粘贴到你的生产代码中，你可能会忘记处理错误。如果你去查看`requests`包的API参考文档，你可以找到`get`方法的文档。该文档（在撰写本书时）没有说明该方法可以引发错误。文档只讨论了方法参数、返回值及其类型。只有当你向下滚动文档页面时，你才会找到一个关于异常的部分。但如果你不向下滚动呢？你可能会最终认为该方法不会引发错误。`get`方法的文档应该被修正，以便它说明该方法可以引发错误，并包含一个链接到描述可能错误的部分。

上述问题在实践*测试驱动开发*（TDD）时至少可以在一定程度上得到缓解。TDD将在下一章中描述，该章涵盖与测试相关的原则。在TDD中，你在实现之前定义测试，这迫使你思考错误场景并为它们创建测试。当你有错误场景的测试时，就不可能在实际实现代码中留下这些场景未处理。

错误处理可能被遗忘的问题的最佳解决方案之一是使引发错误更加明确：

> 如果函数可能引发错误，请在函数名中使用“try”前缀。

这是一个简单的规则。如果一个函数可能引发错误，请将函数命名为以`try`开头。这使每个调用者都清楚该函数可能引发错误，调用者应该为此做好准备。对于函数的调用者，有三种处理抛出错误的替代方案：

1) 捕获被调用函数（或软件组件）的基础错误类并处理错误，例如，如果你正在调用名为`DataFetcher`的类中名为`try_fetch_data`的方法，则捕获`DataFetcher.FetchDataError`。
2) 捕获被调用函数（或软件组件）的基础错误类并在更高的抽象级别上引发新错误。现在你还必须用`try`前缀命名调用函数。
3) 不捕获错误。让它们在调用堆栈中向上传播。现在你还必须用`try`前缀命名调用函数。

以下是替代方案1的示例：

```
from Config import Config
from ConfigParser import ConfigParser
from DataFetcher import DataFetcher

class ConfigFetcher:
    def fetch_config(self, url: str) -> Config:
        try:
            config_str = self.__data_fetcher.try_fetch_data(url)
            return self.__config_parser.try_parse(config_str)
        except (
            DataFetcher.FetchDataError,
            ConfigParser.ParseError
        ) as error:
            # 你也可以在两个不同的except块中捕获错误
            # 你也可以捕获软件组件的基础错误类'DataExporterError'
```

以下是替代方案2的示例：

```
from Config import Config
from ConfigParser import ConfigParser
from DataFetcher import DataFetcher

class ConfigFetcher:
    class FetchConfigError(DataExporterError):
        pass

    def try_fetch_config(self, url: str) -> Config:
        try:
            config_str = self.__data_fetcher.try_fetch_data(url)
            return self.__config_parser.try_parse(config_str)
        except (
            DataFetcher.FetchDataError,
            ConfigParser.ParseError
        ) as error:
            # 在更高的抽象级别上引发错误
            # 此函数必须用'try'前缀命名
            # 以表明它可以引发错误
            raise self.FetchConfigError(error)
```

以下是替代方案3的示例：

## 编码原则

```python
from Config import Config

class ConfigFetcher:
    def try_fetch_config(self, url: str) -> Config:
        # 无 try-except，所有来自 try_fetch_data 和 try_parse 方法调用的
        # 抛出错误都会传播给调用者，
        # 并且此函数必须以 'try' 前缀命名，
        # 以表明它可能引发错误
        config_str = self.__data_fetcher.try_fetch_data(url)
        return self.__config_parser.try_parse(config_str)

from DataExporterError import DataExporterError

class DataExporter:
    def initialize(self) -> None:
        try:
            config = self.__config_fetcher.try_fetch_config(url)
        except DataExporterError as error:
            # 在这种情况下，你必须捕获软件组件的基错误类
            # (DataExporterError)，因为你不知道 try_fetch_config
            # 可能引发什么错误，因为 ConfigFetcher 类中
            # 没有定义 FetchConfigError 类
```

如果我们回到 `requests` 包的使用示例，抛出错误的方法 `requests.get` 和 `Response.json` 可以重命名为 `requests.try_get` 和 `Response.try_parse_json`。这将使前面的示例看起来像这样：

```python
import requests

r = requests.try_get('https://api.github.com/events')
r.try_parse_json()
# [{'repository': {'open_issues': 0, 'url': 'https://github.com/...
```

现在我们可以看到这两个方法可能引发错误，因此我们可以将它们放在 `try` 块中：

```python
import requests

try:
    r = requests.try_get('https://api.github.com/events')
    r.try_parse_json()
    # [{'repository': {'open_issues': 0, 'url': 'https://github.com/...
except ...
    # ...
```

为了使 `try` 前缀约定更好，可以开发一个强制执行抛出错误函数正确命名的 lint 规则。该规则应强制函数名具有 `try` 前缀，如果该函数引发或传播错误。当函数在 `try-except` 块之外调用一个抛出错误（带 `try` 前缀）的方法时，该函数会传播错误。

你也可以创建一个库，其中包含带 `try` 前缀的函数，这些函数包装了不遵循 `try` 前缀规则的抛出错误的函数：

```python
class JsonParser:
    class ParseError(Exception):
        pass

    @staticmethod
    def try_parse(
        s,
        *,
        cls=None,
        object_hook=None,
        parse_float=None,
        parse_int=None,
        parse_constant=None,
        object_pairs_hook=None,
        **kwargs
    ):
        try:
            return json.loads(s)
        except json.JSONDecodeError as error:
            raise JsonParser.ParseError(error)
```

现在，如果你使用 `JsonParser` 的 `try_parse` 方法，你可以轻松推断出可能引发的错误的类名，而无需查阅任何文档。

使用 Web 框架时，该框架通常提供错误处理机制。框架在处理请求时会捕获所有可能的错误，并将它们映射到带有指示失败的 HTTP 状态码的 HTTP 响应。通常，默认状态码是 500 *Internal Server Error*。当你利用 Web 框架的错误处理机制时，使用 `try` 前缀命名抛出错误的函数并没有太大好处，因为如果你忘记捕获错误，这通常不会成为问题，而且很多时候这正是你想要做的——将错误传递给 Web 框架的错误处理器。通常你会提供自己的错误处理器而不是使用默认的，这样你就能得到所需格式的响应。所以，如果你愿意，你可以选择不使用 `try` 前缀规则，但当然为了保持一致性，你也可以使用它。你也可以将错误类放在自己的模块中，并放在特定的包（目录）中。

通常，在软件组件文档中记录所使用的错误处理机制是一个好习惯。

避免忘记处理错误的最佳方法是实践严格的*测试驱动开发*（TDD），这将在下一章中描述。另一个不忘记处理错误的好方法是逐行检查代码，检查特定行是否可能产生错误。如果它可能产生错误，是什么类型的错误，以及该行可能产生多种不同的错误。让我们看一个以下代码的例子（我们只关注可能的错误，而不关注函数的功能）：

```python
from typing import Any

import requests
from jwt import PyJWKClient, decode

class JwtAuthorizer:
    # ...

    def __try_get_jwt_claims(
            self, auth_header: str | None
    ) -> dict[str, Any]:
        if not self.__jwks_client:
            oidc_config_response = requests.get(self.__oidc_config_url)
            oidc_config = oidc_config_response.json()
            self.__jwks_client = PyJWKClient(oidc_config['jwks_uri'])

        jwt = auth_header.split('Bearer ')[1] if auth_header else ''
        signing_key = self.__jwks_client.get_signing_key_from_jwt(jwt)
        jwt_claims = decode(jwt, signing_key.key, algorithms=['RS256'])
        return jwt_claims
```

第一行代码不会产生错误。在第二行，`requests.get` 方法在连接失败时可能引发错误。它还能产生其他错误吗？它可能产生以下错误：

-   URL 格式错误 (`requests.URLRequired`)
-   连接错误 (`requests.ConnectionError`)
-   连接超时 (`requests.ConnectTimeout`)
-   读取超时 (`requests.ReadTimeout`)

它还可能产生错误响应，例如内部服务器错误。我们的代码目前没有处理这种情况，这就是为什么我们应该在 `requests.get` 方法调用后添加以下行：`oidc_config_response.raise_for_status()`，如果响应状态码 >= 400，它可能会引发 `HttpError`。第三行如果响应不是有效的 JSON，可能会引发 `JSONDecodeError`。第四行可能会引发 `KeyError`，因为响应 JSON 中可能不存在键 `jwks_uri`。第五行可能会引发 `IndexError`，因为 `split` 返回的列表不一定在索引一处有元素。此外，第六行在 JWKS 客户端无法连接到 IAM 系统或 JWT 无效时也可能引发错误。倒数第二行在 JWT 无效时可能会引发 `PyJWKClientError`。总之，除了第一行和最后一行，上面代码中的所有行都可能至少产生一种错误。

让我们重构代码以实现错误处理，而不是将所有可能的错误和异常传递给调用者：

```python
from typing import Any

import requests
from jwt import PyJWKClient, PyJWKClientError, decode
from jwt.exceptions import InvalidTokenError

class JwtAuthorizer:
    class GetJwtClaimsError(Exception):
        pass

    def __try_get_jwt_claims(
        self, auth_header: str | None
    ) -> dict[str, Any]:
        try:
            if not self.__jwks_client:
                oidc_config_response = requests.get(self.__oidc_config_url)
                oidc_config_response.raise_for_status()
                oidc_config = oidc_config_response.json()
                self.__jwks_client = PyJWKClient(oidc_config['jwks_uri'])

            jwt = auth_header.split('Bearer ')[1] if auth_header else ''
            signing_key = self.__jwks_client.get_signing_key_from_jwt(jwt)
            jwt_claims = decode(jwt, signing_key.key, algorithms=['RS256'])
            return jwt_claims
        except (
            # RequestException 是 requests 包中所有错误的基错误类
            requests.RequestException,
            KeyError,
            IndexError,
            PyJWKClientError,
            # 当 decode() 在令牌上失败时的基异常
            InvalidTokenError,
        ) as error:
            raise self.GetJwtClaimsError(error)
```

我建议你养成一个习惯：一旦你认为函数准备就绪，就逐行检查函数的代码，以发现你是否意外地遗漏了对某些错误的处理。

### 5.9.1：返回错误

作为引发错误的替代方案，可以使用返回值向函数调用者传达错误行为。使用异常处理机制比返回错误提供了一些优势。当一个函数可以返回错误时，你必须在函数调用后立即检查错误。这可能导致代码包含嵌套的 if 语句，从而阻碍代码的可读性。异常处理机制允许你将错误传播到调用栈的更高层。你还可以在单个 `try` 块中执行多个可能失败的函数调用，并在 `except` 块中提供单个错误处理器。

## 5.9.1.1：返回失败指示器

当可失败函数不需要返回任何额外值时，你可以从该函数返回一个失败指示器。当没有需要返回特定错误代码或消息的需求时，仅从函数返回一个失败指示器就足够了。这可能是因为函数失败的原因只有一个，或者函数调用者对错误细节不感兴趣。要返回失败指示器，请从函数返回一个布尔值：*True* 表示操作成功，*False* 表示失败：

```
def perform_task(...) -> bool:
    # 执行任务并设置
    # 'task_was_performed' 变量的值

    return task_was_performed
```

## 5.9.1.2：返回可选值

假设一个函数应该返回一个值，但函数调用可能失败，并且函数调用失败的原因恰好只有一个。在这种情况下，请从函数返回一个可选值。在下面的示例中，从缓存获取值只有在缓存中没有存储特定键的值时才会失败。我们不需要返回任何错误代码或消息。

```
from typing import Protocol, TypeVar

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')

class Cache(Protocol[TKey, TValue]):
    def add(self, key: TKey, value: TValue) -> None:
        pass

    def get(self, key: TKey) -> TValue | None:
        pass
```

或者，如果你想使用更函数式的方法，可以返回一个 Optional 对象。（Optional 类在前一章中已定义）

```
from typing import Protocol, TypeVar

from Optional import Optional

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')

class Cache(Protocol[TKey, TValue]):
    def add(self, key: TKey, value: TValue) -> None:
        pass

    def get(self, key: TKey) -> Optional[TValue]:
        pass
```

## 5.9.1.3：返回错误对象

当你需要向函数调用者提供有关错误的详细信息时，可以从函数返回一个错误对象：

```
from dataclasses import dataclass

@dataclass
class BackendError:
    http_status_code: int
    error_code: int
    message: str
```

如果一个函数不返回任何值但可能产生错误，你可以返回一个错误对象或 None：

```
from typing import Awaitable, TypeVar

from BackendError import BackendError
from Entity import Entity

T = TypeVar('T', bound=Entity)

class DataStore:
    async def update_entity(
        self,
        id: int,
        entity: T
    ) -> Awaitable[BackendError | None]:
        # ...
```

或者，返回一个可选错误，如下所示。（这些示例中使用的 Optional 类在前一章中已定义。如你从导入语句中看到的，我们使用的是那个类，而不是 typing 模块中的 Optional）

```
from typing import Awaitable, TypeVar

from BackendError import BackendError
from Entity import Entity
from Optional import Optional

T = TypeVar('T', bound=Entity)

class DataStore:
    async def update_entity(
        self,
        id: int,
        entity: T
    ) -> Awaitable[Optional[BackendError]]:
        # ...
```

假设一个函数需要返回一个值或一个错误。在这种情况下，你可以使用一个二元组（即一对）类型，其中元组中的第一个值是实际值或在错误情况下的 None，第二个值是错误对象或在成功操作情况下的 None 值。下面是一个示例。

```
from typing import Awaitable, TypeVar, Union

from BackendError import BackendError
from Entity import Entity

T = TypeVar('T', bound=Entity)

class DataStore:
    async def create_entity(
        self,
        entity: T
    ) -> Awaitable[Union[(T, None), (None, BackendError)]]:
        # ...
```

如果我们想让我们的方法更具函数式风格，我们应该从中返回一个 Either 类型，但 Python 没有这个类型。Either 类型包含两个值中的一个，要么是左值，要么是右值。Either 类型可以定义如下。（下面示例中使用的 Optional 类与前一章中定义的相同，而不是 typing 模块中的 Optional）。

编码原则

317

```
from collections.abc import Callable
from typing import Any, TypeVar, Generic

from Optional import Optional

TLeft = TypeVar('TLeft')
TRight = TypeVar('TRight')
T = TypeVar('T')
U = TypeVar('U')
```

```
class PrivateConstructor(type):
    def __call__(
        cls: type[T],
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any]
    ):
        raise TypeError('Constructor is private')

    def _create(
        cls: type[T],
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any]
    ) -> T:
        return super().__call__(*args, **kwargs)
```

```
class Either(Generic[TLeft, TRight], metaclass=PrivateConstructor):
    def __init__(
        self,
        maybe_left_value: Optional[TLeft],
        maybe_right_value: Optional[TRight]
    ):
        self.__maybe_left_value = maybe_left_value
        self.__maybe_right_value = maybe_right_value

    @classmethod
    def with_left(cls, value: TLeft) -> 'Either[TLeft, TRight]':
        return cls._create(Optional.of(value), Optional.empty())

    @classmethod
    def with_right(cls, value: TRight) -> 'Either[TLeft, TRight]':
        return cls._create(Optional.empty(), Optional.of(value))

    def has_left_value(self) -> bool:
        return self.__maybe_left_value.is_present()

    def has_right_value(self) -> bool:
        return self.__maybe_right_value.is_present()

    def map_left(
        self,
        to_value: Callable[[TLeft], U]
    ) -> 'Either[U, TRight]':
        return Either._create(
            self.__maybe_left_value.map(to_value),
            self.__maybe_right_value
        )

    def map_right(
        self,
        to_value: Callable[[TRight], U]
    ) -> 'Either[TLeft, U]':
        return Either._create(
            self.__maybe_left_value,
            self.__maybe_right_value.map(to_value)
        )

    def map(
        self,
        left_to_value: Callable[[TLeft], U],
        right_to_value: Callable[[TRight], U]
    ) -> U:
        return self.__maybe_left_value.map(left_to_value).or_else_get(
            lambda: self.__maybe_right_value.map(right_to_value).try_get()
        )

    def apply(
        self,
        consume_left_value: Callable[[TLeft], None],
        consume_right_value: Callable[[TRight], None]
    ) -> None:
        self.__maybe_left_value.if_present(consume_left_value)
        self.__maybe_right_value.if_present(consume_right_value)
```

以下是一些如何使用 Either 类的示例：

```
class Error(Exception):
    pass
```

```
int_or_error: Either[int, Error] = Either.with_left(3)
int_or_error2: Either[int, Error] = Either.with_right(Error())
```

```
print(int_or_error.has_left_value()) # 输出 True
print(int_or_error2.has_right_value()) # 输出 True
print(
    int_or_error.map_left(lambda number: number * 2).has_left_value()
)
# 输出 True
```

```
print(int_or_error.map(lambda number: number * 2, lambda error: 0))
# 输出 6
```

```
print(int_or_error2.map(lambda number: number * 2, lambda error: 0))
# 输出 0
```

现在我们可以使用新的 Either 类型并重写示例如下：

```
from typing import Awaitable, TypeVar

from BackendError import BackendError
from Entity import Entity

T = TypeVar('T', bound=Entity)

class DataStore:
    async def create_entity(
        self,
        entity: T
    ) -> Awaitable[Either[T, BackendError]]:
        # ...
```

## 5.9.1.4：适配所需的错误处理机制

你可以通过创建一个适配器类来适配所需的错误处理机制。例如，如果一个库有一个抛出错误的方法，你可以创建一个适配器类，其方法返回一个可选值。下面的 Url 类有一个 try_create_url 工厂方法，该方法可能抛出错误：

```
class Url:
    # ...

    class CreateUrlError(Exception):
        pass

    @classmethod
    def try_create_url(
        cls,
        scheme: str,
        host: str,
        port: int,
        path: str,
        query: str
    ) -> 'Url':
        # ...
        # 这里可能抛出 CreateError ...
```

我们可以创建一个 UrlFactory 适配器类，其方法 create_url 不会抛出错误。

```
from Url import Url

class UrlFactory:
    def create_url(
        self,
        scheme: str,
        host: str,
        port: int,
        path: str,
        query: str
    ) -> Url | None:
        try:
```

```python
return Url.try_create_url(scheme, host, port, path, query)
except Url.CreateUrlError:
    return None
```

如果使用 `UrlFactory` 的代码对错误详情感兴趣，我们也可以创建一个不抛出错误，而是返回值或错误的方法：

```python
from typing import Union

from Url import Url

class UrlFactory:
    def create_url_or_error(
        self,
        scheme: str,
        host: str,
        port: int,
        path: str,
        query: str
    ) -> Union[(Url, None), (None, Url.CreateError)]:
        try:
            return (
                Url.try_create_url(scheme, host, port, path, query),
                None,
            )
        except Url.CreateUrlError as error:
            return None, error
```

## 5.9.1.5：函数式异常处理

下面的 `Failable` 类可用于函数式错误处理。`Failable` 对象表示一个类型为 `T` 的值或一个 `Exception` 类的实例，即 `Failable[T]` 等同于 `Either[T, Exception]`。

```python
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from Either import Either

T = TypeVar('T')

class PrivateConstructor(type):
    def __call__(
        cls: type[T],
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any]
    ):
        raise TypeError('Constructor is private')

    def _create(
        cls: type[T],
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any]
    ) -> T:
        return super().__call__(*args, **kwargs)

TError = TypeVar('TError', bound=Exception)
U = TypeVar('U')

class Failable(Generic[T], metaclass=PrivateConstructor):
    def __init__(self, value_or_error: Either[T, Exception]):
        self.__value_or_error = value_or_error

    @classmethod
    def with_value(cls, value: T) -> 'Failable[T]':
        return cls._create(Either.with_left(value))

    @classmethod
    def with_error(cls, error: Exception) -> 'Failable[T]':
        return cls._create(Either.with_right(error))

    def __raise(self, error: Exception) -> None:
        raise error

    def or_raise(self, error_cls: type[TError]) -> T:
        return self.__value_or_error.map(
            lambda value: value,
            lambda error: self.__raise(error_cls(error))
        )

    def or_else(self, other_value: T) -> T:
        return self.__value_or_error.map(
            lambda value: value,
            lambda error: other_value
        )

    def map_value(
        self,
        to_value: Callable[[T], U]
    ) -> 'Failable[U]':
        return Failable._create(self.__value_or_error.map_left(to_value))

    def map_error(
        self,
        to_error: Callable[[Exception], Exception]
    ) -> 'Failable[T]':
        if self.__value_or_error.has_left_value():
            error = to_error(Exception())
            return Failable.with_error(error)
        else:
            return Failable._create(
                self.__value_or_error.map_right(to_error)
            )
```

在下面的示例中，`read_config` 方法返回一个 `Failable[Configuration]`。`try_initialize` 方法要么获取一个 `Configuration` 实例，要么抛出一个类型为 `Application.InitializeError` 的错误。

## 编码原则

```python
from DataExporterError import DataExporterError

class Application:
    # ...

    class InitializeError(DataExporterError):
        pass

    def try_initialize(self) -> None:
        configuration = self.__config_reader \
            .read_config(...) \
            .or_raise(self.InitializeError)
```

上述函数式方法的好处是它比整个 `try-catch` 块更简短。上述函数式方法也与 `try-catch` 块一样易于理解。请记住，你应该编写最简短、最易理解的代码。当一个方法返回一个 `Failable` 实例时，你不必用 `try` 前缀来命名该方法，因为该方法不会抛出异常。对 `Failable` 调用 `or_raise` 方法用于将函数式代码转换回命令式代码。

你也可以使用 `Failable` 类的其他方法。例如，可以使用 `or_else` 方法返回一个默认值：

```python
from DefaultConfig import DefaultConfig

class Application:
    # ...

    def initialize(self) -> None:
        configuration = self.__config_reader.read_config(...).or_else(DefaultConfig())
```

你也可以将多个命令式错误抛出语句转换为函数式可失败语句。例如，与其写：

```python
from DataExporterError import DataExporterError

class Application:
    # ...

    class InitializeError(DataExporterError):
        pass

    def try_initialize(self) -> None:
        try:
            config_json = self.__data_fetcher.try_fetch_data(self.__config_url)
            configuration = self.__config_parser.try_parse(config_json)
        except DataExporterError as error:
            raise self.InitializeError(error)
```

你可以写：

```python
from DataExporterError import DataExporterError

class Application:
    # ...

    class InitializeError(DataExporterError):
        pass

    def try_initialize(self) -> None:
        configuration = (
            self.__data_fetcher.fetch_data(self.__config_url)
            .map_value(self.__config_parser.parse)
            .or_raise(self.InitializeError)
        )
```

上述函数式代码比相同的命令式代码更短，但可读性较差，因此你可能希望使用命令式方法而不是函数式方法。

将错误抛出的命令式代码与函数式编程构造一起使用可能容易出错。假设我们有下面的代码，它使用函数式编程构造 `reduce` 来读取和解析多个配置文件到一个配置对象。我们将配置读取函数命名为 `try_read_config`，使用了 `try` 前缀，因为它可能抛出错误。当我们使用 `reduce` 函数时，我们必须记住用 `try-except` 块将其包围，因为 `reduce` 函数将调用可能抛出异常的 `try_read_config` 函数。

```python
import json
from functools import reduce
from typing import Any

def try_read_config(
    accumulated_config: dict[str, Any],
    config_file_path_name: str
):
    with open(config_file_path_name) as config_file:
        config_json = config_file.read()

    config = json.loads(config_json)
    return accumulated_config | config

def get_config(
    config_file_path_names: list[str]
) -> dict[str, Any]:
    try:
        return reduce(try_read_config, config_file_path_names, {})
    except:
        # ...
```

我们可以通过让 `get_config` 函数返回一个 `Failable` 实例，将上述示例转变为更函数式的风格：

```python
import json
from functools import reduce
from typing import Any

from Failable import Failable

def to_config_or_error(
    accum_config_or_error: Failable[dict[str, Any]],
    config_file_path_name: str
) -> Failable[dict[str, Any]]:
    try:
        with open(config_file_path_name) as config_file:
            config_json = config_file.read()

        config = json.loads(config_json)

        return accum_config_or_error.map_value(
            lambda accum_config: accum_config | config
        )
    except (OSError, json.JSONDecodeError) as error:
        return accum_config_or_error.map_error(
            lambda accum_error: RuntimeError(
                f'{str(accum_error)}\n{config_file_path_name}: {str(error)}'
            )
        )

def get_config(
    config_file_path_names: list[str]
) -> Failable[dict[str, Any]]:
    return reduce(
        to_config_or_error,
        config_file_path_names,
        Failable.with_value({})
    )
```

如果我们有一个 `config1.json` 文件，内容如下：

```json
{
    "foo": 1,
    "bar": 2
}
```

并且我们有一个 `config2.json` 文件，内容如下：

```json
{
    "xyz": 3
}
```

那么我们可以运行以下代码：

```python
config_file_path_names = ['config1.json', 'config2.json']
maybeConfig = get_config(config_file_path_names)
print(maybeConfig.or_raise(RuntimeError))
# 输出 {'foo': 1, 'bar': 2, 'xyz': 3}
```

让我们在 *config1.json* 文件中引入一个错误（第一个属性后缺少逗号）：

```json
{
    "foo": 1
    "bar": 2
}
```

让我们也尝试提供一个不存在的配置文件 *config3.json*：

```python
config_file_path_names = ['config1.json', 'config3.json']
maybeConfig = get_config(config_file_path_names)
print(maybeConfig.or_raise(RuntimeError))
# 抛出一个 RuntimeError，包含以下消息：
# config1.json: Expecting ',' delimiter: line 3 column 3 (char 16)
# config3.json: [Errno 2] No such file or directory: 'config3.json'
```

## 5.10：避免差一错误原则

差一错误通常源于编程语言中的集合使用从零开始的索引进行索引。从零开始的索引对人类来说不自然，但对计算机来说非常出色。然而，编程语言的设计应该以人为本。人们从不谈论获取数组的第零个值。我们谈论的是获取数组中的第一个值。就像空值被称为价值十亿美元的错误一样，我会称从零开始的索引为另一个价值十亿美元的错误。让我们希望有一天我们能拥有一种使用从一索引的编程语言！但那时我们必须改掉从零索引的习惯……这又是一个问题。

在某些语言中，你可以创建带有循环计数器的 `for` 循环。以下是两个 JavaScript 中的编程错误示例，如果你不够小心，很容易犯：

```javascript
for (let index = 0; index <= values.length; index++) {
    // ...
}

for (let index = 0; index < values.length - 1; index++) {
    // ...
}
```

在第一个示例中，应该用 `<` 代替 `<=`，而在后一个示例中，应该用 `<=` 代替 `<`。幸运的是，上述错误在 Python 中可以避免：for value in values:
    # ...

在Python的`range`函数中，你必须记住它从零开始，并且范围的终点是**不包含**的。如果你不记得这一点，并假设起点为一或终点是包含的，这两点都可能导致**差一错误**。差一错误产生的原因在于，人们默认认为给定的范围在两端都是包含的。因此，`range(6)`给出的值是从0到5，而不是从1到6。同样，`range(1, 6)`给出的值是从1到5，而不是从1到6。切片也是如此，例如`values[:6]`从索引0开始，到索引5结束。如果你想获取除最后一项之外的所有切片，可以使用负索引：`values[:-1]`给出除最后一项外的所有值。使用-1比使用`values[:len(values) - 1]`更安全，后者如果忘记-1可能会产生差一错误。类似地，使用`values[:-2]`比使用`values[:len(values) - 2]`更不容易出错。你也可以使用负索引，例如用`values[-1]`而不是`values[len(values) - 1]`来获取最后一个值。你可以将负索引理解为从列表末尾开始的、基于1的索引。

此外，单元测试是帮助你发现差一错误的好帮手。因此，记得也要为边界情况编写单元测试。

## 5.11：在使用谷歌搜索或生成式AI时要保持批判性原则

> 你应该始终分析和重构从网上获取的代码，以确保其符合生产代码的标准。不要让AI成为主人，而要让它成为学徒。

我们都做过这件事，而且做过成百上千次：用谷歌搜索答案。通常，通过谷歌搜索能找到好的资源，但问题往往在于，搜索结果中的示例不一定是生产级别的质量。其中一个具体缺失的部分是错误处理。如果你从网站上复制粘贴代码，错误可能没有得到恰当处理。你应该始终分析复制粘贴的代码，看看是否需要添加错误处理。

当你为其他人提供答案时，尽量让代码尽可能接近生产环境。在Stack Overflow上，你会在问题下方找到票数最高的答案。如果答案缺少错误处理，你可以评论指出，并让作者改进他们的答案。你也可以给那个看起来最接近生产就绪状态的答案点赞。通常，票数最高的答案都相当古老。因此，向下滚动查看是否有更现代的解决方案更适合你的需求是很有用的。你也可以给那个更现代的解决方案点赞，这样它在答案列表中的排名就会更高。

关于开源库，其文档中的第一个示例可能只描述了“快乐路径”使用场景，错误处理只在文档的后续部分描述。如果你从“快乐路径”示例中复制粘贴代码并忘记添加错误处理，这可能会导致问题。因此，开源库作者应该在文档早期就提供生产质量的示例。

关于生成式AI和ChatGPT，我有一些经验。我曾让ChatGPT生成简单的Django代码。生成的代码大约95%是正确的，但它无法工作。问题在于ChatGPT忘记了生成创建数据库表的代码（makemigrations, migrate）。如果你对Django框架没有经验，这类bug可能很难解决。在这种情况下，我建议你先发现问题，然后让ChatGPT为你解决。

我与ChatGPT的另一个实验是使用*Ariadne*库生成GraphQL服务器代码。ChatGPT生成的代码是针对旧版本Ariadne的，与较新版本的Ariadne库不能正确工作。（请注意，用于训练ChatGPT的数据中，旧数据比新数据更多。ChatGPT不知道应该优先考虑较少且较新的数据，而不是较多且较旧的数据。）它还生成了一些顺序错误的代码行，导致GraphQL API完全无法工作。对于这样一个小程序，花了相当多的调试时间才最终发现问题所在：可执行模式是在查询解析器之前创建的。它应该在定义解析器之后才创建。

在使用ChatGPT或其他生成式AI工具时，你应该熟悉生成的代码，否则你不知道你的程序在做什么，如果AI生成的代码包含bug，这些bug将很难发现，因为你没有清晰理解代码实际在做什么。不要让AI成为主人，而要让它成为学徒。

防止从网上获取的代码相关bug的最佳方法是实践测试驱动开发（TDD）。TDD将在下一章描述。但TDD背后的理念是首先指定函数，并为不同的场景编写单元测试用例：边界/角落情况、错误场景、安全场景。例如，假设你是Python新手，通过谷歌搜索一个向API端点发送HTTP请求的代码片段。一旦你搜索到代码，就可以将其复制粘贴到你的函数中。很可能现在错误场景没有被处理。你还应该做的是实践TDD，并为不同的场景编写单元测试用例，例如，如果远程服务器无法连接或连接导致超时，或者如果远程服务器返回错误（状态码大于或等于400的HTTP响应）怎么办？如果你需要解析API返回的结果（例如解析JSON）并且失败了怎么办？一旦你为所有这些场景编写了单元测试用例，你就可以确信实际函数实现中的错误处理没有被遗忘。

## 5.12：优化原则

代码优化使代码运行更快和/或消耗更少的内存。更快的代码改善最终用户体验，优化减少了对资源（CPU/内存）的需求，从而降低软件运行成本。

> 避免过早优化。过早优化可能会妨碍为软件组件设计合适的面向对象设计。

首先测量未优化的性能。然后决定是否需要优化。逐个实施优化，并在每轮优化后测量性能，以确定优化是否有效。然后，你可以在未来的项目中利用获得的知识，只进行能带来显著性能提升的优化。有时，如果你知道需要进行特定优化（例如，根据以往经验），并且该优化可以在不影响面向对象设计的情况下实施，你可以在项目的早期阶段进行性能优化。

### 5.12.1：优化模式

本节描述以下优化模式：

- 仅优化繁忙循环模式
- 移除不必要功能模式
- 对象池模式
- 共享相同对象，即享元模式

### 5.12.1.1：仅优化繁忙循环模式

优化应主要针对软件组件中的繁忙循环。繁忙循环是线程中反复执行的循环，可能每秒执行数千次或更多次迭代。性能优化不应针对在软件组件生命周期内只执行一次或几次的功能，并且运行该功能不需要很长时间。例如，一个应用程序在启动时可能有配置读取和解析功能。该功能执行时间很短。优化该功能是不合理的，因为它只运行一次。即使性能有50%的差异，你能在200毫秒还是300毫秒内读取和解析配置也无关紧要。

让我们以数据导出器微服务为例。我们的数据导出器微服务由输入、转换和输出部分组成。输入部分从数据源读取消息。如果我们使用第三方库来完成此目的，我们无法影响消息读取部分。当然，如果有多个第三方库可用，可以设计性能测试来评估哪个第三方库提供最佳性能。如果有多个第三方库可用于相同功能，我们倾向于使用最流行的库或我们事先了解的库。如果性能是个问题，我们应该评估不同的库并比较它们的性能。

数据导出器微服务在其繁忙循环中具有以下功能：将输入消息解码为内部消息，执行转换，以及编码输出消息。解码输入消息需要解码消息中的每个字段。假设每秒处理5000条消息，每条消息有100个字段。在一秒钟内，必须解码50000个字段。这表明解码功能的优化至关重要。输出消息编码也是如此。我们诺基亚自己实现了Avro二进制字段的解码和编码。我们能够使它们比第三方库提供的更快。

## 5.12.1.2：移除不必要功能模式

移除不必要的功能将提升性能。你应该停下来批判性地思考你的软件组件：考虑到所有情况，我的软件组件是否只在做必要的事情？

让我们考虑数据导出器的功能。它目前将输入消息解码为内部消息。这个内部消息在对数据进行各种转换时使用。转换后的数据被编码为所需的输出格式。最终输出消息的内容可能只是原始输入消息的一小部分。这意味着只使用了解码消息的极小部分。在这种情况下，如果只有10%的字段在转换和输出消息中被使用，那么解码输入消息的所有字段就是不必要的。通过移除不必要的解码，我们可以提高数据导出器微服务的性能。

## 5.12.1.3：对象池模式

在像Python这样具有垃圾回收机制的语言中，从垃圾回收的角度来看，使用对象池的好处是显而易见的。在对象池模式中，对象只创建一次，然后重复使用。这将减轻垃圾回收的压力。如果我们不使用对象池，新对象可能会在繁忙的循环中被反复创建，并且在创建后不久就可能被丢弃。这将导致大量对象在短时间内被标记为可回收。垃圾回收会占用处理器时间，如果垃圾收集器有大量垃圾需要回收，它可能会在未知的时间间隔内，导致应用程序在未知的时间段内变慢。

## 5.12.1.4：使用最优数据结构模式

如果你在应用程序中执行数值计算，不要使用常规的Python数据结构，而是寻找一个合适的库，比如*numpy*，它包含针对特定目的优化的数据结构。

## 5.12.1.5：算法复杂度降低模式

选择一个使用大O表示法衡量的、复杂度降低的算法。这可以减少CPU/内存的使用。在下面的例子中，我们使用列表的查找算法：

```
values = [1, 2, 3, 4, 5, ..., 2000]
if 2000 in values:
    print("Value 2000 found")
```

上述算法必须遍历列表，这使得它比使用集合的查找算法慢：

```
values = {1, 2, 3, 4, 5, ..., 2000}
if 2000 in values:
    print("Value 2000 found")
```

下面的算法（列表推导式）将生成一个包含20000个值的列表：

```
values = [value for value in range(20_000)]
```

如果我们不需要同时将所有20000个值都保存在内存中，我们可以使用另一种算法（生成器表达式），它消耗的内存要少得多，因为并非所有20000个值都在内存中：

```
values = (value for value in range(20_000))
```

上面例子中`values`对象的类型是`Generator`，它继承自`Iterator`。你可以在任何需要迭代器的地方使用`values`。

## 5.12.1.6：缓存函数结果模式

如果你有一个开销很大的纯函数，它对于相同的输入总是返回相同的结果，且没有任何副作用，那么缓存函数结果将对你有益。你可以使用`@cache`或`lru_cache`装饰器来缓存函数结果，例如：

```
from functools import lru_cache

# 函数最近500次调用的结果将被缓存
@lru_cache(maxsize=500)
def make_expensive_calc(value: int):
    # ...

print(make_expensive_calc(1))
# 第一次调用后，
# 输入值1的函数结果将被缓存

print(make_expensive_calc(1))
# 函数调用的结果从缓存中获取
```

`@cache`与`@lru_cache(maxsize=None)`相同，即缓存没有最大大小限制。

## 5.12.1.7：缓冲文件I/O模式

如果你正在读取/写入非常大的文件，设置自定义缓冲区大小将对你有益。下面的例子将缓冲区大小设置为1MB：

```
python
with open('data.json', 'r', buffering=1_048_576) as data_file:
    data = data_file.read()

with open('data.json', 'w', buffering=1_048_576) as data_file:
    data_file.write(data)
```

## 5.12.1.8：共享相同对象（又称享元模式）

如果你的应用程序有许多具有某些相同属性的对象，那么这些具有相同属性的对象部分就在浪费内存。你应该将公共属性提取到一个新的类中，并让原始对象引用该新类的一个共享对象。现在你的对象共享一个公共对象，并且可能显著减少内存消耗。这种设计模式称为*享元模式*，在前面的章节中有更详细的描述。

## 6：测试原则

测试传统上分为两类：功能测试和非功能测试。本章将首先描述功能测试原则，然后是非功能测试原则。

## 6.1：功能测试原则

功能测试分为三个阶段：

- 单元测试
- 集成测试
- 端到端（E2E）测试

功能测试阶段可以用*测试金字塔*来描述：

![](img/cbd069395d7b824346b69b1f92e0fb4a_344_0.png)

测试金字塔描述了每个阶段测试的相对数量。大多数测试是单元测试。第二多的是集成测试，最少的是E2E测试。单元测试应覆盖软件组件的整个代码库。单元测试侧重于将单个公共函数作为（代码）单元进行测试。软件组件集成测试涵盖了经过单元测试的函数集成为一个完整工作软件组件的过程，包括测试与外部服务的接口。外部服务的例子包括数据库、消息代理和其他微服务。E2E测试侧重于测试完整软件系统的端到端功能。

关于不同的测试级别和阶段，使用了各种不同的术语：

- 模块测试（单元测试的旧称）
- （软件）组件测试（与集成测试相同）
- 系统（集成）测试（与E2E测试相同）

术语*组件测试*也用于仅表示软件组件中经过单元测试的模块的集成，而不测试外部接口。并且与*组件测试*术语相关联的是*集成测试*术语，用于表示软件组件外部接口的测试。这里我使用*集成测试*术语来表示经过单元测试的模块的集成和外部接口的测试。通常没有理由将这些测试分成单独的测试阶段。

### 6.1.1：单元测试原则

> *单元测试应将公共函数作为隔离的单元进行测试，并尽可能提高覆盖率。隔离意味着依赖项（其他类/模块/服务）被模拟。*

单元测试应仅为公共函数编写。不要试图单独测试私有函数。它们应该在测试公共函数时被间接测试。单元测试应测试函数规范，即函数在不同场景下做什么，而不是函数如何实现。当你只对公共函数进行单元测试时，你可以更容易地重构函数实现，例如，重写公共函数使用的私有函数，而无需修改相关的单元测试。

下面是一个使用私有函数的公共函数示例：

图 6.2. parse_config.py

```
from other_module import do_something

def __read_file(...):
    # ...

def parse_config(...):
    # ...
    # __read_file(...)
    # do_something(...)
    # ...
```

在上面的`parse_config.py`模块中，有一个公共函数`parse_config`和一个私有函数`__read_file`。在单元测试中，你应该隔离地测试公共的`parse_config`函数，并模拟从另一个模块导入的`do_something`函数。并且你在测试公共的`parse_config`函数时，间接测试了私有的`__read_file`函数。

下面是使用类编写的上述示例。你以与上述版本类似的方式测试基于类的版本。你只为公共的`parse_config`方法编写单元测试。这些测试将间接测试私有的`__read_file`方法。你必须为`ConfigParser`构造函数提供一个`OtherClass`类的模拟实例。

```
class OtherClass:
    # ...

    def do_something(self, ...) -> None:
        # ...

class ConfigParser:
    def __init__(self, other_class: OtherClass):
        self.__other_class = other_class

    # ...

    def parse_config(self, ...):
        # ...
        # self.__read_file(...)
        # self.__other_class.do_something(...)
        # ...

    def __read_file(self, ...):
        # ...
```

单元测试应测试公共函数的所有功能：正常路径、可能的故障情况、安全问题和边缘情况，以便函数的每一行代码都至少被一个单元测试覆盖。函数中的安全问题主要与函数接收的输入有关。该输入是否安全？如果你的函数从最终用户接收未经验证的输入数据，则必须根据恶意最终用户的可能攻击来验证该数据。

下面列出了一些边缘情况的示例：-   最后的循环计数器值是否正确？此测试应能检测出可能的“差一”错误
-   用空数组进行测试
-   用允许的最小值进行测试
-   用允许的最大值进行测试
-   用负值进行测试
-   用零值进行测试
-   用非常长的字符串进行测试
-   用空字符串进行测试
-   用具有不同精度的浮点值进行测试
-   用舍入方式不同的浮点值进行测试
-   用非常小的浮点值进行测试
-   用非常大的浮点值进行测试

单元测试不应测试依赖项的功能。那是集成测试要测试的内容。单元测试应隔离地测试一个函数。如果一个函数依赖于在不同类（或模块）中定义的一个或多个其他函数，则应模拟这些依赖项。*模拟*是模仿真实对象或函数行为的东西。模拟将在本节后面更详细地描述。

隔离测试函数有两个好处。它使测试更快。这是一个真正的好处，因为你可能有大量的单元测试，并且经常运行它们，因此单元测试的执行时间尽可能短至关重要。另一个好处是你不需要设置外部依赖项，如数据库、消息代理和其他微服务，因为你模拟了依赖项的功能。

单元测试为你提供安全保障，防止在重构代码时引入意外的错误。单元测试确保实现代码符合函数规范。并且应该记住，第一次就写出完美的代码是很难的。如果你想保持代码库的整洁并避免技术债务，你必然会进行重构。而当你重构时，单元测试会站在你这边，防止意外引入错误。

## 6.1.1.1：测试驱动开发（TDD）

测试驱动开发（TDD）是一种软件开发过程，其中软件需求在软件实现之前被表述为测试用例。这与先实现软件，然后再编写测试用例的做法相反。

我在该行业工作了近30年，当我开始编码时，没有自动化测试或测试驱动开发。直到2010年，我才开始编写自动化单元测试。由于这个背景，TDD对我来说相当困难，因为我已经习惯了一些事情：先实现软件，然后再进行测试。我猜你们中的许多人也是这样学习的，这使得转向TDD相当困难。很少有材料使用TDD来教授主题。互联网上充斥着书籍、课程、视频、博客和其他文章，它们没有教你正确的开发方式：TDD。这本书也是如此。我在书中展示了代码示例，但我没有使用TDD来展示它们，因为这会使一切变得复杂和冗长。

我建议你从小处着手开始使用TDD。开始使用TDD的最佳方式是在你实现一个全新的软件组件时。你必须系统地持续练习TDD，即使一开始感觉不自然。只有这样，你才能养成始终使用TDD的习惯。

纯粹的TDD循环包括以下步骤：

1.  为指定的功能添加一个测试
2.  运行所有测试（刚添加的测试应该失败，因为它测试的功能尚未实现）
3.  编写最简单的代码使测试通过
4.  运行所有测试。（它们现在应该通过了）
5.  根据需要进行重构（现有测试应确保任何东西都不会被破坏）
6.  从第一步重新开始，直到所有功能都已实现、重构和测试

让我们继续一个例子。假设待办事项列表中有以下用户故事等待实现：

> 从配置字符串解析配置属性到配置对象。可以从配置对象访问配置属性。如果解析配置失败，则应产生一个错误。

让我们首先为指定功能的“正常路径”编写一个测试：

```python
import unittest

from ConfigParserImpl import ConfigParserImpl

class ConfigParserTests(unittest.TestCase):
    config_parser = ConfigParserImpl()

    def test_parse(self):
        # GIVEN
        config_str = 'propName1=value1\npropName2=value2'

        # WHEN
        config = self.config_parser.parse(config_str)

        # THEN
        self.assertEqual(config.get_property_value('propName1'), 'value1')
        self.assertEqual(config.get_property_value('propName2'), 'value2')
```

现在，如果我们运行所有测试，会得到一个编译错误，这意味着我们编写的测试用例还不会通过。接下来，我们将编写最简单的代码，使测试用例既能编译又能通过：

```python
from typing import Protocol, Final

class Configuration(Protocol):
    def get_property_value(self, property_name: str) -> str:
        pass

class ConfigurationImpl(Configuration):
    def __init__(self, prop_name_to_value_dict: dict[str, str]):
        self.__prop_name_to_value_dict: Final = prop_name_to_value_dict

    def get_property_value(self, property_name: str) -> str | None:
        return self.__prop_name_to_value_dict.get(property_name);

class ConfigParser(Protocol):
    def parse(self, config_str: str) -> Configuration:
        pass

class ConfigParserImpl(ConfigParser):
    def parse(self, config_str: str) -> Configuration:
        # Parse config_str and assign properties to
        # 'prop_name_to_value_dict' variable

        return ConfigurationImpl(prop_name_to_value_dict)
```

现在测试通过了，我们可以添加新功能。让我们为解析失败的情况添加一个测试。我们现在可以通过首先创建一个失败的测试来重复TDD循环：

```python
import unittest

from ConfigParser import ConfigParser

class ConfigParserTests(unittest.TestCase):
    # ...

    def test_try_parse_when_parsing_fails(self):
        # GIVEN
        config_str = 'invalid'

        try:
            # WHEN
            self.config_parser.try_parse(config_str)

            # THEN
            self.fail('ConfigParser.ParseError should have been raised')
        except ConfigParser.ParseError:
            # THEN error was successfully raised
```

接下来，我们应该重构实现以使第二个测试通过：

```python
from typing import Protocol

from Configuration import Configuration
from DataExporterError import DataExporterError

class ConfigParser(Protocol):
    class ParseError(DataExporterError):
        pass

    def try_parse(self, config_str: str) -> Configuration:
        pass

class ConfigParserImpl(ConfigParser):
    def try_parse(self, config_str: str) -> Configuration:
        # Try parse config_str and if successful
        # assign config properties to 'prop_name_to_value_dict'
        # variable

        if prop_name_to_value_dict is None:
            raise self.ParseError()
        else:
            return ConfigurationImpl(prop_name_to_value_dict)
```

我们还需要重构第一个单元测试，使其调用`try_parse`而不是`parse`。我们可以继续为其他功能添加测试用例。

对我来说，上面描述的TDD循环听起来有点繁琐。但是，预先创建测试有明显的好处。当测试首先被定义时，人们通常不太可能忘记测试或实现某些东西。这是因为TDD更好地迫使你思考函数规范：正常路径、可能的安全问题、边界和失败情况。

如果你不实践TDD，总是先进行实现，那么你更有可能忘记一个边界情况或特定的失败/安全场景。当你不实践TDD时，你会直接进入实现，并且你往往只考虑正常路径并努力让它们工作。当你专注于让正常路径工作时，你不会过多地考虑边界情况和失败/安全场景，因为你在精神上如此强烈地专注于正常路径。如果你忘记实现一个边界情况或失败场景，你就不会测试它。你可能对一个函数有100%的单元测试覆盖率，但某个特定的边界情况或失败/安全场景既未实现也未测试。这也发生在我身上。而且不止一次。只有在意识到TDD可以让我避免这类错误之后，我才开始认真对待TDD。在那之前，我没有意识到TDD的实际价值，认为它是一个有点过于繁琐的过程。如果这本书对你来说只有一个要点，那应该是TDD。实践TDD将使你编写更少的错误，并使编写代码压力更小（这很重要！），因为你在开始编写任何代码之前就已经处理了错误情况和边界情况。

作为上述TDD循环的替代方案，你可以进行简化版的TDD。在简化版的TDD中，你首先像在完整的TDD中一样指定函数。从函数规范中，你提取所有需要的测试，包括“正常路径”、边界情况和失败/安全场景。然后在所有测试中放入一个 `fail` 调用，以免忘记稍后实现它们。此外，你可以在每个测试中添加一个注释，说明在给定输入下的预期结果。例如，在失败场景中，你可以添加一个注释说明预期会引发哪种错误；在边界情况中，你可以添加一个注释说明当输入为 x 时，预期输出为 y。一旦你实现了某个测试，就可以移除该注释。

假设我们有以下函数规范：

> 配置解析器的 `parse` 函数将 JSON 格式的配置解析为配置对象。如果配置 JSON 无法解析，该函数应产生一个错误。配置 JSON 由可选属性和必需属性（具有特定类型的名称和值）组成。缺少必需属性应产生一个错误，而缺少可选属性应使用默认值。额外的属性应被丢弃。具有无效值类型的属性应产生一个错误。支持两种属性类型：整数和字符串。整数必须在指定范围内，字符串有最大长度。必需的配置属性如下：名称（类型）... 可选的配置属性如下：名称（类型）...

让我们首先为“正常路径”场景编写一个失败的测试用例：

```
import unittest

class ConfigParserTests(unittest.TestCase):
    def test_try_parse(self):
        # 正常路径场景
        self.fail()
```

接下来，让我们为其他情况编写失败的测试用例：

```
import unittest

class ConfigParserTests(unittest.TestCase):
    # ...

    def test_try_parse__when_json_parsing_fails(self):
        # 失败场景，应产生一个错误
        self.fail()

    def test_try_parse__when_mandatory_prop_is_missing(self):
        # 失败场景，应产生一个错误
        self.fail()

    def test_try_parse__when_optional_prop_is_missing(self):
        # 应使用默认值
        self.fail()

    def test_try_parse__with_extra_props(self):
        # 额外的属性应被丢弃
        self.fail()

    def test_try_parse__when_prop_has_invalid_type(self):
        # 失败场景，应产生一个错误
        self.fail()

    def test_try_parse__when_integer_prop_out_of_range(self):
        # 输入验证安全场景，应产生一个错误
        self.fail()

    def test_try_parse__when_string_prop_too_long(self):
        # 输入验证安全场景，应产生一个错误
        self.fail()
```

现在你以场景的形式获得了函数的高级规范。接下来，你可以继续进行函数实现。完成函数实现后，逐一实现测试，并移除 `fail` 调用。

这种方法的好处在于，你不必在实现源代码文件和测试源代码文件之间不断切换。在每个阶段，你可以专注于一件事：

1) 函数规范

- 函数做什么？（正常路径场景）
- 可能出现哪些失败？（失败场景）
- 是否存在安全问题？（安全场景）
- 是否存在边界情况？（边界情况场景）
- 在指定函数时，并不强制要求将规范写下来。你可以在脑海中完成，特别是如果函数相当简单。对于更复杂的函数，写下规范可能有助于你完全理解函数真正应该做什么。

2) 将不同场景实现为失败的单元测试
3) 函数实现
4) 单元测试的实现

在实际生活中，初始的函数规范并非总是 100% 正确或完整。在函数实现过程中，你可能会发现例如一个初始规范中没有的新错误场景。你应该立即为该新场景添加一个新的失败单元测试，以免忘记稍后实现它。一旦你认为函数实现完成，请逐行检查函数代码，检查是否有任何行可能产生尚未考虑到的错误。养成这个习惯将减少你意外地在函数代码中留下某些未处理错误的可能性。

有时你需要修改现有函数，因为由于各种原因（如不可能或不可行），你并不总是能够遵循开闭原则。当你需要修改现有函数时，请遵循以下步骤：

1) 函数变更规范

- 函数正常路径场景有何变化？
- 失败场景有何变化？
- 安全场景有何变化？
- 边界情况有何变化？

2) 添加/删除/修改测试

- 将新场景添加为失败测试
- 删除已移除场景的测试
- 修改现有测试

3) 对函数进行实现变更

4) 实现单元测试

让我们举一个例子，我们修改配置解析器，使其在配置包含额外属性时应产生一个错误。现在我们已经定义了变更的规范。接下来我们需要修改测试。我们需要按如下方式修改 `test_try_parse__with_extra_props` 方法：

```
import unittest

class ConfigParserTests(unittest.TestCase):
    # ...

    def test_try_parse__with_extra_props(self):
        self.fail()
```

接下来，我们实现所需的变更，然后实现上述单元测试。

让我们再举一个例子，我们修改配置解析器，使其除了 JSON 外，还可以接受 YAML 格式的配置。我们需要添加以下失败的单元测试：

```
import unittest

class ConfigParserTests(unittest.TestCase):
    # ...

    def test_try_parse__when_config_in_yaml_format(self):
        self.fail()

    def test_try_parse__when_yaml_parsing_fails(self):
        # 应产生一个错误
        self.fail()
```

我们还应该重命名以下测试方法：`test_try_parse` 和 `test_try_parse__when_parsing_fails` 改为 `test_try_parse__when_config_in_json_format` 和 `test_try_parse__when_json_parsing_fails`。接下来我们实现对函数的变更，最后我们实现这两个新测试。（根据实际的测试实现，你可能需要也可能不需要对 JSON 解析相关的测试进行小的修改以使其通过。）

作为最后一个例子，让我们进行以下变更：配置没有任何可选属性，所有属性都是必需的。这意味着我们可以移除以下测试：`test_try_parse__when_optional_prop_is_missing`。我们还需要更改 `test_try_parse__when_mandatory_prop_is_missing` 测试。为了记住修改测试，我们可以最初将测试修改为一个失败测试：

```
import unittest

class ConfigParserTests(unittest.TestCase):
    # ...

    def test_try_parse__when_mandatory_prop_is_missing(self):
        self.fail()
        # 此处为现有实现 ...
```

一旦我们实现了变更，我们就可以完成测试的实现并移除 `fail` 调用。

在上面的例子中，我们的函数规范包含了正常路径和失败/安全场景。让我们看一个包含边界情况的函数规范示例。我们应该为一个字符串类实现一个 `contains` 方法。该方法应执行以下操作：

> 该方法接受一个字符串参数，如果该字符串在字符串对象表示的字符串中被找到，则返回 `True`，否则返回 `False`。

我们可以立即注意到有两个正常路径，我们可以创建以下失败测试：

```
import unittest

class StringTests(unittest.TestCase):
    def test_contains__when_arg_string_is_found(self):
        # 应返回 True
        self.fail()

    def test_contains__when_arg_string_is_not_found(self):
        # 应返回 False
        self.fail()
```

我们可能还想测试几个边界情况，以 100% 确保函数在每种情况下都能正确工作：- 字符串相等
- 其中一个或两个字符串为空字符串
- 参数字符串出现在另一个字符串的开头
- 参数字符串出现在另一个字符串的结尾
- 参数字符串比另一个字符串长

我们可以将上述边界情况转化为失败的测试：

```python
import unittest

class StringTests(unittest.TestCase):
    # ...

    def test_contains__strings_are_equal(self):
        self.fail()
        # 应返回 True

    def test_contains__when_both_strings_are_empty(self):
        self.fail()
        # 应返回 True

    def test_contains__when_arg_string_is_empty(self):
        self.fail()
        # 应返回 False

    def test_contains__when_this_string_is_empty(self):
        self.fail()
        # 应返回 False

    def test_contains__when_arg_string_is_found_at_begin(self):
        self.fail()
        # 应返回 True

    def test_contains__when_arg_string_is_found_at_end(self):
        self.fail()
        # 应返回 True

    def test_contains__when_arg_string_is_longer_than_this_string(self):
        self.fail()
        # 应返回 False
```

## 6.1.1.2：命名约定

当被测试的函数位于一个类中时，应创建一个相应命名的单元测试类。例如，如果有一个 `ConfigParser` 类，相应的单元测试类应为 `ConfigParserTests`。这样，可以轻松定位包含特定实现类单元测试的文件。

测试方法名称应以 `test` 前缀开头，后跟被测试方法的名称。例如，如果被测试的方法是 `try_parse`，测试方法名称应为 `test_try_parse`。通常，单个函数会有多个测试。所有测试方法名称都应以 `test_<function-name>` 开头，但测试方法名称还应包含测试方法所测试的具体场景的描述，例如：`test_try_parse__when_parsing_fails`。被测试场景的名称与被测试函数名称之间用两个下划线分隔。

## 6.1.1.3：模拟（Mocking）

Python 有一个用于单元测试中模拟的 `unittest.mock` 库。它允许你用模拟对象替换被测试系统的部分，并对它们的使用方式进行断言。模拟库提供以下模拟方式：

- 使用 `@patch` 修补类/对象/属性/方法
- 使用 `@patch` 修补对象属性/方法
- 使用 `Mock` 构造函数创建模拟函数
- 修补字典

让我们通过涵盖所有四种不同模拟方式的示例来说明。首先，我们将有一个 Kafka 客户端，允许在 Kafka 代理上创建 Kafka 主题。我们希望主题创建是幂等的，即如果主题已存在，则不执行任何操作。在本练习中，我们将使用简化的 TDD 版本，首先将 Kafka 客户端的功能指定为失败的单元测试：

```python
from unittest import TestCase

class KafkaClientTests(TestCase):
    def test_try_create_topic__when_create_succeeds(self):
        self.fail()

    def test_try_create_topic__when_create_fails(self):
        # 抛出错误
        self.fail()

    def test_try_create_topic__when_topic_exists(self):
        self.fail()
```

接下来，我们将编写 `KafkaClient` 类的实现：

```python
from confluent_kafka import KafkaError, KafkaException
from confluent_kafka.admin import AdminClient
from confluent_kafka.cimpl import NewTopic

from DataExporterError import DataExporterError

class KafkaClient:
    def __init__(self, kafka_host: str):
        self.__admin_client = AdminClient(
            {'bootstrap.servers': kafka_host}
        )

class CreateTopicError(DataExporterError):
    pass

def try_create_topic(
    self,
    name: str,
    num_partitions: int,
    replication_factor: int,
    retention_in_secs: int,
    retention_in_gb: int
):
    topic = NewTopic(
        name,
        num_partitions,
        replication_factor,
        config={
            'retention.ms': str(retention_in_secs * 1000),
            'retention.bytes': str(retention_in_gb * pow(10, 9))
        }
    )

    try:
        topic_name_to_creation_dict = (
            self.__admin_client.create_topics([topic])
        )
        topic_name_to_creation_dict[name].result()
    except KafkaException as error:
        if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
            raise self.CreateTopicError(error)
```

让我们实现第一个测试方法来测试 `try_create_topic` 方法的成功执行：

```python
from unittest import TestCase
from unittest.mock import Mock, patch

from KafkaClient import KafkaClient


class KafkaClientTests(TestCase):
    @patch('asyncio.Future')
    @patch('KafkaClient.NewTopic')
    @patch('KafkaClient.AdminClient')
    def test_try_create_topic__when_create_succeeds(
        self,
        admin_client_class_mock: Mock,
        new_topic_class_mock: Mock,
        future_class_mock: Mock,
    ):
        # 给定
        admin_client_mock = admin_client_class_mock.return_value
        future_mock = future_class_mock.return_value
        admin_client_mock.create_topics.return_value = {
            'test': future_mock
        }
        kafka_client = KafkaClient('localhost:9092')

        # 当
        kafka_client.try_create_topic(
            'test',
            num_partitions=3,
            replication_factor=2,
            retention_in_secs=5 * 60,
            retention_in_gb=100,
        )

        # 那么
        admin_client_class_mock.assert_called_once_with(
            {'bootstrap.servers': 'localhost:9092'}
        )

        new_topic_class_mock.assert_called_once_with(
            'test',
            3,
            2,
            config={
                'retention.ms': str(5 * 60 * 1000),
                'retention.bytes': str(100 * pow(10, 9)),
            },
        )

        admin_client_mock.create_topics.assert_called_once_with(
            [new_topic_class_mock.return_value]
        )

        future_mock.result.assert_called_once()
```

在上面的示例中，我们使用了来自 Confluent Kafka 库的两个类 `AdminClient` 和 `NewTopic`。我们无法在单元测试中直接访问这些真实依赖项，但必须模拟它们。这意味着我们修补了从 `KafkaClient` 导入的 `NewTopic` 和 `AdminClient` 类，而 `KafkaClient` 分别从 `confluent_kafka.cimpl` 和 `confluent_kafka.admin` 导入它们。模拟是使用 `@patch` 装饰器创建的。我们还模拟了 `asyncio.Future` 类，因为 `AdminClient.create_topics` 返回一个包含 `Future` 实例的字典。模拟类的版本作为参数提供给 `test_try_create_topic` 方法。我们可以使用 `return_value` 属性从模拟类访问模拟的 `AdminClient` 和 `Future` 实例。执行测试后，我们需要验证对模拟对象的调用。

让我们为创建主题失败的情况添加另一个测试：

```python
from unittest import TestCase
from unittest.mock import Mock, patch

from confluent_kafka import KafkaError, KafkaException
from KafkaClient import KafkaClient

class KafkaClientTests(TestCase):
    @patch('asyncio.Future')
    @patch('KafkaClient.NewTopic')
    @patch('KafkaClient.AdminClient')
    def test_try_create_topic__when_create_fails(
        self,
        admin_client_class_mock: Mock,
        new_topic_class_mock: Mock,
        future_class_mock: Mock,
    ):
        # 给定
        kafka_client = KafkaClient('localhost:9092')
        admin_client_mock = admin_client_class_mock.return_value
        future_mock = future_class_mock.return_value
        admin_client_mock.create_topics.return_value = {
            'test': future_mock
        }
        future_mock.result.side_effect = KafkaException(KafkaError(1))

        # 当
        try:
            kafka_client.try_create_topic(
                'test',
                num_partitions=3,
                replication_factor=2,
                retention_in_secs=5 * 60,
                retention_in_gb=100,
            )
            self.fail('应抛出 KafkaException')
        except KafkaClient.CreateTopicError:
            pass

        # 那么
        admin_client_class_mock.assert_called_once_with(
            {'bootstrap.servers': 'localhost:9092'}
        )

        new_topic_class_mock.assert_called_once_with(
            'test',
            3,
            2,
            config={
                'retention.ms': str(5 * 60 * 1000),
                'retention.bytes': str(100 * pow(10, 9)),
            },
        )

        admin_client_mock.create_topics.assert_called_once_with(
            [new_topic_class_mock.return_value]
        )
```

上述测试的关键是使 `Future` 模拟实例的 `result` 方法作为副作用抛出 `KafkaException`。然后在实际的测试代码中，我们确保抛出了 `KafkaException`，如果没有，则使用消息“应抛出 KafkaException”使测试失败。

上述两个测试方法包含重复的代码。我们也应该保持测试代码的整洁。让我们重构测试用例以移除重复的代码。我们引入一个 `set_up` 方法，该方法将进行模拟的设置和 `KafkaClient` 实例的创建。我们将常见的模拟调用断言重构到一个单独的私有方法中，供两个测试使用。修补器（patchers）是为整个类设置的，这意味着单元测试框架将修补每个以 `test` 前缀开头的方法。

## 测试原则

```python
from unittest import TestCase
from unittest.mock import Mock, patch

from confluent_kafka import KafkaError, KafkaException
from KafkaClient import KafkaClient


@patch('asyncio.Future')
@patch('KafkaClient.NewTopic')
@patch('KafkaClient.AdminClient')
class KafkaClientTests(TestCase):
    def set_up(
        self,
        admin_client_class_mock: Mock,
        future_class_mock: Mock,
    ) -> None:
        # 给定
        self.admin_client_mock = admin_client_class_mock.return_value
        self.future_mock = future_class_mock.return_value
        self.admin_client_mock.create_topics.return_value = {
            'test': self.future_mock
        }
        self.topic_params = {
            'num_partitions': 3,
            'replication_factor': 2,
            'retention_in_secs': 5 * 60,
            'retention_in_gb': 100,
        }
        self.kafka_client = KafkaClient('localhost:9092')

    def test_try_create_topic__when_create_succeeds(
        self,
        admin_client_class_mock: Mock,
        new_topic_class_mock: Mock,
        future_class_mock: Mock,
    ):
        # 给定
        self.set_up(admin_client_class_mock, future_class_mock)

        # 当
        self.kafka_client.try_create_topic('test', **self.topic_params)

        # 则
        self.__assert_mock_calls(
            admin_client_class_mock, new_topic_class_mock
        )
        self.future_mock.result.assert_called_once()

    def test_try_create_topic__when_create_fails(
        self,
        admin_client_class_mock: Mock,
        new_topic_class_mock: Mock,
        future_class_mock: Mock,
    ):
        # 给定
        self.set_up(admin_client_class_mock, future_class_mock)
        self.future_mock.result.side_effect = KafkaException(KafkaError(1))

        # 当
        try:
            self.kafka_client.try_create_topic('test', **self.topic_params)
            self.fail('应该抛出 KafkaException 异常')
        except KafkaClient.CreateTopicError:
            pass

        # 则
        self.__assert_mock_calls(
            admin_client_class_mock, new_topic_class_mock
        )

    def __assert_mock_calls(
        self, admin_client_class_mock: Mock, new_topic_class_mock: Mock
    ):
        admin_client_class_mock.assert_called_once_with(
            {'bootstrap.servers': 'localhost:9092'}
        )

        new_topic_class_mock.assert_called_once_with(
            'test',
            3,
            2,
            config={
                'retention.ms': str(5 * 60 * 1000),
                'retention.bytes': str(100 * pow(10, 9)),
            },
        )

        self.admin_client_mock.create_topics.assert_called_once_with(
            [new_topic_class_mock.return_value]
        )
```

让我们为最后一个测试方法添加实现：

```python
class KafkaClientTests(TestCase):
    def test_try_create_topic__when_topic_exists(self):
        # 给定
        self.future_mock.result.side_effect = KafkaException(
            KafkaError(KafkaError.TOPIC_ALREADY_EXISTS)
        )

        # 当
        self.kafka_client.try_create_topic('test', **self.topic_params)

        # 则
        self.__assert_mock_calls()
```

在上面的例子中，我们使用了 `patch` 来为类创建模拟对象。让我们再看一个直接修补库方法的例子。我们需要实现一个 HTTP 客户端，该客户端允许从 URL 获取 JSON 数据并解析为字典。让我们利用简化的 TDD 并列出 HTTP 客户端所有可能的场景：

- 成功从 URL 获取并解析 JSON 数据。
- 成功从 URL 获取数据，但解析数据失败。应抛出错误。
- 从 URL 获取 JSON 数据失败，HTTP 状态码 >=400。应抛出错误。
- 无法成功连接到 URL（例如，URL 格式错误、连接被拒绝、连接超时等）。应抛出错误。

让我们编写一个包含失败测试方法的测试用例：

```python
from unittest import TestCase

class HttpClientTests(TestCase):
    def test_try_fetch_dict__when_fetch_succeeds(self):
        self.fail()

    def test_try_fetch_dict__when_json_parse_fails(self):
        # 应抛出错误
        self.fail()

    def test_try_fetch_dict__when_response_has_error(self):
        # 应抛出错误
        self.fail()

    def test_try_fetch_dict__when_remote_connection_fails(self):
        # 应抛出错误
        self.fail()
```

现在我们可以实现 `HttpClient` 类，使其提供上述测试方法指定的功能。

```python
from typing import Any

import requests

class HttpClient:
    # 将下面的 'Exception' 替换为软件组件的
    # 基础错误类
    class Error(Exception):
        pass

    def try_fetch_dict(self, url: str) -> dict[str, Any]:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise self.Error(error)
```

如果我们没有使用简化的 TDD，我们很容易最终得到以下只关注正常路径的实现：

```python
from typing import Any

import requests

class HttpClient:
    def fetch_dict(self, url: str) -> dict[str, Any]:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.json()
```

问题在于，很容易忘记处理 `requests.get` 和 `Response.json` 方法可能抛出的错误。使用 TDD 迫使我们在实现任何功能之前停下来，除了正常路径场景外，还要思考可能的错误场景和边缘情况。

让我们实现第一个测试方法：

```python
from unittest import TestCase
from unittest.mock import Mock, patch

from HttpClient import HttpClient

URL = 'https://localhost:8080/'
DICT = {'test': 'test'}

class HttpClientTests(TestCase):
    @patch('requests.Response.__new__')
    @patch('requests.get')
    def test_try_fetch_dict__when_fetch_succeeds(
        self, requests_get_mock: Mock, response_mock: Mock
    ):
        # 给定
        requests_get_mock.return_value = response_mock
        response_mock.status_code = 200
        response_mock.raise_for_status.return_value = None
        response_mock.json.return_value = DICT

        # 当
        response_dict = HttpClient().try_fetch_dict(URL)

        # 则
        requests_get_mock.assert_called_once_with(URL, timeout=60)
        self.assertDictEqual(response_dict, DICT)
```

让我们实现第二个测试方法：

```python
import json
from unittest import TestCase
from unittest.mock import Mock, patch

import requests
from HttpClient import HttpClient

URL = 'https://localhost:8080/'
DICT = {'test': 'test'}

class HttpClientTests(TestCase):
    @patch('requests.Response.__new__')
    @patch('requests.get')
    def test_try_fetch_dict__when_json_parse_fails(
        self, requests_get_mock: Mock, response_mock: Mock
    ):
        # 给定
        requests_get_mock.return_value = response_mock
        response_mock.status_code = 200
        response_mock.raise_for_status.return_value = None
        response_mock.json.side_effect = requests.JSONDecodeError(
            'JSON decode error', json.dumps(DICT), 1
        )

        # 当
        try:
            HttpClient().try_fetch_dict(URL)
            self.fail('应该抛出 HttpClient.Error 异常')
        except HttpClient.Error as error:
            # 则
            self.assertIn('JSON decode error', str(error))

        # 则
        requests_get_mock.assert_called_once_with(URL, timeout=60)
```

现在我们再次有了重复的测试代码，必须重构测试：

```python
import json
from unittest import TestCase
from unittest.mock import Mock, patch

import requests
from HttpClient import HttpClient

URL = 'https://localhost:8080/'
DICT = {'test': 'test'}

@patch('requests.Response.__new__')
@patch('requests.get')
class HttpClientTests(TestCase):
    def test_try_fetch_dict__when_fetch_succeeds(
        self, requests_get_mock: Mock, response_mock: Mock
    ):
        # 给定
        requests_get_mock.return_value = response_mock
        response_mock.status_code = 200
```

response_mock.raise_for_status.return_value = None
response_mock.json.return_value = DICT

# WHEN
dict_ = HttpClient().try_fetch_dict(URL)

# THEN
requests_get_mock.assert_called_once_with(URL, timeout=60)
self.assertDictEqual(dict_, DICT)

def test_try_fetch_dict__when_json_parse_fails(
    self, requests_get_mock: Mock, response_mock: Mock
):
    # GIVEN
    requests_get_mock.return_value = response_mock
    response_mock.status_code = 200
    response_mock.raise_for_status.return_value = None
    response_mock.json.side_effect = requests.JSONDecodeError(
        'JSON decode error', json.dumps(DICT), 1
    )

    # WHEN
    self.assertRaises(
        HttpClient.Error, HttpClient().try_fetch_dict, URL
    )

    # THEN
    requests_get_mock.assert_called_once_with(URL, timeout=60)

让我们添加最后两个测试方法来完成这个测试用例。我还改用了 try-except 块，使用 assertRaises 方法来展示验证函数调用是否引发错误的另一种方式。

```python
import json
from unittest import TestCase
from unittest.mock import Mock, patch

import requests
from HttpClient import HttpClient

URL = 'https://localhost:8080/'
DICT = {'test': 'test'}


@patch('requests.Response.__new__')
@patch('requests.get')
class HttpClientTests(TestCase):
    # ...

    def test_try_fetch_dict__when_response_has_error(
        self, requests_get_mock: Mock, response_mock: Mock
    ):
        # GIVEN
        requests_get_mock.return_value = response_mock
        response_mock.status_code = 500
        response_mock.raise_for_status.side_effect = requests.HTTPError()

        # WHEN
        self.assertRaises(
            HttpClient.Error, HttpClient().try_fetch_dict, URL
        )

        # THEN
        requests_get_mock.assert_called_once_with(URL, timeout=60)

    def test_try_fetch_dict__when_remote_connection_fails(
        self, requests_get_mock: Mock, response_mock: Mock
    ):
        # GIVEN
        requests_get_mock.side_effect = requests.ConnectionError()

        # WHEN
        self.assertRaises(
            HttpClient.Error, HttpClient().try_fetch_dict, URL
        )

        # THEN
        requests_get_mock.assert_called_once_with(URL, timeout=60)
```

让我们看一个使用 `@patch.dict` 的例子。假设我们有以下没有单元测试的代码：

```python
import os
import sys

from KafkaClient import KafkaClient


def get_environ_var(name: str) -> str:
    return (
        os.environ.get(name)
        or f'Environment variable {name} is not defined'
    )


def main():
    kafka_client = KafkaClient(get_environ_var('KAFKA_HOST'))

    try:
        kafka_client.try_create_topic(
            get_environ_var('KAFKA_TOPIC'),
            num_partitions=3,
            replication_factor=2,
            retention_in_secs=5 * 60,
            retention_in_gb=100,
        )
    except KafkaClient.CreateTopicError:
        sys.exit(1)


if __name__ == '__main__':
    main()
```

在单元测试用例中，我们使用 `@patch.dict` 来修补 `os.environ` 字典。在第二个测试方法中，我们还使用了 `@patch.object` 装饰器，而不是普通的 `@patch` 装饰器。`@patch.object` 方法用于修补 `KafkaClient` 类型对象中的方法/属性。

## 测试原则

```python
import os
from unittest import TestCase
from unittest.mock import Mock, patch

from KafkaClient import KafkaClient
from main import main

KAFKA_HOST = 'localhost:9092'
KAFKA_TOPIC = 'test'


@patch.dict(os.environ, {'KAFKA_HOST': KAFKA_HOST})
@patch.dict(os.environ, {'KAFKA_TOPIC': KAFKA_TOPIC})
class MainTests(TestCase):
    @patch('main.KafkaClient')
    def test_main__when_exec_succeeds(self, kafka_client_class_mock: Mock):
        # GIVEN
        kafka_client_mock = kafka_client_class_mock.return_value

        # WHEN
        main()

        # THEN
        kafka_client_class_mock.assert_called_once_with(KAFKA_HOST)
        kafka_client_mock.try_create_topic.assert_called_once_with(
            KAFKA_TOPIC,
            num_partitions=3,
            replication_factor=2,
            retention_in_secs=5 * 60,
            retention_in_gb=100,
        )

    @patch.object(KafkaClient, '__init__')
    @patch.object(KafkaClient, 'try_create_topic')
    @patch('sys.exit')
    def test_main__when_exec_failed(
        self,
        sys_exit_mock: Mock,
        try_create_topic_mock: Mock,
        kafka_client_init_mock: Mock,
    ):
        # GIVEN
        kafka_client_init_mock.return_value = None
        try_create_topic_mock.side_effect = KafkaClient.CreateTopicError()

        # WHEN
        main()

        # THEN
        kafka_client_init_mock.assert_called_once_with(KAFKA_HOST)
        sys_exit_mock.assert_called_once_with(1)
```

让我们为使用依赖注入的代码创建单元测试。我们有来自前面章节的以下代码，并且我们想为 `Application` 类的 `run` 方法创建一个单元测试。在下面的例子中，我们假设每个类都在自己的模块中，模块名根据类名命名，并且 `di_container = DiContainer()` 的定义在一个名为 `di_container` 的模块中。

```python
from enum import Enum
from typing import Protocol

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

class LogLevel(Enum):
    ERROR = 1
    WARN = 2
    INFO = 3
    # ...

class Logger(Protocol):
    def log(self, log_level: LogLevel, message: str):
        pass

class StdOutLogger(Logger):
    def log(self, log_level: LogLevel, message: str):
        # Log to standard output

class DiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=['Application']
    )

    logger = providers.Singleton(StdOutLogger)

di_container = DiContainer()

class Application:
    @inject
    def __init__(self, logger: Logger = Provide['logger']):
        self.__logger = logger

    def run(self):
        self.__logger.log(LogLevel.INFO, 'Starting application')
        # ...
```

在下面的单元测试中，我们首先创建一个 `Logger` 类的模拟实例，然后用该模拟对象覆盖 DI 容器中的 logger 提供者。我们使用 `override` 上下文管理器来定义覆盖的范围。

```python
from unittest import TestCase
from unittest.mock import Mock

from Application import Application, Logger
from di_container import di_container
from Logger import LogLevel

class ApplicationTests(TestCase):
    def test_run__when_execution_succeeds(self):
        logger_mock = Mock(Logger)

        with di_container.logger.override(logger_mock):
            # GIVEN
            application = Application()

            # WHEN
            application.run()

            # THEN
            logger_mock.log.assert_called_once_with(
                LogLevel.INFO, 'Starting application'
            )
```

### 6.1.1.4：Web UI 组件单元测试

UI 组件单元测试与常规单元测试不同，因为如果你有一个 React 函数式组件，你不一定能孤立地测试 UI 组件的功能。你必须通过将组件挂载到 DOM 来进行 UI 组件单元测试，然后通过触发事件等方式执行测试。这样，你就可以测试 UI 组件的事件处理函数。渲染部分也应该被测试。可以通过生成渲染组件的快照并将其存储在版本控制中来进行测试。进一步的渲染测试应将渲染结果与存储在版本控制中的快照进行比较。

下面是测试 React 组件 `NumberInput` 渲染的示例：

图 6.3. NumberInput.test.jsx

```javascript
import renderer from 'react-test-renderer';
// ...

describe('NumberInput', () => {
  // ...

  describe('render', () => {
    it('renders with buttons on left and right', () => {
      const numberInputAsJson =
        renderer
          .create(<NumberInput buttonPlacement="leftAndRight"/>)
          .toJSON();

      expect(numberInputAsJson).toMatchSnapshot();
    });

    it('renders with buttons on right', () => {
```

const numberInputAsJson =
  renderer
    .create(<NumberInput buttonPlacement="right"/>)
    .toJSON();

expect(numberInputAsJson).toMatchSnapshot();
});
});
```

以下是针对数字输入框减量按钮点击事件处理函数 `decrementValue` 的一个单元测试示例：

**图 6.4. NumberInput.test.jsx**

```
import { render, fireEvent, screen } from '@testing-library/react'
// ...

describe('NumberInput') () => {
  // ...

  describe('decrementValue', () => {
    it('should decrement value by given step amount', () => {
      render(<NumberInput value="3" stepAmount={2} />);
      fireEvent.click(screen.getByText('-'));
      const numberInputElement = screen.getByDisplayValue('1');
      expect(numberInputElement).toBeTruthy();
    });
  });
});
```

在上面的例子中，我们使用了 *testing-library*，它为所有常见的 UI 框架（React、Vue 和 Angular）都提供了实现。这意味着无论你使用哪种 UI 框架，都可以使用基本相同的测试 API。它们之间只有细微的差别，基本上只在于 `render` 方法的语法。如果你已经用 React 实现了一些 UI 组件并为它们编写了单元测试，现在想用 Vue 重新实现，你不需要重新编写所有的单元测试。你只需要对它们进行轻微的修改（例如，修改 `render` 函数的调用）。否则，现有的单元测试应该仍然有效，因为 UI 组件的行为没有改变，只是其内部实现从 React 变成了 Vue。

## 6.1.2：软件组件集成测试原则

> *集成测试旨在测试一个软件组件是否能与实际依赖项协同工作，以及其公共方法是否能正确理解它们所使用的其他公共方法的目的和签名。*

在软件组件集成测试中，一个软件组件的所有公共函数都应至少被一个集成测试所覆盖。但并非所有公共函数的功能都需要被测试，因为这些已经在单元测试阶段完成过了。这就是为什么集成测试的数量少于单元测试。术语 *集成测试* 有时指完整软件系统或产品的集成。然而，它应该仅用于描述软件组件的集成。在测试产品或软件系统时，应使用术语 *E2E 测试* 以避免混淆和误解。

定义集成测试的最佳方式是使用 *行为驱动开发*（BDD）。BDD 鼓励团队使用领域驱动设计和具体示例来形式化对软件组件应如何行为的共同理解。在 BDD 中，行为规范是集成测试的根基。团队可以在初始的领域驱动设计阶段创建行为规范。这种做法将使集成测试左移，意味着编写集成测试可以尽早开始，并与实际实现并行进行。一种广泛使用且推荐的编写行为规范的方式是 *Gherkin* 语言。

使用 Gherkin 语言时，软件组件的行为被描述为特性（features）。每个特性应该有一个单独的文件。这些文件具有 *.feature* 扩展名。每个特性文件描述一个特性以及该特性的一个或多个场景。第一个场景应该是所谓的“快乐路径”场景，其他可能的场景应处理需要测试的额外快乐路径、失败和边缘情况。请记住，你不必测试每一个失败和边缘情况，因为这些已经在单元测试阶段测试过了。

下面是一个 *data-visualization-configuration-service* 中某个特性的简化示例。我们假设该服务是一个 REST API。该特性用于创建一个新的图表。（在实际场景中，图表包含更多属性，例如图表的数据源以及图表中显示的度量和维度等）。在我们的简化示例中，图表包含以下属性：布局 ID、类型、显示的 X 轴类别数量，以及应从作为图表数据源的数据库中获取的图表数据行数。

```
Feature: Create chart
  Creates a new chart

  Scenario: Creates a new chart successfully
    Given chart layout id is 1
    And chart type is "line"
    And X-axis categories shown count is 10
    And fetched row count is 1000

    When I create a new chart

    Then I should get the chart given above
      with response code 201 "Created"
```

上面的例子展示了特性名称是如何在 `Feature` 关键字之后给出的。你可以在特性名称下方添加自由格式的文本，以更详细地描述该特性。接下来，在 `Scenario` 关键字之后定义一个场景。首先给出场景的名称。然后是场景的步骤。每个步骤使用以下关键字之一来定义：`Given`、`When`、`Then`、`And` 和 `But`。一个场景应遵循以下模式：

- 描述初始上下文/设置的步骤（Given/And 步骤）
- 描述事件的步骤（When 步骤）
- 描述事件预期结果的步骤（Then/And 步骤）

我们可以在上面的例子中添加另一个场景：

```
Feature: Create chart
  Creates a new chart

  Scenario: Creates a new chart successfully
    Given chart layout id is 1
    And chart type is "line"
    And X-axis categories shown count is 10
    And fetched row count is 1000

    When I create a new chart

    Then I should get the chart given above
      with status code 201 "Created"

  Scenario: Chart creation fails due to missing mandatory parameter
    When I create a new chart

    Then I should get a response with status code 400 "Bad Request"
    And response body should contain error object with
      "is mandatory field" entry for following fields
      | layout_id              |
      | fetched_row_count      |
      | x_axis_categ_shown_count |
      | type                   |
```

现在我们指定了一个包含两个场景的特性。接下来，我们将实现这些场景。我们希望用 Python 实现集成测试，因此我们将使用支持 Gherkin 语言的 Behave BDD 工具。

我们将集成测试代码放入源代码仓库的 *integration-tests* 目录中。特性文件放在 *integration-tests/features* 目录中。特性目录的组织方式应与源代码组织子目录的方式相同：使用领域驱动设计，为子领域创建子目录。我们可以将上面的 *create_chart.feature* 文件放入 *integration-tests/features/chart* 目录。

首先，我们在 *integration-tests/features* 中创建一个 *environment.py* 文件，用于存储所有步骤实现共有的内容：

```
BASE_URL = 'http://localhost:8080/data-visualization-configuration-service/'
```

接下来，我们需要为场景中的每个步骤提供实现。让我们从第一个场景开始。我们将在 *src/integration-tests/features/chart/steps* 目录中创建一个 *create_chart_steps.py* 文件来实现这些步骤：

```
import requests
from behave import given, then, when
from behave.runner import Context
from environment import BASE_URL

input_chart = {}
```

```
@given('chart layout id is {layout_id:d}')
def step_impl1(context: Context, layout_id: int):
    input_chart['layout_id'] = layout_id
```

```
@given('chart type is "{type}"')
def step_impl2(context: Context, type: str):
    input_chart['type'] = type
```

```
@given('X-axis categories shown count is {x_axis_categ_shown_count:d}')
def step_impl3(context: Context, x_axis_categ_shown_count: int):
    input_chart['x_axis_categ_shown_count'] = x_axis_categ_shown_count
```

```
@given('fetched row count is {fetched_row_count:d}')
def step_impl4(context: Context, fetched_row_count: int):
    input_chart['fetched_row_count'] = fetched_row_count
```

```
@when('I create a new chart')
def step_impl5(context: Context):
    context.response = requests.post(BASE_URL + 'charts', data=input_chart)
    context.response_dict = context.response.json()
```

```
@then(
    'I should get the chart given above with status code {status_code:d} "{reason}"'
)
def step_impl6(context: Context, status_code: int, reason: str):
    assert context.response.status_code == status_code
    assert context.response.reason == reason
    output_chart = context.response_dict
    assert output_chart['id'] > 0
    assert output_chart['layout_id'] == input_chart['layout_id']
    assert output_chart['type'] == input_chart['type']
    assert output_chart['x_axis_categ_shown_count'] == (
        input_chart['x_axis_categ_shown_count']
    )
    assert output_chart['fetched_row_count'] == (
        input_chart['fetched_row_count']
    )
```

上述实现包含一个对应每个步骤的函数。每个函数都用特定 Gherkin 关键字的注解进行标注：@given、@when 和 @then。请注意，场景中的步骤可以是模板化的。例如，步骤 `Given chart layout id is 1` 是模板化的，定义在函数 `@Given("chart layout id is {layout_id:d}") def step_impl1(context: Context, layout_id: int)` 中，其中实际的布局 ID 作为参数传递给函数。你可以在不同的场景中使用这个模板化步骤，为布局 ID 提供不同的值，例如：Given chartlayout_id 为 8。layout_id 后的 `:d` 修饰符告诉 Behave，该变量应转换为整数。

`@when('I create a new chart')` 步骤实现使用 `requests` 包向数据可视化配置服务提交 HTTP POST 请求。而 `@then('I should get the chart given above with status code {status_code:d} "{reason}"')` 步骤实现则获取存储在上下文中的 HTTP POST 响应，并验证状态码和响应体中的属性。

第二个场景是一个常见的失败场景，即创建时缺少参数。由于此场景很常见（即我们可以在其他功能中使用相同的步骤），我们将步骤定义放在 `integration-tests/features/steps` 目录的 `common` 子目录中名为 `common_steps.py` 的文件中。

以下是步骤实现：

```
from behave import then
from behave.runner import Context

@then(
    'I should get a response with status code {status_code:d} "{reason}"'
)
def step_impl1(context: Context, status_code: int, reason: str):
    assert context.response.status_code == status_code
    assert context.response.reason == reason

@then(
    'response body should contain error object with {error} entry for following fields'
)
def step_impl2(context: Context, error: str):
    error_description = context.response_dict.error_description
    for field in context.table:
        assert f'{field} {error}' in error_description
```

要使用 Behave 执行集成测试，请在 `integration-tests` 目录中运行 `behave` 命令。这里介绍一下 `behave` 命令行参数以及如何使用标签仅测试特定测试。

一些框架提供了自己的创建集成测试的方式。例如，Django Web 框架提供了自己的集成测试方法。我不推荐使用特定于框架的测试工具有两个原因。第一个原因是，这样你的集成测试就与框架耦合了，如果你决定使用不同的语言或不同的框架重新实现你的微服务，你也需要重新实现集成测试。当你使用像 Behave 这样的通用 BDD 集成测试工具时，你的集成测试不会与任何微服务实现编程语言或框架耦合。第二个原因是，当 QA/测试工程师不必掌握多个特定于框架的集成测试工具时，他们的学习和信息负担会更少。如果你在软件系统的所有微服务中都使用像 Behave 这样的单一 BDD 集成测试工具，QA/测试工程师将更容易处理不同的微服务。

对于 API 微服务，实现集成测试的另一个替代方案是使用像 Postman¹ 这样的 API 开发平台。Postman 可以用来使用 JavaScript 编写集成测试。
假设我们有一个名为 _sales-itemsservice 的 API 微服务，它提供对销售项目的 CRUD 操作。下面是一个创建新销售项目的示例 API 请求。你可以在 Postman 中将其定义为一个新请求：

```
POST http://localhost:3000/sales-item-service/sales-items HTTP/1.1
Content-Type: application/json

{
    "name": "Test sales item",
    "price": 10,
}
```

这是一个用于验证上述请求响应的 Postman 测试用例：

```
pm.test("Status code is 201 Created", function () {
    pm.response.to.have.status(201);
});

const salesItem = pm.response.json();
pm.collectionVariables.set("salesItemId", salesItem.id)

pm.test("Sales item name", function () {
    return pm.expect(salesItem.name).to.eql("Test sales item");
})

pm.test("Sales item price", function () {
    return pm.expect(salesItem.price).to.eql(10);
})
```

在上面的测试用例中，首先验证响应状态码，然后从响应体中解析出 `salesItem` 对象。设置了变量 `salesItemId` 的值。此变量将在后续测试用例中使用。最后，检查 `name` 和 `price` 属性的值。
接下来，可以在 Postman 中创建一个新的 API 请求来检索刚刚创建的销售项目：

```
GET http://localhost:3000/sales-item-service/sales-items/{{salesItemId}} HTTP/1.1
```

我们在请求 URL 中使用了存储在 `salesItemId` 变量中的值。变量可以在 URL 和请求体中使用以下表示法：`{{<variable-name>}}`。让我们为上述请求创建一个测试用例：

¹https://www.postman.com/

```
pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});

const salesItem = pm.response.json();

pm.test("Sales item name", function () {
    return pm.expect(salesItem.name).to.eql("Test sales item");
})

pm.test("Sales item price", function () {
    return pm.expect(salesItem.price).to.eql(10);
})
```

在 Postman 中编写的 API 集成测试可以在 CI 流水线中使用。一个简单的方法是将 Postman 集合导出到一个包含所有 API 请求和相关测试的文件中。Postman 集合文件是一个 JSON 文件。Postman 提供了一个名为 Newman² 的 Node.js 命令行实用程序。它可以用来在导出的 Postman 集合文件中运行 API 请求和相关测试。

你可以在 CI 流水线中使用以下命令在导出的 Postman 集合文件中运行集成测试：

```
newman run integration-tests/integrationTestsPostmanCollection.json
```

在上面的例子中，我们假设一个名为 *integrationTestsPostmanCollection.json* 的文件已导出到源代码仓库的 *integration-tests* 目录中。

## 6.1.2.1：Web UI 集成测试

在指定 UI 功能时，你也可以使用 Gherkin 语言。例如，*TestCafe* UI 测试工具可以与 *gherkin-testcafe* 工具一起使用，使 TestCafe 支持 Gherkin 语法。让我们创建一个简单的 UI 功能：

```
Feature: Greet user
  Entering user name and clicking submit button
  displays a greeting for the user

  Scenario: Greet user successfully
    Given there is "John Doe" entered in the input field
    When I press the submit button
    Then I am greeted with text "Hello, John Doe"
```

接下来，我们可以使用 TestCafe 测试 API 在 JavaScript 中实现上述步骤：

```
// Imports...

// 'Before' hook runs before the first step of each scenario.
// 't' is the TestCafe test controller object
Before('Navigate to application URL', async (t) => {
  // Navigate browser to application URL
  await t.navigateTo('...');
});

Given('there is {string} entered in the input field',
      async (t, [userName]) => {
  // Finds an HTML element with CSS id selector and
  // enters text to it
  await t.typeText('#user-name', userName);
});

When('I press the submit button', async (t) => {
  // Finds an HTML element with CSS id selector and clicks it
  await t.click('#submit-button');
});

When('I am greeted with text {string}', async (t, [greeting]) => {
  // Finds an HTML element with CSS id selector
  // and compares its inner text
  await t.expect(Selector('#greeting').innerText).eql(greeting);
});
```

还有另一个与 TestCafe 类似的工具，即 *Cypress*。你也可以通过 `cypress-cucumber-preprocessor` 包在 Cypress 中使用 Gherkin。然后你可以这样编写你的 UI 集成测试：

```
Feature: Visit duckduckgo.com website

  Scenario: Visit duckduckgo.com website successfully
    When I visit duckduckgo.com
    Then I should see the search bar
```

```
import { When, Then } from
  '@badeball/cypress-cucumber-preprocessor';

When("I visit duckduckgo.com", () => {
  cy.visit("https://www.duckduckgo.com");
});

Then("I should see the search bar", () => {
  cy.get("input").should(
    "have.attr",
    "placeholder",
    "Search the web without being tracked"
  );
});
```

## 6.1.2.2：设置集成测试环境

在运行集成测试之前，必须设置集成测试环境。集成测试环境是被测试的微服务及其所有依赖项运行的地方。为容器化微服务设置集成测试环境最简单的方法是使用 *Docker Compose*，这是一个用于单主机的简单容器编排工具。

让我们为 *sales-item-service* 微服务创建一个 *docker-compose.yml* 文件，该微服务有一个 MySQL 数据库作为依赖项。该数据库被微服务用来存储销售项目。

图 6.5. docker-compose.yaml

```
version: "3.8"

services:
  wait-for-services-ready:
    image: dokku/wait
  sales-item-service:
    restart: always
    build:
      context: .
    env_file: .env.ci
    ports:
      - "3000:3000"
    depends_on:
      - mysql
  mysql:
    image: mysql:8.0.22
    command: --default-authentication-plugin=mysql_native_password
    restart: always
    cap_add:
      - SYS_NICE
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_PASSWORD}
    ports:
      - "3306:3306"
```

在上面的例子中，我们首先定义了一个服务 *wait-for-services-ready*，我们稍后会用到它。接下来，我们定义了我们的微服务 *sales-item-service*。我们要求 Docker Compose 使用当前目录中的 *Dockerfile* 为 *sales-item-service* 构建一个容器镜像。然后我们定义微服务的环境从一个 *.env.ci* 文件中读取。我们暴露端口 3000 并说明我们的微服务依赖于 *mysql* 服务。

接下来，我们定义了 *mysql* 服务。我们指定了要使用的镜像，给出了一个命令行参数，并定义了环境和暴露了一个端口。

在运行集成测试之前，我们必须使用 `docker-compose up` 命令启动集成测试环境：

```
docker-compose up --env-file .env.ci --build -d
```

我们告诉 `docker-compose` 命令从一个 *.env.ci* 文件中读取环境变量，该文件应包含一个名为 `MYSQL_PASSWORD` 的环境变量。我们要求 `docker-compose` 始终构建

## 6.1.3：端到端（E2E）测试原则

端到端（E2E）测试应测试完整的软件系统（即微服务的集成），使每个测试用例都是端到端的（从软件系统的南向接口到软件系统的北向接口）。

顾名思义，在E2E测试中，测试用例应该是端到端的。它们应测试每个微服务是否正确部署到测试环境，并与其依赖的服务连接。E2E测试用例的目的不是测试微服务功能的细节，因为这些细节已经在单元测试和软件组件集成测试中测试过了。

让我们考虑一个由以下应用程序组成的电信网络分析软件系统：

- 数据摄入
- 数据关联
- 数据聚合
- 数据导出器
- 数据可视化

### 北向接口

![](img/cbd069395d7b824346b69b1f92e0fb4a_382_0.png)

图6.7. 电信网络分析软件系统

该软件系统的南向接口是数据摄入应用程序。数据可视化应用程序提供一个Web客户端作为北向接口。此外，数据导出器应用程序为软件系统提供了另一个北向接口。

E2E测试的设计和实现方式与软件组件集成测试类似。我们只是集成不同的东西（微服务而不是函数）。E2E测试应从E2E功能的规范开始。这些功能可以使用Gherkin语言指定，并放入`.feature`文件中。

你可以在软件系统的架构设计完成后立即开始指定和实现E2E测试。这样，你可以将E2E测试的实现左移，并加速开发阶段。你不应该等到整个软件系统实现后才开始指定和实现E2E测试。

我们的示例软件系统应至少有两个正常路径的E2E功能。一个用于测试从数据摄入到数据可视化的数据流，另一个功能用于测试从数据摄入到数据导出的数据流。以下是第一个E2E功能的规范：

```
Feature: Visualize ingested, correlated and
         aggregated data in web UI's dashboard's charts

Scenario: Data ingested, correlated and aggregated is visualized
          successfully in web UI's dashboard's charts

Given southbound interface simulator is configured
      to send input messages that contain data...
And data ingester is configured to read the input messages
    from the southbound interface
And data correlator is configured to correlate
    the input messages
And data aggregator is configured to calculate
    the following counters...
And data visualization is configured with a dashboard containing
    the following charts viewing the following counters/KPIs...

When southbound interface simulator sends the input messages
And data aggregation period is waited
And data content of each data visualization web UI's dashboard's
    chart is exported to a CSV file

Then the CSV export file of the first chart should
     contain following values...
And the CSV export file of the second chart should
     contain following values...
.
.
.
And the CSV export file of the last chart should
     contain following values...
```

然后，我们可以创建另一个功能来测试从数据摄入到数据导出的E2E路径：

```
Feature: Export ingested, correlated and transformed data
         to Apache Pulsar

Scenario: Data ingested, correlated and transformed is
          successfully exported to Apache Pulsar
Given southbound interface simulator is configured to send
      input messages that contain data...
And data ingester is configured to read the input messages
    from the southbound interface
And data correlator is configured to correlate
    the input messages
And data exporter is configured to export messages with
    the following transformations to Apache Pulsar...

When southbound interface simulator sends the input messages
And messages from Apache Pulsar are consumed

Then first message from Apache Pulsar should have
     the following fields with following values...
And second message from Apache Pulsar should have
    the following fields with following values...
.
.
.
And last message from Apache Pulsar should have
    the following fields with following values...
```

接下来，可以实现E2E测试。可以使用任何与Gherkin语法兼容的编程语言和工具，例如Python的Behave。如果开发团队中的QA/测试工程师已经使用Behave进行集成测试，那么自然也可以使用Behave进行E2E测试。

我们想要进行E2E测试的软件系统必须位于一个类似生产的测试环境中。通常，E2E测试在CI和预发布环境中进行。在运行E2E测试之前，需要将软件部署到测试环境。

如果我们考虑上面的第一个功能，实现E2E测试步骤可以这样进行：场景中Given部分的步骤使用外部化配置来实现。如果我们的软件系统运行在Kubernetes集群中，我们可以通过创建所需的ConfigMap来配置微服务。南向接口模拟器可以通过启动一个Kubernetes Job来控制，或者如果南向接口模拟器是一个带有API的微服务，则通过其API来控制。在等待所有摄入的数据被聚合和可视化之后，E2E测试可以启动一个适合Web UI测试的工具（如TestCafe）将图表数据从Web UI导出到下载的文件中。然后，E2E测试将这些文件的内容与预期值进行比较。

你可以在每次提交到主分支后（即微服务CI/CD流水线运行完成后）在CI环境中运行E2E测试，以测试新的提交是否破坏了任何E2E测试。或者，如果E2E测试很复杂且执行时间很长，你可以按计划（例如每小时一次）在CI环境中运行E2E测试。

你可以在预发布环境中使用CI/CD工具中的单独流水线运行E2E测试。

## 6.2：非功能测试原则

除了多层次的功能测试外，还应尽可能自动化地对软件系统进行非功能测试。

非功能测试最重要的类别如下：

- 性能测试
- 数据量测试
- 稳定性测试
- 可靠性测试
- 压力和可扩展性测试
- 安全性测试

## 6.2.1：性能测试

性能测试的目标是验证软件系统的性能。这种验证可以在不同层面、以不同方式进行，例如，单独验证每个性能关键型微服务。

要测量微服务的性能，可以创建性能测试来对微服务中的一个或多个繁忙循环进行基准测试。以数据导出器微服务为例，它有一个执行消息解码、转换和编码的繁忙循环。我们可以使用单元测试框架为这个繁忙循环创建一个性能测试。该性能测试应执行繁忙循环中的代码若干轮次，并验证执行时间不超过首次运行性能测试时获得的指定阈值。性能测试旨在验证性能是否保持在原有水平。如果性能恶化，测试将不会通过。这样，你就不会在不知情的情况下意外引入对性能产生负面影响的更改。同样的性能测试也可用于衡量优化效果。首先，你编写未经优化的繁忙循环代码，测量性能，并将该测量值作为参考点。之后，你开始逐一引入优化，并观察它们是否以及如何影响性能。

性能测试的执行时间阈值必须为每台开发者的计算机单独指定。这可以通过为运行测试的每台计算机主机名设置不同的阈值来实现。

你也可以在 CI/CD 流水线中运行性能测试，但必须首先在该流水线中测量性能并相应设置阈值。此外，运行 CI/CD 流水线的计算实例必须是同构的。否则，你将在不同的 CI/CD 流水线运行中得到不同的结果。

上述性能测试是针对单元（一个公共函数）的，但性能测试也可以在软件组件层面进行。如果软件组件具有需要测量性能的外部依赖项，这将非常有用。在电信网络分析软件系统中，我们可以为 *data-ingester-service* 引入一个性能测试，以测量处理一定数量消息（例如一百万条）所需的时间。执行该测试后，我们就有了一个可供参考的性能测量值。当我们尝试优化微服务时，可以测量优化后微服务的性能，并将其与参考值进行比较。如果我们进行了一项已知会降低性能的更改，我们有一个参考值可以与之比较，以查看性能下降是否可接受。当然，这个参考值将防止开发者意外做出对微服务性能产生负面影响的更改。

我们还可以测量端到端性能。例如，在电信网络分析软件系统中，我们可以测量从数据摄入到数据导出的性能。

## 6.2.2：数据量测试

数据量测试的目标是测量数据库在空状态与存储了大量数据时的性能对比。通过数据量测试，我们可以衡量数据量对软件组件性能的影响。通常，空数据库的性能优于包含大量数据的数据库。当然，这取决于数据库本身及其在大数据量下的扩展能力。

## 6.2.3：稳定性测试

稳定性测试旨在验证软件系统在负载下长时间运行时是否保持稳定。此测试也称为负载测试、耐力测试或浸泡测试。“长时间”一词根据软件系统不同可能有不同的解释。但这个时间段应该是许多小时，最好是几天，甚至长达一周。稳定性测试旨在发现诸如偶发性错误或内存泄漏等问题。偶发性错误是指仅在特定条件下或不规则间隔发生的错误。内存泄漏可能非常小，以至于需要软件组件运行数十小时后才变得明显。建议在长时间运行软件系统时，施加给软件系统的负载应遵循自然模式（模拟生产负载），即负载存在高峰和低谷。

稳定性测试可以部分自动化。可以使用为此目的创建的工具（例如 Apache JMeter）来生成系统负载。每个软件组件都可以测量崩溃次数，这些统计数据可以在稳定性测试完成后自动或手动分析。分析内存泄漏可能更棘手，但应记录因内存不足导致的崩溃，以及软件组件因内存不足而进行横向扩展的情况。

## 6.2.4：可靠性测试

可靠性测试旨在验证软件系统是否可靠运行。当软件系统能够抵御故障并尽可能快地从故障中自动恢复时，它就是可靠的。可靠性测试也称为可用性、恢复或弹性测试。

可靠性测试涉及混沌工程，以在软件系统的环境中诱发各种故障。它还应确保软件系统保持可用，并能从故障中自动恢复。

假设你有一个部署到 Kubernetes 集群的软件系统。你可以通过配置无状态服务运行多个 Pod 来使其高度可用。如果一个节点宕机，它将终止其中一个 Pod，但由于至少还有一个 Pod 在另一个节点上运行，该服务仍然可用且可操作。此外，过一小会儿，当 Kubernetes 注意到缺少一个 Pod 时，它将在新节点上创建一个新的 Pod，这样运行的 Pod 数量将恢复到原始数量，节点宕机的恢复就成功了。

可靠性测试的许多部分可以自动化。你可以使用现成的混沌工程工具或创建自己的工具。使用工具在环境中诱发故障。然后验证服务是否保持高度可用或至少能从故障中快速恢复。
考虑到电信网络分析软件系统，我们可以引入一个测试用例，其中消息代理（例如 Kafka）被关闭。然后我们期望在一段时间后，尝试使用不可用消息代理的微服务会触发警报。在消息代理启动后，警报应自动取消，微服务应继续正常运行。

## 6.2.5：压力与可扩展性测试

压力测试旨在验证软件系统在高负载下运行。在压力测试中，软件系统被暴露于高于其通常负载的负载下。软件系统应设计为可扩展的，这意味着软件系统也应在高负载下运行。因此，压力测试应测试软件系统的可扩展性，并观察微服务在需要时是否进行横向扩展。在压力测试结束时，负载恢复正常水平，同时也可以验证微服务的横向缩减。
你可以为 Kubernetes Deployment 指定一个 HorizontalPodAutoscaler (HPA)。在 HPA 清单中，你必须指定最小副本数。如果你希望微服务高度可用，这至少应为两个。你还需要指定最大副本数，以便在某些异常故障情况下，你的微服务不会消耗过多的计算资源。你可以通过指定 CPU 和内存的目标利用率来实现横向扩展（缩减和扩展）。下面是一个定义 Kubernetes HPA 的 Helm chart 模板示例：

```
{{- if eq .Values.env "production" }}
apiVersion: autoscaling/v2beta1
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "microservice.fullname" . }}
  labels:
    {{- include "microservice.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "microservice.fullname" . }}
  minReplicas: {{ .Values.hpa.minReplicas }}
  maxReplicas: {{ .Values.hpa.maxReplicas }}
  metrics:
    {{- if .Values.hpa.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        targetAverageUtilization: {{ .Values.hpa.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.hpa.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        targetAverageUtilization: {{ .Values.hpa.targetMemoryUtilizationPercentage }}
    {{- end }}
```

也可以指定自动扩缩容使用外部指标。例如，外部指标可以是Kafka消费者滞后。如果Kafka消费者滞后增长过高，HPA可以扩展微服务，为Kafka消费者组提供更多处理能力；当Kafka消费者滞后降至定义的阈值以下时，HPA可以缩减微服务，以减少Pod数量。

## 6.2.6：安全测试

安全测试的目标是验证软件系统是安全的，并且不包含安全漏洞。安全测试的一部分是执行软件制品的漏洞扫描。通常，这意味着使用自动漏洞扫描工具扫描微服务容器。安全测试的另一个重要部分是渗透测试，它模拟恶意方的攻击。渗透测试可以使用自动化工具（如OWASP ZAP或Burp Suite）进行。

渗透测试工具试图查找以下类别的安全漏洞：

- 跨站脚本攻击
- SQL注入
- 路径泄露
- 拒绝服务
- 代码执行
- 内存损坏
- 跨站请求伪造（CSRF）
- 信息泄露
- 本地/远程文件包含

OWASP ZAP工具发现的可能安全漏洞的完整列表可在 https://www.zaproxy.org/docs/alerts/ 找到。

## 6.2.7：其他非功能测试

其他非功能测试包括文档测试和多个与UI相关的非功能测试，包括可访问性（A11Y）测试、视觉测试、可用性测试以及本地化和国际化（I18N）测试。

### 6.2.7.1：视觉测试

我想在这里提一下视觉测试，因为它很重要。*Backstop.js* 和 *cypress-plugin-snapshots* 使用快照测试来测试Web UI的HTML和CSS。快照是Web UI的截图。通过比较快照，可以确保应用程序的视觉外观保持不变，并且HTML或CSS更改没有引入错误。

## 7：安全原则

本章描述了与安全相关的原则，并阐述了与软件开发者相关的主要安全特性。

## 7.1：安全左移原则

> *将安全实现左移。尽早实现安全相关功能，而不是推迟。*

安全是生产级软件不可或缺的一部分，就像源代码本身和所有测试一样。假设安全相关功能仅在项目的非常后期阶段才实现。在这种情况下，可能没有时间实现它们，或者忘记实现它们的可能性更大。因此，安全相关功能应该尽早实现，而不是最后才做。下一节描述的威胁建模过程应用于识别潜在威胁，并提供需要作为威胁对策实施的安全功能列表。

## 7.2：设立产品安全负责人原则

> *每个产品团队都应任命一名安全负责人。安全负责人的角色是确保产品是安全的。*

安全负责人与开发团队紧密合作。他/她向团队传授安全相关流程和安全功能。安全负责人协助团队进行下文描述的威胁建模过程，但遵循该过程是团队的责任，安全功能的实际实施也是团队的责任。

## 7.3：使用威胁建模过程原则

威胁建模过程使你能够识别、量化和解决与软件组件相关的安全风险。威胁建模过程由三个高级步骤组成：

- 分解应用程序
- 确定和排列威胁
- 确定对策和缓解措施

### 7.3.1：分解应用程序

应用程序分解步骤是为了了解应用程序由哪些部分组成、外部依赖关系以及它们如何被使用。此步骤可以在应用程序架构设计完成后执行。此步骤的结果是：

- 识别攻击者进入应用程序的入口点
- 识别受威胁的资产。这些资产是攻击者感兴趣的东西
- 识别信任级别，例如，具有不同用户角色的用户可以做什么

### 7.3.2：确定和排列威胁

要确定可能的威胁，应使用威胁分类方法。*STRIDE* 方法将威胁分为以下类别：

| 类别 | 描述 |
| :--- | :--- |
| 欺骗 | 攻击者在没有真正认证的情况下冒充其他用户，或使用窃取的凭据 |
| 篡改 | 攻击者恶意更改数据 |
| 抵赖 | 攻击者能够执行被禁止的操作 |
| 信息泄露 | 攻击者获得对敏感数据的访问权限 |
| 拒绝服务 | 攻击者试图使服务不可用 |
| 权限提升 | 攻击者获得不需要的访问权限 |

#### 7.3.2.1：STRIDE威胁示例

- 欺骗
  - 当缺少适当的授权时，攻击者能够使用其他用户的ID读取其他用户的数据
  - 由于使用了不安全的协议（如HTTP而不是HTTPS），攻击者能够在网络上窃取用户凭据
  - 攻击者创建一个假的网站登录页面来窃取用户凭据
  - 攻击者能够拦截网络流量，并重放某些用户的请求（原样或修改后）
- 篡改
  - 攻击者通过SQL注入访问数据库并能够更改现有数据
  - 当缺少适当的授权时，攻击者能够使用其他用户的ID修改其他用户的数据
- 抵赖
  - 当缺少审计日志时，攻击者能够在不被注意的情况下执行恶意操作
- 信息泄露
  - 敏感信息在请求响应中被意外发送（如错误堆栈跟踪或业务关键数据）
  - 敏感信息未正确加密
  - 在没有适当授权（例如基于角色）的情况下可以访问敏感信息
- 拒绝服务
  - 当缺少适当的请求速率限制时，攻击者可以创建无限数量的请求
  - 当数据大小完全没有限制时，攻击者可以发送包含大量数据的请求
  - 攻击者可以通过发送可能导致正则表达式评估消耗大量CPU时间的字符串，尝试进行正则表达式DoS攻击
  - 如果没有适当的输入验证，攻击者可以在请求中发送无效值，以尝试使服务崩溃或导致无限循环
- 权限提升
  - 没有用户账户的攻击者由于缺少身份验证/授权而能够访问服务
  - 由于服务不检查用户是否具有适当的角色，攻击者能够以管理员身份行事
  - 由于进程以root用户权限运行，攻击者能够获得操作系统root权限。

*应用程序安全框架*（ASF）将应用程序安全功能分为以下类别：

| 类别 | 描述 |
| --- | --- |
| 审计与日志记录 | 记录用户操作以检测，例如，抵赖攻击 |
| 身份验证 | 防止身份欺骗攻击 |
| 授权 | 防止权限提升攻击 |
| 配置管理 | 正确存储机密信息，并以最小权限配置系统 |
| 传输和静态数据保护 | 使用安全协议（如TLS），加密数据库中的敏感信息（如PII） |
| 数据验证 | 验证来自用户的数据，以防止，例如，注入和ReDoS攻击 |
| 异常管理 | 不要在错误消息中向最终用户透露实现细节 |

当使用上述任一威胁分类方法时，应根据关于分解应用程序的信息列出每个类别中的威胁：应用程序的入口点和需要保护的资产是什么？列出每个类别中的潜在威胁后，应对威胁进行排列。有几种排列威胁的方法。最简单的排列威胁的方法是根据风险将它们分为三类：高、中、低。作为排列的基础，你可以使用关于威胁概率及其产生的不利影响（影响）大小的信息。排列的目的是优先考虑安全功能。高风险威胁的安全功能应首先实现。

### 7.3.3：确定对策和缓解措施

确定对策步骤应提供所需安全功能的用户故事列表。这些安全功能应消除或至少缓解威胁。如果你有一个无法消除或缓解的威胁，并且该威胁被归类为低风险威胁，你可以接受该风险。低风险威胁是对应用程序影响较小且威胁实现概率较低的威胁。假设你在应用程序中发现了一个风险非常高的威胁，并且你无法消除或缓解该威胁。在这种情况下，你应该通过完全从应用程序中移除与威胁相关的功能来消除该威胁。

### 7.3.4：使用STRIDE的威胁建模示例

让我们举一个实际威胁建模的简单例子。我们将对一个名为 *order-service* 的REST API微服务进行威胁建模。该微服务用于处理电子商务软件系统中的订单（对订单实体的CRUD操作）。订单持久化在数据库中。该微服务与其他微服务通信。威胁建模过程的第一步是分解应用程序。

## 7.3.4.1：分解应用程序

在本阶段，我们将分解 *order-service*，以了解其组成部分及其依赖关系。

基于上述对 *order-service* 的分解视图，我们接下来需要识别以下内容：

-   识别攻击者进入应用程序的入口点
-   识别受威胁的资产。这些资产是攻击者感兴趣的东西
-   识别信任级别，例如，具有不同用户角色的用户可以做什么

如上图所示，攻击者的入口点来自互联网（*order-service* 通过 API 网关暴露在公共互联网上），并且内部攻击者也可能能够嗅探服务之间的网络流量。

受威胁的资产包括 API 网关、*order-service*、其数据库以及未加密的网络流量。*order-service* 具有以下信任级别：

-   用户可以为自己下单（不能为其他用户下单）
-   用户可以查看自己的订单（不能查看其他用户的订单）
-   用户只能在订单打包发货前更新自己的订单
-   管理员可以创建/读取/更新/删除任何订单

## 7.3.4.2：确定并排列威胁

接下来，我们应该列出 STRIDE 方法每个类别中可能的威胁。我们还为每个可能的威胁定义了风险。

1.  欺骗
    1.  攻击者试图为他人创建订单（风险：高）
    2.  攻击者试图读取/更新他人的订单（风险：高）
    3.  攻击者使用窃取的凭证冒充他人（风险：中）
2.  篡改
    1.  攻击者试图使用 SQL 注入篡改数据库（风险：高）
    2.  攻击者能够捕获并修改未加密的互联网流量（风险：高）
    3.  攻击者能够捕获并修改未加密的内部网络流量（风险：低）
3.  抵赖
    1.  攻击者能够进行恶意操作而不被发现（风险：高）
4.  信息泄露
    1.  攻击者能够访问敏感信息，因为这些信息未被正确加密（风险：中）
    2.  攻击者在请求响应中接收到敏感信息，如详细的堆栈跟踪。（风险：中）攻击者可以利用这些信息并利用实现中可能存在的安全漏洞
    3.  信息泄露给攻击者，因为互联网流量是明文的，即未加密（风险：高）
    4.  信息泄露给攻击者，因为内部网络流量是明文的，即未加密（风险：低）
5.  拒绝服务
    1.  攻击者试图发送过多请求（风险：高）
    2.  攻击者试图在数据大小完全不受限制的情况下发送包含大量数据的请求（风险：高）
    3.  攻击者试图通过发送可能导致正则表达式评估消耗大量 CPU 时间的字符串来进行正则表达式 DoS (ReDoS) 攻击（风险：高）
    4.  攻击者试图在请求中发送无效值，以尝试使服务崩溃或在没有适当输入验证的情况下导致无限循环（风险：高）
6.  权限提升
    1.  没有用户账户的攻击者可以访问服务，因为缺少身份验证/授权（风险：高）
    2.  攻击者能够以管理员身份行事，因为服务未检查用户是否具有适当的角色（风险：高）
    3.  攻击者能够获得操作系统 root 权限，因为进程以 root 用户权限运行（风险：中）

## 7.3.5：确定对策和缓解措施

接下来，我们将为每个威胁定义对策用户故事。对策所针对的威胁列在对策之后。

1.  仅允许拥有特定资源的用户访问该资源 (1.1, 1.2)
2.  为创建/修改/删除订单的操作实施审计日志 (1.3, 3.1)
3.  对 SQL 使用参数化语句或使用 ORM，并为数据库用户配置最小权限 (2.1) 普通数据库用户应该只能执行仅与管理员相关的操作，如删除、创建/删除表等。
4.  仅允许安全的互联网流量访问 API 网关（TLS 在 API 网关终止）(1.3, 2.2)
5.  使用像 Istio 这样的服务网格在服务之间实施 mTLS (2.3, 4.4)
6.  在数据库中加密所有敏感信息，如个人身份信息 (PII) 和关键业务数据 (4.1)
7.  当微服务在生产环境中运行时，不要返回错误堆栈跟踪 (4.2)
8.  实施请求速率限制，例如在 API 网关中 (5.1.)
9.  验证微服务的输入数据，并定义允许的最大字符串、数组和请求长度 (5.2) 此外，考虑记录输入验证失败的审计日志
10. 不要在验证中使用正则表达式，或使用不会导致 ReDoS 的正则表达式 (5.3.)
11. 验证微服务的输入数据，例如正确的类型、数值的最小/最大值、允许的值列表 (5.4) 此外，考虑记录输入验证失败的审计日志
12. 使用 JWT 实施用户身份验证和授权 (1.1, 1.2, 6.1) 考虑记录身份验证/授权失败的审计日志以检测可能的攻击
13. 对于仅限管理员的操作，在允许操作之前，验证 JWT 是否包含管理员角色 (1.1, 1.2, 6.2) 此外，配置系统，使管理员操作无法从互联网访问，除非绝对必要。
14. 对于容器化的微服务，定义以下内容：
    -   容器不应是特权容器
    -   丢弃所有 capabilities
    -   容器文件系统是只读的
    -   仅允许非 root 用户在容器内运行
    -   定义容器应在其中运行的非 root 用户和组
    -   禁止权限提升
    -   使用 distroless 或尽可能小的容器基础镜像

接下来，我们应该根据相关威胁风险级别对上述用户故事进行优先级排序。让我们使用以下威胁风险级别值为每个用户故事计算优先级指数 (PI)：

-   高 = 3
-   中 = 2
-   低 = 1

以下是按优先级指数 (PI) 从高到低排序的用户故事：

1.  使用 JWT 实施用户身份验证和授权 (PI: 9)
2.  对于仅限管理员的操作，在允许操作之前，验证 JWT 是否包含管理员角色 (PI: 9) 此外，配置系统，使管理员操作无法从互联网访问，除非绝对必要
3.  仅允许安全的互联网流量访问 API 网关（TLS 在 API 网关终止）(PI: 6)
4.  仅允许拥有特定资源的用户访问该资源 (PI: 6)
5.  为创建/修改/删除订单的操作实施审计日志 (PI: 5)
6.  实施请求速率限制，例如在 API 网关中 (PI: 3)
7.  验证微服务的输入数据，并定义允许的最大字符串和数组长度 (PI: 3)
8.  对 SQL 使用参数化语句或使用 ORM，并为数据库用户配置最小权限 (PI: 3) 普通数据库用户应该只能执行仅与管理员相关的操作，如删除、创建/删除表等。
9.  不要在验证中使用正则表达式，或使用不会导致 ReDoS 的正则表达式 (PI: 3)
10. 验证微服务的输入数据，例如数值的最小/最大值、允许的值列表 (PI: 3)
11. 在数据库中加密所有敏感信息，如个人身份信息 (PII) 和关键业务数据 (PI: 2)
12. 使用像 Istio 这样的服务网格在服务之间实施 mTLS (PI: 2)
13. 当微服务在生产环境中运行时，不要返回错误堆栈跟踪 (PI: 2)
14. 对于容器化的微服务，定义以下内容：... (PI: 2)

团队应与产品安全负责人一起审查优先级排序的安全用户故事列表。由于安全是软件系统的组成部分，至少所有 PI 大于 2 的用户故事都应在交付第一个生产版本之前实现。PI <= 2 的用户故事可以在初始交付后的第一个功能包中立即交付。这只是一个例子。一切都取决于所需和/或要求的安全级别，相关利益相关者应参与决定安全级别。

在上面的例子中，我们没有列出与缺少 API 安全相关 HTTP 响应头相关的威胁。这是因为对于任何 REST API，它们都是相同的。这些 API 安全相关的 HTTP 响应头将在本章后面的章节中讨论。这些头的发送应整合到 API 网关中，这样所有 API 微服务就不必自己实现它们。

## 7.3.6：使用 ASF 进行威胁建模示例

使用 ASF 进行威胁建模与使用 STRIDE 方法的前一个示例相同。唯一的区别是威胁的分类方式不同。我们应该能够找到所有相同的威胁，并使用 ASF 将它们放入类别中。那么，让我们尝试将之前发现的威胁放入 ASF 类别：

-   审计与日志
    -   攻击者能够进行恶意操作而不被发现（风险：高）
    -   攻击者使用窃取的凭证冒充他人（风险：中）
-   身份验证
    -   没有用户账户的攻击者可以访问服务，因为缺少身份验证/授权（风险：高）
-   授权

## 7.4：安全特性

本节重点关注与软件开发人员相关的安全特性。它列出了典型软件系统中需要实现的最常见安全特性，并提供了一些关于如何实现这些安全特性的指导。例如，当您实现加密时，应使用安全的算法和安全的加密密钥。

### 7.4.1：身份验证与授权

在为应用程序实现用户身份验证和授权时，请使用第三方授权服务。不要尝试自己构建授权服务，因为很容易出错。此外，如果您的应用程序处理明文用户凭证，这可能构成安全风险。最好使用经过实战检验的解决方案，该方案已修正了最重要的缺陷，并能安全地存储用户凭证。在接下来的示例中，我们将使用 *Keycloak* 作为授权服务。

同时，尽可能使用成熟的第三方库，而不是自己编写所有授权相关代码。创建一个单一的前端身份验证/授权库并在多个项目中使用同一个库，而不是在不同项目中不断从头实现身份验证和授权相关功能，这也是有帮助的。类似地，如果跨域调用不被支持/预期，应在 API 网关处禁用 CORS 相关的 HTTP 响应头。

### 7.4.1.1：前端中的 OpenID Connect 身份验证与授权

关于前端授权，必须注意安全存储授权相关的机密信息，如 *code verifier* 和 *tokens*。这些必须存储在浏览器中的安全位置。以下是一些不安全的存储机制列表：

- Cookies
    - 自动发送，存在 CSRF 威胁
- Session/Local Storage
    - 容易被恶意代码窃取（XSS 威胁）
- 加密的 session/local storage
    - 容易被恶意代码窃取，因为加密密钥是明文存储的
- 全局变量
    - 容易被恶意代码窃取（XSS 威胁）

将机密信息存储在闭包变量中本身并非不安全，但机密信息会在页面刷新或新页面时丢失。

下面是一个使用服务工作者作为机密信息的安全存储的示例。服务工作者的额外好处是它不允许恶意第三方代码修改服务工作者的 fetch 方法，从而防止其窃取访问令牌等行为。

恶意代码很容易更改全局 fetch 函数：

```
fetch = () => console.log('hacked');
fetch() // prints 'hacked' to console
```

下面是一个更现实的例子：

```
originalFetch = fetch;
fetch = (url, options) => {
  // Implement malicious attack here
  // For example: change some data in the request body

  // Then call original fetch implementation
  return originalFetch(url, options);
}
```

当然，有人可能会问：为什么可以这样修改全局对象上的内置方法？当然，这本不应该可能，但不幸的是，它确实可以。

让我们创建一个使用 OpenID Connect 协议（OAuth2 协议的扩展）执行身份验证和授权的 Vue.js 应用程序。

在下面的主模块中，我们设置全局 fetch 始终返回错误，并只允许我们的 tryMakeHttpRequest 函数使用原始的全局 fetch 方法。然后我们注册一个服务工作者。如果服务工作者已经注册，则不会再次注册。最后，我们创建应用程序（App 组件），激活路由器，激活用于状态管理的 Pinia 中间件，并将应用程序挂载到 DOM 节点：

图 7.2. main.ts

```
import { setupFetch } from "@/tryMakeHttpRequest";
setupFetch();
import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "@/App.vue";
import router from "@/router";

if ("serviceWorker" in navigator) {
  await navigator.serviceWorker.register("/serviceWorker.js");
}

const app = createApp(App);
const pinia = createPinia();
app.use(pinia);
app.use(router);
app.mount("#app");
```

下面是 App 组件的定义。挂载后，它将检查用户是否已授权。

如果用户已授权，其授权信息将从服务工作者获取，并且用户的名将更新到授权信息存储中。用户将被转发到主页。

如果用户未授权，则将执行授权。

```
Figure 7.3. App.vue
<template>
  <HeaderView />
  <router-view></router-view>
</template>

<script setup>
import { onMounted } from "vue";
import { useRouter } from "vue-router";
import authorizationService from "@/authService";
import { useAuthInfoStore } from "@/stores/authInfoStore";
import HeaderView from "@/HeaderView.vue";
import tryMakeHttpRequest from "@/tryMakeHttpRequest";

const router = useRouter();
const route = useRoute();

onMounted(async () => {
  const response = await tryMakeHttpRequest("/authorizedUserInfo");
  const responseBody = await response.text();
  if (responseBody !== "") {
    const authorizedUserInfo = JSON.parse(responseBody);
    const { setFirstName } = useAuthInfoStore();
    setFirstName(authorizedUserInfo.firstName);
    router.push({ name: "home" });
  } else if (route.path !== '/auth') {
    authorizationService
      .tryAuthorize()
      .catch(() => router.push({ name: "auth-error" }));
  }
});
</script>
```

```
Figure 7.4. authInfoStore.ts
import { ref } from "vue";
import { defineStore } from "pinia";

export const useAuthInfoStore =
  defineStore("authInfoStore", () => {
    const firstName = ref('');

    function setFirstName(newFirstName: string) {
      firstName.value = newFirstName;
    }

    return { firstName, setFirstName };
  });
```

- 攻击者能够以管理员身份行事，因为服务未检查用户是否具有适当的角色（风险：高）
- 攻击者试图为他人创建订单（风险：高）
- 攻击者试图读取/更新他人的订单（风险：高）

- 配置管理
    - 攻击者能够访问操作系统 root 权限，因为进程以 root 用户权限运行（风险：中）

- 传输中和静态数据保护
    - 攻击者试图使用 SQL 注入篡改数据库（风险：高）
    - 攻击者能够捕获和修改未加密的互联网流量（风险：高）
    - 攻击者能够捕获和修改未加密的内部网络流量（风险：低）
    - 攻击者能够访问敏感信息，因为这些信息未被正确加密（风险：中）
    - 信息泄露给攻击者，因为互联网流量是明文的，即未受保护（风险：高）
    - 信息泄露给攻击者，因为内部网络流量是明文的，即未受保护（风险：低）

- 数据验证
    - 攻击者试图发送过多请求（风险：高）
    - 攻击者试图在数据大小完全不受限制的情况下发送包含大量数据的请求（风险：高）
    - 攻击者试图使用 SQL 注入篡改数据库（风险：高）
    - 攻击者试图通过发送可能导致正则表达式求值的字符串来进行正则表达式 DoS (ReDoS) 攻击
    - 攻击者试图在请求中发送无效值，以尝试使服务崩溃或在没有适当输入验证的情况下导致无限循环（风险：高）

- 异常管理
    - 攻击者在请求响应中收到敏感信息，如详细的堆栈跟踪。（风险：中）

您甚至可以同时使用两种不同的威胁分类方法，如 STRIDE 和 ASF，因为使用多种方法更有可能发现所有可能的威胁。现在考虑 ASF 分类，我们可以看到配置管理类别涉及机密信息的存储。当我们使用 STRIDE 时，我们没有发现任何与机密信息相关的威胁。但如果我们仔细想想，我们的 *order-service* 应该至少有三个机密信息：数据库用户名、数据库用户密码以及用于加密数据库中敏感数据的加密密钥。我们必须将这些机密信息存储在安全的地方，例如 Kubernetes 环境中的 Secret。这些机密信息都不应硬编码在源代码中。

应用程序的头部显示已登录用户的名和一个用于注销用户的按钮：

**图 7.5. HeaderView.vue**

```html
<template>
  <span>{{authInfoStore.firstName}}</span>
  &nbsp;
  <button @click="logout">Logout</button>
</template>

<script setup>
import { useRouter } from "vue-router";
import authorizationService from "@/authService";
import { useAuthInfoStore } from "@/stores/authInfoStore";

const authInfoStore = useAuthInfoStore();
const router = useRouter();

function logout() {
  authorizationService
    .tryLogout()
    .catch(() => router.push({ name: "auth-error" }));
}
</script>
```

`tryMakeHttpRequest` 函数是对浏览器全局 `fetch` 方法的封装。如果 HTTP 请求返回 HTTP 状态码 403 *Forbidden*，它将启动授权流程。

**图 7.6. tryMakeHttpRequest.ts**

```typescript
import authorizationService from "@/authService";

let originalFetch: typeof fetch;

export default function tryMakeHttpRequest(
  url: RequestInfo,
  options?: RequestInit
): Promise<Response> {
  return originalFetch(url, options).then(async (response) => {
    if (response.status === 403) {
      try {
        await authorizationService.tryAuthorize();
      } catch {
        // 处理授权错误，返回状态为 403 的响应
      }
    }

    return response;
  });
}

export function setupFetch() {
  originalFetch = fetch;
  // @ts-ignore
  // eslint-disable-next-line no-global-assign
  fetch = () => Promise.reject(new Error('Global fetch not implemented'));
}
```

以下是服务工作者的实现：

## 图 7.7. serviceWorker.js

```javascript
const allowedOrigins = [
  "http://localhost:8080", // 开发环境中的 IAM
  "http://localhost:3000", // 开发环境中的 API
  "https://software-system-x.domain.com" // 生产环境
];

const apiEndpointRegex = /\/api\//;
const tokenEndpointRegex = /\/openid-connect\/token$/;
const data = {};

// 监听包含需要存储在服务工作者内部的数据的消息
addEventListener("message", (event) => {
  if (event.data) {
    data[event.data.key] = event.data.value;
  }
});

function respondWithUserInfo(event) {
  const response =
    new Response(data.authorizedUserInfo
                    ? JSON.stringify(data.authorizedUserInfo)
                    : '');
  event.respondWith(response);
}

function respondWithIdToken(event) {
  const response = new Response(data.idToken
                                    ? data.idToken
                                    : '');
  event.respondWith(response);
}

function respondWithTokenRequest(event) {
  let body = "grant_type=authorization_code";
  body += `&code=${data.code}`;
  body += `&client_id=app-x`;
  body += `&redirect_uri=${data.redirectUri}`;
  body += `&code_verifier=${data.codeVerifier}`;
  const tokenRequest = new Request(event.request, { body });

  // 验证从授权服务器收到的 state 是否与本应用之前发送的相同
  if (data.state === data.receivedState) {
    event.respondWith(fetch(tokenRequest));
  } else {
    // 处理错误
  }
}

function respondWithApiRequest(event) {
  const headers = new Headers(event.request.headers);

  // 添加包含访问令牌的 Authorization 头
  if (data.accessToken) {
    headers.append("Authorization",
                  `Bearer ${data.accessToken}`);
  }

  const authorizedRequest = new Request(event.request, {
    headers
  });

  event.respondWith(fetch(authorizedRequest));
}

function fetchHandler(event) {
  const requestUrl = new URL(event.request.url);

  if (event.request.url.endsWith('/authorizedUserInfo') &&
      !apiEndpointRegex.test(requestUrl.pathname)) {
    respondWithUserInfo(event);
  } else if (event.request.url.endsWith('/idToken') &&
             !apiEndpointRegex.test(requestUrl.pathname)) {
    respondWithIdToken(event);
  } else if (allowedOrigins.includes(requestUrl.origin)) {
    if (tokenEndpointRegex.test(requestUrl.pathname)) {
      respondWithTokenRequest(event);
    } else if (apiEndpointRegex.test(requestUrl.pathname)) {
      respondWithApiRequest(event);
    }
  } else {
    event.respondWith(fetch(event.request));
  }
}

// 拦截所有 fetch 请求并使用 'fetchHandler' 处理它们
addEventListener("fetch", fetchHandler);
```

使用 OAuth2 授权码流程的授权是通过浏览器重定向到以下类型的 URL 开始的：

```
https://authorization-server.com/auth?response_type=code&client_id=CLIENT_ID&redirect_uri=https://example-app.com/cb&scope=photos&state=1234zyx...ghvx3&code_challenge=CODE_CHALLENGE&code_challenge_method=SHA256
```

上述 URL 中的查询参数如下：

- `response_type=code` - 表示您期望接收授权码
- `client_id` - 您在授权服务器上创建客户端时使用的客户端 ID
- `redirect_uri` - 表示授权完成后将浏览器重定向到的 URI。您还需要在授权服务器中定义此 URI。
- `scope` - 一个或多个范围值，表示您希望访问用户账户的哪些部分。范围应由 URL 编码的空格字符分隔
- *state* - 您的应用程序生成的随机字符串，您稍后将对其进行验证
- *code_challenge* - PKCE 扩展：代码验证器的 URL 安全的 base64 编码的 SHA256 哈希值。代码验证器是您生成的随机字符串密钥
- *code_challenge_method=S256* - PKCE 扩展：指示使用哪种哈希方法（S256 表示 SHA256）

我们需要使用 PKCE 扩展作为额外的安全措施，因为我们在前端而不是后端执行授权码流程。

如果授权成功，授权服务器将使用作为 URL 查询参数给出的 *code* 和 *state* 将浏览器重定向到上面给出的 *redirect_uri*，例如：

```
https://example-app.com/cb?code=AUTH_CODE_HERE&state=1234zyx...ghvx3
```

- *code* - 从授权服务器返回的授权码
- *state* - 与您之前传递的相同的 state 值

应用程序成功授权后，可以使用以下类型的 HTTP POST 请求请求令牌：

```
POST https://authorization-server.com/token HTTP/1.1
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=AUTH_CODE_HERE&
redirect_uri=REDIRECT_URI&
client_id=CLIENT_ID&
code_verifier=CODE_VERIFIER
```

- *grant_type=authorization_code* - 此流程的授权类型是 _authorization_code
- *code=AUTH_CODE_HERE* - 这是当浏览器从授权服务器重定向回您的应用程序时收到的代码。
- *redirect_uri=REDIRECT_URI* - 必须与授权期间提供的重定向 URI 相同
- *client_id=CLIENT_ID* - 您在授权服务器上创建客户端时使用的客户端 ID
- *code_verifier=CODE_VERIFIER* - 您之前生成的随机字符串密钥

以下是 `AuthorizationService` 类的实现。它提供了授权、获取令牌和注销的方法。

## 安全原则

396

## 图 7.8. AuthorizationService.ts

```typescript
import pkceChallenge from "pkce-challenge";
import jwt_decode from "jwt-decode";
import tryMakeHttpRequest from "@/tryMakeHttpRequest";
import type { useAuthInfoStore } from "@/stores/authInfoStore";

interface AuthorizedUserInfo {
  readonly userName: string;
  readonly firstName: string;
  readonly lastName: string;
  readonly email: string;
}

export default class AuthorizationService {
  constructor(
    private readonly oidcConfigurationEndpoint: string,
    private readonly clientId: string,
    private readonly authRedirectUrl: string,
    private readonly loginPageUrl: string
  ) {}

  // 尝试使用 OpenID Connect 授权码流程授权用户
  async tryAuthorize(): Promise<void> {
    // 将重定向 URI 存储在服务工作者中
    navigator.serviceWorker?.controller?.postMessage({
      key: "redirectUri",
      value: this.authRedirectUrl
    });

    // 将 state 密钥存储在服务工作者中
    const state = crypto.randomUUID();
    navigator.serviceWorker?.controller?.postMessage({
      key: "state",
      value: state,
    });

    // 生成 PKCE 挑战并将代码验证器存储在服务工作者中
    const challenge = pkceChallenge(128);
    navigator.serviceWorker?.controller?.postMessage({
      key: "codeVerifier",
      value: challenge.code_verifier,
    });

    const authUrl = await this.tryCreateAuthUrl(state, challenge);

    // 将浏览器重定向到授权服务器的授权 URL
    location.href = authUrl;
  }

  // 尝试从授权服务器的令牌端点获取访问令牌、刷新令牌和 ID 令牌
  async tryGetTokens(
    authInfoStore: ReturnType<typeof useAuthInfoStore>
  ): Promise<void> {
    const oidcConfiguration = await this.getOidcConfiguration();
```const response =
  await tryMakeHttpRequest(oidcConfiguration.token_endpoint, {
    method: "post",
    mode: "cors",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
});

const tokens = await response.json();
this.storeTokens(tokens);
this.storeAuthorizedUserInfo(tokens.id_token, authInfoStore);
}

// 登出并重定向到登录页面
async tryLogout(): Promise<void> {
  const oidcConfiguration = await this.getOidcConfiguration();

  // 清除服务工作线程中的授权用户信息
  navigator.serviceWorker?.controller?.postMessage({
    key: "authorizedUserInfo",
    value: undefined
  });

  // 从服务工作线程获取ID令牌
  const response = await tryMakeHttpRequest("/idToken");
  const idToken = await response.text();

  // 将浏览器重定向到授权服务器的
  // 登出端点
  if (idToken !== "") {
    location.href =
      oidcConfiguration.end_session_endpoint +
      `?post_logout_redirect_uri=${this.loginPageUrl}` +
      `&id_token_hint=${idToken}`;
  } else {
    location.href = oidcConfiguration.end_session_endpoint;
  }
}

private async getOidcConfiguration(): Promise<any> {
  const response =
    await tryMakeHttpRequest(this.oidcConfigurationEndpoint);

  return response.json();
}

private async tryCreateAuthUrl(
  state: string,
  challenge: ReturnType<typeof pkceChallenge>
) {
  const oidcConfiguration = await this.getOidcConfiguration();
  let authUrl = oidcConfiguration.authorization_endpoint;

  authUrl += "?response_type=code";
  authUrl += "&scope=openid+profile+email";
  authUrl += `&client_id=${this.clientId}`;
  authUrl += `&redirect_uri=${this.authRedirectUrl}`;
  authUrl += `&state=${state}`;
  authUrl += `&code_challenge=${challenge.code_challenge}`;
  authUrl += "&code_challenge_method=S256";

  return authUrl;
}

private storeTokens(tokens: any) {
    navigator.serviceWorker?.controller?.postMessage({
        key: "accessToken",
        value: tokens.access_token,
    });

    navigator.serviceWorker?.controller?.postMessage({
        key: "refreshToken",
        value: tokens.refresh_token,
    });

    navigator.serviceWorker?.controller?.postMessage({
        key: "idToken",
        value: tokens.id_token,
    });
}

private storeAuthorizedUserInfo(
    idToken: any,
    authInfoStore: ReturnType<typeof useAuthInfoStore>
) {
    const idTokenClaims: any = jwt_decode(idToken);

    const authorizedUserInfo = {
        userName: idTokenClaims.preferred_username,
        firstName: idTokenClaims.given_name,
        lastName: idTokenClaims.family_name,
        email: idTokenClaims.email,
    };

    navigator.serviceWorker?.controller?.postMessage({
        key: "authorizedUserInfo",
        value: authorizedUserInfo
    });

    authInfoStore.setFirstName(idTokenClaims.given_name);
}

以下是在 `tryGetTokens` 方法中执行 `tryMakeHttpRequest` 函数时获得的响应示例：

```json
{
  "access_token": "eyJz93a...k41aUWw",
  "id_token": "UFn43f...c5vvfGF",
  "refresh_token": "GEbRxBN...edjnXbL",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

`AuthorizationCallback` 组件是在授权服务器成功授权后将浏览器重定向回应用程序时渲染的组件。该组件将授权码和接收到的 state 存储在服务工作线程中，并发起令牌请求。收到令牌后，它会将应用程序路由到主页。作为额外的安全措施，只有当原始 state 和接收到的 state 相等时，才会执行令牌请求。此检查在服务工作线程代码中完成。

图 7.9. AuthorizationCallback.vue

```vue
<template>
  <div></div>
</template>

<script setup>
import { onMounted } from "vue";
import { useRouter, useRoute } from "vue-router";
import authorizationService from "@/authService";
import { useAuthInfoStore } from "@/stores/authInfoStore";

const { query } = useRoute();
const router = useRouter();
const authInfoStore = useAuthInfoStore();

onMounted(async () => {
  // 将授权码存储在服务工作线程中
  navigator.serviceWorker?.controller?.postMessage({
    key: "code",
    value: query.code,
  });

  // 将接收到的 state 存储在服务工作线程中
  navigator.serviceWorker?.controller?.postMessage({
    key: "receivedState",
    value: query.state,
  });

  // 尝试获取令牌
  try {
    await authorizationService.tryGetTokens(authInfoStore);
    router.push({ name: "home" });
  } catch (error) {
    router.push({ name: "auth-error" });
  }
});
</script>
```

应用程序使用的其他 UI 组件定义如下：

图 7.10. AuthorizationError.vue

```vue
<template>
  <div>Error</div>
</template>
```

图 7.11. LoginView.vue

```vue
<template>
  <div>Login</div>
</template>
```

图 7.12. HomeView.vue

```vue
<template>
  <div>Home</div>
</template>
```

应用程序的路由如下：

图 7.13. router.ts

```typescript
import { createRouter, createWebHistory } from "vue-router";
import AuthorizationCallback from "@/AuthorizationCallback.vue";
import AuthorizationError from "@/AuthorizationError.vue";
import HomeView from "@/HomeView.vue";
import LoginView from "@/LoginView.vue";

const routes = [
  {
    path: "/",
    name: "login",
    component: LoginView,
  },
  {
    path: "/auth",
    name: "auth",
    component: AuthorizationCallback,
  },
  {
    path: "/auth-error",
    name: "auth-error",
    component: AuthorizationError,
  },
  {
    path: "/home",
    name: "home",
    component: HomeView,
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
```

以下 `authService` 模块包含所需常量的定义，并创建了 `AuthorizationService` 类的一个实例。以下代码包含用于本地开发环境的值。在实际应用中，这些值应从环境变量中获取。以下值适用于在 `localhost:8080` 运行 Keycloak 服务且 Vue 应用在 `localhost:5173` 运行的情况。你必须在 Keycloak 中创建一个名为 ‘app-x’ 的客户端。此外，你必须定义一个有效的重定向 URI 并添加一个允许的 Web 来源。最后，你必须配置一个有效的登出后重定向 URI（参见下图）。Keycloak 中的默认访问令牌有效期仅为一分钟。你可以在领域设置（令牌选项卡）中为测试目的增加该时间。

## 访问设置

![](img/cbd069395d7b824346b69b1f92e0fb4a_413_0.png)

图 7.14. 客户端的 Keycloak 设置

图 7.15. authService.ts

```typescript
import AuthorizationService from "@/AuthorizationService";

const oidcConfigurationEndpoint =
    "http://localhost:8080/realms/master/.well-known/openid-configuration";

const clientId = "app-x";
const redirectUrl = "http://127.0.0.1:5173/auth";
const loginPageUrl = "http://127.0.0.1:5173";

const authorizationService = new AuthorizationService(
    oidcConfigurationEndpoint,
    clientId,
    redirectUrl,
    loginPageUrl
);

export default authorizationService;
```

## 7.4.1.2：后端中的 OAuth2 授权

只允许授权用户访问资源。避免忘记实现授权的最佳方法是默认拒绝访问资源。你可以要求所有控制器方法都必须存在授权注解。如果 API 端点不需要授权，可以使用像 `@allow_any_user` 这样的特殊注解。如果控制器方法缺少授权注解，则可以抛出异常。这样，你就永远不会忘记为控制器方法添加授权注解。

失效的访问控制在 OWASP 2021 年十大安全风险中排名第一。特别要记住禁止用户为其他用户创建资源。同时禁止用户查看、编辑或删除属于其他人的资源（也称为不安全的直接对象引用 (IDOR) 防护）。仅使用通用唯一标识符 (UUID) 作为资源的 ID 而不使用基本整数是不够的。这是因为如果攻击者可以获得一个带有 UUID 的对象的 URL，他就可以访问该 URL 后面的对象，因为没有访问控制。

以下是一个基于 JWT 的授权器类，可用于 FastAPI 后端 API 服务。在示例中，我们使用了以下额外的 Python 库：`python-benedict` 和 `pyjwt`。以下示例利用了基于角色的访问控制 (RBAC)，但还有更现代的替代方案，包括基于属性的访问控制 (ABAC) 和基于关系的访问控制 (ReBAC)。有关这些的更多信息可在 OWASP 授权备忘单¹ 中找到。

¹https://cheatsheetsseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html

from typing import Protocol

class Authorizer(Protocol):
    pass

## 图 7.16. jwt_authorizer.py

```python
import os
from typing import Any, Final
from collections.abc import Callable

import requests
from Authorizer import Authorizer
from benedict import benedict
from fastapi import HTTPException, Request
from jwt import PyJWKClient, PyJWKClientError, decode
from jwt.exceptions import InvalidTokenError

class __JwtAuthorizer(Authorizer):
    IAM_ERROR: Final = 'IAM error'

    def __init__(self):
        # IAM 系统中的 OpenId Connect 配置端点
        self.__oidc_config_url = os.environ['OIDC_CONFIG_URL']
        self.__jwks_client = None

        # 使用 Keycloak 时，你可以使用例如 realm_access.roles
        self.__roles_claim_path = os.environ['JWT_ROLES_CLAIM_PATH']

        # 这是你可以根据访问令牌中特定的 'sub' 声明值
        # 获取用户 ID 的 URL
        # 例如：http://localhost:8082/user-service/users
        self.__get_users_url = os.environ['GET_USERS_URL']

    def authorize(self, request: Request) -> None:
        self.__decode_jwt_claims(request.headers.get('Authorization'))

    # 授权用户为自己创建资源
    # 检查提供的 user_id 是否与拥有该 JWT 的用户的 user_id 相同
    # 注意！对于某些非 Keycloak 的 IAM 系统，
    # 你可能需要使用 'uid' 声明而不是 'sub' 来获取唯一用户 ID
    def authorize_for_self(
        self, user_id: int, request: Request
    ) -> None:
        jwt_user_id = self.__get_jwt_user_id(request)
        user_is_authorized = user_id == jwt_user_id
        if not user_is_authorized:
            raise HTTPException(status_code=403, detail='Unauthorized')

    # 仅授权用户访问其自己的资源
    # 如果找不到具有给定 id 和 user_id 组合的实体，
    # 则引发认证错误
    def authorize_for_user_own_resources_only(
        self,
        id: int,
        get_entity_by_id_and_user_id: Callable[[int, int], Any],
        request: Request
    ) -> None:
        jwt_user_id = self.__get_jwt_user_id(request)

        try:
            get_entity_by_id_and_user_id(id, jwt_user_id)
        except HTTPException as error:
            if error.status_code == 404:
                raise HTTPException(status_code=403, detail='Unauthorized')
            # 记录错误详情
            raise HTTPException(status_code=500, detail=self.IAM_ERROR)

    def authorize_if_user_has_one_of_roles(
        self, allowed_roles: list[str], request: Request
    ) -> None:
        claims = self.__decode_jwt_claims(
            request.headers.get('Authorization')
        )

        try:
            roles = benedict(claims)[self.__roles_claim_path]
        except KeyError as error:
            # 记录错误详情
            raise HTTPException(status_code=500, detail=self.IAM_ERROR)

        user_is_authorized = any(
            [True for role in roles if role in allowed_roles]
        )
        if not user_is_authorized:
            raise HTTPException(status_code=403, detail='Unauthorized')

    def __decode_jwt_claims(
        self, auth_header: str | None
    ) -> dict[str, Any]:
        if not auth_header:
            raise HTTPException(status_code=401, detail='Unauthenticated')

        try:
            if not self.__jwks_client:
                oidc_config_response = requests.get(self.__oidc_config_url)
                oidc_config_response.raise_for_status()
                oidc_config = oidc_config_response.json()
                self.__jwks_client = PyJWKClient(oidc_config['jwks_uri'])

            jwt = auth_header.split('Bearer ')[1]
            signing_key = self.__jwks_client.get_signing_key_from_jwt(jwt)
            jwt_claims = decode(jwt, signing_key.key, algorithms=['RS256'])
        except (
            requests.RequestException,
            KeyError,
            PyJWKClientError
        ) as error:
            # 记录错误详情
            raise HTTPException(status_code=500, detail=self.IAM_ERROR)
        except (IndexError, InvalidTokenError):
            raise HTTPException(status_code=403, detail='Unauthorized')

        return jwt_claims

    def __get_jwt_user_id(self, request: Request) -> int:
        claims = self.__decode_jwt_claims(
            request.headers.get('Authorization')
        )

        try:
            sub_claim = claims['sub']
            users_response = requests.get(
                f'{self.__get_users_url}?sub={sub_claim}&fields=id'
            )
            users_response.raise_for_status()
            # 预期的响应 JSON 格式为 [{ "id": 12345 }]
            users = users_response.json()
        except (KeyError, requests.RequestException) as error:
            # 记录错误详情
            raise HTTPException(status_code=500, detail=self.IAM_ERROR)

        try:
            return users[0]['id']
        except (IndexError, AttributeError):
            raise HTTPException(status_code=403, detail='Unauthorized')

authorizer = __JwtAuthorizer()
```

以下是一个使用上述定义的 JwtAuthorizer 的示例 API 服务：

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from InputOrder import InputOrder
from jwt_authorizer import authorizer
from order_service import order_service
# OrderUpdate 是一个数据传输对象，不应包含 user_id 属性，
# 因为它不能被更改
from OrderUpdate import OrderUpdate

app = FastAPI()

# 定义一个自定义的 HTTPException 处理器，
# 提供管理员日志记录和指标更新
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, error: StarletteHTTPException
):
    if error.status_code == 403:
        # 审计日志记录未授权请求
        pass
    # 使用以下指标标签将 'HTTP request failures' 计数器加一：
    # error.status_code, error.detail
    return JSONResponse({'error: ': str(error.detail)}, status_code=error.status_code)

@app.get('/sales-item-service/sales-items')
async def get_sales_items():
    # 无需认证/授权
    # 发送销售商品
    pass

@app.post('/messaging-service/messages')
async def create_message(request: Request):
    authorizer.authorize(request)
    # 已认证用户可以创建消息

@app.get('/order-service/orders/{id}')
async def get_order(id: int, request: Request):
    authorizer.authorize_for_user_own_resources_only(
        id,
        order_service.get_order_by_id_and_user_id,
        request
    )
    # 获取由 'id' 标识且拥有 JWT 所有者用户 ID 的订单

@app.post('/order-service/orders')
async def create_order(order: InputOrder, request: Request):
    authorizer.authorize_for_self(
        order.user_id,
        request
    )
    # 为用户创建订单
    # 用户不能为其他用户创建订单

@app.put('/order-service/orders/{id}')
async def update_order(id: int, order: OrderUpdate, request: Request):
    authorizer.authorize_for_user_own_resources_only(
        id,
        order_service.get_order_by_id_and_user_id,
        request
    )
    # 更新由 'id' 标识且拥有 JWT 所有者用户 ID 的订单

@app.delete('/order-service/orders/{id}')
async def delete_order(id: int, request: Request):
    authorizer.authorize_if_user_has_one_of_roles(
        ['admin'], request
    )
    # 只有管理员用户可以删除订单
```

在上面的示例中，授权逻辑是分别编码在每个请求处理器内部的。我们可以将授权逻辑从请求处理器方法中提取出来，放到可以与方法结合使用的装饰器中。这些装饰器可以在一个单独的库中实现，并且它们可以接受任何实现了 Authorizer 协议的授权器，而不仅仅是 JwtAuthorizer：

```python
from collections.abc import Callable
from functools import wraps
from typing import Any

from Authorizer import Authorizer


class AuthDecorException(Exception):
    pass


def allow_any_user(handle_request):
    return handle_request

def allow_authorized_user(authorizer: Authorizer):
    def decorate(handle_request):
        @wraps(handle_request)
        async def wrapped_handle_request(*args, **kwargs):
            try:
                authorizer.authorize(kwargs['request'])
            except KeyError:
                raise AuthDecorException(
                    "Request handler must accept 'request' parameter"
                )
            return await handle_request(*args, **kwargs)
        return wrapped_handle_request
    return decorate

def allow_for_self(authorizer: Authorizer):
    def decorate(handle_request):
        @wraps(handle_request)
        async def wrapped_handle_request(*args, **kwargs):
            try:
                user_id = (
                    kwargs['user_id']
                    if kwargs.get('user_id')
                    else kwargs[
                        [
                            key
                            for key in kwargs.keys()
                            if key.endswith('dto')
                        ][0]
                    ].user_id
                )
                authorizer.authorize_for_user_own_resources_only(
                    user_id, kwargs['request']
                )
            except (AttributeError, IndexError, KeyError):
                raise AuthDecorException("""
                    Request handler must accept 'request' parameter,
                    'user_id' integer parameter or DTO parameter
                    with name ending with 'dto'. DTO parameter
                    must have attribute 'user_id'
                    """)
            return await handle_request(*args, **kwargs)
        return wrapped_handle_request
    return decorate

def allow_for_user_own_resources_only(
```

## 7.4.2：密码策略

实施密码策略，要求使用强密码，并优先选择密码短语而非普通密码。密码短语应包含多个单词。与强密码相比，密码短语更难被攻击者猜测，也更容易被用户记住。允许密码短语包含Unicode字符，这使用户能够使用母语创建密码短语。

应要求密码足够强，并符合以下标准：

- 至少12个字符长
- 至少包含一个大写字母
- 至少包含一个小写字母
- 至少包含一个数字
- 至少包含一个特殊字符
- 不得包含用户名
- 不得包含过多相同的数字或字母，例如，包含“111111”、“aaaaaa”或“1a1a1a1a1a”的密码应被拒绝
- 不得包含过多连续的数字或字母，例如，包含“12345”、“56789”、“abcdef”或“klmno”的密码应被拒绝
- 不得包含键盘上过多相邻的字母，例如，包含“qwerty”的密码应被拒绝
- 不得包含黑名单中的单词。将所有常用、易猜的密码列入黑名单。

机器对机器（非人类相关）的密码（如数据库密码）应在部署期间为每个生产环境自动生成。这些密码应是随机的，且长度应显著超过12个字符。

## 7.4.3：密码学

以下是与密码学相关的关键安全功能：

- 不要以明文形式传输数据
- 无需在所有微服务中都实现HTTPS，因为可以设置服务网格并配置其在服务间实现mTLS
- 不要以明文形式存储敏感信息，如个人身份信息（PII）
    - 在将敏感数据存储到数据库之前进行加密，并在从数据库获取时解密
    - 记住根据隐私法律、监管要求或业务需求来识别哪些数据被归类为敏感数据
    - 不要使用FTP和SMTP等旧协议传输敏感数据
    - 尽快丢弃敏感数据，或使用令牌化（例如，符合PCI DSS标准）甚至截断
    - 不要缓存敏感数据
- 不要使用旧的/弱的密码算法。使用强大的算法，如SHA-256或AES-256
- 不允许在生产环境中使用默认/弱密码或默认加密密钥
    - 可以在微服务运行于生产环境时，为密码/加密密钥实现验证逻辑。如果微服务使用的密码/加密密钥不够强，微服务不应运行，而应报错退出

### 7.4.3.1：加密密钥生命周期与轮换

当满足以下一个或多个条件时，应轮换（即更改）加密密钥：

- 已知当前密钥被泄露或存在泄露嫌疑
- 已经过指定的时间段（这被称为密码周期）
- 密钥已用于加密特定数量的数据
- 所使用的加密算法提供的安全性发生重大变化（例如，宣布了新的攻击方法）

加密密钥轮换应确保所有现有数据被解密并使用新密钥重新加密。这当然是逐步发生的，因此，例如，每个加密的数据库表行都必须包含所使用的加密密钥的ID。当所有现有数据都使用新密钥加密，意味着所有对旧密钥的引用都被移除后，旧密钥可以被销毁。

## 7.4.4：拒绝服务（DoS）防护

DoS防护应至少通过以下方式实现：

- 为微服务建立请求速率限制。这可以在API网关层面或由云提供商完成
- 使用验证码（Captcha）防止非人类（机器人）用户执行可能代价高昂的操作，例如创建新资源或获取大型资源（如大文件）

## 7.4.5：数据库安全

-   连接到数据库的微服务连接必须通过TLS保护。在Kubernetes环境中，可以通过使用服务网格（如Istio）并在环境中的所有服务之间配置mTLS来实现这一点。
-   数据库凭证（用户名和密码）必须存储在安全的位置，例如Kubernetes环境中的Secrets中。切勿将凭证存储在源代码中。
-   使用强密码，最好是为特定环境自动生成的、随机且足够长的密码。如果数据库引擎允许，密码应至少为32个字符。
-   为管理员和常规使用配置不同的数据库用户账户。为两个用户账户分配最小权限。每个账户使用单独的密码。常规数据库用户通常只能执行以下SQL语句：SELECT、INSERT、UPDATE和DELETE。只有使用管理员用户账户登录的用户才能创建/修改/删除表/索引等。

## 7.4.6：SQL注入预防

以下是防止SQL注入攻击需要实施的关键特性：

-   使用参数化SQL语句。不要将用户提供的数据直接拼接到SQL语句字符串中。
-   请记住，并非SQL语句的所有部分都可以使用参数化。如果必须将用户提供的数据放入SQL语句而不使用参数化，请先对其进行清理/验证。例如，对于LIMIT，必须验证用户提供的值是否为整数且在给定范围内。
-   迁移使用ORM（对象关系映射）。
-   在查询中对获取的记录数量使用适当的限制，以防止大量记录泄露。
-   验证第一个查询结果行的正确形状。如果第一行数据的形状错误（例如，包含错误的字段），则不要将查询结果发送给客户端。

## 7.4.7：操作系统命令注入预防

在shell中执行操作系统命令时，不应允许使用用户提供的数据。例如，不允许以下情况：

```python
import os

user_supplied_dir = ...
os.system(f'mkdir {user_supplied_dir}')
```

恶意用户可以提供例如以下类型的目录：`some_dir && rm -rf /`。相反，应使用`os`模块提供的特定函数：

```python
import os

user_supplied_dir = ...
os.mkdir(user_supplied_dir)
```

## 7.4.8：安全配置

默认情况下，容器的安全上下文应如下：

-   容器不应具有特权。
-   丢弃所有能力。
-   容器文件系统为只读。
-   仅允许非root用户在容器内运行。
-   定义容器应在其中运行的非root用户和组。
-   禁止权限提升。

上述Docker容器安全配置的示例将在本书后面的*DevSecOps原则*章节中给出。

在API网关中实施发送安全相关的HTTP响应头：

-   X-Content-Type-Options: nosniff
-   Strict-Transport-Security: max-age: ; includeSubDomains
-   X-Frame-Options: DENY
-   Content-Security-Policy: frame-ancestors 'none'
-   Content-Type: application/json
-   如果未特别启用和配置缓存，则应设置以下头：Cache-Control: no-store
-   Access-Control-Allow-Origin: https://your_domain_here

如果返回的是HTML而不是JSON，还应替换/添加以下响应头：

-   Content-Security-Policy: default-src 'none'
-   Referrer-Policy: no-referrer

使用`Permissions-Policy`响应头禁用不需要/不想要的浏览器功能。以下示例禁用了所有列出的功能：

```
Permissions-Policy: accelerometer=(), ambient-light-sensor=(),
    autoplay=(), battery=(), camera=(), cross-origin-isolated=(),
    display-capture=(), document-domain=(), encrypted-media=(),
    execution-while-not-rendered=(), execution-while-out-of-viewport=(),
    fullscreen=(), geolocation=(), gyroscope=(), keyboard-map=(),
    magnetometer=(), microphone=(), midi=(), navigation-override=(),
    payment=(), picture-in-picture=(), publickey-credentials-get=(),
    screen-wake-lock=(), sync-xhr=(), usb=(), web-share=(), xr-spatial-tracking=()
```

## 7.4.9：自动漏洞扫描

在微服务CI流水线和容器注册表中定期实施自动漏洞扫描。在容器注册表（例如Docker或云供应商提供的注册表）中配置容器漏洞扫描非常重要。此扫描最好每天进行一次。软件系统的所有软件组件都应被扫描。至少立即修复所有严重和高危漏洞。

## 7.4.10：完整性

仅使用带有SHA摘要标签的容器镜像。如果攻击者成功发布具有相同标签的恶意容器镜像，SHA摘要可防止使用该恶意镜像。确保从受信任的来源（如NPM或Maven）使用库和依赖项。您还可以托管仓库的内部镜像，以避免意外使用任何不受信任的仓库。确保对所有代码（源代码、部署、基础设施）和配置更改存在审查流程，以便不会将恶意代码引入您的软件系统。

## 7.4.11：错误处理

确保API响应中的错误消息不包含敏感信息或实现细节。不要在生产环境中传输给客户端的错误响应中添加堆栈跟踪。

例如，如果API请求产生与IAM系统连接相关的内部服务器错误，您不应在错误响应中透露实现细节，例如提及您使用的*Keycloak 18.06*，而应使用抽象术语，如*IAM系统*。如果攻击者获得揭示所用软件组件某些细节的错误响应，攻击者就能够利用该特定软件组件的可能漏洞。

## 7.4.12：日志记录

编写日志条目时，切勿将以下任何内容写入日志：

-   会话ID
-   访问令牌
-   个人身份信息（PII）
-   密码
-   数据库连接字符串
-   加密密钥
-   不合法收集的信息
-   最终用户选择退出收集的信息

## 7.4.13：审计日志

应记录可审计的最终用户相关事件，例如登录、登录失败、未授权或无效请求以及高价值交易，并将其存储在外部审计日志系统中。审计日志系统应自动检测与最终用户相关的可疑操作并发出警报。另请参阅[OWASP日志记录词汇表速查表](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Vocabulary_Cheat_Sheet.html)。

## 7.4.14：输入验证

始终验证来自不受信任来源的输入，例如来自最终用户。有许多方法可以实现验证，并且有多个库可用于此目的。假设您使用ORM并实现实体和数据传输对象。确保正确验证的最佳方法是要求每个实体/DTO属性都必须具有验证注解。如果实体或DTO中的属性不需要任何验证，请使用特殊注解（例如`@any_value`）注解该属性。始终在两个方向使用DTO，从客户端到服务器以及从服务器到客户端。这是因为实体可能包含一些敏感数据，这些数据不是客户端预期的或不应暴露给客户端。例如，`User`实体可能有一个属性`is_admin`。您不应期望从客户端接收`User`实体作为输入，而应期望接收DTO `InputUser`，它与`User`实体相同，但缺少某些字段，如`id`、`created_at_timestamp`和`is_admin`。同样，如果`User`实体包含`password`属性，则该属性不应发送回客户端。因此，您需要定义一个`OutputUser` DTO，它与`User`实体相同，但缺少`password`属性。当您在输入和输出中都使用DTO时，您可以保护服务免受以下情况的影响：您向实体添加新的敏感属性，而该新敏感属性不应从客户端接收或传输给客户端。

记住验证来自所有可能来源的未验证数据：

-   命令行参数
-   环境变量
-   标准输入（stdin）

[2] https://cheatsheetseries.owasp.org/cheatsheets/Logging_Vocabulary_Cheat_Sheet.html

-   来自文件系统的文件
-   来自套接字的数据（网络输入）
-   用户界面输入

## 7.4.14.1：验证数字

验证数值时，务必验证该值是否在指定范围内。例如，如果使用未经验证的数字来检查循环是否应结束，而该数字非常大，则可能导致拒绝服务（DoS）攻击。如果数字应为整数，则不允许使用浮点值。

## 7.4.14.2：验证字符串

验证字符串时，务必首先验证字符串的最大长度。之后再执行其他验证。使用正则表达式验证长字符串可能导致正则表达式拒绝服务（ReDoS）攻击。应避免为验证目的自行编写正则表达式，而应使用包含经过实战检验代码的现成库。也可考虑使用 [Google RE2 库](https://github.com/google/re2/tree/abseil/python)。它比许多语言运行时提供的正则表达式功能更安全，你的代码也更不易受到 ReDoS 攻击。

## 7.4.14.3：验证时间戳

时间戳（或时间、日期）通常以整数或字符串形式给出。对时间戳/时间/日期值应用所需的验证。例如，你可以验证时间戳是在未来还是过去，或者时间戳是早于还是晚于特定时间戳。

## 7.4.14.4：验证数组

验证数组时，应验证数组的大小是否过小或过大，并且可以根据需要验证值的唯一性。此外，在验证数组的最大大小后，记得单独验证数组中的每个值。

## 7.4.14.5：验证对象

通过单独验证对象的每个属性来验证对象。记得也要验证嵌套对象。

## 7.4.14.6：验证上传到服务器的文件

-   确保上传文件的文件名扩展名是允许的扩展名之一
-   确保文件不大于定义的最大大小
-   检查上传的文件是否有病毒和恶意软件
-   如果上传的文件是压缩文件（例如 zip 文件）并且你打算解压缩它，请在解压缩前验证以下内容：
    -   目标路径是可接受的
    -   估计的解压缩大小不会太大
-   在服务器端存储上传的文件时，请注意以下事项：
    -   不要使用用户提供的文件名，而是使用新文件名在服务器上存储文件
    -   不要让用户选择上传文件在服务器上的存储路径

## 8：API 设计原则

本章介绍了面向前端和后端 API 的设计原则。首先讨论面向前端的 API 设计，然后介绍微服务间 API 设计。

## 8.1：面向前端的 API 设计原则

大多数面向前端的 API 应该是基于 HTTP 的 JSON-RPC、REST 或 GraphQL API。当 API 处理深度嵌套的资源，或者客户端希望决定查询应返回哪些字段时，尤其应使用 GraphQL。对于基于订阅的 API，使用服务器发送事件（SSE）或 GraphQL 订阅；对于实时双向通信，使用 WebSocket。如果在前端和后端之间传输大量数据或二进制数据，可以考虑使用 gRPC 和 gRPC Web 实现 API。本书不涵盖 gRPC Web¹。

在接下来的示例中，我们对 JSON 属性名称使用驼峰命名法，因为这是 API 中的事实标准。使用驼峰命名法而非蛇形命名法的另一个好处是，不会立即向客户端暴露实现编程语言（Python）。尽可能隐藏实现细节总是好的。

### 8.1.1：JSON-RPC API 设计原则

> *设计一个 JSON-RPC API，为每个 API 端点执行单个操作（过程）。*

顾名思义，JSON-RPC API 用于执行远程过程调用。远程过程参数是 HTTP 请求体中的一个 JSON 对象。远程过程返回值是 HTTP 响应体中的一个 JSON 对象。客户端通过发出 HTTP POST 请求来调用远程过程，在 URL 路径中指定过程名称，并在请求体中以 JSON 形式提供远程过程调用的参数。

以下是翻译服务 *translate* 过程的请求示例：

```
POST /translation-service/translate HTTP/1.1
Content-Type: application/json

{
    "text": "Ich liebe dich",
    "fromLanguage": "German",
    "toLanguage": "English"
}
```

API 服务器应使用 HTTP 状态码响应，并在 HTTP 响应体中以 JSON 形式包含过程的响应。

对于上述请求，你将得到以下响应：

```
HTTP/1.1 200 OK
Content-Type: application/json

{
    "translatedText": "I love you"
}
```

让我们再看一个 *web-page-search-service* 的例子：

```
POST /web-page-search-service/search-web-pages HTTP/1.1
Content-Type: application/json

{
    "containingText": "Software design patterns"
}
```

```
HTTP/1.1 200 OK
Content-Type: application/json

[
    {
        "url": "https://...",
        "title": "...",
        "date": "...",
        "contentExcerpt": "..."
    },
    更多结果在此 ...
]
```

你可以使用 JSON-RPC 而不是 REST 或 GraphQL 来创建完整的服务。以下是为 *sales-item-service* 定义的五个远程过程。这些过程用于基本的 CRUD 操作。使用 JSON-RPC 而非 REST、GraphQL 或 gRPC 的好处是，你无需学习任何特定技术。

```
POST /sales-item-service/create-sales-item HTTP/1.1
Content-Type: application/json
{
    "name": "Sample sales item",
    "price": 20
}
```

```
POST /sales-item-service/get-sales-items HTTP/1.1
```

```
POST /sales-item-service/get-sales-item-by-id HTTP/1.1
Content-Type: application/json
{
    "id": 1
}
```

```
POST /sales-item-service/update-sales-item HTTP/1.1
Content-Type: application/json
{
    "id": 1,
    "name": "Sample sales item name modified",
    "price": 30
}
```

```
POST /sales-item-service/delete-sales-item-by-id HTTP/1.1
Content-Type: application/json
{
    "id": 1
}
```

```
POST /sales-item-service/delete-sales-items HTTP/1.1
```

你可以轻松地为上述服务创建 API 端点。以下是使用 FastAPI 实现的示例。

```
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputSalesItem(BaseModel):
    name: str
    price: int

class OutputSalesItem(BaseModel):
    id: int
    name: str
    price: int

class SalesItemUpdate(InputSalesItem):
    id: int

class Id(BaseModel):
    id: int

@app.post("/sales-item-service/create-sales-item", response_model=OutputSalesItem)
async def create_sales_items(sales_item: InputSalesItem) -> Any:
    # ...

@app.post("/sales-item-service/get-sales-items", response_model=list[OutputSalesItem])
async def get_sales_items() -> Any:
    # ...

@app.post("/sales-item-service/get-sales-item-by-id", response_model=OutputSalesItem)
async def get_sales_item_by_id(id: Id) -> Any:
    # ...

@app.post("/sales-item-service/update-sales-item", response_model=OutputSalesItem)
async def update_sales_item(sales_item_update: SalesItemUpdate) -> Any:
    # ...

@app.post("/sales-item-service/delete-sales-item-by-id")
async def delete_sales_item_by_id(id: Id) -> None:
    # ...
```

你可以通过在 URL 路径中添加版本号来对 API 进行版本控制。在下面的示例中，新的 API 版本 2 允许为 `search-web-pages` 过程提供新的过程参数 `someNewParam`。

```
POST /web-page-search-service/v2/search-web-pages HTTP/1.1
Content-Type: application/json

{
    "containingText": "Software design patterns",
    "someNewParam": "..."
}
```

### 8.1.2：REST API 设计原则

设计一个 REST API，用于使用 CRUD（创建、读取、更新、删除）操作与资源（或多个资源）进行交互。

许多 API 属于对资源执行 CRUD 操作的类别。让我们创建一个名为 *sales-item-service* 的示例 REST API，用于对销售项执行 CRUD 操作。

#### 8.1.2.1：创建资源

使用 REST API 创建新资源是通过向 API 的资源端点发送 HTTP POST 请求来完成的。API 的资源端点应根据其处理的资源命名。资源端点名称应为名词，并始终使用复数形式，例如，对于 *sales-item-service*，资源端点应为 *sales-items*；对于处理订单的 *order-service*，资源端点应称为 *orders*。

你在 HTTP 请求体中以 JSON 形式提供要创建的资源。要创建新的销售项，你可以发出以下请求：

```
POST /sales-item-service/sales-items HTTP/1.1
Content-Type: application/json

{
    "name": "Sample sales item",
    "price": 20
}
```

服务器将使用 HTTP 状态码 201 *Created* 响应。服务器可以在创建时向资源添加字段。通常，服务器会向创建的资源添加 *id* 属性，但也可以添加其他属性。服务器将在 HTTP 响应体中以 JSON 形式响应创建的资源。以下是销售项创建请求的响应。你可以注意到服务器向资源添加了 *id* 属性。通常添加的其他属性包括创建时间戳和资源的版本（新创建资源的版本应为 1）。

HTTP/1.1 201 Created
Content-Type: application/json

{
    "id": 1,
    "name": "示例销售商品",
    "price": 20
}

如果提供的待创建资源在某种程度上无效，服务器应返回HTTP状态码400 *Bad Request*，并在响应体中解释错误。响应体应为JSON格式，包含有关错误的信息，如错误代码和消息。为了使API错误响应保持一致，如果可能，应在软件系统的所有API中使用相同的错误响应格式。以下是错误响应的示例：

```
{
    "statusCode": 500,
    "statusText": "Internal Server Error",
    "errorCode": "IAMError",
    "errorMessage": "Unable to connect to the Identity and Access Management service",
    "errorDescription": "Describe the error in more detail here, if relevant/needed...",
    "stackTrace": "Call stack trace here...."
}
```

注意！在上面的示例中，`stackTrace`属性在生产环境中默认不应包含，因为它可能向潜在攻击者泄露内部实现细节。仅在开发和其他内部环境中使用它，并且如果绝对需要，仅在生产环境中启用它一小段时间以进行调试。`errorCode`属性对于更新错误计数器指标很有用。将其用作错误计数器的标签。在接下来的*DevSecOps原则*章节中，将有更多关于指标的讨论。

如果创建的资源很大，则无需将资源返回给调用方并浪费网络带宽。你可以只返回添加的属性。例如，如果服务器只添加了`id`属性，则可以在响应体中只返回`id`：

```
HTTP/1.1 201 Created
Content-Type: application/json

{
    "id": 1
}
```

请求发送者可以通过将发送的资源对象与接收到的资源对象合并来构造创建的资源。

> 确保不创建重复资源。

当客户端尝试创建新资源时，资源创建请求可能会失败，导致资源在服务器上成功创建，但客户端未及时收到响应，请求因超时而失败。从服务器的角度来看，请求是成功的，但从客户端的角度来看，请求的状态是不确定的。客户端当然需要重新发出超时的请求，如果成功，相同的资源将在服务器端创建两次，这可能是不希望发生的。

假设一个资源包含一个唯一属性，例如用户的电子邮件。在这种情况下，如果服务器正确实现（=唯一属性在数据库表定义中被标记为唯一列），则不可能创建重复资源。在许多情况下，资源中不存在这样的唯一字段。在这些情况下，客户端可以提供一个通用唯一标识符（UUID），例如命名为`creationUuid`。服务器的作用是检查是否已创建具有相同`creationUuid`的资源，并阻止创建重复资源。作为UUID方法的替代方案，如果服务器在短时间内从同一客户端收到两个相同的资源，服务器可以向客户端请求验证是否打算创建两个相同的资源。

## 8.1.2.2：读取资源

使用REST API读取资源是通过向API的资源端点发送HTTP GET请求来完成的。要读取所有销售商品，你可以发出以下请求：

```
GET /sales-item-service/sales-items HTTP/1.1
```

服务器将返回HTTP状态码200 OK。服务器将在响应体中返回资源的JSON数组，如果未找到资源，则返回空数组。以下是获取销售商品的请求的示例响应：

```
HTTP/1.1 200 OK
Content-Type: application/json
[
  {
    "id": 1,
    "name": "示例销售商品",
    "price": 20
  }
]
```

要通过其id读取单个资源，请将资源id添加到请求URL路径中，如下所示：

```
GET /sales-item-service/sales-items/<id> HTTP/1.1
```

可以发出以下请求来读取标识为id 1的销售商品：

```
GET /sales-item-service/sales-items/1 HTTP/1.1
```

对上述请求的响应将包含单个资源：

```
HTTP/1.1 200 OK
Content-Type: application/json

{
    "id": 1,
    "name": "示例销售商品",
    "price": 20
}
```

如果未找到请求的资源，服务器将返回HTTP状态码404 *Not Found*。

你可以在URL查询字符串中定义参数来过滤要读取的资源。查询字符串是URL的最后一部分，通过问号（?）字符与URL路径分隔。查询字符串可以包含一个或多个由与号（&）字符分隔的参数。每个查询字符串参数具有以下格式：`<查询参数名>=<查询参数值>`。以下是包含两个查询参数的示例请求：*name-contains*和*price-greater-than*。

```
GET /sales-item-service/sales-items?name-contains=Sample&price-greater-than=10 HTTP/1.1
```

上述请求获取名称包含字符串*Sample*且价格大于10的销售商品。

要定义过滤器，你可以使用以下格式指定查询参数：`<字段名>[-<条件>]=<值>`，例如：

- price=10
- price-not-equal=10
- price-less-than=10
- price-less-than-equal=10
- price-greater-than=10
- price-greater-than-equal=10
- name-starts-with=Sample
- name-ends-with=item
- name-contains=Sample
- createdTimestamp-before=2022-08-02T05:18:00Z
- createdTimestamp-after=2022-08-02T05:18:00Z
- images.url-starts-with=https

请记住，在实现服务器端并将上述参数添加到SQL查询时，必须使用参数化SQL查询以防止SQL注入攻击，因为攻击者可以在查询参数中发送恶意数据。

其他操作，如查询资源的投影、排序和分页，也可以在URL中使用查询参数来定义：

```
GET /sales-item-service/sales-items?fields=id,name&sort-by=price:asc&offset=0&limit=100 HTTP/1.1
```

上述请求获取按价格排序（升序）的销售商品。获取的销售商品数量限制为100。销售商品从偏移量0开始获取，响应仅包含每个销售商品的*id*和*name*字段。

*fields*参数定义响应中返回哪些资源字段（属性）。所需的字段定义为以逗号分隔的字段名列表。如果要定义子资源字段，可以使用点表示法定义，例如：

```
fields=id,name,images.url
```

*sort-by*查询参数使用以下格式定义排序：

```
sort-by=<字段名>:asc|desc,[<字段名>:asc|desc]
```

例如：

```
sort-by=price:asc,images.rank:asc
```

在上面的示例中，资源首先按价格升序排序，其次按图像的排名排序。

*limit*和*offset*参数用于分页。*limit*查询参数定义可返回的最大资源数。*offset*查询参数指定返回资源的偏移量。你也可以通过以<子资源>:<数量>的形式给出*offset*和*limit*来对子资源进行分页。以下是使用分页查询参数的示例：

```
offset=0&limit=50,images:5
```

上述查询参数定义获取第一页的50个销售商品，每个销售商品包含该销售商品的前五张图像。你可以使用*page*和*pageSize*参数代替*offset*和*limit*参数。*page*参数定义页码，*pageSize*定义每页应包含多少资源。

在实现服务器端并将查询参数中的数据添加到SQL查询时，请记住验证用户提供的数据以防止SQL注入攻击。例如，*fields*查询参数中的字段名应仅包含SQL列名中允许的字符。同样，*sort-by*参数的值应仅包含SQL列名中允许的字符以及单词*asc*和*desc*。最后，*offset*和*limit*（或*page*和*pageSize*）参数的值必须是整数。你还应该根据允许的最大值验证*limit/pageSize*参数，因为你不应该允许最终用户一次获取太多资源。

一些HTTP服务器会记录HTTP GET请求的URL。因此，不建议在URL中放置敏感信息。敏感信息应放入请求体中。此外，浏览器可能对URL的最大长度有限制。如果你有一个长达数千个字符的查询字符串，你应该在请求体中给出参数。你不应该在HTTP GET请求中放置请求体。你应该做的是使用HTTP POST方法发出请求，例如：

## 8.1.2.3：更新资源

通过 REST API 更新资源，需要向 API 的资源端点发送 HTTP PUT 或 PATCH 请求。要更新标识为 id 1 的销售商品，可以发出以下请求：

```
PUT /sales-item-service/sales-items/1 HTTP/1.1
Content-Type: application/json

{
    "name": "Sample sales item name modified",
    "price": 30
}
```

服务器将响应无内容：

```
HTTP/1.1 204 No Content
```

服务器也可以在响应中返回更新后的资源，特别是当资源被服务器以某种方式修改时，这可能是必需的。如果请求的资源未找到，服务器将响应 HTTP 状态码 404 *Not Found*。

如果请求中提供的资源无效，服务器应响应 HTTP 状态码 400 *Bad Request*。响应体应包含一个 JSON 格式的错误对象。

HTTP PUT 请求将用提供的资源替换现有资源。你也可以使用 HTTP PATCH 方法部分修改现有资源：

```
PATCH /sales-item-service/sales-items/1 HTTP/1.1
Content-Type: application/json

{
    "price": 30
}
```

上述请求仅修改标识为 id 1 的销售商品的价格属性。
你可以通过在 URL 中指定过滤器来进行批量更新，例如：

```
PATCH /sales-item-service/sales-items?price-less-than=10 HTTP/1.1
Content-Type: application/json

{
    "price": 10
}
```

上述示例将更新当前价格小于十的每个资源的价格属性。
在服务器端，API 端点可以使用以下参数化 SQL 语句来实现更新功能：

```
UPDATE salesitems SET price = %s WHERE price < %s
```

上述 SQL 语句只会修改价格列，其他列将保持不变。

> 需要时使用资源版本控制。

当你从服务器获取一个资源然后尝试更新它时，有可能在你获取之后、尝试更新之前，其他人已经更新了它。如果你不关心其他客户端的更新，有时这可能是可以接受的。但有时，你希望确保在你更新资源之前没有其他人更新过它。在这种情况下，你应该使用资源版本控制。在资源版本控制中，资源中有一个版本字段，每次更新时递增一。如果你获取了一个版本为 x 的资源，然后尝试将资源更新回服务器并提供相同的版本 x，但其他人已将资源更新到版本 x + 1，你的更新将因版本不匹配（x != x + 1）而失败。服务器应响应 HTTP 状态码 409 Conflict。收到冲突响应后，你可以从服务器获取资源的最新版本，并根据资源的新状态决定你的更新是否仍然相关。

服务器应将资源版本值分配给 HTTP 响应头 ETag。客户端可以通过将接收到的 ETag 值分配给请求头 If-None-Match，在条件 HTTP GET 请求中使用该值。现在，服务器将仅在资源有更新版本时才返回请求的资源。否则，服务器将返回无内容并附带 HTTP 状态码 304 Not Modified。这样做的优点是无需将未修改的资源从服务器传输到客户端。当资源很大或服务器与客户端之间的连接较慢时，这尤其有益。

## 8.1.2.4：删除资源

通过 REST API 删除资源，需要向 API 的资源端点发送 HTTP DELETE 请求。要删除标识为 id 1 的销售商品，可以发出以下请求：

```
DELETE /sales-item-service/sales-items/1 HTTP/1.1
```

服务器将响应无内容：

```
HTTP/1.1 204 No Content
```

如果请求删除的资源已被删除，API 仍应响应 HTTP 状态码 204 No Content，表示操作成功。它不应响应 HTTP 状态码 404 Not Found。

要删除所有销售商品，可以发出以下请求：

```
DELETE /sales-item-service/sales-items HTTP/1.1
```

要使用过滤器删除销售商品，可以发出以下类型的请求：

```
DELETE /sales-item-service/sales-items?price-less-than=10 HTTP/1.1
```

在服务器端，API 端点处理程序可以使用以下参数化 SQL 查询来实现删除功能：

```
DELETE FROM salesitems WHERE price < %s
```

## 8.1.2.5：在资源上执行非 CRUD 操作

有时你需要对资源执行非 CRUD 操作。在这些情况下，你可以发出 HTTP POST 请求，并在 URL 中资源名称后放置操作名称（一个动词）。以下示例将对账户资源执行存款操作：

```
POST /account-balance-service/accounts/12345678912/deposit HTTP/1.1
Content-Type: application/json

{
    "amountInCents": 2510
}
```

类似地，你可以执行取款操作：

```
POST /account-balance-service/accounts/12345678912/withdraw HTTP/1.1
Content-Type: application/json

{
    "amountInCents": 2510
}
```

## 8.1.2.6：资源组合

一个资源可以由其他资源组合而成。有两种实现资源组合的方式：嵌套资源或链接资源。让我们先看一个嵌套资源的例子。一个销售商品资源可以包含一个或多个图像资源。我们不希望在客户端请求销售商品时返回所有图像，因为图像可能很大，且客户端不一定需要使用它们。我们可以返回一组小的缩略图。为了客户端查看销售商品的图像，我们可以为图像资源实现一个 API 端点。要获取特定销售商品的图像，可以发出以下 API 调用：

```
GET /sales-item-service/sales-items/<id>/images HTTP/1.1
```

你也可以为销售商品添加新图像：

```
POST /sales-item-service/sales-items/<id>/images HTTP/1.1
```

此外，其他 CRUD 操作也可以提供：

```
PUT /sales-item-service/sales-items/<salesItemId>/images/<imageId> HTTP/1.1
```

```
DELETE /sales-item-service/sales-items/<salesItemId>/images/<imageId> HTTP/1.1
```

这种方法的问题在于 *sales-item-service* 的规模会增长，如果你将来需要添加更多嵌套资源，规模会增长得更大，使得微服务过于复杂并负责过多的事情。

一个更好的替代方案是为嵌套资源创建一个单独的微服务。这将能够利用最合适的技术来实现微服务。关于销售商品图像，*sales-item-image-service* 可以采用云对象存储来存储图像，而 *sales-item-service* 可以利用标准关系数据库来存储销售商品。

当为销售商品图像拥有单独的微服务时，你可以通过发出以下请求来获取销售商品的图像：

```
GET /sales-item-image-service/sales-item-images?salesItemId=<salesItemId> HTTP/1.1
```

你可以注意到，*sales-item-service* 和 *sales-item-image-service* 现在通过 *salesItemId* 链接在一起。

## 8.1.2.7：HTTP 状态码

使用以下 HTTP 状态码：

| HTTP 状态码 | 何时使用 |
|---|---|
| 200 OK | 使用 GET 方法的 API 操作成功 |
| 201 Created | 使用 POST 方法的 API 操作成功 |
| 202 Accepted | 请求已被接受处理，但处理尚未完成。这可以用作异步操作请求的响应状态码。例如，POST 请求可以收到带有此状态码和指向最终将创建的资源的链接的响应。在异步创建完成之前，该链接将返回 404 Not Found。 |
| 204 No Content | 使用 PUT、PATCH 或 DELETE 方法的 API 操作成功 |
| 400 Bad Request | API 操作中的客户端错误，例如，客户端提供了无效数据 |
| 401 Unauthorized | 客户端在请求中未提供授权头 |
| 403 Forbidden | 客户端在请求中提供了授权头，但用户未被授权执行 API 操作 |
| 404 Not Found | 使用 GET、PUT 或 PATCH 方法请求不存在的资源时 |
| 405 Method Not Allowed | 当客户端尝试对 API 端点使用错误的方法时 |
| 406 Not Acceptable | 当客户端请求服务器无法生成的格式的响应时，例如，请求 XML，但服务器仅提供 JSON |
| 409 Conflict | 当客户端尝试更新在客户端获取资源后已被更新的资源时 |
| 413 Payload Too Large | 当客户端尝试在请求中提供过大的负载时。为防止 DoS 攻击，不要接受客户端任意大的负载 |
| 429 Too Many Requests | 在你的 API 网关中配置速率限制，当请求速率超出限制时发送此状态码 |
| 500 Internal Server Error | 当发生服务器错误时，例如，抛出异常 |
| 503 Service Unavailable | 服务器与依赖服务的连接失败。这表明客户端应稍后重试请求，因为此问题通常是暂时的。 |

## 8.1.2.8: HATEOAS 与 HAL

超媒体作为应用状态引擎（HATEOAS）可用于为请求的资源添加超媒体/元数据。超文本应用语言（HAL）是一种定义超媒体（元数据）的约定，例如指向外部资源的链接。以下是一个示例响应，该响应针对一个获取 id 为 1234 的销售项目的请求。该销售项目由 id 为 5678 的用户拥有。响应提供了指向所获取资源本身的链接，以及另一个用于获取拥有该销售项目的用户（账户）的链接：

```json
{
  "_links": {
    "self": {
      "href": "https://.../sales-item-service/sales-items/1234"
    },
    "userAccount": {
      "href": "https://.../user-account-service/user-accounts/5678"
    }
  },
  "id": 1234,
  "name": "Sales item xyz",
  "userAccountId": 5678
}
```

当使用 HAL 获取第 3 页的销售项目集合时，我们可以得到如下类型的响应：

```json
{
  "_links": {
    "self": {
      "href": "https://.../sales-items?page=3"
    },
    "first": {
      "href": "https://.../sales-items"
    },
    "prev": {
      "href": "https://.../sales-items?page=2"
    },
    "next": {
      "href": "https://.../sales-items?page=4"
    }
  },
  "count": 25,
  "total": 1500,
  "_embedded": {
    "salesItems": [
      {
        "_links": {
          "self": {
            "href": "https://.../sales-items/123"
          }
        },
        "id": 123,
        "name": "Sales item 123"
      },
      ...
    ]
  }
}
```

```json
{
    "_links": {
        "self": {
            "href": "https://.../sales-items/124"
        }
    },
    "id": 124,
    "name": "Sales item 124"
},
...
]
}
```

## 8.1.2.9: 版本控制

你可以使用版本化的 URL 路径段来引入 API 的新版本。以下是 API 版本 2 的示例端点：

```
GET /sales-item-service/v2/sales-items HTTP/1.1
...
```

## 8.1.2.10: 文档

如果你需要为 REST API 编写文档或提供交互式在线文档，有两种方式：

1.  规范优先：为 API 创建规范，然后根据规范生成代码
2.  代码优先：实现 API，然后根据代码生成 API 规范

像 Swagger 和 Postman 这样的工具可以根据 API 规范为你的 API 生成静态和交互式文档。你应该使用 OpenAPI 规范² 来定义 API。

使用第一种方式时，你可以使用 OpenAPI 规范语言来定义你的 API。你可以使用 SwaggerHub 或 Postman 等工具来编写 API 规范。Swagger 提供了多种语言的代码生成工具。代码生成器根据 OpenAPI 规范生成代码。代码生成器不仅能够生成服务器端代码，还能生成客户端代码。

使用第二种方式时，你可以使用特定于 Web 框架的方式从 API 实现构建 API 规范。例如，使用 FastAPI，你可以获得自动生成的 API 规范。默认情况下，OpenAPI 模式以 JSON 格式在以下端点提供：`/openapi.json`。此 URL 是可配置的。FastAPI 还在以下 URL 提供 Swagger UI 交互式文档和客户端：`/docs`。此 URL 也是可自定义的。在 `/redoc` 处也有 ReDoc 交互式文档和客户端可用，该 URL 同样可配置。例如，要将 OpenAPI 模式设置为在 `/my-service/v1/openapi.json` 提供，将 Swagger UI 设置为在 `/my-service/v1/docs` 提供，并禁用 ReDoc：

> ²https://swagger.io/specification/

```python
from fastapi import FastAPI

app = FastAPI(
    openapi_url='/my-service/v1/openapi.json',
    docs_url='/my-service/v1/docs',
    redoc_url=None
)
```

我更喜欢使用第二种方式，即先编写代码。我不喜欢同时处理自动生成的代码和手写代码，而且许多 Web 框架都提供 OpenAPI 模式和交互式文档（如 Swagger UI）的自动生成。

## 8.1.2.11: 实现示例

让我们使用 FastAPI 来实现 *sales-item-service* API 端点，用于对销售项目进行 CRUD 操作。我们使用前面介绍的*清洁微服务设计原则*，并在控制器类中编写 API 端点：

图 8.1. controllers/RestSalesItemController.py

```python
from dependency_injector.wiring import Provide
from fastapi import APIRouter, Request

from ..decorators.audit_log import audit_log
from ..decorators.increment_counter import increment_counter
from ..dtos.InputSalesItem import InputSalesItem
from ..dtos.OutputSalesItem import OutputSalesItem
from ..service.SalesItemService import SalesItemService


class RestSalesItemController:
    # 销售项目服务通过依赖注入提供
    __sales_item_service: SalesItemService = Provide['sales_item_service']

    def __init__(self):
        self.__router = APIRouter()

        self.__router.add_api_route(
            '/sales-items/',
            self.create_sales_item,
            methods=['POST'],
            status_code=201,
            response_model=OutputSalesItem,
        )

        self.__router.add_api_route(
            '/sales-items/',
            self.get_sales_items,
            methods=['GET'],
            response_model=list[OutputSalesItem],
        )

        self.__router.add_api_route(
            '/sales-items/{id_}',
            self.get_sales_item,
            methods=['GET'],
            response_model=OutputSalesItem,
        )

        self.__router.add_api_route(
            '/sales-items/{id_}',
            self.update_sales_item,
            methods=['PUT'],
            status_code=204,
            response_model=None,
        )

        self.__router.add_api_route(
            '/sales-items/{id_}',
            self.delete_sales_item,
            methods=['DELETE'],
            status_code=204,
            response_model=None,
        )

    @property
    def router(self):
        return self.__router

    def create_sales_item(
        self, input_sales_item: InputSalesItem
    ) -> OutputSalesItem:
        return self.__sales_item_service.create_sales_item(
            input_sales_item
        )

    def get_sales_items(self) -> list[OutputSalesItem]:
        return self.__sales_item_service.get_sales_items()

    def get_sales_item(self, id_: str) -> OutputSalesItem:
        return self.__sales_item_service.get_sales_item(id_)

    def update_sales_item(
        self, id_: str, sales_item_update: InputSalesItem
    ) -> None:
        return self.__sales_item_service.update_sales_item(
            id_, sales_item_update
        )

    def delete_sales_item(self, id_: str, request: Request) -> None:
        return self.__sales_item_service.delete_sales_item(id_)
```

上述控制器尚未达到生产质量。必须添加以下内容：

-   可能的审计日志记录
-   可观测性，即更新指标
-   授权

以上所有内容都可以并且很可能应该使用装饰器来实现，例如：

```python
@allow_for_user_roles(['admin'], authorizer)
@audit_log
@increment_counter(Counters.request_attempts)
def create_sales_item(
    self,
    input_sales_item: InputSalesItem,
    request: Request
) -> OutputSalesItem:
    return self.__sales_item_service.create_sales_item(
        input_sales_item
    )
```

授权装饰器 `@allow_for_user_roles(['admin'], authorizer)` 与我们之前章节讨论的相同。`@audit_log` 可以这样实现：

```python
from functools import wraps

def audit_log(handle_request):
    @wraps(handle_request)
    def wrapped_handle_request(*args, **kwargs):
        method = kwargs['request'].method
        url = kwargs['request'].url
        client_host = kwargs['request'].client.host
        # 下面打印的文本应写入审计日志
        print(f'API endpoint: {method} {url} accessed from: {client_host}')
        return handle_request(*args, **kwargs)

    return wrapped_handle_request
```

`@increment_counter` 装饰器可以这样实现：

```python
from functools import wraps

def increment_counter(counter):
    def decorate(handle_request):
        @wraps(handle_request)
        def wrapped_handle_request(*args, **kwargs):
            method = kwargs['request'].method
            url = kwargs['request'].url
            # 使用 'api_endpoint' 标签将计数器加一
            counter.increment(1, {'api_endpoint': f'{method} {url}'})
            return handle_request(*args, **kwargs)

        return wrapped_handle_request

    return decorate
```

DTO 使用 pydantic 定义：

## 图 8.2. dtos/SalesItemImage.py

```python
from pydantic import BaseModel, HttpUrl, PositiveInt

from ..entities.SalesItemImage import (
    SalesItemImage as SalesItemImageEntity,
)


class SalesItemImage(BaseModel):
    id: PositiveInt
    rank: PositiveInt
    url: HttpUrl

    class Config:
        orm_mode = True

    class Meta:
        orm_model = SalesItemImageEntity
```

## 图 8.3. dtos/InputSalesItem.py

```python
from pydantic import BaseModel, Field

from .SalesItemImage import SalesItemImage


class InputSalesItem(BaseModel):
    name: str = Field(max_length=256)
    # We accept negative prices for sales items that act
    # as discount items
    priceInCents: int
    images: list[SalesItemImage] = Field(max_items=25)

    class Config:
        orm_mode = True
```

## 图 8.4. dtos/OutputSalesItem.py

```python
from pydantic import BaseModel, Field, PositiveInt

from .SalesItemImage import SalesItemImage


class OutputSalesItem(BaseModel):
    id: str
    createdAtTimestampInMs: PositiveInt
    name: str = Field(max_length=256)
    priceInCents: int
    images: list[SalesItemImage] = Field(max_items=25)

    class Config:
        orm_mode = True
```

请注意，我们在所有类中都对每个属性进行了验证。这很重要，因为涉及安全性。例如，字符串和列表属性应具有最大长度验证器，以防止可能的拒绝服务攻击。请记住也要为输出 DTO 添加验证。这很重要，因为涉及安全性。输出验证可以防止试图返回形状无效数据的注入攻击。FastAPI 中的输出验证也用于 API 模式的自动文档和自动生成客户端代码。

`SalesItemService` 协议如下所示：

## 图 8.5. service/SalesItemService.py

```python
from typing import Protocol

from ..dtos.InputSalesItem import InputSalesItem
from ..dtos.OutputSalesItem import OutputSalesItem

class SalesItemService(Protocol):
    def create_sales_item(
        self, input_sales_item: InputSalesItem
    ) -> OutputSalesItem:
        pass

    def get_sales_items(self) -> list[OutputSalesItem]:
        pass

    def get_sales_item(self, id_: str) -> OutputSalesItem:
        pass

    def update_sales_item(
        self, id_: str, sales_item_update: InputSalesItem
    ) -> None:
        pass

    def delete_sales_item(self, id_: str) -> None:
        pass
```

接下来，我们可以实现上述协议：

## 图 8.6. service/SalesItemServiceImpl.py

```python
from dependency_injector.wiring import Provide

from ..dtos.InputSalesItem import InputSalesItem
from ..dtos.OutputSalesItem import OutputSalesItem
from ..errors.EntityNotFoundError import EntityNotFoundError
from ..repositories.SalesItemRepository import SalesItemRepository
from ..service.SalesItemService import SalesItemService

class SalesItemServiceImpl(SalesItemService):
    # Sales item repository is provided by DI
    __sales_item_repository: SalesItemRepository = Provide[
        'sales_item_repository'
    ]

    def create_sales_item(
        self, input_sales_item: InputSalesItem
    ) -> OutputSalesItem:
        sales_item = self.__sales_item_repository.save(input_sales_item)
        return OutputSalesItem.from_orm(sales_item)

    def get_sales_items(self) -> list[OutputSalesItem]:
        return [
            OutputSalesItem.from_orm(sales_item)
            for sales_item in self.__sales_item_repository.find_all()
        ]

    def get_sales_item(self, id_: str) -> OutputSalesItem:
        sales_item = self.__sales_item_repository.find(id_)
        if sales_item is None:
            raise EntityNotFoundError('Sales item', id_)
        return OutputSalesItem.from_orm(sales_item)

    def update_sales_item(
        self, id_: str, sales_item_update: InputSalesItem
    ) -> None:
        return self.__sales_item_repository.update(id_, sales_item_update)

    def delete_sales_item(self, id_: str) -> None:
        return self.__sales_item_repository.delete(id_)
```

以下是 `SalesItemRepository` 协议的定义：

## 图 8.7. repositories/SalesItemRepository.py

```python
from typing import Protocol

from ..dtos.InputSalesItem import InputSalesItem
from ..entities.SalesItem import SalesItem

class SalesItemRepository(Protocol):
    def save(self, input_sales_item: InputSalesItem) -> SalesItem:
        pass

    def find_all(self) -> list[SalesItem]:
        pass

    def find(self, id_: str) -> SalesItem | None:
        pass

    def update(self, id_: str, sales_item_update: InputSalesItem) -> None:
        pass

    def delete(self, id_: str) -> None:
        pass
```

`SalesItemRepository` 的实现将在下一章中介绍，我们将重点讨论数据库原理。下一章将提供仓库的三种不同实现：对象关系映射（ORM）、参数化 SQL 查询和 MongoDB。

在错误处理方面，我们依赖于 FastAPI Web 框架提供的 `except` 块。我们可以在业务逻辑中抛出 FastAPI `HTTPException` 类型的错误，但那样会将我们的 Web 框架与业务逻辑耦合，这是不可取的。请记住 *clean microservice design principle* 中的原则，依赖关系仅从 Web 框架（控制器）指向业务逻辑，而不是相反。如果我们使用特定于 Web 框架的错误类在业务逻辑中，并且我们希望将微服务迁移到不同的 Web 框架，我们将不得不重构整个业务逻辑中关于错误抛出的部分。

我们应该做的是为我们的微服务引入一个错误基类，并为 FastAPI 提供一个自定义错误处理器。自定义错误处理器将我们业务逻辑特定的错误转换为 HTTP 响应。微服务可能引发的所有错误都应该派生自基错误类。`ApiError` 类是一个通用的 API 基错误类。

## 图 8.8. errors/ApiError.py

```python
from typing import Final

class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        status_text: str,
        message: str,
        code: str | None = None,
        description: str | None = None,
        cause: Exception | None = None,
    ):
        self.__status_code: Final = status_code
        self.__status_text: Final = status_text
        self.__message: Final = message
        self.__code: Final = code
        self.__description: Final = description
        self.__cause: Final = cause

    @property
    def status_code(self) -> int:
        return self.__status_code

    @property
    def status_text(self) -> str:
        return self.__status_text

    @property
    def message(self) -> str:
        return self.__message

    @property
    def cause(self) -> Exception | None:
        return self.__cause

    @property
    def code(self) -> str | None:
        return self.__code

    @property
    def description(self) -> str | None:
        return self.__description

    def __str__(self) -> str:
        return self.__message
```

`code` 属性也可以命名为 `type`。该属性背后的理念是传达这是哪种类型的错误。此属性可以在服务器端用作失败指标的标签，在客户端可以实现对特定类型错误的特殊处理。如果您愿意，您甚至可以向上述类添加另一个属性，即 `recovery_action`。这是一个可选属性，包含有关可操作错误的恢复步骤的信息。例如，数据库连接错误可能具有 `recovery_action` 属性值：“请稍后重试。如果问题持续存在，请联系技术支持：”。

以下是 *sales-item-service* 的基错误类：

## 图 8.9. errors/SalesItemServiceError.py

```python
from ..errors.ApiError import ApiError

class SalesItemServiceError(ApiError):
    pass
```

然后让我们定义一个 API 使用的错误类：

## 图 8.10. errors/EntityNotFoundError.py

```python
from .SalesItemServiceError import SalesItemServiceError

class EntityNotFoundError(SalesItemServiceError):
    def __init__(self, entity_name: str, entity_id: str):
        super().__init__(
            404,
            'Not Found',
            f'{entity_name} with id {entity_id} not found',
            'EntityNotFound',
        )
```

让我们为我们的 API 实现一个自定义错误处理器：

```python
# Imports ...

app = FastAPI()

@app.exception_handler(SalesItemServiceError)
def handle_sales_item_service_error(
    request: Request, error: SalesItemServiceError
):
    # Log error.cause

    # Increment 'request_failures' counter by one
    # with labels:
    # api_endpoint=f'{request.method} {request.url}'
```

### 8.1.3：GraphQL API 设计

> 将 API 端点划分为查询和变更。与 REST 相比，REST GET 请求对应 GraphQL 查询，而 REST POST/PUT/PATCH/DELETE 请求对应 GraphQL 变更。使用 GraphQL，你可以为查询和变更赋予描述性的名称。

让我们创建一个 GraphQL 模式，为 *sales-item-service* 定义所需的类型和 API 端点。我们将在示例之后讨论该模式的细节以及模式语言的一般内容。

```python
# status_code=error.status_code
# error_code=error.code

return JSONResponse(
    status_code=error.status_code,
    content={
        'statusCode': error.status_code,
        'statusText': error.status_text,
        'errorCode': error.code,
        'errorMessage': error.message,
        'errorDescription': error.description,
        # get_stack_trace returns stack trace only
        # when environment is not production
        # otherwise it returns None
        'stackTrace': get_stack_trace(error.cause),
    },
)
```

现在，如果业务逻辑抛出以下错误：

```python
raise EntityNotFoundError('Sales item', '10')
```

在生产环境中应预期以下 API 响应（注意在生产环境中 stackTrace 为 null）：

```
HTTP/1.1 404 Not Found
Content-Type: application/json

{
    "statusCode": 404,
    "statusText": "Not Found",
    "errorCode": "EntityNotFound",
    "errorMessage": "Sales item with id 10 not found",
    "errorDescription": null,
    "stackTrace": null
}
```

你还应该为验证消息和其他可能的错误添加特定的错误处理器：

```python
@app.exception_handler(RequestValidationError)
def handle_request_validation_error(
    request: Request, error: RequestValidationError
):
    # Audit log

    # Increment 'request_failures' counter by one
    # with labels:
    # api_endpoint=f'{request.method} {request.url}'
    # status_code=400
    # error_code='RequestValidationError'

    return JSONResponse(
        status_code=400,
        content={
            'statusCode': 400,
            'statusText': 'Bad Request',
            'errorCode': 'RequestValidationError',
            'errorMessage': 'Request validation failed',
            'errorDescription': str(error),
            'stackTrace': None,
        },
    )
```

```python
@app.exception_handler(Exception)
def handle_unspecified_error(request: Request, error: Exception):

    # Increment 'request_failures' counter by one
    # with labels:
    # api_endpoint=f'{request.method} {request.url}'
    # status_code=500
    # error_code='UnspecifiedError'

    return JSONResponse(
        status_code=500,
        content={
            'statusCode': 500,
            'statusText': 'Internal Server Error',
            'errorCode': 'UnspecifiedError',
            'errorMessage': 'Unspecified internal error',
            'errorDescription': str(error),
            'stackTrace': get_stack_trace(error),
        },
    )
```

API 服务源代码文件的其余部分如下所示：

### 图 8.11. DiContainer.py

```python
from dependency_injector import containers, providers

from .controllers.RestSalesItemController import RestSalesItemController
from .controllers.StrawberryGraphQlSalesItemController import (
    StrawberryGraphQlSalesItemController,
)
from .repositories.MongoDbSalesItemRepository import (
    MongoDbSalesItemRepository,
)
from .repositories.OrmSalesItemRepository import OrmSalesItemRepository
from .repositories.ParamSqlSalesItemRepository import (
    ParamSqlSalesItemRepository,
)
from .service.SalesItemServiceImpl import SalesItemServiceImpl

class DiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            '.service.SalesItemServiceImpl',
            '.controllers.RestSalesItemController',
            '.controllers.AriadneGraphQlSalesItemController',
            '.controllers.StrawberryGraphQlSalesItemController',
            '.controllers.GrpcSalesItemController',
            '.repositories.OrmSalesItemRepository',
            '.repositories.ParamSqlSalesItemRepository',
            '.repositories.MongoDbSalesItemRepository',
        ]
    )

    sales_item_service = providers.Singleton(SalesItemServiceImpl)
    sales_item_repository = providers.Singleton(
        ParamSqlSalesItemRepository
    )
    order_controller = providers.Singleton(RestSalesItemController)
```

### 图 8.12. utils.py

```python
import os
import traceback

from pydantic import BaseModel


def is_pydantic(object: object):
    return type(object).__class__.__name__ == 'ModelMetaclass'


def to_entity_dict(dto: BaseModel):
    entity_dict = dict(dto)

    for key, value in entity_dict.items():
        try:
            if (
                isinstance(value, list)
                and len(value)
                and is_pydantic(value[0])
            ):
                entity_dict[key] = [
                    item.Meta.orm_model(**to_entity_dict(item))
                    for item in value
                ]
            elif is_pydantic(value):
                entity_dict[key] = value.Meta.orm_model(
                    **to_entity_dict(value)
                )
        except AttributeError:
            raise AttributeError(
                f'Found nested Pydantic model in {dto.__class__} but Meta.orm_model was not specified.'
            )

    return entity_dict


def get_stack_trace(error: Exception | None):
    return (
        repr(traceback.format_exception(error))
        if error and os.environ.get('ENV') != 'production'
        else None
    )
```

### 图 8.13. app.py

```python
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .DiContainer import DiContainer
from .errors.SalesItemServiceError import SalesItemServiceError
from .utils import get_stack_trace

# Remove the below setting of the env variable for production code!
# mysql+pymysql://root:password@localhost:3306/salesitemservice
# mongodb://localhost:27017/salesitemservice
os.environ[
    'DATABASE_URL'
] = 'mysql+pymysql://root:password@localhost:3306/salesitemservice'


di_container = DiContainer()
app = FastAPI()


@app.exception_handler(SalesItemServiceError)
def handle_sales_item_service_error(
    request: Request, error: SalesItemServiceError
):
    # Log error.cause

    # Increment 'request_failures' counter by one
    # with labels:
    # api_endpoint=f'{request.method} {request.url}'
    # status_code=error.status_code
    # error_code=error.code

    return JSONResponse(
        status_code=error.status_code,
        content={
            'statusCode': error.status_code,
            'statusText': error.status_text,
            'errorCode': error.code,
            'errorMessage': error.message,
            'errorDescription': error.description,
            # get_stack_trace returns stack trace only
            # when environment is not production
            # otherwise it returns None
            'stackTrace': get_stack_trace(error.cause),
        },
    )


@app.exception_handler(RequestValidationError)
def handle_request_validation_error(
    request: Request, error: RequestValidationError
):
    # Audit log

    # Increment 'request_failures' counter by one
    # with labels:
    # api_endpoint=f'{request.method} {request.url}'
    # status_code=400
    # error_code='RequestValidationError'

    return JSONResponse(
        status_code=400,
        content={
            'statusCode': 400,
            'statusText': 'Bad Request',
            'errorCode': 'RequestValidationError',
            'errorMessage': 'Request validation failed',
            'errorDescription': str(error),
            'stackTrace': None,
        },
    )


@app.exception_handler(Exception)
def handle_unspecified_error(request: Request, error: Exception):

    # Increment 'request_failures' counter by one
    # with labels:
    # api_endpoint=f'{request.method} {request.url}'
    # status_code=500
    # error_code='UnspecifiedError'

    return JSONResponse(
        status_code=500,
        content={
            'statusCode': 500,
            'statusText': 'Internal Server Error',
            'errorCode': 'UnspecifiedError',
            'errorMessage': 'Unspecified internal error',
            'errorDescription': str(error),
            'stackTrace': get_stack_trace(error),
        },
    )


order_controller = di_container.order_controller()
app.include_router(order_controller.router)
```

## API 设计原则

```graphql
type Image {
  id: Int!
  rank: Int!
  url: String!
}

type SalesItem {
  id: ID!
  createdAtTimestampInMs: String!
  name: String!
  priceInCents: Int!
  images(
    sortByField: String = "rank",
    sortDirection: SortDirection = ASC,
    offset: Int = 0,
    limit: Int = 5
  ): [Image!]!
}

input InputImage {
  id: Int!
  rank: Int!
  url: String!
}

input InputSalesItem {
  name: String!
  priceInCents: Int!
  images: [InputImage!]!
}

enum SortDirection {
  ASC
  DESC
}

type IdResponse {
  id: ID!
}

type Query {
  salesItems(
    sortByField: String = "createdAtTimestamp",
    sortDirection: SortDirection = DESC,
    offset: Int = 0,
    limit: Int = 50
  ): [SalesItem!]!

  salesItem(id: ID!): SalesItem!

  salesItemsByFilters(
    nameContains: String,
    priceGreaterThan: Float
  ): [SalesItem!]!
}

type Mutation {
  createSalesItem(salesItem: InputSalesItem!): SalesItem!

  updateSalesItem(
    id: ID!,
    salesItem: InputSalesItem
  ): IdResponse!

  deleteSalesItem(id: ID!): IdResponse!
}
```

在上面的 GraphQL 模式中，我们定义了几个用于 API 请求和响应的类型。GraphQL 类型指定了一个对象类型：该对象具有哪些属性以及这些属性的类型。使用 `input` 关键字指定的类型是仅用于输入的类型（输入 DTO 类型）。GraphQL 定义了以下原始（标量）类型：`Int`（32位）、`Float`、`String`、`Boolean` 和 `ID`。你可以使用 `[<Type>]` 表示法定义数组类型。默认情况下，类型是可空的。如果想要一个非空类型，必须在类型名称后添加感叹号（`!`）。你可以使用 `enum` 关键字定义枚举类型。`Query` 和 `Mutation` 类型是用于定义查询和变更的特殊 GraphQL 类型。上面的示例定义了三个查询和四个变更，客户端可以执行它们。你可以为类型属性添加参数。我们为所有查询（查询是 `Query` 类型的属性）、变更（变更是 `Mutation` 类型的属性）以及 `SalesItem` 类型的 `images` 属性都添加了参数。

在上面的示例中，我将所有查询命名为描述其返回值的名称，即查询名称中没有动词。也可以使用动词开头来命名查询（就像变更一样）。例如，如果你愿意，可以在上面定义的查询名称前加上 `get`。

有两种实现 GraphQL API 的方式：

-   模式优先
-   代码优先（模式从代码生成）

让我们首先关注模式优先的实现，并使用 Ariadne 库实现上面指定的 API。我们将首先为一些 API 端点（查询/变更）定义伪实现：

```python
import time

from ariadne import MutationType, QueryType, gql, make_executable_schema
from ariadne.asgi import GraphQL

schema = gql(
    """
type Image {
    id: Int!
    rank: Int!
    url: String!
}

type SalesItem {
    id: ID!
    createdAtTimestampInMs: String!
    name: String!
    priceInCents: Int!
    images(
      sortByField: String = "rank",
      sortDirection: SortDirection = ASC,
      offset: Int = 0,
      limit: Int = 5
    ): [Image!]!
}

input InputImage {
  id: Int!
  rank: Int!
  url: String!
}

input InputSalesItem {
  name: String!
  priceInCents: Int!
  images: [InputImage!]!
}

enum SortDirection {
  ASC
  DESC
}

type IdResponse {
  id: ID!
}

type Query {
  salesItems(
    sortByField: String = "createdAtTimestamp",
    sortDirection: SortDirection = DESC,
    offset: Int = 0,
    limit: Int = 50
  ): [SalesItem!]!

  salesItem(id: ID!): SalesItem!

  salesItemsByFilters(
    nameContains: String,
    priceGreaterThan: Float
  ): [SalesItem!]!
}

type Mutation {
  createSalesItem(inputSalesItem: InputSalesItem!): SalesItem!

  updateSalesItem(
    id: ID!,
    inputSalesItem: InputSalesItem
  ): IdResponse!

  deleteSalesItem(id: ID!): IdResponse!
}
"""
)

query = QueryType()

@query.field('salesItems')
def resolve_sales_items(**, **kwargs):
    if kwargs['offset'] == 0:
        return [
            {
                'id': 1,
                'createdAtTimestampInMs': '12345678999877',
                'name': 'sales item',
                'priceInCents': 1095,
                'images': [{'id': 1, 'rank': 2, 'url': 'url'}],
            }
        ]
    return []

@query.field('salesItem')
def resolve_sales_item(**, id):
    return {
        'id': id,
        'createdAtTimestampInMs': '12345678999877',
        'name': 'sales item',
        'priceInCents': 1095,
        'images': [{'id': 1, 'rank': 2, 'url': 'url'}],
    }

mutation = MutationType()

@mutation.field('createSalesItem')
def resolve_create_sales_item(**, **kwargs):
    return {
        'id': 100,
        'createdAtTimestampInMs': str(round(time.time() * 1000)),
        **kwargs['inputSalesItem'],
    }

@mutation.field('deleteSalesItem')
def resolve_delete_sales_item(**, id):
    return {'id': id}

executable_schema = make_executable_schema(schema, [query, mutation])

app = GraphQL(executable_schema)
```

在上面的示例中，`gql` 函数会验证模式，如果存在问题则抛出描述性的 `GraphQLSyntaxError`，如果正确则返回原始字符串。我们为模式中的前两个查询创建了解析器函数，也为创建和删除销售项创建了解析器。你可以使用以下命令启动 GraphQL 服务器（你应该已经使用 `pip` 安装了 `uvicorn`）：

```
uvicorn app:app
```

服务器运行后，使用浏览器访问以下 URL：http://127.0.0.1:8000/。你将看到 GraphiQL UI，并能够执行查询和变更。在 UI 的左窗格中输入以下查询。

```graphql
query salesItems {
  salesItems(offset: 0) {
    id
    createdAtTimestampInMs
    name
    priceInCents,
    images {
      url
    }
  }
}
```

你应该在右侧窗格中得到以下响应：

```json
{
  "data": {
    "salesItems": [
      {
        "id": "1",
        "createdAtTimestampInMillis": "12345678999877",
        "name": "sales item",
        "priceInCents": 1095,
        "images": [
          {
            "url": "url"
          }
        ]
      }
    ]
  }
}
```

你也可以尝试创建一个新的销售项：

```graphql
mutation create {
  createSalesItem(inputSalesItem: {
    priceInCents: 4095
    name: "test sales item"
    images: []
  }) {
    id,
    createdAtTimestampInMs,
    name,
    priceInCents,
    images {
      id
    },
  }
}
```

这是你会得到的响应，除了代表当前时间的时间戳：

```json
{
  "data": {
    "createSalesItem": {
      "id": "100",
      "createdAtTimestampInMillis": "1694798999418",
      "name": "test sales item",
      "priceInCents": 4095,
      "images": []
    }
  }
}
```

要删除一个销售项：

```graphql
mutation delete {
  deleteSalesItem(id: 1) {
    id
  }
}
```

```json
{
  "data": {
    "deleteSalesItem": {
      "id": "1"
    }
  }
}
```

让我们将 Ariadne GraphQL 控制器中的伪静态实现替换为对销售项服务的真实调用：

图 8.14. controllers/AriadneGraphQLSalesItemController.py

```python
from ariadne import MutationType, QueryType, gql, make_executable_schema
from dependency_injector.wiring import Provide

from ..dtos.InputSalesItem import InputSalesItem
from ..service.SalesItemService import SalesItemService

sales_item_service: SalesItemService = Provide['sales_item_service']

schema = gql(
    """
type Image {
  id: Int!
  rank: Int!
  url: String!
}

type SalesItem {
  id: ID!
  createdAtTimestampInMs: String!
  name: String!
  priceInCents: Int!
  images: [Image!]!
}

input InputImage {
  id: Int!
  rank: Int!
  url: String!
}

input InputSalesItem {
  name: String!
  priceInCents: Int!
  images: [InputImage!]!
}

type IdResponse {
  id: ID!
}

type Query {
  salesItems: [SalesItem!]!
  salesItem(id: ID!): SalesItem!
}

type Mutation {
  createSalesItem(inputSalesItem: InputSalesItem!): SalesItem!

  updateSalesItem(
    id: ID!,
    inputSalesItem: InputSalesItem
  ): IdResponse!

  deleteSalesItem(id: ID!): IdResponse!
}
"""
)

query = QueryType()

@query.field('salesItems')
def resolve_sales_items(*_):
    return sales_item_service.get_sales_items()

@query.field('salesItem')
def resolve_sales_item(*_, id: str):
    return sales_item_service.get_sales_item(id)
```

## API 设计原则

```python
def resolve_update_sales_item(*_, id: str, inputSalesItem):
    sales_item_update = InputSalesItem.parse_obj(inputSalesItem)
    sales_item_service.update_sales_item(id, sales_item_update)
    return {'id': id}

@mutation.field('deleteSalesItem')
def resolve_delete_sales_item(*_, id: str):
    sales_item_service.delete_sales_item(id)
    return {'id': id}

executable_schema = make_executable_schema(schema, [query, mutation])
```

请注意上面的代码，我们必须记住在两个变更操作中验证输入。我们可以在使用 `parse_obj` 方法将输入字典转换为 Pydantic 模型时进行验证。为了使示例更接近生产环境，我们应该添加授权、审计日志和指标更新。所有这些都可以通过创建装饰器来实现，其方式类似于我们在 REST API 示例中创建的方式。装饰器可以从 `info.context` 字典中获取请求对象：`info.context['request']`

GraphQL 的错误处理与 REST API 的错误处理不同。GraphQL API 响应不提供不同的 HTTP 响应状态码。GraphQL API 响应始终以状态码 `200 OK` 发送。当处理 GraphQL API 请求时发生错误，响应体对象将包含一个 `errors` 数组。在你的 GraphQL 类型解析器中，当查询或变更失败时，你应该引发一个错误。你可以使用与之前 REST API 示例中相同的 `ApiError` 基础错误类。为了处理自定义 API 错误，我们需要添加一个错误格式化器，如下所示。错误对象应始终包含一个 `message` 字段。关于错误的附加信息可以在 `extensions` 对象中提供，该对象可以包含任何属性。

假设一个 `salesItem` 查询导致了 `EntityNotFoundError`，API 响应的 `data` 属性将为 `null`，并且会存在 `errors` 属性：

```json
{
    "data": null,
    "errors": [
        {
            "message": "Sales item not found with id 1",
            "extensions": {
                "statusCode": 404,
                "statusText": "Not Found",
                "errorCode": "EntityNotFound",
                "errorDescription": null,
                "stackTrace": null
            }
        }
    ]
}
```

图 8.15. app_graphql.py

```python
import os
from typing import Any

from ariadne import format_error, unwrap_graphql_error
from ariadne.asgi import GraphQL
from pydantic import ValidationError

from .controllers.AriadneGraphQLSalesItemController import (
    executable_schema,
)
from .DiContainer import DiContainer
from .errors.SalesItemServiceError import SalesItemServiceError
from .utils import get_stack_trace

# Remove this setting of env variable for production code!
# mysql+pymysql://root:password@localhost:3306/salesitemservice
# mongodb://localhost:27017/salesitemservice
os.environ['DATABASE_URL'] = 'mongodb://localhost:27017/salesitemservice'

di_container = DiContainer()

def format_custom_error(
    graphql_error, debug: bool = False
) -> dict[str, Any]:
    error = unwrap_graphql_error(graphql_error)

    if isinstance(error, SalesItemServiceError):
        return {
            'message': error.message,
            'extensions': {
                'statusCode': error.status_code,
                'statusText': error.status_text,
                'errorCode': error.code,
                'errorDescription': error.description,
                'stackTrace': get_stack_trace(error.cause),
            },
        }

    if isinstance(error, ValidationError):
        return {
            'message': 'Request validation failed',
            'extensions': {
                'statusCode': 400,
                'statusText': 'Bad Request',
                'errorCode': 'RequestValidationError',
                'errorDescription': str(error),
                'stackTrace': None,
            },
        }

    if isinstance(error, Exception):
        return {
            'message': 'Unspecified internal error',
            'extensions': {
                'statusCode': 500,
                'statusText': 'Internal Server Error',
                'errorCode': 'UnspecifiedError',
                'errorDescription': str(error),
                'stackTrace': get_stack_trace(error),
            },
        }

    else:
        return format_error(graphql_error, debug)

app = GraphQL(executable_schema, error_formatter=format_custom_error)
```

Ariadne GraphQL 版本的 *sales-item-service* 可以通过以下命令运行。（我们假设服务源代码位于一个名为 *salesitemservice* 的 Python 包中，并且我们位于该包的父目录中）。

```
uvicorn salesitemservice.app_graphql:app
```

也可以将错误作为查询/变更的返回值返回。这可以通过例如从查询或变更返回一个联合类型来实现。这种方法需要更复杂的 GraphQL 模式和服务器端更复杂的解析器。例如：

```graphql
# ...

type Error {
    message: String!
    # Other possible properties
}

union SalesItemOrError = SalesItem | Error

type Mutation {
    createSalesItem(inputSalesItem: InputSalesItem!): SalesItemOrError!
}
```

在 *createSalesItem* 查询解析器中，你必须添加一个 try-except 块来处理错误情况，并在发生错误时返回一个 *Error* 对象。

你也可以指定多种错误类型：

```graphql
# ...
type ErrorType1 {
    # ...
}

type ErrorType2 {
    # ...
}

type ErrorType3 {
    # ...
}

union SalesItemOrError = SalesItem | ErrorType1 | ErrorType2 | ErrorType3

type Mutation {
    createSalesItem(inputSalesItem: InputSalesItem!): SalesItemOrError!
}
```

上述示例将要求 *createSalesItem* 解析器捕获多种不同的错误，并返回相应的错误对象作为结果。

客户端代码也会变得更加复杂，因为需要处理单个操作（查询/变更）的不同类型的响应。例如：

```graphql
mutation {
    createSalesItem(inputSalesItem: {
        price: 200
        name: "test sales item"
        images: []
    }) {
        __typename
        ...on SalesItem {
            id,
            createdAtTimestampInMillis
        }
        ...on ErrorType1 {
            # Specify fields here
        }
        ...on ErrorType2 {
            # Specify fields here
        }
        ...on ErrorType3 {
            # Specify fields here
        }
    }
}
```

这种方法还有一个缺点，即客户端仍然必须能够处理响应的 errors 数组中报告的可能错误。

在 GraphQL 模式中，你也可以为原始（标量）属性添加参数。这对于实现转换很有用。例如，我们可以定义一个带有参数化 price 属性的 SalesItem 类型：

```graphql
enum Currency {
  USD,
  GBP,
  EUR,
  JPY
}

type SalesItem {
  id: ID!
  createdAtTimestampInMillis: String!
  name: String!
  price(currency: Currency = USD): Float!
  images(
    sortByField: String = "rank",
    sortDirection: SortDirection = ASC,
    offset: Int = 0,
    limit: Int = 5
  ): [Image!]!
}
```

现在客户端可以在查询中为 price 属性提供一个货币参数，以获取不同货币的价格。如果未提供货币参数，则默认货币为 USD。

以下是客户端可以对前面定义的 GraphQL 模式执行的两个示例查询：

```graphql
{
  # gets the name, price in euros and the first 5 images
  # for the sales item with id "1"
  salesItem(id: "1") {
    name
    price(currency: EUR)
    images
  }

  # gets the next 5 images for the sales item 1
  salesItem(id: "1") {
    images(offset: 5)
  }
}
```

在实际应用中，请考虑将资源获取限制为仅上一页或下一页（或者如果你在客户端实现无限滚动，则仅获取下一页）。这样，客户端就无法获取随机页面。这可以防止恶意用户试图获取页码巨大（例如 10,000）的页面的攻击，这种攻击可能导致服务器额外负载，或者在极端情况下导致拒绝服务。

下面是一个示例，客户端只能查询第一页、下一页或上一页。当客户端请求第一页时，页面游标可以为空，但当客户端请求上一页或下一页时，它必须将当前页面游标作为查询参数提供。

type PageOfSalesItems {
  # 包含经过加密并编码为 Base64 值的页码。
  pageCursor: String!

  salesItems: [SalesItem!]!
}

enum Page {
  FIRST,
  NEXT,
  PREVIOUS
}

type Query {
  pageOfSalesItems(
    page: Page = FIRST,
    pageCursor: String = ""
  ): PageOfSalesItems!
}

让我们再看一个 GraphQL 的例子，这次使用 Strawberry 库的代码优先方法。在实现生产代码时，我们应该遵循清晰的微服务设计原则。我们应该能够与之前的 sales-item-service REST API 示例共享服务、仓库、DTO、错误和实体，仅为 GraphQL API 定义一个单独的控制器。下面的示例只实现了两个 API 端点（获取销售项和创建销售项），以保持示例的简洁。

图 8.16. controllers/StrawberryGraphQLSalesItemController.py

```python
import strawberry
from dependency_injector.wiring import Provide
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

from ..graphqltypes.IdResponse import IdResponse
from ..graphqltypes.InputSalesItem import InputSalesItem
from ..graphqltypes.OutputSalesItem import OutputSalesItem
from ..service.SalesItemService import SalesItemService

sales_item_service: SalesItemService = Provide['sales_item_service']


class StrawberryGraphQLSalesItemController:
    @strawberry.type
    class Query:
        @strawberry.field
        def salesItems(self, info: Info) -> list[OutputSalesItem]:
            output_sales_items = sales_item_service.get_sales_items()

            return [
                OutputSalesItem.from_pydantic(output_sales_item)
                for output_sales_item in output_sales_items
            ]

        @strawberry.field
        def salesItem(self, info: Info, id: str) -> OutputSalesItem:
            output_sales_item = sales_item_service.get_sales_item(id)
            return OutputSalesItem.from_pydantic(output_sales_item)

    @strawberry.type
    class Mutation:
        @strawberry.mutation
        def createSalesItem(
            self, info: Info, inputSalesItem: InputSalesItem
        ) -> OutputSalesItem:
            output_sales_item = sales_item_service.create_sales_item(
                inputSalesItem.to_pydantic()
            )

            return OutputSalesItem.from_pydantic(output_sales_item)

        @strawberry.mutation
        def updateSalesItem(
            self, info: Info, id: str, inputSalesItem: InputSalesItem
        ) -> IdResponse:
            sales_item_service.update_sales_item(
                id, inputSalesItem.to_pydantic()
            )
            return IdResponse(id=id)

        @strawberry.mutation
        def deleteSalesItem(self, info: Info, id: str) -> IdResponse:
            sales_item_service.delete_sales_item(id)
            return IdResponse(id=id)

__schema = strawberry.Schema(query=Query, mutation=Mutation)
__router = GraphQLRouter(__schema, path='/graphql')

@property
def router(self):
    return self.__router
```

为了使我们的控制器更接近生产环境，我们必须添加授权、审计日志和指标更新。我们可以实现与之前 REST API 示例中类似的装饰器。当装饰器需要访问请求时，可以通过 *info* 参数完成：info.context['request']

除了上述控制器，我们必须定义 strawberry 类型，这些类型可以基于现有的 pydantic 类。以下是 strawberry 类型：

图 8.17. graphqltypes/InputSalesItem.py

```python
import strawberry

from ..dtos.InputSalesItem import InputSalesItem
from .InputSalesItemImage import InputSalesItemImage


@strawberry.experimental.pydantic.input(model=InputSalesItem)
class InputSalesItem:
    name: strawberry.auto
    priceInCents: strawberry.auto
    images: list[InputSalesItemImage]
```

图 8.18. graphqltypes/InputSalesItemImage.py

```python
import strawberry

from ..dtos.SalesItemImage import SalesItemImage


@strawberry.experimental.pydantic.input(
    model=SalesItemImage, all_fields=True
)
class InputSalesItemImage:
    pass
```

图 8.19. graphqltypes/OutputSalesItem.py

```python
import strawberry

from ..dtos.OutputSalesItem import OutputSalesItem
from .OutputSalesItemImage import OutputSalesItemImage


@strawberry.experimental.pydantic.type(model=OutputSalesItem)
class OutputSalesItem:
    id: strawberry.auto
    createdAtTimestampInMs: str
    name: strawberry.auto
    priceInCents: strawberry.auto
    images: list[OutputSalesItemImage]
```

图 8.20. graphqltypes/OutputSalesItemImage.py

```python
import strawberry

from ..dtos.SalesItemImage import SalesItemImage

@strawberry.experimental.pydantic.type(
    model=SalesItemImage, all_fields=True
)
class OutputSalesItemImage:
    pass
```

## 8.1.4：基于订阅的 API 设计

> 当你希望客户端能够订阅大型对象的小幅增量变化，或者客户端希望接收低延迟的实时更新时，请设计基于订阅的 API。

### 8.1.4.1：服务器发送事件（SSE）

服务器发送事件（SSE）是一种单向推送技术，使客户端能够通过 HTTP 连接从服务器接收更新。

让我们用一个实际例子来展示 SSE 的功能。下面的示例定义了一个 subscribe-to-loan-app-summaries API 端点，供客户端订阅贷款申请摘要。客户端将在其 UI 的列表视图中显示贷款申请摘要。每当有新的贷款申请摘要可用时，服务器将向客户端发送一个贷款申请摘要事件，客户端将通过添加新的贷款申请摘要来更新其 UI。下面的示例使用 FastAPI 和 sse-starlette 库。

```python
import json

from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse

loan_app_summaries = []

app = FastAPI()

def get_loan_app_summary():
    if len(loan_app_summaries) > 0:
        return loan_app_summaries.pop(0)
    return None

@app.get('/subscribe-to-loan-app-summaries')
async def subscribe_to_loan_app_summaries(request: Request):
    async def generate_loan_app_summary_events():
        while True:
            if await request.is_disconnected():
                break

            loan_app_summary = get_loan_app_summary()
            if loan_app_summary:
                yield json.dumps(loan_app_summary)

    return EventSourceResponse(
        generate_loan_app_summary_events()
    )

@app.post('/loan-app-summaries')
async def create_loan_app_summary(
    request: Request
) -> None:
    loan_app_summary = await request.json()
    loan_app_summaries.append(loan_app_summary)
```

接下来，我们可以用 JavaScript 实现 Web 客户端，并定义以下 React 函数式组件：

```javascript
import React, { useEffect, useState } from 'react';

export default function LoanAppSummaries() {
  const [ loanAppSummaries, setLoanAppSummaries ] = useState([]);

  // 定义一个在组件挂载时执行的效果
  useEffect(() => {
    // 创建新的事件源
    // 这里使用硬编码的开发环境 URL 仅用于演示目的
    const eventSource =
      new EventSource('http://localhost:8000/subscribe-to-loan-app-summaries');

    // 监听服务器发送的事件，并将新的
    // 贷款申请摘要添加到 loanAppSummaries 数组的头部
    eventSource.addEventListener('message', (messageEvent) => {
      try {
        const loanAppSummary = JSON.parse(messageEvent.data);

        if (loanAppSummary) {
          setLoanAppSummaries([loanAppSummary, ...loanAppSummaries]);
        }
      } catch {
        // 处理错误
      }
    });

    eventSource.addEventListener('error', (errorEvent) => {
      // 处理错误
    });

    // 在组件卸载时关闭事件源
    return function cleanup() { eventSource.close(); }
  }, [loanAppSummaries]);

  // 渲染贷款申请摘要列表项
  const loanAppSummaryListItems =
    loanAppSummaries.map(({ ... }) =>
      (<li key={key here...}>render here...</li>));

  return (
    <ul>{loanAppSummaryListItems}</ul>
  );
}
```

### 8.1.4.2：GraphQL 订阅

让我们看一个 GraphQL 订阅的例子。下面的 GraphQL 模式定义了一个用于帖子评论的订阅。帖子是什么并不重要，它可以是博客文章或社交媒体帖子。我们希望客户端能够订阅帖子的评论。

```graphql
type PostComment {
  id: ID!,
  text: String!
}

type Subscription {
  postComment(postId: ID!): PostComment
}
```

在客户端，我们可以有以下 JavaScript 代码来定义一个名为 postCommentText 的订阅，该订阅订阅帖子的评论并返回评论的 text 属性：

```javascript
import { gql } from '@apollo/client';

const POST_COMMENT_SUBSCRIPTION = gql`
  subscription postCommentText($postId: ID!) {
    postComment(postID: $postId) {
      text
    }
  }
`;
```

如果客户端为特定帖子（通过 postId 参数定义）执行上述查询，可以预期得到以下类型的响应：

## 8.1.4.3：WebSocket 示例

下面是一个聊天消息应用，包含一个使用 FastAPI、Kafka 和 Redis 实现的 WebSocket 服务器，以及一个使用 React 实现的 WebSocket 客户端。服务器可以运行多个实例。这些实例是无状态的，除了为本地连接的客户端存储 WebSocket 连接。首先，我们列出服务器端的源代码文件。

使用 *redis-py* 库创建一个新的 Redis 客户端：

**图 8.21. redis_client.py**

```python
import os

from redis import Redis

# The current version of official Python documentation
# does not tell what errors the 'int' constructor can raise,
# but it can raise a 'TypeError' if the argument type
# is not convertible to an integer, and it can raise
# 'ValueError' if the argument value is not convertible
# to an integer
try:
    port = int(os.environ.get('REDIS_PORT'))
except (TypeError, ValueError):
    port = 6379

redis_client = Redis(
    host=os.environ.get('REDIS_HOST') or 'localhost',
    port=port,
    username=os.environ.get('REDIS_USERNAME'),
    password=os.environ.get('REDIS_PASSWORD'),
)
```

下面的 KafkaMsgBrokerAdminClient 类用于在 Kafka 中创建主题：

**图 8.22. ChatMsgBrokerAdminClient.py**

```python
from typing import Protocol

from .WebSocketExampleError import WebSocketExampleError


class ChatMsgBrokerAdminClient(Protocol):
    class CreateTopicError(WebSocketExampleError):
        pass

    def try_create_topic(self, name: str) -> None:
        pass
```

**图 8.23. KafkaChatMsgBrokerAdminClient.py**

```python
import os

from ChatMsgBrokerAdminClient import ChatMsgBrokerAdminClient
from confluent_kafka import KafkaError, KafkaException
from confluent_kafka.admin import AdminClient
from confluent_kafka.cimpl import NewTopic


class KafkaChatMsgBrokerAdminClient(ChatMsgBrokerAdminClient):
    def __init__(self):
        self.__admin_client = AdminClient(
            {
                'bootstrap.servers': os.environ.get('KAFKA_BROKERS'),
                'client.id': 'chat-messaging-service',
            }
        )

    def try_create_topic(self, name: str) -> None:
        topic = NewTopic(name)

        try:
            topic_name_to_creation_dict = (
                self.__admin_client.create_topics([topic])
            )
            topic_name_to_creation_dict[name].result()
        except KafkaException as error:
            if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise self.CreateTopicError(error)
```

聊天消息应用的用户通过电话号码进行标识。在服务器端，我们将每个用户的 WebSocket 连接存储在 phone_nbr_to_conn_map 中：

**图 8.24. phone_nbr_to_conn_map.py**

```python
from Connection import Connection

phone_nbr_to_conn_map: dict[str, Connection] = {}
```

**图 8.25. Connection.py**

```python
from typing import Any, Protocol

from WebSocketExampleError import WebSocketExampleError


class Connection(Protocol):
    class Error(WebSocketExampleError):
        pass

    async def try_connect(self) -> None:
        pass

    async def try_send_json(self, message: dict[str, Any]) -> None:
        pass

    async def try_send_text(self, message: str) -> None:
        pass

    async def try_receive_json(self) -> dict[str, str]:
        pass

    async def try_close(self) -> None:
        pass
```

**图 8.26. WebSocketConnection.py**

```python
from typing import Any

from Connection import Connection
from fastapi import WebSocket, WebSocketException

class WebSocketConnection(Connection):
    def __init__(self, websocket: WebSocket):
        self.__websocket = websocket

    async def try_connect(self) -> None:
        try:
            await self.__websocket.accept()
        except WebSocketException:
            raise self.Error()

    async def try_send_json(self, message: dict[str, Any]) -> None:
        try:
            await self.__websocket.send_json(message)
        except WebSocketException:
            raise self.Error()

    async def try_send_text(self, message: str) -> None:
        try:
            await self.__websocket.send_text(message)
        except WebSocketException:
            raise self.Error()

    async def try_receive_json(self) -> dict[str, str]:
        try:
            return await self.__websocket.receive_json()
        except WebSocketException:
            raise self.Error()

    async def try_close(self) -> None:
        try:
            return await self.__websocket.close()
        except WebSocketException:
            raise self.Error()
```

下面的模块是 WebSocket 服务器。服务器接受来自客户端的连接。当它从客户端接收到聊天消息时，它会首先解析并验证该消息。对于聊天消息，服务器会将消息存储在持久化存储中（使用一个单独的 *chat-message-service* REST API，此处未实现）。服务器从 Redis 缓存中获取接收者的服务器信息，并将聊天消息发送到接收者的 WebSocket 连接，或者将聊天消息生产到 Kafka 主题中，另一个微服务实例可以消费该聊天消息并将其发送到接收者的 WebSocket 连接。Redis 缓存存储一个哈希映射，其中用户的电话号码映射到他们当前连接的服务器实例。一个 UUID 标识一个微服务实例。

```python
from typing import Protocol

from .Connection import Connection

class ChatMsgServer(Protocol):
    async def handle(
        self, connection: Connection, phone_number: str
    ) -> None:
        pass
```

```python
import json
from typing import Final

from ChatMsgBrokerProducer import ChatMsgBrokerProducer
from ChatMsgServer import ChatMsgServer
from Connection import Connection
from fastapi import WebSocket, WebSocketDisconnect, WebSocketException
from KafkaChatMsgBrokerProducer import KafkaChatMsgBrokerProducer
from phone_nbr_to_conn_map import phone_nbr_to_conn_map
from PhoneNbrToInstanceUuidCache import PhoneNbrToInstanceUuidCache
from redis_client import redis_client
from RedisPhoneNbrToInstanceUuidCache import (
    RedisPhoneNbrToInstanceUuidCache,
)
from WebSocketConnection import WebSocketConnection

class WebSocketChatMsgServer(ChatMsgServer):
    def __init__(self, instance_uuid: str):
        self.__instance_uuid: Final = instance_uuid
        self.__conn_to_phone_nbr_map: Final[dict[Connection, str]] = {}
        self.__chat_msg_broker_producer: Final = (
            KafkaChatMsgBrokerProducer()
        )
        self.__cache: Final = RedisPhoneNbrToInstanceUuidCache(
            redis_client
        )

    async def handle(
        self, connection: Connection, phone_number: str
    ) -> None:
        try:
            await connection.try_connect()
            phone_nbr_to_conn_map[phone_number] = connection
```

要能够使用 GraphQL 订阅，你必须在服务器端和客户端都实现对它们的支持。实际上，这意味着要建立 WebSocket 通信，因为 GraphQL 使用该协议来实现订阅。对于服务器端，你可以在这里找到 *Ariadne* 库的说明：[https://ariadnegraphql.org/docs/subscriptions](https://ariadnegraphql.org/docs/subscriptions)。对于客户端，你可以在这里找到 *Apollo client* 的说明：[https://www.apollographql.com/docs/react/data/subscriptions/setting-up-the-transport](https://www.apollographql.com/docs/react/data/subscriptions/setting-up-the-transport)³

在服务器端和客户端都实现了订阅支持之后，你可以在你的 React 组件中使用订阅：

```javascript
import { useState } from 'react';
import { gql, useSubscription } from '@apollo/client';

const POST_COMMENT_SUBSCRIPTION = gql`
  subscription subscribeToPostComment($postId: ID!) {
    postComment(postID: $postId) {
      id
      text
    }
  }
`;

export default function SubscribedPostCommentsView({ postId }) {
  const [ postComments, setPostComments ] = useState([]);

  const { data } = useSubscription(POST_COMMENT_SUBSCRIPTION,
                                    { variables: { postId } });

  if (data?.postComment) {
    setPostComments([...postComments, data.postComment]);
  }

  const postCommentListItems =
    postComments.map(( { id, text }) =>
      (<li key={id}>{text}</li>));

  return <ul>{postCommentListItems}</ul>;
}
```

³ [https://www.apollographql.com/docs/react/data/subscriptions/#setting-up-the-transport](https://www.apollographql.com/docs/react/data/subscriptions/#setting-up-the-transport)

## API 设计原则

```python
self.__conn_to_phone_nbr_map[connection] = phone_number
self.__cache.try_store(phone_number, self.__instance_uuid)

while True:
    chat_message: dict[
        str, str
    ] = await connection.try_receive_json()

    # Validate chat_message ...
    # Store chat message permanently using another API ...
    recipient_phone_nbr = chat_message.get('recipientPhoneNbr')

    recipient_instance_uuid = (
        self.__cache.retrieve_instance_uuid(
            recipient_phone_nbr
        )
    )

    await self.__try_send(
        chat_message, recipient_instance_uuid
    )
except WebSocketDisconnect:
    self.__disconnect(connection)
except PhoneNbrToInstanceUuidCache.Error:
    # Handle error ...
except Connection.Error:
    # Handle error ...
except ChatMsgBrokerProducer.Error:
    # Handle error ...

def close(self) -> None:
    for connection in self.__conn_to_phone_nbr_map.keys():
        try:
            connection.try_close()
        except Connection.Error:
            pass

    self.__chat_msg_broker_producer.close()

async def __try_send(
    self,
    chat_message: dict[str, str],
    recipient_instance_uuid: str | None,
) -> None:
    if recipient_instance_uuid == self.__instance_uuid:
        # Recipient has active connection on
        # the same server instance as sender
        recipient_conn = phone_nbr_to_conn_map.get(
            chat_message.get('recipientPhoneNbr')
        )

        if recipient_conn:
            await recipient_conn.try_send_json(chat_message)
    elif recipient_instance_uuid:
        # Recipient has active connection on different
        # server instance compared to sender
        chat_message_json = json.dumps(chat_message)

        self.__chat_msg_broker_producer.try_produce(
            chat_message_json, topic=recipient_instance_uuid
        )

def __disconnect(self, connection: Connection) -> None:
    phone_number = self.__conn_to_phone_nbr_map.get(connection)

    if phone_number:
        del phone_nbr_to_conn_map[phone_number]

    del self.__conn_to_phone_nbr_map[connection]

    try:
        self.__cache.try_remove(phone_number)
    except PhoneNbrToInstanceUuidCache.Error:
        # Handle error ...
```

图 8.29. PhoneNbrToInstanceUuidCache.py

```python
from typing import Protocol

from WebSocketExampleError import WebSocketExampleError


class PhoneNbrToInstanceUuidCache(Protocol):
    class Error(WebSocketExampleError):
        pass

    def retrieve_instance_uuid(
        self, phone_number: str | None
    ) -> str | None:
        pass

    def try_store(self, phone_number: str, instance_uuid: str) -> None:
        pass

    def try_remove(self, phone_number: str) -> None:
        pass
```

图 8.30. RedisPhoneNbrToInstanceUuidCache.py

```python
from PhoneNbrToInstanceUuidCache import PhoneNbrToInstanceUuidCache
from redis import Redis, RedisError


class RedisPhoneNbrToInstanceUuidCache(PhoneNbrToInstanceUuidCache):
    def __init__(self, redis_client: Redis):
        self.__redis_client = redis_client

    def retrieve_instance_uuid(
        self, phone_number: str | None
    ) -> str | None:
        if phone_number:
            try:
                return self.__redis_client.hget(
                    'phoneNbrToInstanceUuidMap', phone_number
                )
            except RedisError:
                pass

        return None

    def try_store(self, phone_number: str, instance_uuid: str) -> None:
        try:
            self.__redis_client.hset(
                'phoneNbrToInstanceUuidMap', phone_number, instance_uuid
            )
        except RedisError:
            raise self.Error()

    def try_remove(self, phone_number: str) -> None:
        try:
            self.__redis_client.hdel(
                'phoneNbrToInstanceUuidMap', [phone_number]
            )
        except RedisError:
            raise self.Error()
```

图 8.31. ChatMsgBrokerProducer.py

```python
from typing import Protocol

from WebSocketExampleError import WebSocketExampleError


class ChatMsgBrokerProducer(Protocol):
    class Error(WebSocketExampleError):
        pass

    def try_produce(self, chat_message_json: str, topic: str):
        pass

    def close(self):
        pass
```

图 8.32. KafkaChatMsgBrokerProducer.py

```python
import os

from ChatMsgBrokerProducer import ChatMsgBrokerProducer
from confluent_kafka import KafkaException, Producer


class KafkaChatMsgBrokerProducer(ChatMsgBrokerProducer):
    def __init__(self):
        config = {
            'bootstrap.servers': os.environ.get('KAFKA_BROKERS'),
            'client.id': 'chat-messaging-service',
        }

        self.__producer = Producer(config)

    def try_produce(self, chat_message_json: str, topic: str):
        def handle_error(error: KafkaException):
            if error is not None:
                raise self.Error()

        try:
            self.__producer.produce(
                topic, chat_message_json, on_delivery=handle_error
            )

            self.__producer.poll()
        except KafkaException:
            raise self.Error()

    def close(self):
        try:
            self.__producer.flush()
        except KafkaException:
            pass
```

`KafkaChatMsgBrokerConsumer` 类定义了一个 Kafka 消费者，它从特定的 Kafka 主题消费聊天消息，并将其发送到接收者的 WebSocket 连接：

图 8.33. ChatMsgBrokerConsumer.py

```python
from typing import Protocol

class ChatMsgBrokerConsumer(Protocol):
    def consume_chat_msgs(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass
```

图 8.34. KafkaChatMsgBrokerConsumer.py

```python
import json
import os

from ChatMsgBrokerConsumer import ChatMsgBrokerConsumer
from confluent_kafka import Consumer, KafkaException
from Connection import Connection
from phone_nbr_to_conn_map import phone_nbr_to_conn_map

class KafkaChatMsgBrokerConsumer(ChatMsgBrokerConsumer):
    def __init__(self, topic: str):
        self.__topic = topic

        config = {
            'bootstrap.servers': os.environ.get('KAFKA_BROKERS'),
            'group.id': 'chat-messaging-service',
            'auto.offset.reset': 'smallest',
            'enable.partition.eof': False,
        }

        self.__consumer = Consumer(config)
        self.__is_running = True

    def consume_chat_msgs(self) -> None:
        self.__consumer.subscribe([self.__topic])

        while self.__is_running:
            try:
                messages = self.__consumer.poll(timeout=1)

                if messages is None:
                    continue

                for message in messages:
                    if message.error():
                        raise KafkaException(message.error())
                    else:
                        chat_message_json = message.value()
                        chat_message = json.loads(chat_message_json)

                        recipient_conn = phone_nbr_to_conn_map.get(
                            chat_message.get('recipientPhoneNbr')
                        )

                        if recipient_conn:
                            recipient_conn.try_send_text(chat_message_json)
            except KafkaException:
                # Handle error ...
            except Connection.Error:
                # Handle error ...

    def stop(self) -> None:
        self.__is_running = False

    def close(self):
        self.__consumer.close()
```

图 8.35. app.py

```python
import sys
from threading import Thread
from uuid import uuid4

from fastapi import FastAPI, WebSocket
from KafkaChatMsgBrokerAdminClient import KafkaChatMsgBrokerAdminClient
from KafkaChatMsgBrokerConsumer import KafkaChatMsgBrokerConsumer
from WebSocketChatMsgServer import WebSocketChatMsgServer
from WebSocketConnection import WebSocketConnection

instance_uuid = str(uuid4())

# Create a Kafka topic for this particular microservice instance
try:
    KafkaChatMsgBrokerAdminClient().try_create_topic(instance_uuid)
except KafkaChatMsgBrokerAdminClient.CreateTopicError:
    # Log error
    sys.exit(1)

# Create and start a Kafka consumer to consume and send
# chat messages for recipients that are connected to
# this microservice instance
chat_msg_broker_consumer = KafkaChatMsgBrokerConsumer(topic=instance_uuid)

chat_msg_consumer_thread = Thread(
    target=chat_msg_broker_consumer.consume_chat_msgs
)

chat_msg_consumer_thread.start()

app = FastAPI()
chat_msg_server = WebSocketChatMsgServer(instance_uuid)

@app.websocket('/chat-messaging-service/{phone_number}')
async def handle_websocket(websocket: WebSocket, phone_number: str):
    connection = WebSocketConnection(websocket)
    await chat_msg_server.handle(connection, phone_number)

@app.on_event('shutdown')
def shutdown_event():
    chat_msg_broker_consumer.stop()
    chat_msg_consumer_thread.join()
    chat_msg_broker_consumer.close()
    chat_msg_server.close()
```

对于 Web 客户端，我们有以下代码。`ChatMessagingService` 类的一个实例通过 WebSocket 连接到聊天消息服务器。它监听从服务器接收的消息，并在收到聊天消息时分发一个动作。该类还提供了一个向服务器发送聊天消息的方法。

图 8.36. ChatMessagingService.js

```javascript
import store from "./store";

class ChatMessagingService {
  wsConnection;
  connectionIsOpen = false;
  lastChatMessage;

  constructor(dispatch, userPhoneNbr) {
    this.wsConnection =
        new WebSocket(`ws://localhost:8080/chat-messaging-service/${userPhoneNbr}`);

    this.wsConnection.addEventListener('open', () => {
      this.connectionIsOpen = true;
    });

    this.wsConnection.addEventListener('error', () => {
      this.lastChatMessage = null;
    });
```

this.wsConnection.addEventListener(
  'message',
  ({ data: chatMessageJson }) => {
    const chatMessage = JSON.parse(chatMessageJson);

    store.dispatch({
      type: 'receivedChatMessageAction',
      chatMessage
    });
  });

this.wsConnection.addEventListener('close', () => {
  this.connectionIsOpen = false;
});
}

send(chatMessage) {
  this.lastChatMessage = chatMessage;

  if (this.connectionIsOpen) {
    this.wsConnection.send(JSON.stringify(chatMessage));
  } else {
    // 将消息发送到 REST API
  }
}

close() {
  this.connectionIsOpen = false;
  this.wsConnection.close();
}
}

export let chatMessagingService;

export default function createChatMessagingService(
  userPhoneNbr
) {
  chatMessagingService =
    new ChatMessagingService(store.dispatch, userPhoneNbr);

  return chatMessagingService;
}
```

图 8.37. index.jsx

```
import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux'
import ChatApp from './ChatApp';
import store from './store'
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <Provider store={store}>
    <ChatApp/>
  </Provider>
);
```

聊天应用 ChatApp 从 URL 中解析用户和联系人的电话号码，然后渲染用户与联系人之间的聊天视图：

图 8.38. ChatApp.jsx

```
import React, { useEffect } from 'react';
import queryString from "query-string";
import ContactChatView from "./ContactChatView";
import createChatMessagingService from "./ChatMessagingService";

const { userPhoneNbr, contactPhoneNbr } =
  queryString.parse(window.location.search);

export default function ChatApp() {
  useEffect(() => {
    const chatMessagingService =
      createChatMessagingService(userPhoneNbr);

    return function cleanup() {
      chatMessagingService.close();
    }
  }, []);

  return (
    <div>
      用户: {userPhoneNbr}
      <ContactChatView
        userPhoneNbr={userPhoneNbr}
        contactPhoneNbr={contactPhoneNbr}
      />
    </div>
  );
}
```

ContactChatView 组件渲染用户与联系人之间的聊天消息：

图 8.39. ContactChatView.jsx

```
import React, { useEffect, useRef } from 'react';
import { connect } from "react-redux";
import store from './store';

function ContactChatView({
  userPhoneNbr,
  contactPhoneNbr,
  chatMessages,
  fetchLatestChatMessages
}) {
  const inputElement = useRef(null);

  useEffect(() => {
    fetchLatestChatMessages(userPhoneNbr, contactPhoneNbr);
  }, [contactPhoneNbr,
    fetchLatestChatMessages,
    userPhoneNbr]
  );

  function sendChatMessage() {
    if (inputElement?.current.value) {
      store.dispatch({
        type: 'sendChatMessageAction',
        chatMessage: {
          senderPhoneNbr: userPhoneNbr,
          recipientPhoneNbr: contactPhoneNbr,
          message: inputElement.current.value
        }
      });
    }
  }

  const chatMessageElements = chatMessages
    .map(({ message, senderPhoneNbr }, index) => {
      const messageIsReceived =
        senderPhoneNbr === contactPhoneNbr;

      return (
        <li
          key={index}
          className={messageIsReceived ? 'received' : 'sent'}>
          {message}
        </li>
      );
    });

  return (
    <div className="contactChatView">
      联系人: {contactPhoneNbr}
      <ul>{chatMessageElements}</ul>
      <input ref={inputElement}/>
      <button onClick={sendChatMessage}>发送</button>
    </div>
  );
}

function mapStateToProps(state) {
  return {
    chatMessages: state
  };
}

export default connect(mapStateToProps)(ContactChatView);
```

图 8.40. store.js

```
import { createStore } from 'redux';
import { chatMessagingService } from "./ChatMessagingService";

function chatMessagesReducer(state = [], { type, chatMessage }) {
  switch (type) {
    case 'receivedChatMessageAction':
      return state.concat([chatMessage]);
    case 'sendChatMessageAction':
      chatMessagingService.send(chatMessage);
      return state.concat([chatMessage]);
    default:
      return state;
  }
}

const store = createStore(chatMessagesReducer)
export default store;
```

图 8.41. index.css

```
.contactChatView {
  width: 420px;
}

.contactChatView ul {
  padding-inline-start: 0;
  list-style-type: none;
}

.contactChatView li {
  margin-top: 15px;
  width: fit-content;
  max-width: 180px;
  padding: 10px;
  border: 1px solid #888;
  border-radius: 20px;
}

.contactChatView li.received {
  margin-right: auto;
}

.contactChatView li.sent {
  margin-left: auto;
}
```

用户: 0504877334
联系人: 0501234567

fsfd

111

2222

3333

sdfsdfdsf
fsadfsdafdfsdfsdfsdfsdf

sdfsdafdsafsda
fsdafsdafsdafsadf s
fsadfsdafas afsdf

sdfsdfdsf fsadfsdafdfsdfsdfsdf

发送

图 8.42. 两个用户的聊天消息应用视图

![](img/cbd069395d7b824346b69b1f92e0fb4a_494_0.png)

图 8.43. 两个用户的聊天消息应用视图

## 8.2: 微服务间 API 设计原则

微服务间 API 可以根据通信类型分为两类：同步和异步。当期望对发出的请求立即得到响应时，应使用同步通信。当不期望对请求做出响应，或响应不是立即需要时，应使用异步通信。

### 8.2.1: 同步 API 设计原则

当请求和响应不是很大且不包含大量二进制数据时，使用基于 HTTP 的 RPC、REST 或 GraphQL API，并采用 JSON 数据编码，最好使用 HTTP/2 或 HTTP/3 传输。如果你有大型请求或响应，或者大量二进制数据，最好将数据编码为 Avro 二进制格式（Content-Type: avro/binary）而不是 JSON，或者使用基于 gRPC 的 API。gRPC 始终以二进制格式（Protocol Buffers）编码数据。

#### 8.2.1.0.1: 基于 gRPC 的 API 设计示例

让我们看一个基于 gRPC 的 API 示例。首先，我们必须定义所需的 Protocol Buffers 类型。它们在扩展名为 .proto 的文件中定义。proto 文件的语法非常简单。我们通过列出远程过程来定义服务。远程过程使用以下语法定义：`rpc <procedure-name> (<argument-type>) returns (<return-type>) {}`。类型使用以下语法定义：

```
message <type-name> {
    <field-type> <field-name> [= <field-index>];
    ...
}
```

图 8.44. sales_item_service.proto

```
syntax = "proto3";

option objc_class_prefix = "SIS";

package salesitemservice;

service SalesItemService {
    rpc createSalesItem (InputSalesItem) returns (OutputSalesItem) {}
    rpc getSalesItems (GetSalesItemsOptions) returns (OutputSalesItems) {}
    rpc getSalesItem (Id) returns (OutputSalesItem) {}
    rpc updateSalesItem (SalesItemUpdate) returns (Nothing) {}
    rpc deleteSalesItem (Id) returns (Nothing) {}
}

message GetSalesItemsOptions {
    optional string sortByField = 1;
    optional string sortDirection = 2;
    optional uint64 offset = 3;
    optional uint64 limit = 4;
}

message Nothing {}

message Image {
    uint64 id = 1;
    uint64 rank = 2;
    string url = 3;
}

message InputSalesItem {
  string name = 1;
  float price = 2;
  repeated Image images = 3;
}

message SalesItemUpdate {
  uint64 id = 1;
  string name = 2;
  float price = 3;
  repeated Image images = 4;
}

message OutputSalesItem {
  uint64 id = 1;
  uint64 createdAtTimestampInMillis = 2;
  string name = 3;
  float price = 4;
  repeated Image images = 5;
}

message Id {
  uint64 id = 1;
}

message OutputSalesItems {
  repeated OutputSalesItem salesItems = 1;
}

message ErrorDetails {
  optional string code = 1;
  optional string description = 2;
}
```

在上面的示例中，`getSalesItems` 方法返回一个包含销售项目数组的对象。gRPC 还提供了在两个方向上流式传输数据的可能性。例如，我们可以将 `getSalesItems` 方法设为流式方法，这样我们就不需要在 `GetSalesItemsArg` 中包含 `offset` 和 `limit` 属性。要定义一个流式的 `getSalesItems` 方法：

```
// ...
service SalesItemService {
  // ...
  rpc getSalesItems (GetSalesItemsArg) returns (stream OutputSalesItem) {}
  // ...
}
// ...
```

完成 proto 文件后，我们必须为 gRPC 服务器生成代码。让我们安装 grpcio-tools 库：

pip install grpcio-tools

然后我们可以生成代码：

```
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. sales_item_service.proto
```

执行上述命令后，目录中应该会生成三个文件。创建实际的服务器代码需要以下两个步骤：

- 实现生成的 *servicer* 接口，编写执行服务实际“工作”的函数。
- 运行一个 gRPC 服务器，监听客户端请求并传输响应。

我们需要安装：

pip install grpcio grpcio-status

图 8.45. controllers/GrpcSalesItemSController.py

```
from dependency_injector.wiring import Provide
from google.protobuf import any_pb2, json_format
from google.rpc import code_pb2, status_pb2
from grpc_status import rpc_status
from pydantic import ValidationError

from ..dtos.InputSalesItem import InputSalesItem as PydanticInputSalesItem
from ..errors.SalesItemServiceError import SalesItemServiceError
from ..grpc.proto_to_dict import proto_to_dict
from ..grpc.sales_item_service_pb2 import (
    ErrorDetails,
    GetSalesItemsArg,
    Id,
    InputSalesItem,
    Nothing,
    OutputSalesItem,
    OutputSalesItems,
    SalesItemUpdate,
)
from ..grpc.sales_item_service_pb2_grpc import SalesItemServiceServicer
from ..service.SalesItemService import SalesItemService
from ..utils import get_stack_trace

def map_http_status_code_to_grpc_status_code(error: Exception):
    # Map HTTP status code here to
    # respective gRPC status code ...
    # Mapping info is available here:
    # https://cloud.google.com/apis/design/errors#error_model
    return code_pb2.INTERNAL

def create_status_from(error: Exception) -> status_pb2.Status:
    detail = any_pb2.Any()

    if isinstance(error, SalesItemServiceError):
        grpc_status_code = map_http_status_code_to_grpc_status_code(error)
        message = error.message

        detail.Pack(
            ErrorDetails(
                code=error.code,
                description=error.description,
                # get_stack_trace returns stack trace only
                # when environment is not production
                # otherwise it returns None
                stackTrace=get_stack_trace(error.cause),
            )
        )
    elif isinstance(error, ValidationError):
        grpc_status_code = code_pb2.INVALID_ARGUMENT
        message = 'Request validation failed'
        detail.Pack(
            ErrorDetails(
                code='RequestValidationError', description=str(error)
            )
        )
    else:
        grpc_status_code = code_pb2.INTERNAL
        message = 'Unspecified internal error'
        detail.Pack(
            ErrorDetails(
                code='UnspecifiedError',
                description=str(error),
                stackTrace=get_stack_trace(error),
            )
        )

    return status_pb2.Status(
        code=grpc_status_code,
        message=message,
        details=[detail],
    )

class GrpcSalesItemController(SalesItemServiceServicer):
    __sales_item_service: SalesItemService = Provide['sales_item_service']

    def createSalesItem(
        self, input_sales_item: InputSalesItem, context
    ) -> OutputSalesItem:
        try:
            input_sales_item_dict = proto_to_dict(input_sales_item)

            input_sales_item = PydanticInputSalesItem.parse_obj(
                input_sales_item_dict
            )

            output_sales_item_dict = (
                self.__sales_item_service.create_sales_item(
                    input_sales_item
                ).dict()
            )

            output_sales_item = OutputSalesItem()

            json_format.ParseDict(
                output_sales_item_dict, output_sales_item
            )

            return output_sales_item
        except Exception as error:
            self.__abort_with(error, context)

    def getSalesItems(
        self, get_sales_items_arg: GetSalesItemsArg, context
    ) -> OutputSalesItems:
        try:
            # NOTE! Here we don't use the input message
            # 'get_sales_items_arg' because our current
            # business logic does not support it
            output_sales_items = (
                self.__sales_item_service.get_sales_items()
            )

            output_sales_items = [
                json_format.ParseDict(
                    output_sales_item.dict(), OutputSalesItem()
                )
                for output_sales_item in output_sales_items
            ]

            return OutputSalesItems(salesItems=output_sales_items)
        except Exception as error:
            self.__abort_with(error, context)

    def getSalesItem(self, id: Id, context):
        try:
            output_sales_item_dict = (
                self.__sales_item_service.get_sales_item(id.id).dict()
            )

            output_sales_item = OutputSalesItem()

            json_format.ParseDict(
                output_sales_item_dict, output_sales_item
            )

            return output_sales_item
        except Exception as error:
            self.__abort_with(error, context)

    def updateSalesItem(self, sales_item_update: SalesItemUpdate, context):
        try:
            id_ = sales_item_update.id
            sales_item_update_dict = proto_to_dict(sales_item_update)

            sales_item_update = PydanticInputSalesItem.parse_obj(
                sales_item_update_dict
            )

            self.__sales_item_service.update_sales_item(
                id_, sales_item_update
            )

            return Nothing()
        except Exception as error:
            self.__abort_with(error, context)

    def deleteSalesItem(self, id: Id, context):
        try:
            self.__sales_item_service.delete_sales_item(id.id)
            return Nothing()
        except Exception as error:
            self.__abort_with(error, context)

    @staticmethod
    def __abort_with(error: Exception, context):
        status = create_status_from(error)
        context.abort_with_status(rpc_status.to_status(status))
```

对于生产环境，你需要为每个 gRPC 过程实现添加审计日志、指标更新和授权。你可以为此目的使用装饰器。同时，在处理错误时，记得进行必要的审计日志记录（例如，记录错误请求的审计日志）和更新失败相关的指标。

以下是 gRPC 服务器代码：

图 8.46. controllers/app_grpc.py

```
import os
from concurrent import futures

import grpc

from .controllers.GrpcSalesItemController import GrpcSalesItemController
from .DiContainer import DiContainer
from .grpc.sales_item_service_pb2_grpc import (
    add_SalesItemServiceServicer_to_server,
)

di_container = DiContainer()

# Remove this setting of env variable for production code!
# mysql+pymysql://root:password@localhost:3306/salesitemservice
# mongodb://localhost:27017/salesitemservice
os.environ[
    'DATABASE_URL'
] = 'mysql+pymysql://root:password@localhost:3306/salesitemservice'

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_SalesItemServiceServicer_to_server(
        GrpcSalesItemController(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

serve()
```

你可以从 *salesitemservice* 目录的上级目录运行以下命令来启动服务器：

```
python -m salesitemservice.app_grpc
```

以下是使用上述服务器执行操作的 gRPC 客户端示例：

图 8.47. grpc_client.py

```
from grpc_status import rpc_status

import grpc

from .grpc.sales_item_service_pb2 import (
    ErrorDetails,
    GetSalesItemsArg,
    Id,
    Image,
    InputSalesItem,
    SalesItemUpdate,
)
from .grpc.sales_item_service_pb2_grpc import SalesItemServiceStub


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        sales_item_service = SalesItemServiceStub(channel)
        input_sales_item = InputSalesItem(
            name='Test',
            priceInCents=950,
            images=[
                Image(id=11, rank=1, url='http://server.com/images/1')
            ],
        )

        try:
            sales_item = sales_item_service.createSalesItem(
                input_sales_item
            )

            id_ = sales_item.id
            print(f'Sales item with id {id_} created')

            sales_items_response = sales_item_service.getSalesItems(
                GetSalesItemsArg()
            )

            print(
                f'Nbr of sales items fetched: {len(sales_items_response.salesItems)}'
            )
```

sales_item_service.updateSalesItem(
    SalesItemUpdate(
        id=id_,
        name='Test 2',
        priceInCents=1950,
        images=[
            Image(
                id=11, rank=1, url='http://server.com/images/1'
            )
        ],
    )
)
print(f'Sales item with id {id_} updated')
sales_item = sales_item_service.getSalesItem(Id(id=id_))
print(f'Sales item named {sales_item.name} fetched')
sales_item_service.deleteSalesItem(Id(id=id_))
print(f'Sales item with id {id_} deleted')
except grpc.RpcError as error:
    status = rpc_status.from_call(error)
    if status:
        print(f'gRPC status code: {status.code}')
        for detail in status.details:
            error_details = ErrorDetails()
            detail.Unpack(error_details)
            print(f'Error code: {error_details.code}')
            print(f'Error message: {status.message}')
            print(
                f'Error description: {error_details.description}'
            )
    else:
        print(str(error))

if __name__ == '__main__':
    run()

你可以从 *salesitemservice* 目录的上级目录运行客户端，使用以下命令：

```
python -m salesitemservice.grpc_client
```

## 8.2.2：异步 API 设计原则

> 当请求是仅发送请求（即“发后即忘”，不期望收到响应）或响应不是立即需要时，应使用异步 API。

### 8.2.2.1：仅请求异步 API 设计

在仅请求异步 API 中，请求发送方不期望收到响应。此类 API 通常使用消息代理实现。请求发送方将向消息代理中的一个主题发送 JSON 格式的请求，请求接收方异步消费该请求。

例如，可以在请求中使用 `procedure` 属性来指定不同的 API 端点。你可以随意命名 `procedure` 属性，例如 `action`、`operation`、`apiEndpoint` 等。过程的参数可以在 `parameters` 属性中提供。下面是一个 JSON 格式的请求示例：

```json
{
  "procedure": "<procedure name>",
  "parameters": {
    "parameterName1": <parameter value>,
    "parameterName2": <parameter value>,
    // ...
  }
}
```

让我们以一个实现仅请求异步 API 并处理邮件发送的电子邮件发送微服务为例。我们首先为该微服务定义一个消息代理主题。主题应以微服务命名，例如：`email-sending-service`。在 `email-sending-service` 中，我们为发送电子邮件的 API 端点定义以下请求模式：

```json
{
  "procedure": "sendEmailMessage",
  "parameters": {
    "fromEmailAddress": "...",
    "toEmailAddresses": ["...", "...", ...],
    "subject": "...",
    "message": "..."
  }
}
```

下面是一个示例请求，其他微服务可以将其发送到消息代理的 `email-sending-service` 主题：

```json
{
  "procedure": "sendEmailMessage",
  "parameters": {
    "fromEmailAddress": "sender@domain.com",
    "toEmailAddresses": ["receiver@domain.com"],
    "subject": "Status update",
    "message": "Hi, Here is my status update ..."
  }
}
```

### 8.2.2.2：请求-响应异步 API 设计

请求-响应异步 API 微服务接收来自其他微服务的请求，然后异步生成响应。请求-响应异步 API 通常使用消息代理实现。请求发送方将请求发送到一个主题，请求接收方异步消费该请求，然后向一个或多个消息代理主题生成响应。每个参与的微服务都应在消息代理中拥有一个以该微服务命名的主题。

请求格式与前面定义的相同，但响应使用 `response` 属性代替 `parameters` 属性。因此，响应具有以下格式：

```json
{
    "procedure": "<procedure name>",
    "response": {
        "propertyName1": <property value>,
        "propertyName2": <property value>,
        // ...
    }
}
```

下面是一个示例，其中 `loan-application-service` 请求 `loan-eligibility-assessment-service` 评估贷款资格。`loan-application-service` 向消息代理的 `loan-eligibility-assessment-service` 主题发送以下 JSON 格式请求：

```json
{
    "procedure": "assessLoanEligibility",
    "parameters": {
        "userId": 123456789012,
        "loanApplicationId": 5888482223,
        // 其他参数...
    }
}
```

`loan-eligibility-assessment-service` 通过向消息代理的 `loan-application-service` 主题发送以下 JSON 格式响应来回复上述请求：

```json
{
    "procedure": "assessLoanEligibility",
    "response": {
        "loanApplicationId": 5888482223,
        "isEligible": true,
        "amountInDollars": 10000,
        "interestRate": 9.75,
        "termInMonths": 120
    }
}
```

下面是贷款申请被拒绝时的示例响应：

```json
{
    "procedure": "assessLoanEligibility",
    "response": {
        "loanApplicationId": 5888482223,
        "isEligible": false
    }
}
```

或者，请求和响应消息可以被视为带有某些数据的事件。当我们在微服务之间发送事件时，我们拥有一个*事件驱动架构*。在事件驱动架构中，我们必须决定软件系统在消息代理中是使用单个主题还是多个主题。如果使用一个由软件系统中所有微服务共享的单个主题，那么每个微服务将消费消息代理中的每条消息，并决定是否应该对其采取行动。这种方法是合适的，除非向消息代理生成大型事件。当生成大型事件时，即使微服务不需要对其做出反应，也必须消费这些大型事件。当微服务数量也很多时，这将不必要地消耗大量网络带宽。另一个极端是在消息代理中为每个微服务创建一个主题。如果必须将大型事件生成到多个主题以由多个微服务处理，这种方法会导致额外的网络带宽消耗。你也可以创建一个混合模型，其中包含一个广播主题以及特定微服务的单独主题。

以下是前面作为事件编写的请求和响应消息：

```json
{
    "event": "assessLoanEligibility",
    "data": {
        "userId": 123456789012,
        "loanApplicationId": 5888482223,
        // ...
    }
}
```

```json
{
    "event": "LoanApproved",
    "data": {
        "loanApplicationId": 5888482223,
        "isEligible": true,
        "amountInDollars": 10000,
        "interestRate": 9.75,
        "termInMonths": 120
    }
}
```

```json
{
  "procedure": "LoanRejected",
  "response": {
    "loanApplicationId": 5888482223,
    "isEligible": false
  }
}
```

## 9：数据库与数据库原则

本章介绍选择和使用数据库的原则。针对以下数据库类型提出了原则：

- 关系型数据库
- 文档数据库
- 键值数据库
- 宽列数据库
- 搜索引擎

关系型数据库也称为 SQL 数据库，因为访问关系型数据库是通过发出 SQL 语句来完成的。其他类型的数据库称为 NoSQL 数据库，因为它们要么完全不支持 SQL，要么只支持 SQL 的一个子集，可能还有一些附加和修改。

## 9.1：关系型数据库

> *关系型数据库是适用于多种需求的通用数据库。如果你不清楚自己对数据库的所有需求，可以选择关系型数据库。*

例如，如果你不知道现在或将来需要什么样的数据库查询，你应该考虑使用适合各种查询的关系型数据库。

### 9.1.1：关系型数据库的结构

关系型数据库中的数据按以下层次结构组织：

- 逻辑数据库/模式
  - 表
    - 列

表由列和行组成。数据库中的数据存储为表中的行。每行在表中的每一列都有一个值。如果某行对于特定列没有值，则使用特殊的 `NULL` 值。你可以指定列是否允许空值。

一个微服务应该拥有一个逻辑数据库（或模式）。一些关系型数据库默认提供一个逻辑数据库（或模式），而在其他数据库中，你必须自己创建逻辑数据库（或模式）。

## 9.1.2：使用对象关系映射器（ORM）原则

> 使用对象关系映射器（ORM）以避免编写SQL的需要，并防止你的微服务可能受到SQL注入攻击。使用ORM可以将数据库行自动映射为可序列化为JSON的对象。

本节将使用SQLAlchemy库的ORM功能进行示例说明。ORM使用实体作为数据库模式的构建块。微服务中的每个实体类都对应数据库中的一张表。实体和数据库表应使用相同的名称，但表名应使用复数形式。下面是一个`SalesItem`实体类的示例。在定义实际的实体类之前，我们需要声明一个`Base`实体类：

图 9.1. Base.py

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

图 9.2. SalesItem.py

```python
from sqlalchemy import BigInteger, Double, String
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from Base import Base

class SalesItem(Base):
    __tablename__ = 'salesitems'

    id: Mapped[int] = mapped_column(
        BigInteger(), primary_key=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(String(256))
    price: Mapped[float] = mapped_column(Double())
```

相关表名使用复数形式，例如`SalesItem`实体存储在名为`salesitems`的表中。在本书中，我使用不区分大小写的数据库标识符，并将所有标识符写为小写。数据库的大小写敏感性取决于数据库及其运行的操作系统。例如，MySQL仅在Linux系统上区分大小写。

实体的属性映射到实体表的列，这意味着`salesitems`表包含以下列：

- id
- name
- price

每个实体表都应定义一个主键。主键在表的每一行中必须是唯一的。在上面的示例中，我们为`mapped_column`函数提供了`primary_key=True`参数，以定义该列应为主键，并为每一行包含唯一值。我们还定义了数据库应为`id`列自动生成一个自动递增的值（`autoincrement`参数的默认值为True，因此在后续示例中不再指定）。

ORM可以根据代码中的实体规范创建数据库表。下面是ORM生成的用于创建存储`SalesItem`实体的表的示例SQL语句：

```sql
CREATE TABLE salesitems (
    id BIGINT NOT NULL AUTO_INCREMENT,
    name VARCHAR(256) NOT NULL,
    price DOUBLE NOT NULL,
    PRIMARY KEY (id)
)
```

表的列可以指定为唯一且可为空。下面是一个示例，我们定义`salesitems`表中`name`列的值必须是唯一的。我们不想存储名称为空的销售项目，并且希望存储具有唯一名称的销售项目。我们还添加了一个可为空的`description`列。

```python
from typing import Optional

from Base import Base
from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

class SalesItem(Base):
    __tablename__ = 'salesitems'
    __table_args__ = (UniqueConstraint('name'))

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    price: Mapped[float] = mapped_column(Double())
    description: Mapped[Optional[str]] = mapped_column(String(1024))
```

ORM为创建上述定义的`salesitems`表生成以下SQL：

```sql
CREATE TABLE salesitems (
    id BIGINT NOT NULL AUTO_INCREMENT,
    name VARCHAR(256) NOT NULL,
    price DOUBLE NOT NULL,
    description VARCHAR(1024),
    PRIMARY KEY (id),
    UNIQUE (name)
)
```

让我们尝试创建一个实体并将其存储到数据库中。首先，我们需要创建一个数据库引擎：

```python
import os

from sqlalchemy import create_engine

engine = create_engine(os.environ.get('DATABASE_URL'))
```

为了演示目的，我们可以使用内存中的SQLite数据库：

```python
from sqlalchemy import create_engine

engine = create_engine('sqlite://', echo=True)
```

`echo=True`参数定义ORM生成和使用的SQL语句将被记录到标准输出。这对于调试非常方便。创建数据库引擎后，我们必须在数据库中创建数据库表。可以使用以下命令完成：

```python
from Base import Base

Base.metadata.create_all(engine)
```

接下来，我们可以创建一个销售项目并将其持久化到数据库中：

```python
from SalesItem import SalesItem
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

sales_item = SalesItem(name='Sample sales item', price='10')

try:
    with Session(engine) as session:
        session.add(sales_item)
        session.commit()
except SQLAlchemyError:
    # 处理错误
```

ORM将代表你生成并执行所需的SQL语句。下面是ORM生成的用于持久化销售项目的示例SQL语句（请记住，数据库会自动生成id列）。

```sql
INSERT INTO salesitems (name, price)
VALUES ('Sample sales item', 10)
```

你可以在数据库中搜索创建的销售项目：

```python
from SalesItem import SalesItem
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

statement = select(SalesItem).where(SalesItem.id == sales_item.id)

try:
    with Session(engine) as session:
        sales_item = session.scalars(statement).one()
except SQLAlchemyError:
    # 处理错误
```

对于上述操作，ORM将生成以下SQL查询：

```sql
SELECT id, name, price, description FROM salesitems WHERE id = 1
```

然后你可以修改实体并使用`commit`来更新数据库：

```python
from SalesItem import SalesItem
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

try:
    with Session(engine) as session:
        sales_item = session.get(SalesItem, 1)
        sales_item.price = 20
        session.commit()
except SQLAlchemyError:
    # 处理错误
```

对于上述操作，ORM将生成以下SQL语句：

```sql
UPDATE salesitems SET price = 20 WHERE id = 1
```

最后，你可以删除销售项目：

```python
try:
    with Session(engine) as session:
        sales_item = session.get(SalesItem, 1)
        session.delete(sales_item)
        session.commit()
except SQLAlchemyError:
    # 处理错误
```

ORM将执行以下SQL语句：

```sql
DELETE FROM salesitems WHERE id = 1
```

假设你的微服务执行的SQL查询在WHERE子句中不包含主键列。在这种情况下，数据库引擎必须执行全表扫描来查找所需的行。假设你想查询价格低于10的销售项目。这可以通过以下查询实现：

```python
# price = ...

statement = select(SalesItem).where(SalesItem.price < price)

try:
    with Session(engine) as session:
        sales_items = session.scalars(statement).all()
except SQLAlchemyError:
    # 处理错误
```

数据库引擎必须执行全表扫描来查找所有价格列值低于`price`变量值的销售项目。如果数据库很大，这可能会很慢。如果你经常执行上述查询，你应该通过创建索引来优化这些查询。为了使上述查询快速，我们必须为`price`列创建索引：

```python
from typing import Optional

from Base import Base
from sqlalchemy import BigInteger, Double, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

class SalesItem(Base):
    __tablename__ = 'salesitems'
    __table_args__ = (UniqueConstraint('name'),)

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    price: Mapped[float] = mapped_column(Double(), index=True)
    description: Mapped[Optional[str]] = mapped_column(String(1024))
```

## 9.1.3：实体/表关系

关系型数据库中的表可以与其他表建立关系。关系主要有三种类型：

- 一对一
- 一对多
- 多对多

## 9.1.3.1：一对一/一对多关系

本节我们将重点讨论一对一和一对多关系。在一对一关系中，一个表中的一行可以与另一个表中的一行建立关系。在一对多关系中，一个表中的一行可以与另一个表中的多行建立关系。

让我们以一个*订单服务*为例，该服务可以将订单存储在数据库中。每个订单由一个或多个订单项组成。订单项包含了所购销售商品的信息。

```python
from Base import Base
from sqlalchemy import BigInteger, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

class Order(Base):
    __tablename__ = 'orders'

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True)
    # 其他字段 ...
    items: Mapped[list['OrderItem']] = relationship(lazy='joined')

class OrderItem(Base):
    __tablename__ = 'orderitems'
    __table_args__ = (
        PrimaryKeyConstraint('orderId', 'id', name='orderitems_pk'),
    )

    id: Mapped[int]
    salesitemid: Mapped[int] = mapped_column(BigInteger())
    orderid: Mapped[int] = mapped_column(ForeignKey('orders.id'))
```

订单存储在 `orders` 表中，订单项存储在 `orderitems` 表中，该表包含一个名为 `orderid` 的连接列。通过这个连接列，我们可以将特定的订单项映射到特定的订单。每个订单项恰好映射到一个销售商品。因此，`orderitems` 表还包含一个名为 `salesitemid` 的列。销售商品存储在另一个独立微服务的不同数据库中。

以下是 ORM 为创建 `orderitems` 表生成的 SQL 语句。一对一和一对多关系体现在外键约束中：

```sql
CREATE TABLE orderitems (
    id INTEGER NOT NULL,
    salesitemid BIGINT NOT NULL,
    orderid BIGINT NOT NULL,
    CONSTRAINT orderitems_pk PRIMARY KEY (orderid, id),
    FOREIGN KEY (orderid) REFERENCES orders (id)
)
```

以下是 ORM 执行的 SQL 查询，用于获取 id 为 123 的订单及其订单项：

```sql
SELECT o.id, oi.id
FROM orders o
LEFT JOIN orderitems oi ON o.id = oi.orderid
WHERE o.id = 123
```

## 9.1.3.2：多对多关系

在多对多关系中，一个实体与另一个类型的多个实体建立关系，而这些实体又与第一个实体类型的多个实体建立关系。例如，一个学生可以参加多门课程，一门课程也可以有众多学生参加。

假设我们有一个服务，用于在数据库中存储学生和课程实体。每个学生实体包含该学生参加过的课程。同样，每个课程实体包含参加过该课程的学生列表。我们有一个多对多关系，即一个学生可以参加多门课程，多个学生可以参加一门课程。这意味着必须创建一个额外的关联表 `studentcourse`。这个新表将特定的学生映射到特定的课程。

```python
from sqlalchemy import BigInteger, Column, ForeignKey, Table
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

class Base(DeclarativeBase):
    pass

student_course_assoc_table = Table(
    'studentcourse',
    Base.metadata,
    Column('studentid', ForeignKey('students.id'), primary_key=True),
    Column('courseid', ForeignKey('courses.id'), primary_key=True),
)

class Student(Base):
    __tablename__ = 'students'

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True)
    # 其他字段...

    courses: Mapped[list['Course']] = relationship(
        secondary=student_course_assoc_table, back_populates='students'
    )

class Course(Base):
    __tablename__ = 'courses'

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True)
    # 其他字段...

    students: Mapped[list[Student]] = relationship(
        secondary=student_course_assoc_table, back_populates='courses'
    )
```

ORM 除了创建 `students` 和 `courses` 表外，还会创建 `studentcourse` 映射表：

```sql
CREATE TABLE studentcourse (
    studentid BIGINT NOT NULL,
    courseid BIGINT NOT NULL,
    PRIMARY KEY (studentid, courseid),
    FOREIGN KEY (studentid) REFERENCES students (id),
    FOREIGN KEY (courseid) REFERENCES courses (id)
)
```

以下是 ORM 执行的示例 SQL 查询，用于获取 id 为 123 的用户参加的课程：

```sql
SELECT s.id, c.id
FROM students s
LEFT JOIN studentcourse sc ON s.id = sc.studentid
LEFT JOIN courses c ON c.id = sc.courseid
WHERE s.id = 123
```

以下是 ORM 执行的示例 SQL 查询，用于获取 id 为 123 的课程的学生：

```sql
SELECT c.id, s.id
FROM courses c
LEFT JOIN studentcourse sc ON c.id = sc.courseid
LEFT JOIN students s ON s.id = sc.studentid
WHERE c.id = 123
```

在实际场景中，我们不一定需要或应该在单个微服务内实现多对多的数据库关系。例如，上述处理学生和课程的服务在课程和学生的抽象层面上违反了*单一职责原则*。（但是，如果我们创建一个*学校*微服务，那么学生和课程表就会在同一个微服务中）我们应该为学生创建一个单独的微服务，为课程创建另一个单独的微服务。这样，在单个微服务内就不会有数据库表之间的多对多关系了。

## 9.1.3.3：销售商品仓储库示例

让我们使用 SQLAlchemy 的 ORM 功能，为上一章定义的销售商品服务 API 定义一个 `SalesItemRepository` 实现。首先定义 `Base`、`SalesItem` 和 `SalesItemImage` 实体：

```python
# 图 9.3. entities/Base.py
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

```python
# 图 9.4. entities/SalesItem.py
from sqlalchemy import BigInteger, Double, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .Base import Base
from .SalesItemImage import SalesItemImage

class SalesItem(Base):
    __tablename__ = 'salesitems'

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True)
    createdAtTimestampInMs: Mapped[int] = mapped_column(BigInteger())
    name: Mapped[str] = mapped_column(String(256))
    priceInCents: Mapped[int]
    images: Mapped[list[SalesItemImage]] = relationship(
        cascade='all, delete-orphan', lazy='joined'
    )
```

```python
# 图 9.5. entities/SalesItemImage.py
from sqlalchemy import ForeignKey, PrimaryKeyConstraint, String
from sqlalchemy.orm import Mapped, mapped_column

from .Base import Base

class SalesItemImage(Base):
    __tablename__ = 'salesitemimages'
    __table_args__ = (
        PrimaryKeyConstraint(
            'salesItemId', 'id', name='salesitemimages_pk'
        ),
    )

    id: Mapped[int]
    rank: Mapped[int]
    url: Mapped[str] = mapped_column(String(2084))
    salesItemId: Mapped[int] = mapped_column(ForeignKey('salesitems.id'))
```

以下是 `OrmSalesItemRepository` 的实现：

### 图 9.6. repositories/OrmSalesItemRepository.py

```python
import os
import time

from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from ..dtos.InputSalesItem import InputSalesItem
from ..entities.Base import Base
from ..entities.SalesItem import SalesItem
from ..errors.DatabaseError import DatabaseError
from ..errors.EntityNotFoundError import EntityNotFoundError
from ..utils import to_entity_dict
from .SalesItemRepository import SalesItemRepository


class OrmSalesItemRepository(SalesItemRepository):
    def __init__(self):
        try:
            engine = create_engine(os.environ.get('DATABASE_URL'))
            self.__SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=engine
            )
            Base.metadata.create_all(bind=engine)
        except SQLAlchemyError as error:
            # 记录错误
            raise error

    def save(self, input_sales_item: InputSalesItem) -> SalesItem:
        try:
            with self.__SessionLocal() as db_session:
                sales_item = SalesItem(**to_entity_dict(input_sales_item))
                sales_item.createdAtTimestampInMs = (
                    time.time_ns() / 1_000_000
                )
                db_session.add(sales_item)
                db_session.commit()
                db_session.refresh(sales_item)
                return sales_item
        except SQLAlchemyError as error:
            raise DatabaseError(error)

    def find_all(self) -> list[SalesItem]:
        try:
            with self.__SessionLocal() as db_session:
                return db_session.scalars(select(SalesItem)).unique().all()
        except SQLAlchemyError as error:
            raise DatabaseError(error)

    def find(self, id_: str) -> SalesItem | None:
        try:
            with self.__SessionLocal() as db_session:
                return db_session.get(SalesItem, id_)
        except SQLAlchemyError as error:
            raise DatabaseError(error)

    def update(self, id_: str, sales_item_update: InputSalesItem) -> None:
        try:
```

with self.__SessionLocal() as db_session:
    sales_item = db_session.get(SalesItem, id_)

    if sales_item is None:
        raise EntityNotFoundError('Sales item', id_)

    new_sales_item = SalesItem(
        **to_entity_dict(sales_item_update)
    )

    sales_item.name = new_sales_item.name
    sales_item.priceInCents = new_sales_item.priceInCents
    sales_item.images = new_sales_item.images
    db_session.commit()
except SQLAlchemyError as error:
    raise DatabaseError(error)

def delete(self, id_: str) -> None:
    try:
        with self.__SessionLocal() as db_session:
            sales_item = db_session.get(SalesItem, id_)
            if sales_item is not None:
                db_session.delete(sales_item)
                db_session.commit()
    except SQLAlchemyError as error:
        raise DatabaseError(error)

## 9.1.4：使用参数化 SQL 语句原则

> 如果你没有使用 ORM 进行数据库访问，请使用参数化 SQL 语句来防止潜在的 SQL 注入攻击。

让我们使用 Python MySQL 连接器库 *mysql-connector-python*。首先，我们向 salesitems 表插入数据：

```python
from mysql.connector import connect, Error

connection = None

try:
    connection = connect(
        host='...',
        database='...',
        user='...',
        password='...'
    )

    cursor = connection.cursor(prepared=True)
    sql_statement = 'INSERT INTO salesitems (name, price) VALUES (%s, %s)'
    cursor.execute(sql_statement, ('Sample sales item 1', 20))
    connection.commit()
except Error as error:
    # Handle error
finally:
    if connection:
        connection.close()
```

上述 SQL 语句中的 %s 是参数化 SQL 语句中参数的占位符。execute 方法的第二个参数是一个元组，包含参数值。当数据库引擎接收到参数化查询时，它会将 SQL 语句中的占位符替换为提供的参数值。

接下来，我们可以更新 salesitems 表中的一行。下面的示例将 id 为 123 的销售商品的价格更改为 20：

```python
from mysql.connector import connect, Error

connection = None

try:
    connection = connect(
        host='...',
        database='...',
        user='...',
        password='...'
    )

    cursor = connection.cursor(prepared=True)
    sql_statement = 'UPDATE salesitems SET PRICE = %s WHERE id = %s'
    cursor.execute(sql_statement, (20, 123))
    connection.commit()
except Error as error:
    # Handle error
finally:
    if connection:
        connection.close()
```

让我们执行一个 SELECT 语句来获取价格超过 20 的销售商品：

```python
from mysql.connector import connect, Error

connection = None

try:
    connection = connect(
        host='...',
        database='...',
        user='...',
        password='...'
    )

    cursor = connection.cursor(prepared=True)
    sql_statement = 'SELECT id, name, price FROM salesitems WHERE price >= %s'
    cursor.execute(sql_statement, (20,))
    result = cursor.fetchall()
except Error as error:
    # Handle error
finally:
    if connection:
        connection.close()
```

在 SQL SELECT 语句中，你不能在所有地方都使用参数。你可以在 WHERE 子句中将它们用作值占位符。如果你想在 SQL SELECT 语句的其他部分使用用户提供的数据，你需要使用字符串拼接。你不应该在没有进行净化的情况下拼接用户提供的数据，因为这会为 SQL 注入攻击打开可能性。假设你允许微服务客户端指定一个排序列：

```python
import string

class ValidateColNameError(Exception):
    pass

def try_validate_col_name(column_name: str) -> str:
    allowed_chars = string.ascii_letters + string.digits + '_' + '$'

    if all(
        col_name_char in allowed_chars for col_name_char in column_name
    ):
        return column_name

    raise ValidateColNameError()

sort_column_name = # Unvalidated data got from client

sql_query = (
    'SELECT id, name, price FROM salesitems ORDER BY '
    + try_validate_col_name(sort_column_name)
)

# ...
```

如上所示，你需要验证 sort_column 的值，使其仅包含 MySQL 列名的有效字符。如果你需要从客户端获取排序方向，你应该验证该值是 ASC 还是 DESC。在下面的示例中，我们假设存在一个 validateSortDirection 函数：

```python
class ValidateSortDirError(Exception):
    pass

def try_validate_sort_dir(sort_dir: str) -> str:
    lower_case_sort_dir = sort_dir.lower()

    if lower_case_sort_dir == 'asc' or lower_case_sort_dir == 'desc':
        return sort_dir

    raise ValidateSortDirError()

sort_column_name = # Unvalidated data got from client
sort_direction = # Unvalidated data got from client

validated_sort_col_name = try_validate_col_name(sort_column_name)
validated_sort_dir = try_validate_sort_dir(sort_direction)

sql_query = (
    'SELECT id, name, price'
    'FROM salesitems'
    'ORDER BY'
    f'{validated_sort_col_name}'
    f'{validated_sort_dir}'
)

# ...
```

当你从客户端获取 MySQL 查询的 LIMIT 子句的值时，你必须验证这些值是整数且在有效范围内。不要允许客户端提供随机的、非常大的值。在下面的示例中，我们假设存在两个验证函数：try_validate_row_offset 和 try_validate_row_count。如果验证失败，这些验证函数将引发异常。

```python
def try_validate_row_offset(row_offset: str) -> str:
    # Implement ...

def try_validate_row_count(row_limit: str) -> str:
    # Implement ...

row_offset = # Unvalidated data got from client
row_count = # Unvalidated data got from client

validated_row_offset = try_validate_row_offset(row_offset)
validated_row_count = try_validate_row_count(row_count)

sql_query = (
    'SELECT id, name, price'
    'FROM salesitems'
    f'LIMIT {validated_row_offset}, {validated_row_count}'
)

# ...
```

当你从客户端获取所需列名的列表时，你必须验证每个列名都是有效的列标识符：

```python
column_names = # Unvalidated data got from client
validated_col_names = [try_validate_col_name(column_name) for column_name in column_names]
sql_query = f'SELECT {", ".join(validated_col_names)} FROM salesitems'

# ...
```

让我们使用参数化 SQL 实现 SalesItemRepository：

## 图 9.7. repositories/ParamSqlSalesItemRepository.py

```python
import os
import time
from typing import Any

from mysql.connector import connect
from mysql.connector.errors import Error

from ..dtos.InputSalesItem import InputSalesItem
from ..entities.SalesItem import SalesItem
from ..entities.SalesItemImage import SalesItemImage
from ..errors.DatabaseError import DatabaseError
from ..errors.EntityNotFoundError import EntityNotFoundError
from ..utils import to_entity_dict
from .SalesItemRepository import SalesItemRepository


class ParamSqlSalesItemRepository(SalesItemRepository):
    def __init__(self):
        try:
            self.__conn_config = self.__try_create_conn_config()
            self.__try_create_db_tables_if_needed()
        except Exception as error:
            # Log error
            raise (error)

    def save(self, input_sales_item: InputSalesItem) -> SalesItem:
        connection = None

        try:
            connection = connect(**self.__conn_config)
            cursor = connection.cursor(prepared=True)

            sql_statement = (
                'INSERT INTO salesitems'
                '(createdAtTimestampInMs, name, priceInCents)'
                ' VALUES (%s, %s, %s)'
            )

            created_at_timestamp_in_ms = time.time_ns() / 1_000_000

            cursor.execute(
                sql_statement,
                (
                    created_at_timestamp_in_ms,
                    input_sales_item.name,
                    input_sales_item.priceInCents,
                ),
            )

            id_ = cursor.lastrowid

            self.__try_insert_sales_item_images(
                id_, input_sales_item.images, cursor
            )

            connection.commit()

            return SalesItem(
```

```python
def to_entity_dict(input_sales_item),
    id=id_,
    createdAtTimestampInMs=created_at_timestamp_in_ms,
)
except Error as error:
    raise DatabaseError(error)
finally:
    if connection:
        connection.close()

def find_all(self) -> list[SalesItem]:
    connection = None

    try:
        connection = connect(**self.__conn_config)
        cursor = connection.cursor()

        sql_statement = (
            'SELECT s.id, s.createdAtTimestampInMs, s.name, s.priceInCents,'
            'si.id, si.rank, si.url '
            'FROM salesitems s '
            'LEFT JOIN salesitemimages si ON si.salesItemId = s.id'
        )

        cursor.execute(sql_statement)
        return self.__get_sales_item_entities(cursor)
    except Error as error:
        print(error)
        raise DatabaseError(error)
    finally:
        if connection:
            connection.close()

def find(self, id_: str) -> SalesItem | None:
    if not id_.isnumeric():
        raise EntityNotFoundError('Sales item', id_)

    connection = None

    try:
        connection = connect(**self.__conn_config)
        cursor = connection.cursor(prepared=True)

        sql_statement = (
            'SELECT s.id, s.createdAtTimestampInMs, s.name, s.priceInCents,'
            'si.id, si.rank, si.url '
            'FROM salesitems s '
            'LEFT JOIN salesitemimages si ON si.salesItemId = s.id '
            'WHERE s.id = %s'
        )

        cursor.execute(sql_statement, (id_,))

        sales_item_entities = self.__get_sales_item_entities(cursor)
        return sales_item_entities[0] if sales_item_entities else None
    except Error as error:
        raise DatabaseError(error)
    finally:
        if connection:
            connection.close()

def update(self, id_: str, sales_item_update: InputSalesItem) -> None:
    if not id_.isnumeric():
        raise EntityNotFoundError('Sales item', id_)

    connection = None

    try:
        connection = connect(**self.__conn_config)
        cursor = connection.cursor(prepared=True)

        sql_statement = (
            'UPDATE salesitems SET name = %s, priceInCents = %s '
            'WHERE id = %s'
        )

        cursor.execute(
            sql_statement,
            (
                sales_item_update.name,
                sales_item_update.priceInCents,
                id_,
            ),
        )

        sql_statement = (
            'DELETE FROM salesitemimages WHERE salesItemId = %s'
        )

        cursor.execute(sql_statement, (id_,))

        self.__try_insert_sales_item_images(
            id_, sales_item_update.images, cursor
        )

        connection.commit()
    except Error as error:
        raise DatabaseError(error)
    finally:
        if connection:
            connection.close()

def delete(self, id_: str) -> None:
    if not id_.isnumeric():
        return

    connection = None

    try:
        connection = connect(**self.__conn_config)
        cursor = connection.cursor()

        sql_statement = (
            'DELETE FROM salesitemimages WHERE salesItemId = %s'
        )

        cursor.execute(sql_statement, (id_,))
        sql_statement = 'DELETE FROM salesitems WHERE id = %s'
        cursor.execute(sql_statement, (id_,))
        connection.commit()
    except Error as error:
        raise DatabaseError(error)
    finally:
        if connection.is_connected():
            connection.close()

@staticmethod
def __try_create_conn_config() -> dict[str, Any]:
    database_url = os.environ.get('DATABASE_URL')

    user_and_password = (
        database_url.split('@')[0].split('//')[1].split(':')
    )

    host_and_port = database_url.split('@')[1].split('/')[0].split(':')
    database = database_url.split('/')[3]

    return {
        'user': user_and_password[0],
        'password': user_and_password[1],
        'host': host_and_port[0],
        'port': host_and_port[1],
        'database': database,
        'pool_name': 'salesitems',
        'pool_size': 25,
    }

def __try_create_db_tables_if_needed(self) -> None:
    connection = connect(**self.__conn_config)
    cursor = connection.cursor()

    sql_statement = (
        'CREATE TABLE IF NOT EXISTS salesitems ('
        'id BIGINT NOT NULL AUTO_INCREMENT,'
        'createdAtTimestampInMs BIGINT NOT NULL,'
        'name VARCHAR(256) NOT NULL,'
        'priceInCents INTEGER NOT NULL,'
        'PRIMARY KEY (id)'
        ')'
    )

    cursor.execute(sql_statement)

    sql_statement = (
        'CREATE TABLE IF NOT EXISTS salesitemimages ('
        'id BIGINT NOT NULL,'
        '`rank` INTEGER NOT NULL,'
        'url VARCHAR(2084) NOT NULL,'
        'salesItemId BIGINT NOT NULL,'
        'PRIMARY KEY (salesItemId, id),'
        'FOREIGN KEY (salesItemId) REFERENCES salesitems(id)'
        ')'
    )

    cursor.execute(sql_statement)
    connection.commit()
    connection.close()

def __try_insert_sales_item_images(
    self, sales_item_id: str | int, images, cursor
):
    for image in images:
        sql_statement = (
            'INSERT INTO salesitemimages'
            '(id, `rank`, url, salesItemId)'
            'VALUES (%s, %s, %s, %s)'
        )

        cursor.execute(
            sql_statement,
            (image.id, image.rank, image.url, sales_item_id),
        )

def __get_sales_item_entities(self, cursor):
    id_to_sales_items_dict = dict()

    for (
        id_,
        created_at_timestamp_in_ms,
        name,
        price_in_cents,
        image_id,
        image_rank,
        image_url,
    ) in cursor:
        if id_to_sales_items_dict.get(id_) is None:
            id_to_sales_items_dict[id_] = {
                'id': id_,
                'createdAtTimestampInMs': created_at_timestamp_in_ms,
                'name': name,
                'priceInCents': price_in_cents,
                'images': [],
            }

        if image_id is not None:
            id_to_sales_items_dict[id_]['images'].append(
                SalesItemImage(
                    id=image_id, rank=image_rank, url=image_url
                )
            )

    return [
        SalesItem(**sales_item_dict)
        for sales_item_dict in id_to_sales_items_dict.values()
    ]
```

## 9.1.5：规范化规则

将规范化规则应用于你的数据库设计。

以下列出了三个最基本的规范化规则：

- 第一范式（1NF）
- 第二范式（2NF）
- 第三范式（3NF）

如果一个数据库关系满足第一、第二和第三范式，通常就称其为“规范化”的。

### 9.1.5.1：第一范式（1NF）

第一范式要求在行与列的每个交叉点上，必须存在单个值，而绝不能是值的列表。以销售商品为例，第一范式规定 `price` 列中不能有两个不同的价格值，`name` 列中也不能有多个销售商品名称。如果你需要为一个销售商品提供多个名称，你必须在 `SalesItem` 实体和 `SalesItemName` 实体之间建立一对多关系。在实践中，这意味着你需要从 `SalesItem` 实体类中移除 `name` 属性，并创建一个新的 `SalesItemName` 实体类来存储销售商品的名称。然后，你在 `SalesItem` 实体和 `SalesItemName` 实体之间创建一对多映射。

### 9.1.5.2：第二范式（2NF）

第二范式要求每个非键列完全依赖于主键。假设我们在 `orderitems` 表中有以下列：

- orderid（主键）
- productid（主键）
- orderstate

`orderstate` 列仅依赖于 `orderid` 列，而不是整个主键。`orderstate` 列放错了表。它当然应该放在 `orders` 表中。

### 9.1.5.3：第三范式（3NF）

第三范式要求非键列之间相互独立。
假设我们在 `salesitems` 表中有以下列：

- id（主键）
- name
- price
- category
- discount
```

假设折扣取决于商品类别。此表违反了第三范式，因为非键列`discount`依赖于另一个非键列`category`。列独立性意味着你可以更改任何非键列的值而不影响其他列。如果更改了类别，折扣也需要相应更改，从而违反了第三范式规则。

折扣列应移至一个新的`categories`表中，该表包含以下列：

- id（主键）
- name
- discount

然后，我们应该更新`salesitems`表，使其包含以下列：

- id（主键）
- name
- price
- categoryid（引用`categories`表中`id`列的外键）

## 9.2：文档数据库原则

> *在通常需要将完整文档（例如JSON对象）作为整体存储和检索的情况下，使用文档数据库。*

文档数据库（如MongoDB）适用于存储完整文档。文档通常是一个JSON对象，包含数组和嵌套对象中的信息。文档按原样存储，查询时会获取整个文档。

让我们考虑一个销售商品的微服务。每个销售商品包含id、名称、价格、图片URL和用户评论。

以下是一个销售商品的JSON对象示例：

```
{
    "id": "507f191e810c19729de860ea",
    "category": "Power tools",
    "name": "Sample sales item",
    "price": 10,
    "imageUrls": ["https://url-to-image-1...",
                  "https://url-to-image-2..."],
    "averageRatingInStars": 5,
    "reviews": [
        {
            "reviewerName": "John Doe",
            "date": "2022-09-01",
            "ratingInStars": 5,
            "text": "Such a great product!"
        }
    ]
}
```

文档数据库通常对单个文档有大小限制。因此，上述示例并未直接在文档中存储销售商品图片，而只存储图片的URL。实际图片存储在更适合存储图片的另一个数据存储中，例如Amazon S3。

在创建销售商品的微服务时，我们可以选择文档数据库，因为我们通常存储和访问整个文档。当创建销售商品时，它们以上述形状的JSON对象创建，评论数组为空。当获取销售商品时，整个文档从数据库中检索。当客户为销售商品添加评论时，首先从数据库中获取该销售商品。将新评论附加到评论数组，计算新的平均评分，最后将修改后的文档持久化。

以下是在名为salesItems的MongoDB集合中插入一个销售商品的示例。MongoDB使用术语集合而不是表。一个MongoDB集合可以存储多个文档。

```
from pymongo import MongoClient

URL = "mongodb://localhost:27017"
client = MongoClient(URL)

# Create the database for our example
database = client['sales_item_service']
sales_items_coll = database['sales_items']

sales_items_coll.insert_one({
    'category': 'Power tools',
    'name': 'Sample sales item 1',
    'price': 10,
    'images': ['https://url-to-image-1...',
               'https://url-to-image-2...'],
    'averageRatingInStars': None,
    'reviews': []
})

client.close()
```

你可以使用以下查询查找Power tools类别的销售商品：

```
sales_items = sales_items_coll.find({ 'category': 'Power tools' })
print(sales_items.next())
```

如果客户通常按类别查询销售商品，明智的做法是为该字段创建索引：

```
# 1 means ascending index, -1 means descending index
sales_items_coll.create_index([('category', 1)])
```

当客户想要为销售商品添加新评论时，你首先获取该销售商品的文档：

```
sales_items_coll.find({ '_id': ObjectId('507f191e810c19729de860ea') })
```

然后，你使用现有评分和新评分为`averageRatingInStars`字段计算新值，并将新评论添加到`reviews`数组，最后使用以下命令更新文档：

```
sales_items_coll.update_one(
    {'_id': ObjectId('6527a461bd3c27d2d1822508')},
    {
        '$set': {'averageRatingInStars': 5},
        '$push': {
            'reviews': {
                'reviewerName': 'John Doe',
                'date': '2022-09-01',
                'ratingInStars': 5,
                'text': 'Such a great product!',
            }
        },
    },
)
```

客户可能希望按平均评分降序检索销售商品。因此，你可能希望将索引更改为以下形式：

```
sales_items_coll.create_index( [ ('category', 1), ('averageRatingInStars', -1) ] )
```

客户可以发出请求，例如获取`power tools`类别中评分最高的销售商品。此请求可以使用上述创建的索引通过以下查询来满足：

```
sales_items_coll.find({'category': 'Power tools'}).sort(
    [('averageRatingInStars', -1)]
)
```

让我们使用MongoDB实现`SalesItemRepository`：

### 图 9.8. repositories/MongoDbSalesItemRepository.py

```
import os
import time
from typing import Any

from bson.errors import BSONError, InvalidId
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from ..dtos.InputSalesItem import InputSalesItem
from ..entities.SalesItem import SalesItem
from ..entities.SalesItemImage import SalesItemImage
from ..errors.DatabaseError import DatabaseError
from ..errors.EntityNotFoundError import EntityNotFoundError
from .SalesItemRepository import SalesItemRepository


class MongoDbSalesItemRepository(SalesItemRepository):
    def __init__(self):
        try:
            database_url = os.environ.get('DATABASE_URL')
            self.__client = MongoClient(database_url)
            database_name = database_url.split('/')[3]
            database = self.__client[database_name]
            self.__sales_items = database['salesitems']
        except Exception as error:
            # Log error
            raise (error)

    def save(self, input_sales_item: InputSalesItem) -> SalesItem:
        try:
            sales_item = input_sales_item.dict() | {
                'createdAtTimestampInMs': time.time_ns() / 1_000_000
            }

            self.__sales_items.insert_one(sales_item)
            return self.__create_sales_item_entity(sales_item)

        except PyMongoError as error:
            raise DatabaseError(error)

    def find_all(self) -> list[SalesItem]:
        try:
            sales_items = self.__sales_items.find()
            return [
                self.__create_sales_item_entity(sales_item)
                for sales_item in sales_items
            ]
        except PyMongoError as error:
            raise DatabaseError(error)

    def find(self, id_: str) -> SalesItem | None:
        try:
            sales_item = self.__sales_items.find_one(
                {'_id': ObjectId(id_)}
            )

            return (
                None
                if sales_item is None
                else self.__create_sales_item_entity(sales_item)
            )
        except InvalidId:
            raise EntityNotFoundError('Sales item', id_)
        except (BSONError, PyMongoError) as error:
            raise DatabaseError(error)

    def update(self, id_: str, sales_item_update: InputSalesItem) -> None:
        try:
            self.__sales_items.update_one(
                {'_id': ObjectId(id_)}, {'$set': sales_item_update.dict()}
            )
        except InvalidId:
            raise EntityNotFoundError('Sales item', id_)
        except (BSONError, PyMongoError) as error:
            raise DatabaseError(error)

    def delete(self, id_: str) -> None:
        try:
            self.__sales_items.delete_one({'_id': ObjectId(id_)})
        except InvalidId:
            pass
        except (BSONError, PyMongoError) as error:
            raise DatabaseError(error)

    @staticmethod
    def __create_sales_item_entity(sales_item: dict[str, Any]):
        id_ = sales_item['_id']
        del sales_item['_id']

        images = [
            SalesItemImage(**image) for image in sales_item['images']
        ]

        return SalesItem(
            **(sales_item | {'id': str(id_)} | {'images': images})
        )
```

## 9.3：键值数据库原则

使用键值数据库可以快速实时访问通过键存储的数据。键值存储通常将数据存储在内存中，并支持持久化。

键值数据库的一个简单用例是将其用作关系数据库的缓存。例如，微服务可以将关系数据库的SQL查询结果存储在缓存中。Redis是一个流行的开源键值存储。让我们以Redis为例来缓存SQL查询结果。在下面的示例中，我们假设SQL查询结果以字典形式可用：

import json
from redis import Redis

redis_client = Redis(host='localhost', port=6379, decode_responses=True)
redis_client.set(sql_query_statement, json.dumps(sql_query_result))

可以从 Redis 中获取缓存的 SQL 查询结果：

```
sql_query_result_json = redis_client.get(sql_query_statement)
```

使用 Redis，你可以创建在特定时间后自动过期的键值对。如果你将键值数据库用作缓存，这是一个非常有用的功能。你可能希望缓存的项目在一段时间后过期。

除了纯字符串，Redis 还支持其他数据结构。例如，你可以为一个键存储一个列表、队列或哈希映射。如果你在 Redis 中存储一个队列，你可以将其用作一个简单的单消费者消息代理。以下是一个向消息代理中的主题生产消息的示例：

```
# RPUSH 命令（= 右推）将一条新消息
# 推送到由键 *topic* 标识的列表末尾。
redis_client.rpush(topic, message)
```

以下是一个从消息代理中的主题消费消息的示例：

```
# LPOP 命令（= 左弹）从
# 由键 *topic* 标识的列表开头弹出一条消息
# LPOP 命令会从列表中移除该值
message = redis_client.lpop(topic)
```

## 9.4：宽列数据库原理

> 当你清楚需要执行哪些查询，并且希望这些查询速度很快时，可以使用宽列数据库。

宽列数据库的表结构针对特定查询进行了优化。使用宽列数据库时，存储重复数据是可以接受的，以使查询更快。宽列数据库也具有良好的水平扩展能力。

在本节中，我们以 Apache Cassandra 作为宽列数据库的示例。Cassandra 是一个可扩展的多节点数据库引擎。在 Cassandra 中，表的数据根据表的分区键被划分为分区。分区键由表的一列或多列组成。每个分区存储在单个 Cassandra 节点上。你可以将 Cassandra 视为一个键值存储，其中键是分区键，值是另一个“嵌套”表。“嵌套”表中的行由聚簇列唯一标识，默认按升序排序。如果需要，可以将排序顺序更改为降序。

分区键和聚簇列构成了表的主键。主键唯一标识一行。让我们看一个用于存储特定兴趣点（POI）附近酒店的示例表：

```
CREATE TABLE hotels_by_poi (
    poi_name text,
    hotel_distance_in_meters_from_poi int,
    hotel_id uuid,
    hotel_name text,
    hotel_address text,
    PRIMARY KEY (poi_name, hotel_distance_in_meters_from_poi, hotel_id)
);
```

在上面的示例中，主键由三列组成。第一列（poi_name）始终是分区键。查询时必须给出分区键。否则，查询会很慢，因为 Cassandra 必须执行全表扫描，它不知道数据位于哪个节点。当在 SELECT 语句的 WHERE 子句中给出分区键时，Cassandra 可以找到存储该特定分区数据的相应节点。另外两个主键列 hotel_distance_in_meters_from_poi 和 hotel_id 是聚簇列。它们定义了“嵌套”表中行的顺序和唯一性。

![](img/cbd069395d7b824346b69b1f92e0fb4a_534_0.png)

上图显示，当你给出一个分区键值（poi_name）时，你就可以访问相应的“嵌套”表，其中的行首先按 hotel_distance_in_meters_from_poi（升序）排序，其次按 hotel_id（升序）排序。

现在，酒店客房预订客户端可以轻松地请求服务器执行查询，以查找用户给定的 POI 附近的酒店。以下查询将返回距离 *Piccadilly Circus* POI 最近的前 15 家酒店：

```
SELECT
    hotel_distance_in_meters_from_poi,
    hotel_id,
    hotel_name,
    hotel_address
FROM hotels_by_poi
WHERE poi_name = 'Piccadilly Circus'
LIMIT 15
```

当用户从上述查询结果中选择一家特定酒店时，客户端可以请求执行另一个查询来获取所选酒店的信息。用户希望看到所选酒店附近的其他 POI。对于该查询，我们应该创建另一个表：

```
CREATE TABLE pois_by_hotel_id (
    hotel_id uuid,
    poi_distance_in_meters_from_hotel int,
    poi_id uuid,
    poi_name text,
    poi_address text,
    PRIMARY KEY (hotel_id, poi_distance_in_meters_from_hotel, poi_id)
);
```

现在，客户端可以请求服务器执行查询，为所选酒店（id 为 c5a49cb0-8d98-47e3-8767-c30bc075e529 的酒店）获取最近的 20 个 POI：

```
SELECT
    poi_distance_in_meters_from_hotel,
    poi_id,
    poi_name,
    poi_address
FROM pois_by_hotel_id
WHERE hotel_id = c5a49cb0-8d98-47e3-8767-c30bc075e529
LIMIT 20
```

在实际场景中，用户希望在特定时间段内搜索特定 POI 附近的酒店。服务器应返回在选定时间段内有空房的最近酒店。对于这类查询，我们创建一个额外的表来存储酒店客房可用性：

```
CREATE TABLE availability_by_hotel_id (
    hotel_id uuid,
    accommodation_date date,
    available_room_count counter,
    PRIMARY KEY (hotel_id, accommodation_date)
);
```

每当特定日期的房间被预订或预订被取消时，上述表就会更新。在更新过程中，`available_room_count` 列的值会递减或递增一。

假设已经执行了以下查询：

```
SELECT
    hotel_distance_in_meters_from_poi,
    hotel_id,
    hotel_name,
    hotel_address
FROM hotels_by_poi
WHERE poi_name = 'Piccadilly Circus'
LIMIT 30
```

接下来，我们应该从这 30 家酒店的结果中，找出在 2022 年 9 月 1 日至 2022 年 9 月 3 日期间有空房的酒店。我们不能在 Cassandra 中使用连接，但我们可以执行以下查询，其中我们明确列出了上述查询返回的酒店 ID：

```
SELECT hotel_id, MIN(available_room_count)
FROM availability_by_hotel_id
WHERE hotel_id IN (在此列出 30 个 hotel_ids...) AND
      accommodation_date >= '2022-09-01' AND
      accommodation_date <= '2022-09-03'
GROUP BY hotel_id
LIMIT 15
```

上述查询的结果是，我们得到了一个最多包含 15 家酒店的列表，其中列出了最小可用房间数。我们可以将最小可用房间数为一或更多的最多 15 家酒店列表返回给用户。

如果 Cassandra 的查询语言支持 `HAVING` 子句（目前不支持），我们本可以发出以下查询来获得我们想要的结果：

```
SELECT hotel_id, MIN(available_room_count)
FROM availability_by_hotel_id
WHERE hotel_id IN (在此列出 30 个 hotel_ids...) AND
      accommodation_date >= '2022-09-01' AND
      accommodation_date <= '2022-09-03'
GROUP BY hotel_id
HAVING MIN(available_room_count) >= 1
LIMIT 15
```

宽列数据库在存储来自物联网设备和传感器的时间序列数据时也很有用。以下是一个用于电信网络分析系统中存储测量数据的表定义：

```
CREATE TABLE measurements (
    measure_name text,
    dimension_name text,
    aggregation_period text,
    measure_timestamp timestamp,
    measure_value double,
    dimension_value text,
    PRIMARY KEY ((measure_name, dimension_name, aggregation_period),
                 measure_timestamp,
                 measure_value,
                 dimension_value)
) WITH CLUSTERING ORDER BY (
    measure_timestamp DESC,
    measure_value DESC,
    dimension_value ASC
);
```

在上面的表中，我们定义了一个包含三列的*复合分区键*：measure_name、dimension_name 和 aggregation_period。复合分区键的列在括号中给出。

假设我们已经实现了一个可视化测量数据的客户端。在客户端中，用户可以首先选择要可视化的计数器/KPI（= 度量名称），然后选择一个维度和聚合周期。假设用户希望查看 2022-02-03 16:00 计算的一分钟周期内 *cells* 的 *dropped_call_percentage*。可以执行以下类型的查询：

```
SELECT measure_value, dimension_value
FROM measurements
WHERE measure_name = 'dropped_call_percentage' AND
      dimension_name = 'cell' AND
      aggregation_period = '1min' AND
      measureTimestamp = '2022-02-03T16:00+0000'
LIMIT 50;
```

上述查询返回在给定分钟内掉话率最高的前 50 个小区。

我们可以创建另一个表来保存特定维度值（例如，特定小区 ID）的测量值。该表可用于深入到特定维度，并查看历史测量值。

CREATE TABLE measurements_by_dimension (
    measure_name text,
    dimension_name text,
    aggregation_period text,
    dimension_value text,
    measure_timestamp timestamp,
    measure_value double,
    PRIMARY KEY ((measure_name,
                  dimension_name,
                  aggregation_period,
                  dimension_value), measure_timestamp)
) WITH CLUSTERING ORDER BY (measureTimestamp DESC);

以下查询将返回过去30分钟内，由小区ID 3000标识的小区的掉话率值：

```sql
SELECT measure_value, measureTimestamp
FROM measurements_by_dimension
WHERE measure_name = 'dropped_call_percentage' AND
      dimension_name = 'cell' AND
      aggregation_period = '1min' AND
      dimension_value = '3000'
LIMIT 30;
```

## 9.5：搜索引擎原理

> 如果你拥有用户应该能够查询的自由格式文本数据，请使用搜索引擎。

搜索引擎（例如 Elasticsearch）对于存储从微服务收集的日志条目等信息非常有用。你通常希望根据日志消息中的文本来搜索收集到的日志数据。

当需要搜索文本数据时，并非必须使用搜索引擎。其他数据库，无论是文档型还是关系型，都有一种特殊的索引类型，可以对列中的自由格式文本数据进行索引。考虑到前面的 MongoDB 示例，我们可能希望客户端能够根据销售商品名称中的文本来搜索销售商品。我们不需要将销售商品存储在搜索引擎数据库中。我们可以继续将它们存储在文档数据库（MongoDB）中，并为 name 字段引入一个文本类型的索引。该索引可以使用以下 MongoDB 命令创建：

```javascript
sales_items.create_index( { 'name': 'text' } )
```

## 10：并发编程原则

本章介绍以下并发编程原则：

- 线程原则
- 线程安全原则
- 发布/订阅共享状态变更原则

### 10.1：线程原则

> *现代云原生微服务应主要通过添加更多进程来水平扩展，而不是通过添加更多线程来垂直扩展。仅在需要或作为良好优化时才使用线程。*

在开发现代云原生软件时，微服务应该是无状态的，并能够自动水平扩展（通过添加和移除进程来实现扩缩容）。线程在现代云原生微服务中的作用，不像早期软件由运行在裸金属服务器上的单体应用组成、主要能够垂直扩展时那样突出。如今，你应该在它是良好优化或确实需要时才使用线程。除了微服务之外，如果你有一个库、独立应用程序或客户端软件组件，情况就不同了，你当然可以使用线程。

假设我们有一个采用事件驱动架构的软件系统。多个微服务使用异步消息传递相互通信。每个微服务实例只有一个线程，该线程从消息代理消费消息，然后进行处理。如果某个微服务的消息代理队列开始变得过长，该微服务应该通过添加新实例来水平扩展。当微服务的负载减少时，它可以通过移除实例来缩减。完全不需要使用线程。

如果数据导出微服务的输入消费者和输出生产者是同步的，我们可以在其中使用线程。使用线程的原因是优化。如果我们将所有内容都放在一个线程中，并且微服务正在执行网络 I/O（无论是输入还是输出相关），那么微服务在等待某些网络 I/O 完成时将无事可做。使用线程，我们可以优化微服务的执行，使其在等待 I/O 操作完成时可能仍有工作可做。

许多现代的输入消费者和输出生产者都提供了异步实现。如果我们在数据导出微服务中使用异步的消费者和生产者，我们就可以消除线程，因为网络 I/O 将不再阻塞主线程的执行。作为经验法则，请首先考虑使用异步代码，如果不可行或不可行，再考虑使用线程。

你可能需要一个微服务在后台按特定计划执行维护任务。与其使用线程并在微服务中实现维护功能，不如考虑在单独的微服务中实现它，以确保遵循*单一职责原则*。例如，你可以配置维护微服务使用 Kubernetes CronJob 定期运行。

线程也会给微服务带来复杂性，因为微服务必须确保线程安全。如果你忘记实现线程安全，将会遇到大麻烦。与线程和同步相关的错误很难发现。线程安全是本章稍后讨论的主题。线程也会给微服务的部署带来复杂性，因为微服务请求的 vCPU 数量取决于所使用的线程数。

### 10.1.1：并行执行器

并行执行器通过隐藏多个子进程的创建来简化并发。在 Python 中，你可以使用多进程来创建并行执行器。以下是一个使用包含4个子进程的进程池的示例：

```python
import multiprocessing
import os

def print_stdout(number: int) -> None:
    print (f'{number} {os.getpid()}')

if __name__ == '__main__':
    numbers = [1, 2, 3, 4]
    pool = multiprocessing.Pool(4)
    pool.map(print_stdout, numbers)
```

上述代码的输出可能如下所示：

```
1 97672
2 97671
4 97672
3 97670
```

### 10.2：线程安全原则

> *如果你使用线程，你必须确保线程安全。线程安全意味着一次只能有一个线程访问共享数据，以避免竞态条件。*

如果你使用数据结构或库，不要假设它是线程安全的。你必须查阅文档以确认是否保证了线程安全。如果文档中没有提到线程安全，则不能假设其存在。向开发者传达线程安全的最佳方式是通过命名使其显式化。例如，你可以创建一个线程安全的集合库，并将一个类命名为 `ThreadSafeList`，以表明该类是线程安全的。

在 Python 中确保线程安全的主要方法是使用锁。Python 没有原子变量。

#### 10.2.1：使用锁实现互斥

Python 在 `threading` 模块中有一个 `Lock` 类。该类实现了原始锁对象以实现互斥。一旦一个线程获取了锁，后续尝试获取它的线程将被阻塞，直到锁被释放。任何线程都可以释放它。

让我们使用锁对象实现一个线程安全的计数器：

```python
from threading import Lock

class ThreadSafeCounter:
    def __init__(self):
        self.__lock = Lock()
        self.__counter = 0

    def increment(self) -> None:
        with self.__lock:
            self.__counter += 1

    @property
    def value(self) -> int:
        with self.__lock:
            return self.__counter
```

Python 在 `multiprocessing` 模块中也包含一个 `Lock` 类，它可用于以类似方式同步多个进程。

#### 10.2.2：原子变量

Python 没有原子变量，但你可以使用锁来定义自己的原子变量类。以下是一个使用锁的 `AtomicInt` 类示例。

```python
from threading import Lock

class AtomicInt():
    def __init__(self, value: int):
        self.__value = value
        self.__lock = Lock()

    def increment(self, amount: int) -> int:
        with self.__lock:
            self.__value += amount
            return self.__value

    def decrement(self, amount: int) -> int:
        with self.__lock:
            self.__value -= amount
            return self.__value

    @property
    def value(self) -> int:
        with self.__lock:
            return self.__value

    @value.setter
    def value(self, new_value: int):
        with self.__lock:
            self.__value = new_value
```

现在我们可以使用它了。想象一下，下面最后三个操作都可以从不同的线程安全地执行：

```python
my_int = AtomicInt(0)
my_int.increment(1)
my_int.decrement(2)
print(my_int.value) # 输出 -1
```

#### 10.2.3：并发集合

并发集合可以被多个线程使用，无需任何额外的同步。以下是一个线程安全列表的部分示例：

```python
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar('T')

class ThreadSafeList(Generic[T]):
    def __init__(self):
        self.__list: list[T] = []
        self.__lock = Lock()

    def append(self, value: T) -> None:
        with self.__lock:
```

self.__list.append(value)

def pop(self, index: int) -> T:
    with self.__lock:
        return self.__list.pop(index)

def get(self, index: int) -> T:
    with self.__lock:
        return self.__list[index]

# 实现其余所需方法

## 10.3：发布/订阅共享状态变更原则

*使用条件对象来发布和等待共享状态的变更*

当你有一个队列，并且在不同线程中存在该队列的生产者和消费者时，条件对象非常有用。生产者线程可以在队列中有新项目时通知消费者线程，而消费者线程则等待队列中有可用项目。如果没有条件对象，你就必须在消费者中使用睡眠来实现这一点。这并非最佳方案，因为你不一定知道多长的睡眠时间是最优的。

```python
from threading import Condition, Lock
from typing import Final, Generic, TypeVar

T = TypeVar('T')

class ThreadSafeQueue(Generic[T]):
    def __init__(self):
        self.__items: Final[list[T]] = []
        self.__item_waiter: Final = Condition()
        self.__lock: Final = Lock()

    def append(self, item: T) -> None:
        with self.__lock:
            self.__items.append(item)

    def pop_front(self) -> T:
        with self.__lock:
            return self.__items.pop(0)

    def has_item(self) -> bool:
        return len(self.__items) > 0

    @property
    def item_waiter(self):
        return self.__item_waiter

class MsgQueueProducer(Generic[T]):
    def __init__(self, queue: ThreadSafeQueue[T]):
        self.__queue = queue

    def produce(self, item: T) -> None:
        with self.__queue.item_waiter:
            self.__queue.append(item)
            self.__queue.item_waiter.notify()

class MsgQueueConsumer(Generic[T]):
    def __init__(self, queue: ThreadSafeQueue[T]):
        self.__queue = queue

    def consume(self) -> T:
        with self.__queue.item_waiter:
            self.__queue.item_waiter.wait_for(self.__queue.has_item)
            return self.__queue.pop_front()
```

## 11：团队协作原则

本章介绍团队协作原则。以下原则将被描述：

- 使用敏捷框架原则
- 定义完成原则
- 为他人编写代码原则
- 避免技术债务原则
- 软件组件文档原则
- 代码审查原则
- 统一代码格式原则
- 高并发开发原则
- 结对编程原则
- 明确定义开发团队角色原则
- 能力转移原则

### 11.1：使用敏捷框架原则

> *使用敏捷框架可以为组织带来诸多益处，包括提高生产力、改善质量、缩短上市时间以及提升员工满意度。*

以上陈述来自一些采用[规模化敏捷框架（SAFe）](https://www.scaledagileframework.com/)的公司的[客户案例](https://scaledagile.com/insights-customer-stories/)。

敏捷框架描述了一种标准化的软件开发方式，这在大型组织中尤为重要。在当今的工作环境中，人们频繁更换工作，团队也经常变动，这可能导致除非使用特定的敏捷框架，否则对工作方式缺乏共同理解的情况。敏捷框架建立了明确的职责分工，每个人都可以专注于自己最擅长的事情。

例如，在*SAFe*中，在一个项目增量（PI）规划期间，开发团队为下一个PI（包含4个迭代，每个迭代两周，共8周）规划特性。在PI规划中，团队将特性拆分为用户故事，并查看哪些特性适合放入PI。规划好的用户故事将被分配故事点（例如，以人天为单位衡量），并将故事放入迭代中。这个规划阶段产生了一个团队在PI期间应遵循的计划。初级的SAFe实践者可能会犯一些错误，比如低估完成一个用户故事所需的工作量。但这是一个自我纠正的问题。随着团队和个人的发展，他们将更好地估算所需的工作量，计划也会变得更加稳固。团队和开发者了解到他们必须使所有工作可见。例如，预留时间学习新事物，如编程语言或框架，并预留时间进行重构。能够遵循计划进度，有时甚至提前完成工作，是非常令人满意的。这会让你感觉自己像一个真正的专业人士，并提升自尊心。

我个人使用SAFe超过五年的经验都是积极的。我感觉我可以更专注于“真正的工作”，这让我更快乐。会议、无关的电子邮件和干扰减少了。这主要是因为团队有一个产品负责人和Scrum Master，他们的角色是保护团队成员免受任何“浪费”或“管理事务”的干扰，让团队成员能够专注于他们的工作。

### 11.2：定义完成原则

> 对于用户故事和特性，定义“完成”的含义。

在最理想的情况下，开发团队对于宣布一个*用户故事*或*特性*完成所需的内容有共同的理解。当有一个共同的完成定义时，每个开发团队都可以确保一致的结果和质量。

在考虑一个用户故事时，至少可以定义以下完成用户故事的要求：

- 源代码已提交到源代码仓库
- 源代码已审查
- 已执行静态代码分析（无阻塞/严重/主要问题）
- 单元测试覆盖率至少为X%
- CI/CD流水线通过
- 无第三方软件漏洞

团队中产品负责人（PO）的角色是接受一个用户故事为完成状态。上述一些要求可以自动检查。例如，静态代码分析应成为每个CI/CD流水线的一部分，并且也可以自动检查单元测试覆盖率。如果静态代码分析未通过或单元测试覆盖率不可接受，CI/CD流水线就不会通过。

在考虑一个特性时，应定义一些额外的完成要求，因为特性可以交付给客户。以下是一些完成特性的要求列表：

- 架构设计文档已更新
- 集成测试已添加/更新
- 如有需要，端到端测试已添加/更新
- 非功能性测试已完成
- 用户文档已就绪
- 威胁建模已完成，并且威胁对策（安全特性）已实施

为了完成所有所需的完成要求，开发团队可以使用工具来帮助他们记住需要完成的事项。例如，在Jira等工具中创建新的用户故事时，可以克隆一个现有的原型故事（或使用模板）。原型或模板故事应包含在用户故事被批准之前必须完成的任务。

### 11.3：为他人编写代码原则

*你为他人和未来的自己编写代码。*

你独自负责一段软件的情况相对较少。你无法预测未来会发生什么。可能会有其他人负责你曾经编写的代码。也存在这样的情况：你与某些代码共事一段时间后，可能在几年后，需要重新接触那段代码。因此，编写易于他人和未来的自己阅读和理解的整洁代码至关重要。请记住，代码不仅是为计算机编写的，也是为人类编写的。人们应该能够轻松地阅读和理解代码。请记住，最好的代码读起来就像优美的散文！

### 11.4：避免技术债务原则

避免技术债务最常见的实践如下：

- 架构团队应设计高层架构（每个团队应在架构团队中有一名代表。通常是团队的技术负责人）
- 开发团队应先进行面向对象设计，然后再进行实现
- 在团队内进行面向对象设计，让相关的资深和初级开发人员参与其中
- 不要立即采用最新的第三方软件，而是使用在市场上已确立地位的成熟第三方软件
- 设计时考虑易于用另一个第三方组件替换某个第三方软件组件
- 为可扩展性设计（为未来的负载）
- 为扩展设计：新功能放在新类中，而不是修改现有类
- 利用插件架构（为以后添加新功能创建插件的可能性）
- 预留时间进行重构

技术债务的主要原因如下：

-   使用小众技术或不成熟的新技术
-   未使软件具备可扩展性以满足未来的处理需求
-   当替换第三方软件组件相对困难时（例如，使用自定义SQL语法导致无法更换数据库，未在第三方库中使用*适配器模式*）
-   未进行架构评审
-   在开始编码前未进行任何面向对象设计
-   在面向对象设计阶段未让足够资深的开发人员参与
-   未理解并运用相关的设计原则和模式
    -   未针对接口编程
    -   依赖项难以更改（缺少依赖注入）
    -   未使用外观模式
-   未评审代码变更
-   未预留重构时间
-   工作量估算过小
-   来自管理层的时间压力
-   管理层不理解重构的价值
-   将重构推迟到永远不会到来的时间点
-   忘记重构（至少应将所需的重构工作项存储在源代码仓库的TODO.MD文件中）
-   没有单元测试，重构更困难
-   重复代码
-   未践行童子军规则
-   懒惰（使用最先想到的方法，或总是试图寻找最简单快捷的做事方式）

## 11.5：软件组件文档原则

> *每个软件组件都需要文档化。文档化的主要目的是让新成员快速上手开发工作。*

至关重要的是，为软件组件搭建开发环境的过程应有完善的文档记录，并尽可能简单。另一个重要方面是让人们能轻松理解该软件组件试图解决的问题领域。此外，软件组件的面向对象设计也应被记录下来。

软件组件文档应与源代码存放在同一个源代码仓库中。推荐的方式是在源代码仓库的根目录下使用README.MD文件，以Markdown格式编写文档。你可以将文档拆分为多个文件，并将额外的文件存储在源代码仓库的docs目录中。

以下是记录软件组件时可使用的目录表示例：

-   软件组件的简要描述及其用途
-   功能列表
-   描述不同子域及每个子域中接口/类的面向对象设计图
    -   设计说明（如需要）
-   API文档（针对库）
-   实现相关文档
    -   错误处理机制
    -   使用的特殊算法
    -   性能考量
    -   主要安全特性
-   搭建开发环境的说明
    -   搭建开发环境最简单的方法是使用开发容器，这是Visual Studio Code编辑器支持的一个概念。使用开发容器的好处是，你无需在本地安装开发工具，也不存在使用错误版本开发工具的风险
-   本地构建软件的说明
-   本地运行单元测试的说明
-   本地运行集成测试的说明
-   部署到测试环境的说明
-   配置
    -   环境变量
    -   配置文件
    -   密钥

## 11.6：代码评审原则

> 在代码评审中，应关注机器无法为你发现的问题。

在评审代码之前，应先进行静态代码分析，以找出机器能发现的所有问题。实际的代码评审应聚焦于静态代码分析器无法发现的问题。你无需评审代码格式，但团队中的每个人都应使用相同的代码格式，这应通过自动格式化工具来保证。你不能评审自己的代码。至少应有一位评审者处于高级或负责人角色。代码评审中应关注的要点将在后续章节中介绍。

### 11.6.1：关注面向对象设计

在开始编码之前，建议先设计软件：定义子域、所需的接口和类。这个初始设计阶段的产物应在开始编码前提交到源代码仓库并进行评审。这样更容易在早期纠正设计缺陷，避免技术债务。在后期阶段修复设计缺陷可能需要大量精力，甚至需要重写现有代码。至少应有一位高级开发人员参与设计。

如果在评审中遇到设计缺陷，但没有时间立即修复。应向团队的待办事项列表中添加一个或多个重构用户故事。

### 11.6.2：关注通过单元测试体现的功能规格

对于要评审的每个公共函数，评审者应从单元测试开始，查看它们是否覆盖了所有功能。是否缺少针对错误场景、安全场景或边界情况的单元测试用例？

### 11.6.3：关注正确且统一的命名

静态代码分析工具只能部分做到的一件事是确保事物（如类、函数和变量）的命名正确且统一。命名是代码评审中应重点关注的地方。重命名是一项非常直接且快速的重构任务，可以由现代IDE自动执行。

### 11.6.4：不要关注过早优化

在常规代码评审中不要关注优化。优化通常在代码就绪并首先测量性能后，根据需要进行。只有在你评审的提交专门用于优化某些内容时，才应关注与优化相关的问题。

### 11.6.5：检测可能的恶意代码

代码评审者必须验证提交的代码不包含恶意代码。

## 11.7：统一代码格式原则

> 在软件开发团队中，必须就源代码格式化制定共同规则。

一致的代码格式至关重要，因为如果团队成员有不同的源代码格式化规则，一个团队成员对文件的微小更改可能会使用他/她的格式化规则重新格式化整个文件，这可能导致另一位开发人员面临重大的合并冲突，从而减慢开发进程。应始终就共同的源代码格式化规则达成一致，并最好使用像*Prettier*这样的工具来强制执行格式化规则。如果没有自动格式化工具，你可以为团队成员使用的IDE创建源代码格式化规则，并将这些规则存储在源代码仓库中。

## 11.8：高并发开发原则

> 每个团队成员都可以处理某部分代码。任何人都不应长时间等待他人的工作完成。

当不同的人修改不同的源代码文件时，就实现了并发开发。当多个人需要修改相同的文件时，可能会导致合并冲突。这些合并冲突会导致额外的工作，因为它们通常必须手动解决。这种手动工作可能很慢，而且容易出错。最好尽可能避免合并冲突。这可以通过以下章节描述的方式实现。

### 11.8.1：专用的微服务和微库

微服务本质上是小型的，可以将微服务的职责分配给单个团队成员。这位团队成员可以全速推进该微服务，并且可以放心没有其他人会修改代码库。库也是如此。你应该创建小型的微库（=具有单一职责的库），并将开发微库的职责分配给单个人。

### 11.8.2：专用领域

有时无法将单个微服务或库分配给单个开发人员。这可能是因为微服务或库相对较大，将其拆分为多个微服务或库不可行。在这些情况下，应通过进行领域驱动设计，将微服务或库划分为几个子域。每个源代码目录反映一个不同的子域。然后可以将单个子域的职责分配给单个人。子域的分配不应是固定的，而是可以并且应该随着时间的推移而改变。为了传播不同领域的知识，建议在团队成员之间轮换职责。假设你有一个由三名开发人员组成的团队，正在开发一个由三个子域组成的数据导出微服务：输入、转换和输出。团队可以通过将单个领域的职责分配给单个开发人员来实现该微服务。现在所有开发人员都可以高度独立且并发地进行实现。在早期阶段，他们必须定义不同子域之间的接口。

未来，当开发新功能时，团队成员可以负责其他领域，以在团队中传播关于该微服务的知识。

### 11.8.3：遵循开闭原则

有时你可能会遇到这样的情况：单个子域非常大，需要多名开发人员。当然，这应该是相对罕见的情况。当多名开发人员修改属于同一子域（即在同一目录中）的源代码文件时，可能会出现合并冲突。尤其是在修改现有源代码文件时，这种情况更可能发生。但是，当开发人员遵循*开闭原则*时，他们不应修改现有的类（源代码文件），而应在新的类（源代码文件）中实现新功能。使用*开闭原则*使开发人员能够更并发地开发，因为他们主要处理不同的源代码文件，使得合并冲突很少发生或至少不那么频繁。

## 11.9：结对编程原则

> *结对编程有助于生产出质量更好、设计更佳、技术债务更少、测试更好、缺陷更少的软件。*

结对编程是一些开发人员喜欢，而另一些开发人员讨厌的事情。所以它不是一个放之四海而皆准的解决方案。它也不是非此即彼的选择。你可以有一个团队，其中一些开发人员结对编程，而另一些则不结对。此外，人们对结对编程的看法可能存在偏见。也许他们从未尝试过结对编程，那么他们怎么知道自己是否喜欢呢？选择合适的搭档结对也确实很重要。有些搭档比其他搭档更有默契。

结对编程只是增加了开发成本吗？结对编程带来了哪些好处？

我认为结对编程非常有价值，尤其是在初级开发者与资深开发者搭档的情况下，这样初级开发者能更快地融入团队。他可以“向最优秀的人学习”。结对编程能提升软件设计质量，因为设计始终至少有两个人的视角。缺陷更容易被发现，且通常能在更早阶段被发现（四只眼睛总比两只眼睛强）。因此，即使结对编程会增加一些成本，它通常能带来更高质量的软件：更好的设计、更少的技术债务、更完善的测试以及更少的缺陷。

## 11.10：开发团队角色明确原则

> 软件开发团队应为每位成员设定明确的角色。

如果团队中每个人都做所有事，或者期望任何人都能胜任任何工作，那么软件开发团队就无法发挥最佳效能。没有人是全才。当团队针对不同任务拥有专门人才时，才能取得最佳成果。团队成员需要有自己擅长且乐于专注的领域。当你成为某个领域的专家时，你就能更快、更高质量地完成该领域的任务。

以下是开发团队所需的角色列表：

- 产品负责人（PO）
- Scrum Master（SM）
- 软件开发者（初级/资深/负责人）
- 测试自动化开发者
- DevOps 工程师
- UI 设计师（如果团队开发 UI 软件组件）

我们将在以下章节详细讨论每个角色的职责。

### 11.10.1：产品负责人

产品负责人（PO）充当开发团队与业务部门（通常指产品管理部门，PM）之间的接口。PO 通过讨论从 PM 处收集需求（非功能性、功能性）。PO 负责确定团队待办事项的优先级，并与团队共同定义用户故事。PO 角色通常不是全职的，因此一个 PO 可以服务于两个小型团队。PO 从 PM 处收集需求（非功能性、功能性）。PM 通常不会简单地列出所有需求，PO 必须通过提出正确的问题并与 PM 讨论来找出所有需求。PO 应具备技术背景，能够创建例如 Gherkin 功能文件作为待办功能的验收标准。

### 11.10.2：Scrum Master

Scrum Master（SM）是开发团队的仆人式领导和教练。Scrum Master 确保团队遵循相关的敏捷实践和敏捷流程。他们向团队传授敏捷实践。如果团队有直线经理，直线经理可以担任 Scrum Master，但任何团队成员都可以成为 Scrum Master。

### 11.10.3：软件开发者

软件开发者负责设计、实现和测试软件（包括单元测试，在大多数情况下还包括集成测试）。软件开发者通常专注于一两种编程语言和几个技术框架。通常，软件开发者分为以下几类：

- 后端开发者
- 前端开发者
- 全栈开发者
- 移动开发者
- 嵌入式开发者

后端开发者开发在后端运行的微服务，如 API。前端开发者开发 Web 客户端。通常，前端开发者使用 JavaScript 或 TypeScript、React/Angular/Vue、HTML 和 CSS。全栈开发者是后端和前端开发者的结合，能够开发后端微服务和前端客户端。移动开发者为移动设备（如手机和平板电脑）开发软件。

团队应拥有不同资历水平的软件开发者。每个团队都应有一位在所用技术和领域拥有最丰富经验的负责人开发者。负责人开发者通常属于由系统架构师领导的虚拟架构团队。团队中只有初级开发者或只有资深开发者是没有意义的。其理念是将技能和知识从资深开发者传递给初级开发者。反之亦然。初级开发者可能掌握一些资深开发者所缺乏的最新技术和实践知识。因此，总的来说，最好的团队是由初级和资深开发者良好混合组成的团队。

### 11.10.4：测试自动化开发者

测试自动化开发者负责开发各种类型的自动化测试。通常，测试自动化开发者开发集成测试、端到端（E2E）测试和自动化非功能性测试。测试自动化开发者必须精通至少一种用于开发自动化测试的编程语言，例如 Python。测试自动化开发者必须熟练掌握 BDD（行为驱动开发）和一些常见的测试框架，例如 Cucumber-JVM 或 Behate。了解一些测试工具（如 Apache JMeter）是加分项。测试自动化开发者也可以开发内部测试工具，例如接口模拟器和数据生成器。测试自动化开发者应组成一个虚拟团队，以促进 E2E 和自动化非功能性测试的开发。

### 11.10.5：DevOps 工程师

DevOps 工程师充当软件开发团队与软件运维之间的接口。DevOps 工程师通常为微服务创建 CI/CD 流水线，并编写与基础设施和部署相关的代码。DevOps 工程师还定义告警规则和指标可视化仪表板，这些可在监控生产环境中的软件时使用。DevOps 工程师帮助运维人员监控生产环境中的软件。他们可以帮助解决技术支持组织无法解决的问题。DevOps 工程师了解软件部署的环境（即基础设施和平台），这意味着需要至少了解一个云服务提供商（AWS/Azure/Google Cloud 等）以及可能的 Kubernetes 的基础知识。DevOps 工程师应组成一个虚拟团队，以促进制定与 DevOps 相关的实践和指南。

### 11.10.6：UI 设计师

UI 设计师负责基于更高层次的 UX/UI 设计/线框图设计最终的 UI。UI 设计师还将进行软件的可用性测试。

## 11.11：能力转移原则

能力转移的一个极端情况是，当一个人离开公司时，必须将软件组件的责任移交给另一个人。很多时候，这个另一个人对该软件组件完全不熟悉或知之甚少。为确保能力顺利转移，作为能力转移的一部分，至少应执行以下操作：

- 演示软件组件的功能
- 解释软件组件的架构（它如何与其他服务交互）
- 解释软件组件的（面向对象）设计
  - 软件组件如何划分为子域
  - 主要接口/类
- 解释软件组件的主要实现决策：
  - 使用的特殊算法
  - 并发处理
  - 错误处理机制
  - 主要安全特性
  - 性能考量
- 解释软件组件的配置
- 根据 README.MD 中的说明设置开发环境，以确保说明正确且是最新的。
- 将软件组件部署到测试环境
- 构建软件、执行单元测试和执行集成测试
- 解释 CI 流水线（如果与其他软件组件的 CI 流水线不同）
- 解释其他可能的自动化功能性和非功能性测试，例如 E2E 测试、性能和稳定性测试
- 解释软件组件的可观测性，例如日志记录、审计日志、指标、仪表板和告警

## 12：DevSecOps

DevOps 描述了将软件开发（Dev）和软件运维（Ops）集成的实践。它旨在通过开发并行化和自动化来缩短软件开发生命周期，并提供高质量软件的持续交付。DevSecOps 通过在软件生命周期中增加安全方面来增强 DevOps。

## 12.1：安全运维生命周期

安全运维生命周期分为以下阶段：

- 威胁建模
    - 确定需要哪些安全特性和测试
    - 实施威胁对策和缓解措施。这一方面在之前的*安全原则*章节中有更详细的介绍
- 扫描
    - 静态安全分析（也称为 SAST = 静态应用安全测试）
    - 安全测试（也称为 DAST = 动态应用安全测试）
    - 容器漏洞扫描
- 分析
    - 分析扫描阶段的结果，检测并移除误报，并对漏洞修复进行优先级排序
- 修复
    - 根据优先级修复发现的漏洞
- 监控
    - 定义与安全运维相关的指标并进行监控

## 12.2：DevOps 生命周期

DevOps 生命周期分为以下阶段：

- 规划
- 编码
- 构建
- 测试
- 发布
- 部署
- 运维
- 监控

后续章节将更详细地描述每个阶段。

### 12.2.1：规划

*规划*是 DevOps 生命周期的第一个阶段。在此阶段，规划软件功能，并设计高层架构和用户体验。此阶段涉及业务（产品管理）和软件开发组织。

### 12.2.2：编码

*编码*是软件实现阶段。它包括软件组件的设计与实现、编写单元测试、集成测试、端到端测试以及其他自动化测试。此阶段还包括使软件可部署所需的所有其他编码工作。大部分工作在此阶段完成，因此应尽可能简化流程。

缩短此阶段的关键是尽可能地并行化一切。在*规划*阶段，软件在架构上被拆分为更小的部分（微服务），以便不同团队可以并行开发。对于单个微服务的开发，也应尽可能并行化。这意味着，如果一个微服务可以拆分为多个子域，那么这些子域的开发可以非常并行地进行。以数据导出器微服务为例，我们识别出几个子域：输入、解码、转换、编码和输出。如果你能并行开发这五个子域，就能显著缩短完成微服务实现所需的时间。

为了进一步缩短此阶段，团队应配备专门的测试自动化开发人员，以便在实现阶段的早期并行开始开发自动化测试。

提供高质量的软件依赖于高质量的设计、技术债务少的实现以及全面的功能和非功能测试。所有这些方面在前面的章节中已经讨论过。

### 12.2.3：构建与测试

*构建与测试*阶段应实现自动化，并作为*持续集成*（CI）流水线运行。软件系统中的每个软件组件都应有自己的 CI 流水线。CI 流水线由 CI 工具（如 *Jenkins* 或 *GitHub Actions*）运行。CI 流水线使用存储在软件组件源代码仓库中的声明式代码定义。每次向源代码仓库的主分支提交代码时，都应触发一次 CI 流水线运行。

软件组件的 CI 流水线应至少执行以下任务：

- 从源代码仓库检出最新的源代码
- 构建软件
- 执行静态代码分析。可以使用 *SonarQube/SonarCloud* 等工具
    - 执行静态应用安全测试（SAST）。
- 执行单元测试
- 执行集成测试
- 执行动态应用安全测试（DAST）。可以使用 OWASP ZAP 等工具
- 验证第三方许可证合规性并提供物料清单（BOM）。可以使用 Fossa 等工具

### 12.2.4：发布

在发布阶段，经过构建和测试的软件会自动发布。软件组件的 CI 流水线成功执行后，该软件组件即可自动发布。这称为持续交付（CD）。持续交付通常与 CI 流水线结合，为软件组件创建 CI/CD 流水线。持续交付意味着软件组件的制品被交付到制品仓库，如 Artifactory、Docker Hub 或 Helm Chart 仓库。CD 流水线应执行以下任务：

- 对构建容器镜像的代码（例如 Dockerfile）执行静态代码分析。可以使用 Hadolint 等工具处理 Dockerfile。
- 为软件组件构建容器镜像
- 将容器镜像发布到容器注册表（例如 Docker Hub、Artifactory 或云提供商提供的注册表）
- 执行容器镜像漏洞扫描
    - 同时，请记得在容器注册表中定期启用容器漏洞扫描
- 对部署代码执行静态代码分析。可以使用 Helm 的 lint 命令、Kubesec 和 Checkov 等工具
- 打包并发布部署代码（例如，打包 Helm Chart 并将其发布到 Helm Chart 仓库）

#### 12.2.4.1：Dockerfile 示例

下面是一个使用 FastAPI 库编写的 API 微服务的 Dockerfile 示例。该 Dockerfile 使用了 Docker 的多阶段构建特性。首先（在 install_deps 阶段），它安装依赖项。然后（在 intermediate 阶段），它创建一个中间镜像，该镜像从 JavaScript 源代码文件复制源代码文件。最后一个阶段（final）将文件从 install-deps 阶段复制到一个无发行版的 Python 基础镜像中。你应该使用无发行版的基础镜像，以减小镜像大小和攻击面。无发行版镜像内部不包含任何 Linux 发行版。遗憾的是，下面提到的 gcr.io/distroless/python 镜像目前（在撰写本书时）被认为是实验性的，不建议用于生产环境。

```
FROM python:3.11 as install-deps

WORKDIR /microservice
COPY ./requirements.txt /microservice/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /microservice/requirements.txt
COPY ./app /microservice/app

FROM gcr.io/distroless/python3.11 as final
COPY --from=install-deps /microservice /microservice
WORKDIR /microservice
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

#### 12.2.4.2：Kubernetes Deployment 示例

下面是一个用于 Kubernetes Deployment 的 Helm Chart 模板 *deployment.yaml* 示例。模板代码包含在双大括号中。

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "microservice.fullname" . }}
  labels:
    {{- include "microservice.labels" . | nindent 4 }}
spec:
  {{- if ne .Values.env "production" }}
  replicas: 1
  {{- end }}
  selector:
    matchLabels:
      {{- include "microservice.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      {{- with .Values.deployment.pod.annotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "microservice.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.deployment.pod.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "microservice.serviceAccountName" . }}
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.imageRegistry }}/{{ .Values.imageRepository }}:{{ .Values.imageTag }}"
          imagePullPolicy: {{ .Values.deployment.pod.container.imagePullPolicy }}
          securityContext:
            {{- toYaml .Values.deployment.pod.container.securityContext | nindent 12 }}
          {{- if .Values.httpServer.port }}
          ports:
            - name: http
              containerPort: {{ .Values.httpServer.port }}
              protocol: TCP
```

{{- end }}
env:
  - name: ENV
    value: {{ .Values.env }}
  - name: ENCRYPTION_KEY
    valueFrom:
      secretKeyRef:
        name: {{ include "microservice.fullname" . }}
        key: encryptionKey
  - name: MICROSERVICE_NAME
    value: {{ include "microservice.fullname" . }}
  - name: MICROSERVICE_NAMESPACE
    valueFrom:
      fieldRef:
        fieldPath: metadata.namespace
  - name: MICROSERVICE_INSTANCE_ID
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: NODE_NAME
    valueFrom:
      fieldRef:
        fieldPath: spec.nodeName
  - name: MYSQL_HOST
    value: {{ .Values.database.mySql.host }}
  - name: MYSQL_PORT
    value: "{{ .Values.database.mySql.port }}"
  - name: MYSQL_USER
    valueFrom:
      secretKeyRef:
        name: {{ include "microservice.fullname" . }}
        key: mySqlUser
  - name: MYSQL_PASSWORD
    valueFrom:
      secretKeyRef:
        name: {{ include "microservice.fullname" . }}
        key: mySqlPassword
livenessProbe:
  httpGet:
    path: /isAlive
    port: http
  failureThreshold: 3
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /isReady
    port: http
  failureThreshold: 3
  periodSeconds: 5
startupProbe:
  httpGet:
    path: /isStarted
    port: http
  failureThreshold: {{ .Values.deployment.pod.container.startupProbe.failureThreshold }}
  periodSeconds: 10
resources:
  {{- if eq .Values.env "development" }}
  {{- toYaml .Values.deployment.pod.container.resources.development | nindent 12 }}
  {{- else if eq .Values.env "integration" }}
  {{- toYaml .Values.deployment.pod.container.resources.integration | nindent 12 }}
  {{- else }}
  {{- toYaml .Values.deployment.pod.container.resources.production | nindent 12 }}
  {{- end }}
  {{- with .Values.deployment.pod.nodeSelector }}
  nodeSelector:
    {{- toYaml . | nindent 8 }}
  {{- end }}
  {{- with .Values.deployment.pod.affinity }}
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              app.kubernetes.io/name: {{ include "microservice.name" . }}
          topologyKey: "kubernetes.io/hostname"
    {{- toYaml . | nindent 8 }}
  {{- end }}
  {{- with .Values.deployment.pod.tolerations }}
  tolerations:
    {{- toYaml . | nindent 8 }}
  {{- end }}

上述模板中的值（由 `.Values.<something>` 表示）来自一个 *values.yaml* 文件。下面是一个与上述 Helm Chart 模板配合使用的 *values.yaml* 示例文件。

```yaml
imageRegistry: docker.io
imageRepository: pksilen2/backk-example-microservice
imageTag:
env: production
auth:
  # 授权服务器颁发者 URL
  # 例如
  # http://keycloak.platform.svc.cluster.local:8080/auth/realms/<my-realm>
  issuerUrl:

  # 用户角色的 JWT 路径，
  # 例如 'realm_access.roles'
  jwtRolesClaimPath:
secrets:
  encryptionKey:
database:
  mySql:
    # 例如：
    # my-microservice-mysql.default.svc.cluster.local 或
    # 云数据库主机
    host:
    port: 3306
    user:
    password: &mySqlPassword ""
mysql:
  auth:
    rootPassword: *mySqlPassword
deployment:
  pod:
    annotations: {}
    imagePullSecrets: []
    container:
      imagePullPolicy: Always
      securityContext:
        privileged: false
        capabilities:
          drop:
            - ALL
        readOnlyRootFilesystem: true
        runAsNonRoot: true
        runAsUser: 65532
        runAsGroup: 65532
        allowPrivilegeEscalation: false
      env:
      startupProbe:
        failureThreshold: 30
      resources:
        development:
          limits:
            cpu: '1'
            memory: 768Mi
          requests:
            cpu: '1'
            memory: 384Mi
        integration:
          limits:
            cpu: '1'
            memory: 768Mi
          requests:
            cpu: '1'
            memory: 384Mi
        production:
          limits:
            cpu: 1
            memory: 768Mi
          requests:
            cpu: 1
            memory: 384Mi
    nodeSelector: {}
    tolerations: []
    affinity: {}
```

特别注意上述文件中的 `deployment.pod.container.securityContext` 对象。它用于定义微服务容器的安全上下文。

默认情况下，安全上下文应如下所示：

-   容器不应是特权模式
-   丢弃所有能力
-   容器文件系统是只读的
-   仅允许非 root 用户在容器内运行
-   定义容器应运行的非 root 用户和组
-   禁止权限提升

只有当微服务有强制性要求时，你才可以从上述列表中移除某些项。例如，如果微服务因某些合理原因必须写入文件系统，则不应将文件系统定义为只读。

## 12.2.4.3：CI/CD 流水线示例

下面是一个用于 Python 微服务的 GitHub Actions CI/CD 工作流。声明式工作流以 YAML 编写。工作流文件应位于微服务源代码仓库的 `.github/workflows` 目录中。示例之后将更详细地描述工作流中的步骤。

```yaml
name: CI/CD workflow
on:
  workflow_dispatch: {}
  push:
    branches:
      - main
    tags-ignore:
      - '**'
jobs:
  build:
    runs-on: ubuntu-latest
    name: Build with Python 3.11
    steps:
      - name: Checkout Git repo
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint source code
        run: pylint src

      - name: Run unit tests
        run: python -m coverage run -m unittest

      - name: Report unit test coverage
        run: python -m coverage xml

      - name: Setup integration testing environment
        run: docker-compose --env-file .env.ci up --build -d

      - name: Run integration tests
        run: scripts/run-integration-tests-in-ci.sh

      - name: OWASP ZAP API scan
        uses: zaproxy/action-api-scan@v0.5.0
        with:
          target: http://localhost:8080/openapi.json
          fail_action: true
          cmd_options: -I -z "-config replacer.full_list(0).description=auth1"
          -config replacer.full_list(0).enabled=true
          -config replacer.full_list(0).matchtype=REQ_HEADER
          -config replacer.full_list(0).matchstr=Authorization
          -config replacer.full_list(0).regex=false
          -config 'replacer.full_list(0).replacement=Bearer ZX1K...aG\JHZ='"

      - name: Tear down integration testing environment
        run: docker-compose --env-file .env.ci down -v

      - name: Static code analysis with SonarCloud scan
        uses: sonarsource/sonarcloud-github-action@master
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

      - name: 3rd party software license compliance analysis with FOSSA
        uses: fossas/fossa-action@main
        with:
          api-key: ${{ secrets.FOSSA_API_KEY }}
          run-tests: false

      - name: Lint Dockerfile
        uses: hadolint/hadolint-action@v3.1.0

      - name: Log in to Docker registry
        uses: docker/login-action@v3
        with:
          registry: docker.io
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_TOKEN }}

      - name: Extract latest Git tag
        uses: actions-ecosystem/action-get-latest-tag@v1
        id: extractLatestGitTag

      - name: Set up Docker Buildx
        id: setupBuildx
        uses: docker/setup-buildx-action@v3

      - name: Cache Docker layers
        uses: actions/cache@v3
        with:
          path: /tmp/.buildx-cache
          key: ${{ runner.os }}-buildx-${{ github.sha }}
          restore-keys: |
            ${{ runner.os }}-buildx-

      - name: Extract metadata for building and pushing Docker image
        id: dockerImageMetadata
        uses: docker/metadata-action@v5
        with:
          images: ${{ secrets.DOCKER_REGISTRY_USERNAME }}/example-microservice
          tags: |
            type=semver,pattern={{version}},value=${{ steps.extractLatestGitTag.outputs.value }}

      - name: Build and push Docker image
        id: dockerImageBuildAndPush
```

## DevSecOps

```yaml
uses: docker/build-push-action@v5
with:
  context: .
  builder: ${{ steps.setupBuildx.outputs.name }}
  push: true
  cache-from: type=local,src=/tmp/.buildx-cache
  cache-to: type=local,dest=/tmp/.buildx-cache
  tags: ${{ steps.dockerImageMetadata.outputs.tags }}
  labels: ${{ steps.dockerImageMetadata.outputs.labels }}
```

```yaml
- name: 使用 Anchore 进行 Docker 镜像漏洞扫描
  id: anchoreScan
  uses: anchore/scan-action@v3
  with:
    image: ${{ secrets.DOCKER_REGISTRY_USERNAME }}/example-microservice:latest
    fail-build: false
    severity-cutoff: high
```

```yaml
- name: 上传 Anchore 扫描 SARIF 报告
  uses: github/codeql-action/upload-sarif@v1
  with:
    sarif_file: ${{ steps.anchoreScan.outputs.sarif }}
```

```yaml
- name: 安装 Helm
  uses: azure/setup-helm@v3
  with:
    version: v3.13.0
```

```yaml
- name: 从 Git 标签中提取微服务版本
  id: extractMicroserviceVersionFromGitTag
  run: |
    value="${{ steps.extractLatestGitTag.outputs.value }}"
    value=${value:1}
    echo "::set-output name=value::$value"
```

```yaml
- name: 更新 Chart.yaml 中的 Helm chart 版本
  run: |
    sed -i "s/^version:.*/version: ${{ steps.extractMicroserviceVersionFromGitTag.outputs.value }}/g" helm/example-microservice/Chart.yaml
    sed -i "s/^appVersion:.*/appVersion: ${{ steps.extractMicroserviceVersionFromGitTag.outputs.value }}/g" helm/example-microservice/Chart.yaml
```

```yaml
- name: 更新 values.yaml 中的 Docker 镜像标签
  run: |
    sed -i "s/^imageTag:.*/imageTag: ${{ steps.extractMicroserviceVersionFromGitTag.outputs.value }}@${{ steps.dockerImageBuildAndPush.outputs.digest }}/g" helm/example-microservice/values.yaml
```

```yaml
- name: 对 Helm chart 进行代码检查
  run: helm lint -f helm/values/values-minikube.yaml helm/example-microservice
```

```yaml
- name: 使用 Checkov 对 Helm chart 进行静态代码分析
  uses: bridgecrewio/checkov-action@master
  with:
    directory: helm/example-microservice
    quiet: false
    framework: helm
    soft_fail: false
```

```yaml
- name: 上传 Checkov SARIF 报告
  uses: github/codeql-action/upload-sarif@v1
  with:
    sarif_file: results.sarif
    category: checkov-iac-sca
```

```yaml
- name: 配置 Git 用户
  run: |
    git config user.name "$GITHUB_ACTOR"
    git config user.email "$GITHUB_ACTOR@users.noreply.github.com"
```

```yaml
- name: 打包并发布 Helm chart
  uses: helm/chart-releaser-action@v1.5.0
  with:
    charts_dir: helm
  env:
    CR_TOKEN: "${{ secrets.GITHUB_TOKEN }}"
```

1.  检出微服务的 Git 仓库
2.  设置 Python 3.11
3.  安装依赖
4.  对源代码进行代码检查
5.  执行单元测试（需要在 `requirements.txt` 中包含 `coverage` 包）
6.  报告测试覆盖率
7.  使用 Docker 的 `docker-compose up` 命令设置集成测试环境。执行该命令后，微服务将被构建，并且所有位于独立容器中的依赖项都将启动。这些依赖项可以包括其他微服务，以及例如数据库和消息代理（如 Apache Kafka）
8.  执行集成测试。此脚本将首先等待所有依赖项启动并就绪。此等待过程通过运行一个使用 `dokku/wait` (https://hub.docker.com/r/dokku/wait) 镜像的容器来完成。
9.  使用 OWASP ZAP API 扫描执行动态应用安全测试。对于扫描，我们定义了 OpenAPI 3.0 规范的 URL，扫描将基于此进行。我们还提供命令选项，为扫描发出的 HTTP 请求设置有效的 Authorization 头
10. 拆卸集成测试环境
11. 使用 SonarCloud 执行静态代码分析。你需要在源代码仓库的根目录中包含以下文件：
12. 使用 FOSSA 检查第三方软件许可证合规性
13. 对 Dockerfile 进行代码检查
14. 登录 Docker Hub
15. 提取最新的 Git 标签以供后续使用
16. 设置 Docker Buildx 并缓存 Docker 层

```
sonar.projectKey=<sonar-project-key>
sonar.organization=<sonar-organization>

sonar.python.coverage.reportPaths=coverage.xml
```

17. 提取元数据，例如用于构建和推送 Docker 镜像的标签和标签
18. 构建并推送 Docker 镜像
19. 使用 Anchore 执行 Docker 镜像漏洞扫描
20. 将 Anchore 扫描报告上传到 GitHub 仓库
21. 安装 Helm
22. 从 Git 标签中提取微服务版本（移除版本号前的 'v' 字母）
23. 使用 *sed* 命令替换 Helm chart 的 *Chart.yaml* 文件中的版本
24. 更新 *values.yaml* 文件中的 Docker 镜像标签
25. 对 Helm chart 进行代码检查并执行静态代码分析
26. 将静态代码分析报告上传到 GitHub 仓库，并为下一步执行 git 用户配置
27. 打包 Helm chart 并将其发布到 GitHub Pages

上述部分步骤可以并行执行，但 GitHub Actions 工作流目前不支持作业中的并行步骤。在 *Jenkins* 中，你可以使用 *parallel* 块轻松地并行化阶段。

你也可以在构建 Docker 镜像时执行单元测试和代码检查，使用如下所示的 *Dockerfile*：

```dockerfile
FROM python:3.11 as builder

WORKDIR /microservice
COPY ./requirements.txt /microservice/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /microservice/requirements.txt
COPY ./app /microservice/app
RUN pylint src
RUN python -m coverage run -m unittest
RUN python -m coverage xml
# 你必须在此处实现将单元测试覆盖率报告
# 发送到 SonarQube/SonarCloud 的功能

FROM gcr.io/distroless/python3.11 as final
COPY --from=builder /microservice /microservice
WORKDIR /microservice
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

上述解决方案的问题在于，你无法清楚地了解构建中哪部分失败了。你必须检查 Docker 构建命令的输出，以查看是代码检查还是单元测试失败。此外，你将无法再使用 SonarCloud GitHub Action。你必须在 *Dockerfile* 的 *builder* 阶段实现 SonarCloud 报告（在完成单元测试后，将单元测试覆盖率报告发送给 SonarCloud）。

## 12.2.5：部署

在 *部署* 阶段，已发布的软件会被自动部署。在 CI/CD 流水线成功运行后，软件组件可以被自动部署。这被称为 *持续部署*。请注意，*持续交付* 和 *持续部署* 的缩写都是 CD。这可能会导致不幸的误解。持续交付是关于自动发布软件，而持续部署是关于将已发布的软件自动部署到一个或多个环境。这些环境包括，例如，CI/CD 环境、预发布环境以及最终的生产环境。自动化软件部署有多种方式。一种现代且流行的方式是使用 GitOps，它使用一个或多个 Git 仓库，通过声明式方法定义到不同环境的自动部署。GitOps 可以配置为在发布新软件时自动更新环境。这通常用于 CI/CD 环境，该环境应始终保持最新并包含最新的软件组件版本。

GitOps 也可以配置为定期自动部署到预发布环境。预发布环境复制了生产环境。它是在软件部署到生产环境之前执行端到端功能和非功能测试的环境。你可以使用多个预发布环境来加速向生产环境的持续部署。至关重要的是，在部署到生产环境之前完成所有必要的测试。测试可能需要几天时间来验证软件的稳定性。如果在预发布环境中测试需要三天，并且你设置了三个预发布环境，那么你就可以每天部署到生产环境。另一方面，如果在预发布环境中测试需要一周时间，而你只有一个预发布环境，那么你只能每周部署一次到生产环境（此处假设所有测试都成功执行）。部署到生产环境也可以自动化。或者，可以在预发布环境中成功完成所有测试后手动触发。

## 12.2.6：运维

*运维* 是软件在生产环境中运行的阶段。在此阶段，需要确保软件更新（如安全补丁）得到及时部署。此外，生产环境的基础设施和平台应保持最新和安全。

## 12.2.7：监控

*监控* 是对已部署的软件系统进行监控以检测任何可能问题的阶段。监控应尽可能自动化。可以通过定义规则来实现自动化，当软件系统运行需要人工干预时触发警报。这些警报通常基于从微服务、基础设施和平台收集的各种指标。*Prometheus* 是一个流行的系统，用于收集指标、可视化它们并触发警报。

基本的监控工作流遵循以下路径：

1.  监控警报
2.  如果触发了警报，在相关仪表板中调查指标
3.  检查相关服务的日志以查找错误

## 12.2.7.1：记录到标准输入

每个服务必须记录到标准输出。如果你的微服务使用的是将日志输出到标准输出的第三方库，请选择一个允许你配置日志格式的库，或者将日志格式的可配置性作为对该库的增强功能提出请求。选择一个标准化的日志格式并在所有微服务中使用，例如，使用Syslog格式或OpenTelemetry日志数据模型（在后续章节中定义）。将每个微服务的日志收集到一个集中的位置，比如ElasticSearch数据库。

## 12.2.7.2：分布式追踪

将微服务与分布式追踪工具（如Jaeger）集成。分布式追踪工具收集微服务发出的网络请求的相关信息。

## 12.2.7.3：指标收集

定义需要从每个微服务收集哪些指标。典型的指标要么是计数器（例如，处理的请求数或请求错误数），要么是仪表盘（例如，当前的CPU/内存使用率）。收集计算服务级别指标（SLIs）所需的指标。下面列出了SLIs的五个类别以及每个类别的几个示例。

- 可用性
    - 服务是否宕机？
    - 依赖的服务是否宕机？
- 错误率
    - 服务因崩溃或无响应而重启的次数
    - 消息处理错误
    - 请求错误
    - 其他错误
    - 可以通过设置指标标签来监控不同的错误。例如，如果你有一个`_requesterrors`计数器，并且一个请求产生了内部服务器错误，你可以将`_requesterrors`计数器增加一个带有标签`_internal_servererror`的值。
- 延迟
    - 消息或请求处理时长
- 吞吐量
    - 处理的消息/请求数量
- 饱和度
    - 资源使用情况，例如CPU/内存/磁盘使用率与请求量的对比

使用必要的代码对你的微服务进行插桩以收集指标。这可以使用指标收集库（如Prometheus）来完成。

## 12.2.7.4：指标可视化

为每个微服务创建一个主仪表板来展示SLIs。你还必须展示*服务级别目标*（SLOs）。当所有SLOs都满足时，仪表板应以绿色显示SLI值。如果某个SLO未满足，相应的SLI值应以红色显示。你也可以使用黄色和橙色来表示SLO仍然满足，但SLI值不再是最优的。使用与指标收集工具集成的可视化工具，例如与Prometheus集成的Grafana。你通常可以将指标仪表板作为微服务部署的一部分进行部署。如果你使用Kubernetes、Prometheus和Grafana，你可以在使用Grafana Operator时将Grafana仪表板创建为自定义资源（CRs）。

## 12.2.7.5：告警

要定义告警规则，首先定义服务级别目标（SLOs），并基于它们制定告警规则。SLO的一个例子：“服务错误率必须低于x%”。如果某个SLO无法满足，则应触发告警。如果你使用Kubernetes和Prometheus，你可以使用Prometheus Operator和PrometheusRule CRs来定义告警。

## 12.2.8：软件系统告警仪表板示例

下面是一个Grafana仪表板的示例，用于可视化软件系统中的活动告警。

## 告警仪表板

![](img/cbd069395d7b824346b69b1f92e0fb4a_574_0.png)

图12.2. DevSecOps图

## 12.2.9：微服务Grafana仪表板示例

下面是一个Grafana仪表板的示例，用于可视化单个微服务的SLOs和SLIs。仪表板顶部展示SLOs，其下方是五个手风琴（accordion），第一个手风琴已打开以显示其内部的图表。在下图中，SLO 2以红色背景显示，可能表示过去一小时内的错误数量，例如。

## 告警仪表板

| SLO 1 | SLO 2 | SLO 3 | SLO 4 |
| :---: | :---: | :---: | :---: |
| 0 | 25 | 1 | 3 |
| SLO 5 | SLO 6 | SLO 7 | SLO 8 |
| 10,000 | 100 | 0 | 10 |

| ▽ 可用性 |
| :--- |
| SLI 1 图表 | SLI 2 图表 |
| SLI 3 图表 | SLI 4 图表 |

- ▷ 错误率
- ▷ 延迟
- ▷ 吞吐量
- ▷ 饱和度
- ▷ 其他

图12.3. DevSecOps图

软件运维人员通过以下方式与DevOps生命周期的软件开发端进行联系：

- 寻求技术支持
- 提交错误报告
- 提交改进建议

第一种方式将导致一个已解决的案例或错误报告。后两种方式将进入DevOps生命周期的*计划*阶段。错误报告通常会立即进入*编码*阶段，具体取决于故障的严重程度。

## 12.2.9.1：日志记录

在软件组件中实现日志记录，使用以下日志严重级别：

- (CRITICAL/FATAL)
- ERROR
- WARNING
- INFO
- DEBUG
- TRACE

我通常根本不使用CRITICAL/FATAL严重级别。最好使用ERROR严重级别报告所有错误，因为这样可以使用单个关键字轻松查询错误日志，例如：

```
kubectl logs <pod-name> | grep ERROR
```

你可以在日志消息本身中添加关于错误严重性/致命性的信息。当你记录一个有可用解决方案的错误时，你应该在日志消息中告知用户该解决方案，例如，提供指向故障排除指南的链接或给出可用于搜索故障排除指南的错误代码。

不要使用INFO严重级别记录太多信息，因为日志太多可能会难以阅读。仔细考虑哪些应该使用INFO严重级别记录，哪些可以使用DEBUG严重级别记录。微服务的默认日志级别应为WARNING或INFO。

使用TRACE严重级别仅记录跟踪信息，例如，与处理单个请求、事件或消息相关的详细信息。

如果你正在实现一个第三方库，该库应该允许在记录内容时自定义日志。应该有一种方法来设置日志级别，并允许使用该库的代码自定义日志条目的写入格式。否则，第三方库的日志条目将以与微服务本身日志条目不同的格式出现在日志中。

## 12.2.9.2：OpenTelemetry日志数据模型

本节描述OpenTelemetry日志数据模型1.12.0版本的精髓（请查看[https://github.com/open-telemetry/opentelemetry-specification](https://github.com/open-telemetry/opentelemetry-specification)以获取可能的更新）。

一个日志条目是一个JSON对象，包含以下属性：

| 字段名 | 描述 |
|---|---|
| Timestamp | 事件发生的时间。自Unix纪元以来的纳秒数 |
| TraceId | 请求跟踪ID |
| SpanId | 请求跨度ID |
| SeverityText | 严重性文本（也称为日志级别） |
| SeverityNumber | 严重性的数值 |
| Body | 日志条目的主体。你可以在实际日志消息之前包含ISO 8601时间戳和严重性/日志级别 |
| Resource | 描述日志条目的来源 |
| Attributes | 关于日志事件的附加信息。这是一个JSON对象，可以在其中给出自定义属性 |

下面是一个根据OpenTelemetry日志数据模型的示例日志条目。

```json
{
  "Timestamp": "1586960586000000000",
  "TraceId": "f4dbb3edd765f620",
  "SpanId": "43222c2d51a7abe3",
  "SeverityText": "ERROR",
  "SeverityNumber": 9,
  "Body": "20200415T072306-0700 ERROR Error message comes here",
  "Resource": {
    "service.namespace": "default",
    "service.name": "my-microservice",
    "service.version": "1.1.1",
    "service.instance.id": "my-microservice-34fggd-56faae"
  },
  "Attributes": {
    "http.status_code": 500,
    "http.url": "http://example.com",
    "myCustomAttributeKey": "myCustomAttributeValue"
  }
}
```

上述JSON格式的日志条目在控制台上可能难以作为纯文本阅读，例如，在Kubernetes集群中使用`kubectl logs`命令查看Pod日志时。你可以创建一个小脚本，仅从每个日志条目中提取`Body`属性值。

## 12.2.9.3：PrometheusRule示例

*PrometheusRule*自定义资源（CRs）可用于定义触发告警的规则。在下面的示例中，当请求延迟的中位数（以秒为单位）大于1（request_latencies_in_seconds{quantile="0.5"} > 1）时，将触发一个主要严重级别的*example-microservice-high-request-latency*告警。

## 13：附录 A

图 13.1. utils.py

```python
import os
import traceback

from pydantic import BaseModel


def is_pydantic(object: object):
    return type(object).__class__.__name__ == 'ModelMetaclass'


def to_entity_dict(dto: BaseModel):
    entity_dict = dict(dto)
    for key, value in entity_dict.items():
        try:
            if (
                isinstance(value, list)
                and len(value)
                and is_pydantic(value[0])
            ):
                entity_dict[key] = [
                    item.Meta.orm_model(**to_entity_dict(item))
                    for item in value
                ]
            elif is_pydantic(value):
                entity_dict[key] = value.Meta.orm_model(
                    **to_entity_dict(value)
                )
        except AttributeError:
            raise AttributeError(
                f'Found nested Pydantic model in {dto.__class__} but Meta.orm_model was not specified.'
            )
    return entity_dict


def get_stack_trace(error: Exception | None):
    return (
        repr(traceback.format_exception(error))
        if error and os.environ.get('ENV') != 'production'
        else None
    )
```

图 13.2. dtos/InputOrder.py

```python
from pydantic import BaseModel

from .OrderItem import OrderItem

class InputOrder(BaseModel):
    userId: str
    orderItems: list[OrderItem]

    class Config:
        orm_mode = True
```

图 13.3. dtos/OrderItem.py

```python
from pydantic import BaseModel, PositiveInt

from ..entities.OrderItem import OrderItem as OrderItemEntity

class OrderItem(BaseModel):
    id: int
    salesItemId: str
    quantity: PositiveInt

    class Config:
        orm_mode = True

    class Meta:
        orm_model = OrderItemEntity
```

图 13.4. dtos/OutputOrder.py

```python
from pydantic.main import BaseModel

from .OrderItem import OrderItem

class OutputOrder(BaseModel):
    id: str
    userId: str
    orderItems: list[OrderItem]

    class Config:
        orm_mode = True
```

图 13.5. entities/Base.py

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

图 13.6. entities/Order.py

```python
from sqlalchemy import BigInteger
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .Base import Base
from .OrderItem import OrderItem


class Order(Base):
    __tablename__ = 'orders'

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True)
    userId: Mapped[int] = mapped_column(BigInteger(), index=True)
    orderItems: Mapped[list[OrderItem]] = relationship(lazy='joined')
```

图 13.7. entities/OrderItem.py

```python
from sqlalchemy import BigInteger, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .Base import Base


class OrderItem(Base):
    __tablename__ = 'orderitems'
    __table_args__ = (
        PrimaryKeyConstraint('orderId', 'id', name='orderitems_pk'),
    )

    id: Mapped[int]
    salesItemId: Mapped[int] = mapped_column(BigInteger())
    quantity: Mapped[int]
    orderId: Mapped[int] = mapped_column(ForeignKey('orders.id'))
```

图 13.8. errors/OrderServiceError.py

```python
from typing import Final

class OrderServiceError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        cause: Exception | None = None,
    ):
        self.__status_code: Final = status_code
        self.__message: Final = message
        self.__cause: Final = cause

    @property
    def status_code(self) -> int:
        return self.__status_code

    @property
    def message(self) -> str:
        return self.__message

    @property
    def cause(self) -> Exception | None:
        return self.__cause
```

图 13.9. errors/DatabaseError.py

```python
from .OrderServiceError import OrderServiceError

class DatabaseError(OrderServiceError):
    def __init__(self, cause: Exception):
        super().__init__(500, 'Database error', cause)
```

图 13.10. errors/EntityNotFoundError.py

```python
from .OrderServiceError import OrderServiceError

class EntityNotFoundError(OrderServiceError):
    def __init__(self, entity_name: str, entity_id: int):
        super().__init__(
            404, f'{entity_name} with id {entity_id} not found'
        )
```

图 13.11. graphqltypes/InputOrder.py

```python
import strawberry

from ..dtos.InputOrder import InputOrder
from .InputOrderItem import InputOrderItem


@strawberry.experimental.pydantic.input(model=InputOrder)
class InputOrder:
    userId: strawberry.auto
    orderItems: list[InputOrderItem]
```

图 13.12. graphqltypes/InputOrderItem.py

```python
import strawberry

from ..dtos.OrderItem import OrderItem


@strawberry.experimental.pydantic.input(model=OrderItem, all_fields=True)
class InputOrderItem:
    pass
```

图 13.13. graphqltypes/OutputOrder.py

```python
import strawberry

from ..dtos.OutputOrder import OutputOrder
from .OutputOrderItem import OutputOrderItem


@strawberry.experimental.pydantic.type(model=OutputOrder)
class OutputOrder:
    id: strawberry.auto
    userId: strawberry.auto
    orderItems: list[OutputOrderItem]
```

图 13.14. graphqltypes/OutputOrderItem.py

```python
import strawberry

from ..dtos.OrderItem import OrderItem


@strawberry.experimental.pydantic.type(model=OrderItem, all_fields=True)
class OutputOrderItem:
    pass
```

## 14：附录 B

以下是 `proto_to_dict` 函数的源代码：

**图 14.1. grpc/proto_to_dict.py**

```python
# This is free and unencumbered software released into the public domain
# by its author, Ben Hodgson <ben@benhodgson.com>.

# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.

# In jurisdictions that recognise copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# For more information, please refer to <http://unlicense.org/>

from google.protobuf.descriptor import FieldDescriptor

EXTENSION_CONTAINER = '__X'

TYPE_CALLABLE_MAP = {
    FieldDescriptor.TYPE_DOUBLE: float,
    FieldDescriptor.TYPE_FLOAT: float,
    FieldDescriptor.TYPE_INT32: int,
    FieldDescriptor.TYPE_INT64: int,
    FieldDescriptor.TYPE_UINT32: int,
    FieldDescriptor.TYPE_UINT64: int,
    FieldDescriptor.TYPE_SINT32: int,
    FieldDescriptor.TYPE_SINT64: int,
    FieldDescriptor.TYPE_FIXED32: int,
    FieldDescriptor.TYPE_FIXED64: int,
    FieldDescriptor.TYPE_SFIXED32: int,
    FieldDescriptor.TYPE_SFIXED64: int,
    FieldDescriptor.TYPE_BOOL: bool,
    FieldDescriptor.TYPE_STRING: str,
    FieldDescriptor.TYPE_BYTES: lambda b: b.encode('base64'),
    FieldDescriptor.TYPE_ENUM: int,
}
```

def repeated(type_callable):
    return lambda value_list: [
        type_callable(value) for value in value_list
    ]

def enum_label_name(field, value):
    return field.enum_type.values_by_number[int(value)].name

def proto_to_dict(
    pb, type_callable_map=TYPE_CALLABLE_MAP, use_enum_labels=False
):
    result_dict = {}
    extensions = {}
    for field, value in pb.ListFields():
        type_callable = _get_field_value_adaptor(
            pb, field, type_callable_map, use_enum_labels
        )
        if field.label == FieldDescriptor.LABEL_REPEATED:
            type_callable = repeated(type_callable)

        if field.is_extension:
            extensions[str(field.number)] = type_callable(value)
            continue

        result_dict[field.name] = type_callable(value)

    if extensions:
        result_dict[EXTENSION_CONTAINER] = extensions
    return result_dict

def _get_field_value_adaptor(
    pb, field, type_callable_map=TYPE_CALLABLE_MAP, use_enum_labels=False
):
    if field.type == FieldDescriptor.TYPE_MESSAGE:
        # 递归编码 protobuf 子消息
        return lambda pb: proto_to_dict(
            pb,
            type_callable_map=type_callable_map,
            use_enum_labels=use_enum_labels,
        )

    if use_enum_labels and field.type == FieldDescriptor.TYPE_ENUM:
        return lambda value: enum_label_name(field, value)

    if field.type in type_callable_map:
        return type_callable_map[field.type]

    raise TypeError(
        '字段 %s.%s 的类型 id %d 无法识别'
        % (pb.__class__.__name__, field.name, field.type)
    )

def get_bytes(value):
    return value.decode('base64')

REVERSE_TYPE_CALLABLE_MAP = {
    FieldDescriptor.TYPE_BYTES: get_bytes,
}

def _get_field_mapping(pb, dict_value, strict):
    field_mapping = []
    for key, value in dict_value.items():
        if key == EXTENSION_CONTAINER:
            continue
        if key not in pb.DESCRIPTOR.fields_by_name:
            if strict:
                raise KeyError(
                    '%s 没有名为 %s 的字段' % (pb, key)
                )
            continue
        field_mapping.append(
            (
                pb.DESCRIPTOR.fields_by_name[key],
                value,
                getattr(pb, key, None),
            )
        )

    for ext_num, ext_val in dict_value.get(
        EXTENSION_CONTAINER, {}
    ).items():
        try:
            ext_num = int(ext_num)
        except ValueError:
            raise ValueError('扩展键必须是整数。')
        if ext_num not in pb._extensions_by_number:
            if strict:
                raise KeyError(
                    '%s 没有编号为 %s 的扩展。也许你忘记导入它了？'
                    % (pb, key)
                )
            continue
        ext_field = pb._extensions_by_number[ext_num]
        pb_val = None
        pb_val = pb.Extensions[ext_field]
        field_mapping.append((ext_field, ext_val, pb_val))

    return field_mapping

def _string_to_enum(field, input_value):
    enum_dict = field.enum_type.values_by_name
    try:
        input_value = enum_dict[input_value].number
    except KeyError:
        raise KeyError(
            '`%s` 不是字段 `%s` 的有效值'
            % (input_value, field.name)
        )
    return input_value