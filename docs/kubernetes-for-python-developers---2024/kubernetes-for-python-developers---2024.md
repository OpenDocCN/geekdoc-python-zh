

# 面向Python开发者的Kubernetes

Lakshmi Narasimhan

版权所有 © 2024 Lakshmi Narasimhan
保留所有权利。

## 目录

- 第1章：元信息
  - 本书的目标读者
  - 代码示例
  - 开发者为何应学习Kubernetes
  - 我们将构建什么
  - 你需要哪些先决条件
- 第2章：容器回顾
  - 容器如何工作
    - 命名空间与cgroups
      - 联合文件系统
        - 联合文件系统的结构
        - 只读层的作用
        - 清理
    - 容器网络
      - 示例：桥接网络
    - 进程隔离
      - 示例场景
    - 构建我们的应用容器
- ENTRYPOINT与CMD的区别
  - 结合ENTRYPOINT与CMD
- 练习1：配置gunicorn
- 其他容器引擎
  - Docker
  - containerd
  - CRI-O
  - Podman
  - rkt（已弃用）
  - LXD
- docker compose
- 简化的多容器编排
- 轻松配置
- 环境标准化与版本控制
- 开发效率提升
- 便捷测试与CI/CD集成
- 本地开发与测试
- 示例用例
  - 练习2：设置实时重载
  - 练习3：配置Nginx
- 注册表
- 为何需要容器注册表？
- 注册表示例
- 设置容器注册表
  - 练习4：在Ghcr中构建并推送新标签。
- 总结
- 第3章：Kubernetes入门
- Kubernetes如何工作
  - 自动化部署
  - 扩缩容
  - 健康监控与自愈
  - 网络
  - 服务发现
  - 配置管理
  - 资源分配
  - 更新与回滚
- Kubernetes集群的组件
  - kube-apiserver
  - etcd
  - kube-scheduler
  - Kube Controller Manager
  - Kubelet
  - Kube-proxy
- 容器运行时
- 其他容器编排工具
- 创建你的第一个Kubernetes集群
  - 使用真实集群的理由
  - 在DigitalOcean创建集群
- 设置你的kubernetes集群
- 总结
- 第4章：Pod入门
- Kubectl
- Kubeconfig
- Kubeconfig的组成部分
  - 集群
  - 上下文
  - 用户
  - 当前上下文
  - 偏好设置
- 为何使用TLS认证？
  - 加密
  - 认证
  - 完整性
  - 吊销与过期
- 合规性与标准
- 与集群交互
- 关于kubernetes版本的说明
- 你与Kubernetes的首次交互
- Pod
  - 容器分组
  - 共享资源与网络
  - 简化管理与通信
  - 实现细粒度控制
  - 促进服务发现与负载均衡
  - 运行一个简单的pod
  - 练习5：更改pod规格并重新应用
  - 练习6：更改pod的名称
  - 练习7：运行Apache httpd服务器
  - Pod重启行为
  - CrashLoopBackOff是什么意思？
  - 练习8：配置sleepy pod使其不重启
- 描述一个pod
  - 练习9：描述bash-sleep pod。
- 事件
- Kubernetes资源
- Kubectl的工作原理
- Pod中的多个容器
  - 练习10：在pod中运行3个容器。
  - 练习11：2个容器的变体
- 一个pod中应有多少容器
  - 实际考虑因素
  - 最佳实践
  - 技术限制
- 端口转发
  - 练习12：将8008端口转发
  - 练习13：转发多个容器的端口
- 获取pod的日志
  - 简化日志收集
  - 便于日志管理
  - 支持十二要素应用方法论
- 总结
- 第5章：Pod进阶
- 进入pod内部
  - exec的用例
  - 练习14：打印端口号
- 在不同节点上运行pod
- 存活探针
- 就绪探针
- 存活探针与就绪探针的最佳实践
- 命名空间
- 为何需要命名空间
  - 环境隔离
  - 资源组织
  - 访问控制
  - 配额管理
- 创建我们的第一个命名空间
- 为何需要命名空间？
  - 集群杂乱
  - 访问控制挑战
  - 资源配额复杂性
- 默认命名空间
  - 默认命名空间的特性与用例
  - 注意事项
  - 练习15：在新命名空间中部署nginx pod
  - 练习16：删除一个命名空间
- Pod网络
  - CNI（容器网络接口）
  - Pod间通信
  - 节点到Pod通信
  - 外部到Pod通信
  - Pod到外部通信
  - 网络策略
- Pod网络示例
  - 练习17
- 部署flask应用
  - 关于标签的说明
  - 镜像摘要
  - 练习18：通过指定不同标签部署pod
  - 练习19：通过指定镜像SHA摘要部署pod
- 镜像拉取策略
  - Always
  - IfNotPresent
  - Never
  - 默认行为
  - 选择正确的imagePullPolicy
- 集群中应有多少pod
  - 每个节点的Pod考虑因素
  - 每个集群的Pod考虑因素
  - 可扩展性指南
- 总结
- 引入的关键概念
- 第6章：超越Pod
- 副本集
- 为何需要副本集
- 部署
- 为何需要部署
- 副本集与部署的区别
- 应使用哪一个？
- 副本集为何仍然存在？
- 部署策略
- 滚动更新（默认策略）
- 重建策略
- 练习20：使用重建策略为nginx创建部署。
- 蓝绿部署
- 金丝雀部署
- 练习21：将flask应用部署为部署
- 服务
- 为何需要服务
- 服务类型
- CoreDNS入门
- 示例
- 关于端口名称的说明
  - 端口名称的目的
  - 端口名称示例
  - 在Pod探针中使用端口名称
- 功能完备的Flask redis应用
  - 练习22：进行更改并重新部署
  - 练习23：重新部署redis flask应用
- 总结
- 第7章：更进一步
- 配置映射
- 为何需要配置映射
- 我们的第一个配置映射
- 卷与卷挂载
  - 卷
  - 卷挂载
  - 它们如何协同工作
- 从部署创建配置映射
  - 练习24：使用配置映射并在同一pod中运行2个nginx。
- 从配置映射获取环境变量
  - 练习25：应用程序错误
- 配置映射与十二要素原则
- 支持其他十二要素原则
- 密钥
- 为何使用密钥？
- 示例：nginx认证
  - 步骤1：创建htpasswd文件
  - 步骤2：从htpasswd文件创建Kubernetes密钥
  - 步骤3：配置Nginx部署以使用密钥
- 检查密钥
- 为何加密而非编码
- 加密的必要性
- 密钥管理的最佳实践
- 其他类型的密钥
  - 不透明密钥
  - 服务账户令牌密钥
  - Docker注册表密钥
  - 基本身份验证密钥
  - SSH身份验证密钥
  - TLS密钥
- 为何有不同类型的密钥？
- 将密钥注入为环境变量
  - 练习26：配置redis部署

## 镜像拉取密钥

- 为何需要镜像拉取密钥
- 镜像拉取密钥的工作原理
- 创建镜像拉取密钥
- 在部署中使用镜像拉取密钥
- 练习 27：提供不同的密钥
- 练习 28：使用多个镜像拉取密钥

## 初始化容器

### 为何需要初始化容器

### 在 Flask-Redis 部署中使用初始化容器

- Redis 检查脚本
- 与非初始化容器的对比
  - 相似之处
  - 不同之处

### 使用场景

- 等待服务可用
- 数据库迁移与模式更新
- 配置文件准备
- 动态凭证收集
- 访问控制与安全设置
- 数据预加载
- 服务注册
- 健康检查与就绪探针

## 存储持久化数据

- 抽象与解耦
- 共享资源管理
- 动态配置与可移植性
- 访问控制与安全性
- 与其他集群资源的一致性
- ReadWriteOnce (RWO)
- ReadOnlyMany (ROX)
- ReadWriteMany (RWX)
- ReadWriteOncePod (RWOP)（自 Kubernetes v1.22 起可用）
- 注意事项
  - Retain
  - Delete
  - Recycle（已弃用）

## 动态配置

- 动态配置的工作原理
- 动态配置的优势
- 示例场景
- 实际说明
- PV 与 PVC 的关键区别
- 关键区别总结

## 总结

## 第 8 章：更健壮的设置

### 将 Redis 作为有状态集

- 组件
- 复制过程
- 临时 Pod 身份的问题
  - 无有序、优雅的扩缩容或更新
- 对有状态集的需求
  - 将 Redis 建模为有状态集
- 关于持久卷声明模板
  - 持久化存储的自动化
  - 模板结构
- 无头服务
  - 练习 29：扩展 Redis 有状态集。
  - 练习 30：使用 Redis Sentinel 为 Redis 设置高可用性。
- 其他场景中的有状态集
- 服务账户与 RBAC
  - 示例
  - 创建服务账户
  - Kubernetes JWT 的组成部分
  - 在 Kubernetes 中的使用
  - 练习 31：更新角色以获取服务。
- Kubernetes 注解
  - 注解的关键特性
  - 注解的使用场景
  - 练习 32：创建长期令牌以访问 Pod 日志

### Pod 内的服务账户

- 强大的角色

### 将 Pod 分散到各个节点

## Ingress

- 练习 33：多服务 Ingress

## Ingress 控制器

## 总结

- 理解 Redis 与有状态集
- 创建和管理服务账户
- 将 Pod 分散到各个节点
- Ingress 与 Ingress 控制器

## 第 9 章：结语

## 自信地部署到 Kubernetes

## 设置 CD 流水线

## 设置脚手架

## GitHub Actions

- GitHub Actions 的核心组件
- GitHub Actions 的工作原理

### 从这里开始

## 展望未来

## 附录 A：常见场景故障排除

### Pod 未启动

- 原因
- 诊断
- 修复

### 镜像未找到

- 原因
- 诊断
- 修复

### 标签不正确

- 原因
- 诊断
- 修复

### 未暴露正确的端口

- 原因
- 诊断
- 修复

### 资源不足

- 原因
- 诊断
- 修复

### 镜像架构不同

- 原因
- 诊断
- 修复

### 未找到 CRD

- 原因
- 诊断
- 修复

## 未授权

- 原因
- 诊断
- 修复

## 结论

## 附录 B：练习解答

- 练习 1：配置 gunicorn
- 练习 2：设置实时重载
- 练习 3：配置 Nginx
- 练习 4：在 Ghcr 中构建并推送新标签。
- 练习 5：更改 Pod 规范并重新应用
- 练习 6：更改 Pod 的名称
- 练习 7：运行 Apache httpd 服务器
- 练习 8：配置休眠 Pod 使其不崩溃
- 练习 9：描述 bash-sleep Pod。
- 练习 10：在 Pod 中运行 3 个容器。
- 练习 11：2 个容器的扭曲变体
- 练习 12：端口转发到 8008
- 练习 13：端口转发多个容器
- 练习 14：打印端口号
- 练习 15：在新命名空间中部署 nginx Pod
- 练习 16：删除命名空间
- 练习 17：相同 IP 还是不同 IP？
- 练习 18：通过指定不同标签部署 Pod
- 练习 19：通过指定镜像 SHA 摘要部署 Pod。
- 练习 20：使用 recreate 策略为 nginx 创建部署。
- 练习 21：将 Flask 应用部署为 Deployment
- 练习 22：进行更改并重新部署
- 练习 23：重新部署 Redis Flask 应用
- 练习 24：使用 ConfigMap 并在同一个 Pod 中运行 2 个 nginx。
- 练习 25：应用程序错误
- 练习 26：配置 Redis Deployment
- 练习 27：提供不同的密钥
- 练习 28：使用多个镜像拉取密钥
- 练习 29：扩展 Redis 有状态集。
- 练习 30：使用 Redis Sentinel 为 Redis 设置高可用性。
- 练习 31：更新角色以获取服务。
- 练习 32：创建长期令牌以访问 Pod 日志
- 练习 33：多服务 Ingress

## 第 1 章：元信息

### 本书的目标读者

如果你是一名渴望学习 Kubernetes 的 Python 开发者，或者只是对其感到好奇，那么这本书就是为你量身定制的。我假设你之前没有 Kubernetes 的知识，不过如果你了解一些也无妨。

话虽如此，来自其他语言（如 PHP、Ruby/Rails、Golang、Node.js 等）的开发者也会发现这本书很有用。我希望在整本书中使用特定的技术栈/编程语言，这样就不会通过在不同技术栈中给出不同概念的例子来分散你（读者）的注意力。如果有足够的兴趣，我计划发布其他语言特定版本的本书（例如面向 PHP 开发者的 Kubernetes、面向 Node.js 开发者的 Kubernetes 等）。请通过 [lakshmi@lakshminp.com](mailto:lakshmi@lakshminp.com) 给我发邮件，或者考虑在 [https://www.lakshminp.com/kubernetes-for-python-developers/](https://www.lakshminp.com/kubernetes-for-python-developers/) 注册以获取本书的更新。

如果你是一名 DevOps 工程师，你将从另一个角度（即应用上下文以及在 Kubernetes 中容器化、部署和维护应用程序所涉及的内容）获得一两个见解。我不会讨论 Kubernetes 的运维方面（例如，如何使用基础设施即代码启动 Kubernetes 集群，如何在不停机的情况下升级集群等），因为那是一个全新的话题，需要另一整本书来讨论。

# 代码示例

本书各章节的代码示例可在 [https://github.com/badri/kubernetes-for-python-devs](https://github.com/badri/kubernetes-for-python-devs) 找到。

最后一章包含一个 CD 流水线和一个独立的代码仓库，可在 [https://github.com/badri/flask-redis-cd](https://github.com/badri/flask-redis-cd) 找到。

# 为什么开发者应该学习 Kubernetes

我不必向你推销 Kubernetes 有多受欢迎，或者它如何从根本上改变了应用程序在云中的部署、扩展和管理方式。但是，作为开发者，你为什么要学习它呢？你的组织里已经有人在做这件事了，那为什么还要费心去学呢？

它能让你深入理解现代的、云原生的架构，从而能够设计出具有弹性、可扩展性和高可用性的系统。对于本书的主要读者——Python 开发者来说，无论是使用 Django 还是 Flask 这样的 Web 框架，Kubernetes 都能简化容器化应用程序的过程，并通过 ConfigMaps 和 Secrets 管理特定环境的配置，使你的应用程序更具可移植性，也更容易在不同环境中管理。总的来说，它会让你成为一名更好的开发者。引用保罗·格雷厄姆在其文章《超越平庸》（[https://paulgraham.com/avg.html](https://paulgraham.com/avg.html)）中的话：

> Lisp 值得学习，因为当你最终理解它时，你会获得深刻的启蒙体验；这种体验将使你在余生中成为一名更好的程序员，即使你实际上很少使用 Lisp 本身。

他谈论的是为什么学习 Lisp 编程语言会让你成为更好的开发者，即使你最终并不会像日常工作那样经常使用 Lisp。在我们的情况下，唯一的区别是，你将在日常工作中大量使用 Kubernetes。

此外，Kubernetes 在整个行业被广泛采用，得到了 AWS、Google Cloud 和 Azure 等云提供商的支持，确保了 Kubernetes 技能具有高度的相关性和需求。通过掌握 Kubernetes，你可以提高自己的市场竞争力，开辟新的职业机会，并更有效地为组织的数字化转型战略做出贡献。本质上，学习 Kubernetes 不仅仅是为了跟上最新的技术趋势；更是为了让自己掌握构建下一代 Web 应用程序所需的工具和方法论。

# 我们将构建什么

在本书中，我们将踏上一段激动人心的旅程，将 Python 开发的世界与 Kubernetes 强大的编排能力连接起来。我们的冒险最终将创建一个动态的、可扩展的 Flask 应用程序，并借助 Kubernetes 带来的弹性和灵活性得到增强。以下是你将遇到的内容：

**一个 Flask 应用程序：** 我们将从基础开始——开发一个简单但功能齐全的 Flask 应用程序。这个 Python Web 应用程序将作为我们探索的基石，为 Python 开发者提供一个熟悉的起点。随着我们的进展，你将看到即使是最简单的应用程序也能从 Kubernetes 中受益。

**使用 Docker 进行容器化：** 接下来，我们将把 Flask 应用程序容器化，将其封装在 Docker 容器中。这一步对于理解应用程序如何为在 Kubernetes 环境中部署做准备至关重要，它确保了不同环境之间的一致性，并简化了部署流程。

**部署到 Kubernetes：** 现在是时候将我们的应用程序部署到 Kubernetes 了。你将学习如何定义和管理部署，使我们的 Flask 应用程序能够在 Kubernetes 生态系统中无缝部署、管理和扩展。这包括创建和配置 Pod、Deployment 和 Service，以确保我们的应用程序可访问且可扩展。

**高级 Kubernetes 功能：** 随着我们的应用程序变得越来越复杂，我们将引入 Redis 来存储状态，展示有状态应用程序如何在 Kubernetes 中运行。你将获得关于持久化存储、密钥管理以及配置服务间通信的实践经验，这体现了真实世界的应用程序需求。

**持续部署（CD）：** 最后，我们将使用 GitHub Actions 自动化部署流程，实现一个持续部署流水线，将你的代码从 git push 一直带到部署在 Kubernetes 中的应用程序，展示 DevOps 的最佳实践。

**结论：** 在本书结束时，你将拥有一个功能齐全的 Flask 应用程序，其架构设计使其能够在 Kubernetes 环境中蓬勃发展。你不仅将理解“如何做”，还将理解“为什么这样做”，从而让你能够自信且熟练地将 Kubernetes 应用于你自己的 Python 项目。

# 你需要哪些先决条件

为了充分利用本书，你应该熟悉以下内容：

- 基本的命令行使用
- Flask 或 Django 框架
- 基本的 Git 命令
- Docker 客户端设置

以下是你需要拥有的一套工具：

1. git
2. 终端
3. 一个不错的 IDE（我推荐 VSCode）
4. Docker 客户端。有关如何设置的说明，请参阅 [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)。

最后，你需要一个云账户。许多书籍和教程教你如何使用本地或像 Minikube 这样受控的 Kubernetes 环境来设置某些东西。对此我有明确的观点，我强烈建议你创建一个真实的 Kubernetes 集群来运行本书中的示例。有很多云供应商提供免费额度来设置 Kubernetes，我推荐 Digitalocean，你可以通过推荐链接（[https://m.do.co/c/dfe1fd4e27d6](https://m.do.co/c/dfe1fd4e27d6)）或不使用推荐链接（[https://www.digitalocean.com/](https://www.digitalocean.com/)）进行注册。如果由于某些原因，你无法使用云提供商创建真实的 Kubernetes 集群，你可以使用 Minikube。但你将无法使用 Minikube 运行最后一章中的持续交付示例。

# 第 2 章：容器回顾

与每个虚拟机都需要一份完整的操作系统副本不同，容器共享主机系统的内核，并将应用程序进程彼此隔离。这带来了更低的开销、在相同硬件上运行更高密度的应用程序，并最终节省基础设施成本。此外，容器的轻量级特性和快速启动时间非常适合根据需求扩展应用程序，从而提高应用程序的可用性和性能。

使用容器也简化了应用程序的管理和更新。使用容器，更新应用程序或回滚到以前的版本可以像停止一个容器并使用更新后的镜像启动一个新容器一样简单。这种管理应用程序版本的原子性和速度在当今快节奏的开发环境中非常宝贵，因为敏捷性和响应能力是保持竞争力的关键。

在开发期间，你也应该在本地运行容器吗？简短的回答是肯定的。但请允许我详细说明。通过在本地容器化应用程序，你可以确保它在一个与生产环境非常相似的隔离环境中运行。这种“一次构建，随处运行”的方法显著减少了“在我的机器上可以运行”的问题，这是一个常见的陷阱，即由于环境差异，代码在开发环境和生产环境中的行为不同。容器抽象了底层操作系统、库和依赖项的差异，使应用程序更具可移植性，也更不容易出现环境问题。

此外，容器促进了从开发到生产的更平滑过渡。开发者在本地使用的容器与部署到生产环境的容器是同一个，这简化了开发生命周期，并最大限度地降低了部署错误的风险。这种无缝集成对于实施持续集成和持续部署（CI/CD）流水线是一个福音，允许更快、更可靠的软件发布。

# 容器如何工作

在底层，容器利用了几个关键的 Linux 内核特性：命名空间、控制组和联合文件系统，每个特性都在提供使容器如此强大的隔离和资源管理方面发挥着至关重要的作用。

## 容器架构

命名空间是 Linux 内核的一项功能，它将内核资源进行分区，使得每个分区内的进程都拥有其独立的资源实例。这意味着在一个命名空间中运行的进程无法看到或影响另一个命名空间中运行的进程，从而使得多个容器可以在同一台主机上运行而互不干扰。例如，PID（进程 ID）命名空间隔离了进程 ID 编号空间，这意味着一个 PID 命名空间中的进程可以与另一个命名空间中的进程拥有相同的 PID。这种隔离性还扩展到其他系统资源，如网络接口、挂载点和进程间通信资源。

简而言之，命名空间使得在其中运行的进程相信它拥有该资源的独立实例。命名空间可以限制对某些进程树、网络接口、用户 ID 或文件系统挂载的可见性。

控制组（cgroups）是容器用于管理和限制进程或进程组可使用资源量的另一项关键特性。这确保了没有单个容器能够独占系统资源，如 CPU、内存和 I/O 带宽，从而维护主机系统的整体稳定性和性能。例如，你可以有一个运行 Web 服务器的容器，其 CPU 使用限制为 50%，内存限制为 1GB，从而确保同一主机上运行的其他容器能够访问剩余的资源。

要运行一个 Ubuntu 容器实例，并将内存限制设置为 1 GB，CPU 访问限制为 1 个核心，可以使用以下命令：

```
$ docker run -it --memory="1g" -cpus="1.0" ubuntu
```

联合文件系统使容器变得轻量且快速。联合文件系统将多个目录组合成一个单一的目录视图，从而可以将一个可写目录（容器写入其更改的地方）叠加在一个只读目录（容器镜像）之上。这意味着当容器修改文件时，更改会被记录在可写层中，而不会修改底层的镜像。这种方法支持容器的不可变性和可移植性，因为基础镜像保持不变，可用于创建新的容器，而只有更改被单独存储和管理。

为了说明这些组件如何协同工作，想象一下在容器中部署一个简单的 Python Web 应用程序。当容器启动时，命名空间确保 Web 应用程序的进程在隔离的环境中运行，无法感知其他容器或主机系统上的进程。如果应用程序尝试访问网络，它将使用其命名空间隔离的网络接口进行访问。控制组限制应用程序的 CPU 和内存使用，防止其耗尽主机的资源。最后，应用程序从联合文件系统读取数据，其基础镜像以只读方式挂载，任何更改都写入单独的可写层。当容器停止或删除时，可写层可以被移除，而基础镜像保持不变，随时可以重用。

总而言之，容器依赖命名空间实现进程隔离，依赖控制组进行资源管理，并利用联合文件系统实现高效存储，三者协同工作，为运行应用程序提供了一个轻量、隔离且高效的环境。

让我们通过一个实际操作示例来说明命名空间、控制组和联合文件系统的概念，我们将运行一个简单的 Docker 容器。Docker 抽象了这些底层细节，使得无需手动操作即可更容易地理解容器如何利用这些特性。在这个示例中，我们将运行一个 Ubuntu 容器，安装一个基本工具，并检查它如何与主机隔离。

## 命名空间和控制组

首先，让我们以交互模式运行一个 Ubuntu 容器，并为其附加一个终端。这将允许我们在容器内执行命令。

```
docker run -it --rm ubuntu bash
```

此命令执行以下操作：

- `docker run`：创建并启动一个容器。
- `-it`：使容器具有交互性并为其附加一个终端。
- `--rm`：容器退出时自动删除它。
- `ubuntu`：指定要使用的 Docker 镜像（本例中为最新的 Ubuntu 镜像）。
- `bash`：执行 bash shell，为你提供容器内的命令提示符。

进入容器后，让我们检查 bash shell 的进程 ID：

```
echo $$ # 这显示当前 bash 会话的 PID。
```

你可能会看到一个较小的数字，如 1 或其他小数字，这表明在容器的 PID 命名空间内，该 shell 认为它是第一个（或早期的）进程，与主机的进程隔离。

作为对比，在你的主机机器上（在一个新的终端窗口中），你可以使用以下命令列出 Docker 容器中运行的进程：

```
docker ps
```

要从主机的角度查看容器内 bash 进程的实际 PID：

```
ps aux | grep bash
```

你会注意到主机上的 PID 不同，并且通常大得多，这说明了容器的进程是如何在其自己的命名空间内隔离的。

在容器内，你还可以探索 Docker 如何使用控制组来限制资源使用。直接演示这一点稍微复杂一些，因为它涉及检查 `/sys/fs/cgroup/` 目录中的特定文件。当你运行容器时，Docker 会根据默认或指定的配置自动设置这些限制。

## 联合文件系统

要查看联合文件系统的实际运作，请在容器内创建一个新文件：

```
touch /my-container-file.txt
```

此文件被写入容器文件系统的可写层。当你退出容器并使用相同的镜像启动一个新容器时，你会看到 `/my-container-file.txt` 在新的容器实例中不存在，这展示了联合文件系统如何将镜像与容器特定的可写层分离。

## 联合文件系统的结构

联合文件系统

联合文件系统是一种联合文件系统，允许多个文件系统相互叠加。在容器的上下文中，这通常涉及：

**只读层：** 这些层包含容器镜像本身，其中包括应用程序及其依赖项。当你从容器注册表拉取镜像时，你实际上是在下载这些只读层。这些层在所有从同一镜像实例化的容器之间共享，确保基础文件系统保持不变且不可变。这种不可变性对于一致性至关重要，确保每个容器实例都从完全相同的状态启动。

**可写层：** 在只读层之上，当容器启动时会添加一个可写层。此层捕获容器运行期间所做的任何更改。例如，如果应用程序写入文件，更改会记录在此可写层中。这允许每个容器实例拥有自己独特的更改集，与底层镜像和任何其他容器分开。

## 只读层的目的

基础层的只读特性有几个重要目的：

**不可变性：** 通过保持基础镜像层只读，确保原始内容无法被更改，为容器提供一致且可靠的基础。这种不可变性对于跨不同环境和部署的可预测行为至关重要。

**效率：** 只读层可以在多个容器实例之间共享。这意味着如果你有多个基于相同镜像的容器，它们都在主机系统上共享相同的底层层，从而减少存储和内存使用。

**安全性：** 只读层通过防止对容器基础文件系统的未授权更改来增强安全性。任何修改基础层的尝试都会导致写时复制操作，其中更改仅应用于活动容器的可写层。这种隔离有助于将潜在的安全漏洞限制在单个容器的范围内。

## 清理

输入 `exit` 退出容器。由于我们使用了 `--rm` 标志，Docker 会自动清理并删除容器，包括其可写层。

这个实际操作示例展示了 Docker 如何利用 Linux 内核特性，如用于进程隔离的命名空间、用于资源管理的控制组以及用于高效存储管理的联合文件系统。通过抽象这些复杂性，Docker 为强大的底层容器技术提供了用户友好的接口。

## 容器网络

容器网络是容器编排的一个关键方面，它使容器能够相互通信以及与外部世界通信。这种通信通过各种网络模式和概念来实现，旨在确保容器、服务和应用程序之间安全、高效且可扩展的交互。

最常见的容器网络模型包括：

**桥接网络**：容器的默认网络模式。在此模式下，在主机上创建一个虚拟桥，容器通过虚拟网络接口连接到此桥。此设置允许同一主机上的容器相互通信。

## 网络模式

容器可以使用不同的网络模式相互通信以及与主机通信，而主机可以使用NAT（网络地址转换）来管理外部访问。

**主机网络：** 容器共享主机的网络命名空间。这意味着容器的网络栈与主机的网络栈没有隔离，允许容器使用主机的IP地址和端口空间。这种模式适用于需要高性能并直接在主机网络上访问的服务。

**覆盖网络：** 覆盖网络跨越多个主机，允许不同主机上的容器像在同一网络上一样通信。这在运行容器的机器集群中特别有用，例如在Kubernetes或Docker Swarm环境中，它抽象了主机间网络的复杂性。

## 示例：桥接网络

让我们考虑一个简单的例子，涉及两个需要使用桥接网络相互通信的Docker容器。假设你正在开发一个微服务架构，其中一个容器运行Python Flask应用程序（一个Web API），另一个容器运行Redis数据库用于缓存。

### 创建桥接网络

首先，创建一个自定义桥接网络。与默认桥接网络相比，这提供了更好的隔离和网络控制。

```
$ docker network create --driver bridge my_bridge_network
```

### 运行Redis容器

启动一个Redis容器并将其连接到 **my_bridge_network**。

```
$ docker run --name my-redis-container --network my_bridge_network -d redis
```

### 运行Flask容器

启动一个Flask容器，同样连接到 **my_bridge_network**。该容器将与Redis容器通信。

```
$ docker run --name my-flask-app --network my_bridge_network -d -e REDIS_HOST=my-redis-container my-flask-image
```

在此设置中，Flask应用程序可以使用主机名 `my-redis-container` 访问Redis容器，该主机名解析为Redis容器在 **my_bridge_network** 上分配的内部IP地址。这是可能的，因为Docker的内部DNS解析器确保容器可以在同一自定义网络中通过名称找到彼此。

此示例演示了桥接网络如何促进同一主机上容器之间的轻松通信。每个容器可以根据需要安全地与网络上的其他容器通信，而无需在外部暴露不必要的访问。使用自定义桥接网络还允许在不同容器组之间进行网络级隔离，从而增强微服务架构中容器化组件的安全性和组织性。

## 进程隔离

在上面的例子中，为什么我不能在同一个容器中运行redis和flask？

在同一个容器中运行Redis和Flask应用程序在技术上是可能的，但这违背了容器化哲学，该哲学提倡“每个容器一个进程”的原则。这一原则植根于更广泛的微服务架构概念，旨在实现模块化、可扩展性和可维护性。以下是通常不鼓励在同一个容器中运行Redis和Flask的几个原因：

1.  **关注点隔离** 容器旨在封装单一职责或服务。这种隔离简化了开发、调试、扩展和维护。通过将Redis和Flask一起运行，你模糊了职责界限，使得更难独立管理每个进程的生命周期（例如，更新、重启和监控）。
2.  **可扩展性** 在微服务架构中，不同的组件可能需要根据需求独立扩展。例如，你可能需要更多的Flask实例来处理Web请求，但不需要那么多Redis实例，因为Redis可以处理大量连接。将两者结合在单个容器中会使扩展复杂化，并可能导致资源使用效率低下。
3.  **可重用性** 容器旨在成为可在不同环境和场景中部署的可重用组件。通过将Redis和Flask结合，你失去了在只需要其中一个服务的上下文中重用该容器的能力。它还使分布式系统中的部署过程复杂化，因为不同的服务可能由不同的团队管理。
4.  **安全性** 在单个容器中运行多个服务可能带来安全风险。最小权限原则建议一个进程应仅具有执行其功能所需的权限。如果一个服务被攻破，同一容器中的另一个服务可能为攻击者提供不必要的访问权限或可利用的资源。将服务分离到不同的容器中有助于限制安全漏洞的影响。
5.  **生命周期管理** 容器旨在表示一个单一的、无状态的服务，可以轻松启动、停止或替换。这与无状态应用程序（如Flask）非常契合，但与有状态服务（如Redis）不太契合，后者可能需要持久存储和更细致的生命周期管理。在单个容器中结合有状态和无状态服务会使备份、恢复和数据持久化策略复杂化。

## 示例场景

想象一下需要在不影响Redis的情况下更新Flask应用程序。如果两者都在同一个容器中运行，你将不得不重建并重启整个容器，导致Redis服务不必要的停机。将它们分离到不同的容器中允许独立更新，减少停机时间并降低对Redis服务可用性的潜在影响。

虽然容器技术提供了将多个进程一起运行的灵活性，但坚持每个容器一个进程的原则有助于提高应用程序的可管理性、安全性、可扩展性和模块化。这种方法符合云原生开发和微服务架构的最佳实践，确保应用程序具有弹性、可维护性和可扩展性。

## 构建我们的应用容器

我们现在将构建一个简单的“Hello world”容器，该容器运行一个Flask应用程序。在本教程中，我将故意保持Python应用程序部分简单。

```python
from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')
```

让我们看看用于构建应用程序的Dockerfile。

```dockerfile
FROM python:3.12 # (1)

WORKDIR /code # (2)
COPY requirements.txt requirements.txt # (3)
RUN pip install -r requirements.txt # (4)
EXPOSE 5000 # (5)
COPY . . # (6)
CMD ["flask", "run"] # (7)
```

(1) -> 此行指定构建新镜像所基于的基础镜像。这里，它使用了标记为版本3.12的官方Python Docker镜像。此基础镜像包含Python 3.12和运行Python应用程序所需的其他基本工具。这通常是Dockerfile的第一行，并为后续指令奠定了基础。基础镜像是构建新镜像的起点。

(2) -> 设置Docker容器内的工作目录。未来的命令将从此目录运行。它本质上将当前目录更改为容器文件系统内的 `/code`。如果该目录不存在，它将被创建。此目录的名称是任意的，但按照惯例，它通常是 **/app** 或 **/code**。

(3) -> 将 `requirements.txt` 文件从主机复制到容器中。此文件通常包含应用程序所依赖的Python包列表。通过将其放置在Dockerfile中的此位置，它允许后续的RUN指令在构建过程的早期安装这些依赖项。

(4) -> 使用pip（Python的包安装器）安装 `requirements.txt` 中定义的Python依赖项。这确保了应用程序运行所需的所有必要库在容器中都可用。

(5) -> 通知Docker容器在运行时监听端口5000。虽然这实际上并不发布端口，但它作为文档，并且可以被运行容器的人用来知道在运行时使用 **-p** 标志映射哪个端口。

(6) -> 将其余的应用程序代码（不包括在 `.dockerignore` 中指定的文件和目录）从主机复制到容器内的当前工作目录（`/code`）。这确保了应用程序的源代码存在于镜像中。

这是我们的 **.dockerignore** 文件的样子：

```
venv
__pycache__
```

它指示docker在从主机复制文件到容器时忽略 **__pycache__** 和虚拟环境文件。

此命令在安装依赖项后执行，以确保应用程序代码的更改无需重新安装所有依赖项，从而利用 Docker 的缓存机制。与应用程序代码相比，依赖项更改的可能性较小，因此稍后复制应用程序代码可以通过避免不必要的缓存失效来优化构建过程。

(7) -> 指定容器启动时运行的默认命令。在本例中，它启动一个 Flask 开发服务器。**CMD** 指令允许在运行时覆盖命令，这为容器的执行方式提供了灵活性。

这通常是 Dockerfile 中的最后一条指令，因为它定义了容器的运行时行为——基于镜像启动容器时实际应该发生什么。将其放在末尾可确保在应用程序开始运行之前完成所有设置和配置。

现在让我们构建我们的镜像。

```
$ docker build -t flask-app .
```

这将拉取基础镜像并运行 **Dockerfile** 中指定的指令。包含我们代码的最终镜像将存储在您的笔记本电脑中，并且仅从那里引用。

一旦我们成功构建了镜像，让我们运行它。

```
$ docker run --publish 5000:5000 flask-app
```

**--publish 5000:5000** 部分告诉 docker 将容器内的 5000 端口暴露给本地主机的 5000 端口。此语法为 **--publish <host-port>:<container-port>**。

让我们在 [http://localhost:5000/](http://localhost:5000/) 上查看应用程序。

如果您在 Mac 上运行此命令，docker run 命令很可能会失败，并显示类似以下的错误消息：

```
$ docker run --publish 5000:5000 flask-app
docker: Error response from daemon: Ports are not available: exposing port TCP 0.0.0.0:5000 -> 0.0.0.0:0: listen tcp 0.0.0.0:5000: bind: address already in use.
```

此错误表明我们主机（笔记本电脑）上的端口 5000 已绑定到其他应用程序。[这篇文章](https://example.com) 提到了如何解决此问题。一旦我们解决了这个问题，我们应该能够成功运行应用程序。

```
$ docker run --publish 5000:5000 flask-app
* Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
* Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

让我们再次在 [http://localhost:5000/](http://localhost:5000/) 上查看应用程序。

![](img/91c4bde832e221ba5bc429caf9587286_46_0.png)

## 运行失败

刚才发生了什么？我的端口现在是空闲的，但我仍然无法访问应用程序。这是因为 Flask 在**容器内部**的 127.0.0.1 上运行它。当我在主机的浏览器中访问 localhost 时，它被解析为主机的 127.0.0.1。

默认的 flask run 命令启动 Flask 开发服务器，并且仅在回环接口（127.0.0.1 或 localhost）上监听连接。出于安全原因，此行为是有意为之的，因为它将服务器的访问限制为仅来自运行 Flask 应用程序的同一台机器的请求。当在 Docker 容器内运行 Flask 时，该容器被视为具有自己网络接口集的独立机器。因此，当 Flask 仅在容器内的 127.0.0.1 上监听时，它无法接收来自主机或任何其他机器的请求，因为这些请求并非来自容器的 localhost。

要允许 Flask 应用程序接受来自主机 localhost（或其他机器）的请求，您需要告诉 Flask 在所有网络接口上监听。这可以通过在启动 Flask 服务器时将 –host 选项设置为 0.0.0.0 来实现：

```
$ flask run --host=0.0.0.0
```

通过绑定到 0.0.0.0，您是在指示 Flask 开发服务器在 Docker 容器可用的所有网络接口上接受请求，这包括 Docker 主机。这样，当您使用 Docker 的端口映射功能将 Docker 容器的端口映射到主机上的端口时，发送到主机 IP 地址上映射端口的请求将被成功转发到容器内的 Flask 应用程序。

让我们进行此更改，并使端口可配置。

我们将添加一个新文件 **entrypoint.sh**。

```
#!/bin/sh

exec flask run -h 0.0.0.0 -p $PORT
```

这是一个 shell 脚本，用于在监听所有网络接口的同时运行 flask，并在 **$PORT** 环境变量指定的端口上运行它。

为什么不运行类似这样的命令？

```
CMD ["flask", "run", "-h", "0.0.0.0", "-p", "$PORT"]
```

这在启动容器时不会评估 **$PORT** 变量。此外，我们在 shell 脚本中使用 **exec** 运行它，以便我们用于终止容器的 **Ctrl+C** 信号能被正确解释。

我们还对 Dockerfile 进行了以下更改：

```
COPY entrypoint.sh /entrypoint.sh # (1)
RUN chmod +x /entrypoint.sh

COPY . . # (2)

ENTRYPOINT ["/entrypoint.sh"] # (3)
```

(1) -> 将 entrypoint.sh 文件复制到容器内部。（否则我们无法运行它 :)）。同时，赋予其执行权限。(2) -> 注意 **COPY** 命令已移至此步骤下方。这是为了确保我们在构建容器时利用层缓存功能。(3) -> 最后，我们运行 **entrypoint.sh** 脚本。

## ENTRYPOINT 和 CMD 的区别

**ENTRYPOINT** 和 **CMD** 都是 Dockerfile 指令，用于指定 Docker 容器启动时将执行的默认命令。然而，它们的用途不同，使用方式也略有差异。

**ENTRYPOINT** 设置容器启动时将运行的可执行文件。它旨在设置容器的主命令，允许该容器像该命令或应用程序一样运行。

如果使用了 **ENTRYPOINT**，任何传递给 **docker run <image>** 的命令和参数都将作为参数附加到 **ENTRYPOINT**。

**示例：** 考虑一个设计用于运行 Python 脚本作为其主要目的的 Dockerfile：

```
FROM python:3.10
COPY . /app
WORKDIR /app
ENTRYPOINT ["python", "myscript.py"]
```

在这种情况下，运行 **docker run myimage** 将在容器中执行 **python myscript.py**。如果您在 **myimage** 后添加参数，它们将被传递给 **myscript.py**。

**CMD** 指定仅在未向 **docker run <image>** 提供参数时执行的默认命令和/或参数。如果您在 **docker run <image>** 中的镜像名称后指定命令，该命令将覆盖 **CMD** 指令。

**示例：** 使用 **CMD** 为脚本提供默认参数，或提供默认命令：

```
FROM ubuntu
CMD ["echo", "Hello, Docker!"]
```

运行 **docker run myimage** 将打印 "**Hello, Docker!**"，因为未指定命令，所以执行 **CMD**。但是，如果您运行 **docker run myimage echo "Another command"**，**CMD** 将被覆盖，转而打印 "**Another command**"。

## 结合使用 ENTRYPOINT 和 CMD

它们可以一起使用，以提供一个充当可执行文件并允许覆盖默认参数的容器。**ENTRYPOINT** 定义可执行文件，**CMD** 提供默认参数。

```
FROM ubuntu

ENTRYPOINT ["echo"]

CMD ["Hello, Docker!"]
```

运行不带任何额外参数的 **docker run myimage** 将打印 "**Hello, Docker!**"，因为 **CMD** 为 **ENTRYPOINT** 提供了默认参数。

运行 **docker run myimage "Another message"** 将覆盖 **CMD**，转而打印 "**Another message**"。

这种组合既提供了灵活性，又可以控制容器默认执行的内容以及如何在运行时进行自定义。

现在，有了这些新的更改，让我们重新构建容器。

```
$ docker build -t flask-app .
```

并运行它。

```
$ docker run -e PORT=8080 --publish 8080:8080 flask-app
* Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server
```

相反。

* 在所有地址上运行 (0.0.0.0)
* 在 http://127.0.0.1:8080 上运行
* 在 http://172.17.0.2:8080 上运行

按 CTRL+C 退出

让我们在浏览器中访问 [http://localhost:8080/](http://localhost:8080/)。

![](img/91c4bde832e221ba5bc429caf9587286_51_0.png)

*在端口 8080 上运行的应用*

### 练习 1：配置 gunicorn

生产环境中的 Flask 应用运行 gunicorn。请配置你的入口点以运行 gunicorn。

## 其他容器引擎

术语 docker 和 container 经常可以互换使用。这是因为 Docker 是最流行的容器运行时。你的笔记本电脑上可以使用多种容器引擎，每种都有其自身的特性和生态系统。以下是一些最受欢迎的容器引擎的总结：

## Docker

Docker 是使用最广泛的容器引擎，以其易用性、丰富的文档和强大的社区支持而闻名。它为开发者和系统管理员提供了一个使用容器开发、部署和运行应用程序的平台。

## containerd

containerd 最初是 Docker 的一部分，现在是云原生计算基金会（CNCF）下的一个独立容器运行时。它专注于简洁性、健壮性和可移植性。Docker、Kubernetes 和其他平台都使用 containerd 作为容器运行时。

## CRI-O

CRI-O 是一个专为 Kubernetes 设计的轻量级容器运行时。它实现了 Kubernetes 容器运行时接口（CRI），允许 Kubernetes 使用任何符合 OCI 标准的运行时作为运行 Pod 的容器运行时。

## Podman

Podman 是一个无守护进程的容器引擎，用于在你的系统上开发、管理和运行 OCI 容器。它与 Docker 完全兼容，但不需要运行守护进程。它支持以非 root 用户身份运行容器。

## rkt（已弃用）

描述：rkt 是由 CoreOS 开发的容器引擎，旨在实现安全性、高效性和可组合性。然而，它已被官方弃用并归档，最后一次发布是在 2019 年。尽管它曾具有影响力，但鼓励用户和开发者迁移到其他容器引擎。

## LXD

LXD 是来自 Canonical（Ubuntu 的制造商）的容器和虚拟机管理器。它提供了更接近虚拟机的用户体验，同时兼具容器的轻量级和快速特性。它特别适用于需要完整系统容器的场景。

这些容器引擎各有其独特功能，各自更适合某些特定场景。Docker 仍然是通用场景下最受欢迎的选择，但像 Podman 这样的工具因其无 root 模式和无守护进程架构而日益流行。对于 Kubernetes 环境，containerd 和 CRI-O 是常用选择。选择合适的容器引擎取决于你的具体需求，例如与 Kubernetes 的集成、安全特性或对以非 root 用户身份运行容器的支持。

## docker compose

Docker Compose 是一个用于定义和运行多容器 Docker 应用程序的工具。使用 Compose，你可以通过一个 YAML 文件来配置应用的服务、网络和卷，然后只需一个命令，就能创建并启动该配置中指定的所有组件。以下是 Docker Compose 为何特别有用，有时甚至被认为是开发和测试环境必备工具的原因：

### 简化的多容器编排

现代应用程序通常由多个需要一起运行的组件或服务组成，例如 Web 服务器、数据库，可能还有缓存层。Docker Compose 将管理这些多个容器的过程简化为管理单个服务。如果没有 Compose，你需要手动启动每个容器、设置网络并确保它们正确连接，这既容易出错又耗时。

### 易于配置

Compose 允许你在单个 **docker-compose.yml** 文件中定义你的多容器应用。该文件指定了关于应用的所有信息——使用哪些镜像、暴露哪些端口、容器如何链接、挂载哪些卷等等。这种声明式方法使得理解应用架构以及与他人共享变得更加容易。

### 环境标准化和版本控制

使用 Docker Compose，整个环境配置可以与你的应用代码一起进行版本控制。这确保了任何检出应用的人都可以通过 `docker-compose up` 命令启动一个完全相同的环境。这种一致性通过标准化团队间的开发环境，消除了“在我机器上能运行”的问题。

### 提高开发效率

Docker Compose 通过自动化容器生命周期管理任务来简化开发流程。开发者可以使用简单的命令轻松地启动、停止、重建和扩展服务。这意味着花在设置上的时间更少，而花在实际开发上的时间更多。此外，Compose 支持开发环境的实时重载，允许开发者无需重建容器即可实时查看代码更改。

### 易于测试和 CI/CD 集成

Compose 使得搭建和拆除整个环境变得非常简单，这对于自动化测试和持续集成/持续部署（CI/CD）流水线特别有用。你可以为开发、测试和生产定义不同的配置，确保应用程序在与生产环境高度相似的环境中进行测试。

### 本地开发和测试

虽然 Kubernetes 和其他编排工具在管理生产环境中的容器方面功能强大，但它们设置复杂，对于本地开发和测试来说可能过于复杂。Docker Compose 填补了这一空白，提供了一个更简单但功能强大的工具，用于定义和运行具有多个相互依赖容器的复杂应用程序。

### 示例用例

考虑一个使用 Flask 框架开发的 Web 应用程序，使用 Redis 进行缓存，PostgreSQL 作为数据库。使用 Docker Compose，你可以在一个 **docker-compose.yml** 文件中定义并链接所有三个组件。然后，开发者可以使用 `docker-compose up` 运行整个技术栈，而无需手动启动每个组件或设置容器间的网络链接。

Docker Compose 通过提供一种简化的、声明式的配置和编排方法，解决了管理多容器应用程序的复杂性。它提高了开发效率，确保了环境一致性，并简化了与 CI/CD 流水线的集成，使其成为现代开发工作流中不可或缺的工具。

让我们构建一个简单的 Flask 应用，它与 Redis 服务通信，并使用 *docker-compose* 来管理它。

```python
import time
import os

from flask import Flask
from flask import render_template

import redis # (1)

redis_host = os.getenv('REDIS_HOST') # (2)
redis_password = os.getenv('REDIS_PASSWORD') # (3)

app = Flask(__name__)

cache = redis.Redis(host=redis_host, password=redis_password, port=6379) # (4)

def get_hit_count(): # (5)
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

@app.route("/")
def hello():
    count = get_hit_count() # (6)
    return render_template('index.html', count=count) # (7)
```

(1) -> 导入 Python 的 Redis 客户端库，使其能够与 Redis 服务器交互。
(2) -> 从环境变量中获取 Redis 服务器的主机地址。
(3) -> 从环境变量中获取 Redis 服务器的密码。
(4) -> 使用从环境变量获取的主机和密码创建一个 Redis 连接对象。
(5) -> 定义一个函数，用于检索并增加存储在 Redis 中的访问计数。
(6) -> 调用 **get_hit_count** 函数以获取当前的访问次数。
(7) -> 渲染一个 HTML 模板，并将访问计数传递给它以在网页上显示。

让我们构建 Flask 容器镜像。

```bash
$ docker build -t flask-redis:1.0 .
```

最后，是 **docker-compose.yml** 文件。

这个 Docker Compose 文件定义了两个服务：flask-app 和 cache。其中 flask-app 是一个 Flask 应用，配置为连接到由 cache 服务定义的 Redis 实例。Flask 应用的端口根据 PORT 环境变量动态映射，并且两个服务共享一个用于 Redis 密码的密钥以确保通信安全。

```yaml
version: '3' # (1)
services:
  flask-app:
    image: flask-redis:1.0 # (2)
    ports:
      - "8080:${PORT}" # (3)
```

environment:
  - PORT=8080
  - REDIS_HOST=cache # (4)
  - REDIS_PASSWORD=SuperS3cret # (5)
depends_on: # (6)
  - cache

cache:
  image: "bitnami/redis:7.2.4" # (7)
  environment:
    - REDIS_PASSWORD=SuperS3cret # (8)
  ports: # (9)
    - "6379:6379"

(1) -> 指定 Docker Compose 文件格式的版本。(2) -> 指定 'flask-app' 使用的 Docker 镜像。这是我们上一步构建的镜像。(3) -> 映射容器与主机之间的端口。具体来说，将主机的 8080 端口映射到容器的环境变量 ${PORT}。(4) -> 将容器的 REDIS_HOST 环境变量设置为 'cache'，即 Redis 容器的服务名称。(5) -> 设置容器的 REDIS_PASSWORD 环境变量。(6) -> 指定 'flask-app' 依赖于 'cache' 服务。这确保了 'cache' 服务在 'flask-app' 之前启动。(7) -> 指定 Redis 服务使用的 Docker 镜像。我们指定 Bitnami 的 Redis 镜像，因为它被广泛使用。Bitnami 镜像还允许用户通过环境变量配置 Redis 服务的认证。(8) -> 设置容器的 REDIS_PASSWORD 环境变量。(9) -> 将主机的 6379 端口映射到容器内的 6379 端口，这是 Redis 的默认端口。

现在，我们启动这两个服务。

```
$ export PORT=8080
$ docker-compose up
```

你的输出将与此类似。

![](img/91c4bde832e221ba5bc429caf9587286_60_0.png)

Flask 应用将可通过 [http://localhost:8080/](http://localhost:8080/) 访问。

![](img/91c4bde832e221ba5bc429caf9587286_61_0.png)

由 Redis 缓存支持的 docker-compose flask 应用

### 练习 2：设置实时重载

修改 docker-compose 配置，使其在开发期间支持实时重载。这确保了主机上源代码的更改无需重启容器即可在容器内生效。

### 练习 3：配置 Nginx

生产环境的 Flask 应用不使用 `flask run` 来提供应用服务。它们要么使用 gunicorn，要么使用像 Nginx 这样的 Web 服务器进行代理。不要使用默认的 Flask run 命令来提供服务，而是使用一个 Nginx 容器进行代理。对 docker-compose 文件进行必要的更改。

## 注册表

![](img/91c4bde832e221ba5bc429caf9587286_62_0.png)

一个容器注册表

容器注册表是一个存储和内容分发系统，用于存放容器镜像。这些镜像包含可在容器化环境中运行的软件的可执行包，其中包含运行所需的所有元素，例如代码、运行时、系统工具、系统库和设置。注册表是容器生态系统中的关键组件，作为存储、组织和管理镜像的仓库。以下是对容器注册表及其重要性的更深入了解：

## 为什么我们需要容器注册表？

**集中存储：** 容器注册表提供了一个集中存储和管理容器镜像的位置。这种集中化便于开发团队和部署环境之间更轻松地访问和分发镜像，确保部署的一致性和可靠性。

**版本控制和镜像管理：** 注册表跟踪容器镜像的不同版本，允许团队部署特定版本的应用程序或服务。此功能对于维护应用程序稳定性、在必要时回滚到以前的版本以及实施蓝/绿或金丝雀部署策略至关重要。

**安全性和合规性：** 许多容器注册表提供安全功能，例如漏洞扫描，它会检查镜像是否存在已知的安全问题。这有助于维护应用程序的安全态势。注册表还可以强制执行合规策略，确保只有经过批准和审查的镜像才能在生产环境中使用。

**与 CI/CD 管道集成：** 容器注册表与持续集成/持续部署 (CI/CD) 管道无缝集成。这种集成自动化了构建、测试和部署应用程序的过程，显著加快了开发周期并减少了人为错误的可能性。

**访问控制：** 注册表提供控制谁可以推送和拉取镜像的机制，使组织能够实施访问策略来保护敏感的应用程序和数据。这种访问控制对于维护容器化应用程序的完整性和安全性至关重要。

**分发效率：** 容器注册表通常具有高效分发镜像的机制，例如层缓存和镜像压缩。这些功能减少了带宽使用并加快了镜像的下载速度，这在分布式或云原生环境中尤其有益，因为应用程序可能部署在多个区域或云提供商中。

## 示例注册表

**Docker Hub：** 最大的公共容器镜像注册表，托管数百万个容器镜像。它被广泛用于公共和私有仓库，为各种软件和服务提供大量社区生成的镜像。

**Google Container Registry (GCR)：** 一个私有容器注册表，提供对容器镜像的高速访问，集成在 Google Cloud Platform 中，确保 Docker 镜像的安全高效存储和管理。

**Amazon Elastic Container Registry (ECR)：** AWS 提供的完全托管的 Docker 容器注册表，允许开发人员轻松存储、管理和部署 Docker 容器镜像。

**Azure Container Registry (ACR)：** Microsoft Azure 提供的服务，提供私有 Docker 注册表功能，并与 Azure 服务集成，支持 Windows 和 Linux 容器。

总之，容器注册表通过提供可靠、安全和高效的方式来存储、管理和分发容器镜像，在容器化生态系统中发挥着关键作用。它们支持部署过程的自动化，通过镜像扫描和访问控制增强安全性，并促进开发和运维团队之间的协作。

## 设置容器注册表

Github 自带一个免费的容器注册表。假设你已经拥有一个 Github 账户并已注册，请从其网站创建一个新令牌 (https://github.com/settings/tokens/new?scopes=write:packages)。复制并保存此令牌以登录 Github 注册表。

以下是使用你刚获得的令牌登录 Github 注册表的命令。

```
$ export CR_PAT=ghp_XXXX # 你刚才获得的令牌
$ echo $CR_PAT | docker login ghcr.io -u <your-github-username> --password-stdin # 将 "<your-github-username>" 替换为你的 github 用户名。
```

登录后，你可以重新标记你的 Flask 容器镜像并将其推送到 ghcr。

```
$ docker tag flask-app ghcr.io/<your-github-username>/flask-app:1.0 # 将 "<your-github-username>" 替换为你的 github 用户名。
$ docker push ghcr.io/<your-github-username>/flask-app:1.0 # 将 "<your-github-username>" 替换为你的 github 用户名。
```

### 练习 4：在 Ghcr 中构建并推送新标签。

对你的代码进行任意更改，构建它并将新标签推送到 Github 注册表。

接下来，在 **requirements.txt** 中添加一个新的依赖项，重新构建镜像并推送一个新标签。比较两种情况下的构建步骤输出，并记录你的观察结果。

## 总结

本章为理解容器技术的关键组件奠定了基础，这对于掌握 Kubernetes 至关重要。它从容器如何工作的基础知识开始，利用 Linux 内核特性（如用于隔离的命名空间和用于资源管理的 cgroups）来在单个主机系统上高效地运行多个隔离环境。这种理解对于你掌握容器化在实现跨各种环境的一致性和简化应用程序部署方面的意义至关重要。

我们进一步深入探讨了容器网络、覆盖文件系统以及在容器内构建和运行应用程序的实际方面。它说明了不同的网络模式，例如桥接和覆盖网络，这些对于配置容器间通信和将容器连接到外部世界至关重要。此外，它还引导读者完成将 Flask 应用程序容器化的过程，重点介绍了创建 Docker 镜像和运行容器的过程，从而提供了一种实践方法来理解容器操作的细微差别。

最后，Docker Compose 作为在本地开发环境中管理多容器应用程序的工具被引入，为 Kubernetes 的编排能力奠定了基础。现在，你已经具备了容器技术的基础知识，这为探索 Kubernetes 的高级编排功能铺平了道路，这些功能对于在生产环境中部署、扩展和管理容器化应用程序至关重要。

## 第 3 章：Kubernetes 入门

虽然 Docker Compose 非常适合开发和小规模生产部署，但 Kubernetes 旨在处理具有复杂管理和编排需求的大规模、生产级部署。以下是为什么对于相同的 Flask 和 Redis 示例，尤其是在扩展时，Kubernetes 是必要的：

**编排与调度：** Kubernetes 可以根据容器的资源需求和其他约束自动决定在哪里运行它们，从而提高整个基础设施的利用率和效率。

**高可用性与故障转移：** Kubernetes 确保你的应用程序始终可用，并能自动替换失败的容器，这是生产环境的关键特性，而 Docker Compose 本身并不提供此功能。

**可扩展性：** Kubernetes 擅长根据需求通过水平 Pod 自动伸缩来扩展应用程序。与 Docker Compose 相比，它允许更复杂的伸缩策略，这对于处理可变工作负载至关重要。

**服务发现与负载均衡：** Kubernetes 提供内置的服务发现和负载均衡功能，自动将流量分配到你的容器，以确保操作顺畅高效。随着应用程序复杂性和流量的增长，这一点尤为重要。

**配置与密钥管理：** Kubernetes 提供了强大的机制来管理配置和密钥，允许你在不重建容器镜像的情况下更新应用程序配置和敏感信息。

**自愈能力：** Kubernetes 持续监控 Pod 和服务的状态，自动重启失败的容器，在节点故障时替换和重新调度容器，终止未通过用户定义健康检查的容器，并在它们准备好服务之前不向客户端通告。

本质上，虽然 Docker Compose 为在开发和单机生产环境中定义和运行多容器应用程序提供了一个很好的起点，但 Kubernetes 提供了在大规模、分布式和动态生产环境中管理复杂应用程序所需的可扩展性、可靠性和灵活性。当你的应用程序增长时，从 Docker Compose 过渡到 Kubernetes 可确保你能够充分利用容器化部署的潜力。

## Kubernetes 的工作原理

简单来说，Kubernetes 是一个分布式系统，旨在跨一台或多台虚拟机（VM）或物理机编排容器。编排涉及什么？

在容器化应用程序和云计算的背景下，编排是指对计算机系统、应用程序和服务的自动化管理、协调和安排。它涉及一系列广泛的操作：在集群内的多个主机上部署、管理、扩展、联网和连接容器。编排的目标是简化和优化这些容器的部署和运行，确保它们作为更大分布式应用程序或系统的一部分和谐地工作。

以下是编排所涵盖的关键方面：

### 自动化部署

编排工具根据预定义的配置自动处理容器的部署。这包括决定哪些主机（虚拟或物理机）应该运行容器，同时考虑资源可用性和约束等因素。

### 扩展

编排通过根据需求自动调整运行的容器实例数量，实现应用程序的动态扩展。这可以包括扩展（添加更多容器）以处理增加的负载，或缩减（移除容器）以在较安静的时期节省资源。

### 健康监控与自愈

编排系统持续监控容器的健康状况，并自动用新实例替换不健康的容器，确保应用程序无需人工干预即可保持可用和可靠。

### 网络

编排工具管理容器之间的网络，使它们能够相互通信以及与外部网络通信。这包括分配 IP 地址、配置网络策略以及设置负载均衡器来分发流量。

### 服务发现

在容器频繁创建和销毁的动态环境中，编排促进了服务发现，允许容器使用逻辑名称而不是依赖静态 IP 地址来查找和相互通信。

### 配置管理

编排涉及在运行时管理和向容器注入配置。这可以包括环境变量、配置文件和密钥（如密码和 API 密钥），确保容器配置正确，而无需硬编码敏感信息。

### 资源分配

编排工具根据容器的需求和可用容量为其分配资源（CPU、内存、存储），优化整个集群的资源利用率。

### 更新与回滚

编排支持应用程序的滚动更新，即逐步推出新版本的容器而无需停机。如果更新失败或引入问题，它还支持回滚到以前的版本。

本质上，编排自动化了大规模部署和管理容器化应用程序所涉及的复杂手动过程。它将一个潜在混乱的环境——容器不断启动、停止和移动——转变为一个协调良好的系统。

一个 Kubernetes 集群至少包含 1 个控制平面节点和工作平面节点。

控制平面节点负责管理集群的状态。它调度应用程序，管理部署和扩展，并维护应用程序的期望状态。控制平面的组件包括 kube-apiserver、etcd 存储、kube-scheduler、kube-controller-manager 和 cloud-controller-manager。

工作节点运行实际的应用程序和工作负载。每个工作节点都有一个 Kubelet，它是管理节点状态并与 Kubernetes 主节点通信的代理。工作节点还运行路由流量和允许容器相互通信所需的服务，例如 kube-proxy 和容器运行时。

## Kubernetes 集群的组件

![Kubernetes 架构](img/91c4bde832e221ba5bc429caf9587286_73_0.png)

## kube-apiserver

kube-apiserver 充当 Kubernetes 控制平面的前端。它是 Kubernetes API 暴露的组件，作为用户、外部组件和集群内部部分之间通信和管理 Kubernetes 对象（Pod、服务、部署等）状态的主要接口。kube-apiserver 处理 REST 操作，验证它们，并更新 etcd 中的相应对象，使其成为通往集群操作状态的关键网关。

## etcd

etcd 是一个分布式键值存储，被 Kubernetes 用作所有集群数据的后备存储。它提供了一种可靠的方式来存储需要被像 Kubernetes 这样的分布式系统访问的数据。etcd 的主要工作是维护和管理 Kubernetes 集群的状态，跟踪所有 Kubernetes 对象。它被设计为高可用性，并且是集群的最终事实来源，确保集群状态数据的持久性、一致性和容错性。

## kube-scheduler

kube-scheduler 负责将 Pod 分配给节点。它观察没有分配节点的新 Pod 的创建，并根据多个调度标准为它们选择运行节点。这些标准包括单个和集体资源需求、硬件/软件/策略约束、亲和性和反亲和性规范、数据局部性、工作负载间干扰和截止日期。kube-scheduler 确保在可能的情况下，工作负载在可用的计算资源之间保持平衡。

## Kube Controller Manager

Kube Controller Manager 是运行在主节点上的组件，负责执行控制器进程。这些控制器包括节点控制器、副本控制器、端点控制器以及服务账户与令牌控制器等。每个控制器负责管理集群操作的特定方面，通过 apiserver 监视集群的共享状态，并做出变更以尝试将当前状态推向期望状态。

## Kubelet

Kubelet 运行在集群中的每个节点上，负责确保存储在 Pod 中的容器正常运行。它接收通过各种机制（主要通过 apiserver）提供的一组 PodSpec，并确保这些 PodSpec 中描述的容器处于运行且健康的状态。Kubelet 监视 Pod 和容器的状态，向控制平面报告，处理 Pod 和容器的生命周期事件，并管理容器运行时环境。

## Kube-proxy

Kube-proxy 是运行在集群每个节点上的网络代理，实现了 Kubernetes Service 概念的一部分。Kube-proxy 在节点上维护网络规则，这些规则允许来自集群内部或外部网络会话的流量与你的 Pod 进行通信。如果操作系统存在并可用，Kube-proxy 会使用操作系统的数据包过滤层；否则，kube-proxy 会自行转发流量。

## 容器运行时

容器运行时是负责运行容器的软件。Kubernetes 支持多种容器运行时：Docker、containerd、CRI-O 以及任何符合 Kubernetes CRI（容器运行时接口）的实现。容器运行时负责从仓库拉取容器镜像、解压容器并运行应用程序。Kubernetes 将管理容器的过程抽象到容器运行时，使其能够根据 Pod 规范管理容器的生命周期。

## 其他容器编排工具

在不断发展的容器编排领域，Kubernetes 无疑已成为领跑者，但它并非唯一的选择。其他各种容器编排工具也各具特色，各自拥有独特的功能和优势，以满足不同的用例和偏好。

**Docker Swarm** 是 Docker 原生的集群和调度工具。其主要吸引力在于其简洁性以及与 Docker 生态系统的紧密集成。Swarm 采用了与 Docker 类似的命令行界面，这降低了已经熟悉 Docker 命令的团队的学习门槛。它以简单的设置过程而闻名，其声明式模型允许你定义服务的期望状态。然而，它可能无法匹敌 Kubernetes 的高级功能和可扩展性，因此更适合较小规模的部署或刚开始容器化之旅的组织。

**Nomad** 是 HashiCorp 推出的另一款著名工具，提供工作负载编排功能。与专注于容器的 Kubernetes 不同，Nomad 与工作负载无关，这意味着它不仅可以调度容器，还可以调度非容器化应用程序。它与其他 HashiCorp 工具（如用于服务发现的 Consul 和用于密钥管理的 Vault）无缝集成，提供了一个全栈解决方案。其在调度多样化工作负载方面的简洁性和灵活性，使其成为多平台部署的一个引人注目的选择。

**Kamal** 是一个专门用于在各种平台（从云虚拟机到裸金属）上部署 Web 应用程序的部署工具，注重简洁性和可移植性。它提供了零停机部署和资产桥接等功能，并能够使用 Docker 处理生产环境中的复杂任务。Kamal 被描述为“容器的 Capistrano”，允许在无需预先准备服务器的情况下实现部署自动化。对于那些希望避免 Kubernetes 和 Docker Swarm 复杂性的人来说，它提供了一个更简单的替代方案，倾向于使用命令式方法以提高易用性和清晰度。

## 创建你的第一个 Kubernetes 集群

理论讲得够多了。让我们开始动手实践吧！现在我们将创建我们的第一个 Kubernetes 集群。互联网上有无数关于如何使用 minikube 集群（一个流行的、在笔记本电脑上运行的 Kubernetes 集群）的教程。对此我有一些个人看法。我们将不使用 Minikube。我们将创建一个真实的集群来学习和实践 Kubernetes。在你被这个想法吓到之前，请先听听我支持这种方法的理由。它可能并没有那么糟糕。

## 使用真实集群的理由

启动实际的 Kubernetes 集群，而不是仅仅依赖像 Minikube 这样的本地开发环境，能为开发者提供大量实践经验，这对于理解 Kubernetes 在真实场景中的细微差别至关重要。虽然 Minikube 是一个学习 Kubernetes 基础知识和在本地测试应用程序的绝佳工具，但它简化了 Kubernetes 管理和网络的许多方面。通过在更真实的环境中部署集群，开发者能够获得关于集群架构、网络、存储和安全复杂性的宝贵见解，这种方式更接近于模拟生产环境。

此外，管理真实的 Kubernetes 集群让开发者接触到扩展性、弹性和多租户方面的挑战。在本地集群上部署应用程序是一回事；而确保同一应用程序在面对真实用户流量和数据量的不可预测变量时，仍然保持高可用性、安全性和可扩展性，则完全是另一回事。这种实践经验对于理解如何设计和管理能够承受生产环境严苛考验的应用程序至关重要。

此外，使用实际的 Kubernetes 集群让开发者有机会熟悉云原生技术以及 AWS、Google Cloud 和 Azure Kubernetes Service (AKS) 等云平台提供的服务。每个云提供商都提供与 Kubernetes 集成的独特工具和服务，例如托管数据库、CI/CD 流水线和监控工具，这些都是全面的云原生应用程序栈的重要组成部分。学习如何使用这些服务并理解它们如何补充 Kubernetes 部署，对于希望充分利用云能力的开发者来说至关重要。

本质上，虽然像 Minikube 这样的工具在 Kubernetes 教育中扮演着关键角色，但过渡到管理实际的 Kubernetes 集群是开发者的关键一步。它弥合了理论知识与实践技能之间的差距，为开发者在云中部署和管理生产级应用程序的挑战做好了准备。这种实践经验不仅提高了技术熟练度，也增强了使用 Kubernetes 解决真实世界应用程序部署和管理挑战的信心。

我希望你在真实集群中部署真实应用程序后能获得成就感。任何低于此标准的体验，对于一本旨在让开发者亲自动手实践 Kubernetes 的书籍来说，都是不够的。

## 在 DigitalOcean 创建集群

我强烈建议你在 DigitalOcean 创建一个账户。这将为你提供 200 美元的信用额度，有效期为 60 天，这在本书的整个学习过程中绰绰有余。我将在下一节简要介绍如何在 DigitalOcean 创建集群。

## 设置你的 Kubernetes 集群

下载并设置 **doctl** 命令行工具，以便与你的 DigitalOcean 云账户进行交互。[此页面](https://cloud.digitalocean.com/account/api/tokens) 包含针对你的操作系统进行设置和账户认证的详细说明。

```bash
$ doctl auth init
Please authenticate doctl for use with your DigitalOcean account. You can generate a token in the control panel at https://cloud.digitalocean.com/account/api/tokens

> Enter your access token: ************************************

Validating token... ✓

$ doctl kubernetes options regions
Slug    Name
nyc1    New York 1
sgp1    Singapore 1
lon1    London 1
nyc3    New York 3
ams3    Amsterdam 3
fra1    Frankfurt 1
tor1    Toronto 1
sfo2    San Francisco 2
blr1    Bangalore 1
sfo3    San Francisco 3
syd1    Sydney 1
$ 
```

*使用 DigitalOcean 令牌认证 doctl 并列出区域*

一旦成功认证，你将获得类似以下的输出：

```
$ doctl account get
Email              Droplet Limit  Email Verified  UUID                                    Status
sammy@example.org  10             true            3a56c5e109736b50e823eaebca85708ca0e5087c  active
```

接下来，我们可以继续创建集群。

```
$ doctl kubernetes cluster create first-cluster --region blr1 --size s-2vcpu-4gb --count 3
```

**doctl kubernetes cluster create** 是集群创建子命令。**first-cluster** 是集群的名称。**blr1** 表示集群应在班加罗尔区域配置（注意，要查看其他区域选项，你可以运行 **doctl kubernetes options regions**，如上所示）。**s-2vcpu-4gb** 是集群的大小，包含 2 个 vCPU 和 4GB 内存。最后，**--count** 参数表示将创建多少个节点。

![](img/91c4bde832e221ba5bc429caf9587286_80_0.png)

*使用 doctl 创建 Kubernetes 集群*

为什么不使用 UI？许多产品的 UI 经常变化，因此本节内容可能会过时并让人困惑。如果我们使用 CLI，指令会更精确，我们清楚自己请求和获得的是什么，因此它能在更长时间内保持相关性。

## 总结

Kubernetes 通过提供一个弹性框架而区别于 Docker Compose，该框架不仅能随需求波动进行扩展，还能以最小的停机时间维持应用程序运行。我们将这个强大的基础设施与 Docker Compose 的简洁性进行了对比，为你欣赏 Kubernetes 在实际应用中的复杂性和必要性奠定了基础。本章还深入探讨了 Kubernetes 集群的运行组件，详细说明了控制平面、工作节点以及 kube-apiserver、etcd、kube-scheduler、Kube Controller Manager、Kubelet 和 Kube-proxy 等关键组件的作用。

在展望下一章时，我们将聚焦于 Kubernetes 最基本的概念之一：Pod。理解 Pod 至关重要，因为它们是 Kubernetes 部署的基本构建块。我们将深入探讨 Pod 的生命周期、它在 Kubernetes 生态系统中的交互方式，以及如何有效地管理和调试它们。准备好深入探索 Kubernetes 的微观层面机制，这些机制支撑着我们在本章探讨的宏观层面操作。

## 第 4 章：Pod 基础

到目前为止，我们已经构建了 Python 应用程序的容器镜像并在本地运行了它们。我们还搭建了一个生产级的 Kubernetes 集群。让我们将这两者结合起来，在这个新创建的集群中部署 Python 应用程序。

但在我们深入之前，必须先熟悉几个概念。

### Kubectl

第一个概念是一个名为 **kubectl** 的工具。有关如何为你的操作系统下载它，有直接的说明。

**kubectl** 是一个命令行工具，作为与 Kubernetes 交互的主要接口，允许用户部署应用程序、检查和管理集群资源以及查看日志。它与 Kubernetes API 服务器通信以控制 Kubernetes 集群，提供广泛的命令，使用户能够执行诸如创建和删除 Pod、扩展部署以及管理配置等操作。

**kubectl** 旨在用户友好，支持多种输出格式来显示集群信息，并包含用于调试和故障排除集群中运行应用程序的功能。通过提供强大而灵活的接口，**kubectl** 使开发人员和管理员能够高效地管理其应用程序和 Kubernetes 集群本身的生命周期。

### Kubeconfig

简单来说，Kubeconfig 就像一个钥匙链，保存着访问和与 Kubernetes 集群通信所需的钥匙和方向。想象一下，你有几把不同房子（集群）的钥匙，每栋房子都有自己的安全系统（认证方法）。kubeconfig 文件会跟踪所有这些钥匙和房子的地址，这样你就知道哪把钥匙打开哪栋房子以及如何到达那里。当你使用 kubectl 命令时，它会查看你的 kubeconfig 来确定你试图与哪个集群通信，以及使用哪个钥匙（凭据）进行该对话。它本质上是一个配置文件，使得管理和切换不同的 Kubernetes 集群及其访问细节变得轻而易举，确保你始终使用正确的访问权限与正确的集群通信。

假设你已经登录到你的 DigitalOcean 账户，你可以下载你创建的集群的 kubeconfig。

```
$ doctl kubernetes cluster kubeconfig save first-cluster
```

![](img/91c4bde832e221ba5bc429caf9587286_83_0.png)

*将集群的 kubeconfig 保存到本地*

**first-cluster** 是你之前创建的集群的名称。

你也可以通过运行以下命令将 kubeconfig 保存到文件：

```
$ doctl kubernetes cluster kubeconfig show first-cluster > ~/k8s-for-devs/kubeconfig.first-cluster
```

这会将 kubeconfig 保存到文件 `~/k8s-for-devs/kubeconfig.first-cluster`。

### Kubeconfig 的组成部分

kubeconfig 是一个 YAML 文件，看起来像这样：

```
apiVersion: v1
kind: Config
clusters:
- name: development-cluster
  cluster:
    server: https://development.example.com
    certificate-authority-data: ...
contexts:
- name: development
  context:
    cluster: development-cluster
    user: developer
    namespace: dev
users:
- name: developer
  user:
    client-certificate-data: ...
    client-key-data: ...
current-context: development
preferences: {}
```

以下是其关键组件的分解：

#### Clusters（集群）

此部分列出了你可以访问的所有 Kubernetes 集群。对于每个集群，它包含一个用于引用的名称和集群的端点，即 Kubernetes API 服务器的 URL。它通常包含额外信息，如用于验证 API 服务器 SSL 证书的证书颁发机构（CA）数据，以确保安全通信。

#### Contexts（上下文）

上下文结合了关于集群、用户和命名空间的信息，以简化在不同工作环境之间的切换。每个上下文都有一个名称，并包含：来自 clusters 部分的集群名称，指示你正在访问哪个集群；来自 users 部分的用户名，指定用于向集群进行身份验证的凭据；以及要设置为当前活动命名空间的命名空间，允许你专注于集群内的特定资源子集。

#### Users（用户）

此部分定义了用于向集群进行身份验证的凭据。每个用户条目包含一个名称和身份验证详细信息，例如客户端证书、客户端密钥、令牌或用户名和密码。这些凭据用于向集群的 API 服务器证明你的身份。

#### Current Context（当前上下文）

指定 kubectl 应用于命令的默认上下文（来自 contexts 部分）。这使得在不同的集群和用户组合之间切换变得容易，无需在每次运行命令时都指定它们。你可以使用 kubectl config use-context 命令更改当前上下文。

#### Preferences（偏好设置）

包含 kubectl 的全局配置选项。虽然不如其他部分常用，但它允许设置影响 kubectl 在所有上下文中行为方式的默认值，例如输出中的颜色。

### 为什么使用 TLS 认证？

Kubernetes API 服务器使用 TLS（传输层安全）认证来确保客户端与服务器之间以及集群内服务之间的安全通信。TLS 提供了几个关键的安全功能，使其比基本认证更适合大多数场景，尤其是在像 Kubernetes 这样的分布式系统中，需要保护敏感数据和系统完整性。以下是 TLS 优于基本认证的关键原因：

#### 加密

TLS 提供传输中数据的端到端加密。这意味着在 Kubernetes API 服务器和客户端之间（或服务之间）发送的所有数据都是加密的，防止窃听者截获和读取敏感信息。

## 身份验证

TLS 支持双向身份验证。客户端通过受信任的证书颁发机构（CA）签发的数字证书来验证服务器的身份。此外，服务器也可以选择要求客户端出示证书，以验证客户端的身份。这种双向身份验证确保了通信双方都是其声称的身份，从而降低了中间人攻击的风险。基本身份验证仅提供单向认证，客户端不会验证服务器的身份，因此更容易受到冒充攻击。

## 完整性

TLS 通过加密消息摘要确保传输数据的完整性。这意味着在传输过程中对数据的任何篡改都能被检测到，确保接收到的数据与发送的数据完全一致。基本身份验证不提供这种级别的完整性检查，使得传输的数据容易受到未授权的修改。

## 吊销与过期

TLS 中使用的证书可以被吊销或设置为在一定期限后过期，从而实现对访问和安全性的更好管理。如果证书被泄露，可以将其吊销以防止进一步滥用。证书过期会强制定期更新，确保身份验证机制得到定期更新。基本身份验证缺乏这种机制，使得凭证的安全管理随时间推移变得更加困难。

## 合规性与标准

许多法规和安全标准要求使用 TLS 来保护通信安全，因为它具有全面的安全特性。依赖未加密的基本身份验证不符合这些安全标准，并可能导致违反数据保护法规。

## 与集群交互

连接到集群的方法是下载 kubeconfig 文件，并将 **KUBECONFIG** 环境变量指向下载的 kubeconfig 文件的绝对路径。

```
$ export KUBECONFIG=~/k8s-for-devs/kubeconfig.first-cluster
```

之后，你就可以运行 **kubectl** 命令了。

## 关于 Kubernetes 版本的说明

在撰写本文时，Kubernetes 的最新稳定版本是 1.29。Kubernetes 版本采用 x.y.z 的格式表示，其中 x 是主版本号，y 是次版本号，z 是补丁版本号，遵循语义化版本控制术语。

Kubernetes 项目为最近三个次版本维护发布分支，例如在我们的案例中是 1.29、1.28 和 1.27。使用较新或较旧版本的 Kubernetes 并不强制要求升级或降级 kubectl。

## 与 Kubernetes 的首次交互

假设你已经设置了 **KUBECONFIG** 环境变量，运行一个命令来获取集群中的所有节点。

```
$ kubectl get nodes
NAME                       STATUS   ROLES    AGE   VERSION
first-cluster-default-pool-oeo85   Ready    <none>   6m32s   v1.29.1
first-cluster-default-pool-oeo8k   Ready    <none>   6m39s   v1.29.1
first-cluster-default-pool-oeo8s   Ready    <none>   6m42s   v1.29.1
```

**NAME** 列显示了 3 节点集群中每个节点的名称。**STATUS** 列显示节点处于 **Ready** 状态。我们暂时忽略 **ROLES** 列。**AGE** 列是节点创建以来经过的时间。最后一列 **VERSION** 显示了节点上运行的 Kubernetes 版本。

## Pod

Pod 是 Kubernetes 中最小的可部署单元，它封装了一个或多个共享存储、网络以及如何运行容器规范的容器。我们稍后会详细分解这句话。现在，你可以将 Pod 想象成一个包含 1 个或多个容器的集合，可以在 Kubernetes 集群中部署。

为什么 Pod 而不是单个容器被视为最小的可部署单元？这一设计决策基于几个关键考虑，这些考虑增强了平台以分布式和可扩展方式管理应用程序的能力。以下是 Pod 在 Kubernetes 架构中扮演如此关键角色的原因：

### 容器分组

Pod 允许将多个容器分组到一个可部署的实体中。这对于涉及需要协同工作、共享资源或作为单个单元进行管理的紧密相关容器的应用程序特别有用。例如，一个主应用程序容器可能与一个处理日志记录或监控的 sidecar 容器一起运行。通过将这些容器分组在单个 Pod 中，Kubernetes 可以将它们的部署、扩展和网络作为一个统一的实体来管理。

### 共享资源和网络

Pod 内的容器共享相同的网络命名空间，这意味着它们共享一个 IP 地址、端口空间，并且可以使用 localhost 相互通信。这简化了容器间通信和资源共享，例如磁盘卷可以挂载到同一 Pod 内的多个容器中。这种共享上下文对于设计为作为一个内聚单元工作的应用程序至关重要，这些应用程序需要组件之间的紧密耦合和高效通信。

### 简化管理和通信

通过将 Pod 视为最小的可部署单元，Kubernetes 简化了应用程序的部署和管理。Pod 是 Kubernetes 的单一管理点，扩展、网络和生命周期管理等操作应用于整个 Pod，而不是单个容器。这种方法简化了部署过程，并更容易确保相关容器被调度到同一主机上，从而提高性能并降低复杂性。

### 实现细粒度控制

Pod 提供了一种抽象级别，允许开发人员指定其应用程序应如何运行以及如何与系统的其他组件交互。这包括在每个 Pod 的基础上指定资源请求和限制，这有助于 Kubernetes 做出智能的调度决策，以优化整个集群的资源利用率。

### 促进服务发现和负载均衡

Kubernetes 使用 Pod 来促进服务发现和负载均衡。当在 Kubernetes 中创建一个服务时，它会在逻辑上将提供相同功能的 Pod 分组，并通过单个 DNS 名称或 IP 地址自动管理对它们的访问。这将应用程序的前端访问与运行应用程序的容器的单个实例解耦，从而实现无缝的扩展和管理。

总之，Pod 不仅仅是容器的分组机制；它们是 Kubernetes 中部署的基本单元，封装了应用程序的运行时环境，简化了资源共享，并增强了平台在分布式计算环境中高效管理、扩展和维护应用程序的能力。

## 运行一个简单的 Pod

让我们创建并运行一个简单的 Nginx Pod。

```
$ kubectl run nginx --image=nginx
pod/nginx created
```

**run** 子命令创建并运行一个特定的资源。在我们的例子中，它恰好是一个独立的 Pod。

**nginx** 参数是创建的 Pod 的名称。

**--image=nginx** 此选项指定用于 Pod 的容器镜像。在这种情况下，它使用的是来自 Docker Hub 的官方 Nginx 镜像。如果该镜像在节点上尚不可用，Kubernetes 将从容器注册表（Docker Hub）拉取此镜像。

让我们检查一下我们刚刚创建的 Pod。

```
$ kubectl get pod
NAME   READY   STATUS    RESTARTS   AGE
nginx  1/1     Running   0          26s
```

"get" 告诉 kubectl 获取一个资源，类型为 "pod"。

**NAME** 列显示 Pod 的名称。**READY** 列指示容器已就绪（即已启动并正在运行）。数字 **1/1** 表示 Pod 中的 1 个容器中有 1 个处于 "Ready" 状态。约定 **X/Y** 表示 Pod 中的 Y 个容器中有 X 个正在成功运行。对于健康状态，X 和 Y 应该相同。

**STATUS** 列指示 Pod 的状态，即它们处于 "Running" 状态。关于 Pod 状态的更多信息见下文。**RESTARTS** 是 Pod 被重启的次数。在我们的例子中是 0，这是好的。**AGE** 列显示 Pod 的年龄（废话！）。

虽然直接从命令行在集群中创建资源既简单又容易，但我们通常不会这样做。我们使用 YAML 来描述 Pod 的各种规范，并将该 YAML 清单 *应用* 到我们的集群中。

以下是其工作原理的简化模型：我们创建一个表示 Pod 期望状态的 YAML 构件。Kubernetes 集群会创建/修改 Pod，使其与我们的期望状态相匹配。Kubernetes 会尽最大可能将现有状态与期望状态进行匹配。有时，它无法做到这一点。例如，如果你只有 1 个节点，却请求集群创建数百个 Pod，集群会放弃，并表示它无法“将集群状态与你的期望状态匹配”。

以下是我们从命令行创建的 nginx Pod 对应的 YAML 构件。

```
apiVersion: v1 # (1)
kind: Pod # (2)
metadata: # (3)
  labels: # (4)
    run: nginx
  name: nginx # (5)
spec: # (6)
  containers: # (7)
  - image: nginx
    name: nginx
```

(1) -> 在 Kubernetes 中，API 是定义、管理和控制集群内各种资源的基础。

(2) -> Kubernetes 中的“资源”指的是系统知道如何管理的实体，例如 Pod、Deployment、Service 等，我们将在后续章节中看到。每种资源都由一个“kind”定义，它指定了所讨论资源的类型。

除了 kind，每个 Kubernetes 资源都与一个 API 版本相关联。API 版本至关重要，因为 Kubernetes 的 API 会随时间演进，添加新功能并修改现有功能。API 版本以 **apiVersion: <group>/<version>** 的格式指定，例如 **apiVersion: v1** 或 **apiVersion: apps/v1**，它告知 Kubernetes 资源定义所使用的 API 模式版本。这种版本控制系统允许 Kubernetes 提供向后兼容性，并平滑地管理向新 API 版本的过渡，确保用户可以按自己的节奏采用新功能，同时保持其现有部署的稳定性。

(3) -> metadata 是一组数据，提供有关 Kubernetes 系统内管理的其他数据的信息。它是一个结构化的信息集合，用于标识、描述和组织 Pod、Service 和 Deployment 等资源。元数据在资源管理中起着至关重要的作用，能够实现 Kubernetes 生态系统内资源的高效查询、选择和操作。

(4) -> 标签是附加到 Kubernetes 对象（如 Pod、Service 和 Deployment）的键值对。它们充当标识符和组织标签，使得管理、选择和操作一组对象变得更加容易。标签可以根据用户或应用程序的需要，以有意义的方式组织资源。例如，标签可以表示环境（例如 **env=production**）、应用程序版本（例如 **version=1.2.3**）或任何其他有助于大规模管理资源的分类。标签的一个关键特性是它们在选择器中的使用，选择器是允许用户根据标签标准过滤资源的查询。

(5) -> 你要创建的 Pod 的名称。是元数据的一部分。

(6) -> Kubernetes 中的“spec”是对象定义的一个必需部分，描述了资源的期望状态。你在这里指定要创建或配置的资源的特性。

例如，在 Pod 定义中，spec 可能包括要使用的容器镜像、副本数量、网络设置和卷挂载等配置。spec 本质上是一个蓝图，告诉 Kubernetes 如何构建和管理所讨论的资源。

(7) -> 容器是 Pod spec 的一部分，它告诉 Kubernetes Pod 包含哪些容器，以及它们的名称和镜像是什么。**请注意**，镜像应该是一个有效的容器注册表镜像，带有可选的标签。Name 是人类用来标识容器的任意名称。

你也可以通过运行以下命令生成此 YAML：

```
$ kubectl run nginx --image=nginx --dry-run=client -o yaml > nginx-pod.yaml
```

**--dry-run=client** 参数告诉 kubectl 输出资源的定义，而不是通过 Kubernetes API 创建资源。**-o yaml** 参数告诉 kubectl 以 YAML 格式输出资源规范。你也可以尝试使用不太有用的 **-o json** 参数，看看会发生什么。

当我们尝试应用这些更改时，得到以下结果：

```
$ kubectl apply -f nginx-pod.yaml
Warning: resource pods/nginx is missing the kubectl.kubernetes.io/last-applied-configuration annotation which is required by kubectl apply. kubectl apply should only be used on resources created declaratively by either kubectl create --save-config or kubectl apply. The missing annotation will be patched automatically.
pod/nginx configured
```

该消息明确指出，只有当前状态中不存在的更改才会被应用。这与 kubectl 工作方式背后的理念紧密相关。它尝试将系统的当前状态与 YAML 文件中的期望状态进行匹配。如果两者相同，则无需“应用”。

### 练习 5：更改 Pod 规范并重新应用

更改 Pod 规范 YAML 中 nginx 容器的名称，然后再次运行 **kubectl apply**。

### 练习 6：更改 Pod 的名称

尝试在 Pod 规范 YAML 的元数据中将 Pod 的名称从 **nginx** 更改为 **webapp**，然后运行 **kubectl apply**。

### 练习 7：运行 Apache httpd 服务器

类似于 Nginx，以名称“httpd”运行 Apache httpd 服务器作为 Pod，并带有标签 **app=webserver**。

## Pod 重启行为

让我们创建一个“休眠”5 秒的 Pod。

```
apiVersion: v1
kind: Pod
metadata:
  name: bash-sleep
spec:
  containers:
  - name: sleepy-app # (1)
    image: bash # (2)
    command: ["bash", "-c", "sleep 5"] # (3)
```

(1) -> Pod 内容器的名称。

(2) -> 我们将使用官方的 bash 镜像。

(3) -> **command** 规范类似于 Dockerfile 中的 **CMD** 指令。我们运行一个在 5 秒后终止的进程。

当我们应用这个 Pod 时，

```
$ kubectl apply -f bash-sleep-pod.yaml
```

Kubernetes 会愉快地运行 Pod，直到它停止运行。一分钟后，

```
$ kubectl get pod
NAME         READY   STATUS             RESTARTS   AGE
bash-sleep   0/1     CrashLoopBackOff   2 (18s ago)   55s
nginx        1/1     Running            0            4h20m
```

## 几分钟后...

```
$ kubectl get pod
NAME         READY   STATUS             RESTARTS   AGE
bash-sleep   0/1     CrashLoopBackOff   6 (45s ago)   7m15s
nginx        1/1     Running            0            4h27m
```

你可以观察到重启次数不断增加（在 **RESTARTS** 列中）。**STATUS** 是 **CrashLoopBackOff**。Kubernetes 看到 Pod 已停止运行（因为容器在 5 秒后终止），并尝试重启它。**READY** 列表明 Pod 中 1 个容器中有 0 个已就绪。

## CrashLoopBackOff 是什么意思？

CrashLoopBackOff 是你在 Kubernetes 中可能看到的一种状态消息，当 Pod 中的容器反复启动、崩溃，然后在循环中重启时会出现。当 Kubernetes 遵循指定的 restartPolicy（对于 Pod，默认为 Always）但发现容器无法保持运行时，就会发生这种行为。每次崩溃后，Kubernetes 会等待更长的时间再尝试重启容器，从而导致 CrashLoopBackOff 中的“back-off”部分。

导致 CrashLoopBackOff 状态的原因可能多种多样，但通常包括：

- 容器内的应用程序在启动过程中遇到错误，原因可能是配置错误、缺少环境变量、依赖项不正确或缺失等。
- 容器尝试绑定到一个已被使用的端口，或者没有使用该端口的权限。
- 应用程序设计为在完成任务后退出，但 Pod 的 restartPolicy 设置为 Always，导致 Kubernetes 不断重启容器。

## 练习 8：配置休眠 Pod 不重启

修复 bash-sleep 容器的配置，使 Pod 停止运行。

## 描述 Pod

kubectl 有一个“describe”操作，用于描述资源。让我们描述我们的 nginx Pod。

### 练习 9：描述 bash-sleep Pod。

即使对于崩溃的 Pod，你也可以运行 describe 命令。你将在后面的故障排除章节中了解到，尤其应该对崩溃的 Pod 运行 describe。试着理解并推断 kubectl 关于这个 Pod 给出了哪些信息。

```
$ kubectl describe pod nginx
Name:         nginx # (1)
Namespace:    default # (2)
Priority:     0
Service Account: default
Node:         first-cluster-default-pool-oeo8s/10.122.0.4 # (3)
Start Time:   Mon, 19 Feb 2024 11:32:56 +0530 # (4)
Labels:       run=nginx # (5)
Annotations:  <none>
Status:       Running # (6)
IP:           10.244.0.187 # (7)
IPs:
  IP: 10.244.0.187
Containers:
  nginx:
    Container ID:   containerd://49122f24393291163f861fd366dc027395ed636d3442e5705e84875a340117af # (8)
    Image:          nginx # (9)
...
```

(1) -> **nginx** - Pod 在 Kubernetes 集群中的唯一标识符。

(2) -> **default** - Pod 运行所在的命名空间，用于将资源组织成组以便进行管理和访问控制。我们将在后面的章节中更详细地讨论命名空间。

(3) -> **first-cluster-default-pool-oeo8s/10.122.0.4** - Pod 被调度并运行所在的集群节点的名称和 IP 地址。

(4) -> **Mon, 19 Feb 2024 11:32:56 +0530** - Pod 启动的时间戳。

(5) -> **run=nginx** - 附加到 Pod 的键值对，用于组织、选择和管理 Kubernetes 对象。

(6) -> **Running** - Pod 的当前状态，表示它正在活跃运行。

(7) -> **10.244.0.187** - 分配给 Pod 的 IP 地址，使其能够与集群内的其他 Pod 和服务进行通信。

(8) -> **containerd://49122f24393291163f861fd366dc027395ed636d3442e5705e84875a340117af** - 在 Pod 内运行的容器的唯一标识符，由容器运行时（本例中为 containerd）管理。

(9) -> **nginx** - 用于创建 Pod 中运行的容器的 Docker 镜像，指定了它旨在运行的应用程序或服务。

## 事件

如果你尝试了前面的练习，你会注意到 describe 输出包含一个按时间倒序排列的事件列表。这些是 Kubernetes 事件，当集群中发生事情时就会产生。让我们看看在启动 nginx Pod 时发生的事件：

```
$ kubectl get events
LAST SEEN   TYPE     REASON      OBJECT      MESSAGE
14m         Normal   Scheduled   pod/nginx   Successfully assigned default/nginx to first-cluster-default-pool-oeo8s # (1)
14m         Normal   Pulling     pod/nginx   Pulling image "nginx" # (2)
14m         Normal   Pulled      pod/nginx   Successfully pulled image "nginx" in 8.203s (8.203s including waiting) # (3)
14m         Normal   Created     pod/nginx   Created container nginx # (4)
14m         Normal   Started     pod/nginx   Started container nginx # (5)
44s         Normal   Killing     pod/nginx   Stopping container nginx # (6)
32s         Normal   Scheduled   pod/nginx   Successfully assigned default/nginx to first-cluster-default-pool-oeo8s # (7)
32s         Normal   Pulling     pod/nginx   Pulling image "nginx" # (8)
30s         Normal   Pulled      pod/nginx   Successfully pulled image "nginx" in 1.634s (1.634s including waiting) # (9)
30s         Normal   Created     pod/nginx   Created container nginx
30s         Normal   Started     pod/nginx   Started container nginx
```

(1) -> Kubernetes 调度器将 nginx Pod 分配给了名为 first-cluster-default-pool-oeo8s 的节点。

(2) -> **first-cluster-default-pool-oeo8s** 上的 kubelet 开始为 Pod 拉取名为 nginx 的 Docker 镜像。

(3) -> **first-cluster-default-pool-oeo8s** 上的 kubelet 在 8.203 秒内（包括任何等待时间）成功从容器镜像仓库拉取了 nginx 镜像。

(4) -> 拉取镜像后，kubelet 在 Pod 内创建了 nginx 容器。

(5) -> kubelet 启动了 nginx 容器，启动了它旨在运行的应用程序或服务。

(6) -> kubelet 正在停止 nginx 容器，可能是由于 Pod 删除请求。我后来想起，我删除了 Pod 只是为了验证一个练习 :)

(7) -> nginx Pod 再次被调度到同一个节点，表明 Pod 重启或重新调度。我在删除 Pod 后重新创建了它。

(8) -> kubelet 再次开始为 Pod 拉取 nginx 镜像，这是 Pod 重启或重新调度过程的一部分。

(9) -> 这次镜像拉取更快，表明该节点上可能有缓存的层。

**请注意**，事件也是 Kubernetes 中的一种资源，就像 Pod 一样，这引出了一个关于 Kubernetes API 和资源的简短插曲。

## Kubernetes 资源

在 Kubernetes 中将一切建模为资源是一个基本的设计选择，这与其管理应用程序和基础设施的声明式方法相一致。这种模型提供了几个优势：

1.  **一致性：** 通过将所有实体（Pod、服务、部署等）视为资源，Kubernetes 提供了一种一致的方式来管理系统的不同方面。用户以类似的方式与每个资源交互，无论是创建、读取、更新还是删除资源，这简化了学习曲线和操作实践。

2.  **声明式管理：** Kubernetes 采用声明式模型，用户以资源的形式指定其应用程序或基础设施的期望状态。然后 Kubernetes 努力确保实际状态与期望状态匹配。这种方法抽象了实现期望状态所需的程序步骤，简化了扩展、滚动更新和自愈等复杂操作。

3.  **可扩展性：** 资源模型允许 Kubernetes 高度可扩展。可以定义自定义资源来扩展 Kubernetes 的新功能，以适应特定需求，而无需更改 Kubernetes 核心代码。这使开发人员能够基于 Kubernetes 的功能进行构建，并与其他工具和系统无缝集成。

4.  **自动化和编排：** 将实体视为资源允许 Kubernetes 有效地自动化其管理。Kubernetes 控制器持续监控资源状态，并采取行动来协调期望状态和当前状态之间的差异。这实现了复杂的编排功能，例如自动推出更新、根据需求扩展应用程序以及维护系统健康。

5.  **互操作性：** 资源模型促进了 Kubernetes 生态系统内不同组件之间的互操作性。以标准方式定义的资源可以被各种工具和操作员轻松理解和操作，增强了生态系统的丰富性和实用性。

6.  **访问控制和策略执行：** Kubernetes 可以对所有资源统一应用访问控制、策略和配额。这确保了整个集群的安全性、合规性和高效资源使用，无论资源类型或其在系统中的用途如何。

本质上，在 Kubernetes 中将一切建模为资源提供了一种统一、可扩展且可管理的方式来自动化容器化应用程序的部署、扩展和运行。它为 Kubernetes 强大的编排能力奠定了基础，使用户更容易以声明式方式描述复杂系统，并让 Kubernetes 随时间维护这些系统的期望状态。

## Kubectl，它如何工作

让我们剖析一下 **kubectl** 命令是如何工作的。

## Kubectl 流程

```
$ kubectl get pod --v=8
I0219 16:55:21.244844 83925 loader.go:395] Config loaded from file: /Users/lakshminp/.kube/config
I0219 16:55:21.252782       83925 round_trippers.go:463] GET https://xxxxxxxx-abcd-1234-yyyy-c88487140b6f.k8s.ondigitalocean.com/api/v1/namespaces/default/pods?limit=500
I0219 16:55:21.252791       83925 round_trippers.go:469] Request Headers:
I0219 16:55:21.252798       83925 round_trippers.go:473]     User-Agent: kubectl/v1.28.1 (darwin/arm64) kubernetes/8dc49c4
I0219 16:55:21.252803       83925 round_trippers.go:473]     Accept: application/json;as=Table;v=v1;g=meta.k8s.io,application/json;as=Table;v=v1beta1;g=meta.k8s.io,application/json
I0219 16:55:22.358534       83925 round_trippers.go:574] Response Status: 200 OK in 1105 milliseconds
I0219 16:55:22.358698       83925 round_trippers.go:577] Response Headers:
I0219 16:55:22.358723       83925 round_trippers.go:580]     X-Kubernetes-Pf-Flowschema-Uid: a15935ee-895f-420e-8da3-2c4ddfff3643
I0219 16:55:22.358763       83925 round_trippers.go:580]     X-Kubernetes-Pf-Prioritylevel-Uid: 11b53add-ab9f-4eac-b48e-d2605272f1f7
I0219 16:55:22.358785       83925 round_trippers.go:580]     Date: Mon, 19 Feb 2024 11:25:22 GMT
I0219 16:55:22.358803       83925 round_trippers.go:580]     Audit-Id: 4d4bf345-84d9-43f0-a829-b6e0f0190f9a
I0219 16:55:22.358820       83925 round_trippers.go:580]     Cache-Control: no-cache, private
I0219 16:55:22.358905       83925 round_trippers.go:580]     Content-Type: application/json
I0219 16:55:22.361063       83925 request.go:1212] Response Body: {"kind":"Table","apiVersion":"meta.k8s.io/v1","metadata":{"resourceVersion":"147606"},"columnDefinitions":[{"name":"Name","type":"string","format":"name","description":"Name must be unique within a namespace. Is required when creating resources, although some resources may allow a client to request the generation of an appropriate name automatically. Name is primarily intended for creation idempotence and configuration definition. Cannot be updated. More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/names#names","priority":0},{"name":"Ready",
```

## Pod 中的多个容器

在 Pod 中运行多个容器之前，我们先简要了解一下容器端口的概念。

在 Kubernetes Pod 规范中，**containerPort** 参数是容器规范的一部分。它声明了容器监听的端口。虽然此声明对于容器打开端口并非强制性的，但它有几个用途：

**文档记录**：它记录了容器预期监听的端口，使检查 Pod 规范的开发人员和运维人员能更清楚地了解 Pod 的配置。

**服务发现**：与服务结合使用时，**containerPort** 可用于简化 Kubernetes 集群内的服务发现。当 Service 选择一个 Pod 时，**containerPort** 可用于将流量定向到 Pod 容器的正确端口，我们将在后续章节中看到这一点。

**安全**：在强制执行网络策略的环境中，指定 **containerPort** 可以是定义和执行控制 Pod 出入站流量安全规则的一部分。

让我们尝试在同一个 Pod 中运行 2 个 nginx 容器。以下是相应的 YAML 规范。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: two-containers
  labels:
    purpose: demonstrate-multi-container
spec:
  containers:
  - name: nginx-container-1 # (1)
    image: nginx
    ports: # (2)
    - containerPort: 80
  - name: nginx-container-2 # (3)
    image: nginx
    ports: # (4)
    - containerPort: 81
```

(1) -> 第一个容器命名为 **nginx-container-1**。

(2) -> 从 **containerPort** 值可以看出，它暴露了端口 **80**。

(3) -> 第二个容器命名为 **nginx-container-2**。请注意，两者运行相同的镜像。这是完全有效的。只有名称应该是唯一的。

(4) -> 第二个容器暴露了一个不同的端口。

当我们尝试运行这个 Pod 时，会遇到一个令人头疼的时刻。

```bash
$ kubectl apply -f two-containers.yaml
```

几分钟后...

```bash
$ kubectl get pod
NAME                READY   STATUS             RESTARTS   AGE
bash-sleep          0/1     CrashLoopBackOff   16         4m13s ago   62m
nginx               1/1     Running            0          5h22m
two-containers      1/2     CrashLoopBackOff   5          94s ago     5m5s
```

你可以看到我们的 **two-containers** Pod 处于 **CrashLoopBackOff** 状态，2 个容器中只有 1 个是活跃的，显然 Kubernetes 正在徒劳地尝试对这个 Pod 进行“心肺复苏”，而它将永远无法成功运行！

为什么会这样？因为 Pod 中的容器共享相同的 IP 和端口空间。如果第一个 nginx 容器占用了端口 **80**，那么第二个不幸的容器就无法再占用该端口，因此会失败。你可能会争辩说你为两个容器指定了不同的 **containerPort** 值，但该容器镜像中的 nginx 进程被硬编码为在端口 **80** 上运行，所以无论你给 **containerPort** 指定什么任意端口值，第二个容器都注定会被终止。

解决方案是在不同的端口上运行 nginx。官方的 nginx 镜像开箱即用不支持这种配置，但 Bitnami 的 nginx 镜像允许你通过环境变量配置端口号。让我们尝试一下这种方法。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: two-containers-better
  labels:
    purpose: demonstrate-multi-container
spec:
  containers:
  - name: nginx-container-1
    image: bitnami/nginx # (1)
    env: # (2)
    - name: NGINX_HTTP_PORT_NUMBER # (3)
      value: "8080"
    ports:
    - containerPort: 8080
  - name: nginx-container-2
    image: bitnami/nginx
    env:
    - name: NGINX_HTTP_PORT_NUMBER
      value: "8081" # (4)
    ports:
    - containerPort: 8081 # (5)
```

(1) -> 如前所述，我们运行 **bitnami/nginx** 镜像。

(2) -> 为了指定运行时环境变量，类似于 docker-compose 的 **environment**，Pod 规范有一个每个容器的 **env:** 部分。这是一个用于指定环境变量的名称/值对列表。

(3) -> Bitnami nginx 镜像允许你在选择的端口上运行 nginx，这可以通过设置 **NGINX_HTTP_PORT_NUMBER** 变量来配置。我们在每个容器的 **env** 部分提供此值。

(4) -> 第一个容器在端口 **8080** 上运行 nginx，因此我们在不同的端口上运行此容器，即 **8081**。

**注意：** 这里需要理解的是，**NGINX_HTTP_PORT_NUMBER** 表示 nginx 在 Pod 内部运行并暴露的端口。这应该与 Pod 中每个容器的 **containerPort** 值相匹配。

(5) -> 在这种情况下，**containerPort** 值将生效，因为容器确实在指定的端口上运行了进程。

当我们 kubectl apply 这个 Pod 时，可以看到它运行成功了。

Pod 内部是什么样子

```bash
$ kubectl apply -f two-containers-better.yaml
pod/two-containers-better created
```

一分钟后...

```bash
$ kubectl get pod
NAME                     READY   STATUS             RESTARTS   AGE
bash-sleep               0/1     CrashLoopBackOff   17         3m14s ago   67m
nginx                    1/1     Running            0          5h27m
two-containers           1/2     CrashLoopBackOff   5          41s ago     4m4s
two-containers-better    2/2     Running            0          35s
```

### 练习 10：在 Pod 中运行 3 个容器。

使用与上一节相同的规范，添加另一个容器并运行它，方式如下：

a. 通过更新上面的 Pod 规范，在同一个 Pod 中运行

b. 通过创建一个新的 Pod 规范，在不同的 Pod 中运行

尝试理解任务 (a) 和 (b) 之间的区别是什么。

### 练习 11：2 个容器的变体

再次在 Pod 中运行 2 个容器。但一个是 nginx，另一个是一个 bash 容器，它运行 "sleep 5" 然后终止。Pod 会崩溃吗？如果会，你如何在不更改 bash 容器镜像的情况下修复它？

## 一个 Pod 中有多少个容器

在 Kubernetes 中，Pod 被设计用来运行一个或多个紧密耦合且需要共享资源的容器。虽然 Kubernetes 没有对 Pod 中可以运行的容器数量设置硬性限制，但实际限制通常由 Pod 所调度到的节点上的可用资源，以及你的应用程序的设计和需求来决定。

## 实际考虑因素

**资源约束：** Pod 中的每个容器都会消耗节点上的 CPU、内存和其他资源。Pod 内所有容器的总资源消耗不得超过节点的可用资源。

**网络：** Pod 中的所有容器共享同一个网络命名空间，这意味着它们共享相同的 IP 地址和端口空间。你需要确保 Pod 内运行的应用程序不会尝试使用相同的端口。

**管理复杂性：** Pod 中的容器越多，管理日志、卷和容器生命周期事件的复杂性就越高。通常更简单的方法是让 Pod 专注于运行一个主要的应用程序容器，并在必要时（例如日志收集器或监控代理）附带几个辅助容器。

**安全性：** 同一 Pod 中的容器共享相同的生命周期、网络和存储。这种紧密耦合意味着，如果一个容器被攻破，Pod 中的其他容器也可能面临风险。设计 Pod 时只包含必要的最少容器，有助于限制安全问题的范围。

## 最佳实践

**单一职责原则：** 每个 Pod 应专注于运行给定应用程序或服务的单个实例。如果你的应用程序由多个可以独立扩展的组件组成，它们应该被部署在单独的 Pod 中。

**边车容器：** 在 Pod 中运行一个或多个边车容器来支持主应用程序是很常见的，例如处理日志记录、监控或数据同步任务。然而，边车容器的数量应保持最少，并与应用程序的需求相关。

## 技术限制

虽然 Kubernetes 本身没有明确限制每个 Pod 的容器数量，但必须考虑 kubelet 的性能以及管理 Pod 中多个容器的开销。Kubernetes 文档和社区建议通常建议将每个 Pod 的容器数量保持在最少，而是专注于需要紧密共享资源的容器的逻辑分组。

总之，虽然从技术上讲你可以在单个 Pod 中运行多个容器，但最佳实践和实际资源考虑建议保持数量较少，并专注于应用程序中紧密耦合的组件。

## 端口转发

我们已经介绍了 Kubernetes Pod 的基本功能，它们是 Kubernetes 集群中最小的构建单元。但我们究竟如何访问这些 Pod/容器中运行的应用程序呢？

在 Kubernetes 中有一些惯用的方法，我们将在后面的章节中介绍。目前，kubectl 通过端口转发为我们提供了实现这一目标的途径。

要在 Kubernetes 中转发 Pod 的端口，你可以使用 `kubectl port-forward` 命令。该命令将一个或多个本地端口转发到 Pod。端口转发对于调试或从本地机器临时访问 Pod 内运行的应用程序特别有用。

![](img/91c4bde832e221ba5bc429caf9587286_118_0.png)

## 端口转发

以下是该命令的一般语法：

```
$ kubectl port-forward pod/<pod-name> <local-port>:<pod-port>
```

**<pod-name>** 是你要转发端口的 Pod 的名称。

**<local-port>** 是你本地机器上的端口号。

**<pod-port>** 是 Pod 上的端口号。

让我们为 nginx Pod 进行端口转发。

```
$ kubectl port-forward pod/nginx 8080:80
Forwarding from 127.0.0.1:8080 -> 80
Forwarding from [::1]:8080 -> 80
```

现在，如果我们访问 [http://localhost:8080/](http://localhost:8080/)，我们应该能够在本地机器上访问 Kubernetes 集群中 nginx Pod 内运行的 nginx 容器。

![](img/91c4bde832e221ba5bc429caf9587286_120_0.png)

*nginx 端口转发*

### 练习 12：在 8008 端口进行端口转发

尝试将同一个 Pod 转发到你主机的 8008 端口。

## 练习 13：转发多个容器

使用 2 个 Pod 的 bitnami nginx 设置。分别将 8080 nginx 转发到 localhost 的 8080，将 8081 nginx 容器转发到 localhost 的 8081。

## 获取 Pod 的日志

kubectl 帮助你获取 Pod 的日志。

获取日志的语法是：

```
$ kubectl logs nginx
/docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
/docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/
/docker-entrypoint.sh: Launching /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
10-listen-on-ipv6-by-default.sh: info: Getting the checksum of /etc/nginx/conf.d/default.conf
10-listen-on-ipv6-by-default.sh: info: Enabled listen on IPv6 in /etc/nginx/conf.d/default.conf
/docker-entrypoint.sh: Sourcing /docker-entrypoint.d/15-local-resolvers.envsh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/20-envsubst-on-templates.sh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/30-tune-worker-processes.sh
/docker-entrypoint.sh: Configuration complete; ready for start up
2024/02/19 06:16:40 [notice] 1#1: using the "epoll" event method
2024/02/19 06:16:40 [notice] 1#1: nginx/1.25.4
2024/02/19 06:16:40 [notice] 1#1: built by gcc 12.2.0 (Debian 12.2.0-14)
2024/02/19 06:16:40 [notice] 1#1: OS: Linux 6.1.0-17-amd64
2024/02/19 06:16:40 [notice] 1#1: getrlimit(RLIMIT_NOFILE): 1048576:1048576
2024/02/19 06:16:40 [notice] 1#1: start worker processes
2024/02/19 06:16:40 [notice] 1#1: start worker process 30
2024/02/19 06:16:40 [notice] 1#1: start worker process 31
127.0.0.1 - - [19/Feb/2024:12:41:20 +0000] "GET / HTTP/1.1" 200 615 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" "-"
127.0.0.1 - - [19/Feb/2024:12:41:27 +0000] "GET / HTTP/1.1" 200 615 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" "-"
```

请确保在运行 port-forward 命令的窗口之外的另一个窗口中执行此操作。

我们也可以通过向上述命令传递 `-f` 标志来实时跟踪日志。

```
$ kubectl logs -f nginx
/docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
...
127.0.0.1 - - [19/Feb/2024:12:41:20 +0000] "GET / HTTP/1.1" 200 615 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" "-"
127.0.0.1 - - [19/Feb/2024:12:41:27 +0000] "GET / HTTP/1.1" 200 615 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" "-"

127.0.0.1 - - [19/Feb/2024:12:52:17 +0000] "GET /some-page HTTP/1.1" 404 555 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" "-"
2024/02/19 12:52:17 [error] 31#31: *3 open() "/usr/share/nginx/html/some-page" failed (2: No such file or directory), client: 127.0.0.1, server: localhost, request: "GET /some-page HTTP/1.1", host: "localhost:8080"
```

注意日志跟踪是如何工作的，以及当我访问一个不存在的页面（如 [http://localhost:8080/some-page](http://localhost:8080/some-page)）时 nginx 返回 404 错误。

在容器化应用程序中将日志打印到 stdout（标准输出）被认为是一种良好的实践，原因有几个，它符合容器化环境的设计原则，并有助于有效的日志管理。以下是其优势：

## 简化日志收集

像 Kubernetes 这样的容器编排系统和像 Docker 这样的容器运行时被设计为自动捕获容器的 stdout 和 stderr 流。这种设计简化了日志收集，因为日志被集中管理，可以轻松地聚合和监控，而无需在容器内添加额外的日志驱动程序或代理。

## 简化日志管理

通过将日志定向到 stdout，管理日志变得更加直接。编排工具可以处理日志的轮转、聚合和保留，而无需在容器内进行额外的配置或工具。这种方法降低了日志管理的复杂性，并避免了日志文件填满容器存储。

## 支持十二要素应用方法论

十二要素应用方法论 ([https://12factor.net/](https://12factor.net/)) 是一套构建可扩展、

## 总结

本章我们全面介绍了 Kubernetes 的基本构建模块：Pod。本章详细阐述了在 Kubernetes 生态系统中运行容器的机制，强调了 Pod 作为最小可部署单元的关键作用。它详细说明了 Pod 如何允许共享存储、网络和生命周期的容器共存，阐明了 Pod 在分组容器以实现高效管理和通信方面的重要性。

我们介绍了 kubectl 作为 Kubernetes 必备的命令行工具，它允许你与集群交互并有效管理应用程序。通过理解 kubeconfig，你学会了如何安全地连接到集群，并开始与 Kubernetes 进行动手交互，为迄今为止所涵盖概念的实际应用奠定了基础。

展望下一章，我们将深入探讨 Pod 管理的复杂细节。我们将探索 Pod 的高级功能，深入研究 Pod 健康检查、生命周期管理、Pod 间通信以及在生产环境中使用 Pod 的最佳实践等方面。

## 第 5 章：Pod 201

我们现在知道了什么是 Pod，Pod 如何融入 Kubernetes 生态系统的大局，以及 Pod 背后的概念和设计原则。我们还对比了许多 docker 容器/docker compose 与 Pod 之间的相似之处。在本章中，我们将超越这些相似点，深入探讨 Pod 的具体细节，并更进一步。

### 进入 Pod 内部

我们可以使用 **kubectl exec** 在权限和边界内进入 Pod 并在其中执行操作。

kubectl exec 命令用于在 Kubernetes Pod 中的容器内执行命令。此命令对于调试和交互式故障排除 Kubernetes 管理的容器非常有用。它允许开发人员和管理员直接访问容器的运行时环境，从而更容易诊断问题、检查应用程序状态或与容器内运行的进程进行交互。

kubectl exec 的基本语法是：

```
$ kubectl exec [POD_NAME] -- [COMMAND]
```

**[POD_NAME]** 是包含你要访问的容器的 Pod 的名称。

**[COMMAND]** 是你想要在容器内执行的命令。如果命令包含参数，它们跟在命令名称之后。

让我们在 nginx Pod 内运行一个简单的命令。

```
$ kubectl exec nginx -- ls /
bin
boot
dev
docker-entrypoint.d
docker-entrypoint.sh
etc
home
lib
lib64
media
mnt
opt
proc
root
run
sbin
srv
sys
tmp
usr
var
```

为什么需要双破折号 ( -- )？

双破折号 ( -- ) 将 kubectl exec 参数与你想要在容器内运行的命令分开。

如果我们运行上述命令的变体而不使用 -- ，它看起来会像这样：

```
$ kubectl exec nginx ls -l
```

在这个例子中，-l 可能会被错误地解释为 kubectl exec 的选项，而不是作为要在容器内运行的 ls 命令的一部分。

因此，正确的方式是，

```
$ kubectl exec nginx -- ls -l
```

如果你想在特定容器内运行命令，请添加 -c [container-name] 选项。

让我们打印双容器 bitnami nginx-es Pod 中的 nginx 端口环境变量。

```
$ kubectl exec two-containers-better -c nginx-container-1 -- sh -c 'echo $NGINX_HTTP_PORT_NUMBER'
8080
```

对于交互式命令，例如打开 shell，请添加 -i（交互式）和 -t（TTY）标志以分配伪 TTY 并保持 stdin 打开：

```
$ kubectl exec -it nginx -- /bin/bash
```

此命令在 **nginx** Pod 的 **nginx** 容器内打开一个交互式 bash shell。然后，你可以像登录到容器上的 shell 会话一样在 shell 中运行任何命令。要退出 shell，请输入 **exit**。

### exec 的用例

- **调试应用程序：** 快速检查应用程序的当前状态、检查环境变量或查看存储在容器本地的日志。
- **文件操作：** 在容器的文件系统内执行文件操作，如查看、编辑或删除文件，以排查配置问题或临时修改应用程序行为。
- **数据库交互：** 访问并交互运行在 Pod 内的数据库，用于维护任务、查询或更新，而无需将数据库服务暴露到外部。

## 练习 14：打印端口号

打印双容器 bitnami nginx-es Pod 中第 2 个容器的端口号。

### 在不同节点上运行 Pod

Kubernetes 使用其 kube-scheduler 组件（控制平面的一部分）来决定在何处调度 Pod。kube-scheduler 根据多个因素确定最适合 Pod 的节点，确保集群内的最佳运行和资源利用。以下是该过程更详细的说明：

调度器评估 Pod 的资源需求，例如 CPU、内存和存储，并将其与节点上的可用资源进行匹配。它旨在将 Pod 放置在具有足够资源以满足 Pod 需求且不会使任何节点过载的节点上。

![Kubernetes 调度器工作原理（简化版）](img/91c4bde832e221ba5bc429caf9587286_130_0.png)

如果有多个节点符合这些标准，则调度器会随机将 Pod 放置在其中一个节点上，以避免任何单个节点过载。

让我们尝试将 Pod 调度到特定节点。为此，我们先选择一个节点。

```
$ kubectl get nodes
NAME                           STATUS   ROLES    AGE   VERSION
first-cluster-default-pool-oeo85   Ready    <none>   15h   v1.29.1
first-cluster-default-pool-oeo8k   Ready    <none>   15h   v1.29.1
first-cluster-default-pool-oeo8s   Ready    <none>   15h   v1.29.1
```

我们选择 **first-cluster-default-pool-oeo8s** 节点来调度我们的 nginx Pod。

以下是我们将对 Pod 规范进行的更改：

```yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: nginx
  name: nginx-201
spec:
  containers:
  - image: nginx
    name: nginx
  nodeName: first-cluster-default-pool-oeo8s # (1)
```

(1) -> 指示将调度 Pod 的节点名称。

当指定 **nodeName** 时，Pod 会绕过 kube-scheduler，直接由命名节点上的 kubelet 尝试将 Pod 放置在该节点上。这通常不是一个好主意，因为我们违反了 DevOps 的基本原则，即我们应该将基础设施视为牛群（可随意替换），而不是宠物（需要精心呵护）。

在这种情况下，我们通过记住节点名称并在 Pod 规范中指定它，将节点视为宠物。云环境中的节点名称是不可预测的。如果指定的节点不存在，Pod 将不会运行。如果指定的节点没有足够的资源来容纳 Pod，Pod 将失败，且不会有任何关于失败原因的提示。

还有其他方法可以将 Pod 调度到特定节点，我们将在后面讨论部署时介绍。

目前，只需理解我们可以对 Pod 在节点上的放置方式发表意见即可。

### 存活探针

Kubernetes 中使用存活探针来确定容器是否按预期运行。如果存活探针失败，kubelet 会终止容器，该容器将根据其重启策略重新启动。存活探针对于应用程序的自愈至关重要，因为即使应用程序进程仍在运行，应用程序也可能由于死锁或其他问题而变得无响应。

这是一个包含 Nginx 容器存活探针的 Pod 规范示例。此示例使用 HTTP GET 请求作为存活探针，这是检查容器健康状况的常用方法之一。kubelet 向容器内的 Nginx 服务器发送 HTTP GET 请求；如果服务器未能以成功状态（200-399）响应，容器将被重新启动。

apiVersion: v1
kind: Pod
metadata:
  name: nginx-with-liveness
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
    livenessProbe: # (1)
      httpGet: # (2)
        path: /
        port: 80 # (3)
      initialDelaySeconds: 15 # (4)
      timeoutSeconds: 2 # (5)
      periodSeconds: 5 # (6)
      failureThreshold: 3 # (7)

(1) -> 这指定了用于存活探针的 HTTP GET 请求。kubelet 会向容器的 IP 地址发送此请求，请求指定的端口和路径。

(2) -> 探针请求的 URL 路径。此示例使用根路径（ / ），这是检查 Web 服务器健康状况的常用端点。

(3) -> kubelet 将发送 HTTP GET 请求的端口。这应与 Nginx 服务器监听的端口匹配，在此示例中为 80。

(4) -> 容器启动后，在启动存活探针之前等待的秒数。这为应用程序提供了启动时间，然后再开始进行检查。在此示例中，设置为 **15** 秒。

(5) -> 探针超时的秒数。如果 Nginx 服务器在 **2** 秒内没有响应，则认为探针失败。

(6) -> 指定执行探针的频率（以秒为单位）。此示例每 **5** 秒检查一次 Nginx 服务器的健康状况。

(7) -> kubelet 决定终止并重启容器之前，探针可以失败的次数。在此示例中，如果 Nginx 容器连续 3 次未能通过存活检查，则将重启该容器。

此配置确保如果容器内的 Nginx 服务器变得无响应（例如，开始返回错误代码或完全停止响应），kubelet 将自动重启容器，从而可能无需手动干预即可解决问题。存活探针是确保在 Kubernetes 中运行的应用程序可靠性和弹性的关键部分。

## 就绪探针

Kubernetes 中的就绪探针用于确定容器何时准备好开始接收流量。

与用于了解容器是否存活的存活探针不同，就绪探针用于了解容器是否准备好处理请求。这种区别对于启动后需要一些时间才能完全运行或可能暂时无法处理流量的服务至关重要。就绪探针失败不会终止容器，但会将 Pod 从服务的负载均衡器中移除，直到它通过就绪检查，确保流量不会被发送到尚未准备好处理它的 Pod。

以下是一个包含 Nginx 容器就绪探针的示例 Pod 规范。此示例使用 HTTP GET 请求进行就绪探针，类似于存活探针，但目的是检查是否准备好处理流量。

```
apiVersion: v1
kind: Pod
metadata:
  name: nginx-with-readiness
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
    readinessProbe:
      httpGet: # (1)
        path: /
        port: 80
      initialDelaySeconds: 5 # (2)
      timeoutSeconds: 1 # (3)
      periodSeconds: 5 # (4)
      failureThreshold: 3 # (5)
```

(1) ->
httpGet：指定就绪探针通过 HTTP GET 请求进行。
path：探针请求的 URL 路径。此示例检查根路径（/），这是确认 Web 服务器准备好提供内容的典型方式。
port：发送请求的容器端口。对于 Nginx，通常是端口 80。

(2) -> 容器启动后，在启动就绪探针之前等待的秒数。将此值设置为 5 秒，可以让 Nginx 服务器在需要响应就绪检查之前有时间启动。

(3) -> 探针等待响应的时间。如果就绪检查在 1 秒内未完成，则认为探针失败。

(4) -> 指定就绪探针的频率（以秒为单位）。此示例每 5 秒执行一次检查，以持续验证 Nginx 服务器是否准备好接收流量。

(5) -> 将容器标记为未就绪所需的连续失败次数。在此示例中，连续 3 次失败后，Pod 将从任何匹配它的服务的端点列表中移除，因此在再次就绪之前不会接收任何流量。

使用具有这些设置的就绪探针可确保 Pod 内的 Nginx 容器在完全准备好处理请求之前不会接收流量。此设置对于确保流量仅定向到完全运行的 Pod 特别有用，从而增强了在 Kubernetes 中部署的应用程序的整体可靠性和稳定性。

## 存活探针和就绪探针的最佳实践

存活探针和就绪探针对于维护 Kubernetes 中应用程序的健康和可用性至关重要。有效使用它们需要理解其目的和细微差别。以下是利用这两种探针的一些最佳实践：

- 1. 对复杂应用程序同时使用两种探针 存活探针：确保用于检查您的应用程序是否存活并需要重启。存活探针失败应表明您的应用程序处于无法恢复的状态，必须重启。 就绪探针：使用就绪探针来确定您的应用程序何时准备好开始接收流量。这对于启动后需要时间才能处理流量或可能因负载或依赖项中断而暂时不可用的应用程序至关重要。
- 2. 根据应用程序要求配置探针 了解应用程序的启动时间、依赖项和典型故障模式，以适当配置探针的参数（initialDelaySeconds、periodSeconds、timeoutSeconds 和 failureThreshold）。避免使用可能导致不必要的重启或流量被阻止到达健康 Pod 的激进探针配置。
- 3. 区分存活检查和就绪检查 设计应用程序的健康端点以准确反映存活状态和就绪状态。存活端点可能检查关键进程的可用性，而就绪端点可以检查数据库连接、缓存就绪性或其他依赖项。
- 4. 避免对两种探针使用相同的端点 虽然可能倾向于对存活探针和就绪探针使用相同的端点，但这可能导致问题，例如暂时过载的应用程序因无响应而被终止，而不是给予其恢复时间。不同的端点可以反映“存活”与“准备好处理流量”的细微状态差异。
- 5. 从保守的阈值开始 从 failureThreshold 和 timeoutSeconds 的保守阈值开始，以防止抖动（频繁的状态变化）和不必要的重启。随着您对应用程序在生产环境中的行为有更深入的了解，可以调整这些值。
- 6. 为启动缓慢的容器利用启动探针 对于启动时间较长的应用程序，考虑使用启动探针。启动探针允许您为就绪和存活检查的启动设置较长的延迟，为您的应用程序提供足够的初始化时间，而不会被过早终止。
- 7. 基于观察进行监控和调整 定期监控应用程序的性能和探针的有效性。根据观察到的数据，必要时调整探针配置。过于敏感的探针可能造成与其旨在预防的问题一样多的停机时间。

## 命名空间

到目前为止，从讨论 Kubernetes 架构到将 Pod 放置在节点中，我们一直假设命名空间的概念不存在。Kubernetes 中的命名空间是在多个用户和项目之间划分集群资源的一种方式。它们本质上是在 Kubernetes 集群内限定访问和资源范围的一种方法。命名空间有助于组织集群中的对象（例如 Pod、服务和部署），并可以简化访问控制和资源管理，尤其是在拥有许多用户和应用程序的环境中。

## 命名空间

## 为什么我们需要命名空间

### 环境隔离

命名空间通常用于在同一集群内隔离不同的环境，例如开发、预发布和生产环境，否则这些环境可能会相互干扰。

### 资源组织

它们有助于组织和分类 Kubernetes 集群中的资源。这对于部署了多个项目或团队资源的大型集群尤其有用。

### 访问控制

Kubernetes 将命名空间与基于角色的访问控制（RBAC）结合使用，以控制谁可以访问集群内的特定资源。这允许管理员将用户权限限制为其角色所需的最小权限。

### 配额管理

管理员可以对命名空间应用资源配额，限制特定命名空间内资源可以消耗的资源量（CPU、内存等）。这有助于在多个用户或团队之间高效地管理和分配集群资源。

### 创建我们的第一个命名空间

要创建一个名为 **development** 的新命名空间，你可以使用以下命令：

```
$ kubectl create namespace development
```

要列出集群中的所有命名空间：

```
$ kubectl get namespaces
```

请记住，命名空间是 Kubernetes 管理的另一种资源，就像 Pod、节点或事件一样。同样重要的是要理解，Kubernetes 命名空间与我们在前面章节中看到的 Linux 命名空间不同。这两个概念之间唯一的共同点是它们共享相同的名称，“命名空间”。

部署资源时，你可以指定应在其创建的命名空间。例如，在 development 命名空间中创建一个 Pod：

```
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  namespace: development
spec:
  containers:
  - name: my-container
    image: nginx
```

或者你可以使用带有 `-n` 或 `--namespace` 标志的 kubectl 命令来指定命名空间：

```
$ kubectl apply -f my-pod.yaml -n development
```

## 为什么我们需要命名空间？

想象一个没有命名空间的世界。会发生什么？

### 集群混乱

没有命名空间，在同一集群内管理多个环境或项目将变得繁琐，可能导致潜在的名称冲突以及隔离资源和应用特定策略的困难。

### 访问控制挑战

没有命名空间，实施细粒度的访问控制将具有挑战性，因为 RBAC 策略通常应用于命名空间级别。

### 资源配额复杂性

应用资源配额的灵活性会降低，因为管理员必须将其应用于整个集群，而不是特定的命名空间，这使得更难以有效地管理资源分配。

本质上，命名空间对于在 Kubernetes 集群中组织资源、管理访问以及应用策略和配额至关重要。它们提供了高效运行多租户环境所需的必要隔离和管理能力。

### 默认命名空间

“但是等等！到目前为止，我部署所有这些 Pod 时从未指定过任何命名空间”，你可能会问。别担心，Kubernetes 为你提供了默认命名空间。

在 Kubernetes 中，**default** 命名空间是一个在集群设置期间自动创建的预定义命名空间。它旨在容纳未分配给任何其他命名空间的资源（如 Pod、服务、部署等）。当你使用 kubectl 或其他 API 客户端与 Kubernetes 集群交互而未指定命名空间时，你的操作将指向 **default** 命名空间。

### 默认命名空间的特性和用例

1.  **Kubernetes 资源的隐式命名空间**

    如果你在创建资源时未指定命名空间，它将被放置在默认命名空间中。这使得快速部署或 Kubernetes 初学者使用起来很方便，因为它减少了为简单应用程序或学习环境管理多个命名空间的需要。

2.  **简单性和便利性**

    默认命名空间简化了小规模部署或不需要命名空间细分的场景的操作。它允许用户部署和管理资源，而无需创建和指定额外的命名空间。

3.  **访问和管理**

    就像其他命名空间中的资源一样，默认命名空间中的资源可以使用 kubectl 命令进行管理。例如，要列出默认命名空间中的所有 Pod，你可以使用 `kubectl get pods`。要与其他命名空间中的资源交互，你需要在 kubectl 命令中使用 **-n** 或 **--namespace** 标志。

## 注意事项

1.  **命名空间隔离**

    虽然默认命名空间很方便，但它缺乏自定义命名空间提供的隔离。在生产环境或多租户集群中，最佳实践是为不同的应用程序、团队或开发阶段创建专用命名空间。这有助于组织资源、应用特定的访问控制和管理配额。

2.  **资源配额和限制**

    与任何其他命名空间一样，可以对默认命名空间应用资源配额和限制。但是，为了更好的资源管理和隔离，建议使用具有特定配额的自定义命名空间。

3.  **安全性**

    对所有部署使用默认命名空间可能会增加意外暴露或配置错误的风险，尤其是在多个用户或应用程序共享的集群中。定义明确的命名空间有助于应用细粒度的安全策略和访问控制。

总之，Kubernetes 中的默认命名空间提供了一种直接的方式来开始部署和管理资源。但是，为了增强组织性、隔离性和安全性，特别是在较大或共享的环境中，建议使用根据特定需求和上下文量身定制的自定义命名空间。

## 练习 15：在新命名空间中部署 nginx Pod

创建一个新命名空间 **python-dev** 并在该命名空间中部署一个 nginx Pod。尝试两种变体：

- a. 在命令行中指定命名空间。
- b. 在 Pod YAML 中指定命名空间。

现在，尝试部署一个 Pod，在 YAML 中指定了 **python-dev** 命名空间，而在命令行中指定了不同的命名空间（**development**）。看看会发生什么。

## 练习 16：删除命名空间

删除上面创建的 **python-dev** 命名空间。找出命名空间被删除后，在该命名空间中部署的 nginx Pod 发生了什么。

## Pod 网络

Kubernetes 中的 Pod 网络旨在允许 Pod 以一致且可预测的方式相互通信以及与外部世界通信。Kubernetes 施加了一组基本的网络要求，以确保集群内的无缝交互：

1.  所有 Pod 可以在不使用 NAT（网络地址转换）的情况下与所有其他 Pod 通信。
2.  所有节点可以在不使用 NAT 的情况下与所有 Pod 通信。
3.  Pod 自身看到的 IP 地址与其他人看到的 IP 地址相同。

为了满足这些要求，Kubernetes 采用扁平网络模型，其中每个 Pod 被分配一个唯一的 IP 地址。这种设计简化了网络配置，并允许 Pod 之间直接通信，无论它们位于哪个主机节点。以下是 Pod 网络如何实现的更详细说明：

### CNI（容器网络接口）

Kubernetes 使用 CNI 插件来设置 Pod 网络。当创建 Pod 时，Kubernetes kubelet 调用配置的 CNI 插件为 Pod 分配 IP 地址并设置网络路由，以确保符合 Kubernetes 网络模型。有许多可用的 CNI 插件，例如 Calico、Flannel、Weave Net 和 Cilium，每个插件都提供不同的功能和能力。

### Pod 到 Pod 通信

由于每个 Pod 在集群网络中获得一个唯一的 IP 地址，Pod 可以使用此 IP 地址直接相互通信。CNI 插件设置的路由规则确保从一个 Pod 发送到另一个 Pod 的数据包到达其目的地，无论 Pod 运行在哪个节点上。

### 节点到 Pod 通信

节点可以使用 Pod 的 IP 地址与 Pod 通信。这通过将目标为 Pod IP 的流量定向到适当节点的路由规则来实现。一旦流量到达节点，额外的规则会将流量路由到正确的 Pod。

### 外部到 Pod 通信

为了允许从集群外部到 Pod 的通信，Kubernetes 提供了一种称为服务的资源。我们将在下一章更详细地介绍服务。

## Pod到外部的通信

Pod可以通过其运行所在节点的路由与外部实体通信。可以配置出站规则，以允许连接到互联网或其他网络的出站连接。

## 网络策略

Kubernetes允许管理员使用网络策略在IP地址或端口级别控制流量。这提供了一种根据特定要求限制Pod进出连接的方法。这是一个高级概念，我们不会在本书中深入探讨，但了解它很有好处。

## Pod网络示例

为了通过一个涉及Nginx Pod和curl Pod的示例来演示Kubernetes中的Pod网络，我们将逐步设置这两个Pod，确保curl Pod能够使用Kubernetes网络与Nginx Pod通信。假设你已经运行了nginx Pod，让我们获取nginx Pod的IP地址。

```
$ kubectl get pod -o wide
NAME           READY   STATUS             RESTARTS   AGE   IP            NODE                            NOMINATED NODE   READINESS GATES
bash-sleep     0/1     CrashLoopBackOff   296        25h   10.244.1.126  first-cluster-default-pool-oeo85   <none>           <none>
nginx          1/1     Running            0          29h   10.244.0.146  first-cluster-default-pool-oeo8s   <none>           <none>
nginx-201      1/1     Running            0          20h   10.244.0.204  first-cluster-default-pool-oeo8s   <none>           <none>
two-containers     1/2     CrashLoopBackOff   286 (4m3s ago)   24h   10.244.0.117   first-cluster-default-pool-oeo8k   <none>           <none>
two-containers-better   2/2     Running            0                24h   10.244.0.178   first-cluster-default-pool-oeo8s   <none>           <none>
```

此命令显示更多信息，例如IP地址、Pod运行所在的节点等。从该命令的输出中我们可以看到，nginx节点的IP是**10.244.0.146**。

我们现在将部署一个curl Pod。

```
apiVersion: v1
kind: Pod
metadata:
  name: curl-pod
spec:
  containers:
  - name: curl-container
    image: curlimages/curl # (1)
    command: [ "sleep", "infinity" ] # (2)
```

(1) -> **curlimages/curl**是一个公共容器镜像，自带curl。

(2) -> 上述镜像没有默认命令。因此，如果我们尝试运行它，Pod将会崩溃。为了保持运行，我们执行**sleep infinity**。

让我们创建Pod并进入Pod内部。

```
$ kubectl apply -f curl-pod.yaml
pod/curl-pod created
```

```
# 一分钟后 ...
```

```
$ kubectl exec -it curl-pod -- sh # (1)
~ $ curl http://10.244.0.146 # (2)
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
<style>
html { color-scheme: light dark; }
body { width: 35em; margin: 0 auto;
font-family: Tahoma, Verdana, Arial, sans-serif; }
</style>
</head>
<body>
<h1>Welcome to nginx!</h1>
<p>If you see this page, the nginx web server is successfully installed and
working. Further configuration is required.</p>
<p>For online documentation and support please refer to
<a href="http://nginx.org/">nginx.org</a>.<br/>
Commercial support is available at
<a href="http://nginx.com/">nginx.com</a>.</p>

<p><em>Thank you for using nginx.</em></p>
</body>
</html>
~ $
```

(1) -> 我们使用sh进入Pod。（这个curl镜像没有bash）。

(2) -> 我们使用nginx Pod的IP地址运行了curl命令，并得到了默认的nginx索引页。

## 练习17

进行相同的curl练习，尝试从twin-nginx Pod访问两个容器。每个容器的IP地址会不同，还是相同？

## 部署Flask应用程序

至此，你已经掌握了足够的Kubernetes知识，可以在Kubernetes中部署一个简单的Flask应用程序，而这正是我们现在要做的。还记得我们在前面章节构建的容器镜像**flask-app**吗？我们将使用该镜像并部署一个Pod。

我们构建了容器镜像并将其推送到ghcr注册表。让我们构建一个Pod清单。

```
apiVersion: v1
kind: Pod
metadata:
  name: simple-flask
  labels:
    app: simple-flask # (1)
spec:
  containers:
  - name: flask # (2)
    image: ghcr.io/badri/flask-app:1.0 # (3)
    env:
    - name: PORT # (4)
      value: "8080"
    ports:
    - containerPort: 8080 # (5)
```

(1) -> Pod的标签。我们在这里发挥了一些创意自由。

(2) -> 容器的名称。我们将只运行1个容器。

(3) -> 带标签的镜像注册表名称。即使你是随意观察，你也会注意到我们之前所有的Pod都没有使用这种镜像模式。这是Kubernetes世界中的一个约定。镜像的语法是：

```
<registry-url>:<optional-port>/<image-namespace>/<image-name>:<optional-tag>
```

让我们分解一下。

**registry-url**表示存储容器镜像的注册表URL。如果未给出，则假定为Dockerhub注册表URL，**docker.io**。

**optional-port**有时，注册表在不同的端口上运行，例如**5000**。如果是这种情况，那么我们需要明确指定端口。否则，假定为**443**。

**image-namespace**是在容器注册表中组织多个容器镜像的一种方式。可以将其视为操作系统中的文件夹或目录。如果你的注册表被多个团队使用（通常如此），那么会有一个**team-a**镜像命名空间和一个**team-b**镜像命名空间。**注意**，不要将其与kubernetes命名空间混淆，那是完全不同的东西。对于curl Pod，**curlimages**是镜像命名空间。对于nginx，这将是**library**，因为它属于Dockerhub官方常用容器镜像库。

**image-name**镜像名称。无需多言！

**optional-tag**镜像的标签。对于我们的flask应用，它是**1.0**。如果未指定标签，则会拉取latest标签。我们稍后将了解为什么指定标签是不好的做法。

几个例子来强化这个约定。

**nginx**翻译为**docker.io:443/library/nginx:latest**。

**curlimages/curl**翻译为**docker.io:443/curlimages/curl:latest**。

现在让我们部署这个Pod。

```
$ kubectl apply -f simple-flask-pod.yaml
pod/simple-flask created
```

```
# 几秒钟后 ...
```

```
$ kubectl get pod simple-flask
NAME            READY   STATUS             RESTARTS   AGE
simple-flask    0/1     ImagePullBackOff   0          27s
```

我们遇到了一个新错误，**ImagePullBackOff**。让我们深入挖掘，找出这意味着什么。

```
$ kubectl describe pod simple-flask
...
Events:
  Type    Reason     Age                From               Message
  ----    ------     ----               ----               -------
  Normal  Scheduled  49s                default-scheduler  Successfully assigned default/simple-flask to first-cluster-default-pool-oeo85
  Normal  BackOff    21s (x2 over 47s)  kubelet            Back-off pulling image "ghcr.io/badri/flask-app:1.0"
  Warning Failed     21s (x2 over 47s)  kubelet            Error: ImagePullBackOff
  Normal  Pulling    9s (x3 over 48s)   kubelet            Pulling image "ghcr.io/badri/flask-app:1.0"
  Warning Failed     9s (x3 over 48s)   kubelet            Failed to pull image "ghcr.io/badri/flask-app:1.0": failed to pull and unpack image "ghcr.io/badri/flask-app:1.0": failed to resolve reference "ghcr.io/badri/flask-app:1.0": failed to authorize: failed to fetch anonymous token: unexpected status from GET request to https://ghcr.io/token?scope=repository%3Abadri%2Fflask-app%3Apull&service=ghcr.io: 401 Unauthorized
  Warning Failed     9s (x3 over 48s)   kubelet            Error: ErrImagePull
```

节点**first-cluster-default-pool-oeo85**中的kubelet尝试从ghcr的注册表中拉取**ghcr.io/badri/flask-app:1.0**镜像，但它从注册表收到了401未授权响应。因此，它失败了，并表示：“我无法启动Pod，因为我无法从你给我的注册表中拉取该镜像”。

当我们发布镜像到ghcr注册表时，我们将其作为私有镜像发布。因此，未经身份验证无法拉取和使用它。我们可以向kubelet提供身份验证凭据，或者我们可以将镜像设为公开。前者，我们将在下一章了解如何操作。现在，我们将把镜像设为公开。

访问**https://github.com/users/<your-user-name>/packages/container/package/flask-app/settings**，假设你将镜像发布为**flask-app**，向下滚动到“Danger Zone”，然后单击“Change visibility”。
将包的可见性更改为“Public”。
现在，让我们删除Pod并重新部署它。

```
$ kubectl delete pod simple-flask
pod "simple-flask" deleted

# 重新创建 ..

$ kubectl apply -f simple-flask.yaml
pod/simple-flask created
```

我们可以看到镜像现在被成功拉取了，因为它是一个公开镜像。

## 关于标签的说明

在容器化环境中，采用容器镜像标签的最佳实践可以显著提升软件部署流程的可靠性、可追溯性和安全性。以下是一些关键的容器镜像标签最佳实践：

1.  **使用语义化版本号。** 为标签应用语义化版本号（SemVer），使其具有信息性并能反映所包含的变更。语义化版本号遵循 **主版本号.次版本号.修订号** 格式，其中：
    *   当进行不兼容的 API 变更时递增 **主版本号**，以向后兼容的方式添加功能时递增 **次版本号**，进行向后兼容的错误修复时递增 **修订号**。
    *   此实践有助于用户和自动化系统理解新镜像版本中的变更范围。
2.  **包含具体版本号，避免使用** latest。虽然将镜像标记为 **latest** 很常见，但仅依赖此标签进行部署可能导致不可预测性，因为 **latest** 可能随时间指向不同的镜像版本。务必为镜像添加具体版本标签，同时保留 latest 标签，以确保可重现的部署和更轻松的回滚。在开发或快速测试中使用 latest，但在预发布和生产环境中指定确切版本。
3.  **使用短 Git SHA。** 在标签中包含短 Git SHA（提交哈希）可提供从镜像构建时追溯到源代码的能力。此实践增强了将镜像追溯到其代码库的能力，提高了可审计性和调试过程的效率。
4.  **包含构建时间戳。** 在镜像标签中添加时间戳（例如 **20231005-1345**）可以提供镜像构建时间的信息。这对于识别最新构建或将镜像部署与特定事件关联起来特别有用。
5.  **为不同环境使用描述性标签。** 对于针对特定环境（开发、预发布、生产）定制的镜像，在标签中包含环境信息（例如 1.2.3-staging）。这有助于快速识别每个环境的正确镜像，并防止意外的跨环境部署。
6.  **在 CI/CD 流水线中自动化标签。** 将标签过程作为持续集成/持续部署（CI/CD）流水线的一部分进行自动化。自动化可确保标签实践的一致性并减少人为错误。在 CI/CD 系统中实施规则，根据分支名称、提交 SHA、构建编号或与构建上下文相关的其他有意义的标识符为镜像打标签。
7.  **遵守不可变标签原则。** 一旦标签与镜像关联并推送到注册表，避免将该标签用于不同的镜像。不可变标签确保特定标签始终指向完全相同的镜像内容，这对于可靠的部署和安全审计至关重要。
8.  **定期清理旧标签。** 定期审查并清理容器注册表中的旧标签或未使用标签，以管理存储成本并减少杂乱。但是，请确保任何自动清理过程都尊重因合规性、回滚能力或历史原因而必须保留的标签。

通过遵循这些标签最佳实践，团队可以显著改善容器化应用程序的管理、部署和安全性，确保正确的版本部署在适当的环境中，并具有完整的可追溯性直至源代码。

推送镜像后，更新 Pod 中的镜像标签，并重新应用 Pod。

## 镜像摘要

注册表中的容器镜像可以通过标签或 SHA（安全哈希算法）摘要来标识。两者都用作标识符，但具有不同的特性和用途：

1.  **镜像标签**

    标签是附加到镜像版本的人类可读标签，例如 nginx:1.19，其中 nginx 是镜像名称，1.19 是表示版本的标签。标签是可变的，这意味着同一个标签可以随时间重新分配给不同的镜像（例如 latest 标签）。

    标签用于更方便地引用和管理镜像，允许开发者以人类友好的格式指定版本。它们在开发和部署脚本中特别有用，用于指定应使用哪个版本的镜像。

    标签的主要特性是其可变性；同一个标签在不同时间可以指向不同的镜像内容。这使得它们便于表示不断演变的状态（如最新版本），但如果管理不当，可能导致部署的不可预测性。

2.  **镜像 SHA（摘要）**

    SHA 摘要是镜像内容的不可变标识符，在镜像推送时由注册表生成。它是镜像内容的哈希值。例如，一个镜像可能具有如下 SHA 摘要：**sha256:45b23dee0a....**

    当你需要对镜像进行不可变引用时，可以使用 SHA。由于摘要是镜像内容的哈希值，它唯一且可靠地标识特定的内容集。使用 SHA 可确保始终使用镜像的确切版本，消除了“在我的机器上可以运行”的问题。

    SHA 的不可变性意味着一旦创建了镜像摘要，它就始终指向相同的镜像内容。这对于安全性、可审计性和可靠性至关重要，确保镜像未被更改或篡改。

    可以通过运行以下命令找到镜像 SHA 摘要。

    ```
    $ docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/badri/flask-app:1.0
    ghcr.io/badri/flask-app@sha256:eda44382a1cfee8810a0dd0f5452ae239b59a489dc9001081254ae97ee5ba3b7
    ```

3.  **关键区别**

    a. **可变性与不可变性**

    标签是可变的，可以从一个镜像移动到另一个镜像，而 SHA 是不可变的，永久链接到特定的镜像内容。

    b. **人类可读性**

    标签设计为人类可读且有意义，例如版本号或环境名称。SHA 是加密哈希值，对人类不友好，但提供更高程度的精确性和可靠性。

    c. **用途**

    标签适用于开发和更通用意义上的版本标识。SHA 用于安全性，确保使用的是预期的确切内容，没有意外更改的风险。

在实践中，标签和 SHA 在容器镜像管理中扮演互补的角色。标签为识别和处理镜像提供了便利性和可读性，而 SHA 提供了不可变的引用，以确保所使用镜像的完整性和可靠性。正确管理两者是构建健壮的容器化应用程序生命周期的关键。

## 练习 18：通过指定不同的标签部署 Pod

在 Pod YAML 清单中指定不同的标签来部署你的 flask/nginx 容器。

## 练习 19：通过指定镜像 SHA 摘要部署 Pod

在 Pod YAML 清单中通过指定镜像的 SHA 摘要来部署你的 Flask 容器。

## 镜像拉取策略

每次构建新镜像（即使使用相同的标签），只要任何层发生变化，都会生成一个唯一的 SHA 摘要。我们需要一种方式来控制 kubelet 何时为我们的 Pod 拉取镜像。这就引入了 **imagePullPolicy**。

Kubernetes 中 Pod 的 imagePullPolicy 决定了 kubelet 在启动容器之前如何从容器镜像仓库拉取镜像。此策略对于确保使用正确版本的镜像创建容器至关重要，它在利用缓存镜像以提高效率与拉取新镜像以获取更新或出于安全原因之间取得平衡。imagePullPolicy 有三个可能的值：

- Always
- IfNotPresent
- Never

## Always

使用 **imagePullPolicy: Always** 时，Kubernetes 每次启动或重启 Pod 时，都会尝试从容器镜像仓库拉取最新版本的镜像。这确保了你始终运行最新的镜像，但可能会增加启动时间和容器镜像仓库的负载。

## IfNotPresent

**imagePullPolicy: IfNotPresent** 告诉 Kubernetes 仅在节点上不存在该镜像时才从容器镜像仓库拉取。如果镜像已存在，Kubernetes 将使用缓存版本，从而加快 Pod 启动时间。此策略在效率和确保镜像相对最新之间取得了良好平衡。

## Never

使用 **imagePullPolicy: Never** 时，Kubernetes 永远不会尝试拉取镜像。它严格使用本地可用的镜像。如果节点上不存在该镜像，Pod 将无法启动。此策略很少使用，但在外部网络访问受限的严格控制环境中，或使用本地构建的镜像进行测试时可能很有用。

此属性可以赋予 Pod 中的每个容器条目。换句话说，你可以让 Pod 中的一个容器使用 **Always** 拉取策略，而另一个容器使用 **IfNotPresent**。

## 默认行为

默认的 imagePullPolicy 由镜像标签决定：

如果镜像标签是 **:latest**，Kubernetes 默认使用 **Always**。如果镜像标签是特定的且不是 **:latest**，Kubernetes 默认使用 **IfNotPresent**。

## 选择合适的 imagePullPolicy

在开发环境中，当你频繁推送和测试镜像更改时，使用 **Always**。

**IfNotPresent** 适用于生产环境，或当网络效率和速度至关重要且镜像更新受控时。

**Never** 主要用于使用本地镜像进行测试，或在具有严格网络限制的环境中。

理解和正确设置 imagePullPolicy 可确保你的部署运行预期版本的容器镜像，在稳定性、安全性和效率之间取得平衡。

## 集群中 Pod 的数量

可以在 Kubernetes 节点或集群中运行的 Pod 数量取决于多种因素，包括节点的资源（CPU、内存、存储）、集群的总体资源和配置，以及 Pod 本身的资源需求和限制。Kubernetes 对每个节点或集群的 Pod 数量没有固定限制；相反，它由你已有的资源和配置决定。

## 每个节点 Pod 数量的考虑因素

**节点资源：** 最基本的限制是节点上可用的 CPU 和内存资源量。每个 Pod 及其容器都会消耗资源，节点只能支持其资源容量范围内的 Pod。

**Pod 资源请求和限制：** 定义 Pod 时，你可以指定资源请求和限制。请求保证 Pod 获得最低资源量，这会影响调度决策。限制确保 Pod 不会消耗超过指定量的资源，防止 Pod 之间出现资源饥饿。

**Kubelet 设置：** Kubelet 有一个默认设置，用于每个节点的最大 Pod 数量，可以进行调整。例如，默认值可能设置为每个节点 110 个 Pod，但这可能因 Kubernetes 设置和版本而异。

**网络资源：** Pod 通过网络相互通信，网络资源和配置（如可用的 IP 地址数量）也可能限制每个节点的 Pod 数量。

## 每个集群 Pod 数量的考虑因素

**集群规模和资源：** 集群的总容量（所有节点资源的总和）最终决定了它可以运行多少 Pod。添加更多节点或使用具有更多资源的节点可以增加集群可以支持的 Pod 数量。

**Kubernetes Master 组件的可扩展性：** Master 组件（如 API 服务器、调度器和 etcd）的可扩展性也影响集群可以高效处理多少 Pod。大型集群可能需要对这些组件进行调优和优化。

**网络和服务限制：** Kubernetes 网络和集群中的服务数量可能带来限制。例如，最大服务数和端点数可能影响集群的可扩展性。

**Etcd 存储：** Kubernetes 使用 etcd 存储所有运行时数据。随着对象（包括 Pod）数量的增长，etcd 数据库的大小和性能可能成为限制因素。

## 可扩展性指南

Kubernetes 官方提供了可扩展性指南，提供了集群在不同负载下性能的见解。在撰写本文时，Kubernetes 支持最多 5000 个节点和总共 150,000 个 Pod 的集群，但随着 Kubernetes 及其底层基础设施的改进，这些数字在不断演变。

## 总结

在本章中，我们通过探索 Pod——Kubernetes 中最小和最基本的可部署对象——深入 Kubernetes 编排能力的核心。Pod 作为 Kubernetes 应用程序的基础构建块，封装了一个或多个容器，这些容器共享存储、网络以及如何运行容器的规范。

下一章将从对 Pod 的基础理解过渡到探索 Deployment 和 Service，重点介绍 Kubernetes 如何大规模编排和管理容器化应用程序，确保它们在集群中的可用性和可访问性。

## 引入的关键概念

**Pod 基础知识：** 我们首先介绍了 Pod 是什么以及它们在 Kubernetes 中为何至关重要。与可能关注单个容器的传统容器部署不同，Kubernetes 使用 Pod 作为最小的可部署单元，从而实现复杂的管理、扩展和网络功能。

**Pod 生命周期和管理：** 理解 Pod 的生命周期对于在 Kubernetes 中有效管理应用程序至关重要。我们介绍了 Pod 如何被创建、调度、运行以及最终终止。本节还介绍了就绪探针和存活探针对于维护 Pod 健康的重要性。

**多容器 Pod：** 我们深入探讨了 Pod 可能包含多个容器的场景。这种设置允许将紧密耦合的应用程序组件打包在一起，共享资源和网络空间，从而促进相互通信并支持常见的模式，如 sidecar、适配器和 ambassador。

**Pod 中的网络：** 网络是 Pod 的一个关键方面，使它们能够相互通信并与外部世界通信。我们探讨了如何为每个 Pod 分配一个唯一的 IP 地址，以及如何使用 Service 为 Pod 组提供稳定的接口。

**最佳实践：** 本章最后介绍了使用 Pod 的最佳实践，包括为故障而设计、理解不可变性和无状态性的重要性，以及在 Pod 内进行调试和日志记录的策略。

## 第 6 章：超越 Pod

在为 Pod——Kubernetes 中的调度原子单元——打下坚实基础之后，我们现在深入 Kubernetes 生态系统。本章将展开 Kubernetes 的编排能力，向你介绍一系列资源，这些资源赋予了可扩展性、管理和服务发现能力。这些资源对于将独立的 Pod 转变为协调的、自愈的应用程序系统至关重要，体现了云原生原则的精髓。

**Deployment** 成为我们深入探讨 Pod 声明式管理的核心。你将了解 Deployment 如何抽象化滚动更新、管理副本和确保应用程序可用性的复杂性。我们将剖析 Deployment 的结构，说明它如何监督 Pod 的期望状态，从而自动化你的部署策略和扩展操作。

过渡到 **Service**，我们探索 Kubernetes 中网络通信的支柱。Service 通过提供一致的端点来访问一组 Pod 的功能，从而实现 Pod 之间的松耦合。无论你是路由内部流量还是将应用程序暴露给外部世界，Service 都能确保你的应用程序组件高效通信并保持可发现性。

在探索这些概念的同时，我们还触及了标签和选择器的重要性——Kubernetes 组织和选择对象子集的方式。理解标签和选择器至关重要，因为它们构成了将 Deployment 与 Pod 关联、将 Service 与它们路由流量的 Pod 关联的基础机制。

随后，叙述扩展到 Ingress，这是管理服务外部访问不可或缺的资源。Ingress 为服务提供 HTTP 和 HTTPS 路由，使你能够定义既安全又可扩展的应用程序入口点。你将学习如何配置 Ingress 规则以将应用程序暴露给外部世界、实现负载均衡、终止 SSL/TLS 以及提供基于名称的虚拟主机。

通过结合实际示例和真实场景，本章旨在让你掌握在 Kubernetes 上自信地设计、扩展和暴露应用程序的知识。从使用 Deployments 管理应用程序的多个副本，到通过 Ingress 使服务可访问，你将获得在 Kubernetes 上构建和部署弹性、可访问且可扩展应用程序所需的见解。

## ReplicaSets

Kubernetes 中的 ReplicaSets 是一种基本的工作负载 API 对象，它确保在任何给定时间都有指定数量的 Pod 副本在运行。它们提供了一种管理 Pod 生命周期的方式，确保基于模板的指定数量的 Pod 实例正在运行。ReplicaSets 主要用于维护一组相同 Pod 的可用性，使其在 Kubernetes 环境中对于容错和可扩展性至关重要。

## 为什么我们需要 ReplicaSets

**高可用性：** ReplicaSets 确保始终有指定数量的 Pod 副本在运行，即使某些 Pod 发生故障。这对于维护应用程序的可用性至关重要。

**负载分布：** 通过维护 Pod 的多个副本，ReplicaSets 允许将请求分布到各个副本上，从而提高应用程序的性能和可靠性。

**可扩展性：** ReplicaSets 使得通过简单地更改所需副本数量来轻松扩展应用程序成为可能，使应用程序能够高效处理不同的负载。

**自愈能力：** ReplicaSets 提供了一种自愈机制，通过自动替换因任何原因而失败、被删除或被终止的 Pod，确保应用程序继续按预期运行。

这是一个简单的 ReplicaSet 示例，它确保始终运行三个 Nginx 服务器副本：

```yaml
apiVersion: apps/v1 # (1)
kind: ReplicaSet # (2)
metadata:
  name: nginx-replicaset
spec: # (3)
  replicas: 3 # (4)
  selector: # (5)
    matchLabels:
      app: nginx
  template: # (6)
    metadata:
      labels:
        app: nginx
    spec:
      containers:
        - name: nginx
          image: nginx:latest
```

(1) -> 指定 API 版本；apps/v1 是 ReplicaSets 的版本。

(2) -> 指定 Kubernetes 对象的类型；在本例中是 ReplicaSet。

(3) -> 提供关于 ReplicaSet 的元数据，包括其名称。

(4) -> 定义 ReplicaSet 的期望状态。

(5) -> 指定期望的 Pod 副本数量。

(6) -> 定义 ReplicaSet 如何识别其管理的 Pod，使用标签选择正确的 Pod。

(7) -> 描述副本将基于的 Pod 模板，包括要使用的容器镜像（**nginx:latest**）和标签。

此 ReplicaSet 配置告诉 Kubernetes 维持三个 Nginx 服务器 Pod 的副本。如果任何 Pod 崩溃，ReplicaSet 将自动创建一个新的 Pod 来替换它，从而维持三个副本的期望状态。

## Deployments

Kubernetes 中的 Deployments 提供了一种声明式的方式来管理 Pod 和 ReplicaSets 的生命周期。它们允许你在 Deployment 配置中描述应用程序的期望状态，然后 Kubernetes 会努力维持该状态。Deployments 处理 Pod 的滚动更新，使你能够轻松更新在 Pod 中运行的应用程序版本、回滚到先前版本、向上或向下扩展以及轻松管理应用程序的状态。

## 为什么我们需要 Deployments

**自动推出和回滚：** Deployments 自动化了更新在 Pod 中运行的应用程序的过程，包括滚动更新（确保更新期间零停机）以及在出现问题时回滚到先前版本的能力。

**扩展：** Deployments 使得根据需求轻松扩展应用程序成为可能，只需更改 Deployment 配置中的副本数量即可。

**自愈能力：** 与 ReplicaSets 类似，Deployments 确保始终有指定数量的 Pod 在运行，即使某些 Pod 发生故障。它们提供了更高层次的抽象，包括 ReplicaSets 的自愈机制。

**Pod 版本管理：** Deployments 管理不同版本的 Pod，使得更新容器镜像、配置或资源限制变得容易，而无需手动删除和重新创建 Pod。

让我们创建一个包含 3 个 nginx Pod 的 Deployment。

```yaml
apiVersion: apps/v1 # (1)
kind: Deployment # (2)
metadata:
  name: nginx-deployment
spec:
  replicas: 3 # (3)
  selector:
    matchLabels:
      app: nginx
  template: # (4)
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.16.1
        ports:
        - containerPort: 80
```

(1) -> 指定 Deployments 的 API 版本。

(2) -> 指定 Kubernetes 资源类型，在本例中是 Deployment。

(3) -> spec 部分详细说明了 Deployment 的期望状态。

replicas: 期望的 Pod 副本数量。selector: 定义 Deployment 如何识别其管理的 Pod。它应与模板中描述的 Pod 的标签匹配。

(4) -> 用于创建副本的 Pod 模板。这包括元数据（如标签）和 spec（如容器镜像和端口）。

此 Deployment 确保始终运行三个 Nginx 服务器（版本 1.16.1）的副本。如果 Pod 失败，Deployment 将自动创建一个新的 Pod 来替换它。你可以更新 Deployment 以推出应用程序的新版本、扩展副本数量，并在必要时回滚到先前版本。

Kubernetes 中的 Deployments 提供了一种强大且灵活的方式来管理你的应用程序。它们抽象了管理单个 Pod 和 ReplicaSets 的复杂性，为部署、扩展和维护应用程序提供了更简单、更健壮的机制。Deployments 对于任何希望利用 Kubernetes 运行生产级应用程序的人来说都是必不可少的，因为它们具有自动化管理功能，包括无缝的推出和回滚、扩展以及自愈能力。

## ReplicaSets 和 Deployments 之间的区别

Kubernetes 中 ReplicaSet 和 Deployment 之间的主要区别在于它们的目的和功能。虽然两者都确保有指定数量的 Pod 副本在运行，但 Deployments 提供了额外的功能，例如更新这些 Pod 以及更全面地管理其生命周期的能力。

**ReplicaSet** 确保在任何给定时间都有指定数量的 Pod 副本在运行。它主要用于维护一组相同 Pod 的可用性。它可以根据需要创建和销毁 Pod 以达到所需的副本计数。它监视与其标准匹配的 Pod，并在数量未达到时采取行动。但是，ReplicaSet 不提供使用新版本容器镜像更新 Pod 的内置机制。要更新由 ReplicaSet 管理的 Pod，你需要手动删除 Pod，以便 ReplicaSet 使用新镜像重新创建它们，或者使用 ReplicaSet 本身之外的其他机制。

**Deployment** 为 Pod 和 ReplicaSets 提供声明式更新。Deployments 管理一组 Pod 的部署和扩展，并允许对 Pod 及其配置进行声明式更新。Deployment 包含 ReplicaSet 的所有功能（管理副本），并增加了对滚动更新、回滚和版本控制的支持。Deployments 保留更新历史记录，允许在需要时轻松回滚到先前版本。Deployments 非常适合管理需要轻松扩展、滚动更新和回滚功能的无状态应用程序。

## 你应该使用哪一个？

**对于简单的 Pod 副本管理：** 如果你的唯一要求是确保一定数量的 Pod 副本在运行，并且你不需要在创建后更新这些 Pod，那么 ReplicaSet 可能就足够了。然而，这种用例在实践中相对较少，因为大多数应用程序在部署后都需要更新。

**对于通用的应用程序部署和管理：** 在大多数情况下，Deployments 是推荐的选择。Deployments 提供了 ReplicaSet 功能的超集，包括管理副本以及无缝更新和回滚的额外好处。它们提供了更高层次的抽象，简化了应用程序管理的许多方面，使 Deployments 适用于 Kubernetes 中的大多数应用程序部署需求。

总之，虽然 ReplicaSets 对于维护特定数量的副本很有用，但由于其全面的功能集涵盖了 Pod 和 ReplicaSets 的生命周期管理，Deployments 通常是 Kubernetes 中应用程序部署和管理的首选。

请**注意**，Kubernetes 中的 Deployment 会自动创建一个 ReplicaSet 来管理 Pod。Deployment 指定了应用程序的期望状态，例如副本数量，而 ReplicaSet 则确保在任何给定时间，基于 Deployment 中的 Pod 模板，指定数量的 Pod 都在运行。当你创建一个 Deployment 时，它会创建一个 ReplicaSet 来编排 Pod 实例。如果你更新 Deployment（例如，更改 Pod 模板），它将为新版本的 Pod 创建一个新的 ReplicaSet，并根据 Deployment 的更新策略（例如滚动更新）管理从旧版本到新版本的过渡。

Deployment 和 ReplicaSet 之间的关系简化了部署、扩展和管理无状态应用程序的过程。它提供了滚动更新、回滚、扩展和 Pod 自愈的机制，使得 Deployment 成为比 ReplicaSet 更高级别的概念，而 ReplicaSet 则更专注于维护特定数量的 Pod 副本。

## 为什么 ReplicaSet 仍然存在？

尽管 Deployment 提供了其功能的超集，并附带了滚动更新和回滚等额外功能，但 ReplicaSet 仍然是 Kubernetes 生态系统的基础部分，原因如下。以下是 ReplicaSet 保持重要性和使用率的原因：

- 1. **基础构建块：** ReplicaSet 作为 Kubernetes 中的基础构建块。它们提供了一种简单、专注的机制，确保在任何给定时间都有指定数量的 Pod 副本在运行。这是维护应用程序可用性和扩展性的关键功能。
- 2. **Deployment 的抽象层：** Deployment 通过在底层使用 ReplicaSet 来管理应用程序的部署和更新。当你创建一个 Deployment 时，它会创建一个 ReplicaSet 来管理 Pod。这种关注点分离允许 Deployment 抽象掉 Pod 生命周期管理和更新策略的复杂性，同时仍然利用 ReplicaSet 维护所需副本数量的能力。
- 3. **直接使用场景：** 在用户需要简单的复制控制器而不需要滚动更新或版本历史记录的场景中，可以直接使用 ReplicaSet。它们更简单，对于某些需要固定副本数量但不需要 Deployment 提供的额外功能的无状态应用程序可能更合适。
- 4. **学习和调试：** 理解 ReplicaSet 对于学习 Kubernetes 内部工作原理至关重要。它们通常用于教育目的，以教授 Pod 复制和扩展的基础知识。此外，在调试时，理解 Deployment 及其底层 ReplicaSet 之间的关系可能很有价值。
- 5. **API 稳定性和演进：** Kubernetes 旨在保持 API 稳定性和向后兼容性。因此，一旦像 ReplicaSet 这样的资源类型被引入并广泛使用，它就会保留在生态系统中，以确保现有应用程序和系统继续运行。这种稳定性对于依赖特定 Kubernetes 资源的企业环境和生产应用程序至关重要。
- 6. **高级用例的灵活性：** 虽然 Deployment 涵盖了管理无状态应用程序的大多数用例，但在某些高级场景中，可能需要 ReplicaSet 提供的细粒度控制。例如，用户可能希望自行管理滚动更新逻辑，或实现 Deployment API 未直接支持的自定义部署策略。

总之，ReplicaSet 因其简单性、作为 Deployment 构建块的角色，以及在需要更专注于副本管理的特定场景中的使用，仍然是 Kubernetes 的关键部分。它们的持续存在确保了 Kubernetes 生态系统内的灵活性、稳定性和向后兼容性。

## Deployment 策略

Kubernetes 中的 Deployment 策略决定了如何将应用程序的更新推出，以确保可用性并最小化中断。两种主要策略是滚动更新和重建，但也可以使用 Kubernetes 功能实现其他高级策略，如蓝/绿部署和金丝雀发布。以下是每种策略的工作原理，以及使用 Nginx Pod 的示例：

### 滚动更新（默认策略）

滚动更新逐步用新版本的 Pod 替换旧版本，确保应用程序在更新期间保持可用，并且不超过可用资源。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  strategy:
    type: RollingUpdate # (1)
    rollingUpdate: # (2)
      maxUnavailable: 1 # (3)
      maxSurge: 1 # (4)
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.24
        ports:
        - containerPort: 80
```

(1) -> 部署策略的类型。

(2) -> 我们为滚动更新策略提供特定配置。

(3) -> 在更新过程中可以不可用的 Pod 数量。

(4) -> 在更新期间可以创建的超出期望 Pod 数量的 Pod 数量。

**maxSurge** 和 **maxUnavailable** 都可以指定为整数（在我们的例子中是 **1**）或百分比（例如 25%），并且它们不能同时为零。当指定为整数时，它表示实际的 Pod 数量；当指定百分比时，使用期望 Pod 数量的该百分比，向下取整。

需要注意的是，maxSurge 表示在部署时将创建的新 Pod 的最大数量，而 maxUnavailable 是在部署期间将删除的旧 Pod 的最大数量。

例如，如果我们使用更新的 nginx 镜像（**nginx:1.25**）重新应用部署，kubelet 将为部署创建最多 1 个使用 **1.25** 镜像的新 Pod，并且在**部署期间**的任何给定时间点，最多删除 1 个使用 **1.24** 标签的 Pod。

让我们检查发生的一系列事件来确认这一点。

```bash
$ kubectl apply -f deployment-1.yaml
deployment.apps/nginx-deployment created

# 几秒钟后...

$ kubectl get pod -l app=nginx
NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-5948f7484f-7wtj7   1/1     Running   0          29s
nginx-deployment-5948f7484f-88kkp  1/1     Running   0          29s
nginx-deployment-5948f7484f-9m2fx  1/1     Running   0          29s
```

```bash
$ kubectl apply -f deployment-2.yaml
deployment.apps/nginx-deployment configured
```

# 几秒钟后 ..

```bash
$ kubectl get pod -l app=nginx
NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-cd5968d5b-btjcg   1/1     Running   0          28s
nginx-deployment-cd5968d5b-x7pdb   1/1     Running   0          30s
nginx-deployment-cd5968d5b-xmjw2   1/1     Running   0          30s
```

```bash
$ kubectl get events --sort-by='.lastTimestamp'
...
4m9s        Normal  ScalingReplicaSet  deployment/nginx-deployment              Scaled up replica set nginx-deployment-cd5968d5b to 2 from 1 # (A)
4m9s        Normal  Pulling            pod/nginx-deployment-cd5968d5b-x7pdb    Pulling image "nginx:1.25"
4m9s        Normal  SuccessfulDelete   replicaset/nginx-deployment-5948f7484f  Deleted pod: nginx-deployment-5948f7484f-88kkp # (B)
4m9s        Normal  SuccessfulCreate   replicaset/nginx-deployment-cd5968d5b   Created pod: nginx-deployment-cd5968d5b-xmjw2
4m9s        Normal  Scheduled          pod/nginx-deployment-cd5968d5b-xmjw2   Successfully assigned default/nginx-deployment-cd5968d5b-xmjw2 to first-cluster-default-pool-oeo85
4m9s    Normal  Killing     pod/nginx-deployment-5948f7484f-88kkp  Stopping container nginx
4m9s    Normal  SuccessfulCreate replicaset/nginx-deployment-cd5968d5b  Created pod: nginx-deployment-cd5968d5b-x7pdb
4m9s    Normal  ScalingReplicaSet deployment/nginx-deployment    Scaled down replica set nginx-deployment-5948f7484f to 2 from 3
4m9s    Normal  ScalingReplicaSet deployment/nginx-deployment    Scaled up replica set nginx-deployment-cd5968d5b to 1
4m9s    Normal  Scheduled     pod/nginx-deployment-cd5968d5b-x7pdb  Successfully assigned default/nginx-deployment-cd5968d5b-x7pdb to first-cluster-default-pool-oeo8k
4m8s    Normal  Pulling     pod/nginx-deployment-cd5968d5b-xmjw2  Pulling image "nginx:1.25"
4m7s    Normal  ScalingReplicaSet deployment/nginx-deployment    Scaled down replica set nginx-deployment-5948f7484f to 1 from 2
4m7s    Normal  Pulled      pod/nginx-deployment-cd5968d5b-x7pdb  Successfully pulled image "nginx:1.25" in 1.677s (1.677s including waiting)
4m7s    Normal  Created     pod/nginx-deployment-cd5968d5b-x7pdb  Created container nginx
4m7s    Normal  Started     pod/nginx-deployment-cd5968d5b-x7pdb  Started container nginx
4m7s    Normal  Scheduled     pod/nginx-deployment-cd5968d5b-btjcg  Successfully assigned default/nginx-deployment-cd5968d5b-btjcg to first-cluster-default-pool-oeo8s
4m7s    Normal  ScalingReplicaSet deployment/nginx-deployment    Scaled up replica set nginx-deployment-cd5968d5b to 3 from 2
4m7s    Normal  SuccessfulDelete replicaset/nginx-deployment-5948f7484f  Deleted pod: nginx-deployment-5948f7484f-7wtj7
```

## 关于部署 Pod 名称的说明

在 Kubernetes 中，当 Deployment 创建 Pod 时，这些 Pod 的名称是通过在管理这些 Pod 的 ReplicaSet 名称后附加一个唯一标识符来自动生成的。这个唯一标识符由一个随机字符串组成，其中包含十六进制字符，确保同一 ReplicaSet 内的每个 Pod 名称都是唯一的。

以下是 Pod 名称的构建方式分解：

**Deployment 名称：** 你为 Deployment 指定的基础名称。

**ReplicaSet 后缀：** 当 Deployment 创建 ReplicaSet 时，它会在 Deployment 名称后附加一个唯一的哈希值来形成 ReplicaSet 的名称。这个哈希值是 Deployment Pod 模板的确定性函数（意味着它仅在 Pod 模板更改时才会更改）。

**Pod 后缀：** ReplicaSet 创建的每个 Pod 都会在 ReplicaSet 名称后附加一个进一步的唯一字母数字字符串，确保每个 Pod 的名称都是唯一的。这个后缀通常包含十六进制字符。

例如，如果你有一个名为 `nginx-deployment` 的 Deployment，它可能会创建一个名为 **nginx-deployment-75675f5897** 的 ReplicaSet（其中 **75675f5897** 是基于 Pod 模板的哈希值）。此 ReplicaSet 管理的每个 Pod 可能具有类似 **nginx-deployment-75675f5897-qwerty** 和 **nginx-deployment-75675f5897-asdfgh** 的名称，其中 **qwerty** 和 **asdfgh** 是包含十六进制字符的唯一字母数字字符串。

Pod 名称中的十六进制字符（唯一哈希和唯一字母数字字符串的一部分）代表十六进制数字系统（基数为 16）中的值，由数字 0 到 9 和字母 a 到 f（或 A 到 F）组成。这些字符是用于以紧凑且人类可读的形式表示二进制数据的编码的一部分。

值得注意的是，Kubernetes 用于生成这些哈希值和唯一字符串的确切方法可能会随时间演变，但使用基础 Deployment 名称，后跟 ReplicaSet 哈希值，再后跟唯一 Pod 标识符的原则，仍然是确保 Deployment 内 Pod 唯一性和可追溯性的一致方法。

## Recreate 策略

Recreate 策略在创建新 Pod 之前终止所有旧版本的 Pod，这会导致停机时间，但确保不会同时运行两个版本。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  strategy:
    type: Recreate # (1)
  selector:
    matchLabels:
    app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.25.1
        ports:
        - containerPort: 80
```

(1) -> 此配置确保在创建任何新 Pod 之前删除所有旧 Pod。

## 练习 20：使用 recreate 策略为 nginx 创建一个部署。

观察事件并确认所有旧 Pod 同时被删除，然后新的 Pod 在之后再次同时重新创建。

## 蓝/绿部署

蓝/绿部署涉及同时运行应用程序的两个版本（“蓝色”和“绿色”），然后在绿色版本经过充分测试并准备就绪后，将流量从蓝色切换到绿色。

Kubernetes 不提供内置的蓝/绿部署策略，但你可以使用服务和标签来实现它。我们将在服务部分了解如何实现这一点。

## 金丝雀部署

金丝雀部署在将更改推广到整个基础设施之前，先将其推广给一小部分用户。

我们在稳定版本旁边部署一个金丝雀版本，并将一部分流量路由到它。这可以通过调整入口来将一小部分流量路由到金丝雀版本来实现。我们将在处理入口时查看这一点。

## 练习 21：将 flask 应用部署为一个部署

在新的命名空间 **dev** 中，将简单的 flask 应用部署为一个具有 4 个副本的部署。删除 1 或 2 个 Pod 并观察发生了什么。

## 服务

Kubernetes 中的服务是一种将运行在一组 Pod 上的应用程序作为网络服务暴露的抽象方式。使用 Kubernetes，你无需修改应用程序以使用不熟悉的服务发现机制。Kubernetes 为 Pod 提供自己的 IP 地址和一组 Pod 的单一 DNS 名称，并且可以跨它们进行负载均衡。服务资源跨一组 Pod 路由流量。服务使用 YAML 或 JSON 定义，就像所有 Kubernetes 对象一样。

## 为什么我们需要服务

**用于 Pod 通信的稳定端点：** Pod 是临时性的，可以随时创建或销毁。服务提供一个稳定的 IP 地址和 DNS 名称，客户端可以使用它们与 Pod 进行通信，即使 Pod 被替换或重新调度也是如此。

**负载均衡：** 服务可以将网络流量分配到与指定标签选择器匹配的所有 Pod。这平衡了负载，并确保应用程序具有可扩展性和弹性。

**抽象：** 服务抽象了你访问应用程序的方式。你不需要知道在任何给定时间哪些特定的 Pod 正在提供应用程序服务。

**服务发现：** Kubernetes 服务集成到其 DNS 系统中，允许应用程序通过 DNS 查找轻松发现其他服务，使用服务名称。

## 服务类型

**ClusterIP（默认）：** 在集群内的内部 IP 上暴露服务。此类型使服务仅可从集群内部访问。

**NodePort：** 使用 NAT 在集群中每个选定节点的相同端口上暴露服务。它使服务可以使用 `<NodeIP>:<NodePort>` 从集群外部访问。

**LoadBalancer：** 使用云提供商的负载均衡器将服务暴露到外部。外部负载均衡器路由到的 NodePort 和 ClusterIP 服务会自动创建。

**ExternalName：** 通过返回一个 CNAME 记录及其值，将服务映射到 externalName 字段的内容（例如，`foo.bar.example.com`）。不设置任何类型的代理。

## CoreDNS 入门

CoreDNS 是 Kubernetes 中的默认 DNS 服务器，负责集群内基于 DNS 的服务发现。它在将服务名称解析为其对应的 IP 地址方面起着至关重要的作用，使 Pod 能够无缝地相互通信以及与服务通信。以下是 CoreDNS 在 Kubernetes 服务上下文中的工作方式：

当你在 Kubernetes 中创建一个服务时，它会被分配一个唯一的 DNS 名称。此名称遵循标准模式：`<service-name>.<namespace-name>.svc.cluster.local`，其中：

- `<service-name>` 是你的服务名称。
- `<namespace-name>` 是你的服务所在的命名空间。
- `svc.cluster.local` 是 Kubernetes 集群内服务的默认域。

CoreDNS 被配置为监视 Kubernetes API 中服务和端点的更改。当创建新的服务时，CoreDNS 会自动为其创建 DNS 记录。因此，当集群中的 Pod 通过其 DNS 名称查询服务时，CoreDNS 会将该名称解析为服务的 ClusterIP（对于 ClusterIP 服务）或适当的 IP 地址（对于 NodePort 或 LoadBalancer 服务）。

## 示例

一个 **ClusterIP** 服务使 Nginx 应用程序可以在集群内通过稳定的内部 IP 地址访问。这对于不应暴露在集群外部的内部应用程序或服务很有用。

## Cluster IP 服务

让我们为之前创建的 nginx 部署创建一个 ClusterIP 类型的服务作为前端。

```yaml
apiVersion: v1 # (1)
kind: Service # (2)
metadata:
  name: nginx-clusterip
spec:
  selector: # (3)
    app: nginx
  ports: # (4)
    - protocol: TCP
      port: 80
      targetPort: 80
  type: ClusterIP # (5)
```

(1) -> 指定用于创建 Service 的 Kubernetes API 版本。**v1** 版本表示这是一个稳定的资源定义版本。

(2) -> 定义要创建的 Kubernetes 资源类型。此处为 **Service**，用于将运行在一组 Pod 上的应用程序作为网络服务暴露出来。

(3) -> 用于选择此 Service 将定位并路由流量到的 Pod。Service 将管理所有标签与此选择器匹配的 Pod。**请注意**，这仅适用于与 Service 处于同一命名空间中的 Pod。

(4) -> 指定 Service 将监听的网络端口。每个条目包含端口规范，例如协议（TCP/UDP）、Service 上的端口号以及要转发到的 Pod 上的目标端口。

(5) -> 定义 Service 的类型。**ClusterIP** 是默认类型，它为 Service 提供一个稳定的内部 IP 地址，以便从集群内部访问。

```bash
$ kubectl apply -f svc-cluster-ip.yaml
service/nginx-clusterip created
```

```bash
$ kubectl get svc
NAME                TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
kubernetes          ClusterIP   10.245.0.1      <none>        443/TCP   3d7h
nginx-clusterip     ClusterIP   10.245.216.117  <none>        80/TCP    9s
```

我们可以看到 **nginx-clusterip** 被分配了一个新的集群 IP（10.245.216.117，你的可能不同）并暴露了 80 端口。让我们尝试从一个 curl Pod 访问它。

```bash
$ kubectl exec -it curl-pod -- sh
~ $ curl http://10.245.216.117
<!DOCTYPE html>
...
<p><em>Thank you for using nginx.</em></p>
</body>
</html>
```

通过 IP 访问服务并不是一个好的设计选择，因为在环境中 IP 会不断变化。为了解决这个问题，我们使用指定的 DNS 名称。我们之前讨论过，CoreDNS 会为每个服务分配一个内部 DNS 名称。根据 DNS 命名约定，此服务的 DNS 名称将是 **nginx-clusterip.default.svc.cluster.local**，或者如果你从同一命名空间内的 Pod 访问它，只需使用 **nginx-clusterip**。

让我们测试一下。

```bash
$ kubectl exec -it curl-pod -- sh # (1)
~ $ curl -I http://nginx-clusterip.default.svc.cluster.local # (2)
HTTP/1.1 200 OK
Server: nginx/1.25.4
Date: Thu, 22 Feb 2024 07:45:28 GMT
Content-Type: text/html
Content-Length: 615
Last-Modified: Wed, 14 Feb 2024 16:03:00 GMT
Connection: keep-alive
ETag: "65cce434-267"
Accept-Ranges: bytes

~ $ curl -I http://nginx-clusterip # (3)
HTTP/1.1 200 OK
Server: nginx/1.25.4
Date: Thu, 22 Feb 2024 07:45:34 GMT
Content-Type: text/html
Content-Length: 615
Last-Modified: Wed, 14 Feb 2024 16:03:00 GMT
Connection: keep-alive
ETag: "65cce434-267"
Accept-Ranges: bytes
```

```bash
~ $ curl -I http://non-existent-service # (4)
curl: (6) Could not resolve host: non-existent-service
```

- (1) -> 首先进入 curl Pod 内部。
- (2) -> 使用集群 FQDN（完全限定域名）进行 curl 访问。
- (3) -> 仅使用服务名称进行 curl 访问，因为 curl Pod 与服务处于同一命名空间。
- (4) -> 确保 curl 工作正常 :)

## 关于端口名称的说明

**Service** 或 **Pod** 定义中的端口规范可以为每个端口包含一个 name 参数。name 参数是一个描述性字符串，用于在规范中标识该端口。当单个 Service 或 Pod 暴露多个端口时，它特别有用，提供了一种通过名称（而不仅仅是数字）来引用每个端口的方式。

## 端口名称的目的

**清晰性和可读性：** 为端口命名使得更容易理解每个端口的用途，尤其是在处理多个端口时。例如，像 http 或 metrics 这样的名称可以指示特定端口的用途。

**在其他配置中引用：** 端口名称可用于在 Kubernetes 配置的其他部分中引用端口，例如在 Ingress 规则中或定义探针时。这允许配置在端口号更改时仍然有效。

## 带端口名称的示例

这是一个包含命名端口的 Service 定义示例：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app
  ports:
    - name: http
      protocol: TCP
      port: 80
      targetPort: http-port
    - name: https
      protocol: TCP
      port: 443
      targetPort: https-port
```

在此示例中，Service 暴露了两个端口：用于 HTTP 流量的端口 **80** 和用于 HTTPS 流量的端口 **443**。每个端口都有一个对应的名称（**http** 和 **https**），清楚地表明了每个端口的用途。

## 在 Pod 探针中使用端口名称

端口名称也可以在 Pod 规范中的存活和就绪探针中使用，通过名称引用端口。例如：

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: http-port
```

这里，存活探针使用名为 **http-port** 的端口向 **/healthz** 端点发送 HTTP GET 请求。

在 Service 或 Pod 定义中为端口包含 **name** 参数可增强清晰度，并允许更灵活的配置管理。它有助于确保你的配置更易于维护和理解，尤其是在具有多个端口的复杂应用程序中。

让我们创建另一种服务类型，**NodePort**。

NodePort 服务在集群的所有节点上的特定端口上暴露 Nginx 应用程序。这允许通过 **<NodeIP>:<NodePort>** 从外部访问该服务。

节点端口服务

以下是 NodePort 服务的一些主要用例：

- **用于开发和测试的外部访问：** NodePort 服务提供了一种从 Kubernetes 集群外部访问应用程序的简单方法，这对于开发和测试目的非常有用。

- **小型应用程序的公共暴露：** 对于小型或概念验证应用程序，NodePort 服务可以作为一种简单的方法，使应用程序可被外部用户访问，而无需外部负载均衡器或 Ingress 控制器。

- **LoadBalancer 服务的基础：** 在云环境中，NodePort 服务通常是 LoadBalancer 服务使用的底层机制。云提供商的负载均衡器针对所有节点上的 NodePort，将流量分发到应用程序。我们将在下一节中看到这种服务类型。

- **与现有基础设施集成：** 在已经存在外部负载均衡器或其他路由基础设施的场景中，NodePort 服务可用于通过将流量定向到指定节点端口的节点 IP 地址，将外部流量路由到 Kubernetes 集群。例如：可以通过创建 2 个不同的服务，将同一应用程序同时作为外部负载均衡器 IP 和节点端口暴露。

- **非 HTTP/TCP 应用程序：** 对于不使用 HTTP/TCP 协议因此无法通过 Ingress 轻松暴露的应用程序，NodePort 可以提供一种从 Kubernetes 集群外部直接访问服务的方式。

这是一个 NodePort 服务的示例。

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-nodeport
spec:
  selector:
    app: nginx
  ports:
    - port: 80
      nodePort: 30007 # (1)
  type: NodePort # (2)
```

(1) -> 要在哪个端口号上暴露服务。

(2) -> 类型应为 **NodePort**。

现在，让我们创建服务并尝试通过 curl 访问它，分别从 Pod 内部和外部世界进行访问。

```bash
$ kubectl apply -f svc-node-port.yaml
service/nginx-nodeport created
```

```bash
$ kubectl get svc
NAME               TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
kubernetes         ClusterIP  10.245.0.1      <none>        443/TCP          3d8h
nginx-clusterip    ClusterIP  10.245.216.117  <none>        80/TCP           36m
nginx-nodeport     NodePort   10.245.196.9    <none>        80:30007/TCP     17s
```

新服务具有不同的类型，请注意端口如何呈现 **X:Y** 模式，其中 **X** 是 Pod 暴露的端口，**Y** 是路由通过的节点端口。

需要指出的是，此服务还将被分配一个集群IP，通过该IP可以从集群内部访问它，并且会遵循相同的命名约定分配一个DNS名称。

让我们查看集群中所有3个节点的节点IP。

```
$ kubectl get nodes -o wide
NAME                           STATUS   ROLES    AGE   VERSION   INTERNAL-IP   EXTERNAL-IP   OS-IMAGE                       KERNEL-VERSION                CONTAINER-RUNTIME
first-cluster-default-pool-oeo85   Ready    <none>   3d8h   v1.29.1   10.122.0.6    139.59.1.188   Debian GNU/Linux 12 (bookworm)   6.1.0-17-amd64   containerd://1.6.28
first-cluster-default-pool-oeo8k   Ready    <none>   3d8h   v1.29.1   10.122.0.5    139.59.6.116   Debian GNU/Linux 12 (bookworm)   6.1.0-17-amd64   containerd://1.6.28
first-cluster-default-pool-oeo8s   Ready    <none>   3d8h   v1.29.1   10.122.0.4    139.59.2.214   Debian GNU/Linux 12 (bookworm)   6.1.0-17-amd64   containerd://1.6.28
```

我们更关注 **EXTERNAL-IP** 列，它代表集群中每个节点唯一的外部可访问IP。这些IP在你的环境中可能不同。

让我们从笔记本电脑（**而非**从Pod内部）通过这些IP访问我们的服务。

```
local-laptop $ curl -I http://139.59.1.188:30007
HTTP/1.1 200 OK
Server: nginx/1.25.4
Date: Thu, 22 Feb 2024 08:12:01 GMT
Content-Type: text/html
Content-Length: 615
Last-Modified: Wed, 14 Feb 2024 16:03:00 GMT
Connection: keep-alive
ETag: "65cce434-267"
Accept-Ranges: bytes
```

```
local-laptop $ curl -I http://139.59.6.116:30007
HTTP/1.1 200 OK
Server: nginx/1.25.4
Date: Thu, 22 Feb 2024 08:12:01 GMT
Content-Type: text/html
Content-Length: 615
Last-Modified: Wed, 14 Feb 2024 16:03:00 GMT
Connection: keep-alive
ETag: "65cce434-267"
Accept-Ranges: bytes
```

```
local-laptop $ curl -I http://139.59.2.214:30007
HTTP/1.1 200 OK
Server: nginx/1.25.4
Date: Thu, 22 Feb 2024 08:12:01 GMT
Content-Type: text/html
Content-Length: 615
Last-Modified: Wed, 14 Feb 2024 16:03:00 GMT
Connection: keep-alive
ETag: "65cce434-267"
Accept-Ranges: bytes
```

最后，让我们来看看 **LoadBalancer** 服务类型。

**LoadBalancer** 类型的服务通过云提供商的负载均衡器（在我们的例子中是 [DigitalOcean Load Balancer](https://www.digitalocean.com/)）将Nginx应用暴露到外部。云提供商会自动为服务分配一个外部IP地址，从而允许外部访问该应用。

![](img/91c4bde832e221ba5bc429caf9587286_202_0.png)

## 负载均衡器服务

```
apiVersion: v1
kind: Service
metadata:
  name: nginx-loadbalancer
spec:
  selector:
    app: nginx
  ports:
    - port: 80
  type: LoadBalancer # (1)
```

(1) -> 在内部，它通过 **nginx-loadbalancer.<namespace>.svc.cluster.local** 进行解析；在外部，则通过负载均衡器的IP或主机名进行访问。

让我们创建这个服务。

```
$ kubectl apply -f svc-loadbalancer.yaml
service/nginx-loadbalancer created
```

```
$ kubectl get svc
NAME                TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
kubernetes          ClusterIP   10.245.0.1      <none>        443/TCP   3d8h
nginx-clusterip     ClusterIP   10.245.216.117  <none>        80/TCP    51m
nginx-loadbalancer  LoadBalancer 10.245.178.209 <pending>     80:30919/TCP 9s
nginx-nodeport      NodePort    10.245.196.9    <none>        80:30007/TCP 15m
```

你可以看到，我们的LoadBalancer服务的 **EXTERNAL-IP** 列处于 **<pending>** 状态。云提供商（在我们的例子中是DigitalOcean）正在尝试创建一个负载均衡器云资源（不是Kubernetes资源）并将其分配给我们的服务。

```
$ kubectl describe svc nginx-loadbalancer
...

Events:
Type    Reason              Age   From               Message
----    ------              ----  ----               -------
Warning SyncLoadBalancerFailed  2m33s  service-controller  Error syncing load balancer: failed to ensure load balancer: load-balancer is currently being created
Warning SyncLoadBalancerFailed  2m33s  service-controller  Error syncing load balancer: failed to ensure load balancer: failed to update load-balancer with ID d581bcb5-6266-4b33-a17f-f59d2009636c: PUT https://api.digitalocean.com/v2/load_balancers/d581bcb5-6266-4b33-a17f-f59d2009636c: 403 (request "db402ac5-0e4f-459c-a263-33a785942c7d") Load Balancer can't be updated while it processes previous actions
Warning SyncLoadBalancerFailed  2m33s  service-controller  Error syncing load balancer: failed to ensure load balancer: failed to update load-balancer with ID d581bcb5-6266-4b33-a17f-f59d2009636c: PUT https://api.digitalocean.com/v2/load_balancers/d581bcb5-6266-4b33-a17f-f59d2009636c: 403 (request "9c1e3d0f-3482-4e99-b2f1-576442c9e5c1") Load Balancer can't be updated while it processes previous actions
Warning SyncLoadBalancerFailed  2m7s   service-controller  Error syncing load balancer: failed to ensure load balancer: failed to update load-balancer with ID d581bcb5-6266-4b33-a17f-f59d2009636c: PUT https://api.digitalocean.com/v2/load_balancers/d581bcb5-6266-4b33-a17f-f59d2009636c: 403 (request "bcb2931a-7f90-4e12-a90a-c168765e1460") Load Balancer can't be updated while it processes previous actions
Warning SyncLoadBalancerFailed  85s    service-controller  Error syncing load balancer: failed to ensure load balancer: failed to update load-balancer with ID d581bcb5-6266-4b33-a17f-f59d2009636c: PUT https://api.digitalocean.com/v2/load_balancers/d581bcb5-6266-4b33-a17f-f59d2009636c: 403 (request "f5742e47-a760-425f-bab6-15b628f67c52") Load Balancer can't be updated while it processes previous actions
Normal    EnsuringLoadBalancer  5s (x6 over 2m35s)  service-controller  Ensuring load balancer
Normal    EnsuredLoadBalancer   4s    service-controller  Ensured load balancer
```

几分钟后，你应该会有一个外部负载均衡器IP连接到你的服务。

```
$ kubectl get svc
NAME                TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)                      AGE
kubernetes          ClusterIP      10.245.0.1      <none>          443/TCP                      3d8h
nginx-clusterip     ClusterIP      10.245.216.117  <none>          80/TCP                       54m
nginx-loadbalancer  LoadBalancer   10.245.178.209  174.138.122.25  80:30919/TCP                 2m50s
nginx-nodeport      NodePort       10.245.196.9    <none>          80:30007/TCP                 18m
```

你还会注意到它被自动分配了一个节点端口 **30919**。

以下是新创建的负载均衡器的截图：

![](img/91c4bde832e221ba5bc429caf9587286_206_0.png)

*digitalocean为我们外部服务创建的负载均衡器*

请注意，这个IP在你的环境中可能不同。

当你对外部IP执行curl命令时，你将访问到集群中3个nginx Pod中的一个。

```
localhost$ curl -I http://174.138.122.25
HTTP/1.1 200 OK
Server: nginx/1.25.4
Date: Thu, 22 Feb 2024 08:31:34 GMT
Content-Type: text/html
Content-Length: 615
Last-Modified: Wed, 14 Feb 2024 16:03:00 GMT
Connection: keep-alive
ETag: "65cce434-267"
Accept-Ranges: bytes
```

## 功能完备的Flask Redis应用

掌握了关于部署和服务的新知识后，让我们将由Redis支持的Flask应用部署到集群中。首先，让我们梳理一下部署所需的所有构件：

1.  一个flask-redis应用容器镜像。确保将此容器镜像设为公开，类似于你对 **flask-app** 镜像所做的操作。
2.  一个用于flask-redis镜像的部署。
3.  一个用于flask-redis镜像的服务。
4.  一个用于redis服务的部署。
5.  一个用于redis部署的服务。

首先，我们将创建一个名为 **full-stack** 的独立命名空间来部署这些构件。

```
$ kubectl create namespace full-stack
namespace/full-stack created
```

首先，让我们创建最后两项。如果我们的flask应用需要访问Redis服务，那么集群中必须存在一个Redis服务。该服务应由一个Redis Pod支持。

以下是打包为部署的Bitnami redis镜像。

apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache # (1)
spec:
  replicas: 1 # (2)
  selector:
    matchLabels:
      app: cache # (3)
  template:
    metadata:
      labels:
        app: cache # (4)
    spec:
      containers:
      - name: redis
        image: bitnami/redis:7.2.4 # (5)
        env:
        - name: REDIS_PASSWORD # (6)
          value: SuperS3cret
        ports:
        - containerPort: 6379 # (7)

(1) -> 将 Deployment 命名为 "cache"。

(2) -> 指定只维护一个 Pod 副本。我们将在后续章节中了解如何增加副本数。

(3) -> Deployment 的选择器，用于匹配带有标签 **app=cache** 的 Pod。

(4) -> 为 Pod 模板添加标签 **app=cache**，使其能被 Deployment 的选择器选中。

(5) -> 指定容器使用的 Docker 镜像，这里是 **bitnami/redis** 镜像的 **7.2.4** 标签，就像我们在 docker-compose 对应部分所做的那样。

(6) -> 为容器定义一个名为 **REDIS_PASSWORD** 的环境变量。Bitnami redis 镜像允许我们通过环境变量设置密码。

(7) -> 在容器上暴露端口 **6379**，这是 Redis 的默认端口。

```
$ kubectl apply -f redis-deployment.yaml -n full-stack
deployment.apps/cache created
```

```
# 几分钟后 ...
```

```
$ kubectl get pod -n full-stack
NAME                       READY   STATUS    RESTARTS   AGE
cache-7cd8b6b9dc-ftkgj    1/1     Running   0          96s
```

让我们创建一个服务来为 redis pod 提供前端访问。

```
apiVersion: v1
kind: Service
metadata:
  name: cache # (1)
spec:
  selector: # (2)
    app: cache
  ports:
    - port: 6379 # (3)
      targetPort: 6379
```

(1) -> 我们将服务命名为 **cache**。

(2) -> 我们将使用与部署相同的选择器，以便将流量路由到正确的 pod。

(3) -> 我们为服务暴露相同的端口。请注意，服务类型是 **clusterIP**，因为我在 spec 中没有明确指定任何服务类型。

```
$ kubectl apply -f redis-service.yaml -n full-stack
service/cache created
```

```
$ kubectl get svc -n full-stack
NAME    TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
cache   ClusterIP      10.245.218.215  <none>        6379/TCP   2s
```

接下来，我们将进行与应用程序相关的步骤。

其中，第一步是构建 flask redis docker 镜像。

以下是步骤回顾。

```
$ export DOCKER_DEFAULT_PLATFORM=linux/amd64 # (1)
$ docker build -t ghcr.io/<GH-USERNAME>/flask-redis:1.0 . # (2)
$ export CR_PAT=ghp_XXXyyyZZZ # (3)
$ docker push ghcr.io/<GH-USERNAME>/flask-redis:1.0 # (4)
```

(1) -> 指示你将在哪个目标架构上构建容器镜像。**注意**，仅在使用 Apple Silicon 笔记本电脑时才运行此步骤。

(2) -> 构建你的镜像。将 **<GH-USERNAME>** 替换为你的 Github 用户名。

(3) -> 将镜像推送到 ghcr.io 容器注册表。

![](img/91c4bde832e221ba5bc429caf9587286_211_0.png)

将你的镜像设为公开，就像你在上一章为 **flask-app** 镜像所做的那样。

现在，让我们为 flask-redis 应用创建一个部署。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-redis # (1)
spec:
  replicas: 1 # (2)
  selector:
    matchLabels:
      app: flask-redis # (3)
  template:
    metadata:
      labels:
        app: flask-redis
    spec:
      containers:
      - name: flask
        image: ghcr.io/badri/flask-redis:1.0 # (4)
        env:
        - name: PORT # (5)
          value: "8080"
        - name: REDIS_HOST # (6)
          value: cache
        - name: REDIS_PASSWORD # (7)
          value: SuperS3cret
        ports:
          - containerPort: 8080 # (8)
```

(1) -> 部署的名称。

(2) -> 我们只运行 1 个副本。请随意更改。

(3) -> 选择器应匹配的标签。

(4) -> 我们刚刚构建的容器镜像。

(5) -> 我们在运行时提供的 **PORT** 环境变量。

(6) -> **REDIS_HOST**，指向 redis 服务名称。

(7) -> **REDIS_PASSWORD**，必须与我们提供给 redis 部署的密码相同，否则应用程序将无法访问 Redis 服务。

(8) -> 暴露的端口，**注意** 这应该与 **PORT** 环境变量相同。

然后部署它。

```
$ kubectl apply -f flask-deployment.yaml -n full-stack
deployment.apps/flask-redis created
```

```
# 一分钟后 ...
```

```
$ kubectl get pod -n full-stack
NAME                       READY   STATUS    RESTARTS   AGE
flask-redis-bdb576b95-d2h69   1/1     Running   0          31s
```

最后，让我们创建 flask redis 服务。我们将创建一个 **NodePort** 类型的服务。

```
apiVersion: v1
kind: Service
metadata:
  name: flask-redis # (1)
spec:
  type: NodePort # (2)
  selector:
    app: flask-redis # (3)
  ports:
    - port: 8080
      targetPort: 8080
      nodePort: 30080 # (4)
```

(1) -> 服务的名称

(2) -> 类型是 **NodePort**。

(3) -> 应用选择器。确保这是正确的，如果没有找到匹配的 pod，服务将指向一个黑洞，不会有任何请求被服务。

(4) -> 我们想要暴露的 nodeport。

... 然后部署它。

```
$ kubectl apply -f flask-service.yaml -n full-stack
service/flask-redis created
```

```
$ kubectl get svc -n full-stack
NAME           TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
cache          ClusterIP  10.245.218.215  <none>        6379/TCP         68m
flask-redis    NodePort   10.245.199.247  <none>        8080:30080/TCP   10s
```

让我们通过 http://<any-node-public-ip>:30080/ 访问应用程序。

![](img/91c4bde832e221ba5bc429caf9587286_216_0.png)

运行中的 flask redis 应用

## 练习 22：进行更改并重新部署

将 `<title>` 标签内容更改为 "Flask Redis app" 并重新部署新镜像。

## 练习 23：重新部署 redis flask 应用

通过将 redis 服务指向以下地址来重新部署 redis flask 应用：
a. pod ip。b. 服务的 cluster ip。

并确保一切正常工作。

## 总结

在本章中，我们通过一个部署由 Redis 数据库支持的 Python Flask 应用程序的实际示例，探讨了 Kubernetes 部署和服务的基础知识。我们介绍了部署如何管理应用程序的期望状态，确保指定数量的副本始终处于运行状态，并促进平滑的更新和回滚。我们还深入探讨了服务，它抽象了 Pod 内部通信的方式，并将应用程序暴露到外部，为你的 Flask 应用程序提供一个稳定的端点。

通过部署 Flask 应用程序及其后端服务 Redis，你亲眼目睹了 Kubernetes 如何简化从扩展到更新的复杂应用程序管理任务。这种实践经验为更高级的 Kubernetes 特性和最佳实践奠定了基础，确保你的应用程序不仅具有弹性和可扩展性，而且安全且高效地管理。

## 第 7 章：更进一步

随着我们继续前进，我们对 Kubernetes 的探索将继续揭开应用程序管理的层次，重点关注增强安全性、配置管理以及确保数据持久性。本章将向你介绍 ConfigMaps 和 Secrets，这是两个用于外部化配置数据和安全管理敏感信息的关键资源。你将学习如何将配置与应用程序代码解耦，使你的部署更加灵活和安全。

此外，我们将探讨 Init Containers 在为你的应用程序容器准备环境方面的强大功能。无论是等待服务就绪、配置权限还是执行初步设置任务，Init Containers 都提供了一种顺序、受控的方法来初始化你的应用程序环境。

最后，我们将解决 Kubernetes 中数据持久性的关键方面，重点是为 Redis 实现持久性。尽管容器具有临时性，但某些应用程序需要数据在单个 Pod 的生命周期之外持久存在。我们将深入探讨持久卷和持久卷声明，探索如何使用它们来确保你的 Redis 数据在 Pod 被重新调度或重新部署时保持完整和可用。

### 配置映射

ConfigMaps 是 Kubernetes 对象，用于以键值对的形式存储非机密数据。ConfigMaps 允许你将配置与 Pod 和容器规范分离，使你的应用程序...

## 为什么我们需要 ConfigMap

**配置解耦：** ConfigMap 将配置与容器镜像解耦，因此你可以在不修改容器镜像的情况下更改配置。这简化了应用程序的部署和管理流程。

**环境一致性：** 它们有助于在不同环境（开发、测试、生产）之间保持一致性，允许你应用特定于环境的配置，而无需修改应用程序代码。

**配置管理：** ConfigMap 提供了一种集中管理应用程序配置的方式，可以轻松更新和部署，而无需重启 Pod 或服务。

## 我们的第一个 ConfigMap

假设你想用 Nginx Pod 提供一个自定义的 **index.html** 页面。最直接的方法是修改 index.html，在本地测试，构建新的容器镜像标签，将其推送到容器注册表，最后重新部署该标签。这是一个好方法，直到它变得繁琐和令人疲惫。

让我们看看如何用 ConfigMap 解决这个问题。

首先，创建一个包含 **index.html** 的 ConfigMap。

这是我们想要制作成 ConfigMap 的 index.html 文件。

```html
<!doctype html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title> Kubernetes for Python devs</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
  <h1 class="text-3xl font-bold underline">
    My first configmap
  </h1>
</body>
</html>
```

现在，从这个文件创建一个 ConfigMap。

让我们创建一个专门用于本章的新命名空间。

```bash
$ kubectl create ns notch-further
namespace/notch-further created
```

并在那里创建 ConfigMap。

```bash
$ kubectl create configmap nginx-config --from-file=index.html -n notch-further
configmap/nginx-config created
```

**nginx-config** 是 ConfigMap 的名称，ConfigMap 的内容将从 **index.html** 加载，这就是 **--from-file** 标志的作用。

让我们检查一下我们创建的 ConfigMap。

```bash
$ kubectl get configmap nginx-config -n notch-further -o yaml
apiVersion: v1 # (1)
data: # (2)
  index.html: |
    <!doctype html>
    <html>
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title> Kubernetes for Python devs</title>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body>
      <h1 class="text-3xl font-bold underline">
        My first configmap
      </h1>
    </body>
    </html>
kind: ConfigMap # (3)
metadata:
  creationTimestamp: "2024-02-22T11:36:12Z" # (4)
  name: nginx-config
  namespace: notch-further
  resourceVersion: "1056760" # (5)
  uid: ce499044-e349-4cb8-8f04-4fdce7b5d5d3 # (6)
```

(1) -> 指定 ConfigMap 资源的 API 版本，表明它是 Kubernetes API 中的一个稳定版本。

(2) -> 包含存储在 ConfigMap 中的数据，其中每个键值对代表一个配置项。在这种情况下，我们只有 1 个键值对。键是文件名（**index.html**），值是文件的内容。| 是 YAML 中用于多行字符串的语法。

(3) -> 将 Kubernetes 资源类型标识为 ConfigMap，用于以键值对形式存储非机密数据。

(4) -> 自动生成的时间戳，标记 ConfigMap 的创建时间。

(5) -> ConfigMap 当前状态的唯一版本标识符，Kubernetes 内部用于跟踪和并发控制。

(6) -> Kubernetes 分配给 ConfigMap 的唯一标识符，用于内部跟踪和管理。

**请注意**，属性 (4)、(5) 和 (6) 也存在于其他 Kubernetes 资源中，例如 Pod、Service 和 Deployment。

我们刚刚创建了一个 ConfigMap，它目前漂浮在虚空中，尚未被任何应用程序使用。为了使其有用，我们必须将 ConfigMap 附加到 Pod。

这正是我们将在最后一步中做的事情。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
spec:
  containers:
  - name: nginx
    image: nginx
    volumeMounts: # (1)
    - name: config-volume # (2)
      mountPath: /usr/share/nginx/html # (3)
  volumes: # (4)
  - name: config-volume # (5)
    configMap: # (6)
      name: nginx-config # (7)
```

(1) -> 指定要挂载到容器中的卷。

(2) -> 标识要挂载的卷，将其链接到 volumes 部分。一个容器可以在 **volumeMounts** 部分有多个值（例如，另一个 ConfigMap 挂载在不同的路径）。

(3) -> 卷应在容器内挂载的路径。

(4) -> 定义 Pod 中容器可挂载的卷。**请注意**，这是 **spec** 的同级节点，而不是 **containers** 的。

(5) -> 为卷命名，用于在 **volumeMounts** 中引用它。

(6) -> 指定卷源是 ConfigMap。

(7) -> 要作为卷挂载的 ConfigMap 的名称。

```bash
$ kubectl apply -f nginx-with-configmap-1.yaml -n notch-further
pod/nginx-pod created
```

## 检查 Pod

```bash
$ kubectl port-forward pod/nginx-pod 8080:80 -n notch-further
Forwarding from 127.0.0.1:8080 -> 80
Forwarding from [::1]:8080 -> 80
```

![我的第一个 ConfigMap](img/91c4bde832e221ba5bc429caf9587286_225_0.png)

## 卷和卷挂载

在上面的 Pod 规范中，我们在 spec 中引入了 **volumes** 和 **volumeMounts**。两者在管理 Pod 内容器的存储方面扮演着互补的角色。以下是它们的区别以及如何协同工作的详细说明：

### volumes

Pod 规范中的 volumes 字段定义了 Pod 可以使用的存储卷。这种存储可以来自多种来源，例如上面案例中的 ConfigMap，或者持久卷声明（PVC）、Secret 等，我们将在后续章节中学习。volumes 部分是 Pod 规范的一部分，并非特定于 Pod 内的任何单个容器。

其主要目的是声明将提供给 Pod 中容器使用的存储卷。这些卷可以保存应用程序数据、配置文件、Secret 以及其他需要在容器之间共享或在 Pod 生命周期之外持久化的数据。

卷的作用范围在 **Pod 级别**。一旦定义，Pod 中的任何容器都可以挂载声明的卷。

### volumeMounts

volumeMounts 字段是 Pod 内容器规范的一部分。它指定 Pod 级别定义的卷（在 volumes 部分）如何挂载到每个容器的文件系统中。它允许你定义挂载路径（卷在容器文件系统中的可用位置）以及其他与挂载相关的选项。

其目的是使特定卷在指定路径对容器可访问，允许容器从卷读取和写入数据。

volumeMounts 的作用范围在 Pod 内的 **容器级别**。Pod 中的每个容器都可以有自己的 volumeMounts 部分，允许它将 Pod 的任何卷挂载到自己的文件系统路径。

### 它们如何协同工作

**定义卷：** 在 Pod 规范中，你首先在 volumes 字段下定义一个卷列表。在这里，你指定卷的类型（例如 PVC、Secret、ConfigMap）以及该类型的特定详细信息。

**将卷挂载到容器：** 对于需要访问卷的每个容器，你指定一个 volumeMounts 部分。在这里，你通过名称（在 volumes 部分中定义）引用卷，并指定容器内的挂载路径。

## 从 Deployment 配置 ConfigMap

我们可以从 Deployment 配置 ConfigMap 吗？是的，我们可以。以下是同一个 ConfigMap，配置为 Deployment。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3 # 指定副本数量
  selector:
    matchLabels:
      app: nginx # 此标签用于选择 Deployment 管理的 Pod
  template: # template 字段包含 Pod 规范
    metadata:
      labels:
        app: nginx # Pod 标签，必须与上面的 selector 匹配
    spec:
      containers:
```

## 练习 24：使用 ConfigMap 并在同一个 Pod 中运行两个 Nginx。

使用前几章中的双 Nginx 示例，不运行 Bitnami 的 Nginx，而是运行官方 Nginx，并使用不同的 **nginx.conf** 文件，将它们作为 ConfigMap 挂载，并在同一个 Pod 中的不同端口上运行两个 Nginx 容器。

## 从 ConfigMap 获取环境变量

将环境变量建模为 ConfigMap 是一种常见做法，尤其是在环境变量数量过多时。让我们以 flask-redis 示例为例，将几个变量作为 ConfigMap 注入。

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: flask-config
data:
  DEBUG: "False" # (1)
  REDIS_HOST: "cache" # (2)
  PORT: "8080" # (3)
```

(1) -> 调试标志已关闭。在 ConfigMap 中，即使是布尔值或数字，也建议使用字符串值。

**注意**，我们这里使用 **DEBUG** 作为示例来演示布尔标志。这可以是任何环境变量。在此上下文中，Flask 应用可以读取 **DEBUG** 环境变量，并以调试模式运行服务器。

(2) -> 配置 **REDIS_HOST**。

(3) -> Flask 应用端口。**注意**，这也是一个被建模为字符串的数字。

让我们在 **flask-redis** 部署中进行必要的更改以适应这一点。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flask-redis
  template:
    metadata:
      labels:
        app: flask-redis
    spec:
      containers:
        - name: flask
          image: ghcr.io/badri/flask-redis:1.0
          env:
            - name: DEBUG # (1)
              valueFrom: # (2)
                configMapKeyRef: # (3)
                  name: flask-config # (4)
                  key: DEBUG # (5)
            - name: PORT
              valueFrom: # (6)
                configMapKeyRef:
                  name: flask-config
                  key: PORT
            - name: REDIS_HOST
              valueFrom: # (7)
                configMapKeyRef:
                  name: flask-config
                  key: REDIS_HOST
            - name: REDIS_PASSWORD # (8)
              value: SuperS3cret
          ports:
            - containerPort: 8080
```

(1) -> 第一个环境变量 **DEBUG** 没有直接设置。

(2) -> 我们没有使用 **value**，而是使用了一个新的构造 **valueFrom**，它用于引用一个 ConfigMap（或 Secret，我们将在下一节中展示）。

(3) -> **configMapKeyRef** 指向应从中推断此值的 ConfigMap。

(4) -> ConfigMap 的名称，它**必须存在于同一个命名空间中**。

(5) -> ConfigMap 键值集中的键。

(6) -> 我们对 **PORT** 重复相同的操作，确保此值与下面的 **containerPort** 值相同。

(7) -> ...以及 **REDIS_HOST**。

(8) -> **REDIS_PASSWORD** 是一个像以前一样的普通环境变量。

部署后，我们可以在应用内部引用这些环境变量：

```python
import os
DEBUG = os.environ.get('DEBUG', 'True') # 如果未找到，默认为 'True'
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost') # 提供一个默认值
# ...
```

## 练习 25：应用程序错误

你认为上面 ConfigMap 中的 **REDIS_HOST** 值会起作用吗？如果不会，你会做什么更改来修复它？

提示：**cache** 服务是存在的，我们在上一章中确实创建了它。但是你的 **flask-redis** 应用能否跨命名空间访问它？

## ConfigMap 与 12-Factor 原则

[12-Factor 应用方法论](https://12factor.net/) 是一套旨在构建可扩展、可移植和可维护应用程序的原则，尤其适用于云环境。Kubernetes 中的 ConfigMap 与 12-Factor 方法论的多个方面保持一致，特别是在配置、代码库、依赖项和后端服务方面。

12-Factor 应用方法论的第三个原则强调配置与代码的严格分离。配置可能包括数据库 URL、外部服务凭据或特定于部署的参数。根据 12-Factor 方法论，此类配置应在不同部署之间有所不同，但应独立于应用程序的代码，从而可以轻松更改配置而无需更改代码。

让我们看看 ConfigMap 如何与这一原则保持一致。

**外部配置：** ConfigMap 提供了一种将配置数据与容器镜像和应用程序代码分开存储的方式，这与 12-Factor 应用将配置与代码分离的指令完美契合。这允许相同的应用程序镜像在不同环境（开发、测试、生产）中推广而无需重建，并在部署时应用特定于环境的配置。

**环境一致性：** 通过外部化配置，ConfigMap 有助于保持不同环境之间的一致性，这是 12-Factor 应用的一个关键原则。配置可以在不影响应用程序二进制文件的情况下进行更改或更新，确保环境之间的差异仅限于存储在 ConfigMap 中的配置。

**可移植性：** ConfigMap 增强了应用程序在不同 Kubernetes 集群或环境之间的可移植性。由于应用程序代码没有硬编码配置值，移动应用程序只需在新环境中应用相应的 ConfigMap 即可。

## 支持其他 12-Factor 原则

虽然 ConfigMap 最直接的对应是原则 III：配置，但它们的使用也支持其他 12-Factor 原则：

**原则 I：代码库和原则 II：依赖项：** 尽管 ConfigMap 与配置更直接相关，但它们的使用支持在版本控制中跟踪单一代码库以及显式声明和隔离依赖项的原则。通过将配置保持分离和外部化，ConfigMap 确保代码库保持干净和专注，并且依赖项（对外部资源或配置的依赖）在应用程序外部被清晰地定义和管理。

## 因素四：后端服务

ConfigMap 可以存储关于后端服务的信息，将这些资源视为可随时替换或重新配置的附加资源，而无需更改代码。这有助于实现与后端服务的松耦合。这正是我们之前将 **REDIS_HOST** 外部化为基于 ConfigMap 的环境变量时所做的。

考虑一个需要外部 API URL 和凭据的 Flask 应用程序。与其在应用程序或其 Docker 镜像中硬编码这些信息，不如将它们存储在 ConfigMap 中：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: flask-config
data:
  API_URL: "https://api.example.com"
  API_KEY: "secret-api-key"
```

应用程序从其环境中读取这些值，而 Kubernetes 在运行时从 ConfigMap 中填充这些环境变量。这种设置通过将配置（API URL 和密钥）与应用程序代码分离，遵循了 12 因素应用方法论，从而提升了可配置性、可移植性和环境一致性。

例如，你可能在开发命名空间中运行一个部署，并将其指向一个具有不同 **API_URL** 和 **API_KEY** 集的开发版本 ConfigMap。你也可以在生产命名空间中运行相同的实例，并将其指向另一个具有不同变量值的 ConfigMap。所有这些都无需更改代码。

ConfigMap 通过支持外部的、特定于环境的配置，体现了 12 因素应用方法论的原则。这增强了应用程序在云原生环境中的可移植性、可扩展性和可维护性。

你可能会考虑是否应该将 **REDIS_PASSWORD** 和 **API_KEY** 等明文信息外部化为 ConfigMap，你的想法是对的。这就是为什么我们有 Secrets。

## Secrets

Kubernetes Secrets 是允许你存储和管理敏感信息（如密码、OAuth 令牌和 ssh 密钥）的对象，避免了将敏感数据存储在 Pod 规范或容器镜像中的需要。与用于非敏感配置的 ConfigMap 相比，使用 Secrets 更安全，因为 Secrets 可以静态加密，并通过安全的 API 进行访问。

## 为什么使用 Secrets？

- **安全性：** Secrets 旨在保存敏感信息，降低了泄露机密数据的风险。
- **管理：** 它们提供了一种集中管理敏感信息的方式，使得在不修改应用程序代码或部署配置的情况下更新或轮换凭据变得更加容易。

## 示例：nginx 认证

让我们通过为 Nginx 服务器设置基本认证来演示如何使用 Kubernetes Secrets。

我们将创建一个 Secret 来存储凭据，并配置 Nginx 使用此 Secret 进行认证。

### 步骤 1：创建 htpasswd 文件

首先，创建一个包含用户名和加密密码的文件。你可以使用 **openssl** 或 **htpasswd**（Apache 工具的一部分）等工具进行加密。例如：

```bash
$ htpasswd -c ./auth myuser
```

此命令会提示你输入密码，并创建一个包含加密密码的 auth 文件。

**注意：** 如果你的系统中没有安装 **htpasswd**，也不必担心，你可以直接从源代码中复制生成的文件，而无需执行此步骤。密码是 **mypassword**。

### 步骤 2：从 htpasswd 文件创建 Kubernetes Secret

将 auth 文件转换为 Kubernetes Secret：

```bash
$ kubectl create secret generic nginx-auth --from-file=auth -n notch-further
secret/nginx-auth created
```

此命令创建一个名为 **nginx-auth** 的 Secret，其中包含你的 auth 文件的内容。

### 步骤 3：配置 Nginx 部署以使用 Secret

修改你的 Nginx 部署以挂载 Secret，并配置 Nginx 使用基本认证。

你需要稍微调整你的 **nginx.conf** 以支持你刚刚创建的认证。

```nginx
user nginx;
worker_processes auto;

events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name localhost;

        location / {
            root /usr/share/nginx/html;
            index index.html;
            auth_basic "Restricted Content"; # (1)
            auth_basic_user_file /etc/nginx/auth/auth; # (2)
        }
    }
}
```

(1) -> 显示给用户的消息。
(2) -> 用于检查凭据的认证文件。我们将将其作为 Kubernetes secret 挂载。

```bash
$ kubectl create configmap nginx-auth-conf --from-file=nginx-auth.conf -n notch-further
configmap/nginx-auth-conf created
```

最后，让我们看看 nginx 部署的变更。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-auth
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        volumeMounts: # (1)
        - name: auth-volume # (2)
          mountPath: "/etc/nginx/auth" # (3)
          readOnly: true # (4)
        - name: conf-volume # (5)
          mountPath: /etc/nginx/nginx.conf # (6)
          subPath: nginx-auth.conf # (7)
        ports:
        - containerPort: 80
      volumes:
      - name: auth-volume # (8)
        secret: # (9)
          secretName: nginx-auth # (10)
      - name: conf-volume # (11)
        configMap:
          name: nginx-auth-conf
```

(1) -> 指定要挂载到容器中的卷。
(2) -> 命名包含认证数据的卷，将其链接到 volumes 部分。
(3) -> 定义认证卷将挂载到的容器目录。
(4) -> 指示挂载的卷是只读的。
(5) -> 命名另一个用于 Nginx 配置的卷，将其链接到 volumes 部分。
(6) -> 指定 Nginx 配置文件要挂载到的容器路径。
(7) -> 使用卷中的特定文件，允许从 ConfigMap 或 Secret 挂载单个文件。
(8) -> 声明一个源自 Secret 的卷，用于认证数据。
(9) -> 指示卷源是 Kubernetes Secret。
(10) -> 指定包含认证数据的 Secret 的名称。
(11) -> 声明另一个卷，这次源自 ConfigMap，用于 Nginx 配置。

部署此配置后，让我们检查 secrets 是否有效。

```bash
$ kubectl apply -f nginx-auth-deployment.yaml -n notch-further
deployment.apps/nginx-auth created
```

```bash
$ kubectl port-forward deploy/nginx-auth 8080:80 -n notch-further
Forwarding from 127.0.0.1:8080 -> 80
Forwarding from [::1]:8080 -> 80
```

让我们从浏览器检查 http://localhost:8080/。

![nginx auth in action](img/91c4bde832e221ba5bc429caf9587286_244_0.png)

**注意：** 你很有可能看到的是旧的缓存副本，而不会提示你输入凭据。因此，建议在隐身窗口中打开此页面。

## 检查 secrets

让我们更仔细地看看 secrets。

```bash
$ kubectl get secret nginx-auth -n notch-further -o yaml
apiVersion: v1
data:
  auth: bXl1c2VyOiRhcHIxJEZKUVFyLy95JE5PRVdsTkFPUjBhWW9ORkowandKcTEK # (1)
kind: Secret
metadata:
  creationTimestamp: "2024-02-23T04:55:27Z"
  name: nginx-auth
  namespace: notch-further
  resourceVersion: "1277520"
  uid: bedd4dc1-044a-470e-9212-0da650c56c78
type: Opaque
```

(1) -> **auth** 键的值看起来是乱码，但它只是你的 auth 文件的 base64 编码版本。

让我们确认一下。

```bash
$ echo "bXl1c2VyOiRhcHIxJEZKUVFyLy95JE5PRVdsTkFPUjBhWWOORkowandKcTEK" | base64 --decode
myuser:$apr1$FJQQr//y$NOEWlNAOR0aYm4FJ0jwJq1
```

Kubernetes Secrets 默认是 base64 编码的，而不是加密的。Base64 编码用于数据表示，使二进制数据与仅处理文本数据的系统兼容。需要指出的是，base64 编码不是一种安全措施；它只是确保可能本身不适合文本表示的数据能够安全地传输和存储。

## 为什么是加密而不是编码

- **简单性和兼容性：** 编码简化了跨不同系统和软件处理二进制或特殊字符数据的过程。它确保 secret 数据可以在 Kubernetes 清单或 API 中安全地表示而不会损坏。
- **遗留设计选择：** Kubernetes Secrets 的初始设计并未包含内置的静态加密，而是侧重于易用性和防止随意浏览的基本保护。保护对 Secrets 访问的责任在很大程度上被委托给了底层的存储系统和集群访问控制。

**性能考量：** 编码的计算强度低于加密。虽然它不能替代加密的安全性，但在基础机制中避免使用加密可以实现更快的访问和更简单的密钥管理，前提是根据需要实施了额外的安全措施。

## 加密的必要性

虽然编码本身不提供机密性，但 Kubernetes 已经发展到支持对密钥进行静态加密，以解决安全问题。静态加密确保密钥数据不会以明文形式存储在磁盘上，从而提供更高水平的安全性。此功能必须在 Kubernetes 中显式启用和配置，允许集群管理员选择加密提供者和密钥。

## 密钥管理的最佳实践

尽管这超出了本书的范围，但以下是在 Kubernetes 中管理密钥时需要考虑的一些最佳实践。

**启用静态加密：** 使用 Kubernetes 在 etcd 数据存储中对密钥进行静态加密的能力。这确保了即使发生对数据存储的未授权访问，密钥在没有加密密钥的情况下仍然不可读。

**使用 RBAC 和访问控制：** 正确配置基于角色的访问控制（RBAC）策略，以限制谁可以访问密钥。将访问权限最小化到绝对需要的人员。

**考虑外部密钥管理：** 对于高级用例或增强的安全要求，考虑将 Kubernetes 与外部密钥管理解决方案集成，如 HashiCorp Vault、AWS Secrets Manager 或 Azure Key Vault。这些工具提供了强大的机制来存储、访问和管理敏感信息，包括动态密钥生成、自动轮换和详细的审计日志。

**传输中加密：** 确保客户端与 Kubernetes API 服务器之间的通信以及集群内服务之间的通信是加密的，以防止密钥在传输过程中被截获。

通过理解 base64 编码的局限性并实施加密和适当的访问控制，您可以显著增强 Kubernetes 中密钥的安全性。

## 另一种密钥

我们刚刚执行的是一个通用密钥。Kubernetes 密钥旨在保护其他敏感数据，如密码、OAuth 令牌和 SSH 密钥。了解不同类型的密钥及其用例有助于在 Kubernetes 集群中正确管理敏感数据。以下是 Kubernetes 提供的主要密钥类型及其示例：

### Opaque 密钥

默认的密钥类型；用于存储任意的用户定义数据。

示例用例：存储不适合其他预定义密钥类型的应用程序凭据，如 API 密钥或数据库密码。

示例：我们刚才看到的那个。它将具有以下结构。

```
apiVersion: v1
kind: Secret
metadata:
  name: my-app-secret
type: Opaque
data:
  api-key: <base64-encoded-api-key>
```

### Service Account Token 密钥

用于存储标识服务账户的令牌。

示例用例：当创建 ServiceAccount 时由 Kubernetes 自动创建，允许 Pod 向 Kubernetes API 进行身份验证。

示例：由 Kubernetes 自动管理；通常，您不会手动创建这种类型的密钥。我们将在处理服务账户时看到一个详细的示例。

### Docker Registry 密钥

用于存储访问私有 Docker 注册表的凭据。

示例用例：在 Pod/Deployment 定义中从私有 Docker 注册表拉取容器镜像。

以下是我们如何创建一个。

```
$ kubectl create secret docker-registry my-reg-secret \
--docker-server=DOCKER_REGISTRY_SERVER \
--docker-username=DOCKER_USER \
--docker-password=DOCKER_PASSWORD \
--docker-email=DOCKER_EMAIL
```

### Basic Authentication 密钥

用于存储基本身份验证的凭据。

示例用例：对需要基本身份验证的外部系统进行身份验证，例如 Web 应用程序的管理界面。

它将具有以下形式：

```
apiVersion: v1
kind: Secret
metadata:
  name: my-basic-auth
type: kubernetes.io/basic-auth
stringData:
  username: admin
  password: secret
```

### SSH Authentication 密钥

用于存储 SSH 身份验证所需的数据。

示例用例：对 SSH 服务器进行身份验证，例如在 CI/CD 管道中进行安全的 Git 访问。

```
apiVersion: v1
kind: Secret
metadata:
  name: my-ssh-key
type: kubernetes.io/ssh-auth
data:
  ssh-privatekey: <base64-encoded-private-key>
```

### TLS 密钥

用于存储证书及其关联的私钥。

示例用例：在集群内通过 HTTPS 安全地提供流量，或对需要 TLS 的外部系统进行身份验证。

```
apiVersion: v1
kind: Secret
metadata:
  name: my-tls-secret
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-certificate>
  tls.key: <base64-encoded-key>
```

我们将在后面的章节中处理 Ingress 时看到一个详细的示例。

通过为您的用例选择正确的密钥类型，您可以增强应用程序的安全性，并利用 Kubernetes 的内置机制来管理敏感数据。

## 为什么有不同类型的密钥？

将其分类为不同类型有几个目的：

**专门处理：** Kubernetes 可以优化每种类型密钥的处理方式。例如，自动将服务账户令牌附加到 Pod，或确保证书密钥包含有效的证书和密钥对。

**简化用户操作：** 通过为常见用例（如访问 Docker 注册表或存储 TLS 证书）提供特定类型，Kubernetes 简化了创建和管理密钥的过程。

**提高安全性：** 不同类型的密钥通常意味着不同的安全考虑和访问模式。通过区分类型，Kubernetes 可以实施更严格的验证和访问策略，降低配置错误和暴露的风险。

**与 Kubernetes 功能集成：** 某些密钥类型与 Kubernetes 功能和组件紧密集成，如服务账户和 Ingress 控制器。这种集成允许在集群内无缝且安全地使用这些功能。

通过提供各种类型的密钥，Kubernetes 能够实现更安全、高效且不易出错的敏感数据管理，以适应广泛的应用需求和安全要求。

## 将密钥注入为环境变量

在 **flask-redis** 示例中，我们将 **REDIS_PASSWORD** 环境变量作为明文配置注入。让我们将其转换为密钥。

首先，我们从 **REDIS_PASSWORD** 值创建一个密钥。这将是一个通用密钥。

```
$ kubectl create secret generic redis-credentials --from-literal=REDIS_PASSWORD=SuperS3cret -n notch-further
secret/redis-credentials created
```

让我们更新 **flask-redis** 部署以获取此密钥。

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flask-redis
  template:
    metadata:
      labels:
        app: flask-redis
    spec:
      containers:
        - name: flask
          image: ghcr.io/badri/flask-redis:1.0
          env:
            - name: DEBUG
              valueFrom:
                configMapKeyRef:
                  name: flask-config
                  key: DEBUG
            - name: PORT
              valueFrom:
                configMapKeyRef:
                  name: flask-config
                  key: PORT
            - name: REDIS_HOST
              valueFrom:
                configMapKeyRef:
                  name: flask-config
                  key: REDIS_HOST
            - name: REDIS_PASSWORD # (1)
              valueFrom:
                secretKeyRef: # (2)
                  name: redis-credentials # (3)
                  key: REDIS_PASSWORD # (4)
          ports:
            - containerPort: 8080
```

虽然文件的其余部分相同，但我将只介绍此上下文中的特定更改。

(1) -> **REDIS_PASSWORD** 是一个 **valueFrom**，就像其他环境变量一样。

(2) -> 它从密钥中获取值，该密钥由 **secretKeyRef** 部分引用。

(3) -> 指向同一命名空间中的密钥，即我们刚刚创建的 **redis-credentials** 密钥。

(4) -> 我们想要获取的密钥资源中的键。

重新应用此部署后，您的 **flask-redis** 应用程序将运行*大多数*符合 Kubernetes 最佳实践的规范。

## 练习 26：配置 redis 部署

使 redis 部署也从 **redis-credentials** 密钥中获取密码。完成后，凭据仅存储在一个位置，并将被两个部署（flask 和 redis）引用。

## 镜像拉取密钥

到目前为止的所有部署中，我们一直在处理公共容器镜像。即使 ghcr.io 中的 **flask-redis** 和 **simple-flask** 镜像在前面的章节中也被转换为公共镜像。现在我们已经了解了密钥，让我们利用 Docker Registry 密钥并使用凭据来拉取这些镜像。

在 Kubernetes 中，镜像拉取密钥用于存储访问私有容器镜像仓库的凭据。当 Kubernetes 集群需要从私有仓库拉取容器镜像以部署 Pod 时，它需要认证详情。镜像拉取密钥提供了一种安全的方式来存储和使用这些凭据。

## 为什么我们需要镜像拉取密钥

**访问控制：** 私有容器镜像仓库限制对镜像的访问，要求用户进行身份验证。镜像拉取密钥确保 Kubernetes 能够向仓库进行身份验证以拉取镜像。

**安全性：** 将凭据存储在密钥中，而不是存储在 Pod 规范或 Dockerfile 中，可以避免敏感数据暴露，从而增强安全性。

**自动化：** 无需手动进行身份验证干预，即可促进使用私有镜像的应用程序的自动化部署和管理。

## 镜像拉取密钥如何工作

当你创建一个使用来自私有仓库镜像的 Pod 时，你需要在 Pod 的规范中引用镜像拉取密钥。Kubernetes 使用来自镜像拉取密钥的凭据向私有仓库进行身份验证并拉取镜像。

## 创建镜像拉取密钥

首先，你需要使用仓库凭据创建一个镜像拉取密钥。

```
$ kubectl create secret docker-registry ghcr-creds \
  --docker-server=ghcr.io \
  --docker-username=<GH-USERNAME> \
  --docker-password=<GH-TOKEN> \
  --docker-email=<GH-EMAIL> \
  -n notch-further
secret/ghcr-creds created
```

我们创建了一个类型为 **docker-registry** 的密钥，名称为 **ghcr-creds**。

**docker-server** 的值是仓库的 URL，即 **ghcr.io**。

用户名是你的 Github 用户名，**<GH-TOKEN>** 是你在将镜像推送到仓库时创建的用于身份验证的令牌。

**<GH-EMAIL>** 是你的 Github 账户邮箱。最后，我们必须在尝试拉取私有镜像的同一个命名空间中创建此密钥。

在我们向 **flask-redis** 部署中添加此镜像拉取密钥之前，我们必须先将该镜像设为私有镜像。访问 **https://github.com/users/<your-user-name>/packages/container/package/flask-redis/settings**，假设你已将镜像发布为 **flask-redis**，向下滚动到“危险区域”，然后点击“更改可见性”。

将包的可见性更改为“私有”。

## 在部署中使用镜像拉取密钥

让我们修改 **flask-redis** 部署以使用镜像拉取密钥 **ghcr-creds**。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-redis-pvt
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flask-redis
  template:
    metadata:
      labels:
        app: flask-redis
    spec:
      containers:
      - name: flask
        image: ghcr.io/badri/flask-redis:1.0
        imagePullPolicy: Always # (1)
        env:
        - name: DEBUG
          valueFrom:
            configMapKeyRef:
              name: flask-config
              key: DEBUG
        - name: PORT
          valueFrom:
            configMapKeyRef:
              name: flask-config
              key: PORT
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: flask-config
              key: REDIS_HOST
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: REDIS_PASSWORD
        ports:
          - containerPort: 8080
      imagePullSecrets: # (2)
      - name: ghcr-creds # (3)
```

(1) -> 我们将 **imagePullPolicy** 的值设为 **Always**，否则镜像拉取密钥将被忽略，因为镜像已存在于节点上。

(2) -> **imagePullSecrets** 部分与 **containers** 是同级的，其中包含镜像拉取凭据。

(3) -> 我们刚刚创建的镜像拉取密钥的名称。

## 练习 27：使用不同的密钥

尝试为上述部署提供一个不同类型的密钥，看看会发生什么变化。

## 练习 28：多个镜像拉取密钥

在另一个仓库中构建另一个私有镜像。将其作为另一个容器添加到同一个 Pod/部署中，并为新的容器在配置中添加另一个镜像拉取密钥。

## init 容器

Init 容器是在 Pod 中应用容器之前运行的专用容器。它们按顺序执行，每个容器成功完成后才会启动下一个。如果任何 init 容器失败，Kubernetes 会反复重启 Pod，直到 init 容器成功为止。它们非常适合用于设置脚本或工具，这些脚本或工具不是主应用程序的一部分，但必须在应用程序启动之前运行。

![](img/91c4bde832e221ba5bc429caf9587286_260_0.png)

![](img/91c4bde832e221ba5bc429caf9587286_260_1.png)

![](img/91c4bde832e221ba5bc429caf9587286_260_2.png)

![](img/91c4bde832e221ba5bc429caf9587286_260_3.png)

![](img/91c4bde832e221ba5bc429caf9587286_260_4.png)

init 容器

## 为什么我们需要 Init 容器

**顺序初始化：** Init 容器确保在应用程序启动之前完成某些设置任务。这对于等待服务启动、初始化数据库或确保配置文件就位非常有用。

**安全性：** 通过将初始化任务分离到单独的容器中，你可以使主应用程序容器尽可能干净和安全，通过不包含仅在启动时需要的工具或脚本来减少攻击面。

**简洁性：** 它们通过卸载启动前任务来简化主应用程序容器。这使主应用程序容器可以专注于运行应用程序本身。

## 在 Flask-Redis 部署中使用 Init 容器

假设我们希望在启动 Flask 应用程序之前检查 Redis 服务是否已启动并运行。在这里，你可以使用一个 init 容器来等待 Redis 服务完全运行，然后再启动 Flask 应用程序。这确保了你的应用程序不会在其依赖项就绪之前启动。

## Redis 检查脚本

首先，你需要一个简单的脚本来检查 Redis 是否已启动并运行。这个脚本可以是一个简单的 shell 脚本，尝试连接到 Redis 服务器：

```bash
# check-redis.sh
until nc -z -v -w30 $REDIS_HOST 6379
do
  echo "Waiting for redis connection..."
  sleep 1
done
echo "Connected to Redis"
```

此脚本使用 **nc** (netcat) 来检查是否可以连接到默认端口 (**6379**) 上的 Redis。**$REDIS_HOST** 是你在 Kubernetes 中的 Redis 服务的名称。

接下来，我们希望将这个脚本制作成一个 configmap，以便我们可以将其作为 init 容器的一部分挂载。

```
$ kubectl create configmap redis-check --from-file=check-redis.sh -n notch-further
configmap/redis-check created
```

最后，我们修改部署以容纳 init 容器。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-redis-init
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flask-redis-init
  template:
    metadata:
      labels:
        app: flask-redis-init
    spec:
      initContainers: # (1)
        - name: check-redis # (2)
          image: alpine # (3)
          command: ['/bin/sh', '-c', 'apk add --no-cache netcat-openbsd && /usr/local/bin/check-redis.sh'] # (4)
          env: # (5)
            - name: REDIS_HOST
              value: redis-service
          volumeMounts: # (6)
            - name: redis-check-script
              mountPath: /usr/local/bin/check-redis.sh
              subPath: check-redis.sh
      containers:
        - name: flask
          image: ghcr.io/badri/flask-redis:1.0
          env:
            - name: DEBUG
              valueFrom:
                configMapKeyRef:
                  name: flask-config
                  key: DEBUG
            - name: PORT
              valueFrom:
                configMapKeyRef:
                  name: flask-config
                  key: PORT
            - name: REDIS_HOST
              valueFrom:
                configMapKeyRef:
                  name: flask-config
                  key: REDIS_HOST
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: redis-credentials
                  key: REDIS_PASSWORD
          ports:
            - containerPort: 8080
      volumes: # (7)
        - name: redis-check-script # (8)
          configMap:
            name: redis-check # (9)
            defaultMode: 0755 # (10)
```

(1) 指定在主应用容器之前运行的初始化容器。我们可以根据需要创建任意数量的初始化容器。

(2) 将初始化容器命名为 **check-redis**，其目的是在启动主 Flask 应用容器之前检查 Redis 服务是否就绪。

(3) 由于 Alpine Linux 镜像体积小且软件包齐全，因此被用作初始化容器的镜像。

(4) 执行 shell 命令安装 **netcat** 并运行脚本来检查 Redis 的可用性。

(5) 设置初始化容器可访问的环境变量。这些变量类似于容器的环境变量。

(6) 在初始化容器内挂载卷，用于设置 Redis 检查脚本。

(7) 定义可被初始化容器和主应用容器共同挂载的卷。

(8) 命名一个源自 ConfigMap 的卷，其中包含 Redis 检查脚本。

(9) 指定包含检查 Redis 可用性脚本的 ConfigMap。

(10) 设置卷中文件的权限模式，使脚本可执行。

当我们应用这个新的部署并检查日志时，

```
$ kubectl apply -f flask-redis-with-init.yaml -n notch-further
deployment.apps/flask-redis-init created
```

```
$ kubectl get pod -n notch-further -l app=flask-redis-init
NAME                          READY   STATUS    RESTARTS   AGE
flask-redis-init-5cc7f7646d-fs42x   0/1     Init:0/1   0          4s
```

```
$ kubectl logs -f -l app=flask-redis-init -n notch-further -c check-redis
Waiting for redis connection...
nc: getaddrinfo for host "redis-service" port 6379: Name does not resolve
nc: getaddrinfo for host "redis-service" port 6379: Name does not resolve
Waiting for redis connection...
nc: getaddrinfo for host "redis-service" port 6379: Name does not resolve
Waiting for redis connection...
...
```

请注意，Pod 的 **STATUS** 显示它正在首先运行初始化容器。除非初始化容器正常退出（错误码为 **0**），否则主容器不会启动。

初始化容器将处于无限循环中，直到 Redis 服务可用。这是因为我们没有名为 **redis-service** 的服务。这将防止主容器启动后不断崩溃，然后被 kubelet 机械地重启的问题。

让我们通过将 **REDIS_HOST** 指向 **full-stack** 命名空间中的 **cache** 服务来修复初始化容器，使其成功启动。

```
$ kubectl logs -f -l app=flask-redis-init -n notch-further -c check-redis
fetch https://dl-cdn.alpinelinux.org/alpine/v3.19/main/x86_64/APKINDEX.tar.gz
fetch https://dl-cdn.alpinelinux.org/alpine/v3.19/community/x86_64/APKINDEX.tar.gz
(1/3) Installing libmd (1.1.0-r0)
(2/3) Installing libbsd (0.11.7-r3)
(3/3) Installing netcat-openbsd (1.226-r0)
Executing busybox-1.36.1-r15.trigger
OK: 8 MiB in 18 packages
Connection to cache.full-stack.svc.cluster.local (10.245.218.215) 6379 port [tcp/redis] succeeded!
Connected to Redis
```

## 与非初始化容器的对比

Kubernetes 中的初始化容器和常规（非初始化）容器共享一些共同功能，但在 Pod 的生命周期中被设计用于不同的目的。以下是它们的异同点：

## 相同点

**容器镜像：** 初始化容器和常规容器都可以使用注册表中任何可用的容器镜像。

**资源控制：** Kubernetes 允许你为这两种容器指定资源请求和限制。

**环境变量：** 两者都可以配置环境变量来控制其行为。

**卷支持：** 初始化容器和常规容器都可以挂载和访问卷，从而允许在 Pod 内共享数据。

**Kubernetes API：** 两者都使用相同的 Kubernetes API 进行定义和管理，并且可以使用 Kubernetes 清单中的 YAML 或 JSON 进行配置。

## 不同点

1.  **执行顺序和生命周期**

    **初始化容器：** 在任何常规容器启动之前按顺序运行。每个初始化容器必须在下一个启动之前成功完成。如果初始化容器失败，Kubernetes 将根据其重启策略重启 Pod，直到初始化容器成功。

    **常规容器：** 在所有初始化容器成功完成后运行。它们可以并行运行，旨在运行 Pod 的主要应用工作负载。

2.  **目的和用例**

    **初始化容器：** 用于需要在应用容器启动之前运行完成的设置任务。常见用例包括等待服务就绪、执行数据库迁移和设置文件权限。

    **常规容器：** 托管 Pod 要运行的实际应用或服务。只要 Pod 在运行，它们就会持续运行，处理主要工作负载。

3.  **重启行为**

    **初始化容器：** 不支持诸如 restartPolicy=Always 之类的生命周期策略。如果初始化容器失败，整个 Pod 将被重启（假设 Pod 的重启策略允许），并且所有初始化容器将再次执行。

    **常规容器：** Pod 的 restartPolicy 影响常规容器在失败时如何重启。例如，使用 restartPolicy=Always 时，失败的容器会独立于 Pod 中的其他容器进行重启。

4.  **资源共享**

    **初始化容器：** 不能直接与其他初始化容器或常规容器通信，因为它们在应用容器启动之前按顺序运行。

    **常规容器：** 可以通过 localhost 和共享卷相互通信，因为它们在同一个 Pod 网络命名空间中并发运行。

5.  **存活探针和就绪探针**

    **初始化容器：** 不支持存活探针或就绪探针，因为它们必须在 Pod 就绪之前完成其任务。

    **常规容器：** 支持存活探针、就绪探针和启动探针，以管理容器生命周期事件，例如重启失败的容器或在 Pod 能够处理请求时将其标记为就绪。

## 用例

初始化容器为在主应用容器启动之前准备环境提供了一种灵活的解决方案。它们按顺序执行，并且每个都必须在下一个启动之前成功完成，这使得它们非常适合需要运行完成的设置任务。以下是 Kubernetes 中初始化容器的几个示例用例：

### 等待服务可用

初始化容器可以在启动应用容器之前等待依赖服务变得可用，确保应用程序在其依赖项就绪之前不会启动。这在微服务架构中特别有用，因为服务之间相互依赖。

### 数据库迁移和模式更新

在应用程序启动之前，初始化容器可以执行数据库迁移或模式更新。这确保数据库处于应用程序的正确状态，防止与数据库模式不匹配相关的运行时错误。

### 配置文件准备

初始化容器可以根据环境或其他动态输入生成或修改配置文件。这允许在不构建新镜像的情况下自定义应用程序配置。

### 动态凭证收集

对于需要凭证且不应存储在镜像或标准 Kubernetes Secrets 中的应用程序（例如，来自保险库的令牌），初始化容器可以获取这些凭证并将其写入应用程序容器可访问的共享卷。

### 访问控制和安全设置

它们可用于设置安全上下文、文件权限或生成应用程序所需的 SSL 证书，确保应用程序以正确的安全设置运行。

### 数据预加载

在应用程序需要在启动前本地获取特定数据的场景中，初始化容器可以从远程源预填充数据到共享卷中，或执行必要的数据转换。

### 服务注册

对于需要向发现服务或反向代理注册自身的应用程序，初始化容器可以在应用程序启动之前处理注册过程，确保应用程序在开始运行时即可被发现。

### 健康检查和就绪探针

虽然 Kubernetes 支持存活探针和就绪探针，但初始化容器可用于复杂的启动检查，这些检查可能无法通过标准探针机制轻松表达。

### 存储持久数据

我们的 Redis 缓存存储页面访问计数，我们从 flask-redis 应用中检索它并显示在页面上。到现在为止，它应该显示一个非零数字。

让我们继续删除 redis 缓存 pod。在本章的这一部分，我们将坚持使用 **full-stack** 命名空间。

```
$ kubectl get pod -n full-stack
NAME                READY   STATUS    RESTARTS   AGE
```

cache-7cd8b6b9dc-ftkgj    1/1     Running   0          23h
flask-redis-bdb576b95-dmrr9 1/1     Running   0          33s

```
$ kubectl delete pod cache-7cd8b6b9dc-ftkgj -n full-stack
pod "cache-7cd8b6b9dc-ftkgj" deleted
```

```
# 几秒钟后 ...
```

```
$ kubectl get pod -n full-stack
NAME                        READY   STATUS    RESTARTS   AGE
cache-7cd8b6b9dc-g4p6b     1/1     Running   0          52s
flask-redis-bdb576b95-dmrr9 1/1     Running   0          97s
```

这确实是一个部署。因此，Kubernetes 检测到 Pod 已停止，并重新创建了相同的规格。这很酷。但是，当我们访问应用程序 URL 时，

```
$ kubectl port-forward deploy/flask-redis 8080:8080 -n full-stack
Forwarding from 127.0.0.1:8080 -> 8080
Forwarding from [::1]:8080 -> 8080
```

![](img/91c4bde832e221ba5bc429caf9587286_273_0.png)

## 缓存重置

我们发现计数又被重置为 **1**。这个实验展示了容器的短暂性。容器被设计为短暂的，这意味着当容器停止或被删除时，存储在容器文件系统中的任何数据都会丢失。这对于像 Redis 这样的缓存和存储应用程序来说是一个挑战，因为它们需要在重启和部署之间保留数据。

如果是这样，我们如何使用容器和 Kubernetes 来处理任何有状态的应用程序？这就是卷的用武之地。

Kubernetes 中的**持久卷 (PV)** 是一种集群级资源，它将存储的提供方式与其使用方式抽象开来。它代表集群中由管理员预配置或使用存储类动态配置的一块存储（我们稍后会看到这个术语的含义）。PV 旨在供需要持久存储的应用程序使用，这些存储的生命周期比单个 Pod 更长。

**持久卷声明 (PVC)** 是 Pod 用于请求和消耗 PV 资源的命名空间资源。我们将看到 PVC 如何让开发者抽象底层存储基础设施的细节，使他们能够专注于应用程序的需求，而不是存储提供方式的具体细节。

但我们有点超前了。让我们首先通过手动创建 PV 和底层存储来说明对 PVC 和 PV 的需求。

Kubernetes 中的持久卷 (PV) 被建模为集群级资源，有几个基本原因，每个原因都符合 Kubernetes 的设计原则以及在分布式、多租户环境中管理存储的需求。原因如下：

## 抽象与解耦

PV 在底层存储基础设施之上提供了一个抽象层。这种解耦允许开发者和管理员在不需要了解底层存储实现细节（无论是云块存储、NFS、iSCSI 等）的情况下使用存储。作为集群级资源，PV 可以独立于使用它们的单个 Pod 和应用程序进行配置、管理和分配，从而促进清晰的关注点分离。

## 共享资源管理

存储通常是集群中的共享资源，可能被多个应用程序和命名空间使用。将 PV 建模为集群级资源符合这一范式，能够对存储资源进行集中管理、分配和回收。这种方法简化了在集群规模上管理存储生命周期和容量的任务，而不是在更细粒度的单个 Pod 或容器级别上进行管理。

## 动态配置与可移植性

PV 的动态配置允许根据持久卷声明 (PVC) 的需求自动创建存储资源。这个系统级功能要求 PV 是集群级的，以便有效地将来自不同命名空间的 PVC 与适当的存储资源匹配。此外，PV 提供的抽象通过标准化存储资源的请求和使用方式，增强了应用程序在集群之间的可移植性，无论底层基础设施如何。

## 访问控制与安全

Kubernetes 的基于角色的访问控制 (RBAC) 可以应用于作为集群级资源的 PV，从而对谁可以创建、查看或删除 PV 进行细粒度控制。这种级别的控制在多租户环境中至关重要，因为不同的团队或用户可能具有不同的访问和权限级别。它确保敏感的存储资源得到充分保护，并且只能由授权实体访问。

## 与其他集群资源的一致性

Kubernetes 将其他基础设施组件（如节点和 PersistentVolumeClaims）建模为集群级资源。这种一致性简化了对 Kubernetes API 的理解和使用，因为相同的交互模式适用于不同类型的资源。它还促进了在集群范围内运行的高级编排和管理工具的开发。

在 DigitalOcean Kubernetes 集群中，你可以利用 Digitalocean 块存储 (https://docs.digitalocean.com/glossary/block-storage/) 作为应用程序的持久存储解决方案。但在我们从块存储实例创建持久卷之前，我们首先必须创建该实例。

以下是执行此操作的命令：

```
$ doctl compute volume create cache --fs-type ext4 --region blr1 --size 2GiB
ID                                    Name   Size  Region Filesystem Type Filesystem Label Droplet IDs Tags
d5716c46-d22a-11ee-b619-0a58ac14b586  cache  2 GiB blr1   ext4
```

此命令创建一个名为 **cache** 的块存储，使用 **ext4** 文件系统，大小为 **2GiB**，位于 **blr1** 区域（因为我也在那里创建了我的 Kubernetes 集群）。请根据你的区域修改命令。记下创建的卷的唯一 ID。

你可以在 UI 中看到这个新创建的卷。

![](img/91c4bde832e221ba5bc429caf9587286_276_0.png)

*新的块存储卷*

接下来，我们创建一个指向此块存储实例的 **PersistentVolume** 资源。

```yaml
apiVersion: v1 # (1)
kind: PersistentVolume # (2)
metadata:
  name: cache # (3)
spec:
  capacity:
    storage: 2Gi # (4)
  accessModes:
    - ReadWriteOnce # (5)
  persistentVolumeReclaimPolicy: Retain # (6)
  storageClassName: do-block-storage # (7)
  csi:
    driver: dobs.csi.digitalocean.com # (8)
    volumeHandle: d5716c46-d22a-11ee-b619-0a58ac14b586 # (9)
    fsType: ext4 # (10)
```

(1) -> 指定 PersistentVolume 资源的 API 版本，**v1** 表示它是一个稳定的 API 资源。

(2) -> 将 Kubernetes 资源类型标识为 **PersistentVolume**。

(3) -> 将 PersistentVolume 命名为 "cache"。**请注意**，此名称是任意的，与我们创建的块存储实例的名称无关。

(4) -> 将 PersistentVolume 的存储容量设置为 2GiB。

(5) -> 允许卷以读写方式挂载到单个节点。

**accessModes** 字段指定卷如何在节点上挂载以及 Pod 中的容器如何访问该字段。此字段对于确保卷的访问级别与应用程序的需求相匹配，同时遵守底层存储系统的能力至关重要。有四种主要的访问模式：

### ReadWriteOnce (RWO)

描述：卷可以以读写方式挂载到单个节点。用例：适用于需要独占读写访问存储的单 Pod 应用程序，例如数据库或其他不支持数据复制的有状态应用程序。

### ReadOnlyMany (ROX)

描述：卷可以以只读方式挂载到多个节点。用例：适用于需要共享数据而不修改数据的应用程序，例如为 Web 应用程序提供静态内容或在多个 Pod 之间共享配置文件。

### ReadWriteMany (RWX)

描述：卷可以以读写方式挂载到多个节点。用例：适用于需要来自不同节点的多个 Pod 并发读写访问的应用程序，例如协作应用程序、共享工作区或分布式文件系统。

### ReadWriteOncePod (RWOP) (从 Kubernetes v1.22 开始可用)

描述：卷可以以读写方式挂载到单个 Pod。此访问模式比 ReadWriteOnce 更严格，因为它将卷限制为单个 Pod，而不仅仅是单个节点。用例：设计用于需要卷被同一 Pod 中的多个容器访问的用例，确保卷不会被其他 Pod 意外挂载。

## 注意事项

支持的访问模式取决于存储提供程序或所使用的 StorageClass 的功能。并非所有存储系统都支持所有访问模式。

访问模式的选择会影响应用程序与存储的交互方式。例如，选择 ReadWriteMany 可以实现需要共享存储的场景，但并非所有应用程序或存储后端都需要或支持此模式。

在定义 PersistentVolumeClaim (PVC) 以从 PV 申请存储时，PVC 中请求的访问模式必须与 PV 提供的访问模式兼容。Kubernetes 仅在 PVC 和 PV 的访问模式兼容时才会将它们绑定。

选择正确的访问模式对于依赖 Kubernetes 持久化存储的应用程序的性能、可扩展性和功能至关重要。

(6) -> 指定卷在从声明中释放后将被保留。

Kubernetes PersistentVolume (PV) 中的 **persistentVolumeReclaimPolicy** 字段指定了当 PersistentVolumeClaim (PVC) 被释放时，底层物理存储会发生什么。此策略有助于高效管理存储资源，确保根据应用程序和集群的需求妥善处理存储的生命周期。主要有三种回收策略：

## Retain

当 PVC 被释放时，PV 不会自动删除。卷上的数据保持完整，卷被视为“已释放”但仍绑定到集群。需要手动干预来回收资源并删除数据。

**使用场景：** 当您希望在删除前手动审查数据，或需要确保数据不会被意外删除时非常有用。此策略适用于在删除前需要手动备份或审计的敏感数据。

## Delete

当 PVC 被释放时，PV 和底层基础设施中的关联存储资产（例如 AWS EBS 卷、Google Compute Engine 磁盘或 Azure 磁盘）会被自动删除。

**使用场景：** 非常适合动态环境和开发场景，其中首选自动清理资源以防止未使用的卷和相关成本的积累。

## Recycle (已弃用)

卷会被清除（即数据被删除）并再次可用于新的声明。由于其局限性和潜在的安全问题，此策略已被弃用，不建议在新部署中使用。

**使用场景：** 以前用于需要自动擦除数据和重用卷而无需创建新存储资产的简单场景。由于已弃用，请考虑使用带有 Delete 策略的动态配置来实现类似行为。

(7) -> 将 PersistentVolume 与 **do-block-storage** StorageClass 关联。

Kubernetes 中的 StorageClass 是一种资源，允许管理员根据其集群中可用的底层存储系统定义不同的存储类别。StorageClass 抽象了存储后端和配置机制的细节，提供了一种根据 PersistentVolumeClaim (PVC) 的需求来配置 PersistentVolume 的方法。

以下命令列出了我们集群中所有可用的存储类：

```
$ kubectl get storageclass
```

| NAME | PROVISIONER | RECLAIMPOLICY | VOLUMEBINDINGMODE | ALLOWVOLUMEEXPANSION | AGE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| do-block-storage (default) | dobs.csi.digitalocean.com | Delete | Immediate | true | 27d |
| do-block-storage-retain | dobs.csi.digitalocean.com | Retain | Immediate | true | 27d |
| do-block-storage-xfs | dobs.csi.digitalocean.com | Delete | Immediate | true | 27d |
| do-block-storage-xfs-retain | dobs.csi.digitalocean.com | Retain | Immediate | true | 27d |

(8) -> 标识 DigitalOcean 块存储的容器存储接口 (CSI) 驱动程序。

容器存储接口 (CSI) 是一个标准，用于向容器编排系统 (COS)（如 Kubernetes、Mesos、Docker 和 Cloud Foundry）上的容器化工作负载暴露任意块和文件存储系统。CSI 作为容器生态系统中各利益相关者的协作项目开发，旨在为容器化应用程序提供一致、标准化的 API 来访问存储资源。

(9) -> 指定 DigitalOcean 卷的唯一标识符。这对您来说会有所不同。

(10) -> 将卷的文件系统类型设置为 **ext4**。

Kubernetes 中持久卷 (PV) 的文件系统类型取决于底层存储系统以及正在使用的容器存储接口 (CSI) 驱动程序或内联卷插件。当您配置 PV 时，您可以指定卷应格式化的文件系统类型，前提是底层存储系统支持该文件系统类型。

让我们应用这个 PV。

```
$ kubectl apply -f pv.yaml
persistentvolume/cache created
```

```
$ kubectl get pv
NAME    CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS      CLAIM   STORAGECLASS   VOLUMEATTRIBUTESCLASS   REASON   AGE
cache   2Gi        RWO            Retain           Available           do-block-storage   <unset>                 4s
```

我们已经将一个 PersistentVolume **cache** 与块存储对象 **cache** 关联起来。

接下来，让我们将一个持久卷声明与此持久卷关联。

```
apiVersion: v1 # (1)
kind: PersistentVolumeClaim # (2)
metadata:
  name: cache # (3)
spec:
  accessModes: # (4)
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi # (5)
  storageClassName: do-block-storage # (6)
```

(1) -> 指定 PersistentVolumeClaim 资源的 API 版本。

(2) -> 将 Kubernetes 资源类型标识为 PersistentVolumeClaim。

(3) -> 将 PersistentVolumeClaim 命名为 "cache"。同样，这是任意的。

(4) -> 定义卷的访问方式；在这种情况下，是单节点读写。

(5) -> 为声明请求一个大小为 2 GB 的卷。

(6) -> 指定用于配置卷的 StorageClass，名为 **do-block-storage**。

您会注意到 PVC 和 PV 之间有很多参数是相似的。一旦我们应用此资源，Kubernetes 将从其库存中找到一个 PV 并将其与此 PVC 关联，*前提是* PVC 规范与任何现有的 PV 匹配。例如，Kubernetes 仅在 PVC 和 PV 的访问模式兼容时才会将它们绑定。如果我们给 **accessMode** 属性赋予 **ReadWriteMany** 值，这个 PVC 将不会绑定到我们创建的 PV。

```
$ kubectl apply -f pvc.yaml -n full-stack
persistentvolumeclaim/cache created
```

```
$ kubectl get pvc -n full-stack
NAME    STATUS   VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS       VOLUMEATTRIBUTESCLASS   AGE
cache   Bound    cache    2Gi        RWO            do-block-storage   <unset>                 33s
```

**STATUS** 显示为 **Bound**，意味着此 PVC 已与一个 PV 绑定。让我们检查一下 PV 的状态。

```
$ kubectl get pv
NAME    CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                 STORAGECLASS       VOLUMEATTRIBUTESCLASS   REASON   AGE
cache   2Gi        RWO            Retain           Bound    full-stack/cache      do-block-storage   <unset>                       2m57s
```

它显示 PV 已与 PVC **full-stack/cache**（**full-stack** 命名空间中的 cache PVC）绑定。

我们有一个功能齐全的 PVC，现在可以与 Pod 关联了。这就是我们接下来要做的。

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cache
  template:
    metadata:
      labels:
        app: cache
    spec:
      containers:
      - name: redis
        image: bitnami/redis:7.2.4
        env:
        - name: REDIS_PASSWORD
          value: SuperS3cret
        ports:
        - containerPort: 6379
        volumeMounts: # (1)
        - name: redis-data # (2)
          mountPath: /bitnami/redis/data # (3)
      volumes:
      - name: redis-data # (4)
        persistentVolumeClaim: # (5)
          claimName: cache  # (6)
```

(1) -> 指定要挂载到 redis 容器的卷。

(2) -> 命名将要挂载的卷，并将其与下面的 `volumes` 部分关联起来。

(3) -> 定义卷在容器内的挂载路径。**/bitnami/redis/data** 是 Bitnami redis 镜像存储 redis 数据的路径。

(4) -> 通过名称声明一个卷，该卷将由 PersistentVolumeClaim 提供。

(5) -> 表明卷的来源是一个 PersistentVolumeClaim。

(6) -> 指定提供该卷的 PersistentVolumeClaim 的名称。这就是我们在上一步中创建的 claim。

让我们应用这个更新后的部署。

```
$ kubectl apply -f redis-deployment-with-pvc.yaml -n full-stack
deployment.apps/cache configured
```

但我们得到了一个意外的响应。

```
$ kubectl get pod -n full-stack
NAME                                READY   STATUS             RESTARTS   AGE
cache-77d8d48c6b-g6dpj              0/1     CrashLoopBackOff   1 (11s ago)   21s
cache-7cd8b6b9dc-g4p6b              1/1     Running            0            23m
flask-redis-bdb576b95-dmrr9          1/1     Running            0            24m
```

让我们检查容器，看看是否有任何异常。

```
$ kubectl describe pod cache-77d8d48c6b-g6dpj -n full-stack
```

Name: cache-77d8d48c6b-g6dpj
Namespace: full-stack

Events:

| Type | Reason | Age | From | Message |
|---|---|---|---|---|
| Normal | Scheduled | 36s | default-scheduler | Successfully assigned full-stack/cache-77d8d48c6b-g6dpj to first-cluster-default-pool-oeo85 |
| Normal | SuccessfulAttachVolume | 30s | attachdetach-controller | AttachVolume.Attach succeeded for volume "cache" |
| Normal | Pulled | 10s (x3 over 27s) | kubelet | Container image "bitnami/redis:7.2.4" already present on machine |
| Normal | Created | 10s (x3 over 27s) | kubelet | Created container redis |
| Normal | Started | 10s (x3 over 27s) | kubelet | Started container redis |
| Warning | BackOff | 9s (x3 over 25s) | kubelet | Back-off restarting failed container redis in pod cache-77d8d48c6b-g6dpj_full-stack(2c0932d8-635f-4264-9933-2d1a3809b920) |

它成功挂载了卷并启动了容器。但是容器崩溃了。是时候检查日志了。

```
$ kubectl logs -f cache-77d8d48c6b-g6dpj -n full-stack
```

```
redis 09:17:10.74 INFO ==>
redis 09:17:10.74 INFO ==> Welcome to the Bitnami redis container
redis 09:17:10.74 INFO ==> Subscribe to project updates by watching https://github.com/bitnami/containers
redis 09:17:10.75 INFO ==> Submit issues and feature requests at https://github.com/bitnami/containers/issues
redis 09:17:10.75 INFO ==>
redis 09:17:10.75 INFO ==> ** Starting Redis setup **
redis 09:17:10.78 INFO ==> Initializing Redis
redis 09:17:10.80 INFO ==> Setting Redis config file
redis 09:17:10.84 INFO ==> ** Redis setup finished! **

redis 09:17:10.87 INFO ==> ** Starting Redis **
1:C 23 Feb 2024 09:17:10.892 * oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
1:C 23 Feb 2024 09:17:10.892 * Redis version=7.2.4, bits=64, commit=00000000, modified=0, pid=1, just started
1:C 23 Feb 2024 09:17:10.892 * Configuration loaded
1:M 23 Feb 2024 09:17:10.893 * monotonic clock: POSIX clock_gettime
1:M 23 Feb 2024 09:17:10.894 * Running mode=standalone, port=6379.
1:M 23 Feb 2024 09:17:10.895 * Server initialized
1:M 23 Feb 2024 09:17:10.895 # Can't open or create append-only dir appendonlydir: Permission denied
```

看起来 redis 进程无法读取或写入挂载的卷。但在我们修复之前，让我们看看为什么会发生这种情况。

这个问题通常是由于 Kubernetes 持久卷（PV）和某些存储后端处理挂载目录的所有权和权限的方式造成的。

默认情况下，卷的文件系统所有者是 **root:root**。如果 Pod 以非 root 用户身份运行，并且需要在卷上创建文件或目录，这将由于权限不足或不正确而失败。

解决方案是使用 initContainers 创建一个临时容器来更改卷文件系统的权限/所有权。

我们将在 redis 容器启动并使用该卷之前，创建一个 init 容器来修复此权限问题。

这将是更新后的 redis 缓存部署。

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cache
  template:
    metadata:
      labels:
        app: cache
    spec:
      initContainers: # (1)
      - name: volume-permissions # (2)
      image: busybox # (3)
      command: ['sh', '-c', 'chmod -R 777 /data'] # (4)
      volumeMounts: # (5)
      - name: redis-data # (6)
        mountPath: /data # (7)
      containers:
      - name: redis
        image: bitnami/redis:7.2.4
        env:
        - name: REDIS_PASSWORD
          value: SuperS3cret
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /bitnami/redis/data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: cache
```

文件的其余部分相同，让我们来看看更改的部分。

(1) -> 定义一个或多个在主应用容器启动之前运行的 init 容器。

(2) -> 命名负责设置卷权限的 init 容器。取一个有意义的名字即可。

(3) -> 指定 init 容器的 Docker 镜像，在本例中是 **busybox**。一个最小化的镜像，可用于运行此类小任务。

(4) -> 执行一个 shell 命令，将 **/data** 目录的权限更改为所有用户可写。

(5) -> 指定要挂载到 init 容器内的卷。

(6) -> 标识将要挂载到 init 容器中的卷。这应与 **volumes** 部分中的名称匹配。

(7) -> 设置 init 容器内卷的挂载路径。这不必与主容器中的挂载路径相同。我们只需要一个快速的地方来挂载卷并运行 **chmod** 命令。

一旦我们应用这个新的部署。

```
$ kubectl get pod -n full-stack
NAME                                READY   STATUS    RESTARTS   AGE
cache-77dd5d666f-hcfjr              1/1     Running   0          39s
flask-redis-bdb576b95-dmrr9         1/1     Running   0          32m
```

```
$ kubectl port-forward deploy/flask-redis 8080:8080 -n full-stack
Forwarding from 127.0.0.1:8080 -> 8080
Forwarding from [::1]:8080 -> 8080
```

让我们用几次页面访问来填充 redis 缓存。

![](img/91c4bde832e221ba5bc429caf9587286_292_0.png)

*缓存删除前的页面访问计数*

停掉 Pod。

```
$ kubectl delete pod cache-77dd5d666f-hcfjr -n full-stack
pod "cache-77dd5d666f-hcfjr" deleted
```

```
# 新的 Pod 几乎立即启动...
```

```
$ kubectl get pod -n full-stack
NAME                       READY   STATUS    RESTARTS   AGE
cache-77dd5d666f-pnmwf    1/1     Running   0          5s
flask-redis-bdb576b95-dmrr9 1/1     Running   0          34m
```

```
$ kubectl port-forward deploy/flask-redis 8080:8080 -n full-stack
Forwarding from 127.0.0.1:8080 -> 8080
Forwarding from [::1]:8080 -> 8080
```

现在检查页面计数。数据应该已经持久化了。

![](img/91c4bde832e221ba5bc429caf9587286_293_0.png)

*缓存 Pod 重启后的页面访问计数*

## 动态配置

让我们回顾一下我们所做的步骤。我们首先使用 **doctl** 创建了一个块存储对象，然后创建了一个与之关联的 PV，接着是一个 PVC。然后我们将 PVC 与部署关联起来。如果我们只创建一个 PVC 并更新部署，而 Kubernetes 负责创建块存储、PV 等所有繁重的工作呢？

这正是动态配置的含义。

Kubernetes 中的动态配置是一个功能，它可以根据 Persistent Volume Claims (PVCs) 的需求自动创建存储资源（持久卷，PVs）。这个过程消除了管理员手动预配置存储的需要，并允许应用程序动态地请求和使用存储。

## 动态配置如何工作

![](img/91c4bde832e221ba5bc429caf9587286_294_0.png)

## 动态供应的工作原理

**StorageClass：** 管理员在集群中定义一个或多个 StorageClass 对象。每个 StorageClass 指定一个供应器（例如，AWS EBS、Google 持久磁盘、Azure 磁盘存储或本地存储选项）以及一组特定于该供应器的参数，如磁盘类型、大小和 IOPS。StorageClass 充当创建 PersistentVolumes 的模板。

**PersistentVolumeClaim (PVC) 请求：** 当用户或应用程序需要存储时，他们会创建一个 PVC，指定所需的特性，如大小和访问模式。重要的是，PVC 引用一个 StorageClass 来指示所需的存储类型。

**自动供应：** Kubernetes 控制平面监视那些无法由现有 PV 满足的 PVC。当它检测到这样的 PVC 时，它会使用指定的 StorageClass 动态供应一个新的 PV，以匹配 PVC 的要求。供应过程由 StorageClass 中定义的存储供应器处理。

**绑定：** 一旦新的 PV 被创建，Kubernetes 会自动将其绑定到 PVC，使存储可供应用程序使用。此绑定过程根据 PVC 和 PV 的规格和声明进行匹配。

## 动态供应的优势

**效率：** 消除了手动预供应存储的需要。存储在需要时自动创建，确保资源的高效利用。

**速度：** 通过减少手动步骤，加快了需要持久存储的应用程序的部署速度。

**灵活性：** 允许用户透明地使用存储资源，无论底层存储基础设施如何。应用程序可以精确指定它们需要的存储类型（例如，SSD 与 HDD、大小、IOPS），而无需担心供应的细节。

**可扩展性：** 通过自动化供应过程，支持应用程序的可扩展部署，这在动态和大规模环境中尤为重要。

## 示例场景

一个在云中运行 Kubernetes 的组织想要部署一个需要高 IOPS 持久存储的数据库应用程序。与其手动创建云磁盘并将其绑定到应用程序：

- 管理员创建一个特定于其云提供商提供的高 IOPS SSD 的 StorageClass。
- 应用程序的部署包含一个 PVC，该 PVC 指定所需的存储大小并引用高 IOPS StorageClass。
- 当应用程序部署时，Kubernetes 根据 StorageClass 参数自动供应一个合适的云磁盘，并将其绑定到应用程序的 PVC。
- 数据库应用程序使用动态供应的存储，无需在存储供应过程中进行任何手动干预。

动态供应简化并自动化了存储分配过程，使 Kubernetes 环境更灵活且更易于管理。

此外，想想我们的情况，每个需要为其应用程序存储的开发者都需要拥有 DigitalOcean 云密钥和访问 **doctl** 的权限，以便在创建任何 PVC 之前创建块对象存储。听起来相当复杂，对吧？动态供应将所有这些从你的开发者那里抽象出来。

## 实际示例

这次我们只创建 PVC。

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dynamic-cache
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
  storageClassName: do-block-storage
```

```
$ kubectl apply -f pvc-dynamic.yaml -n full-stack
persistentvolumeclaim/dynamic-cache created
```

PVC 被创建了，PV 也被创建了！

```
$ kubectl get pvc -n full-stack
NAME            STATUS   VOLUME          CAPACITY   ACCESS MODES   STORAGECLASS      VOLUMEATTRIBUTESCLASS AGE
cache           Bound    cache           2Gi        RWO            do-block-storage  <unset>                157m
dynamic-cache   Bound    pvc-991f35c9-c4d2-4e9f-9fdf-9aa65e819935  2Gi  RWO  do-block-storage  <unset>  9s
```

```
$ kubectl get pv
NAME                                       CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                       STORAGECLASS      VOLUMEATTRIBUTESCLASS   REASON   AGE
cache                                      2Gi        RWO            Retain           Bound    full-stack/cache            do-block-storage  <unset>                       160m
pvc-991f35c9-c4d2-4e9f-9fdf-9aa65e819935   2Gi        RWO            Delete           Bound    full-stack/dynamic-cache    do-block-storage  <unset>                       13s
```

让我们检查一下后端是否也创建了存储对象。

```
$ doctl compute volume list
ID                                    Name                                      Size   Region   Filesystem Type   Filesystem Label   Droplet IDs   Tags
d5716c46-d22a-11ee-b619-0a58ac14b586  cache                                     2 GiB  blr1     ext4               [402016564]        k8s:72ad2064-31b4-4552-a630-c88487140b6f
c993be03-d241-11ee-bb89-0a58ac14b93e  pvc-991f35c9-c4d2-4e9f-9fdf-9aa65e819935  2 GiB  blr1     unknown            k8s:72ad2064-31b4-4552-a630-c88487140b6f
```

接下来的步骤与此相同。

## PV 和 PVC 之间的关键区别

持久卷 (PV) 和持久卷声明 (PVC) 是 Kubernetes 中与存储相关的两个基本概念。它们协同工作以在集群中提供和管理存储，但它们扮演着不同的角色：

### 持久卷 (PV)

**集群资源：** PV 是集群级别的资源，代表集群中的一块存储。它们独立于任何单个 Pod 的生命周期，可以由管理员手动或通过 StorageClass 动态供应。

**供应：** 可以由管理员预供应，也可以由 Kubernetes 使用 StorageClass 动态供应。供应方法决定了 PV 的创建方式及其代表的内容（例如，云存储中的磁盘、节点上的本地路径等）。

**生命周期和管理：** 在集群级别进行管理，这意味着它们存在于 Kubernetes 集群中，直到被手动删除或根据回收策略被回收。PV 有其自己的生命周期，独立于使用它们的 Pod。

**访问策略：** 包括访问模式（例如，ReadWriteOnce、ReadOnlyMany、ReadWriteMany）和回收策略，该策略规定了 PV 从声明中释放后会发生什么（例如，Retain、Delete、Recycle）。

### 持久卷声明 (PVC)

**Pod 资源：** PVC 是用户对存储的请求。它们指定大小、访问模式，有时还指定特定的 StorageClass。PVC 是命名空间资源，这意味着它们存在于特定的命名空间中，并由该命名空间中的 Pod 使用。

**请求存储：** 充当用户对存储的请求。它指定大小和访问模式（例如，卷是否可以作为读写挂载到单个节点，或作为只读挂载到多个节点），并可选地请求特定的 StorageClass。

**绑定：** PVC 会自动绑定到满足其要求的可用 PV。此绑定由 Kubernetes 完成，并基于 PVC 中指定的大小、访问模式和 StorageClass。

**生命周期：** PVC 的生命周期与其所在的命名空间和使用它的 Pod 相关联。只要 Pod 需要，PVC 就存在，当不再使用时会被删除。

### 关键区别总结

**范围：** PV 是集群范围的资源，而 PVC 是命名空间范围的。

**目的：** PV 代表集群中可用的物理或虚拟存储空间，而 PVC 代表用户或应用程序对存储的请求。

**生命周期：** PV 的生命周期独立于 Pod 进行管理，但 PVC 与使用它的 Pod 的生命周期相关联。

**供应：** PV 可以静态供应（由管理员预供应）或动态供应（由 Kubernetes 响应 PVC 自动创建）。PVC 始终由用户或应用程序创建以请求存储资源。

**使用：** PV 绑定到 PVC 以满足存储请求。PVC 指定其在存储方面的需求，Kubernetes 将其绑定到适当的 PV。

理解 PV 和 PVC 之间的关系和区别对于在 Kubernetes 中有效管理存储至关重要，确保应用程序拥有其运行所需的必要存储资源。

## 总结

在本章中，我们探讨了几个关键概念，这些概念是管理 Kubernetes 中状态和配置的基础，增强了应用程序的可移植性、可扩展性和安全性。通过深入研究 ConfigMaps、Secrets、Init Containers 和持久性，我们掌握了创建复杂的云原生应用程序所需的工具，这些应用程序可以优雅地处理配置数据、安全管理敏感信息、确保正确的初始化，并在应用程序的整个生命周期中维护数据。

**ConfigMaps** 允许我们将配置工件与镜像内容解耦，使应用程序能够动态配置。这种灵活性对于在不同环境中部署相同的应用程序而无需重新构建或更改代码至关重要。

**Secrets** 提供了一种安全的机制来存储和管理敏感信息，如密码、OAuth 令牌和 ssh 密钥，保护它们免受暴露和未经授权的访问。我们学习了如何使用 Secrets 来保护应用程序的机密数据，确保其在传输和静态时保持加密。

**Init Containers** 提供了一种强大的方式，在 Pod 的主容器启动之前执行设置任务。我们看到 init 容器可用于诸如等待依赖项就绪、设置环境，或准备配置文件，确保我们的应用程序以能够处理请求的状态启动。

**持久化**通过持久卷（PV）和持久卷声明（PVC）向我们介绍了 Kubernetes 中持久存储的概念。我们探讨了 PV 如何对物理存储资源进行抽象，使得存储能够超越单个 Pod 的生命周期而持续存在。PVC 使应用程序能够根据其存储需求动态地请求和使用持久存储。

通过实际示例，特别是 Redis 缓存服务的设置和配置，我们看到了这些概念在现实场景中的应用。我们学习了如何使用 PV 和 PVC 动态配置存储，使用 ConfigMaps 和 Secrets 配置 Redis，并通过初始化容器确保其就绪状态。

总之，本章强调了在 Kubernetes 中有效管理配置和存储的重要性。通过掌握 ConfigMaps、Secrets、初始化容器和持久化，开发人员和管理员可以构建具有弹性、灵活性和安全性的应用程序，为应对现代云环境的挑战做好准备。这些工具和概念是充分利用 Kubernetes 潜力的基础，使应用程序能够在动态且可扩展的云原生生态系统中蓬勃发展。

## 第 8 章：更健壮的设置

随着我们进一步深入 Kubernetes 生态系统，我们的旅程进入了更高级的领域，这对于大规模编排复杂应用程序至关重要。本章建立在我们迄今所获得的基础知识之上，深入探讨 Kubernetes 提供的用于管理有状态应用程序、增强安全性和扩展 Kubernetes 能力的复杂机制。这个循序渐进的介绍将为理解 StatefulSets、服务账户、基于角色的访问控制（RBAC）、Ingress、自定义资源定义（CRD）和证书管理器铺平道路——它们都是 Kubernetes 拼图中的关键部分。

**StatefulSets** 是我们将要探索的第一个概念。与 Deployments 不同，StatefulSets 专为有状态应用程序（如数据库）设计，这些应用程序需要唯一的、持久的身份、稳定的网络主机名以及有序的部署和扩缩容。我们将深入探讨 StatefulSets 如何管理一组 Pod 的部署和扩缩容，并确保这些 Pod 的顺序性和唯一性得到保证。

接下来是**服务账户**和 **RBAC**，它们向我们介绍了 Kubernetes 的安全机制，用于控制对集群 API 的访问。服务账户为在 Pod 中运行的进程提供身份，允许 Kubernetes 为 Pod 分配特定的权限和访问权限，以便与 Kubernetes API 交互。结合 RBAC，我们可以定义细粒度的访问控制策略，以限制用户或服务账户可以执行的操作，确保我们集群的操作是安全且管理良好的。

**Ingress** 是我们将涵盖的另一个关键概念。它为集群内的服务提供 HTTP 和 HTTPS 路由，允许外部流量到达在 Kubernetes 中运行的应用程序。理解 Ingress 对于以受控且高效的方式将您的服务暴露给外界至关重要。

在本章中，我们将看到这些高级功能如何相互作用，为在 Kubernetes 中部署和管理应用程序提供一个健壮、安全且可扩展的平台。在本章结束时，您将更深入地了解如何利用 Kubernetes 满足复杂现实应用程序的需求，确保它们安全、可扩展且可维护。

### Redis 作为有状态集

让我们以之前章节中实现的 Redis 缓存为例。我们决定使其更具弹性，并将副本数增加到 2。我们遇到了以下问题：

```
$ kubectl describe pod cache-77dd5d666f-scrrl -n robust-setup
...
Events:
  Type    Reason     Age   From               Message
  ----    ------     ----  ----               -------
  Normal  Scheduled  62s   default-scheduler  Successfully assigned robust-setup/cache-77dd5d666f-scrrl to first-cluster-default-pool-oeo8s
  Warning FailedAttachVolume  56s   attachdetach-controller  Multi-Attach error for volume "pvc-cb035503-13c4-4740-9262-7e65837999cd" Volume is already used by pod(s) cache-77dd5d666f-2d8t2
```

我们的 PVC **cache** 配置为 **ReadWriteOnce**，这意味着第一个启动的 Pod 会挂载该卷，而第二个 Pod 无法启动，因为它无法挂载已经挂载到第一个 Pod 的卷。

我们可以将卷配置为 **ReadWriteMany** 访问模式，但 Digitalocean Kubernetes 的存储策略不允许我们创建此类卷，至少开箱即用不支持。在我们朝那个方向努力之前，让我们讨论一下 Redis 以及通常情况下，大多数有状态应用程序在扩缩容时是如何工作的。

无状态应用程序不内部存储数据或会话信息。处理所需的任何状态或数据通常在每个请求中提供，响应生成不依赖于存储的数据。在 Kubernetes 中扩缩容无状态应用程序相对简单。您可以增加或减少 Pod 副本的数量，而无需担心数据持久性或同步问题，因为每个实例都是独立且相同的。

示例：nginx 部署被认为是无状态的，因为它不需要在请求之间记住有关客户端会话的信息。即使是 Flask hello world 和 redis 页面计数应用程序也是无状态的。**请注意**，尽管页面计数在每次访问页面时都会递增，但页面计数应用程序仍被认为是无状态的。这是因为页面计数并非直接存储在该应用程序中，而是从 Redis 缓存中获取的。

有状态应用程序保存交互过程中生成的数据，并依赖这些数据进行未来的事务处理。扩缩容有状态应用程序更为复杂。它需要仔细管理数据持久性、复制和一致性。

示例：Redis 缓存部署是有状态的。

Redis 主从架构是在 Redis 部署中实现数据复制和高可用性的一种方法。它是 Redis 能够水平扩展并为存储数据提供容错能力的基础部分。以下是其工作原理的概述：

### 组件

**主节点：** 主节点是接收所有写操作的主节点。它保存原始数据集，并负责处理客户端请求，包括读和写。主节点还将更改复制到一个或多个从节点。

**从节点：** 从节点是主节点的副本。它们连接到主节点以执行初始同步（从主节点复制整个数据集），随后在发生更改时从主节点接收更新。从节点可以接受读请求，以分担读负载并提高系统的读吞吐量。在主节点发生故障时，可以将从节点提升为新的主节点，从而确保服务的可用性。

### 复制过程

**初始同步：** 当从节点连接到主节点时，它会启动一个同步过程。主节点创建其当前数据集的快照并将其发送给从节点。从节点将此快照加载到其内存中，有效地克隆了主节点的数据集。

**命令传播：** 初始同步后，主节点继续将所有写命令传播到已连接的从节点。这确保了对主节点所做的所有更改都会在从节点上镜像，从而保持主节点及其所有副本之间数据集的一致性。

**部分重新同步：** 如果从节点暂时与主节点失去连接，但在主节点缓冲区耗尽之前重新连接，它可以请求部分重新同步。然后主节点仅发送断开连接期间发生的更改，从而避免了完整数据集传输的需要。

通过将数据从主节点复制到一个或多个从节点，Redis 确保即使在硬件故障或网络问题的情况下数据也不会丢失，并且它支持跨多个节点扩展读操作。

这表明每个 redis 实例（主节点或从节点）都需要有自己的持久卷来存储数据。现在共享同一个 PVC 是不可能的了。

### 临时 Pod 身份的问题

Deployment 中的每个 Pod 都是可互换的，没有稳定的网络身份或存储。当 Pod 被终止并替换时（这可能在扩缩容操作、更新或节点故障期间发生），新 Pod 被视为一个全新的实例。即使 PVC 确保了数据在单个 Pod 生命周期之外的持久性，新 Pod 也可能不会重新挂载到与旧 Pod 相同的持久卷（PV），特别是当 Deployment 扩展到多个副本时。这可能导致数据不一致或丢失，因为新 Pod 可能以空数据集或过时的数据快照启动。

### 没有有序、优雅的扩缩容或更新

Deployment 不保证 Pod 启动、停止或更新的顺序。对于通常依赖主从架构进行复制和高可用性的 Redis 来说，这可能会破坏复制拓扑。例如，主实例可能在新的副本完全同步之前被终止，从而导致潜在的数据丢失或服务中断。

## 有状态集的需求

如果部署中的每个 Pod 都能拥有自己的 PVC 会怎样？此外，如果我们能获得关于每个副本启动顺序的强力保证呢？例如，主节点总是先启动，然后是从副本节点。另外，如果 Kubernetes 能给我们提供可预测的 Pod 标识，比如 **redis-master-0**，而不是随机的 **cache-cd5968d5b-btjcg**（我们不知道 **cache-cd5968d5b-btjcg** 是主节点还是从节点 Pod），那将解决我们所有的问题。这正是有状态集所提供的。

让我们看看有状态集如何解决这些问题以及更多问题。

**稳定、唯一的网络标识符：** 有状态集中的每个 Pod 都有一个稳定的主机名，该主机名源自有状态集的名称和一个序号索引（例如 **redis-0**、**redis-1**）。这确保了即使在重启或重新调度后，Pod 的标识也保持一致，从而保留了网络通信和数据复制配置。

**持久存储绑定：** 有状态集中的每个 Pod 都可以通过 PVC 模板与一个专用的 PersistentVolume 关联。这确保了每个 Pod 始终重新连接到相同的 PV，从而在重启和重新调度过程中保留其状态。

**有序、优雅的部署和扩缩容：** 有状态集确保 Pod 以严格、有序的方式（根据其序号索引）创建、扩缩容和终止。对于 Redis，这意味着可以管理主节点和副本实例，以确保数据完整性和一致的复制。

## 将 Redis 建模为有状态集

![](img/91c4bde832e221ba5bc429caf9587286_309_0.png)

Redis 作为有状态集

让我们将 Redis 建模为主从架构以实现更高的弹性。我们现在确信，这无法通过部署来实现。所有这些条件在 Redis 有状态集中将如何实现？

如果我们把 Redis 集群建模为一个包含 1 个主节点（例如 **redis-0**）和 2 个副本（例如 **redis-1**、**redis-2**）的 3 Pod 有状态集，Kubernetes 将确保：

- 1. 每个 Redis Pod 都绑定到自己的 PersistentVolume，即使 Pod 移动到另一个节点也能保留数据。
- 2. Pod 以有序的方式启动和终止，尊重 Redis 的复制和数据完整性要求。
- 3. 网络标识（主机名）保持一致，确保复制和客户端连接稳定且可预测。

这次我们想做一些改变。我们将不使用 Bitnami Redis 镜像。尽管可以使用该镜像构建有状态集，但我们必须在其设置中调整大量配置参数。我还想提一下，Bitnami 使得构建这个主从 Redis 集群变得轻而易举，但为了教学目的，我们将构建一个简化版本。此外，重点将主要放在建模有状态集上，而不是 Redis 配置（这本身就是一个独立的主题）。因此，我们将坚持使用官方 Redis 镜像。

官方 Redis 镜像不包含 **redis.conf**。我们必须创建一个最小版本并将其附加到 Pod，因为主节点和从节点 Redis 实例都需要 Redis 配置。正如我们稍后将了解到的，此文件将包含主节点 Pod 的网络标识（网络可访问服务名称的花哨说法），以便从节点实例知道谁是主节点以及如何连接到它。

我们还将编写一个脚本，该脚本将找出当前 Pod 是否是主节点，并相应地将主节点的网络标识附加到 **redis.conf** 文件中。之后，它将启动 Redis 进程。我们将把这两者都建模为配置映射。

```
masterauth SuperS3cret # (1)
requirepass SuperS3cret # (2)
bind 0.0.0.0 # (3)
protected-mode no # (4)
port 6379 # (5)
dir "/data" # (6)
```

(1) -> 设置从节点向主节点进行身份验证所需的密码。

(2) -> 配置客户端连接到 Redis 服务器所需的密码。

(3) -> 允许 Redis 在所有网络接口上接受连接。

(4) -> 禁用保护模式，允许来自任何 IP 地址的连接。

(5) -> 指定 Redis 服务器监听连接的端口。

(6) -> 设置 Redis 将写入持久数据文件的目录。

接下来是 Redis 初始化脚本。

```
#!/bin/sh
mkdir -p /etc/redis # (1)
cp /tmp/redis/redis.conf /etc/redis/redis.conf # (2)

POD_NAME=$(hostname) # (3)

POD_BASE_NAME=$(echo $POD_NAME | sed 's/-[0-9]*$//') # (4)
POD_INDEX=$(echo $POD_NAME | sed 's/^.*-//') # (5)

MASTER_POD_NAME="${POD_BASE_NAME}-0.redis" # (6)

REDIS_CONF="/etc/redis/redis.conf" # (7)

if [ "$POD_INDEX" = "0" ]; then # (8)
    echo "Starting as master..."
else # (9)
    echo "Setting up as replica of $MASTER_POD_NAME..."
    echo "replicaof $MASTER_POD_NAME 6379" >> $REDIS_CONF
fi

redis-server $REDIS_CONF  # (10)
```

(1) -> 如果 /etc/redis 目录不存在，则创建它。

(2) -> 将 redis.conf 文件从 /tmp/redis 复制到 /etc/redis。我们稍后将了解到，我们在有状态集中将此文件作为配置映射挂载。

(3) -> 将当前 Pod 的主机名存储在 **POD_NAME** 中。

(4) -> 通过移除数字后缀提取 Pod 的基本名称。因此，如果你的 Pod 名称是 **redis-1**，基本名称将是 **redis**。

(5) -> 从其主机名中检索 Pod 的索引号。对于 Pod 名称 **redis-1**，索引号将是 **1**。

(6) -> 通过将 **-0.redis** 附加到基本名称来构造主节点 Pod 的名称。**请注意**，如果这是一个部署，步骤 4、5 和 6 将无法实现，因为我们将无法获得可预测的 Pod 名称/网络标识符。

(7) -> 设置 Redis 配置文件的路径。

(8) -> 根据其索引检查当前 Pod 是否是第一个（主节点）。同样，如果此 Pod 是部署的一部分，这将无法实现。

(9) -> 对于第一个以外的 Pod，将它们配置为主节点的副本。

(10) -> 使用配置文件启动 Redis 服务器。

然后，我们继续创建有状态集。

```yaml
apiVersion: apps/v1
kind: StatefulSet # (1)
metadata:
  name: redis
spec:
  serviceName: "redis" # (2)
  replicas: 3 # (3)
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      initContainers:
      - name: volume-permissions # (4)
        image: busybox
        command: ['sh', '-c', 'chmod -R 777 /data']
        volumeMounts:
        - name: redis-data
          mountPath: /data
      containers:
      - name: redis
        image: redis:7.2.4 # (5)
        command: ['/scripts/redis-init.sh'] # (6)
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data # (7)
          mountPath: /data
        - name: redis-script # (8)
          mountPath: /scripts
        - name: redis-config # (9)
          mountPath: /tmp/redis/redis.conf
          subPath: redis.conf
      volumes:
      - name: redis-script # (10)
        configMap:
          name: redis-custom
          items:
          - key: redis-init.sh
            path: redis-init.sh
          defaultMode: 0755
      - name: redis-config # (11)
        configMap:
          name: redis-custom
          items:
          - key: redis.conf
            path: redis.conf
  volumeClaimTemplates: # (12)
  - metadata:
      name: redis-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 1Gi
```

(1) -> 将资源定义为有状态集，用于管理有状态应用程序。

(2) -> 指定将管理此有状态集的 Service 名称，用于网络标识。此有状态集中的每个 Pod 都将使用此名称作为基本名称，并结合序号来寻址其他 Pod。

(3) -> 将所需的 Pod 副本数设置为 3。在我们的例子中是 1 个主节点和 2 个从节点。

(4) -> 声明一个名为 volume-permissions 的初始化容器，用于更改 /data 目录的权限。

(5) -> 我们使用官方 Redis Docker 镜像，而不是 Bitnami 镜像，原因如上所述。

(6) -> 覆盖默认命令以运行用于 Redis 配置的自定义初始化脚本。

(7) -> 在 Redis 容器中的 /data 路径挂载一个名为 redis-data 的卷，用于持久存储。这在之前的 redis.conf 中已经配置过。

(8) -> 在 /scripts 挂载一个名为 redis-script 的配置映射卷，自定义初始化脚本位于此处。

(9) -> 将特定的 **redis.conf** 文件从 configMap 挂载到容器内的 **/tmp/redis/redis.conf**。

(10) -> 定义一个名为 redis-script 的卷，其数据源为 ConfigMap，其中包含 **redis-init.sh** 脚本。

(11) -> 定义另一个名为 redis-config 的卷，其数据源为同一个 ConfigMap，其中包含 redis.conf 文件。

(12) -> 指定用于动态卷配置的 PersistentVolumeClaim 模板，确保 StatefulSet 中的每个 Pod 都拥有自己的持久化存储。

## 关于持久卷声明模板

StatefulSet 规范中的 **volumeClaimTemplates** 部分是一个强大的功能，它能自动为 StatefulSet 中的每个 Pod 配置 PersistentVolume (PV)。这确保了每个 Pod 都拥有自己独特的、持久化的存储，即使 Pod 重启或重新调度，数据也能得以保留，这对于需要在重启和故障中维持状态的有状态应用至关重要。以下是 volumeClaimTemplates 工作原理及其重要性的详细说明：

## 持久化存储的自动化

**动态配置：** 当部署 StatefulSet 时，对于创建的每个 Pod，Kubernetes 会根据 volumeClaimTemplates 中定义的规格动态配置一个新的 PersistentVolume (PV)。这意味着你无需为有状态应用中的每个 Pod 手动创建 PV。

**一对一映射：** StatefulSet 中的每个 Pod 都会获得自己的 PersistentVolumeClaim (PVC)，该 PVC 又会绑定到一个唯一的 PV。这种一对一映射确保了存储在 PV 上的数据为该 Pod 所独占，即使 Pod 被创建和销毁，也能提供稳定的存储系统。

## 模板结构

**声明规范：** volumeClaimTemplates 将包含一个或多个 PVC 定义，这些定义指定了所需的存储类、访问模式（例如 ReadWriteOnce）和资源请求（例如存储大小）。这些定义被用作模板，为每个 Pod 创建实际的 PVC。

**稳定且唯一的存储：** 从 volumeClaimTemplates 创建的 PVC 会根据 Pod 的序数索引自动添加唯一后缀进行命名。这确保了每个 Pod 的存储是隔离且持久的，从而维护了应用的有状态特性。

**存储类使用：** 模板可以指定 storageClassName，以确定使用哪个 StorageClass 来配置 PV。这允许使用不同类型的存储（例如 SSD、标准磁盘驱动器）以及来自不同提供商（云、本地）的存储。

在 Redis StatefulSet 示例的上下文中，**volumeClaimTemplates** 部分确保 StatefulSet 中的每个 Redis Pod 都有自己的 PVC，命名为 **redis-data-\<pod-index\>**，该 PVC 又绑定到一个唯一配置的、至少具有 **1Gi** 存储空间、以 **ReadWriteOnce** 模式访问的 PV。

StatefulSet 中的 **volumeClaimTemplates** 功能对于在 Kubernetes 中管理有状态应用至关重要，它提供了一种可扩展且自动化的方式来处理持久化存储。它抽象了管理单个 PV 和 PVC 的复杂性，使开发人员能够专注于应用逻辑，而非底层基础设施。

## Headless 服务

在 Kubernetes 中 Redis StatefulSet 的上下文中包含一个 Service 条目，特别是 headless 服务，对于有状态应用及其发现机制有几个关键目的。

Kubernetes 中的 headless 服务不提供负载均衡或单个 IP 地址。相反，它允许通过 Pod 自身的 IP 地址直接访问它们。对于像 Redis 这样的有状态应用，每个实例可能需要单独访问（例如用于复制或分片），headless 服务为每个 Pod 提供了一个基于其主机名和服务名称的稳定 DNS 条目。

![](img/91c4bde832e221ba5bc429caf9587286_320_0.png)

## Headless 服务

这种稳定的网络标识对于 Redis 节点之间的通信至关重要，尤其是在集群或复制设置中，节点需要发现并连接到主节点或其他副本。

该服务充当集群内服务发现的一种方法。Pod 可以使用 DNS 查询来发现其他 Pod 的 IP 地址。例如，如果知道主节点的主机名，Redis 副本可以使用 DNS 发现主节点的 IP 地址。

在提供的示例中，如果你将服务命名为 **redis**，并且你的 StatefulSet Pod 的命名模式为 **redis-0**、**redis-1** 等，那么一个 Pod 可以查询 DNS 以获取 **redis-0.redis** 的 IP 地址，从而得到 **redis-0** Pod 的 IP 地址。这对于初始配置以及处理 Redis 集群内的故障转移或拓扑变更特别有用。

我们如下定义一个 headless 服务：

```
apiVersion: v1
kind: Service
metadata:
  name: redis # (1)
spec:
  ports:
    - port: 6379 # (2)
      targetPort: 6379
  clusterIP: None # (3)
  selector:
    app: redis # (4)
```

(1) -> 服务的名称。这是任意的。

(2) -> 我们希望暴露服务的端口。

(3) -> 此配置文件中最重要的行。当我们设置 **clusterIP** 为 **None** 时，Kubernetes 将其视为 headless 服务，并且不会为其分配任何 IP。

(4) -> 选择器标签应与我们的 statefulset 标签匹配。

让我们按顺序应用 configmap、statefulset 和 service。

```
$ kubectl create configmap redis-custom --from-file=redis.conf=redis.conf --from-file=redis-init.sh=redis-init.sh -n robust-setup
configmap/redis-custom created
```

```
$ kubectl apply -f redis-sts.yaml -n robust-setup
statefulset.apps/redis created
```

```
$ kubectl apply -f redis-headless-svc.yaml -n robust-setup
service/redis created
```

你会注意到 statefulset 的 Pod 是逐个启动的，而不是同时启动。

```
$ kubectl get pod -n robust-setup --watch
NAME      READY   STATUS              RESTARTS   AGE
redis-0   0/1     Init:0/1            0          4s
redis-0   0/1     PodInitializing     0          12s
redis-0   1/1     Running             0          13s
redis-1   0/1     Pending             0          0s
redis-1   0/1     Pending             0          3s
redis-1   0/1     Init:0/1            0          3s
```

```
$ kubectl get pod -n robust-setup
NAME      READY   STATUS    RESTARTS   AGE
redis-0   1/1     Running   0          119s
redis-1   1/1     Running   0          106s
redis-2   1/1     Running   0          92s
```

第一个序号 **0** 的 Pod 最先启动，然后是 **1** 和 **2**。这是 statefulset 的固有属性。让我们检查日志，看看为什么这在 Redis 的上下文中很重要。

```
$ kubectl logs -f redis-0 -n robust-setup # (1)
Defaulted container "redis" out of: redis, volume-permissions (init)
Starting as as master... # (2)
17:C 29 Feb 2024 06:23:42.714 * oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
17:C 29 Feb 2024 06:23:42.714 * Redis version=7.2.4, bits=64, commit=00000000, modified=0, pid=17, just started
17:C 29 Feb 2024 06:23:42.714 * Configuration loaded
17:M 29 Feb 2024 06:23:42.715 * monotonic clock: POSIX clock_gettime
17:M 29 Feb 2024 06:23:42.715 * Running mode=standalone, port=6379.
17:M 29 Feb 2024 06:23:42.716 * Server initialized
17:M 29 Feb 2024 06:23:42.716 * Ready to accept connections tcp
17:M 29 Feb 2024 06:23:56.580 * Replica 10.244.0.20:6379 asks for synchronization # (3)
17:M 29 Feb 2024 06:23:56.580 * Full resync requested by replica 10.244.0.20:6379
17:M 29 Feb 2024 06:23:56.580 * Replication backlog created, my new replication IDs are 'c23ad16898eaf58472ee2dd577d908119df94c07' and '0000000000000000000000000000000000000000'
17:M 29 Feb 2024 06:23:56.580 * Delay next BGSAVE for diskless SYNC
17:M 29 Feb 2024 06:24:01.787 * Starting BGSAVE for SYNC with target: replicas sockets
17:M 29 Feb 2024 06:24:01.788 * Background RDB transfer started by pid 22
22:C 29 Feb 2024 06:24:01.789 * Fork CoW for RDB: current 0 MB, peak 0 MB, average 0 MB
17:M 29 Feb 2024 06:24:01.789 * Diskless rdb transfer, done reading from pipe, 1 replicas still up.
17:M 29 Feb 2024 06:24:01.796 * Background RDB transfer terminated with success
17:M 29 Feb 2024 06:24:01.796 * Streamed RDB transfer with replica 10.244.0.20:6379 succeeded (socket). Waiting for REPLCONF ACK from replica to enable streaming
17:M 29 Feb 2024 06:24:01.796 * Synchronization with replica 10.244.0.20:6379 succeeded
17:M 29 Feb 2024 06:24:09.700 * Replica 10.244.0.189:6379 asks for synchronization # (4)
17:M 29 Feb 2024 06:24:09.700 * Full resync requested by replica 10.244.0.189:6379
17:M 29 Feb 2024 06:24:09.700 * Delay next BGSAVE for diskless SYNC
17:M 29 Feb 2024 06:24:14.842 * Starting BGSAVE for SYNC with target: replicas sockets
17:M 29 Feb 2024 06:24:14.843 * Background RDB transfer started by pid 23
23:C 29 Feb 2024 06:24:14.845 * Fork CoW for RDB: current 0 MB, peak 0 MB, average 0 MB
17:M 29 Feb 2024 06:24:14.845 * Diskless rdb transfer, done reading from pipe, 1 replicas still up.
17:M 29 Feb 2024 06:24:14.850 * Background RDB transfer terminated with success
17:M 29 Feb 2024 06:24:14.850 * Streamed RDB transfer with replica 10.244.0.189:6379 succeeded (socket). Waiting for REPLCONF ACK from replica to enable streaming
```

等待来自副本的 REPLCONF ACK 以启用流式传输

17:M 29 Feb 2024 06:24:14.850 * 与副本 10.244.0.189:6379 的同步成功

(1) -> 我们正在检查主 Redis 的启动过程。我们在初始化脚本中指定 **0** 作为主 Redis。

(2) -> 这行由脚本打印，因为它是 **redis-0**。

(3) -> 另一个副本 **redis-1** 通过其网络标识符 **redis-0.redis** 发现主节点并请求同步。如果我们在非主 Pod 中运行脚本，我们会在 **redis.conf** 文件中添加此标识。

(4) -> **redis-2** 中也发生了相同的事件。

让我们检查其中一个副本的日志。

```
$ kubectl logs -f redis-1 -n robust-setup # (1)
Defaulted container "redis" out of: redis, volume-permissions (init)
Setting up as replica of redis-0.redis... # (2)
16:C 29 Feb 2024 06:23:56.533 * oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
16:C 29 Feb 2024 06:23:56.533 * Redis version=7.2.4, bits=64, commit=00000000, modified=0, pid=16, just started
16:C 29 Feb 2024 06:23:56.533 * Configuration loaded
16:S 29 Feb 2024 06:23:56.533 * monotonic clock: POSIX clock_gettime
16:S 29 Feb 2024 06:23:56.533 * Running mode=standalone, port=6379.
16:S 29 Feb 2024 06:23:56.534 * Server initialized
16:S 29 Feb 2024 06:23:56.534 * Ready to accept connections tcp
16:S 29 Feb 2024 06:23:56.535 * Connecting to MASTER redis-0.redis:6379 # (3)
16:S 29 Feb 2024 06:23:56.541 * MASTER <-> REPLICA sync started
16:S 29 Feb 2024 06:23:56.545 * Non blocking connect for SYNC fired the event.
16:S 29 Feb 2024 06:23:56.547 * Master replied to PING, replication can continue...
16:S 29 Feb 2024 06:23:56.552 * Partial resynchronization not possible (no cached master)
16:S 29 Feb 2024 06:24:01.761 * Full resync from master: c23ad16898eaf58472ee2dd577d908119df94c07:0
16:S 29 Feb 2024 06:24:01.763 * MASTER <-> REPLICA sync: receiving streamed RDB from master with EOF to disk
16:S 29 Feb 2024 06:24:01.763 * MASTER <-> REPLICA sync: Flushing old data
16:S 29 Feb 2024 06:24:01.763 * MASTER <-> REPLICA sync: Loading DB in memory
16:S 29 Feb 2024 06:24:01.768 * Loading RDB produced by version 7.2.4
16:S 29 Feb 2024 06:24:01.768 * RDB age 0 seconds
16:S 29 Feb 2024 06:24:01.768 * RDB memory usage when created 0.90 Mb
16:S 29 Feb 2024 06:24:01.769 * Done loading RDB, keys loaded: 0, keys expired: 0.
16:S 29 Feb 2024 06:24:01.769 * MASTER <-> REPLICA sync: Finished with success  # (4)
```

(1) -> 检查其中一个副本日志的命令。

(2) -> 如果 Pod 是非主 Pod，则由脚本写入。

(3) -> 副本连接到主节点。这大约发生在主节点在之前的日志中收到同步请求的同时。

(4) -> 同步成功完成。

我们还可以观察到每个 Redis Pod 都分配了自己的 PVC。这是由 StatefulSet 自动管理的。

```
$ kubectl get pvc -n robust-setup
NAME                  STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS       VOLUMEATTRIBUTESCLASS   AGE
redis-data-redis-0    Bound    pvc-952d1749-c2bb-4101-a66b-90ea5fdf1e5e   1Gi        RWO            do-block-storage   <unset>                 3m23s
redis-data-redis-1    Bound    pvc-28584793-a096-4366-b8d2-a30e8068e02f   1Gi        RWO            do-block-storage   <unset>                 3m10s
redis-data-redis-2    Bound    pvc-f0612a6c-f43d-433a-89a9-75f798df401f   1Gi        RWO            do-block-storage   <unset>                 2m56s
```

关于 StatefulSet 的最后一个观察点是 Kubernetes 如何在所有节点之间均匀分配 Pod。

```
$ kubectl get pod -n robust-setup -o wide
NAME      READY   STATUS    RESTARTS   AGE    IP           NODE                             NOMINATED NODE   READINESS GATES
redis-0   1/1     Running   0          2m53s  10.244.1.42  first-cluster-default-pool-oeo85   <none>           <none>
redis-1   1/1     Running   0          2m40s  10.244.0.20  first-cluster-default-pool-oeo8k   <none>           <none>
redis-2   1/1     Running   0          2m26s  10.244.0.189 first-cluster-default-pool-oeo8s   <none>           <none>
```

## 练习 29：扩展 Redis StatefulSet。

扩展 Redis StatefulSet 的规模。观察并记录对 Redis 数据和 Pod 命名约定的影响。

## 练习 30：使用 Redis Sentinel 为 Redis 设置高可用性。

在 Redis StatefulSet 旁边部署 Redis Sentinel，以监控主节点并在主节点故障时执行自动故障转移。记录部署过程和故障转移测试。

## 其他上下文中的 StatefulSet

上述示例展示了 StatefulSet 在高可用 Redis 主从设置中的概念。除了 Redis，Kubernetes 中的 StatefulSet 旨在管理各种用例中的有状态应用程序和服务，特别是数据库，这些数据库从 StatefulSet 提供的特定功能中受益匪浅。

这些功能包括为每个 Pod 提供稳定、唯一的网络标识符，确保可预测和一致的寻址；稳定、持久的存储，允许每个 Pod 永久连接到其存储，在 Pod 重新调度和重启时保留数据；有序、优雅的部署和扩展，这对于管理依赖关系和维护有状态应用程序的完整性至关重要；以及有序、自动化的滚动更新，允许无缝、最小停机时间的更新。这些能力对于任何有状态服务都至关重要，确保复杂、依赖状态的应用程序（如数据库）在应用程序的整个生命周期中保持其状态，提高可靠性和运营效率。

例如，对于 PostgreSQL，使用 StatefulSet 实现高可用性（HA）涉及利用这些功能来确保数据库实例得到一致且可靠的管理。典型的设置可能包括一个由一个 StatefulSet 管理的主 PostgreSQL 实例和多个由另一个 StatefulSet 管理的副本，或者根据架构全部在单个 StatefulSet 中。PersistentVolume 确保在 Pod 被移动或重启时数据不会丢失。稳定的网络标识符使得配置和维护主实例和副本实例之间的复制更加容易。Kubernetes 服务可用于公开数据库实例，并将流量管理到主节点进行写入，副本节点进行读取，从而提高性能。

## 服务账户和 RBAC

Kubernetes 被设计为一个容器编排系统。其重点在于管理集群内的资源，而不是管理单个用户账户。因此，Kubernetes 没有用户账户的概念。嗯，至少不是直接的。可以将用户身份验证委托给外部系统。Kubernetes 依赖外部机制（如 OpenID Connect、LDAP、x509 证书等）来对用户进行身份验证。这种设计允许 Kubernetes 与各种身份验证提供者集成，而无需直接管理用户账户。

在许多情况下，我们需要一个非人类用户账户来对集群执行特定操作，例如在 Kubernetes 中运行 CI/CD 流水线，或者一个 Pod 与其运行所在的同一集群内的 Kubernetes 资源进行交互。这些特殊类型的账户称为服务账户。

服务账户（以及用户账户）有两个特点：

**作用域访问：** 服务账户的作用域限定于特定的命名空间，或者可以配置为集群范围，从而更容易管理集群内运行的应用程序和进程的权限和访问控制。

**细粒度权限：** Kubernetes 使用基于角色的访问控制（RBAC）来定义经过身份验证的用户或服务账户可以在集群内做什么。Role 和 ClusterRole 定义权限（如创建、列出或删除资源），而 RoleBinding 和 ClusterRoleBinding 将这些权限分配给用户、组或服务账户。

## 示例

### 示例 1：与 Kubernetes API 通信的内部应用程序

需要查询 Kubernetes API 以发现其他 Pod、服务或配置的集群内应用程序可以使用服务账户来认证其请求。例如，运行在集群内从其他 Pod 收集指标的监控工具可以使用一个具有跨命名空间读取 Pod 指标权限的服务账户。

### 示例 2：CI/CD 流水线自动化

将应用程序部署到 Kubernetes 集群的持续集成/持续部署（CI/CD）工具将使用服务账户来创建或更新部署和服务。此服务账户将被授予管理特定命名空间或整个集群内资源的必要权限，具体取决于 CI/CD 工具操作的范围。

### 示例 3：访问 Kubernetes 资源的外部应用程序

需要与 Kubernetes 集群交互的外部应用程序（例如，在集群外运行的仪表板或管理工具）也可以使用服务账户。你可以安全地公开必要的 API 端点，并让外部应用程序使用具有所需权限的服务账户的令牌进行身份验证。

## 创建服务账户

我们将只关注服务账户，因为它足以在生产环境中部署一个可运行的应用程序。处理人类用户和 Kubernetes 是一个庞大的主题，值得单独写一本书。

服务账户是 Kubernetes 中的一个命名空间资源。这意味着服务账户是在特定命名空间的范围内定义和操作的。每个服务账户都绑定到一个命名空间，可用于授予在该命名空间内操作的权限，从而实现细粒度的访问控制，并限制 Pod 中运行的进程可以访问或修改的范围。

创建服务账户是一个简单的命令。

```
$ kubectl create serviceaccount test -n robust-setup
serviceaccount/test created
```

就这样。我们在 **robust-setup** 命名空间中创建了一个服务账户。但目前，这是一个没有权限的实体。它没有告诉我们这个服务账户能做什么或不能做什么。现在让我们来定义这部分。首先，我们必须定义它可以拥有什么样的“角色”。为此，我们创建一个 **Role** 资源对象。

**Role** 是一个 Kubernetes 资源，它指定了代表一组权限的规则。这些权限包括对 Pod、Service、Deployment 等资源执行 get、list、create、update、delete 等操作。Role 中的规则决定了允许对哪些资源执行哪些操作，从而提供对命名空间内访问的细粒度控制。

这是一个允许对 Pod 进行读取操作的简单角色。

```
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader # (1)
rules:
- apiGroups: [""] # (2)
  resources: ["pods"] # (3)
  verbs: ["get", "watch", "list"] # (4)
```

(1) -> 角色的名称。应具有意义。

(2) -> 角色控制的资源所属的 API 组。我们定义了一个空字符串 `""`，这表示核心 API 组。Pod、Service、Deployment 等都属于这个组。

(3) -> 权限适用的资源列表。目前，只有 Pod。

(4) -> 我们可以对资源执行的操作。

我们在 **robust-setup** 命名空间中创建这个角色。

```
$ kubectl apply -f pod-reader-role.yaml -n robust-setup
role.rbac.authorization.k8s.io/pod-reader created
```

我们现在有了一个服务账户和一个角色。但 Kubernetes 并不知道它需要将 **pod-reader** 角色“绑定”到 **test** 服务账户。为此，我们创建一个 **RoleBinding** 资源对象。

这是一个将 **pod-reader** Role 绑定到名为 **test** 的服务账户的 **RoleBinding** 示例：

```
apiVersion: rbac.authorization.k8s.io/v1 # (1)
kind: RoleBinding # (2)
metadata:
  name: read-pods # (3)
subjects:
- kind: ServiceAccount # (4)
  name: test # (5)
roleRef:
  kind: Role # (6)
  name: pod-reader # (7)
  apiGroup: rbac.authorization.k8s.io # (8)
```

(1) -> 指定 RBAC 授权 API 组中 RoleBinding 资源的 API 版本。

(2) -> 将资源类型定义为 RoleBinding，它将 Role 中定义的权限授予一组主体。

(3) -> 角色绑定的名称。

(4) -> 标识 Role 被绑定到的主体类型；在本例中是 ServiceAccount。其他可能的值：**User**、**Group**，用于将角色绑定到单个用户或组。我们不会在本书中讨论这些。

(5) -> 指定 Role 绑定到的 ServiceAccount（**test**）的名称。

(6) -> 指示引用的角色类型；这里是一个用于命名空间范围权限的 Role。正如我们将在本节后面看到的，这也可以是 **ClusterRole**。

(7) -> 定义被授予权限的 Role（**pod-reader**）的名称。

(8) -> 指定被引用 Role 的 API 组，确认它是 RBAC API 组的一部分。通常，对于 RoleBinding，这始终是 **rbac.authorization.k8s.io**，因为它引用的是 RBAC 资源。

服务账户与 RBAC

让我们创建这个资源对象。

```
$ kubectl apply -f pod-reader-role-binding.yaml -n robust-setup
rolebinding.rbac.authorization.k8s.io/read-pods created
```

现在让我们来测试一下这个服务账户。为此，我们必须以该服务账户的身份发出 kubectl 命令。Kubernetes 使用令牌来帮助你做到这一点。

Kubernetes JSON Web Token (JWT) 是一种紧凑、URL 安全的方式，用于表示要在双方之间传输的声明。在 Kubernetes 的上下文中，JWT 被广泛用于身份验证，特别是与服务账户一起使用。

## Kubernetes JWT 的组成部分

JWT 通常由三个部分组成，用点（.）分隔：

**Header（头部）：** 头部通常由两部分组成：令牌类型（即 JWT）和使用的签名算法，例如 HMAC SHA256 或 RSA。

**Payload（负载）：** 负载包含声明。声明是关于实体（通常是用户）和附加元数据的陈述。在 Kubernetes 中，负载通常包含有关服务账户的信息（如其名称和命名空间）、令牌的颁发者以及访问范围。

**Signature（签名）：** 要创建签名部分，需要使用编码后的头部、编码后的负载、一个密钥以及头部中指定的算法来对令牌进行签名。这确保了令牌在颁发后未被篡改。

## 在 Kubernetes 中的使用

**身份验证：** Kubernetes 使用 JWT 对服务账户进行身份验证以访问 Kubernetes API。当与服务账户关联的 Pod 向 Kubernetes API 发出请求时，它会在 HTTP 请求的 Authorization 头中附加 JWT 令牌作为持有者令牌。

**服务账户：** Kubernetes 中的每个服务账户都会自动生成并分配一个 JWT 令牌。可以在 Pod 内的 `/var/run/secrets/kubernetes.io/serviceaccount/token` 路径找到此令牌，Pod 内运行的应用程序使用它来向 Kubernetes API 进行身份验证。

**安全性：** JWT 令牌允许 Kubernetes 验证请求者的身份，而无需每次都查询外部身份验证服务。JWT 的签名确保了令牌未被篡改。

让我们为我们的服务账户生成一个令牌。

```
$ kubectl create token test --duration=600s -n robust-setup > token.txt
```

此命令为 **test** 服务账户创建一个有效期为 10 分钟的令牌（在撰写本文时，这是我们可以设置的最小持续时间值）。生成的令牌被写入 **token.txt** 文件。

让我们解码这个 JWT 令牌并了解更多关于它的信息。你可以通过将令牌粘贴到 [jwt.io](https://jwt.io) 来自行解码。

头部部分指明了它使用的算法。

```
{
  "alg": "RS256",
  "kid": "quaAkoqCy2G-lp-YZYGv1UCJZnW1EcPZiCrDb197uZo"
}
```

负载部分更值得关注。

```
{
  "aud": [
    "system:konnectivity-server"
  ],
  "exp": 1709302847, // (1)
  "iat": 1709302247,
  "iss": "https://kubernetes.default.svc.cluster.local", // (2)
  "kubernetes.io": {
    "namespace": "robust-setup", // (3)
    "serviceaccount": {
      "name": "test", // (4)
      "uid": "67a83c4d-e55d-4c4d-821e-0f5acbff094e"
    }
  },
  "nbf": 1709302247,
  "sub": "system:serviceaccount:robust-setup:test" // (5)
}
```

(1) -> 令牌的过期时间（exp），指示令牌何时失效（Unix 时间戳）。

(2) -> 令牌的颁发者（iss），标识颁发令牌的机构；这里是 Kubernetes API 服务器。

(3) -> 指定与此令牌关联的服务账户所在的 Kubernetes 命名空间。

(4) -> 令牌颁发给的 Kubernetes 服务账户的名称。

(5) -> 令牌的主题（sub），通常标识作为 JWT 主体的实体；这里它以 `system:serviceaccount:<namespace>:<service account name>` 的格式表示服务账户。

Kubernetes API 服务器是这些令牌的主要消费者。当 Pod 或外部实体向 Kubernetes API 发出请求时，它可以使用服务账户令牌进行身份验证。API 服务器验证令牌以确保其有效且未过期，然后使用令牌中的信息（如服务账户名称和命名空间）来执行基于与该服务账户关联的角色绑定的访问控制策略。

让我们使用这个令牌向我们的集群发出请求。

```
$ export KUBE_TOKEN=$(cat token.txt) # (1)
$ kubectl --token=$KUBE_TOKEN get pods # (2)
Error from server (Forbidden): pods is forbidden: User "system:serviceaccount:robust-setup:test" cannot list resource "pods" in API group "" in the namespace "default"
```

## 练习 31：更新角色以获取服务

除了 Pod 之外，增加角色/令牌的范围以包含服务。

每次令牌过期都重新创建确实非常不便。因此 Kubernetes 允许我们创建长期有效的令牌。

```yaml
apiVersion: v1
kind: Secret # (1)
metadata:
  name: test-llt # (2)
  annotations:
    kubernetes.io/service-account.name: test # (3)
type: kubernetes.io/service-account-token # (4)
```

(1) -> 这将是一个 **Secret** 资源。
(2) -> 我们 Secret 的名称。
(3) -> 我们为 Secret 添加注解，以告知 Kubernetes 此 Secret 将为 **test** 服务账户生成并持有令牌。
(4) -> 此 Secret 的类型为 **kubernetes.io/service-account-token**。如果还记得，我们之前讨论过 Kubernetes 中 Secret 有不同的类型。这就是其中一种。

## Kubernetes 注解

Kubernetes 注解是一种灵活且可选的机制，允许你将任意的非标识性元数据附加到 Kubernetes 对象上。与用于组织和选择 Kubernetes 内对象和资源的标签不同，注解不用于标识和选择资源。注解可以比标签容纳更大、更复杂的数据，非常适合存储可能被与 Kubernetes 对象交互的工具、库、客户端或其他外部来源使用的附加信息。

### 注解的关键特性

- **任意元数据：** 注解可以存储任意元数据，这些元数据可以包括关于对象的操作信息，例如构建/发布时间戳、客户端库信息、调试数据或负责该人员的联系信息等。
- **不用于选择：** 与标签不同，注解不用于查询或组织资源集合。它们纯粹是信息性的。
- **更大的数据容量：** 注解可以比标签容纳更大量的数据，使其适合存储序列化的 JSON 或 YAML 数据、冗长的描述和其他非紧凑数据。
- **工具集成：** 它们通常被工具和扩展用于存储配置数据、定义或 Kubernetes 核心资源不直接支持的命令选项。

注解与标签一样，是键值对。它们在 Kubernetes 对象定义的 metadata 部分中定义：

```yaml
metadata:
  annotations:
    example.com/annotation: "This is an example annotation"
    buildInfo: |
      {
        "version": "v1.2.3",
        "buildDate": "2024-03-01T12:00:00Z"
      }
```

以上代码是一个包含构建信息的注解示例。

在我们用于服务账户的 Secret 示例中，Kubernetes API 服务器是此注解的主要消费者。当创建类型为 **kubernetes.io/service-account-token** 的 Secret 时，API 服务器会读取注解以将 Secret 与指定的服务账户（本例中为 **test**）关联起来。这种关联对于 Kubernetes 中服务账户令牌的自动管理至关重要。

### 注解的用例

- **跟踪信息：** 注解可用于跟踪部署所用的工具、构建信息、时间戳、发布信息等。
- **配置和管理：** 一些 Kubernetes 组件和第三方工具使用注解来配置行为、集成附加功能或管理特定于组件的信息。例如，Ingress 控制器通常使用注解来配置 SSL、负载均衡器行为和身份验证。
- **开发和调试：** 开发人员和运维人员可以使用注解添加调试信息、注释或外部引用（如工单 ID 或文档链接），这些信息有助于理解某些资源的用途和管理。

当我们创建 Secret 并查看它时，

```bash
$ kubectl apply -f long-lived-token.yaml -n robust-setup
secret/test-llt created
```

```bash
$ kubectl describe secret test-llt -n robust-setup
Name:         test-llt
Namespace:    robust-setup
Labels:       <none>
Annotations:  kubernetes.io/service-account.name: test
              kubernetes.io/service-account.uid: 67a83c4d-e55d-4c4d-821e-0f5acbff094e

Type:  kubernetes.io/service-account-token

Data
====
ca.crt:     1155 bytes
namespace:  12 bytes
token:      eyJhbGciOiJSUzI1NiIsImtpZCI6InF1YUFrb3FDeTJHLWxwLVlaWUd2MVVDSlpuVzFFY1BaaUNyRGIxOTd1Wm8ifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJyb2J1c3Qtc2V0dXAiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlY3JldC5uYW1lIjoidGVzdC1sbHQiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoidGVzdCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6IjY3YTgzYzRkLWU1NWQtNGM0ZC04MjFlLTBmNWFjYmZmMDk0ZSIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpyb2J1c3Qtc2V0dXA6dGVzdCJ9.pyskXzOFwQkVVcdsENfeuuuze0dNCaX38ePIDSbIwUdaMUDE-jKSm7XyuBRee_w6r3K8fPdCFAUjXyQ7wJocr3XWOyjzmyVJ3gf-gSgK7OjKsPe5JbcvlOyo8Q0zI7UZZ7CydKgcI-hfTMNpUYWeBXOyLOAaSSenkANWB5Dqwftg_BRKwE6-UQRXeOTkVBHNoOZ9Nut7eh6yIB_H_5EsswYk-bLgyBTrHxVPOXds_bTNctmimJOU2MRzl6S8Y842MKIgsBH-KkPYXeuk_i6oZ_HNlh8t1f3tDOnnuSis-nlPOm-fcgL1Lm20sA8LlG-j8nSdoWSOcUEZ1NwIuNIx4q21EA
```

我们可以看到 Kubernetes 已经将一个令牌附加到了这个 Secret 上。我们可以像之前的步骤一样使用这个令牌来运行 kubectl 命令。

## 练习 32：创建长期有效的令牌以访问 Pod 日志

你会注意到我们无法使用此角色访问 Pod 的日志。更改角色，以便我们可以使用为此角色创建的长期有效令牌来访问 Pod 日志。

## Pod 内部的服务账户

在许多场景中，我们希望将服务账户与 Pod 关联起来。Kubernetes 使用关联的服务账户自动将服务账户令牌和相关凭证（CA 证书和命名空间）挂载到 Pod 中。这消除了手动管理凭证的需要，使得构建和部署安全的云原生应用程序变得更加容易。

这在两种主要场景中非常有用。

- **身份验证：** 服务账户为 Pod 在向 Kubernetes API 发出请求时提供身份，从而实现安全且经过身份验证的 API 访问。这对于需要与 Kubernetes 系统交互以读取或修改资源的应用程序至关重要。
- **将 Secret 与 Pod 关联：** 当我们创建服务账户时，我们可以在 YAML 文件中将现有的 Secret 添加到该账户。然后，我们将服务账户与 Pod 关联起来。这种关联对于自动化将凭证提供给在 Pod 内运行的应用程序的过程特别有用，无需硬编码敏感信息。Kubernetes 使用指定的服务账户自动将这些 Secret 挂载到 Pod 中，提供了一种安全的方式来存储和管理敏感信息，如数据库密码、API 密钥和 SSH 密钥。

这是一个带有 Secret 的服务账户示例。

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-service-account
secrets:
- name: my-secret
```

我们将看到一个更详细的例子。在此之前，我们将一个 Pod 与 **test** 服务账户关联起来，并检查会发生什么。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: service-account-test-pod # (1)
spec:
  containers:
```

## 强化角色

在为服务账户分配角色时，始终遵循[最小权限原则](https://www.paloaltonetworks.com/cyberpedia/what-is-the-principle-of-least-privilege)是一个好习惯，即不要授予超出完成工作所需的权限。例如，如果我们希望一个服务账户对服务、部署、配置映射、Pod、密钥和持久卷声明执行CRUD操作，我们应该只明确授予这些资源的权限。如果我想通过数据库团队处理有状态集，那么我会给他们一个单独的服务账户，该账户可以访问Pod、持久卷声明、配置映射/密钥和有状态集，仅此而已。

有时，我们希望授予对大量资源的访问权限。Kubernetes允许你使用通配符（*）来指定权限，这可能会让你自掘坟墓（尽管我仍然建议编写详细的规则）。以下是一个你**不**想在家尝试的示例。

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: megalomaniac-villain
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
```

简而言之；Kubernetes允许在角色中使用*，请谨慎使用。

有些情况下，我们希望授予对某些资源的集群范围访问权限，而不是将其限制在某个命名空间内。这就需要一种单独的资源类别，称为**ClusterRole**。它们与**Role**相同，但是角色的集群范围对应物。因为它们是集群范围的，所以不受任何命名空间的限制。

**ClusterRole**主要可以通过**ClusterRoleBinding**与服务账户关联。**ClusterRole**定义了整个集群的权限，而**ClusterRoleBinding**则是在集群级别将这些权限授予用户、组或服务账户。

Kubernetes自带了几个内置的**ClusterRoles**和**Roles**，它们是预定义的，为各种集群操作提供了通用的权限集。这些内置角色旨在通过提供现成的权限集来简化访问控制的管理，这些权限集涵盖了与Kubernetes资源交互的常见用例。

一些值得注意的内置ClusterRoles包括：

**cluster-admin：** 授予超级用户权限，可以在任何资源上执行任何操作。它提供对集群中每个资源和所有命名空间的完全控制。

**admin：** 允许管理员访问，旨在通过RoleBinding在命名空间内授予。它允许对命名空间中的大多数资源进行读/写访问，包括在该命名空间内创建角色和角色绑定的能力。它不允许对资源配额或命名空间本身进行写访问。

**edit：** 允许对命名空间中的大多数对象进行读/写访问。它不允许查看或修改角色或角色绑定。

**view：** 允许只读访问以查看命名空间中的大多数对象。它不允许查看角色或角色绑定。它不允许查看密钥，因为这些被认为封装了敏感信息。

**system:node：** 授予节点（kubelet）权限，允许它们与API服务器通信并管理其运行节点上的Pod资源。

除了**ClusterRoles**，还有特定于命名空间的Roles，它们在Kubernetes中也是预定义的。这些角色包含为管理单个命名空间内的资源而量身定制的权限，对于在该命名空间范围内授予权限非常有用。

这些内置角色可以通过**RoleBindings**或**ClusterRoleBindings**分配给集群中的用户、组或服务账户，具体取决于你希望权限在集群范围内还是在特定命名空间内生效。

例如，要授予用户在特定命名空间中的管理员权限，你可以在该命名空间中创建一个引用admin **ClusterRole**的**RoleBinding**：

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: namespace-admin
  namespace: mynamespace
subjects:
- kind: ServiceAccount
  name: mynamespace-admin
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: admin
  apiGroup: rbac.authorization.k8s.io
```

使用内置角色的主要优势在于它们提供的便利性和最佳实践。它们由Kubernetes项目维护，旨在遵循最小权限原则，仅授予集群内常见任务和角色所需的权限。这有助于简化RBAC设置，并默认确保更高水平的安全性。

## 将Pod分散到各个节点

想象一个场景，你的集群在微服务架构中托管多个应用程序，多个节点容纳你的应用程序Pod。虽然调度器确保你的Pod被放置在健康且资源充足的节点上，但它并不考虑这些Pod在节点上的分布情况。

属于特定部署的Pod很有可能全部被调度到单个节点上。这种情况的明显问题是，当该节点宕机时，你的用户会经历停机时间，因为没有可用的Pod来处理请求。

Kubernetes 为我们提供了一种将 Pod 均匀分布到各个节点上的方法，从而避免单点故障。你的集群中所有节点同时宕机的可能性非常小，对吧？

我们将此作为 Pod/部署规范的一部分进行指定，称为 **podAntiAffinity**。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3 # (1)
  selector:
    matchLabels:
      app: my-app # (2)
  template:
    metadata:
      labels:
        app: my-app # (3)
    spec:
      affinity: # (4)
        podAntiAffinity: # (5)
          requiredDuringSchedulingIgnoredDuringExecution: # (6)
          - labelSelector:
            matchExpressions: # (7)
            - key: app
              operator: In
              values:
              - my-app
            topologyKey: "kubernetes.io/hostname" # (8)
      containers:
      - name: server
        image: nginx
```

(1) -> 指定部署所需的副本 Pod 数量。

(2) -> 定义用于识别属于此部署的 Pod 的标签选择器。

(3) -> 设置从此模板创建的 Pod 的标签，用于与部署选择器进行匹配。

(4) -> 开始指定关于 Pod 放置的调度偏好。

(5) -> 指定使用 Pod 反亲和性规则来影响 Pod 调度决策。

(6) -> 表示在调度期间必须考虑反亲和性规则，但在 Pod 放置后可能会被忽略。

(7) -> 定义标签选择器应如何匹配现有 Pod 的标签。

(8) -> 指定系统用于表示应用 Pod 反亲和性规则边界的节点标签键。

让我们应用此部署并检查 Pod 的调度情况。

```bash
$ kubectl apply -f pod-anti-affinity.yaml -n robust-setup
deployment.apps/my-deployment created
```

```bash
$ kubectl get pod -n robust-setup -l app=my-app -o wide
NAME                          READY   STATUS    RESTARTS   AGE   IP           NODE                             NOMINATED NODE   READINESS GATES
my-deployment-bbb987cc-l8mm5  1/1     Running   0          19s   10.244.1.90  first-cluster-default-pool-oeo85 <none>           <none>
my-deployment-bbb987cc-nzcfx  1/1     Running   0          19s   10.244.0.234 first-cluster-default-pool-oeo8s  <none>           <none>
my-deployment-bbb987cc-p6q2q  1/1     Running   0          19s   10.244.0.1   first-cluster-default-pool-oeo8k <none>           <none>
```

我们可以看到每个 Pod 都被放置在不同的节点上。如果副本数多于集群中的节点数会发生什么？让我们通过增加部署数量来运行这个场景。

```bash
$ kubectl scale deployment my-deployment --replicas=5 -n robust-setup
```

几秒钟后，

```bash
$ kubectl get pod -n robust-setup -l app=my-app -o wide
NAME                          READY   STATUS    RESTARTS   AGE   IP           NODE                             NOMINATED NODE   READINESS GATES
my-deployment-bbb987cc-5jz9b  0/1     Pending   0          3s    <none>       <none>                           <none>           <none>
my-deployment-bbb987cc-l8mm5  1/1     Running   0          7m40s  10.244.1.90   first-cluster-default-pool-oeo85   <none>           <none>
my-deployment-bbb987cc-nzcfx  1/1     Running   0          7m40s  10.244.0.234   first-cluster-default-pool-oeo8s   <none>           <none>
my-deployment-bbb987cc-p6q2q  1/1     Running   0          7m40s  10.244.0.1     first-cluster-default-pool-oeo8k   <none>           <none>
my-deployment-bbb987cc-s7k2v  0/1     Pending   0          3s     <none>         <none>                              <none>           <none>
```

我们可以看到两个新 Pod 处于 **Pending** 状态。

当我们描述其中一个 Pod 时，我们发现调度器无法为该 Pod 分配节点，因为反亲和性规则阻止了它这样做。

```bash
$ kubectl describe pod my-deployment-bbb987cc-5jz9b -n robust-setup
Name:         my-deployment-bbb987cc-5jz9b
Namespace:    robust-setup
Priority:     0
....
Events:
  Type     Reason            Age   From               Message
  ----     ------            ----  ----               -------
  Warning  FailedScheduling  21s   default-scheduler  0/3 nodes are available: 3 node(s) didn't match pod anti-affinity rules. preemption: 0/3 nodes are available: 3 No preemption victims found for incoming pod.
```

Kubernetes 允许我们使用 **preferredDuringSchedulingIgnoredDuringExecution** 规范稍微放宽 Pod 反亲和性规则。我们稍微修改一下部署。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 5 # (1)
  ...
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100 # (3)
        podAffinityTerm: # (4)
          labelSelector:
            matchExpressions:
            - key: app
              operator: In
              values:
              - my-app
          topologyKey: "kubernetes.io/hostname"
  ...
```

(1) -> 我们将副本数更改为 5。

(2) -> 我们将 Pod 反亲和性策略更改为更多基于尽力而为的基础。这意味着调度器将尝试尽可能地将 Pod 分散到各个节点上，但如果找不到匹配的节点，它仍然会调度该 Pod。

(3) -> 指定此偏好的权重。**preferredDuringSchedulingIgnoredDuringExecution** 列表中的每个项目都必须有一个 weight 字段。此值必须在 1 到 100 之间，表示此规则与其他规则相比的偏好级别。

(4) -> 将 labelSelector 和 **topologyKey** 包装在 **podAffinityTerm** 中。这种封装对于 **preferredDuringSchedulingIgnoredDuringExecution** 规范的结构是必要的。

让我们删除旧的部署并应用这个新的部署。

```bash
$ kubectl apply -f pod-anti-affinity-preferred.yaml -n robust-setup
deployment.apps/my-deployment created
```

```bash
# 几秒钟后 ...
$ kubectl get pod -n robust-setup -l app=my-app -o wide
NAME                                      READY   STATUS    RESTARTS   AGE   IP           NODE                             NOMINATED NODE   READINESS GATES
my-deployment-5b878b869f-8v2jc            1/1     Running   0          17s   10.244.0.112 first-cluster-default-pool-oeo8k <none>           <none>
my-deployment-5b878b869f-gs2j4            1/1     Running   0          17s   10.244.0.16  first-cluster-default-pool-oeo8k <none>           <none>
my-deployment-5b878b869f-hzm29            1/1     Running   0          17s   10.244.1.59   first-cluster-default-pool-oeo85   <none>           <none>
my-deployment-5b878b869f-qqj2f            1/1     Running   0          17s   10.244.0.201  first-cluster-default-pool-oeo8s   <none>           <none>
my-deployment-5b878b869f-wv8rt            1/1     Running   0          17s   10.244.1.113  first-cluster-default-pool-oeo85   <none>           <none>
```

我们可以看到调度器已经将 Pod 部署到了各个节点上。

## Ingress

Kubernetes Ingress 是一个 API 对象，它为 Kubernetes 集群内的服务提供 HTTP 和 HTTPS 路由。它允许你以更灵活和统一的方式将服务暴露到互联网，管理对集群中服务的外部访问，通常是 HTTP(S) 流量。Ingress 可以提供负载均衡、SSL 终止和基于名称的虚拟主机。

没有 Ingress，将每个服务暴露给外部世界将需要为每个服务配置 LoadBalancer 或 NodePort，这可能效率低下且成本高昂。Ingress 通过作为所有服务的单一入口点简化了这一点。

要使 Ingress 资源工作，你需要在集群中有一个 Ingress 控制器。Ingress 控制器负责实现 Ingress，通常使用负载均衡器，尽管它可能配置边缘路由器或额外的前端来帮助处理流量。流行的 Ingress 控制器包括 Nginx、HAProxy、Traefik 和 Istio。

我们可以在没有 Ingress 控制器的情况下创建 Ingress。为了体会 Ingress 控制器的实用性，让我们先在没有它的情况下创建 Ingress。

```
$ cd robust-setup/ingress
$ kubectl apply -f deployment.yaml -n robust-setup
deployment.apps/nginx-deployment created
$ kubectl apply -f nginx-lb-svc.yaml -n robust-setup
service/nginx-loadbalancer created
$ kubectl apply -f nginx-np-svc.yaml -n robust-setup
service/nginx-nodeport created
```

几分钟后...

```
$ kubectl get svc -n robust-setup
NAME                 TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)          AGE
nginx-loadbalancer   LoadBalancer   10.245.80.181   139.59.53.169   80:30864/TCP     3m35s
nginx-nodeport       NodePort       10.245.23.243   <none>          80:30908/TCP     3m9s
redis                ClusterIP      None            <none>          6379/TCP         46h
```

首先，我们创建一个指向 NodePort 服务的 Ingress。

```yaml
apiVersion: networking.k8s.io/v1 # (1)
kind: Ingress # (2)
metadata:
  name: nginx-ingress-nodeport
spec:
  rules: # (3)
  - host: nginx-nodeport.example.com # (4)
    http: # (5)
      paths: # (6)
      - path: / # (7)
        pathType: Prefix # (8)
        backend: # (9)
          service:
            name: nginx-nodeport
            port:
              number: 80
```

(1) -> 指定 Ingress 资源的 API 版本，表示 networking.k8s.io/v1 组。

(2) -> 将资源类型定义为 Ingress，用于通过 HTTP/HTTPS 管理对集群中服务的外部访问。

(3) -> 开始定义 Ingress 规则，这些规则定义了如何将外部 HTTP 流量路由到服务。

(4) -> 指定用于路由 HTTP 流量的主机名。发送到此主机的请求将根据定义的规则进行匹配和路由。这可以是任何有效的域名或子域名，例如 **example.com** 或 **api.example.com**。

(5) -> 表示为指定主机定义 HTTP 路径和后端的部分。

(6) -> 列出要匹配传入请求的路径以及流量应转发到的相应后端。

(7) -> 指定要匹配以路由到后端服务的 URL 路径。此示例匹配根路径。或者，这可以是你的应用程序提供的任何有效 URL 路径，例如 `/api`、`/users` 等。

(8) -> 指定要使用的路径匹配类型。"Prefix" 匹配任何以指定值开头的路径。如果你想要精确匹配，请使用 **Exact**。

(9) -> 定义匹配路径的流量应转发到的后端服务和端口。

此 Ingress 资源配置指定，发送到 **nginx-nodeport.example.com** 任何路径 (/) 的 HTTP 流量，应使用基于前缀的路径匹配路由到 **nginx-nodeport** 服务的 80 端口。

让我们 curl 这个端点。

```
$ curl -I -H "Host: nginx-nodeport.example.com" http://139.59.6.116:30908
HTTP/1.1 200 OK
Server: nginx/1.25.4
Date: Sat, 02 Mar 2024 03:35:38 GMT
Content-Type: text/html
Content-Length: 615
Last-Modified: Wed, 14 Feb 2024 16:03:00 GMT
Connection: keep-alive
ETag: "65cce434-267"
Accept-Ranges: bytes
```

我使用 **Host** 头来指定 DNS 名称。IP 可以是集群中任何节点的外部 IP。我们可以从上面的 **kubectl get service** 命令获取端口号。

我们也可以为负载均衡器服务构建一个类似的 Ingress。

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nginx-ingress-loadbalancer
spec:
  rules:
  - host: nginx-loadbalancer.example.com # (1)
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nginx-loadbalancer # (2)
            port:
              number: 80
```

(1) -> 我们给出一个不同的主机域名。

(2) -> 此 Ingress 指向 **nginx-loadbalancer** 服务。

我们可以执行类似的 curl 命令从外部访问此服务。

```
$ curl -I -H "Host: nginx-loadbalancer.example.com" http://139.59.53.169
HTTP/1.1 200 OK
Server: nginx/1.25.4
Date: Sat, 02 Mar 2024 03:39:44 GMT
Content-Type: text/html
Content-Length: 615
Last-Modified: Wed, 14 Feb 2024 16:03:00 GMT
Connection: keep-alive
ETag: "65cce434-267"
Accept-Ranges: bytes
```

该 IP 是 **nginx-loadbalancer** 服务的负载均衡器 IP。

## 练习 33：多服务 Ingress

创建一个主机为 **multiservice.example.com** 的 Ingress，其中 **/nginx** 路径指向 nginx 部署，**/flask** 路径指向 redis flask 服务。

## Ingress 控制器

如果我们为集群中的许多服务创建 Ingress，那么这会开始变得繁琐，因为我们必须为每个服务识别节点端口和 IP。我们不能为每个服务使用外部负载均衡器 IP，因为 IP 是稀缺资源。为了解决这个问题，我们有了 Ingress 控制器的概念。

Ingress 控制器充当 Kubernetes 服务与外部世界之间的桥梁。它提供了一种集中、高效的方式来管理 HTTP(S) 流量的路由、SSL 终止和负载均衡，使得在 Kubernetes 中运行的应用程序更容易暴露和安全访问。

Ingress 控制器持续监控 Kubernetes API 服务器，以获取新的、更新的或删除的 Ingress 资源。它读取这些 Ingress 资源的规范，其中包含将外部流量路由到集群内服务的规则。

根据 Ingress 资源中定义的规则，Ingress 控制器充当反向代理或负载均衡器。它在预定义或动态分配的 IP 地址和端口上监听传入的 HTTP(S) 流量。

当收到外部请求时，Ingress 控制器会检查请求的头信息（例如 **Host**）和路径，以确定应如何根据 Ingress 规则进行路由。然后，控制器根据匹配的 Ingress 规则将请求转发到集群内适当的后端服务。

除此之外，Ingress 控制器可以管理集群中服务的 SSL/TLS 终止。这意味着 Ingress 控制器解密传入的 HTTPS 流量，然后将其作为 HTTP 转发给后端服务，从而将 SSL/TLS 工作负载从服务本身卸载。

除了基本路由，Ingress 控制器还可以提供高级功能，如 URL 重写、会话保持、WebSockets 支持、速率限制以及与身份验证和授权服务的集成。

让我们在集群中安装 nginx ingress controller，这是一个流行的 Ingress 控制器。

```
$ kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.0/deploy/static/provider/cloud/deploy.yaml
namespace/ingress-nginx created
serviceaccount/ingress-nginx created
serviceaccount/ingress-nginx-admission created
role.rbac.authorization.k8s.io/ingress-nginx created
role.rbac.authorization.k8s.io/ingress-nginx-admission created
clusterrole.rbac.authorization.k8s.io/ingress-nginx created
clusterrole.rbac.authorization.k8s.io/ingress-nginx-admission created
rolebinding.rbac.authorization.k8s.io/ingress-nginx created
rolebinding.rbac.authorization.k8s.io/ingress-nginx-admission created
clusterrolebinding.rbac.authorization.k8s.io/ingress-nginx created
clusterrolebinding.rbac.authorization.k8s.io/ingress-nginx-admission created
configmap/ingress-nginx-controller created
service/ingress-nginx-controller created
service/ingress-nginx-controller-admission created
deployment.apps/ingress-nginx-controller created
job.batch/ingress-nginx-admission-create created
job.batch/ingress-nginx-admission-patch created
ingressclass.networking.k8s.io/nginx created
validatingwebhookconfiguration.admissionregistration.k8s.io/ingress-nginx-admission created
```

上面的 URL 包含了创建 nginx ingress controller 所需的所有 YAML 工件。几分钟后，你应该会在 **ingress-nginx** 命名空间中看到一个正在运行的 Ingress 控制器。

## 总结

在本章中，我们探讨了几个关键的 Kubernetes 概念，以及如何将它们应用于在 Kubernetes 环境中部署、管理和暴露像 Redis 这样的应用程序。让我们回顾一下关键要点，并理解它们如何在更广泛的 Kubernetes 编排背景下协同工作。

## 通过 Redis 理解 StatefulSet

我们深入探讨了 StatefulSet，这是管理像 Redis 这样有状态应用程序的 Kubernetes 核心资源。与 Deployment 不同，StatefulSet 为其每个 Pod 维护一个稳定的标识，这使得它们非常适合需要 Pod 标识、稳定网络标识符和持久化存储的场景。通过为 Redis 使用 StatefulSet，我们看到了如何确保 Redis 实例得到可预测和可靠的管理，即使在重新调度和重启过程中也能保留数据并确保一致的网络连接。

```
$ kubectl get deploy -n ingress-nginx
NAME                       READY   UP-TO-DATE   AVAILABLE   AGE
ingress-nginx-controller   1/1     1            1           3m14s
```

```
$ kubectl get service -n ingress-nginx
NAME                       TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)                      AGE
ingress-nginx-controller   LoadBalancer   10.245.39.164   174.138.120.165  80:31372/TCP,443:30382/TCP   3m44s
ingress-nginx-controller-admission   ClusterIP   10.245.230.107   <none>          443/TCP                      3m44s
```

Ingress 控制器部署正在运行，同时还有一个代表 Ingress 的负载均衡器服务。为了检查通过 Ingress 控制器路由的外部主机，我们将创建一个新的 ClusterIP 服务和一个 Ingress。

```
$ kubectl apply -f ingress-clusterip.yaml -n robust-setup
ingress.networking.k8s.io/nginx-ingress-clusterip created
```

这次的 Ingress 只有细微差别。

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nginx-ingress-clusterip
spec:
  ingressClassName: nginx # (1)
  rules:
  - host: nginx-clusterip.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nginx-clusterip # (2)
            port:
              number: 80
```

(1) -> 我们指定了一个 Ingress Class。Kubernetes 中的 Ingress Class 是一种根据 Ingress 控制器实现的一组特性或行为对其进行分类或归类的方式。此机制允许你在单个 Kubernetes 集群中使用多个 Ingress 控制器，并选择哪个 Ingress 控制器应处理特定的 Ingress 资源。引入 Ingress Class 概念是为了提供更大的灵活性，并支持在同一集群中运行多个 Ingress 控制器。

(2) -> 我们指向了创建的 **nginx-clusterip** 服务。

让我们利用 Ingress 控制器提供的一些高级功能。我们将从实现速率限制开始。

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nginx-ingress-ratelimit
  annotations:
    nginx.ingress.kubernetes.io/limit-connections: "1" # (1)
    nginx.ingress.kubernetes.io/limit-rps: "1" # (2)
spec:
  ingressClassName: nginx
  rules:
  - host: nginx-ratelimit.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nginx-clusterip
            port:
              number: 80
```

我们为这个新的 Ingress 添加了两个注解。

(1) -> 将同时连接数限制为 1。(2) -> 将每秒请求数限制为 1。

现在，当我们尝试同时发起 10 个 curl 请求时。

```
$ for i in {1..10}; do curl -i -H "Host: nginx-ratelimit.example.com" http://174.138.120.165; done
```

我们可能会收到一些请求返回 503，这是当达到速率限制时 nginx 控制器配置的默认响应。控制器将会有如下日志条目：

```
10.122.0.9 - - [02/Mar/2024:06:05:52 +0000] "GET / HTTP/1.1" 200 615 "-" "curl/8.4.0" 90 0.000 [robust-setup-nginx-clusterip-80][] 10.244.1.101:80 615 0.001 200 f4a0d3f0a1bc9d211c5d8a09256b50f6
2024/03/02 06:05:52 [error] 239#239: *52996 limiting requests, excess: 5.320 by zone "robust-setup_nginx-ingress-ratelimit_8330be13-c601-4a32-9687-57d4a2f2898b_rps", client: 10.122.0.9, server: nginx-ratelimit.example.com, request: "GET / HTTP/1.1", host: "nginx-ratelimit.example.com"
10.122.0.9 - - [02/Mar/2024:06:05:52 +0000] "GET / HTTP/1.1" 503 190 "-" "curl/8.4.0" 90 0.000 [robust-setup-nginx-clusterip-80][] - - - - 865fa11eb057a978a2592e94b544f1af
```

第一行是正常的请求-响应日志。第二行表明控制器触发了速率限制，因此它以 **503** 响应。

接下来，让我们为 Ingress 添加 TLS 证书。为了说明 TLS 终止如何在 Ingress 控制器本身发生，我们将为本次练习生成自签名证书。

```
$ openssl genrsa -out nginx-clusterip.example.com.key 4096 # (1)

$ openssl req -new -x509 -days 365 -key nginx-clusterip.example.com.key -out nginx-clusterip.example.com.crt -addext "subjectAltName = DNS:nginx-clusterip.example.com" -subj "/CN=nginx-clusterip.example.com" # (2)

$ kubectl create secret tls nginx-clusterip-tls --key="nginx-clusterip.example.com.key" --cert="nginx-clusterip.example.com.crt" -n robust-setup # (3)
```

(1) ->

**openssl**: 这是一个命令行工具，用于从 shell 中使用 OpenSSL 加密库的各种加密函数。genrsa: 此选项告诉 OpenSSL 生成一个 RSA 私钥。

**-out nginx-clusterip.example.com.key** 指定保存生成的私钥的文件名。在这种情况下，它以预期的域名命名，但此名称可以是任意的。

**4096** 这是私钥的位数大小。4096 位的密钥比更常见的 2048 位密钥更长，以略微增加加密和解密过程的计算开销为代价，提供了更高的安全性。

(2) ->

**req**: 此选项用于指定正在创建证书请求。

**-new** 表示这是一个新请求。

**-x509** 此标志告诉 OpenSSL 生成自签名证书，而不是证书签名请求 (CSR)。

**-days 365** 将证书的有效期设置为 365 天。在此期限之后，证书将过期并需要续订。

**-key nginx-clusterip.example.com.key** 指定生成证书时要使用的私钥。此密钥是在第一个命令中创建的。

**-out nginx-clusterip.example.com.crt** 指定生成的证书的文件名。

**-addext "subjectAltName = DNS:nginx-clusterip.example.com"** 此扩展为证书添加了主题备用名称 (SAN)，允许证书不仅对通用名称 (CN) 有效，也对其他名称有效。在这种情况下，它指定了与 CN 相同的域名。

**-subj "/CN=nginx-clusterip.example.com"** : 将证书的通用名称设置为 **nginx-clusterip.example.com**，这通常是证书将保护的主要域名。

(3) ->

使用密钥文件和证书文件在相同命名空间中创建一个类型为 **tls** 的 Kubernetes Secret。

我们现在用这个 TLS Secret 更新 Ingress 规范。

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nginx-ingress-clusterip
spec:
  tls: # (1)
  - hosts:
    - nginx-clusterip.example.com # (2)
    secretName: nginx-clusterip-tls # (3)
  ingressClassName: nginx
... # 其余部分相同
```

(1) -> 定义用于保护 Ingress 的 TLS 设置，指定应如何处理 HTTPS 流量。

(2) -> 列出此 TLS 配置涵盖的主机名，指示哪些域名应使用指定的 TLS Secret 进行 HTTPS 访问。

(3) -> 指定包含所列主机的 TLS 证书和私钥的 Kubernetes Secret 的名称，从而启用 HTTPS。

让我们测试一下 TLS 现在是否有效。

```
$ curl -I -H "Host: nginx-clusterip.example.com" https://174.138.120.165 -k
HTTP/2 200
date: Sat, 02 Mar 2024 12:25:56 GMT
content-type: text/html
content-length: 615
last-modified: Wed, 14 Feb 2024 16:03:00 GMT
etag: "65cce434-267"
accept-ranges: bytes
strict-transport-security: max-age=31536000; includeSubDomains
```

如果使用 http 调用服务，Ingress 控制器也会进行重定向。

```
$ curl -I -H "Host: nginx-clusterip.example.com" http://174.138.120.165
HTTP/1.1 308 Permanent Redirect
Date: Sat, 02 Mar 2024 12:27:04 GMT
Content-Type: text/html
Content-Length: 164
Connection: keep-alive
Location: https://nginx-clusterip.example.com
```

**注意：** 如果你使用的是自签名证书，客户端（如 Web 浏览器）很可能会显示安全警告，因为该证书不是由受信任的证书颁发机构 (CA) 签发的。此设置通常用于开发、测试环境或内部应用程序，在这些场景中你可以安全地绕过警告。

在生产环境中，你会使用 [certbot](https://certbot.eff.org/) 生成证书，并将其作为 Secret 安装在集群中。更好的做法是，你会使用名为 [Cert manager](https://cert-manager.io/) 的集群内控制器来自动生成证书并续订它们。

## 创建和管理服务账户

服务账户对于在 Kubernetes 中授予权限和管理访问至关重要。我们学习了如何创建和管理这些账户，为我们的应用程序和 Pod 提供与 Kubernetes API 交互的特定权限。这对于维护安全性和确保应用程序仅拥有其所需的访问权限（不多也不少）至关重要。

## 将 Pod 分布在节点上

我们探讨了有效将 Pod 分布在节点上的策略，以增强容错能力和提高应用程序可用性。通过使用亲和性和反亲和性规则，我们可以影响调度器的决策，确保我们的应用程序组件被分散开，以避免单点故障。这在生产环境中尤为重要，因为高可用性是至关重要的。

## Ingress 和 Ingress 控制器

我们详细介绍了 Ingress 资源和 Ingress 控制器，强调了它们在管理 Kubernetes 集群内服务外部访问方面的作用。我们看到了 Ingress 如何定义规则，将外部 HTTP(S) 流量路由到内部服务，使应用程序可以从外部访问。结合 Ingress 控制器（如 NGINX），我们学习了如何实现高级流量管理功能，例如 SSL/TLS 终止、基于路径的路由和速率限制，显著增强了应用程序暴露的安全性和效率。

本章全面概述了使用 StatefulSets 管理有状态应用程序、利用服务账户进行安全的 API 访问、战略性地将 Pod 分布在集群中以实现高可用性，以及使用 Ingress 资源和控制器有效地将应用程序暴露给外部世界。这些组件中的每一个都在 Kubernetes 生态系统中扮演着至关重要的角色，提供了在云原生环境中可靠且安全地部署、管理和扩展应用程序所需的工具和机制。

通过掌握这些概念，你已准备好充分利用 Kubernetes 的潜力，为可扩展、有弹性且高效的应用程序部署和管理策略铺平道路，这些策略对于现代软件环境至关重要。

## 第 9 章：结语

随着我们结束《面向 Python 开发者的 Kubernetes》之旅，我们已经穿越了使 Kubernetes 成为容器化应用程序编排强大工具的基础领域。从使用 StatefulSets 部署有状态应用程序，到掌握服务账户的复杂性，再到将 Pod 战略性地分布在节点上，以及利用 Ingress 和 Ingress 控制器的动态世界，我们已经涵盖了重要的内容。这些知识不仅仅是理论性的；它非常实用，并构成了我们可以自信地部署生产级应用程序的基石。

## 自信地部署到 Kubernetes

我们的最终任务是将我们的学习转化为行动，为 Flask-Redis 应用程序构建一个 GitHub 持续部署 (CD) 流水线。这条流水线体现了现代 DevOps 实践的精髓，将我们应用程序的构建、测试和部署过程集成到一个无缝的工作流中。通过自动化部署，我们最大限度地减少人为错误，提高效率，并确保我们的应用程序可以随时可靠地部署到生产环境。

这条流水线不仅自动化了部署过程，还展示了 Kubernetes 核心概念在现实场景中的应用。它强调了容器化、编排部署和网络配置的重要性，突出了 Kubernetes 如何促进每一步，以确保我们的应用程序具有弹性、可扩展且易于管理。

## 设置 CD 流水线

应用程序和随附的 CD 流水线可以在 [https://github.com/badri/flask-redis-cd](https://github.com/badri/flask-redis-cd) 仓库中找到。

结构如下所示：

```
.
├── .dockerignore
├── .github
│   └── workflows
│       └── cd.yml # (1)
├── .gitignore
├── Dockerfile # (2)
├── app.py # (3)
├── docker-compose.yml # (4)
├── entrypoint.sh
├── kubernetes # (5)
│   ├── check-redis.sh
│   ├── flask-config.yaml
│   ├── flask-redis.example.com.crt
│   ├── flask-redis.example.com.key
│   └── flask.yaml # (6)
├── redis-init.sh
├── redis.conf
├── redis.yaml
├── requirements.txt # (7)
└── templates
    └── index.html
```

- (1) -> GitHub CD 流水线代码的标准位置。我们将详细讲解此文件。
- (2) -> 用于构建应用程序镜像的 Dockerfile。
- (3) -> 主要的应用程序源代码。
- (4) -> 用于在本地测试设置的 Docker Compose 配置。
- (5) -> **kubernetes** 文件夹包含在 Kubernetes 中部署应用程序所需的工件。
- (6) -> 包含 flask-redis 部署、服务和 Ingress 的主要 YAML 文件。这将用于 CD 流水线。其他文件仅用于设置应用程序的第一个版本。
- (7) -> 用于安装应用程序库和依赖项。

## 设置脚手架

在构建流水线之前，我们首先必须创建运行该流水线所需的脚手架。当开发者 "git push" 代码时，流水线将构建一个新的容器镜像并将其推送到 ghcr 注册表。随后将使用新的容器镜像重新部署应用程序，并在之后运行任何可选的测试。

该流水线假设，

1. 命名空间已创建。
2. 所有其他脚手架（包括 secrets（包括镜像拉取 secrets）、configmaps 和 Redis 服务）都已存在。

流水线还需要一个服务账户来在该命名空间内创建和管理资源。

让我们创建所有这些。我们将从命名空间开始。

```
$ kubectl create namespace dev
```

接下来，我们将创建一个 **ClusterRole**，并将其与服务账户关联。由于这是一个集群范围的资源，我们不需要指定命名空间。

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: continuous-deployment # (1)
rules:
  - apiGroups: # (2)
    - ""
    - apps
    - networking.k8s.io
  resources: # (3)
    - namespaces
    - deployments
    - replicasets
    - ingresses
    - services
    - secrets
    - pvcs
    - statefulsets
    - configmaps
  verbs: # (4)
    - create
    - delete
    - deletecollection
    - get
    - list
    - patch
    - update
    - watch
```

- (1) -> 我们的集群角色的名称。
- (2) -> 所有资源所属的 API 组列表。大多数核心资源不需要 API 组。**apps** 用于 deployments，**networking.k8s.io** 用于 ingresses。
- (3) -> 我们需要访问权限的资源列表。这些是我们最终将通过流水线创建和管理的资源。请根据需要随意调整其权限范围。
- (4) -> 拥有此集群角色的服务账户可以对上述资源执行的动词/操作列表。

```
$ kubectl apply -f cd-clusterrole.yaml
clusterrole.rbac.authorization.k8s.io/continuous-deployment created
```

我们还将创建一个服务账户。

```
$ kubectl create serviceaccount github
serviceaccount/github created
```

接下来，我们将创建一个集群角色绑定，将 **github** 服务账户与 **continuous-deployment** 集群角色关联起来。

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding # (1)
metadata:
  name: continuous-deployment # (2)
subjects:
- kind: ServiceAccount # (3)
  name: github # (4)
  namespace: default # (5)
roleRef:
  kind: ClusterRole # (6)
  name: continuous-deployment # (7)
  apiGroup: rbac.authorization.k8s.io
```

- (1) -> 这是一个 **ClusterRoleBinding** 类型的资源。
- (2) -> 名称是任意的。最好是有意义的名称。
- (3) -> 我们将集群角色绑定与哪种类型的主体关联。在我们的例子中，是 **ServiceAccount**。
- (4) -> 主体的名称。
- (5) -> 主体所属的命名空间。我们在默认命名空间中创建了 **github** 服务账户。
- (6) -> 我们关联的角色是什么。它是 **ClusterRole** 类型。
- (7) -> 其名称为 **continuous-deployment**。

我们将创建此集群角色绑定。

```
$ kubectl apply -f github-clusterrole-binding.yaml
clusterrolebinding.rbac.authorization.k8s.io/continuous-deployment created
```

接下来，我们将创建一个 secret，用于保存此服务账户的令牌。

```yaml
apiVersion: v1
kind: Secret
metadata:
```

## GitHub Actions

GitHub Actions 是集成在 GitHub 中的自动化与 CI/CD 平台，为开发者提供了一个强大的工具，可以直接从他们的 GitHub 仓库自动化工作流程。通过为构建、测试和部署应用程序定义工作流程，GitHub Actions 简化了将持续集成（CI）和持续部署（CD）实践融入软件开发生命周期的过程。

### GitHub Actions 的核心组件

**工作流程：** 工作流程是一个可配置的自动化过程，由一个或多个作业组成。它在 GitHub 仓库的 `.github/workflows` 目录中的一个 YAML 文件中定义，由特定事件触发，例如推送到仓库、创建拉取请求或按计划触发。每个工作流程指定了应采取的操作、运行环境以及触发条件。

**事件：** 事件是触发工作流程的特定活动。GitHub Actions 支持广泛的事件，包括代码推送、问题创建、拉取请求事件，甚至通过 Webhook 触发的外部事件。开发者可以指定触发工作流程的事件，使自动化高度依赖上下文并能响应开发过程。

**作业：** 工作流程由作业组成，作业是在同一个运行器上执行的一组步骤。作业可以按顺序或并行运行，具体取决于工作流程中定义的依赖关系。每个作业都在运行器指定的虚拟环境的新实例中运行。

**步骤：** 步骤是在作业中运行命令或操作的单个任务。步骤可以是自定义脚本或操作。操作是可重用的代码单元，可以在仓库内或与 GitHub 社区共享。开发者可以利用庞大的操作市场来执行常见任务，例如设置开发环境、运行测试或部署代码。

**运行器：** 运行器是 GitHub 托管的虚拟机或自托管环境，工作流程在其中执行。GitHub 提供了不同操作系统的运行器，包括 Linux、Windows 和 macOS。开发者也可以使用自托管运行器来适应特定环境或利用特定硬件。

### GitHub Actions 的工作原理

当 GitHub 仓库中发生事件时，例如推送到主分支，GitHub Actions 会查找由该事件触发的工作流程文件。然后，平台会将工作流程中定义的作业加入队列，并根据指定的操作系统和环境将它们分配给可用的运行器。每个作业按顺序执行其步骤，使用操作或脚本来完成构建应用程序、运行测试或部署到服务器等任务。

运行中的工作流程的反馈和日志可以实时查看，提供了对自动化过程的可见性。如果某个步骤失败，工作流程可以停止、发送通知，或者根据工作流程文件中定义的失败处理逻辑触发其他工作流程。

```yaml
name: CD # (1)

on:
  push:
    branches: [ "main" ] # (2)

jobs: # (3)
  build:
    runs-on: ubuntu-latest # (4)
    permissions: # (5)
      contents: read
      packages: write
    steps: # (6)
      - name: Checkout # (7)
        uses: actions/checkout@v3

      - name: Set up Docker Buildx # (8)
        uses: docker/setup-buildx-action@v3

      - name: Login to GitHub Container Registry # (9)
        uses: docker/login-action@v3
        with: # (10)
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push # (11)
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64
          file: ./Dockerfile
          push: true
          tags: ghcr.io/${{ github.repository }}/flask-redis:${{ github.sha }} # (12)
          cache-from: type=gha # (13)
          cache-to: type=gha,mode=max # (14)

      - name: Set kubernetes context # (15)
        uses: azure/k8s-set-context@v3
        with:
          method: service-account
          k8s-url: https://72ad2064-31b4-4552-a630-c88487140b6f.k8s.ondigitalocean.com # (16)
          k8s-secret: ${{ secrets.KUBERNETES_SECRET }} # (17)

      - name: Install kubectl # (18)
        uses: azure/setup-kubectl@v3
        id: install

      - name: Deploy to the Kubernetes cluster # (19)
        uses: azure/k8s-deploy@v4
        with:
          namespace: dev # (20)
          manifests: |
            kubernetes/flask.yaml # (21)
          images: |
            ghcr.io/badri/flask-redis-cd/flask-redis:${{ github.sha }} # (22)
```

这个 GitHub Actions 工作流程是为持续部署（CD）设计的，特别针对一个场景：当提交到 **main** 分支时，构建 Docker 镜像并推送到 GitHub 容器注册表（GHCR），然后部署到我们的 Kubernetes 集群。为了清晰起见，我们来分解每个带注释的行：

(1) -> name: **CD** 设置了工作流程的名称。此名称会出现在 GitHub 仓库的 Actions 页面上，有助于识别工作流程的用途。

(2) -> 此部分在推送到 **main** 分支的事件上触发工作流程。它指定工作流程仅在更改被推送到 **main** 时运行。

(3) -> jobs: 定义了工作流程将执行的一组任务（作业）。作业默认彼此独立运行。

(4) -> runs-on: **ubuntu-latest** 指定该作业应在 GitHub Actions 虚拟环境中可用的最新版本 Ubuntu 上运行。

(5) -> permissions: 显式设置工作流程中 GitHub 令牌的权限，将令牌访问限制为仓库内容（只读）和包上传。

(6) -> steps: 概述了作业将执行的顺序任务或操作。每个步骤可以运行命令或操作。

(7) -> 此步骤检出仓库代码，使其可供工作流程使用。

(8) -> 设置 Docker Buildx，这是一个 Docker CLI 插件，用于通过 BuildKit 提供扩展的构建功能，增强 Docker 镜像的构建和推送。

(9) -> 登录到 GitHub 容器注册表（GHCR），以允许后续步骤将 Docker 镜像推送到注册表。

(10) -> with: 指定了 **docker/login-action** 的输入，包括注册表 URL、用户名和密码（本例中为 GitHub 令牌）。`${{ secrets.GITHUB_TOKEN }}` 是在流水线中访问密钥的语法。此特定密钥默认可用。这可以替换为任何其他容器注册表。

(11) -> 此步骤从仓库中的 **Dockerfile** 构建 Docker 镜像，并使用 build-push-action 将其推送到 GHCR。

(12) -> 指定要推送的 Docker 镜像的标签。它使用仓库路径和提交 SHA 作为唯一标识符。

(13) -> **cache-from: type=gha** 启用从先前运行中缓存构建层，以提高构建性能。

(14) -> **cache-to: type=gha,mode=max** 指定将当前构建的缓存存储到何处，以优化后续构建。

(15) -> 准备用于部署的 Kubernetes 上下文，配置对 Kubernetes 集群的访问。

(16) -> kubectl 命令将指向的 Kubernetes 集群 API URL。

(17) -> 包含用于向 Kubernetes 集群进行身份验证的服务账户令牌的密钥。

(18) -> 安装 kubectl 命令行工具，用于与 Kubernetes 集群交互。

(19) -> 使用指定的清单和镜像将应用程序部署到 Kubernetes 集群。

(20) -> namespace: **dev** 指定资源将部署到的 Kubernetes 命名空间。

(21) -> manifests: 指定定义应用程序所需状态的 Kubernetes 清单文件的路径。

(22) -> images: 指定部署中要使用的 Docker 镜像，允许操作更新清单中的镜像引用。

(17) -> 用于与 Kubernetes 集群进行身份验证的 Kubernetes 密钥（存储在 GitHub Secrets 中）。`${{ secrets.KUBERNETES_SECRET }}` 将存储 `kubectl get secret github-cd-token -o yaml` 命令的密钥值。

(18) -> 安装 kubectl。

(19) -> 使用 k8s-deploy action 执行部署到 Kubernetes 集群。

(20) -> 指定部署应在其中发生的 Kubernetes 命名空间。

(21) -> 指向定义应用程序在集群中期望状态的 Kubernetes 清单文件。

(22) -> 指定 Kubernetes 部署中使用的 Docker 镜像，引用工作流早期推送到 GHCR 的镜像。

![](img/91c4bde832e221ba5bc429caf9587286_395_0.png)

github actions 成功运行

**注意** 如果你计划将此用于你的代码，请在流水线文件中更改注册表 URL、Kubernetes API 服务器 URL、命名空间等。

## 接下来去哪里

我们到目前为止所学的内容构成了 Kubernetes 的核心。理解 Pod、Deployment、Service、Ingress、ConfigMap、Secret 和 StatefulSet 至关重要。但除了理解之外，有效应用这些资源的能力确保我们能够在 Kubernetes 中部署和管理生产级应用程序。这些概念是 Kubernetes 的构建块，使开发人员能够充分利用其潜力来编排复杂的应用程序。

Kubernetes 的设计以声明式配置和自动化原则为中心。通过 YAML 清单定义应用程序的期望状态，并将这些清单应用到我们的集群，我们指示 Kubernetes 如何管理应用程序的生命周期。Kubernetes 接管了调度 Pod、管理其生命周期、根据负载进行扩缩容以及在故障时进行自愈的繁重工作。这种自动化正是 Kubernetes 在部署生产应用程序方面如此强大的原因。

虽然涵盖的概念提供了坚实的基础，但 Kubernetes 生态系统庞大且不断发展。为了进一步增强你的 Kubernetes 之旅，并真正利用其能力来构建复杂的、生产就绪的应用程序，请考虑深入研究以下领域：

**编写 Helm Charts 并使用 Helm：** Helm 通过将应用程序及其依赖项打包到 charts 中来简化 Kubernetes 应用程序部署。学习为你的应用程序编写 Helm charts 可以简化你的部署流程，并使管理应用程序版本更容易。

**Metrics Server 和水平 Pod 自动扩缩容：** 理解资源指标并基于 CPU 和内存使用情况实现水平 Pod 自动扩缩容，可以确保你的应用程序根据需求动态扩展，保持性能并降低成本。

**集群自动扩缩容：** 除了 Pod 自动扩缩容之外，集群自动扩缩容会根据工作负载的需求动态调整集群的大小，添加或移除节点。这对于高效管理集群资源至关重要。

**日志记录和监控：** 实施全面的日志记录和监控对于保持应用程序性能的可见性和故障排除至关重要。像 Prometheus 这样的监控工具和 Grafana 这样的可视化工具是强大可观测性策略不可或缺的一部分。

**卷快照、备份和恢复：** 通过卷快照、备份和恢复过程确保应用程序的数据弹性，对于灾难恢复计划和维护数据完整性至关重要。

**使用 Vault 等外部工具管理密钥：** 虽然 Kubernetes Secrets 提供了存储敏感信息的机制，但与 HashiCorp Vault 等外部密钥管理工具集成可以为管理密钥提供增强的安全性和灵活性。

**用户身份验证、配额和策略：** 通过用户身份验证、配额和网络策略来保护你的集群并管理资源消耗，确保你的 Kubernetes 环境既安全又高效利用。

**API 网关：** 实施 API 网关可以简化微服务管理，为你的应用程序 API 提供单一入口点，并促进速率限制、身份验证和服务发现等服务。

## 展望未来

Kubernetes 之旅仍在继续。随着生态系统的发展，我们可用的工具和实践也将随之发展。紧跟 Kubernetes 社区内的发展，并不断尝试新的工具和资源，将确保你的技能保持敏锐，你的应用程序受益于 Kubernetes 提供的最佳功能。

在结束本章时，请记住 Kubernetes 不仅仅是一项技术；它是一种关于部署、扩展和管理应用程序的新思维方式。它使开发人员能够专注于构建出色的应用程序，并放心地让编排层处理大规模运行这些应用程序的复杂性。

以好奇心和热情拥抱前方的旅程。这条道路充满了学习机会，并有可能彻底改变我们在云原生时代思考和部署软件的方式。

## 附录 A：常见场景故障排除

在使用 Kubernetes 时，在应用程序的部署和运行过程中遇到问题是很常见的。了解如何有效地排除这些场景的故障对于维护服务的健康和可用性至关重要。本附录深入探讨了常见的故障排除场景，提供了关于这些问题发生原因、如何诊断它们以及解决策略的见解。通过掌握这些故障排除技能，你可以确保你的 Kubernetes 应用程序平稳高效地运行。

### Pod 无法启动

#### 为什么发生

Pod 可能由于配置错误、资源不足、镜像问题或安全约束而无法启动。

### 诊断

使用 `kubectl describe pod <pod-name>` 检查与 Pod 相关的事件和状态。检查诸如 `ImagePullBackOff`、`CrashLoopBackOff` 或资源限制之类的错误。

### 修复

确保 Pod 的配置正确，包括镜像名称、资源请求和限制。根据需要调整配置或资源配额。

### 镜像未找到

#### 为什么发生

当 Kubernetes 无法拉取容器镜像时，会发生此错误，可能是由于镜像名称、标签或注册表访问问题不正确。

### 诊断

检查 Pod 描述中的 **ImagePullBackOff** 或 **ErrImagePull** 状态。确保镜像名称和标签正确，并且注册表可访问。

### 修复

更正部署配置中的镜像名称/标签。对于私有注册表，确保镜像拉取密钥配置正确。

### 标签不正确

#### 为什么发生

标签是用于组织和选择对象子集的键值对。不正确或缺失的标签可能会破坏服务发现或 Pod 选择。

### 诊断

使用 **kubectl get pods --show-labels** 列出 Pod 及其标签。验证它们是否与你预期的选择器匹配。

### 修复

使用正确的标签更新你的部署或 Pod 清单，并重新应用配置。

### 未暴露正确的端口

#### 为什么发生

服务根据端口号将流量路由到 Pod。如果 Pod 未暴露预期的端口，可能会导致连接问题。

### 诊断

检查 Pod 和服务配置以确保端口匹配。

### 修复

更新 Pod 的容器规范以暴露正确的端口，并确保服务指向此端口。

### 资源不足

#### 为什么发生

如果节点缺乏足够的 CPU 或内存资源，Pod 可能无法启动，通常由于不可调度条件而显示 **Pending** 状态。

### 诊断

使用 `kubectl describe pod <pod-name>` 检查指示资源不足的事件。

### 修复

调整 Pod 的资源请求和限制，或扩展集群以添加更多资源。

### 镜像架构不同

#### 为什么发生

当容器镜像的架构与节点的架构不匹配时，会出现此问题，例如，在 x86 节点上运行 ARM 镜像。检查 Pod 日志中的 **exec format error**。

### 诊断

检查 Pod 日志或描述 Pod 以识别架构不匹配错误。

### 修复

使用多架构镜像或确保镜像架构与你的节点匹配。

### CRD 未找到

#### 为什么发生

自定义资源定义扩展了 Kubernetes API。如果缺少 CRD，则无法创建依赖于它的资源。

### 诊断

在应用程序日志或部署过程中检查指示缺少 CRD 的错误。

### 修复

在部署依赖于它们的资源之前，请将缺失的 CRD 定义应用到您的集群。

## 未授权

### 为何发生

Pod 或操作员可能尝试在没有适当权限的情况下执行操作，从而导致授权错误。

### 诊断

检查日志中的授权相关错误，这些错误表明操作被拒绝。

### 修复

调整基于角色的访问控制（RBAC）设置，确保服务账户拥有执行所需操作的必要权限。

## 结论

在 Kubernetes 中进行故障排除需要采用系统化的方法来有效诊断和解决问题。通过理解常见场景及其解决方案，开发人员和运维人员可以最大限度地减少停机时间并确保应用程序的稳定性。始终从使用 kubectl 命令收集信息开始，检查配置中的常见错误，并应用 Kubernetes 应用程序部署和管理的最佳实践。

## 附录 B：练习解答

### 练习 1：配置 gunicorn

生产环境中的 Flask 应用运行 gunicorn。配置您的入口点以运行 gunicorn。

**解答**

需要进行两项更改。将 gunicorn 库添加到 **requirements.txt**。

我们修改 **entrypoint.sh** 脚本以运行 Gunicorn。

```
exec gunicorn -b 0.0.0.0:$PORT app:app
```

### 练习 2：设置实时重载

更改 docker-compose 配置，使其在开发期间允许实时重载。这确保了主机上源代码的更改无需重启容器即可在容器内反映出来。

**解答**

需要在 **entrypoint.sh** 中的 flask 命令添加 –reload 标志。

```
exec flask run -h 0.0.0.0 -p $PORT –reload
```

在 **docker-compose.yml** 中将源代码作为卷挂载到容器中

**services:**

```
flask-app:
  image: flask-redis:1.0
  ports:
    - "8080:${PORT}"
  environment:
    - PORT=8080
    - REDIS_HOST=cache
    - REDIS_PASSWORD=SuperS3cret
  depends_on:
    - cache
  volumes:
    - ./:/code # <--
```

### 练习 3：配置 Nginx

生产环境的 Flask 应用不使用 **flask run** 来提供应用程序服务。它们要么使用 gunicorn，要么使用像 Nginx 这样的 Web 服务器进行代理。不要使用默认的 Flask run 命令提供服务，而是使用 Nginx 容器进行代理。对 docker-compose 文件进行必要的更改。

**解答**

不要将 flask 端口暴露到外部。添加一个新的 nginx 容器，并挂载一个自定义的 **nginx.conf** 来代表 Flask 进行代理。

**docker-compose.yml** 的结构

```
version: '3'

services:

  flask-app:

    image: flask-redis:1.0

    expose:

      - "8080"

    environment:

      - PORT=8080

      - REDIS_HOST=cache

      - REDIS_PASSWORD=SuperS3cret

    depends_on:

      - cache

cache:
  image: "bitnami/redis:7.2.4"
  environment:
    - REDIS_PASSWORD=SuperS3cret
  ports:
    - "6379:6379"

nginx:
  image: nginx:alpine
  ports:
    - "8080:80"
  volumes:
    - ./nginx.conf:/etc/nginx/conf.d/default.conf
  depends_on:
    - flask-app
```

nginx.conf 文件。

```
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://flask-app:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 练习 4：在 Ghcr 中构建并推送新标签。

对您的代码进行任意更改，构建它并将新标签推送到 Github 注册表。

**解答**

根据需要对您的应用程序代码进行更改。

使用新标签构建新的 Docker 镜像。如果您的 GitHub 用户名是 your-username，您的应用程序名为 my-flask-app，您将使用：

```
docker build -t ghcr.io/your-username/my-flask-app:new-tag .
```

将新标签推送到 GHCR。如果您尚未进行身份验证，您需要使用具有包读/写权限的个人访问令牌对 GHCR 进行身份验证。

```
echo $CR_PAT | docker login ghcr.io -u your-username --password-stdin
docker push ghcr.io/your-username/my-flask-app:new-tag
```

### 练习 5：更改 Pod 规范并重新应用

更改 Pod 规范 YAML 中 nginx 容器的名称，然后再次运行 `kubectl apply`。

**解答**

```
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: nginx
  name: nginx
spec:
  containers:
  - image: nginx
    name: a-new-name
```

### 练习 6：更改 Pod 的名称

尝试在 Pod 规范 YAML 的 metadata 中将 Pod 的名称从 `nginx` 更改为 `webapp`，然后运行 `kubectl apply`。

**解答**

```
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: nginx
  name: webapp
spec:
  containers:
  - image: nginx
    name: nginx
```

### 练习 7：运行 Apache httpd 服务器

与 Nginx 类似，运行 Apache httpd 服务器作为名为 "httpd" 的 Pod，标签为 **app=webserver**。

**解答**

```
apiVersion: v1
kind: Pod
metadata:
  labels:
    app: webserver
  name: httpd
spec:
  containers:
  - image: httpd
    name: apache
```

### 练习 8：配置 sleep pod 使其不崩溃

修复 bash-sleep 容器的配置，使 Pod 不会宕机。

**解答**

```
apiVersion: v1
kind: Pod
metadata:
  name: bash-sleep
spec:
  containers:
  - name: sleepy-app
    image: bash
    command: ["bash", "-c", "sleep infinity"]
```

### 练习 9：描述 bash-sleep Pod。

您甚至可以在崩溃的 Pod 上运行 describe。您将在后面的故障排除章节中了解到，您应该特别在崩溃的 Pod 上运行 describe。尝试理解并推断 kubectl 关于此 Pod 给出的信息。

**解答**

```
$ kubectl describe bash-sleep
```

### 练习 10：在 Pod 中运行 3 个容器。

使用与上一节相同的规范，添加另一个容器并将其运行为：

- a. 通过更新上面的 Pod 规范在同一个 Pod 中
- b. 通过创建新的 Pod 规范在不同的 Pod 中

尝试理解任务 (a) 和 (b) 之间的区别。

**解答**

对于 (a)，您必须如下编辑 Pod 规范，

```
apiVersion: v1

kind: Pod

metadata:

  name: two-containers-better

  labels:

    purpose: demonstrate-multi-container

spec:
  containers:
    - name: nginx-container-1
      image: bitnami/nginx
      env:
        - name: NGINX_HTTP_PORT_NUMBER
          value: "8080"
      ports:
        - containerPort: 8080
    - name: nginx-container-2
      image: bitnami/nginx
      env:
        - name: NGINX_HTTP_PORT_NUMBER
          value: "8081"
      ports:
        - containerPort: 8081
    - name: nginx-container-3
      image: bitnami/nginx
      env:
        - name: NGINX_HTTP_PORT_NUMBER
          value: "8082"
      ports:
        - containerPort: 8082
```

然后 **kubectl apply** 它。

对于 (b)，您必须创建一个新的 Pod 规范文件，使用新的 Pod 名称，并运行 **kubectl create**。

kubectl create 和 kubectl create 命令都用于在 Kubernetes 集群中创建资源，但它们的设计目的略有不同，行为也不同。

当您使用 kubectl create 时，它期望资源在集群中尚不存在。如果同名资源已存在，该命令将因错误而失败。

kubectl apply 可以在资源不存在时创建它们，如果存在则更新它们。它通过跟踪您在注解中应用的先前配置来实现这一点。这允许 kubectl apply 计算上次应用的配置与您提供的新所需配置之间的差异（delta），并仅应用这些更改。

### 练习 11：2 个容器的变体

再次在 Pod 中运行 2 个容器。但 1 个是 nginx，另一个是运行 "sleep 5" 然后终止的 bash 容器。Pod 会崩溃吗？如果是，您将如何在不更改 bash 容器镜像的情况下修复它？

**解答**

即使其中一个容器的进程被终止，Pod 也会崩溃。在这种情况下，bash sleep 容器将在 5 秒后关闭，向 kubelet 指示该 Pod 需要重新启动，从而导致无限的 **CrashloopBackoff** 情况。改为在 bash 容器中添加 **sleep infinity**。

### 练习 12：在 8008 端口进行端口转发

尝试将同一个 Pod 端口转发到您主机的 8008 端口。

**解答**

```
$ kubectl port-forward pod/nginx 8008:80
```

### 练习 13：端口转发多个容器

采用 2-pod Bitnami nginx 设置。将 8080 nginx 端口转发到 localhost 的 8080，将 8081 nginx 容器端口转发到 localhost 的 8081。

## 练习 14：打印端口号

打印双容器 bitnami nginx-es pod 中第二个容器的端口号。

## 解答

```
$ kubectl exec two-containers-better -c nginx-container-2 -- sh -c 'echo $NGINX_HTTP_PORT_NUMBER'
```

## 练习 15：在新命名空间中部署 nginx pod

创建一个新的命名空间 **python-dev**，并在该命名空间中部署一个 nginx pod。尝试两种变体：
a. 在命令行中指定命名空间。
b. 在 pod YAML 中指定命名空间。
现在，尝试在 YAML 中指定 **python-dev** 命名空间，同时在命令行中指定一个不同的命名空间（**development**）。看看会发生什么。

## 解答

```
$ kubectl create namespace python-dev

$ kubectl apply -f nginx-pod.yaml -n python-dev # (a)
```

```
$ kubectl apply -f nginx-pod.yaml # (b) 假设元数据部分已包含 "namespace: python-dev"
```

(a) 和 (b) 都将按预期工作。

最后一部分会出错，因为 Kubernetes 会优先采用 YAML 规范中的命名空间，而不是命令行标志中的命名空间。

## 练习 16：删除命名空间

删除上面创建的 **python-dev** 命名空间。找出命名空间被删除后，部署在该命名空间中的 nginx pod 发生了什么。

## 解答

```
$ kubectl delete namespace python-dev
```

当 Kubernetes 命名空间被删除时，与该命名空间关联的所有资源，包括 pod、服务、部署等，也会被一并删除。命名空间是在多个用户和项目之间划分集群资源的一种方式。它们为名称提供了一个作用域，并提供了一种将授权和策略附加到集群子部分的机制。

## 练习 17：相同 IP 还是不同 IP？

进行相同的 curl 练习，尝试从 twin-nginx pod 访问两个容器。每个容器的 IP 会不同，还是相同？

## 解答

Pod 内所有容器的 IP 都将是相同的，即 Pod IP。Pod 中的容器共享相同的 IP 和端口空间。

## 练习 18：通过指定不同的标签部署 pod

在 pod YAML 清单中指定一个不同的标签来部署你的 flask/nginx 容器。

## 解答

更改 **image** 部分中冒号 ":" 后的标签，然后执行 kubectl apply。

## 练习 19：通过指定镜像 SHA 摘要部署 pod

在 pod YAML 清单中指定镜像的 SHA 摘要来部署你的 flask 容器。

## 解答

将 **image** 值更改为指向你的镜像 SHA。

```
apiVersion: v1
kind: Pod
metadata:
  name: simple-flask
  labels:
    app: simple-flask
spec:
  containers:
  - name: flask
    image: ghcr.io/badri/flask-app@sha256:<sha>
    imagePullPolicy: Always
    env:
    - name: PORT
      value: "8080"
    ports:
    - containerPort: 8080
```

## 练习 20：使用 recreate 策略为 nginx 创建一个 deployment。

观察事件并确认所有旧 pod 被同时删除，然后新的 pod 被重新创建，同样是同时进行。

## 解答

将 .spec.strategy.type 更改为 Recreate。

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.24
        ports:
          - containerPort: 80
```

## 练习 21：将 flask 应用部署为 deployment

在一个新的命名空间 **dev** 中，将简单的 flask 应用部署为具有 4 个副本的 deployment。删除 1 或 2 个 pod 并观察发生了什么。

## 解答

首先创建一个新的命名空间。

```
$ kubectl create namespace dev
```

然后应用以下 deployment 规范。

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: simple-flask
  namespace: dev
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: flask
  template:
    metadata:
      labels:
        app: flask
    spec:
      containers:
      - name: flask
        image: ghcr.io/<your-username>/flask-app:1.0
        ports:
          - containerPort: 8080
```

## 练习 22：进行更改并重新部署

将 `<title>` 标签内容更改为 "Flask Redis app"，并重新部署一个新镜像。

## 解答

1. 更改 title HTML 标签。
2. 构建并将新的镜像标签推送到 ghcr.io 仓库。
3. 更新 deployment 并重新应用新的规范。

## 练习 23：重新部署 redis flask 应用

通过将 redis 服务指向以下地址来重新部署 redis flask 应用：
a. pod IP。b. 服务的 cluster IP。

并确保这能正常工作。

## 解答

对于 (a)，我们可以使用以下命令获取 pod IP：

```
$ kubectl get pod -n full-stack -o wide
```

然后将规范中 env 部分的 **REDIS_HOST** 环境变量更改为指向此 IP。

对于 (b)，执行相同的操作，但使用 cluster IP，可以通过运行以下命令获取：

```
$ kubectl get svc -n full-stack
```

## 练习 24：使用 configmaps 并在同一个 pod 中运行 2 个 nginx。

使用前面章节中的 twin-nginx 示例，不运行 Bitnami 的 nginx，而是运行官方 nginx，并使用不同的 **nginx.conf** 文件，将它们挂载为 configmaps，并在同一个 pod 中的不同端口上运行两个 nginx 容器。

## 解答

要在单个 Kubernetes pod 中运行两个不同端口的 NGINX 容器，你需要：

1. 创建两个 NGINX 配置文件，每个配置为监听不同的端口。
2. 创建一个 Kubernetes ConfigMap 来存储这些配置文件。
3. 定义一个 pod 规范，使用此 ConfigMap 将配置文件挂载到每个容器内的适当位置。

nginx1.conf

```
user nginx;

worker_processes auto;

events {
    worker_connections 1024;
}

http {
    server {
        listen 8080;
        server_name localhost;

        location / {
            root /usr/share/nginx/html;
            index index.html;
        }
    }
}
```

nginx2.conf

与 **nginx1.conf** 相同，但 **listen 8081;**

Pod 规范。

```
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
spec:
  volumes:
  - name: nginx-config-volume
    configMap:
      name: nginx-configmap
  containers:
  - name: nginx1
    image: nginx:latest
    ports:
    - containerPort: 8080
    volumeMounts:
    - name: nginx-config-volume
      mountPath: /etc/nginx/nginx.conf
      readOnly: true
      subPath: nginx1.conf
  - name: nginx2
    image: nginx
    ports:
    - containerPort: 8081
    volumeMounts:
    - name: nginx-config-volume
      mountPath: /etc/nginx/nginx.conf
      readOnly: true
      subPath: nginx2.conf
```

## 练习 25：应用程序错误

你认为上面 configmap 中的 **REDIS_HOST** 值会起作用吗？如果不会，你会做什么更改来修复它？

提示：**cache** 服务存在，我们在上一章中确实创建了它。但是你的 **flask-redis** 应用能否跨命名空间访问它？

## 解答

将 **REDIS_HOST** 指向 cache 不会起作用，因为 **cache** 服务在当前命名空间中不存在。但是，我们可以使用其内部 FQDN **cache.full-stack.svc.cluster.local** 从另一个命名空间访问 cache 服务。

## 练习 26：配置 redis deployment

使 redis deployment 也从 **redis-credentials** secret 中获取密码。完成后，凭据仅存储在一个位置，并将被两个 deployment（flask 和 redis）引用。

## 解答

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cache
  template:
    metadata:
      labels:
        app: cache
    spec:
      containers:
        - name: redis
          image: bitnami/redis:7.2.4
          env:
            - name: REDIS_PASSWORD
              valueFrom:
```

secretKeyRef:
    name: redis-credentials
    key: REDIS_PASSWORD
ports:
    - containerPort: 6379

## 练习 27：提供不同的密钥

尝试为上面的部署提供一个不同类型的密钥，看看会发生什么变化。

## 解答

如果你提供一个不同类型的密钥（例如，一个通用密钥或结构不同的密钥）作为 imagePullSecret，会发生以下情况：

Kubernetes 认证失败：Kubernetes 尝试使用提供的密钥向镜像仓库进行认证。如果密钥没有预期的格式或内容（即，它不是一个有效的 `.dockerconfigjson`），认证尝试将会失败。

镜像拉取失败：由于认证失败，Kubernetes 将无法从私有仓库拉取所需的镜像。这将导致在尝试启动 Pod 时出现错误，你可能会在 Pod 的事件或日志中看到与镜像拉取失败相关的错误。

Pod 保持等待状态：Pod 将保持在“等待”状态，具体表现为 `ImagePullBackOff` 或 `ErrImagePull` 状态，因为 Kubernetes 会定期重试镜像拉取，并由于认证错误而持续失败。

## 练习 28：多个镜像拉取密钥

在另一个仓库中构建另一个私有镜像。将其作为另一个容器添加到同一个 Pod/部署中，并为新的容器在配置中添加另一个镜像拉取密钥。

## 解答

以下是如何实现此设置的方法：

首先，为每个私有仓库在你的 Kubernetes 集群中创建所需的 Docker 仓库密钥。使用 `kubectl create secret docker-registry` 命令创建每个密钥。例如：

```
$ kubectl create secret docker-registry secret1 --docker-server=REGISTRY1_URL --docker-username=USER1 --docker-password=PASSWORD1 --docker-email=EMAIL1

$ kubectl create secret docker-registry secret2 --docker-server=REGISTRY2_URL --docker-username=USER2 --docker-password=PASSWORD2 --docker-email=EMAIL2
```

将 `REGISTRY1_URL`、`USER1`、`PASSWORD1`、`EMAIL1`、`REGISTRY2_URL`、`USER2`、`PASSWORD2` 和 `EMAIL2` 替换为你的实际仓库 URL 和凭据。

在定义 Pod 时，在 `imagePullSecrets` 字段中包含两个密钥。这使得两个密钥对 Pod 内的所有容器都可用。以下是一个 Pod 定义示例：

```
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  imagePullSecrets:
  - name: secret1
  - name: secret2
  containers:
  - name: container1
    image: registry1.example.com/myimage1:latest
  - name: container2
    image: registry2.example.com/myimage2:latest
```

在此设置中，`container1` 将使用 `secret1` 中的凭据从 `registry1.example.com` 拉取其镜像，而 `container2` 将使用 `secret2` 中的凭据从 `registry2.example.com` 拉取其镜像。Kubernetes 将尝试使用 `imagePullSecrets` 中提供的所有密钥对每个仓库进行认证，直到成功或用尽所有密钥。

**注意：** 虽然此方法允许每个容器可能使用不同的密钥，但它并不限制容器必须使用特定的密钥。Pod 中的每个容器都可以使用任何提供的密钥进行镜像拉取。关键在于，Pod 内容器使用的所有私有仓库所需的凭据都是可用的，从而允许镜像拉取操作成功。

## 练习 29：扩展 Redis 有状态集。

扩展 Redis 有状态集的副本数。观察并记录对 Redis 数据和 Pod 命名约定的影响。

## 解答

扩展有状态集的一般语法是：

```
$ kubectl scale statefulsets <statefulset-name> --replicas=<new-replica-count>
```

扩容将创建序号更高的新 Pod。缩容将移除序号最高的 Pod。

## 练习 30：使用 Redis Sentinel 为 Redis 设置高可用性。

在 Redis 有状态集旁边部署 Redis Sentinel，以监控主节点并在主节点故障时执行自动故障转移。记录部署过程和故障转移测试。

## 解答

使用 **redis-init.sh** 中的 sentinel 脚本来实现这一点。参考 [https://faun.pub/redis-high-availability-with-sentinel-on-kubernetes-k8s-a1d67842e0ce](https://faun.pub/redis-high-availability-with-sentinel-on-kubernetes-k8s-a1d67842e0ce)。

## 练习 31：更新角色以获取服务。

扩大角色/令牌的范围，除了 Pod 之外，还包括服务。

## 解答

```
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "watch", "list"]
```

## 练习 32：创建长期令牌以访问 Pod 日志

你会注意到，使用此角色我们无法访问 Pod 的日志。更改角色，以便我们可以使用为此角色创建的长期令牌访问 Pod 日志。

## 解答

```
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "watch", "list"]
```

## 练习 33：多服务入口

创建一个入口，主机为 **multiservice.example.com**，其中 **/nginx** 路径指向 nginx 部署，**/flask** 路径指向 redis flask 服务。

## 解答

```
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: multiservice-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$1
spec:
  rules:
  - host: multiservice.example.com
    http:
      paths:
      - path: /nginx(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: nginx
            port:
              number: 80
      - path: /flask(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: redis-flask
            port:
              number: 8080
```

此入口定义执行以下操作：

- 定义了一个名为 **multiservice-ingress** 的入口。确保底层服务已部署。
- 使用注解 **nginx.ingress.kubernetes.io/rewrite-target** 修改传递给后端服务的路径。这对于可能不期望接收路径前缀的服务很有帮助。
- 为主机 `multiservice.example.com` 指定了一个规则，包含两个路径：
    - 到 **multiservice.example.com/nginx** 的流量被路由到端口 80 上的 **nginx** 服务。
    - 到 **multiservice.example.com/flask** 的流量被路由到端口 8080 上的 **redis-flask** 服务。