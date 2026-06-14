

# 使用Python和Django构建后端REST API - 入门篇

通过本速成课程，你将创建一个应用程序并构建一个功能完整的用户数据库，用于构建REST API。

## 内容指南

**第1部分：简介**
- 第1章：课程概览
- 第2章：Vagrant与Docker

**第2部分：设置你的开发环境**
- 第3章：Windows：安装Git、VirtualBox、Vagrant、Atom和ModHeader
- 第4章：macOS：安装Git、VirtualBox、Vagrant、Atom和ModHeader

**第3部分：设置你的项目**
- 第5章：创建工作区
- 第6章：创建Git项目
- 第7章：推送到GitHub

**第4部分：创建开发服务器**
- 第8章：创建Vagrant文件
- 第9章：配置我们的Vagrant box
- 第10章：运行并连接到我们的开发服务器
- 第11章：运行一个Hello World脚本

**第5部分：创建Django应用**
- 第12章：创建Python虚拟环境
- 第13章：安装所需的Python包
- 第14章：创建新的Django项目和应用
- 第15章：在Django设置文件中启用我们的应用
- 第16章：测试并提交我们的更改

**第6部分：设置数据库**
- 第17章：什么是Django模型？
- 第18章：创建我们的用户数据库模型
- 第19章：添加用户模型管理器
- 第20章：设置我们的自定义用户模型
- 第21章：创建迁移并同步数据库

**第7部分：设置Django管理后台**
- 第22章：创建超级用户
- 第23章：启用Django管理后台
- 第24章：测试Django管理后台

**第8部分：API视图简介**
- 第25章：什么是APIView？
- 第26章：创建第一个APIView
- 第27章：配置视图URL
- 第28章：测试我们的API视图
- 第29章：创建序列化器
- 第30章：为APIView添加POST方法
- 第31章：测试POST功能
- 第32章：添加PUT、PATCH和DELETE方法
- 第33章：测试PUT、PATCH和DELETE方法

**第9部分：视图集简介**
- 第34章：什么是视图集？
- 第35章：创建一个简单的视图集
- 第36章：添加URL路由器
- 第37章：测试我们的视图集
- 第38章：添加创建、检索、更新、部分更新和销毁功能
- 第39章：测试视图集

**第10部分：创建个人资料API**
- 第40章：规划我们的个人资料API
- 第41章：创建用户资料序列化器
- 第42章：创建个人资料视图集
- 第43章：将个人资料视图集注册到URL路由器
- 第44章：测试创建个人资料
- 第45章：创建权限类
- 第46章：为视图集添加身份验证和权限
- 第47章：测试新权限
- 第48章：添加搜索个人资料功能
- 第49章：测试搜索个人资料

**第11部分：创建登录API**
- 第50章：创建登录API视图集
- 第51章：测试登录API
- 第52章：使用ModHeader扩展设置令牌头

**第12部分：创建个人资料动态API**
- 第53章：规划个人资料动态API
- 第54章：添加新模型Item
- 第55章：创建并运行模型迁移
- 第56章：将个人资料动态模型添加到管理后台
- 第57章：创建个人资料动态项序列化器
- 第58章：为我们的个人资料动态项创建视图集
- 第59章：测试动态API
- 第60章：为动态API添加权限
- 第61章：测试动态API权限
- 第62章：将状态更新的查看权限限制为仅登录用户
- 第63章：测试新的私有动态

**第13部分：将我们的API部署到AWS服务器**
- 第64章：将我们的应用部署到AWS简介
- 第65章：向AWS添加密钥对
- 第66章：创建EC2服务器实例
- 第67章：向我们的项目添加部署脚本和配置
- 第68章：部署到服务器
- 第69章：更新允许的主机并部署更改
- 结论

# 第1部分：简介

### 课程概览

欢迎来到“使用Python和Django构建后端REST API - 入门篇”的导论章节。本章旨在让你概览在学习本课程时可以期待的内容。学完本章后，你应该对涵盖的主题有清晰的了解，并为接下来的实践部分做好准备。

#### 1.1 课程目的

本课程的主要目的是指导你使用Django和Django REST Framework构建一个健壮、安全且可扩展的后端REST API。你在这里获得的知识和技能在当今的应用和Web开发世界中至关重要。无论你是想推出下一个大产品的初创公司，还是想扩展技能集的开发者，掌握构建后端的艺术都能成倍增加你应用的潜力和能力。

#### 1.2 为什么选择Django和Django REST Framework？

Django是一个用Python编写的高级开源Web框架。它遵循“不要重复自己”（DRY）原则，并强调代码的可重用性。这使其成为开发Web应用程序最高效的框架之一。

Django REST Framework，或称DRF，是一个建立在Django之上的强大工具包，专门用于构建Web API。它为你提供了使数据序列化、身份验证、视图集、路由器等变得轻而易举的工具。Django和DRF的组合是开发现代Web API的成熟技术栈。

#### 1.3 课程亮点

在整个课程中，我们将涵盖以下内容：

- 设置开发环境：在深入编码之前，我们将确保你的开发环境（无论是在Windows还是macOS上）已准备就绪，并配备了所有必要的工具。
- 理解和实现Django：从设置你的第一个Django项目到创建复杂的数据模型并与Django的管理站点集成，你将学习这个优秀框架的方方面面。
- 深入Django REST Framework：在你的Django知识基础上，你将被介绍到DRF的高级功能，从创建APIView到实现功能齐全的视图集。
- 构建端到端的REST API：这不仅仅是一门理论课程。学完后，你将构建一个完整的REST API，能够处理用户资料、身份验证、状态更新等。
- 部署到AWS：一旦你的API准备就绪，我们将引导你完成将其部署到AWS（世界领先的云服务提供商之一）的过程。

#### 1.4 谁应该学习本课程

本课程专为以下人群量身定制：

- 旨在为其应用或MVP构建后端的开发者。
- 渴望学习后端以成为全栈开发者的前端开发者。
- 希望增强其作品集和职业前景的科技领域初学者。

#### 1.5 先决条件

虽然本课程是为初学者设计的，但具备Python的基础知识会有所帮助。不过，如果你完全是新手也不用担心。课程的结构是从基础开始，确保每个人都能跟上进度。

#### 1.6 结论

本概述让你得以一窥即将开始的激动人心的旅程。通过专注和努力，学完本课程后，你不仅将拥有一个功能正常的REST API，还将对后端开发有深刻的理解，这将使你在科技行业中具有竞争优势。

现在，让我们卷起袖子，深入后端开发的世界！在下一章中，我们将讨论两种流行的开发工具：Vagrant和Docker，帮助你做出明智的选择。

### Vagrant与Docker

目标：学完本章后，读者将清楚地了解Vagrant和Docker之间的区别、它们的用例，以及在Web开发的特定场景中哪个可能更合适。

资产、资源和材料：

1. Vagrant：
    - 获取：从[Vagrant官方网站](https://www.vagrantup.com/)下载
    - 用途：用于创建和管理虚拟化开发环境的虚拟化工具。
2. Docker：
    - 获取：从[Docker官方网站](https://www.docker.com/)下载
    - 用于在容器内开发、交付和运行应用程序的平台。

### 简介

在现代开发世界中，确保你构建的软件在所有环境中一致运行至关重要。Vagrant和Docker都旨在解决这个问题，但方式略有不同。要了解哪个更适合你的需求，了解每个工具的基础知识及其差异至关重要。

#### 什么是Vagrant？

Vagrant是一个允许开发者创建、配置和管理完整虚拟化环境的工具。其核心是，Vagrant使用VirtualBox、VMware等虚拟化技术来提供尽可能接近生产环境的环境。

Vagrant的主要特点：

- 可重现的环境：通过利用Vagrantfile（虚拟机的脚本化定义），Vagrant

#### 什么是 Docker？

另一方面，Docker 则专注于容器。容器是一个独立的软件包，包含了运行它所需的一切：代码、运行时、系统工具、库等。容器是隔离的，但运行在共享的操作系统上，这使得它们比虚拟机更轻量。

#### Docker 的主要特性：

-   轻量级：容器共享宿主系统的操作系统内核，无需为每个应用配备独立的操作系统。
-   Docker Hub：一个基于云的注册服务，允许你关联代码仓库、构建详情等。
-   可移植性：Docker 容器可以在任何地方运行——开发者的机器、物理硬件、虚拟机、公有云、私有云等等。
-   微服务：Docker 的架构天生支持微服务，这是一种现代的软件设计模式。

#### 比较 Vagrant 和 Docker

1.  开销与性能：Docker 容器是轻量级的，因为它们共享宿主操作系统。相比之下，Vagrant 创建的是完整的虚拟机，可能会消耗更多资源。
2.  隔离性：Vagrant 提供更好的隔离性，因为它为每个环境运行独立的操作系统实例。Docker 容器虽然也是隔离的，但共享同一个操作系统内核。
3.  生态系统与社区：Docker 拥有更广泛的社区和生态系统，尤其是在 Kubernetes 等容器编排工具流行之后。Vagrant 的生态系统则更侧重于虚拟机提供者和配置工具。
4.  可移植性：Docker 的“一次构建，随处运行”理念确保了无论 Docker 容器在哪里启动，你的应用都能以相同方式运行。Vagrant 环境虽然可复现，但根据提供者的不同，可能仍会存在细微差异。
5.  学习曲线：Docker 的学习曲线可能更陡峭，尤其是在深入高级功能时。Vagrant 在设置可复现的开发环境方面则更简单直接。

#### 何时使用哪个工具？

-   Vagrant：如果你需要完整的虚拟机，或者需要模拟复杂的网络配置，又或者你的应用需要不同的操作系统，那么 Vagrant 可能是更好的选择。
-   Docker：如果你正在构建微服务，需要更广泛的生态系统来实现可扩展性和编排，或者只是想要一个轻量快速的环境，那么 Docker 是不二之选。

## 结论

Vagrant 和 Docker 都是旨在让开发者工作更轻松的强大工具。它们之间的选择应取决于你项目的具体需求。虽然 Vagrant 非常适合设置与生产服务器镜像的虚拟机，但 Docker 的轻量级容器正在改变我们对部署和可扩展性的思考方式。

随着你深入后端开发，你会发现有些场景下其中一个工具会比另一个更合适。请记住，最好的工具永远是那个能让你工作更轻松、应用更健壮的工具。

# 第 2 节：设置你的开发环境

## Windows：安装 Git、VirtualBox、Vagrant、Atom 和 ModHeader

本章所需资源/材料：

-   稳定的互联网连接。
-   一台 Windows 电脑。
-   你电脑上的管理员权限。

### 1. Git（源代码管理）

如何获取：[Git 官方网站](https://git-scm.com/)
用途：Git 是一个免费且开源的分布式版本控制系统，旨在以速度和效率处理从小到非常大的项目。
安装 Git 的步骤：
1.  访问官方网站，点击 Windows 版本的“下载”按钮。
2.  下载完成后，运行安装程序。
3.  选择安装位置。默认位置通常没问题。
4.  选择要安装的组件。对于大多数用户，默认组件就足够了。
5.  选择 Git 的默认编辑器。对于初学者，Windows 默认编辑器应该可以。
6.  调整你的 PATH 环境。我建议选择“Use Git from the Windows Command Prompt”以便于访问。
7.  选择 HTTPS 传输后端。
8.  勾选“Check out Windows-style, commit Unix-style line endings”选项。
9.  使用 MinTTY 作为默认终端模拟器。
10. 完成安装。

### 2. VirtualBox（虚拟化工具）

如何获取：[VirtualBox 官方网站](https://www.virtualbox.org/)
用途：VirtualBox 是一款强大的虚拟化产品，适用于企业和家庭使用。
安装 VirtualBox 的步骤：
1.  访问 VirtualBox 官方网站，下载 Windows 安装程序。
2.  运行安装程序。
3.  在整个安装过程中确认默认设置。
4.  完成安装。

### 3. Vagrant（开发环境管理器）

如何获取：[Vagrant 官方网站](https://www.vagrantup.com/)
用途：Vagrant 是一个用于在单一工作流中构建和管理虚拟机环境的工具。
安装 Vagrant 的步骤：
1.  访问 Vagrant 官方网站，下载 Windows 版本。
2.  双击安装程序。
3.  按照安装向导操作。保持默认设置。
4.  完成安装。可能需要重启。

### 4. Atom（文本编辑器）

如何获取：[Atom 官方网站](https://atom.io/)
用途：Atom 是一个免费且开源的文本和源代码编辑器，支持用 Node.js 编写的插件。
安装 Atom 的步骤：
1.  访问 Atom 官方网站。
2.  下载 Windows 安装程序。
3.  下载完成后，运行安装程序。安装应会自动开始。
4.  安装完成后，Atom 应会自动打开。如果没有，你可以在应用程序文件夹中找到它。

### 5. ModHeader（用于修改 HTTP 头的浏览器扩展）

如何获取：ModHeader 可在 Chrome 网上应用店获取。
用途：ModHeader 允许你在测试 Web 应用程序时修改请求和响应头。
安装 ModHeader 的步骤：
1.  打开 Google Chrome，访问 Chrome 网上应用店。
2.  搜索“ModHeader”。
3.  点击 ModHeader 扩展程序上的“添加到 Chrome”。
4.  确认任何提示。

结论：
设置开发环境需要仔细关注细节。按照上述步骤操作将确保你安装了必要的工具，以便继续学习本课程的后续章节。当我们深入探讨使用 Django 创建后端 REST API 时，这些工具中的每一个都将发挥关键作用。在继续之前，请确保所有工具都已正确安装。

## macOS：安装 Git、VirtualBox、Vagrant、Atom 和 ModHeader

资源、材料和物品：

-   macOS 计算机：这是必不可少的，因为本章的安装说明是针对 macOS 定制的。
-   Git：一个分布式版本控制系统，开发者用它来跟踪源代码的更改。
-   VirtualBox：一个免费且开源的虚拟化软件包。
-   Vagrant：一个用于构建和分发开发环境的工具。
-   Atom：一个免费且开源的文本编辑器，现代、易用且可定制。
-   ModHeader：一个用于修改 HTTP 请求和响应头的浏览器扩展。

> （如何获取：可以从浏览器的扩展或插件商店添加到浏览器。在本章中，我们将重点介绍如何将其添加到 Chrome。可在 [Chrome 网上应用店](https://chrome.google.com/webstore/detail/modheader/idgpnmonknjnojddfkpgkljpfnnfcklj?hl=en) 获取）

简介：
在深入后端开发之前，设置开发环境是第一个必要的步骤。在本章中，我们将指导你安装关键工具和软件，确保你拥有构建健壮后端系统所需的一切。

### 1. 安装 Git：

用途：Git 允许开发者跟踪和管理项目的更改，促进协作和版本控制。
步骤：
1.  访问 [Git 官方网站](https://git-scm.com/download/mac)，下载适用于 macOS 的最新版本。
2.  下载完成后，双击 `.dmg` 文件开始安装。
3.  将 Git 图标拖到你的应用程序文件夹。
4.  要确认安装成功，请打开终端并输入：
   ```bash
   git --version
   ```
   你应该会看到已安装的 Git 版本显示出来。

### 2. 安装 VirtualBox：

用途：VirtualBox 让你可以在 macOS 上运行多个虚拟机，这对于创建隔离的开发环境至关重要。

### 3. 安装 Vagrant：

目的：Vagrant 与 VirtualBox 配合使用，可自动化设置虚拟环境，从而简化开发流程。

步骤：

1.  前往 [Vagrant 官方网站](https://www.vagrantup.com/downloads.html) 并下载 macOS 版本。
2.  打开 `.dmg` 文件，并将 Vagrant 图标拖拽到你的“应用程序”文件夹中。
3.  要验证安装是否成功，请打开终端并输入：

```
vagrant --version
```

你应该会看到已安装的 Vagrant 版本信息。

### 4. 安装 Atom：

目的：Atom 是一款功能强大的文本编辑器，专为编码优化，提供语法高亮、自动补全以及与 Git 的集成。

步骤：

1.  访问 [Atom 官方网站](https://atom.io/) 下载 macOS 版本。
2.  打开下载的 `.zip` 文件，它将解压出 Atom 应用程序。
3.  将 Atom 应用程序移动到你的“应用程序”文件夹。
4.  启动 Atom 以确认安装成功。

### 5. 为 Chrome 添加 ModHeader 扩展：

目的：ModHeader 允许你修改和添加 HTTP 头信息，这对于测试和调试 API 至关重要。

步骤：

1.  打开 Chrome 浏览器并导航至 [Chrome 网上应用店中的 ModHeader 扩展](https://chrome.google.com/webstore/detail/modheader/idgpnmonknjnojddfkpgkljpfnnfcklj?hl=en)。
2.  点击“添加到 Chrome”并确认添加。
3.  添加成功后，你将在 Chrome 浏览器的右上角看到 ModHeader 图标。

结论：
恭喜！你已成功在 macOS 上为你的开发环境设置了必备工具。在后续章节中，当你深入后端开发时，这些工具将起到关键作用。请确保你熟悉每个软件，因为它们将在你的 REST API 开发和测试中不可或缺。

# 第 3 节：设置你的项目

## 创建工作区

资产、资源和材料：

1.  计算机：任何现代计算机（Mac、Windows 或 Linux）均可。
2.  操作系统：确保你拥有最新版本的 Windows、macOS 或 Linux。
3.  文件资源管理器：操作系统内置的资源管理器（例如 Windows 的资源管理器，macOS 的 Finder）。
4.  文本编辑器：Atom（你可以从 [atom.io](https://atom.io/) 下载）。我们将使用 Atom 来编写和管理代码。
5.  专用文件夹：一个干净的目录/文件夹，用于存放你所有的项目文件。

引言：

在深入实际编码和服务器配置之前，在你的计算机上设置一个干净、有序的工作区至关重要。一个专用的工作区将使文件管理、跟踪更改以及在需要时与他人协作变得更加容易。在本章中，我们将指导你完成为 Django REST API 项目创建有序工作区的过程。

创建工作区的步骤：

1.  在你的计算机上选择一个合适的位置：
    首先决定你想将项目存储在哪里。它可以位于你的用户目录、一个专用的 `projects` 目录，或任何你认为方便的位置。只需确保有足够的空间并且易于访问。
2.  为你的项目创建一个新目录：
    使用你的文件资源管理器导航到所选位置。创建一个新目录（或文件夹），并为其命名一个描述性的名称，例如 `django_rest_api_project`。这个名称将帮助你在未来一眼识别出该项目。
3.  打开 Atom 文本编辑器：
    启动 Atom 文本编辑器。如果你尚未下载 Atom，可以从 [atom.io](https://atom.io/) 获取。
4.  将你的项目目录添加到 Atom：
    - 在 Atom 中，转到 `File` > `Add Project Folder`。
    - 导航到你刚刚创建的 `django_rest_api_project` 目录并选择它。
    - 你现在应该能在 Atom 的侧边栏中看到该目录，确认它已被添加为项目文件夹。
5.  创建必要的子目录：
    为了组织性，请在你的主项目文件夹内创建几个子目录：
    - `source`：用于存放你项目的所有源代码。
    - `docs`：用于存放任何文档、笔记或参考资料。
    - `assets`：用于存储你可能需要的任何图像、样式表或其他静态文件。
    - `scripts`：用于存放你可能编写或获取的任何辅助脚本或实用程序。
    使用 Atom 的侧边栏，右键单击 `django_rest_api_project`，选择 `New Folder`，并相应地命名。对每个子目录重复此操作。
6.  初始化一个 README 文件：
    在项目根目录下拥有一个 `README.md` 文件是一个好习惯。这个 Markdown 文件将包含有关项目的信息、其目的、设置说明以及任何其他相关细节。
    - 在 Atom 中右键单击 `django_rest_api_project`。
    - 选择 `New File` 并将其命名为 `README.md`。
    - 打开文件并输入你项目的简要介绍。随着项目的进展，你可以对此进行扩展。

结论：
恭喜！你现在已经为你的 Django REST API 项目设置了一个结构化且有序的工作区。这种有序的设置将确保你能够轻松地管理、更新和协作你的项目，而不会有任何麻烦。随着我们继续前进，请始终记住保持这种结构，因为一个有序的工作区是成功开发过程的关键。

## 创建一个 Git 项目

资产、资源和材料：

- Git：一个分布式版本控制系统（VCS）。
  （获取/获取：从 [Git 官方网站](https://git-scm.com/) 下载并安装）
- GitHub：一个用于托管和版本控制代码的平台。
  （获取/获取：在 [GitHub](https://github.com/) 注册一个免费账户）

引言：
创建一个 Git 项目意味着初始化一个新的仓库，你的项目文件及其历史记录将存储在此仓库中。这个仓库充当你项目的容器，跟踪你所做的更改。在软件开发中使用像 Git 这样的 VCS 至关重要，它允许你管理项目的版本、与他人协作，并在需要时回滚到以前的版本。

创建 Git 项目的步骤：

1.  安装 Git：
    在初始化 Git 项目之前，请确保你已安装 Git。如果尚未安装，请参考第 3 章或第 4 章（取决于你的操作系统）来安装 Git。
2.  导航到你的工作区：
    使用你的终端或命令提示符，使用 `cd`（更改目录）命令导航到你希望项目所在的目录（文件夹）。例如：
    ```
    cd path/to/your/workspace
    ```
3.  初始化一个新的 Git 仓库：
    在你的工作区内，运行以下命令：
    ```
    git init
    ```
    这将初始化一个新的 Git 仓库并开始跟踪现有目录。你会看到一条类似这样的消息：“Initialized empty Git repository in /path/to/your/workspace/.git/”
4.  添加你的文件：
    在 Git 能够跟踪你的文件之前，你需要将它们添加到仓库中。首先创建一个新文件或添加现有文件。
    要添加目录中的所有文件：
    ```
    git add .
    ```
    要添加特定文件：
    ```
    git add filename.ext
    ```
5.  提交你的文件：
    在 Git 中提交会创建你所做更改的快照。这就像保存项目的一个版本。要提交你已添加的文件，请使用：
    ```
    git commit -m "Initial commit"
    ```
    将“Initial commit”替换为你所做更改的简要描述。
6.  在 GitHub 上创建一个远程仓库：
    要将你的本地仓库推送到 GitHub：
    - 登录你的 GitHub 账户。
    - 点击右上角的‘+’图标并选择‘New repository’。
    - 填写仓库名称、描述和其他设置。
    - 点击‘Create repository’。
7.  将本地仓库链接到远程仓库：
    在你的终端中，添加远程仓库：
    ```
    git remote add origin https://github.com/YourUsername/YourRepoName.git
    ```
    将‘YourUsername’替换为你的 GitHub 用户名，将‘YourRepoName’替换为你刚刚创建的仓库名称。
8.  推送到 GitHub：
    要将你的本地仓库上传到 GitHub，请使用：git push -u origin master

这将把你的代码推送到 GitHub 仓库的 master 分支。

## 结论

你现在已经成功创建了一个 Git 项目，并将其与 GitHub 上的远程仓库连接起来。通过这样做，你不仅可以在本地跟踪更改，还能为你的代码提供一个在线备份。随着项目的进展，请记得定期提交更改并将其推送到 GitHub，以保持本地和远程仓库同步更新。

在下一章中，我们将探讨如何将项目推送到 GitHub，确保我们所有的代码都安全地存储在一个在线仓库中。这将为协作以及与他人分享你的项目奠定基础。

## 推送到 GitHub

本章的资产、资源和材料：

- 1. GitHub 账户
  （获取方式：在 [GitHub](https://github.com/) 上创建一个免费账户。这是一个数百万开发者存储、共享和协作代码的平台。）
  （用途：创建仓库和推送代码所必需。）
- 2. Git
  （获取方式：从 [Git 官方网站](https://git-scm.com/) 下载并安装。）
  （用途：一个版本控制工具，帮助跟踪源代码的更改并与他人协作。）

### 介绍

将你的项目推送到 GitHub 不仅可以作为备份，还允许其他开发者协作、查看甚至使用你的项目。在前面的章节中设置好开发环境后，我们现在准备好将项目推送到 GitHub。让我们开始吧！

### 1. 设置你的 GitHub 仓库

1.1 登录 GitHub：打开浏览器并登录你的 GitHub 账户。

1.2 创建新仓库：在右上角，点击‘+’符号并选择‘New repository’。为你的仓库命名，例如“django_rest_api_beginner”。如果需要，可以提供描述。在本教程中，我们将仓库设为公开。不要用 README、.gitignore 或 license 初始化——我们稍后会手动添加这些。

### 2. 初始化本地 Git 仓库

GitHub 仓库设置好后，是时候准备我们的本地项目了。

2.1 打开终端或命令提示符：使用 `cd` 命令导航到你的项目目录。例如：

```bash
cd path_to_your_project_directory/django_rest_api_project/
```

2.2 初始化 Git：运行以下命令初始化一个新的 git 仓库：

```bash
git init
```

### 3. 将本地仓库链接到 GitHub

3.1 添加远程仓库：要将你的本地仓库链接到你的 GitHub 仓库，请使用以下命令：

```bash
git remote add origin https://github.com/your_username/django_rest_api_beginner.git
```

将 `your_username` 替换为你的 GitHub 用户名。

### 4. 暂存和提交更改

在推送到 GitHub 之前，你需要暂存（跟踪）你的项目文件并将其提交到本地仓库。

4.1 暂存文件：使用以下命令暂存所有项目文件：

```bash
git add .
```

4.2 提交文件：使用提交消息提交暂存的文件：

```bash
git commit -m "Initial commit"
```

### 5. 推送到 GitHub

5.1 推送：使用以下命令将提交的更改推送到你的 GitHub 仓库：

```bash
git push -u origin master
```

### 6. 在 GitHub 上验证

在浏览器中打开你的 GitHub 仓库。你现在应该能看到所有项目文件都已列出。

### 7. 处理 README、.gitignore 和 License（可选）

包含一个用于项目描述的 `README.md`、一个用于排除不必要文件/文件夹的 `.gitignore` 以及一个 license 文件是一个好习惯。

7.1 创建 README.md：在你的项目根目录中，创建一个 `README.md` 文件。编辑此文件以提供项目的简要描述、如何设置等。

7.2 设置 .gitignore：在你的项目根目录中创建一个 `.gitignore` 文件。在这里，你可以指定不想跟踪的文件或文件夹，例如：

```
*.log
*.cache
__pycache__/
```

7.3 添加 License：此步骤是可选的，但如果你想让其他人使用你的代码，这是必不可少的。你可以从 [Choose a License](https://choosealicense.com/) 选择一个许可证并将其添加到你的项目中。

一旦你添加了这些文件，请记得暂存、提交并将它们推送到 GitHub。

## 结论

恭喜！你已成功将项目推送到 GitHub。这不仅作为版本控制，还允许其他开发者查看、使用或为你的项目做出贡献。当你对项目进行更改时，请始终记得将这些更改推送到 GitHub 以保持其更新。

# 第4节：创建开发服务器

## 创建 Vagrant 文件

欢迎来到第8章！在本章中，我们将深入探讨创建 Vagrantfile 的过程。这个 Vagrantfile 将作为我们开发服务器的蓝图，确保我们的环境保持一致，并且易于在各种机器上复制。

本章的资产、资源和材料：

- Vagrant：我们在本节中使用的主要软件。如果你还没有安装 Vagrant，可以从 [Vagrant 官方网站](https://www.vagrantup.com/) 下载。
- VirtualBox：Vagrant 与 VirtualBox 等提供程序协同工作来管理虚拟化。从 [VirtualBox 官方网站](https://www.virtualbox.org/) 获取。
- 文本编辑器：你需要一个文本编辑器来创建和修改 Vagrantfile。我们推荐 Atom，正如本书前面提到的。从[这里](https://atom.io/)下载 Atom。

步骤1：初始化 Vagrantfile

1. 打开你的终端或命令提示符。
2. 使用 `cd` 命令导航到你的项目根目录。
3. 进入项目目录后，输入以下命令：

```bash
vagrant init
```

运行此命令将在你的项目目录中创建一个新的 `Vagrantfile`，其中包含默认设置。

## 步骤2：配置 Vagrantfile

使用 Atom 或你首选的文本编辑器打开 `Vagrantfile`。你会注意到它包含大量注释和一个基本配置。

让我们编辑此文件以满足我们的需求：

1. 设置 Box：找到类似这样的行：

```ruby
### config.vm.box = "base"
```

取消注释（删除开头的 `#`）并将 `"base"` 更改为你要使用的 box 的名称。在本教程中，我们将使用 `ubuntu/bionic64`，这是一个 Ubuntu 18.04 LTS 64位 box。它应该看起来像这样：

```ruby
config.vm.box = "ubuntu/bionic64"
```

2. 网络：允许端口转发，以便我们的应用程序可以从主机访问。找到并取消注释或添加：

```ruby
config.vm.network "forwarded_port", guest: 8000, host: 8000
```

这意味着在虚拟机上运行在端口 8000 的应用程序可以在主机上的端口 8000 访问。

3. 同步文件夹：Vagrant 将项目目录（在你的主机上）与 `/vagrant` 目录（在客户机上）同步。这意味着你在主机上所做的任何更改都会反映在虚拟机中，反之亦然。

```ruby
config.vm.synced_folder ".", "/vagrant"
```

## 步骤3：验证 Vagrantfile

保存 `Vagrantfile` 后，返回你的终端或命令提示符，并导航到包含 `Vagrantfile` 的目录。

运行以下命令以验证配置：

```bash
vagrant validate
```

如果一切正常，Vagrant 将确认并显示一条消息，表明配置有效。

## 步骤4：启动虚拟机

现在我们的 Vagrantfile 已设置好，是时候使用以下命令启动虚拟机了：

```bash
vagrant up
```

此命令将下载指定的 box（如果尚未下载），并使用我们在 Vagrantfile 中设置的配置启动一个虚拟机。

## 总结

在本章中，我们成功创建了一个根据我们项目需求定制的 Vagrantfile，对其进行了配置，并使用它启动了我们的开发服务器。这个 Vagrant 设置确保了我们的开发环境保持一致，使团队协作更容易，并减少了“在我的机器上可以运行”的问题。

在下一章中，我们将深入探讨配置我们的 Vagrant box，使其成为我们 Django REST API 开发的强大环境。

## 配置我们的 Vagrant box

本章所需的资产、资源和材料：

- Vagrant：这是我们用于管理和配置虚拟开发环境的工具。如果你还没有，请从 [Vagrant 网站](https://www.vagrantup.com/) 下载并安装。
- VirtualBox：此软件将允许我们的 Vagrant box 作为虚拟机在我们的计算机上运行。确保它已安装。如果没有，你可以从 [VirtualBox 网站](https://www.virtualbox.org/) 下载。
- 文本编辑器：正如前面提到的，我们将在本课程中使用 Atom。确保 Atom（或你首选的编辑器）已安装并准备就绪。
- Vagrant Base Box：这本质上是我们的 Vagrant box 所基于的操作系统镜像。在本教程中，我们将使用 `ubuntu/bionic64`，这是一个流行的 Ubuntu 18.04 LTS 镜像。

#### 介绍：

既然我们已经创建了 Vagrant 文件，是时候配置我们的 Vagrant box 了。这涉及指定虚拟机的设置（如内存、CPU、

### 分步配置：

1.  虚拟机配置：
    打开你在上一章创建的 Vagrantfile。首先，为我们的 Vagrant 虚拟机指定基础镜像。

    ```ruby
    config.vm.box = "ubuntu/bionic64"
    ```

2.  设置虚拟机资源：
    根据你的主机性能调整虚拟机的资源。一个标准设置可能如下所示：

    ```ruby
    config.vm.provider "virtualbox" do |v|
      v.memory = "1024"
      v.cpus = 2
    end
    ```

    这将为虚拟机分配 1 GB 内存和 2 个 CPU。

3.  网络配置：
    我们可以为 Vagrant 虚拟机设置一个私有网络，这样我们就可以通过私有 IP 地址访问它。

    ```ruby
    config.vm.network "private_network", type: "dhcp"
    ```

    使用上述配置，Vagrant 将自动为虚拟机分配一个 IP 地址。

    或者，你可以指定一个静态 IP：

    ```ruby
    config.vm.network "private_network", ip: "192.168.33.10"
    ```

### 4. 文件夹同步：

    默认情况下，Vagrant 会将你的项目目录共享到虚拟机内的 `/vagrant` 目录。这使得主机和 Vagrant 虚拟机之间可以轻松共享文件。如果需要，你可以进一步自定义此设置：

    ```ruby
    config.vm.synced_folder "src/", "/srv/website"
    ```

    这将把主机上的 `src` 目录同步到虚拟机上的 `/srv/website` 目录。

### 5. 配置脚本：

    Vagrant 允许你通过配置脚本自动化虚拟机上的软件安装和配置。目前，我们将使用一个简单的 shell 脚本来更新包管理器并安装一些必要的软件包。

    将以下内容添加到 Vagrantfile：

    ```ruby
    config.vm.provision "shell", inline: <<-SHELL
      apt-get update
      apt-get install -y python3 python3-pip
    SHELL
    ```

    此脚本将在你第一次运行 `vagrant up` 时执行。

### 6. 附加插件：

    如果你想使用其他 Vagrant 插件，可以通过 Vagrant 命令行安装它们，然后在 Vagrantfile 中进行配置。例如：

    ```bash
    $ vagrant plugin install vagrant-vbguest
    ```

    `vagrant-vbguest` 插件会自动在虚拟机系统上安装主机的 VirtualBox Guest Additions。

### 启动 Vagrant 虚拟机：

    完成 Vagrantfile 的配置后，保存文件并打开终端。导航到 Vagrantfile 所在的目录并运行：

    ```bash
    $ vagrant up
    ```

    此命令将读取 Vagrantfile，如果尚未下载 `ubuntu/bionic64` 镜像则会下载它，并使用你指定的所有配置启动虚拟机。

## 结论：

    你的 Vagrant 虚拟机现在已按照我们刚才的设置启动并运行。这将作为我们 Django 开发服务器的基础。通过一些简单的配置，我们创建了一个一致且可复现的开发环境，这对于防止常见的“在我机器上能运行”问题至关重要。在下一章中，我们将深入探讨如何运行和连接到我们的开发服务器。

    （注意：请务必记住，在完成工作后运行 `vagrant halt` 来关闭虚拟机，或者如果你想保存当前状态以便稍后恢复，则使用 `vagrant suspend`。要完全销毁虚拟机（并释放磁盘空间），请使用 `vagrant destroy`。你随时可以使用 `vagrant up` 重新创建虚拟机。）

## 运行并连接到我们的开发服务器

本章的资产、资源和材料：

- 1. Vagrant：这是我们管理虚拟开发环境的主要工具。（从 [Vagrant 官方网站](https://www.vagrantup.com/) 获取）
- 2. VirtualBox：这是我们运行虚拟机的首选虚拟机管理程序。（可在 [VirtualBox 官方网站](https://www.virtualbox.org/) 获取）
- 3. 终端或命令提示符：用于运行命令。
- 4. SSH 密钥对：允许我们安全连接到 Vagrant 虚拟机的安全 Shell 密钥。（你应该在安装 Vagrant 时已经设置好了，但我们将简要回顾如何确保它们已就位。）

### 简介

设置好 Vagrant 和 VirtualBox 后，下一个合乎逻辑的步骤是启动你的开发服务器并能够连接到它。在本章结束时，你将知道如何启动 Vagrant 环境、通过 SSH 连接到它，以及如何高效地管理它。

### 1. 启动你的 Vagrant 虚拟机

在启动 Vagrant 虚拟机之前，我们需要导航到包含 `Vagrantfile` 的目录。`Vagrantfile` 包含了我们虚拟环境的配置详情。

```bash
$ cd path/to/your/vagrant/project
```

现在，使用以下命令启动 Vagrant 虚拟机：

```bash
$ vagrant up
```

此命令将读取 `Vagrantfile` 并相应地启动虚拟机。第一次运行时，此过程可能需要几分钟，因为 Vagrant 需要下载必要的虚拟机镜像文件。

### 2. 通过 SSH 连接到你的 Vagrant 虚拟机

虚拟机启动并运行后，连接到它很简单：

```bash
$ vagrant ssh
```

这将启动一个安全 Shell (SSH) 会话，连接到你的 Vagrant 虚拟机。你现在处于一个独立于主机操作系统的完整 Linux 环境中。

注意：你不需要提供任何 SSH 密钥或密码，因为 Vagrant 已经为你方便地设置了基于密钥的身份验证。如果你需要访问这些密钥，它们通常位于你项目目录下的 `.vagrant/machines/default/virtualbox/private_key`。

### 3. 管理你的 Vagrant 环境

以下是一些管理 Vagrant 虚拟机的基本命令：

- 暂停虚拟机：这将保存虚拟机的当前运行状态。

```bash
$ vagrant suspend
```

- 停止或关闭虚拟机：这将优雅地关闭虚拟机。

```bash
$ vagrant halt
```

- 销毁虚拟机：这将从你的系统中移除虚拟机的所有痕迹。下次运行 `vagrant up` 时，它将从头开始创建一个新的虚拟机。

```bash
$ vagrant destroy
```

- 检查状态：这将显示虚拟机的状态。

```bash
$ vagrant status
```

### 4. 共享文件夹

默认情况下，Vagrant 会将项目目录（你的 `Vagrantfile` 所在的位置）共享到虚拟机内的 `/vagrant` 目录。这意味着你可以轻松地在主机和虚拟机之间共享文件。

例如：如果你在主机上的项目目录中有一个名为 `hello-world.py` 的文件，你会在虚拟机的 `/vagrant/hello-world.py` 下找到它。

## 结论

至此，你已经成功使用 Vagrant 启动并连接到了你的开发服务器。你还学习了一些基本的管理命令来有效地处理你的虚拟环境。在继续之前，请记住这个环境是隔离的；你在这里的操作不会影响你的主操作系统，让你可以自由地进行实验和学习。

## 运行一个 Hello World 脚本

所需的资产、资源和材料：

- Python：如果你遵循了前面的章节，你的环境中应该已经安装了 Python。如果没有，你可以从 [Python 官方网站](https://www.python.org/downloads/) 下载。我们将使用 Python 来运行我们的简单 Hello World 脚本。
- Atom：一个流行的开源文本编辑器，可用于编码。你已经在第 3 章或第 4 章安装了它。如果没有，请从 [Atom 官方网站](https://atom.io/) 下载。
- Vagrant 开发服务器：确保你的 Vagrant 开发服务器已启动并运行。如果没有，请参考本节前面的章节。

简介：
开始任何编码之旅最传统的方式之一就是编写一个简单的脚本，显示消息：“Hello, World!”。在本章中，我们将在开发服务器的环境中做同样的事情。这将介绍如何在我们的环境中执行 Python 脚本。

### 在开发服务器上运行 Hello World 脚本的步骤：

1.  访问你的开发服务器：
    如果你还没有启动，请先启动你的 Vagrant 开发服务器：

    ```bash
    vagrant up
    ```

    服务器启动后，通过 SSH 连接到它：

    ```bash
    vagrant ssh
    ```

2.  导航到你的工作区：
    现在，导航到你的工作区目录，也就是你在第 5 章创建的那个目录。这将是我们脚本的工作目录。

    ```bash
    cd /path_to_your_workspace/
    ```

3.  创建脚本：
    使用 Atom，在你的工作区内创建一个名为 `hello_world.py` 的新文件。记住，你可以在你的主操作系统上操作，而不是在 Vagrant SSH 会话中。

    在 Atom 中打开 `hello_world.py` 并编写以下 Python 代码：

    ```python
    print("Hello, World!")
    ```

    保存并关闭文件。

4.  运行脚本：
    回到你的 Vagrant SSH 会话，确保你仍然在你的工作区目录中。使用 Python 运行脚本：

    ```bash
    python hello_world.py
    ```

    你应该会看到输出：

    ```
    Hello, World!
    ```

    恭喜！你已成功在开发服务器内运行了一个 Python 脚本。

### 理解 Hello World 脚本：

其核心是，Hello World 脚本介绍了：

-   文件创建与管理：你学习了如何在工作区内创建、保存和管理文件。
-   Python 的 Print 函数：Python 中的 `print()` 函数将文本输出到控制台。这是一个至关重要的函数，你在调试和构建应用程序时会经常用到。
-   运行 Python 脚本：通过执行 `python hello_world.py`，你调用了 Python 解释器来执行脚本，展示了 Python 脚本通常是如何被执行的。

## 结论：

虽然本章只是对在开发环境中运行脚本进行了基础介绍，但它为后续更复杂的任务奠定了基础。随着我们的深入，我们将使用 Python 和 Django 来创建更复杂的应用程序和功能。
记住，每一次伟大的旅程都始于一个简单的步骤，或者在编程世界里，一个“Hello, World!”。

练习：
1.  修改 `hello_world.py` 脚本以显示你的名字，例如，“Hello, [Your Name]!”。
2.  尝试在脚本中运行其他简单的 Python 命令，例如基本算术，以熟悉执行 Python 脚本的过程。

# 第 5 节：创建一个 Django 应用

### 创建 Python 虚拟环境

资产、资源和材料：

-   Python：在我们创建虚拟环境之前，你需要安装 Python。你可以从 [Python 官方网站](https://www.python.org/downloads/) 下载最新版本。众所周知，Python 是我们本课程使用的基础语言。
-   pip (Python 包安装器)：大多数 Python 安装默认包含 pip。但是，如果缺失，你可以 [按照这些说明操作](https://pip.pypa.io/en/stable/installing/)。Pip 允许我们轻松安装 Python 包。
-   virtualenv：这是我们用来创建隔离 Python 环境的主要工具。如果你还没有安装，别担心，我们会介绍的。

虚拟环境的目的：
虚拟环境是你计算机上的一个隔离空间，你可以在其中独立于系统级的 Python 安装来安装软件和 Python 包。这种隔离防止了版本之间的冲突，并确保了一个干净、受控的开发环境，使得管理特定项目的依赖项变得更加容易。

让我们开始吧：
1.  安装 virtualenv：
    如果你还没有安装 virtualenv，可以使用 pip 进行安装：

    ```bash
    pip install virtualenv
    ```

2.  创建虚拟环境：
    安装好 virtualenv 后，导航到你希望虚拟环境所在的目录（通常在你的项目目录内），然后运行：

    ```bash
    virtualenv myenv
    ```

    这里，`myenv` 是你的虚拟环境的名称。你可以随意命名，但保持描述性会很有帮助。

3.  激活虚拟环境：
    -   在 Windows 上：

        ```bash
        myenv\Scripts\activate
        ```

    -   在 macOS 和 Linux 上：

        ```bash
        source myenv/bin/activate
        ```

    当虚拟环境被激活时，你会在命令行提示符的开头看到你的虚拟环境名称（本例中是 `myenv`）。这表明环境处于活动状态，你在其活动期间安装的任何 Python 包都将仅安装在此环境中。

4.  停用虚拟环境：
    当你在虚拟环境中完成工作并希望返回全局 Python 环境时，只需运行：

    ```bash
    deactivate
    ```

5.  在虚拟环境中安装包：
    在虚拟环境处于活动状态时，你可以像往常一样使用 pip 安装包。例如，要安装 Django：

    ```bash
    pip install django
    ```

    这将仅在活动的虚拟环境中安装 Django，而不会影响系统级的 Python 安装。

为什么使用虚拟环境？
在同一个系统上开发多个项目时，虚拟环境至关重要，尤其是当这些项目具有不同的依赖项时。它们允许：

-   隔离：确保每个项目都有自己的一组依赖项，互不干扰。
-   版本控制：允许不同的项目使用不同版本的包，而不会产生任何冲突。
-   干净的环境：当你开始一个新项目时，你可以确保你是在一个干净的基础上工作，没有任何不需要的包。
-   部署：在将应用程序部署到生产环境时，更容易管理依赖项。

回顾：
在本章中，我们学习了如何使用 virtualenv 设置 Python 虚拟环境。我们现在有了一个专门用于 Django 项目的空间，可以在其中隔离地管理我们的依赖项。在下一章中，我们将开始安装必要的 Python 包来启动我们的 Django 项目。

### 安装所需的 Python 包

资产、资源和材料：

1.  Python：确保你的开发服务器上安装了 Python（最好是 3.7 或更高版本）。要检查，请在终端中运行 `python --version` 或 `python3 --version`。（你可以从 [python.org](https://www.python.org/downloads/) 下载并安装 Python）。
2.  pip：Python 的包安装器。它通常随 Python 一起安装。要检查其安装情况，请在终端中运行 `pip --version` 或 `pip3 --version`。（如果未安装，[请按照这些说明操作](https://pip.pypa.io/en/stable/installing/)）。
3.  虚拟环境：虽然不是严格要求，但为你的项目使用虚拟环境可以确保依赖项在项目之间不会冲突。（安装说明如下）。

介绍：
要开始构建我们的 Django 应用程序，我们首先需要确保已安装所有必要的 Python 包。这些包将促进我们 Django 应用的创建、管理和增强。在本章中，我们将逐步介绍安装必要 Python 包的过程。

步骤 1：设置虚拟环境
虚拟环境是一个隔离的环境，你可以在其中安装包，而不会影响全局 Python 环境或其他项目。为每个 Python 项目使用虚拟环境是最佳实践。

a. 要安装虚拟环境包，请运行：
```bash
pip install virtualenv
```

b. 导航到你的项目目录（或你希望创建虚拟环境的位置）并运行：
```bash
virtualenv venv
```

c. 激活虚拟环境。激活命令因操作系统而异：
-   Windows：
```bash
venv\Scripts\activate
```

-   macOS 和 Linux：
```bash
source venv/bin/activate
```

激活后，你应该会在终端提示符的开头看到 `(venv)`，这表明你现在正在虚拟环境中工作。

步骤 2：安装 Django 和 Django REST Framework
在我们的虚拟环境处于活动状态后，我们可以继续安装 Django 和 Django REST Framework。

a. 通过运行以下命令安装 Django：
```bash
pip install django==2.2
```

通过指定 `==2.2`，我们确保安装的是我们课程所基于的 Django 2.2 版本。

b. 接下来，安装 Django REST Framework：
```bash
pip install djangorestframework==3.9
```

步骤 3：验证安装
安装后，最好验证一切是否正确安装。

a. 使用以下命令检查已安装的 Django 版本：
```bash
django-admin --version
```

## 创建新的 Django 项目与应用

所需资源、材料与工具：

- Python：确保已安装 Python。（从 [Python 官方网站](https://www.python.org/downloads/) 下载）
- Django：我们将在本章中使用 pip 进行安装。
- 命令行界面 (CLI)：通过终端 (macOS/Linux) 或命令提示符/PowerShell (Windows) 访问。
- 文本编辑器：Atom（或任何您偏好的文本编辑器）

简介：

在本章中，我们将引导您完成创建新 Django 项目的过程，然后在该项目中创建一个应用。Django 项目本质上是一系列设置、配置和应用的集合。而应用则是一个独立的模块，可以代表任何事物——从博客到用户认证系统。每个应用通常都有自己的模型、视图和控制器。

步骤 1：安装 Django

在开始创建项目之前，我们需要确保 Django 已安装。如果尚未安装，请使用 Python 的包管理器 `pip` 进行安装。

```bash
pip install django==2.2
```

步骤 2：启动新的 Django 项目

Django 安装完成后，我们现在可以创建一个新项目。导航到您希望项目所在的目录，然后运行以下命令：

```bash
django-admin startproject myproject
```

将 `myproject` 替换为您期望的项目名称。此命令将创建一个名为 `myproject` 的新目录，其中包含 Django 项目所需的所有必要文件。

### 步骤 3：启动新的 Django 应用

现在项目已创建，让我们进入项目目录：

```bash
cd myproject
```

在项目目录内，我们可以创建应用。假设我们正在构建一个博客；我们可能会将应用命名为 ‘blog’。要创建此应用，请使用以下命令：

```bash
python manage.py startapp blog
```

此命令将在您的项目目录内生成一个名为 `blog` 的新目录，其中包含新应用的结构。

### 步骤 4：探索项目与应用结构

现在，如果您使用文件资源管理器或文本编辑器 (Atom) 导航到项目目录，您将看到以下结构：

```
myproject/
|
├── manage.py
|
├── myproject/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
|
└── blog/
    ├── migrations/
    │   └── __init__.py
    ├── admin.py
    ├── apps.py
    ├── models.py
    ├── tests.py
    └── views.py
```

- manage.py：这是一个用于与项目交互的命令行实用程序。
- myproject/settings.py：此文件包含项目的设置。
- myproject/urls.py：此文件将包含项目的所有 URL 模式。
- blog/models.py：这是您定义应用数据模型的地方。
- blog/views.py：在这里，您将定义处理请求和响应的逻辑与控制流。

### 步骤 5：将应用注册到项目

为了让 Django 识别我们新的 ‘blog’ 应用，我们需要对其进行注册。在 Atom（或您偏好的文本编辑器）中打开 `myproject/settings.py`，并找到 `INSTALLED_APPS` 设置。将您的应用名称添加到此列表中：

```python
INSTALLED_APPS = [
    # ... 其他默认应用在此 ...
    'blog',
]
```

## 结论：

恭喜！您现在已经设置了一个新的 Django 项目，并在其中创建了您的第一个应用。
请记住，DRY（不要重复自己）原则是 Django 的核心。仅通过几个命令，我们就建立了一个强大的结构，这将在我们推进项目时节省时间和精力。

## 在 Django 设置文件中启用我们的应用

在本章中，我们将引导您完成在 Django 设置文件中启用新创建的 Django 应用的过程，确保它被 Django 框架识别和管理。这一步对于将任何新应用集成到 Django 项目中至关重要。

本章所需资源、材料与工具：

- Django：我们选择的 Web 框架。（已在前面的章节中安装。如果未安装，请回顾第 14 章。）
- Django 项目：包含所有 Django 应用和配置的顶层项目文件夹。
- Django 应用：您刚刚创建的应用。（如果尚未创建，请参考第 14 章。）
- 文本编辑器 (Atom)：用于编辑 Django 设置文件。（如果未安装，请根据您的操作系统参考第 3 章或第 4 章的安装步骤。）

启用 Django 应用的分步指南：

1. 定位您的 Django 设置文件
   - 在文本编辑器 (Atom) 中打开 Django 项目。
   - 导航到主项目文件夹（这是以您的项目命名的文件夹，而不是以应用命名的文件夹）。
   - 在此文件夹内，您将找到一个名为 `settings.py` 的文件。此文件包含整个 Django 项目的配置。
2. 打开 `settings.py` 文件
   - 双击 `settings.py` 文件以在文本编辑器中打开它。
3. 定位 `INSTALLED_APPS` 设置
   - 向下滚动或搜索 `INSTALLED_APPS`。这是 Django 中的一个列表，包含当前已启用并被您的 Django 项目识别的所有应用。
4. 将您的应用添加到列表中
   - 在 `INSTALLED_APPS` 列表的末尾，添加一个新行，其中包含用单引号或双引号括起来的 Django 应用名称。

```python
INSTALLED_APPS = [
    ... # 其他已存在的应用在此
    'your_app_name',
]
```

将 `your_app_name` 替换为您的应用名称。

### 5. 保存文件

- 将应用名称添加到列表后，保存 `settings.py` 文件。

### 6. 验证您的更改

- 为确保 Django 识别您的应用，请在终端或命令提示符中运行以下命令：

```python
python manage.py check
```

此命令将执行系统检查。如果没有问题，它应该不会返回任何错误。

### 7. 理解目的

- 我们为什么这样做？`INSTALLED_APPS` 告诉 Django 哪些应用程序在此项目中处于活动状态，并应在各种操作（如数据库迁移、管理界面生成等）中被考虑。每当您创建一个新的 Django 应用时，您都需要确保它在 `INSTALLED_APPS` 设置中列出，以便 Django 知道它的存在。

### 总结：

通过遵循这些步骤，您已将应用集成到 Django 项目中。确保您创建的每个应用都添加到 `INSTALLED_APPS` 列表中至关重要，这样 Django 才能正确管理它。现在您的应用已被 Django 正式识别，您可以在接下来的章节中开始定义特定于您应用的模型、视图和其他功能。

在下一章中，我们将开始编写测试以确保我们的应用按预期运行，并使用 Git 提交我们的更改。

## 测试并提交我们的更改

本章所需资源、材料与工具：

- 终端或命令提示符（大多数操作系统预装）
- Python（已安装，参见第 12 章）
- Django 框架（在第 14 章中安装）
- Git（根据操作系统在第 3 章或第 4 章中安装）
- Atom 编辑器（根据操作系统在第 3 章或第 4 章中安装）
- GitHub 账户（在第 7 章中创建，用于推送项目）

简介：
创建 Django 项目和应用后，确保一切按预期工作至关重要。本章将引导您如何测试应用，然后使用 Git 提交更改。

1. 测试您的 Django 应用
在提交任何代码之前，让我们确保一切正常。Django 自带一个用于本地开发的内置服务器，这非常适合我们的需求。
步骤 1：启动 Django 开发服务器。打开终端或命令提示符，导航到项目目录，然后输入：

### 2. 使用 Git 提交你的更改

既然我们已经验证了应用程序可以正常工作，现在是时候提交我们的更改了。

步骤 1：首先，在你的项目目录中打开终端或命令提示符。

步骤 2：检查你的 git 仓库状态：

```bash
git status
```

此命令显示自上次提交以来你所做的更改。你应该能看到与你创建的 Django 项目和应用相关的文件。

步骤 3：将更改的文件添加到暂存区：

```bash
git add .
```

`add` 后的 `.` 表示你正在将当前目录（及其子目录）中的所有更改添加到暂存区。

步骤 4：提交更改：

```bash
git commit -m "Created Django project and app"
```

此命令使用暂存区中的更改创建一个新的提交。`-m` 标志允许你添加一条描述提交的消息，这对于跟踪更改和了解项目历史非常有用。

步骤 5：将更改推送到 GitHub：

确保你已经在 GitHub 上设置了一个远程仓库（参见第 7 章）。使用以下命令推送更改：

```bash
git push origin master
```

`origin` 指的是你设置的远程仓库的默认名称，`master` 是你项目的主分支。

结论：

你已经成功测试了你的 Django 应用，并使用 Git 提交了更改。定期测试和提交更改可以确保你始终拥有一个可工作的应用版本，并且你的进度保存在像 GitHub 这样的平台上。这个习惯在开发世界中至关重要，它能让你的代码安全且可随时随地访问。

# 第 6 节：设置数据库

#### 什么是 Django 模型？

资产、资源和材料：

- Django（如何获取：通过 `pip install django` 安装。用途：我们正在使用的 Web 框架。）
- Python（如何从 [Python 官方网站](https://www.python.org/downloads/) 下载。用途：Django 是用这种编程语言编写的。）
- Django 关于模型的文档（如何获取：可在 [Django 官方文档网站](https://docs.djangoproject.com/en/2.2/topics/db/models/) 在线获取。用途：关于 Django 模型的综合资源。）

### 简介

在 Web 应用程序的世界里，数据为王。无论你是在跟踪用户资料、产品库存还是任何其他类型的信息，这些数据都需要存储在某个地方。在大多数现代 Web 应用程序中，这个“某个地方”就是一个关系型数据库。为了以一种有组织、结构化和高效的方式与这个数据库交互，Django 为我们提供了一个强大的功能，称为“模型”。

#### 什么是 Django 模型？

Django 模型是关于你的数据的单一、权威的来源。它包含了你所存储数据的基本字段和行为。本质上，Django 模型是一种定义应用程序数据结构和行为的方式。

每个模型映射到一个单独的数据库表，可以看作是一个继承自 `django.db.models.Model` 的 Python 类。这个类的每个属性代表数据库表中的一个字段。

#### 为什么使用 Django 模型？

1.  抽象化：你无需编写原始 SQL 查询，而是可以利用 Django 的对象关系映射（ORM）以更 Pythonic 的方式与数据交互。
2.  效率：Django 会为你处理数据库中记录的创建、读取、更新和删除，最大限度地减少人为错误的可能性。
3.  一致性：通过模型定义数据结构，你可以确保你的数据遵循特定的格式或模式。
4.  验证：模型允许数据验证，确保数据库中的数据是干净且可靠的。
5.  查询 API：Django 模型内置了查询 API，让你能够以复杂的方式检索和操作数据。

#### 定义一个简单的 Django 模型

让我们以一个简单的 `Blog` 应用程序为例，其中你想存储关于 `Post` 的信息。

```python
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    pub_date = models.DateTimeField('date published')
```

在这个例子中：

- `Post` 是我们的 Django 模型，对应于数据库中的一个表。
- `title` 是一个最多允许 200 个字符的字段。
- `content` 是一个允许无限字符数的字段。
- `pub_date` 是一个日期时间字段，用于存储帖子的发布时间。

一旦定义，这个模型就为 Django 提供了创建相应数据库表所需的所有信息。

#### 迁移：在数据库中反映模型更改

Django 内置了一个迁移框架，用于跟踪模型随时间的变化。当你创建或更新模型时，Django 可以自动生成在数据库中进行相应更改所需的 SQL。这使得随着应用程序数据结构的演变，能够进行平滑且受控的过渡。

## 结论

Django 模型是任何 Django Web 应用程序的基石。它们定义了应用程序数据的形状和行为，并内置了许多功能，使得与关系型数据库的交互变得简单高效。在后续章节中，我们将深入探讨为模型定义自定义行为、不同模型之间的关系以及高级查询技术。

### 创建我们的用户数据库模型

资产、资源和材料：

- Django：本课程中使用的主要 Web 框架。[在此处获取 Django](https://www.djangoproject.com/download/)。
- Python：使用的编程语言。确保你拥有 Python 3.7 或更新版本。[在此处下载 Python](https://www.python.org/downloads/)。
- Django 文档：你获取任何澄清或额外信息的首选之地。[查看 Django 文档](https://docs.djangoproject.com/)。
- 代码编辑器：例如 Atom，如前面章节所述。

简介：
在本章中，我们将重点使用 Django 的对象关系映射（ORM）系统创建一个用户数据库模型。Django 中的模型是表示数据库表的一种方式。通过定义一个 User 模型，你本质上是在勾勒数据库中用户表的模式。

1.  理解 Django 模型：
    Django 模型是关于你的数据的单一、权威的信息来源。它们包含了你所存储数据的基本字段和行为。通常，每个模型映射到一个单独的数据库表。
    每个模型都是一个继承自 `django.db.models.Model` 的 Python 类。模型的属性代表数据库字段。

2.  创建我们的用户模型：
    对于我们的 REST API，我们希望创建一个自定义用户模型，以便将来需要时可以扩展。Django 确实内置了一个用户模型，但通过创建自定义模型，你可以获得更大的灵活性。
    以下是分步指南：

    步骤 1：在你的 Django 应用目录中创建一个名为 `models.py` 的新文件（如果它尚不存在）。

    步骤 2：打开 `models.py` 并首先导入必要的模块：

    ```python
    from django.db import models
    from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
    ```

    步骤 3：定义自定义用户管理器：
    在我们创建自定义用户模型之前，我们需要为它定义一个管理器。这个管理器将包含像 `create_user` 和 `create_superuser` 这样的辅助函数。

    ```python
    class UserManager(BaseUserManager):
        def create_user(self, email, password=None, extra_fields):
            if not email:
                raise ValueError("The Email field must be set")
            email = self.normalize_email(email)
            user = self.model(email=email, extra_fields)
            user.set_password(password)
            user.save(using=self._db)
            return user

        def create_superuser(self, email, password=None, extra_fields):
            extra_fields.setdefault('is_staff', True)
            extra_fields.setdefault('is_superuser', True)
            return self.create_user(email, password, extra_fields)
    ```

步骤 4：定义 `User` 模型：
现在，让我们定义我们的 `User` 模型，它将继承自 `AbstractBaseUser` 和 `PermissionsMixin`。

```python
class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=30, blank=True)
    last_name = models.CharField(max_length=30, blank=True)
    date_joined = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    objects = UserManager()
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    def __str__(self):
        return self.email
```

在上面的代码中：

- 我们为用户定义了基本字段，如 `email`、`first_name`、`last_name` 等。
- 我们将 `USERNAME_FIELD` 设置为 email，它将用作身份验证的唯一标识符。
- `REQUIRED_FIELDS` 列表指明了通过命令行创建用户时必须提供的字段。由于 email 是我们的 USERNAME_FIELD，因此它不需要包含在此列表中。

结论：
模型定义完成后，你就有效地为用户数据库奠定了基础。接下来的步骤将涉及创建迁移（我们将在后续章节中处理），以将此模型应用到实际数据库，然后利用 Django Admin 来管理这些用户。

## 添加用户模型管理器

所需资产、资源和材料：

- Python：我们将使用的主要编程语言。（通过官方网站 [Python website](https://www.python.org/downloads/) 获取）
- Django：我们用于构建后端的 Web 框架。（通过 pip 安装：`pip install django`）
- Django 的 UserManager：Django 提供的一个内置管理器，用于处理用户相关任务。（随 Django 的 auth 框架一起提供）
- 文本编辑器/IDE：Atom、Visual Studio Code、PyCharm 或任何其他首选编辑器来编写代码。（在我们的示例中，我们使用 Atom，可以从官方网站 [Atom website](https://atom.io/) 下载）
- 命令行终端：你操作系统的默认终端/命令提示符即可。

引言
随着我们进一步深入设置数据库，我们需要考虑的一个关键方面是用户模型的管理。Django 提供了一个默认的 `UserManager`，但因为我们使用的是自定义用户模型，所以我们需要设置自己的用户模型管理器。
模型管理器本质上是一个类，它包含有助于创建和管理关联模型对象的函数。在用户模型的上下文中，这可以意味着创建用户、创建超级用户、检查用户凭证等函数。

### 添加用户模型管理器的步骤

1. 导入必要的模块和库：
打开你定义用户模型的 models.py 文件。在此文件的顶部，你需要导入一些模块：

```python
from django.contrib.auth.models import BaseUserManager
```

2. 定义自定义用户管理器：
在你的 User 模型下方，定义你的自定义用户管理器类。此类将继承自 `BaseUserManager`。

```python
class CustomUserManager(BaseUserManager):
    pass
```

3. 向管理器添加方法：
在 `CustomUserManager` 类中，我们将添加有助于创建用户和超级用户的方法。

```python
def create_user(self, email, password=None, extra_fields):
    """
    创建并返回一个具有给定电子邮件和密码的用户。
    """
    if not email:
        raise ValueError("用户必须拥有电子邮件地址")
    email = self.normalize_email(email)
    user = self.model(email=email, extra_fields)
    user.set_password(password)
    user.save(using=self._db)
    return user

def create_superuser(self, email, password=None, extra_fields):
    """
    创建并返回一个具有给定电子邮件和密码的超级用户。
    """
    extra_fields.setdefault('is_staff', True)
    extra_fields.setdefault('is_superuser', True)
    if extra_fields.get('is_staff') is not True:
        raise ValueError('超级用户必须设置 is_staff=True。')
    if extra_fields.get('is_superuser') is not True:
        raise ValueError('超级用户必须设置 is_superuser=True。')
    return self.create_user(email, password, extra_fields)
```

4. 将自定义用户管理器链接到用户模型：
现在我们已经创建了自定义用户管理器，我们需要将其与用户模型链接起来。在用户模型内部，创建一个自定义用户管理器的实例。

```python
class CustomUser(AbstractBaseUser, PermissionsMixin):
    ...
    objects = CustomUserManager()
```

`objects` 属性允许我们使用模型来调用自定义用户管理器方法。

### 解释：

- `create_user` 方法用于创建普通用户。它接收一个电子邮件（我们用作主要识别方式）和一个密码，以及你可能在用户模型中定义的任何其他字段。
- `create_superuser` 方法用于创建管理员用户。它确保 `is_staff` 和 `is_superuser` 字段设置为 True，这是需要访问 Django 管理站点的用户的要求。
- `normalize_email` 函数是 Django 提供的一个辅助函数，它规范化电子邮件的域名部分，将其转换为小写。这确保了数据库中的一致性。

## 结论

用户模型管理器就位后，我们的自定义用户模型现在能够处理用户创建和管理任务。这为在我们的 REST API 中管理用户奠定了基础。随着我们继续前进，你将看到在 Django 中拥有自定义用户模型和管理器的真正力量和灵活性，允许你根据应用程序的需求自定义用户行为。

（注：本章提供了自定义用户模型管理器的基本实现。根据你项目的具体需求，你可能需要添加更多方法或调整现有方法以适应你的需求。）

## 设置我们的自定义用户模型

本章所需的资产、资源和材料：

1. Django：（从 [Django 官方网站](https://www.djangoproject.com/download/) 获取）。我们用于构建应用程序的 Web 框架。
2. 文本编辑器（例如 Atom）：（从 [Atom 官方网站](https://atom.io/) 下载）。我们将在此编写代码。
3. Python 环境：确保你已安装 Python 并且可以运行 Django。（如果尚未设置，请参考第 12 章）。
4. 先前的用户模型：在第 18 章中创建。我们将修改和扩展此模型。
5. Django 文档：（可在 [Django 官方文档](https://docs.djangoproject.com/) 获取）。对于任何额外的细节或澄清都很有帮助。

引言：

Django 自带一个用于身份验证的内置用户模型。然而，我们通常希望扩展此模型或完全替换它以满足我们的应用程序需求。在本章中，我们将设置我们的自定义用户模型，使其成为身份验证和用户相关操作的默认用户模型。

步骤：

1. 查看现有用户模型：
    首先，导航到你在第 18 章中创建的用户模型。这是我们构建的基础。

2. 定义自定义用户模型：

在你的 models.py 文件中（在你的用户模型所在的应用程序内），首先导入必要的模块：

```python
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
```

为我们的自定义用户模型扩展 AbstractBaseUser 和 PermissionsMixin。这是一个基于你可能已有的用户模型构建的示例：

```python
class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']
    objects = CustomUserManager()
```

在这里，我们将 email 设置为 USERNAME_FIELD，这意味着用户将使用他们的电子邮件而不是用户名来登录。

3. 为用户模型创建自定义管理器：

CustomUser 模型需要一个管理器来处理其数据库操作。扩展 BaseUserManager 以创建我们的自定义管理器：

### 4. 更新 Django 设置：

现在我们的自定义用户模型已经设置好了，我们需要通知 Django 将其用作默认用户模型。在你的项目 `settings.py` 文件中，添加或更新以下行：

```python
AUTH_USER_MODEL = 'your_app_name.CustomUser'
```

将 `your_app_name` 替换为你的 `CustomUser` 模型所在的应用名称。

### 5. 数据库迁移：

由于我们对模型进行了更改，我们需要创建迁移并应用它们。可以通过以下命令完成：

```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. 测试：

测试我们的自定义用户模型以确保一切按预期工作至关重要。你可以通过 Django 管理站点或使用 Django 的 shell 创建一个新用户来完成此操作。如果一切设置正确，应该不会有任何错误，并且用户数据应该会被保存到数据库中。

## 结论：

在 Django 中自定义用户模型允许开发者根据应用程序的需求来定制身份验证系统。通过设置我们的自定义用户模型，我们对用户数据、其存储方式以及应包含的字段有了更多的控制权。

## 创建迁移并同步数据库

所需的资产、资源和材料：

- Python（用于运行 Django 命令）_（已在前面章节中安装）_
- Django 项目设置 _（已在前面章节中涵盖）_
- 终端或命令提示符
- 代码编辑器（如 Atom）_（已在前面章节中使用）_

### 1. 简介

Django 的数据库抽象 API 允许你创建、检索、更新和删除记录，而无需编写一行 SQL。这得益于其对象关系映射（ORM）层。Django 最强大的部分之一就是其 ORM，而迁移在其中扮演着关键角色。迁移是 Django 将你对模型所做的更改（添加字段、删除模型等）传播到数据库模式的方式。

### 2. 为什么需要迁移？

当我们在 Django 中定义或修改模型类时，实际的数据库表并不会立即受到影响。相反，Django 会跟踪这些更改，并允许我们使用迁移以系统的方式应用它们。迁移解决了在无需手动干预的情况下演进数据库模式的问题。

### 3. `makemigrations` 命令

在应用迁移之前，我们需要先创建它们。在对 Django 模型进行更改后：

1. 打开你的终端或命令提示符。
2. 导航到你的 Django 项目的根目录。
3. 输入以下命令：

```bash
python manage.py makemigrations
```

这将检查你的模型定义与数据库的当前状态，并生成迁移文件——这些脚本在运行时会修改你的数据库模式以反映你对模型所做的更改。

你应该会看到类似以下的输出：

```
Migrations for 'app_name':
    app_name/migrations/0002_auto_YYYYMMDD_HHMM.py
        - Create model YourModelName
```

### 4. 检查迁移的 SQL 代码

如果你想查看 Django 为特定迁移生成的 SQL 代码，可以使用以下命令：

```bash
python manage.py sqlmigrate app_name 0002_auto_YYYYMMDD_HHMM
```

将 `app_name` 替换为你的应用名称，将 `0002_auto_YYYYMMDD_HHMM` 替换为你想要检查的迁移文件名称。这将显示应用此迁移时将运行的 SQL 语句。

### 5. 应用迁移

要应用迁移并同步你的数据库，请使用以下命令：

```bash
python manage.py migrate
```

此命令会查看 `INSTALLED_APPS` 设置，找到尚未应用的迁移，并按正确的顺序对你的数据库运行它们。你应该会看到输出，指示哪些迁移正在被应用以及它们是否成功。

### 6. 回滚迁移

如果你犯了错误或想恢复到之前的数据库状态，Django 允许你“取消应用”迁移。要回滚一个迁移：

```bash
python manage.py migrate app_name 0001_initial
```

这将取消应用 `app_name` 在 `0001_initial` 之后的所有迁移。

### 7. 常见问题和提示

- 未检测到更改：如果你运行 `makemigrations` 并看到“未检测到更改”，请仔细检查你是否已保存 `models.py` 文件，并且模型的应用已包含在 `INSTALLED_APPS` 设置中。
- 依赖关系：迁移可能依赖于其他迁移甚至其他应用。Django 通常很智能，能确定正确的顺序，但有时你可能需要给它一些提示。
- 冲突的迁移：如果你在团队中工作，并且多个开发者正在对同一模型进行更改，你可能会遇到“冲突的迁移”。你需要与团队协调以手动解决这些问题。

### 8. 总结

在本章中，我们深入探讨了 Django 数据库迁移的世界。迁移允许我们以受控和系统的方式演进数据库模式。通过使用 `makemigrations` 和 `migrate` 命令，我们可以高效地处理模型的更改并保持数据库同步。请记住，在将迁移应用于生产环境之前，始终在开发环境中进行测试，以避免意外问题。

# 第 7 节：设置 Django 管理后台

## 创建超级用户

目标：在本章结束时，你应该能够为你的 Django 项目创建一个超级用户账户，该账户允许你访问 Django 管理界面。

资产和材料：

1. 命令行界面（CLI）：大多数操作系统都预装了此工具。我们将使用它来运行创建超级用户的命令。
   - 如何获取：大多数操作系统预装。
   - 用途：运行各种命令。
2. Django 项目：你应该已经在前面的章节中设置好了。
   - 如何获取：如果你一直跟着做，你应该已经设置好了。如果没有，请参考前面的章节。
   - 用途：超级用户将与此 Django 项目关联。
3. Django 的 manage.py 脚本：每个 Django 项目都预装了此脚本。
   - 如何获取：创建新的 Django 项目时自动生成。
   - 用途：管理 Django 项目的各个方面，包括创建超级用户。

简介

在 Django 中，“超级用户”一词指的是拥有所有权限且对 Django 管理界面拥有无限制访问权限的用户。Django 管理界面是一个强大的内置工具，允许你轻松管理应用程序中的数据。但是，要访问它，你需要一个具有必要权限的用户账户——即超级用户。

### 创建超级用户的步骤：

1. 打开你的命令行界面（CLI）：
   导航到你的 Django 项目所在的目录。如果你设置了虚拟环境，请确保已激活它。
2. 运行 `createsuperuser` 命令：
   在你的 Django 项目的根目录（即 `manage.py` 文件所在的位置），运行以下命令：
   ```bash
   python manage.py createsuperuser
   ```
   此命令告诉 Django 开始创建新超级用户的过程。
3. 填写所需信息：
   运行命令后，系统会提示你输入超级用户的详细信息：
   - 用户名：这将用于登录 Django 管理界面。选择一个你能记住的。
   - 电子邮件地址：一个有效的电子邮件地址。这对于密码恢复和通知很有用。
   - 密码：选择一个强密码。你需要输入两次以确认。

> 注意：如果你输入的任何信息与现有数据冲突（例如，具有相同用户名或电子邮件地址的现有用户），Django 会提醒你并要求你更正。

### 确认：

一旦你成功填写了所需信息且没有冲突，你应该会看到一条成功消息，确认超级用户已创建。

### 访问 Django 管理后台：

既然你已经创建了一个超级用户，你就可以访问 Django 管理后台界面了。如果 Django 服务器尚未运行，请使用以下命令启动它：

```bash
python manage.py runserver
```

打开一个网页浏览器，导航到 `http://127.0.0.1:8000/admin/` 或你为项目设置的相应地址。在这里，你可以使用刚刚创建的超级用户凭据登录。

## 结论：

恭喜！你已经成功为你的 Django 项目创建了一个超级用户。通过这个账户，你可以通过 Django 管理后台界面管理应用程序的各个方面。随着你继续开发应用程序，这项能力将被证明是无价的。

注意：请记住，超级用户账户对 Django 管理后台拥有完全访问权限，并且可以对应用程序及其数据进行重大更改。在使用超级用户账户时务必谨慎，并考虑为日常任务创建具有有限权限的其他用户。

## 启用 Django 管理后台

资产、资源和材料：

- Django 框架（安装方式：`pip install django==2.2`）
- Django 管理后台站点（内置于 Django 2.2；无需额外安装）
- 一个 Django 项目和应用（请参考前面的章节了解如何创建）

## 简介：

Django 管理后台是 Django 提供的一个功能强大且可用于生产环境的管理界面。它让开发者和站点管理员能够轻松地在数据库中创建、读取、更新和删除记录。在本章中，我们将学习如何为我们的项目启用和自定义 Django 管理后台站点。

### 启用 Django 管理后台的步骤：

#### 1. 确保 `django.contrib.admin` 在 `INSTALLED_APPS` 中：

打开你的 Django 项目的 `settings.py` 文件。你应该能找到一个名为 `INSTALLED_APPS` 的列表。默认情况下，`django.contrib.admin` 应该已经是该列表的一部分：

```python
INSTALLED_APPS = [
    ...
    'django.contrib.admin',
    ...
]
```

如果它不存在，请添加它。

#### 2. 确保中间件已设置：

管理后台站点需要某些中间件类才能正常运行。确保你的 `settings.py` 中的 `MIDDLEWARE` 设置包含：

```python
MIDDLEWARE = [
    ...
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    ...
]
```

#### 3. 设置 URL：

为了使管理后台站点可以通过 URL 访问，你需要将其 URL 包含在你的项目的 `urls.py` 文件中。

首先，确保你在文件顶部导入了 `admin` 和 `include`：

```python
from django.contrib import admin
from django.urls import path, include
```

然后，将以下路径添加到 `urlpatterns` 列表中：

```python
urlpatterns = [
    ...
    path('admin/', admin.site.urls),
    ...
]
```

此配置将使管理后台站点在你的网站上可通过 `/admin/` 访问。

#### 4. 运行迁移：

确保你已经应用了最新的迁移，因为 Django 管理后台使用了一些内置模型。运行：

```bash
python manage.py migrate
```

#### 5. 访问管理后台站点：

启动你的开发服务器：

```bash
python manage.py runserver
```

现在，在你的浏览器中导航到 `http://127.0.0.1:8000/admin/`。你应该会看到 Django 管理后台的登录页面。

### 自定义管理后台界面：

虽然 Django 管理后台站点的默认配置是可用的，但你可以做很多事情来自定义其外观和行为：

#### 1. 品牌标识：

更改站点标题、标题和索引标题：

```python
admin.site.site_header = "MyApp Administration"
admin.site.site_title = "MyApp Admin Portal"
admin.site.index_title = "Welcome to MyApp Admin"
```

将上述代码放在你的应用的 `admin.py` 文件中。

#### 2. 注册模型：

如果你已经构建了模型（比如我们第 18 章的用户数据库模型），你可以将它们注册到管理后台站点以管理其记录：

```python
from .models import UserProfile
admin.site.register(UserProfile)
```

将此代码添加到你的应用的 `admin.py` 文件中，你将能够从管理后台界面管理 `UserProfile` 记录。

## 结论：

Django 管理后台站点是管理应用程序数据的强大工具。通过如上所述启用和自定义它，你已经为自己提供了一个强大的界面来管理应用程序的后端数据。

## 测试 Django 管理后台

所需的资产、资源和材料：

- 来自前面章节的 Django 项目设置。
- Django 的内置开发服务器。
- 网页浏览器（例如 Google Chrome、Mozilla Firefox）。
- 在第 22 章创建的超级用户账户。
- *（可选）* 我们创建的用户模型的示例数据。

目的：Django 管理后台站点提供了一个基于 Web 的界面来管理你的应用程序数据。测试 Django 管理后台可以确保你已正确设置它，并且可以轻松地与应用程序的数据模型进行交互。

### 1. Django 管理后台测试简介

Django 管理后台站点是 Django 最受赞誉的功能之一。它提供了一种快速的方法来为你的应用程序数据创建管理控制台，无需任何额外的编码。测试确保你已正确设置它，并且它按预期运行。

### 2. 启动 Django 开发服务器

在测试 Django 管理后台之前，你必须确保你的开发服务器正在运行：

```bash
python manage.py runserver
```

运行此命令后，你的服务器将启动，你将看到一条输出，指示服务器运行的 IP 地址和端口，通常是 `http://127.0.0.1:8000/`。

### 3. 访问 Django 管理后台界面

打开你的网页浏览器，通过在服务器地址后附加 `/admin` 来导航到管理后台页面：`http://127.0.0.1:8000/admin`

你将看到一个登录页面。输入你在第 22 章创建的超级用户凭据。

### 4. 导航管理后台仪表板

成功登录后，你将进入 Django 管理后台仪表板。在这里，你将看到代表项目中每个应用的组，以及你已注册到管理后台界面的数据库模型列表。如果你按照前面的章节操作，你应该会看到我们的用户模型被列出。

### 5. 添加和编辑数据

#### 1. 创建新的用户记录：

- 点击 `Users` 或相应的模型名称。
- 从右上角选择“添加用户”或类似选项。
- 在表单中填写用户详细信息。
- 点击底部的“保存”。

#### 2. 编辑现有记录：

- 从用户列表中，点击一个用户名。
- 根据需要修改任何字段。
- 点击底部的“保存”。

### 6. 过滤和搜索记录

在用户列表页面，你会注意到右侧边栏有过滤选项。这允许你根据某些条件快速过滤记录。此外，在顶部有一个搜索栏。尝试搜索用户名或电子邮件，看看搜索功能是如何工作的。

### 7. 删除记录

要删除记录：

- 从用户列表中，勾选你想要删除的用户旁边的复选框。
- 从顶部的“操作”下拉菜单中，选择“删除选定的用户”并点击“执行”。
- 在下一页确认删除。

### 8. 从 Django 管理后台注销

完成操作后，请务必从管理后台界面注销，尤其是在公共或共享机器上。点击仪表板右上角的“注销”链接。

### 9. 确保安全

请注意：Django 管理后台站点功能强大，你应该谨慎对待谁有权访问它。在实际部署中，请考虑：

- 为超级用户使用强而唯一的密码。
- 通过 IP 限制对管理后台界面的访问。
- 定期审查和更新用户权限。

### 10. 结论

Django 管理后台界面提供了一种强大的方式来管理应用程序数据。通过彻底测试其功能，你可以确保数据管理任务可以轻松高效地执行。

请记住，Django 管理后台只是你 Django 工具箱中的一个工具。虽然它非常适合快速的数据管理任务，但你可能经常需要更定制化的解决方案，我们将在后面的章节中探讨。

# 第 8 节：API 视图简介

### 什么是 APIView？

资产、资源和材料：

- 官方 Django REST Framework 文档：官方文档是一个全面的资源，提供了对包括 APIView 在内的各种组件的深入理解。可以在这里访问 [here](https://www.django-rest-framework.org/)。
- Python：我们将使用的主要语言。如果尚未安装，你可以下载并安装 Python。

### 引言

在我们深入探讨 APIView 及其强大功能之前，有必要先阐明 Django 中“视图”这一基本概念。在 Django 中，视图本质上是一个 Python 函数（或类），它接收一个 Web 请求并返回一个 Web 响应。这个响应可以是一个 HTML 页面、一个重定向、一个 404 错误，甚至在 API 场景下可以是一个 JSON 响应。借助 Django REST Framework（DRF），这一概念得到了增强，使其在处理 API 时更加强大和灵活。

## APIView：为 API 提升 Django 视图

APIView 是 Django REST Framework 提供的一个基于类的视图，专为构建 API 而设计。Django 的传统视图侧重于处理 HTML 响应和管理表单，而 APIView 则专注于返回 JSON 或 XML 等结构化数据。

APIView 的主要特性：

- 1. HTTP 方法处理：无需为不同的 HTTP 方法（GET、POST、PUT 等）定义单独的视图，使用 APIView，你可以通过在视图类上定义相应的方法（`get()`、`post()` 等）来处理这些方法。
- 2. 请求解析：APIView 将传入的 HTTP 请求转换为更易于 Django 处理的格式。传入的请求数据被解析为 Python 数据类型，从而简化了数据处理过程。
- 3. 响应包装：它提供了一个 `Response` 对象，确保你返回的数据被渲染为适当的内容类型（如 JSON）。
- 4. 异常处理：无需手动处理异常，APIView 提供了内置的异常处理机制，能将许多 Python 异常转换为适当的 HTTP 响应。
- 5. 认证：你可以轻松地为视图添加认证，确保只有授权用户才能与你的 API 端点交互。
- 6. 权限处理：APIView 提供了定义“谁可以做什么”的机制。例如，你可以指定只有经过认证的用户才能发布数据，但任何人都可以读取。
- 7. 内容协商：根据客户端请求确定最佳的响应格式。例如，虽然你的 API 可能同时支持 JSON 和 XML，但它可以根据客户端的请求头使用内容协商来决定返回哪种格式。

## 为什么使用 APIView？

考虑到 Django 已经拥有一个强大的视图创建系统，人们可能会疑惑 APIView 的必要性。原因如下：

- 结构化数据：由于我们的目标是构建 API，我们需要大量处理结构化数据（如 JSON）。APIView 简化了这一过程。
- 可重用性：通过基于类的视图，你可以通过扩展基类视图或混入额外行为来重用通用功能。
- 更好的抽象：APIView 抽象了 Web API 的许多复杂性，使开发者能够专注于应用逻辑，而非样板代码。

## 结论

APIView 是 Django REST Framework 中的一个核心工具，它提升了构建 Web API 的过程。通过提供一系列专为 API 开发量身定制的功能，它使开发者的工作变得更加简单和高效。

## 创建第一个 APIView

资产与材料：

- Django 框架（通过在终端或命令提示符中执行 `pip install django` 获取）
- Django REST Framework (DRF)（通过在终端或命令提示符中执行 `pip install djangorestframework` 获取）
- Atom 编辑器（从 [atom.io](https://atom.io/) 下载）
- Python（已安装在开发机器上）
- 虚拟环境（在前面的章节中创建）
- ModHeaders（作为浏览器扩展安装）

简介：

APIView 是 Django REST Framework 的核心部分，它提供了一种定义针对不同类型 HTTP 方法执行逻辑的方式。与主要处理网页请求和响应的常规 Django 视图不同，APIView 旨在处理 API 端点，返回适合其他软件或前端框架使用的 JSON 或 XML 响应。

### 步骤 1：搭建舞台

在我们开始创建 APIView 之前，请确保你的 Django 项目已设置好并且 Django REST Framework 已安装。如果没有，请回顾前面的章节以确保你的环境已准备就绪。

### 步骤 2：为 API 创建一个应用

为了清晰起见，我们将为一个简单的模型——比如“消息”——创建一个 API。首先创建一个应用：

```bash
python manage.py startapp messages_app
```

创建应用后，将其添加到项目 `settings.py` 的 `INSTALLED_APPS` 列表中：

```python
INSTALLED_APPS = [
    ...
    'rest_framework',
    'messages_app',
]
```

### 步骤 3：定义一个模型

在 `messages_app` 的 `models.py` 中，定义一个简单的模型：

```python
from django.db import models

class Message(models.Model):
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.content[:50]
```

运行迁移：

```bash
python manage.py makemigrations
python manage.py migrate
```

### 步骤 4：创建一个序列化器

序列化器允许将复杂的数据类型（如 Django 模型）转换为可以轻松渲染为 JSON 的格式。在 `messages_app` 目录中，创建一个名为 `serializers.py` 的文件：

```python
from rest_framework import serializers
from .models import Message

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'content', 'created_at']
```

### 步骤 5：创建我们的第一个 APIView

现在，让我们深入主题——APIView。在 `messages_app` 的 `views.py` 文件中，首先导入必要的库：

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Message
from .serializers import MessageSerializer
```

接下来，定义你的 APIView：

```python
class MessageList(APIView):
    """
    列出所有消息或创建新消息。
    """
    def get(self, request):
        messages = Message.objects.all()
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = MessageSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

在上面的代码中：

- `get` 方法获取所有消息并返回它们。
- `post` 方法创建一个新消息。

### 步骤 6：配置 URL

要使你的 APIView 可访问，你需要将其与一个 URL 关联起来。在 `messages_app` 的 `urls.py` 中：

```python
from django.urls import path
from .views import MessageList

urlpatterns = [
    path('messages/', MessageList.as_view(), name='message-list'),
]
```

确保此应用的 URL 包含在主项目的 `urls.py` 中。

### 步骤 7：使用 ModHeaders 进行测试

一切设置就绪后，启动你的 Django 服务器：

```bash
python manage.py runserver
```

打开浏览器，导航到 ModHeaders 扩展，并将请求头 “Content-Type” 设置为 “application/json”。现在，你可以访问 `http://127.0.0.1:8000/messages/` 来查看你的消息。你也可以使用 Postman 或 CURL 等工具来测试 POST 方法并创建新消息。

## 结论

恭喜你使用 Django REST Framework 创建了你的第一个 APIView！这个基础构建块将在你后续章节中深入构建更复杂 API 时发挥重要作用。

## 配置视图 URL

本章的资产、资源与材料：

- 1. Django：我们将使用 Django，这个用于构建 Web 应用程序的 Web 框架。（获取方式：在终端中使用命令 `pip install django==2.2`）。

### 简介

Django 在 Web 开发中如此受欢迎的主要原因之一是其内置的 URL 路由。本章将教你如何在 Django 中为你的 API 视图配置 URL，以便用户能够访问你设置的 API 端点。

### 配置视图 URL 的分步指南

#### 1. 导入所需的库

在设置 URL 之前，我们需要导入一些必要的库。打开你 Django 应用目录中的 `urls.py` 文件。如果你的应用目录中没有 `urls.py`，请创建一个。然后，导入必要的库：

```python
from django.urls import path
from . import views
```

这里，`path` 是我们用来定义 URL 模式的函数，我们导入视图以便将它们链接到这些 URL。

#### 2. 定义 URL 模式列表

在 Django 中，URL 在一个名为 `urlpatterns` 的列表中定义。这个列表将包含我们应用程序的所有路由。如果你有现有的路由，我们只需添加新的路由即可。否则，初始化列表：

```python
urlpatterns = []
```

#### 3. 将 API 视图添加到 URL 模式

回想一下我们在上一章创建的第一个 APIView。我们现在将把这个视图链接到一个 URL。在 `urlpatterns` 列表中添加以下行：

```python
path('api-view/', views.OurApiView.as_view(), name='api-view')
```

这里：

- `api-view/` 是 URL 端点。当用户访问此 URL 时，他们将访问我们创建的 API 视图。
- `views.OurApiView.as_view()` 告诉 Django 使用我们在视图中创建的 `OurApiView`。`as_view()` 方法是 Django 的内置方法，用于将基于类的视图转换为基于函数的视图，这是 URL 路由所需要的。
- `name='api-view'` 是我们给这个 URL 模式起的名字。它对于反向 URL 匹配很有用。

#### 4. 在项目 URL 中包含应用 URL

如果你还没有这样做，你必须在项目的主 `urls.py`（位于主项目文件夹中，而不是应用文件夹）中包含应用的 URL。打开项目的 `urls.py` 并确保你有：

```python
from django.contrib import admin
from django.urls import path, include
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('<your_app_name>.urls')),
]
```

这里，`include('<your_app_name>.urls')` 本质上导入了你在应用 `urls.py` 中定义的所有 URL 模式。

## 测试你的设置

要验证一切是否设置正确，请运行开发服务器：

```bash
python manage.py runserver
```

在你的 Web 浏览器中访问 `http://127.0.0.1:8000/api-view/`。你应该会看到来自你的 API 视图的响应。

## 结论

恭喜！你已成功为你的 API 视图配置了 URL。这个基本步骤确保了你的后端服务可以通过清晰且逻辑的 Web 地址访问。在接下来的章节中，我们将深入探讨 Django 和 Django REST 框架的强大功能，使你能够创建更复杂和健壮的 Web API。

## 测试我们的 API 视图

**资产、资源和材料：**

- Django REST Framework (DRF)：用于在 Django 中构建 API。（通过运行命令 `pip install djangorestframework` 获取）
- Postman：一个流行的 API 测试工具。（从 [Postman 官方网站](https://www.postman.com/downloads/) 下载并安装）
- Python 的内置 unittest 库：用于在 Django 中编写测试用例。（随 Python 预装）
- 示例数据：一些用于测试 POST、PUT、PATCH 和 DELETE 操作的示例数据。
- ModHeader 浏览器扩展：帮助设置请求头。（从浏览器的扩展商店安装）

**简介：**

在设置好我们的 APIView 后，确保它按预期工作至关重要。测试不仅仅是软件开发的一个阶段，它还是决定我们应用程序健壮性和可靠性的关键实践。在本章中，我们将深入探讨测试 API 视图的功能。

### 步骤 1：使用 Postman 进行手动测试

在深入编写自动化测试之前，使用 Postman 进行一些手动测试是有益的。

#### 1.1. 设置 Postman

- 安装后启动 Postman。
- 创建一个新请求。
- 设置请求类型。对于我们的第一个测试，我们将使用 `GET`。
- 输入我们 API 端点的 URL。
- 如果需要身份验证，请导航到 `Authorization` 选项卡并提供必要的凭据。

#### 1.2. 发送请求

- 对于 `GET`：只需点击 `Send` 按钮。检查响应。它应该与我们 API 的预期数据相对应。
- 对于 `POST`：将请求类型更改为 `POST`。转到 `Body` 选项卡并以 JSON 格式提供所需的数据。点击 `Send` 并检查响应。

记得类似地测试其他 HTTP 方法，如 `PUT`、`PATCH` 和 `DELETE`。

### 步骤 2：使用 Django 的 Unittest 进行自动化测试

虽然手动测试非常适合快速检查，但我们需要自动化测试来确保我们 API 的健壮性。

#### 2.1. 设置测试文件

导航到应用目录并创建一个名为 `test_api.py` 的文件。

#### 2.2. 编写测试用例

这是一个如何测试 GET 方法的简单示例：

```python
from rest_framework.test import APIClient
from rest_framework import status
from django.urls import reverse
class APIViewTestCase(unittest.TestCase):
    def setUp(self):
        self.client = APIClient()
        self.api_url = reverse('name_of_the_view')
    def test_api_get_request(self):
        response = self.client.get(self.api_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
```

记得为 `POST`、`PUT`、`PATCH` 和 `DELETE` 编写类似的测试。

#### 2.3. 运行测试

导航到项目的根目录并运行以下命令：

```bash
python manage.py test app_name
```

将 `app_name` 替换为你的应用名称。

### 步骤 3：使用 ModHeader 设置请求头

在某些情况下，你可能需要为请求设置请求头。使用 ModHeader 浏览器扩展，你可以轻松设置基于令牌的身份验证等请求头。

#### 3.1. 设置 ModHeader

安装扩展后：

- 通过点击浏览器中的图标打开它。
- 通过点击 `+` 号添加新请求头。
- 输入请求头名称和值。

#### 3.2. 将 ModHeader 与 Postman 一起使用

如果你的 API 需要像令牌这样的请求头进行身份验证，请确保在手动测试期间在 Postman 中设置它。

## 结论

测试是 API 开发过程中的关键实践。通过 Postman 和 Django 的 unittest 等工具，我们可以确保我们的 API 按预期运行，为进一步的开发奠定坚实的基础。请记住，在继续开发新功能或修改现有功能时，始终要迭代和完善你的测试。

## 创建序列化器

**资产、资源和材料：**

- Django（从 [Django 官方网站](https://www.djangoproject.com/download/) 获取）
- Django REST framework（使用 pip 安装：`pip install djangorestframework`）
- Atom 或任何其他你选择的代码编辑器（从 [Atom 官方网站](https://atom.io/) 获取 Atom）

**简介：**

序列化是将复杂数据类型（如查询集和模型实例）转换为可以轻松渲染为 JSON、XML 或其他内容类型的原生数据类型的过程。在 Django REST framework 中，序列化器允许将复杂数据类型转换为易于渲染的类型，类似于 Django 的 Form 和 ModelForm 类。在本章中，我们将重点介绍如何使用 Django REST framework 创建一个序列化器，以适合渲染为 JSON 的方式表示我们的数据。

**创建序列化器的步骤：**

### 1. 设置：

首先，确保你已安装 Django REST framework。如果没有，请使用 pip 安装：

```bash
pip install djangorestframework
```

### 2. 创建基础序列化器：

导航到你的 Django 应用目录，创建一个名为 `serializers.py` 的新文件。该文件将包含你应用的所有序列化器。

在 `serializers.py` 中，首先导入所需的模块：

```python
from rest_framework import serializers
from .models import YourModelName
```

将 `YourModelName` 替换为你希望序列化的模型名称。

### 3. 定义序列化器类：

接下来，你将为你的模型定义一个序列化器类：

```python
class YourModelNameSerializer(serializers.Serializer):
    field_name1 = serializers.CharField(max_length=100)
    field_name2 = serializers.DateField()
    # 根据你的模型添加更多字段。
```

此处，将 `YourModelNameSerializer` 替换为一个合适的名称，并在序列化器中定义你模型的所有字段。

### 4. 使用 ModelSerializer 以简化操作：

你可以使用 `ModelSerializer` 类来代替手动定义每个字段，它将自动创建一个包含与模型字段相对应字段的序列化器。这类似于 Django 的 `ModelForm` 的工作方式：

```python
class YourModelNameModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = YourModelName
        fields = ['field_name1', 'field_name2', 'field_name3']
        # 你也可以使用 fields = '__all__' 来包含所有字段。
```

### 5. 使用序列化器验证数据：

序列化器还允许你轻松地验证数据。例如，如果你想确保某个字符字段不包含任何数字，你可以这样做：

```python
def validate_field_name1(self, value):
    if any(char.isdigit() for char in value):
        raise serializers.ValidationError("This field should not contain numbers.")
    return value
```

将此方法添加到你的序列化器类内部。

### 测试序列化器：

测试你的序列化器以确保其正常工作至关重要。你可以使用 Django 的 shell 来完成此操作：

1.  运行 Django shell：

```bash
python manage.py shell
```

2.  导入你的模型和序列化器：

```python
from your_app_name.models import YourModelName
from your_app_name.serializers import YourModelNameSerializer
```

3.  创建一个模型实例并将其序列化：

```python
instance = YourModelName.objects.create(field_name1="Test Data", field_name2="2023-10-20")
serializer = YourModelNameSerializer(instance)
print(serializer.data)
```

这将在控制台中显示序列化后的数据。

## 结论：

Django REST 框架中的序列化器在将复杂数据转换为易于在客户端应用程序上渲染和处理的格式方面起着关键作用。通过遵循本章概述的步骤，你已经为有效地序列化数据奠定了基础，使其更易于与 API 交互。

### 练习：

1.  为你的应用程序中的另外两个模型创建序列化器。
2.  在 Django shell 中测试你的序列化器，确保它们正常工作。
3.  为你的序列化器中的一个字段添加自定义验证。在 Django shell 中测试此验证。

## 向 APIView 添加 POST 方法

### 本章的资产、资源和材料：

- Django 和 Django REST Framework：使用 pip 下载并安装它们（`pip install django djangorestframework`）。
- 开发环境：我们在本书中使用 Atom，但任何 IDE 或文本编辑器都可以。
- 一个功能正常的 Django 项目（如前面章节所设置）。
- `serializers.py` 文件（在第 29 章中创建）。

### 目的：
POST 方法允许客户端向服务器提交数据，以作为新实体进行处理。在我们的 REST API 上下文中，这意味着在我们的数据库中创建新记录。

### 简介

Django REST Framework 中的 APIView 提供了一种简单的方法来按方法处理 HTTP 方法，如 GET、POST、PUT 等，而不是使用单个类或函数。在本章中，我们将重点实现 POST 方法，以允许客户端在我们的系统中创建新实体。

### 1. 理解流程

在深入代码之前，理解流程至关重要：

1.  客户端发送带有数据的 POST 请求。
2.  APIView 处理此请求。
3.  序列化器验证数据。
4.  如果有效，数据将被保存到数据库中。
5.  向客户端返回响应。

### 2. 更新 `serializers.py`

在添加 POST 方法之前，请确保你已经为将要接收的数据定义了一个序列化器。我们在第 29 章中创建了一个基础序列化器，因此我们将使用它。

```python
#### serializers.py
from rest_framework import serializers

class YourEntitySerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    description = serializers.CharField()
    # 根据需要添加其他字段。
```

### 3. 在 APIView 中实现 POST

导航到你的视图文件，找到你的 APIView 所在位置。我们将在这里添加一个 post 方法。

```python
#### views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import YourEntitySerializer

class YourEntityApiView(APIView):
    def post(self, request):
        serializer = YourEntitySerializer(data=request.data)
        if serializer.is_valid():
            name = serializer.validated_data.get('name')
            # 处理将数据保存到数据库。
            # 目前，我们只返回名称。
            return Response({'name': name},
                            status=status.HTTP_201_CREATED)
        else:
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )
```

### 4. 处理数据库保存

目前，我们只在响应中返回了名称。在实际场景中，你会将此数据保存到数据库中。

例如，如果你正在使用 Django 模型：

```python
from .models import YourEntity

#### 在 post 方法内部，检查序列化器有效后：
entity = YourEntity.objects.create(name=name, description=description)
return Response({'id': entity.id, 'name': entity.name}, status=status.HTTP_201_CREATED)
```

### 5. 更新 URL

确保你的 APIView 已连接到一个 URL 端点以测试 POST 方法。在 `urls.py` 中：

```python
from django.urls import path
from .views import YourEntityApiView

urlpatterns = [
    path('entity/', YourEntityApiView.as_view(), name='entity-create')
]
```

### 6. 测试 POST 请求

要进行测试，你可以使用 Postman 或 CURL 等工具：

```bash
curl -X POST -H "Content-Type: application/json" -d '{"name": "Sample Name", "description": "Sample Description"}' http://localhost:8000/entity/
```

你应该会收到一个 HTTP 201 状态的响应，其中包含你提供的名称。

## 结论

POST 方法对于在 REST API 中创建新资源至关重要。借助 Django REST Framework 的 APIView 和序列化器，我们可以高效地实现和验证传入数据。

## 测试 POST 功能

### 资产、资源和材料：

- Postman（或任何其他 API 测试工具，如 Insomnia）：一个 API 测试工具，用于向我们的 API 发送请求并查看响应。可从 [https://www.postman.com/downloads/](https://www.postman.com/downloads/) 免费下载。
- 源代码：来自前面的章节，特别是第 30 章，我们向 APIView 添加了 POST 方法。

## 简介：
在上一章中，我们向 APIView 添加了一个 POST 方法，允许客户端创建新的数据条目。本章重点测试该 POST 功能，确保其按预期运行。

### 步骤 1：设置 Postman：

1.1. 安装并打开 Postman：
- 如果你还没有安装，请从上面提供的链接下载并安装 Postman。
- 安装完成后启动 Postman。

1.2. 设置新请求：
- 点击 ‘+’ 标签页以打开一个新请求。
- 使用下拉菜单将请求类型设置为 “POST”。
- 输入你的 API 端点 URL。例如，如果你在本地运行 Django 服务器，它可能类似于 `http://127.0.0.1:8000/api/your-endpoint/`。

### 步骤 2：构建 POST 请求：

2.1. 请求头：
- 将 “Content-Type” 设置为 “application/json”。这告诉我们的 API 我们正在以 JSON 格式发送数据。

2.2. 请求体：
- 点击 URL 字段下方的 “Body” 标签页。
- 选择 “raw” 输入，并确保右侧下拉菜单中选择了 “JSON (application/json)”。
- 以 JSON 格式输入你的数据。例如，如果你正在测试一个用户创建 API，它可能看起来像这样：
```json
{
    "name": "John Doe",
    "email": "john.doe@example.com"
}
```

步骤 3：发送 POST 请求：
3.1. 点击蓝色的“Send”按钮。Postman 将向你的 API 发送 POST 请求。
3.2. 分析响应：
- 请求完成后，Postman 将在下方显示响应。通常，对于成功的 POST 请求，你会期望看到一个 201 状态码，表示资源已成功创建。
- 检查响应体中的返回数据。理想情况下，它应该显示创建的对象及其任何系统生成的字段，例如 ID。

步骤 4：处理错误：
4.1. 如果 POST 请求返回错误，请仔细分析响应体。Django 和 Django REST Framework 通常会返回描述性的错误信息，可以指导你解决问题。
4.2. 常见错误包括：
- 400 Bad Request：通常表示你发送的数据无效。检查所有必需字段是否存在且具有有效值。
- 403 Forbidden：表示权限问题。确保你的 API 已设置为允许来自测试源的 POST 请求。
- 500 Internal Server Error：一个通用错误。检查你的 Django 服务器日志以获取更具体的信息。

步骤 5：附加测试：
5.1. 为了彻底测试 POST 功能：
- 尝试发送不完整或无效的数据，以检查你的 API 是否正确验证传入数据并返回适当的错误消息。
- 如果你的 API 使用身份验证，请同时测试经过身份验证和未经身份验证的请求，以确保安全措施正常工作。
- 测试幂等性。如果你多次发送相同的 POST 请求，它是否会创建多个条目（如果这是预期的）或者是否会拒绝重复项？

结论：
测试是开发过程中的关键步骤，确保我们的 API 按预期运行并准备好投入生产。通过彻底测试我们的 POST 功能，我们可以对其行为以及处理来自用户或其它服务的真实数据的能力充满信心。

## 添加 PUT、PATCH 和 DELETE 方法

本章所需的资产、资源和材料：
- Python（从 [https://www.python.org/downloads/](https://www.python.org/downloads/) 获取）：Django 所使用的编程语言。
- Django（通过 pip 安装：`pip install django==2.2`）：我们的主要 Web 框架。
- Django REST Framework (DRF)（通过 pip 安装：`pip install djangorestframework==3.9`）：提供创建 Web API 的工具。
- Postman（从 [https://www.postman.com/downloads/](https://www.postman.com/downloads/) 获取）：用于测试的 API 客户端。
- 你现有的 Django 项目：你应该已经从前面的章节中设置好了项目。

### 介绍

到目前为止，你已经了解了 Django Rest Framework (DRF) 中的 APIView，并且已经创建了一个用于检索信息的 GET 端点。但大多数 API 不仅仅是检索数据；它们还允许客户端修改数据。在本章中，我们将向我们的 APIView 添加三种方法：PUT（用于完全更新对象）、PATCH（用于部分更新）和 DELETE（用于删除对象）。

### 步骤 1：理解 HTTP 方法

在深入编码之前，让我们理解每种方法的用途：
- PUT：此方法用于更新整个对象。如果请求中缺少任何字段，则假定缺失的字段应设置为其默认值。
- PATCH：此方法允许部分更新。因此，如果你只想更改对象的某个特定部分而保持其余部分不变，你会使用 PATCH。
- DELETE：此方法很直接——用于删除对象。

### 步骤 2：修改你的 APIView

现在，让我们向现有的 APIView 添加这三种方法。导航到你的 `views.py`，你的 APIView 就位于其中。

```python
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
#### (假设你已经导入了一个序列化器和模型)
class YourModelNameAPIView(APIView):
    ... # 你之前编写的方法在这里
    def put(self, request, pk=None):
        """完全更新一个对象"""
        obj = self.get_object(pk)
        serializer = YourModelNameSerializer(obj, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    def patch(self, request, pk=None):
        """部分更新一个对象"""
        obj = self.get_object(pk)
        serializer = YourModelNameSerializer(obj, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    def delete(self, request, pk=None):
        """删除一个对象"""
        obj = self.get_object(pk)
        obj.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
```

### 步骤 3：测试方法

添加方法后，必须测试它们以确保它们按预期工作。
1. 测试 PUT：
    - 启动 Postman。
    - 将请求类型设置为 PUT。
    - 提供端点 URL。
    - 在正文中，设置你要更新的数据。
    - 发送请求。
    - 响应应反映更新后的数据。
2. 测试 PATCH：
    - 类似地，在 Postman 中，将请求类型设置为 PATCH。
    - 提供端点 URL 以及你只想更新的特定数据。
    - 发送请求。
    - 响应应反映部分更新。
3. 测试 DELETE：
    - 在 Postman 中，将请求类型设置为 DELETE。
    - 提供你要删除的对象的端点 URL。
    - 发送请求。
    - 你应该收到一个 204 No Content 响应，表示删除成功。

## 结论

PUT、PATCH 和 DELETE 方法对于功能完备的 API 至关重要，允许用户根据需要更新和删除数据。到目前为止，你的 APIView 应该能够处理所有 CRUD 操作，使其成为一个完整且功能齐全的端点。

在下一章中，我们将更广泛地测试这些方法，以确保它们处理所有场景和边缘情况。

以上就是我们关于在 APIView 中添加 PUT、PATCH 和 DELETE 方法的章节。请记住，在实现新功能后，务必进行彻底测试，以确保一切按预期工作。

## 测试 PUT、PATCH 和 DELETE 方法

资产、资源和材料：
- Postman（用于向我们的 API 发送 PUT、PATCH 和 DELETE 请求。从 [Postman 官方网站](https://www.postman.com/downloads/) 下载并安装。用途：Postman 是一个 API 测试工具，有助于向 API 发送请求并接收响应。）
- 你的 Django 开发服务器（必须正在运行。我们在前面的章节中已经设置好了。）
- API URL（我们之前创建的 APIView 的端点。）

#### 介绍：

在前面的章节中，我们设置了 APIView 并向其添加了 PUT、PATCH 和 DELETE 方法。在本章中，我们将测试这些方法以确保它们正常工作。测试是 Web 开发的一个关键方面，以确保你的应用程序按预期运行。

### 1. 设置 Postman：

在开始之前，确保你已安装 Postman。安装后：
1. 打开 Postman。
2. 创建一个新请求。
3. 从左侧的下拉菜单中，选择请求类型。我们将从 PUT 开始。

### 2. 测试 PUT 方法：

PUT 方法用于更新资源。让我们测试它：
1. 在 Postman 中，输入你要更新的资源的 API URL。
2. 选择 PUT 请求方法。
3. 转到“Body”选项卡。
4. 选择“raw”和“JSON”格式。
5. 以 JSON 格式添加你要更新的数据。
6. 点击“Send”。

你应该收到一个响应，表明资源已更新。如果没有，请检查错误消息以调试问题。

示例：
假设你有一个 API 端点位于 `http://localhost:8000/api/profile/1/`，代表

### 3. 测试 PATCH 方法：

PATCH 方法用于部分更新，与完全更新资源的 PUT 方法不同。测试 PATCH 的步骤与 PUT 类似：

1.  在 Postman 中，输入你想要部分更新的资源的 API URL。
2.  选择 PATCH 请求方法。
3.  转到 ‘Body’ 选项卡。
4.  选择 ‘raw’ 和 ‘JSON’ 格式。
5.  以 JSON 格式添加你想要更新的数据。
6.  点击 ‘Send’。

同样，确保你收到一个表明部分更新成功的响应。

示例：
使用与之前相同的端点，如果你只想更新个人资料的电子邮件而保持其他属性不变，你的 JSON 数据可能是：

```json
{
    "email": "updated.email@example.com"
}
```

### 4. 测试 DELETE 方法：

DELETE 方法很直接。它用于删除资源：

1.  在 Postman 中，输入你想要删除的资源的 API URL。
2.  选择 DELETE 请求方法。
3.  点击 ‘Send’。

发送 DELETE 请求后，你应该会收到一个表明资源已被删除的响应。

注意：在测试期间请谨慎使用 DELETE 方法，因为它会删除数据。如果你想继续测试其他功能，可能需要重新填充数据库或重新创建资源。

## 结论：

至此，你应该已经成功测试了你的 APIView 的 PUT、PATCH 和 DELETE 方法。始终建议进行严格的测试，特别是针对不同的场景和边缘情况，以确保你的 API 的健壮性。

### 作业：

- 尝试使用不同的数据输入并观察 API 响应。
- 看看如果你尝试发送一个没有请求体数据或数据类型不正确的 PATCH 或 PUT 请求，你的 API 会如何响应。

# 第 9 节：Viewset 简介

### 什么是 Viewset？

### 资产、资源和材料：

- Django REST Framework (DRF)：你需要安装 Django REST Framework。它是一个强大且灵活的工具包，用于在 Django 中构建 Web API。（安装命令：`pip install djangorestframework`）
- Django 项目：确保你已经设置了一个集成了 DRF 的 Django 项目。如果你遵循了前面的章节，你应该已经准备好了。
- Python 环境：你应该在 Python 虚拟环境中工作，以避免潜在的包冲突。（使用 `virtualenv` 或内置的 `venv` 模块）
- 文本编辑器：我们将使用 Atom，如前所述，但任何你选择的文本编辑器或 IDE 都可以。
- 文档：[DRF 官方文档](https://www.django-rest-framework.org/) 是一个宝贵的资源。将其添加为书签以便轻松访问。

### 简介

在 Web API 领域，尤其是在使用 Django REST Framework (DRF) 时，你可能经常听到“视图”和“视图集”。虽然你熟悉 Django 中视图的概念，但“视图集”这个术语对你来说可能是新的。本章致力于揭开视图集背后的神秘面纱，使其成为你 DRF 工具包中一个易于理解且不可或缺的工具。

### 什么是 Viewset？

简单来说，Viewset 是 DRF 提供的一个高级抽象，它将处理 HTTP 方法（如 GET、POST、PUT、DELETE）的逻辑组合到一个类中。它是 Django 视图之上的一层，专门设计用于处理序列化数据和查询集。

可以将 Viewset 视为你的模型（数据库）和序列化器（表示）之间的桥梁。它们决定你的数据如何被处理，以及什么作为响应发送或作为输入接收。

### Viewset 的主要特性：

1.  CRUD 操作：Viewset 本质上理解 CRUD 操作（创建、读取、更新、删除）的概念。当你定义一个视图集时，你本质上是在说：“我想创建一组视图来处理特定模型和序列化器的这些 CRUD 操作。”
2.  更少的代码行数：使用视图集的一个显著优势是减少了代码量。使用传统视图，你可能需要为列表视图和详情视图分别编写类或方法。视图集将这些行为封装在一个类中。
3.  路由器：视图集的另一个强大之处在于它们与 DRF 的路由器无缝集成。这允许为你的 API 端点自动创建 URL 模式。
4.  权限和认证：就像使用 APIView 一样，你可以轻松地将认证和权限与视图集集成，确保你的数据安全且仅对授权用户可访问。
5.  可定制性：虽然视图集提供了许多开箱即用的功能，但它们也是高度可定制的。你可以重写方法、添加额外的操作或与第三方包集成。

### APIView 和 Viewset 的区别

为了进一步阐明视图集的概念，让我们将它们与 APIView 区分开来：

- APIView：这是 DRF 的一个抽象，它将 HTTP 方法作为单独的类方法来处理。例如，对于 `GET` 请求，你会实现 `get()` 方法。它提供了对响应和请求逻辑的更多控制。
- Viewset：这比 APIView 更高级，它抽象了基于模型查询集处理 CRUD 操作的逻辑。它最适合标准的数据库操作。对于任何非标准行为，你可能仍然需要回到 APIView 或进一步自定义视图集。

## 结论

本质上，Django REST Framework 中的视图集提供了一种强大且简化的方式来为你的序列化数据创建 API 端点。它们减少了样板代码，并提供了与路由器的轻松集成，使得构建健壮的 Web API 的过程更加顺畅和直观。

在接下来的章节中，我们将深入探讨如何有效地创建和利用视图集，确保你全面理解这个关键的 DRF 组件。

## 创建一个简单的 Viewset

### 资产、资源和材料：

- Python：这是我们后端将使用的主要语言。（可以从 [python.org](https://www.python.org/downloads/) 下载并安装）。
- Django：一个高级的 Python Web 框架，我们将用它来创建我们的应用程序。（通过 pip 安装：`pip install django`）。
- Django REST Framework (DRF)：Django 的一个扩展，它使构建 RESTful API 更简单。（通过 pip 安装：`pip install djangorestframework`）。
- 代码编辑器（例如 Atom）：你将在这里编写和编辑代码。（[atom.io](https://atom.io/)）。
- 终端或命令行：用于运行我们的服务器、应用迁移等。
- Web 浏览器：用于查看和测试我们的 API。（你可以使用 Chrome、Firefox、Safari 或任何其他你选择的浏览器）。

## 简介：

Viewset 是 Django REST Framework 中的一个关键组件。它们允许开发者快速构建 CRUD 操作，而无需为每个特定的 HTTP 动词定义方法。Viewset 将处理 HTTP 方法（GET、POST、PUT、DELETE）的逻辑组合到一个特定模型的类中。

在本章中，我们将为一个假设的 `Book` 模型构建一个简单的 Viewset，该模型将是书籍的基本表示，包含 title、author 和 publication_date 等字段。让我们开始吧。

### 步骤 1：定义 Book 模型

在创建 Viewset 之前，我们需要一个模型。

在你的 models.py 中：

```python
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    publication_date = models.DateField()

    def __str__(self):
        return self.title
```

### 步骤 2：创建序列化器

序列化器允许将复杂的数据类型转换为可以轻松渲染为 JSON 的 Python 数据类型。

在你的 serializers.py 中：

```python
from rest_framework import serializers
from .models import Book

class BookSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = ['id', 'title', 'author', 'publication_date']
```

### 步骤 3：创建一个简单的 Viewset

模型和序列化器准备好后，是时候创建 Viewset 了。

在你的 views.py 中：

```python
from rest_framework import viewsets
from .models import Book
from .serializers import BookSerializer

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
```

这里，`viewsets.ModelViewSet` 提供了默认的 `list()`、`create()`、`retrieve()`、`update()` 和 `destroy()` 操作。`queryset` 和 `serializer_class` 属性告诉

### 第四步：将视图集与URL关联

在你的 `urls.py` 文件中：

```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import BookViewSet

router = DefaultRouter()
router.register(r'books', BookViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

DRF 路由器会自动为我们的视图集操作生成 URL。

## 结论

通过以上步骤，你已经为 `Book` 模型设置了一个基本的视图集。当你运行 Django 服务器并导航到相关端点（例如 `/books/`）时，你现在可以通过 API 对 `Book` 模型执行 CRUD 操作。

由于这是一个入门介绍，我们使用了最简单的视图集版本。随着你学习的深入，你将发现 Django REST Framework 中视图集和其他组件的更多功能和细微差别。

## 添加 URL 路由器

### 本章的资产、资源和材料

- Django：一个用于构建 Web 应用程序的 Python Web 框架。（如何获取：你可以使用 pip 安装 Django，运行 `pip install django`）
- Django REST Framework (DRF)：Django 的一个扩展，用于构建 Web API。（如何获取：通过 pip 安装，使用 `pip install djangorestframework`）
- Django 的默认路由器：Django REST Framework 的一部分，有助于自动确定视图集的 URL 模式。（随 DRF 安装包一起提供）

### 引言

在 Django REST Framework (DRF) 中使用视图集时，URL 路由器在自动生成 URL 模式方面起着至关重要的作用。与你可能使用 APIViews 手动定义每个 URL 模式不同，视图集与 DRF 的路由器相结合有助于简化此过程，使其既高效又优雅。

## 理解 DRF 的路由器

Django REST Framework 包含一组路由器，可帮助自动为你的视图集创建适当的 URL。最常用的路由器是 `DefaultRouter`。`DefaultRouter` 类将自动为你创建 URL 模式，并提供一个简单的接口来注册你的视图集。

## 添加 URL 路由器的分步指南

### 1. 导入必要的库

在你的 `urls.py` 文件顶部（定义 URL 模式的地方），你需要导入必要的模块：

```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
```

### 2. 创建 DefaultRouter 的实例

接下来，你将创建一个 `DefaultRouter` 的实例：

```python
router = DefaultRouter()
```

### 3. 将视图集注册到路由器

在这个例子中，假设你有一个名为 `UserProfileViewSet` 的视图集，位于一个名为 `views.py` 的文件中。首先，在你的 `urls.py` 顶部导入该视图集：

```python
from .views import UserProfileViewSet
```

现在，你可以将此视图集注册到路由器：

```python
router.register(r'profiles', UserProfileViewSet)
```

第一个参数 `r'profiles'` 指定了我们 URL 模式的前缀。这意味着我们的用户资料 API 端点将类似于 `http://yourdomain.com/profiles/`。

### 4. 将路由器 URL 添加到 Django 的 URL 模式中

最后，你需要将路由器的 URL 模式包含到 Django 的 URL 模式中：

```python
urlpatterns = [
    path('', include(router.urls)),
]
```

这告诉 Django 包含与路由器关联的所有 URL。

### 最终结果

将所有内容整合在一起，你的 `urls.py` 应该看起来像这样：

```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserProfileViewSet

router = DefaultRouter()
router.register(r'profiles', UserProfileViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

通过此设置，你的 `UserProfileViewSet` 将拥有一组标准化的 URL。例如：

- 列出所有资料：`GET /profiles/`
- 检索特定资料：`GET /profiles/{id}/`
- 更新资料：`PUT /profiles/{id}/`
- 部分更新资料：`PATCH /profiles/{id}/`
- 删除资料：`DELETE /profiles/{id}/`

## 结论

在本章中，你学习了 DRF 路由器（特别是 `DefaultRouter`）的重要性，以及它如何极大地简化视图集的 URL 管理。通过这种设置，它不仅使代码更整洁，还确保了 Web API 中 URL 模式的一致性，使其更具可预测性和用户友好性。

## 测试我们的视图集

### 资产、资源和材料

- Python：我们将用于后端脚本的主要语言。（可在 [python.org](https://www.python.org/) 获取）
- Django：用于构建 Web 应用程序的高级 Web 框架。（使用 pip 安装：`pip install django==2.2`）
- Django REST Framework (DRF)：一个强大且灵活的 Web API 构建工具包。（使用 pip 安装：`pip install djangorestframework==3.9`）
- Postman：一个流行的 API 测试工具。（可在 [postman.com](https://www.postman.com/downloads/) 获取）
- unittest：Python 内置的单元测试库。（随 Python 一起提供）

### 引言

在前面的章节中，我们已经介绍了视图集的创建和设置。现在，我们必须确保我们编写的函数不仅工作正常，而且提供正确的响应。测试对于在添加或修改功能时保持代码的可靠性和稳定性至关重要。在本章中，我们将仔细研究如何在 Django 中使用手动和自动方法测试你的视图集。

## 使用 Postman 进行手动测试

在深入自动化测试之前，让我们先使用 Postman 手动测试我们的视图集端点。

### 步骤

1. 安装和设置 Postman：从其官方网站下载并安装 Postman。安装后，打开它。
2. 设置请求：根据你要测试的视图集功能，选择请求类型（GET、POST、PUT、PATCH、DELETE）。输入你的 API 端点 URL。
3. 添加数据（如果需要）：对于 POST、PUT 和 PATCH 请求，转到“Body”选项卡，并以 JSON 格式输入任何必要的数据。
4. 发送请求并检查响应：点击“Send”按钮。请求处理后，你将在下方看到服务器的响应。确保响应数据和状态码符合你的预期。

## 使用 `unittest` 进行自动化测试

虽然手动测试对于初步检查很好，但我们不能依赖它来处理大型应用程序。它耗时，并且总是有可能错过测试某个特定场景。自动化测试一旦编写，就可以运行任意多次，确保每次测试的一致性。

### 设置你的测试环境

1. 在你的 Django 项目中，在你的应用文件夹内创建一个名为 `tests.py` 的文件。
2. 在 `tests.py` 中，导入必要的库：

```python
from django.test import TestCase
from rest_framework.test import APIClient
from .models import YourModel
```

### 编写你的第一个测试

让我们测试当发出 GET 请求时，我们的视图集是否返回一个项目列表：

```python
class ViewsetTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        YourModel.objects.create(name="TestItem1", description="TestDescription1")
        YourModel.objects.create(name="TestItem2", description="TestDescription2")
    def test_get_items(self):
        response = self.client.get('/your_endpoint_url/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 2)
```

1. 设置：在这里，我们初始化 APIClient（它将模拟我们的 API 调用）并使用我们的模型设置一些示例数据。
2. 测试：我们向端点发送一个 GET 请求。然后我们检查响应状态码是否为 200（OK），以及我们是否在响应中收到了两个项目。

### 运行你的测试

要运行你的测试，在终端中导航到项目的根目录并运行：

```bash
python manage.py test your_app_name
```

确保所有测试通过。如果任何测试失败，它们会提供错误信息，你可以利用这些信息来调试和修复代码的相应部分。

## 结论

测试确保你的视图集（Viewsets），进而确保你的API，能够按预期运行。手动测试提供了一种快速检查流程的方式，而自动化测试则确保了一致的可靠性。随着你为API添加更多功能或进行更改，请确保更新并运行你的测试，从而为用户保证稳定性和可靠性。

## 添加创建、检索、更新、部分更新和销毁功能

资产、资源和材料：

- Django：一个用于构建Web应用程序的高级Web框架。你可以通过运行 `pip install django==2.2` 来下载和安装Django。
- Django REST framework：一个用于构建Web API的灵活工具包。使用 `pip install djangorestframework==3.9` 进行安装。
- Postman：一个流行的API测试工具。你可以从 [Postman 官方网站](https://www.postman.com/downloads/) 下载它。
- 文本编辑器（Atom）：从 [这里](https://atom.io/) 下载并安装Atom。

#### 介绍：

视图集（Viewsets）是Django REST框架中的一个高级组件，它将处理HTTP方法（GET、POST、PUT等）的逻辑组合到一个类中。在本章中，我们将为我们的视图集实现核心功能（创建、检索、更新、部分更新和销毁）。这些功能将使我们的API能够处理不同类型的请求，包括创建、检索、更新和删除对象。

#### 1. 视图集函数的基础知识：

在深入实现之前，理解每个函数的目的至关重要：

- create：处理HTTP POST请求，用于创建一个新对象。
- retrieve：处理针对单个对象的HTTP GET请求。通过ID返回一个特定对象。
- update：处理HTTP PUT请求。完全更新一个对象。
- partial_update：处理HTTP PATCH请求。更新对象的特定字段。
- destroy：处理HTTP DELETE请求。删除一个对象。

#### 2. 设置视图集：

在你的 `views.py` 文件中，确保你有以下导入：

```python
from rest_framework import viewsets
from .models import YourModel
from .serializers import YourModelSerializer
```

将 `YourModel` 替换为你的模型名称，将 `YourModelSerializer` 替换为该模型的序列化器。

#### 3. 实现函数：

a) 创建函数：

```python
def create(self, request):
    serializer = YourModelSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data,
                        status=status.HTTP_201_CREATED)
    return Response(serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST)
```

此函数将处理POST请求，使用我们的序列化器验证数据，并将其保存到数据库。

b) 检索函数：

```python
def retrieve(self, request, pk=None):
    queryset = YourModel.objects.all()
    item = get_object_or_404(queryset, pk=pk)
    serializer = YourModelSerializer(item)
    return Response(serializer.data)
```

这里，我们使用主键（ID）获取一个特定项目，并返回其序列化数据。

c) 更新函数：

```python
def update(self, request, pk=None):
    queryset = YourModel.objects.all()
    item = get_object_or_404(queryset, pk=pk)
    serializer = YourModelSerializer(item,
                                     data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST)
```

如果提供的数据有效，此函数将完全更新一个对象。

d) 部分更新函数：

```python
def partial_update(self, request, pk=None):
    queryset = YourModel.objects.all()
    item = get_object_or_404(queryset, pk=pk)
    serializer = YourModelSerializer(item,
                                     data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST)
```

此函数将更新特定字段，前提是请求数据仅包含需要更新的字段。

e) 销毁函数：

```python
def destroy(self, request, pk=None):
    queryset = YourModel.objects.all()
    item = get_object_or_404(queryset, pk=pk)
    item.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)
```

它将删除由其主键标识的特定项目。

#### 4. 使用Postman测试函数：

一旦你实现了这些函数，你就可以使用Postman向你的API发送各种HTTP请求。确保你已正确设置URL以测试上述每个功能。

## 结论：

有了这些功能，我们的视图集现在能够处理各种HTTP请求，从而增强了我们REST API的功能。这些核心功能是大多数Web应用程序中CRUD操作的基础，掌握它们将使你在后端开发中领先一步。

## 测试视图集

本章的资产、资源和材料：

- Python：本书使用的主要语言。如果未安装，你可以从 [python.org](https://www.python.org/) 下载。
- Django：用于构建我们应用程序的Web框架。如果未安装，你可以使用pip下载：`pip install django`。
- Django REST Framework (DRF)：为我们提供创建RESTful服务的工具。使用pip下载：`pip install djangorestframework`。
- Postman：用于测试API端点的工具。你可以从 [Postman 官方网站](https://www.postman.com/downloads/) 下载它。
- ViewSet 和 Router：已在前面的章节中介绍。

### 介绍

随着你深入研究Django REST框架（DRF），你会遇到一个名为“视图集（Viewsets）”的重要组件。它们为你的视图提供了高级抽象，使得在数据库中创建、读取、更新和删除对象变得更加容易。但这种抽象也带来了责任，即确保你的端点按预期运行。测试作为一种实践，确保你暴露给外部世界的端点是可靠且一致的。在本章中，我们将探讨如何测试你已设置好的视图集。

#### 1. 理解要测试什么

在直接开始编写测试用例之前，让我们先确定我们视图集的核心功能：

- 列表（List）：它能否检索并列出所有对象？
- 检索（Retrieve）：它能否通过ID获取特定对象？
- 创建（Create）：它能否成功创建一个新对象？
- 更新（Update）：它能否修改一个现有对象？
- 部分更新（Partial Update）：它能否修改现有对象的部分内容？
- 删除（Delete）：它能否删除一个对象？

这些功能中的每一个都代表一个潜在的测试用例。

#### 2. 设置测试环境

在开始测试之前，确保你有一个单独的测试数据库。Django会为你处理这个问题，因此每当你运行测试时，它都会创建一个单独的数据库，以确保你的主数据库保持不变。

#### 3. 编写测试用例

这是一个测试我们视图集 `list` 功能的简化示例：

```python
from rest_framework.test import APITestCase
from rest_framework import status
from .models import YourModel

class ViewsetTestCase(APITestCase):
    def setUp(self):
        # 这会创建一个示例对象，我们可以用它来
        # 测试我们的列表功能。
        YourModel.objects.create(name="Sample",
                                description="Sample description")

    def test_list_objects(self):
        response = self.client.get('/path-to-your-viewset/')
        self.assertEqual(response.status_code,
                         status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
```

这里，我们向视图集发送一个 `GET` 请求，然后断言两件事：

1. 我们响应的状态码应该是200（OK）。
2. 我们响应数据的长度应该是1，因为我们在 `setUp` 方法中创建了一个对象。

#### 4. 测试其他功能

遵循类似的方法来测试 `retrieve`、`create`、`update`、`partial update` 和 `delete`。记住根据你正在测试的操作调整你的HTTP方法（GET、POST、PUT、PATCH、DELETE）。

### 5. 运行测试

在终端中导航到你的 Django 项目根目录。运行以下命令执行测试：

```bash
python manage.py test
```

Django 会搜索所有以 `test` 开头的文件并执行它们。确保所有测试都通过。如果有任何测试失败，控制台会提供详细的错误信息，帮助你识别并修复问题。

测试你的视图集可以确保你的 API 行为符合预期。这是构建健壮可靠应用程序的关键步骤。请养成在引入新功能或修改现有功能时编写测试的习惯。这个习惯将为你未来节省无数调试时间，并让你对应用程序在所有场景下都能正常工作充满信心。

# 第10节：创建用户资料 API

### 规划我们的用户资料 API

资产、资源和材料：

- 用户故事或用例：（这些是详细描述用户如何与功能或系统交互的假设场景。可以使用 JIRA、Trello 甚至纸笔等工具创建。目的：帮助理解用户需求。）
- UML 图工具：（统一建模语言工具，如 Lucidchart、Draw.io，甚至纸笔。目的：创建实体关系图以规划我们的数据结构。）
- API 设计工具：（如 Swagger 或 Postman 等工具。目的：可视化 API 端点和方法。）

#### 介绍：

规划 API 是其开发生命周期中的关键步骤。适当的规划可以确保开发流程顺畅，并降低遗漏关键功能或引入错误的可能性。对于我们的用户资料 API，我们的主要目标是提供处理用户资料的功能——包括创建、读取、更新和删除它们。

#### 1. 用户故事：

在开始编码之前，了解最终用户对我们 API 的期望至关重要。用户故事是捕获这些需求的有效方式。以下是与我们的用户资料 API 相关的一些用户故事：

- 作为用户，我希望能用我的姓名、电子邮件和个人资料图片创建一个资料。
- 作为用户，我希望能查看我的资料。
- 作为用户，我希望能更新我的资料信息。
- 作为用户，我希望能删除我的资料。
- 作为用户，我希望能通过姓名或电子邮件地址搜索其他用户。

#### 2. 定义数据模型：

有了用户故事，我们需要为我们的用户资料 API 设计数据模型。根据需求，我们的 Profile 模型可能如下所示：

- `User_ID`：用户的唯一标识符。
- `Name`：用户的全名。
- `Email`：用户的电子邮件地址。
- `Profile_Picture`：个人资料图片的 URL。
- `Created_At`：资料创建的日期和时间。
- `Updated_At`：资料最后更新的日期和时间。

#### 3. 设计 API 端点：

利用我们的用户故事和数据模型，我们现在可以规划用户资料 API 的端点：

- 创建资料：
  - 端点：`POST /profiles/`
  - 载荷：`{ "name": "John Doe", "email": "john@example.com", "profile_picture": "url_to_picture" }`
- 获取资料：
  - 端点：`GET /profiles/<User_ID>/`
- 更新资料：
  - 端点：`PUT /profiles/<User_ID>/`
  - 载荷：`{ "name": "John A. Doe", "email": "john.a@example.com", "profile_picture": "new_url_to_picture" }`
- 删除资料：
  - 端点：`DELETE /profiles/<User_ID>/`
- 搜索资料：
  - 端点：`GET /profiles/?search=<query>`
  - 这将按姓名或电子邮件搜索资料。

#### 4. API 响应和状态码：

为了确保我们的 API 与前端或其他服务有效通信，我们需要使用适当的状态码和响应消息。例如：

- `200 OK`：成功的 GET 请求。
- `201 Created`：资料创建成功。
- `204 No Content`：成功的 DELETE 请求。
- `400 Bad Request`：如果请求载荷不正确。
- `404 Not Found`：如果找不到具有给定 ID 的资料。

#### 5. 安全考虑：

在规划我们的 API 时，牢记安全至关重要：

- 认证：确保只有经过认证的用户才能创建、更新或删除资料。
- 授权：用户应该只能更新或删除自己的资料，而不是别人的。
- 数据验证：验证所有传入的数据以防止恶意输入。
- 速率限制：实施速率限制以防止滥用。

## 结论：

通过提前规划我们的用户资料 API，我们为其开发奠定了清晰的路线图。下一步将涉及实际开发，我们将根据此计划实现数据模型、API 端点和安全措施。请记住，规划良好的 API 通常意味着实现良好的 API！

### 创建用户资料序列化器

资产和资源：

- Django REST Framework (DRF)：一个强大且灵活的 Web API 构建工具包。你可以使用 pip 安装它：`pip install djangorestframework`。
- Django：确保你已安装 Django。如果没有，请使用 pip：`pip install Django`。
- Python：Django 和 DRF 编写所用的核心语言。
- 用户模型：我们将在第18章中创建它。用户模型是数据库中存储用户信息的结构。

### 介绍

Django REST Framework (DRF) 中的序列化器允许将复杂数据类型（如 Django 模型）转换为 Python 数据类型。在 API 的上下文中，这些 Python 数据类型随后可以轻松渲染为 JSON 或 XML 供客户端使用。对于我们的项目，序列化器的主要用途是将我们的用户资料实例转换为 JSON 格式。

在本章中，我们将专注于为我们的用户资料创建一个序列化器，它将在客户端和服务器之间的数据交换中扮演关键角色。

#### 步骤 1：创建序列化器文件

1. 在你的 Django 应用文件夹内（models.py、views.py 等所在的位置），创建一个名为 `serializers.py` 的新文件。

#### 步骤 2：设置序列化器

在 `serializers.py` 中：

```python
from rest_framework import serializers
from .models import UserProfile

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ('id', 'email', 'name', 'password')
        extra_kwargs = {
            'password': {
                'write_only': True,
                'style': {'input_type': 'password'}
            }
        }
```

让我们分解一下我们做了什么：

- 导入必要模块：我们从 DRF 导入了 `serializers`，并导入了我们的 `UserProfile` 模型。
- UserProfileSerializer：这是我们新的序列化器类，它继承自 `serializers.ModelSerializer`。
    - 在嵌套的 `Meta` 类中：
        - `model = UserProfile`：这将我们的序列化器与 `UserProfile` 模型关联起来。
        - `fields`：指定我们希望在序列化表示中包含模型的哪些字段。
        - `extra_kwargs`：为我们的序列化器提供额外选项。在这种情况下，我们指定密码字段应该是只写的（出于安全考虑），并且其输入类型应该是密码类型（因此在表单中会被隐藏）。

#### 步骤 3：序列化器方法（可选但推荐）

为了增强我们 API 的功能和安全性，我们可以向序列化器添加自定义方法。例如，让我们确保当使用序列化器创建用户资料时，密码以哈希格式保存：

```python
def create(self, validated_data):
    """创建并返回一个新用户，密码已加密"""
    user = UserProfile.objects.create_user(
        email=validated_data['email'],
        name=validated_data['name'],
        password=validated_data['password']
    )
    return user
```

通过这个方法，当使用此序列化器创建新用户时，密码会被安全地存储。

## 结论

序列化器是 Django REST Framework 应用程序的重要组成部分，使我们能够轻松地在 Python 对象和 JSON 表示之间验证和转换数据。有了我们的 `UserProfileSerializer`，我们现在准备好在后续章节的 API 视图和视图集中处理用户资料数据。请记住，序列化器不仅确保我们的数据格式正确，还在数据交换的安全性方面发挥着关键作用。

### 创建用户资料视图集

资产、资源和材料：

- Django REST Framework (DRF)：要安装，请使用 pip 包管理器运行 `pip install djangorestframework`。我们将使用 DRF 来创建我们的视图集。
- Django 项目和应用：应该已经在前面的章节中设置好了。

## 引言：

在本章中，我们将深入探讨如何为我们的用户资料创建一个 ViewSet。ViewSet 是 Django REST Framework 提供的便利工具，它允许开发者快速构建针对模型的 CRUD（创建、读取、更新、删除）操作。在本章结束时，你将拥有一个针对你在第 41 章创建的 `UserProfile` 模型的可运行 ViewSet。

## 1. Django REST Framework 中的 ViewSets 是什么？

在 Django REST Framework 中，ViewSets 是一种高级抽象，它将处理 HTTP 方法（如 GET、POST、PUT 等）的逻辑组合到类中。无需为每个可能的操作（列出所有资料、查看单个资料、更新资料等）编写单独的视图，一个 ViewSet 可以通过开箱即用的方式提供这些操作，从而替代多个视图。

## 2. 设置 UserProfileViewSet

导航到你的应用 `views.py` 文件，让我们开始设置 ViewSet。

```python
from rest_framework import viewsets
from .models import UserProfile
from .serializers import UserProfileSerializer

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
```

以下是详细说明：

- 我们导入必要的模块：来自 DRF 的 `viewsets`、我们的 `UserProfile` 模型，以及我们在上一章创建的 `UserProfileSerializer`。
- 然后我们创建一个名为 `UserProfileViewSet` 的新类，它继承自 `viewsets.ModelViewSet`。
- 在这个类中，我们定义了两个类级别的变量：
    - `queryset`：这是我们定义要提供哪些记录的地方。在本例中，我们希望提供所有用户资料。
    - `serializer_class`：这告诉 ViewSet 使用我们的 `UserProfileSerializer` 将数据库记录序列化为可以渲染成响应的格式。

## 3. 将 UserProfileViewSet 注册到我们的 URL 配置中

为了让我们的 ViewSet 可以通过网络访问，我们需要将其注册到 URL 配置中。首先，我们需要设置一个路由器。打开你的 `urls.py`：

```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'profiles', views.UserProfileViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

以下是我们所做的：

- 我们导入必要的模块。
- 创建一个 `DefaultRouter` 实例。
- 将我们的 `UserProfileViewSet` 注册到这个路由器。第一个参数 `r'profiles'` 决定了我们 API 端点的 URL 前缀。
- 最后，我们将路由器的 URL 包含在我们应用的 URL 模式中。

## 4. 测试 UserProfileViewSet

运行你的开发服务器：

```
python manage.py runserver
```

在你的网络浏览器中访问 `http://127.0.0.1:8000/profiles/`。你应该会看到所有用户资料的列表（如果存在的话）。你也可以使用 Postman 或浏览器等工具来测试 POST、PUT 和 DELETE 请求。

## 结论：

恭喜！你现在已经为你的 `UserProfile` 模型创建了一个功能完整的 ViewSet。这个 ViewSet 提供了列出所有资料、查看单个资料、更新、创建和删除资料的端点。

## 后续步骤：

在下一章中，我们将把我们的 `UserProfileViewSet` 注册到 URL 路由器，使我们的 API 可以通过网络访问。我们还将介绍如何处理不同的 HTTP 方法来与我们的 API 交互，例如使用 GET 检索数据，使用 POST 添加新资料。

## 将个人资料 Viewset 注册到 URL 路由器

资产、资源和材料：

- Django REST Framework：提供构建 Web API 的工具。([获取地址](https://www.django-rest-framework.org/))
- Python：本书使用的核心编程语言。([获取地址](https://www.python.org/downloads/))
- Django：为有截止日期的完美主义者准备的 Web 框架。([获取地址](https://www.djangoproject.com/download/))
- 代码编辑器（例如 Atom）：用于编写和查看代码。([获取地址](https://atom.io/))
- 你之前的代码：来自前面的章节，以确保连续性。

引言：

在创建了我们的个人资料 Viewset 之后，下一步是使其可以通过 URL 访问。在 Django 中，这通常通过使用 URL 路由器来实现。Django REST Framework (DRF) 包含一个专门为 Viewsets 设计的 URL 路由器，它会为典型操作（例如创建、读取、更新、删除）自动生成适当的 URL。

在本章中，我们将引导你完成将你的个人资料 Viewset 注册到此路由器的过程，从而使你的 API 可以从网络访问。

### 1. 导入必要的库和模块：

首先，我们需要导入 URL 配置所需的模块。如果你还记得，`urls.py` 文件是 Django 项目的主要路由文件。

```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
```

这里：

- `path` 和 `include` 是标准的 Django URL 工具。
- `DefaultRouter` 是来自 DRF 的一个类，用于为我们的 Viewsets 自动创建路由。
- `views` 是包含个人资料 Viewset 的模块。

### 2. 创建路由器对象：

接下来，我们需要实例化我们的路由器。这个路由器将负责为我们的 Viewset 生成 URL。

```python
router = DefaultRouter()
```

### 3. 将 Viewset 注册到路由器：

路由器创建后，我们现在可以注册我们的个人资料 Viewset。

```python
router.register(r'profile', views.ProfileViewSet)
```

这里，`r'profile'` 指定此资源的基 URL 将是 “profile”。例如，列出所有资料的 URL 将是 `http://yourapi.com/profile/`，而特定资料的 URL 可能是 `http://yourapi.com/profile/1/`。

### 4. 将路由器的 URL 包含在主 URL 配置中：

最后，确保生成的路由包含在项目的主 `urls.py` 中。

```python
urlpatterns = [
    path('', include(router.urls)),
]
```

此代码将路由器生成的所有路由包含在我们项目的 URL 配置中。

### 5. 测试配置：

现在我们的个人资料 Viewset 已注册到 URL 路由器，最好测试并确保一切按预期工作。

#### 1. 启动开发服务器：

```
python manage.py runserver
```

#### 2. 打开你的网络浏览器或像 Postman 这样的 API 客户端工具，并导航到 `http://localhost:8000/profile/`。

如果一切配置正确，你应该会看到资料列表，或者如果尚未创建任何资料，则至少会看到一个空白页面。

## 结论：

URL 路由对于使你的 API 端点可访问至关重要。Django REST Framework 的 `DefaultRouter` 通过为你的 Viewsets 自动生成路由来简化此过程。随着个人资料 Viewset 的注册，我们距离拥有一个功能完整的 Profiles API 又近了一步。

> 给读者的提示：
始终确保你的 URL 模式清晰且一致。这不仅有助于你的开发过程，还会使你的 API 对于将来可能与之交互的其他开发者更加直观。

## 测试创建个人资料

本章所需的资产、资源和材料：

- Python（版本 3.x）：确保你已安装 Python。如果没有，请从 [Python 官方网站](https://www.python.org/downloads/) 下载。Python 是我们用于开发 REST API 的主要语言。
- Django 和 Django REST Framework：你应该在前面的章节中已经安装了这些。如果没有，请回顾第 12 章和第 13 章的安装细节。这些工具帮助我们构建 Web 应用程序和 API 的结构。
- Postman：一个流行的 API 测试工具。从 [官方网站](https://www.postman.com/downloads/) 下载。我们将使用此工具向我们的 API 发送 HTTP 请求并分析响应。
- 终端或命令提示符：用于运行服务器端命令。

目标：在本章结束时，你应该能够测试你的 REST API 的个人资料创建功能，确保其行为符合预期。

### 1. 引言

在应用程序中部署任何功能之前，必须对其进行严格测试，以确保其按预期运行。在本章中，我们将使用 Postman 来测试我们 REST API 的个人资料创建功能。Postman 提供了一个用户友好的界面来向我们的 API 发送请求，使测试各种场景变得更加容易。

### 2. 准备 API 进行测试

首先，确保你的 Django 开发服务器正在运行。如果没有，请在终端中导航到你的项目目录并运行：

```bash
python manage.py runserver
```

这将启动你的服务器，通常可通过 `http://127.0.0.1:8000/` 访问。

### 3. 设置 Postman 进行测试

1.  打开 Postman：安装后，启动 Postman。
2.  创建新请求：点击“+”标签页。这将打开一个新标签页用于创建请求。
3.  将请求类型设置为 POST：由于我们正在测试个人资料创建，我们将发送一个 POST 请求。从 URL 栏左侧的下拉菜单中，选择“POST”。
4.  输入 API URL：在 URL 栏中，输入你用于创建个人资料的 API 端点，例如 `http://127.0.0.1:8000/profiles/`。
5.  设置请求头：在许多情况下，你的 API 可能需要特定的请求头（如 Content-Type）。确保其设置为“application/json”。
6.  输入 JSON 数据：在“Body”标签页中，选择“raw”，然后以 JSON 格式输入个人资料数据。以下是一个示例：

```json
{
    "name": "John Doe",
    "email": "johndoe@example.com",
    "password": "strongpassword"
}
```

### 4. 发送请求并分析响应

在 Postman 中设置好请求后：

1.  发送请求：点击“Send”按钮。
2.  分析响应：发送请求后，Postman 将在下方显示 API 的响应。你应该检查以下内容：
    - 状态码：确保你收到 `201 Created` 状态，表示个人资料创建成功。
    - 响应体：检查 API 是否返回了预期的数据。例如，新个人资料的 ID、姓名和电子邮件（出于安全原因，不包含密码）。
    - 错误消息：如果出现问题，API 应返回一条错误消息，详细说明出了什么问题。这可能是验证错误或服务器端错误。

### 5. 测试边界情况

仅测试成功场景是不够的；你还必须确保你的 API 能够优雅地处理失败情况：

1.  缺失数据：尝试发送一个缺少必填字段的请求，例如没有电子邮件。API 应返回 `400 Bad Request` 状态和清晰的错误消息。
2.  无效数据：输入无效数据，例如格式不正确的电子邮件。同样，API 应返回 `400 Bad Request` 状态和适当的错误消息。
3.  重复数据：尝试使用数据库中已存在的电子邮件创建个人资料。API 应防止重复条目并发送错误消息。

## 结论

测试是软件开发中的关键阶段。通过确保个人资料创建功能按预期工作并能优雅地处理失败，我们离拥有一个健壮可靠的 API 又近了一步。在接下来的章节中，我们将把测试扩展到其他功能，并确保我们的 API 已准备好部署。
记住，“这不是一个 bug——这是一个未记录的功能！”你测试得越多，最终产品中的“未记录的功能”就越少。

## 创建权限类

资产、资源和材料：

- Django：（通过 pip 安装。Django 是我们用于构建后端的核心框架。）
- Django REST Framework (DRF)：（通过 pip 安装。DRF 扩展了 Django 处理 API 的能力。）
- 文本编辑器（例如 Atom）：（可从 Atom 网站免费下载。我们将使用它来编写和查看代码。）
- 终端或命令行界面：（内置于你的操作系统中。用于运行命令。）
- Django 的权限和认证模块：（随 DRF 安装提供。用于创建和管理权限。）

引言：

在使用 REST API 时，实施必要的权限以保护敏感数据并确保前端和后端之间的正确数据流至关重要。在本章中，我们将重点使用 Django REST Framework 为我们的 Profiles API 创建一个自定义权限类。

### 什么是权限类？

Django REST Framework 中的权限类决定是否应授予或拒绝请求的访问权限。DRF 提供了一组内置的权限类，如 `IsAuthenticated`、`IsAdminUser` 和 `IsAuthenticatedOrReadOnly`。虽然这些内置类很有用，但有时你需要更自定义的行为，这就是创建自己的权限类发挥作用的地方。

### 创建自定义权限类的步骤：

1.  创建新的权限文件：
    在你的 `profiles_api` 应用目录中，创建一个名为 `permissions.py` 的新文件。

    ```bash
    touch profiles_api/permissions.py
    ```

2.  设置权限类：
    在 `permissions.py` 中，首先导入必要的模块，然后定义你的自定义权限类。

    ```python
    from rest_framework import permissions

    class UpdateOwnProfile(permissions.BasePermission):
        """允许用户编辑自己的个人资料"""
        def has_object_permission(self, request, view, obj):
            """检查用户是否试图编辑自己的个人资料"""
            if request.method in permissions.SAFE_METHODS:
                return True
            return obj.id == request.user.id
    ```

    这里，我们创建了一个名为 `UpdateOwnProfile` 的权限，它检查请求是否是安全方法（GET、HEAD 或 OPTIONS）。如果是安全方法，则授予请求权限；否则，它通过比较对象的 id 与用户的 id 来检查用户试图修改的对象是否属于他们。

3.  实现权限：
    转到 `profiles_api` 应用目录中的 `views.py`。在这里，你需要将自定义权限类添加到你的 `ProfileViewSet` 中。
    首先，在顶部导入自定义权限：

    ```python
    from .permissions import UpdateOwnProfile
    ```

    然后，在你的 `ProfileViewSet` 中，将权限添加到 `permission_classes` 属性：

    ```python
    permission_classes = [UpdateOwnProfile, ]
    ```

    现在，每当向 `ProfileViewSet` 发出请求时，`UpdateOwnProfile` 权限类将用于检查是否应授予或拒绝该请求的访问权限。

测试权限：
权限类就位后，对其进行测试非常重要。可以使用 Postman 或你的浏览器等工具来完成。尝试更新你自己的个人资料以外的个人资料，并观察结果。如果权限类工作正常，对于此类请求，你应该会收到一个禁止响应。

结论：
权限类是在 Django REST Framework 中处理权限和访问控制的强大方式。通过理解并有效使用它们，你可以为你的数据和 API 用户创建一个安全的环境。我们刚刚创建的自定义权限类确保用户只能修改自己的数据，使我们的 Profiles API 保持安全和功能正常。

## 向 Viewset 添加认证和权限

资产和资源：

- Django REST Framework (DRF) 文档：[官方 DRF 文档](https://www.django-rest-framework.org/)
  - 深入了解认证、权限和视图集。
- Django 认证：[官方 Django 文档](https://docs.djangoproject.com/en/3.0/topics/auth/default/)
  - 了解内置的 Django 用户认证系统。
- Postman 或类似的 API 测试工具：[下载 Postman](https://www.postman.com/downloads/)
  - 用于发送请求和观察 API 行为。

引言：

Django REST Framework (DRF) 提供了开箱即用的工具，用于将认证和权限集成到你的 API 中。确保只有经过认证的用户才能发出某些请求，这对于我们的 Profiles API 的安全性和功能性至关重要。在本章中，我们将为我们的个人资料视图集设置认证和权限。

### 1. 设置认证：

为确保只有经过认证的用户才能访问某些视图或执行特定操作，我们首先需要设置认证。DRF 提供了多种认证类，包括基本认证、令牌认证和会话认证。

对于我们的 Profiles API，我们将使用令牌认证。以下是设置方法：

a. 安装必要的包：

```bash
pip install djangorestframework[authtoken]
```

b. 在 `settings.py` 的 `INSTALLED_APPS` 中添加 `rest_framework.authtoken`：

```python
INSTALLED_APPS = [
    ...
    'rest_framework.authtoken',
    ...
]
```

c. 运行迁移以将令牌模型添加到你的数据库：

```bash
python manage.py migrate
```

### 2. 设置权限：

DRF 中的权限决定了用户被允许执行哪种类型的请求（GET、POST、PUT 等）。让我们为我们的 profile 视图集定义必要的权限。

a. 在 `settings.py` 中定义默认权限类：

```python
REST_FRAMEWORK = {
    ...
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    ...
}
```

上述设置将确保只有经过身份验证的用户才能访问视图集。但是，如果我们想要更细粒度的权限，比如允许任何人查看 profiles，但只有经过身份验证的用户才能编辑它们呢？

b. 创建自定义权限：

在你的应用中，创建一个名为 `permissions.py` 的文件，并添加以下代码：

```python
from rest_framework import permissions

class UpdateOwnProfile(permissions.BasePermission):
    """允许用户编辑自己的 profile"""
    def has_object_permission(self, request, view, obj):
        """检查用户是否试图编辑自己的 profile"""
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.id == request.user.id
```

这里，我们允许任何被认为是“安全”的请求方法（即 GET 请求）。但对于其他方法，如 PUT 或 DELETE，我们确保被编辑的 profile 属于经过身份验证的用户。

c. 将自定义权限添加到我们的 Profile 视图集：

在你的 `views.py` 中，导入自定义权限并将其添加到 profile 视图集：

```python
from .permissions import UpdateOwnProfile

class UserProfileViewSet(viewsets.ModelViewSet):
    ...
    permission_classes = (UpdateOwnProfile, )
    ...
```

### 3. 测试我们的设置：

使用 Postman 或其他 API 工具，你现在可以向你的 Profiles API 发送请求，以查看权限和身份验证的实际效果。确保：

- 用户无法编辑其他用户的 profile。
- 没有令牌，用户无法编辑任何 profile。
- 任何用户，即使未登录，也可以查看 profiles。

## 结论：

身份验证和权限是任何健壮 API 的支柱，确保数据安全和适当的数据访问。通过利用 Django 和 Django REST Framework，你已经能够以最小的麻烦实现它们，专注于最重要的事情——提供无缝的用户体验。

## 测试新权限

所需的资产、资源和材料：

- Django 项目环境：到这个阶段，你应该已经设置好了你的 Django 项目环境，并创建了 profiles API。
- Django Rest Framework：确保你已经安装并配置了 Django Rest Framework (DRF)，因为我们将使用 DRF 的测试工具。
- Postman（或任何 API 测试工具）：你可以从[这里](https://www.postman.com/downloads/)下载 Postman。它是一个广泛使用的 API 测试工具。
- Python 单元测试框架：Django 自带 Python 的 `unittest` 框架，我们将使用它。
- 用户资料：在你的 Django 后端创建一个用户资料，用于测试权限。
- ModHeader 扩展：确保它已安装在你的浏览器中，用于设置令牌头。

### 测试权限的目的

在深入测试过程之前，让我们理解为什么我们要这样做。在 API 的上下文中，权限定义了谁可以做什么。这确保了数据的完整性和安全性。通过测试权限，我们旨在验证：

1. 未经过身份验证的请求被拒绝访问。
2. 来自未授权用户的经过身份验证的请求被拒绝访问。
3. 经过身份验证和授权的请求被授予访问权限。

### 1. 设置测试环境

首先，我们需要为测试设置环境。Django 自带一个内置的测试数据库，每次运行测试时都会创建，并在测试结束后销毁。

步骤：

- 在你的 profiles 应用目录中创建一个名为 `test_permissions.py` 的新文件。
- 导入必要的模块：

```python
from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model
from .models import Profile
from rest_framework.authtoken.models import Token
```

### 2. 创建测试用户

为了我们的测试，我们将创建两个用户：一个我们将分配权限，另一个则不会。

```python
class ProfilePermissionTests(APITestCase):
    def setUp(self):
        self.user1 = get_user_model().objects.create_user(
            username='user1',
            password='testpass123'
        )
        self.user2 = get_user_model().objects.create_user(
            username='user2',
            password='testpass123'
        )
        self.profile1 = Profile.objects.create(user=self.user1, ... 其他 profile 详情...)
        self.profile2 = Profile.objects.create(user=self.user2, ... 其他 profile 详情...)
        self.token1 = Token.objects.create(user=self.user1)
        self.token2 = Token.objects.create(user=self.user2)
```

### 3. 测试未经过身份验证的请求

我们需要确保未经过身份验证的请求无法访问我们的 profiles。

```python
def test_unauthenticated_permissions(self):
    response = self.client.get('/path_to_profiles_api/')
    self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
```

### 4. 测试未授权的请求

现在，测试 `user2` 试图修改 `user1` 的 profile 的场景。这应该被拒绝。

```python
def test_unauthorized_user_permissions(self):
    self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token2.key)
    response = self.client.put('/path_to_profiles_api/profile1_id/', {...data...})
    self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
```

### 5. 测试授权的请求

这里，`user1` 将尝试修改他们自己的 profile。这应该成功。

```python
def test_authorized_user_permissions(self):
    self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token1.key)
    response = self.client.put('/path_to_profiles_api/profile1_id/', {...data...})
    self.assertEqual(response.status_code, status.HTTP_200_OK)
```

### 6. 运行测试

现在你已经设置好了测试，是时候运行它们了。

在你的终端或命令提示符中，导航到你的 Django 项目目录并运行：

```bash
python manage.py test profiles
```

你应该在终端上看到测试的结果。成功的测试将表明权限按预期工作。

## 结论

测试权限对于确保 API 的安全性和可靠性至关重要。始终记住在更改权限配置后进行测试，以保证用户数据的安全。

在下一章中，我们将探讨为 profiles API 添加搜索功能。这将允许客户端查询 API 以获取特定的用户资料。敬请期待！

## 添加搜索 profiles 功能

本章所需的资产、资源和材料：

1. Django（需要安装：可以使用 pip 获取，`pip install django==2.2`）
   - 用途：构建我们的 Web 应用程序的主框架。
2. Django REST Framework（需要安装：可以使用 pip 获取，`pip install djangorestframework==3.9`）
   - 用途：Django 的一个附加工具包，用于轻松创建 RESTful API。
3. 数据库（你应该已经设置好了，可以是 SQLite（Django 自带）或你选择的另一个数据库。）
   - 用途：存储我们的用户资料数据。
4. Postman（需要安装：可以从 [Postman 官方网站](https://www.postman.com/downloads/) 下载）
   - 用途：测试我们的搜索 API 的工具。

### 介绍

搜索是任何应用程序的关键部分。随着我们的用户群增长，我们希望用户能够轻松找到并与其他用户的资料进行交互。在本章中，我们将为我们的 Profiles API 集成一个搜索功能。在本章结束时，你将了解如何实现搜索过滤器，根据某些条件定位 profiles。

### 分步指南

1. 更新 `views.py` 文件：
首先，我们需要修改我们的 `profiles` 视图集以支持搜索。

```python
from rest_framework import viewsets, filters
```

## 测试搜索配置文件

资产、资源和材料：

- 1. Django 开发环境：（至此，您已使用前面的章节搭建好环境。若未搭建，请参考第 5 - 16 章。）
- 2. Postman：（一款流行的 API 测试工具。可从 [Postman 官方网站](https://www.postman.com/downloads/) 下载并安装。）- 我们将使用 Postman 来测试我们的 API 端点。
- 3. ModHeader 扩展：（如果您一直跟着操作，应该已经安装了此扩展。若未安装，请从您浏览器的网上应用店获取。）- 用于设置令牌头并模拟经过身份验证的请求。
- 4. 示例配置文件数据：（请确保数据库中有一些配置文件。这可以是您之前输入的测试数据，或者您可以为本章添加一些新的配置文件。）

### 简介

在前面的章节中，我们构建了一个机制，允许用户在我们的 API 中搜索配置文件。搜索是任何存储用户相关数据的 API 的核心功能。本章将重点介绍如何测试搜索功能，以确保其按预期工作。

#### 测试搜索配置文件的步骤：

- 1. 确保存在配置文件：在测试搜索功能之前，请确保数据库中有多个配置文件。如果没有，请通过 API 或 Django 管理后台创建几个。
- 2. 打开 Postman：
    - 启动 Postman 并创建一个新请求。
    - 将请求类型设置为 `GET`。
    - 使用获取配置文件的端点 URL。它可能看起来像这样：`http://127.0.0.1:8000/api/profiles/`。
- 3. 设置请求头：
    - 由于我们的 API 受到保护，需要设置请求头以包含用于身份验证的令牌。使用 ModHeader 扩展添加您的令牌。
- 4. 搜索配置文件：
    - 在 URL 末尾添加查询参数以搜索配置文件。这将基于您实现的搜索功能。例如，如果您要搜索名为 "John" 的用户，您的 URL 可能看起来像：`http://127.0.0.1:8000/api/profiles/?search=John`。
    - 在 Postman 中发送请求。
- 5. 分析结果：
    - 发送请求后，查看响应。您应该看到与搜索条件匹配的配置文件。以我们的示例为例，应显示包含 "John" 名称的配置文件。
    - 确保返回的配置文件与搜索词相关。
- 6. 测试边缘情况：
    - 大小写敏感性：尝试使用不同的字母大小写进行搜索，例如 "john"、"JOHN" 和 "JoHn"，以确保搜索不区分大小写。
    - 部分匹配：尝试仅搜索名称或电子邮件的一部分，看看是否会出现配置文件。例如，搜索 "Jo" 仍应显示名为 "John" 的配置文件。
    - 无匹配项：搜索一个在您的配置文件中不存在的词。您应该得到一个空的结果集。
- 7. 检查错误响应：确保处理可能发生的错误情况。这可能是由于无效的令牌、过期的令牌或任何其他错误。API 应向用户返回清晰的错误消息。

测试是开发过程中的一个关键部分。通过确保搜索功能按预期工作，我们正在朝着为用户提供可靠且有效的 API 迈出又一步。请始终记住测试 API 的每个新功能和更改，以确保其健壮性和可靠性。

注意：这是测试配置文件搜索功能的基本概述。根据您的 API 的复杂性和您已实现的搜索功能，可能还有更多的测试用例和考虑因素需要纳入。

# 第 11 节：创建登录 API

### 创建登录 API 视图集

本章的资产、资源和材料：

- Django REST Framework (DRF)：用于创建我们的 API 视图集。（获取方式：`pip install djangorestframework`）
- ModHeader 浏览器扩展：用于在测试期间设置请求头中的令牌。（获取方式：从 Chrome 网上应用店或 Firefox 附加组件商店安装）
- Postman：一款流行的 API 测试工具。（获取方式：从 Postman 官方网站下载）

简介：

登录是任何 Web 应用程序的关键部分。本章将引导您创建登录 API 视图集。目标是让用户发送其凭据，验证它们，并作为响应，发回一个用于经过身份验证的请求的令牌。

#### 1. 设置登录序列化器：

在深入研究视图集之前，我们需要一个序列化器来验证来自用户的数据。这将是一个简单的序列化器，包含用户名和密码字段。

```python
from rest_framework import serializers

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)
```

这里，`write_only=True` 确保密码在序列化时不会被显示。

#### 2. 创建登录 API 视图集：

现在，我们将创建一个视图集来处理登录。
首先，导入必要的模块：

```python
from rest_framework import viewsets, status
from rest_framework.response import Response
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
```

然后，创建视图集：

```python
class LoginViewSet(viewsets.ViewSet):
    serializer_class = LoginSerializer

    def create(self, request):
        """处理用户登录并返回认证令牌。"""
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            user = authenticate(
                username=serializer.validated_data['username'],
                password=serializer.validated_data['password']
            )
            if user:
                token, created = Token.objects.get_or_create(user=user)
                return Response({'token': token.key}, status=status.HTTP_200_OK)
            else:
```

### 3. 将登录 API 视图集注册到 URL：

在你的 `urls.py` 文件中：

```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register('login', views.LoginViewSet, basename='login')

urlpatterns = [
    path('', include(router.urls)),
]
```

现在，端点 `/login/` 已准备好接收用于用户登录的 POST 请求。

### 4. 测试登录 API：

我们将使用 Postman 进行测试：

1.  打开 Postman。
2.  将请求类型设置为 `POST`。
3.  输入 URL：`http://localhost:8000/login/`。
4.  在 ‘Body’ 下，选择 ‘raw’ 和 ‘JSON (application/json)’。
5.  输入以下内容：
    ```json
    {
        "username": "yourUsername",
        "password": "yourPassword"
    }
    ```
    将 `yourUsername` 和 `yourPassword` 替换为你数据库中某个用户的凭据。
6.  点击 ‘Send’。如果成功，你将在响应中收到一个令牌。否则，你将收到一条错误消息。

**结论：**

在本章结束时，你已成功设置了一个登录 API 视图集，该视图集可以对用户进行身份验证，并为经过身份验证的请求返回一个令牌。此令牌对于在应用程序的后续部分中进行安全的 API 调用至关重要。

请记住，安全至关重要。始终确保你的令牌安全，切勿在客户端代码中暴露它们，并确保你的 API 在生产环境中使用 HTTPS 以确保加密通信。

在下一章中，我们将讨论如何测试我们的登录 API，确保其按预期运行并能抵御潜在威胁。

### 测试登录 API

**本章的资产、资源和材料：**

1.  Postman - 一个流行的 API 测试工具。（你可以从 [Postman 官方网站](https://www.postman.com/downloads/) 下载）
2.  登录 API 端点 - 你为登录功能设置的 URL。
3.  Django 开发服务器 - 确保你的开发服务器正在运行。
4.  凭据 - 一组用于测试登录功能的用户名和密码。如果你还没有创建任何用户，你需要在 Django 管理后台中创建。

### 简介

一旦你设置了登录 API，就必须确保它按预期运行。测试是任何开发过程中的关键阶段，它确保了系统的正确性和可靠性。在本章中，我们将使用一个强大的 API 测试工具 Postman 来测试我们的登录 API。

### 设置 Postman

1.  安装：如果你还没有安装，请从提供的链接下载并安装 Postman。
2.  启动和初始设置：
    - 安装后启动 Postman。
    - 如果你是首次用户，除非你想将集合保存到云端，否则可以跳过初始注册。
3.  设置新请求：
    - 点击 ‘+’ 标签页以创建一个新请求。
    - 从 URL 输入字段旁边的下拉菜单中，选择 ‘POST’，因为我们的登录 API 通常使用 POST 方法。
    - 在 URL 字段中输入你的登录 API 端点（例如，`http://127.0.0.1:8000/api/login/`）。

### 测试登录 API

1.  添加请求头：
    - 在 Postman 中，在 ‘Headers’ 标签下，你需要添加一个键值对：
        - 键：`Content-Type`
        - 值：`application/json`
    这告诉服务器你正在发送 JSON 数据。
2.  输入请求体数据：
    - 切换到 Postman 的 ‘Body’ 标签。
    - 选择 ‘raw’ 选项，并确保从下拉菜单中选择了 ‘JSON’。
    - 输入一组有效的凭据。例如：
        ```json
        {
          "username": "sampleuser",
          "password": "samplepassword123"
        }
        ```
3.  发送请求：
    - 点击蓝色的 ‘Send’ 按钮。
    - 观察响应。如果你的 API 和服务器运行正常，你应该会收到一个令牌作为响应的一部分。此令牌代表一次成功的登录，并可用于后续的经过身份验证的请求。
4.  负面测试：
    - 测试不仅仅是关于正面场景（一切正常）。你需要测试当事情出错时会发生什么。
    - 尝试输入错误的密码或用户名。预期的响应应该是一个错误，表明凭据无效。
    - 输入不完整的数据，例如省略密码或用户名。API 应该返回一个错误，指出缺少字段。

### 5. 测试令牌过期（可选）：

- 根据你在 Django Rest Framework 中的设置，令牌可能有有效期。如果你设置了令牌过期，请等待该时间段，然后尝试使用过期的令牌访问一个需要身份验证的端点。你应该会收到一个 ‘invalid token’ 或 ‘token expired’ 的响应。

## 结论

测试是一个持续且不断进行的过程。即使在部署 API 之后，你也应该定期进行测试，尤其是在进行更改或更新之后。使用像 Postman 这样的工具可以显著简化和精简流程，确保你的登录 API 强健、安全且可靠。

在下一章中，我们将探讨如何在 ModHeader 扩展中设置成功登录后收到的令牌，以便在进一步的测试和开发过程中轻松访问需要身份验证的路由。

> 注意：本章重点介绍使用 Postman 进行手动测试。在实际场景中，特别是对于较大的应用程序，你还应该为你的 API 实现自动化测试，以确保稳定性和一致性。

## 使用 ModHeader 扩展设置令牌头

**本章的资产、资源和材料：**

- ModHeader 扩展：这是一个浏览器扩展，允许你操作浏览器请求头。它适用于 Google Chrome 和 Mozilla Firefox。
- 你的 API：确保你的 Django REST API 正在运行，特别是我们在前面章节中开发的登录 API 端点。

### 简介

通过我们的登录 API 端点成功登录后，通常会返回一个令牌以对未来的请求进行身份验证。此令牌向服务器保证请求来自合法的、经过身份验证的用户。为了在每个请求中发送此令牌，我们需要将其包含在请求头中。在开发和测试期间，一种流行且简单的方法是使用 ModHeader 浏览器扩展。

在本章中，我们将介绍如何使用 ModHeader 扩展来设置令牌头，确保我们的 API 请求经过身份验证。

### 1. 安装 ModHeader

如果你还没有安装 ModHeader：

1.  打开你的浏览器。
2.  访问相应的商店：
    - Chrome 用户：访问 [Chrome 网上应用店](https://chrome.google.com/webstore/)
    - Firefox 用户：访问 [Firefox 附加组件网站](https://addons.mozilla.org/)
3.  搜索 “ModHeader”。
4.  在搜索结果中找到 ModHeader 扩展，然后点击 “添加到 Chrome” 或 “添加到 Firefox”。
5.  按照浏览器的提示完成安装。

### 2. 获取你的令牌

在我们在 ModHeader 中设置令牌之前，我们首先需要获取它：

1.  通过你的登录 API 登录。
2.  成功登录后，响应将包含一个身份验证令牌。复制此令牌；我们将在下一步中使用它。

### 3. 在 ModHeader 中设置令牌头

1.  点击浏览器工具栏中的 ModHeader 图标。这将打开 ModHeader 界面。
2.  在 ModHeader 界面中，你会看到两个主要的输入字段：“Request Headers” 和 “Filter”。
3.  在 “Request Headers” 下，点击 `+` 按钮以添加一个新的请求头。
4.  对于请求头名称，输入 `Authorization`。
5.  对于请求头值，输入 `Token <Your-Token>`。将 `<Your-Token>` 替换为你从登录 API 响应中复制的令牌。它应该看起来像 `Token 1234abcd5678efgh`。

注意：“Token” 和你的实际令牌之间的空格是必不可少的。

### 4. 发送经过身份验证的请求

在 ModHeader 中设置好令牌头后：

1.  向你的 API 发送任何需要身份验证的请求，例如获取用户资料或发布

### 5. 移除或禁用令牌头

在测试未认证请求或测试完成后，你可能需要移除或禁用令牌头：

1.  打开 ModHeader 界面。
2.  你可以选择：
    -   点击 `Authorization` 头旁边的垃圾桶图标将其移除。
    -   将右上角的绿色开关切换到关闭位置，以临时禁用 ModHeader。当你需要再次使用时，再将其切换回开启状态。

## 结论

使用 ModHeader 设置令牌头是在开发期间测试认证请求的一种简单而强大的方法。请始终确保你使用的是有效的令牌，并在必要时移除或禁用它们，以防止对你的 API 进行未经授权的访问。作为最佳实践，切勿分享包含可见令牌的 ModHeader 界面截图或视频，以维护安全性。

# 第 12 节：创建个人资料动态 API

## 规划个人资料动态 API

资产、资源和材料：

-   API 设计工具（例如 Postman、Swagger）：这些工具有助于规划和记录 API。你可以从各自的网站下载。
-   流程图软件（例如 Lucidchart、Draw.io）：有助于设计 API 的功能流程。在线提供免费版本。
-   笔记应用程序（例如 Notepad、Evernote）：用于记录想法和需求。

### 简介

规划是任何开发过程的重要组成部分。对于 API 来说，确保正确的端点、适当的方法和身份验证检查到位至关重要。在本章中，我们将逐步规划个人资料动态 API，这是我们应用程序的重要组成部分。

### 什么是个人资料动态？

从高层次来看，个人资料动态类似于你在 Twitter 或 Instagram 等社交媒体平台上看到的内容。它是一系列来自用户的帖子或状态更新。这些帖子可以被创建、读取、更新或删除。在我们的案例中，个人资料动态将展示用户发布的状态更新。

### 用户故事

在深入探讨技术细节之前，让我们先定义我们的用户故事。用户故事是一种敏捷开发工具，用于从最终用户的角度捕捉产品功能。

1.  作为一名注册用户，我想发布一条状态更新，以便其他人可以看到我在做什么。
2.  作为一名注册用户，我想查看我过去的状态更新，以回忆过去的事件。
3.  作为一名注册用户，我想在出错时更新一条状态。
4.  作为一名注册用户，我想删除一条不再相关的状态。
5.  作为一名浏览者，我想查看其他用户的状态更新，以了解他们的最新动态。

### 端点和方法

根据我们的用户故事，以下是我们可能需要的端点和方法的初步列表：

1.  POST /feed/：创建新的状态更新。
2.  GET /feed/{user_id}/：检索特定用户的所有状态更新。
3.  PUT /feed/{status_id}/：更新特定的状态更新。
4.  DELETE /feed/{status_id}/：删除特定的状态更新。

### 数据结构

接下来，我们需要确定与状态更新相关的数据。以下是初步的数据结构：

-   user_id：发布状态的用户的标识符。
-   status_id：每个状态的唯一标识符。
-   content：状态更新的文本内容。
-   timestamp：状态发布的日期和时间。

### 身份验证和权限

确保我们的 API 安全至关重要。根据我们的用户故事，以下是一些必要的权限：

-   所有用户都可以查看状态更新。
-   只有状态更新的所有者才能更新或删除它。
-   注册用户可以创建新的状态更新。

### 流程

使用流程图工具有助于可视化用户可能采取的操作顺序以及 API 将如何处理这些操作。例如：

1.  用户登录并收到身份验证令牌。
2.  用户向 /feed/ 发送带有状态内容的 POST 请求。
3.  API 检查令牌，验证其有效性，并将状态更新与该用户关联。
4.  状态更新保存到数据库，并返回给用户一个唯一的 status_id。
5.  对于后续操作（如更新或删除），API 会同时检查令牌以及用户是否是该状态更新的所有者。

## 结论

规划 API 需要结合理解最终用户的需求、定义清晰的端点和方法，以及确保数据完整性和安全性。有了这个针对个人资料动态 API 的计划，我们已准备好开始开发！在接下来的章节中，我们将深入探讨我们计划的 API 的实际实现。

## 添加新模型 Item

本章的资产、资源和材料：

-   Python（用途：编程语言。你可以从 Python 官方网站下载。）
-   Django（用途：Web 框架。可以使用 pip 安装 - “pip install Django”。）
-   Django 的 ORM（对象关系映射）（用途：数据库管理。包含在 Django 中。）

### 简介

在我们使用 Django 创建健壮的 REST API 的旅程中，我们现在处于一个关键步骤，我们将引入一个新模型来处理用户个人资料动态中的动态项。这个 “Item” 模型将存储用户发布的每条动态或状态更新。

让我们深入探讨如何创建这个模型。

### 步骤 1：打开你的 Models.py 文件

导航到你的 Django 应用目录中的 `models.py` 文件。我们将在这里定义我们的新模型。

```python
from django.db import models
from django.conf import settings
```

### 步骤 2：定义 Item 模型

对于我们的动态项，我们需要以下字段：

1.  `user_profile`：一个外键，链接到我们的用户模型。这将让我们知道是哪个用户发布了动态项。
2.  `status_text`：一个字符字段，用于存储用户的状态。
3.  `created_on`：一个日期字段，将在创建动态项时自动设置当前日期和时间。

代码如下所示：

```python
class Item(models.Model):
    user_profile = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    status_text = models.CharField(max_length=255)
    created_on = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.status_text
```

### 步骤 3：模型字段说明

1.  ForeignKey：此字段用于创建一对多关系。在这里，它用于将每个动态项链接到一个用户。`on_delete=models.CASCADE` 参数意味着如果删除了用户配置文件，其所有动态项也将被删除。
2.  CharField：这是一个用于存储字符数据的字段。我们将状态文本限制为 255 个字符。
3.  DateTimeField：此字段用于存储日期和时间数据。`auto_now_add=True` 参数意味着在创建新的动态项时，该字段将被设置为当前日期和时间。

### 步骤 4：迁移

定义模型后，下一步是为其创建迁移。迁移是 Django 将对模型所做的更改（添加、删除或更改字段）传播到数据库模式的方式。

在你的终端中，导航到项目的根目录并运行：

```bash
python manage.py makemigrations
```

之后，运行：

```bash
python manage.py migrate
```

这将应用迁移并在数据库中创建相应的表。

### 步骤 5：在 Admin 中注册模型

要通过 Django admin 界面管理我们的 `Item` 模型，你需要注册该模型。打开你应用目录中的 `admin.py` 并按如下方式修改：

```python
from django.contrib import admin
from .models import Item
admin.site.register(Item)
```

现在，当你登录 Django admin 界面时，你将能够查看、添加、编辑和删除动态项。

## 结论

随着 `Item` 模型的添加，我们处理用户动态的后端结构正在成型。随着你的进展，你将把这个模型与序列化器和视图集集成，以创建一个功能齐全的动态项 API。请记住，任何应用程序的核心通常在于其数据结构。通过花时间深思熟虑地设计你的模型，你将为成功奠定基础。

## 创建并运行模型迁移

资产、资源与材料：

-   Python：（你可以从[官方网站](https://www.python.org/downloads/)获取Python。用于运行我们的Django项目。）
-   Django：（通过pip安装，命令为`pip install django`。我们的Web应用框架。）
-   Django REST Framework：（通过pip安装，命令为`pip install djangorestframework`。用于构建API。）
-   终端或命令提示符：（操作系统预装。用于运行命令。）
-   文本编辑器：（我推荐Atom，你可以从[官方网站](https://atom.io/)获取。用于编写和编辑代码。）

简介：

Django中的迁移是其一大亮点功能。本质上，迁移允许你管理对应用模型所做的更改（如添加新字段、删除模型等），并将这些更改同步到数据库模式中。本章将指导你为我们的个人资料动态API创建一个新模型，并随后运行迁移以将此新模型反映到我们的数据库中。

### 步骤1：定义模型

在运行迁移之前，我们需要一个可供迁移的模型。鉴于本章是关于创建个人资料动态API，让我们创建一个名为`ProfileFeedItem`的模型：

```python
#### In models.py
from django.conf import settings
from django.db import models

class ProfileFeedItem(models.Model):
    user_profile = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    status_text = models.CharField(max_length=255)
    created_on = models.DateTimeField(auto_now_add=True)
```

以下是每行代码的作用：

-   `user_profile`：这是一个外键，将每个动态项链接到一个用户资料。
-   `status_text`：这将存储我们状态更新的实际文本。
-   `created_on`：这记录了状态创建的日期和时间。

### 步骤2：准备迁移

定义好模型后，下一步是告诉Django根据你所做的更改准备一个迁移。这通过`makemigrations`命令完成：

```bash
$ python manage.py makemigrations
```

运行此命令后，Django将在相应应用的`migrations/`目录中生成一个迁移文件。该文件将包含在数据库中进行更改（在本例中，是为我们的`ProfileFeedItem`模型创建一个新表）的步骤。

### 步骤3：应用迁移

迁移文件准备好后，我们现在可以将其应用到数据库模式中。运行`migrate`命令：

```bash
$ python manage.py migrate
```

这将应用所有待处理的迁移，包括我们刚刚为`ProfileFeedItem`创建的那个。此命令完成后，更改将反映在数据库中。

### 步骤4：验证迁移

为确保迁移成功，我们可以使用Django内置的管理界面或直接检查数据库。目前，让我们将`ProfileFeedItem`注册到管理站点以便于可视化：

1.  在`admin.py`中，添加：

```python
from .models import ProfileFeedItem
admin.site.register(ProfileFeedItem)
```

2.  现在，运行你的开发服务器：

```bash
$ python manage.py runserver
```

3.  导航到Django管理站点（`http://127.0.0.1:8000/admin/`）并登录。你应该能看到`ProfileFeedItem`被列出，并且能够添加和查看项目。

## 结论：

迁移是Django开发者工具包中不可或缺的工具，它允许从模型定义无缝过渡到一个活跃、功能齐全的数据库表。至此，你应该已经很好地理解了如何创建和运行迁移，让你的`ProfileFeedItem`模型栩栩如生。

## 将个人资料动态模型添加到管理后台

资产、资源与材料：

-   Django Admin：（Django自带，无需额外设置）。用途：用于管理数据库条目的内置管理界面。
-   Django Model：（课程前面已介绍）。用途：表示数据库模式和数据结构。
-   代码编辑器：（如Atom，在第3和第4章介绍）。用途：用于编写和修改你的Python代码。
-   运行开发服务器：确保你的Django开发服务器正在运行以便测试。

简介：

Django自带一个强大的管理界面，允许开发者轻松地与他们的模型进行交互。该界面提供了数据库的可视化表示，使CRUD操作（创建、读取、更新、删除）变得简单。在本章中，我们将逐步介绍将个人资料动态模型添加到此管理界面的步骤。

### 将个人资料动态模型添加到Django管理后台的步骤：

1.  导入所需模块：

    首先，我们需要确保在应用的`admin.py`文件中导入了我们的模型和Django管理模块。

```python
from django.contrib import admin
from .models import ProfileFeedItem
```

2.  注册模型：

    导入后，下一步是将模型注册到Django管理站点。通过注册模型，我们告诉Django这个模型应该通过管理界面可用并可管理。

```python
admin.site.register(ProfileFeedItem)
```

你的`admin.py`现在应该看起来像这样：

```python
from django.contrib import admin
from .models import ProfileFeedItem
admin.site.register(ProfileFeedItem)
```

3.  自定义管理界面（可选）：

    为了在管理界面中获得更定制化的体验，你可以创建一个管理类，为模型在管理站点中的显示和行为提供配置。

```python
class ProfileFeedItemAdmin(admin.ModelAdmin):
    list_display = ['user_profile', 'status_text', 'created_on']
    search_fields = ['status_text', 'user_profile__name']
admin.site.register(ProfileFeedItem, ProfileFeedItemAdmin)
```

这将在模型的列表视图中将`user_profile`、`status_text`和`created_on`字段显示为列。`search_fields`属性将在页面顶部添加一个搜索栏，允许你按`status_text`和关联的`user_profile`的名称搜索动态项。

4.  访问管理站点：

    要通过管理界面查看和操作个人资料动态模型：

    1.  确保你的开发服务器正在运行。
    2.  打开一个网页浏览器并导航至：`http://127.0.0.1:8000/admin/`
    3.  使用你的超级用户凭据登录。（如果你还没有创建超级用户，请参考第22章）。
    4.  登录后，你应该能在应用名称下看到你的`ProfileFeedItem`。
    5.  点击`ProfileFeedItem`将显示所有动态项的列表，你将有选项来添加、修改或删除条目。

## 结论：

随着你的个人资料动态模型现在已添加到Django管理界面，管理动态条目变得更加简单直接。管理界面不仅限于CRUD操作。随着你对Django的熟悉程度加深，你会发现你可以执行许多其他复杂任务，如导出数据、自定义外观以及添加内联相关模型。这些功能使Django的管理界面成为开发者不可或缺的工具。

### 下一步：

在下一章中，我们将深入探讨序列化个人资料动态模型的过程，这是为我们的应用程序创建API端点的重要一步。敬请期待！

## 创建个人资料动态项序列化器

本章的资产、资源与材料：

-   Django Rest Framework (DRF)：可以通过使用pip安装（`pip install djangorestframework`）来获取。我们使用DRF来创建序列化器，它允许复杂的数据类型（如Django模型）轻松转换为可以渲染为JSON的Python数据类型。
-   Models.py文件：我们从创建Django应用时就应该已经有了这个文件。我们的`ProfileFeedItem`模型就位于此处。
-   Serializers.py文件：我们将在这里创建序列化器。如果它不存在，你需要创建它。

简介：

序列化器允许复杂的数据类型（如Django模型）转换为可以渲染为JSON的Python数据类型。本质上，序列化器允许我们轻松地将数据库对象转换为可以渲染为响应的格式，反之亦然。在本章中，我们将为我们的`ProfileFeedItem`模型创建一个序列化器，使我们能够在API中轻松处理个人资料动态数据。

### 分步指南：

1.  设置你的序列化器文件：

    如果你还没有在应用目录中创建`serializers.py`文件，现在就创建它。

### 4. 序列化器详解：

- 我们的 `ProfileFeedItemSerializer` 将负责把模型实例转换为 JSON 格式。
- 当我们想在响应中输出一个动态项时，会将其传递给这个序列化器，由它来处理转换。
- 同样，当我们想根据接收到的 JSON 数据创建新的动态项时，这个序列化器将负责验证和解析传入的数据。

### 5. 测试序列化器（可选但推荐）：

虽然在生产环境中不是强制性的，但在开发阶段测试序列化器可以避免很多麻烦。可以使用 Django 的 shell 来进行测试。

```bash
python manage.py shell
```

然后，在 shell 中：

```python
from your_app_name.serializers import ProfileFeedItemSerializer
from your_app_name.models import ProfileFeedItem
feed_item = ProfileFeedItem.objects.first() # 或任何其他获取动态项的查询
serializer = ProfileFeedItemSerializer(feed_item)
print(serializer.data)
```

这应该会打印出该动态项的序列化版本。

## 结论：

序列化器是 Django REST Framework 的核心组成部分，它使我们能够轻松管理复杂的数据类型，并将其转换为更易于使用的格式。至此，你应该已经很好地理解了如何为特定模型创建序列化器。这将作为我们 API 视图的基础，我们将在接下来的章节中深入探讨。

> 注意：记得始终检查代码中的错误并进行彻底测试。确保我们的序列化器完美运行至关重要，因为它们是 API 逻辑的基础部分。

下一步：在下一章中，我们将为个人资料动态项创建 ViewSet。这将是 API 逻辑的核心，利用我们刚刚创建的序列化器。

## 为个人资料动态项创建 ViewSet

资产、资源和材料：

- Django REST Framework (DRF)：使用 ViewSet 和其他有助于快速创建 API 的实用工具需要此框架。可以使用 pip 安装 (`pip install djangorestframework`)。
- Django 项目和应用：确保你已经设置好了 Django 项目和应用。如果你是从前面的章节开始学习，应该已经准备就绪。
- 个人资料动态模型：这代表我们在前面章节中为个人资料动态项创建的数据库模型。
- Python 环境：确保你的 Python 虚拟环境已激活。

简介：
在 Django REST Framework 的世界里，`ViewSet` 是一个基于类的视图，它本身不提供任何方法处理器，而是继承自基本的类视图，如 `.list()`、`.create()`、`.retrieve()`、`.update()` 和 `.destroy()`。这些方法为我们的 API 提供了 CRUD 操作。
对于我们的个人资料动态 API，我们将使用 `ViewSet` 来管理动态项，实现创建新动态项、查看所有动态项、更新动态项和删除动态项等功能。

### 步骤 1：导入所需模块

在开始之前，我们需要确保导入了所有必要的模块：

```python
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from .models import ProfileFeedItem
from .serializers import ProfileFeedItemSerializer
```

### 步骤 2：创建个人资料动态项 ViewSet

现在，我们将为个人资料动态项创建一个 ViewSet：

```python
class ProfileFeedItemViewSet(viewsets.ModelViewSet):
    """处理个人资料动态项的创建、读取和更新。"""
    serializer_class = ProfileFeedItemSerializer
    queryset = ProfileFeedItem.objects.all()
```

在这个 `ProfileFeedItemViewSet` 中，我们做了以下工作：

- 使用 `ProfileFeedItemSerializer` 来确定数据如何与 JSON 格式相互转换。
- 指定了 `queryset`，它告诉 ViewSet 从哪里获取数据（在本例中，是所有 `ProfileFeedItem` 实例）。

### 步骤 3：将个人资料动态项 ViewSet 添加到 URL

为了让我们的 API 识别这个 ViewSet，我们需要将其与一个 URL 关联起来。我们将使用 DRF 提供的 `DefaultRouter` 来自动生成 URL 模式。

在你的 `urls.py` 中：

```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'feed', views.ProfileFeedItemViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

`DefaultRouter` 会自动为我们的 ViewSet 生成 URL 模式，减少了我们需要编写的代码量。参数 `r'feed'` 指定了 API 端点的名称。

### 步骤 4：测试 ViewSet

此时，你应该启动 Django 开发服务器并导航到端点（通常是 `http://127.0.0.1:8000/feed/`）来查看所有个人资料动态项的列表。你也可以使用 Postman 等工具，或者直接使用你的网页浏览器与 API 进行交互。尝试通过向 API 发送数据来创建新的动态项，或者更新/删除现有的动态项。

结论：
恭喜！你已经成功创建了一个用于管理 API 中个人资料动态项的 ViewSet。仅仅几行代码，你就启用了 CRUD 功能。Django REST Framework 的 ViewSet 的强大之处在于对常见模式的抽象，使开发者能够快速构建健壮的 API。
记住，随着你的应用程序增长，你可能需要向 ViewSet 添加自定义功能。DRF 提供了各种 mixin 和方法来帮助你实现这一点。请始终参考官方文档以探索更多高级功能。

## 测试动态 API

本章的资产、资源和材料：

- Postman：（你可以从官方网站 `https://www.postman.com/downloads/` 下载 Postman。这是一个流行的 API 端点测试工具。）
- ModHeader 浏览器扩展：（可在 Chrome 网上应用店和 Firefox 附加组件商店中找到。此扩展允许你修改 HTTP 请求头，这对于设置授权令牌非常方便。）
- Python 和 Django 设置：（假设你已经从前面的章节设置好了。）
- 我们 API 的 URL：（我们将用它来发送测试请求。它应该在前面的章节中已经定义。）

简介

既然我们已经设置了个人资料动态 API，测试其功能以确保一切按预期工作就至关重要了。测试是 API 开发中的关键步骤，因为它有助于识别潜在问题并确认 API 行为正确。在本章中，我们将使用 Postman 来测试我们的动态 API。

### 设置 Postman

1. 安装 Postman：如果你还没有安装，请从上面提供的链接下载并安装 Postman。
2. 创建新请求：启动 Postman 并点击 `+` 号打开一个新标签页。这将是我们发送测试请求的工作区。

### 测试我们的动态 API

1. 授权：
   - 在我们测试个人资料动态 API 之前，需要先进行授权。如果你的 API 使用基于令牌的认证，你应该在登录或用户注册时收到一个令牌。
   - 在 Postman 中，在 `Headers` 标签下，将键设置为 `Authorization`，值设置为 `Token <your_token>`。（将 `<your_token>` 替换为你的实际令牌。）或者，如果你使用 ModHeader 浏览器扩展，可以在那里设置令牌。
2. 测试 `GET` 请求：
   - 在 Postman 中将请求类型设置为 `GET`。
   - 输入你的 API 动态 URL（例如，`http://localhost:8000/feed/`）。
   - 点击 `Send` 按钮。你应该会在响应中看到返回的动态项列表，如果没有添加任何动态项，则返回空列表。
3. 测试 `POST` 请求：
   - 将请求类型切换为 `POST`。

### 4. 测试 `PUT` 请求：

- 首先，使用 `GET` 请求获取一个动态项，以记录其 ID。
- 将请求类型切换为 `PUT`。
- 修改 URL 以指向特定的动态项（例如，`http://localhost:8000/feed/1/`，其中 `1` 是动态项的 ID）。
- 在 `Body` 标签页中，输入更新后的状态文本：
```json
{
    "status_text": "This is an updated test status!"
}
```
- 点击 `Send` 按钮。响应应显示更新后的动态项。

### 5. 测试 `DELETE` 请求：

- 使用与 `PUT` 测试相同的目标 URL。
- 将请求类型切换为 `DELETE`。
- 点击 `Send` 按钮。您应收到一个响应，表明动态项已被删除。

通过尝试对同一项进行另一次 `GET` 请求来确认；它应返回 `404 Not Found`。

### 处理测试失败

如果任何测试失败：

- 查看响应中提供的错误消息。这通常可以提供关于问题所在的线索。
- 检查您的 API 代码，确保端点定义正确，并且身份验证和权限设置得当。
- 查阅 Django 日志以查找任何服务器端错误。

## 结论

彻底的测试对于确保 API 的可靠性和稳健性至关重要。请记住在任何更改或更新后进行测试，并考虑在未来实施自动化测试以简化此流程。随着我们的动态 API 测试完成，我们可以自信地推进项目的下一步。

## 为动态 API 添加权限

资产、资源和材料：

- Django 和 Django REST Framework：如果您尚未安装这些包，它们对于 API 的开发至关重要。您可以通过运行 `pip install django django-rest-framework` 使用 pip 获取它们。
- Postman 或类似工具：用于 API 测试的工具，以查看权限的实际效果。您可以从 [Postman 官方网站](https://www.postman.com/downloads/) 下载 Postman。
- 您现有的项目：确保您在前面章节中介绍过的现有 Django 项目中工作。
- 自定义用户模型和动态模型：这些应在前面的章节中已经创建。

### 权限的目的：

Django REST Framework (DRF) 中的权限用于授予或拒绝不同类别的用户对 API 不同部分的访问权限。对于我们的动态 API，我们希望确保只有经过身份验证的用户才能创建、查看或修改动态项。

### 步骤 1：了解默认权限

默认情况下，DRF 提供了一组可以应用于我们视图的权限。一些最常用的权限包括：

- AllowAny：向任何用户提供无限制的访问。
- IsAuthenticated：仅授予经过身份验证的用户访问权限。
- IsAdminUser：仅授予管理员用户访问权限。
- IsAuthenticatedOrReadOnly：向未经过身份验证的用户提供只读访问权限，但向经过身份验证的用户提供完全访问权限。

对于我们的动态 API，我们将使用 `IsAuthenticated` 权限，以确保只有登录用户才能与动态交互。

### 步骤 2：设置全局权限

首先，您可以在项目设置中设置全局权限，以默认将 `IsAuthenticated` 权限应用于所有视图：

```python
#### settings.py
REST_FRAMEWORK = {
    ...
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}
```

但是，对于本章，我们将在视图中显式设置权限。

### 步骤 3：使用权限更新动态 API 视图

导航到您的动态 API 视图（在前面的章节中创建）。在这里，我们将显式设置权限类。

```python
#### api/views.py（或您的动态视图所在的位置）
from rest_framework.permissions import IsAuthenticated
class ProfileFeedViewSet(viewsets.ModelViewSet):
    ...
    permission_classes = (IsAuthenticated,)
    ...
```

这确保只有经过身份验证的用户才能访问此视图。

### 步骤 4：测试权限

现在我们已经应用了权限，测试并确保它们正常工作至关重要。

1. 未经过身份验证的访问测试：使用 Postman 或您首选的 API 测试工具。尝试在未登录的情况下访问动态 API。您应收到 `401 Unauthorized` 错误。
2. 经过身份验证的访问测试：首先，使用用户配置文件登录。使用收到的令牌访问动态 API。您应该能够毫无问题地执行所有 CRUD 操作（创建、检索、更新、删除）。

### 步骤 5：自定义权限（可选）

为了更细粒度的控制，DRF 允许您创建自定义权限。例如，如果您希望只有动态项的创建者才能编辑或删除它，您需要一个自定义权限。

这是一个基本示例：

```python
from rest_framework.permissions import BasePermission

class IsOwner(BasePermission):
    """
    自定义权限，仅允许对象的所有者编辑或删除它。
    """
    def has_object_permission(self, request, view, obj):
        return obj.user_profile == request.user
```

然后，您可以将此权限添加到您的视图：

```python
permission_classes = (IsAuthenticated, IsOwner,)
```

请记住，权限是按列出的顺序进行检查的。因此，`IsAuthenticated` 将在 `IsOwner` 之前被检查。

## 结论

权限在确保 API 的安全性和功能性方面起着至关重要的作用。到现在，您应该对如何在 Django REST Framework 中实施和测试权限有了很好的理解。一如既往，请确保在开发和生产环境中彻底测试所有权限，以确保应用程序的安全性和完整性。

## 测试动态 API 权限

资产、资源和材料：

- Postman：（一个流行的 API 测试工具。可以从 [Postman 官方网站](https://www.postman.com/downloads/) 下载）。使用此工具发送 HTTP 请求并验证来自我们 API 的响应。
- 您的 Django 后端：确保您的 Django 服务器正在运行，并且您已完成动态 API 的设置。
- API 端点：指向您动态 API 的 URL（通常是类似 `http://localhost:8000/api/feed/` 的地址，具体取决于您的配置）。

引言
测试是软件开发不可或缺的一部分。在构建 API 的背景下，测试确保端点正常运行，并在响应各种请求时提供预期的数据。在本章中，我们将重点测试动态 API 的权限。确保适当的权限对于数据隐私和完整性至关重要。

目标：

- 测试未经过身份验证的用户无法访问动态数据。
- 测试经过身份验证的用户可以访问动态并发布内容。
- 验证用户只能修改或删除自己的动态数据。

### 1. 设置 Postman 以进行 API 测试

1. 下载并安装 Postman：如果您尚未安装 Postman，请从上面提供的链接下载并安装它。
2. 打开 Postman：安装后，启动 Postman。
3. 创建新请求：点击 '+' 标签页以创建新请求。

### 2. 测试未经过身份验证的请求

目标：确保未经过身份验证的请求（未提供有效用户令牌的请求）无法访问动态。

1. 在 Postman 中设置请求：
   - 将请求类型设置为 `GET`。
   - 在 URL 字段中输入您的动态 API 端点（`http://localhost:8000/api/feed/`）。
2. 发送请求：点击 `Send` 按钮。
3. 查看响应：API 应返回 `403 Forbidden` 状态，表明用户没有适当的权限在未经过身份验证的情况下查看动态。

### 3. 测试经过身份验证的请求

目标：对用户进行身份验证，并测试他们是否可以查看动态并发布内容。

1. 登录以获取令牌：
   - 将请求类型设置为 `POST`。

## 测试新的私密动态

资产与资源：

- Python：用于REST API后端开发的编程语言。
  - 获取方式：从[官方网站](https://www.python.org/downloads/)下载并安装。
  - 用途：构建REST API的核心语言。
- Django：Python的高级Web框架。
  - 获取方式：通过pip安装（`pip install django`）。
  - 用途：构建Web应用和API的框架。
- Django REST Framework：用于在Django中构建Web API的工具包。
  - 获取方式：通过pip安装（`pip install djangorestframework`）。
  - 用途：提供构建REST API的工具。
- Postman或ModHeader：用于测试API端点的工具。
  - 获取方式：下载[Postman](https://www.postman.com/downloads/)或从Chrome网上应用店安装[ModHeader扩展](https://chrome.google.com/webstore/detail/modheader/idgpnmonknjnojddfkpgkljpfnnfcklj)。
  - 用途：用于测试API端点、发送请求和查看响应。
- SQLite：Django在开发环境中使用的默认数据库。
  - 获取方式：Django预配置。
  - 用途：存储和检索API数据。

简介

在实现了限制状态更新仅对登录用户可见的新功能后，我们现在必须测试此功能。确保隐私和安全是任何应用程序的重要组成部分，尤其是在用户内容和个人信息领域。在本章中，我们将进行测试，以验证我们的私密动态系统是否按预期运行。

### 步骤1：设置

通过在终端或命令提示符中导航到项目目录并运行以下命令，确保Django服务器正在运行：

```bash
python manage.py runserver
```

确保您已准备好Postman或ModHeader以测试API端点。

### 步骤2：未认证测试

1. 打开Postman。
2. 将请求方法设置为GET。
3. 输入获取个人资料动态的端点URL。
4. 点击发送。

预期结果：
您应该收到403禁止错误。这表明未认证用户无法访问动态。

### 步骤3：认证测试

为确保只有登录用户才能查看个人资料动态，我们需要使用认证请求进行测试：

1. 注册新用户或使用现有用户的凭据。
2. 登录以获取令牌。
3. 将令牌添加到请求头中，格式为`Authorization: Token <your_token_here>`。
4. 再次向个人资料动态端点发送GET请求。

预期结果：
您现在应该收到200 OK状态，并在响应正文中看到动态数据。

### 步骤4：测试用户特定权限

目标：确保用户只能修改或删除自己的动态数据。

1. 创建第二个用户（如果尚未创建）：此用户将用于测试他们无法修改其他用户的动态数据。
2. 测试修改他人的动态数据：
   - 以第二个用户身份登录并获取其令牌。
   - 尝试修改（使用`PUT`或`PATCH`请求）或删除（使用`DELETE`请求）第一个用户创建的动态数据。
   - API应返回`403 Forbidden`状态，表明第二个用户没有适当的权限修改其他用户的动态数据。

## 结论

在本章结束时，您应该已经成功测试了与动态API相关的各种权限。正确测试权限可确保数据完整性和隐私性，这是可信后端系统的重要组成部分。请始终记住，构建API只是流程的一部分；严格的测试才能确保它在所有场景中按预期工作。

## 限制状态更新仅对登录用户可见

所需资产与资源：

1. Django和Django REST Framework（DRF）：（用于构建和测试我们的API。如果尚未安装，可以通过`pip install django djangorestframework`获取。）
2. Postman或浏览器：（用于测试我们的API视图和端点。可以从[此处](https://www.postman.com/downloads/)下载Postman。）
3. Django内置认证：（用于处理用户认证，包含在Django包中。）
4. 前几章构建的个人资料动态API。

简介：
在开始编码部分之前，让我们了解此功能的重要性。现代Web应用程序最常见的要求之一是将某些内容或功能限制为仅认证用户可用。对于社交平台或基于用户的平台，状态更新或动态通常是个人的。确保这些更新仅对登录用户可用可以保护用户数据。

### 步骤1：更新ViewSet权限

1. 打开定义了`ProfileFeedItemViewSet`的views.py。
2. 在文件顶部添加所需的导入：
3. 更新`ProfileFeedItemViewSet`以包含我们的新权限：

```python
from rest_framework.permissions import IsAuthenticated
```

```python
class ProfileFeedItemViewSet(viewsets.ModelViewSet):
    ...
    permission_classes = (IsAuthenticated,)
```

我们在这里做的是指定`ProfileFeedItemViewSet`中的所有操作都需要`IsAuthenticated`权限。

### 步骤2：测试限制

在继续之前，让我们确保我们的限制正在工作：

1. 打开Postman或您的浏览器。
2. 尝试在未登录的情况下访问个人资料动态API。您应该收到`403 Forbidden`错误。
3. 现在，使用用户登录（使用我们之前设置的登录API）。
4. 使用登录时收到的令牌来认证您的请求。
5. 再次访问个人资料动态API。这次，由于您已认证，您应该能够查看状态更新。

### 步骤3：自定义权限

虽然`IsAuthenticated`权限是一个很好的开始，但有时您可能希望向权限添加自定义逻辑。例如，您可能希望将某些操作限制为个人资料的所有者。

要实现此目的：

1. 如果还没有，请创建一个新的权限文件（通常在应用目录中命名为`permissions.py`）。
2. 定义一个新的权限类：

```python
from rest_framework.permissions import BasePermission

class IsOwnerOrReadOnly(BasePermission):
    """
    自定义权限，仅允许所有者编辑其个人资料。
    """
    def has_object_permission(self, request, view, obj):
        # 任何请求都允许只读权限
        if request.method in ['GET']:
            return True
        # 仅允许所有者进行写入权限
        return obj.user == request.user
```

3. 更新`views.py`中的`ProfileFeedItemViewSet`以使用此新权限：

```python
permission_classes = (IsAuthenticated, IsOwnerOrReadOnly,)
```

### 步骤4：其他考虑事项

虽然我们将视图限制为认证用户，但请考虑以下事项以增强安全性：

- 分页：限制返回的结果数量以防止数据抓取。
- 速率限制：限制端点的访问频率以防止滥用。
- 安全令牌：定期轮换和过期认证令牌。

## 结论：

通过几行代码，我们成功地为应用程序添加了关键的安全层。定期测试和重新审视应用程序的权限可确保您维护一个强大且用户信任的平台。下次考虑功能时，请始终记住，有时重要的不是您添加了什么功能，而是您限制了什么。

## 测试新的私密动态

资产与资源：

- Python：用于REST API后端开发的编程语言。
  - 获取方式：从[官方网站](https://www.python.org/downloads/)下载并安装。
  - 用途：构建REST API的核心语言。
- Django：Python的高级Web框架。
  - 获取方式：通过pip安装（`pip install django`）。
  - 用途：构建Web应用和API的框架。
- Django REST Framework：用于在Django中构建Web API的工具包。
  - 获取方式：通过pip安装（`pip install djangorestframework`）。
  - 用途：提供构建REST API的工具。
- Postman或ModHeader：用于测试API端点的工具。
  - 获取方式：下载[Postman](https://www.postman.com/downloads/)或从Chrome网上应用店安装[ModHeader扩展](https://chrome.google.com/webstore/detail/modheader/idgpnmonknjnojddfkpgkljpfnnfcklj)。
  - 用途：用于测试API端点、发送请求和查看响应。
- SQLite：Django在开发环境中使用的默认数据库。
  - 获取方式：Django预配置。
  - 用途：存储和检索API数据。

简介

在实现了限制状态更新仅对登录用户可见的新功能后，我们现在必须测试此功能。确保隐私和安全是任何应用程序的重要组成部分，尤其是在用户内容和个人信息领域。在本章中，我们将进行测试，以验证我们的私密动态系统是否按预期运行。

### 步骤1：设置

通过在终端或命令提示符中导航到项目目录并运行以下命令，确保Django服务器正在运行：

```bash
python manage.py runserver
```

确保您已准备好Postman或ModHeader以测试API端点。

### 步骤2：未认证测试

1. 打开Postman。
2. 将请求方法设置为GET。
3. 输入获取个人资料动态的端点URL。
4. 点击发送。

预期结果：
您应该收到403禁止错误。这表明未认证用户无法访问动态。

### 步骤3：认证测试

为确保只有登录用户才能查看个人资料动态，我们需要使用认证请求进行测试：

1. 注册新用户或使用现有用户的凭据。
2. 登录以获取令牌。
3. 将令牌添加到请求头中，格式为`Authorization: Token <your_token_here>`。
4. 再次向个人资料动态端点发送GET请求。

预期结果：
您现在应该收到200 OK状态，并在响应正文中看到动态数据。

### 步骤4：测试用户特定权限

目标：确保用户只能修改或删除自己的动态数据。

1. 创建第二个用户（如果尚未创建）：此用户将用于测试他们无法修改其他用户的动态数据。
2. 测试修改他人的动态数据：
   - 以第二个用户身份登录并获取其令牌。
   - 尝试修改（使用`PUT`或`PATCH`请求）或删除（使用`DELETE`请求）第一个用户创建的动态数据。
   - API应返回`403 Forbidden`状态，表明第二个用户没有适当的权限修改其他用户的动态数据。

## 结论

在本章结束时，您应该已经成功测试了与动态API相关的各种权限。正确测试权限可确保数据完整性和隐私性，这是可信后端系统的重要组成部分。请始终记住，构建API只是流程的一部分；严格的测试才能确保它在所有场景中按预期工作。

### 步骤 4：跨用户隐私测试

为确保用户只能查看自己的私人动态，而无法查看其他用户的动态：

1. 创建或使用第二个用户账户。
2. 使用该账户登录以获取其身份验证令牌。
3. 在 Postman 中，使用与上面相同的 GET 请求，将 Authorization 头中的原始令牌替换为第二个用户的令牌。
4. 点击发送。

预期结果：
返回的动态应仅包含第二个用户的状态更新，而不包含原始用户的。这证实了每个用户只能访问自己的私人动态。

### 步骤 5：登出测试

1. 从用户账户登出。
2. 使用 Postman，尝试使用相同的 GET 请求和令牌访问私人动态。

预期结果：
你应该再次收到 403 Forbidden 错误。这证实了一旦用户登出，即使他们拥有之前有效的令牌，也无法访问私人动态。

## 结论

完成这些测试后，你应该对私人资料动态的功能和安全性充满信心。彻底测试新功能至关重要，尤其是那些与用户隐私和身份验证相关的功能，以确保应用程序的稳健性和可靠性。请始终记住，经过充分测试的应用程序才是可靠的应用程序。

# 第 13 节：将我们的 API 部署到 AWS 服务器

## 将我们的应用部署到 AWS 简介

本章的资产、资源和材料：

- AWS 账户：你需要一个 AWS 账户。如果没有，请在 [Amazon Web Services](https://aws.amazon.com/) 注册（用途：访问和使用 AWS 服务）。
- AWS EC2 文档：[官方文档](https://docs.aws.amazon.com/ec2/index.html)（用途：作为参考指南，并深入了解特定功能）。
- AWS CLI：[安装指南](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)（用途：从命令行管理 AWS 服务）。
- SSH 客户端：根据你的操作系统，可能已经安装了一个。如果没有，你可以使用 Windows 的 PuTTY 或 macOS 和 Linux 的内置终端等工具（用途：连接到 AWS EC2 实例）。

### 1. 简介

将应用程序部署到云端最初可能看起来是一项艰巨的任务，特别是如果你不熟悉云概念或像 AWS 这样的特定云提供商。然而，AWS 使得部署应用程序变得相对简单，本章旨在向你介绍将 Django REST API 部署到 AWS 服务器所需的基本概念和步骤。

在本章结束时，你将清楚地了解：

1. 为什么我们选择 AWS 进行部署。
2. 我们将使用的主要 AWS 服务：Amazon EC2（弹性计算云）。
3. 与 AWS EC2 相关的关键概念。

### 2. 为什么选择 AWS？

当今有许多云提供商可用，包括 Google Cloud Platform (GCP)、Microsoft Azure 等。那么，为什么有人会选择 AWS 呢？

1. 成熟度和覆盖范围：AWS 于 2006 年推出，是最早且最成熟的云平台之一。它提供广泛的服务和工具，几乎满足所有需求。
2. 可扩展性：AWS 的服务，尤其是 EC2，是为扩展而构建的。无论你是服务十个用户还是一千万用户，AWS 都能处理。
3. 安全性：AWS 提供各种工具和最佳实践，以确保应用程序和数据的安全。
4. 成本效益：通过 AWS 的按需付费定价模型，你只需为使用的资源付费。

### 3. Amazon EC2（弹性计算云）简介

Amazon EC2 是一项在云中提供可调整计算容量的 Web 服务。简单来说，它允许你运行虚拟服务器，并根据需求扩展计算能力。

与 AWS EC2 相关的关键概念：

1. 实例：Amazon EC2 环境中的虚拟服务器。
2. Amazon 系统映像 (AMI)：启动实例所需的预配置包。这包括操作系统、服务器、应用服务器和应用程序。
3. 实例类型：定义用于实例的主机计算机的硬件。它影响内存、CPU、存储等方面。
4. 安全组：充当虚拟防火墙，控制一个或多个实例的流量。
5. 密钥对：实例的安全登录信息。它包含 AWS 存储的公钥和你存储的私钥文件。

### 4. 部署准备

在我们深入后续章节的部署过程之前，有必要熟悉一些先决条件：

1. 了解你的应用程序：了解你的应用程序的需求。它需要特定的操作系统吗？硬件要求是什么？
2. 选择合适的 EC2 实例：AWS 提供针对不同需求量身定制的各种实例类型。根据你的应用程序需求，选择一个合适的实例。
3. 安全性：确保了解 AWS 的安全最佳实践。配置错误可能会使你的应用程序面临不必要的风险。

### 5. 结论

AWS 为部署像我们的 Django REST API 这样的 Web 应用程序提供了一个强大的平台。在本章中，我们介绍了选择 AWS 的原因，并概述了我们将用于部署应用程序的 EC2 服务。

在接下来的章节中，我们将更深入地探讨在 AWS 上设置应用程序、保护其安全以及确保其平稳运行以服务用户的细节。

> *给读者的说明*：AWS 及其服务随时间推移而发展，会定期添加新功能。参考 AWS 的官方文档以获取最新和最深入的信息是一个好习惯。

# 将密钥对添加到 AWS

本章的资产、资源和材料：

- AWS 管理控制台（通过在 AWS 官方网站注册获取）
- 带有 SSH 客户端的计算机（macOS 和 Linux 内置。对于 Windows，请使用 PuTTY 等软件）

## 简介：

在我们将 Django REST API 部署到 AWS 之前，一个关键步骤是设置密钥对。AWS 使用公钥密码学来加密和解密登录信息。密钥对由一个公钥（AWS 用于加密数据）和一个私钥（你用于解密数据）组成。简单来说，可以将其视为一把独特的数字钥匙，让你安全地访问服务器。

本章将指导你在 AWS 上创建和下载密钥对，你稍后将使用它通过 SSH 连接到你的服务器实例。

### 1. 登录 AWS 管理控制台

首先导航到 AWS 管理控制台。如果你没有账户，需要注册。登录后，在右上角选择你要部署服务的区域。

### 2. 导航到 EC2 仪表板

从 AWS 管理控制台，通过在搜索栏中输入“EC2”或在“最近访问的服务”部分找到 EC2 服务。

### 3. 访问密钥对部分

在 EC2 仪表板上，在侧边栏的“网络和安全”部分下找到“密钥对”选项。

### 4. 创建新的密钥对

- 点击“创建密钥对”按钮。
- 为你的密钥对提供一个名称，例如“MyDjangoRESTAPIKey”。
- 从“文件格式”下拉菜单中，根据你的操作系统选择合适的文件格式：
  - `.pem` 用于 Linux 和 macOS
  - `.ppk` 用于 Windows（如果你使用 PuTTY）
- 点击“创建密钥对”按钮。

### 5. 下载并保护私钥

点击“创建密钥对”后，私钥文件将自动下载到你的计算机。此私钥对于访问你的服务器至关重要，因此请务必妥善保管。

- 切勿分享你的私钥。
- 将其移动到计算机上的安全位置。
- 如果你使用的是 macOS 或 Linux，你可能还需要使用以下命令为密钥设置正确的权限：

```bash
chmod 400 path_to_your_key.pem
```

### 6. 如果丢失私钥怎么办？

重要的是要明白，如果你丢失了对私钥的访问权限，AWS 无法帮助你找回它。在这种情况下，你需要创建一个新的密钥对并将其与你的 EC2 实例关联，或者使用新的密钥对启动一个新实例。

## 结论：

你现在已经成功从 AWS 创建并下载了你的密钥对。当我们继续创建 EC2 服务器实例并安全地连接到它时，这个密钥对将发挥关键作用。在接下来的章节中，我们将更深入地探讨使用我们刚刚生成的密钥对在 AWS 上设置服务器。请记住，始终将你的私钥保密，以确保服务器的安全。

## 创建 EC2 服务器实例

资产、资源和材料：

- 1. AWS 账户：亚马逊网络服务上的一个账户。如果您还没有，可以[在此](https://aws.amazon.com/)注册。这将允许您访问 EC2 服务并创建服务器实例。
- 2. 密钥对：这在第 65 章中创建并讨论过。它用于安全地通过 SSH 访问您的 EC2 实例。
- 3. SSH 客户端：用于安全连接到您的 EC2 服务器实例的软件。流行的选择包括 Windows 上的 `PuTTY` 或 macOS 和 Linux 内置的 `ssh` 命令。

简介：

在本章中，我们将指导您在 AWS 上创建一个 EC2（弹性计算云）服务器实例。Amazon EC2 在云中提供可扩展的计算能力，使开发者更容易进行网络规模的计算。这将是我们部署 Django REST API 的远程服务器。

创建 EC2 服务器实例的分步指南：

- 1. 登录 AWS 管理控制台：
登录您的 AWS 账户。如果您是 AWS 新用户，需要注册并提供账单详情。别担心；AWS 为 EC2 提供免费套餐，允许您在不产生费用的情况下探索该服务。
- 2. 导航到 EC2：
从 AWS 管理控制台，点击 `Services`，然后在 `Compute` 类别下选择 `EC2`。
- 3. 启动新实例：
点击 `Launch Instance` 按钮开始流程。
- 4. 选择 Amazon 机器映像 (AMI)：
AMI 是实例的预配置模板。出于我们的目的，选择 `Ubuntu Server`，因为它是一个流行且支持良好的选择。确保选择最新的稳定版本。
- 5. 选择实例类型：
出于本教程的目的，您可以选择 `t2.micro` 实例类型，它符合 AWS 免费套餐的条件。它适合开发和测试目的。稍后，根据您的生产需求，您可以选择更强大的实例。
- 6. 配置实例：
    - 实例数量：1（因为我们只设置一个服务器）
    - 网络：默认 VPC（虚拟私有云）
    - 子网：目前选择默认值。子网代表您 VPC 的 IP 地址块内的一个范围。
    - 确保 `Auto-assign Public IP` 设置为 `Enable`。这会给您的实例一个公共 IP 地址，您可以使用它从互联网访问实例。
- 7. 添加存储：
默认存储（根卷）是通用 SSD 上的 8GB。对于我们的教程来说应该足够了。但是，在实际场景中，请确保根据应用程序的需求进行调整。
- 8. 添加标签：
标签对于计费和组织很有用。例如：
    - 键：`Name`
    - 值：`DjangoRESTAPI-Server`
- 9. 配置安全组：
安全组充当防火墙，控制进出实例的流量。出于我们的目的：
    - 创建一个新的安全组。
    - 给它起一个描述性的名称，例如 `DjangoRESTAPISG`。
    - 添加规则：
        - `SSH`：允许安全访问您的服务器。
        - `HTTP`：允许 HTTP 流量。
        - `HTTPS`：允许 HTTPS 流量。
    - 对于源，选择 `Anywhere`。这允许来自任何 IP 的连接。（在生产场景中，出于安全原因，您会将其限制为已知 IP。）
- 10. 审查并启动：
审查您的设置。满意后，点击 `Launch` 按钮。将出现一个提示，要求提供密钥对。使用您在第 65 章创建的密钥对。这将允许您安全地通过 SSH 连接到您的服务器。
- 11. 等待初始化：
启动后，EC2 实例需要几分钟来初始化。一旦实例状态变为“running”，您就可以使用其公共 IP 连接到它。

结论：
恭喜！您已成功在 AWS 上创建了一个 EC2 服务器实例。在接下来的章节中，我们将指导您将 Django REST API 部署到此服务器上，使其可以从世界任何地方访问。请记住保管好您的密钥对，因为它是您访问服务器的门户。

## 向我们的项目添加部署脚本和配置

资产、资源和材料：

- Amazon EC2（弹性计算云）：这是一项在云中提供可扩展计算能力的网络服务。如果您还没有，可以[在此](https://aws.amazon.com/)注册 AWS 账户。
- 部署脚本：文件中的一组指令，用于自动将我们的项目部署到服务器。
- `requirements.txt`：列出我们项目所需的所有 Python 依赖项。
- SSH（安全外壳）：一种加密网络协议，用于在远程服务器上安全地启动会话。可以使用 Windows 上的 [PuTTY](https://www.putty.org/) 等工具或 macOS 和 Linux 内置的 SSH 客户端。
- `gunicorn`：用于 UNIX 的 Python WSGI HTTP 服务器。这将是我们生产环境中 Django 的应用服务器。
- AWS CLI（命令行界面）：AWS 提供的用于管理 AWS 服务的工具。您可以[在此](https://aws.amazon.com/cli/)下载。
- Virtualenv：用于创建隔离 Python 环境的工具。
- 配置文件：对于设置特定于环境的配置、密钥和变量至关重要。

### 1. 简介：

部署是将我们本地开发工作呈现给世界的阶段。由于我们将在 AWS EC2 上部署，我们需要自动化该过程以确保一致且快速的部署。

### 2. 设置环境：

在开始编写部署脚本之前，让我们确保环境已准备就绪：
*确保已安装 AWS CLI。*
您可以通过以下方式安装：

```
pip install awscli
```

确保您已使用凭据设置 AWS CLI。运行：

```
aws configure
```

系统将提示您输入 AWS 访问密钥、秘密密钥、区域和默认输出格式。

### 3. 创建 `requirements.txt`：
我们的 EC2 实例需要知道要安装哪些 Python 包。在您的项目目录内，运行：

```
pip freeze > requirements.txt
```

这将创建一个包含所有 Python 依赖项的文件。

### 4. 编写部署脚本：
让我们创建一个名为 `deploy.sh` 的脚本。此脚本将为我们处理部署过程。

```bash
#!/bin/bash
#### 导航到我们的项目目录
cd /path/to/your/django/project
#### 激活我们的虚拟环境
source venv/bin/activate
#### 安装必要的 Python 包
pip install -r requirements.txt
#### 收集静态文件
python manage.py collectstatic --noinput
#### 最后，使用 gunicorn 运行我们的应用程序
gunicorn your_project_name.wsgi:application --bind 0.0.0.0:8000
```

别忘了使此脚本可执行：

```
chmod +x deploy.sh
```

### 5. 配置文件：

部署时，我们需要与开发环境不同的配置。这就是配置文件发挥作用的地方。

在您的 Django 项目的设置目录中创建一个 `prod_settings.py`。此文件将包含特定于生产的配置。例如，`DEBUG` 应设置为 `False`。

确保您已隐藏密钥和数据库配置，可以使用环境变量或 AWS 提供的密钥管理工具（如 AWS Secrets Manager）。

### 6. 将脚本和配置上传到 AWS：

现在，我们将部署脚本和配置上传到 EC2 实例。确保您已设置好 EC2 实例并准备好 `.pem` 密钥文件。

使用 SSH：

```
scp -i path_to_your_key.pem deploy.sh ubuntu@your_ec2_ip:/path/on/server
scp -i path_to_your_key.pem requirements.txt ubuntu@your_ec2_ip:/path/on/server
scp -i path_to_your_key.pem -r your_django_project ubuntu@your_ec2_ip:/path/on/server
```

### 7. 结论：

至此，您应该已在 EC2 实例上设置好部署脚本和配置。在下一章中，我们将看到如何执行此脚本并让我们的 Django 项目上线！

## 部署到服务器

资产、资源和材料：

- AWS EC2 实例：这是我们部署 Django 项目的虚拟服务器。您可以通过 AWS 管理控制台创建和管理 EC2 实例。（获取方式：注册或登录您的 AWS 账户并导航到 EC2 仪表板）。
- Git：我们将使用 Git 进行版本控制。（获取方式：从 [git-scm.com](https://git-scm.com/) 安装）。
- 虚拟环境：我们需要 Python 虚拟环境来管理项目的依赖项。（获取方式：使用 `pip install virtualenv`）。
- Requirements.txt：包含项目所需所有依赖项列表的文件。这将在我们的项目目录中使用 `pip freeze > requirements.txt` 创建。
- SSH 密钥对：用于连接 EC2 实例的安全密钥。这是在创建 EC2 实例时从 AWS 生成的。

### 引言

将我们的 API 部署到服务器上，尤其是在 AWS 上，是使我们的应用程序可供用户访问的关键一步。在本章中，我们将逐步介绍如何在 AWS EC2 实例上部署我们的 Django REST API。

### 1. 为部署准备你的 Django 项目

在部署之前，请确保以下几点：

- 你的 Django 项目能在本地无错误地运行。
- 在 `settings.py` 中，`DEBUG` 已设置为 `False`。
- 你已将服务器的 IP 地址添加到 `settings.py` 的 `ALLOWED_HOSTS` 中。
- 你已使用以下命令生成了 `requirements.txt`：

```bash
pip freeze > requirements.txt
```

### 2. 启动一个 EC2 实例

1.  登录 AWS 管理控制台并导航到 EC2 仪表板。
2.  点击“启动实例”并选择你偏好的操作系统（在本指南中，我们将使用 Ubuntu 18.04）。
3.  选择一个实例类型（对于新 AWS 用户，在免费套餐内 t2.micro 是免费的）。
4.  点击“下一步”，直到你到达“配置安全组”。在这里，添加一条规则以允许 HTTP（端口 80）和 HTTPS（端口 443）。
5.  启动实例，如果还没有，请创建一个新的密钥对。下载密钥对（这将是一个 .pem 文件）并妥善保管。

### 3. 连接到你的 EC2 实例

1.  导航到你的 EC2 仪表板并找到你实例的“公有 IP”。
2.  打开你的终端并导航到你的 .pem 文件所在的位置。
3.  使用 SSH 连接到你的实例：

```bash
ssh -i “your-key.pem” ubuntu@your-ec2-public-ip
```

### 4. 在 EC2 上设置环境

1.  更新软件包列表并安装必要的软件包：

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y
```

2.  从 GitHub 克隆你的项目或将你的项目文件传输到服务器。
3.  导航到你的项目目录并创建一个虚拟环境：

```bash
python3 -m venv venv
```

4.  激活虚拟环境：

```bash
source venv/bin/activate
```

5.  安装项目依赖项：

```bash
pip install -r requirements.txt
```

### 5. 设置 Web 服务器

我们将使用 Gunicorn 作为我们的应用服务器，Nginx 作为反向代理：

1.  在你的虚拟环境中安装 Gunicorn：

```bash
pip install gunicorn
```

2.  测试 Gunicorn 是否可以服务你的项目：

```bash
gunicorn your_project_name.wsgi:application —bind 0.0.0.0:8000
```

3.  安装 Nginx：

```bash
sudo apt install nginx -y
```

4.  配置 Nginx 将请求转发给 Gunicorn。在 `/etc/nginx/sites-available/your_project_name` 创建一个新配置文件并添加以下内容：

```
server {
    listen 80;
    server_name your-ec2-public-ip;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

5.  在 `sites-enabled` 中为此配置创建一个符号链接并重启 Nginx：

```bash
sudo ln -s /etc/nginx/sites-available/your_project_name /etc/nginx/sites-enabled
sudo nginx -t && sudo systemctl restart nginx
```

### 6. 使用 SSL 保护你的应用程序

1.  在你的 Django 设置中，将 HTTPS 添加到允许的协议中。
2.  安装 Certbot 以获取来自 Let's Encrypt 的免费 SSL 证书：

```bash
sudo apt install certbot python3-certbot-nginx -y
```

3.  请求一个证书：

```bash
sudo certbot —nginx
```

按照提示操作，Certbot 将自动配置 Nginx 以使用 SSL。

### 7. 完成部署

1.  使用 `systemd` 确保 Gunicorn 作为服务运行。这意味着即使服务器重启，它也会自动启动。
2.  通过定期从你的代码仓库拉取更改并根据需要重启 Gunicorn 和 Nginx 来保持你的应用程序更新。

恭喜！你的 Django REST API 现在已在 AWS 上线！请确保监控你的应用程序并定期备份你的服务器和数据库。

## 更新允许的主机并部署更改

本章所需的资产、资源和材料：

- AWS 管理控制台访问权限（获取方式：在 [https://aws.amazon.com/](https://aws.amazon.com/) 注册或登录）
- Python（获取方式：[下载并安装 Python](https://www.python.org/downloads/)）
- 包含你的 REST API 的 Django 项目（如前面章节所构建）
- 代码编辑器（本书中，我们一直使用 Atom）

引言：
当部署 Django 项目时，尤其是部署到面向公众的服务器时，确保你的应用程序设置针对安全性进行正确配置至关重要。其中一个设置就是 `ALLOWED_HOSTS`。在 Django 中，`ALLOWED_HOSTS` 设置是一项安全措施，用于防止 HTTP 主机头攻击。在我们进行实际部署之前，让我们先处理这个设置。

理解 `ALLOWED_HOSTS`：
Django 使用此设置将 `Host` 头与一个字符串列表进行匹配，该列表代表此 Django 站点可以服务的 `主机/域名`。这是一种安全机制，用于防止特定类型的安全漏洞。
在开发环境中，你可能遇到过未设置 `ALLOWED_HOSTS` 并尝试从不同机器或域访问应用程序时出现的错误。对于生产环境，我们需要非常明确地指定哪些主机被允许托管我们的应用程序。

步骤 1：确定你的 AWS EC2 公有 IP 和域名（如果可用）
当你创建一个 AWS EC2 实例时，AWS 会为你提供一个公有 IPv4 地址和一个公有 IPv4 DNS。你需要这些来更新 `ALLOWED_HOSTS` 设置。

1.  导航到 AWS 管理控制台。
2.  在“服务”下拉菜单下，选择“EC2”。
3.  在 EC2 仪表板中，在“实例”下，点击“实例”以查看你的实例列表。
4.  选择你用于 Django 部署的实例。在下方的“描述”选项卡中，记下“IPv4 公有 IP”和“公有 IPv4 DNS”。

步骤 2：在 settings.py 中更新 `ALLOWED_HOSTS`
在 Atom 或你偏好的代码编辑器中打开你的 Django 项目。导航到 `settings.py` 文件，该文件通常位于主应用程序目录中。
找到 `ALLOWED_HOSTS` 设置。默认情况下，它是一个空列表：

```python
ALLOWED_HOSTS = []
```

将其更新为包含你的 AWS EC2 实例的 IP 和 DNS：

```python
ALLOWED_HOSTS = ['your-ec2-ipv4-public-ip', 'your-ec2-public-ipv4-dns']
```

将 `your-ec2-ipv4-public-ip` 和 `your-ec2-public-ipv4-dns` 替换为你在上一步中记下的值。

注意：如果你有一个域名指向此服务器，你也应该将该域名添加到列表中。

### 步骤 3：提交更改

1.  使用你的终端或命令提示符，导航到你的项目目录。
2.  使用以下命令提交你的更改：

```bash
git add .
git commit -m "Updated ALLOWED_HOSTS for AWS deployment."
```

### 步骤 4：将更改部署到 AWS

1.  将你的代码更改推送到你的代码仓库。

```bash
git push origin master
```

2.  SSH 连接到你的 AWS EC2 实例：

```bash
ssh -i path-to-your-aws-key-pair.pem ec2-user@your-ec2-ipv4-public-ip
```

将 `path-to-your-aws-key-pair.pem` 替换为你的 AWS 密钥对的路径，将 `your-ec2-ipv4-public-ip` 替换为你的 EC2 实例的公有 IP。

3.  导航到你的 Django 项目所在的目录并拉取最新的更改：

```bash
cd path-to-your-django-project
git pull origin master
```

4.  重启你的服务器（这可能是 Gunicorn、uWSGI 等）以应用更改。

结论：
通过正确设置 `ALLOWED_HOSTS`，你已采取了确保 Django 应用程序安全性的关键一步。每当你要添加一个新的域名或 IP 来托管你的应用程序时，更新此列表至关重要。请始终保持谨慎，只允许受信任的域名或 IP，以确保你的应用程序的安全。

## ~ 结语

随着我们关于使用 Python 和 Django 构建 REST API 的旅程的最后一章结束，花点时间反思一下你所获得的技能和知识，以及你现在作为后端开发者所拥有的强大能力，这很重要。

回顾：从基础到精通
从基础开始，你了解了后端 REST API 在当今技术中的重要性

从Facebook和Instagram这样的巨头到小型初创公司，强大的后端是许多成功数字平台的基石。随着你深入学习本课程，你已经探索了开发环境的搭建、掌握Django和Django REST框架、理解数据库等诸多内容。

### 关键要点

1.  **开发环境**：通过设置Vagrant、VirtualBox、Atom和ModHeaders等工具，你已经建立了一个坚实的基础，这不仅对本项目有用，也对未来无数的开发工作大有裨益。
2.  **精通Django**：Django与Django REST框架相结合，提供了一套强大的工具集，用于构建健壮且可扩展的Web应用。通过本课程，你已经从创建一个简单的“Hello World”脚本，进阶到将整个API部署到AWS。
3.  **理解数据库**：任何后端系统的核心都是其数据库。你对Django模型的深入研究和对数据库的理解，将成为你职业生涯中的一个关键知识点。
4.  **API与端点**：通过实践方法，你已经创建、测试并迭代了多个API视图和视图集。你处理了用户认证、个人资料管理以及信息流功能——这些都是现代Web应用的核心组成部分。
5.  **部署**：软件开发的重要步骤之一就是部署应用。通过将你的API部署到AWS上线，你已经跨越了一个里程碑，这标志着你从一名爱好者转变为一名专业人士。

### 展望未来：后端开发的前景

虽然你已经取得了许多成就，但后端开发的世界广阔无垠且不断演进。技术会更迭，新的方法论也会涌现。保持好奇心并持续寻求知识至关重要。可以考虑进一步探索：

-   **容器与微服务**：像Docker这样的工具可能是你旅程的下一步。
-   **数据库扩展**：更深入地研究数据库，探索NoSQL选项，并理解扩展策略。
-   **高级安全**：随着你构建更复杂的应用，理解网络安全威胁和缓解措施将变得至关重要。

### 结语

本课程旨在提供使用Django和Python构建REST API的基础知识。你现在已具备设计、开发和部署后端系统的能力，可以将你的应用想法变为现实，或与前端团队高效协作。请记住，学习是一段持续的旅程。技术领域日新月异，作为一名开发者，你的适应能力和好奇心将决定你的成功。

最后，无论你是经验丰富的开发者还是刚刚起步，请记住始终保持热情，坚持构建，最重要的是，永不停止学习。数字世界正等待着你的下一次创新。