# 核心 Python 开发:如何提交补丁

> 原文：<https://www.blog.pythonlibrary.org/2012/05/22/core-python-development-how-to-submit-a-patch/>

正如我在上一篇文章中提到的，我想我应该试着找到一些可以用 Python 修补的东西并提交。在写另一篇文章时，我在 Python devguide 的 [Windows 部分](http://docs.python.org/devguide/setup.html#windows)中发现了一个小错误。虽然修补一个文档远没有我想象的修补 Python 那么酷，但我认为它对我来说相当合适，因为我最近倾向于贡献更多的文档。所以我将解释我发现的过程。

### 入门指南

首先，你需要用 Python 的 [Bug 追踪器](http://bugs.python.org/)获得一个账户。如果您希望成为核心开发人员，那么您需要确保您的用户名遵循他们的指导原则，这非常简单:

 `firstname.lastname` 

一旦你有了这些，你就可以开始寻找修补的东西了。有一个链接写着“简单问题”,这是一个很好的起点。你也可以搜索一个你能胜任使用的组件，看看是否有你认为可以修复的 bug。一旦你发现了什么，你需要确保更新你的本地回购，然后阅读 devguide 的[补丁页面](http://docs.python.org/devguide/patch.html)。

### 创建补丁

假设您已经在本地机器上签出了必要的存储库，那么您需要做的就是编辑适当的文件。在我的例子中，我必须检查 devguide(你可以在这里阅读)并编辑 **setup.rst** 文件。如果你正在编辑 Python 代码，那么你必须遵守 PEP8。编辑完文件后，我保存了我的更改，然后必须使用 Mercurial 来创建补丁。这是我根据 Python [补丁指令](http://docs.python.org/devguide/patch.html)使用的命令。

 `hg diff > setup.patch` 

下面是该补丁文件的内容:

 `diff -r b1c1d15271c0 setup.rst
--- a/setup.rst Tue May 22 00:33:42 2012 +0200
+++ b/setup.rst Tue May 22 13:55:09 2012 -0500
@@ -173,7 +173,7 @@
To build from the Visual Studio GUI, open pcbuild.sln to load the project
files and choose the Build Solution option from the Build menu, often
associated with the F7 key. Make sure you have chosen the "Debug" option from
-the build configuration drop-down first.
+the configuration toolbar drop-down first.`

构建完成后，您可能希望将 Python 设置为一个启动项目。在
Visual Studio 中按 F5，或者从调试菜单中选择开始调试，将启动

现在我们有了一个补丁，我们需要提交它！

### 提交补丁

举起你的盾牌，我们要进去了！提交补丁有点令人生畏。别人会怎么看你？我怀疑如果你打算做一些重大的事情，那么你最好开始厚脸皮。在我的情况下，我将提交一个非常简单的错别字修复，所以我希望这种事情不值得大动干戈。话说回来，这是我的第一个补丁，所以我可能会以一种完全错误的方式提交它。因为我的补丁将会是新的，所以我做了一个快速搜索以确保它没有被报道过。什么也没看到，我战战兢兢地点击了“Create New”链接，并选择“devguide”作为我的组件。我也选择了最新版本的 Python。我在 devguide 中没有看到任何说它只适用于一组 Python 版本的内容，所以我打算就此打住。我没有真正看到适合 devguide 编辑“类型”,所以我把空白留给我的上级来修复。最后，我把我的补丁文件附在了 bug 单上。如果你愿意，你可以在这里看到我的虫票[。](http://bugs.python.org/issue14884)

向 Python 贡献一个补丁时，你应该填写一份[贡献者协议表](http://www.python.org/psf/contrib/)，它允许 Python 软件基金会许可你的代码与 Python 一起使用，而你可以保留版权。是的，你也可以因为写 Python 代码而出名！假设人们阅读了源代码或那些确认页。

### 包扎

我不知道我那相当蹩脚的贡献会怎么样。也许会被接受，也许不会。但是我想我会花一些时间尝试找出一些其他的 bug，看看我能做些什么来帮助 Python 社区。欢迎加入我的冒险之旅！