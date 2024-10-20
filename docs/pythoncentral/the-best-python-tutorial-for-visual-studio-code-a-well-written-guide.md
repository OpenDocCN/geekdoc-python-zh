# Visual Studio 代码的最佳 Python 教程(一本写得很好的指南)

> 原文：<https://www.pythoncentral.io/the-best-python-tutorial-for-visual-studio-code-a-well-written-guide/>

开始您的 Python 之旅可能会非常激动人心，尤其是如果您以前从未安装过 IDE 的话。这个过程非常简单，不需要太多时间就可以完成。

在本指南中，我们将引导您安装 Visual Studio 代码和 Python 开发所需的扩展。我们还概述了如何输入代码、保存文件和运行程序。

## **为 Python 安装和配置 Visual Studio 代码**

你可能听说过微软的 Visual Studio，但是除了名字之外，VS Code 与这个工具没有任何共同之处。由于它内置的对扩展的支持，你可以用 VS 代码编写多种语言的程序。

Windows、Mac、Linux 上都有 VS 代码，IDE 每月更新。

要在您的操作系统上安装该程序，请访问 [官方网站](https://code.visualstudio.com/) ，将鼠标悬停在下载按钮上的箭头上即可找到该工具的稳定版本。

在 Windows 上，运行安装程序并遵循安装说明将完成安装。在 Mac 上，您需要将下载的“Visual Studio Code.app”文件拖到 applications 文件夹中。

要在 Ubuntu 或 Debian 机器上安装 VS 代码，可以在终端中运行以下命令:

*sudo apt 安装。/ <文件>。deb*

### **VS 代码 Python 扩展**

为了使 VS 代码能够与您选择的编程语言一起工作，您需要安装相关的扩展。安装 Python 扩展可以解锁 VS 代码中的以下特性:

*   自动 conda 使用和虚拟环境
*   使用智能感知完成代码
*   Jupyter 环境和 Jupyter 笔记本中的代码编辑
*   代码片段
*   调试支持
*   林挺
*   支持 Python 3.4 及更高版本，以及 Python 2.7
*   单元测试支持

要安装扩展，按 Ctrl+Shift+X 打开扩展视图。您也可以通过单击活动栏上的扩展图标来完成此操作。

搜索“Python ”,点击安装按钮，从 VS 代码市场安装这个扩展。你也可以在这里 找到市集上的分机 [。](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

## **用 VS 代码编写并运行 Python 程序**

单击顶部栏上的文件，然后单击新建，打开一个新文件。您可以使用的快捷键是 Ctrl+n。

现在，您可以开始在新文件中输入代码。但是需要注意的是，VS 代码不会识别你写代码的语言。这是因为它不知道文件类型是什么。

您可以通过保存文件来激活 Python 扩展。“文件”菜单中提供了“保存”选项，但是您可以使用 Ctrl+S。py 扩展名附加在文件上，VS 代码会识别你的代码。

输入代码后，您可以在代码编辑器窗口内单击鼠标右键，然后单击“在终端中运行 Python 文件”来运行代码终端将出现在屏幕底部，显示程序的输出。