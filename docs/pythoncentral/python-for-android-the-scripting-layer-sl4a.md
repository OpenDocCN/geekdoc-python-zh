# Python for Android:脚本层(SL4A)

> 原文：<https://www.pythoncentral.io/python-for-android-the-scripting-layer-sl4a/>

Android 的脚本层 SL4A 是一个开源应用程序，允许用一系列解释语言编写的程序在 Android 上运行。它还提供了一个高级 API，允许这些程序与 Android 设备进行交互，使访问传感器数据、发送短信、渲染用户界面等事情变得容易。

它真的很容易安装，可以在任何安卓设备上运行，所以你不需要 root 或者类似的权限。

目前脚本层支持 Python、Perl、Ruby、Lua、BeanShell、JavaScript 和 Tcl。它还提供对 Android 系统外壳的访问，这实际上只是一个最小的 Linux 外壳。

你可以从他们的网站上找到更多关于 SL4A 项目的信息。

## **为什么安卓的 SL4A 不一样**

在 Android 上运行 Python 还有其他一些选择，其中一些非常好，但是没有一个提供脚本层的灵活性和特性。备选方案真正关注的是让您能够使用一些不受支持的语言创建和打包本机应用程序，有些做得非常好。例如，使用 [Kivy](https://kivy.org/#home "Kivy") ，你可以用 Python 创建一个应用程序，它可以运行在许多流行的操作系统、桌面和智能手机上，包括 Android。然而，因为它是多平台的，你无法直接访问 Android API，所以你无法使用许多让智能手机如此有趣的功能。

SL4A 是围绕 Android 操作系统设计的:它要求 Android 有用，但允许与操作系统更紧密地集成。

使用 SL4A，你仍然可以打包你的应用，如果你喜欢，你可以将它们发布到 Play 等应用商店，但这只是一个选项。将打包作为一个选项而不是一个目的的一个优点是，大多数 Python 脚本实际上都没有通过应用商店发布。它们应该被用作常规的 Python 程序。你通常只想写一点代码，保存它，然后运行它，并不断迭代。不得不不断构建应用程序是非常乏味的。

有了脚本层，您可以像在任何其他系统上一样开始工作，只是编辑和执行文件。

本系列主要关注 Python，Python 是 SL4A 上最受欢迎和最受支持的语言，但是其他语言也有非常有用的特性。例如，BeanShell 是一种编译为 Java 的高级语言，它能够绕过脚本层 API，直接访问 Android Java API。脚本层的 Ruby 和 JavaScript 解释器 JRuby 和 Rhino 也运行在 JVM 上，所以这些语言也可以这样做。有一个具备这些特性的环境是很好的。

### **投入:做好准备**

SL4A 真的很好装。该应用程序以 APK 的形式发布，这是 Android 应用程序的标准格式，因此它可以以同样的方式安装。但是，在安装来自“未知来源”的应用程序之前，您需要在您的设备上允许这样做。如果您尚未安装，请打开设备的主设置菜单，打开安全菜单，然后通过选中未知来源选项“允许安装非市场应用程序”。现在您已经准备好安装脚本层了。

### **脚本层(SL4A)**

如果您转到 SL4A 项目的主页，只需扫描页面上的条形码并在出现提示时确认下载，即可将脚本层 APK 的副本下载到您的设备上。如果你的设备上没有条形码扫描仪，任何应用商店都有一堆免费的扫描仪。

下载完 APK 后，您应该能够直接在设备的通知面板中安装它，下载内容将出现在通知面板中。您的设备可能略有不同，但只要您启用了从未知来源安装，就可以清楚地知道如何安装 APK。

一旦您安装了脚本层，您将能够打开它，用内置编辑器创建和编辑小 shell 脚本并运行它们。开始非常容易。

每当您第一次打开 SL4A 时，您会看到您的脚本目录的内容，它位于`/sdcard/sl4a/scripts`。这是您通常放置自己的脚本以便于访问的地方。如果你想构建更复杂的应用程序，或者如果你有很多简单的脚本，你可以在这里创建目录来帮助保持事情正常。

### **SL4A 解释器**

SL4A 只将 shell 作为标准配置，但是在应用程序中安装其他解释器很容易。如果您打开 SL4A，然后点击您的设备主菜单按钮，SL4A 菜单将弹出。如果你按下 View，你会看到一个有三个选项的菜单，解释器，触发器和 Logcat。选择解释器会将您的视图从脚本目录移动到解释器列表。在此视图中点击设备的主菜单按钮将打开一个新菜单，其中有一个添加翻译的选项。在这里，您可以选择要安装的解释器，这将打开您的浏览器并为该解释器下载 APK。按照安装 SL4A 的相同方式安装该 APK。

每个解释器都作为一个独立的 Android 应用程序存在，并将作为一个应用程序出现在设备的菜单中。Python 解释器的应用叫做 Python for Android，或者简称 PY4A。每个解释器的应用程序至少有安装或卸载解释器的能力。PY4A 也可以管理。egg 文件，这为您提供了一种在脚本层安装 Python C 模块的简单方法，否则您只能使用纯 Python。任何 Python C 模块都必须首先针对 ARM 内核进行交叉编译，这是一个复杂的过程，但一些有用的包[是从 Python for Android 项目页面预编译的](https://code.google.com/archive/p/python-for-android/wikis/Modules.wiki)，以及关于如何编译其他包的[指令](https://code.google.com/archive/p/python-for-android/wikis/BuildingModules.wiki)。

如果你还没有，打开 Python for Android 应用程序，点击安装按钮。您的脚本层现在支持 Python。

### **SL4A 上的 Python 库**

如果你想添加新的模块来扩展设备的 Python 库，你必须做一些与你习惯的不同的事情。没有 root，你不能直接修改系统文件，所以脚本层在`/sdcard/com.googlecode.pythonforandroid/extras/python`有自己的包目录，这些包在你的 sdcard 上。您可以随时从该目录导入，对于您安装的每种语言，都存在类似的目录。

Python extras 目录预装了大量有用的模块，您可以将任何您喜欢的纯 Python 模块放在这里。

如果您将一个模块添加到 extras 目录，请确保它是可导入的 Python 模块本身。当你从存储库中获取一个库时，它通常会被构造成这样，应用程序的根目录包含一堆东西，如 READMEs、docs、tests 和 setup 文件，以及你需要的实际模块，可能是文件或目录。请记住，它必须是纯 Python，您必须解决任何依赖关系。

*注意*:如果你的设备上没有像样的文件浏览器，那就马上去抢一个。我通常使用免费的 [ES 文件浏览器](https://es-file-explorer.en.uptodown.com/android)。如果你想认真研究黑客机器人，最好也早点在开发盒上安装 Android SDK。它包括一些工具，可以轻松完成一些常见的任务。这是一个相当复杂的工具包，但每个操作系统的指南都可以在网上找到，你将需要它来打包应用程序。

### **Android 的 SL4A Hello World:你的第一个脚本和 API**

每当您添加一个新的解释器时，就会在脚本目录中自动安装一些示例脚本。这些是开始寻找代码示例的好地方，一旦完成，您可以安全地删除它们。

在 Python 中使用 SL4A 真的很简单。大多数脚本以下面两行开始:

```py

from android import Android

droid = Android()

```

从这里开始，`droid`这个名字就是一个`Android`对象，并作为 Android 设备的挂钩。您可以使用它来访问整个脚本层 API。API 被划分为称为 *facades* 的部分，每个 facades 覆盖 API 的一些区域，比如网络视图、电话、WiFi、事件、相机、电池等等。

例如，您可以使用文本到语音外观让 droid 说话:

```py

droid.ttsSpeak('Hello World')

```

*注意* : Tasker 还支持 SL4A 脚本，因此您可以使用它来监控系统，并在满足特定条件时启动 Python 程序。

### **狂野西部:开放平台，无文档**

原始 API 在网上有完整的文档记录。然而，关于如何使用它的信息很少。SL4A 和 PY4A 项目网站上有更多的信息，但还远远不够。就我个人而言，我认为开源开发者足够聪明，知道永远不要写文档~这样，他们可以通过写书谋生...

Paul Ferrill 所著的 Apress book[Pro Android Scripting with SL4A 是学习正确使用脚本层的极好资源，并且使用 Python 作为示例语言。Apress 还发布了由 SL4A 项目的首席开发人员 Robbie Matthews](http://www.amazon.com/Android-Python-SL4A-Paul-Ferrill/dp/1430235691 "Pro Android Python with SL4A") 撰写的[开始 Android 平板电脑编程。这本书更侧重于平板电脑，而不是 Python，但对于真正了解其主题的人来说，这仍然是一本很好的书。](http://www.amazon.com/Beginning-Android-Tablet-Programming-Apress/dp/143023783X "Beginning Android Tablet Programming (Beginning Apress)")

还有 SL4A 和 PY4A 谷歌小组可以寻求帮助。他们可能会有点孤独，但是如果你把你的问题写得很好，你通常很快就会得到一个有用的回复。那里也有大量的存档材料。

这个 Python 中心系列旨在涵盖 Python 程序员需要了解的关于 Android 和脚本层的所有最重要的事情，以帮助您快速上手。而且，一如既往，我们真的很感谢社区对我们项目的投入，所以如果有什么你想看的，一定要让我们知道。