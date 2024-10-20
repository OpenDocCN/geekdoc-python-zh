# PyCon 2011:周五上午会谈(与 Bicking 和 Beazley)

> 原文：<https://www.blog.pythonlibrary.org/2011/03/12/pycon-2011-friday-morning-talks-with-bicking-and-beazley/>

[![](img/404a7b432bc1cc0abd29b26f7d7351fc.png)](http://us.pycon.org)

我参加了迈克尔·福德的模拟演讲，开始了我的晨会，但是由于我的绿色房间志愿者职位，我不得不提前离开。我们有一些幕后问题需要处理。唉！总之，我最终跳过了大部分内容，但是我提到了另一个 Python 名人:

## Ian Bicking 关于 Javascript 的演讲

Ian Bicking 在 Python 社区和 PyCon 都很受欢迎。今年他在 Pythonistas 的 Javascript 上发言。以下是我从中得到的收获:

*   Javascript 到处都有对象，很像 Python
*   Javascript 有一个类似于 Python 字典的对象，至少从语法上看是这样的
*   他谈到了可变范围，但我没有抓住那张幻灯片的要点
*   **undefined** 是 falshish，不同于 null，不神奇，是一个对象！
*   typeof 有点神奇
*   原型就像 Python 类？
*   **这个**就像是 Python 的**自我**。**这个**即使没用也总有价值
*   Javascript 中的数组糟透了
*   如果你喜欢 Python，你可能会喜欢 CoffeeScript

## 大卫·比兹利的演讲

David Beazley 做了一个关于使用 Python 3 为我的 Superboard II 构建云计算服务的演讲。他也一直在他的博客上谈论这个项目，我觉得这听起来很有趣。他谈到 Superboard II 是他 12 岁时的第一台电脑。如果我的笔记是准确的，它有以下规格:1 兆赫中央处理器，8k 内存，300 波特磁带。

他发现他的父母仍然在他们的地下室里有这个东西，所以他把它拿出来，并试图找出如何处理它。他的主意？使用 Python 将它的程序存储在云中！或者类似的东西。它使用录音带来告诉它该做什么，所以他必须将 pyaudio 移植到 Python 3，然后使用他的 Mac 来模拟这些声音。最终，他出于测试目的编写了 Superboard II 的模拟器(我想)。他还谈到用 Python 3 用 500 行左右的代码编写 6502 汇编程序。

这里的要点是，他必须将大约 6 个库移植到 Python 3(包括 Redis 和 pyPng)。他使用 Redis 创建了他的云，并展示了许多录制的演示，展示了他如何与 Superboard 通信，并最终如何将程序存储在 Redis 云中，甚至将程序从云中恢复到 Superboard 中。总的来说，这个演讲棒极了！我绝对推荐试着找一下这个的视频。