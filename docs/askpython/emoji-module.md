# Python:表情符号模块

> 原文：<https://www.askpython.com/python-modules/emoji-module>

嘿，同学们！今天我们将学习一些有趣的东西！我们将看看 python 中的表情模块。

所以让我们开始吧！

## 表情符号简介

在当今世界，人们通过表情符号来交流情感，而不是打出很长的段落。表情符号已经成为我们日常交流的一个主要部分。

今天在这个教程中，我将教你如何使用简单的代码行和 python 的表情符号模块自己打印表情符号。

## 使用表情符号模块打印自己的表情符号

我们首先从导入`emoji`模块开始。如果导入时出现任何错误，我们需要在命令提示符下使用`pip`命令来安装模块。

为了在屏幕上打印表情符号，我们将使用`emojize()`函数，该函数将冒号(:)内的表情符号名称作为参数。

该功能会自动返回表情符号作为结果。

如果出于某种原因，你不知道某个特定表情符号的文本，我们可以使用`demojize()`函数，并将表情符号作为参数传递。

相同的代码如下所示:

```
import emoji
print(emoji.demojize('😃'))
print(emoji.emojize("Hello there friend! :grinning_face_with_big_eyes:"))

```py

代码的输出如下所示。你可以看到第一个函数将表情符号转换为文本，而第二个函数将文本转换为表情符号。

```
:grinning_face_with_big_eyes:
Hello there friend! 😃

```py

## 什么是 Unicodes？

我们可以直接使用`unicode`作为表情符号，而不是使用长文本。每个表情符号都有一个唯一的对应于该表情符号的 Unicode。

你可以从这个网站获得任何表情符号[的统一码。](https://unicode-table.com/en/sets/emoji/)我们需要做的就是用`000`替换 Unicode 中的`+`来获得正确的 Unicode。

使用 Unicode 打印表情符号非常简单，除了在 Unicode 前加上一个反斜杠(`\`)的`print`语句之外，不需要任何函数就可以打印表情符号。

下面的代码显示了在 unicodes 的帮助下打印一堆表情符号。

```
print("\U0001F680 \U0001F649 \U0001F698 \U0001F6C1")

```py

上面代码的输出如下:

```
🚀 🙉 🚘 🛁

```

## 结论

所以今天我们学会了自己在句子中打印表情符号。您可以使用我们文本中有趣的表情符号来构建有趣的[文本游戏](https://www.askpython.com/python/examples/easy-games-in-python)！希望你喜欢这个教程！

坚持读书学习！谢谢你。