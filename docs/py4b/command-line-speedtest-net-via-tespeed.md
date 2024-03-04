# 通过 tespeed 的 speedtest.net 命令行

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/command-line-speedtest-net-via-tespeed>

## 概观

在四处寻找命令行工具来检查我的网速时，
我无意中发现了[lowendtalk.com](https://www.lowendtalk.com/discussion/comment/172526#Comment_172526 "lowendtalk")上的一个帖子。

## 这是什么？

这是 speedtest.net 服务器的命令行界面，可以在这里找到:
[https://github.com/Janhouse/tespeed](https://github.com/Janhouse/tespeed "tespeed")

## 什么是 Speedtest？

Speedtest.net 是一个连接分析网站，用户可以在这里通过分布在世界各地的数百台服务器测试他们的互联网 T2 速度。

在每次测试结束时，用户会看到他们的下载(数据从服务器到他们计算机的速度)和上传(数据从用户计算机发送到服务器的速度)带宽速度。

## 要求

这个脚本需要 lxml 和 argparse，所以确保首先安装它们。

参见此[链接](https://code.activestate.com/pypm/lxml/ "pypm")了解操作方法。

在大多数情况下，从终端运行这个命令就足够了: **pypm install lxml**

## 脚本是如何工作的？

有关如何使用该程序，请参见 this [url](https://github.com/Janhouse/tespeed#readme "tespeed")

## 运行脚本

下载需求后，我决定给这个程序一个机会。

将代码从[复制并粘贴到这里](https://raw.github.com/Janhouse/tespeed/master/tespeed.py "tespeed.py")。

## 运行脚本

打开你的编辑器，复制并粘贴上面的代码，将文件保存为 **speed.py**

要执行该程序，请在 shell 中键入 python 和文件名。
**$python speed.py**

##### 来源

[http://www . Jan house . LV/blog/coding/tes speed-terminal-network-speed-test/](http://www.janhouse.lv/blog/coding/tespeed-terminal-network-speed-test/ "janhouse.lv")
https://github.com/Janhouse/tespeed