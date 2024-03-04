# 操作系统模块:概述

> 原文：<https://www.pythonforbeginners.com/os/os-module-overview>

## 操作系统模块

Python 中的 OS 模块提供了一种使用操作系统相关功能的方式。

os 模块还提供查找关于您的位置或过程的重要信息的功能。在这篇文章中，我将展示其中的一些功能。

## 操作系统功能

```py
 import os

os.system()		Executing a shell command
os.environ()		Get the users environment
os.getcwd()	 	Returns the current working directory.
os.getgid()		Return the real group id of the current process. 
os.getuid() 		Return the current process’s user id.
os.getpid() 		Returns the real process ID of the current process.
os.uname()		Return information identifying the current OS.
os.chroot(path)		Change the root directory of the current process to path
os.listdir(path)	Return a list of the entries in the directory by path.
os.mkdir(path)		Create a directory named path with numeric mode mode
os.makedirs(path)	Recursive directory creation function
os.remove(path)		Remove (delete) the file path
os.removedirs(path)	Remove directories recursively
os.rename(src, dst)	Rename the file or directory src to dst
os.rmdir(path)		Remove (delete) the directory path 
```

关于 OS 模块的更多用法，请参见官方的 [Python 文档](https://docs.python.org/2/library/os.html "os")。