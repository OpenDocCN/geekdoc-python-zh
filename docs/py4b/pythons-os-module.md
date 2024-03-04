# Python 的操作系统模块

> 原文：<https://www.pythonforbeginners.com/os/pythons-os-module>

## 概观

Python 中的 OS 模块提供了一种使用操作系统相关功能的方式。

OS 模块提供的功能允许您与运行 Python 的底层操作系统进行交互——无论是 Windows、Mac 还是 Linux。

您可以找到关于您所在位置或流程的重要信息。在这篇文章中，我将展示其中的一些功能。

## 操作系统功能

```py
 **import os**

Executing a shell command
**os.system()**    

Get the users environment 
**os.environ()**   

#Returns the current working directory.
**os.getcwd()**   

Return the real group id of the current process.
**os.getgid()**       

Return the current process’s user id.
**os.getuid()**    

Returns the real process ID of the current process.
**os.getpid()**     

Set the current numeric umask and return the previous umask.
**os.umask(mask)**   

Return information identifying the current operating system.
**os.uname()**     

Change the root directory of the current process to path.
**os.chroot(path)**   

Return a list of the entries in the directory given by path.
**os.listdir(path)** 

Create a directory named path with numeric mode mode.
**os.mkdir(path)**    

Recursive directory creation function.
**os.makedirs(path)**  

Remove (delete) the file path.
**os.remove(path)**    

Remove directories recursively.
**os.removedirs(path)** 

Rename the file or directory src to dst.
**os.rename(src, dst)**  

Remove (delete) the directory path.
**os.rmdir(path)** 
```