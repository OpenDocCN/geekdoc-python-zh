# 模块:os.stat()

> 原文：<https://www.pythonforbeginners.com/os/os-module-os-stat>

## 操作系统统计

要在给定的路径上执行 stat 系统调用，我们可以使用 os 的 os.stat()函数。

首先导入 os 模块，然后简单地指定要在其上执行系统调用的文件的路径。

## 例子

让我们看一个如何使用 os.stat 函数的例子

```py
import os
print "-" * 30
print "os.stat    = status of a file 	" , os.stat('/usr/bin/vi')
print "-" * 30 
```