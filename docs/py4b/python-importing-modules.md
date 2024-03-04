# 如何在 Python 中导入模块

> 原文：<https://www.pythonforbeginners.com/basics/python-importing-modules>

### 模块

Python 模块使得编程更加容易。

它基本上是一个由已经编写好代码组成的文件。

当 Python 导入一个模块时，它首先检查模块注册表(sys.modules)
以查看该模块是否已经被导入。

如果是这种情况，Python 将使用现有的模块对象。

### 导入模块

导入模块有不同的方法

```py
import sys    		
# access module, after this you can use sys.name to refer to
# things defined in module sys.

from sys import stdout  
# access module without qualifying name. 
# This reads >> from the module "sys" import "stdout", so that
# we would be able to refer "stdout" in our program.

from sys import *       
# access all functions/classes in the sys module. 
```