# 命令行参数(sys.argv)

> 原文：<https://www.pythonforbeginners.com/argv/more-fun-with-sys-argv>

Python 中的 sys 模块是许多可用的库代码模块之一。

### sys.argv 是什么？

[sys.argv](https://www.pythonforbeginners.com/system/python-sys-argv) 是传递给 Python 程序的命令行参数列表。

argv 表示通过命令行输入的所有项目，它基本上是一个数组，保存我们程序的命令行参数。

不要忘记计数是从零(0)而不是一(1)开始的。

### 我如何使用它？

要使用它，您必须首先导入它(导入系统)

第一个参数 sys.argv[0]始终是程序被调用时的名称，
，sys.argv[1]是传递给程序的第一个参数。

通常，您会对列表进行切片以访问实际的命令行参数:

```py
import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

```

这是一个如何从命令行读取参数的例子

```py
import sys
for x in sys.argv:
     print "Argument: ", x

```

```py
 len(sys.argv) , checks how many arguments that have been entered. 

len(sys.argv) != 2 just checks whether you entered at least two elements 
```

```py
import sys
if len (sys.argv) != 2 :
    print "Usage: python ex.py "
    sys.exit (1)

```

要运行它，只需输入:

```py
 >python ex.py
Argument:  ex.py

>python ex.py hello
Argument:  ex.py
Argument:  hello

>python ex.py hello world
Argument:  ex.py
Argument:  hello
Argument:  world 
```