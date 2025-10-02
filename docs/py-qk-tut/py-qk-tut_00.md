## Python 基础 01 Hello World!

[`www.cnblogs.com/vamei/archive/2012/05/28/2521650.html`](http://www.cnblogs.com/vamei/archive/2012/05/28/2521650.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei…

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

简单的‘Hello World!’

1\. 直接运行 python

假设你已经安装好了 python, 那么在 Linux 命令行输入:

$python

将直接进入 python。然后在命令行提示符>>>后面输入:

>>>print 'Hello World!'

可以看到，python 随后在屏幕上输出:

print 是一个常用的 python 关键字(keyword)，其功能就是输出。

（在 Python 3.x 中，print 的语法会有所变化，作为一个函数使用， 所以上面应写成 print('Hello World!')，以此类推 ）

2\. 写一段小程序

另一个使用 Python 的方法，用文本编辑器写一个.py 结尾的文件，比如说*hello.py*

在*hello.py*中写入如下，并保存:

退出文本编辑器，然后在命令行输入:

$python hello.py

来运行 hello.py。可以看到 python 随后输出

总结：

print

命令行模式: 运行 python，在命令行输入命令并执行。

程序模式: 写一段 python 程序并运行。