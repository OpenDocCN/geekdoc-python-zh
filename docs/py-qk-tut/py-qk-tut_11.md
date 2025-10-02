## Python 进阶 02 文本文件的输入输出

[`www.cnblogs.com/vamei/archive/2012/06/06/2537868.html`](http://www.cnblogs.com/vamei/archive/2012/06/06/2537868.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

Python 具有基本的文本文件读写功能。Python 的标准库提供有更丰富的读写功能。

文本文件的读写主要通过 open()所构建的文件对象来实现。

1\. 打开文件，创建文件对象。

f = open(文件名，模式)

最常用的模式有：

"r"     # 只读

“w”     # 写入

2\. 文件对象的方法：

读取方法：

content = f.read(N)          # 读取 N bytes 的数据

content = f.readline()       # 读取一行

content = f.readlines()      # 读取所有行，储存在表中，每个元素是一行。

写入方法：

f.write('I like apple')      # 将'I like apple'写入文件

f.write(list)                # 将一个包含有多个字符串的表写入文件，每个元素成为文件中的一行。

关闭文件：

f.close()

3\. 循环读入文件：

```py
for line in file(文件名): print line

```

利用 file()函数，我们创建了一个循环对象。在循环中，文件的每一行依次被读取，赋予给 line 变量。

总结：

f    = open(name, "r")

line = f.readline()

f.write('abc')

f.close()

for line in file(name): ...