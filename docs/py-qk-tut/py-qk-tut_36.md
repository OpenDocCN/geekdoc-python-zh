## Python 标准库 05 存储对象 (pickle 包，cPickle 包)

[`www.cnblogs.com/vamei/archive/2012/09/15/2684781.html`](http://www.cnblogs.com/vamei/archive/2012/09/15/2684781.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

谢谢[reverland](http://home.cnblogs.com/u/469140/)纠错

在之前对 Python 对象的介绍中 ([面向对象的基本概念](http://www.cnblogs.com/vamei/archive/2012/06/02/2531515.html)，[面向对象的进一步拓展](http://www.cnblogs.com/vamei/archive/2012/06/02/2532018.html))，我提到过 Python“一切皆对象”的哲学，在 Python 中，无论是变量还是函数，都是一个对象。当 Python 运行时，对象存储在内存中，随时等待系统的调用。然而，内存里的数据会随着计算机关机和消失，如何将对象保存到文件，并储存在硬盘上呢？

计算机的内存中存储的是二进制的序列 (当然，在 Linux 眼中，是[文本流](http://www.cnblogs.com/vamei/archive/2012/09/14/2683756.html))。我们可以直接将某个对象所对应位置的数据抓取下来，转换成文本流 (这个过程叫做 serialize)，然后将文本流存入到文件中。由于 Python 在创建对象时，要参考对象的类定义，所以当我们从文本中读取对象时，必须在手边要有该对象的类定义，才能懂得如何去重建这一对象。从文件读取时，对于 Python 的内建(built-in)对象 (比如说整数、词典、表等等)，由于其类定义已经载入内存，所以不需要我们再在程序中定义类。但对于用户自行定义的对象，就必须要先定义类，然后才能从文件中载入对象 (比如[面向对象的基本概念](http://www.cnblogs.com/vamei/archive/2012/06/02/2531515.html)中的对象那个 summer)。

1\. pickle 包
对于上述过程，最常用的工具是 Python 中的 pickle 包。

1) 将内存中的对象转换成为文本流：

```py
import pickle # define class
class Bird(object):
    have_feather = True
    way_of_reproduction = 'egg'

summer       = Bird()                 # construct an object
picklestring = pickle.dumps(summer)   # serialize object

```

使用 pickle.dumps()方法可以将对象 summer 转换成了字符串 picklestring(也就是文本流)。随后我们可以用普通文本的存储方法来将该字符串储存在文件([文本文件的输入输出](http://www.cnblogs.com/vamei/archive/2012/06/06/2537868.html))。

当然，我们也可以使用 pickle.dump()的方法，将上面两部合二为一:

```py
 define class
class Bird(object):
    have_feather = True
    way_of_reproduction = 'egg' summer = Bird()                        # construct an object
fn           = 'a.pkl' with open(fn, 'w') as f:                     # open file with write-mode
    picklestring = pickle.dump(summer, f)   # serialize and save object

```

对象 summer 存储在文件 a.pkl

2) 重建对象

首先，我们要从文本中读出文本，存储到字符串 ([文本文件的输入输出](http://www.cnblogs.com/vamei/archive/2012/06/06/2537868.html))。然后使用 pickle.loads(str)的方法，将字符串转换成为对象。要记得，此时我们的程序中必须已经有了该对象的类定义。

此外，我们也可以使用 pickle.load()的方法，将上面步骤合并:

```py
import pickle # define the class before unpickle
class Bird(object):
    have_feather = True
    way_of_reproduction = 'egg' fn = 'a.pkl' with open(fn, 'r') as f:
    summer = pickle.load(f)   # read file and build object

```

2\. cPickle 包

cPickle 包的功能和用法与 pickle 包几乎完全相同 (其存在差别的地方实际上很少用到)，不同在于 cPickle 是基于 c 语言编写的，速度是 pickle 包的 1000 倍。对于上面的例子，如果想使用 cPickle 包，我们都可以将 import 语句改为:

就不需要再做任何改动了。

总结:

对象 -> 文本 -> 文件

pickle.dump(), pickle.load(), cPickle