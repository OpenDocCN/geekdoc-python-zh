## Python 标准库 12 数学与随机数 (math 包，random 包)

[`www.cnblogs.com/vamei/archive/2012/10/26/2741702.html`](http://www.cnblogs.com/vamei/archive/2012/10/26/2741702.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

我们已经在[Python 运算](http://www.cnblogs.com/vamei/archive/2012/05/29/2524376.html)中看到 Python 最基本的数学运算功能。此外，math 包补充了更多的函数。当然，如果想要更加高级的数学功能，可以考虑选择标准库之外的 numpy 和 scipy 项目，它们不但支持数组和矩阵运算，还有丰富的数学和物理方程可供使用。 

此外，random 包可以用来生成随机数。随机数不仅可以用于数学用途，还经常被嵌入到算法中，用以提高算法效率，并提高程序的安全性。 

1\. math 包 

math 包主要处理数学相关的运算。math 包定义了两个常数:

math.e   # 自然常数 e

math.pi  # 圆周率 pi

此外，math 包还有各种运算函数 (下面函数的功能可以参考数学手册)： 

math.ceil(x)       # 对 x 向上取整，比如 x=1.2，返回 2 

math.floor(x)      # 对 x 向下取整，比如 x=1.2，返回 1 

math.pow(x,y)      # 指数运算，得到 x 的 y 次方 

math.log(x)        # 对数，默认基底为 e。可以使用 base 参数，来改变对数的基地。比如 math.log(100,base=10) 

math.sqrt(x)       # 平方根 

三角函数: math.sin(x), math.cos(x), math.tan(x), math.asin(x), math.acos(x), math.atan(x) 

这些函数都接收一个弧度(radian)为单位的 x 作为参数。 

角度和弧度互换: math.degrees(x), math.radians(x) 

双曲函数: math.sinh(x), math.cosh(x), math.tanh(x), math.asinh(x), math.acosh(x), math.atanh(x) 

特殊函数： math.erf(x), math.gamma(x)

2\. random 包 

如果你已经了解伪随机数(psudo-random number)的原理，那么你可以使用如下: 

random.seed(x) 

来改变随机数生成器的种子 seed。如果你不了解其原理，你不必特别去设定 seed，Python 会帮你选择 seed。 

1) 随机挑选和排序 

random.choice(seq)   # 从序列的元素中随机挑选一个元素，比如 random.choice(range(10))，从 0 到 9 中随机挑选一个整数。 

random.sample(seq,k) # 从序列中随机挑选 k 个元素

random.shuffle(seq)  # 将序列的所有元素随机排序 

2）随机生成实数

下面生成的实数符合均匀分布(uniform distribution)，意味着某个范围内的每个数字出现的概率相等: 

random.random()          # 随机生成下一个实数，它在[0,1)范围内。 

random.uniform(*a,b*)      # 随机生成下一个实数，它在[a,b]范围内。 

下面生成的实数符合其它的分布 (你可以参考一些统计方面的书籍来了解这些分布): 

random.gauss(*mu,sigma*)    # 随机生成符合高斯分布的随机数，mu,sigma 为高斯分布的两个参数。 

random.expovariate(*lambd*) # 随机生成符合指数分布的随机数，lambd 为指数分布的参数。

此外还有对数分布，正态分布，Pareto 分布，Weibull 分布，可参考下面链接: 

[`docs.python.org/library/random.html`](http://docs.python.org/library/random.html)

假设我们有一群人参加舞蹈比赛，为了公平起见，我们要随机排列他们的出场顺序。我们下面利用 random 包实现： 

```py
import random
all_people = ['Tom', 'Vivian', 'Paul', 'Liya', 'Manu', 'Daniel', 'Shawn']
random.shuffle(all_people) for i,name in enumerate(all_people): print(i,':'+name)

```

 练习:

设计下面两种彩票号码生成器:

1\. 从 1 到 22 中随机抽取 5 个整数 （这 5 个数字不重复）

2\. 随机产生一个 8 位数字，每位数字都可以是 1 到 6 中的任意一个整数。  

总结:

math.floor(), math.sqrt(), math.sin(), math.degrees()

random.random(), random.choice(), random.shuffle()