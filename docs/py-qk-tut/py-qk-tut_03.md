## Python 基础 04 运算

[`www.cnblogs.com/vamei/archive/2012/05/29/2524376.html`](http://www.cnblogs.com/vamei/archive/2012/05/29/2524376.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

Python 的运算符和其他语言类似

（我们暂时只了解这些运算符的基本用法，方便我们展开后面的内容，高级应用暂时不介绍）

1\. 数学运算

>>>print 1+9        # 加法

>>>print 1.3-4      # 减法

>>>print 3*5        # 乘法

>>>print 4.5/1.5    # 除法

>>>print 3**2       # 乘方     

>>>print 10%3       # 求余数

2.  判断

判断是真还是假，返回 True/False

>>>print 5==6               # =， 相等

>>>print 8.0!=8.0           # !=, 不等

>>>print 3<3, 3<=3          # <, 小于; <=, 小于等于

>>>print 4>5, 4>=0          # >, 大于; >=, 大于等于

>>>print 5 in [1,3,5]       # 5 是 list [1,3,5]的一个元素

（还有 is, is not 等, 暂时不深入）

3\. 逻辑运算

True/False 之间的运算

>>>print True and True, True and False      # and, “与”运算， 两者都为真才是真

>>>print True or False                      # or, "或"运算， 其中之一为真即为真

>>>print not True                           # not, “非”运算， 取反

可以和上一部分结合做一些练习，比如：

>>>print 5==6 or 3>=3

总结：

数学 +, -, *, /, **, %

判断 ==, !=, >, >=, <, <=, in

逻辑 and, or, not