## Python 基础 06 循环

[`www.cnblogs.com/vamei/archive/2012/05/30/2526357.html`](http://www.cnblogs.com/vamei/archive/2012/05/30/2526357.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

从上一讲的选择结构，我们已经看到了如何用缩进来表示隶属关系。循环也会用到类似的表示方法。

1\. for 循环

for 循环需要预先设定好循环的次数(n)，然后执行隶属于 for 的语句 n 次。

基本构造是

举例来说，我们编辑一个叫 forDemo.py 的文件

```py
for a in [3,4.4,'life']: print a

```

这个循环就是每次从表[3,4.4,'life'] 中取出一个元素（回忆：表是一种序列），然后将这个元素赋值给 a，之后执行隶属于 for 的操作(print)。

介绍一个新的 python 函数 range()，来帮助你建立表。

可以看到 idx 是[0,1,2,3,4]

这个函数的功能是新建一个表。这个表的元素都是整数，从 0 开始，下一个元素比前一个大 1， 直到函数中所写的上限 （不包括该上限本身）

(关于 range()，还有丰富用法，有兴趣可以查阅， python 3 中， range()用法有变化，见评论区)

```py
for a in range(10): print a**2

```

2\. while 循环

while 的用法是

while 会不停地循环执行隶属于它的语句，直到条件为假(False)
举例

```py
while i < 10: print i
    i = i + 1

```

3\. 中断循环

(定义一个环的说法。循环是相同的一组操作重复多次，我们把其中的一组操作叫做一环)

continue   # 在同一循环的某一环，如果遇到 continue, 那么跳过这一环，进行下一次环的操作

break      # 停止执行整个循环

```py
for i in range(10): if i == 2: 
 continue
    print i

```

当循环执行到 i = 2 的时候，if 条件成立，触发 continue, 跳过本环(不执行 print)，继续进行下一环(i = 3)

```py
for i in range(10): if i == 2: break
    print i

```

当循环执行到 i = 2 的时候，if 条件成立，触发 break, 循环停止执行。

range() 

for 元素 in 序列:

while 条件:

continue

break