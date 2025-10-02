## Python 进阶 05 循环设计

[`www.cnblogs.com/vamei/archive/2012/07/09/2582435.html`](http://www.cnblogs.com/vamei/archive/2012/07/09/2582435.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

之前在“循环”一节，我们已经讨论了 Python 最基本的循环语法。这一节，我们将接触更加灵活的循环方式。

1\. 利用 range(), 得到下标

在 Python 中，for 循环后的 in 跟随一个序列的话，循环每次使用的序列元素，而不是序列的下标。

之前我们已经使用过 range 来控制 for 循环。现在，我们继续开发 range 的功能，以实现下标对循环的控制：

```py
S = ‘abcdefghijk’ for i in range(0,len(S),2): print S[i]

```

在该例子中，我们利用 len()函数和 range()函数，用 i 作为 S 序列的下标来控制循环。在 range 函数中，分别定义上限，下限和每次循环的步长。这就和 C 语言中的 for 循环相类似了。

2\. 利用 enumerate(), 同时得到下标和元素

利用 enumerate()函数，可以在每次循环中同时得到下标和元素：

```py
S = 'abcdefghijk'
for (index,char) in enumerate(S): print index print char

```

实际上，enumerate()在每次循环中，返回的是一个包含两个元素的定值表(tuple)，两个元素分别赋予 index 和 char

3\. 利用 zip(), 实现并行循环

如果你多个等长的序列，然后想要每次循环时从各个序列分别取出一个元素，可以利用 zip()方便地实现：

```py
ta = [1,2,3]
tb = [9,8,7]
tc = ['a','b','c'] for (a,b,c) in zip(ta,tb,tc): print a,b,c

```

实际上，zip()在每次循环时，从各个序列分别从左到右取出一个元素，合并成一个 tuple，然后 tuple 的元素赋予给 a,b,c

总结：

range()

enumerate()

zip()