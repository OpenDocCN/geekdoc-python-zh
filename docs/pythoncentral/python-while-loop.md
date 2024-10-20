# Python While 循环

> 原文：<https://www.pythoncentral.io/python-while-loop/>

Python 中的 while 循环基本上只是一种构造代码的方法，当某个表达式为真时，代码会不断重复*。要创建一个 while 循环，您需要一个目标语句和一个条件，目标语句是只要条件为真就会一直执行的代码。*

while 循环的语法如下所示:

而**条件
目标声明**

然而，理解 while 循环的最好方法是看它在上下文中做了什么。在下面的例子中看看它是如何工作的:

```py
count = 0
while (count < 4):
   print count
   count = count + 1

print "Bye!"
```

因此，上面 while 循环的输出应该如下所示:

```py
0
1
2
3
Bye!
```

Count 从 0 开始，每执行一次循环，计数就按代码的指示增加 1(count = count+1)。由于这种情况，只有在 count 小于 4 时才会执行循环。所以在第四次执行后，count 变成了 4，循环被打破。然后下一行代码(打印“拜拜！”)可以执行。