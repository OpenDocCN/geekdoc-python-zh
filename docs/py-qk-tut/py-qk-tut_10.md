## Python 进阶 01 词典

[`www.cnblogs.com/vamei/archive/2012/06/06/2537436.html`](http://www.cnblogs.com/vamei/archive/2012/06/06/2537436.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

通过我们的基础教程，我们已经对 Python 建立了基本概念，也对对象和类有一个相对明确的认识。

我们的进阶教程就是对基础教程的进一步拓展，进一步了解 Python 的细节。希望在进阶教程之后，你可以对 Python 的基本语法有一个相对全面的认识。

之前我们说了，表是 Python 里的一个类。一个特定的表，比如说 nl = [1,3,8]，就是这个类的一个对象。我们可以调用这个对象的一些方法，比如 nl.append(15)。
现在，我们要介绍一个新的类，就是词典 (dictionary)。与表相类似，词典也可以储存多个元素。这种可以用来储存多个元素的对象统称为容器(container)。

1\. 基本概念 

常见的创建词典的方法:

>>>dic = {'tom':11, 'sam':57,'lily':100} 

>>>print type(dic) 

词典和表类似的地方，是包含有多个元素，每个元素以逗号分隔。但词典的元素包含有两部分，键和值，常见的是以字符串来表示键，也可以使用数字或者真值来表示键（不可变的对象可以作为键）。值可以是任意对象。键和值两者一一对应。 

（实际上，表的元素也可以是任意对象） 

比如上面的例子中，‘tom’对应 11，'sam 对应 57，'lily'对应 100 

与表不同的是，词典的元素没有顺序。你不能通过下标引用元素。词典是通过键来引用。 

>>>print dic['tom'] 

>>>dic['tom'] = 30 

>>>print dic 

可以构建一个新的空的词典： 

>>>dic = {} 

>>>print dic 

在词典中增添一个新元素的方法： 

>>>dic['lilei'] = 99 

>>>print dic 

（引用一个新的键，赋予对应的值） 

2\. 对 dictionary 的元素进行循环调用： 

```py
dic = {'lilei': 90, 'lily': 100, 'sam': 57, 'tom': 90} for key in dic: print dic[key]

```

可以看到，在循环中，dict 的一个键会提取出来赋予给 key 变量。

通过 print 的结果，我们可以再次确认，dic 中的元素是没有顺序的。

3\. 词典的其它常用方法

>>>print dic.keys()           # 返回 dic 所有的键

>>>print dic.values()         # 返回 dic 所有的值

>>>print dic.items()          # 返回 dic 所有的元素（键值对）

>>>dic.clear()                # 清空 dic，dict 变为{}

另外有一个很常用的用法：

>>>del dic['tom']             # 删除 dic 的‘tom’元素

del 是 Python 中保留的关键字，用于删除对象。

与表类似，你可以用 len()查询词典中的元素总数。

>>>print(len(dic))

总结：

词典的每个元素是键值对。元素没有顺序。

dic = {'tom':11, 'sam':57,'lily':100}

dic['tom'] = 99 

for key in dic: ... 

del, len()