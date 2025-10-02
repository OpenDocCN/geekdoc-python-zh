# python 数据结构与算法 2 栈的概念

 <wbr>

## Stacks 栈

## What is a Stack? 什么是栈？

A <wbr>stack <wbr>(sometimescalled a “push-down stack”) is an orderedcollection of items where the addition of new items and the removalof existing items always takes place at the same end. This end iscommonly referred to as the “top.” The end opposite the top isknown as the “base.”

栈（也叫下推栈）一种线性有序的数据元素集合，它的特点是，数据的增加删除操作都在同一端进行。进行操作的这一端，我们一般叫做“顶”，另一端叫做“底”。

The base of the stack is significant since items stored in thestack that are closer to the base represent those that have been inthe stack the longest. The most recently added item is the one thatis in position to be removed first. This ordering principle issometimes called <wbr>LIFO, <wbr>last-infirst-out. It provides an ordering based on lengthof time in the collection. Newer items are near the top, whileolder items are near the base.

栈的底部很有象征性，因为元素越接近底部，就意味着在栈里的时间越长。最近进来的，总是最早被移走，这种排列规律叫做先进后出，综合为 LIFO。所以栈的排序是按时间长短来排列元素的。新来的在栈顶，老家伙们在栈底（译者注：中国有句成语，叫后来居上。原典故是这样说的：陛下用群臣，如积薪耳，后来者居上。栈的方式，可不就是堆柴草吗？）

Many examples of stacks occur in everyday situations. Almost anycafeteria has a stack of trays or plates where you take the one atthe top, uncovering a new tray or plate for the next customer inline. Imagine a stack of books on a desk ([Figure1](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#fig-bookstack)). The only book whose cover is visible is the oneon top. To access others in the stack, we need to remove the onesthat are sitting on top ofthem. <wbr>[Figure2](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#fig-objectstack)shows another stack. This one contains a number ofprimitive Python data objects.

栈的例子经常看到。比如自助餐厅的盘，人们总是从上面拿盘子，拿走一个后面的人再拿下面的一个，（服务员端来一些新的，又堆在上面了）。又如一堆书，（图 1）你只能看到最上面一本的封面，要看下面一本，就要把上面的先拿 <wbr>走。图 2 是另一种栈，存储的是几个主要的 python 语言数据对象。

 <wbr>![](img/a10ed103a1ebd065affdd647b79bdf2d.jpg)

Figure 1: A Stack of Books

![](img/9c198e877adc882f423d1344db0e6fc2.jpg)

Figure 2: A Stack of Primitive Python Objects

One of the most useful ideas related to stacks comes from thesimple observation of items as they are added and then removed.Assume you start out with a clean desktop. Now place books one at atime on top of each other. You are constructing a stack. Considerwhat happens when you begin removing books. The order that they areremoved is exactly the reverse of the order that they were placed.Stacks are fundamentally important, as they can be used to reversethe order of items. The order of insertion is the reverse of theorder of removal. <wbr>[Figure3](http://interactivepython.org/courselib/static/pythonds/BasicDS/stacks.html#fig-reversal) <wbr>shows the Pythondata object stack as it was created and then again as items areremoved. Note the order of the objects.

与栈有关的思想来自生活中的观察，假设你从一张干净的桌子开始，一本一本地放上书，这就是在建立栈。当你一本一本地拿走，想像一下，是不是先进后出？由于这种结构具有翻转顺序的作用，所以非常重要。图 3 显示了 Python 数据栈，加入和移走的顺序。

![](img/8b2d9ba2e0620644afda8f33f22e8866.jpg)

 <wbr>

Figure 3: The Reversal Property of Stacks

Considering this reversal property, you can perhaps think ofexamples of stacks that occur as you use your computer. Forexample, every web browser has a Back button. As you navigate fromweb page to web page, those pages are placed on a stack (actuallyit is the URLs that are going on the stack). The current page thatyou are viewing is on the top and the first page you looked at isat the base. If you click on the Back button, you begin to move inreverse order through the pages.

栈这种翻转性，在你用电脑上网的时候也用到了。浏览器软件上都有“返回”按钮，当你从一个链接到另一个链接，这时网址（URL）就被存进了栈。正在浏览的页就存在栈顶，点“返回”的时候，返回到刚刚浏览的页面。最早浏览的页面，要一直到最后才能看到。

 <wbr>

 <wbr>