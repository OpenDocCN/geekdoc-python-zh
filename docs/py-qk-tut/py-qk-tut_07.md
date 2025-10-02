## Python 基础 08 面向对象的基本概念

[`www.cnblogs.com/vamei/archive/2012/06/02/2531515.html`](http://www.cnblogs.com/vamei/archive/2012/06/02/2531515.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

谢谢逆水寒龙,topmad 和 Liqing 纠错

（面向对象并不难，不要被“面向对象”吓跑）

Python 中通过使用类(class)和对象(object)来实现面向对象（object-oriented programming，简称 OOP）的编程。

面向对象编程的最主要目的是提高程序的重复使用性，这和函数的目的相类似。

我们这么早切入面向对象编程的原因是，Python 的整个概念是基于对象的。了解 OOP 对于我们深入了解 Python 很关键。

下面是我对面向对象的理解。

1\. 类是属性相近的对象的归类

在人类认知中，会根据属性相近把东西归类，并且给类别命名。比如说，鸟类的共同属性是有羽毛，通过产卵生育后代。任何一只特别的鸟都在鸟类的原型基础上的。

面向对象就是模拟了以上人类认知过程。在 Python 语言，为了听起来酷，我们把上面说的“东西”称为对象（object）。

先定义鸟类

```py
class Bird(object):
    have_feather = True
    way_of_reproduction = 'egg'

```

我们定义了一个类别（class），就是鸟（Bird）。在隶属于这个类比的语句块中，我们定义了两个变量，一个是有羽毛（have_feather），一个是生殖方式（way_of_reproduction）,这两个变量对应我们刚才说的属性（attribute）。我们暂时先不说明括号以及其中的内容，记为问题 1。

假设我养了一只小鸡，叫 summer。它是个对象，属于鸟类。使用前面定义的类。

```py
summer = Bird() print summer.way_of_reproduction

```

通过第一句创建对象，并说明 summer 是类别鸟中的一个对象，summer 就有了鸟的类属性，对属性的引用是通过 对象.属性（object.attribute） 的形式实现的。

（可怜的 summer，你就是个有毛产蛋的东西，好不精致）

2\. 属性可以是变量，也可以是动作（方法）。

在人类日常认知中，我们在通过属性识别类别的时候，有时候会根据这个东西能做什么事情来区分类别。比如说，鸟会移动 （这样就和房屋的类别区分开了）。而这些动作又会带来一定的结果，通过移动会带来位置的变化。

为了酷起见，我们叫这样的一些属性为方法（method）。Python 中通过在类的内部定义函数，来说明方法。

```py
class Bird(object):
 have_feather = True
    way_of_reproduction = 'egg' def move(self, dx, dy):
        position = [0,0]
        position[0] = position[0] + dx
        position[1] = position[1] +

```

dy
return position

summer = Bird()

```py
print 'after move:',summer.move(5,8)

```

我们重新定义了鸟这个类别。

鸟新增一个方法属性，就是移动（函数 move）。（我承认这个方法很傻，你可以在看过下一讲之后定义个有趣些的方法）

（它的参数中有一个 self，它是为了方便我们引用对象自身。方法的第一个参数必须是 self，无论是否用到。有关 self 的内容会在下一讲展开）

另外两个参数，dx, dy 表示在 x、y 两个方向移动的距离。move 方法会最终返回运算过的 position。

在最后调用 move 方法的时候，我们只传递了 dx 和 dy 两个参数，不需要传递 self 参数（因为 self 只是为了内部使用）。

（我的 summer 现在可以跑一下了）

3\. 类别本身还可以进一步细分成子类

比如说，鸟类可以进一步分成鸡，大雁，黄鹂。

在 OOP 中，我们通过继承(inheritance)来表达上述概念。

```py
class Chicken(Bird):
    way_of_move = 'walk'
    possible_in_KFC = True class Oriole(Bird):
    way_of_move = 'fly' possible_in_KFC = False

summer = Chicken() print summer.have_feather print summer.move(5,8)

```

我们新定义的鸡（Chicken）类的，新增加了两个属性，移动方式（way_of_move）和可能在 KFC 找到（possible_in_KFC）

在类定义时，括号里改为了 Bird，用来说明，Chicken 是属于鸟类（Bird）的一个子类（酷点的说法，Chicken 继承自 Bird），而自然而然，Bird 就是 Chicken 的父类。通过这个说明，Python 就知道，Chicken 具有 Bird 的所有属性。我们可以看到，尽管我只声明了 summer 是鸡类，它依然具有鸟类的属性（无论是变量属性 have_feather 还是方法属性 move）

另外定义黄鹂(Oriole)类，同样继承自鸟类。这样，我们在有一个属于黄鹂的对象时，也会自动拥有鸟类的属性。

通过继承制度，我们可以避免程序中的重复信息和重复语句。如果我们分别定义两个类，而不继承自鸟类，那么我们就必须把鸟类的属性分别敲到鸡类和黄鹂类的定义中，累啊。

（回到问题 1, 括号中的 object，当括号中为 object 时，说明这个类没有父类（到头了））

所以说，面向对象提高了程序的可重复使用性。

我们可以看到，面向对象实际上基于人类认知时的习惯，将各种各样的东西分类，从而了解世界。我们从祖先开始可能已经练习了这个认知过程有几百万年，所以面向对象是很符合人类思维习惯的编程方法。所谓面向过程（也就是执行完一个语句再执行下一个）实际上是机器思维。通过面向对象的编程，我们实际上是更贴近我们自然的思维方式，也更方便和其他人交流我们程序里所包含的想法，甚至于那个人并不是程序员。

总结：

将东西根据属性归类 ( 将 object 归为 class )

方法是一种属性，表示动作

用继承来说明父类-子类关系。子类自动具有父类的所有属性。

self 代表了根据该类定义而创建的对象。

定义类：

```py
class class_name(parent_class):
    a = ...
    b = ... def method1():
        ... def method2():
        ...

```

建立对一个对象： 对象名 = 类名()

引用对象的属性： object.attribute