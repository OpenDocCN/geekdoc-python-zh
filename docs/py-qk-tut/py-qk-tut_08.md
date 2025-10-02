## Python 基础 09 面向对象的进一步拓展

[`www.cnblogs.com/vamei/archive/2012/06/02/2532018.html`](http://www.cnblogs.com/vamei/archive/2012/06/02/2532018.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

上一讲我们熟悉了对象和类的基本概念。这一讲我们将进一步拓展，以便我们真正能实际运用对象和类。

1\. 在方法内调用类属性（变量以及其它方法）：

上一讲我们已经提到，在定义方法时，必须有 self 这一参数，这个参数指的是对象。由于对象拥有类的所有性质，那么我们就可以在方法内部通过 self 来调用类的其它属性。

```py
class Human(object):
    laugh = 'hahahaha'
    def show_laugh(self): print self.laugh def laugh_100th(self): for i in range(100): self.show_laugh()
li_lei = Human()              # 李雷
li_lei.laugh_100th()

```

我们这里有一个变量属性 laugh，在方法 show_laugh()中通过 self.laugh 使用该属性的值。方法 show_laugh 则在 laugh_100th 中通过 self.show_laugh()被调用。

（通过对象来修改类属性是危险的，这样可能会影响根据这个类定义的所有对象的这一属性！！）

2\. __init__()方法

__init__()是一个特殊方法(special method)。Python 里会有一些特殊方法，Python 会以特别的方式处理它们。特殊方法的名字的特点是前后都有两个下划线。

__init__()方法的特殊在于，如果你在类中定义了这个方法，一旦你根据这个类建立对象，Python 就会自动调用这个方法（这个过程也叫初始化）。（在上一讲中，我们手动调用了 move()方法）

```py
class happyBird(Bird): def __init__(self,more_words): print 'We are happy birds.',more_words 
summer = happyBird('Happy,Happy!')

```

（Bird 类的定义见上一讲）

屏幕上打印出：

```py
We are happy birds.Happy,Happy!

```

我们看到，尽管我们只是创建了 summer 对象，但 __init__()方法被自动调用了。最后一行的语句(summer = happyBird...)先创建了对象，然后执行：

summer.__init__(more_words)

'Happy,Happy!' 被传递给了 __init__()的参数 more_words

3\. 对象的性质

上一讲我们讲了变量属性和方法属性。要注意，这些属性是类的属性。所有属于一个类的对象都会共享这些属性。比如说，鸟都有羽毛，鸡都不会飞。

在一些情况下，我们需要用到对象的性质。比如说，人是一个类别，我们知道，性别是人类的一个性质，但并不是所有的人类都是男性或者所有的人类都是女性。这个性质的值会随着对象的不同而不同。（李雷是人类的一个对象，性别是男；韩美美也是人类的一个对象，性别是女）。

从上一讲中，我们已经知道了，当定义类的方法时，必须要传递一个 self 的参数。这个参数指代的就是类的一个对象。当然，这是一个很模糊的一个概念。但一旦我们用类来新建一个对象（比如说我们下面例子中的 li_lei）, 那么 li_lei 就是 self 所代表的东西。我们已经知道了，li_lei 会拥有 Human 类的属性。进一步，我们通过赋值给 self.attribute，给 li_lei 这一对象增加一些性质（比如说性别）。由于 self 强制传递给各个方法，方法可以通过引用 self.attribute 很方便地查询到这些性质，并进行进一步的操作。

这样，我们在类的属性统一的基础上，又给每个对象增添了各自特色的性质，从而能描述多样的世界。

```py
class Human(object): def __init__(self, input_gender):
        self.gender = input_gender def printGender(self): print self.gender

li_lei = Human('male') # 这里，'male'作为参数传递给 __init__()方法的 input_gender 变量。
print li_lei.gender
li_lei.printGender()

```

首先，在初始化中，将参数 input_gender 赋值给对象 li_lei 的性质 gender。（上一讲，我们已经提到，self 指示的是对象, 也就是 li_lei）

我们发现，li_lei 拥有了属性 gender。在类 human 的定义中，并没有这样一个变量属性。Python 是在建立了 li_lei 这一对象之后，专门为 li_lei 建立的属性。我们称此为对象的性质。（也有人以类属性，对象属性来区分）。

对象的性质也可以被其它方法调用，正如我们在 printGender 方法中所看到的那样。

总结：

通过 self 调用类属性

__init__(): 在建立对象时自动执行

类属性和对象的性质的区别