# 五、类

## 类(1)

类，这个词如果是你第一次听到，把它作为一个单独的名词，总感觉怪怪的，因为在汉语体系中，很常见的是说“鸟类”、“人类”等词语，而单独说“类”，总感觉前面缺点修饰成分。其实，它对应的是英文单词 class，“类”是这个 class 翻译过来的，你就把它作为一个翻译术语吧。

除了“类”这个术语，从现在开始，还要经常提到一个 OOP，即面向对象编程（或者“面向对象程序设计”）。

为了理解类和 OOP，需要对一些枯燥的名词有了解。

### 术语

必须了解这些术语的基本含义，因为后面经常用到。下面的术语定义均来自维基百科。

#### 问题空间

**定义：**

> 问题空间是问题解决者对一个问题所达到的全部认识状态，它是由问题解决者利用问题所包含的信息和已贮存的信息主动地构成的。

一个问题一般有下面三个方面来定义：

*   初始状态——一开始时的不完全的信息或令人不满意的状况；
*   目标状态——你希望获得的信息或状态；
*   操作——为了从初始状态迈向目标状态，你可能采取的步骤。

这三个部分加在一起定义了问题空间（problem space）。

#### 对象

**定义：**

> 对象（object），台湾译作物件，是面向对象（Object Oriented）中的术语，既表示客观世界问题空间（Namespace）中的某个具体的事物，又表示软件系统解空间中的基本元素。

把 object 翻译为“对象”，是比较抽象的。因此，有人认为，不如翻译为“物件”更好。因为“物件”让人感到一种具体的东西。

这种看法在某些语言中是非常适合的。但是，在 Python 中，则无所谓，不管怎样，Python 中的一切都是对象，不管是字符串、函数、模块还是类，都是对象。“万物皆对象”。

都是对象有什么优势吗？太有了。这说明 Python 天生就是 OOP 的。也说明，Python 中的所有东西，都能够进行拼凑组合应用，因为对象就是可以拼凑组合应用的。

对于对象这个东西，OOP 大师 Grandy Booch 的定义，应该是权威的，相关定义的内容包括：

*   **对象**：一个对象有自己的状态、行为和唯一的标识；所有相同类型的对象所具有的结构和行为在他们共同的类中被定义。
*   **状态（state）**：包括这个对象已有的属性（通常是类里面已经定义好的）在加上对象具有的当前属性值（这些属性往往是动态的）
*   **行为（behavior）**：是指一个对象如何影响外界及被外界影响，表现为对象自身状态的改变和信息的传递。
*   **标识（identity）**：是指一个对象所具有的区别于所有其它对象的属性。（本质上指内存中所创建的对象的地址）

大师的话的确有水平，听起来非常高深。不过，初学者可能理解起来就有点麻烦了。我就把大师的话化简一下，但是化简了之后可能在严谨性上就不足了，我想对于初学者来讲，应该是影响不很大的。随着学习和时间的深入，就更能理解大师的严谨描述了。

简化之，对象应该具有属性（就是上面的状态，因为属性更常用）、方法（就是上面的行为，方法跟常被使用）和标识。因为标识是内存中自动完成的，所以，平时不用怎么管理它。主要就是属性和方法。

任何一个对象都要包括这两部分：属性（是什么）和方法（能做什么）。

#### 面向对象

**定义：**

> 面向对象程序设计（英语：Object-oriented programming，缩写：OOP）是一种程序设计范型，同时也是一种程序开发的方法。对象指的是类的实例。它将对象作为程序的基本单元，将程序和数据封装其中，以提高软件的重用性、灵活性和扩展性。
> 
> 面向对象程序设计可以看作一种在程序中包含各种独立而又互相调用的对象的思想，这与传统的思想刚好相反：传统的程序设计主张将程序看作一系列函数的集合，或者直接就是一系列对电脑下达的指令。面向对象程序设计中的每一个对象都应该能够接受数据、处理数据并将数据传达给其它对象，因此它们都可以被看作一个小型的“机器”，即对象。
> 
> 目前已经被证实的是，面向对象程序设计推广了程序的灵活性和可维护性，并且在大型项目设计中广为应用。 此外，支持者声称面向对象程序设计要比以往的做法更加便于学习，因为它能够让人们更简单地设计并维护程序，使得程序更加便于分析、设计、理解。反对者在某些领域对此予以否认。
> 
> 当我们提到面向对象的时候，它不仅指一种程序设计方法。它更多意义上是一种程序开发方式。在这一方面，我们必须了解更多关于面向对象系统分析和面向对象设计（Object Oriented Design，简称 OOD）方面的知识。

下面再引用一段来自维基百科中关于 OOP 的历史。

> 面向对象程序设计的雏形，早在 1960 年的 Simula 语言中即可发现，当时的程序设计领域正面临着一种危机：在软硬件环境逐渐复杂的情况下，软件如何得到良好的维护？面向对象程序设计在某种程度上通过强调可重复性解决了这一问题。20 世纪 70 年代的 Smalltalk 语言在面向对象方面堪称经典——以至于 30 年后的今天依然将这一语言视为面向对象语言的基础。
> 
> 计算机科学中对象和实例概念的最早萌芽可以追溯到麻省理工学院的 PDP-1 系统。这一系统大概是最早的基于容量架构（capability based architecture）的实际系统。另外 1963 年 Ivan Sutherland 的 Sketchpad 应用中也蕴含了同样的思想。对象作为编程实体最早是于 1960 年代由 Simula 67 语言引入思维。Simula 这一语言是奥利-约翰·达尔和克利斯登·奈加特在挪威奥斯陆计算机中心为模拟环境而设计的。（据说，他们是为了模拟船只而设计的这种语言，并且对不同船只间属性的相互影响感兴趣。他们将不同的船只归纳为不同的类，而每一个对象，基于它的类，可以定义它自己的属性和行为。）这种办法是分析式程序的最早概念体现。在分析式程序中，我们将真实世界的对象映射到抽象的对象，这叫做“模拟”。Simula 不仅引入了“类”的概念，还应用了实例这一思想——这可能是这些概念的最早应用。
> 
> 20 世纪 70 年代施乐 PARC 研究所发明的 Smalltalk 语言将面向对象程序设计的概念定义为，在基础运算中，对对象和消息的广泛应用。Smalltalk 的创建者深受 Simula 67 的主要思想影响，但 Smalltalk 中的对象是完全动态的——它们可以被创建、修改并销毁，这与 Simula 中的静态对象有所区别。此外，Smalltalk 还引入了继承性的思想，它因此一举超越了不可创建实例的程序设计模型和不具备继承性的 Simula。此外，Simula 67 的思想亦被应用在许多不同的语言，如 Lisp、Pascal。
> 
> 面向对象程序设计在 80 年代成为了一种主导思想，这主要应归功于 C++——C 语言的扩充版。在图形用户界面（GUI）日渐崛起的情况下，面向对象程序设计很好地适应了潮流。GUI 和面向对象程序设计的紧密关联在 Mac OS X 中可见一斑。Mac OS X 是由 Objective-C 语言写成的，这一语言是一个仿 Smalltalk 的 C 语言扩充版。面向对象程序设计的思想也使事件处理式的程序设计更加广泛被应用（虽然这一概念并非仅存在于面向对象程序设计）。一种说法是，GUI 的引入极大地推动了面向对象程序设计的发展。
> 
> 苏黎世联邦理工学院的尼克劳斯·维尔特和他的同事们对抽象数据和模块化程序设计进行了研究。Modula-2 将这些都包括了进去，而 Oberon 则包括了一种特殊的面向对象方法——不同于 Smalltalk 与 C++。
> 
> 面向对象的特性也被加入了当时较为流行的语言：Ada、BASIC、Lisp、Fortran、Pascal 以及种种。由于这些语言最初并没有面向对象的设计，故而这种糅合常常会导致兼容性和维护性的问题。与之相反的是，“纯正的”面向对象语言却缺乏一些程序员们赖以生存的特性。在这一大环境下，开发新的语言成为了当务之急。作为先行者，Eiffel 成功地解决了这些问题，并成为了当时较受欢迎的语言。
> 
> 在过去的几年中，Java 语言成为了广为应用的语言，除了它与 C 和 C++ 语法上的近似性。Java 的可移植性是它的成功中不可磨灭的一步，因为这一特性，已吸引了庞大的程序员群的投入。
> 
> 在最近的计算机语言发展中，一些既支持面向对象程序设计，又支持面向过程程序设计的语言悄然浮出水面。它们中的佼佼者有 Python、Ruby 等等。
> 
> 正如面向过程程序设计使得结构化程序设计的技术得以提升，现代的面向对象程序设计方法使得对设计模式的用途、契约式设计和建模语言（如 UML）技术也得到了一定提升。

列位看官，当您阅读到这句话的时候，我就姑且认为您已经对面向对象有了一个模糊的认识了。那么，类和 OOP 有什么关系呢？

#### 类

**定义：**

> 在面向对象程式设计，类（class）是一种面向对象计算机编程语言的构造，是创建对象的蓝图，描述了所创建的对象共同的属性和方法。
> 
> 类的更严格的定义是由某种特定的元数据所组成的内聚的包。它描述了一些对象的行为规则，而这些对象就被称为该类的实例。类有接口和结构。接口描述了如何通过方法与类及其实例互操作，而结构描述了一个实例中数据如何划分为多个属性。类是与某个层的对象的最具体的类型。类还可以有运行时表示形式（元对象），它为操作与类相关的元数据提供了运行时支持。
> 
> 支持类的编程语言在支持与类相关的各种特性方面都多多少少有一些微妙的差异。大多数都支持不同形式的类继承。许多语言还支持提供封装性的特性，比如访问修饰符。类的出现，为面向对象编程的三个最重要的特性（封装性，继承性，多态性），提供了实现的手段。

看到这里，看官或许有一个认识，要 OOP 编程，就得用到类。可以这么说，虽然不是很严格。但是，反过来就不能说了。不是说用了类就一定是 OOP。

### 编写类

首先要明确，类是对某一群具有同样属性和方法的对象的抽象。比如这个世界上有很多长翅膀并且会飞的生物，于是聪明的人们就将它们统一称为“鸟”——这就是一个类，虽然它也可以称作“鸟类”。

还是以美女为例子，因为这个例子不仅能阅读本课程不犯困，还能兴趣昂然。

要定义类，就要抽象，找出共同的方面。

```py
class 美女:        #用 class 来声明，后面定义的是一个类
    pass 
```

好，现在就从这里开始，编写一个类，不过这次我们暂时不用 Python，而是用伪代码，当然，这个代码跟 Python 相去甚远。如下：

```py
class 美女:
    胸围 = 90
    腰围 = 58
    臀围 = 83
    皮肤 = white
    唱歌()
    做饭() 
```

定义了一个名称为“美女”的类，其中我约定，没有括号的是属性，带有括号的是方法。这个类仅仅是对美女的通常抽象，并不是某个具体美女.

对于一个具体的美女，比如前面提到的苍老师或者王美女，她们都是上面所定义的“美女”那个类的具体化，这在编程中称为“美女类”的实例。

```py
王美女 = 美女() 
```

我用这样一种表达方式，就是将“美女类”实例化了，对“王美女”这个实例，就可以具体化一些属性，比如胸围；还可以具体实施一些方法，比如做饭。通常可以用这样一种方式表示：

```py
a = 王美女.胸围 
```

用点号`.`的方式，表示王美女胸围的属性，得到的变量 a 就是 90.另外，还可以通过这种方式给属性赋值，比如

```py
王美女.皮肤 = black 
```

这样，这个实例（王美女）的皮肤就是黑色的了。

通过实例，也可以访问某个方法，比如：

```py
王美女.做饭() 
```

这就是在执行一个方法，让王美女这个实例做饭。现在也比较好理解了，只有一个具体的实例才能做饭。

至此，你是否对类和实例，类的属性和方法有初步理解了呢？

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 类(2)

现在开始不用伪代码了，用真正的 Python 代码来理解类。当然，例子还是要用读者感兴趣的例子。

### 新式类和旧式类

因为 Python 是一个不断发展的高级语言（似乎别的语言是不断发展的，甚至于自然语言也是），导致了在 Python2.x 的版本中，有“新式类”和“旧式类（也叫做经典类）”之分。新式类是 Python2.2 引进的，在此后的版本中，我们一般用的都是新式类。本着知其然还要知其所以然的目的，简单回顾一下两者的差别。

```py
>>> class AA:
...     pass
... 
```

这是定义了一个非常简单的类，而且是旧式类。至于如何定义类，下面会详细说明。读者姑且囫囵吞枣似的的认同我刚才建立的名为 `AA` 的类，为了简单，这个类内部什么也不做，就是用 `pass` 一带而过。但不管怎样，是一个类，而且是一个旧式类（或曰经典类）

然后，将这个类实例化（还记得上节中实例化吗？对，就是那个王美女干的事情）：

```py
>>> aa = AA() 
```

不要忘记，实例化的时候，类的名称后面有一对括号。接下来做如下操作：

```py
>>> type(AA)
<type 'classobj'>
>>> aa.__class__
<class __main__.AA at 0xb71f017c>
>>> type(aa)
<type 'instance'> 
```

解读一下上面含义：

*   `type(AA)`：查看类 `AA` 的类型，返回的是`'classobj'`
*   `aa.__class__`：aa 是一个实例，也是一个对象，每个对象都有`__class__`属性，用于显示它的类型。这里返回的结果是`<class __main__.AA at 0xb71f017c>`，从这个结果中可以读出的信息是，aa 是类 AA 的实例，并且类 AA 在内存中的地址是 `0xb71f017c`。
*   `type(aa)`：是要看实例 aa 的类型，它显示的结果是`'instance`，意思是告诉我们它的类型是一个实例。

在这里是不是有点感觉不和谐呢？`aa.__class__`和 `type(aa)` 都可以查看对象类型，但是它们居然显示不一样的结果。比如，查看这个对象：

```py
>>> a = 7
>>> a.__class__
<type 'int'>
>>> type(a)
<type 'int'> 
```

别忘记了，前面提到过的“万物皆对象”，那么一个整数 7 也是对象，用两种方式查看，返回的结果一样。为什么到类（严格讲是旧式类）这里，居然返回不一样呢？太不和谐了。

于是乎，就有了新式类，从 Python2.2 开始，变成这样了：

```py
>>> class BB(object):
...     pass
... 

>>> bb = BB()

>>> bb.__class__
<class '__main__.BB'>
>>> type(bb)
<class '__main__.BB'> 
```

终于把两者统一起来了，世界和谐了。

这就是新式类和旧式类的不同。

当然，不同点绝非仅仅于此，这里只不过提到一个现在能够理解的不同罢了。另外的不同还在于两者对于多重继承的查找和调用方法不同，旧式类是深度优先，新式类是广度优先。可以先不理解，后面会碰到的。

不管是新式类、还是旧式类，都可以通过这样的方法查看它们在内存中的存储空间信息

```py
>>> print aa
<__main__.AA instance at 0xb71efd4c>

>>> print bb
<__main__.BB object at 0xb71efe6c> 
```

分别告诉了我们两个实例是基于谁生成的，不过还是稍有区别。

知道了旧式类和新式类，那么下面的所有内容，就都是对新式类而言。“喜新厌旧”不是编程经常干的事情吗？所以，旧式类就不是我们讨论的内容了。

还要注意，如果你用的是 Python3，就不用为新式类和旧式类而担心了，因为在 Python3 中压根儿就没有这个问题存在。

如何定义新式类呢？

第一种定义方法，就是如同前面那样：

```py
>>> class BB(object):
...     pass
... 
```

跟旧式类的区别就在于类的名字后面跟上 `(object)`，这其实是一种名为“继承”的类的操作，当前的类 BB 是以类 object 为上级的（object 被称为父类），即 BB 是继承自类 object 的新类。在 Python3 中，所有的类自然地都是类 object 的子类，就不用彰显出继承关系了。对了，这里说的有点让读者糊涂，因为冒出来了“继承”、“父类”、“子类”，不用着急，继续向下看。下面精彩，并且能解惑。

第二种定义方法，在类的前面写上这么一句：`__metaclass__ == type`，然后定义类的时候，就不需要在名字后面写`(object)`了。

```py
>>> __metaclass__ = type
>>> class CC:
...     pass
... 
>>> cc = CC()
>>> cc.__class__
<class '__main__.CC'>
>>> type(cc)
<class '__main__.CC'> 
```

两种方法，任你选用，没有优劣之分。

### 创建类

因为在一般情况下，一个类都不是两三行能搞定的。所以，下面可能很少使用交互模式了，因为那样一旦有一点错误，就前功尽弃。我改用编辑界面。你用什么工具编辑？Python 自带一个 IDE，可以使用。我习惯用 vim。你用你习惯的工具即可。如果你没有别的工具，就用安装 Python 是自带的那个 IDE。

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type

class Person:
    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    def color(self, color):
        print "%s is %s" % (self.name, color) 
```

上面定义的是一个比较常见的类，一般情况下，都是这样子的。下面对这个“大众脸”的类一一解释。

#### 新式类

`__metaclass__ = type`，意味着下面的类是新式类。

#### 定义类

`class Person`，这是在声明创建一个名为"Person"的类。类的名称一般用大写字母开头，这是惯例。如果名称是两个单词，那么两个单词的首字母都要大写，例如 `class HotPerson`，这种命名方法有一个形象的名字，叫做“驼峰式命名”。当然，如果故意不遵循此惯例，也未尝不可，但是，会给别人阅读乃至于自己以后阅读带来麻烦，不要忘记“代码通常是给人看的，只是偶尔让机器执行”。既然大家都是靠右走的，你就别非要在路中间睡觉了。

接下来，分别以缩进表示的，就是这个类的内容了。其实那些东西看起来并不陌生，你一眼就认出它们了——就是已经学习过的函数。没错，它们就是函数。不过，很多程序员喜欢把类里面的函数叫做“方法”。是的，就是上节中说到的对象的“方法”。我也看到有人撰文专门分析了“方法”和“函数”的区别。但是，我倒是认为这不重要，重要的是类的中所谓“方法”和前面的函数，在数学角度看，丝毫没有区别。所以，你尽可以称之为函数。当然，听到有人说方法，也不要诧异和糊涂。它们本质是一样的。

需要再次提醒，函数的命名方法是以 `def` 发起，并且函数名称首字母不要用大写，可以使用 `aa_bb` 的样式，也可以使用 `aaBb` 的样式，一切看你的习惯了。

不过，要注意的是，类中的函数（方法）的参数跟以往的参数样式有区别，那就是每个函数必须包括 `self` 参数，并且作为默认的第一个参数。这是需要注意的地方。至于它的用途，继续学习即可知道。

#### 初始化

`def __init__`，这个函数是一个比较特殊的，并且有一个名字，叫做**初始化函数**（注意，很多教材和资料中，把它叫做构造函数，这种说法貌似没有错误，但是一来从字面意义上看，它对应的含义是初始化，二来在 Python 中它的作用和其它语言比如 java 中的构造函数还不完全一样，因为还有一个`__new__`的函数，是真正地构造。所以，在本教程中，我称之为初始化函数）。它是以两个下划线开始，然后是 init，最后以两个下划线结束。

> 所谓初始化，就是让类有一个基本的面貌，而不是空空如也。做很多事情，都要初始化，让事情有一个具体的起点状态。比如你要喝水，必须先初始化杯子里面有水。在 Python 的类中，初始化就担负着类似的工作。这个工作是在类被实例化的时候就执行这个函数，从而将初始化的一些属性可以放到这个函数里面。

此例子中的初始化函数，就意味着实例化的时候，要给参数 name 提供一个值，作为类初始化的内容。通俗点啰嗦点说，就是在这个类被实例化的同时，要通过 name 参数传一个值，这个值被一开始就写入了类和实例中，成为了类和实例的一个属性。比如：

```py
girl = Person('wangguniang') 
```

girl 是一个实例对象，就如同前面所说的一样，它有属性和方法。这里仅说属性吧。当通过上面的方式实例化后，就自动执行了初始化函数，让实例 girl 就具有了 name 属性。

```py
print girl.name 
```

执行这句话的结果是打印出 `wangguniang`。

这就是初始化的功能。简而言之，通过初始化函数，确定了这个实例（类）的“基本属性”（实例是什么样子的）。比如上面的实例化之后，就确立了实例 girl 的 name 是"wangguniang"。

初始化函数，就是一个函数，所以，它的参数设置，也符合前面学过的函数参数设置规范。比如

```py
def __init__(self,*args):
    pass 
```

这种类型的参数：*args 和前面讲述函数参数一样，就不多说了。忘了的看官，请去复习。但是，self 这个参数是必须的。

很多时候，并不是每次都要从外面传入数据，有时候会把初始化函数的某些参数设置默认值，如果没有新的数据传入，就应用这些默认值。比如：

```py
class Person:
    def __init__(self, name, lang="golang", website="www.google.com"):
        self.name = name
        self.lang = lang
        self.website = website
        self.email = "qiwsir@gmail.com"

laoqi = Person("LaoQi")     
info = Person("qiwsir",lang="python",website="qiwsir.github.io")

print "laoqi.name=",laoqi.name
print "info.name=",info.name
print "-------"
print "laoqi.lang=",laoqi.lang
print "info.lang=",info.lang
print "-------"
print "laoqi.website=",laoqi.website
print "info.website=",info.website

#运行结果

laoqi.name= LaoQi
info.name= qiwsir
-------
laoqi.lang= golang
info.lang= python
-------
laoqi.website= www.google.com
info.website= qiwsir.github.io 
```

在编程界，有这样一句话，说“类是实例工厂”，什么意思呢？工厂是干什么的？生产物品，比如生产电脑。一个工厂可以生产好多电脑。那么，类，就能“生产”好多实例，所以，它是“工厂”。比如上面例子中，就有两个实例。

#### 函数（方法）

还是回到本节开头的那个类。构造函数下面的两个函数：`def getName(self)`,`def color(self, color)`，这两个函数和前面的初始化函数有共同的地方，即都是以 self 作为第一个参数。

```py
def getName(self):
    return self.name 
```

这个函数中的作用就是返回在初始化时得到的值。

```py
girl = Person('wangguniang')
name = girl.getName() 
```

`girl.getName()`就是调用实例 girl 的方法。调用该方法的时候特别注意，方法名后面的括号不可少，并且括号中不要写参数，在类中的 `getName(self)` 函数第一个参数 self 是默认的，当类实例化之后，调用此函数的时候，第一个参数不需要赋值。那么，变量 name 的最终结果就是 `name = "wangguniang"`。

同样道理，对于方法：

```py
def color(self, color):
    print "%s is %s" % (self.name, color) 
```

也是在实例化之后调用：

```py
girl.color("white") 
```

这也是在执行实例化方法，只是由于类中的该方法有两个参数，除了默认的 self 之外，还有一个 color，所以，在调用这个方法的时候，要为后面那个参数传值了。

至此，已经将这个典型的类和调用方法分解完毕，把全部代码完整贴出，请读者在从头到尾看看，是否理解了每个部分的含义：

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type             #新式类

class Person:                    #创建类
    def __init__(self, name):    #构造函数
        self.name = name

    def getName(self):           #类中的方法（函数）
        return self.name

    def color(self, color):
        print "%s is %s" % (self.name, color)

girl = Person('wangguniang')      #实例化
name = girl.getName()            #调用方法（函数） 
print "the person's name is: ", name
girl.color("white")              #调用方法（函数）

print "------"
print girl.name                  #实例的属性 
```

保存后，运行得到如下结果：

```py
$ python 20701.py 
the person's name is:  wangguniang
wangguniang is white
------
wangguniang 
```

#### 类和实例

有必要总结一下类和实例的关系：

*   “类提供默认行为，是实例的工厂”（源自 Learning Python），这句话非常经典，一下道破了类和实例的关系。所谓工厂，就是可以用同一个模子做出很多具体的产品。类就是那个模子，实例就是具体的产品。所以，实例是程序处理的实际对象。
*   类是由一些语句组成，但是实例，是通过调用类生成，每次调用一个类，就得到这个类的新的实例。
*   对于类的：`class Person`，class 是一个可执行的语句。如果执行，就得到了一个类对象，并且将这个类对象赋值给对象名（比如 Person）。

也许上述比较还不足以让看官理解类和实例，没关系，继续学习，在前进中排除疑惑。

### self 的作用

类里面的函数，第一个参数是 self，但是在实例化的时候，似乎没有这个参数什么事儿，那么 self 是干什么的呢？

self 是一个很神奇的参数。

在 Person 实例化的过程中 `girl = Person("wangguniang")`，字符串"wangguniang"通过初始化函数（`__init__()`）的参数已经存入到内存中，并且以 Person 类型的面貌存在，组成了一个对象，这个对象和变量 girl 建立引用关系。这个过程也可说成这些数据附加到一个实例上。这样就能够以:`object.attribute`的形式，在程序中任何地方调用某个数据，例如上面的程序中以 `girl.name` 的方式得到`"wangguniang"`。这种调用方式，在类和实例中经常使用，点号“.”后面的称之为类或者实例的属性。

这是在程序中，并且是在类的外面。如果在类的里面，想在某个地方使用实例化所传入的数据（"wangguniang"），怎么办？

在类内部，就是将所有传入的数据都赋给一个变量，通常这个变量的名字是 self。注意，这是习惯，而且是共识，所以，看官不要另外取别的名字了。

在初始化函数中的第一个参数 self，就是起到了这个作用——接收实例化过程中传入的所有数据，这些数据是初始化函数后面的参数导入的。显然，self 应该就是一个实例（准确说法是应用实例），因为它所对应的就是具体数据。

如果将上面的类稍加修改，看看效果：

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type

class Person:
    def __init__(self, name):
        self.name = name
        print self           #新增
        print type(self)     #新增 
```

其它部分省略。当初始化的时候，就首先要运行构造函数，同时就打印新增的两条。结果是：

```py
<__main__.Person object at 0xb7282cec>
<class '__main__.Person'> 
```

证实了推理。self 就是一个实例（准确说是实例的引用变量）。

self 这个实例跟前面说的那个 girl 所引用的实例对象一样，也有属性。那么，接下来就规定其属性和属性对应的数据。上面代码中：

```py
self.name = name 
```

就是规定了 self 实例的一个属性，这个属性的名字也叫做 name，这个属性的值等于初始化函数的参数 name 所导入的数据。注意，`self.name` 中的 name 和初始化函数的参数 `name` 没有任何关系，它们两个一样，只不过是一种起巧合（经常巧合，其实是为了省事和以后识别方便，故意让它们巧合。），或者说是写代码的人懒惰，不想另外取名字而已，无他。当然，如果写成 `self.xxxooo = name`，也是可以的。

其实，从效果的角度来理解，这么理解更简化：类的实例 girl 对应着 self，girl 通过 self 导入实例属性的所有数据。

当然，self 的属性数据，也不一定非得是由参数传入的，也可以在构造函数中自己设定。比如：

```py
#!/usr/bin/env Python
#coding:utf-8

__metaclass__ = type

class Person:
    def __init__(self, name):
        self.name = name
        self.email = "qiwsir@gmail.com"     #这个属性不是通过参数传入的

info = Person("qiwsir")              #换个字符串和实例化变量
print "info.name=",info.name
print "info.email=",info.email      #info 通过 self 建立实例，并导入实例属性数据 
```

运行结果

```py
info.name= qiwsir
info.email= qiwsir@gmail.com    #打印结果 
```

通过这个例子，其实让我们拓展了对 self 的认识，也就是它不仅仅是为了在类内部传递参数导入的数据，还能在初始化函数中，通过 `self.attribute` 的方式，规定 self 实例对象的属性，这个属性也是类实例化对象的属性，即做为类通过初始化函数初始化后所具有的属性。所以在实例 info 中，通过 info.email 同样能够得到该属性的数据。在这里，就可以把 self 形象地理解为“内外兼修”了。或者按照前面所提到的，将 info 和 self 对应起来，self 主内，info 主外。

怎么样？是不是明白了类的奥妙？

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 类(3)

在上一节中，对类有了基本的或者说是模糊的认识，为了能够对类有更深刻的认识，本节要深入到一些细节。

### 类属性和实例属性

正如上节的案例中，一个类实例化后，实例是一个对象，有属性。同样，类也是一个对象，它也有属性。

```py
>>> class A(object):
...     x = 7
... 
```

在交互模式下，定义一个很简单的类（注意观察，有`(object)`，是新式类），类中有一个变量 `x = 7`，当然，如果愿意还可以写别的。因为一下操作中，只用到这个，我就不写别的了。

```py
>>> A.x
7 
```

在类 A 中，变量 x 所引用的数据，能够直接通过类来调用。或者说 x 是类 A 的属性，这种属性有一个名称，曰“类属性”。类属性仅限于此——类中的变量。它也有其他的名字，如静态数据。

```py
>>> foo = A()
>>> foo.x
7 
```

实例化，通过实例也可以得到这个属性，这个属性叫做“实例属性”。对于同一属性，可以用类来访问（类属性），在一般情况下，也可以通过实例来访问同样的属性。但是：

```py
>>> foo.x += 1
>>> foo.x
8
>>> A.x
7 
```

实例属性更新了，类属性没有改变。这至少说明，类属性不会被实例属性左右，也可以进一步说“类属性与实例属性无关”。那么，`foo.x += 1` 的本质是什么呢？其本质是该实例 foo 又建立了一个新的属性，但是这个属性（新的 foo.x）居然与原来的属性（旧的 foo.x）重名，所以，原来的 foo.x 就被“遮盖了”，只能访问到新的 foo.x，它的值是 8.

```py
>>> foo.x
8
>>> del foo.x
>>> foo.x
7 
```

既然新的 foo.x“遮盖”了旧的 foo.x，如果删除它，旧的不久显现出来了？的确是。删除之后，foo.x 就还是原来的值。此外，还可以通过建立一个不与它重名的实例属性：

```py
>>> foo.y = foo.x + 1
>>> foo.y
8
>>> foo.x
7 
```

foo.y 就是新建的一个实例属性，它没有影响原来的实例属性 foo.x。

但是，类属性能够影响实例属性，这点应该好理解，因为实例就是通过实例化调用类的。

```py
>>> A.x += 1
>>> A.x
8
>>> foo.x
8 
```

这时候实例属性跟着类属性而改变。

以上所言，是指当类中变量引用的是不可变数据。如果类中变量引用可变数据，情形会有所不同。因为可变数据能够进行原地修改。

```py
>>> class B(object):
...     y = [1,2,3]
... 
```

这次定义的类中，变量引用的是一个可变对象。

```py
>>> B.y         #类属性
[1, 2, 3]
>>> bar = B()
>>> bar.y       #实例属性
[1, 2, 3]

>>> bar.y.append(4)
>>> bar.y
[1, 2, 3, 4]
>>> B.y
[1, 2, 3, 4]

>>> B.y.append("aa")
>>> B.y
[1, 2, 3, 4, 'aa']
>>> bar.y
[1, 2, 3, 4, 'aa'] 
```

从上面的比较操作中可以看出，当类中变量引用的是可变对象是，类属性和实例属性都能直接修改这个对象，从而影响另一方的值。

对于类属性和实例属性，除了上述不同之外，在下面的操作中，也会有差异。

```py
>>> foo = A()
>>> dir(foo)
['__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'x'] 
```

实例化类 A，可以查看其所具有的属性（看最后一项，x），当然，执行 `dir(A)` 也是一样的。

```py
>>> A.y = "hello"
>>> foo.y
'hello' 
```

增加一个类属性，同时在实例属性中也增加了一样的名称和数据的属性。如果增加通过实例增加属性呢？看下面：

```py
>>> foo.z = "python"
>>> foo.z
'python'
>>> A.z
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: type object 'A' has no attribute 'z' 
```

类并没有收纳这个属性。这进一步说明，类属性不受实例属性左右。另外，在类确定或者实例化之后，也可以增加和修改属性，其方法就是通过类或者实例的点号操作来实现，即 `object.attribute`，可以实现对属性的修改和增加。

### 数据流转

在类的应用中，最广泛的是将类实例化，通过实例来执行各种方法。所以，对此过程中的数据流转一定要弄明白。

回顾上节已经建立的那个类，做适当修改，读者是否能够写上必要的注释呢？如果你把注释写上，就已经理解了类的基本结构。

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type

class Person:
    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    def breast(self, n):
        self.breast = n

    def color(self, color):
        print "%s is %s" % (self.name, color)

    def how(self):
        print "%s breast is %s" % (self.name, self.breast)

girl = Person('wangguniang')
girl.breast(90)

girl.color("white")
girl.how() 
```

运行后结果：

```py
$ python 20701.py 
wangguniang is white
wangguniang breast is 90 
```

一图胜千言，有图有真相。通过图示，我们看一看数据的流转过程。

![](img/20801.png)

创建实例 `girl = Person('wangguniang')`，注意观察图上的箭头方向。girl 这个实例和 Person 类中的 self 对应，这正是应了上节所概括的“实例变量与 self 对应，实例变量主外，self 主内”的概括。"wangguniang"是一个具体的数据，通过初始化函数中的 name 参数，传给 self.name，前面已经讲过，self 也是一个实例，可以为它设置属性，`self.name` 就是一个属性，经过初始化函数，这个属性的值由参数 name 传入，现在就是"wangguniang"。

在类 Person 的其它方法中，都是以 self 为第一个或者唯一一个参数。注意，在 Python 中，这个参数要显明写上，在类内部是不能省略的。这就表示所有方法都承接 self 实例对象，它的属性也被带到每个方法之中。例如在方法里面使用 `self.name` 即是调用前面已经确定的实例属性数据。当然，在方法中，还可以继续为实例 self 增加属性，比如 `self.breast`。这样，通过 self 实例，就实现了数据在类内部的流转。

如果要把数据从类里面传到外面，可以通过 `return` 语句实现。如上例子中所示的 `getName` 方法。

因为实例名称(girl)和 self 是对应关系，实际上，在类里面也可以用 girl 代替 self。例如，做如下修改：

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type

class Person:
    def __init__(self, name):
        self.name = name

    def getName(self):
        #return self.name
        return girl.name    #修改成这个样子，但是在编程实践中不要这么做。

girl = Person('wangguniang')
name = girl.getName()
print name 
```

运行之后，打印：

```py
wangguniang 
```

这个例子说明，在实例化之后，实例变量 girl 和函数里面的那个 self 实例是完全对应的。但是，提醒读者，千万不要用上面的修改了的那个方式。因为那样写使类没有独立性，这是大忌。

### 命名空间

命名空间，英文名字：namespaces。在研究类或者面向对象编程中，它常常被提到。虽然在《函数(2)中已经对命名空间进行了解释，那时是在函数的知识范畴中对命名空间的理解。现在，我们在类的知识范畴中理解“类命名空间”——定义类时，所有位于 class 语句中的代码都在某个命名空间中执行，即类命名空间。

在研习命名空间以前，请打开在 Python 的交互模式下，输入：`import this`，可以看到:

```py
>>> import this
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those! 
```

这里列位看到的就是所谓《Python 之禅》，请看最后一句： Namespaces are one honking great idea -- let's do more of those!

这是为了向看官说明 Namespaces、命名空间值重要性。

把在《函数(2)》href="https://github.com/qiwsir/StarterLearningPython/blob/master/202.md")中已经阐述的命名空间用一句比较学术化的语言概括：

**命名空间是从所定义的命名到对象的映射集合。**

不同的命名空间，可以同时存在，当彼此相互独立互不干扰。

命名空间因为对象的不同，也有所区别，可以分为如下几种：

*   内置命名空间(Built-in Namespaces)：Python 运行起来，它们就存在了。内置函数的命名空间都属于内置命名空间，所以，我们可以在任何程序中直接运行它们，比如前面的 id(),不需要做什么操作，拿过来就直接使用了。
*   全局命名空间(Module:Global Namespaces)：每个模块创建它自己所拥有的全局命名空间，不同模块的全局命名空间彼此独立，不同模块中相同名称的命名空间，也会因为模块的不同而不相互干扰。
*   本地命名空间(Function&Class: Local Namespaces)：模块中有函数或者类，每个函数或者类所定义的命名空间就是本地命名空间。如果函数返回了结果或者抛出异常，则本地命名空间也结束了。

从网上盗取了一张图，展示一下上述三种命名空间的关系

![

那么程序在查询上述三种命名空间的时候，就按照从里到外的顺序，即：Local Namespaces --> Global Namesspaces --> Built-in Namesspaces

```py
>>> def foo(num,str):
...     name = "qiwsir"
...     print locals()
... 
>>> foo(221,"qiwsir.github.io")
{'num': 221, 'name': 'qiwsir', 'str': 'qiwsir.github.io'}
>>> 
```

这是一个访问本地命名空间的方法，用 `print locals()` 完成，从这个结果中不难看出，所谓的命名空间中的数据存储结构和 dictionary 是一样的。

根据习惯，看官估计已经猜测到了，如果访问全局命名空间，可以使用 `print globals()`。

### 作用域

作用域是指 Python 程序可以直接访问到的命名空间。“直接访问”在这里意味着访问命名空间中的命名时无需加入附加的修饰符。（这句话是从网上抄来的）

程序也是按照搜索命名空间的顺序，搜索相应空间的能够访问到的作用域。

```py
def outer_foo():
    b = 20
    def inner_foo():
        c = 30
a = 10 
```

假如我现在位于 inner_foo() 函数内，那么 c 对我来讲就在本地作用域，而 b 和 a 就不是。如果我在 inner_foo() 内再做：b=50，这其实是在本地命名空间内新创建了对象，和上一层中的 b=20 毫不相干。可以看下面的例子：

```py
#!/usr/bin/env Python
#coding:utf-8

def outer_foo():
    a = 10
    def inner_foo():
        a = 20
        print "inner_foo,a=",a      #a=20

    inner_foo()
    print "outer_foo,a=",a          #a=10

a = 30
outer_foo()
print "a=",a                #a=30

#运行结果

inner_foo,a= 20
outer_foo,a= 10
a= 30 
```

如果要将某个变量在任何地方都使用，且能够关联，那么在函数内就使用 global 声明，其实就是曾经讲过的全局变量。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 类(4)

本节介绍类中一个非常重要的东西——继承，其实也没有那么重要，只是听起来似乎有点让初学者晕头转向，然后就感觉它属于很高级的东西，真是情况如何？学了之后你自然有感受。

在现实生活中，“继承”意味着一个人从另外一个人那里得到了一些什么，比如“继承革命先烈的光荣传统”、“某人继承他老爹的万贯家产”等。总之，“继承”之后，自己就在所继承的方面省力气、不用劳神费心，能轻松得到，比如继承了万贯家产，自己就一夜之间变成富豪。如果继承了“革命先烈的光荣传统”，自己是不是一下就变成革命者呢？

当然，生活中的继承或许不那么严格，但是编程语言中的继承是有明确规定和稳定的预期结果的。

> 继承（Inheritance）是面向对象软 件技术当中的一个概念。如果一个类别 A“继承自”另一个类别 B，就把这个 A 称为“B 的子类别”，而把 B 称为“A 的父类别”，也可以称“B 是 A 的超类”。
> 
> 继承可以使得子类别具有父类别的各种属性和方法，而不需要再次编写相同的代码。在令子类别继承父类别的同时，可以重新定义某些属性，并重写某些方法，即覆盖父类别的原有属性和方法，使其获得与父类别不同的功能。另外，为子类别追加新的属性和方法也是常见的做法。 （源自维基百科）

由上面对继承的表述，可以简单总结出继承的意图或者好处：

*   可以实现代码重用，但不是仅仅实现代码重用，有时候根本就没有重用
*   实现属性和方法继承

诚然，以上也不是全部，随着后续学习，对继承的认识会更深刻。好友令狐虫曾经这样总结继承：

> 从技术上说，OOP 里，继承最主要的用途是实现多态。对于多态而言，重要的是接口继承性，属性和行为是否存在继承性，这是不一定的。事实上，大量工程实践表明，重度的行为继承会导致系统过度复杂和臃肿，反而会降低灵活性。因此现在比较提倡的是基于接口的轻度继承理念。这种模型里因为父类（接口类）完全没有代码，因此根本谈不上什么代码复用了。
> 
> 在 Python 里，因为存在 Duck Type，接口定义的重要性大大的降低，继承的作用也进一步的被削弱了。
> 
> 另外，从逻辑上说，继承的目的也不是为了复用代码，而是为了理顺关系。

他是大牛，或许读者感觉比较高深，没关系，随着你的实践经验的积累，你也能对这个问题有自己独到的见解。

或许你也要问我的观点是什么？我的观点就是：走着瞧！怎么理解？继续向下看，只有你先深入这个问题，才能跳到更高层看这个问题。小马过河的故事还记得吧？只有亲自走入河水中，才知道河水的深浅。

对于 Python 中的继承，前面一直在使用，那就是我们写的类都是新式类，所有新式类都是继承自 object 类。不要忘记，新式类的一种写法：

```py
class NewStyle(object):
    pass 
```

这就是典型的继承。

### 基本概念

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type

class Person:
    def speak(self):
        print "I love you."

    def setHeight(self, n):
        self.length = n

    def breast(self, n):
        print "My breast is: ",n

class Girl(Person):
    def setHeight(self):
        print "The height is:1.70m ."

if __name__ == "__main__":
    cang = Girl()
    cang.setHeight()
    cang.speak()
    cang.breast(90) 
```

上面这个程序，保存之后运行：

```py
$ python 20901.py 
The height is:1.70m .
I love you.
My breast is:  90 
```

对以上程序进行解释，从中体会继承的概念和方法。

首先定义了一个类 Person，在这个类中定义了三个方法。注意，没有定义初始化函数，初始化函数在类中不是必不可少的。

然后又定义了一个类 Girl，这个类的名字后面的括号中，是上一个类的名字，这就意味着 Girl 继承了 Person，Girl 是 Person 的子类，Person 是 Girl 的父类。

既然是继承了 Person，那么 Girl 就全部拥有了 Person 中的方法和属性（上面的例子虽然没有列出属性）。但是，如果 Girl 里面有一个和 Person 同样名称的方法，那么就把 Person 中的同一个方法遮盖住了，显示的是 Girl 中的方法，这叫做方法的**重写**。

实例化类 Girl 之后，执行实例方法 `cang.setHeight()`，由于在类 Girl 中重写了 setHeight 方法，那么 Person 中的那个方法就不显作用了，在这个实例方法中执行的是类 Girl 中的方法。

虽然在类 Girl 中没有看到 speak 方法，但是因为它继承了 Person，所以 `cang.speak()` 就执行类 Person 中的方法。同理 `cang.breast(90)`，它们就好像是在类 Girl 里面已经写了这两个方法一样。既然继承了，就是我的了。

### 多重继承

所谓多重继承，就是只某一个类的父类，不止一个，而是多个。比如：

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type

class Person:
    def eye(self):
        print "two eyes"

    def breast(self, n):
        print "The breast is: ",n

class Girl:
    age = 28
    def color(self):
        print "The girl is white"

class HotGirl(Person, Girl):
    pass

if __name__ == "__main__":
    kong = HotGirl()
    kong.eye()
    kong.breast(90)
    kong.color()
    print kong.age 
```

在这个程序中，前面有两个类：Person 和 Girl，然后第三个类 HotGirl 继承了这两个类，注意观察继承方法，就是在类的名字后面的括号中把所继承的两个类的名字写上。但是第三个类中什么方法也没有。

然后实例化类 HotGirl，既然继承了上面的两个类，那么那两个类的方法就都能够拿过来使用。保存程序，运行一下看看

```py
$ python 20902.py 
two eyes
The breast is:  90
The girl is white
28 
```

值得注意的是，这次在类 Girl 中，有一个 `age = 28`，在对 HotGirl 实例化之后，因为继承的原因，这个类属性也被继承到 HotGirl 中，因此通过实例属性 `kong.age` 一样能够得到该数据。

由上述两个实例，已经清楚看到了继承的特点，即将父类的方法和属性全部承接到子类中；如果子类重写了父类的方法，就使用子类的该方法，父类的被遮盖。

### 多重继承的顺序

多重继承的顺序很必要了解。比如，如果一个子类继承了两个父类，并且两个父类有同样的方法或者属性，那么在实例化子类后，调用那个方法或属性，是属于哪个父类的呢？造一个没有实际意义，纯粹为了解决这个问题的程序：

```py
#!/usr/bin/env Python
# coding=utf-8

class K1(object):
    def foo(self):
        print "K1-foo"

class K2(object):
    def foo(self):
        print "K2-foo"
    def bar(self):
        print "K2-bar"

class J1(K1, K2):
    pass

class J2(K1, K2):
    def bar(self):
        print "J2-bar"

class C(J1, J2):
    pass

if __name__ == "__main__":
    print C.__mro__
    m = C()
    m.foo()
    m.bar() 
```

这段代码，保存后运行：

```py
$ python 20904.py 
(<class '__main__.C'>, <class '__main__.J1'>, <class '__main__.J2'>, <class '__main__.K1'>, <class '__main__.K2'>, <type 'object'>)
K1-foo
J2-bar 
```

代码中的 `print C.__mro__`是要打印出类的继承顺序。从上面清晰看出来了。如果要执行 foo() 方法，首先看 J1，没有，看 J2，还没有，看 J1 里面的 K1，有了，即 C==>J1==>J2==>K1；bar() 也是按照这个顺序，在 J2 中就找到了一个。

这种对继承属性和方法搜索的顺序称之为“广度优先”。

新式类用以及 Python3.x 中都是按照此顺序原则搜寻属性和方法的。

但是，在旧式类中，是按照“深度优先”的顺序的。因为后面读者也基本不用旧式类，所以不举例。如果读者愿意，可以自己模仿上面代码，探索旧式类的“深度优先”含义。

### super 函数

对于初始化函数的继承，跟一般方法的继承，还有点不同。可以看下面的例子：

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type

class Person:
    def __init__(self):
        self.height = 160

    def about(self, name):
        print "{} is about {}".format(name, self.height)

class Girl(Person):
    def __init__(self):
        self.breast = 90

    def about(self, name):
        print "{} is a hot girl, she is about {}, and her breast is {}".format(name, self.height, self.breast)

if __name__ == "__main__":
    cang = Girl()
    cang.about("wangguniang") 
```

在上面这段程序中，类 Girl 继承了类 Person。在类 Girl 中，初始化设置了 `self.breast = 90`，由于继承了 Person，按照前面的经验，Person 的初始化函数中的 `self.height = 160` 也应该被 Girl 所继承过来。然后在重写的 about 方法中，就是用 `self.height`。

实例化类 Girl，并执行 `cang.about("wangguniang")`，试图打印出一句话 `wangguniang is a hot girl, she is about 160, and her bereast is 90`。保存程序，运行之：

```py
$ python 20903.py 
Traceback (most recent call last):
  File "20903.py", line 22, in <module>
    cang.about("wangguniang")
  File "20903.py", line 18, in about
    print "{} is a hot girl, she is about {}, and her breast is {}".format(name, self.height, self.breast)
AttributeError: 'Girl' object has no attribute 'height' 
```

报错！

程序员有一句名言：不求最好，但求报错。报错不是坏事，是我们长经验的时候，是在告诉我们，那么做不对。

重要的是看报错信息。就是我们要打印的那句话出问题了，报错信息显示 `self.height` 是不存在的。也就是说类 Girl 没有从 Person 中继承过来这个属性。

原因是什么？仔细观察类 Girl，会发现，除了刚才强调的 about 方法重写了,`__init__`方法，也被重写了。不要认为它的名字模样奇怪，就不把它看做类中的方法（函数），它跟类 Person 中的`__init__`重名了，也同样是重写了那个初始化函数。

这就提出了一个问题。因为在子类中重写了某个方法之后，父类中同样的方法被遮盖了。那么如何再把父类的该方法调出来使用呢？纵然被遮盖了，应该还是存在的，不要浪费了呀。

Python 中有这样一种方法，这种方式是被提倡的方法：super 函数。

```py
#!/usr/bin/env python
# coding=utf-8

__metaclass__ = type

class Person:
    def __init__(self):
        self.height = 160

    def about(self, name):
        print "{} is about {}".format(name, self.height)

class Girl(Person):
    def __init__(self):
        super(Girl, self).__init__()
        self.breast = 90

    def about(self, name):
        print "{} is a hot girl, she is about {}, and her breast is {}".format(name, self.height, self.breast)
        super(Girl, self).about(name)

if __name__ == "__main__":
    cang = Girl()
    cang.about("wangguniang") 
```

在子类中，`__init__`方法重写了，为了调用父类同方法，使用 `super(Girl, self).__init__()`的方式。super 函数的参数，第一个是当前子类的类名字，第二个是 self，然后是点号，点号后面是所要调用的父类的方法。同样在子类重写的 about 方法中，也可以调用父类的 about 方法。

执行结果：

```py
$ python 20903.py 
wangguniang is a hot girl, she is about 160, and her breast is 90
wangguniang is about 160 
```

最后要提醒注意：super 函数仅仅适用于新式类。当然，你一定是使用的新式类。“喜新厌旧”是程序员的嗜好。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 类(5)

在前面几节讨论类的时候，经常要将类实例化，然后通过实例来调用类的方法（函数）。在此，把前面经常做的这类事情概括一下：

*   方法是类内部定义函数，只不过这个函数的第一个参数是 self。（可以认为方法是类属性，但不是实例属性）
*   必须将类实例化之后，才能通过实例调用该类的方法。调用的时候在方法后面要跟括号（括号中默认有 self 参数，但是不写出来。）

通过实例调用方法（在前面曾用了一个不严谨的词语：实例方法），我们称这个方法**绑定**在实例上。

### 调用绑定方法

前面一直在这样做。比如：

```py
class Person(object):
    def foo(self):
        pass 
```

如果要调用 Person.foo() 方法，必须：

```py
pp = Person()    #实例化
pp.foo() 
```

这样就实现了方法和实例的绑定，于是通过 `pp.foo()` 即可调用该方法。

### 调用非绑定方法

在《类(4)》中，介绍了一个函数 super。为了描述方便，把代码复制过来：

```py
#!/usr/bin/env python
# coding=utf-8

__metaclass__ = type

class Person:
    def __init__(self):
        self.height = 160

    def about(self, name):
        print "{} is about {}".format(name, self.height)

class Girl(Person):
    def __init__(self):
        super(Girl, self).__init__()
        self.breast = 90

    def about(self, name):
        print "{} is a hot girl, she is about {}, and her breast is {}".format(name, self.height, self.breast)
        super(Girl, self).about(name)

if __name__ == "__main__":
    cang = Girl()
    cang.about("wangguniang") 
```

在子类 Girl 中，因为重写了父类的`__init__`方法，如果要调用父类该方法，在上节中不得不使用 `super(Girl, self).__init__()`调用父类中因为子类方法重写而被遮蔽的同名方法。

其实，在子类中，父类的方法就是**非绑定方法**，因为在子类中，没有建立父类的实例，却要是用父类的方法。对于这种非绑定方法的调用，还有一种方式。不过这种方式现在已经较少是用了，因为有了 super 函数。为了方便读者看其它有关代码，还是要简要说明。

例如在上面代码中，在类 Girl 中想调用父类 Person 的初始化函数，则需要在子类中，写上这么一行：

```py
Person.__init__(self) 
```

这不是通过实例调用的，而是通过类 Person 实现了对`__init__(self)`的调用。这就是调用非绑定方法的用途。但是，这种方法已经被 super 函数取代，所以，如果读者在编程中遇到类似情况，推荐使用 super 函数。

### 静态方法和类方法

已知，类的方法第一个参数必须是 self，并且如果要调用类的方法，必须将通过类的实例，即方法绑定实例后才能由实例调用。如果不绑定，一般在继承关系的类之间，可以用 super 函数等方法调用。

这里再介绍一种方法，这种方法的调用方式跟上述的都不同，这就是：静态方法和类方法。看代码：

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type

class StaticMethod:
    @staticmethod
    def foo():
        print "This is static method foo()."

class ClassMethod:
    @classmethod
    def bar(cls):
        print "This is class method bar()."
        print "bar() is part of class:", cls.__name__

if __name__ == "__main__":
    static_foo = StaticMethod()    #实例化
    static_foo.foo()               #实例调用静态方法
    StaticMethod.foo()             #通过类来调用静态方法
    print "********"
    class_bar = ClassMethod()
    class_bar.bar()
    ClassMethod.bar() 
```

对于这部分代码，有一处非常特别，那就是包含了“@”符号。在 Python 中：

*   `@staticmethod`表示下面的方法是静态方法
*   `@classmethod`表示下面的方法是类方法

一个一个来看。

先看静态方法，虽然名为静态方法，但也是方法，所以，依然用 def 语句来定义。需要注意的是文件名后面的括号内，没有 self，这和前面定义的类中的方法是不同的，也正是因着这个不同，才给它另外取了一个名字叫做静态方法，否则不就“泯然众人矣”。如果没有 self，那么也就无法访问实例变量、类和实例的属性了，因为它们都是借助 self 来传递数据的。

在看类方法，同样也具有一般方法的特点，区别也在参数上。类方法的参数也没有 self，但是必须有 cls 这个参数。在类方法中，能够方法类属性，但是不能访问实例属性（读者可以自行设计代码检验之）。

简要明确两种方法。下面看调用方法。两种方法都可以通过实例调用，即绑定实例。也可以通过类来调用，即 `StaticMethod.foo()` 这样的形式，这也是区别一般方法的地方，一般方法必须用通过绑定实例调用。

上述代码运行结果：

```py
$ python 21001.py 
This is static method foo().
This is static method foo().
********
This is class method bar().
bar() is part of class: ClassMethod
This is class method bar().
bar() is part of class: ClassMethod 
```

这是关于静态方法和类方法的简要介绍。

正当我思考如何讲解的更深入一点的时候，我想起了以往看过的一篇文章，觉得人家讲的非常到位。所以，不敢吝啬，更不敢班门弄斧，所以干醋把那篇文章恭恭敬敬的抄录于此。同时，读者从下面的文章中，也能对前面的知识复习一下。文章标题是：Python 中的 staticmethod 和 classmethod 的差异。原载：www.pythoncentral.io/difference-between-staticmethod-and-classmethod-in-Python/。此地址需要你准备梯子才能浏览。后经国人翻译，地址是：[`www.wklken.me/posts/2013/12/22/difference-between-staticmethod-and-classmethod-in-Python.html`](http://www.wklken.me/posts/2013/12/22/difference-between-staticmethod-and-classmethod-in-Python.html)

以下是翻译文章：

#### Class vs static methods in Python

这篇文章试图解释：什么事 staticmethod/classmethod,并且这两者之间的差异.

staticmethod 和 classmethod 均被作为装饰器，用作定义一个函数为"staticmethod"还是"classmethod"

如果想要了解 Python 装饰器的基础，可以看[这篇文章](http://www.pythoncentral.io/python-decorators-overview/)

#### Simple, static and class methods

类中最常用到的方法是 实例方法(instance methods), 即，实例对象作为第一个参数传递给函数

例如，下面是一个基本的实例方法

```py
class Kls(object):
    def __init__(self, data):
        self.data = data

    def printd(self):
        print(self.data)

ik1 = Kls('arun')
ik2 = Kls('seema')

ik1.printd()
ik2.printd() 
```

得到的输出:

```py
arun
seema 
```

调用关系图:

![](img/21001.png)

查看代码和图解:

> 1/2 参数传递给函数
> 
> 3 self 参数指向实例本身
> 
> 4 我们不需要显式提供实例，解释器本身会处理

假如我们想仅实现类之间交互而不是通过实例？我们可以在类之外建立一个简单的函数来实现这个功能，但是将会使代码扩散到类之外，这个可能对未来代码维护带来问题。

例如：

```py
def get_no_of_instances(cls_obj):
    return cls_obj.no_inst

class Kls(object):
    no_inst = 0

    def __init__(self):
        Kls.no_inst = Kls.no_inst + 1

ik1 = Kls()
ik2 = Kls()

print(get_no_of_instances(Kls)) 
```

结果：

```py
2 
```

#### The Python @classmethod

现在我们要做的是在类里创建一个函数，这个函数参数是类对象而不是实例对象.

在上面那个实现中，如果要实现不获取实例,需要修改如下:

```py
def iget_no_of_instance(ins_obj):
    return ins_obj.__class__.no_inst

class Kls(object):
    no_inst = 0

    def __init__(self):
        Kls.no_inst = Kls.no_inst + 1

ik1 = Kls()
ik2 = Kls()
print iget_no_of_instance(ik1) 
```

结果

```py
2 
```

可以使用 Python2.2 引入的新特性，使用 @classmethod 在类代码中创建一个函数

```py
class Kls(object):
    no_inst = 0

    def __init__(self):
        Kls.no_inst = Kls.no_inst + 1

    @classmethod
    def get_no_of_instance(cls_obj):
        return cls_obj.no_inst

ik1 = Kls()
ik2 = Kls()

print ik1.get_no_of_instance()
print Kls.get_no_of_instance() 
```

We get the following output:

```py
2
2 
```

#### The Python @staticmethod

通常，有很多情况下一些函数与类相关，但不需要任何类或实例变量就可以实现一些功能.

比如设置环境变量，修改另一个类的属性等等.这种情况下，我们也可以使用一个函数，一样会将代码扩散到类之外（难以维护）

下面是一个例子:

```py
IND = 'ON'

def checkind():
    return (IND == 'ON')

class Kls(object):
    def __init__(self,data):
        self.data = data

    def do_reset(self):
        if checkind():
            print('Reset done for:', self.data)

    def set_db(self):
        if checkind():
            self.db = 'new db connection'
            print('DB connection made for:',self.data)

ik1 = Kls(12)
ik1.do_reset()
ik1.set_db() 
```

结果:

```py
Reset done for: 12
DB connection made for: 12 
```

现在我们使用 @staticmethod, 我们可以将所有代码放到类中

```py
IND = 'ON'

class Kls(object):
    def __init__(self, data):
        self.data = data

    @staticmethod
    def checkind():
        return (IND == 'ON')

    def do_reset(self):
        if self.checkind():
            print('Reset done for:', self.data)

    def set_db(self):
        if self.checkind():
            self.db = 'New db connection'
        print('DB connection made for: ', self.data)

ik1 = Kls(12)
ik1.do_reset()
ik1.set_db() 
```

得到的结果:

```py
Reset done for: 12
DB connection made for: 12 
```

#### How @staticmethod and @classmethod are different

```py
class Kls(object):
    def __init__(self, data):
        self.data = data

    def printd(self):
        print(self.data)

    @staticmethod
    def smethod(*arg):
        print('Static:', arg)

    @classmethod
    def cmethod(*arg):
        print('Class:', arg) 
```

调用

```py
>>> ik = Kls(23)
>>> ik.printd()
23
>>> ik.smethod()
Static: ()
>>> ik.cmethod()
Class: (<class '__main__.Kls'>,)
>>> Kls.printd()
TypeError: unbound method printd() must be called with Kls instance as first argument (got nothing instead)
>>> Kls.smethod()
Static: ()
>>> Kls.cmethod()
Class: (<class '__main__.Kls'>,) 
```

图解

![](img/21002.png)

### 文档字符串

在写程序的时候，必须要写必要的文字说明，没别的原因，除非你的代码写的非常容易理解，特别是各种变量、函数和类等的命名任何人都能够很容易理解，否则，文字说明是不可缺少的。

在函数、类或者文件开头的部分写文档字符串说明，一般采用三重引号。这样写的最大好处是能够用 help() 函数看。

```py
"""This is python lesson"""

def start_func(arg):
    """This is a function."""
    pass

class MyClass:
    """Thi is my class."""
    def my_method(self,arg):
        """This is my method."""
        pass 
```

这样的文档是必须的。

当然，在编程中，有不少地方要用“#”符号来做注释。一般用这个来注释局部。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 多态和封装

前面讲过的“继承”，是类的一个重要特征，在编程中用途很多。这里要说两个在理解和实践上有争议的话题：多态和封装。所谓争议，多来自于对同一个现象不同角度的理解，特别是有不少经验丰富的程序员，还从其它语言的角度来诠释 Python 的多态等。

### 多态

在网上搜索一下，发现对 Python 的多态问题，的确是仁者见仁智者见智。

作为一个初学者，不一定要也没有必要、或者还没有能力参与这种讨论。但是，应该理解 Python 中关于多态的基本体现，也要对多态有一个基本的理解。

```py
>>> "This is a book".count("s")
2
>>> [1,2,4,3,5,3].count(3)
2 
```

上面的 `count()` 的作用是数一数某个元素在对象中出现的次数。从例子中可以看出，我们并没有限定 count 的参数。类似的例子还有：

```py
>>> f = lambda x,y:x+y 
```

还记得这个 lambda 函数吗？如果忘记了，请复习[函数(4)href="https://github.com/qiwsir/StarterLearningPython/blob/master/204.md")中对此的解释。

```py
>>> f(2,3)
5
>>> f("qiw","sir")
'qiwsir'
>>> f(["python","java"],["c++","lisp"])
['python', 'java', 'c++', 'lisp'] 
```

在那个 lambda 函数中，我们没有限制参数的类型，也一定不能限制，因为如果限制了，就不是 Pythonic 了。在使用的时候，可以给参数任意类型，都能到的不报错的结果。当然，这样做之所以合法，更多的是来自于 `+` 的功能强悍。

以上，就体现了“多态”。当然，也有人就此提出了反对意见，因为本质上是在参数传入值之前，Python 并没有确定参数的类型，只能让数据进入函数之后再处理，能处理则罢，不能处理就报错。例如：

```py
>>> f("qiw", 2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in <lambda>
TypeError: cannot concatenate 'str' and 'int' objects 
```

本教程由于不属于这种概念争论范畴，所以不进行这方面的深入探索，仅仅是告诉各位读者相关信息。并且，本教程也是按照“人云亦云”的原则，既然大多数程序员都在讨论多态，那么我们就按照大多数人说的去介绍（尽管有时候真理掌握在少数人手中）。

“多态”，英文是:Polymorphism，在台湾被称作“多型”。维基百科中对此有详细解释说明。

> 多型（英语：Polymorphism），是指物件导向程式执行时，相同的讯息可能会送給多个不同的类別之物件，而系统可依剧物件所属类別，引发对应类別的方法，而有不同的行为。简单来说，所谓多型意指相同的讯息給予不同的物件会引发不同的动作称之。

再简化的说法就是“有多种形式”，就算不知道变量（参数）所引用的对象类型，也一样能进行操作，来者不拒。比如上面显示的例子。在 Python 中，更为 Pthonic 的做法是根本就不进行类型检验。

例如著名的 `repr()` 函数，它能够针对输入的任何对象返回一个字符串。这就是多态的代表之一。

```py
>>> repr([1,2,3])
'[1, 2, 3]'
>>> repr(1)
'1'
>>> repr({"lang":"python"})
"{'lang': 'Python'}" 
```

使用它写一个小函数，还是作为多态代表的。

```py
>>> def length(x):
...     print "The length of", repr(x), "is", len(x)
... 

>>> length("how are you")
The length of 'how are you' is 11
>>> length([1,2,3])
The length of [1, 2, 3] is 3
>>> length({"lang":"python","book":"itdiffer.com"})
The length of {'lang': 'python', 'book': 'itdiffer.com'} is 2 
```

不过，多态也不是万能的，如果这样做：

```py
>>> length(7)
The length of 7 is
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in length
TypeError: object of type 'int' has no len() 
```

报错了。看错误提示，明确告诉了我们 `object of type 'int' has no len()`。

在诸多介绍多态的文章中，都会有这样关于猫和狗的例子。这里也将代码贴出来，读者去体会所谓多态体现。其实，如果你进入了 Python 的语境，有时候是不经意就已经在应用多态特性呢。

```py
#!/usr/bin/env Python
# coding=utf-8

"the code is from: http://zetcode.com/lang/python/oop/"

__metaclass__ = type

class Animal:
    def __init__(self, name=""):
        self.name = name

    def talk(self):
        pass

class Cat(Animal):
    def talk(self):
        print "Meow!"

class Dog(Animal):
    def talk(self):
        print "Woof!"

a = Animal()
a.talk()

c = Cat("Missy")
c.talk()

d = Dog("Rocky")
d.talk() 
```

保存后运行之：

```py
$ python 21101.py 
Meow!
Woof! 
```

代码中有 Cat 和 Dog 两个类，都继承了类 Animal，它们都有 `talk()` 方法，输入不同的动物名称，会得出相应的结果。

关于多态，有一个被称作“鸭子类型”(duck typeing)的东西，其含义在维基百科中被表述为：

> 在程序设计中，鸭子类型（英语：duck typing）是动态类型的一种风格。在这种风格中，一个对象有效的语义，不是由继承自特定的类或实现特定的接口，而是由当前方法和属性的集合决定。这个概念的名字来源于由 James Whitcomb Riley 提出的鸭子测试（见下面的“历史”章节），“鸭子测试”可以这样表述：“当看到一只鸟走起来像鸭子、游泳起来像鸭子、叫起来也像鸭子，那么这只鸟就可以被称为鸭子。”

对于鸭子类型，也是有争议的。这方面的详细信息，读者可以去看有关维基百科的介绍。

对于多态问题，最后还要告诫读者，类型检查是毁掉多态的利器，比如 type、isinstance 以及 isubclass 函数，所以，一定要慎用这些类型检查函数。

### 封装和私有化

在正式介绍封装之前，先扯个笑话。

> 某软件公司老板，号称自己懂技术。一次有一个项目要交付给客户，但是他有不想让客户知道实现某些功能的代码，但是交付的时候要给人家代码的。于是该老板就告诉程序员，“你们把那部分核心代码封装一下”。程序员听了之后，迷茫了。

不知道你有没有笑。

“封装”，是不是把代码写到某个东西里面，“人”在编辑器中打开，就看不到了呢？除非是你的显示器坏了。

在程序设计中，封装(Encapsulation)是对 object 的一种抽象，即将某些部分隐藏起来，在程序外部看不到，即无法调用（不是人用眼睛看不到那个代码，除非用某种加密或者混淆方法，造成现实上的困难，但这不是封装）。

要了解封装，离不开“私有化”，就是将类或者函数中的某些属性限制在某个区域之内，外部无法调用。

Python 中私有化的方法也比较简单，就是在准备私有化的属性（包括方法、数据）名字前面加双下划线。例如：

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type

class ProtectMe:
    def __init__(self):
        self.me = "qiwsir"
        self.__name = "kivi"

    def __python(self):
        print "I love Python."

    def code(self):
        print "Which language do you like?"
        self.__python()

if __name__ == "__main__":
    p = ProtectMe()
    print p.me
    print p.__name 
```

运行一下，看看效果：

```py
$ python 21102.py
qiwsir
Traceback (most recent call last):
  File "21102.py", line 21, in <module>
    print p.__name
AttributeError: 'ProtectMe' object has no attribute '__name' 
```

查看报错信息，告诉我们没有`__name` 那个属性。果然隐藏了，在类的外面无法调用。再试试那个函数，可否？

```py
if __name__ == "__main__":
    p = ProtectMe()
    p.code()
    p.__python() 
```

修改这部分即可。其中 `p.code()` 的意图是要打印出两句话：`"Which language do you like?"`和`"I love Python."`，`code()` 方法和`__python()` 方法在同一个类中，可以调用之。后面的那个 `p.__Python()` 试图调用那个私有方法。看看效果：

```py
$ python 21102.py 
Which language do you like?
I love Python.
Traceback (most recent call last):
  File "21102.py", line 23, in <module>
    p.__python()
AttributeError: 'ProtectMe' object has no attribute '__python' 
```

如愿以偿。该调用的调用了，该隐藏的隐藏了。

用上面的方法，的确做到了封装。但是，我如果要调用那些私有属性，怎么办？

可以使用 `property` 函数。

```py
#!/usr/bin/env Python
# coding=utf-8

__metaclass__ = type

class ProtectMe:
    def __init__(self):
        self.me = "qiwsir"
        self.__name = "kivi"

    @property
    def name(self):
        return self.__name

if __name__ == "__main__":
    p = ProtectMe()
    print p.name 
```

运行结果：

```py
$ python 21102.py 
kivi 
```

从上面可以看出，用了 `@property` 之后，在调用那个方法的时候，用的是 `p.name` 的形式，就好像在调用一个属性一样，跟前面 `p.me` 的格式相同。

看来，封装的确不是让“人看不见”。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 特殊方法 (1)

探究更多的类属性，在一些初学者的教程中，一般很少见。我之所以要在这里也将这部分奉献出来，就是因为本教程是“From Beginner to Master”。当然，不是学习了类的更多属性就能达到 Master 水平，但是这是通往 Master 的一步，虽然在初级应用中，本节乃至于后面关于类的属性用的不很多，但是，这一步迈出去，你就会在实践中有一个印象，以后需要用到了，知道有这一步，会对项目有帮助的。俗话说“艺不压身”。

### `__dict__`

前面已经学习过有关类属性和实例属性的内容，并且做了区分，如果忘记了可以回头参阅《类(3)》中的“类属性和实例属性”部分。有一个结论，是一定要熟悉的，那就是可以通过 `object.attribute` 的方式访问对象的属性。

如果接着那部分内容，读者是否思考过一个问题：类或者实例属性，在 Python 中是怎么存储的？或者为什么修改或者增加、删除属性，我们能不能控制这些属性？

```py
>>> class A(object):
...     pass
...

>>> a = A()
>>> dir(a)
['__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__']
>>> dir(A)
['__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__'] 
```

用 `dir()` 来查看一下，发现不管是类还是实例，都有很多属性，这在前面已经反复出现，有点见怪不怪了。不过，这里我们要看一个属性：`__dict__`，因为它是一个保存秘密的东西：对象的属性。

```py
>>> class Spring(object):
...     season = "the spring of class"
... 

>>> Spring.__dict__
dict_proxy({'__dict__': <attribute '__dict__' of 'Spring' objects>, 
'season': 'the spring of class', 
'__module__': '__main__', 
'__weakref__': <attribute '__weakref__' of 'Spring' objects>, 
'__doc__': None}) 
```

为了便于观察，我将上面的显示结果进行了换行，每个键值对一行。

对于类 Spring 的`__dict__`属性，可以发现，有一个键`'season'`，这就是这个类的属性；其值就是类属性的数据。

```py
>>> Spring.__dict__['season']
'the spring of class'
>>> Spring.season
'the spring of class' 
```

用这两种方式都能得到类属性的值。或者说 `Spring.__dict__['season']` 就是访问类属性。下面将这个类实例化，再看看它的实例属性：

```py
>>> s = Spring()
>>> s.__dict__
{} 
```

实例属性的`__dict__`是空的。有点奇怪？不奇怪，接着看：

```py
>>> s.season
'the spring of class' 
```

这个其实是指向了类属性中的 `Spring.season`，至此，我们其实还没有建立任何实例属性呢。下面就建立一个实例属性：

```py
>>> s.season = "the spring of instance"
>>> s.__dict__
{'season': 'the spring of instance'} 
```

这样，实例属性里面就不空了。这时候建立的实例属性和上面的那个 `s.season` 只不过重名，并且把它“遮盖”了。这句好是不是熟悉？因为在讲述“实例属性”和“类属性”的时候就提到了。现在读者肯定理解更深入了。

```py
>>> s.__dict__['season']
'the spring of instance'
>>> s.season
'the spring of instance' 
```

此时，那个类属性如何？我们看看：

```py
>>> Spring.__dict__['season']
'the spring of class'
>>> Spring.__dict__
dict_proxy({'__dict__': <attribute '__dict__' of 'Spring' objects>, 'season': 'the spring of class', '__module__': '__main__', '__weakref__': <attribute '__weakref__' of 'Spring' objects>, '__doc__': None})
>>> Spring.season
'the spring of class' 
```

Spring 的类属性没有受到实例属性的影响。

按照前面的讲述类属性和实例熟悉的操作，如果这时候将前面的实例属性删除，会不会回到实例属性`s.__dict__`为空呢？

```py
>>> del s.season
>>> s.__dict__
{}
>>> s.season
'the spring of class' 
```

果然打回原形。

当然，你可以定义其它名称的实例属性，它一样被存储到`__dict__`属性里面：

```py
>>> s.lang = "python"
>>> s.__dict__
{'lang': 'python'}
>>> s.__dict__['lang']
'python' 
```

诚然，这样做仅仅是更改了实例的`__dict__`内容，对 `Spring.__dict__`无任何影响，也就是说通过 `Spring.lang` 或者 `Spring.__dict__['lang']` 是得不到上述结果的。

```py
>>> Spring.lang
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: type object 'Spring' has no attribute 'lang'

>>> Spring.__dict__['lang']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'lang' 
```

那么，如果这样操作，会怎样呢？

```py
>>> Spring.flower = "peach"
>>> Spring.__dict__
dict_proxy({'__module__': '__main__', 
'flower': 'peach', 
'season': 'the spring of class', 
'__dict__': <attribute '__dict__' of 'Spring' objects>, '__weakref__': <attribute '__weakref__' of 'Spring' objects>, '__doc__': None})
>>> Spring.__dict__['flower']
'peach' 
```

在类的`__dict__`被更改了，类属性中增加了一个'flower'属性。但是，实例的`__dict__`中如何？

```py
>>> s.__dict__
{'lang': 'python'} 
```

没有被修改。我也是这么想的，哈哈。你此前这这么觉得吗？然而，还能这样：

```py
>>> s.flower
'peach' 
```

这个读者是否能解释？其实又回到了前面第一个出现 `s.season` 上面了。

通过上面探讨，是不是基本理解了实例和类的`__dict__`，并且也看到了属性的变化特点。特别是，这些属性都是可以动态变化的，就是你可以随时修改和增删。

属性如此，方法呢？下面就看看方法（类中的函数）。

```py
>>> class Spring(object):
...     def tree(self, x):
...         self.x = x
...         return self.x
... 
>>> Spring.__dict__
dict_proxy({'__dict__': <attribute '__dict__' of 'Spring' objects>, 
'__weakref__': <attribute '__weakref__' of 'Spring' objects>, 
'__module__': '__main__', 
'tree': <function tree at 0xb748fdf4>, 
'__doc__': None})

>>> Spring.__dict__['tree']
<function tree at 0xb748fdf4> 
```

结果跟前面讨论属性差不多，方法 `tree` 也在`__dict__`里面呢。

```py
>>> t = Spring()
>>> t.__dict__
{} 
```

又跟前面一样。虽然建立了实例，但是在实例的`__dict__`中没有方法。接下来，执行：

```py
>>> t.tree("xiangzhangshu")
'xiangzhangshu' 
```

在类(3)中有一部分内容阐述“数据流转”，其中有一张图，其中非常明确显示出，当用上面方式执行方法的时候，实例 `t` 与 `self` 建立了对应关系，两者是一个外一个内。在方法中 `self.x = x`，将 x 的值给了 self.x，也就是实例应该拥有了这么一个属性。

```py
>>> t.__dict__
{'x': 'xiangzhangshu'} 
```

果然如此。这也印证了实例 `t` 和 `self` 的关系，即实例方法(`t.tree('xiangzhangshu')`)的第一个参数(self，但没有写出来)绑定实例 t，透过 self.x 来设定值，即给 `t.__dict__`添加属性值。

换一个角度：

```py
>>> class Spring(object):
...     def tree(self, x):
...         return x
... 
```

这回方法中没有将 x 赋值给 self 的属性，而是直接 return，结果是：

```py
>>> s = Spring()
>>> s.tree("liushu")
'liushu'
>>> s.__dict__
{} 
```

是不是理解更深入了？

现在需要对 Python 中一个观点：“一切皆对象”，再深入领悟。以上不管是类还是的实例的属性和方法，都是符合 `object.attribute` 格式，并且属性类似。

当你看到这里的时候，要么明白了类和实例的`__dict__`的特点，要么就糊涂了。糊涂也不要紧，再将上面的重复一遍，特别是自己要敲一敲有关代码。（建议一个最好的方法：用两个显示器，一个显示器看本教程，另外一个显示器敲代码。事半功倍的效果。）

需要说明，我们对`__dict__`的探讨还留有一个尾巴：属性搜索路径。这个留在后面讲述。

不管是类还是实例，其属性都能随意增加。这点在有时候不是一件好事情，或许在某些时候你不希望别人增加属性。有办法吗？当然有，请继续学习。

### `__slots__`

首先声明，`__slots__`能够限制属性的定义，但是这不是它存在终极目标，它存在的终极目标更应该是一个在编程中非常重要的方面：**优化内存使用。**

```py
>>> class Spring(object):
...     __slots__ = ("tree", "flower")
... 
>>> dir(Spring)
['__class__', '__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', 'flower', 'tree'] 
```

仔细看看 `dir()` 的结果，还有`__dict__`属性吗？没有了，的确没有了。也就是说`__slots__`把`__dict__`挤出去了，它进入了类的属性。

```py
>>> Spring.__slots__
('tree', 'flower') 
```

这里可以看出，类 Spring 有且仅有两个属性。

```py
>>> t = Spring()
>>> t.__slots__
('tree', 'flower') 
```

实例化之后，实例的`__slots__`与类的完全一样，这跟前面的`__dict__`大不一样了。

```py
>>> Spring.tree = "liushu" 
```

通过类，先赋予一个属性值。然后，检验一下实例能否修改这个属性：

```py
>>> t.tree = "guangyulan"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Spring' object attribute 'tree' is read-only 
```

看来，我们的意图不能达成，报错信息中显示，`tree` 这个属性是只读的，不能修改了。

```py
>>> t.tree
'liushu' 
```

因为前面已经通过类给这个属性赋值了。不能用实例属性来修改。只能：

```py
>>> Spring.tree = "guangyulan"
>>> t.tree
'guangyulan' 
```

用类属性修改。但是对于没有用类属性赋值的，可以通过实例属性：

```py
>>> t.flower = "haitanghua"
>>> t.flower
'haitanghua' 
```

但此时：

```py
>>> Spring.flower
<member 'flower' of 'Spring' objects> 
```

实例属性的值并没有传回到类属性，你也可以理解为新建立了一个同名的实例属性。如果再给类属性赋值，那么就会这样了：

```py
>>> Spring.flower = "ziteng"
>>> t.flower
'ziteng' 
```

当然，此时在给 `t.flower` 重新赋值，就会爆出跟前面一样的错误了。

```py
>>> t.water = "green"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Spring' object has no attribute 'water' 
```

这里试图给实例新增一个属性，也失败了。

看来`__slots__`已经把实例属性牢牢地管控了起来，但更本质是的是优化了内存。诚然，这种优化会在大量的实例时候显出效果。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 特殊方法 (2)

书接上回，不管是实例还是类，都用`__dict__`来存储属性和方法，可以笼统地把属性和方法称为成员或者特性，用一句笼统的话说，就是`__dict__`存储对象成员。但，有时候访问的对象成员没有存在其中，就是这样：

```py
>>> class A(object):
...     pass
... 
>>> a = A()
>>> a.x
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'A' object has no attribute 'x' 
```

`x` 不是实例的成员，用 `a.x` 访问，就出错了，并且错误提示中报告了原因：“'A' object has no attribute 'x'”

在很多情况下，这种报错是足够的了。但是，在某种我现在还说不出的情况下，你或许不希望这样报错，或许希望能够有某种别的提示、操作等。也就是我们更希望能在成员不存在的时候有所作为，不是等着报错。

要处理类似的问题，就要用到本节中的知识了。

### `__getattr__`、`__setattr__`和其它类似方法

还是用上面的例子，如果访问 `a.x`，它不存在，那么就要转向到某个操作。我们把这种情况称之为“拦截”。就好像“寻隐者不遇”，却被童子“遥指杏花村”，将你“拦截”了。在 Python 中，有一些方法就具有这种“拦截”能力。

*   `__setattr__(self,name,value)`：如果要给 name 赋值，就调用这个方法。
*   `__getattr__(self,name)`：如果 name 被访问，同时它不存在的时候，此方法被调用。
*   `__getattribute__(self,name)`：当 name 被访问时自动被调用（注意：这个仅能用于新式类），无论 name 是否存在，都要被调用。
*   `__delattr__(self,name)`：如果要删除 name，这个方法就被调用。

如果一时没有理解，不要紧，是正常的。需要用例子说明。

```py
>>> class A(object):
...     def __getattr__(self, name):
...         print "You use getattr"
...     def __setattr__(self, name, value):
...         print "You use setattr"
...         self.__dict__[name] = value
... 
```

类 A 是新式类，除了两个方法，没有别的属性。

```py
>>> a = A()
>>> a.x
You use getattr 
```

`a.x`，按照本节开头的例子，是要报错的。但是，由于在这里使用了`__getattr__(self, name)` 方法，当发现 `x` 不存在于对象的`__dict__`中的时候，就调用了`__getattr__`，即所谓“拦截成员”。

```py
>>> a.x = 7
You use setattr 
```

给对象的属性赋值时候，调用了`__setattr__(self, name, value)`方法，这个方法中有一句 `self.__dict__[name] = value`，通过这个语句，就将属性和数据保存到了对象的`__dict__`中，如果在调用这个属性：

```py
>>> a.x
7 
```

它已经存在于对象的`__dict__`之中。

在上面的类中，当然可以使用`__getattribute__(self, name)`，因为它是新式类。并且，只要访问属性就会调用它。例如：

```py
>>> class B(object):
...     def __getattribute__(self, name):
...         print "you are useing getattribute"
...         return object.__getattribute__(self, name)
... 
```

为了与前面的类区分，新命名一个类名字。需要提醒注意，在这里返回的内容用的是 `return object.__getattribute__(self, name)`，而没有使用 `return self.__dict__[name]`像是。因为如果用这样的方式，就是访问 `self.__dict__`，只要访问这个属性，就要调用`**getattribute**``，这样就导致了无线递归下去（死循环）。要避免之。

```py
>>> b = B()
>>> b.y
you are useing getattribute
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 4, in __getattribute__
AttributeError: 'B' object has no attribute 'y'
>>> b.two
you are useing getattribute
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 4, in __getattribute__
AttributeError: 'B' object has no attribute 'two' 
```

访问不存在的成员，可以看到，已经被`__getattribute__`拦截了，虽然最后还是要报错的。

```py
>>> b.y = 8
>>> b.y
you are useing getattribute
8 
```

当给其赋值后，意味着已经在`__dict__`里面了，再调用，依然被拦截，但是由于已经在`__dict__`内，会把结果返回。

当你看到这里，是不是觉得上面的方法有点魔力呢？不错。但是，它有什么具体应用呢？看下面的例子，能给你带来启发。

```py
#!/usr/bin/env Python
# coding=utf-8

"""
study __getattr__ and __setattr__
"""

class Rectangle(object):
    """
    the width and length of Rectangle
    """
    def __init__(self):
        self.width = 0
        self.length = 0

    def setSize(self, size):
        self.width, self.length = size
    def getSize(self):
        return self.width, self.length

if __name__ == "__main__":
    r = Rectangle()
    r.width = 3
    r.length = 4
    print r.getSize()
    r.setSize( (30, 40) )
    print r.width
    print r.length 
```

上面代码来自《Beginning Python:From Novice to Professional,Second Edittion》（by Magnus Lie Hetland），根据本教程的需要，稍作修改。

```py
$ python 21301.py 
(3, 4)
30
40 
```

这段代码已经可以正确运行了。但是，作为一个精益求精的程序员。总觉得那种调用方式还有可以改进的空间。比如，要给长宽赋值的时候，必须赋予一个元组，里面包含长和宽。这个能不能改进一下呢？

```py
#!/usr/bin/env Python
# coding=utf-8

"""
study __getattr__ and __setattr__
"""

class Rectangle(object):
    """
    the width and length of Rectangle
    """
    def __init__(self):
        self.width = 0
        self.length = 0

    def setSize(self, size):
        self.width, self.length = size
    def getSize(self):
        return self.width, self.length

    size = property(getSize, setSize)

if __name__ == "__main__":
    r = Rectangle()
    r.width = 3
    r.length = 4
    print r.size
    r.size = 30, 40
    print r.width
    print r.length 
```

以上代码的运行结果同上。但是，因为加了一句 `size = property(getSize, setSize)`，使得调用方法是不是更优雅了呢？原来用 `r.getSize()`，现在使用 `r.size`，就好像调用一个属性一样。难道你不觉得眼熟吗？在《多态和封装》中已经用到过 property 函数了，虽然写法略有差别，但是作用一样。

本来，这样就已经足够了。但是，因为本节中出来了特殊方法，所以，一定要用这些特殊方法从新演绎一下这段程序。虽然重新演绎的不一定比原来的好，主要目的是演示本节的特殊方法应用。

```py
#!/usr/bin/env Python
# coding=utf-8

class NewRectangle(object):
    def __init__(self):
        self.width = 0
        self.length = 0

    def __setattr__(self, name, value):
        if name == "size":
            self.width, self.length = value
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        if name == "size":
            return self.width, self.length
        else:
            raise AttributeError

if __name__ == "__main__":
    r = NewRectangle()
    r.width = 3
    r.length = 4
    print r.size
    r.size = 30, 40
    print r.width
    print r.length 
```

除了类的样式变化之外，调用样式没有变。结果是一样的。

这就算了解了一些这些属性了吧。但是，有一篇文章是要必须推荐给读者阅读的：[Python Attributes and Methods](http://www.cafepy.com/article/Python_attributes_and_methods/Python_attributes_and_methods.html)，读了这篇文章，对 Python 的对象属性和方法会有更深入的理解。

### 获得属性顺序

通过实例获取其属性（也有说特性的，名词变化了，但是本质都是属性和方法），如果在`__dict__`中有相应的属性，就直接返回其结果；如果没有，会到类属性中找。比如：

```py
#!/usr/bin/env Python
# coding=utf-8

class A(object):
    author = "qiwsir"
    def __getattr__(self, name):
        if name != "author":
            return "from starter to master."

if __name__ == "__main__":
    a = A()
    print a.author
    print a.lang 
```

运行程序：

```py
$ python 21302.py 
qiwsir
from starter to master. 
```

当 `a = A()` 后，并没有为实例建立任何属性，或者说实例的`__dict__`是空的，这在上节中已经探讨过了。但是如果要查看 `a.author`，因为实例的属性中没有，所以就去类属性中找，发现果然有，于是返回其值 `"qiwsir"`。但是，在找 `a.lang` 的时候，不仅实例属性中没有，类属性中也没有，于是就调用了`__getattr__()`方法。在上面的类中，有这个方法，如果没有`__getattr__()`方法呢？如果没有定义这个方法，就会引发 AttributeError，这在前面已经看到了。

这就是通过实例查找特性的顺序。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 迭代器

迭代，对于读者已经不陌生了，曾有专门一节来讲述，如果印象不深，请复习《迭代》。

正如读者已知，对序列（列表、元组）、字典和文件都可以用 `iter()` 方法生成迭代对象，然后用 `next()` 方法访问。当然，这种访问不是自动的，如果用 for 循环，就可以自动完成上述访问了。

如果用 `dir(list)`,`dir(tuple)`,`dir(file)`,`dir(dict)` 来查看不同类型对象的属性，会发现它们都有一个名为`__iter__`的东西。这个应该引起读者的关注，因为它和迭代器（iterator）、内置的函数 iter() 在名字上是一样的，除了前后的双下划线。望文生义，我们也能猜出它肯定是跟迭代有关的东西。当然，这种猜测也不是没有根据的，其重要根据就是英文单词，如果它们之间没有一点关系，肯定不会将命名搞得一样。

猜对了。`__iter__`就是对象的一个特殊方法，它是迭代规则(iterator potocol)的基础。或者说，对象如果没有它，就不能返回迭代器，就没有 `next()` 方法，就不能迭代。

> 提醒注意，如果读者用的是 Python3.x，迭代器对象实现的是`__next__()` 方法，不是 `next()`。并且，在 Python3.x 中有一个内建函数 next()，可以实现 `next(it)`，访问迭代器，这相当于于 python2.x 中的 `it.next()`（it 是迭代对象）。

那些类型是 list、tuple、file、dict 对象有`__iter__()`方法，标着他们能够迭代。这些类型都是 Python 中固有的，我们能不能自己写一个对象，让它能够迭代呢？

当然呢！要不然 python 怎么强悍呢。

```py
#!/usr/bin/env Python
# coding=utf-8

"""
the interator as range()
"""
class MyRange(object):
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def next(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return i
        else:
            raise StopIteration()

if __name__ == "__main__":
    x = MyRange(7)
    print "x.next()==>", x.next()
    print "x.next()==>", x.next()
    print "------for loop--------"
    for i in x:
        print i 
```

将代码保存，并运行，结果是：

```py
$ python 21401.py 
x.next()==> 0
x.next()==> 1
------for loop--------
2
3
4
5
6 
```

以上代码的含义，是自己仿写了拥有 `range()` 的对象，这个对象是可迭代的。分析如下：

类 MyRange 的初始化方法`__init__()` 就不用赘述了，因为前面已经非常详细分析了这个方法，如果复习，请阅读《类(2)》相关内容。

`__iter__()` 是类中的核心，它返回了迭代器本身。一个实现了`__iter__()`方法的对象，即意味着其实可迭代的。

含有 `next()` 的对象，就是迭代器，并且在这个方法中，在没有元素的时候要发起 `StopIteration()` 异常。

如果对以上类的调用换一种方式：

```py
if __name__ == "__main__":
    x = MyRange(7)
    print list(x)
    print "x.next()==>", x.next() 
```

运行后会出现如下结果：

```py
$ python 21401.py 
[0, 1, 2, 3, 4, 5, 6]
x.next()==>
Traceback (most recent call last):
  File "21401.py", line 26, in <module>
    print "x.next()==>", x.next()
  File "21401.py", line 21, in next
    raise StopIteration()
StopIteration 
```

说明什么呢？`print list(x)` 将对象返回值都装进了列表中并打印出来，这个正常运行了。此时指针已经移动到了迭代对象的最后一个，正如在《迭代》中描述的那样，`next()` 方法没有检测也不知道是不是要停止了，它还要继续下去，当继续下一个的时候，才发现没有元素了，于是返回了 `StopIteration()`。

为什么要将用这种可迭代的对象呢？就像上面例子一样，列表不是挺好的吗？

列表的确非常好，在很多时候效率很高，并且能够解决相当普遍的问题。但是，不要忘记一点，在某些时候，列表可能会给你带来灾难。因为在你使用列表的时候，需要将列表内容一次性都读入到内存中，这样就增加了内存的负担。如果列表太大太大，就有内存溢出的危险了。这时候需要的是迭代对象。比如斐波那契数列（在本教程多处已经提到这个著名的数列：《练习》的练习 4）:

```py
#!/usr/bin/env Python
# coding=utf-8
"""
compute Fibonacci by iterator
"""
__metaclass__ = type

class Fibs:
    def __init__(self, max):
        self.max = max
        self.a = 0
        self.b = 1

    def __iter__(self):
        return self

    def next(self):
        fib = self.a
        if fib > self.max:
            raise StopIteration
        self.a, self.b = self.b, self.a + self.b
        return fib

if __name__ == "__main__":
    fibs = Fibs(5)
    print list(fibs) 
```

运行结果是：

```py
$ python 21402.py 
[0, 1, 1, 2, 3, 5] 
```

> 给读者一个思考问题：要在斐波那契数列中找出大于 1000 的最小的数，能不能在上述代码基础上改造得出呢？

关于列表和迭代器之间的区别，还有两个非常典型的内建函数：`range()` 和 `xrange()`，研究一下这两个的差异，会有所收获的。

```py
range(...)
    range(stop) -> list of integers
    range(start, stop[, step]) -> list of integers

>>> dir(range)
['__call__', '__class__', '__cmp__', '__delattr__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__', '__name__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__self__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__'] 
```

从 `range()` 的帮助文档和方法中可以看出，它的结果是一个列表。但是，如果用 `help(xrange)` 查看：

```py
class xrange(object)
 |  xrange(stop) -> xrange object
 |  xrange(start, stop[, step]) -> xrange object
 |  
 |  Like range(), but instead of returning a list, returns an object that
 |  generates the numbers in the range on demand.  For looping, this is 
 |  slightly faster than range() and more memory efficient. 
```

`xrange()` 返回的是对象，并且进一步告诉我们，类似 `range()`，但不是列表。在循环的时候，它跟 `range()` 相比“slightly faster than range() and more memory efficient”，稍快并更高的内存效率（就是省内存呀）。查看它的方法：

```py
>>> dir(xrange)
['__class__', '__delattr__', '__doc__', '__format__', '__getattribute__', '__getitem__', '__hash__', '__init__', '__iter__', '__len__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__'] 
```

看到令人兴奋的`__iter__`了吗？说明它是可迭代的，它返回的是一个可迭代的对象。

也就是说，通过 `range()` 得到的列表，会一次性被读入内存，而 `xrange()` 返回的对象，则是需要一个数值才从返回一个数值。比如这样一个应用：

还记得 `zip()` 吗？

```py
>>> a = ["name", "age"]
>>> b = ["qiwsir", 40]
>>> zip(a,b)
[('name', 'qiwsir'), ('age', 40)] 
```

如果两个列表的个数不一样，就会以短的为准了，比如：

```py
>>> zip(range(4), xrange(100000000))
[(0, 0), (1, 1), (2, 2), (3, 3)] 
```

第一个 `range(4)` 产生的列表被读入内存；第二个是不是也太长了？但是不用担心，它根本不会产生那么长的列表，因为只需要前 4 个数值，它就提供前四个数值。如果你要修改为 `range(100000000)`，就要花费时间了，可以尝试一下哦。

迭代器的确有迷人之处，但是它也不是万能之物。比如迭代器不能回退，只能如过河的卒子，不断向前。另外，迭代器也不适合在多线程环境中对可变集合使用（这句话可能理解有困难，先混个脸熟吧，等你遇到多线程问题再说）。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 生成器

生成器（英文：generator）是一个非常迷人的东西，也常被认为是 Python 的高级编程技能。不过，我依然很乐意在这里跟读者——尽管你可能是一个初学者——探讨这个话题，因为我相信读者看本教程的目的，绝非仅仅将自己限制于初学者水平，一定有一颗不羁的心——要成为 Python 高手。那么，开始了解生成器吧。

还记得上节的“迭代器”吗？生成器和迭代器有着一定的渊源关系。生成器必须是可迭代的，诚然它又不仅仅是迭代器，但除此之外，又没有太多的别的用途，所以，我们可以把它理解为非常方便的自定义迭代器。

最这个关系实在感觉有点糊涂了。稍安勿躁，继续阅读即明了。

### 简单的生成器

```py
>>> my_generator = (x*x for x in range(4)) 
```

这是不是跟列表解析很类似呢？仔细观察，它不是列表，如果这样的得到的才是列表：

```py
>>> my_list = [x*x for x in range(4)] 
```

以上两的区别在于是 `[]` 还是 `()`，虽然是细小的差别，但是结果完全不一样。

```py
>>> dir(my_generator)
['__class__', '__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', 
'__iter__', 
'__name__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'close', 'gi_code', 'gi_frame', 'gi_running', 
'next', 
'send', 'throw'] 
```

为了容易观察，我将上述结果进行了重新排版。是不是发现了在迭代器中必有的方法`__inter__()`和 `next()`，这说明它是迭代器。如果是迭代器，就可以用 for 循环来依次读出其值。

```py
>>> for i in my_generator:
...     print i
... 
0
1
4
9
>>> for i in my_generator:
...     print i
... 
```

当第一遍循环的时候，将 my_generator 里面的值依次读出并打印，但是，当再读一次的时候，就发现没有任何结果。这种特性也正是迭代器所具有的。

如果对那个列表，就不一样了：

```py
>>> for i in my_list:
...     print i
... 
0
1
4
9
>>> for i in my_list:
...     print i
... 
0
1
4
9 
```

难道生成器就是把列表解析中的 `[]` 换成 `()` 就行了吗？这仅仅是生成器的一种表现形式和使用方法罢了，仿照列表解析式的命名，可以称之为“生成器解析式”（或者：生成器推导式、生成器表达式）。

生成器解析式是有很多用途的，在不少地方替代列表，是一个不错的选择。特别是针对大量值的时候，如上节所说的，列表占内存较多，迭代器（生成器是迭代器）的优势就在于少占内存，因此无需将生成器（或者说是迭代器）实例化为一个列表，直接对其进行操作，方显示出其迭代的优势。比如：

```py
>>> sum(i*i for i in range(10))
285 
```

请读者注意观察上面的 `sum()` 运算，不要以为里面少了一个括号，就是这么写。是不是很迷人？如果列表，你不得不：

```py
>>> sum([i*i for i in range(10)])
285 
```

通过生成器解析式得到的生成器，掩盖了生成器的一些细节，并且适用领域也有限。下面就要剖析生成器的内部，深入理解这个魔法工具。

### 定义和执行过程

yield 这个词在汉语中有“生产、出产”之意，在 Python 中，它作为一个关键词（你在变量、函数、类的名称中就不能用这个了），是生成器的标志。

```py
>>> def g():
...     yield 0
...     yield 1
...     yield 2
... 
>>> g
<function g at 0xb71f3b8c> 
```

建立了一个非常简单的函数，跟以往看到的函数唯一不同的地方是用了三个 yield 语句。然后进行下面的操作：

```py
>>> ge = g()
>>> ge
<generator object g at 0xb7200edc>
>>> type(ge)
<type 'generator'> 
```

上面建立的函数返回值是一个生成器(generator)类型的对象。

```py
>>> dir(ge)
['__class__', '__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__iter__', '__name__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'close', 'gi_code', 'gi_frame', 'gi_running', 'next', 'send', 'throw'] 
```

在这里看到了`__iter__()` 和 `next()`，说明它是迭代器。既然如此，当然可以：

```py
>>> ge.next()
0
>>> ge.next()
1
>>> ge.next()
2
>>> ge.next()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration 
```

从这个简单例子中可以看出，那个含有 yield 关键词的函数返回值是一个生成器类型的对象，这个生成器对象就是迭代器。

我们把含有 yield 语句的函数称作生成器。生成器是一种用普通函数语法定义的迭代器。通过上面的例子可以看出，这个生成器（也是迭代器），在定义过程中并没有像上节迭代器那样写`__inter__()` 和 `next()`，而是只要用了 yield 语句，那个普通函数就神奇般地成为了生成器，也就具备了迭代器的功能特性。

yield 语句的作用，就是在调用的时候返回相应的值。详细剖析一下上面的运行过程：

1.  `ge = g()`：除了返回生成器之外，什么也没有操作，任何值也没有被返回。
2.  `ge.next()`：直到这时候，生成器才开始执行，遇到了第一个 yield 语句，将值返回，并暂停执行（有的称之为挂起）。
3.  `ge.next()`：从上次暂停的位置开始，继续向下执行，遇到 yield 语句，将值返回，又暂停。
4.  `gen.next()`：重复上面的操作。
5.  `gene.next()`：从上面的挂起位置开始，但是后面没有可执行的了，于是 `next()` 发出异常。

从上面的执行过程中，发现 yield 除了作为生成器的标志之外，还有一个功能就是返回值。那么它跟 return 这个返回值有什么区别呢？

### yield

为了弄清楚 yield 和 return 的区别，我们写两个没有什么用途的函数：

```py
>>> def r_return(n):
...     print "You taked me."
...     while n > 0:
...         print "before return"
...         return n
...         n -= 1
...         print "after return"
... 
>>> rr = r_return(3)
You taked me.
before return
>>> rr
3 
```

从函数被调用的过程可以清晰看出，`rr = r_return(3)`，函数体内的语句就开始执行了，遇到 return，将值返回，然后就结束函数体内的执行。所以 return 后面的语句根本没有执行。这是 return 的特点，关于此特点的详细说明请阅读《函数(2)》中的返回值相关内容。

下面将 return 改为 yield：

```py
>>> def y_yield(n):
...     print "You taked me."
...     while n > 0:
...         print "before yield"
...         yield n
...         n -= 1
...         print "after yield"
... 
>>> yy = y_yield(3)    #没有执行函数体内语句
>>> yy.next()          #开始执行
You taked me.
before yield
3                      #遇到 yield，返回值，并暂停
>>> yy.next()          #从上次暂停位置开始继续执行
after yield
before yield
2                      #又遇到 yield，返回值，并暂停
>>> yy.next()          #重复上述过程
after yield
before yield
1
>>> yy.next()
after yield            #没有满足条件的值，抛出异常
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration 
```

结合注释和前面对执行过程的分析，读者一定能理解 yield 的特点了，也深知与 return 的区别了。

一般的函数，都是止于 return。作为生成器的函数，由于有了 yield，则会遇到它挂起，如果还有 return，遇到它就直接抛出 SoptIteration 异常而中止迭代。

斐波那契数列已经是老相识了。不论是循环、迭代都用它举例过，现在让我们还用它吧，只不过是要用上 yield：

```py
#!/usr/bin/env Python
# coding=utf-8

def fibs(max):
    """
    斐波那契数列的生成器
    """
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1

if __name__ == "__main__":
    f = fibs(10)
    for i in f:
        print i , 
```

运行结果如下：

```py
$ python 21501.py
1 1 2 3 5 8 13 21 34 55 
```

用生成器方式实现的斐波那契数列是不是跟以前的有所不同了呢？读者可以将本教程中已经演示过的斐波那契数列实现方式做一下对比，体会各种方法的差异。

经过上面的各种例子，已经明确，一个函数中，只要包含了 yield 语句，它就是生成器，也是迭代器。这种方式显然比前面写迭代器的类要简便多了。但，并不意味着上节的就被抛弃。是生成器还是迭代器，都是根据具体的使用情景而定。

### 生成器方法

在 python2.5 以后，生成器有了一个新特征，就是在开始运行后能够为生成器提供新的值。这就好似生成器和“外界”之间进行数据交流。

```py
>>> def repeater(n):
...     while True:
...         n = (yield n)
... 
>>> r = repeater(4)
>>> r.next()
4
>>> r.send("hello")
'hello' 
```

当执行到 `r.next()` 的时候，生成器开始执行，在内部遇到了 `yield n` 挂起。注意在生成器函数中，`n = (yield n)` 中的 `yield n` 是一个表达式，并将结果赋值给 n，虽然不严格要求它必须用圆括号包裹，但是一般情况都这么做，请读者也追随这个习惯。

当执行 `r.send("hello")` 的时候，原来已经被挂起的生成器（函数）又被唤醒，开始执行 `n = (yield n)`，也就是讲 send() 方法发送的值返回。这就是在运行后能够为生成器提供值的含义。

如果接下来再执行 `r.next()` 会怎样？

```py
>>> r.next() 
```

什么也没有，其实就是返回了 None。按照前面的叙述，读者可以看到，这次执行 `r.next()`，由于没有传入任何值，yield 返回的就只能是 None.

还要注意，send() 方法必须在生成器运行后并挂起才能使用，也就是 yield 至少被执行一次。如果不是这样：

```py
>>> s = repeater(5)
>>> s.send("how")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can't send non-None value to a just-started generator 
```

就报错了。但是，可将参数设为 None：

```py
>>> s.send(None)
5 
```

这是返回的是调用函数的时传入的值。

此外，还有两个方法：close() 和 throw()

*   throw(type, value=None, traceback=None):用于在生成器内部（生成器的当前挂起处，或未启动时在定义处）抛出一个异常（在 yield 表达式中）。
*   close()：调用时不用参数，用于关闭生成器。

最后一句，你在编程中，不用生成器也可以。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。