# 三、语句和文件

## 运算符

在编程语言，运算符是比较多样化的，虽然在《常用数学函数和运算优先级》中给出了一个各种运算符和其优先级的表格，但是，那时对 Python 理解还比较肤浅。建议诸位先回头看看那个表格，然后继续下面的内容。

这里将各种运算符总结一下，有复习，也有拓展。

### 算术运算符

前面已经讲过了四则运算，其中涉及到一些运算符：加减乘除，对应的符号分别是：+ - * /，此外，还有求余数的：%。这些都是算术运算符。其实，算术运算符不止这些。根据中学数学的知识，看官也应该想到，还应该有乘方、开方之类的。

下面列出一个表格，将所有的运算符表现出来。不用记，但是要认真地看一看，知道有那些，如果以后用到，但是不自信能够记住，可以来查。

| 运算符 | 描述 | 实例 |
| --- | --- | --- |
| + | 加 - 两个对象相加 | 10+20 输出结果 30 |
| - | 减 - 得到负数或是一个数减去另一个数 | 10-20 输出结果 -10 |
| * | 乘 - 两个数相乘或是返回一个被重复若干次的字符串 | 10 * 20 输出结果 200 |
| / | 除 - x 除以 y | 20/10 输出结果 2 |
| % | 取余 - 返回除法的余数 | 20%10 输出结果 0 |
| ** | 幂 - 返回 x 的 y 次幂 | 10**2 输出结果 100 |
| // | 取整除 - 返回商的整数部分 | 9//2 输出结果 4 , 9.0//2.0 输出结果 4.0 |

列为看官可以根据中学数学的知识，想想上面的运算符在混合运算中，应该按照什么顺序计算。并且亲自试试，是否与中学数学中的规律一致。（应该是一致的，计算机科学家不会另外搞一套让我们和他们一块受罪。）

### 比较运算符

所谓比较，就是比一比两个东西。这在某国是最常见的了，做家长的经常把自己的孩子跟别人的孩子比较，唯恐自己孩子在某方面差了；官员经常把自己的收入和银行比较，总觉得少了。

在计算机高级语言编程中，任何两个同一类型的量的都可以比较，比如两个数字可以比较，两个字符串可以比较。注意，是两个同一类型的。不同类型的量可以比较吗？首先这种比较没有意义。就好比二两肉和三尺布进行比较，它们谁大呢？这种比较无意义。所以，在真正的编程中，我们要谨慎对待这种不同类型量的比较。

但是，在某些语言中，允许这种无意思的比较。因为它在比较的时候，都是将非数值的转化为了数值类型比较。

对于比较运算符，在小学数学中就学习了一些：大于、小于、等于、不等于。没有陌生的东西，Python 里面也是如此。且看下表：

以下假设变量 a 为 10，变量 b 为 20：

| 运算符 | 描述 | 实例 |
| --- | --- | --- |
| == | 等于 - 比较对象是否相等 | (a == b) 返回 False。 |
| != | 不等于 - 比较两个对象是否不相等 | (a != b) 返回 true. |
| > | 大于 - 返回 x 是否大于 y | (a > b) 返回 False。 |
| < | 小于 - 返回 x 是否小于 y | (a < b) 返回 true。 |
| >= | 大于等于 - 返回 x 是否大于等于 y。 | (a >= b) 返回 False。 |
| <= | 小于等于 - 返回 x 是否小于等于 y。 | (a <= b) 返回 true。 |

上面的表格实例中，显示比较的结果就是返回一个 true 或者 false，这是什么意思呢。就是在告诉你，这个比较如果成立，就是为真，返回 True，否则返回 False，说明比较不成立。

请按照下面方式进行比较操作，然后再根据自己的想象，把比较操作熟练熟练。

```py
>>> a=10
>>> b=20
>>> a>b
False
>>> a<b
True
>>> a==b
False
>>> a!=b
True
>>> a>=b
False
>>> a<=b
True 
```

除了数字之外，还可以对字符串进行比较。字符串中的比较是按照“字典顺序”进行比较的。当然，这里说的是英文的字典，不是前面说的字典数据类型。

```py
>>> a = "qiwsir"
>>> b = "Python"
>>> a > b
True 
```

先看第一个字符，按照字典顺序，q 大于 p（在字典中，q 排在 p 的后面），那么就返回结果 True.

在 Python 中，如果是两种不同类型的对象，虽然可以比较。但我是不赞成这样进行比较的。

```py
>>> a = 5
>>> b = "5"
>>> a > b
False 
```

### 逻辑运算符

首先谈谈什么是逻辑，韩寒先生对逻辑有一个分类：

> 逻辑分两种，一种是逻辑，另一种是中国人的逻辑。————韩寒

这种分类的确非常精准。在很多情况下，中国人是有很奇葩的逻辑的。但是，在 Python 中，讲的是逻辑，不是中国人的逻辑。

> 逻辑（logic），又称理则、论理、推理、推论，是有效推论的哲学研究。逻辑被使用在大部份的智能活动中，但主要在哲学、数学、语义学和计算机科学等领域内被视为一门学科。在数学里，逻辑是指研究某个形式语言的有效推论。

关于逻辑问题，看官如有兴趣，可以听一听[《国立台湾大学公开课：逻辑》](http://v.163.com/special/ntu/luoji.html)

#### 布尔类型的变量

在所有的高级语言中，都有这么一类变量，被称之为布尔型。从这个名称，看官就知道了，这是用一个人的名字来命名的。

> 乔治·布尔（George Boole，1815 年 11 月－1864 年，)，英格兰数学家、哲学家。
> 
> 乔治·布尔是一个皮匠的儿子，生于英格兰的林肯。由于家境贫寒，布尔不得不在协助养家的同时为自己能受教育而奋斗，不管怎么说，他成了 19 世纪最重要的数学家之一。尽管他考虑过以牧师为业，但最终还是决定从教，而且不久就开办了自己的学校。
> 
> 在备课的时候，布尔不满意当时的数学课本，便决定阅读伟大数学家的论文。在阅读伟大的法国数学家拉格朗日的论文时，布尔有了变分法方面的新发现。变分法是数学分析的分支，它处理的是寻求优化某些参数的曲线和曲面。
> 
> 1848 年，布尔出版了《The Mathematical Analysis of Logic》，这是他对符号逻辑诸多贡献中的第一次。
> 
> 1849 年，他被任命位于爱尔兰科克的皇后学院（今科克大学或 UCC）的数学教授。1854 年，他出版了《The Laws of Thought》，这是他最著名的著作。在这本书中布尔介绍了现在以他的名字命名的布尔代数。布尔撰写了微分方程和差分方程的课本，这些课本在英国一直使用到 19 世纪末。
> 
> 由于其在符号逻辑运算中的特殊贡献，很多计算机语言中将逻辑运算称为布尔运算，将其结果称为布尔值。

请看官认真阅读布尔的生平，励志呀。

布尔所创立的这套逻辑被称之为“布尔代数”。其中规定只有两种值，True 和 False，正好对应这计算机上二进制数的 1 和 0。所以，布尔代数和计算机是天然吻合的。

所谓布尔类型，就是返回结果为 1(True)、0(False)的数据变量。

在 Python 中（其它高级语言也类似，其实就是布尔代数的运算法则），有三种运算符，可以实现布尔类型的变量间的运算。

#### 布尔运算

看下面的表格，对这种逻辑运算符比较容易理解：

（假设变量 a 为 10，变量 b 为 20）

| 运算符 | 描述 | 实例 |
| --- | --- | --- |
| and | 布尔"与" - 如果 x 为 False，x and y 返回 False，否则它返回 y 的计算值。 | (a and b) 返回 true。 |
| or | 布尔"或" - 如果 x 是 True，它返回 True，否则它返回 y 的计算值。 | (a or b) 返回 true。 |
| not | 布尔"非" - 如果 x 为 True，返回 False。如果 x 为 False，它返回 True。 | not(a and b) 返回 false。 |

*   and

and，翻译为“与”运算，但事实上，这种翻译容易引起望文生义的理解。先说一下正确的理解。A and B，含义是：首先运算 A，如果 A 的值是 true，就计算 B，并将 B 的结果返回做为最终结果，如果 B 是 False，那么 A and B 的最终结果就是 False,如果 B 的结果是 True，那么 A and B 的结果就是 True；如果 A 的值是 False ,就不计算 B 了，直接返回 A and B 的结果为 False.

比如：

`4>3 and 4<9`，首先看 `4>3` 的值，这个值是 `True`，再看 `4<9` 的值，是 `True`，那么最终这个表达式的结果为 `True`.

```py
>>> 4>3 and 4<9
True 
```

`4>3 and 4<2`，先看 `4>3`，返回 `True`，再看 `4<2`，返回的是 `False`，那么最终结果是 `False`.

```py
>>> 4>3 and 4<2
False 
```

`4<3 and 4<9`，先看 `4<3`，返回为 `False`,就不看后面的了，直接返回这个结果做为最终结果（对这种现象，有一个形象的说法，叫做“短路”。这个说法形象吗？不熟悉物理的是不是感觉形象？）。

```py
>>> 4<3 and 4<2
False 
```

前面说容易引起望文生义的理解，就是有相当不少的人认为无论什么时候都看 and 两边的值，都是 true 返回 true，有一个是 false 就返回 false。根据这种理解得到的结果，与前述理解得到的结果一样，但是，运算量不一样哦。

*   or

or，翻译为“或”运算。在 A or B 中，它是这么运算的：

```py
if A==True:
    return True
else:
    if B==True:
        return True
    else if B==False:
        return False 
```

上面这段算是伪代码啦。所谓伪代码，就是不是真正的代码，无法运行。但是，伪代码也有用途，就是能够以类似代码的方式表达一种计算过程。

看官是不是能够看懂上面的伪代码呢？下面再增加上每行的注释。这个伪代码跟自然的英语差不多呀。

```py
if A==True:         #如果 A 的值是 True
    return True     #返回 True,表达式最终结果是 True
else:               #否则，也就是 A 的值不是 True
    if B==True:     #看 B 的值，然后就返回 B 的值做为最终结果。
        return True
    else if B==False:
        return False 
```

举例，根据上面的运算过程，分析一下下面的例子，是不是与运算结果一致？

```py
>>> 4<3 or 4<9
True
>>> 4<3 or 4>9
False
>>> 4>3 or 4>9
True 
```

*   not

not，翻译成“非”，窃以为非常好，不论面对什么，就是要否定它。

```py
>>> not(4>3)
False
>>> not(4<3)
True 
```

关于运算符问题，其实不止上面这些，还有呢，比如成员运算符 in，在后面的学习中会逐渐遇到。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 语句(1)

数据类型已经学的差不多了，但是，到现在为止我们还不能真正的写程序，这就好比小学生学习写作一样，到目前为止仅仅学会了一些词语，还不知道如何造句子。从现在开始就学习如何造句子了。

在编程语言中，句子被称之为“语句”，

### 什么是语句

事实上，前面已经用过语句了，最典型的那句：`print "Hello, World"`就是语句。

为了能够严谨地阐述这个概念，抄一段[维基百科中的词条：命令式编程](http://zh.wikipedia.org/wiki/%E6%8C%87%E4%BB%A4%E5%BC%8F%E7%B7%A8%E7%A8%8B)

> 命令式编程（英语：Imperative programming），是一种描述电脑所需作出的行为的编程范型。几乎所有电脑的硬件工作都是指令式的；几乎所有电脑的硬件都是设计来运行机器码，使用指令式的风格来写的。较高级的指令式编程语言使用变量和更复杂的语句，但仍依从相同的范型。
> 
> 运算语句一般来说都表现了在存储器内的数据进行运算的行为，然后将结果存入存储器中以便日后使用。高级命令式编程语言更能处理复杂的表达式，可能会产生四则运算和函数计算的结合。

一般所有高级语言，都包含如下语句，Python 也不例外：

*   循环语句:容许一些语句反复运行数次。循环可依据一个默认的数目来决定运行这些语句的次数；或反复运行它们，直至某些条件改变。
*   条件语句:容许仅当某些条件成立时才运行某个区块。否则，这个区块中的语句会略去，然后按区块后的语句继续运行。
*   无条件分支语句容许运行顺序转移到程序的其他部分之中。包括跳跃（在很多语言中称为 Goto）、副程序和 Procedure 等。

循环、条件分支和无条件分支都是控制流程。

当然，Python 中的语句还是有 Python 特别之处的（别的语言中，也会有自己的特色）。下面就开始娓娓道来。

### print

在 Python2.x 中，print 是一个语句，但是在 Python3.x 中它是一个函数了。这点请注意。不过，这里所使用的还是 Python2.x。

> 为什么不用 Python3.x？这个问题在开始就回答过。但是还有朋友问。重复回答：因为现在很多工程项目都是 Python2.x，Python3.x 相对 Python2.x 有不完全兼容的地方。学 Python 的目的就是要在真实的工程项目中使用，理所应当要学 Python2.x。此外，学会了 Python2.x，将来过渡到 Python3.x，只需要注意一些细节即可。

print 发起的语句，在程序中主要是将某些东西打印出来，还记得在讲解字符串的时候，专门讲述了字符串的格式化输出吗？那就是用来 print 的。

```py
>>> print "hello, world"
hello, world
>>> print "hello","world"
hello world 
```

请仔细观察，上面两个 print 语句的差别。第一个打印的是"hello, world"，包括其中的逗号和空格，是一个完整的字符串。第二个打印的是两个字符串，一个是"hello"，另外一个是"world"，两个字符串之间用逗号分隔。

本来，在 print 语句中，字符串后面会接一个 `\n` 符号。即换行。但是，如果要在一个字符串后面跟着逗号，那么换行就取消了，意味着两个字符串"hello"，"world"打印在同一行。

或许现在体现的还不时很明显，如果换一个方法，就显示出来了。（下面用到了一个被称之为循环的语句，下一节会重点介绍。

```py
>>> for i in [1,2,3,4,5]:
...     print i
... 
1
2
3
4
5 
```

这个循环的意思就是要从列表中依次取出每个元素，然后赋值给变量 i，并用 print 语句打印打出来。在变量 i 后面没有任何符号，每打印一个，就换行，再打印另外一个。

下面的方式就跟上面的有点区别了。

```py
>>> for i in [1,2,3,4,5]:
...     print i ,
... 
1 2 3 4 5 
```

就是在 print 语句的最后加了一个逗号，打印出来的就在一行了。

print 语句经常用在调试程序的过程，让我们能够知道程序在执行过程中产生的结果。

### import

在《常用数学函数和运算优先级》中，曾经用到过一个 math 模块，它能提供很多数学函数，但是这些函数不是 Python 的内建函数，是 math 模块的，所以，要用 import 引用这个模块。

这种用 import 引入模块的方法，是 Python 编程经常用到的。引用方法有如下几种：

```py
>>> import math
>>> math.pow(3,2)
9.0 
```

这是常用的一种方式，而且非常明确，`math.pow(3,2)`就明确显示了，pow()函数是 math 模块里的。可以说这是一种可读性非常好的引用方式，并且不同模块的同名函数不会产生冲突。

```py
>>> from math import pow
>>> pow(3,2)
9.0 
```

这种方法就有点偷懒了，不过也不难理解，从字面意思就知道 pow()函数来自于 math 模块。在后续使用的时候，只需要直接使用 `pow()`即可，不需要在前面写上模块名称了。这种引用方法，比较适合于引入模块较少的时候。如果引入模块多了，可读性就下降了，会不知道那个函数来自那个模块。

```py
>>> from math import pow as pingfang
>>> pingfang(3,2)
9.0 
```

这是在前面那种方式基础上的发展，将从某个模块引入的函数重命名，比如讲 pow 充命名为 pingfang，然后使用 `pingfang()`就相当于在使用 `pow()`了。

如果要引入多个函数，可以这样做：

```py
>>> from math import pow, e, pi
>>> pow(e,pi)
23.140692632779263 
```

引入了 math 模块里面的 pow,e,pi，pow()是一个乘方函数，e，就是那个欧拉数；pi 就是 π.

> e，作为数学常数，是自然对数函数的底数。有时称他为欧拉函数（Euler's number），以瑞士数学家欧拉命名；也有个较鲜见的名字纳皮尔常数，以纪念苏格兰数学家约翰·纳皮尔引进对数。它是一个无限不循环小数。e = 2.71828182845904523536(《维基百科》)
> 
> e 的 π 次方,是一个数学常数。与 e 和 π 一样，它是一个超越数。这个常数在希尔伯特第七问题中曾提到过。(《维基百科》)

```py
>>> from math import *
>>> pow(3,2)
9.0
>>> sqrt(9)
3.0 
```

这种引入方式是最贪图省事的了，一下将 math 中的所有函数都引过来了。不过，这种方式的结果是让可读性更降低了。仅适用于模块中的函数比较少的时候，并且在程序中应用比较频繁。

在这里，我们用 math 模块为例，引入其中的函数。事实上，不仅函数可以引入，模块中还可以包括常数等，都可以引入。在编程中，模块中可以包括各样的对象，都可以引入。

### 赋值语句

对于赋值语句，应该不陌生，在前面已经频繁使用了，如 `a = 3` 这样的，就是将一个整数赋给了变量。

> 编程中的“=”和数学中的“=”是完全不同的。在编程语言中，“=”表示赋值过程。

除了那种最简单的赋值之外，还可以这么干：

```py
>>> x, y, z = 1, "python", ["hello", "world"]
>>> x
1
>>> y
'python'
>>> z
['hello', 'world'] 
```

这里就一一对应赋值了。如果把几个值赋给一个，可以吗？

```py
>>> a = "itdiffer.com", "python"

>>> a
('itdiffer.com', 'python') 
```

原来是将右边的两个值装入了一个元组，然后将元组赋给了变量 a。这个 Python 太聪明了。

在 Python 的赋值语句中，还有一个更聪明的，它一出场，简直是让一些已经学习过某种其它语言的人亮瞎眼。

有两个变量，其中 `a = 2`,`b = 9`。现在想让这两个变量的值对调，即最终是 `a = 9`,`b = 2`.

这是一个简单而经典的题目。在很多编程语言中，是这么处理的：

```py
temp = a;
a = b;
b = temp; 
```

这么做的那些编程语言，变量就如同一个盒子，值就如同放到盒子里面的东西。如果要实现对调，必须在找一个盒子，将 a 盒子里面的东西（数字 2）拿到那个临时盒子(temp)中，这样 a 盒子就空了，然后将 b 盒子中的东西拿(数字 9)拿到 a 盒子中(a = b)，完成这步之后，b 盒子是空的了，最后将临时盒子里面的那个数字 2 拿到 b 盒子中。这就实现了两个变量值得对调。

太啰嗦了。

Python 只要一行就完成了。

```py
>>> a = 2
>>> b = 9

>>> a, b = b, a

>>> a
9
>>> b
2 
```

`a, b = b, a` 就实现了数值对调，多么神奇。之所以神奇，就是因为我前面已经数次提到的 Python 中变量和数据对象的关系。变量相当于贴在对象上的标签。这个操作只不过是将标签换个位置，就分别指向了不同的数据对象。

还有一种赋值方式，被称为“链式赋值”

```py
>>> m = n = "I use python"
>>> print m,n
I use python I use python 
```

用这种方式，实现了一次性对两个变量赋值，并且值相同。

```py
>>> id(m)
3072659528L
>>> id(n)
3072659528L 
```

用 `id()`来检查一下，发现两个变量所指向的是同一个对象。

另外，还有一种判断方法，来检查两个变量所指向的值是否是同一个（注意，同一个和相等是有差别的。在编程中，同一个就是 `id()`的结果一样。

```py
>>> m is n
True 
```

这是在检查 m 和 n 分别指向的对象是否是同一个，True 说明是同一个。

```py
>>> a = "I use python"
>>> b = a
>>> a is b
True 
```

这是跟上面链式赋值等效的。

但是：

```py
>>> b = "I use python"
>>> a is b
False
>>> id(a)
3072659608L
>>> id(b)
3072659568L

>>> a == b
True 
```

看出其中的端倪了吗？这次 a、b 两个变量虽然相等，但不是指向同一个对象。

还有一种赋值形式，如果从数学的角度看，是不可思议的，如：`x = x + 1`，在数学中，这个等式是不成立的。因为数学中的“=”是等于的含义，但是在编程语言中，它成立，因为"="是赋值的含义，即将变量 x 增加 1 之后，再把得到的结果赋值变量 x.

这种变量自己变化之后将结果再赋值给自己的形式，称之为“增量赋值”。+、-、*、/、% 都可以实现这种操作。

为了让这个操作写起来省点事（要写两遍同样一个变量），可以写成：`x += 1`

```py
>>> x = 9
>>> x += 1
>>> x
10 
```

除了数字，字符串进行增量赋值，在实际中也很有价值。

```py
>>> m = "py"
>>> m += "th"
>>> m
'pyth'
>>> m += "on"
>>> m
'python' 
```

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 语句(2)

所谓条件语句，顾名思义，就是依据某个条件，满足这个条件后执行下面的内容。

### if

if，其含义就是：conj. （表条件）如果。if 发起的就是一个条件，它是构成条件语句的关键词。

```py
>>> a = 8
>>> if a==8:
...     print a
... 
8 
```

在交互模式下，简单书写一下 if 发起的条件语句。特别说明，我上面这样写，只是简单演示一下。如果你要写大段的代码，千万不要在交互模式下写。

`if a==8:`，这句话里面如果条件 `a==8` 返回的是 True，那么就执行下面的语句。特别注意，冒号是必须的。下面一行语句 `print a` 要有四个空格的缩进。这是 Python 的特点，称之为语句块。

唯恐说的不严谨，我还是引用维基百科中的叙述：

> Python 开发者有意让远反了缩排規則的程序不能通过编译，以此来强迫程序员养成良好的编程习惯。並且 Python 語言利用缩排表示语句块的开始和结束（Off-side 规则），而非使用花括号或者某种关键字。增加缩表示语句块的开始，而減少缩排则表示语句块的结束。缩排成为了语法的一部分。例如 if 语句.
> 
> 根剧 PEP 的规定，必须使用 4 个空格来表示每级缩排。使用 Tab 字符和其它数目的空格虽然都可以编译通过，但不符合编码规范。支持 Tab 字符和其它数目的空格仅仅是为兼容很旧的 Python 程式和某些有问题的编辑程式。

从上面的这段话中，提炼出几个关键点：

*   必须要通过缩进方式来表示语句块的开始和结束
*   缩进用四个空格（也是必须的，别的方式或许可以，但不提倡）

### if/else/elif

在进行条件判断的时候，只有 if，往往是不够的。比如下图所示的流程

![](img/12201.png)

这张图反应的是这样一个问题：

输入一个数字，并输出输入的结果，如果这个数字大于 10，那么同时输出大于 10,如果小于 10，同时输出提示小于 10,如果等于 10,就输出表扬的一句话。

从图中就已经显示出来了，仅仅用 if 来判断，是不足的，还需要其它分支。这就需要引入别的条件判断了。所以，有了 if...elif...else 语句。

基本样式结构：

```py
if 条件 1:
    执行的内容 1
elif 条件 2:
    执行的内容 2
elif 条件 3：
    执行的内容 3
else:
    执行的内容 4 
```

elif 用于多个条件时使用，可以没有。另外，也可以只有 if，而没有 else。

下面我们就不在交互模式中写代码了。打开文本编辑界面，你的编辑器也能提供这个功能，如果找不到，请回到《写一个简单的程序》查看。

代码实例如下：

```py
#! /usr/bin/env python
#coding:utf-8

print "请输入任意一个整数数字："

number = int(raw_input())   #通过 raw_input()输入的数字是字符串
                            #用 int()将该字符串转化为整数

if number == 10:
    print "您输入的数字是：%d"%number
    print "You are SMART."
elif number > 10:
    print "您输入的数字是：%d"%number
    print "This number is more than 10."
elif number < 10:
    print "您输入的数字是：%d"%number
    print "This number is less than 10."
else:
    print "Are you a human?" 
```

特别提醒看官注意，前面我们已经用过 raw_input() 函数了，这个是获得用户在界面上输入的信息，而通过它得到的是字符串类型的数据。

上述程序，依据条件进行判断，不同条件下做不同的事情了。需要提醒的是在条件中：number == 10，为了阅读方便，在 number 和 == 之间有一个空格最好了，同理，后面也有一个。这里的 10,是 int 类型，number 也是 int 类型.

把这段程序保存成一个扩展名是.py 的文件，比如保存为 `num.py`,进入到存储这个文件的目录，运行 `Python num.py`，就能看到程序执行结果了。下面是我执行的结果，供参考。

```py
$ Python num.py
请输入任意一个整数数字：
12 
您输入的数字是：12
This number is more than 10.

$ Python num.py
请输入任意一个整数数字：
10
您输入的数字是：10
You are SMART.

$ Python num.py
请输入任意一个整数数字：
9
您输入的数字是：9
This number is less than 10. 
```

不知道各位是否注意到，上面的那段代码，开始有一行：

```py
#! /usr/bin/env python 
```

这是什么意思呢？

这句话以 # 开头，表示本来不在程序中运行。这句话的用途是告诉机器寻找到该设备上的 Python 解释器，操作系统使用它找到的解释器来运行文件中的程序代码。有的程序里写的是 /usr/bin Python，表示 Python 解释器在 /usr/bin 里面。但是，如果写成 /usr/bin/env，则表示要通过系统搜索路径寻找 Python 解释器。不同系统，可能解释器的位置不同，所以这种方式能够让代码更将拥有可移植性。对了，以上是对 Unix 系列操作系统而言。对与 windows 系统，这句话就当不存在。

在“条件”中，就是上节提到的各种条件运算表达式，如果是 True，就执行该条件下的语句。

### 三元操作符

三元操作，是条件语句中比较简练的一种赋值方式，它的模样是这样的：

```py
>>> name = "qiwsir" if "laoqi" else "github"
>>> name
'qiwsir'
>>> name = 'qiwsir' if "" else "python"
>>> name
'Python'
>>> name = "qiwsir" if "github" else ""
>>> name
'qiwsir' 
```

总结一下：A = Y if X else Z

什么意思，结合前面的例子，可以看出：

*   如果 X 为真，那么就执行 A=Y
*   如果 X 为假，就执行 A=Z

如此例

```py
>>> x = 2
>>> y = 8
>>> a = "python" if x>y else "qiwsir"
>>> a
'qiwsir'
>>> b = "python" if x<y else "qiwsir"
>>> b
'python' 
```

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 语句(3)

循环，也是现实生活中常见的现象，我们常说日复一日，就是典型的循环。又如：日月更迭，斗转星移，无不是循环；王朝更迭；子子孙孙，繁衍不息，从某个角度看也都是循环。

编程语言就是要解决现实问题的，因此也少不了要循环。

在 Python 中，循环有一个语句：for 语句。

其基本结构是：

```py
for 循环规则：
    操作语句 
```

从这个基本结构看，有着同 if 条件语句类似的地方：都有冒号；语句块都要缩进。是的，这是不可或缺的。

### 简单的 for 循环例子

前面介绍 print 语句的时候，出现了一个简单例子。重复一个类似的：

```py
>>> hello = "world"
>>> for i in hello:
...     print i
... 
w
o
r
l
d 
```

这个 for 循环是怎么工作的呢？

1.  hello 这个变量引用的是"world"这个 str 类型的数据
2.  变量 i 通过 hello 找到它所引用的对象"world",因为 str 类型的数据属于序列类型，能够进行索引，于是就按照索引顺序，从第一字符开始，依次获得该字符的引用。
3.  当 i="w"的时候，执行 print i，打印出了字母 w，结束之后循环第二次，让 i="e"，然后执行 print i,打印出字母 e，如此循环下去，一直到最后一个字符被打印出来，循环自动结束。注意，每次打印之后，要换行。如果不想换行，怎么办？参见《语句(1)》中关于 print 语句。

因为可以也通过使用索引（偏移量），得到序列对象的某个元素。所以，还可以通过下面的循环方式实现同样效果：

```py
>>> for i in range(len(hello)):
...     print hello[i]
... 
w
o
r
l
d 
```

其工作方式是：

1.  len(hello)得到 hello 引用的字符串的长度，为 5
2.  range(len(hello),就是 range(5),也就是[0, 1, 2, 3, 4],对应这"world"每个字母索引，也可以称之为偏移量。这里应用了一个新的函数 `range()`，关于它的用法，继续阅读，就能看到了。
3.  for i in range(len(hello)),就相当于 for i in [0,1,2,3,4],让 i 依次等于 list 中的各个值。当 i=0 时，打印 hello[0]，也就是第一个字符。然后顺序循环下去，直到最后一个 i=4 为止。

以上的循环举例中，显示了对 str 的字符依次获取，也涉及了 list，感觉不过瘾呀。那好，看下面对 list 的循环：

```py
>>> ls_line
['Hello', 'I am qiwsir', 'Welcome you', '']
>>> for word in ls_line:
...     print word
... 
Hello
I am qiwsir
Welcome you

>>> for i in range(len(ls_line)):
...     print ls_line[i]
... 
Hello
I am qiwsir
Welcome you 
```

### range(start,stop[, step])

这个内建函数，非常有必要给予说明，因为它会经常被使用。一般形式是`range(start, stop[, step])`

要研究清楚一些函数特别是内置函数的功能，建议看官首先要明白内置函数名称的含义。因为在 Python 中，名称不是随便取的，是代表一定意义的。所谓：名不正言不顺。

> range
> 
> n. 范围；幅度；排；山脉 vi. （在...内）变动；平行，列为一行；延伸；漫游；射程达到 vt. 漫游；放牧；使并列；归类于；来回走动

在具体实验之前，还是按照管理，摘抄一段[官方文档的原话](https://docs.Python.org/2/library/functions.html#range)，让我们能够深刻理解之：

> This is a versatile function to create lists containing arithmetic progressions. It is most often used in for loops. The arguments must be plain integers. If the step argument is omitted, it defaults to 1\. If the start argument is omitted, it defaults to 0\. The full form returns a list of plain integers [start, start + step, start + 2 * step, ...]. If step is positive, the last element is the largest start + i * step less than stop; if step is negative, the last element is the smallest start + i * step greater than stop. step must not be zero (or else ValueError is raised).

从这段话，我们可以得出关于 `range()`函数的以下几点：

*   这个函数可以创建一个数字元素组成的列表。
*   这个函数最常用于 for 循环（关于 for 循环，马上就要涉及到了）
*   函数的参数必须是整数，默认从 0 开始。返回值是类似[start, start + step, start + 2*step, ...]的列表。
*   step 默认值是 1。如果不写，就是按照此值。
*   如果 step 是正数，返回 list 的最最后的值不包含 stop 值，即 start+i*step 这个值小于 stop；如果 step 是负数，start+i*step 的值大于 stop。
*   step 不能等于零，如果等于零，就报错。

在实验开始之前，再解释 range(start,stop[,step])的含义：

*   start：开始数值，默认为 0,也就是如果不写这项，就是认为 start=0
*   stop：结束的数值，必须要写的。
*   step：变化的步长，默认是 1,也就是不写，就是认为步长为 1。坚决不能为 0

实验开始，请以各项对照前面的讲述：

```py
>>> range(9)                #stop=9，别的都没有写，含义就是 range(0,9,1)
[0, 1, 2, 3, 4, 5, 6, 7, 8] #从 0 开始，步长为 1,增加，直到小于 9 的那个数 
>>> range(0,9)
[0, 1, 2, 3, 4, 5, 6, 7, 8]
>>> range(0,9,1)
[0, 1, 2, 3, 4, 5, 6, 7, 8]

>>> range(1,9)              #start=1
[1, 2, 3, 4, 5, 6, 7, 8]

>>> range(0,9,2)            #step=2,每个元素等于 start+i*step，
[0, 2, 4, 6, 8] 
```

仅仅解释一下 range(0,9,2)

*   如果是从 0 开始，步长为 1,可以写成 range(9)的样子，但是，如果步长为 2，写成 range(9,2)的样子，计算机就有点糊涂了，它会认为 start=9,stop=2。所以，在步长不为 1 的时候，切忌，要把 start 的值也写上。
*   start=0,step=2,stop=9.list 中的第一个值是 start=0,第二个值是 start+1*step=2（注意，这里是 1，不是 2，不要忘记，前面已经讲过，不论是 list 还是 str，对元素进行编号的时候，都是从 0 开始的），第 n 个值就是 start+(n-1)*step。直到小于 stop 前的那个值。

熟悉了上面的计算过程，看看下面的输入谁是什么结果？

```py
>>> range(-9) 
```

我本来期望给我返回[0,-1,-2,-3,-4,-5,-6,-7,-8],我的期望能实现吗？

分析一下，这里 start=0,step=1,stop=-9.

第一个值是 0；第二个是 start+1*step，将上面的数代入，应该是 1,但是最后一个还是 -9，显然出现问题了。但是，Python 在这里不报错，它返回的结果是：

```py
>>> range(-9)
[]
>>> range(0,-9)
[]
>>> range(0)
[] 
```

报错和返回结果，是两个含义，虽然返回的不是我们要的。应该如何修改呢？

```py
>>> range(0,-9,-1)
[0, -1, -2, -3, -4, -5, -6, -7, -8]
>>> range(0,-9,-2)
[0, -2, -4, -6, -8] 
```

有了这个内置函数，很多事情就简单了。比如：

```py
>>> range(0,100,2)
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98] 
```

100 以内的自然数中的偶数组成的 list，就非常简单地搞定了。

思考一个问题，现在有一个列表，比如是["I","am","a","Pythoner","I","am","learning","it","with","qiwsir"],要得到这个 list 的所有序号组成的 list，但是不能一个一个用手指头来数。怎么办？

请沉思两分钟之后，自己实验一下，然后看下面。

```py
>>> pythoner
['I', 'am', 'a', 'pythoner', 'I', 'am', 'learning', 'it', 'with', 'qiwsir']
>>> py_index = range(len(pythoner))     #以 len(pythoner)为 stop 的值
>>> py_index
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
```

再用手指头指着 Pythoner 里面的元素，数一数，是不是跟结果一样。

**例：**找出 100 以内的能够被 3 整除的正整数。

**分析：**这个问题有两个限制条件，第一是 100 以内的正整数，根据前面所学，可以用 range(1,100)来实现；第二个是要解决被 3 整除的问题，假设某个正整数 n，这个数如果能够被 3 整除，也就是 n%3(% 是取余数)为 0.那么如何得到 n 呢，就是要用 for 循环。

以上做了简单分析，要实现流程，还需要细化一下。按照前面曾经讲授过的一种方法，要画出问题解决的流程图。

![](img/12301.png)

下面写代码就是按图索骥了。

代码：

```py
#! /usr/bin/env python
#coding:utf-8

aliquot = []

for n in range(1,100):
    if n%3 == 0:
        aliquot.append(n)

print aliquot 
```

代码运行结果：

```py
[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99] 
```

上面的代码中，将 for 循环和 if 条件判断都用上了。

不过，感觉有点麻烦，其实这么做就可以了：

```py
>>> range(3,100,3)
[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99] 
```

### 能够用来 for 的对象

所有的序列类型对象，都能够用 for 来循环。比如：

```py
>>> name_str = "qiwsir"
>>> for i in name_str:  #可以对 str 使用 for 循环
...     print i,
...                     
q i w s i r

>>> name_list = list(name_str)
>>> name_list
['q', 'i', 'w', 's', 'i', 'r']
>>> for i in name_list:     #对 list 也能用
...     print i,
... 
q i w s i r

>>> name_set = set(name_str)    #set 还可以用
>>> name_set
set(['q', 'i', 's', 'r', 'w'])
>>> for i in name_set:
...     print i,
... 
q i s r w

>>> name_tuple = tuple(name_str)
>>> name_tuple
('q', 'i', 'w', 's', 'i', 'r')
>>> for i in name_tuple:        #tuple 也能呀
...     print i,
... 
q i w s i r

>>> name_dict={"name":"qiwsir","lang":"python","website":"qiwsir.github.io"}
>>> for i in name_dict:             #dict 也不例外，这里本质上是将字典的键拿出来，成为序列后进行循环
...     print i,"-->",name_dict[i]
... 
lang --> Python
website --> qiwsir.github.io
name --> qiwsir 
```

在用 for 来循环读取字典键值对上，需要多说几句。

有这样一个字典：

```py
>>> a_dict = {"name":"qiwsir", "lang":"python", "email":"qiwsir@gmail.com", "website":"www.itdiffer.com"} 
```

曾记否？在《字典(2)》中有获得字典键、值的函数：items/iteritems/keys/iterkeys/values/itervalues，通过这些函数得到的是键或者值的列表。

```py
>>> for k in a_dict.keys():
...     print k, a_dict[k]
... 
lang python
website www.itdiffer.com
name qiwsir
email qiwsir@gmail.com 
```

这是最常用的一种获得字典键/值对的方法，而且效率也不错。

```py
>>> for k,v in a_dict.items():
...     print k,v
... 
lang python
website www.itdiffer.com
name qiwsir
email qiwsir@gmail.com

>>> for k,v in a_dict.iteritems():
...     print k,v
... 
lang python
website www.itdiffer.com
name qiwsir
email qiwsir@gmail.com 
```

这两种方法也能够实现同样的效果，但是因为有了上面的方法，一般就少用了。但是，用也无妨，特别是第二个 `iteritems()`，效率也是挺高的。

但是，要注意下面的方法：

```py
>>> for k in a_dict.keys():
...     print k, a_dict[k]
... 
lang python
website www.itdiffer.com
name qiwsir
email qiwsir@gmail.com 
```

这种方法其实是不提倡的，虽然实现了同样的效果，但是效率常常是比较低的。切记。

```py
>>> for v in a_dict.values():
...     print v
... 
python
www.itdiffer.com
qiwsir
qiwsir@gmail.com

>>> for v in a_dict.itervalues():
...     print v
... 
python
www.itdiffer.com
qiwsir
qiwsir@gmail.com 
```

单独取 values，推荐第二种方法。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 语句(4)

for 循环在 Python 中应用广泛，所以，要用更多的篇幅来介绍。

### 并行迭代

关于迭代，在《列表(2)》中曾经提到过“可迭代的(iterable)”这个词，并给予了适当解释，这里再次提到“迭代”，说明它在 Python 中占有重要的位置。

迭代，在 Python 中表现就是用 for 循环，从序列对象中获得一定数量的元素。

在前面一节中，用 for 循环来获得列表、字符串、元组，乃至于字典的键值对，都是迭代。

现实中迭代不都是那么简单的，比如这个问题：

**问题：**有两个列表，分别是：a = [1,2,3,4,5], b = [9,8,7,6,5]，要计算这两个列表中对应元素的和。

**解析：**

太简单了，一看就知道结果了。

很好，这是你的方法，如果是 computer 姑娘来做，应该怎么做呢？

观察发现两个列表的长度一样，都是 5。那么对应元素求和，就是相同的索引值对应的元素求和，即 a[i]+b[i],(i=0,1,2,3,4)，这样一个一个地就把相应元素和求出来了。当然，要用 for 来做这个事情了。

```py
>>> a = [1,2,3,4,5]
>>> b = [9,8,7,6,5]
>>> c = []
>>> for i in range(len(a)):
...     c.append(a[i]+b[i])
... 
>>> c
[10, 10, 10, 10, 10] 
```

看来 for 的表现还不错。不过，这种方法虽然解决问题了，但 Python 总不会局限于一个解决之道。于是又有一个内建函数 `zip()`，可以让同样的问题有不一样的解决途径。

zip 是什么东西？在交互模式下用 help(zip),得到官方文档是：

> zip(...) zip(seq1 [, seq2 [...]]) -> [(seq1[0], seq2[0] ...), (...)]
> 
> Return a list of tuples, where each tuple contains the i-th element from each of the argument sequences. The returned list is truncated in length to the length of the shortest argument sequence.

seq1, seq2 分别代表了序列类型的数据。通过实验来理解上面的文档：

```py
>>> a = "qiwsir"
>>> b = "github"
>>> zip(a,b)
[('q', 'g'), ('i', 'i'), ('w', 't'), ('s', 'h'), ('i', 'u'), ('r', 'b')] 
```

如果序列长度不同，那么就以"the length of the shortest argument sequence"为准。

```py
>>> c = [1,2,3]
>>> d = [9,8,7,6]
>>> zip(c,d)
[(1, 9), (2, 8), (3, 7)]

>>> m = {"name","lang"}  
>>> n = {"qiwsir","python"}
>>> zip(m,n)
[('lang', 'python'), ('name', 'qiwsir')] 
```

m，n 是字典吗？当然不是。下面的才是字典呢。

```py
>>> s = {"name":"qiwsir"}
>>> t = {"lang":"python"}
>>> zip(s,t)
[('name', 'lang')] 
```

zip 是一个内置函数，它的参数必须是某种序列数据类型，如果是字典，那么键视为序列。然后将序列对应的元素依次组成元组，做为一个 list 的元素。

下面是比较特殊的情况，参数是一个序列数据的时候，生成的结果样子：

```py
>>> a  
'qiwsir'
>>> c  
[1, 2, 3]
>>> zip(c)
[(1,), (2,), (3,)]
>>> zip(a)
[('q',), ('i',), ('w',), ('s',), ('i',), ('r',)] 
```

很好的 `zip()`！那么就用它来解决前面那个两个列表中值对应相加吧。

```py
>>> d = []
>>> for x,y in zip(a,b):
...     d.append(x+y)
... 
>>> d
[10, 10, 10, 10, 10] 
```

多么优雅的解决！

比较这个问题的两种解法，似乎第一种解法适用面较窄，比如，如果已知给定的两个列表长度不同，第一种解法就出问题了。而第二种解法还可以继续适用。的确如此，不过，第一种解法也不是不能修订的。

```py
>>> a = [1,2,3,4,5]
>>> b = ["python","www.itdiffer.com","qiwsir"] 
```

如果已知是这样两个列表，要讲对应的元素“加起来”。

```py
>>> length = len(a) if len(a)<len(b) else len(b)
>>> length
3 
```

首先用这种方法获得两个列表中最短的那个列表的长度。看那句三元操作，这是非常 Pythonic 的写法啦。写出这句，就可以冒充高手了。哈哈。

```py
>>> for i in range(length):
...     c.append(str(a[i]) + ":" + b[i])
... 
>>> c
['1:python', '2:www.itdiffer.com', '3:qiwsir'] 
```

我还是用第一个思路做的，经过这么修正一下，也还能用。要注意一个细节，在“加”的时候，不能直接用 `a[i]`，因为它引用的对象是一个 int 类型，不能跟后面的 str 类型相加，必须转化一下。

当然，`zip()`也是能解决这个问题的。

```py
>>> d = []
>>> for x,y in zip(a,b):
...     d.append(x + y)
... 
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
TypeError: unsupported operand type(s) for +: 'int' and 'str' 
```

报错！看错误信息，我刚刚提醒的那个问题就冒出来了。所以，应该这么做：

```py
>>> for x,y in zip(a,b):
...     d.append(str(x) + ":" + y)
... 
>>> d
['1:python', '2:www.itdiffer.com', '3:qiwsir'] 
```

这才得到了正确结果。

切记：**computer 是一个姑娘，她非常秀气，需要敲代码的小伙子们耐心地、细心地跟她相处。**

以上两种写法那个更好呢？前者？后者？哈哈。我看差不多了。

```py
>>> result
[(2, 11), (4, 13), (6, 15), (8, 17)]
>>> zip(*result)
[(2, 4, 6, 8), (11, 13, 15, 17)] 
```

`zip()`还能这么干，是不是有点意思？

下面延伸一个问题：

**问题**：有一个 dictionary，myinfor = {"name":"qiwsir","site":"qiwsir.github.io","lang":"python"},将这个字典变换成：infor = {"qiwsir":"name","qiwsir.github.io":"site","python":"lang"}

**解析：**

解法有几个，如果用 for 循环，可以这样做（当然，看官如果有方法，欢迎贴出来）。

```py
>>> infor = {}
>>> for k,v in myinfor.items():
...     infor[v]=k
... 
>>> infor
{'python': 'lang', 'qiwsir.github.io': 'site', 'qiwsir': 'name'} 
```

下面用 zip() 来试试：

```py
>>> dict(zip(myinfor.values(),myinfor.keys()))
{'python': 'lang', 'qiwsir.github.io': 'site', 'qiwsir': 'name'} 
```

呜呼，这是什么情况？原来这个 zip() 还能这样用。是的，本质上是这么回事情。如果将上面这一行分解开来，看官就明白其中的奥妙了。

```py
>>> myinfor.values()    #得到两个 list
['Python', 'qiwsir', 'qiwsir.github.io']
>>> myinfor.keys()
['lang', 'name', 'site']
>>> temp = zip(myinfor.values(),myinfor.keys())     #压缩成一个 list，每个元素是一个 tuple
>>> temp
[('python', 'lang'), ('qiwsir', 'name'), ('qiwsir.github.io', 'site')]

>>> dict(temp)                          #这是函数 dict() 的功能，将上述列表转化为 dictionary
{'Python': 'lang', 'qiwsir.github.io': 'site', 'qiwsir': 'name'} 
```

至此，是不是明白 zip()和循环的关系了呢？有了它可以让某些循环简化。

### enumerate

这是一个有意思的内置函数，本来我们可以通过 `for i in range(len(list))`的方式得到一个 list 的每个元素索引，然后在用 list[i]的方式得到该元素。如果要同时得到元素索引和元素怎么办？就是这样了:

```py
>>> for i in range(len(week)):
...     print week[i]+' is '+str(i)     #注意，i 是 int 类型，如果和前面的用 + 连接，必须是 str 类型
... 
monday is 0
sunday is 1
friday is 2 
```

Python 中提供了一个内置函数 enumerate，能够实现类似的功能

```py
>>> for (i,day) in enumerate(week):
...     print day+' is '+str(i)
... 
monday is 0
sunday is 1
friday is 2 
```

官方文档是这么说的：

> Return an enumerate object. sequence must be a sequence, an iterator, or some other object which supports iteration. The next() method of the iterator returned by enumerate() returns a tuple containing a count (from start which defaults to 0) and the values obtained from iterating over sequence:

顺便抄录几个例子，供看官欣赏，最好实验一下。

```py
>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')] 
```

对于这样一个列表：

```py
>>> mylist = ["qiwsir",703,"python"]
>>> enumerate(mylist)
<enumerate object at 0xb74a63c4> 
```

出现这个结果，用 list 就能实现转换，显示内容.意味着可迭代。

```py
>>> list(enumerate(mylist))
[(0, 'qiwsir'), (1, 703), (2, 'python')] 
```

再设计一个小问题，练习一下这个函数。

**问题：**将字符串中的某些字符替换为其它的字符串。原始字符串"Do you love Canglaoshi? Canglaoshi is a good teacher."，请将"Canglaoshi"替换为"PHP".

**解析：**

```py
>>> raw = "Do you love Canglaoshi? Canglaoshi is a good teacher." 
```

这是所要求的那个字符串，当时，不能直接对这个字符串使用 `enumerate()`，因为它会变成这样：

```py
>>> list(enumerate(raw))
[(0, 'D'), (1, 'o'), (2, ' '), (3, 'y'), (4, 'o'), (5, 'u'), (6, ' '), (7, 'l'), (8, 'o'), (9, 'v'), (10, 'e'), (11, ' '), (12, 'C'), (13, 'a'), (14, 'n'), (15, 'g'), (16, 'l'), (17, 'a'), (18, 'o'), (19, 's'), (20, 'h'), (21, 'i'), (22, '?'), (23, ' '), (24, 'C'), (25, 'a'), (26, 'n'), (27, 'g'), (28, 'l'), (29, 'a'), (30, 'o'), (31, 's'), (32, 'h'), (33, 'i'), (34, ' '), (35, 'i'), (36, 's'), (37, ' '), (38, 'a'), (39, ' '), (40, 'g'), (41, 'o'), (42, 'o'), (43, 'd'), (44, ' '), (45, 't'), (46, 'e'), (47, 'a'), (48, 'c'), (49, 'h'), (50, 'e'), (51, 'r'), (52, '.')] 
```

这不是所需要的。所以，先把 raw 转化为列表：

```py
>>> raw_lst = raw.split(" ") 
```

然后用 `enumerate()`

```py
>>> for i, string in enumerate(raw_lst):
...     if string == "Canglaoshi":
...         raw_lst[i] = "PHP"
... 
```

没有什么异常现象，查看一下那个 raw_lst 列表，看看是不是把"Canglaoshi"替换为"PHP"了。

```py
>>> raw_lst
['Do', 'you', 'love', 'Canglaoshi?', 'PHP', 'is', 'a', 'good', 'teacher.'] 
```

只替换了一个，还有一个没有替换。为什么？仔细观察发现，没有替换的那个是'Canglaoshi?',跟条件判断中的"Canglaoshi"不一样。

修改一下，把条件放宽：

```py
>>> for i, string in enumerate(raw_lst):
...     if "Canglaoshi" in string:
...         raw_lst[i] = "PHP"
... 
>>> raw_lst
['Do', 'you', 'love', 'PHP', 'PHP', 'is', 'a', 'good', 'teacher.'] 
```

好的。然后呢？再转化为字符串？留给读者试试。

### list 解析

先看下面的例子，这个例子是想得到 1 到 9 的每个整数的平方，并且将结果放在 list 中打印出来

```py
>>> power2 = []
>>> for i in range(1,10):
...     power2.append(i*i)
... 
>>> power2
[1, 4, 9, 16, 25, 36, 49, 64, 81] 
```

Python 有一个非常有意思的功能，就是 list 解析，就是这样的：

```py
>>> squares = [x**2 for x in range(1,10)]
>>> squares
[1, 4, 9, 16, 25, 36, 49, 64, 81] 
```

看到这个结果，看官还不惊叹吗？这就是 Python，追求简洁优雅的 Python！

其官方文档中有这样一段描述，道出了 list 解析的真谛：

> List comprehensions provide a concise way to create lists. Common applications are to make new lists where each element is the result of some operations applied to each member of another sequence or iterable, or to create a subsequence of those elements that satisfy a certain condition.

这就是 Python 有意思的地方，也是计算机高级语言编程有意思的地方，你只要动脑筋，总能找到惊喜的东西。

其实，不仅仅对数字组成的 list，所有的都可以如此操作。请在平复了激动的心之后，默默地看下面的代码，感悟一下 list 解析的魅力。

```py
>>> mybag = [' glass',' apple','green leaf ']   #有的前面有空格，有的后面有空格
>>> [one.strip() for one in mybag]              #去掉元素前后的空格
['glass', 'apple', 'green leaf'] 
```

上面的问题，都能用 list 解析来重写。读者不妨试试。

在很多情况下，list 解析的执行效率高，代码简洁明了。是实际写程序中经常被用到的。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 语句(5)

while，翻译成中文是“当...的时候”，这个单词在英语中，常常用来做为时间状语，while ... someone do somthing，这种类型的说法是有的。在 Python 中，它也有这个含义，不过有点区别的是，“当...时候”这个条件成立在一段范围或者时间间隔内，从而在这段时间间隔内让 Python 做好多事情。就好比这样一段情景：

```py
while 年龄大于 60 岁：-------->当年龄大于 60 岁的时候
    退休            -------->凡是符合上述条件就执行的动作 
```

展开想象，如果制作一道门，这道门就是用上述的条件调控开关的，假设有很多人经过这个们，报上年龄，只要年龄大于 60,就退休（门打开，人可以出去），一个接一个地这样循环下去，突然有一个人年龄是 50,那么这个循环在他这里就停止，也就是这时候他不满足条件了。

这就是 while 循环。写一个严肃点的流程，可以看下图：

![](img/12501.png)

### 做猜数字游戏

前不久，有一个在校的大学生朋友（他叫李航），给我发邮件，让我看了他做的游戏，能够实现多次猜数，直到猜中为止。这是一个多么喜欢学习的大学生呀。

我在这里将他写的程序恭录于此，如果李航同学认为此举侵犯了自己的知识产权，可以告知我，我马上撤下此代码。

```py
#! /usr/bin/env Python
#coding:UTF-8           

import random

i=0
while i < 4:
    print'********************************'
    num = input('请您输入 0 到 9 任一个数：')       #李同学用的是 Python3

    xnum = random.randint(0,9)

    x = 3 - i

    if num == xnum:
        print'运气真好，您猜对了！'
        break
    elif num > xnum:
        print'''您猜大了!\n 哈哈,正确答案是:%s\n 您还有 %s 次机会！''' %(xnum,x)
    elif num < xnum:
        print'''您猜小了!\n 哈哈,正确答案是:%s\n 您还有 %s 次机会！''' %(xnum,x)
    print'********************************'

    i += 1 
```

我们就用这段程序来分析一下，首先看 while i<4，这是程序中为猜测限制了次数，最大是三次，请看官注意，在 while 的循环体中的最后一句：i +=1，这就是说每次循环到最后，就给 i 增加 1,当 bool(i<4) 为 False 的时候，就不再循环了。

当 bool(i<4) 为 True 的时候，就执行循环体内的语句。在循环体内，让用户输入一个整数，然后程序随机选择一个整数，最后判断随机生成的数和用户输入的数是否相等，并且用 if 语句判断三种不同情况。

根据上述代码，看官看看是否可以修改？

为了让用户的体验更爽，不妨把输入的整数范围扩大，在 1 到 100 之间吧。

```py
num_input = raw_input("please input one integer that is in 1 to 100:")    #我用的是 Python2.7，在输入指令上区别于李同学 
```

程序用 num_input 变量接收了输入的内容。但是，请列位看官一定要注意，看到这里想睡觉的要打起精神了，我要分享一个多年编程经验：

请牢记：**任何用户输入的内容都是不可靠的。**

这句话含义深刻，但是，这里不做过多的解释，需要各位在随后的编程生涯中体验了。为此，我们要检验用户输入的是否符合我们的要求，我们要求用户输入的是 1 到 100 之间的整数，那么就要做如下检验：

1.  输入的是否是整数
2.  如果是整数，是否在 1 到 100 之间。

为此，要做：

```py
if not num_input.isdigit():     #str.isdigit()是用来判断字符串是否纯粹由数字组成
    print "Please input interger."
elif int(num_input)<0 and int(num_input)>=100:
    print "The number should be in 1 to 100."
else:
    pass       #这里用 pass，意思是暂时省略，如果满足了前面提出的要求，就该执行此处语句 
```

再看看李航同学的程序，在循环体内产生一个随机的数字，这样用户每次输入，面对的都是一个新的随机数字。这样的猜数字游戏难度太大了。我希望是程序产生一个数字，直到猜中，都是这个数字。所以，要把产生随机数字这个指令移动到循环之前。

```py
import random

number = random.randint(1,100)

while True:             #不限制用户的次数了
    ... 
```

观察李同学的程序，还有一点需要向列位显明的，那就是在条件表达式中，两边最好是同种类型数据，上面的程序中有：num>xnum 样式的条件表达式，而一边是程序生成的 int 类型数据，一边是通过输入函数得到的 str 类型数据。在某些情况下可以运行，为什么？看官能理解吗？都是数字的时候，是可以的。但是，这样不好。

那么，按照这种思路，把这个猜数字程序重写一下：

```py
#!/usr/bin/env python
#coding:utf-8

import random

number = random.randint(1,101)

guess = 0

while True:

    num_input = raw_input("please input one integer that is in 1 to 100:")
    guess += 1

    if not num_input.isdigit():
        print "Please input interger."
    elif int(num_input) < 0 or int(num_input) >= 100:
        print "The number should be in 1 to 100."
    else:
        if number == int(num_input):
            print "OK, you are good.It is only %d, then you successed." % guess
            break
        elif number > int(num_input):
            print "your number is more less."
        elif number < int(num_input):
            print "your number is bigger."
        else:
            print "There is something bad, I will not work" 
```

以上供参考，看官还可改进。

### break 和 continue

break,在上面的例子中已经出现了，其含义就是要在这个地方中断循环，跳出循环体。下面这个简要的例子更明显：

```py
#!/usr/bin/env python
#coding:utf-8

a = 8
while a:
    if a%2 == 0:
        break
    else:
        print "%d is odd number"%a
        a = 0 
print "%d is even number"%a 
```

a=8 的时候，执行循环体中的 break，跳出循环，执行最后的打印语句，得到结果：

```py
8 is even number 
```

如果 a=9，则要执行 else 里面的的 print，然后 a=0，循环就在执行一次，又 break 了，得到结果：

```py
9 is odd number
0 is even number 
```

而 continue 则是要从当前位置（即 continue 所在的位置）跳到循环体的最后一行的后面（不执行最后一行），对一个循环体来讲，就如同首尾衔接一样，最后一行的后面是哪里呢？当然是开始了。

```py
#!/usr/bin/env python
#coding:utf-8

a = 9
while a:
    if a%2==0:
        a -=1
        continue    #如果是偶数，就返回循环的开始
    else:
        print "%d is odd number"%a #如果是奇数，就打印出来
        a -=1 
```

其实，对于这两东西，我个人在编程中很少用到。我有一个固执的观念，尽量将条件在循环之前做足，不要在循环中跳来跳去，不仅可读性下降，有时候自己也糊涂了。

### while...else

这两个的配合有点类似 if ... else，只需要一个例子列为就理解了，当然，一遇到 else 了，就意味着已经不在 while 循环内了。

```py
#!/usr/bin/env Python

count = 0
while count < 5:
    print count, " is  less than 5"
    count = count + 1
else:
    print count, " is not less than 5" 
```

执行结果：

```py
0 is less than 5
1 is less than 5
2 is less than 5
3 is less than 5
4 is less than 5
5 is not less than 5 
```

### for...else

除了有 while...else 外，还可以有 for...else。这个循环也通常用在当跳出循环之后要做的事情。

```py
#!/usr/bin/env python
# coding=utf-8

from math import sqrt

for n in range(99, 1, -1):
    root = sqrt(n)
    if root == int(root):
        print n
        break

else:
    print "Nothing." 
```

读 读者是否能够读懂这段代码的含义？

> 阅读代码是一个提升自己编程水平的好方法。如何阅读代码？像看网上新闻那样吗？一目只看自己喜欢的文字，甚至标题看不完就开始喷。
> 
> 绝对不是这样，如果这样，不是阅读代码呢。阅读代码的最好方法是给代码做注释。对，如果可能就给每行代码做注释。这样就能理解代码的含义了。

上面的代码，读者不妨做注释，看看它到底在干什么。如果把 `for n in range(99, 1, -1)`修改为 `for n in range(99, 81, -1)`看看是什么结果？

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 文件(1)

文件，是 computer 姑娘中非常重要的东西，在 Python 里，它也是一种类型的对象，类似前面已经学习过的其它数据类型，包括文本的、图片的、音频的、视频的等等，还有不少没见过的扩展名的。事实上，在 linux 操作系统中，所有的东西都被保存到文件中。

先在交互模式下查看一下文件都有哪些属性：

```py
>>> dir(file)
['__class__', '__delattr__', '__doc__', '__enter__', '__exit__', '__format__', '__getattribute__', '__hash__', '__init__', '__iter__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'close', 'closed', 'encoding', 'errors', 'fileno', 'flush', 'isatty', 'mode', 'name', 'newlines', 'next', 'read', 'readinto', 'readline', 'readlines', 'seek', 'softspace', 'tell', 'truncate', 'write', 'writelines', 'xreadlines'] 
```

然后对部分属性进行详细说明，就是看官学习了。

特别注意观察，在上面有`__iter__`这个东西。曾经在讲述列表的时候，是不是也出现这个东西了呢？是的。它意味着这种类型的数据是可迭代的(iterable)。在下面的讲解中，你就会看到了，能够用 for 来读取其中的内容。

### 打开文件

在某个文件夹下面建立了一个文件，名曰：130.txt，并且在里面输入了如下内容：

```py
learn python
http://qiwsir.github.io
qiwsir@gmail.com 
```

此文件一共三行。

下图显示了这个文件的存储位置：

![](img/12601.png)

在上面截图中，我在当前位置输入了 Python（我已经设置了环境变量，如果你没有，需要写全启动 Python 命令路径），进入到交互模式。在这个交互模式下，这样操作：

```py
>>> f = open("130.txt")     #打开已经存在的文件
>>> for line in f:
...     print line
... 
learn Python

http://qiwsir.github.io

qiwsir@gmail.com 
```

提醒初学者注意，在那个文件夹输入了启动 Python 交互模式的命令，那么，如果按照上面的方法 `open("130.txt")`打开文件，就意味着这个文件 130.txt 是在当前文件夹内的。如果要打开其它文件夹内的文件，请用相对路径或者绝对路径来表示，从而让 python 能够找到那个文件。

将打开的文件，赋值给变量 f，这样也就是变量 f 跟对象文件 130.txt 用线连起来了（对象引用），本质上跟前面所讲述的其它类型数据进行赋值是一样的。

接下来，用 for 来读取文件中的内容，就如同读取一个前面已经学过的序列对象一样，如 list、str、tuple，把读到的文件中的每行，赋值给变量 line。也可以理解为，for 循环是一行一行地读取文件内容。每次扫描一行，遇到行结束符号 \n 表示本行结束，然后是下一行。

从打印的结果看出，每一行跟前面看到的文件内容中的每一行是一样的。只是行与行之间多了一空行，前面显示文章内容的时候，没有这个空行。或许这无关紧要，但是，还要深究一下，才能豁然。

在原文中，每行结束有本行结束符号 \n，表示换行。在 for 语句汇总，print line 表示每次打印完 line 的对象之后，就换行，也就是打印完 line 的对象之后会增加一个 \n。这样看来，在每行末尾就有两个 \n，即：\n\n，于是在打印中就出现了一个空行。

```py
>>> f = open('130.txt')
>>> for line in f:
...     print line,     #后面加一个逗号，就去掉了原来默认增加的 \n 了，看看，少了空行。
... 
learn Python
http://qiwsir.github.io
qiwsir@gmail.com 
```

在进行上述操作的时候，有没有遇到这样的情况呢？

```py
>>> f = open('130.txt')
>>> for line in f:
...     print line,
... 
learn Python
http://qiwsir.github.io
qiwsir@gmail.com

>>> for line2 in f:     #在前面通过 for 循环读取了文件内容之后，再次读取，
...     print line2     #然后打印，结果就什么也显示，这是什么问题？
... 
>>> 
```

如果看官没有遇到上面问题，可以试试。这不是什么错误，是因为前一次已经读取了文件内容，并且到了文件的末尾了。再重复操作，就是从末尾开始继续读了。当然显示不了什么东西，但是 Python 并不认为这是错误，因为后面就会讲到，或许在这次读取之前，已经又向文件中追加内容了。那么，如果要再次读取怎么办？就从新来一边好了。这就好比有一个指针在指着文件中的每一行，每读完一行，指针向移动一行。直到指针指向了文件的最末尾。当然，也有办法把指针移动到任何位置。

特别提醒看官，因为当前的交互模式是在该文件所在目录启动的，所以，就相当于这个实验室和文件 130.txt 是同一个目录，这时候我们打开文件 130.txt，就认为是在本目录中打开，如果文件不是在本目录中，需要写清楚路径。

比如：在上一级目录中（~/Documents/ITArticles/BasicPython），假如我进入到那个目录中，运行交互模式，然后试图打开 130.txt 文件。

![](img/12602.png)

```py
>>> f = open("130.txt")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IOError: [Errno 2] No such file or directory: '130.txt'

>>> f = open("./codes/130.txt")     #必须得写上路径了（注意，windows 的 路径是 \ 隔开，需要转义。对转义符，看官看以前讲座）
>>> for line in f:
...     print line
... 
learn Python

http://qiwsir.github.io

qiwsir@gmail.com

>>> 
```

### 创建文件

上面的实验中，打开的是一个已经存在的文件。如何创建文件呢？

```py
>>> nf = open("131.txt","w")
>>> nf.write("This is a file") 
```

就这样创建了一个文件？并写入了文件内容呢？看看再说：

![](img/12603.png)

真的就这样创建了新文件，并且里面有那句话呢。

看官注意了没有，这次我们同样是用 open() 这个函数，但是多了个"w"，这是在告诉 Python 用什么样的模式打开文件。也就是说，用 open() 打开文件，可以有不同的模式打开。看下表：

| 模式 | 描述 |
| --- | --- |
| r | 以读方式打开文件，可读取文件信息。 |
| w | 以写方式打开文件，可向文件写入信息。如文件存在，则清空该文件，再写入新内容 |
| a | 以追加模式打开文件（即一打开文件，文件指针自动移到文件末尾），如果文件不存在则创建 |
| r+ | 以读写方式打开文件，可对文件进行读和写操作。 |
| w+ | 消除文件内容，然后以读写方式打开文件。 |
| a+ | 以读写方式打开文件，并把文件指针移到文件尾。 |
| b | 以二进制模式打开文件，而不是以文本模式。该模式只对 Windows 或 Dos 有效，类 Unix 的文件是用二进制模式进行操作的。 |

从表中不难看出，不同模式下打开文件，可以进行相关的读写。那么，如果什么模式都不写，像前面那样呢？那样就是默认为 r 模式，只读的方式打开文件。

```py
>>> f = open("130.txt")
>>> f
<open file '130.txt', mode 'r' at 0xb7530230>
>>> f = open("130.txt","r")
>>> f
<open file '130.txt', mode 'r' at 0xb750a700> 
```

可以用这种方式查看当前打开的文件是采用什么模式的，上面显示，两种模式是一样的效果，如果不写那个"r"，就默认为是只读模式了。下面逐个对各种模式进行解释

**"w":以写方式打开文件，可向文件写入信息。如文件存在，则清空该文件，再写入新内容**

131.txt 这个文件是存在的，前面建立的，并且在里面写了一句话：This is a file

```py
>>> fp = open("131.txt")
>>> for line in fp:         #原来这个文件里面的内容
...     print line
... 
This is a file
>>> fp = open("131.txt","w")    #这时候再看看这个文件，里面还有什么呢？是不是空了呢？
>>> fp.write("My name is qiwsir.\nMy website is qiwsir.github.io")  #再查看内容
>>> fp.close() 
```

查看文件内容：

```py
$ cat 131.txt  #cat 是 linux 下显示文件内容的命令，这里就是要显示 131.txt 内容
My name is qiwsir.
My website is qiwsir.github.io 
```

**"a":以追加模式打开文件（即一打开文件，文件指针自动移到文件末尾），如果文件不存在则创建**

```py
>>> fp = open("131.txt","a")
>>> fp.write("\nAha,I like program\n")    #向文件中追加
>>> fp.close()                            #这是关闭文件，一定要养成一个习惯，写完内容之后就关闭 
```

查看文件内容：

```py
$ cat 131.txt
My name is qiwsir.
My website is qiwsir.github.io
Aha,I like program 
```

其它项目就不一一讲述了。看官可以自己实验。

### 使用 with

在对文件进行写入操作之后，一定要牢记一个事情：`file.close()`，这个操作千万不要忘记，忘记了怎么办，那就补上吧，也没有什么天塌地陷的后果。

有另外一种方法，能够不用这么让人揪心，实现安全地关闭文件。

```py
>>> with open("130.txt","a") as f:
...     f.write("\nThis is about 'with...as...'")
... 
>>> with open("130.txt","r") as f:
...     print f.read()
... 
learn python
http://qiwsir.github.io
qiwsir@gmail.com
hello

This is about 'with...as...'
>>> 
```

这里就不用 close()了。而且这种方法更有 Python 味道，或者说是更符合 Pythonic 的一个要求。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 文件(2)

上一节，对文件有了初步认识。读者要牢记，文件无非也是一种类型的数据。

### 文件的状态

很多时候，我们需要获取一个文件的有关状态（也称为属性），比如创建日期，访问日期，修改日期，大小，等等。在 os 模块中，有这样一个方法，专门让我们查看文件的这些状态参数的。

```py
>>> import os
>>> file_stat = os.stat("131.txt")      #查看这个文件的状态
>>> file_stat                           #文件状态是这样的。从下面的内容，有不少从英文单词中可以猜测出来。
posix.stat_result(st_mode=33204, st_ino=5772566L, st_dev=2049L, st_nlink=1, st_uid=1000, st_gid=1000, st_size=69L, st_atime=1407897031, st_mtime=1407734600, st_ctime=1407734600)

>>> file_stat.st_ctime                  #这个是文件创建时间
1407734600.0882277 
```

这是什么时间？看不懂！别着急，换一种方式。在 Python 中，有一个模块 `time`，是专门针对时间设计的。

```py
>>> import time                         
>>> time.localtime(file_stat.st_ctime)  #这回看清楚了。
time.struct_time(tm_year=2014, tm_mon=8, tm_mday=11, tm_hour=13, tm_min=23, tm_sec=20, tm_wday=0, tm_yday=223, tm_isdst=0) 
```

### read/readline/readlines

上节中，简单演示了如何读取文件内容，但是，在用 `dir(file)`的时候，会看到三个函数：read/readline/readlines，它们各自有什么特点，为什么要三个？一个不行吗？

在读者向下看下面内容之前，请想一想，如果要回答这个问题，你要用什么方法？注意，我问的是用什么方法能够找到答案，不是问答案内容是什么。因为内容，肯定是在某个地方存放着呢，关键是用什么方法找到。

搜索？是一个不错的方法。

还有一种，就是在交互模式下使用的，你肯定也想到了。

```py
>>> help(file.read) 
```

用这样的方法，可以分别得到三个函数的说明：

```py
read(...)
    read([size]) -> read at most size bytes, returned as a string.

    If the size argument is negative or omitted, read until EOF is reached.
    Notice that when in non-blocking mode, less data than what was requested
    may be returned, even if no size parameter was given.

readline(...)
    readline([size]) -> next line from the file, as a string.

    Retain newline.  A non-negative size argument limits the maximum
    number of bytes to return (an incomplete line may be returned then).
    Return an empty string at EOF.

readlines(...)
    readlines([size]) -> list of strings, each a line from the file.

    Call readline() repeatedly and return a list of the lines so read.
    The optional size argument, if given, is an approximate bound on the
    total number of bytes in the lines returned. 
```

对照一下上面的说明，三个的异同就显现了。

EOF 什么意思？End-of-file。在[维基百科](http://en.wikipedia.org/wiki/End-of-file)中居然有对它的解释：

```py
In computing, End Of File (commonly abbreviated EOF[1]) is a condition in a computer operating system where no more data can be read from a data source. The data source is usually called a file or stream. In general, the EOF is either determined when the reader returns null as seen in Java's BufferedReader,[2] or sometimes people will manually insert an EOF character of their choosing to signal when the file has ended. 
```

明白 EOF 之后，就对比一下：

*   read：如果指定了参数 size，就按照该指定长度从文件中读取内容，否则，就读取全文。被读出来的内容，全部塞到一个字符串里面。这样有好处，就是东西都到内存里面了，随时取用，比较快捷；“成也萧何败萧何”，也是因为这点，如果文件内容太多了，内存会吃不消的。文档中已经提醒注意在“non-blocking”模式下的问题，关于这个问题，不是本节的重点，暂时不讨论。
*   readline：那个可选参数 size 的含义同上。它则是以行为单位返回字符串，也就是每次读一行，依次循环，如果不限定 size，直到最后一个返回的是空字符串，意味着到文件末尾了(EOF)。
*   readlines：size 同上。它返回的是以行为单位的列表，即相当于先执行 `readline()`，得到每一行，然后把这一行的字符串作为列表中的元素塞到一个列表中，最后将此列表返回。

依次演示操作，即可明了。有这样一个文档，名曰：you.md，其内容和基本格式如下：

> You Raise Me Up When I am down and, oh my soul, so weary; When troubles come and my heart burdened be; Then, I am still and wait here in the silence, Until you come and sit awhile with me. You raise me up, so I can stand on mountains; You raise me up, to walk on stormy seas; I am strong, when I am on your shoulders; You raise me up: To more than I can be.

分别用上述三种函数读取这个文件。

```py
>>> f = open("you.md")
>>> content = f.read()
>>> content
'You Raise Me Up\nWhen I am down and, oh my soul, so weary;\nWhen troubles come and my heart burdened be;\nThen, I am still and wait here in the silence,\nUntil you come and sit awhile with me.\nYou raise me up, so I can stand on mountains;\nYou raise me up, to walk on stormy seas;\nI am strong, when I am on your shoulders;\nYou raise me up: To more than I can be.\n'
>>> f.close() 
```

**提示：养成一个好习惯，**只要打开文件，不用该文件了，就一定要随手关闭它。如果不关闭它，它还驻留在内存中，后面又没有对它的操作，是不是浪费内存空间了呢？同时也增加了文件安全的风险。

> 注意：在 Python 中，'\n'表示换行，这也是 UNIX 系统中的规范。但是，在奇葩的 windows 中，用'\r\n'表示换行。Python 在处理这个的时候，会自动将'\r\n'转换为'\n'。

请仔细观察，得到的就是一个大大的字符串，但是这个字符串里面包含着一些符号 `\n`，因为原文中有换行符。如果用 print 输出这个字符串，就是这样的了，其中的 `\n` 起作用了。

```py
>>> print content
You Raise Me Up
When I am down and, oh my soul, so weary;
When troubles come and my heart burdened be;
Then, I am still and wait here in the silence,
Until you come and sit awhile with me.
You raise me up, so I can stand on mountains;
You raise me up, to walk on stormy seas;
I am strong, when I am on your shoulders;
You raise me up: To more than I can be. 
```

用 `readline()`读取，则是这样的：

```py
>>> f = open("you.md")
>>> f.readline()
'You Raise Me Up\n'
>>> f.readline()
'When I am down and, oh my soul, so weary;\n'
>>> f.readline()
'When troubles come and my heart burdened be;\n'
>>> f.close() 
```

显示出一行一行读取了，每操作一次 `f.readline()`，就读取一行，并且将指针向下移动一行，如此循环。显然，这种是一种循环，或者说可迭代的。因此，就可以用循环语句来完成对全文的读取。

```py
#!/usr/bin/env Python
# coding=utf-8

f = open("you.md")

while True:
    line = f.readline()
    if not line:         #到 EOF，返回空字符串，则终止循环
        break
    print line ,         #注意后面的逗号，去掉 print 语句后面的 '\n'，保留原文件中的换行

f.close()                #别忘记关闭文件 
```

将其和文件"you.md"保存在同一个目录中，我这里命名的文件名是 12701.py，然后在该目录中运行 `Python 12701.py`，就看到下面的效果了：

```py
~/Documents$ python 12701.py 
You Raise Me Up
When I am down and, oh my soul, so weary;
When troubles come and my heart burdened be;
Then, I am still and wait here in the silence,
Until you come and sit awhile with me.
You raise me up, so I can stand on mountains;
You raise me up, to walk on stormy seas;
I am strong, when I am on your shoulders;
You raise me up: To more than I can be. 
```

也用 `readlines()`来读取此文件：

```py
>>> f = open("you.md")
>>> content = f.readlines()
>>> content
['You Raise Me Up\n', 'When I am down and, oh my soul, so weary;\n', 'When troubles come and my heart burdened be;\n', 'Then, I am still and wait here in the silence,\n', 'Until you come and sit awhile with me.\n', 'You raise me up, so I can stand on mountains;\n', 'You raise me up, to walk on stormy seas;\n', 'I am strong, when I am on your shoulders;\n', 'You raise me up: To more than I can be.\n'] 
```

返回的是一个列表，列表中每个元素都是一个字符串，每个字符串中的内容就是文件的一行文字，含行末的符号。显而易见，它是可以用 for 来循环的。

```py
>>> for line in content:
...     print line ,
... 
You Raise Me Up
When I am down and, oh my soul, so weary;
When troubles come and my heart burdened be;
Then, I am still and wait here in the silence,
Until you come and sit awhile with me.
You raise me up, so I can stand on mountains;
You raise me up, to walk on stormy seas;
I am strong, when I am on your shoulders;
You raise me up: To more than I can be.
>>> f.close() 
```

### 读很大的文件

前面已经说明了，如果文件太大，就不能用 `read()`或者 `readlines()`一次性将全部内容读入内存，可以使用 while 循环和 `readlin()`来完成这个任务。

此外，还有一个方法：fileinput 模块

```py
>>> import fileinput
>>> for line in fileinput.input("you.md"):
...     print line ,
... 
You Raise Me Up
When I am down and, oh my soul, so weary;
When troubles come and my heart burdened be;
Then, I am still and wait here in the silence,
Until you come and sit awhile with me.
You raise me up, so I can stand on mountains;
You raise me up, to walk on stormy seas;
I am strong, when I am on your shoulders;
You raise me up: To more than I can be. 
```

我比较喜欢这个，用起来是那么得心应手，简洁明快，还用 for。

对于这个模块的更多内容，读者可以自己在交互模式下利用 `dir()`，`help()`去查看明白。

还有一种方法，更为常用：

```py
>>> for line in f:
...     print line ,
... 
You Raise Me Up
When I am down and, oh my soul, so weary;
When troubles come and my heart burdened be;
Then, I am still and wait here in the silence,
Until you come and sit awhile with me.
You raise me up, so I can stand on mountains;
You raise me up, to walk on stormy seas;
I am strong, when I am on your shoulders;
You raise me up: To more than I can be. 
```

之所以能够如此，是因为 file 是可迭代的数据类型，直接用 for 来迭代即可。

### seek

这个函数的功能就是让指针移动。特别注意，它是以字节为单位进行移动的。比如：

```py
>>> f = open("you.md")
>>> f.readline()
'You Raise Me Up\n'
>>> f.readline()
'When I am down and, oh my soul, so weary;\n' 
```

现在已经移动到第四行末尾了，看 `seek()`的能力：

```py
>>> f.seek(0) 
```

意图是要回到文件的最开头，那么如果用 `f.readline()`应该读取第一行。

```py
>>> f.readline()
'You Raise Me Up\n' 
```

果然如此。此时指针所在的位置，还可以用 `tell()`来显示，如

```py
>>> f.tell()
17L
>>> f.seek(4) 
```

`f.seek(4)`就将位置定位到从开头算起的第四个字符后面，也就是"You "之后，字母"R"之前的位置。

```py
>>> f.tell()
4L 
```

`tell()`也是这么说的。这时候如果使用 `readline()`，得到就是从当前位置开始到行末。

```py
>>> f.readline()
'Raise Me Up\n'
>>> f.close() 
```

`seek()`还有别的参数，具体如下：

> seek(...) seek(offset[, whence]) -> None. Move to new file position.
> Argument offset is a byte count. Optional argument whence defaults to 0 (offset from start of file, offset should be >= 0); other values are 1 (move relative to current position, positive or negative), and 2 (move relative to end of file, usually negative, although many platforms allow seeking beyond the end of a file). If the file is opened in text mode, only offsets returned by tell() are legal. Use of other offsets causes undefined behavior. Note that not all file objects are seekable.

whence 的值：

*   默认值是 0，表示从文件开头开始计算指针偏移的量（简称偏移量）。这是 offset 必须是大于等于 0 的整数。
*   是 1 时，表示从当前位置开始计算偏移量。offset 如果是负数，表示从当前位置向前移动，整数表示向后移动。
*   是 2 时，表示相对文件末尾移动。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 迭代

跟一些比较牛 X 的程序员交流，经常听到他们嘴里冒出一个不标准的英文单词，而 loop、iterate、traversal 和 recursion 如果不在其内，总觉得他还不够牛 X。当让，真正牛 X 的绝对不会这么说的，他们只是说“循环、迭代、遍历、递归”，然后再问“这个你懂吗？”。哦，这就是真正牛 X 的程序员。不过，他也仅仅是牛 X 罢了，还不是大神。大神程序员是什么样儿呢？他是扫地僧，大隐隐于市。

先搞清楚这些名词再说别的：

*   循环（loop），指的是在满足条件的情况下，重复执行同一段代码。比如，while 语句。
*   迭代（iterate），指的是按照某种顺序逐个访问列表中的每一项。比如，for 语句。
*   递归（recursion），指的是一个函数不断调用自身的行为。比如，以编程方式输出著名的斐波纳契数列。
*   遍历（traversal），指的是按照一定的规则访问树形结构中的每个节点，而且每个节点都只访问一次。

对于这四个听起来高深莫测的词汇，其实前面，已经涉及到了一个——循环（loop），本节主要介绍一下迭代（iterate），看官在网上 google，就会发现，对于迭代和循环、递归之间的比较的文章不少，分别从不同角度将它们进行了对比。这里暂不比较，先搞明白 python 中的迭代。

当然，迭代的话题如果要说起来，会很长，本着循序渐进的原则，这里介绍比较初级的。

### 逐个访问

在 python 中，访问对象中每个元素，可以这么做：（例如一个 list）

```py
>>> lst
['q', 'i', 'w', 's', 'i', 'r']
>>> for i in lst:
...     print i,
... 
q i w s i r 
```

除了这种方法，还可以这样：

```py
>>> lst_iter = iter(lst)    #对原来的 list 实施了一个 iter()
>>> lst_iter.next()         #要不厌其烦地一个一个手动访问
'q'
>>> lst_iter.next()
'i'
>>> lst_iter.next()
'w'
>>> lst_iter.next()
's'
>>> lst_iter.next()
'i'
>>> lst_iter.next()
'r'
>>> lst_iter.next()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration 
```

`iter()`是一个内建函数，其含义是：

上面的 `next()`就是要获得下一个元素，但是做为一名优秀的程序员，最佳品质就是“懒惰”，当然不能这样一个一个地敲啦，于是就：

```py
>>> while True:
...     print lst_iter.next()
... 
Traceback (most recent call last):      #居然报错，而且错误跟前面一样？什么原因
  File "<stdin>", line 2, in <module>
StopIteration 
```

先不管错误，再来一遍。

```py
>>> lst_iter = iter(lst)                #上面的错误暂且搁置，回头在研究
>>> while True:
...     print lst_iter.next()
... 
q                                       #果然自动化地读取了
i
w
s
i
r
Traceback (most recent call last):      #读取到最后一个之后，报错，停止循环
  File "<stdin>", line 2, in <module>
StopIteration 
```

首先了解一下上面用到的那个内置函数：iter(),官方文档中有这样一段话描述之：

> iter(o[, sentinel])
> 
> Return an iterator object. The first argument is interpreted very differently depending on the presence of the second argument. Without a second argument, o must be a collection object which supports the iteration protocol (the **iter**() method), or it must support the sequence protocol (the **getitem**() method with integer arguments starting at 0). If it does not support either of those protocols, TypeError is raised. If the second argument, sentinel, is given, then o must be a callable object. The iterator created in this case will call o with no arguments for each call to its next() method; if the value returned is equal to sentinel, StopIteration will be raised, otherwise the value will be returned.

大意是说...(此处故意省略若干字，因为我相信看此文章的看官英语水平是达到看文档的水平了，如果没有，也不用着急，找个词典什么的帮助一下。)

尽管不翻译了，但是还要提炼一下主要的东西：

*   返回值是一个迭代器对象
*   参数需要是一个符合迭代协议的对象或者是一个序列对象
*   next() 配合与之使用

什么是“可迭代的对象”呢？在前面学习的时候，曾经提到过，如果忘记了请往前翻阅。

一般，我们常常将哪些能够用诸如循环语句之类的方法来一个一个读取元素的对象，就称之为可迭代的对象。那么用来循环的如 for 就被称之为迭代工具。

用严格点的语言说：所谓迭代工具，就是能够按照一定顺序扫描迭代对象的每个元素（按照从左到右的顺序）。

显然，除了 for 之外，还有别的可以称作迭代工具。

那么，刚才介绍的 iter() 的功能呢？它与 next() 配合使用，也是实现上述迭代工具的作用。

在 Python 中，甚至在其它的语言中，迭代这块的说法比较乱，主要是名词乱，刚才我们说，那些能够实现迭代的东西，称之为迭代工具，就是这些迭代工具，不少程序员都喜欢叫做迭代器。当然，这都是汉语翻译，英语就是 iterator。

看官看上面的所有例子会发现，如果用 for 来迭代，当到末尾的时候，就自动结束了，不会报错。如果用 iter()...next() 迭代，当最后一个完成之后，它不会自动结束，还要向下继续，但是后面没有元素了，于是就报一个称之为 StopIteration 的错误（这个错误的名字叫做：停止迭代，这哪里是报错，分明是警告）。

看官还要关注 iter()...next() 迭代的一个特点。当迭代对象 lst_iter 被迭代结束，即每个元素都读取了一遍之后，指针就移动到了最后一个元素的后面。如果再访问，指针并没有自动返回到首位置，而是仍然停留在末位置，所以报 StopIteration，想要再开始，需要重新载入迭代对象。所以，当我在上面重新进行迭代对象赋值之后，又可以继续了。这在 for 等类型的迭代工具中是没有的。

### 文件迭代器

现在有一个文件，名称：208.txt，其内容如下：

```py
Learn python with qiwsir.
There is free python course.
The website is:
http://qiwsir.github.io
Its language is Chinese. 
```

用迭代器来操作这个文件，我们在前面讲述文件有关知识的时候已经做过了，无非就是：

```py
>>> f = open("208.txt")
>>> f.readline()        #读第一行
'Learn python with qiwsir.\n'
>>> f.readline()        #读第二行
'There is free python course.\n'
>>> f.readline()        #读第三行
'The website is:\n'
>>> f.readline()        #读第四行
'http://qiwsir.github.io\n'
>>> f.readline()        #读第五行，也就是这真在读完最后一行之后，到了此行的后面
'Its language is Chinese.\n'
>>> f.readline()        #无内容了，但是不报错，返回空。
'' 
```

以上演示的是用 readline() 一行一行地读。当然，在实际操作中，我们是绝对不能这样做的，一定要让它自动进行，比较常用的方法是：

```py
>>> for line in f:     #这个操作是紧接着上面的操作进行的，请看官主要观察
...     print line,    #没有打印出任何东西 
... 
```

这段代码之所没有打印出东西来，是因为经过前面的迭代，指针已经移到了最后了。这就是迭代的一个特点，要小心指针的位置。

```py
>>> f = open("208.txt")     #从头再来
>>> for line in f:
...     print line,
... 
Learn python with qiwsir.
There is free python course.
The website is:
http://qiwsir.github.io
Its language is Chinese. 
```

这种方法是读取文件常用的。另外一个 readlines() 也可以。但是，需要有一些小心的地方，看官如果想不起来小心什么，可以在将关于文件的课程复习一边。

上面过程用 next() 也能够读取。

```py
>>> f = open("208.txt")
>>> f.next()
'Learn python with qiwsir.\n'
>>> f.next()
'There is free python course.\n'
>>> f.next()
'The website is:\n'
>>> f.next()
'http://qiwsir.github.io\n'
>>> f.next()
'Its language is Chinese.\n'
>>> f.next()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration 
```

如果用 next()，就可以直接读取每行的内容。这说明文件是天然的可迭代对象，不需要用 iter() 转换了。

再有，我们用 for 来实现迭代，在本质上，就是自动调用 next()，只不过这个工作，已经让 for 偷偷地替我们干了，到这里，列位是不是应该给 for 取另外一个名字：它叫雷锋。

还有，列表解析也能够做为迭代工具，在研究列表的时候，看官想必已经清楚了。那么对文件，是否可以用？试一试：

```py
>>> [ line for line in open('208.txt') ]
['Learn python with qiwsir.\n', 'There is free python course.\n', 'The website is:\n', 'http://qiwsir.github.io\n', 'Its language is Chinese.\n'] 
```

至此，看官难道还不为列表解析所折服吗？真的很强大，又强又大呀。

其实，迭代器远远不止上述这么简单，下面我们随便列举一些，在 Python 中还可以这样得到迭代对象中的元素。

```py
>>> list(open('208.txt'))
['Learn python with qiwsir.\n', 'There is free python course.\n', 'The website is:\n', 'http://qiwsir.github.io\n', 'Its language is Chinese.\n']

>>> tuple(open('208.txt'))
('Learn python with qiwsir.\n', 'There is free python course.\n', 'The website is:\n', 'http://qiwsir.github.io\n', 'Its language is Chinese.\n')

>>> "$$$".join(open('208.txt'))
'Learn python with qiwsir.\n$$$There is free python course.\n$$$The website is:\n$$$http://qiwsir.github.io\n$$$Its language is Chinese.\n'

>>> a,b,c,d,e = open("208.txt")
>>> a
'Learn python with qiwsir.\n'
>>> b
'There is free python course.\n'
>>> c
'The website is:\n'
>>> d
'http://qiwsir.github.io\n'
>>> e
'Its language is Chinese.\n' 
```

上述方式，在编程实践中不一定用得上，只是向看官展示一下，并且看官要明白，可以这么做，不是非要这么做。

补充一下，字典也可以迭代，看官自己不妨摸索一下（其实前面已经用 for 迭代过了，这次请摸索一下用 iter()...next() 手动一步一步迭代）。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 练习

已经将 Python 的基础知识学习完毕，包含基本的数据类型（或者说对象类型）和语句。利用这些，加上个人的聪明才智，就能解决一些问题了。

#### 练习 1

**问题描述**

有一个列表，其中包括 10 个元素，例如这个列表是[1,2,3,4,5,6,7,8,9,0],要求将列表中的每个元素一次向前移动一个位置，第一个元素到列表的最后，然后输出这个列表。最终样式是[2,3,4,5,6,7,8,9,0,1]

**解析**

或许刚看题目的读者，立刻想到把列表中的第一个元素拿出来，然后追加到最后，不就可以了吗？是的。就是这么简单。主要是联系一下已经学习过的列表操作。

看下面代码之前，不妨自己写一写试试。然后再跟我写的对照。

**注意，我在这里所写的代码不能算标准答案。只能是参考。很可能你写的比我写的还要好。在代码界，没有标准答案。**

参考代码如下，这个我保存为 12901.py 文件

```py
#!/usr/bin/env python
# coding=utf-8

raw = [1,2,3,4,5,6,7,8,9,0]
print raw

b = raw.pop(0)
raw.append(b)
print raw 
```

执行这个文件：

```py
$ python 12901.py
[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
[2, 3, 4, 5, 6, 7, 8, 9, 0, 1] 
```

第一行所打印的是原来的列表，第二行是需要的列表。这里用到的主要是列表的两个函数 `pop()`和 `append()`。如果读者感觉不是很熟悉，或者对这个问题，在我提供的参考之前只有一个模糊认识，但是没有明晰地写出代码，说明对前面的函数还没有烂熟于胸。唯一的方法就是多练习。

#### 练习 2

**问题描述**

按照下面的要求实现对列表的操作：

1.  产生一个列表，其中有 40 个元素，每个元素是 0 到 100 的一个随机整数
2.  如果这个列表中的数据代表着某个班级 40 人的分数，请计算成绩低于平均分的学生人数，并输出
3.  对上面的列表元素从大到小排序

**解析**

这个问题中，需要几个知识点：

第一个是随机产生整数。一种方法是你做 100 个纸片，分别写上 1 到 100 的数字（每张上一个整数），然后放到一个盒子里面。抓出一个，看是几，就讲这个数字写到列表中，直到抓出第 40 个。这样得到的列表是随机了。但是，好像没有 Python 什么事情。那么久要用另外一种方法，让 Python 来做。Python 中有一个模块：random，专门提供随机事件的。

```py
>>> dir(random)
['BPF', 'LOG4', 'NV_MAGICCONST', 'RECIP_BPF', 'Random', 'SG_MAGICCONST', 'SystemRandom', 'TWOPI', 'WichmannHill', '_BuiltinMethodType', '_MethodType', '__all__', '__builtins__', '__doc__', '__file__', '__name__', '__package__', '_acos', '_ceil', '_cos', '_e', '_exp', '_hashlib', '_hexlify', '_inst', '_log', '_pi', '_random', '_sin', '_sqrt', '_test', '_test_generator', '_urandom', '_warn', 'betavariate', 'choice', 'division', 'expovariate', 'gammavariate', 'gauss', 'getrandbits', 'getstate', 'jumpahead', 'lognormvariate', 'normalvariate', 'paretovariate', 'randint', 'random', 'randrange', 'sample', 'seed', 'setstate', 'shuffle', 'triangular', 'uniform', 'vonmisesvariate', 'weibullvariate'] 
```

在这个问题中，只需要 `random.randint()`，专门获取某个范围内的随机整数。

第二个是求平均数，方法是将所有数字求和，然后除以总人数（40）。求和方法就是 `sum()`函数。在计算平均数的时候，要注意，一般平均数不能仅仅是整数，最好保留一位小数吧。这是除法中的知识了。

第三个是列表排序。

下面就依次展开。不忙，在我开始之前，你先试试吧。

```py
#!/usr/bin/env Python
# coding=utf-8

from __future__ import division
import random

score = [random.randint(0,100) for i in range(40)]    #0 到 100 之间，随机得到 40 个整数，组成列表
print score

num = len(score)
sum_score = sum(score)               #对列表中的整数求和
ave_num = sum_score/num              #计算平均数
less_ave = len([i for i in score if i<ave_num])    #将小于平均数的找出来，组成新的列表，并度量该列表的长度
print "the average score is:%.1f" % ave_num
print "There are %d students less than average." % less_ave

sorted_score = sorted(score, reverse=True)    #对原列表排序
print sorted_score 
```

### 练习 3

**问题描述**

如果将一句话作为一个字符串，那么这个字符串中必然会有空格（这里仅讨论英文），比如"How are you."，但有的时候，会在两个单词之间多大一个空格。现在的任务是，如果一个字符串中有连续的两个空格，请把它删除。

**解析**

对于一个字符串中有空格，可以使用《字符串(4)》中提到的 `strip()`等。但是，它不是仅仅去掉一个空格，而是把字符串两遍的空格都去掉。都去掉似乎也没有什么关系，再用空格把单词拼起来就好了。

按照这个思路，我这样写代码，供你参考（更建议你先写出一段来，然后我们两个对照）。

```py
#!/usr/bin/env Python
# coding=utf-8

string = "I love  code."    #在 code 前面有两个空格，应该删除一个
print string                #为了能够清楚看到每步的结果，把过程中的量打印出来

str_lst = string.split(" ")    #以空格为分割，得到词汇的列表
print str_lst

words = [s.strip() for s in str_lst]    #去除单词两边的空格
print words

new_string = " ".join(words)    #以空格为连接符，将单词链接起来
print new_string 
```

保存之后，运行这个代码，结果是：

```py
I love  code.
['I', 'love', '', 'code.']
['I', 'love', '', 'code.']
I love  code. 
```

结果是令人失望的。经过一番折腾，空格根本就没有被消除。最后的输出和一开始的字符串完全一样。泪奔！

查找原因。

从输出中已经清楚表示了。当执行 `string.split(" ")`的时候，是以空格为分割符，将字符串分割，并返回列表。列表中元素是由单词组成。原来字符串中单词之间的空格已经被作为分隔符，那么列表中单词两遍就没有空格了。所以，前面代码中就无需在用 `strip()`去删除空格。另外，特别要注意的是，有两个空格连着呢，其中一个空格作为分隔符，另外一个空格就作为列表元素被返回了。这样一来，分割之后的操作都无作用了。

看官是否明白错误原因了？

如何修改？显然是分割之后，不能用 `strip()`，而是要想办法把那个返回列表中的空格去掉，得到只含有单词的列表。再用空格连接之，就应该对了。所以，我这样修正它。

```py
#!/usr/bin/env Python
# coding=utf-8

string = "I love  code."
print string

str_lst = string.split(" ")
print str_lst

words = [s for s in str_lst if s!=""]    #利用列表解析，将空格检出
print words

new_string = " ".join(words)
print new_string 
```

将文件保存，名为 12903.py，运行之得到下面结果：

```py
I love  code.
['I', 'love', '', 'code.']
['I', 'love', 'code.']
I love code. 
```

OK！完美地解决了问题，去除了 code 前面的一个空格。

### 练习 4

**问题描述**

> 根剧高德纳（Donald Ervin Knuth）的《计算机程序设计艺术》（The Art of Computer Programming），1150 年印度数学家 Gopala 和金月在研究箱子包装物件长宽刚好为 1 和 2 的可行方法数目时，首先描述这个数列。 在西方，最先研究这个数列列的人是比萨的李奥纳多（意大利人斐波那契 Leonardo Fibonacci），他描述兔子生長的数目時用上了这数列。
> 
> 第一个月初有一对刚诞生的兔子;第二个月之后（第三个月初）他们可以生育,每月每对可生育的兔子会诞生下一对新兔子;兔子永不死去
> 
> 假设计 n 月有可生育的兔子总共 a 对，n+1 月就总共有 b 对。在 n+2 月必定总共有 a+b 对： 因为在 n+2 月的时候，前一月（n+1 月）的 b 对兔子可以存留至第 n+2 月（在当月属于新诞生的兔子尚不能生育）。而新生育出的兔子對数等于所有在 n 月就已存在的 a 对

上面故事是一个著名的数列——斐波那契数列——的起源。斐波那契数列用数学方式表示就是：

```py
a0 = 0                (n=0)
a1 = 1                (n=1)
a[n] = a[n-1] + a[n-2]  (n>=2) 
```

我们要做的事情是用程序计算出 n=100 是的值。

在解决这个问题之前，你可以先观看一个[关于斐波那契数列数列的视频](http://swf.ws.126.net/openplayer/v02/-0-2_M9HKRT25D_M9HNA0UNO-vimg1_ws_126_net//image/snapshot_movie/2014/1/6/L/M9HNA8H6L-.swf)，注意，请在墙内欣赏。

**解析**

斐波那契数列是各种编程语言中都要秀一下的东西，通常用在阐述“递归”中。什么是递归？后面的 Python 中也会讲到。不过，在这里不准备讲。

其实，如果用递归来写，会更容易明白。但是，这里我给出一个用 for 循环写的，看看是否能够理解之。

```py
#!/usr/bin/env Python
# coding=utf-8

a, b = 0, 1

for i in range(4):    #改变这里的数，就能得到相应项的结果
    a, b = b, a+b

print a 
```

保存运行之，看看结果和你推算的是否一致。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 自省

特别说明，这一讲的内容不是我写的，是我从[《Python 自省指南》](http://www.ibm.com/developerworks/cn/linux/l-pyint/#ibm-pcon)抄录过来的，当然，为了适合本教程，我在某些地方做了修改或者重写。

### 什么是自省？

在日常生活中，自省（introspection）是一种自我检查行为。自省是指对某人自身思想、情绪、动机和行为的检查。伟大的哲学家苏格拉底将生命中的大部分时间用于自我检查，并鼓励他的雅典朋友们也这样做。他甚至对自己作出了这样的要求：“未经自省的生命不值得存在。”无独有偶，在中国《论语》中，也有这样的名言：“吾日三省吾身”。显然，自省对个人成长多么重要呀。

在计算机编程中，自省是指这种能力：检查某些事物以确定它是什么、它知道什么以及它能做什么。自省向程序员提供了极大的灵活性和控制力。一旦您使用了支持自省的编程语言，就会产生类似这样的感觉：“未经检查的对象不值得实例化。”

整个 Python 语言对自省提供了深入而广泛的支持。实际上，很难想象假如 Python 语言没有其自省特性是什么样子。

学完这节，你就能够轻松洞察到 Python 对象的“灵魂”。

在深入研究更高级的技术之前，我们尽可能用最普通的方式来研究 Python 自省。有些读者甚至可能会争论说：我们开始时所讨论的特性不应称之为“自省”。我们必须承认，它们是否属于自省的范畴还有待讨论。但从本节的意图出发，我们所关心的只是找出有趣问题的答案。

现在让我们以交互方式使用 Python 来开始研究。这是前面已经在使用的一种方式。

### 联机帮助

在交互模式下，用 help 向 Python 请求帮助。

```py
>>> help()

Welcome to Python 2.7!  This is the online help utility.

If this is your first time using Python, you should definitely check out
the tutorial on the Internet at http://docs.python.org/2.7/tutorial/.

Enter the name of any module, keyword, or topic to get help on writing
Python programs and using Python modules.  To quit this help utility and
return to the interpreter, just type "quit".

To get a list of available modules, keywords, or topics, type "modules",
"keywords", or "topics".  Each module also comes with a one-line summary
of what it does; to list the modules whose summaries contain a given word
such as "spam", type "modules spam".

help> 
```

这时候就进入了联机帮助状态，根据提示输入 `keywords`

```py
help> keywords

Here is a list of the Python keywords.  Enter any keyword to get more help.

and                 elif                if                  print
as                  else                import              raise
assert              except              in                  return
break               exec                is                  try
class               finally             lambda              while
continue            for                 not                 with
def                 from                or                  yield
del                 global              pass 
```

现在显示出了 Python 关键词的列表。依照说明亦步亦趋，输入每个关键词，就能看到那个关键词的相关文档。这里就不展示输入的结果了。读者可以自行尝试。要记住，如果从文档说明界面返回到帮助界面，需要按 `q` 键。

这样，我们能够得到联机帮助。从联机帮助状态退回到 Python 的交互模式，使用 `quit` 命令。

```py
help> quit

You are now leaving help and returning to the Python interpreter.
If you want to ask for help on a particular object directly from the
interpreter, you can type "help(object)".  Executing "help('string')"
has the same effect as typing a particular string at the help> prompt.
>>> 
```

联机帮助实用程序会显示关于各种主题或特定对象的信息。

帮助实用程序很有用，并确实利用了 Python 的自省能力。但仅仅使用帮助不会揭示帮助是如何获得其信息的。而且，因为我们的目的是揭示 Python 自省的所有秘密，所以我们必须迅速地跳出对帮助实用程序的讨论。

在结束关于帮助的讨论之前，让我们用它来获得一个可用模块的列表。

模块只是包含 Python 代码的文本文件，其名称后缀是 .py ，关于模块，本教程会在后面有专门的讲解。如果在 Python 提示符下输入 help('modules') ，或在 help 提示符下输入 modules，则会看到一长列可用模块，类似于下面所示的部分列表。自己尝试它以观察您的系统中有哪些可用模块，并了解为什么会认为 Python 是“自带电池”的（自带电池，这是一个比喻，就是说 Python 在被安装时，就带了很多模块，这些模块是你以后开发中会用到的，比喻成电池，好比开发的助力工具），或者说是 Python 一被安装，就已经包含有的模块，不用我们费力再安装了。

```py
>>> help("modules")

Please wait a moment while I gather a list of all available modules...
ANSI                _threading_local    gnomekeyring        repr
BaseHTTPServer      _warnings           gobject             requests
MySQLdb             chardet             lsb_release         sre_parse
......(此处省略一些)
PyQt4               codeop              markupbase          stringprep
Queue               collections         marshal             strop
ScrolledText        colorama            math                struct
......(省略其它的模块)
Enter any module name to get more help.  Or, type "modules spam" to search
for modules whose descriptions contain the word "spam". 
```

因为太多，无法全部显示。你可以子线观察一下，是不是有我们前面已经用过的那个 `math`、`random` 模块呢？

如果是在 Python 交互模式 `>>>` 下，比如要得到有关 math 模块的更多帮助，可以输入 `>>> help("math")`，如果是在帮助模式 `help>` 下，直接输入 `>math` 就能得到关于 math 模块的详细信息。简直太贴心了。

### dir()

尽管查找和导入模块相对容易，但要记住每个模块包含什么却不是这么简单。你或许并不希望总是必须查看源代码来找出答案。幸运的是，Python 提供了一种方法，可以使用内置的 dir() 函数来检查模块（以及其它对象）的内容。

其实，这个东西我们已经一直在使用。

dir() 函数可能是 Python 自省机制中最著名的部分了。它返回传递给它的任何对象的属性名称经过排序的列表。如果不指定对象，则 dir() 返回当前作用域中（这里冒出来一个新名词：“作用域”，暂且不用管它，后面会详解，你就姑且理解为某个范围吧）的名称。让我们将 dir() 函数应用于 keyword 模块，并观察它揭示了什么：

```py
>>> import keyword
>>> dir(keyword)
['__all__', '__builtins__', '__doc__', '__file__', '__name__', '__package__', 'iskeyword', 'kwlist', 'main'] 
```

如果不带任何参数，则 dir() 返回当前作用域中的名称。请注意，因为我们先前导入了 keyword ，所以它们出现在列表中。导入模块将把该模块的名称添加到当前作用域：

```py
>>> dir()
['GFileDescriptorBased', 'GInitiallyUnowned', 'GPollableInputStream', 'GPollableOutputStream', '__builtins__', '__doc__', '__name__', '__package__', 'keyword']
>>> import math
>>> dir()
['GFileDescriptorBased', 'GInitiallyUnowned', 'GPollableInputStream', 'GPollableOutputStream', '__builtins__', '__doc__', '__name__', '__package__', 'keyword', 'math'] 
```

dir() 函数是内置函数，这意味着我们不必为了使用该函数而导入模块。不必做任何操作，Python 就可识别内置函数。

再观察，看到调用 dir() 后返回了这个名称 `__builtins__` 。也许此处有连接。让我们在 Python 提示符下输入名称 `__builtins__` ，并观察 Python 是否会告诉我们关于它的任何有趣的事情：

```py
>>> __builtins__
<module '__builtin__' (built-in)> 
```

因此 `__builtins__` 看起来象是当前作用域中绑定到名为 `__builtin__` 的模块对象的名称。（因为模块不是只有多个单一值的简单对象，所以 Python 改在尖括号中显示关于模块的信息。）

注：如果您在磁盘上寻找 `__builtin__.py` 文件，将空手而归。这个特殊的模块对象是 Python 解释器凭空创建的，因为它包含着解释器始终可用的项。尽管看不到物理文件，但我们仍可以将 dir() 函数应用于这个对象，以观察所有内置函数、错误对象以及它所包含的几个杂项属性。

```py
>>> dir(__builtins__)
['ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'BufferError', 'BytesWarning', 'DeprecationWarning', 'EOFError', 'Ellipsis', 'EnvironmentError', 'Exception', 'False', 'FloatingPointError', 'FutureWarning', 'GeneratorExit', 'IOError', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError', 'KeyError', 'KeyboardInterrupt', 'LookupError', 'MemoryError', 'NameError', 'None', 'NotImplemented', 'NotImplementedError', 'OSError', 'OverflowError', 'PendingDeprecationWarning', 'ReferenceError', 'RuntimeError', 'RuntimeWarning', 'StandardError', 'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError', 'True', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning', 'ValueError', 'Warning', 'ZeroDivisionError', '_', '__debug__', '__doc__', '__import__', '__name__', '__package__', 'abs', 'all', 'any', 'apply', 'ascii', 'basestring', 'bin', 'bool', 'buffer', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'cmp', 'coerce', 'compile', 'complex', 'copyright', 'credits', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'execfile', 'exit', 'file', 'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'intern', 'isinstance', 'issubclass', 'iter', 'len', 'license', 'list', 'locals', 'long', 'map', 'max', 'memoryview', 'min', 'next', 'ngettext', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'quit', 'range', 'raw_input', 'reduce', 'reload', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'unichr', 'unicode', 'vars', 'xrange', 'zip'] 
```

dir() 函数适用于所有对象类型，包括字符串、整数、列表、元组、字典、函数、定制类、类实例和类方法（不理解的对象类型，会在随后的教程中讲解）。例如将 dir() 应用于字符串对象，如您所见，即使简单的 Python 字符串也有许多属性（这是前面已经知道的了，权当复习）

```py
>>> dir("You raise me up")
['__add__', '__class__', '__contains__', '__delattr__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__getslice__', '__gt__', '__hash__', '__init__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_formatter_field_name_split', '_formatter_parser', 'capitalize', 'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill'] 
```

读者可以尝试一下其它的对象类型，观察返回结果，如：`dir(42)`,`dir([])`,`dir(())`,dir({})`,`dir(dir)`。

### 文档字符串

在许多 dir() 示例中，您可能会注意到的一个属性是 `__doc__` 属性。这个属性是一个字符串，它包含了描述对象的注释。Python 称之为文档字符串或 docstring（这个内容，会在下一部分中讲解如何自定义设置）。

如果模块、类、方法或函数定义的第一条语句是字符串，那么该字符串会作为对象的 `__doc__` 属性与该对象关联起来。例如，看一下 str 类型对象的文档字符串。因为文档字符串通常包含嵌入的换行 \n ，我们将使用 Python 的 print 语句，以便输出更易于阅读：

```py
>>> print str.__doc__
str(object='') -> string

Return a nice string representation of the object.
If the argument is a string, the return value is the same object. 
```

### 检查 Python 对象

前面已经好几次提到了“对象（object）”这个词，但一直没有真正定义它。编程环境中的对象很象现实世界中的对象。实际的对象有一定的形状、大小、重量和其它特征。实际的对象还能够对其环境进行响应、与其它对象交互或执行任务。计算机中的对象试图模拟我们身边现实世界中的对象，包括象文档、日程表和业务过程这样的抽象对象。

其实，我总觉得把 object 翻译成对象，让人感觉很没有具象的感觉，因为在汉语里面，对象是一个很笼统的词汇。另外一种翻译，流行于台湾，把它称为“物件”，倒是挺不错的理解。当然，名词就不纠缠了，关键是理解内涵。关于面向对象编程，可以阅读维基百科的介绍——[面向对象程序设计](http://zh.wikipedia.org/zh/%E9%9D%A2%E5%90%91%E5%AF%B9%E8%B1%A1%E7%A8%8B%E5%BA%8F%E8%AE%BE%E8%AE%A1#.E7.89.A9.E4.BB.B6.E5.B0.8E.E5.90.91.E7.9A.84.E8.AF.AD.E8.A8.80)——先了解大概。

类似于实际的对象，几个计算机对象可能共享共同的特征，同时保持它们自己相对较小的变异特征。想一想您在书店中看到的书籍。书籍的每个物理副本都可能有污迹、几张破损的书页或唯一的标识号。尽管每本书都是唯一的对象，但都拥有相同标题的每本书都只是原始模板的实例，并保留了原始模板的大多数特征。

对于面向对象的类和类实例也是如此。例如，可以看到每个 Python 符串都被赋予了一些属性， dir() 函数揭示了这些属性。

于是在计算机术语中，对象是拥有标识和值的事物，属于特定类型、具有特定特征和以特定方式执行操作。并且，对象从一个或多个父类继承了它们的许多属性。除了关键字和特殊符号（象运算符，如 + 、 - 、 * 、 ** 、 / 、 % 、 < 、 > 等）外，Python 中的所有东西都是对象。Python 具有一组丰富的对象类型：字符串、整数、浮点、列表、元组、字典、函数、类、类实例、模块、文件等。

当您有一个任意的对象（也许是一个作为参数传递给函数的对象）时，可能希望知道一些关于该对象的情况。如希望 Python 告诉我们：

*   对象的名称是什么？
*   这是哪种类型的对象？
*   对象知道些什么？
*   对象能做些什么？
*   对象的父对象是谁？

#### 名称

并非所有对象都有名称，但那些有名称的对象都将名称存储在其 `__name__` 属性中。注：名称是从对象而不是引用该对象的变量中派生的。

```py
>>> dir()    #dir() 函数
['GFileDescriptorBased', 'GInitiallyUnowned', 'GPollableInputStream', 'GPollableOutputStream', '__builtins__', '__doc__', '__name__', '__package__', 'keyword', 'math']
>>> directory = dir    #新变量
>>> directory()        #跟 dir() 一样的结果
['GFileDescriptorBased', 'GInitiallyUnowned', 'GPollableInputStream', 'GPollableOutputStream', '__builtins__', '__doc__', '__name__', '__package__', 'directory', 'keyword', 'math']
>>> dir.__name__       #dir() 的名字
'dir'
>>> directory.__name__
'dir'

>>> __name__          #这是不一样的   
'__main__' 
```

模块拥有名称，Python 解释器本身被认为是顶级模块或主模块。当以交互的方式运行 Python 时，局部 `__name__` 变量被赋予值 `'__main__'` 。同样地，当从命令行执行 Python 模块，而不是将其导入另一个模块时，其 `__name__` 属性被赋予值 `'__main__'` ，而不是该模块的实际名称。这样，模块可以查看其自身的 `__name__` 值来自行确定它们自己正被如何使用，是作为另一个程序的支持，还是作为从命令行执行的主应用程序。因此，下面这条惯用的语句在 Python 模块中是很常见的：

```py
if __name__ == '__main__':
    # Do something appropriate here, like calling a
    # main() function defined elsewhere in this module.
    main()
else:
    # Do nothing. This module has been imported by another
    # module that wants to make use of the functions,
    # classes and other useful bits it has defined. 
```

#### 类型

type() 函数有助于我们确定对象是字符串还是整数，或是其它类型的对象。它通过返回类型对象来做到这一点，可以将这个类型对象与 types 模块中定义的类型相比较：

```py
>>> import types
>>> print types.__doc__
Define names for all type symbols known in the standard interpreter.

Types that are part of optional modules (e.g. array) are not listed.

>>> dir(types)
['BooleanType', 'BufferType', 'BuiltinFunctionType', 'BuiltinMethodType', 'ClassType', 'CodeType', 'ComplexType', 'DictProxyType', 'DictType', 'DictionaryType', 'EllipsisType', 'FileType', 'FloatType', 'FrameType', 'FunctionType', 'GeneratorType', 'GetSetDescriptorType', 'InstanceType', 'IntType', 'LambdaType', 'ListType', 'LongType', 'MemberDescriptorType', 'MethodType', 'ModuleType', 'NoneType', 'NotImplementedType', 'ObjectType', 'SliceType', 'StringType', 'StringTypes', 'TracebackType', 'TupleType', 'TypeType', 'UnboundMethodType', 'UnicodeType', 'XRangeType', '__builtins__', '__doc__', '__file__', '__name__', '__package__']
>>> p = "I love Python"
>>> type(p)
<type 'str'>
>>> if type(p) is types.StringType:
...     print "p is a string"
... 
p is a string
>>> type(42)
<type 'int'>
>>> type([])
<type 'list'>
>>> type({})
<type 'dict'>
>>> type(dir)
<type 'builtin_function_or_method'> 
```

#### 标识

先前说过，每个对象都有标识、类型和值。值得注意的是，可能有多个变量引用同一对象，同样地，变量可以引用看起来相似（有相同的类型和值），但拥有截然不同标识的多个对象。当更改对象时（如将某一项添加到列表），这种关于对象标识的概念尤其重要，如在下面的示例中， blist 和 clist 变量引用同一个列表对象。正如您在示例中所见， id() 函数给任何给定对象返回唯一的标识符。其实，这个东东我们也在前面已经使用过了。在这里再次提出，能够让你理解上有提升吧。

```py
>>> print id.__doc__
id(object) -> integer

Return the identity of an object.  This is guaranteed to be unique among
simultaneously existing objects.  (Hint: it's the object's memory address.)
>>> alist = [1,2,3]
>>> blist = [1,2,3]
>>> clist = blist
>>> id(alist)
2979691052L
>>> id(blist)
2993911916L
>>> id(clist)
2993911916L
>>> alist is blist
False
>>> blist is clist
True
>>> clist.append(4)
>>> clist
[1, 2, 3, 4]
>>> blist
[1, 2, 3, 4]
>>> alist
[1, 2, 3] 
```

如果对上面的操作还有疑惑，可以回到前面复习有关深拷贝和浅拷贝的知识。

#### 属性

对象拥有属性，并且 `dir()` 函数会返回这些属性的列表。但是，有时我们只想测试一个或多个属性是否存在。如果对象具有我们正在考虑的属性，那么通常希望只检索该属性。这个任务可以由 hasattr() 和 getattr() 函数来完成.

```py
>>> print hasattr.__doc__
hasattr(object, name) -> bool

Return whether the object has an attribute with the given name.
(This is done by calling getattr(object, name) and catching exceptions.)

>>> print getattr.__doc__
getattr(object, name[, default]) -> value

Get a named attribute from an object; getattr(x, 'y') is equivalent to x.y.
When a default argument is given, it is returned when the attribute doesn't
exist; without it, an exception is raised in that case.
>>> 
>>> hasattr(id, '__doc__')
True

>>> print getattr(id, '__doc__')
id(object) -> integer

Return the identity of an object.  This is guaranteed to be unique among
simultaneously existing objects.  (Hint: it's the object's memory address.) 
```

#### 可调用

可以调用表示潜在行为（函数和方法）的对象。可以用 callable() 函数测试对象的可调用性：

```py
>>> print callable.__doc__
callable(object) -> bool

Return whether the object is callable (i.e., some kind of function).
Note that classes are callable, as are instances with a __call__() method.
>>> callable("a string")
False
>>> callable(dir)
True 
```

#### 实例

这个名词还很陌生，没关系，先看看，混个脸熟，以后会经常用到。

在 type() 函数提供对象的类型时，还可以使用 isinstance() 函数测试对象，以确定它是否是某个特定类型或定制类的实例：

```py
>>> print isinstance.__doc__
isinstance(object, class-or-type-or-tuple) -> bool

Return whether an object is an instance of a class or of a subclass thereof.
With a type as second argument, return whether that is the object's type.
The form using a tuple, isinstance(x, (A, B, ...)), is a shortcut for
isinstance(x, A) or isinstance(x, B) or ... (etc.).
>>> isinstance(42, str)
False
>>> isinstance("python", str)
True 
```

#### 子类

关于类的问题，有一个“继承”概念，有继承就有父子问题，这是在现实生活中很正常的，在编程语言中也是如此。虽然这是后面要说的，但是，为了本讲内容的完整，也姑且把这个内容放在这里。读者可以不看，留着以后看也行。我更建议还是阅读一下，有个印象。

在类这一级别，可以根据一个类来定义另一个类，同样地，这个新类会按照层次化的方式继承属性。Python 甚至支持多重继承，多重继承意味着可以用多个父类来定义一个类，这个新类继承了多个父类。 issubclass() 函数使我们可以查看一个类是不是继承了另一个类：

```py
>>> print issubclass.__doc__
issubclass(C, B) -> Boolean
Return whether class C is a subclass (i.e., a derived class) of class B.
>>> class SuperHero(Person):   # SuperHero inherits from Person...
...     def intro(self):       # but with a new SuperHero intro
...         """Return an introduction."""
...         return "Hello, I'm SuperHero %s and I'm %s." % (self.name, self.age)
...
>>> issubclass(SuperHero, Person)
1
>>> issubclass(Person, SuperHero)
0 
```

### Python 文档

文档，这个词语在经常在程序员的嘴里冒出来，有时候他们还经常以文档有没有或者全不全为标准来衡量一个软件项目是否高大上。那么，软件中的文档是什么呢？有什么要求呢？Python 文档又是什么呢？文档有什么用呢？

文档很重要。独孤九剑的剑诀、易筋经的心法、写着辟邪剑谱的袈裟，这些都是文档。连那些大牛人都要这些文档，更何况我们呢？所以，文档是很重要的。

文档，说白了就是用 word（这个最多了）等（注意这里的等，把不常用的工具都等掉了，包括我编辑文本时用的 vim 工具）文本编写工具写成的包含文本内容但不限于文字的文件。有点啰嗦，啰嗦的目的是为了严谨，呵呵。最好还是来一个更让人信服的定义，当然是来自维基百科。

> 软件文档或者源代码文档是指与软件系统及其软件工程过程有关联的文本实体。文档的类型包括软件需求文档，设计文档，测试文档，用户手册等。其中的需求文档，设计文档和测试文档一般是在软件开发过程中由开发者写就的，而用户手册等非过程类文档是由专门的非技术类写作人员写就的。
> 
> 早期的软件文档主要指的是用户手册，根据 Barker 的定义，文档是用来对软件系统界面元素的设计、规划和实现过程的记录，以此来增强系统的可用性。而 Forward 则认为软件文档是被软件工程师之间用作沟通交流的一种方式，沟通的信息主要是有关所开发的软件系统。Parnas 则强调文档的权威性，他认为文档应该提供对软件系统的精确描述。
> 
> 综上，我们可以将软件文档定义为：

1.文档是一种对软件系统的书面描述； 2.文档应当精确地描述软件系统； 3.软件文档是软件工程师之间用作沟通交流的一种方式； 4.文档的类型有很多种，包括软件需求文档，设计文档，测试文档，用户手册等； 5.文档的呈现方式有很多种，可以是传统的书面文字形式或图表形式，也可是动态的网页形式

那么这里说的 Python 文档指的是什么呢？一个方面就是每个学习者要学习 Python，Python 的开发者们（他们都是大牛）给我们这些小白提供了什么东西没有？能够让我们给他们这些大牛沟通，理解 Python 中每个函数、指令等的含义和用法呢？

有。大牛就是大牛，他们准备了，而且还不止一个。

真诚的敬告所有看本教程的诸位，要想获得编程上的升华，看文档是必须的。文档胜过了所有的教程和所有的老师以及所有的大牛。为什么呢？其中原因，都要等待看官看懂了之后，有了体会感悟之后才能明白。

Python 文档的网址：[`docs.python.org/2/`](https://docs.Python.org/2/)，这是 Python2.x，从这里也可以找到 Python3.x 的文档。

当然，除了看官方文档之外，自己写的东西也可以写上文档。这个先不要着急，我们会在后续的学习中看到。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。