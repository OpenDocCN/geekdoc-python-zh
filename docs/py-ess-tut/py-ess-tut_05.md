# 第五章 条件、循环和其他语句

> 来源：[`www.cnblogs.com/Marlowes/p/5329066.html`](http://www.cnblogs.com/Marlowes/p/5329066.html)
> 
> 作者：Marlowes

读者学到这里估计都有点不耐烦了。好吧，这些数据结构什么的看起来都挺好，但还是没法用它们做什么事，对吧？

下面开始，进度会慢慢加快。前面已经介绍过了几种基本语句(`print`语句、`import`语句、赋值语句)。在深入介绍*条件语句*和*循环语句*之前，我们先来看看这几种基本语句更多的使用方法。随后你会看到列表推倒式(list comprehension)如何扮演循环和条件语句的角色——尽管它本身是表达式。最后介绍`pass`、`del`和`exec`语句的用法。

## 5.1 `print`和`import`的更多信息

随着更加深入第学习 Python，可能会出现这种感觉：有些自己以为已经掌握的知识点，还隐藏着一些让人惊讶的特性。首先来看看`print`(在 Python3.0 中，`print`不再是语句——而是函数(功能基本不变))和`import`的几个比较好的特性。

*注：对于很多应用程序来说，使用`logging`模块记日志比`print`语句更合适。更多细节请参见第十九章。*

### 5.1.1 使用逗号输出

前面的章节中讲解过如何使用`print`来打印表达式——不管是字符串还是其他类型进行自动转换后的字符串。但是事实上打印多个表达式也是可行的，只要将它们用逗号隔开就好：

```py
>>> print "Age", 19 
Age 19 
```

可以看到，每个参数之间都插入了一个空格符。

*注：print 的参数并不能像我们预期那样构成一个元组：*

```py
>>> 1, 2, 3 
(1, 2, 3) 
>>> print 1, 2, 3
1 2 3
>>> print (1, 2, 3)
(1, 2, 3) 
```

如果想要同时输出文本和变量值，却又不希望使用字符串格式化的话，那这个特性就非常有用了：

```py
>>> name = "XuHoo"
>>> salutation = "Mr."
>>> greeting = "Hello,"
>>> print greeting, salutation, name
Hello, Mr. XuHoo 
# 注意，如果 greeting 字符串不带逗号，那么结果中怎么能得到逗号呢？像下面这样做是不行的：
>>> print greeting, ",", salutation, name
Hello , Mr. XuHoo 
# 因为上面的语句会在逗号前加入空格。下面是一种解决方案：
>>> print greeting + ",", salutation, name
Hello, Mr. XuHoo 
# 这样一来，问候语后面就只会增加一个逗号。 
```

如果在结尾处加上逗号，那么接下来的语句会与前一条语句在同一行打印，例如：

```py
print "Hello", print "world!"

# 输出 Hello, world!(这只在脚本中起作用，而在交互式 Python 会话中则没有效果。在交互式会话中，所有的语句都会被单独执行(并且打印出内容)) 
```

### 5.1.2 把某件事作为另一件事导入

从模块导入函数的时候，通常可以使用以下几种方式：

```py
import somemodule # or
from somemodule import somefunction 
# or
from somemodule import somefunction, anotherfunction, yetanotherfunction 
# or
from somemodule import * 
```

只有确定自己想要从给定的模块导入所有功能时，才应该使用最后一个版本。但是如果两个模块都有`open`函数，那又该怎么办？只需要使用第一种方式导入，然后像下面这样使用函数：

```py
import module1 import module2

module1.open(...)
module2.open(...) 
```

但还有另外的选择：可以在语句末尾增加一个`as`子句，在该子句后给出想要使用的别名。例如为整个模块提供别名：

```py
>>> import math as foobar 
>>> foobar.sqrt(4) 
2.0

# 或者为函数提供别名
>>> from math import sqrt as foobar 
>>> foobar(4) 
2.0

# 对于 open 函数，可以像下面这样使用：
from module1 import open as open1 
from module2 import open as open2 
```

注：有些模块，例如`os.path`是分层次安排的(一个模块在另一个模块的内部)。有关模块结构的更多信息，请参见第十章关于包的部分。

## 5.2 赋值魔法

就算是不起眼的赋值语句也有一些特殊的技巧。

### 5.2.1 序列解包

赋值语句的例子已经给过不少，其中包括对变量和数据结构成员的(比如列表中的位置和分片以及字典中的槽)赋值。但赋值的方法还不止这些。比如，多个赋值操作可以*同时*进行：

```py
>>> x, y, z = 1, 2, 3
>>> print x, y, z 1 2 3

# 很有用吧？用它交换两个(或更多个)变量也是没问题的：
>>> x, y = y, x >>> print x, y, z 2 1 3 
```

事实上，这里所做的事情叫做序列解包(sequence unpacking)或*递归解包*——将多个值的序列解开，然后放到变量的序列中。更形象一点的表示就是：

```py
>>> values = 1, 2, 3
>>> values
(1, 2, 3) >>> x, y, z  = values >>> x 1
>>> y 2
>>> z 3 
```

当函数或者方法返回元组(或者其他序列或可迭代对象)时，这个特性尤其有用。假设需要获取(和删除)字典中任意的键-值对，可以使用`popitem`方法，这个方法将键-值作为元组返回。那么这个元组就可以直接赋值到两个变量中：

```py
>>> scoundrel = {"name": "XuHoo", "girlfriend": "None"}  
# =_=
>>> key, value = scoundrel.popitem() 
>>> key 'girlfriend'
>>> value 'None' 
```

它允许函数返回一个以上的值并且打包成元组，然后通过一个赋值语句很容易进行访问。所解包的序列中的元素数量必须和放置在赋值符号=左边的变量数量完全一致，否则 Python 会在赋值时引发异常：

```py
>>> x, y, z = 1, 2 Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  ValueError: need more than 2 values to unpack 
>>> x, y, z = 1, 2, 3, 4 
  Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  ValueError: too many values to unpack 
```

*注：Python3.0 中有另外一个解包的特性：可以像在函数的参数列表中一样使用星号运算符(参见第六章)。例如，`a, b, *rest = [1, 2, 3, 4]`最终会在`a`和`b`都被赋值之后将所有的其他的参数都收集到`rest`中。本例中，`rest`的结果将会是`[3, 4]`。使用星号的变量也可以放在第一个位置，这样它就总会包含一个列表。右侧的赋值语句可以是可迭代对象。*

### 5.2.2 链式赋值

链式赋值(charned assignment)是将同一个值赋给多个变量的捷径。它看起来有些像上节中并行赋值，不过这里只处理一个值：

```py
x = y = somefunction() # 和下面语句的效果是一样的：
y = somefunction()
x = y 
# 注意上面的语句和下面的语句不一定等价：
x = somefunction()
y = somefunction() 
```

有关链式赋值更多的信息，请参见本章中的“同一性运算符”一节。

### 5.2.3 增量赋值

这里没有将赋值表达式写为`x=x+1`，而是将表达式运算符(本例中是`±`)放置在赋值运算符`=`的左边，写成`x+=1`,。这种写法叫做增量赋值(augmented assignmnet)，对于`*`、`/`、`%`等标准运算符都适用：

```py
>>> x = 2
>>> x += 1
>>> x *= 2
>>> x 6

# 对于其他数据类型也适用(只要二元运算符本身适用于这些数据类型即可)：
>>> fnord = "foo"
>>> fnord += "bar"
>>> fnord *= 2
>>> fnord 'foobarfoobar' 
```

增量赋值可以让代码更加紧凑和简练，很多情况下会更易读。

## 5.3 语句块：缩排的乐趣

语句块并非一种语句，而是在掌握后面两节的内容之前应该了解的知识。

语句块是在条件为真(条件语句)时执行或者执行多次(循环语句)的*一组*语句。在代码前放置空格来缩进语句即可创建语句块。

*注：使用`tab`字符也可以缩进语句块。Python 将一个`tab`字符解释为到下一个`tab`字符位置的移动，而一个`tab`字符位置为 8 个空格，但是标准且推荐的方式是只用空格，尤其是在每个缩进需要 4 个空格的时候。*

块中的每行都应该缩进*同样的量*。下面的伪代码(并非真正 Python 代码)展示了缩进的工作方法：

```py
this is a line
this is another line:
    this is another block
    continuing the same block
    the last line of this block
phew, there we escaped the inner block 
```

很多语言使用特殊单词或者字符(比如`begin`或`{`)来表示一个语句块的开始，使用另外的单词或者字符(比如`end`或者`}`)表示语句块的结束。在 Python 中，冒号(`:`)用来标识语句块的开始，块中的每一个语句都是缩进(缩进量相同)。当回退到和已经闭合的块一样的缩进量时，就表示当前块已经结束了(很多程序编辑器和集成开发环境都知道如何缩进语句块，可以帮助用户轻松把握缩进)。

现在我确信你已经等不及想知道语句块怎么使用了。废话不多说，我们来看一下。

## 5.4 条件和条件语句

到目前为止的程序都是一条一条语句顺序执行的。在这部分中会介绍让程序选择是否执行语句块的方法。

### 5.4.1 这就是布尔变量的作用

*真值*(也叫作*布尔值*，这个名字根据在真值上做过大量研究的 George Boole 命名的)是接下来内容的主角。

*注：如果注意力够集中，你就会发现在第一章的“管窥：`if`语句”中就已经描述过`if`语句。到目前为止这个语句还没有被正式介绍。实际上，还有很多`if`语句的内容没有介绍。*

下面的值在作为布尔表达式的时候，会被解释器看做假(`False`)：

```py
False    None    0    ""    ()    []    {} 
```

换句话说，也就是标准值`False`和`None`、所有类型的数字`0`(包括浮点型、长整型和其他类型)、空序列(比如空字符串、元组和列表)以及空的字典都为假。其他的一切(至少当我们讨论內建类型是是这样——第九章內会讨论构建自己的可以被解释为真或假的对象)都被解释为真，包括特殊值`True`(Python 经验丰富的 Laura Creighton 解释说这个区别类似于“有些东西”和“没有东西”的区别，而不是*真*和*假*的区别)。

明白了吗？也就是说 Python 中的所有值都能被解释为真值，初次接触的时候可能会有些搞不明白，但是这点的确非常有用。“标准的”布尔值为`True`和`False`。在一些语言中(例如 C 和 Python2.3 以前的版本)，标准的布尔值为`0`(表示假)和`1`(表示真)。事实上，`True`和`False`只不过是`1`和`0`的一种“华丽”的说法而已——看起来不同，但作用相同。

```py
>>> True
True 
>>> False
False 
>>> True == 1 
True 
>>> False == 0
True 
>>> True + False 
1
>>> True + False + 19
20 
```

那么，如果某个逻辑表达式返回`1`或`0`(在老版本 Python 中)，那么它实际的意思是返回`True`或`False`。

布尔值`True`和`False`属于布尔类型，`bool`函数可以用来(和`list`、`str`以及`tuple`一样)转换其他值。

```py
>>> bool("I think, therefore I am")
True 
>>> bool(19)
True 
>>> bool("")
False 
>>> bool(0)
False 
```

因为所有值都可以用作布尔值，所以几乎不需要对它们进行显示转换(可以说 Python 会自动转换这些值)。

注：尽管`[]`和`""`都是假肢(也就是说`bool([])==bool("")==False`)，它们本身却并不相等(也就是说`[]!=""`)。对于其他不同类型的假值对象也是如此(例如`()!=False`)。

### 5.4.2 条件执行和`if`语句

真值可以联合使用(马上就要介绍)，但还是让我们先看看它们的作用。试着运行下面的脚本：

```py
name = raw_input("What is your name? ") 
if name.endswith("XuHoo"): 
    print "Hello, Mr.XuHoo" 
```

这就是`if`语句，它可以实现*条件执行*，即如果条件(在`if`和冒号之间的表达式)判定为*真*，那么后面的语句块(本例中是单个`print`语句)就会被执行。如果条件为假，语句块就不会被执行(你猜到了，不是吗)。

注：在第一章的“管窥：`if`语句”中，所有语句都写在一行中。这种书写方式和上例中的使用单行语句块的方式是等价的。

### 5.4.3 else 子句

前一节的例子中，如果用户输入了以`XuHoo`作为结尾的名字，那么`name.endswit`方法就会返回真，使得`if`进入语句块，打印出问候语。也可以使用`else`子句增加一种选择(之所以叫做*子句*是因为它不是独立的语句，而只能作为`if`语句的一部分)。

```py
name = raw_input("What is your name? ") 
if name.endswith("XuHoo"): 
    print "Hello, Mr.XuHoo"
else: 
    print "Hello. stranger" 
```

如果第一个语句块没有被执行(因为条件被判定为假)，那么就会站转入第二个语句块，可以看到，阅读 Python 代码很容易，不是吗？大声把代码读出来(从`if`开始)，听起来就像正常(也可能不是很正常)句子一样。

### 5.4.4 elif 子句

如果需要检查多个条件，就可以使用`elif`，它是`else if`的简写，也是`if`和`else`子句的联合使用，也就是具有条件的`else`子句。

```py
name = input("Enter a number: ") 
if num > 0: 
    print "The number is positive"
elif num < 0: 
    print "The number is negative"
else: 
    print "The number is zero" 
```

*注：可以使用`int(raw_input(...))`函数来代替`input(...)`。关于两者的区别，请参见第一章。*

### 5.4.5 嵌套代码块

下面的语句中加入了一些不必要的内容。if 语句里面可以嵌套使用`if`语句，就像下面这样：

```py
name = raw_input("What is your name? ") 
if name.endswith("XuHoo"): 
    if name.startswith("Mr."): 
        print "Hello, Mr. XuHoo"
    elif name.startswith("Mrs."): 
        print "Hello, Mrs. XuHoo"
    else: 
        print "Hello, XuHoo"
else: 
    print "Hello, stranger" 
```

如果名字是以`XuHoo`结尾的话，还要检查名字的开头——在第一个语句块中的单独的`if`语句中。注意这里`elif`的使用。最后一个选项中(`else`子句)没有条件——如果其他的条件都不满足就使用最后一个。可以把任何一个`else`子句放在语句块外面。如果把里面的`else`子句放在外面的话，那么不以`Mr.`或`Mrs.`开头(假设这个名字是`XuHoo`)的名字都被忽略掉了。如果不写最后一个`else`子句，那么陌生人就被忽略掉。

### 5.4.6 更复杂的条件

以上就是有关 if 语句的所有知识。下面让我们回到条件本身，因为它们才是条件执行时真正有趣的部分。

1\. 比较运算符

用在条件中的最基本的运算符就是*比较运算符*了，它们用来比较其他对象。比较运算符已经总结在表 5-1 中。

表 5-1 Python 中的比较运算符

```py
x = y　　　　　　　　　　x 等于 y
x < y　　　　　　   　　 x 小于 y
x > y　　　　　　   　　 x 大于 y
x >= y　　　　　　   　  x 大于等于 y
x <= y　　　　　　   　　x 小于等于 y
x != y　　　　　　　　　 x 不等于 y
x is y　　　　　　　　　 x 和 y 是同一个对象
x is not y　　　　　　　 x 和 y 是不同的对象
x in y　　　　　　　　　 x 是 y 容器(例如，序列)的成员
x not in y　　　　　　　 x 不是 y 容器(例如，序列)的成员 
```

**比较不兼容类型**

理论上，对于相对大小的任意两个对象`x`和`y`都是可以使用比较运算符(例如，`<`和`<=`)比较的，并且都会得到一个布尔值结果。但是只有在`x`和`y`是相同或者近似类型的对象时，比较才有意义(例如，两个整型数或者一个整型数和一个浮点型数进行比较)。

正如将一个整型数添加到一个字符串中是没有意义的，检查一个整型是否比一个字符串小，看起来也是毫无意义的。但奇怪的是，在 Python3.0 之前的版本中这却是可以的。对于此类比较行为，读者应该敬而远之，因为结果完全不可靠，在每次程序执行的时候得到的结果都可能不同。在 Python3.0 中，比较不兼容类型的对象已经不再可行。

*注：如果你偶然遇见 `x <> y` 这样的表达式，它的意思其实就是 `x != y`。不建议使用`<>`运算符，应该尽量避免使用它。*

在 Python 中比较运算符和赋值运算符一样是可以*连接*的——几个运算符可以连在一起使用，比如：`0<age<100`。

*注：比较对象的时候可以使用第二章中介绍的內建的`cmp`函数。*

有些运算符值得特别关注，下面的章节中会对此进行介绍。

2\. 相等运算符

如果想要知道两个东西是否相等，应该使用相等运算符，即两个等号"=="：

```py
>>> "foo" == "foo" True 
>>> "foo" == "bar" False 
# 相等运算符需要使用两个等号，如果使用一个等号会出现下面的情况
>>> "foo" = "foo" 
    File "<stdin>", line 1 
    SyntaxError: can't assign to literal 
```

单个相等运算符是赋值运算符，是用来*改变*值的，而不能用来比较。

3\. `is`：同一性运算符

这个运算符比较有趣。它看起来和`==`一样，事实上却不同：

```py
>>> x = y = [1, 2, 3] 
>>> z = [1, 2, 3] 
>>> x == y
True >>> x == z
True >>> x is y
True >>> x is z
False 
```

到最后一个例子之前，一切看起来都很好，但是最后一个结果很奇怪，`x`和`z`相等却不等同，为什么呢？因为`is`运算符是判定*同一性*而不是*相等性*的。变量`x`和`y`都被绑定到同一列表上，而变量`z`被绑定在另外一个具有相同数值和顺序的列表上。它们的值可能相等，但是却不是同一个*对象*。

这看起来有些不可理喻吧？看看这个例子：

```py
>>> x = [1, 2, 3] 
>>> y = [2, 4] 
>>> x is not y
True 
>>> del x[2] 
>>> y[1] = 1
>>> y.reverse() 
>>> y
[1, 2] 
>>> x
[1, 2] # 本例中，首先包括两个不同的列表 x 和 y。可以看到 x is not y 与(x is y 相反)，这个已经知道了。之后我改动了一下列表，尽管它们的值相等了，但是还是两个不同的列表。
>>> x == y
True 
>>> x is y
False # 显然，两个列表值等但是不等同。 
```

总结一下：使用`==`运算符来判定两个对象是否*相等*。使用`is`判定两者是否*等同*(同一个对象)。

*注：避免将`is`运算符用于比较类似数值和字符串这类不可变值。由于 Python 内部操作这些对象的方式的原因，使用`is`运算符的结果是不可预测的。*

4\. `in`：成员资格运算符

`in`运算符已经介绍过了(在 2.2.5 节)。它可以像其他比较运算符一样在条件语句中使用。

```py
name = raw_input("What is your name? ") 
if "s" in name: 
    print "Your name contains the letter 's'."
else: 
    print "Your name does not contains the letter 's'." 
```

5\. 字符串和序列比较

字符串可以按照字母顺序排列进行比较。

```py
>>> "alpha" < "beta" True 
```

*注：实际的顺序可能会因为使用不同的本地化设置(`locale`)而和上边的例子有所不同(请参见标准库文档中`locale`模块一节)。*

如果字符串內包括大写字母，那么结果就会有点乱(实际上，字符是按照本身的顺序值排列的。一个字母的顺序值可以用`ord`函数查到，`ord`函数与`chr`函数功能相反)。如果要忽略大小写字母的区别，可以使用字符串方法`upper`和`lower`(请参见第三章)。

```py
>>> "FnOrD".lower() == "Fnord".lower()
True # 其他的序列也可以用同样的方式进行比较，不过比较的不是字符而是其他类型的元素。
>>> [1, 2] < [2 ,1]
True # 如果一个序列中包括其他序列元素，比较规则也同样适用于序列元素。
>>> [2, [1, 4]] < [2, [1, 5]]
True 
```

6\. 布尔运算符

返回布尔值的对象已经介绍过许多(事实上，所有值都可以解释为布尔值，所有的表达式也都返回布尔值)。但有时想要检查一个以上的条件。例如，如果需要编写读取数字并且判断该数字是否位于 1~10 之间(也包括 10)的程序，可以像下面这样做：

```py
number = input("Enter a number between 1 and 10: ") 
if number <= 10: 
    if number >= 1: 
        print "Great!"
    else: 
        print "Wrong!"
else: 
    print "Wrong!"

# 这样做没问题，但是方法太笨了。笨在需要写两次 print "Wrong!"。在复制上浪费精力可不是好事。那么怎么办？很简单：
number = input("Enter a number between 1 and 10: ") 
if number <= 10 and number >= 1: 
    print "Great!"
else: 
    print "Wrong!" 
```

*注：本例中，还有(或者说应该使用)更简单的方法，即使用连接比较：`1<=number<=10`。*

`and`运算符就是所谓的布尔运算符。它连接两个布尔值，并且在两者都为真时返回真，否则返回假。与它同类的还有两个运算符，`or`和`not`。使用这三个运算符就可以随意结合真值。

```py
if ((cash > price) or customer_has_good_credit) and not out_of_stock:
    give_goods() 
```

**短路逻辑和条件表达式**

布尔运算符有个有趣的特性：只有在需要求值时才进行求值。举例来说，表达式 `x and y` 需要两个变量都为真时才为真，所以如果 x 为假，表达式就会立刻返回`False`，而不管`y`的值。实际上，如果`x`为假，表达式会返回`x`的值——否则它就返回`y`的值。(能明白它是怎么达到预期效果的吗？)这种行为被称为*短路逻辑*(short-circuit logic)或*惰性求值*(lazy evaluation)：布尔运算符通常被称为逻辑运算符，就像你看到的那样第二个值有时“被短路了”。这种行为对于`or`来说也同样适用。在`x or y`中，`x`为真时，它直接返回`x`值，否则返回`y`值。(应该明白什么意思吧？)注意，这意味着在布尔运算符之后的所有代码都不会执行。

这有什么用呢？它主要是避免了无用地执行代码，可以作为一种技巧使用，假设用户应该输入他/她的名字，但也可以选择什么都不输入，这时可以使用默认值`"<unknown>"`。可以使用`if`语句，但是可以很简洁的方式：

```py
name = raw_input("Please enter your name: ") or "<unknown>" 
```

换句话说，如果`raw_input(...)`语句的返回值为真(不是空字符串)，那么它的值就会赋值给`name`，否则将默认的`"<unknown>"`赋值给`name``。

这类短路逻辑可以用来实现 C 和 Java 中所谓的三元运算符(或条件运算符)。在 Python2.5 中有一个内置的条件表达式，像下面这样：

```py
a if b else c 
```

如果 b 为真，返回 a，否则，返回 c。(注意，这个运算符不用引入临时变量，就可以直接使用，从而得到与`raw_input(...)`例子中同样的结果)

### 5.4.7 断言

`if`语句有个非常有用的“近亲”，它的工作方式多少有点像下面这样(伪代码)：

```py
if not condition:
    crash program 
```

究竟为什么会需要这样的代码呢？就是因为与其让程序在晚些时候崩溃，不如在错误条件出现时直接让它崩溃。一般来说，你可以要求某些条件必须为真(例如，在检查函数参数的属性时，或者作为初期测试和调试过程中的辅助条件)。语句中使用的关键字是`assert`。

```py
>>> age = 10
>>> assert 0 < age < 100
>>> age = -1
>>> assert 0 < age < 100 
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module> AssertionError 
```

如果需要确保程序中的某一个条件一定为真才能让程序正常工作的话，assert 语句就有用了，他可以在程序中置入检查点。

条件后可以添加字符串，用来解释断言：

```py
>>> age = -1
>>> assert 0 < age < 100, "The age must be realistic" 
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module> 
    AssertionError: The age must be realistic 
```

## 5.5 循环

现在你已经知道当条件为真(或假)时如何执行了，但是怎么才能重复执行多次呢？例如，需要实现一个每月提醒你付房租的程序，但是就我们目前学习到的知识而言，需要向下面这样编写程序(伪代码)：

```py
发邮件
等一个月
发邮件
等一个月
发邮件
等一个月
(继续下去······) 
```

但是如果想让程序继续执行直到认为停止它呢？比如想像下面这样做(还是伪代码)：

```py
当我们没有停止时：
    发邮件
    等一个月 
```

或者换个简单些的例子。假设想要打印 1~100 的所有数字，就得再次用这个笨方法：

```py
print 1
print 2
print 3 
······ 
print 99
print 100 
```

但是如果准备用这种笨方法也就不会学 Python 了，对吧？

### 5.5.1 `while`循环

为了避免上例中笨重的代码，可以像下面这样做：

```py
x = 1
while x <= 100
    print x
    x += 1 
```

那么 Python 里面应该如何写呢？你猜对了，就像上面那样。不是很复杂吧？一个循环就可以确保用户输入了名字：

```py
name = ""
while not name:
    name = raw_input("Please enter your name: ") 
    print "Hello, %s!" % name 
```

运行这个程序看看，然后在程序要求输入名字时按下回车键。程序会再次要求输入名字，因为`name`还是空字符串，其求值结果为`False`。

*注：如果直接输入一个空格作为名字又会如何？试试看。程序会接受这个名字，因为包括一个空格的字符串并不是空的，所以不会判定为假。小程序因此出现了瑕疵，修改起来也很简单：只需要把`while not name`改为`while not name or name.isspace()`即可，或者可以使用`while not name.strip()`。*

### 5.2.2 `for`循环

`while`语句非常灵活。它可以用来在*任何条件*为真的情况下重复执行一个代码块。一般情况下这样用就够了，但是有些时候还得量体裁衣。比如要为一个集合(序列和其他可迭代对象)的每个元素都执行一个代码块。

*注：可迭代对象是指可以按次序迭代的对象(也就是用于`for`循环中的)。有关可迭代和迭代器的更多信息，请参见第九章，现在读者可以将其看做序列。*

这个时候可以使用`for`语句：

```py
words = ["this", "is", "an", "ex", "parrot"] 
for word in words: 
    print word 
# 或者
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
for number in numbers: 
    print number 
# 因为迭代(循环的另一种说法)某范围的数字是很常见的，所以有个內建的范围函数提供使用:
>>> range(0, 10)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
# Range 函数的工作方式类似于分片。它包含下限(本例中为 0)，但不包含上限(本例中为 10)。如果希望下限为 0，可以只提供上限:
>>> range(10)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
# 下面的程序会打印 1~100 的数字:
for number in range(1, 100): 
    print number 
# 它比之前的 while 循环更简洁。 
```

*注：如果能使用`for`循环，就尽量不用`while`循环。*

`xrange`函数的循环行为类似于`range`函数，区别在于`range`函数一次创建整个序列，而`xrange`一次只创建一个数(在 Python3.0 中，`range`会被转换成`xrange`风格的函数)。当需要迭代一个巨大的序列时`xrange`会更高效，不过一般情况下不需要过多关注它。

### 5.5.3 循环遍历字典元素

一个简单的`for`语句就能遍历字典的所有键，就像遍历访问序列一样：

```py
d = {"x": 1, "y": 2, "z": 3} 
for key in d: 
    print key, "corresponds to", d[key] 
```

在 Python2.2 之前，还只能用`keys`等字典方法来获取键(因为不允许直接迭代字典)。如果只需要值，可以使用`d.values`替代`d.keys`。`d.items`方法会将键-值对作为元组返回，`for`循环的一大好处就是可以循环中使用序列解包：

```py
for key, value in d.items(): 
    print key, "corrsponds", value 
```

*注：字典元素的顺序通常是没有定义的。换句话说，迭代的时候，字典中的键和值都能保证被处理，但是处理顺序不确定。如果顺序很重要的话，可以将键值保存在单独的列表中，例如在迭代前进行排序。*

### 5.5.4 一些迭代工具

在 Python 中迭代序列(或者其他可迭代对象)时，有一些函数非常好用。有些函数位于`itertools`模块中(第十章中介绍)，还有一些 Python 的內建函数也十分方便。

1\. 并行迭代

程序可以同时迭代两个系列。比如有下面两个列表：

```py
names = ["XuHoo", "Marlowes", "GuoYing", "LeiLa"]
ages = [19, 19, 22, 22] 
# 如果想要打印名字和对应的年龄，可以像下面这样做:
for i in range(len(names)): 
    print names[i], "is", ages[i], "years old" 
```

这里 `i` 是循环索引的标准变量名(可以自己随便定义，一般情况下`for`循环都以 `i` 作为变量名)。

而內建的`zip`函数就可以用来进行并行迭代，可以把两个序列“压缩”在一起，然后返回一个元组的列表：

```py
>>> zip(names, ages)
[("XuHoo", 19), ("Marlowes", 19), ("GuoYing", 22), ("LeiLa", 22)] 
# 现在我可以在循环中解包元组:
for name, age in zip(names, ages): 
    print name, "is", age, "years old"

# zip 函数也可以作用于任意多的序列。关于它很重要的一点是 zip 可以处理不等长的序列，当最短的序列"用完"的时候就会停止:
>>> zip(range(5), xrange(100000000))
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)] 
```

在上面的代码中，不推荐用`range`替换`xrange`——尽管只需要前五个数字，但`range`会计算所有的数字，这要花费很长的时间。而是用`xrange`就没这个问题了，它只计算前五个数字。

2.按索引迭代

有些时候想要迭代访问序列中的对象，同时还要获取当前对象的索引。例如，在一个字符串列表中替换所有包含`"xxx"`的子字符。实现的方法肯定有很多，假设你想像下面这样做：

```py
for string in strings: if "xxx" in string:
        index = strings.index(string)  
        # Search for the string in the list of strings
        strings[index] = "[censored]"

# 没问题，但是在替换前要搜索给定的字符串似乎没必要。如果不替换的话，搜索还会返回错误的索引(前面出现的同一个词的索引)。一个比较好的版本如下:
index = 0 for string in strings: 
    if "xxx" in string:
        strings[index] = "[censored]" 
        index += 1 
```

方法有些笨，不过可以接受。另一种方法是使用內建的`enumerate`函数：

```py
for index, string in enumerate(strings): 
    if "xxx" in string:
        strings[index] = "[censored]" 
```

这个函数可以在提供索引的地方迭代索引-值对。

3\. 翻转和排序迭代

让我们看看另外两个有用的函数：`reversed`和`sorted`。它们同列表的`reverse`和`sort`(`sorted`和`sort`使用同样的参数)方法类似，但作用于任何序列或可迭代对象上，不是原地修改对象，而是返回翻转或排序后的版本：

```py
>>> sorted([4, 3, 6, 8, 3])
[3, 3, 4, 6, 8] 
>>> sorted("Hello, world!")
[' ', '!', ',', 'H', 'd', 'e', 'l', 'l', 'l', 'o', 'o', 'r', 'w'] 
>>> list(reversed("Hello, world!"))
['!', 'd', 'l', 'r', 'o', 'w', ' ', ',', 'o', 'l', 'l', 'e', 'H'] 
>>> "".join(reversed("Hello, world!")) 
'!dlrow ,olleH' 
```

注意，虽然`sorted`方法返回列表，`reversed`方法却返回一个更加不可思议的可迭代对象。它们具体的含义不用过多关注，大可在`for`循环以及`join`方法中使用，而不会有任何问题。不过却不能直接对它使用索引、分片以及调用`list`方法，如果希望进行上述处理，那么可以使用`list`类型转换返回对象，上面的例子中已经给出具体的做法。

### 5.5.5 跳出循环

一般来说，循环会一直执行到条件为假，或者到序列元素用完时。但是有些时候可能会提前中断一个循环，进行新的迭代(新一"轮"的代码执行)，或者仅仅就是像结束循环。

1\. `break`

结束(跳出)循环可以使用`break`语句。假设需要寻找 100 以内的最大平方数，那么程序可以开始从 100 往下迭代到 0.当找到一个平方数时就不需要继续循环了，所以可以跳出循环：

```py
from math import sqrt 
    for n in range(99, 0, -1):
        root = sqrt(n) 
        if root == int(root): 
            print n break 
```

如果执行这个程序的话，会打印出`81`，然后程序停止。注意，上面的代码中`range`函数增加了第三个参数——表示*步长*，步长表示每对相邻数字之间的差别。将其设置为负值的话就会想例子中一样反向迭代。它也可以用来跳过数字：

```py
>>> range(0, 10, 2)
[0, 2, 4, 6, 8] 
```

2\. `continue`

`continue`语句比`break`语句用得要少得多。它会让当前的迭代结束，“跳”到下一轮循环的开始。它最基本的意思是“跳过剩余的循环体，但是不结束循环”。当循环体很大而且很复杂的时候，这会很有用，有些时候因为一些原因可能会跳过它——这个时候可以使用`continue`语句：

```py
for x in seq: 
    if condition1:
    continue
    if condition2:
    continue
    if condition3:
    continue 
    do_something()
    do_something_else()
    do_another_thing()
    etc() 
# 很多时候，只要使用 if 语句就可以了:
for x in seq: 
    if not (condition1 or condition2 or condition3):
        do_something()
        do_something_else()
        do_another_thing()
        etc() 
```

尽管`continue`语句非常有用，它却不是最本质的。应该习惯使用`break`语句，因为在`while True`语句中会经常用到它。下一节会对此进行介绍。

3\. `while True/break`习语

Python 中的`while`和`for`循环非常灵活，但一旦使用`while`语句就会遇到一个需要更多功能的问题。如果需要当用户在提示符下输入单词时做一些事情，并且在用户不输入单词后结束循环。可以使用下面的方法：

```py
word = "dummy"
while word:
    # 处理 word
    word = raw_input("Please enter a word: ") 
    print "The word was " + word 
# 下面是一个会话示例:
Please enter a word: first
The word was first
Please enter a word: second
The word was second
Please enter a word: 
```

代码按要求的方式工作(大概还能做些比直接打印出单词更有用的工作)。但是代码有些丑。在进入循环之前需要给 word 赋一个哑值(未使用的)。使用哑值(dummy value)就是工作没有尽善尽美的标志。让我们试着避免它：

```py
word = raw_input("Please enter a word: ") 
# 处理 word
while word: 
    print "The word was " + word
    word = raw_input("Please enter a word: ") 
    # 哑值没有了。但是有重复的代码(这样也不好):要用一样的赋值语句在两个地方两次调用 raw_input。能否不这么做呢？可以使用 while True/break 语句:
while True:
    word = raw_input("Please enter a word: ") 
    if not word:
    break
    # 处理 word
    print "The word was " + word 
```

`while True`的部分实现了一个永远不会自己停止的循环。但是在循环内部的 if 语句中加入条件也是可以的，在条件满足时使用`break`语句。这样一来就可以在循环内部任何地方而不是只在开头(像普通的`while`循环一样)终止循环。`if/break`语句自然地将循环分为两部分：第一部分负责初始化(在普通的`while`循环中，这部分需要重复)，第二部分则在循环条件为真的情况下使用第一部分內初始化好的数据。

尽管应该避免在代码中频繁使用`break`语句(因为这可能会让循环的可读性降低，尤其是在一个循环中使用多个`break`语句的时候)，但这个特殊的技术用得非常普遍，大多数 Python 程序员(包括你自己)都能理解你的意思。

### 5.5.6 循环中的`else`子句

当在循环内使用`break`语句时，通常是因为“找到”了某物或者因为某事“发生”了。在跳出是做一些事情是很简单的(比如`print n`)，但是有些时候想要在没有跳出之前做些事情。那么怎么判断呢？可以使用布尔变量，再循环前将其设定为`False`，跳出后设定为`True`。然后再使用`if`语句查看循环是否跳出了：

```py
broke_out = False for x in seq:
    do_something(x) 
    if condition(x):
        broke_out = True 
        break 
    do_something_else(x) 
    if not broke_out: 
        print "I didn't break out!"

# 更简单的方式是在循环中增加一个 else 子句——它仅在没有调用`break`时执行。让我们用这个方法重写刚才的例子:
from math import sqrt 
for n in range(99, 81, -1):
    root = sqrt(n) 
    if root == int(root): 
        print n break
else: 
    print "Didn't find it!" 
```

注意我将下限改为 81(不包括 81)以测试`else`子句。如果执行程序的话，它会打印出`"Didn't find it!"`，因为(就像在`break`那节看到的一样)100 以内最大的平方数时 81。`for`和`while`循环中都可以使用`continue`、`break`语句和`else`子句。

## 5.6 列表推导式——轻量级循环

*列表推导式*(list comprehension)是利用其他列表创建新列表(类似于数学术语中的集合推导式)的一种方法。它的工作方式类似于`for`循环，也很简单：

```py
>>> [x*x for x in range(10)]
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81] 
```

列表有`range(10)`中每个 x 的平方组成。太容易了？如果只想打印出那些能被 3 整除的平方数呢？那么可以使用模除运算符——`y%3`，当数字可以被 3 整除时返回 0(注意，`x`能被 3 整除时，`x`的平方必然也可以被 3 整除)。这个语句可以通过增加一个`if`部分添加到列表推导式中：

```py
>>> [x*x for x in range(10) if x % 3 == 0]
[0, 9, 36, 81] 
# 也可以增加更多 for 语句的部分:
>>> [(x, y) for x in range(3) for y in range(3)]
[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)] 
# 作为对比，下面的代码使用两个 for 语句创建了相同的列表:
result = [] 
for x in range(3): 
    for y in range(3):
        result.append((x, y)) 
print result 
# 也可以和 if 子句联合使用，像以前一样:
>>> girls = ["alice", "bernice", "clarice"] >>> boys = ["chris", "arnold", "bob"] 
>>> [b+"+"+g for b in boys for g in girls if b[0] == g[0]]
['chris+clarice', 'arnold+alice', 'bob+bernice'] # 这样就得到了那些名字首字母相同的男孩和女孩。 
```

*注：使用普通的圆括号而不是方括号不会得到“元组推导式”。在 Python2.3 及以前的版本中只会得到错误。在最近的版本中，则会得到一个生成器。请参见 9.7 节获得更多信息。*

**更优秀的方案**

男孩/女孩名字对的例子其实效率不高，因为它会检查每个可能的配对。Python 有很多解决这个问题的方法，下面的方法是 Alex Martelli 推荐的：

```py
girls = ["alice", "bernice", "clarice"]
boys = ["chris", "arnold", "bob"]
letterGirls = {} 
for girl in girls:
    letterGirls.setdefault(girl[0], []).append(girl) 
print [b+"+"+g for b in boys for g in letterGirls[b[0]]] 
```

这个程序创建了一个叫做`letterGirls`的字典，其中每一项都把单字母作为键，以女孩名字组成的列表作为值。(`setdefault`字典方法在前一章中已经介绍过)在字典建立后，列表推导式循环整个男孩集合，并且查找那些和当前男孩名字首字母相同的女孩集合。这样列表推导式就不用尝试所有的男孩女孩的组合，检查首字母是否匹配。

## 5.7 三人行

作为本章的结束，让我们走马观花地看一下另外三个语句：`pass`、`del`和`exec`。

### 5.7.1 什么都没发生

有的时候，程序什么事情都不用做吗。这种情况不多，但是一旦出现，就应该让`pass`语句出马了。

```py
>>> pass
>>> 
```

似乎没什么动静。

那么究竟为什么使用一个什么都不做的语句？它可以在代码中做占位符使用。比如程序需要一个`if`语句，然后进行测试，但是缺少其中一个语句块的代码，考虑下面的情况：

```py
if name == "Ralph Auldus Melish": 
    print "Welcome!"
elif name == "End": 
    # 还没完······
elif name == "Bill Gates": 
    print "Access Denied"

# 代码不会执行，因为 Python 中空代码块是非法的。解决方案就是在语句块中加上一个 pass 语句:
if name == "Ralph Auldus Melish": 
    print "Welcome!"
elif name == "End": 
    # 还没完······
    pass
elif name == "Bill Gates": 
    print "Access Denied" 
```

*注：注释和`pass`语句联合的代替方案是插入字符串。对于那些没有完成的函数(参见第六章)和类(参见第七章)来说这个方法尤其有用，因为它们会扮演文档字符串(docstring)的角色(第六章中会有解释)。*

### 5.7.2 使用 del 删除

一般来说，Python 会删除那些不再使用的对象(因为使用者不会再通过任何变量或数据结构引用它们)：

```py
>>> scoundrel = {"age": 42, "first name": "Robin", "last name": "of Locksley"} >>> robin = scoundrel >>> scoundrel
{'last name': 'of Locksley', 'first name': 'Robin', 'age': 42} 
>>> robin
{'last name': 'of Locksley', 'first name': 'Robin', 'age': 42} 
>>> scoundrel = None >>> robin = None 
```

首先，`robin`和`scoundrel`都被绑定到同一个字典上。所以当设置`scoundrel`为`None`的时候，字典通过`robin`还是可用的。但是当我把`robin`也设置为`None`的时候，字典就“漂”在内存里了，没有任何名字绑定到它上面。没有办法获取和使用它，所以 Python 解释器(以其无穷的智慧)直接删除了那个字典(这种行为被称为*垃圾收集*)。注意，也可以使用`None`之外的其他值。字典同样会“消失不见”。

另外一个方法就是使用`del`语句(我们在第二章和第四章里面用来删除序列和字典元素的语句)，它不仅会移除一个对象的引用，也会移除那个名字本身。

```py
>>> x = 1
>>> del x 
>>> x
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  NameError: name 'x' is not defined 
# 看起来很简单，但有时理解起来有些难度。例如，下面的例子中，x 和 y 都指向同一个列表:
>>> x = ["Hello", "world"] 
>>> y = x 
>>> y[1] = "Python"
>>> x
['Hello', 'Python'] 
# 会有人认为删除 x 后，y 也就随之消失了，但并非如此:
>>> del x 
>>> y
['Hello', 'Python'] 
```

为什么会这样？`x`和`y`都指向同一个列表。但是删除`x`并不会影响`y`。原因就是删除的只是*名称*，而不是列表本身(值)。事实上，在 Python 中是没有办法删除值的(也不需要过多考虑删除值的问题，因为在某个值不再使用的时候，Python 解释器会负责内存的回收)。

### 5.7.3 使用`exec`和`eval`执行和求值字符串

有些时候可能会需要动态地创造 Python 代码，然后将其作为语句执行或作为表达式计算，这可能近似于“黑暗魔法”——在此之前，一定要慎之又慎，仔细考虑。

*警告：本节中，会学到如何执行存储在字符串中的 Python 代码。这样做会有很严重的潜在安全漏洞。如果程序将用户提供的一段内容中的一部分字符串作为代码执行，程序可能会失去对代码执行的控制，这种情况在网络应用程序——比如 CGI 脚本中尤其危险，这部分内容会在第十五章介绍。*

1\. `exec`

执行一个字符串的语句是`exec`(在 Python3.0 中，`exec`是一个函数而不是语句)：

```py
>>> exec "print 'Hello, world!'" 
Hello, world! 
```

但是，使用简单形式的`exec`语句绝不是好事。很多情况下可以给它提供命名空间——可以放置变量的地方。你想这样做，从而使代码不会干扰命名空间(也就是改变你的变量)，比如，下面的代码中使用了名称`sqrt`：

```py
>>> from math import sqrt 
>>> exec "sqrt = 1"
>>> sqrt(4)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  TypeError: 'int' object is not callable 
```

想想看，为什么一开始我们要这样做？`exec`语句最有用的地方在于可以动态地创建代码字符串。如果字符串是从其他地方获得——很有可能是用户——那么几乎不能确定其中到底包含什么代码。所以为了安全起见，可以增加一个字典，起到命名空间的作用。

*注：命名空间的概念，或称为作用域(`scope`)，是非常重要的知识，下一章会深入学习，但是现在可以把它想象成保存变量的地方，类似于不可见的字典。所以在程序执行`x=1`这类赋值语句时，就将键`x`和值`1`放在当前的命名空间內，这个命名空间一般来说都是全局命名空间(到目前为止绝大多数都是如此)，但这并不是必须的。*

可以通过增加 `in<scope>` 来实现，其中`<scope>`就是起到放置代码字符串命名空间作用的字典。

```py
>>> from math import sqrt 
>>> scope = {} 
>>> exec "sqrt = 1" in scope 
>>> sqrt(4) 2.0
>>> scope["sqrt"] 
1 
```

可以看到，潜在的破坏性代码并不会覆盖`sqrt`函数，原来的函数能正常工作，而通过`exec`赋值的变量`sqrt`只在它的作用域内有效。

注意，如果需要将`scope`打印出来的话，会看到其中包含很多东西，因为內建的`__builtins__`字典自动包含所有的內建函数和值：

```py
>>> len(scope) 2
>>> scope.keys()
['__builtins__', 'sqrt'] 
```

2\. `eval`

`eval`(用于“求值”)是类似于`exec`的內建函数。`exec`语句会执行一系列*Python 语句*，而`eval`会计算*Python 表达式*(以字符串形式书写)，并且返回结果值。(`exec`语句并不返回任何对象，因为它本身就是语句)例如，可以使用下面的代码创建一个 Python 计算器：

```py
>>> eval(raw_input("Enter an arithmetic expression: "))
Enter an arithmetic expression: 6 + 18 * 2
42 
```

*注：表达式`eval(raw_input(...))`事实上等同于`input(...)`。在 Python3.0 中，`raw_input`被重命名为`input`。*

跟`exec`一样，`eval`也可以使用命名空间。尽管表达式几乎不像语句那样为变量重新赋值。(实际上，可以给`eval`语句提供两个命名空间，一个全局的一个局部的。全局的必须是字典，局部的可以是任何形式的映射。)

警告：尽管表达式一般不给变量重新赋值，但它们的确可以(比如可以调用函数给全局变量重新赋值)。所以使用`eval`语句对付一些不可信任的代码并不比`exec`语句安全。目前，在 Python 内没有任何执行不可信任代码的安全方式。一个可选的方案是使用 Python 的实现，比如 Jython(参见第十七章)，以及使用一些本地机制，比如 Java 的 sandbox 功能。

**初探作用域**

给`exec`或者`eval`语句提供命名空间时，还可以在真正使用命名空间前放置一些值进去：

```py
>>> scope = {} 
>>> scope["x"] = 2
>>> scope["y"] = 3
>>> eval("x * y", scope) 
6

# 同理，exec 或者 eval 调用的作用域也能在另一个上面使用:
>>> scope = {} 
>>> exec "x = 2" in scope 
>>> eval("x * x", scope) 
4 
```

事实上，`exec`语句和`eval`语句并不常用，但是它们可以作为后兜里的得力工具(当然，这仅仅是比喻而已)。

## 5.8 小结

本章中介绍了几类语句和其他知识。

√ 打印。`print`语句可以用来打印由逗号隔开的多个值。如果语句以逗号结尾，后面的`print`语句会在同一行内继续打印。

√ 导入。有些时候，你不喜欢你想导入的函数名——还有可能由于其他原因使用了这个函数名。可以使用`from ... as ...` 语句进行函数的局部重命名。

√ 赋值。通过序列解包和链式赋值功能，多个变量赋值可以一次性赋值，通过增量赋值可以原地改变变量。

√ 块。块是通过缩排使语句成组的一种方法。它们可以在条件以及循环语句中使用，后面的章节中会介绍，块也可以在函数和类中使用。

√ 条件。条件语句可以根据条件(布尔表达式)执行或不执行一个语句块。几个条件可以串联使用`if/elif/else`。这个主题下还有一种变体叫做条件表达式，形如 `a if b else c`(这种表达式其实类似于三元运算)。

√ 断言。断言简单来说就是肯定某事(布尔表达式)为真。也可在后面跟上这么认为的原因。如果表达式为真，断言就会让程序崩溃(事实上是产生异常——第八章会介绍)。比起错误潜藏在程序中，直到你不知道它源在何处，更好的方法是尽早找到错误。

√ 循环。可以为序列(比如一个范围内的数字)中的每一个元素执行一个语句块，或者在条件为真的时候继续执行一段语句。可以使用`continue`语句跳过块中的其他语句，然后继续下一次迭代，或者使用`break`语句跳出循环。还可以选择在循环结尾加上`else`子句，当没有执行循环内部的`break`语句的时候便会执行`else`子句中的内容。

√ 列表推导式。它不是真正的语句，而是看起来像循环的表达式，这也是我将它归到循环语句中的原因。通过列表推导式，可以从旧列表中产生新的列表、对元素应用函数、过滤不需要的元素，等等。这个功能很强大，但是很多情况下，直接使用循环和条件语句(工作也能完成)，程序会更易读。

√ `pass`、`del`、`exec`和`eval`语句。`pass`语句什么都不做，可以作为占位符使用。`del`语句用来删除变量，或者数据结构的一部分，但是不能用来删除值。`exec`语句用于执行 Python 程序相同的方式来执行字符串。內建的`eval`函数对写在字符串中的表达式进行计算并且返回结果。

### 5.8.1 本章的新函数

本章涉及的新函数如表 5-2 所示。

表 5-2 本章的新函数

```py
chr(n)                                 当传入序号 n 时，返回 n 所代表的包含一个字符的字符串(0≤n&lt;256)。
eval(source[, globals[, locals]])      将字符串作为表达式计算，并且返回值。
enumerate(seq)                         产生用于迭代的(索引，值)对。
ord(c)                                 返回单字符字符串的 int 值。
range([start,] stop[, step])           创建整数的列表。
reversed(seq)                          产生 seq 中值的反向版本，用于迭代。
sorted(seq[, cmp][, key][, reverse])   返回 seq 中值排序后的列表。
xrange([start,] stop[, step])          创造 xrange 对象用于迭代。
zip(seq1, seq2 ...)                    创造用于并行迭代的新序列。 
```

### 5.8.2 接下来学什么

现在基本知识已经学完了。实现任何自己能想到的算法已经没问题了，也可以让程序读取参数并且打印结果。下面两章中，将会介绍可以创建较大程序，却不让代码冗长的知识。这也就是我们所说的*抽象*(abstraction)。