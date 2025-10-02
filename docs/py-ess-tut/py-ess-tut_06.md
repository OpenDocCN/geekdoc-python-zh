# 第六章 抽象

> 来源：[`www.cnblogs.com/Marlowes/p/5351415.html`](http://www.cnblogs.com/Marlowes/p/5351415.html)
> 
> 作者：Marlowes

本章将会介绍如何将语句组织成函数，这样，你可以告诉计算机如何做事，并且只需要告诉一次。有了函数以后，就不必反反复复像计算机传递同样的具体指令了。本章还会详细介绍*参数*(parameter)和*作用域*(scope)的概念，以及递归的概念及其在程序中的用途。

## 6.1 懒惰即美德

目前为止我们缩写的程序都很小，如果想要编写大型程序，很快就会遇到麻烦。考虑一下如果在一个地方编写了一段代码，但在另一个地方也要用到这段代码，这时会发生什么。例如，假设我们编写了一小段代码来计算斐波那契数列(任一个数都是前两数之和的数字序列)：

```py
fibs = [0, 1] for i in range(8):
    fibs.append(fibs[-2] + fibs[-1]) 
    # 运行之后,fibs 会包含斐波那契数列的前 10 个数字:
fibs
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34] 
# 如果想要以此计算前 10 个数的话，没有问题。你甚至可以将用户输入的数字作为动态范围的长度使用，从而改变 for 语句循环的次数:
fibs = [0, 1]
num = input("How many Fibonacci numbers do you want? ") 
for i in range(num - 2):
    fibs.append(fibs[-2] + fibs[-1]) 
    print fibs 
```

*注：在本例中，读取字符串可以使用`raw_input`函数，然后再用`int`函数将其转换为整数。*

但是如果想用这些数字做其他事情呢？当然可以在需要的时候重写同样的循环，但是如果已经编写的是一段复杂的代码——比如下载一系列网页并且计算词频——应该怎么做呢？你是否希望在每次需要的时候把所有的代码重写一遍呢？当然不用，真正的程序员不会这么做的，他们都很懒，但不是用错误的方式犯懒，换句话说就是他们不做无用功。

那么真正的程序员怎么做呢？他们会让自己的程序*抽象*一些。上面的程序可以改写为比较抽象的版本：

```py
num = input("How many numbers do you want? ") 
print fibs(num) 
```

这个程序的具体细节已经写的很清楚了(读入数值，然后打印结果)。事实上计算菲波那切数列是由一种更抽象的方式完成的：只需要告诉计算机去做就好，不用特别说明应该怎么做。名为 fibs 的函数被创建，然后在需要计算菲波那切数列的地方调用它即可。如果这函数要被调用很多次的话，这么做会节省很多精力。

## 6.2 抽象和结构

抽象可以节省很多工作，实际上它的作用还要更大，它是使得计算机程序可以让人读懂的关键(这也是最基本的要求，不管是读还是写程序)。计算机非常乐于处理精确和具体的指令，但是人可就不同了。如果有人问我去电影院怎么走，估计他不会希望我回答“向前走 10 步，左转 90 度，再走 5 步右转 45 度，走 123 步”。弄不好就迷路了，对吧？

现在，如果我告诉他“一直沿着街走，过桥，电影院就在左手边”，这样就明白多了吧！关键在于大家都知道怎么走路和过桥，不需要明确指令来指导这些事。

组织计算机程序也是类似的。程序应该是非常抽象的，就像“下载网页、计算频率、打印每个单词的频率”一样易懂。事实上，我们现在就能把这段描述翻译成 Python 程序：

```py
page = download_page()
freqs = compute_frequencies(page) 
for word, freq in freqs: 
    print word, freq 
```

虽然没有明确地说出它是怎么做的，单读完代码就知道程序做什么了。只需要告诉计算机下载网页并计算词频。这些操作的具体指令细节会在其他地方给出——在单独的*函数定义*中。

## 6.3 创建函数

函数是可以调用的(可能带有参数，也就是放在圆括号中的值)，它执行某种行为并且返回一个值(并非所有 Python 函数都有返回值)。一般来说，内建的`callable`函数可以用来判断函数是否可调用：

```py
>>> import math >>> x = 1
>>> y = math.sqrt 
>>> callable(x)
False 
>>> callable(y)
True 
```

*注：函数`callable`在 Python3.0 中不再可用，需要使用表达式`hasattr(func, __call__)`代替，有关`hasattr`的更多信息，请参见第七章。*

就像前一节内容中介绍的，创建函数是组织程序的关键。那么怎么定义函数呢？使用`def`(或“函数定义”)语句即可：

```py
def hello(name): 
    return "Hello, " + name + "!"

# 运行这段程序就会得到一个名为 hello 的新函数，它可以返回一个将输入的参数作为名字的问候语。可以像使用内建函数一样使用它:
>>> print hello("world")
Hello, world! 
>>> print hello("XuHoo")
Hello, XuHoo! 
```

很精巧吧？那么想想看怎么写个返回斐波那契数列列表的函数吧。简单！只需要使用刚才的代码，把从用户输入获取的数字改为作为参数接收数字：

```py
num = input("How many numbers do you want? ") 
def fibs(num):
    result = [0, 1] 
    for i in range(num - 2):
        result.append(result[-2] + resultp[-1]) 
    return result 
# 执行这段与语句后，编译器就知道如何计算斐波那契数列了——所以现在就不用关注细节了，只要用函数 fibs 就行:
>>> fibs(10)
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34] 
>>> fibs(15)
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377] 
```

本例中的`num`和`result`的名字都是随便起的，但是`return`语句非常重要。`return`语句是用来从函数中返回值的(函数可以返回一个以上的值，元组中返回即可)(前例中的`hello`函数也有用到)。

### 6.3.1 文档化函数

如果想要给函数写文档，让其他使用该函数的人能理解的话，可以加入注释(以`#`开头)。另外一个方式就是直接写上字符串。这类字符串在其他地方可能会非常有用，比如在`def`语句后面(以及在模块或者类的开头——有关类的更多内容请参见第七章，有关模块的更多内容请参见第十章)。如果在函数的开头写下字符串，它就会作为函数的一部分进行存储，这成为*文档字符串*。下面代码演示了如何给函数添加文档字符串：

```py
def square(x): 
    "Calculates the square of the number x."
    return x * x 
# 文档字符串可以按如下方式访问:
>>> square.__doc__
"Calculates the square of the number x." 
```

*注：`__doc__`是函数属性，第七章中会介绍更多关于属性的知识，属性名中的双下划线表示它是个特殊属性。这类特殊和“魔法”属性会在第九章讨论。*

内建的`help`函数是非常有用的。在交互式解释器中使用它，就可以得到关于函数，包括它的文档字符串的信息：

```py
>>> help(square)
Help on function square in module __main__;

square(x)
    Calculates the square of the number x. 
```

第十章中会再次对`help`函数进行讨论。

### 6.3.2 并非真正函数的函数

数学意义上的函数，总在计算其参数后返回点什么。Python 的有些函数却并不返回任何东西。在其他语言中(比如 Pascal)，这类函数可能有其他名字(比如*过程*)。但是 Python 的函数就是函数，即便它从学术上讲并不是函数。没有`return`语句，或者虽有`return`语句但`return`后边没有跟任何值的函数不返回值：

```py
def test(): 
    print "This is printed"
    return
    print "This is not"

# 这里的 return 语句只起到结束函数的作用:
>>> x = test()
This is printed 
# 可以看到，第 2 个 print 语句被跳过了(类似于循环中的 break 语句，不过这里是跳出函数)。但是如果 test 不返回任何值，那么 x 又引用什么呢？让我们看看:
>>> x 
>>>
# 没东西，再仔细看看:
>>> print x 
>>> None 
```

好熟悉的值：`None`。所以所有的函数的确都返回了东西：当不需要它们返回值的时候，它们就返回`None`。看来刚才“有些函数并不真的是函数”的说法有些不公平了。

*注：千万不要被默认行为所迷惑。如果在`if`语句内返回值，那么要确保其他分支也有返回值，这样一来当调用者期待一个序列的时候，就不会意外地返回`None`。*

## 6.4 参数魔法

函数使用起来很简单，创建起来也不复杂。但函数参数的用法有时就有些神奇了。还是先从最基础的介绍起。

### 6.4.1 值从哪里来

函数被定义后，所操作的值是从哪里来的呢？一般来说不用担心这些，编写函数只是给程序需要的部分(也可能是其他程序)提供服务，能保证函数在被提供给可接受参数的时候正常工作就行，参数错误的话显然会导致失败(一般来说这时候要用断言和异常，第八章会介绍异常)。

*注：写在`def`语句中函数名后面的变量通常叫做函数的形参，而调用函数的时候提供的值是实参，或者称为参数。一般来说，本书在介绍的时候对于两者的区别并不会吹毛求疵。如果这种区别影响较大的话，我会将实参称为“值”以区别与形参。*

### 6.4.2 我能改变参数吗

函数通过它的参数获得一系列值。那么这些值能改变吗？如果改变了又会怎么样？参数只是变量而已，所以它们的行为其实和你预想的一样。在函数内为参数赋予新值不会改变外部任何变量的值：

```py
>>> def try_to_change(n):
...     n = "Mr. XuHoo" 
... 
>>> name = "Mr. Marlowes"
>>> try_to_change(name) 
>>> name 
'Mr. Marlowes'

# 在 try_to_change 内，参数 n 获得了新值，但是它没有影响到 name 变量。n 实际上是个完全不同的变量，具体的工作方式类似于下面这样:
>>> name = "Mr. Marlowes"
>>> n = name  
# 这句的作用基本上等于传参数
>>> n = "Mr. XuHoo"  
# 在函数内部完成的
>>> name 
'Mr. Marlowes' 
```

结果是显而易见的。当变量`n`改变的时候，变量`name`不变。同样，当在函数内部把参数重绑(赋值)的时候，函数外的变量是不会受到影响的。

*注：参数存储在局部作用域(local scope)内，本章后面会介绍。*

字符串(以及数字和元组)是*不可变*的，即无法被修改(也就是说只能用新的值覆盖)。所以它们做参数的时候也就无需多做介绍。但是考虑一下如果将可变的数据结构如列表用作参数的时候会发生什么：

```py
>>> def change(n):
...     n[0] = "Mr. XuHoo" 
... 
>>> names = ["Mrs. Marlowes", "Mrs. Something"] 
>>> change(names) 
>>> names
['Mr. XuHoo', 'Mrs. Something'] 
```

本例中，参数被改变了。这就是本例和前面例子中至关重要的区别。前面的例子中，局部变量被赋予了新值，但是这个例子中变量`names`所绑定的列表的确变了。有些奇怪吧？其实这种行为并不奇怪，下面不用函数调用再做一次：

```py
>>> names = ["Mrs. Marlowes", "Mrs. Something"] 
>>> n = names  # 再来一次，模拟传参行为
>>> n[0] = "Mr. XuHoo"  # 改变列表
>>> names
['Mr. XuHoo', 'Mrs. Something'] 
```

这类情况在前面已经出现了多次。当两个变量同时引用一个列表的时候，它们的确是同时引用一个列表。就是这么简单。如果想避免出现这种情况，可以复制一个列表的*副本*。当在序列中做切片的时候，返回的切片总是一个副本。因此，如果你复制了*整个列表*的切片，将会得到一个副本：

```py
>>> names = ["Mrs. Marlowes", "Mrs. Something"] 
>>> n = names[:] 
# 现在 n 和 names 包含两个独立(不同)的列表，其值相等:
>>> n is names
False '
>>> n == names
True 
# 如果现在改变 n(就像在函数 change 中做的一样)，则不会影响到 names:
>>> n[0] = "Mr. XuHoo"
>>> n
['Mr. XuHoo', 'Mrs. Something'] 
>>> names
['Mrs. Marlowes', 'Mrs. Something'] 
# 再用 change 试一下:
>>> change(names[:]) >>> names
['Mrs. Marlowes', 'Mrs. Something'] 
```

现在参数`n`包含一个副本，而原始的列表是安全的。

*注：可能有的读者会发现这样的问题：函数的局部名称——包括参数在内——并不和外面的函数名称(全局的)冲突。关于作用域的更多信息，后面的章节会进行讨论。*

1\. 为什么要修改参数

使用函数改变数据结构(比如列表或字典)是一种将程序抽象化的好方法。假设需要编写一个存储名字并且能用名字、中间名或姓查找联系人的程序，可以使用下面的数据结构：

```py
storage = {}
storage["first"] = {}
storage["middle"] = {}
storage["last"] = {} 
```

`storage`这个数据结构是带有 3 个键`“first”`、`“middle”`、`“last”`的字典。每个键下面都又存储一个字典。子字典中，可以使用名字(名字、中间名或姓)作为键，插入联系人列表作为值。比如要把我自己的名字加入这个数据结构，可以像下面这么做：

```py
>>> me = "Magnus Lie Hetland"
>>> storage["first"]["Magnus"] = [me] 
>>> storage["middle"]["Lie"] = [me] 
>>> storage["last"]["Hetland"] = [me] 
# 每个键下面都存储了一个以人名组成的列表。本例中，列表中只有我。 
# 现在如果想要得到所有注册的中间名为 Lie 的人，可以像下面这么做:
>>> storage["middle"]["Lie"]
['Magnus Lie Hetland'] 
```

将人名加到列表中的步骤有点枯燥乏味，尤其是要加入很多姓名相同的人时，因为需要扩展已经存储了那些名字的列表。例如，下面加入我姐姐的名字，而且假设不知道数据库中已经存储了什么：

```py
>>> my_sister = "Anne Lie Hetland"
>>> storage["first"].setdefault("Anne", []).append(my_sister) 
>>> storage["middle"].setdefault("Lie", []).append(my_sister) 
>>> storage["last"].setdefault("Hetland", []).append(my_sister) 
>>> storage["first"]["Anne"]
['Anne Lie Hetland'] >>> storage["middle"]["Lie"]
['Magnus Lie Hetland', 'Anne Lie Hetland'] 
```

如果要写个大程序来这样更新列表，那么很显然程序很快就会变得臃肿且笨拙不堪了。

抽象的要点就是隐藏更新时繁琐的细节，这个过程可以用函数实现。下面的例子就是初始化数据结构的函数：

```py
def init(data):
    data["first"] = {}
    data["middle"] = {}
    data["last"] = {} 
# 上面的代码只是把初始化语句放到了函数中，使用方法如下：
>>> storage = {} 
>>> init(storage) 
>>> storage
{'middle': {}, 'last': {}, 'first': {}} 
```

可以看到，函数包办了初始化的工作，让代码更易读。

*注：字典的键并没有特定的顺序，所以当字典打印出来的时候，顺序是不同的。如果读者在自己的解释器中打印出的顺序不同，请不要担心，这是很正常的。*

在编写存储名字的函数前，先写个获得名字的函数：

```py
def lookup(data, label, name): 
    return data[label].get(name) 
```

标签(比如`"middle"`)以及名字(比如`"Lie"`)可以作为参数提供给`lookup`函数使用，这样会获得包含全名的列表。换句话说，如果我的名字已经存储了，可以像下面这样做：

```py
>>> lookup(storage, "middle", "Lie")
['Magnus Lie Hetland'] 
```

注意，返回的列表和存储在数据结构中的列表是相同的，所以如果列表被修改了，那么也会影响数据结构(没有查询到人的时候就问题不大了，因为函数返回的是`None`)。

```py
def store(data, full_name):
    names = full_name.split() 
    if len(name) == 2:
        names.insert(1, "")
    labels = "first", "middle", "last"
    for label, name in zip(labels, names):
        people = lookup(data, label, name) 
        if people:
            people.append(full_name) else:
            data[label][name] = [full_name] 
```

store 函数执行以下步骤。

(1) 使用参数`data`和`full_name`进入函数，这两个参数被设置为函数在外部获得的一些值。

(2) 通过拆分`full_name`，得到一个叫做`names`的列表。

(3) 如果`names`的长度为`2`(只有首名和末名)，那么插入一个空字符串作为中间名。

(4) 将字符串`"first"`、`"middle"`和`"last"`作为元组存储在`labels`中(也可以使用列表，这里只是为了方便而去掉括号)。

(5) 使用`zip`函数联合标签和名字，对于每一个`(label, name)`对，进行一下处理：

1) 获得属于给定标签和名字的列表；

2) 将`full_name`添加到列表中，或者插入一个需要的新列表。

来试用一下刚刚实现的程序：

```py
>>> MyNames = {} >>> init(MyNames) 
>>> store(MyNames, "Magnus Lie Hetland") 
>>> lookup(MyNames, "middle", "Lie") 
# 好像可以工作，再试试:
>>> store(MyNames, "Robin Hood") 
>>> store(MyNames, "Robin Locksley") 
>>> lookup(MyNames, "first", "Robin")
['Robin Hood', 'Robin Locks ley'] 
>>> store(MyNames, "Mr. XuHoo") 
>>> lookup(MyNames, "middle", "")
['Robin Hood', 'Robin Locksley', 'Mr. XuHoo'] 
```

可以看到，如果某些人的名字、中间名或姓相同，那么结果中会包含所有这些人的信息。

*注：这类程序很适合进行面向对象程序设计，下一章内会讨论到如何进行面向对象程序设计。*

2.如果我的参数不可变呢

在某些语言(比如 C++、Pascal 和 Ada)中，重新绑定参数并且使这些改变影响到函数外的变量是很平常的事情。但在 Python 中这是不可能的：函数只能修改参数对象本身。但是如果你的参数不可变(比如是数字)，又该怎么办呢？

不好意思，没有办法。这个时候你应该从函数中返回所有你需要的值(如果值多于一个的话就以元组形式返回)。例如，将变量的数值增 1 的函数可以这样写：

```py
>>> def inc(x):    
        return x + 1 
... 
>>> foo = 10
>>> foo = inc(foo) 
>>> foo 11

# 如果真的想改变参数，那么可以使用一点小技巧，即将值放置在列表中:
>>> def inc(x):    
        x[0] = x[0] + 1 
... 
>>> foo = [10] 
>>> inc(foo) 
>>> foo
[11] 
```

这样就会返回新值，代码看起来也比较清晰。

### 6.4.3 关键字参数和默认值

目前为止我们所使用的参数都叫做*位置参数*，因为它们的位置很重要，事实上比它们的名字更加重要。本节中引入的这个功能可以回避位置问题，当你慢慢习惯使用这个功能以后，就会发现程序规模越大，它们的作用也就越大。

```py
# 考虑下面的两个函数:
def hello_1(greeting, name): 
    print "%s, %s!" % (greeting, name) 
def hello_2(name, greeting): 
    print "%s, %s!" % (name, greeting) 
# 两个代码所实现的是完全一样的功能，只是参数顺序反过来了:
>>> hello_1("Hello", "world")
Hello, world! 
>>> hello_2("Hello", "world")
Hello, world! 
# 有些时候(尤其是参数很多的时候)，参数的顺序是很难记住的。为了让事情简单些，可以提供参数的名字:
>>> hello_1(greeting="Hello", name="world")
Hello, world! 
# 这样一来，顺序就完全没影响了:
>>> hello_1(name="world", greeting="Hello")
Hello, world! 
# 但参数名和值一定要对应:
>>> hello_2(greeting="Hello", name="world")
world, Hello! 
```

这类使用参数名提供的参数叫做*关键字参数*。它的主要作用在于可以明确每个参数的作用，也就避免了下面这样的奇怪的函数调用：

```py
>>> store("Mr. Brainsample", 10, 20, 13, 5) 
# 可以使用:
>>> store(patient="Mr. Brainsample", hour=10, minut=20, day=13, month=5) 
```

尽管这么做打的字就多了些，但是很显然，每个参数的含义变得更加清晰。而且就算弄乱了参数的顺序，对于程序的功能也没有任何影响。

关键字参数最厉害的地方在于可以在函数中给参数提供默认值：

```py
def hello_3(greeting="Hello", name="world"): 
    print "%s, %s!" % (greeting, name) 
# 当参数具有默认值的时候，调用的时候就不用提供参数了！可以不提供、提供一些或提供所有的参数:
>>> hello_3()
Hello, world! 
>>> hello_3("Greetings")
Greetings, world! 
>>> hello_3("Greetings", "universe")
Greetings, universe! 
```

可以看到，位置参数这个方法不错，只是在提供名字的时候同时还要提供问候语。但是如果只想提供`name`参数，而让`greeting`使用默认值该怎么办呢？相信此刻你已经猜到了：

```py
>>> hello_3(name="XuHoo")
Hello, XuHoo! 
```

很简洁吧？还没完。位置参数和关键字参数是可以联合使用的。把位置参数放置在前面就可以了。如果不这样做，解释器会不知道它们到底是谁(也就是它们应该处的位置)。

*注：除非完全清除程序的功能和参数的意义，否则应该避免混合使用位置参数和关键字参数。一般来说，只有在强制要求的参数个数比可修改的具有默认值的参数个数少的时候，才使用上面提到的参数书写方法。*

例如，`hello`函数可能需要名字作为参数，但是也允许用户自定义名字、问候语和标点：

```py
def hello_4(name, greeting="Hello", punctuation="!"): 
    print "%s, %s%s" % (greeting, name, punctuation) 
# 调用函数的方式很多，下面是其中一些:
>>> hello_4("Mars")
Hello, Mars! 
>>> hello_4("Mars", "Howdy")
Howdy, Mars! 
>>> hello_4("Mars", "Howdy", "...")
Howdy, Mars... 
>>> hello_4("Mars", punctuation=".")
Hello, Mars. 
>>> hello_4("Mars", greeting="Top of the morning to ya")
Top of the morning to ya, Mars! 
>>> hello_4()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  TypeError: hello_4() takes at least 1 argument (0 given) 
```

*注：如果为`name`也赋予默认值，那么最后一个语句就不会产生异常。*

很灵活吧？我们也不需要做多少工作。下一节中我们可以做得更灵活。

### 6.4.4 收集参数

有些时候让用户提供任意数量的参数是很有用的。比如在名字存储程序中(本章前面“为什么我想要修改参数”一节用到的)，用户每次只能存一个名字。如果能像下面这样存储多个名字就更好了：

```py
>>> store(data, name1, name2, name3) 
# 用户可以给函数提供任意多的参数。实现起来也不难。 
# 试着像下面这样定义函数:
def print_params(*params): 
    print params 
# 这里我只指定了一个参数，但是前面加上了个星号。这是什么意思？让我们用一个参数调用函数看看会发生什么:
>>> print_params("XuHoo")
('XuHoo',) 
# 可以看到，结果作为元组打印出来，因为里面有个逗号(长度为 1 的元组有些奇怪，不是吗)。所以在参数前使用星号就能打印出元组？那么在 Params 中使用多个参数看看会发生什么:
>>> print_params(1, 2, 3)
(1, 2, 3) 
# 参数前的星号将所有值放置在同一个元组中。可以说是将这些值收集起来，然后使用。不知道能不能与普通参数联合使用。让我们再写个函数:
def print_params_2(title, *params): 
    print title 
    print params 
# 试试看
>>> print_params_2("Params:", 1, 2, 3)
Params:
(1, 2, 3) 
# 没问题！所以星号的意思就是"收集其余的位置参数"。如果不提供任何供收集的元素，params 就是个空元组:
>>> print_params_2("Nothing",)
Nothing
() 
# 的确如此，很有用。那么能不能处理关键字参数(也是参数)呢？
>>> print_params_2("XuHoo", something=19)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  TypeError: print_params_2() got an unexpected keyword argument 'something'

# 看来不行。所以我们需要另外一个能处理关键字参数的“收集操作”。那么语法应该怎么写呢？会不会是"**"？
def print_params_3(**params): 
    print params 
# 至少解释器没有报错。调用一下看看:
>>> print_params_3(x=1, y=2, z=3)
{'y': 2, 'x': 1, 'z': 3} 
# 返回的是字典而不是元组。放一起用用看:
def print_params_4(x, y, z=3, *pospar, **keypar): 
    print x, y, z 
    print pospar 
    print keypar 
# 和我们期望的结果别无二致:
>>> print_params_4(1, 2, 3, 4, 5, 6, 7, foo=1, bar=2) 
1 2 3 
(4, 5, 6, 7)
{'foo': 1, 'bar': 2} 
>>> print_params_4(1, 2) 
1 2 3 
()
{} 
```

联合使用这些功能，可以做的事情就多了。如果你想知道几种功能联合起来如何工作(或者说是否允许这么做)，那么就自己动手试试看吧(下一节中，会看到`*`和`**`是怎么用来进行函数调用的，不管是否在函数定义中使用)。

现在回到原来的问题上：怎么实现多个名字同时存储。解决方案如下：

```py
def store(data, *full_names): 
    for full_name in full_names:
        names = full_name.split() 
        if len(names) == 2:
            names.insert(1, "")
        labels = "first", "middle", "last"
        for label, name in zip(labels, names):
            people = lookup(data, label, name) 
            if people:
                people.append(full_name) else:
                data[label][name] = [full_name] 
# 使用这个函数就像上一节中的只接受一个名字的函数一样简单:
>>> d = {} 
>>> init(d) 
>>> store(d, "Han Solo") 
# 但是现在可以这样使用:
>>> store(d, "Luke Skywalker", "Anakin Skywalker") 
>>> lookup(d, "last", "Skywalker")
["Luke Skywalker", "Anakin Skywalker"] 
```

### 6.4.5 参数收集的逆过程

如何将参数收集为元组和字典已经讨论过了，但是事实上，如果使用`*`和`**`的话，也可以执行相反的操作。那么参数收集的逆过程是什么样？假设有如下函数：

```py
def add(x, y):
    return x + y 
```

*注：`operator`模块中包含此函数的效率更高的版本。*

比如说有个包含由两个要相加的数字组成的元组：

```py
params = (1, 2) 
```

这个过程或多或少有点像我们上一节中介绍的方法的逆过程。不是要收集参数，而是*分配*它们在“另一端”。使用`*`运算符就简单了——不过是在调用而不是在定义时使用：

```py
>>> add(*params) 
3 
```

对于参数列表来说工作正常，只要扩展的部分是最新的就可以。可以使用同样的技术来处理字典——使用双星号运算符。假设之前定义了`hello_3`，那么可以这样使用：

```py
>>> params = {"name":"Sir Robin", "greeting":"Well met"} 
>>> hello_3(**params)
Well met, Sir Robin! 
```

在定义或调用函数时使用星号(或者双星号)仅传递元组或字典，所以可能没遇到什么麻烦：

```py
>>> def with_stars(**kwds):
...     print kwds["name"], "is", kwds["age"], "year old" 
... 
>>> def without_stars(kwds):
...     print kwds["name"], "is", kwds["age"], "year old" 
... 
>>> args = {"name": "XuHoo", "age": 19} 
>>> with_stars(**args)
XuHoo is 19 year old 
>>> without_stars(args)
XuHoo is 19 year old 
```

可以看到，在`with_stars`中，我在定义和调用函数时都使用了星号。而在 without_stars 中两处都没用，但得到了同样的效果。所以星号只在定义函数(允许使用不定数目的参数)或者调用(“分割”字典或者序列)时才有用。

*注：使用拼接(Splicing)操作符“传递”参数很有用，因为这样一来就不用关心参数的个数之类的问题，例如：*

```py
def foo(x, y, z, m=0, n=0): 
    print x, y, z, m, n 
def call_foo(*args, **kwds): 
    print "Calling foo!" 
foo(*args, **kwds) 
```

*在调用超类的构造函数时这个方法尤其有用(请参见第九章获取更多信息)。*

### 6.4.6 练习使用参数

有了这么多种提供和接受参数的方法，很容易犯晕吧！所以让我们把这些方法放在一起举个例子。首先，我定义了一些函数：

```py
def story(**kwds): 
    return "Once upon a time, there was a " \ "%(job)s called %(name)s. " % kwds 
def power(x, y, *others): 
    if others: 
        print "Received redundant parameters:", others 
    return pow(x, y) 
def interval(start, stop=None, step=1): 
    "Imitates range() for step > 0"
    if stop is None:  # 如果没有为 stop 指定值······
        start, stop = 0, start  # 指定参数
    result = []
    i = start  # 计算 start 索引
    while i < stop:  # 直到计算到 stop 的索引
        result.append(i)  # 将索引添加到 result 内······
        i += step  # 用 stop(>0)增加索引······
    return result # 让我们试一下:
>>> print story(job="king", name="XuHoo")
Once upon a time, there was a king called XuHoo. 
>>> print story(name="Sir Robin", job="brave knight")
Once upon a time, there was a brave knight called Sir Robin. 
>>> params = {"job": "language", "name": "Python"} 
>>> print story(**params)
Once upon a time, there was a language called Python. 
>>> del params["job"] 
>>> print story(job="stroke of genius", **params)
Once upon a time, there was a stroke of genius called Python. 
>>> power(2, 3) 8
>>> power(3, 2) 9
>>> power(y=3, x=2) 8
>>> params = (5,) * 2
>>> power(*params) 3125
>>> power(3, 3, "Hello, world")
Received redundant parameters: ('Hello, world',) 27
>>> interval(10)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
>>> interval(1, 5)
[1, 2, 3, 4] 
>>> interval(3, 12, 4)
[3, 7, 11] 
>>> power(*interval(3, 7))
Received redundant parameters: (5, 6) 81 
```

这些函数应该多加练习，加以掌握。

## 6.5 作用域

到底什么是变量？你可以把它们看做是值的名字。在执行`x=1`赋值语句后，名称`x`引用到值`1`上。这就像用字典一样，键引用值，当然，变量和所对应的值用的是个“不可见”的字典。实际上这么说已经很接近真是情况了。內建的`vars`函数可以返回这个字典：

```py
>>> x = 1
>>> scope = vars() 
>>> scope["x"] 
1
>>> scope["x"] += 1
>>> x 
2 
```

*注：一般来说，`vars`所返回的字典是不能修改的，因为根据官方 Python 文档的说法，结果是未定义的。换句话说，可能得不到想要的结果。*

这类“不可见字典”叫做*命名空间*或者*作用域*。那么到底有多少个命名空间？除了全局作用域外，每个函数调用都会创建一个新的作用域：

```py
>>> def foo():  
        x = 19 
... 
>>> x = 1
>>> foo() 
>>> x 
1 
```

这里的`foo`函数改变(重绑定)了变量`x`，但是在最后的时候，`x`并没有变。这是因为当调用`foo`的时候，新的命名空间就被创建了，它作用于`foo`内的代码块。赋值语句`x=19`只在内部作用域(*局部*命名空间)起作用，所以它并不影响外部(*全局*)作用域中的`x`。函数内的变量被称为*局部变量*(local variable，这是与全局变量相反的概念)。参数的工作原理类似于局部变量，所以用全局变量的名字作为参数名并没有问题。

```py
>>> def output(x):
        print x
... 
>>> x = 1
>>> y = 2
>>> output(y) 
2 
```

目前为止一切正常。但是如果需要在函数内部访问全局变量怎么办呢？而且只想读取变量的值(也就是说不想重绑定变量)，一般来说是没有问题的：

```py
>>> def combine(parameter):
        print parameter + external
... 
>>> external = "berry"
>>> combine("Shrub")
Shrubberry 
```

*注：像这样引用全局变量是很多错误的引发原因。慎重使用全局变量。*

**屏蔽引发的问题**

读取全局变量一般来说并不是问题，但是还是有个会出问题的事情。如果局部变量或者参数的名字和想要访问的去全局变量相同的话，就不能直接访问了。全局变量会被局部变量屏蔽。

如果的确需要的话，可以使用`globals`函数获取全局变量值，该函数的近亲是`vars`，它可以返回全局变量的字典(`locals`返回局部变量的字典)。例如，如果前例中有个叫做`parameter`的全局变量，那么就不能在`combine`函数内部访问该变量，因为你有一个与之同名的参数。必要时，能使用`globals()["parameter"]`获取：

```py
>>> def combine(parameter):
...     print parameter + globals()["parameter"]
... 
>>> parameter = "berry"
>>> combine("Shrub")
Shrubberry 
```

接下来讨论*重绑定*全局变量(使变量引用其他新值)。如果在函数内部将值赋予一个变量，它会自动生成为局部变量——除非告知 Python 将其声明为全局变量(注意只有在需要的时候才使用全局变量。它们会让代码变得混乱和不灵活。局部变量可以让代码更加抽象，因为它们是在函数中“隐藏”的)。那么怎么才能告诉 Python 这是一个全局变量呢？

```py
>>> x = 1
>>> def change_global():
... global x
...     x = x + 1 ... >>> change_global() >>> x 2 
```

小菜一碟！

**嵌套作用域**

Python 的函数是可以嵌套的，也就是说可以将一个函数放在另一个里面(这个话题稍微有点复杂，如果读者刚刚接触函数和作用域，现在可以先跳过)。下面是一个例子：

```py
>>> def foo():
... def bar():
... print "Hello, world!" ...     bar() 
```

嵌套一般来说并不是那么有用，但它有一个很突出的应用，例如需要一个函数“创建”另一个。也就意味着可以像下面这样(在其他函数内)书写函数：

```py
>>> def multiplier(factor):
... def multiplyByFactor(number):
... return number * factor
... return multiplyByFactor 
```

一个函数位于另外一个里面，外层函数返回里层函数。也就是说函数本身被返回了，但并没有被调用。重要的是返回的函数还可以访问它的定义所在的作用域。换句话说，它“带着”它的环境(和相关的局部变量)。

每次调用外层函数，它内部的函数都被重新绑定，factor 变量每次都有一个新的值。由于 Python 的嵌套作用域，来自(multiplier 的)外部作用域的这个变量，稍后会被内层函数访问。例如：

```py
>>> double = multiplier(2) >>> double(5) 10
>>> triple = multiplier(3) >>> triple(3) 9
>>> multiplier(5)(4) 20 
```

类似 multiplyByFactor 函数存储子封闭作用域的行为叫做*闭包*(closure)。

外部作用域的变量一般来说是不能进行重新绑定的。但在 Python3.0 中，nonlocal 关键字被引入。它和 global 关键字的使用方法类似，可以让用户对外部作用域(但并非全局作用域)的变量进行赋值。

## 6.6 递归

前面已经介绍了很多关于创建和调用函数的知识。函数也可以调用其他函数。令人惊讶的是函数可以调用*自身*，下面将对此进行介绍。

*递归*这个词对于没接触过程序设计的人来说可能会比较陌生。简单来说就是引用(或调用)自身的意思。来看一个有点幽默的定义：

*recur sion \ri-'k&r-zh&n\ n: see recursion.*

*(递归[名词]：见递归)。*

递归的定义(包括递归函数定义)包括它们自身定义内容的引用。由于每个人对递归的掌握程度不同。它可能会让人大伤脑筋，也可能是小菜一碟。为了深入理解它，读者应该买本计算机科学方面的好书，常用 Python 解释器也能帮助理解。

使用“递归”的幽默定义来定义递归递归一般来说是不可行的，因为那样什么也做不了。我们需要查找递归的意思，结果它告诉我们请参见递归，无穷尽也。一个类似的函数定义如下：

```py
def recursion(): return recursion() 
```

显然它做不了任何事情——和刚才那个递归的假定义一样没用。运行一下，会发生什么事情？欢迎尝试：不一会，程序直接就崩溃了(发生异常)。理论上讲，它应该永远运行下去。然而每次调用函数都会用掉一点内存，在足够的函数调用发生后(在之前的调用返回后)，空间就不够了，程序会以一个“超过最大递归深度”的错误信息结束。

这类递归叫做*无穷递归*(infinite recursion)，类似于 while True 开始的*无穷循环*，中间没有 break 或 return 语句。因为(理论上讲)它永远不会结束。我们想要的是能做一些有用的事情的递归函数。有用的递归函数包含以下几部分：

a.当函数直接返回值时有基本实例(最小可能性问题)；

b.*递归实例*，包括一个或者多个问题较小部分的递归调用。

这里关键就是讲问题分解为小部分，递归不能永远继续下去，因为它总是以最小可能性问题结束，而这些问题又存储在基本实例中，所以才会让函数调用自身。

但是怎么将其实现呢？做起来没有看起来这么奇怪。就像我刚才说的那样，每次函数被调用时，针对这个调用的新命名空间会被创建，意味着当函数调用“自身”时，实际上运行的是两个不同的函数(或者说是同一个函数具有两个不同的命名空间)。实际上，可以将它想象成和同种类的一个生物进行对话的另一个生物对话。

### 6.6.1 两个经典：阶乘和幂

本节中，我们会看到两个经典的递归函数。首先，假设想要计算数 n 的*阶乘*。n 的阶乘定义为 n x (n -1) x (n -2) x ··· x 1。很多数学应用中都会用到它(比如计算将 n 个人排为一行共有多少种方法)。那么该怎么计算呢？可以使用循环：

```py
def factorial(n):
    result = n for i in range(1, n):
        result *= i return result 
```

这个方法可行而且容易实现。它的主要过程是：首先，将 result 赋值到 n 上，然后 result 依次与 1~n-1 的数相乘，最后返回结果。下面来看看使用递归的版本。关键在于阶乘的数学定义，下面就是：

a.1 的阶乘是 1；

b.大于 1 的数 n 的阶乘是 n 乘 n-1 的阶乘。

可以看到，这个定义完全符合刚才所介绍的递归的两个条件。

现在考虑如何将定义实现为函数。理解了定义本身以后，实现其实很简单：

```py
def factorial(n): if n == 1: return 1
    else: return n * factorial(n-1) 
```

这是定义的直接实现。只要记住函数调用 factorial(n)是和调用 factorial(n-1)不同的实体就行。

考虑另外一个例子。假设需要计算幂，就像內建的 pow 函数或者**运算符一样。可以用很多种方法定义一个数的(整数)幂。先看一个简单的例子：power(x, n)(x 为 n 的幂次)是 x 自乘 n-1 次的结果(所以 x 用作乘数 n 次)。所以 power(2, 3)是 2 乘以自身两次：2 x 2 x 2 = 8。

实现很简单：

```py
def power(x, n):
    result = 1
    for i in range(n):
        result *= x return result 
```

程序很小巧，接下来把它改编为递归版本：

a.对于任意数字来说，power(x, 0)是 1；

b.对于任何大于 0 的数来说，power(x, n)是 x 乘以(x, n-1)的结果。

同样，可以看到这与简单版本的递归定义的结果相同。

理解定义是最困难的部分——实现起来就简单了：

```py
def power(x, n): if n == 0: return 1
    else: return x * power(x, n-1) 
```

文字描述的定义再次被转换为了程序语言(Python 代码)。

注：如果函数或算法很复杂而且难懂的话，在实现前用自己的话明确地定义一下是很有帮助的。这类使用“准程序语言”编写的程序称为*伪代码。*

那么递归有什么用呢？就不能用循环代替吗？答案是肯定的，在大多数情况下可以使用循环，而且大多数情况下还会更有效率(至少会高一些)。但是在多数情况下，递归更加易读，有时会大大提高可读性，尤其当读程序的人懂得递归函数的定义的时候。尽管可以避免编写使用递归的程序，但作为程序员来说还是要理解递归算法以及其他人写的递归程序，这也是最基本的。

### 6.2.2 另外一个经典：二分法查找

作为递归实践的最后一个例子，来看看这个叫做二分法查找(binary search)的算法例子。

你可能玩过一个游戏，通过询问 20 个问题，被询问者回答是或不是，然后猜测别人在想什么。对于大多数问题来说，都可以将可能性(或多或少)减半。比如已经知道答案是个人，那么可以问“你是不是在想一个女人”，很显然，提问者不会上来就问“你是不是在想约翰·克里斯”——除非提问者会读心术。这个游戏的数学班就是猜数字。例如，被提问者可能在想一个 1~100 的数字，提问者需要猜中它。当然，提问者可以耐心地猜上 100 次，但是真正需要才多少次呢？

答案就是只需要问 7 次即可。第一个问题类似于“数字是否大于 50”，如果被提问者回答说数字大于 50，那么就问“是否大于 75”，然后继续将满足条件的值=等分(排除不满足条件的)，直到找到正确答案。这个不需要太多考虑就能解答出来。

很多其他问题上也能用同样的方法解决。一个很普遍的问题就是查找一个数字是否存在于一个(排过序)的序列中，还要找到具体位置。还可以使用同样的过程。“这个数字是否存在序列正中间的右边”，如果不是的话，“那么是否在第二个 1/4 范围内(左侧靠右)”，然后这样继续下去。提问者对数字可能存在的位置上下限心里有数，然后每个问题继续切分可能的距离。

这个算法的本身就是递归的定义，亦可用递归实现。让我们首先重看定义，以保证知道自己在做什么：

a.如果上下限相同，那么就是数字所在的位置，返回；

b.否则找到两者的中点(上下限的平均值)，查找数字是在左侧还是在右侧，继续查找数字所在的那半部分。

这个递归例子的关键就是顺序，所以当找到中间元素的时候，只需要比较它和所查找的数字，如果查找数字较大，那么该数字一定在右侧，反之则在左侧。递归部分就是“继续查找数字所在的那半部分”，因为搜索的具体实现可能会和定义中完全相同。(注意搜索的算法返回的是数字应该在的位置——如果它本身不在序列中，那么所返回位置上的其实就是其他数字)

下面来实现一个二分法查找：

```py
def search(sequence, number, lower, upper): if lower == upper: assert number == sequence[upper] return upper else:
        middle = (lower + upper) // 2
        if number > sequence[middle]: return search(sequence, number, middle+1, upper) else: return search(sequence, number, lower, middle) 
```

完全符合定义。如果 lower==upper，那么返回 upper，也就是上限。注意，程序假设(断言)所查找的数字一定会被找到(number==sequence[upper])。如果没有到达基本实例，先找到 middle，检查数字是在左边还是在右边，然后使用新的上下限继续调用递归过程。也可以将限制设为可选以方便用。只要在函数定义的开始部分加入下面的条件语句即可：

```py
def search(sequence, number, lower=0, upper=None): if upper is None:   upper = len(sequence) - 1 ······ 
```

如果现在不提供限制，程序会自动设定查找范围为整个序列，看看行不行：

```py
>>> seq = [34, 67, 8, 123, 4, 100, 95] >>> seq.sort() >>> seq
[4, 8, 34, 67, 95, 100, 123] >>> search(seq, 34) 2
>>> search(seq, 100) 5 
```

但不必这么麻烦，一则可以直接使用列表方法 index，如果想要自己实现的话，只要从程序的开始处循环迭代知道找到数字就行了。

当然可以，使用 index 没问题。但是只使用循环可能效率有点低。刚才说过查找 100 内的一个数(或位置)，只需要 7 个问题即可。用循环的话，在最糟糕的情况下要问 100 个问题。“没什么大不了的”，有人可能会这样想。但是如果列表有 100 000 000 000 000 000 000 000 000 000 000 000 个元素，要么循环多次(可能对于 Python 的列表来说这个大小有些不现实)，就“有什么大不了的”了。二分查找法只需要 117 个问题。很有效吧？(事实上，可观测到的宇宙内的粒子总数是 10**87，也就是说只要 290 个问题就能分辨它们了！)

*注：标准库中的 bisect 模块可以非常有效地实现二分查找。*

**函数式编程**

到现在为止，函数的使用方法和其他对象(字符串、数值、序列，等等)基本上一样，它们可以分配给变量、作为参数传递以及从其他函数返回。有些编程语言(比如 Scheme 或者 LISP)中使用函数几乎可以完成所有的事情，尽管在 Python(经常会创建自定义的对象——下一章会讲到)中不用那么倚重函数，但也可以进行函数式程序设计。

Python 在应对这类“函数式编程”方面有一些有用的函数：map、filter 和 reduce 函数(Python3.0 中这些都被移至 functools 模块中(除此之外还有 apply 函数。但这个函数被前面讲到的拼接操作符所取代))。map 和 filter 函数在目前版本的 Python 中并不是特别有用，并且可以使用列表推导式代替。不过读者可以使用 map 函数将序列中的元素全部传递给一个函数：

```py
>>> map(str, range(10))  # Equivalent to [str(i) for i in range(10)]
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
```

filter 函数可以基于一个返回布尔值的函数对元素进行过滤。

```py
>>> def func(x):
... return x.isalnum()
... >>> seq = ["foo", "x41", "?!", "***"] >>> filter(func, seq)
['foo', 'x41'] 
```

本例中，使用列表推导式可以不用专门定义一个函数：

```py
>>> [x for x in seq if x.isalnum()]
['foo', 'x41'] 
```

事实上，还有个叫做 lambda 表达式的特性，可以创建短小的函数("lambda"来源于希腊字母，在数学中表示匿名函数)。

```py
>>> filter(lambda x: x.isalnum, seq)
['foo', 'x41'] 
```

还是列表推导式更易读吧？

reduce 函数一般来说不能轻松被列表推导式代替，但是通常用不到这个功能。它会将序列的前两个元素与给定的函数联合使用，并且将它们的返回值和第 3 个元素继续联合使用，直到整个序列都处理完毕，并且得到一个最终结果。例如，需要计算一个序列的数字的和，可以使用 reduce 函数加上 lambda x,y: x+y(继续使用相同的数字)(事实上，不是使用 lambda 函数，而是在 operator 模块引入每个內建运算符的 add 函数。使用 operator 模块中的函数通常比用自己的函数更有效率)：

```py
>>> numbers = [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33] >>> reduce(lambda x,y: x+y, numbers) 1161 
```

当然，这里也可以使用內建函数 sum。

## 6.7 小结

本章介绍了关于抽象的常见知识以及函数的特殊知识。

☑ 抽象：抽象是隐藏多余细节的艺术。定义处理细节的函数可以让程序更抽象。

☑ 函数定义：函数使用 def 语句定义。它们是由语句组成的块，可以从“外部世界”获取值(参数)，也可以返回一个或多个值作为运算的结果。

☑ 参数：函数从参数中得到需要的信息。也就是函数调用时设定的变量。Python 中有两类参数：位置参数和关键字参数。参数在给定默认值时是可选的。

☑ 作用域：变量存储在作用域(也叫做命名空间)中。Python 中有两类主要的作用域——全局作用域和局部作用域。作用域可以嵌套。

☑ 递归：函数可以调用自身，如果它这么做了就叫递归。一切用递归实现的功能都可以用循环实现，但是有些时候递归函数更易读。

☑ 函数式编程：Python 有一些进行函数性编程的机制。包括 lambda 表达式以及 map、filter 和 reduce 函数。

### 6.7.1 本章的新函数

本章涉及的新函数如表 6-1 所示。

**表 6-1 本章的新函数**

```py
map(func, seq[, seq, ...])                 对序列中的每个元素应用函数
filter(func, seq)                          返回其函数为真的元素的列表
reduce(func, seq[, initial])               等同于 func(func(func(seq[0], seq[1]), seq[2]), ...)
sum(seq)                                   返回 seq 中所有元素的和
apply(func[, args[, kwargs]])              调用函数，可以提供参数 
```

### 6.7.2 接下来学什么

下一章会通过面向对象程序设计，把抽象提升到一个新高度。你将学到如何创建自定义对象的类型(或者说类)，和 Python 提供的类型(比如字符串、列表和字典)一起使用，以及如何利用这些知识编写出运行更快、更清晰的程序。如果你真正掌握了下一章的内容，编写大型程序会毫不费力。