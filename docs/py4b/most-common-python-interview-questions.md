# 2021 年最常见的 Python 面试问题

> 原文：<https://www.pythonforbeginners.com/basics/most-common-python-interview-questions>

Python 是当今市场上最受欢迎的技能之一。

通过掌握 Python 编程语言，你也可以加入编程大师和奇才的行列，雇主们希望在专注于数据科学、机器学习、商业智能等领域或本质上是科学和数字的公司中填补职位。要被录用，你你应该精通 Python 和 Python 面试问题。

了解 Python 编程的所有知识，包括如何从多个类继承到切片，以及如何轻松使用所有数据结构，包括数组、字符串、列表、元组和矩阵。

为了你自己和你的职业生涯，你能做的最好的事情就是好好学习，准备回答一些棘手的 Python 面试问题。

做好准备将有助于缓解紧张，减少技术性面试的威胁。为了帮助你提前准备技术面试，我们整理了 2020 年最常见的 Python 面试问题。

通过让你的朋友问你这些问题来和你的朋友练习。如果可能的话，用你的智能手机拍摄模拟面试，检查你的答案和你回答问题的方式。

Q1:告诉我 Python 的主要特点是什么？

**A1:** 以下是 Python 的一些关键特性。 **Python 面试技巧**:选择至少 5 个，自信地讲述 Python 为什么伟大。

| **解释的 vs 编译的(如 C)** | Python 是一种解释型语言，这意味着代码在使用前不必编译。 |
| **动态键入** | 不需要声明变量的数据类型。Python 通过存储在其中的值自动识别数据类型。 |
| **OOP** | Python 遵循面向对象的范式，这意味着您拥有封装、继承、多态等能力。 |
| **跨平台** | 一旦编写完成，Python 可以移植到另一个平台上，几乎不需要修改。 |
| **通用目的** | Python 是一种通用语言，这意味着它可以用于任何领域，包括数据科学、机器学习、自动化、学习、科学、数学等。 |
| **函数和类是一级对象** | Python 函数和类可以赋给变量，并作为参数传递给其他函数。 |
| **扩展库** | Python 提供了丰富的库列表，用强大的函数和方法来扩展语言。 |

* * *

PYTHONPATH 环境变量的目的是什么？

**A2:** Python 使用环境变量来设置 Python 将要运行的环境。PYTHONPATH 用于包含 Python 的模块和包所在的路径或目录。请注意，PYTHONPATH 变量通常是在安装 Python 时设置的。

* * *

**Q3**:python startup、PYTHONCASEOK、PYTHONHOME 环境变量是什么？

**A3:** PYTHONSTARTUP 是您为 Python 文件设置的路径，该文件将在启动 Python 解释器模式(即口译员)。它用于初始化设置，如颜色、预加载常用模块等。

PYTHONCASEOK 用于在不区分大小写的文件系统中查找模块文件。

PYTHONHOME 是一个替代的模块搜索路径。

* * *

**Q4** :列表和元组有什么区别？

**A4:** 列表和元组的主要区别在于，列表是可变的，元组是不可变的。不能添加元组或更改维度，而列表可以。一个很好的例子是在数学意义上用作轴的(x，y，z)元组。改变这种状况是有意义的。

* * *

**Q5** :解释 Python 中的继承？还有 Python 支持多重继承吗？

继承是指一个(子)类继承了另一个类的某些特征。父类或类继承的类将定义基类变量和方法供子类使用。

与 Java 不同，Python 支持多重继承。

* * *

**Q6**:Python 中的字典是什么，如何创建字典？

**A6**:Python 字典是一种用键值对实现的数据结构。字典在其他语言中被称为关联数组，它将一个值与一个键相关联。要在 Python 中创建字典，花括号用于定义字典，如下所示:

Sampledict = {"make ":起亚，"车型":魂，"发动机尺寸":" 1.6L"}

* * *

**Q7** :负指标有什么用？

**A7** :负指数让你从右边数，而不是从左边数。换句话说，你可以从末尾到开始索引一个字符串或者一个列表。

* * *

**Q8**:range 和 xrange 有什么区别？

**A8** :两个函数都生成一个整数列表供您使用，但不同的是 range()函数返回一个 [Python list](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 对象，而 xrange()返回一个 xrange 对象。

* * *

**Q9** :什么模块有 split()、sub()和 subn()方法？这些方法是做什么的？

**A9** :这些函数来自[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python) (re)模块，用于修改字符串表达式。

| 拆分() | 该函数根据字符或模式的出现次数分割字符串。当 Python 找到字符或模式时，它将字符串中剩余的字符作为结果列表的一部分返回。参见[如何在 Python 中使用 Split](https://www.pythonforbeginners.com/dictionary/python-split) |
| sub() | 这个函数在一个字符串中搜索一个模式，当找到时，用一个替换字符串替换子字符串。 |
| subn() | 这个函数与 sub()类似，但是它的输出不同，因为它返回一个元组，其中包含所有替换和新字符串的总数。 |

* * *

**Q10:**Python 中酸洗和拆洗是什么意思？你什么时候会用它？

**A10**:pickle 模块用于序列化(或 pickle)或反序列化(unpickle)Python 对象结构。pickling 和 unpickling 的一个用途是当您想要将 Python 对象存储到数据库中时。为此，您必须在将对象保存到数据库之前将其转换为字节流。

* * *

**Q11**:map()函数是用来做什么的？

**A11**:map()函数有两个参数，一个函数和一个可迭代列表。该函数应用于 iterable 列表中的每个元素。

* * *

**Q12** :什么是‘with’语句，你会在什么时候使用它？

**A12**:‘with’语句是一个复合语句，用于异常处理。它简化了文件流等资源的管理。

* * *

**Q13**:Python 需要缩进吗，还是为了让代码更容易阅读？

A13 :在其他语言中，缩进是为了让代码更易读，但是在 Python 中，缩进表示代码块。

* * *

**Q14** :你会在哪个模块中找到 random()函数？

**A14** :可以在 NumPy 中找到生成随机数的函数 random()。

* * *

**Q15**:NumPy 是什么？

NumPy 是科学计算的基础包。它是一个包含多维数组(或矩阵数据结构)的开源库。

* * *

**Q16** :解释 Python 中的局部和全局变量。

A16 :局部变量只能在一个块中访问，全局变量可以在整个代码中访问。为了声明一个全局变量，使用了 global 关键字。

* * *

你会用什么来存储一堆不同类型的数据？数组还是列表？

**A17**:Python 中的数组是最具限制性的数据结构，因为数组中存储的元素必须是相同的数据类型。答案是 list，因为只有 list 才能包含不同数据类型的元素。

* * *

**Q18**:Python 是怎样一种解释型语言？

**A18:** Python 代码在运行前不在机器级代码中。这使得 Python 成为一种解释型语言。

* * *

**Q19** :什么是 Python 装饰器？

装饰器是任何用于修改函数或类的可调用对象。对函数或类的引用被传递给装饰器，装饰器返回修改后的函数或类。(有趣的是，Python 通常通过值传递参数，但是在 decorators 的情况下，函数或类是通过引用传递的)。

* * *

**Q20**:Python 的[字符串](https://www.pythonforbeginners.com/basics/strings-formatting)是不可变的还是可变的？

A20: 在 Python 中，字符串是不可变的。不能修改 string 对象，但可以通过重新分配变量来创建新的 string 对象。

* * *

**Q21** :什么是切片？

**A21** :切片在 Python 中的意思是提供序列、字符串、列表或元组的‘切片’或片段。

* * *

**Q22** :你用什么来捕捉和处理异常？

A22 :为了捕捉和处理异常，使用了 try 语句。

* * *

**Q23**:Python 中的 self 是什么？

**A23**:self 用于表示类的实例，以便访问属性和方法。

* * *

**Q24** :你用什么函数给数组添加元素？

**A24:** 有三个函数:append()，insert()和 extend()。append()将元素添加到列表的末尾，insert()将元素添加到列表中，extend()将另一个列表添加到现有列表中。

* * *

**Q25** :浅拷贝和深拷贝有什么区别？

**A25** :浅拷贝复制地址，深拷贝复制地址和地址引用的数据。

* * *

**Q26**:Python 支持多重继承吗？

**A26** :与 Java 不同，Python 支持多重继承。

* * *

**Q27** :如何移除 Python 数组中的值？

A27 :有两个函数可以从数组中移除元素。它们是 pop()和 remove()。

* * *

**Q28** :如果值不在列表中，那么 list.index(value)会返回什么？会是零还是别的？

**A28:** 如果 index()函数中指定的值没有在列表中找到，那么 Python 将返回 ValueError 异常

* * *

**Q29** :什么是 _init_？

A29:_ init _ 是一个构造函数方法，用于在实例化一个对象时初始化一个类的属性。

* * *

**Q30** :什么是 mylist[-1]如果 mylist = [2，244，56，95]？

**A30** :答案是 95。

* * *

**Q31** :什么是 lambda 函数？

**A31** : Lambda 函数是匿名函数，没有名字，可以有任意多的参数。唯一的规定是 lambda 函数只有一个表达式。没有显式返回，因为当计算表达式时，值隐式返回。

* * *

**Q32** :突破、继续、传球有什么区别？

**A32** :当某些条件发生时，break、continue 和 pass 语句用在 FOR 和 WHILE 循环中。break 语句允许您在外部条件发生时退出循环。通常在 if 语句中使用 break 来测试循环中的条件。continue 语句允许您通过在 continue 语句之后不执行任何操作来跳过循环部分。pass 语句允许您继续循环的一部分，就像外部条件不满足一样。

* * *

**Q33:**[::-1]是做什么的？

**A33** :反转一个列表。

* * *

**Q34** :如何检查一个字符串是否为数字？

**A34:** 如果字符串中的所有字符都是数字，函数 isnumeric()返回 true。

* * *

**Q35** :什么函数在换行符处分割一个字符串？

A35:split lines()函数将在换行符处分割字符串。

* * *

**Q36** :如果 mylist = ['hello '，' world '，' in '，' Paris']，print mylist[0]的输出是什么？

**A36** :答案是‘hello’，因为索引 0 代表第一个元素。

* * *

**Q37** :如果 str = 'Hello '，打印 str * 2 的输出是什么？

**A37:** 乘数将一个字符串重复多次。所以答案是“HelloHello”

* * *

**Q38:** 如果 str = 'Hello World '，print str[2:5]的输出是什么？

**A38:** 答案是‘llo’。

* * *

**Q39:** 如何将字符转换成整数？

**A39** :函数 ord()会把字符转换成整数。同时，学习额外的转换，比如将整数转换成八进制。

* * *

**Q40** :列表和数组有什么区别？

在存储大量数据时，数组比 Python 中的列表更有效，但数组中的值必须都是同一类型。数组非常适合数值计算，而列表则不然。例如，可以用一行代码分割数组中的每个元素。对于列表，你需要几行代码来做同样的事情。

* * *

**Q41** :模块无法加载时会引发什么异常？

**A41** :当 import 语句加载模块失败时，引发 ImportError。

* * *

**Q42** :提供代码用零初始化一个 5 乘 5 的 Numpy 数组？

**A42** :答案如下:

将 numpy 作为 np 导入

n1 = np.zeros((5，5))

n1

* * *

**Q43** :提供从字典创建数据帧的代码？

**A43:** 答案如下:

进口熊猫作为 pd

颜色= ["蓝色"、"绿色"、"白色"、"黑色"]

汽车= ["丰田"、"本田"、"起亚"、"宝马"]

d = { "汽车":汽车，"颜色":颜色}

f = pd。数据帧(d)

f

* * *

**Q44** :链表上 append()和 extend()有什么区别？

A44:append()将一个元素添加到一个列表的末尾，而 extend()将另一个列表添加到一个列表中。

列表 1 = [1，2，3]

list1.append(4)

列表 1 = [1，2，3，4]

列表 2 = [5，6，7，8]

list1.extend(list2)

列表 1 = [1，2，3，4，5，6，7，8]

* * *

**Q45**:index()函数和 find()函数有什么区别？

两个函数都将在迭代器中搜索一个值。唯一的区别是，如果值不是迭代器的一部分，index()函数将中断并返回 ValueError 异常。如果没有找到值，find()函数将返回-1。

* * *

**Q46:** 当用户按下 Ctrl-C 或 Delete 键时会引发什么异常？

**A46** :键盘中断异常产生。

* * *

**Q47**:Python 支持递归吗？如果是，你能深入多深？如果超过了会怎么样？

**A47** :是的，Python 支持递归，最高级别 997。如果超过该值，将引发 RecursionError 异常。

* * *

**Q48**:Python 中内存是如何管理的？

所有的 Python 对象和数据结构都保存在一个程序员无法访问的私有堆中。每当一个对象需要内存时，Python 内存管理器就会在堆中分配空间。Python 有一个内置的垃圾收集器，可以回收私有堆中的空间用于更多的分配。

* * *

**Q49:**Python 中可以创建空类吗？如果有，如何实现？

使用 pass 语句可以创建一个空类，但是注意你仍然可以实例化这个类。

* * *

**Q50:** 如何用 Python 代码写注释？

**A50** :注释以“#”为前缀。参见[如何在 Python 中使用注释](https://www.pythonforbeginners.com/comments/comments-in-python)

* * *

总之，学习这些 python 面试问题会让你领先一步。花点时间检查一下答案，这样你就可以对每个答案给出你自己的、自信的解释。