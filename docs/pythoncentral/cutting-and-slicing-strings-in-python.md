# 在 Python 中剪切和切片字符串

> 原文：<https://www.pythoncentral.io/cutting-and-slicing-strings-in-python/>

## **作为字符序列的 Python 字符串**

Python 字符串是单个字符的序列，与其他 Python 序列共享它们的基本访问方法——列表和元组。从字符串中提取单个字符(以及任何序列中的单个成员)的最简单方法是将它们解包到相应的变量中。

```py

>>> s = 'Don'

>>> s

'Don'

>>> a, b, c = s # Unpack into variables

>>> a

'D'

>>> b

'o'

>>> c

'n'

```

不幸的是，为了存储字符串中的每一个字符，我们很难事先知道需要多少个变量。如果我们提供的变量数量与字符串中的字符数量不匹配，Python 会给我们一个错误。

```py

s = 'Don Quijote'

a, b, c = s

Traceback (most recent call last):

File "", line 1, in

ValueError: too many values to unpack

```

### **在 Python 中通过索引访问字符串中的字符**

通常，使用 Python 的类似数组的索引语法来访问字符串的单个字符更有用。这里，和所有序列一样，重要的是要记住索引是从零开始的；也就是说，序列中的第一项是数字 0。

```py

>>> s = 'Don Quijote'

>>> s[4] # Get the 5th character

'Q'

```

如果您想从字符串的末尾开始计数，而不是从开头，请使用负索引。例如，索引-1 表示字符串最右边的字符。

```py

>>> s[-1]

'e'

>>> s[-7]

'Q'

```

Python 字符串是不可变的，这只是一种奇特的说法，一旦它们被创建，你就不能改变它们。尝试这样做会触发错误。

```py

>>> s[7]

'j'

>>> s[7] = 'x'

Traceback (most recent call last):

File "", line 1, in

TypeError: 'str' object does not support item assignment

```

如果你想修改一个字符串，你必须创建一个全新的字符串。在实践中，这很容易。我们一会儿就来看看。

### **切片 Python 字符串**

在此之前，如果你想提取一个以上的字符，位置和大小已知的块呢？这相当简单和直观。我们稍微扩展了方括号语法，这样我们不仅可以指定我们想要的片段的起始位置，还可以指定它的结束位置。

```py

>>> s[4:8]

'Quij'

```

让我们看看这里发生了什么。和以前一样，我们指定要从字符串中的位置 4(从零开始)开始。但是现在，我们不再满足于字符串中的单个字符，而是说我们想要更多的字符，直到**，但不包括位于第 8 位的字符**。

你可能会认为你也会得到 8 号位的字符。但事情不是这样的。别担心，你会习惯的。如果有帮助的话，可以把第二个索引(冒号后面的那个**)想象成指定你**不想要的第一个字符**。顺便说一句，这种机制的一个好处是，您可以通过简单地从第二个索引中减去第一个索引来快速判断您将得到多少个字符。**

使用这种语法，您可以省略一个或两个索引。第一个索引，如果省略，默认为 0，这样您的块从原始字符串的开头开始；第二个默认为字符串中的最高位置，因此您的块在原始字符串的末尾结束。省略这两个指标不太可能有多大的实际用途；正如您可能猜到的，它只是返回整个原始字符串。

```py

>>> s[4:]

'Quijote' # Returns from pos 4 to the end of the string

>>> s[:4]

'Don ' # Returns from the beginning to pos 3

>>> s[:]

'Don Quijote'

```

如果你还在纠结于这样一个事实，例如，s[0:8]返回到**为止的所有内容，但不包括第 8 位的字符**，这可能会有所帮助:对于你选择的任何 index、**n**值，`s[:n] + s[n:]`的值将始终与原始目标字符串相同。如果索引机制是包含的，则位置 **n** 处的字符将出现两次。

```py

>>> s[6]

'i'

>>> s[:6] + s[6:]

'Don Quijote'

```

就像以前一样，您可以使用负数作为索引，在这种情况下，计数从字符串的结尾(索引为-1)开始，而不是从开头开始。

```py

>>> s[-7:-3]

'Quij'

```

### **分割 Python 字符串时跳过字符**

方括号语法的最后一个变化是添加了第三个参数，该参数指定了“步幅”，即在从原始字符串中检索每个字符后要向前移动多少个字符。第一个检索到的字符总是对应于冒号之前的索引；但此后，无论您指定多少个字符作为步幅，指针都会向前移动，并在该位置检索字符。依此类推，直到达到或超过结束索引。就像我们目前遇到的情况一样，如果参数被省略，它默认为 1，这样就可以检索指定段中的每个字符。一个例子更清楚地说明了这一点。

```py

>>> s[4:8]

'Quij'

>>> s[4:8:1] # 1 is the default value anyway, so same result

'Quij'

>>> s[4:8:2] # Return a character, then move forward 2 positions, etc.

'Qi' # Quite interesting!

```

您也可以指定负步幅。正如您所料，这表明您希望 Python 在检索字符时后退。

```py

>>> s[8:4:-1]

'ojiu'

```

正如你所看到的，因为我们是在后退，所以起始索引比结束索引高是有意义的(否则什么都不会返回)。

```py

>>> s[4:8:-1]

''

```

因此，如果您指定了一个负的步幅，但是忽略了第一个或第二个索引，Python 会将缺少的值默认为在这种情况下有意义的值:开始索引到字符串的末尾，结束索引到字符串的开头。我知道，它会让你一想到它就头疼，但是 Python 知道它在做什么。

```py

>>> s[4::-1] # End index defaults to the beginning of the string

'Q noD'

>>> s[:4:-1] # Beginning index defaults to the end of the string

'etojiu'

```

这就是方括号语法，如果你知道你需要的字符块在字符串中的确切位置，它允许你检索字符块。

但是，如果您想基于字符串的**内容**检索一个块，而我们可能事先并不知道，该怎么办呢？

**检查内容**

Python 提供了字符串方法，允许我们根据指定的分隔符将字符串分割。换句话说，我们可以告诉 Python 在我们的目标字符串中寻找某个子字符串，并围绕该子字符串分割目标字符串。它通过返回结果子字符串的列表(减去分隔符)来实现这一点。顺便说一下，我们可以选择不显式指定分隔符，在这种情况下，它默认为空白字符(空格，' \t '，' \n '，' \r '，' \f ')或此类字符的序列。

请记住，这些方法对调用它们的字符串没有影响；它们只是返回一个新的字符串。

```py

>>> s.split()

['Don', 'Quijote']

>>> s

'Don Quijote' # s has not been changed

```

更有用的是，我们可以将返回的列表直接存储到适当的变量中。

```py

>>> title, handle = s.split()

>>> title

'Don'

>>> handle

'Quijote'

```

让我们的西班牙英雄暂时离开他的风车，让我们想象一下，我们有一个字符串，其中包含以小时、分钟和秒表示的时钟时间，用冒号分隔。在这种情况下，我们可以合理地将单独的部分收集到变量中，以便进一步操作。

```py

>>> tim = '16:30:10'

>>> hrs, mins, secs = tim.split(':')

>>> hrs

'16'

>>> mins

'30'

>>> secs

'10'

```

我们可能只想分割目标字符串一次，不管分隔符出现多少次。`split()`方法将接受第二个参数，该参数指定要执行的最大分割数。

```py

>>> tim.split(':', 1) # split() only once

['16', '30:10']

```

这里，字符串在第一个冒号处被拆分，其余部分保持不变。如果我们想让 Python 从字符串的另一端开始寻找分隔符呢？嗯，有一个叫做`rsplit()`的变体方法，它就是这么做的。

```py

>>> tim.rsplit(':', 1)

['16:30', '10']

```

**建造隔墙**

类似的字符串方法是`partition()`。这也基于内容分割一个字符串，不同之处在于结果是一个`tuple`，它保留了分隔符，以及它两边的目标字符串的两个部分。与`split()`不同，`partition()`总是只做一次拆分操作，不管分隔符在目标字符串中出现多少次。

```py

>>> tim = '16:30:10'

>>> tim.partition(':')

('16', ':', '30:10')

```

与`split()`方法一样，`partition()`、`rpartition()`也有一个变体，它从目标字符串的另一端开始搜索分隔符。

```py

>>> tim.rpartition(':')

('16:30', ':', '10')

```

### **使用 Python 的 string.replace()**

现在，回到我们的堂吉诃德。早些时候，当我们试图通过将“x”直接赋给`s[7]`来将“j”改为“x”从而使他的名字英语化时，我们发现我们做不到，因为你不能改变现有的 Python 字符串。但是我们可以通过在旧字符串的基础上创建一个我们更喜欢的新字符串来解决这个问题。允许我们这样做的字符串方法是`replace()`。

```py

>>> s.replace('j', 'x')

'Don Quixote'

>>> s

'Don Quijote' # s has not been changed

```

再说一次，我们的字符串没有被改变。所发生的是 Python 只是根据我们给出的指令返回了一个新的字符串，然后立即丢弃它，留下我们的原始字符串不变。为了保存我们的新字符串，我们需要将它赋给一个变量。

```py

>>> new_s = s.replace('j', 'x')

>>> s

'Don Quijote'

>>> new_s

'Don Quixote'

```

当然，我们可以重用现有的变量，而不是引入新的变量。

```py

>>> s = s.replace('j', 'x')

>>> s

'Don Quixote'

```

这里，虽然看起来我们改变了原来的字符串，但实际上我们只是丢弃了它，并在它的位置存储了一个新的字符串。

注意，默认情况下，`replace()`将用新的子字符串替换搜索子字符串的每一次出现。

```py

>>> s = 'Don Quijote'

>>> s.replace('o', 'a')

'Dan Quijate'

```

我们可以通过添加一个额外的参数来指定搜索子串应该被替换的最大次数，从而控制这种浪费。

```py

>>> s.replace('o', 'a', 1)

'Dan Quijote'

```

最后，replace()方法不限于作用于单个字符。我们可以用某个指定的值替换整个目标字符串。

```py

>>> s.replace(' Qui', 'key ')

'Donkey jote'

```

有关所用字符串方法的参考，请参见以下内容:

*   [str.replace(old，new[，count])](http://docs.python.org/3/library/stdtypes.html#str.replace "Python's str.replace")
*   [字符串分区(sep)](http://docs.python.org/3/library/stdtypes.html#str.partition "Python's str.partiton")
*   [str.rsplit(sep=None，maxsplit=-1)](http://docs.python.org/3/library/stdtypes.html#str.rsplit "Pythons str.rsplit")
*   [str.split(sep=None，maxsplit=-1)](http://docs.python.org/3/library/stdtypes.html#str.split "Python's str.split")