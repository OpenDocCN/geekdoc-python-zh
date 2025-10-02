# 使用正则表达式

## 使用正则表达式

现在我们已经看了一些简单的正则表达式，那么我们实际在 Python 中是如何使用它们的呢？ re 模块提供了一个正则表达式引擎的接口，可以让你将 REs 编译成对象并用它们来进行匹配。

# 编译正则表达式

### 编译正则表达式

正则表达式被编译成 `RegexObject` 实例，可以为不同的操作提供方法，如模式匹配搜索或字符串替换。

```py
#python
>>> import re
>>> p = re.compile('ab*')
>>> print p
<_sre.SRE_Pattern object at 0xb76e1a70> 
```

re.compile() 也接受可选的标志参数，常用来实现不同的特殊功能和语法变更。我们稍后将查看所有可用的设置，但现在只举一个例子：

```py
#!python
>>> p = re.compile('ab*', re.IGNORECASE) 
```

RE 被做为一个字符串发送给 re.compile()。REs 被处理成字符串是因为正则表达式不是 Python 语言的核心部分，也没有为它创建特定的语法。（应用程序根本就不需要 REs，因此没必要包含它们去使语言说明变得臃肿不堪。）而 re 模块则只是以一个 C 扩展模块的形式来被 Python 包含，就象 socket 或 zlib 模块一样

将 REs 作为字符串以保证 Python 语言的简洁，但这样带来的一个麻烦就是象下节标题所讲的。

# 反斜杠的麻烦

### 反斜杠的麻烦

在早期规定中，正则表达式用反斜杠字符 ("\") 来表示特殊格式或允许使用特殊字符而不调用它的特殊用法。这就与 Python 在字符串中的那些起相同作用的相同字符产生了冲突。

让我们举例说明，你想写一个 RE 以匹配字符串 "\section"，可能是在一个 LATEX 文件查找。为了要在程序代码中判断，首先要写出想要匹配的字符串。接下来你需要在所有反斜杠和其它元字符前加反斜杠来取消其特殊意义，结果要匹配的字符串就成了"\section"。 当把这个字符串传递给 re.compile()时必须还是"\section"。然而，作为 Python 的字符串实值(string literals)来表示的话，"\section"中两个反斜杠还要再次取消特殊意义，最后结果就变成了"\\section"。

| 字符 | 阶段 |
| --- | --- |
| \section | 要匹配的字符串 |
| \section | 为 re.compile 取消反斜杠的特殊意义 |
| "\\section" | 为"\section"的字符串实值(string literals)取消反斜杠的特殊意义 |

简单地说，为了匹配一个反斜杠，不得不在 RE 字符串中写 '\\'，因为正则表达式中必须是 "\"，而每个反斜杠在常规的 Python 字符串实值中必须表示成 "\"。在 REs 中反斜杠的这个重复特性会导致大量重复的反斜杠，而且所生成的字符串也很难懂。

解决的办法就是为正则表达式使用 Python 的 raw 字符串表示；在字符串前加个 "r" 反斜杠就不会被任何特殊方式处理，所以 r"\n" 就是包含"\" 和 "n" 的两个字符，而 "\n" 则是一个字符，表示一个换行。正则表达式通常在 Python 代码中都是用这种 raw 字符串表示。

| 常规字符串 | Raw 字符串 |
| --- | --- |
| "ab*" | r"ab*" |
| "\\section" | r"\section" |
| "\w+\s+\1" | r"\w+\s+\1" |

# 执行匹配

### 执行匹配

一旦你有了已经编译了的正则表达式的对象，你要用它做什么呢？`RegexObject` 实例有一些方法和属性。这里只显示了最重要的几个，如果要看完整的列表请查阅 Python Library Reference

| 方法/属性 | 作用 |
| --- | --- |
| match() | 决定 RE 是否在字符串刚开始的位置匹配 |
| search() | 扫描字符串，找到这个 RE 匹配的位置 |
| findall() | 找到 RE 匹配的所有子串，并把它们作为一个列表返回 |
| finditer() | 找到 RE 匹配的所有子串，并把它们作为一个迭代器返回 |

如果没有匹配到的话，match() 和 search() 将返回 None。如果成功的话，就会返回一个 `MatchObject` 实例，其中有这次匹配的信息：它是从哪里开始和结束，它所匹配的子串等等。

你可以用采用人机对话并用 re 模块实验的方式来学习它。如果你有 Tkinter 的话，你也许可以考虑参考一下 Tools/scripts/redemo.py，一个包含在 Python 发行版里的示范程序。

首先，运行 Python 解释器，导入 re 模块并编译一个 RE：

```py
#!python
Python 2.2.2 (#1, Feb 10 2003, 12:57:01)
>>> import re
>>> p = re.compile('[a-z]+')
>>> p
<_sre.SRE_Pattern object at 80c3c28> 
```

现在，你可以试着用 RE 的 [a-z]+ 去匹配不同的字符串。一个空字符串将根本不能匹配，因为 + 的意思是 “一个或更多的重复次数”。 在这种情况下 match() 将返回 None，因为它使解释器没有输出。你可以明确地打印出 match() 的结果来弄清这一点。

```py
#!python
>>> p.match("")
>>> print p.match("")
None 
```

现在，让我们试着用它来匹配一个字符串，如 "tempo"。这时，match() 将返回一个 MatchObject。因此你可以将结果保存在变量里以便後面使用。

```py
#!python
>>> m = p.match( 'tempo')
>>> print m
<_sre.SRE_Match object at 80c4f68> 
```

现在你可以查询 `MatchObject` 关于匹配字符串的相关信息了。MatchObject 实例也有几个方法和属性；最重要的那些如下所示：

| 方法/属性 | 作用 |
| --- | --- |
| group() | 返回被 RE 匹配的字符串 |
| start() | 返回匹配开始的位置 |
| end() | 返回匹配结束的位置 |
| span() | 返回一个元组包含匹配 (开始,结束) 的位置 |

试试这些方法不久就会清楚它们的作用了：

```py
#!python
>>> m.group()
'tempo'
>>> m.start(), m.end()
(0, 5)
>>> m.span()
(0, 5) 
```

group() 返回 RE 匹配的子串。start() 和 end() 返回匹配开始和结束时的索引。span() 则用单个元组把开始和结束时的索引一起返回。因为匹配方法检查到如果 RE 在字符串开始处开始匹配，那么 start() 将总是为零。然而， `RegexObject` 实例的 search 方法扫描下面的字符串的话，在这种情况下，匹配开始的位置就也许不是零了。

```py
#!python
>>> print p.match('::: message')
None
>>> m = p.search('::: message') ; print m
<re.MatchObject instance at 80c9650>
>>> m.group()
'message'
>>> m.span()
(4, 11) 
```

在实际程序中，最常见的作法是将 `MatchObject` 保存在一个变量里，然後检查它是否为 None，通常如下所示：

```py
#!python
p = re.compile( ... )
m = p.match( 'string goes here' )
if m:
print 'Match found: ', m.group()
else:
print 'No match' 
```

两个 `RegexObject` 方法返回所有匹配模式的子串。findall()返回一个匹配字符串行表：

```py
#!python
>>> p = re.compile('\d+')
>>> p.findall('12 drummers drumming, 11 pipers piping, 10 lords a-leaping')
['12', '11', '10'] 
```

findall() 在它返回结果时不得不创建一个列表。在 Python 2.2 中，也可以用 finditer() 方法。

```py
#!python
>>> iterator = p.finditer('12 drummers drumming, 11 ... 10 ...')
>>> iterator
<callable-iterator object at 0x401833ac>
>>> for match in iterator:
...     print match.span()
...
(0, 2)
(22, 24)
(29, 31) 
```

# 模块级函数

### 模块级函数

你不一定要产生一个 `RegexObject` 对象然后再调用它的方法；re 模块也提供了顶级函数调用如 match()、search()、sub() 等等。这些函数使用 RE 字符串作为第一个参数，而后面的参数则与相应 `RegexObject` 的方法参数相同，返回则要么是 None 要么就是一个 `MatchObject` 的实例。

```py
#!python
>>> print re.match(r'From\s+', 'Fromage amk')
None
>>> re.match(r'From\s+', 'From amk Thu May 14 19:12:10 1998')
<re.MatchObject instance at 80c5978> 
```

Under the hood, 这些函数简单地产生一个 RegexOject 并在其上调用相应的方法。它们也在缓存里保存编译后的对象，因此在将来调用用到相同 RE 时就会更快。

你将使用这些模块级函数，还是先得到一个 `RegexObject` 再调用它的方法呢？如何选择依赖于怎样用 RE 更有效率以及你个人编码风格。如果一个 RE 在代码中只做用一次的话，那么模块级函数也许更方便。如果程序包含很多的正则表达式，或在多处复用同一个的话，那么将全部定义放在一起，在一段代码中提前编译所有的 REs 更有用。从标准库中看一个例子，这是从 xmllib.py 文件中提取出来的：

```py
#!python
ref = re.compile( ... )
entityref = re.compile( ... )
charref = re.compile( ... )
starttagopen = re.compile( ... ) 
```

我通常更喜欢使用编译对象，甚至它只用一次，但很少人会像我这样做(如同一个纯粹主义者)。

# 编译标志

### 编译标志

编译标志让你可以修改正则表达式的一些运行方式。在 re 模块中标志可以使用两个名字，一个是全名如 IGNORECASE，一个是缩写，一字母形式如 I。（如果你熟悉 Perl 的模式修改，一字母形式使用同样的字母；例如 re.VERBOSE 的缩写形式是 re.X。）多个标志可以通过按位 OR-ing 它们来指定。如 re.I | re.M 被设置成 I 和 M 标志：

这有个可用标志表，对每个标志后面都有详细的说明。

| 标志 | 含义 |
| --- | --- |
| DOTALL, S | 使 . 匹配包括换行在内的所有字符 |
| IGNORECASE, I | 使匹配对大小写不敏感 |
| LOCALE, L | 做本地化识别（locale-aware）匹配 |
| MULTILINE, M | 多行匹配，影响 ^ 和 $ |
| VERBOSE, X | 能够使用 REs 的 verbose 状态，使之被组织得更清晰易懂 |

**I** **IGNORECASE**

使匹配对大小写不敏感；字符类和字符串匹配字母时忽略大小写。举个例子，[A-Z]也可以匹配小写字母，Spam 可以匹配 "Spam", "spam", 或 "spAM"。这个小写字母并不考虑当前位置。

**L** **LOCALE**

影响 \w, \W, \b, 和 \B，这取决于当前的本地化设置。

locales 是 C 语言库中的一项功能，是用来为需要考虑不同语言的编程提供帮助的。举个例子，如果你正在处理法文文本，你想用 \w+ 来匹配文字，但 \w 只匹配字符类 [A-Za-z]；它并不能匹配 "é" 或 "ç"。如果你的系统配置适当且本地化设置为法语，那么内部的 C 函数将告诉程序 "é" 也应该被认为是一个字母。当在编译正则表达式时使用 LOCALE 标志会得到用这些 C 函数来处理 \w 后的编译对象；这会更慢，但也会象你希望的那样可以用 \w+ 来匹配法文文本。

**M** **MULTILINE**

(此时 ^ 和 $ 不会被解释; 它们将在 4.1 节被介绍.)

使用 "^" 只匹配字符串的开始，而 $ 则只匹配字符串的结尾和直接在换行前（如果有的话）的字符串结尾。当本标志指定后， "^" 匹配字符串的开始和字符串中每行的开始。同样的， $ 元字符匹配字符串结尾和字符串中每行的结尾（直接在每个换行之前）。

**S** **DOTALL**

使 "." 特殊字符完全匹配任何字符，包括换行；没有这个标志， "." 匹配除了换行外的任何字符。

**X** **VERBOSE**

该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解。当该标志被指定时，在 RE 字符串中的空白符被忽略，除非该空白符在字符类中或在反斜杠之后；这可以让你更清晰地组织和缩进 RE。它也可以允许你将注释写入 RE，这些注释会被引擎忽略；注释用 "#"号 来标识，不过该符号不能在字符串或反斜杠之后。

举个例子，这里有一个使用 re.VERBOSE 的 RE；看看读它轻松了多少？

```py
#!python
charref = re.compile(r"""&[[]]           # Start of a numeric entity reference|||here has wrong.i can't fix
(
[0-9]+[⁰-9]      # Decimal form
| 0[0-7]+[⁰-7]   # Octal form
| x[0-9a-fA-F]+[⁰-9a-fA-F] # Hexadecimal form
)
""", re.VERBOSE) 
```

没有 verbose 设置， RE 会看起来象这样：

```py
#!python
charref = re.compile("&#([0-9]+[⁰-9]"
"|0[0-7]+[⁰-7]"
"|x[0-9a-fA-F]+[⁰-9a-fA-F])") 
```

在上面的例子里，Python 的字符串自动连接可以用来将 RE 分成更小的部分，但它比用 re.VERBOSE 标志时更难懂