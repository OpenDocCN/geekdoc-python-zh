# 常见问题

## 常见问题

正则表达式对一些应用程序来说是一个强大的工具，但在有些时候它并不直观而且有时它们不按你期望的运行。本节将指出一些最容易犯的常见错误。

# 使用字符串方式

### 使用字符串方式

有时使用 re 模块是个错误。如果你匹配一个固定的字符串或单个的字符类，并且你没有使用 re 的任何象 IGNORECASE 标志的功能，那么就没有必要使用正则表达式了。字符串有一些方法是对固定字符串进行操作的，它们通常快很多，因为它们都是一个个经过优化的 C 小循环，用以代替大的、更具通用性的正则表达式引擎。

举个 用一个固定字符串替换另一个 的例子，如：你可以把 "deed" 替换成 "word"。re.sub() 似乎正是胜任这个工作的函数，但还是考虑考虑 replace() 方法吧。注意 replace() 也可以在单词里面进行替换，可以把 "swordfish" 变成 "sdeedfish"。不过 RE 也是可以做到的。（为了避免替换单词的一部分，模式将写成 \bword\b，这是为了要求 "word" 两边有一个单词边界。这是个超出 replace 能力的工作）。

另一个常见任务是从一个字符串中删除单个字符或用另一个字符来替代它。你也许可以用 re.sub('\n',' ', s) 这样来实现，但 translate() 能够实现这两个任务，而且比任何正则表达式操作起来更快。 （translate 需要配合 string.maketrans 使用。例如：import string 后 'a1b3'.translate(string.maketrans('ab', 'cd')) ）

总之，在使用 re 模块之前，先考虑一下你的问题是否可以用更快、更简单的字符串方法来解决。

# match() vs search()

### match() vs search()

match() 函数只检查 RE 是否在字符串开始处匹配，而 search() 则是扫描整个字符串。记住这一区别是重要的。记住，match() 只报告一次成功的匹配，它将从 0 处开始；如果匹配不是从 0 开始的，match() 将不会报告它。

```py
#!python
>>> print re.match('super', 'superstition').span()
(0, 5)
>>> print re.match('super', 'insuperable')
None 
```

另一方面，search() 将扫描整个字符串，并报告它找到的第一个匹配。

```py
#!python
>>> print re.search('super', 'superstition').span()
(0, 5)
>>> print re.search('super', 'insuperable').span()
(2, 7) 
```

有时你可能倾向于使用 re.match()，只在 RE 的前面部分添加 .* 。请尽量不要这么做，最好采用 re.search() 代替之。正则表达式编译器会对 REs 做一些分析以便可以在查找匹配时提高处理速度。一个那样的分析机会指出匹配的第一个字符是什么；举个例子，模式 Crow 必须从 "C" 开始匹配。分析机可以让引擎快速扫描字符串以找到开始字符，并只在 "C" 被发现后才开始全部匹配。

添加 .* 会使这个优化失败，这就要扫描到字符串尾部，然后回溯以找到 RE 剩余部分的匹配。使用 re.search() 代替。

# 贪婪 vs 不贪婪

### 贪婪 vs 不贪婪

当重复一个正则表达式时，如用 a*，操作结果是尽可能多地匹配模式。当你试着匹配一对对称的定界符，如 HTML 标志中的尖括号时这个事实经常困扰你。匹配单个 HTML 标志的模式不能正常工作，因为 .* 的本质是“贪婪”的

```py
#!python
>>> s = '<html><head><title>Title</title>'
>>> len(s)
32
>>> print re.match('<.*>', s).span()
(0, 32)
>>> print re.match('<.*>', s).group()
<html><head><title>Title</title> 
```

RE 匹配 在 "`&lt;html&gt;`" 中的 "<"，.* 消耗掉字符串的剩余部分。在 RE 中保持更多的左，虽然 > 不能匹配在字符串结尾，因此正则表达式必须一个字符一个字符地回溯，直到它找到 > 的匹配。最终的匹配从 "<html" 中的 "<" 到 "</title>" 中的 ">",这并不是你所想要的结果。

在这种情况下，解决方案是使用不贪婪的限定符 *?、+?、?? 或 {m,n}?，尽可能匹配小的文本。在上面的例子里， ">" 在第一个 "<" 之后被立即尝试，当它失败时，引擎一次增加一个字符，并在每步重试 ">"。这个处理将得到正确的结果：

```py
#!python
>>> print re.match('<.*?>', s).group()
<html> 
```

注意用正则表达式分析 HTML 或 XML 是痛苦的。变化混乱的模式将处理常见情况，但 HTML 和 XML 则是明显会打破正则表达式的特殊情况；当你编写一个正则表达式去处理所有可能的情况时，模式将变得非常复杂。象这样的任务用 HTML 或 XML 解析器。

# 不用 re.VERBOSE

### 不用 re.VERBOSE

现在你可能注意到正则表达式的表示是十分紧凑，但它们非常不好读。中度复杂的 REs 可以变成反斜杠、圆括号和元字符的长长集合，以致于使它们很难读懂。

在这些 REs 中，当编译正则表达式时指定 re.VERBOSE 标志是有帮助的，因为它允许你可以编辑正则表达式的格式使之更清楚。

re.VERBOSE 标志有这么几个作用。在正则表达式中不在字符类中的空白符被忽略。这就意味着象 dog | cat 这样的表达式和可读性差的 dog|cat 相同，但 [a b] 将匹配字符 "a"、"b" 或 空格。另外，你也可以把注释放到 RE 中；注释是从 "#" 到下一行。当使用三引号字符串时，可以使 REs 格式更加干净：

```py
#!python
pat = re.compile(r"""
\s*                 # Skip leading whitespace
(?P<header>[^:]+)   # Header name
\s* :               # Whitespace, and a colon
(?P<value>.*?)      # The header's value -- *? used to
# lose the following trailing whitespace
\s*$                # Trailing whitespace to end-of-line
""", re.VERBOSE) 
```

这个要难读得多：

```py
#!python
pat = re.compile(r"\s*(?P<header>[^:]+)\s*:(?P<value>.*?)\s*$") 
```