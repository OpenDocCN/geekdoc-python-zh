# 修改字符串

## 修改字符串

到目前为止，我们简单地搜索了一个静态字符串。正则表达式通常也用不同的方式，通过下面的 `RegexObject` 方法，来修改字符串。

| 方法/属性 | 作用 |
| --- | --- |
| split() | 将字符串在 RE 匹配的地方分片并生成一个列表， |
| sub() | 找到 RE 匹配的所有子串，并将其用一个不同的字符串替换 |
| subn() | 与 sub() 相同，但返回新的字符串和替换次数 |

# 将字符串分片

### 将字符串分片

`RegexObject` 的 split() 方法在 RE 匹配的地方将字符串分片，将返回列表。它同字符串的 split() 方法相似但提供更多的定界符；split()只支持空白符和固定字符串。就象你预料的那样，也有一个模块级的 re.split() 函数。

```py
split(string [, maxsplit = 0]) 
```

通过正则表达式将字符串分片。如果捕获括号在 RE 中使用，那么它们的内容也会作为结果列表的一部分返回。如果 maxsplit 非零，那么最多只能分出 maxsplit 个分片。

你可以通过设置 maxsplit 值来限制分片数。当 maxsplit 非零时，最多只能有 maxsplit 个分片，字符串的其余部分被做为列表的最后部分返回。在下面的例子中，定界符可以是非数字字母字符的任意序列。

```py
#!python
>>> p = re.compile(r'\W+')
>>> p.split('This is a test, short and sweet, of split().')
['This', 'is', 'a', 'test', 'short', 'and', 'sweet', 'of', 'split', '']
>>> p.split('This is a test, short and sweet, of split().', 3)
['This', 'is', 'a', 'test, short and sweet, of split().'] 
```

有时，你不仅对定界符之间的文本感兴趣，也需要知道定界符是什么。如果捕获括号在 RE 中使用，那么它们的值也会当作列表的一部分返回。比较下面的调用：

```py
#!python
>>> p = re.compile(r'\W+')
>>> p2 = re.compile(r'(\W+)')
>>> p.split('This... is a test.')
['This', 'is', 'a', 'test', '']
>>> p2.split('This... is a test.')
['This', '... ', 'is', ' ', 'a', ' ', 'test', '.', ''] 
```

模块级函数 re.split() 将 RE 作为第一个参数，其他一样。

```py
#!python
>>> re.split('[\W]+', 'Words, words, words.')
['Words', 'words', 'words', '']
>>> re.split('([\W]+)', 'Words, words, words.')
['Words', ', ', 'words', ', ', 'words', '.', '']
>>> re.split('[\W]+', 'Words, words, words.', 1)
['Words', 'words, words.'] 
```

# 搜索和替换

### 搜索和替换

其他常见的用途就是找到所有模式匹配的字符串并用不同的字符串来替换它们。sub() 方法提供一个替换值，可以是字符串或一个函数，和一个要被处理的字符串。

```py
sub(replacement, string[, count = 0]) 
```

返回的字符串是在字符串中用 RE 最左边不重复的匹配来替换。如果模式没有发现，字符将被没有改变地返回。

可选参数 count 是模式匹配后替换的最大次数；count 必须是非负整数。缺省值是 0 表示替换所有的匹配。

这里有个使用 sub() 方法的简单例子。它用单词 "colour" 替换颜色名。

```py
#!python
>>> p = re.compile( '(blue|white|red)')
>>> p.sub( 'colour', 'blue socks and red shoes')
'colour socks and colour shoes'
>>> p.sub( 'colour', 'blue socks and red shoes', count=1)
'colour socks and red shoes' 
```

subn() 方法作用一样，但返回的是包含新字符串和替换执行次数的两元组。

```py
#!python
>>> p = re.compile( '(blue|white|red)')
>>> p.subn( 'colour', 'blue socks and red shoes')
('colour socks and colour shoes', 2)
>>> p.subn( 'colour', 'no colours at all')
('no colours at all', 0) 
```

空匹配只有在它们没有紧挨着前一个匹配时才会被替换掉。

```py
#!python
>>> p = re.compile('x*')
>>> p.sub('-', 'abxd')
'-a-b-d-' 
```

如果替换的是一个字符串，任何在其中的反斜杠都会被处理。"\n" 将会被转换成一个换行符，"\r"转换成回车等等。未知的转义如 "\j" 则保持原样。逆向引用，如 "\6"，被 RE 中相应的组匹配而被子串替换。这使你可以在替换后的字符串中插入原始文本的一部分。

这个例子匹配被 "{" 和 "}" 括起来的单词 "section"，并将 "section" 替换成 "subsection"。

```py
#!python
>>> p = re.compile('section{ ( [^}]* ) }', re.VERBOSE)
>>> p.sub(r'subsection{\1}','section{First} section{second}')
'subsection{First} subsection{second}' 
```

还可以指定用 (?P<name>...) 语法定义的命名组。"\g<name>" 将通过组名 "name" 用子串来匹配，并且 "\g<number>" 使用相应的组号。所以 "\g<2>" 等于 "\2"，但能在替换字符串里含义不清，如 "\g<2>0"。（"\20" 被解释成对组 20 的引用，而不是对后面跟着一个字母 "0" 的组 2 的引用。）

```py
#!python
>>> p = re.compile('section{ (?P<name> [^}]* ) }', re.VERBOSE)
>>> p.sub(r'subsection{\1}','section{First}')
'subsection{First}'
>>> p.sub(r'subsection{\g<1>}','section{First}')
'subsection{First}'
>>> p.sub(r'subsection{\g<name>}','section{First}')
'subsection{First}' 
```

替换也可以是一个甚至给你更多控制的函数。如果替换是个函数，该函数将会被模式中每一个不重复的匹配所调用。在每次调用时，函数会被传入一个 `MatchObject` 的对象作为参数，因此可以用这个对象去计算出替换字符串并返回它。

在下面的例子里，替换函数将十进制翻译成十六进制：

```py
#!python
>>> def hexrepl( match ):
...     "Return the hex string for a decimal number"
...     value = int( match.group() )
...     return hex(value)
...
>>> p = re.compile(r'\d+')
>>> p.sub(hexrepl, 'Call 65490 for printing, 49152 for user code.')
'Call 0xffd2 for printing, 0xc000 for user code.' 
```

当使用模块级的 re.sub() 函数时，模式作为第一个参数。模式也许是一个字符串或一个 `RegexObject`；如果你需要指定正则表达式标志，你必须要么使用 `RegexObject` 做第一个参数，或用使用模式内嵌修正器，如 sub("(?i)b+", "x", "bbbb BBBB") returns 'x x'。