# Python 连接示例

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-join-examples>

## 概观

这篇文章将展示 Python join 方法的一些例子。

重要的是要记住，连接元素
的字符是调用函数的字符。

## 加入示例

让我们展示一个例子

创建新列表

```py
>>> music = ["Abba","Rolling Stones","Black Sabbath","Metallica"]

>>> print music
['Abba', 'Rolling Stones', 'Black Sabbath', 'Metallica'] 
```

用空格连接列表

```py
>>> print ' '.join(music)
Abba Rolling Stones Black Sabbath Metallica 
```

用新行加入列表

```py
>>> print "
".join(music)
Abba
Rolling Stones
Black Sabbath
Metallica 
```

用选项卡加入列表

```py
>>> print "	".join(music)
Abba	Rolling Stones	Black Sabbath	Metallica
>>> 
```

快乐脚本！