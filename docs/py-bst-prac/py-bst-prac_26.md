# XML 解析

## untangle

[untangle](https://github.com/stchris/untangle) [https://github.com/stchris/untangle] 库可以将 XML 文档映射为一个 Python 对象，该对象于其结构中包含了原文档的节点与属性信息。

作为例子，一个像这样的 XML 文件：

```py
<?xml version="1.0"?>
<root>
    <child name="child1">
</root> 
```

可以被这样载入：

```py
import untangle
obj = untangle.parse('path/to/file.xml') 
```

然后你可以像这样获取 child 元素名称：

```py
obj.root.child['name'] 
```

untangle 也支持从字符串或 URL 中载入 XML。

## xmltodict

[xmltodict](http://github.com/martinblech/xmltodict) [http://github.com/martinblech/xmltodict] 是另一个简易的库， 它致力于将 XML 变得像 JSON。

对于一个像这样的 XML 文件：

```py
<mydocument has="an attribute">
  <and>
    <many>elements</many>
    <many>more elements</many>
  </and>
  <plus a="complex">
    element as well
  </plus>
</mydocument> 
```

可以装载进一个 Python 字典里，像这样：

```py
import xmltodict

with open('path/to/file.xml') as fd:
    obj = xmltodict.parse(fd.read()) 
```

你可以访问元素，属性以及值，像这样：

```py
doc['mydocument']['@has'] # == u'an attribute'
doc['mydocument']['and']['many'] # == [u'elements', u'more elements']
doc['mydocument']['plus']['@a'] # == u'complex'
doc['mydocument']['plus']['#text'] # == u'element as well' 
```

xmltodict 也有 unparse 函数让你可以转回 XML。该函数有一个 streaming 模式适合用来 处理不能放入内存的文件，它还支持命名空间。

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.