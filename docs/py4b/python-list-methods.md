# Python:列出方法

> 原文：<https://www.pythonforbeginners.com/basics/python-list-methods>

就像字符串方法一样，列表方法作用于调用它的列表，例如，如果你有一个名为 list1 = ["Movies "，" Music "，" Pictures"]的列表，那么列表方法就这样调用:list1.list_method()让我们通过在 python 解释器中键入一些列表方法来演示这一点。

```py
>>> list1 = ["Movies", "Music", "Pictures"]

#list1.append(x) will add an element to the end of the list
>>> list1.append("Files")
>>> list1
['Movies', 'Music', 'Pictures', 'Files']

#list1.insert(x, y) will add element y on the place before x
>>> list1.insert(2,"Documents")
>>> list1
['Movies', 'Music', 'Documents', 'Pictures', 'Files']

#list1.remove(x) will remove the element x from the list
>>> list1.remove("Files")
>>> list1
['Movies', 'Music', 'Documents', 'Pictures']

#list1.extend(x) will join the list with list x
>>> list2 = ["Music2", "Movies2"]
>>> list1.extend(list2)
>>> list1
['Movies', 'Music', 'Documents', 'Pictures', 'Music2', 'Movies2'] 
```

请看这个[列表方法](https://www.pythonforbeginners.com/basics/lists-methods "Python List Methods")帖子，它描述了更多的列表方法