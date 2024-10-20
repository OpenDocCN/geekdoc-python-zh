# 如何使用 Enumerate()函数

> 原文：<https://www.pythoncentral.io/how-to-use-the-enumerate-function/>

在 Python 中，enumerate()函数用于遍历列表，同时跟踪列表项的索引。要看到它的实际效果，首先你需要列出一个清单:

```py
pets = ('Dogs', 'Cats', 'Turtles', 'Rabbits')
```

那么您将需要这行代码:

```py
for i, pet in enumerate(pets):
print i, pet
```

您的输出应该如下所示:

```py
0 Dogs
1 Cats
2 Turtles
3 Rabbits
```

如您所见，这些结果不仅打印出了列表的内容，还打印出了它们对应的索引顺序。您还可以使用 enumerate()函数创建索引/值列表的输出，其中的索引根据您的代码进行更改。

对于我，宠物在列举(pets，7):
打印我，宠物

这段代码更改了函数，以便为列表中的第一个值分配索引号 7，以便进行枚举。在这种情况下，您的结果将如下:

```py
7 Dogs
8 Cats
9 Turtles
10 Rabbits
```