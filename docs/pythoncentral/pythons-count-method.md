# Python 的计数方法

> 原文：<https://www.pythoncentral.io/pythons-count-method/>

在 Python 中，count 方法返回对象在列表中出现的次数。count 方法的语法非常简单:

```py
list.count(obj)
```

上面的例子代表了这个方法的基本语法。当您在上下文中使用它时，您需要将“list”关键字替换为包含您的对象的列表的实际名称，将“obj”关键字替换为您想要计数的实际对象。查看下面的示例，了解如何使用 count 方法的真实示例:

首先，从一个列表开始:

```py
myList = ['blue', 'orange', 'purple', 'yellow', 'orange', 'green', 'pink'];
```

现在，要使用 count 方法对列表中的项目进行计数，您的代码应该如下所示:

```py
myList.count('orange');
myList.count('pink');
```

您可能已经猜到，当您运行上面的代码时，输出将分别是 2 和 1。