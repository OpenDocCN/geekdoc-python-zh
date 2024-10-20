# 快速提示:在 Python 中使用集合

> 原文：<https://www.pythoncentral.io/quick-tip-using-sets-in-python/>

在 Python 中，集合是不包含任何重复条目的列表。使用 set 类型是确保列表不包含任何重复项的一种快速而简单的方法。下面是一个如何使用它来检查重复列表的示例:

```py
a = set(["Pizza", "Ice Cream", "Donuts", "Pizza"])
print a
```

因为“Pizza”在列表中出现了两次，所以使用 set 将产生一个“Pizza”只出现一次的列表输出，如下所示:

```py
set(['Pizza', 'Ice Cream', 'Donuts'])
```

如您所见，使用 set 类型是确保列表中的每个值都是独一无二的好方法。