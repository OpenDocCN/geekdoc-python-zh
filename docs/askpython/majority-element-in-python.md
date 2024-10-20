# 在 Python 中找到多数元素

> 原文：<https://www.askpython.com/python/examples/majority-element-in-python>

嘿编码器！因此，在本教程中，我们将了解 python 编程语言中的一个简单问题。这个问题很简单，但在很多求职面试中还是会被问到。

* * *

## 理解多数元素问题

在程序中，用户需要输入具有 N 个元素的数组 A。然后，代码的目标是找到数组中的多数元素。

大小为 N 的数组 A 中的多数元素是在数组中出现了 N/2 次以上的元素。

程序将返回多数元素，或者如果没有找到/存在多数元素，将返回-1。

* * *

## 用 Python 实现多数元素查找器

在代码实现中，我们首先获取数组大小的输入，然后获取数组中由空格分隔的所有元素。

然后，我们将以字典的形式存储数组中每个元素的计数，在字典中完成了元素到元素计数的映射。

最后，我们将用 n/2 检查每个元素的计数，当计数大于 n/2 时，我们返回该数，否则返回-1。

```py
def check_majority(arr, N):
   map = {}
   for i in range(0, N):
      if arr[i] in map.keys():
         map[arr[i]] += 1
      else:
         map[arr[i]] = 1
   for key in map:
      if map[key] > (N / 2):
         return key
   return -1

arr = list(input("Enter elements of array:"))
size = len(arr)
ans = check_majority(arr, size)
if ans != -1:
   print("Majority Element is: ", ans)
else:
   print("No majority element in array")

```

* * *

## 样本输出

```py
Enter elements of array:1111111212121
Majority Element is:  1

```

* * *

## 结论

我希望您已经清楚问题陈述和代码实现。是的，解决这个问题有多种方法。你能想到什么吗？

快乐学习！😇

* * *