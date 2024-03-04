# Python 列表备忘单

> 原文：<https://www.pythonforbeginners.com/lists/python-lists-cheat-sheet-2>

## 什么是列表？

Python 列表用于存储数据集合。Python 可以将多个值赋给一个列表，这在处理大量数据时非常方便。

列表可以保存任何类型的数据，包括整数、字符串，甚至其他列表。列表是动态的，可以更改。使用特殊的方法，我们可以在 Python 列表中添加或删除项目。

列表中的元素是有索引的，每个元素在列表的顺序中都有明确的位置。与 Python 字符串不同，列表的内容是可以改变的。

## 列表创建

Python 列表是用方括号写的。列表中的元素用逗号分隔。稍后，我们将看到如何添加和删除元素。

```py
# a list for days of the work week
weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday"]

# an empty list just waiting to do something
empty_list = []

# lists can hold data of different types
mix_list = ["one","two",1,2] 
```

## 查找列表的长度

使用 **len()** 方法计算列表的长度。这个方法将返回列表中元素的总数。

```py
nums = [0,1,2,3,4,5,6,7,8,9]
# print the total number of items in the list
print("Length of the list: ", len(nums)) 
```

**输出**

```py
Length of the list:  10
```

## 追加列表

我们可以使用 **append()** 方法向列表中添加条目。新元素将出现在列表的末尾。

```py
# a list of popular car manufacturers
car_brands = ["BMW","Ford","Toyota","GM","Honda","Chevrolet"]

# add to a list with append()
car_brands.append("Tesla") 
```

## 列表插入

在上面的例子中，我们看到我们可以将项目添加到列表的末尾。如果我们想把一些东西放在开头，甚至中间呢？

用 **insert()** 方法，我们可以指定在列表中的什么地方添加一个新元素。

```py
letters = ['B','C','D','E','F','G']
letters.insert(0,'A') # add element 'A' at the first index

print(letters) 
```

**输出**

```py
['A', 'B', 'C', 'D', 'E', 'F', 'G']
```

**列表插入语法:**

```py
my_list.insert(x,y) # this will insert y before x
```

```py
# an example of inserting an element into the third position in a list
top_five = ["The Beatles","Marvin Gaye","Gorillaz","Cat Power"]
top_five.insert(2, "Prince")

print(top_five)
```

**输出**

```py
 ['The Beatles', 'Marvin Gaye', 'Prince', 'Nirvana', 'Cat Power']
```

## 从列表中删除元素

从列表中删除一个元素，使用 **remove()** 方法。这个方法将找到列表中第一个出现的条目并删除它。

```py
# a basic to do list
to_do = ["dishes","laundry","dusting","feed the dog"]
# we already fed Fido!
to_do.remove("feed the dog")
print("Things to do: ", to_do)

# remove the first 3 in the list
nums = [1,2,3,3,4,5]
nums.remove(3)
print(nums)
```

**输出**

```py
Things to do:  ['dishes', 'laundry', 'dusting']
[1, 2, 3, 4, 5] 
```

建议阅读:[如何用 Python 制作聊天 app？](https://codinginfinite.com/python-chat-application-tutorial-source-code/)

## 扩展列表

Python 提供了一种用 **extend()** 方法连接列表的方法。使用这种方法，一个列表的元素将被添加到另一个列表的*末端*。

```py
# we need a list of items to send to the  movers
furniture = ["bed","chair","bookcase"]

# add additional elements with extend()
furniture.extend(["couch","desk","coffee table"])
print(furniture) 
```

**输出**

```py
['bed', 'chair', 'bookcase', 'couch', 'desk', 'coffee table']
```

## 使用 pop()删除元素

除了 remove()，我们还可以使用 **pop()** 方法从列表中移除元素。使用 pop()方法移除特定索引处的元素。

```py
nums = [1,2,3,4]
nums.pop(1)
print(nums) 
```

**输出**

```py
[1, 3, 4]
```

位于索引 1 的元素已被移除。如果我们不向 pop()传递索引，它将从列表中删除最后一项。

```py
# generate a list of numbers 1-10
nums = [x for x in range(1,11)]
# pop the last element off the list
nums.pop()
print(nums) 
```

**输出**

```py
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## 关键词

在处理列表时，有几个 Python 关键字很方便。关键字中的**可以用来检查一个项目是否在列表中。**

在中使用**的语法如下:**

```py
list_item in list
```

下面是一个使用关键字中的**来确定列表是否包含特定字符串的示例:**

```py
the_beatles = ["John","Paul","George","Ringo"]
print("Was John in the Beatles? ","John" in the_beatles)
```

**输出**

```py
Was John in the Beatles?  True
```

另一个有用的关键词是**不是**。通过使用 not，我们可以确定字符串中是否缺少某个元素。

```py
print("So Yoko wasn't a member of the Beatles? ","Yoko" not in the_beatles)
```

**输出**

```py
So Yoko wasn't a member of the Beatles?  True
```

## 反转列表

在 Python 中反转列表最简单的方法是使用 **reverse()** 方法。该方法对列表进行重新排序，使最后一个元素成为第一个元素，反之亦然。

或者，我们可以使用 Python 切片符号反向遍历列表。

```py
superheroes = ["Batman", "The Black Panther", "Iron Man"]

# use slice notation to traverse the list in reverse
for hero_name in superheroes[::-1]:
    print(hero_name)

# use the reverse method to reverse a list in place
superheroes.reverse()

print(superheroes) 
```

**输出**

```py
Iron Man
The Black Panther
Batman
['Iron Man', 'The Black Panther', 'Batman']
```

## 列表排序

使用 Python 的 **sort()** 方法对列表中的元素重新排序。默认情况下，sort()将重新排列列表，以便它包含的项目按升序排列。例如，对数字列表使用 sort 会将数字从最小到最大排序。

```py
nums = [100,2003,1997,3,-9,1]

nums.sort()
print(nums) 
```

**输出**

```py
[-9, 1, 3, 100, 1997, 2003]
```

或者，在字符串上使用 sort()将把项目按字母顺序排列。

```py
alphabet = ['B','C','A']

alphabet.sort()
print(alphabet) 
```

**输出**

```py
['A', 'B', 'C']
```

如果需要保持原来的列表不变，选择 **sorted()** 方法。sorted()方法返回一个新的列表，保持原来的列表不变。

```py
nums = [7,2,42,99,77]
# sorted will return a new list
print("Modified list:", sorted(nums))
print("Original list: ", nums) 
```

**输出**

```py
Modified list: [2, 7, 42, 77, 99]
Original list:  [7, 2, 42, 99, 77] 
```

## 列表索引

使用*索引*来引用列表中的项目。索引代表项目在列表中出现的顺序。

列表中的第一项位于索引 0 处。第二个在索引 1，依此类推。

```py
villains = ["Shredder","Darth Vader","The Joker"]

print(villains[0])
print(villains[1])
print(villains[2]) 
```

**输出**

```py
Shredder
Darth Vader
The Joker 
```

与 Python 字符串不同，列表是可以改变的。例如，我们可以使用 Python 来交换列表中第一项和第三项的内容。

```py
# swamp the first and third items of the list
temp = villains[2]
villains[2] = villains[0]
villains[0] = temp 
```

然而，有一种更简单的方法来淹没 Python 中的列表项。

```py
# swap list items with the power of Python!
villains[0],villains[2]=villains[2],villains[0] 
```

## 限幅

Python 切片允许我们从一个列表中检索多个项目。切片的符号是期望范围的开始和结束之间的冒号。

**语法**:

```py
my_list[start:end:step] 
```

对于一个给定的列表，切片符号查找起始索引和结束索引。这告诉 Python 我们要寻找的项目的范围。

可选地，我们可以指定遍历列表的**步骤**。该步骤告诉 Python 如何遍历列表。例如，我们可以提供一个负数来反向遍历列表。

```py
rainbow = ['red','orange','yellow','green','blue','indigo','violet']
print(rainbow[1]) # get the second item in the list
print(rainbow[:1]) # get items at indexes 0 and 1
print(rainbow[1:3]) # items at index 1 and 2
print(rainbow[:-1]) # all items excluding the last 
```

**输出**

```py
orange
['red']
['orange', 'yellow']
['red', 'orange', 'yellow', 'green', 'blue', 'indigo'] 
```

## 循环和列表

因为 Python 中的列表是有索引的，所以我们可以使用循环来遍历它们的元素。

```py
# a list of random numbers in ascending order
nums = [2,4,7,8,9,10,11,12,13,15,16,17]
# a list of prime numbers
primes = [2,3,5,7,11,13,17]

# loop through a Python list
for num in nums:
    if num in primes:
        print(num,end=" ") 
```

**输出**

```py
2 7 11 13 17
```

## 列出方法

我们已经看到了 Python 列表方法的例子，比如 reverse()和 sort()。不幸的是，这篇文章没有足够的篇幅来涵盖它们，但是我们提供了一个您应该知道的列表，并描述了它们的作用。

*   **Append():** 在列表末尾添加一个新项目。
*   **Count()** :返回列表中项目的总数。
*   Clear(): 从列表中删除所有项目。
*   **Extend():** 将一个列表的元素连接到另一个列表的末尾。
*   **Index():** 查找列表中某项的索引。
*   将一个条目添加到列表中给定的索引处。
*   从列表中删除最后一项。
*   从列表中删除一个特定的项目。
*   **Reverse():** 从最后一项到第一项对列表进行重新排序。
*   **Sort():** 对列表进行升序排序。

## 例子

让我们以一些在 Python 中使用列表和列表方法的例子来结束本文。

#### 示例 1:计算数字列表中所有项目的总和

```py
nums = [98,62,77,84,89]

total = 0
for i in range(len(nums)):
    total += nums[i]

print("Total: ", total) 
```

**输出**

```py
Total:  410
```

#### 示例 2:计算一组数字的平均值

```py
# find the average for a list of numbers
nums = [20,22,1.5,2,7,5.2,99]

total = 0
i = 0
while i < len(nums):
    total = total + nums[i]
    i = i + 1

average = total/len(nums)
print("The average to 2 decimal places: {:.2f}".format(average)) 
```

**输出**

```py
The average to 2 decimal places: 22.39
```

## 相关职位

*   [学习 Python 中的列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)
*   [探索 Python 字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)
*   [反转列表和字符串](https://www.pythonforbeginners.com/code-snippets-source-code/reverse-loop-on-a-list)