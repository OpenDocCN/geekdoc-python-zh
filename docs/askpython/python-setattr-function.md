# Python setattr()函数

> 原文：<https://www.askpython.com/python/python-setattr-function>

Python `setattr()`函数用于设置一个对象的属性，给定它的名字。虽然它是一个非常简单的函数，但在 Python 中的[面向对象编程环境中，它被证明是非常有用的。让我们看看如何在我们的 Python 程序中使用这个函数。](https://www.askpython.com/python/oops/object-oriented-programming-python)

* * *

## setattr()函数语法

它以对象名、属性名和值为参数，设置`object.attribute`等于`value`。由于任何对象属性都可以是任何类型，因此该函数不会引发任何异常。

**格式** : `setattr(object, attr, value)`

这里有一个简单的例子来演示`setattr()`的用法。

```py
class MyClass():
    def __init__(self, name, value):
        # Set the attribute of the class object
        setattr(self, name, value)

a = MyClass('KEY', 100)
print('Printing attribute from Class Object:', a.KEY)
print('Printing attribute from getattr():', getattr(a, 'KEY'))

```

**输出**

```py
Printing attribute from Class Object: 100
Printing attribute from getattr(): 100

```

当事先不知道对象的属性，并且不能使用`object.attribute_name = value`设置时，`setattr()`非常有用。

这是一种非常方便的方法，只要对象的属性在运行时会发生变化，就可以使用这种方法，它展示了面向对象编程在这些情况下仍然表现良好。

* * *

## 将 setattr()与 getattrr()一起使用

它通常与`getattr()`方法一起使用，以获取和设置对象的属性。

这里有一个例子来展示当与`getattr()`方法配对时`setattr()`的一些用例。

此示例为单个学生构造对象，并将每个科目的属性设置为其相应的分数。

在使用`setattr()`构建了学生对象之后，我们使用`getattr()`对学生的学科分数进行排序。

```py
class Student():
	def __init__(self, name, results):
		self.name = name
		for key, value in results.items():
                        # Sets the attribute of the 'subject' to
                        # the corresponding subject mark.
                        # For example: a.'Chemistry' = 75
			setattr(self, key, value)

	def update_mark(self, subject, mark):
		self.subject = mark

subjects = ['Physics', 'Chemistry', 'Biology']

a = Student('Amit', {key: value for (key, value) in zip(subjects, [73, 81, 90])})

b = Student('Rahul', {key: value for (key, value) in zip(subjects, [91, 89, 74])})

c = Student('Sunil', {key: value for (key, value) in zip(subjects, [59, 96, 76])})

student_list = [a, b, c]

stud_names = [student.name for student in student_list]

print('Sorted Physics Marks:')
print(sorted([getattr(s, 'Physics') for s in student_list]))

print('\nSorted Marks for all subjects:')
print(sorted([getattr(s, subject) for s in student_list for subject in subjects]))

print('\nSorted Marks for every Student:')
print(dict(zip(stud_names, [sorted([getattr(s, subject) for subject in subjects]) for s in student_list])))

```

虽然一些 Python 一行程序看起来非常复杂，但事实并非如此。第一个`sorted([getattr(s, 'Physics') for s in student_list])`相当于:

```py
ls = []
for s in student_list:
    ls.append(getattr(s, 'Physics'))
# Sort the list
ls.sort()
print(ls)

```

第二个 liner 也非常相似，但是使用了两个嵌套循环而不是一个。

```py
ls = []
for s in student_list:
    for subject in subjects:
        ls.append(getattr(s, subject))
ls.sort()
print(ls)

```

最后一个有点棘手，其中您为每个学生构造了一个字典`name: sorted_subject_marks`。

我们首先遍历每个名称，从 student 对象列表中获取属性，然后在添加到字典之前对中间列表进行排序。

```py
dct = {}
for name, s in zip(subjects, student_list):
    ls = []
    for subject in subjects:
        ls.append(getattr(s, subject))
    ls.sort()
    dct[name] = ls
print(dct)

```

完整代码片段的输出:

```py
Sorted Physics Marks:
[59, 73, 91]

Sorted Marks for all subjects:
[59, 73, 74, 76, 81, 89, 90, 91, 96]

Sorted Marks for every Student:
{'Amit': [73, 81, 90], 'Rahul': [74, 89, 91], 'Sunil': [59, 76, 96]}

```

* * *

## 结论

在本文中，我们学习了用于在运行时动态设置对象属性的`setattr()`方法。当开发人员不知道属性时，这是面向对象编程的一个非常有用的方法，因此用于构建灵活的 API。

## 参考

*   [Python 文档](https://python-reference.readthedocs.io/en/latest/docs/functions/setattr.html)