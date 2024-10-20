# Python 201:如何按值对字典排序

> 原文：<https://www.blog.pythonlibrary.org/2012/06/19/python-201-how-to-sort-a-dictionary-by-value/>

有一天，有人问我是否有办法按值对字典进行排序。如果您经常使用 Python，那么您应该知道字典数据结构根据定义是一种未排序的映射类型。有些人会将 dict 定义为一个散列表。无论如何，我需要一种方法来根据嵌套字典中的值对嵌套字典(即字典中的字典)进行排序，这样我就可以按照指定的顺序遍历这些键。我们将花一些时间来看看我发现的一个实现。

在谷歌上搜索想法后，我在 stack overflow 上找到了一个答案，它完成了我想要的大部分工作。我不得不稍微修改它，让它使用我的嵌套字典值进行排序，但是这出奇的简单。在我们得到答案之前，我们应该快速地看一下数据结构。这是一个野兽的变体，但为了你的安全，去掉了隐私部位:

```py

mydict = {'0d6f4012-16b4-4192-a854-fe9447b3f5cb': 
          {'CLAIMID': '123456789',
           'CLAIMDATE': '20120508', 
           'AMOUNT': '365.64', 'EXPDATE': '20120831'}, 
          'fe614868-d0c0-4c62-ae02-7737dea82dba': 
          {'CLAIMID': '45689654', 
           'CLAIMDATE': '20120508', 
           'AMOUNT': '185.55', 'EXPDATE': '20120831'}, 
          'ca1aa579-a9e7-4ade-80a3-0de8af4bcb21': 
          {'CLAIMID': '98754651',
           'CLAIMDATE': '20120508', 
           'AMOUNT': '93.00', 'EXPDATE': '20120831'},
          'ccb8641f-c1bd-45be-8f5e-e39b3be2e0e3': 
          {'CLAIMID': '789464321',
           'CLAIMDATE': '20120508', 'AMOUNT': '0.00',
           'EXPDATE': ''}, 
          'e1c445c2-5148-4a08-9b7e-ff5ed51c43ed': 
          {'CLAIMID': '897987945', 
           'CLAIMDATE': '20120508', 
           'AMOUNT': '62.66', 'EXPDATE': '20120831'}, 
          '77ad6dd4-5704-4060-9c38-6a93721ef98e': 
          {'CLAIMID': '23212315',
           'CLAIMDATE': '20120508', 
           'AMOUNT': '41.05', 'EXPDATE': '20120831'}
          }

```

现在我们知道我们在对付什么了。让我们快速看一下我想出的稍加修改的答案:

```py

sorted_keys = sorted(mydict.keys(), key=lambda y: (mydict[y]['CLAIMID']))

```

这是一个非常漂亮的一行程序，但我认为它有点令人困惑。以下是我对其工作原理的理解。 **sorted** 函数基于**键**对列表(字典的键)进行排序，在本例中是一个匿名函数(lambda)。向匿名函数传递字典以及一个外部键和一个我们想要排序的内部键，在本例中是“CLAIMID”。一旦排序完毕，它将返回新的列表。就我个人而言，我发现 lambdas 有点令人困惑，所以我通常花一点时间将它们解构为一个命名函数，以便我能更好地理解它们。话不多说，下面是同一脚本的一个函数版本:

```py

#----------------------------------------------------------------------
def func(key):
    """"""
    return mydict[key]['CLAIMID']

sorted_keys = sorted(mydict.keys(), key=func)

for key in sorted_keys:
    print mydict[key]['CLAIMID']

```

为了好玩，让我们编写一个脚本，它可以根据嵌套字典中的任意键对嵌套字典进行排序。

```py

mydict = {'0d6f4012-16b4-4192-a854-fe9447b3f5cb': 
          {'CLAIMID': '123456789',
           'CLAIMDATE': '20120508', 
           'AMOUNT': '365.64', 'EXPDATE': '20120831'}, 
          'fe614868-d0c0-4c62-ae02-7737dea82dba': 
          {'CLAIMID': '45689654', 
           'CLAIMDATE': '20120508', 
           'AMOUNT': '185.55', 'EXPDATE': '20120831'}, 
          'ca1aa579-a9e7-4ade-80a3-0de8af4bcb21': 
          {'CLAIMID': '98754651',
           'CLAIMDATE': '20120508', 
           'AMOUNT': '93.00', 'EXPDATE': '20120831'},
          'ccb8641f-c1bd-45be-8f5e-e39b3be2e0e3': 
          {'CLAIMID': '789464321',
           'CLAIMDATE': '20120508', 'AMOUNT': '0.00',
           'EXPDATE': ''}, 
          'e1c445c2-5148-4a08-9b7e-ff5ed51c43ed': 
          {'CLAIMID': '897987945', 
           'CLAIMDATE': '20120508', 
           'AMOUNT': '62.66', 'EXPDATE': '20120831'}, 
          '77ad6dd4-5704-4060-9c38-6a93721ef98e': 
          {'CLAIMID': '23212315',
           'CLAIMDATE': '20120508', 
           'AMOUNT': '41.05', 'EXPDATE': '20120831'}
          }

outer_keys = mydict.keys()
print "outer keys:"
for outer_key in outer_keys:
    print outer_key

print "*" * 40
inner_keys = mydict[outer_key].keys()

for key in inner_keys:
    sorted_keys = sorted(mydict.keys(), key=lambda y: (mydict[y][key]))
    print "sorted by: " + key
    print sorted_keys
    for outer_key in sorted_keys:
        print mydict[outer_key][key]
    print "*" * 40
    print

```

这段代码可以工作，但是它没有给出我期望的结果。试着运行这个，你会注意到输出有点奇怪。排序是在字符串上进行的，所以所有看起来像数字的值都像字符串一样排序。哎呀！大多数人希望数字像数字一样排序，所以我们需要快速地将类似数字的值转换成整数或浮点数。下面是代码的最终版本(是的，有点马虎):

```py

mydict = {'0d6f4012-16b4-4192-a854-fe9447b3f5cb': 
          {'CLAIMID': '123456789',
           'CLAIMDATE': '20120508', 
           'AMOUNT': '365.64', 'EXPDATE': '20120831'}, 
          'fe614868-d0c0-4c62-ae02-7737dea82dba': 
          {'CLAIMID': '45689654', 
           'CLAIMDATE': '20120508', 
           'AMOUNT': '185.55', 'EXPDATE': '20120831'}, 
          'ca1aa579-a9e7-4ade-80a3-0de8af4bcb21': 
          {'CLAIMID': '98754651',
           'CLAIMDATE': '20120508', 
           'AMOUNT': '93.00', 'EXPDATE': '20120831'},
          'ccb8641f-c1bd-45be-8f5e-e39b3be2e0e3': 
          {'CLAIMID': '789464321',
           'CLAIMDATE': '20120508', 'AMOUNT': '0.00',
           'EXPDATE': ''}, 
          'e1c445c2-5148-4a08-9b7e-ff5ed51c43ed': 
          {'CLAIMID': '897987945', 
           'CLAIMDATE': '20120508', 
           'AMOUNT': '62.66', 'EXPDATE': '20120831'}, 
          '77ad6dd4-5704-4060-9c38-6a93721ef98e': 
          {'CLAIMID': '23212315',
           'CLAIMDATE': '20120508', 
           'AMOUNT': '41.05', 'EXPDATE': '20120831'}
          }

outer_keys = mydict.keys()
print "outer keys:"
for outer_key in outer_keys:
    print outer_key

print "*" * 40
inner_keys = mydict[outer_key].keys()

for outer_key in outer_keys:
    for inner_key in inner_keys:
        if mydict[outer_key][inner_key] == "":
            continue
        try:
            mydict[outer_key][inner_key] = int(mydict[outer_key][inner_key])
        except ValueError:
            mydict[outer_key][inner_key] = float(mydict[outer_key][inner_key])

for key in inner_keys:
    sorted_keys = sorted(mydict.keys(), key=lambda y: (mydict[y][key]))
    print "sorted by: " + key
    print sorted_keys
    for outer_key in sorted_keys:
        print mydict[outer_key][key]
    print "*" * 40
    print

```

所以现在我们用一种对人类感知更自然的方式对它进行了分类。现在还有一种方法可以做到这一点，那就是在将数据放入数据结构之前，按照我们想要的方式对数据进行排序。然而，只有当我们从 Python 2.7 开始使用来自**集合**模块的**ordered direct**时，这才会起作用。你可以在[官方文件](http://docs.python.org/library/collections.html#collections.OrderedDict)中读到。

现在你知道我对这个话题的了解了。我相信我的读者也会有其他的解决方案或方法。欢迎在评论中提及或链接到他们。

### 进一步阅读

*   [Python Lambda](https://www.blog.pythonlibrary.org/2010/07/19/the-python-lambda/)
*   [又一个 Lambda 教程](http://pythonconquerstheuniverse.wordpress.com/2011/08/29/lambda_tutorial/)