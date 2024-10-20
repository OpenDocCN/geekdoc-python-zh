# Python 3 -类型提示介绍

> 原文：<https://www.blog.pythonlibrary.org/2016/01/19/python-3-an-intro-to-type-hinting/>

Python 3.5 增加了一个有趣的新库，叫做 **typing** 。这给 Python 增加了类型提示。类型提示就是声明你的函数参数具有某种类型。但是，类型提示没有绑定。这只是一个提示，所以没有什么可以阻止程序员传递他们不应该传递的东西。这毕竟是 Python。你可以在 [PEP 484](https://www.python.org/dev/peps/pep-0484) 中阅读类型提示规范，或者你可以在 [PEP 483](https://www.python.org/dev/peps/pep-0483) 中阅读其背后的理论。

让我们看一个简单的例子:

```py

>>> def some_function(number: int, name: str) -> None:
    print("%s entered %s" % (name, number))

>>> some_function(13, 'Mike')
Mike entered 13

```

这意味着 **some_function** 需要两个参数，第一个是整数，第二个是字符串。还需要注意的是，我们已经暗示过这个函数不返回任何值。

让我们后退一点，用正常的方法写一个函数。然后我们将添加类型提示。在下面的例子中，我们有一个接受 list 和 name 的函数，在这个例子中是一个字符串。它所做的只是检查名字是否在列表中，并返回一个适当的布尔值。

```py

def process_data(my_list, name):
    if name in my_list:
        return True
    else:
        return False

if __name__ == '__main__':
    my_list = ['Mike', 'Nick', 'Toby']
    print( process_data(my_list, 'Mike') )
    print( process_data(my_list, 'John') )

```

现在让我们给这个函数添加类型提示:

```py

def process_data(my_list: list, name: str) -> bool:
    return name in my_list

if __name__ == '__main__':
    my_list = ['Mike', 'Nick', 'Toby']
    print( process_data(my_list, 'Mike') )
    print( process_data(my_list, 'John') )

```

在这段代码中，我们暗示第一个参数是一个列表，第二个参数是一个字符串，返回值是一个布尔值。

根据 PEP 484，“类型提示可以是内置类(包括标准库或第三方扩展模块中定义的类)、抽象基类、类型模块中可用的类型以及用户定义的类”。这意味着我们可以创建自己的类并添加一个提示。

```py

class Fruit:
    def __init__(self, name, color):
        self.name = name
        self.color = color

def salad(fruit_one: Fruit, fruit_two: Fruit) -> list:
    print(fruit_one.name)
    print(fruit_two.name)
    return [fruit_one, fruit_two]

if __name__ == '__main__':
    f = Fruit('orange', 'orange')
    f2 = Fruit('apple', 'red')
    salad(f, f2)

```

这里我们创建了一个简单的类，然后创建了一个函数，该函数需要该类的两个实例并返回一个列表对象。另一个我认为有趣的话题是你可以创建一个别名。这里有一个超级简单的例子:

```py

Animal = str

def zoo(animal: Animal, number: int) -> None:
    print("The zoo has %s %s" % (number, animal))

if __name__ == '__main__':
    zoo('Zebras', 10)

```

您可能已经猜到了，我们只是用变量**动物**作为**字符串**类型的别名。然后，我们使用动物别名向我们的函数添加了一个提示。

* * *

### 包扎

当我第一次听说类型暗示时，我很感兴趣。这是一个很好的概念，我肯定能看到它的用途。我想到的第一个用例就是自我记录你的代码。我工作过太多的代码库，在那里很难判断一个函数或类接受什么，虽然类型提示不强制任何东西，但它肯定会使一些模糊的代码变得清晰。如果一些 Python IDEs 添加了一个可选标志，可以检查代码的类型提示，并确保正确调用代码，那就太好了。

我强烈建议查看官方文档，因为那里有更多的信息。pep 也包含了很多好的细节。开心快乐编码！

* * *

### 相关阅读

*   关于[打字模块](https://docs.python.org/3/library/typing.html)的官方文档
*   PEP 0484 - [类型提示](https://www.python.org/dev/peps/pep-0484)