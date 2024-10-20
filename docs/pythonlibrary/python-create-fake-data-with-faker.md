# Python:用 Faker 创建假数据

> 原文：<https://www.blog.pythonlibrary.org/2014/06/18/python-create-fake-data-with-faker/>

偶尔，我会遇到需要虚拟数据来测试代码的情况。如果您需要对一个新的数据库或表进行测试，您将经常遇到对虚拟数据的需求。最近偶然发现一个有趣的包，叫做 [Faker](http://www.joke2k.net/faker/) 。Faker 的唯一目的是创建半随机的假数据。Faker 可以创建假的名字，地址，浏览器用户代理，域名，段落等等。在本文中，我们将花一些时间展示 Faker 的一些功能。

* * *

### 入门指南

首先，你需要安装 Faker。如果你有皮普(为什么你不会？)，你需要做的就是这个:

```py

pip install fake-factory

```

现在您已经安装了这个包，我们可以开始使用它了！

* * *

### 制造假数据

用 Faker 创建假数据真的很好做。我们来看几个例子。我们将从几个创造假名字的例子开始:

```py

from faker import Factory

#----------------------------------------------------------------------
def create_names(fake):
    """"""
    for i in range(10):
        print fake.name()

if __name__ == "__main__":
    fake = Factory.create()
    create_names(fake)

```

如果您运行上面的代码，您将看到 10 个不同的名字被打印到 stdout。这是我运行它时得到的结果:

```py

Mrs. Terese Walter MD
Jess Mayert
Ms. Katerina Fisher PhD
Mrs. Senora Purdy PhD
Gretchen Tromp
Winnie Goodwin
Yuridia McGlynn MD
Betty Kub
Nolen Koelpin
Adilene Jerde

```

你可能会收到一些不同的东西。每次我运行这个脚本，结果都不一样。大多数情况下，我不希望名字有前缀或后缀，所以我创建了另一个只产生名和姓的脚本:

```py

from faker import Factory

#----------------------------------------------------------------------
def create_names2(fake):
    """"""
    for i in range(10):
        name = "%s %s" % (fake.first_name(),
                          fake.last_name())
        print name

if __name__ == "__main__":
    fake = Factory.create()
    create_names2(fake)

```

如果您运行第二个脚本，您看到的姓名不应该包含前缀(如女士、先生等)或后缀(如 PhD、Jr .等)。让我们来看看我们可以用这个包生成的一些其他类型的假数据。

* * *

### 创造其他虚假的东西

现在，我们将花一些时间了解 Faker 可以生成的其他一些假数据。下面这段代码将创建六个假数据。让我们来看看:

```py

from faker import Factory

#----------------------------------------------------------------------
def create_fake_stuff(fake):
    """"""
    stuff = ["email", "bs", "address",
             "city", "state",
             "paragraph"]
    for item in stuff:
        print "%s = %s" % (item, getattr(fake, item)())

if __name__ == "__main__":
    fake = Factory.create()
    create_fake_stuff(fake)

```

这里我们使用 Python 内置的 **getattr** 函数来调用 Faker 的一些方法。当我运行这个脚本时，我收到了以下输出:

```py

email = pacocha.aria@kris.com
bs = reinvent collaborative systems
address = 57188 Leuschke Mission
Lake Jaceystad, KY 46291
city = West Luvinialand
state = Oregon
paragraph = Possimus nostrum exercitationem harum eum in. Dicta aut officiis qui deserunt voluptas ullam ut. Laborum molestias voluptatem consequatur laboriosam. Omnis est cumque culpa quo illum.

```

那不是很有趣吗？

* * *

### 包扎

Faker 包还有很多其他的方法，这里没有介绍。你应该看看他们完整的[文档](http://www.joke2k.net/faker/)，看看你还能用这个包做些什么。只需做一点工作，您就可以使用这个包轻松地填充数据库或报告。