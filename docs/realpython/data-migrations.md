# 数据迁移

> 原文：<https://realpython.com/data-migrations/>

这是 Django 迁移系列的最后一篇文章:

*   第 1 部分: [Django 迁移-初级读本](https://realpython.com/django-migrations-a-primer/)
*   第 2 部分:[深入探讨迁移](https://realpython.com/digging-deeper-into-migrations/)
*   **第 3 部分:数据迁移(当前文章)**
*   视频: [Django 1.7 迁移-初级教程](https://realpython.com/django-migrations-a-primer/#video)

又回来了。

迁移主要是为了保持数据库的数据模型是最新的，但是数据库不仅仅是一个数据模型。最值得注意的是，它也是一个大的数据集合。因此，如果不讨论数据迁移，任何关于数据库迁移的讨论都是不完整的。

* * *

**2015 年 2 月 12 日更新**:更改数据迁移，从应用注册表中查找模型。

## 定义的数据迁移

数据迁移在很多情况下都会用到。两个非常受欢迎的是:

1.  当您希望加载应用程序成功运行所依赖的“系统数据”时。
2.  当对数据模型的更改迫使需要更改现有数据时。

> 请注意，加载虚拟数据进行测试不在上述列表中。您可以使用迁移来做到这一点，但是迁移通常在生产服务器上运行，所以您可能不希望在您的生产服务器上创建一堆虚拟测试数据。

[*Remove ads*](/account/join/)

## 示例

继续之前的 Django 项目，作为创建一些“系统数据”的例子，我们来创建一些历史比特币价格。Django migrations 将帮助我们解决这个问题，它创建一个空的迁移文件，如果我们键入:

```py
$ ./manage.py makemigrations --empty historical_data
```

这应该会创建一个名为`historical_data/migrations/003_auto<date_time_stamp>.py`的文件。我们把名字改成`003_load_historical_data.py`再开吧。您将得到一个默认结构，看起来像这样:

```py
# encoding: utf8
from django.db import models, migrations

class Migration(migrations.Migration):

    dependencies = [
        ('historical_data', '0002_auto_20140710_0810'),
    ]

    operations = [
    ]
```

您可以看到它为我们创建了一个基础结构，甚至插入了依赖项。那是有帮助的。现在要进行一些数据迁移，使用`RunPython`迁移操作:

```py
# encoding: utf8
from django.db import models, migrations
from datetime import date

def load_data(apps, schema_editor):
    PriceHistory = apps.get_model("historical_data", "PriceHistory")

    PriceHistory(date=date(2013,11,29),
         price=1234.00,
         volume=354564,
         total_btc=12054375,
         ).save()
    PriceHistory(date=date(2012,11,29),
         price=12.15,
         volume=187947,
         total_btc=10504650,
         ).save()

class Migration(migrations.Migration):

    dependencies = [
        ('historical_data', '0002_auto_20140710_0810'),
    ]

    operations = [
        migrations.RunPython(load_data)
    ]
```

我们从[定义](https://realpython.com/defining-your-own-python-function/)加载数据的函数`load_data`开始。

> 对于一个真正的应用程序，我们可能希望访问 blockchain.info 并获取历史价格的完整列表，但我们只是在那里放了几个来展示迁移是如何进行的。

一旦我们有了这个函数，我们就可以从我们的`RunPython`操作中调用它，然后当我们从命令行运行`./manage.py migrate`时，这个函数就会被执行。

注意这一行:

```py
PriceHistory = apps.get_model("historical_data", "PriceHistory")
```

运行迁移时，获得与您所处的迁移点相对应的`PriceHistory`模型版本非常重要。当您运行迁移时，您的模型(`PriceHistory`)可能会改变，例如，如果您在后续迁移中添加或删除了一个列。这可能会导致您的数据迁移失败，除非您使用上面的代码行来获得模型的正确版本。关于这一点的更多信息，请参见[评论这里](https://realpython.com/data-migrations/#comment-1843026722)。

可以说，这比运行`syncdb`并让它加载一个 fixture 要多得多。事实上，迁移并不尊重 fixturess 这意味着它们不会像`syncdb`那样自动为您加载 fixture。

这主要是哲学原因。

虽然您可以使用迁移来加载数据，但是它们主要是关于迁移数据和/或数据模型。我们展示了一个加载系统数据的示例，主要是因为它简单地解释了如何设置数据迁移，但通常情况下，数据迁移用于更复杂的操作，如转换数据以匹配新的数据模型。

例如，如果我们决定开始存储来自多个交易所而不是一个交易所的价格，那么我们可以添加像`price_gox`、`price_btc`等字段，然后我们可以使用迁移将所有数据从`price`列移动到`price_btc`列。

一般来说，在 Django 1.7 中处理迁移时，最好将加载数据看作是与迁移数据库分开的一个单独的练习。如果您确实想继续使用/加载装置，您可以使用如下命令:

```py
$ ./manage.py loaddata historical_data/fixtures/initial_data.json
```

这将把数据从夹具加载到数据库中。

这不会像数据迁移那样自动发生(这可能是件好事)，但是功能仍然存在；还没丢，有需要可以继续用固定物。不同之处在于，现在您可以在需要时用 fixtures 加载数据。如果您使用 fixtures 来为您的[单元测试](https://realpython.com/python-testing/)加载测试数据，这是需要记住的事情。

[*Remove ads*](/account/join/)

## 结论

这篇文章以及前两篇文章涵盖了使用迁移时最常见的场景。还有很多场景，如果你很好奇并且真的想深入研究迁移，最好的去处(除了代码本身)是[官方文档](https://docs.djangoproject.com/en/1.7/topics/migrations/)。

它是最新的，并且很好地解释了事物是如何工作的。如果你想看一个更复杂的例子，请在下面评论让我们知道。

请记住，在一般情况下，您面对的是以下两种情况之一:

1.  **模式迁移:**对数据库或表的结构进行更改，但不更改数据。这是最常见的类型，Django 通常可以自动为您创建这些迁移。

2.  **数据迁移:**更改数据，或者加载新数据。姜戈不能为你生成这些。必须使用`RunPython`迁移手动创建它们。

所以选择适合你的迁移，运行`makemigrations`,然后确保每次更新你的模型时更新你的迁移文件——差不多就是这样。这将允许您将迁移与代码一起存储在 git 中，并确保您可以更新数据库结构而不会丢失数据。

迁徙快乐！**