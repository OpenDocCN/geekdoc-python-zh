# Django 的新 Postgres 功能带来的乐趣

> 原文：<https://realpython.com/fun-with-djangos-new-postgres-features/>

**这篇博文讲述了如何使用 Django 1.8 中引入的新的 [PostgreSQL 特有的模型字段](https://docs.djangoproject.com/en/1.8/ref/contrib/postgres/)——数组字段、HStoreField 和范围字段。**

*这篇文章献给由[马克·塔姆林](https://twitter.com/mjtamlyn)组织的 [Kickstarter](https://www.kickstarter.com/projects/mjtamlyn/improved-postgresql-support-in-django/posts/803919) 活动的令人敬畏的支持者，真正的 playa 让它发生了。*

## Playaz 俱乐部？

因为我是一个超级极客，而且没有机会进入一个真正的 Playaz 俱乐部(而且因为在那个时候 [4 Tay](https://www.youtube.com/watch?v=2daXghqHgjQ) 是个炸弹)，我决定建立我自己的虚拟在线 Playaz 俱乐部。那到底是什么？一个私人的、只接受邀请的社交网络，面向一小群志同道合的人。

在这篇文章中，我们将关注用户模型，并探索 Django 的新 [PostgreSQL 特性](https://docs.djangoproject.com/en/1.8/ref/contrib/postgres/fields/)如何支持建模。我们所指的新特性是 PostgreSQL 独有的，所以除非您的数据库`ENGINE`等于`django.db.backends.postgresql_psycopg2`，否则不要尝试。你需要`psycopg2`的版本> = 2.5。好孩子，我们开始吧。

如果你和我在一起，你好！:)

[*Remove ads*](/account/join/)

## 塑造 Playa 的形象

每个 playa 都有一个代表，他们希望全世界都知道他们的代表。所以让我们创建一个用户档案(也称为“代表”)，让我们的每个 playaz 都能表达他们的个性。

以下是 playaz 代表的基本模型:

```py
from django.db import models
from django.contrib.auth.models import User

class Rep(models.Model):
    playa = models.OneToOneField(User)
    hood = models.CharField(max_length=100)
    area_code = models.IntegerField()
```

上面 1.8 没什么特别的。只是一个扩展 Django 用户的标准模型，因为 playa 仍然需要用户名和电子邮件地址，对吗？另外，我们添加了两个新字段来存储 playaz hood 和区号。

## 资金和测距仪

对于一个玩家来说，仅仅戴上兜帽是不够的。Playaz 经常喜欢炫耀他们的资金，但同时又不想让人知道资金的确切数额。我们可以用一个新的 [Postgres 值域](https://docs.djangoproject.com/en/1.8/ref/contrib/postgres/fields/#range-fields)来建模。当然，我们将使用`BigIntegerRangeField`来更好地模拟海量数字，对吗？

```py
bankroll = pgfields.BigIntegerRangeField(default=(10, 100))
```

范围字段基于 [psycopg2 范围对象](http://initd.org/psycopg/docs/extras.html#adapt-range)，可用于数字和日期范围。将资金字段迁移到数据库后，我们可以通过向其传递一个范围对象来与范围字段进行交互，因此创建我们的第一个 playa 将如下所示:

>>>

```py
>>> from playa.models import Rep
>>> from django.contrib.auth.models import User
>>> calvin = User.objects.create_user(username="snoop", password="dogg")
>>> calvins_rep = Rep(hood="Long Beach", area_code=213)
>>> calvins_rep.bankroll = (100000000, 150000000)
>>> calvins_rep.playa = calvin
>>> calvins_rep.save()
```

注意这一行:`calvins_rep.bankroll = (100000000, 150000000)`。这里我们使用一个简单的元组来设置一个范围字段。也可以使用`NumericRange`对象来设置值，如下所示:

```py
from psycopg2.extras import NumericRange
br = NumericRange(lower=100000000, upper=150000000)
calvin.rep.bankroll = br
calvin.rep.save()
```

这与使用元组本质上是一样的。然而，了解`NumericRange`对象很重要，因为它用于过滤模型。例如，如果我们想找到所有资金大于 5000 万的玩家(意味着整个资金范围大于 5000 万):

```py
Rep.objects.filter(bankroll__fully_gt=NumericRange(50000000, 50000000))
```

这将返回这些 playas 的列表。或者，如果我们想找到资金“在 1000 万到 1500 万之间”的所有玩家，我们可以使用:

```py
Rep.objects.filter(bankroll__overlap=NumericRange(10000000, 15000000))
```

这将返回资金范围至少在 1000 万到 1500 万之间的所有玩家。一个更绝对的查询是资金完全在一个范围内的所有玩家，即每个人至少赚 1000 万但不超过 1500 万。该查询类似于:

```py
Rep.objects.filter(bankroll__contained_by=NumericRange(10000000, 15000000))
```

关于基于范围的查询的更多信息可以在[这里](https://docs.djangoproject.com/en/1.8/ref/contrib/postgres/fields/#querying-range-fields)找到。

## skillz as array file

这不全是资金的问题，playaz 有 skillz，各种 skillz。让我们用一个[数组字段](https://docs.djangoproject.com/en/1.8/ref/contrib/postgres/fields/#arrayfield)来建模。

```py
skillz = pgfields.ArrayField(
    models.CharField(max_length=100, blank=True),
    blank = True,
    null = True,
)
```

要声明`ArrayField`,我们必须给它一个第一参数，也就是基字段。与 Python 列表不同，ArrayFields 必须将列表中的每个元素声明为相同的类型。Basefield 声明这是哪种类型，它可以是任何标准的模型字段类型。在上面的例子中，我们刚刚使用了一个`CharField`作为我们的基本类型，这意味着`skillz`将是一个字符串数组。

将值存储到`ArrayField`正如您所期望的那样:

>>>

```py
>>> from django.contrib.auth.models import User
>>> calvin = User.objects.get(username='snoop')
>>> calvin.rep.skillz = ['ballin', 'rappin', 'talk show host', 'merchandizn']
>>> calvin.rep.save()
```

[*Remove ads*](/account/join/)

### 斯奇尔兹寻找玩法

如果我们需要一个有特殊技能的球员，我们怎么找到他们？使用`__contains`过滤器:

```py
Rep.objects.filter(skillz__contains=['rappin'])
```

对于拥有任何一项技能['说唱'，' djing '，'制作']但没有其他技能的玩家，您可以执行如下查询:

```py
Rep.objects.filter(skillz__contained_by=['rappin', 'djing', 'producing'])
```

或者，如果您想找到具有某一特定技能列表的任何人:

```py
Rep.objects.filter(skillz__overlap=['rappin', 'djing', 'producing'])
```

你甚至可以找到那些把一项技能列为第一技能的人(因为每个人都把自己最擅长的技能列在第一位):

```py
Rep.objects.filter(skillz__0='ballin')
```

## 游戏 as HStore

游戏可以被认为是玩家可能拥有的各种随机技能的列表。由于游戏跨越了各种各样的东西，让我们把它建模为一个 [HStore 字段](https://docs.djangoproject.com/en/1.8/ref/contrib/postgres/fields/#hstorefield)，这基本上意味着我们可以把任何旧的 [Python 字典](https://realpython.com/python-dicts/)放在那里:

```py
game = pgfields.HStoreField()
```

> 花点时间想想我们刚刚做了什么。HStore 相当大。它基本上允许“NoSQL”类型的数据存储，就在 postgreSQL 内部。另外，由于它在 PostgreSQL 内部，我们可以链接(通过外键)包含 NoSQL 数据的表和存储常规 SQL 类型数据的表。您甚至可以将两者存储在同一个表中的不同列上，就像我们在这里所做的那样。也许玩家们不需要再使用那个只会说废话的 MongoDB 了…

回到实现细节，如果您[试图将新的 HStore 字段迁移到数据库中，并以这个错误结束-](https://realpython.com/django-migrations-a-primer/)

```py
django.db.utils.ProgrammingError: type "hstore" does not exist
```

-那么你的 PostgreSQL 数据库是 8.1 之前(升级时间，playa)或者没有安装 HStore 扩展。请记住，在 PostgreSQL 中，HStore 扩展是针对每个数据库安装的，而不是系统范围的。要从 psql 提示符安装它，请运行以下 sql:

```py
CREATE EXTENSION hstore
```

或者，如果您愿意，可以使用以下迁移文件通过 SQL 迁移来完成(假设您以超级用户身份连接到数据库):

```py
from django.db import models, migrations

class Migration(migrations.Migration):

    dependencies = []

    operations = [
        migrations.RunSQL("CREATE EXTENSION IF NOT EXISTS hstore")
    ]
```

最后，您还需要确保您已经将`'django.contrib.postgres'`添加到`'settings.INSTALLED_APPS'`中，以利用 HStore 字段。

通过这种设置，我们可以像这样用字典向我们的`HStoreField` `game`添加数据:

>>>

```py
>>> calvin = User.objects.get(username="snoop")
>>> calvin.rep.game = {'best_album': 'Doggy Style', 'youtube-channel': \
 'https://www.youtube.com/user/westfesttv', 'twitter_follows' : '11000000'}
>>> calvin.rep.save()
```

请记住，字典必须只对所有的键和值使用字符串。

现在来看一些更有趣的例子…

[*Remove ads*](/account/join/)

## Propz

让我们编写一个“显示游戏”函数来搜索 playaz 游戏，并返回匹配的 playaz 列表。用极客的话来说，我们在 HStore 字段中搜索传递给函数的任何键。它看起来像这样:

```py
def show_game(key):
    return Rep.Objects.filter(game__has_key=key).values('game','playa__username')
```

上面我们已经为 HStore 字段使用了`has_key`过滤器来返回一个 queryset，然后使用 values 函数进一步过滤它(主要是为了说明您可以将`django.contrib.postgres`内容与常规查询集内容链接起来)。

返回值将是一个字典列表:

```py
[
  {'playa__username': 'snoop',
  'game': {'twitter_follows': '11000000',
           'youtube-channel': 'https://www.youtube.com/user/westfesttv',
           'best_album': 'Doggy Style'
        }
  }
]
```

正如他们所说，[游戏识别游戏](http://www.urbandictionary.com/define.php?term=Game+recognizes+game)，现在我们也可以搜索游戏了。

## 豪赌客

如果我们相信 playaz 告诉我们的关于他们资金的信息，那么我们可以用它来对他们进行分类(因为这是一个范围)。让我们根据以下级别的资金添加一个 Playa 排名:

*   年轻的小伙子-资金不足 10 万美元

*   balla——通过“ballin”技能获得 100，000 到 500，000 英镑的资金

*   playa–用两个 skillz 和一些游戏赢得 500，000 到 1，000，000 美元的资金

*   高注玩家–资金超过 100 万英镑

*   有“黑帮”技能和“老派”游戏钥匙

balla 的查询如下。这将是严格的解释，它将只返回那些其整个资金范围在指定限制内的人:

```py
Rep.objects.filter(bankroll__contained_by=[100000, 500000], skillz__contains=['ballin'])
```

自己尝试[休息](https://docs.djangoproject.com/en/1.8/ref/contrib/postgres/)进行一些练习。如果你需要帮助，[阅读文件](https://docs.djangoproject.com/en/1.8/ref/contrib/postgres/fields/#postgresql-specific-model-fields)。***