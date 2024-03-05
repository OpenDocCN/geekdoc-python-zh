# 深入挖掘 Django 移民

> 原文：<https://realpython.com/digging-deeper-into-migrations/>

这是 Django 迁移系列的第二篇文章:

*   第 1 部分: [Django 迁移:初级读本](https://realpython.com/django-migrations-a-primer/)
*   **第 2 部分:深入挖掘 Django 迁移(当前文章)**
*   第 3 部分:[数据迁移](https://realpython.com/data-migrations/)
*   视频: [Django 1.7 迁移-初级教程](https://realpython.com/django-migrations-a-primer/#video)

在本系列的前一篇文章中，您了解了 Django 迁移的目的。您已经熟悉了基本的使用模式，如创建和应用迁移。现在是时候更深入地挖掘迁移系统，看一看它的一些底层机制了。

到本文结束时，你会知道:

*   Django 是如何记录迁徙的
*   迁移如何知道要执行哪些数据库操作
*   如何定义迁移之间的依赖关系

一旦您理解了 Django 迁移系统的这一部分，您就为创建自己的定制迁移做好了充分的准备。让我们从停下来的地方开始吧！

本文使用了 [Django Migrations: A Primer](https://realpython.com/django-migrations-a-primer/) 中内置的`bitcoin_tracker` Django 项目。您可以通过阅读该文章来重新创建该项目，也可以下载源代码:

**下载源代码:** [单击此处下载您将在本文中使用的 Django 迁移项目的代码。](https://realpython.com/bonus/django-migrations/)

## Django 如何知道应用哪些迁移

让我们回顾一下本系列上一篇文章的最后一步。您创建了一个迁移，然后使用`python manage.py migrate`应用了所有可用的迁移。如果该命令成功运行，那么您的数据库表现在与您的模型定义相匹配。

如果再次运行该命令会发生什么？让我们试一试:

```py
$ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, contenttypes, historical_data, sessions
Running migrations:
 No migrations to apply.
```

什么都没发生！一旦一个迁移被应用到一个数据库，Django 就不会再将这个迁移应用到那个特定的数据库。确保迁移仅应用一次需要跟踪已应用的迁移。

Django 使用一个名为`django_migrations`的数据库表。第一次应用迁移时，Django 会在数据库中自动创建这个表。对于每个应用或伪造的迁移，都会在表中插入一个新行。

例如，在我们的`bitcoin_tracker`项目中，这个表格是这样的:

| 身份证明 | 应用 | 名字 | 应用的 |
| --- | --- | --- | --- |
| one | `contenttypes` | `0001_initial` | 2019-02-05 20:23:21.461496 |
| Two | `auth` | `0001_initial` | 2019-02-05 20:23:21.489948 |
| three | `admin` | `0001_initial` | 2019-02-05 20:23:21.508742 |
| four | `admin` | `0002_logentry_remove...` | 2019-02-05 20:23:21.531390 |
| five | `admin` | `0003_logentry_add_ac...` | 2019-02-05 20:23:21.564834 |
| six | `contenttypes` | `0002_remove_content_...` | 2019-02-05 20:23:21.597186 |
| seven | `auth` | `0002_alter_permissio...` | 2019-02-05 20:23:21.608705 |
| eight | `auth` | `0003_alter_user_emai...` | 2019-02-05 20:23:21.628441 |
| nine | `auth` | `0004_alter_user_user...` | 2019-02-05 20:23:21.646824 |
| Ten | `auth` | `0005_alter_user_last...` | 2019-02-05 20:23:21.661182 |
| Eleven | `auth` | `0006_require_content...` | 2019-02-05 20:23:21.663664 |
| Twelve | `auth` | `0007_alter_validator...` | 2019-02-05 20:23:21.679482 |
| Thirteen | `auth` | `0008_alter_user_user...` | 2019-02-05 20:23:21.699201 |
| Fourteen | `auth` | `0009_alter_user_last...` | 2019-02-05 20:23:21.718652 |
| Fifteen | `historical_data` | `0001_initial` | 2019-02-05 20:23:21.726000 |
| Sixteen | `sessions` | `0001_initial` | 2019-02-05 20:23:21.734611 |
| Nineteen | `historical_data` | `0002_switch_to_decimals` | 2019-02-05 20:30:11.337894 |

如您所见，每个应用的迁移都有一个条目。该表不仅包含从我们的`historical_data`应用程序的迁移，还包含从所有其他已安装应用程序的迁移。

下次运行迁移时，Django 将跳过数据库表中列出的迁移。这意味着，即使您手动更改了已经应用的迁移文件，Django 也会忽略这些更改，只要数据库中已经有它的条目。

您可以通过从表中删除相应的行来欺骗 Django 重新运行迁移，但是这并不是一个好主意，并且会给您留下一个损坏的迁移系统。

[*Remove ads*](/account/join/)

## 迁移文件

跑`python manage.py makemigrations <appname>`会怎么样？Django 寻找对你的应用程序`<appname>`中的模型所做的更改。如果它找到了，比如一个已经添加的模型，那么它会在`migrations`子目录中创建一个迁移文件。该迁移文件包含一个操作列表，用于使您的数据库模式与您的模型定义同步。

**注意:**你的 app 必须在`INSTALLED_APPS`设置中列出，并且必须包含一个`migrations`目录和一个`__init__.py`文件。否则 Django 不会为它创建任何迁移。

使用`startapp`管理命令创建新应用时会自动创建`migrations`目录，但手动创建应用时很容易忘记。

迁移文件只是 Python，我们来看看`historical_prices` app 中的第一个迁移文件。可以在`historical_prices/migrations/0001_initial.py`找到。它应该是这样的:

```py
from django.db import models, migrations

class Migration(migrations.Migration):
    dependencies = []
    operations = [
        migrations.CreateModel(
            name='PriceHistory',
            fields=[
                ('id', models.AutoField(
                    verbose_name='ID',
                    serialize=False,
                    primary_key=True,
                    auto_created=True)),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('price', models.DecimalField(decimal_places=2, max_digits=5)),
                ('volume', models.PositiveIntegerField()),
                ('total_btc', models.PositiveIntegerField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
```

如您所见，它包含一个名为`Migration`的类，该类继承自`django.db.migrations.Migration`。当您要求迁移框架应用迁移时，它将查找并执行这个类。

`Migration`类包含两个主要列表:

1.  `dependencies`
2.  `operations`

### 迁移操作

我们先来看一下`operations`榜单。此表包含迁移过程中要执行的操作。操作是类`django.db.migrations.operations.base.Operation`的子类。以下是 Django 中内置的常见操作:

| 操作类 | 描述 |
| --- | --- |
| `CreateModel` | 创建新模型和相应的数据库表 |
| `DeleteModel` | 删除模型并删除其数据库表 |
| `RenameModel` | 重命名模型并重命名其数据库表 |
| `AlterModelTable` | 重命名模型的数据库表 |
| `AlterUniqueTogether` | 更改模型的唯一约束 |
| `AlterIndexTogether` | 更改模型的索引 |
| `AlterOrderWithRespectTo` | 创建或删除模型的`_order`列 |
| `AlterModelOptions` | 在不影响数据库的情况下更改各种模型选项 |
| `AlterModelManagers` | 更改迁移期间可用的管理器 |
| `AddField` | 向模型和数据库中的相应列添加字段 |
| `RemoveField` | 从模型中删除字段，并从数据库中删除相应的列 |
| `AlterField` | 更改字段的定义，并在必要时改变其数据库列 |
| `RenameField` | 重命名字段，如有必要，还重命名其数据库列 |
| `AddIndex` | 在数据库表中为模型创建索引 |
| `RemoveIndex` | 从模型的数据库表中删除索引 |

请注意操作是如何根据对模型定义所做的更改来命名的，而不是在数据库上执行的操作。当您应用迁移时，每个操作负责为您的特定数据库生成必要的 SQL 语句。例如，`CreateModel`将生成一个`CREATE TABLE` SQL 语句。

开箱即用，迁移支持 Django 支持的所有标准数据库。因此，如果您坚持使用这里列出的操作，那么您可以对您的模型做或多或少的任何更改，而不必担心底层的 SQL。这都是为你做的。

**注意:**在某些情况下，Django 可能无法正确检测到您的更改。如果您重命名一个模型并更改它的几个字段，那么 Django 可能会将其误认为是一个新模型。

它将创建一个`DeleteModel`和一个`CreateModel`操作，而不是一个`RenameModel`和几个`AlterField`操作。它不会重命名模型的数据库表，而是将它删除，并用新名称创建一个新表，实际上删除了所有数据！

在生产数据上运行迁移之前，要养成检查生成的迁移并在数据库副本上测试它们的习惯。

Django 为高级用例提供了另外三个操作类:

1.  **`RunSQL`** 允许你在数据库中运行自定义 SQL。
2.  **`RunPython`** 允许你运行任何 Python 代码。
3.  **`SeparateDatabaseAndState`** 是针对高级用途的专门操作。

通过这些操作，您基本上可以对数据库进行任何想要的更改。然而，您不会在使用`makemigrations`管理命令自动创建的迁移中找到这些操作。

从 Django 2.0 开始，`django.contrib.postgres.operations`中也有一些 PostgreSQL 特有的操作，可以用来安装各种 PostgreSQL 扩展:

*   `BtreeGinExtension`
*   `BtreeGistExtension`
*   `CITextExtension`
*   `CryptoExtension`
*   `HStoreExtension`
*   `TrigramExtension`
*   `UnaccentExtension`

请注意，包含这些操作之一的迁移需要具有超级用户权限的数据库用户。

最后但同样重要的是，您还可以创建自己的操作类。如果您想深入了解这一点，那么请看一下关于创建定制迁移操作的 [Django 文档。](https://docs.djangoproject.com/en/2.1/ref/migration-operations/#writing-your-own)

[*Remove ads*](/account/join/)

### 迁移依赖关系

迁移类中的`dependencies`列表包含在应用该迁移之前必须应用的任何迁移。

在上面看到的`0001_initial.py`迁移中，不需要事先应用任何东西，因此没有依赖关系。我们来看看`historical_prices` app 中的第二次迁移。在文件`0002_switch_to_decimals.py`中，`Migration`的`dependencies`属性有一个条目:

```py
from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('historical_data', '0001_initial'),
    ]
    operations = [
        migrations.AlterField(
            model_name='pricehistory',
            name='volume',
            field=models.DecimalField(decimal_places=3, max_digits=7),
        ),
    ]
```

上面的依赖关系表示应用程序`historical_data`的迁移`0001_initial`必须首先运行。这是有意义的，因为迁移`0001_initial`创建了包含迁移`0002_switch_to_decimals`想要改变的字段的表。

迁移也可能依赖于另一个应用程序的迁移，如下所示:

```py
class Migration(migrations.Migration):
    ...

    dependencies = [
        ('auth', '0009_alter_user_last_name_max_length'),
    ]
```

如果一个模型有一个外键指向另一个应用程序中的模型，这通常是必要的。

或者，您也可以使用属性`run_before`强制一个迁移在另一个迁移之前*运行:*

```py
class Migration(migrations.Migration):
    ...

    run_before = [
        ('third_party_app', '0001_initial'),
    ]
```

依赖关系也可以合并，这样你就可以拥有多个依赖关系。这个功能提供了很大的灵活性，因为您可以容纳依赖于不同应用程序模型的外键。

明确定义迁移之间依赖关系的选项也意味着迁移的编号(通常是`0001`、`0002`、`0003`、…)并不严格代表应用迁移的顺序。您可以根据需要添加任何依赖项，从而控制顺序，而不必对所有迁移重新编号。

### 查看迁移

您通常不必担心迁移生成的 SQL。但是，如果您想仔细检查生成的 SQL 是否有意义，或者只是好奇它看起来像什么，那么 Django 会为您提供`sqlmigrate`管理命令:

```py
$ python manage.py sqlmigrate historical_data 0001
BEGIN;
--
-- Create model PriceHistory
--
CREATE TABLE "historical_data_pricehistory" (
 "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
 "date" datetime NOT NULL,
 "price" decimal NOT NULL,
 "volume" integer unsigned NOT NULL
);
COMMIT;
```

这样做将根据您的`settings.py`文件中的数据库，列出由指定迁移生成的底层 SQL 查询。当您传递参数`--backwards`时，Django 生成 SQL 来取消迁移:

```py
$ python manage.py sqlmigrate --backwards historical_data 0001
BEGIN;
--
-- Create model PriceHistory
--
DROP TABLE "historical_data_pricehistory";
COMMIT;
```

一旦您看到稍微复杂一点的迁移的`sqlmigrate`的输出，您可能会意识到您不必手工制作所有这些 SQL！

## Django 如何检测模型的变化

您已经看到了迁移文件的样子，以及它的`Operation`类列表如何定义对数据库执行的更改。但是 Django 怎么知道哪些操作应该放入迁移文件呢？您可能期望 Django 将您的模型与您的数据库模式进行比较，但事实并非如此。

当运行`makemigrations`时，Django 不*也不*检查你的数据库。它也不会将您的模型文件与早期版本进行比较。取而代之的是，Django 检查了所有已经应用的迁移，并构建了模型应该是什么样子的项目状态。然后，将这个项目状态与您当前的模型定义进行比较，并创建一个操作列表，当应用该列表时，将使项目状态与模型定义保持一致。

[*Remove ads*](/account/join/)

### 和姜戈下棋

你可以把你的模型想象成棋盘，Django 是一个国际象棋大师，看着你和自己对弈。但是大师不会监视你的一举一动。大师只在你喊`makemigrations`的时候看棋盘。

因为只有有限的一组可能的走法(而特级大师是特级大师)，她可以想出自她上次看棋盘以来发生的走法。她做了一些笔记，让你玩，直到你再次大喊`makemigrations`。

当下一次看棋盘时，特级大师不记得上一次棋盘是什么样子的，但她可以浏览她以前移动的笔记，并建立棋盘样子的心理模型。

现在，当你喊`migrate`时，特级大师将在另一个棋盘上重放所有记录的移动，并在电子表格中记录她的哪些记录已经被应用。第二个棋盘是您的数据库，电子表格是`django_migrations`表。

这个类比非常恰当，因为它很好地说明了 Django 迁移的一些行为:

*   **Django 迁移努力做到高效:**就像特级大师假设你走的步数最少一样，Django 会努力创造最高效的迁移。如果您向模型中添加一个名为`A`的字段，然后将其重命名为`B`，然后运行`makemigrations`，那么 Django 将创建一个新的迁移来添加一个名为`B`的字段。

*   姜戈的迁移有其局限性:如果你在让特级大师看棋盘之前走了很多步，那么她可能无法追溯每一步的准确移动。类似地，如果您一次进行太多的更改，Django 可能无法实现正确的迁移。

*   **Django migration 希望你遵守游戏规则:**当你做任何意想不到的事情时，比如从棋盘上随便拿走一个棋子或者弄乱音符，大师一开始可能不会注意到，但迟早她会放弃并拒绝继续。当您处理`django_migrations`表或者在迁移之外更改您的数据库模式时，也会发生同样的情况，例如删除模型的数据库表。

### 理解`SeparateDatabaseAndState`

现在您已经了解了 Django 构建的项目状态，是时候仔细看看操作`SeparateDatabaseAndState`了。这个操作可以做到顾名思义:它可以将项目状态(Django 构建的心智模型)从数据库中分离出来。

`SeparateDatabaseAndState`用两个操作列表实例化:

1.  **`state_operations`** 包含只适用于项目状态的操作。
2.  **`database_operations`** 包含只应用于数据库的操作。

此操作允许您对数据库进行任何类型的更改，但是您有责任确保项目状态在之后适合数据库。`SeparateDatabaseAndState`的示例用例是将模型从一个应用程序移动到另一个应用程序，或者[在不停机的情况下在大型数据库上创建索引](https://realpython.com/create-django-index-without-downtime/)。

`SeparateDatabaseAndState`是一项高级操作，您不需要在第一天就进行迁移，也许根本不需要。`SeparateDatabaseAndState`类似于心脏手术。这有相当大的风险，不是你为了好玩而做的事情，但有时这是让病人活下去的必要程序。

## 结论

您对 Django 迁移的深入研究到此结束。恭喜你！您已经讨论了相当多的高级主题，现在已经对迁移的本质有了深入的了解。

你学到了:

*   Django 在 Django 迁移表中跟踪应用的迁移。
*   Django 迁移由包含一个`Migration`类的普通 Python 文件组成。
*   Django 知道要从`Migration`类的`operations`列表中执行哪些更改。
*   Django 将您的模型与它从迁移中构建的项目状态进行比较。

有了这些知识，您就可以开始学习 Django 迁移系列的第三部分了，在这里您将学习如何使用数据迁移来安全地对数据进行一次性更改。敬请期待！

本文使用了 [Django Migrations: A Primer](https://realpython.com/django-migrations-a-primer/) 中内置的`bitcoin_tracker` Django 项目。您可以通过阅读该文章来重新创建该项目，也可以下载源代码:

**下载源代码:** [单击此处下载您将在本文中使用的 Django 迁移项目的代码。](https://realpython.com/bonus/django-migrations/)***