# 如何将 Django 模型移动到另一个应用程序

> 原文：<https://realpython.com/move-django-model/>

如果你曾经想过重构你的 Django 应用，那么你可能会发现自己需要移动一个 Django 模型。使用 Django **迁移**将 Django 模型从一个应用程序迁移到另一个应用程序有几种方法，但不幸的是，没有一种方法是直接的。

在 Django 应用程序之间移动模型通常是一项非常复杂的任务，包括复制数据、改变约束和重命名对象。由于这些复杂性，Django **对象关系映射器(ORM)** 没有提供可以检测和自动化整个过程的内置迁移操作。相反，ORM 提供了一组底层迁移操作，允许 Django 开发人员在迁移框架中自己实现过程。

在本教程中，您将学习:

*   如何将 Django 模型从一个应用程序移动到另一个应用程序
*   如何使用 Django 迁移命令行界面(CLI)的**高级功能**，如`sqlmigrate`、`showmigrations`、`sqlsequencereset`
*   如何制定和检查**迁移计划**
*   如何使迁移可逆以及如何**逆转迁移**
*   什么是内省以及 Django 如何在迁移中使用它

完成本教程后，您将能够根据您的具体用例选择将 Django 模型从一个应用程序迁移到另一个应用程序的最佳方法。

**免费奖励:** [点击此处获取免费的 Django 学习资源指南(PDF)](#) ，该指南向您展示了构建 Python + Django web 应用程序时要避免的技巧和窍门以及常见的陷阱。

## 示例案例:将 Django 模型移动到另一个应用程序

在本教程中，您将使用商店应用程序。你的商店将从两个 Django 应用开始:

1.  **`catalog`** :这个应用是用来存储产品和产品类别的数据。
2.  **`sale`** :这个 app 是用来记录和跟踪产品销售的。

完成这两个应用程序的设置后，您将把一个名为`Product`的 Django 模型转移到一个名为`product`的新应用程序中。在此过程中，您将面临以下挑战:

*   被移动的模型与其他模型有外键关系。
*   其他模型与被移动的模型有外键关系。
*   被移动的模型在其中一个字段上有一个索引(除了主键之外)。

这些挑战受到现实生活中重构过程的启发。在克服了这些困难之后，您就可以为您的特定用例计划一个类似的迁移过程了。

[*Remove ads*](/account/join/)

## 设置:准备您的环境

在您开始移动东西之前，您需要[设置项目的初始状态](https://realpython.com/django-setup/)。本教程使用运行在 Python 3.8 上的 Django 3，但是您可以在其他版本中使用类似的技术。

### 建立一个 Python 虚拟环境

首先，在新目录中创建虚拟环境:

```py
$ mkdir django-move-model-experiment
$ cd django-move-model-experiment
$ python -m venv venv
```

关于创建虚拟环境的逐步说明，请查看 [Python 虚拟环境:初级教程](https://realpython.com/python-virtual-environments-a-primer/)。

### 创建 Django 项目

在您的终端中，激活虚拟环境并安装 Django:

```py
$ source venv/bin/activate
$ pip install django
Collecting django
Collecting pytz (from django)
Collecting asgiref~=3.2 (from django)
Collecting sqlparse>=0.2.2 (from django)
Installing collected packages: pytz, asgiref, sqlparse, django
Successfully installed asgiref-3.2.3 django-3.0.4 pytz-2019.3 sqlparse-0.3.1
```

现在您已经准备好创建您的 Django 项目了。使用`django-admin startproject`创建一个名为`django-move-model-experiment`的项目:

```py
$ django-admin startproject django-move-model-experiment
$ cd django-move-model-experiment
```

运行这个命令后，您会看到 Django 创建了新的文件和目录。关于如何开始一个新的 Django 项目，请查看[开始一个 Django 项目](https://realpython.com/django-setup/)。

### 创建 Django 应用程序

现在你有了一个新的 Django 项目，用你商店的产品目录创建一个应用程序:

```py
$ python manage.py startapp catalog
```

接下来，将以下型号添加到新的`catalog`应用程序中:

```py
# catalog/models.py
from django.db import models

class Category(models.Model):
    name = models.CharField(max_length=100)

class Product(models.Model):
    name = models.CharField(max_length=100, db_index=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
```

您已经在您的`catalog`应用中成功创建了`Category`和`Product`模型。现在你有了目录，你想开始销售你的产品。为销售创建另一个应用程序:

```py
$ python manage.py startapp sale
```

将以下`Sale`型号添加到新的`sale`应用程序中:

```py
# sale/models.py
from django.db import models

from catalog.models import Product

class Sale(models.Model):
    created = models.DateTimeField()
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
```

注意，`Sale`模型使用 [`ForeignKey`](https://docs.djangoproject.com/en/3.0/ref/models/fields/#django.db.models.ForeignKey) 引用了`Product`模型。

[*Remove ads*](/account/join/)

### 生成并应用初始迁移

要完成设置，生成 [**迁移**](https://docs.djangoproject.com/en/3.0/topics/migrations/) 并应用它们:

```py
$ python manage.py makemigrations catalog sale
Migrations for 'catalog':
 catalog/migrations/0001_initial.py
 - Create model Category
 - Create model Product
Migrations for 'sale':
 sale/migrations/0001_initial.py
 - Create model Sale

$ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, catalog, contenttypes, sale, sessions
Running migrations:
 Applying contenttypes.0001_initial... OK
 Applying auth.0001_initial... OK
 Applying admin.0001_initial... OK
 Applying admin.0002_logentry_remove_auto_add... OK
 Applying admin.0003_logentry_add_action_flag_choices... OK
 Applying contenttypes.0002_remove_content_type_name... OK
 Applying auth.0002_alter_permission_name_max_length... OK
 Applying auth.0003_alter_user_email_max_length... OK
 Applying auth.0004_alter_user_username_opts... OK
 Applying auth.0005_alter_user_last_login_null... OK
 Applying auth.0006_require_contenttypes_0002... OK
 Applying auth.0007_alter_validators_add_error_messages... OK
 Applying auth.0008_alter_user_username_max_length... OK
 Applying auth.0009_alter_user_last_name_max_length... OK
 Applying auth.0010_alter_group_name_max_length... OK
 Applying auth.0011_update_proxy_permissions... OK
 Applying catalog.0001_initial... OK
 Applying sale.0001_initial... OK
 Applying sessions.0001_initial... OK
```

关于 Django 迁移的更多信息，请查看 [Django 迁移:初级读本](https://realpython.com/django-migrations-a-primer/)。迁移就绪后，现在就可以创建一些示例数据了！

### 生成样本数据

为了使迁移场景尽可能真实，从终端窗口激活 [Django shell](https://docs.djangoproject.com/en/3.0/ref/django-admin/#shell) :

```py
$ python manage.py shell
```

接下来，创建以下对象:

>>>

```py
>>> from catalog.models import Category, Product
>>> clothes = Category.objects.create(name='Clothes')
>>> shoes = Category.objects.create(name='Shoes')
>>> Product.objects.create(name='Pants', category=clothes)
>>> Product.objects.create(name='Shirt', category=clothes)
>>> Product.objects.create(name='Boots', category=shoes)
```

您创建了两个类别，`'Shoes'`和`'Clothes'`。接下来，您向`'Clothes'`类别添加了两个产品`'Pants'`和`'Shirt'`，向`'Shoes'`类别添加了一个产品`'Boots'`。

恭喜你！您已经完成了项目初始状态的设置。在现实生活中，这是您开始规划重构的地方。本教程中介绍的三种方法都将从这一点开始。

## 漫长的道路:将数据复制到一个新的 Django 模型中

首先，你要走很长的路:

1.  创建新模型
2.  将数据复制到其中
3.  扔掉旧桌子

这种方法有一些你应该知道的陷阱。您将在接下来的小节中详细探索它们。

### 创建新模型

首先创建一个新的`product`应用程序。从您的终端执行以下命令:

```py
$ python manage.py startapp product
```

运行这个命令后，您会注意到一个名为`product`的新目录被添加到项目中。

要将新应用程序注册到您现有的 Django 项目中，请将其添加到 Django 的`settings.py`中的`INSTALLED_APPS`列表中:

```py
--- a/store/store/settings.py +++ b/store/store/settings.py @@ -40,6 +40,7 @@ INSTALLED_APPS = [ 'catalog', 'sale', +    'product', ] MIDDLEWARE = [
```

您的新`product`应用程序现已在 Django 注册。接下来，在新的`product`应用程序中创建一个`Product`模型。您可以从`catalog`应用程序中复制代码:

```py
# product/models.py
from django.db import models

from catalog.models import Category

class Product(models.Model):
    name = models.CharField(max_length=100, db_index=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
```

现在您已经定义了模型，试着为它生成迁移:

```py
$ python manage.py makemigrations product
SystemCheckError: System check identified some issues:

ERRORS:
catalog.Product.category: (fields.E304) Reverse accessor for 'Product.category' clashes with reverse accessor for 'Product.category'.
HINT: Add or change a related_name argument to the definition for 'Product.category' or 'Product.category'.
product.Product.category: (fields.E304) Reverse accessor for 'Product.category' clashes with reverse accessor for 'Product.category'.
HINT: Add or change a related_name argument to the definition for 'Product.category' or 'Product.category'.
```

该错误表明 Django 为字段`category`找到了两个具有相同反向访问器的模型。这是因为有两个名为`Product`的模型引用了`Category`模型，产生了冲突。

当您向模型添加外键时，Django 会在相关模型中创建一个[反向访问器](https://docs.djangoproject.com/en/3.0/ref/models/fields/#django.db.models.ForeignKey.related_name)。在这种情况下，反向访问器是`products`。reverse 访问器允许您像这样访问相关对象:`category.products`。

新模型是您想要保留的模型，因此要解决这个冲突，请在`catalog/models.py`中从旧模型中移除反向访问器:

```py
--- a/store/catalog/models.py +++ b/store/catalog/models.py @@ -7,4 +7,4 @@ class Category(models.Model): class Product(models.Model): name = models.CharField(max_length=100, db_index=True) -    category = models.ForeignKey(Category, on_delete=models.CASCADE) +    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='+')
```

属性 [`related_name`](https://docs.djangoproject.com/en/3.0/ref/models/fields/#django.db.models.ForeignKey.related_name) 可用于显式设置反向存取器的相关名称。这里，您使用特殊值`+`，它指示 Django 不要创建反向访问器。

现在为`catalog`应用程序生成一个迁移:

```py
$ python manage.py makemigrations catalog
Migrations for 'catalog':
 catalog/migrations/0002_auto_20200124_1250.py
 - Alter field category on product
```

**暂时不要应用此迁移！**一旦发生这种变化，使用反向访问器的代码可能会中断。

既然反向访问器之间没有冲突，那么尝试为新的`product`应用程序生成迁移:

```py
$ python manage.py makemigrations product
Migrations for 'product':
 product/migrations/0001_initial.py
 - Create model Product
```

太好了！你已经准备好进入下一步了。

[*Remove ads*](/account/join/)

### 将数据复制到新模型

在上一步中，您创建了一个新的`product`应用程序，其`Product`模型与您想要移动的模型相同。下一步是将数据从旧模型转移到新模型。

要创建数据迁移，请从终端执行以下命令:

```py
$ python manage.py makemigrations product --empty
Migrations for 'product':
 product/migrations/0002_auto_20200124_1300.py
```

编辑新的迁移文件，并添加从旧表中复制数据的操作:

```py
from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ('product', '0001_initial'),
    ]

    operations = [
        migrations.RunSQL("""
 INSERT INTO product_product (
 id,
 name,
 category_id
 )
 SELECT
 id,
 name,
 category_id
 FROM
 catalog_product;
 """, reverse_sql="""
 INSERT INTO catalog_product (
 id,
 name,
 category_id
 )
 SELECT
 id,
 name,
 category_id
 FROM
 product_product;
 """)
    ]
```

要在迁移中执行 SQL，可以使用特殊的 [`RunSQL`迁移命令](https://docs.djangoproject.com/en/3.0/ref/migration-operations/#runsql)。第一个参数是要应用的 SQL。您还可以使用`reverse_sql`参数提供一个动作来反转迁移。

当您发现错误并希望回滚更改时，撤销迁移会很方便。大多数内置迁移操作都可以逆转。例如，添加字段的相反操作是删除字段。创建新表的相反操作是删除表。通常最好提供`reverse_SQL`到`RunSQL`,这样如果出了问题，你可以回溯。

在这种情况下，正向迁移操作将数据从`product_product`插入到`catalog_product`。反向操作将做完全相反的事情，将数据从`catalog_product`插入`product_product`。通过为 Django 提供反向操作，您将能够在发生灾难时反向迁移。

此时，您仍处于迁移过程的中途。但是这里有一个教训，所以继续应用迁移:

```py
$ python manage.py migrate product
Operations to perform:
 Apply all migrations: product
Running migrations:
 Applying product.0001_initial... OK
 Applying product.0002_auto_20200124_1300... OK
```

在进入下一步之前，尝试创建一个新产品:

>>>

```py
>>> from product.models import Product
>>> Product.objects.create(name='Fancy Boots', category_id=2)
Traceback (most recent call last):
  File "/venv/lib/python3.8/site-packages/django/db/backends/utils.py", line 86, in _execute
    return self.cursor.execute(sql, params)
psycopg2.errors.UniqueViolation: duplicate key value violates unique constraint "product_product_pkey"
DETAIL:  Key (id)=(1) already exists.
```

当您使用一个**自动递增主键**时，Django 会在数据库中创建一个[序列](https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-sequences/)来为新对象分配唯一的标识符。例如，请注意，您没有为新产品提供 ID。您通常不希望提供 ID，因为您希望数据库使用序列为您分配主键。然而，在这种情况下，新表为新产品赋予了 ID `1`,即使这个 ID 已经存在于表中。

那么，哪里出了问题？当您将数据复制到新表时，没有同步序列。要同步序列，您可以使用另一个名为 [`sqlsequencereset`](https://docs.djangoproject.com/en/3.0/ref/django-admin/#sqlsequencereset) 的 Django 管理命令。该命令生成一个脚本，根据表中的现有数据设置序列的当前值。该命令通常用于用预先存在的数据填充新模型。

使用`sqlsequencereset`生成一个脚本来同步序列:

```py
$ python manage.py sqlsequencereset product
BEGIN;
SELECT setval(pg_get_serial_sequence('"product_product"','id'), coalesce(max("id"), 1), max("id") IS NOT null)
FROM "product_product";
COMMIT;
```

该命令生成的脚本是特定于数据库的。在本例中，数据库是 PostgreSQL。该脚本将序列的当前值设置为序列应该产生的下一个值，即表中的最大 ID 加 1。

最后，将代码片段添加到数据迁移中:

```py
--- a/store/product/migrations/0002_auto_20200124_1300.py +++ b/store/product/migrations/0002_auto_20200124_1300.py @@ -22,6 +22,8 @@ class Migration(migrations.Migration): category_id FROM catalog_product; + +            SELECT setval(pg_get_serial_sequence('"product_product"','id'), coalesce(max("id"), 1), max("id") IS NOT null) FROM "product_product"; """, reverse_sql=""" INSERT INTO catalog_product ( id,
```

当您应用迁移时，代码片段将同步序列，解决您在上面遇到的序列问题。

这种学习同步序列的弯路给你的代码造成了一点混乱。要清理它，从 Django shell 中删除新模型中的数据:

>>>

```py
>>> from product.models import Product
>>> Product.objects.all().delete()
(3, {'product.Product': 3})
```

现在，您复制的数据已被删除，您可以反向迁移。要撤消迁移，您需要迁移到以前的迁移:

```py
$ python manage.py showmigrations product
product
 [X] 0001_initial
 [X] 0002_auto_20200124_1300

$ python manage.py migrate product 0001_initial
Operations to perform:
 Target specific migration: 0001_initial, from product
Running migrations:
 Rendering model states... DONE
 Unapplying product.0002_auto_20200124_1300... OK
```

您首先使用命令`showmigrations`列出应用于应用程序`product`的迁移。输出显示两个迁移都已应用。然后，您通过迁移到先前的迁移`0001_initial`来反转迁移`0002_auto_20200124_1300`。

如果您再次执行`showmigrations`，那么您将看到第二次迁移不再被标记为已应用:

```py
$ python manage.py showmigrations product
product
 [X] 0001_initial
 [ ] 0002_auto_20200124_1300
```

空框确认第二次迁移已被逆转。现在您已经有了一张白纸，使用新代码运行迁移:

```py
$ python manage.py migrate product
Operations to perform:
 Apply all migrations: product
Running migrations:
 Applying product.0002_auto_20200124_1300... OK
```

迁移已成功应用。确保现在可以在 Django shell 中创建新的`Product`:

>>>

```py
>>> from product.models import Product
>>> Product.objects.create(name='Fancy Boots', category_id=2)
<Product: Product object (4)>
```

太神奇了！你的努力得到了回报，你已经为下一步做好了准备。

[*Remove ads*](/account/join/)

### 更新新模型的外键

旧表当前有其他表使用`ForeignKey`字段引用它。在删除旧模型之前，您需要更改引用旧模型的模型，以便它们引用新模型。

一个仍然引用旧模型的模型是`sale`应用程序中的`Sale`。更改`Sale`模型中的外键以引用新的`Product`模型:

```py
--- a/store/sale/models.py +++ b/store/sale/models.py @@ -1,6 +1,6 @@ from django.db import models -from catalog.models import Product +from product.models import Product class Sale(models.Model): created = models.DateTimeField()
```

生成迁移并应用它:

```py
$ python manage.py makemigrations sale
Migrations for 'sale':
 sale/migrations/0002_auto_20200124_1343.py
 - Alter field product on sale

$ python manage.py migrate sale
Operations to perform:
 Apply all migrations: sale
Running migrations:
 Applying sale.0002_auto_20200124_1343... OK
```

`Sale`模型现在引用了`product`应用中的新`Product`模型。因为您已经将所有数据复制到新模型中，所以不存在约束冲突。

### 删除旧型号

上一步删除了对旧`Product`模型的所有引用。现在可以安全地从`catalog`应用中移除旧型号了:

```py
--- a/store/catalog/models.py +++ b/store/catalog/models.py @@ -3,8 +3,3 @@ from django.db import models class Category(models.Model): name = models.CharField(max_length=100) - - -class Product(models.Model): -    name = models.CharField(max_length=100, db_index=True) -    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='+')
```

生成迁移，但**尚未应用**:

```py
$ python manage.py makemigrations
Migrations for 'catalog':
 catalog/migrations/0003_delete_product.py
 - Delete model Product
```

为了确保旧模型仅在数据被复制后的*被删除，添加以下依赖关系:*

```py
--- a/store/catalog/migrations/0003_delete_product.py +++ b/store/catalog/migrations/0003_delete_product.py @@ -7,6 +7,7 @@ class Migration(migrations.Migration): dependencies = [ ('catalog', '0002_auto_20200124_1250'), +        ('sale', '0002_auto_20200124_1343'), ] operations = [
```

添加这种依赖性极其重要。跳过这一步会有可怕的后果，包括丢失数据。关于迁移文件和迁移之间的依赖关系的更多信息，请查看[深入挖掘 Django 迁移](https://realpython.com/digging-deeper-into-migrations/)。

**注意:**迁移的名称包括其生成的日期和时间。如果您使用自己的代码，那么名称的这些部分将会不同。

现在您已经添加了依赖项，请应用迁移:

```py
$ python manage.py migrate catalog
Operations to perform:
 Apply all migrations: catalog
Running migrations:
 Applying catalog.0003_delete_product... OK
```

传输现在完成了！通过创建一个新模型并将数据复制到新的`product`应用程序中，您已经成功地将`Product`模型从`catalog`应用程序中移动到了新的`catalog`应用程序中。

### 额外收获:逆转迁移

Django 迁移的好处之一是它们是可逆的。迁移可逆意味着什么？如果您犯了一个错误，那么您可以反向迁移，数据库将恢复到应用迁移之前的状态。

还记得你之前是怎么提供`reverse_sql`到`RunSQL`的吗？这就是回报的地方。

在新数据库上应用所有迁移:

```py
$ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, catalog, contenttypes, product, sale, sessions
Running migrations:
 Applying product.0001_initial... OK
 Applying product.0002_auto_20200124_1300... OK
 Applying sale.0002_auto_20200124_1343... OK
 Applying catalog.0003_delete_product... OK
```

现在，使用[特殊关键字`zero`](https://docs.djangoproject.com/en/3.0/topics/migrations/#reverting-migrations) 将它们全部反转:

```py
$ python manage.py migrate product zero
Operations to perform:
 Unapply all migrations: product
Running migrations:
 Rendering model states... DONE
 Unapplying catalog.0003_delete_product... OK
 Unapplying sale.0002_auto_20200124_1343... OK
 Unapplying product.0002_auto_20200124_1300... OK
 Unapplying product.0001_initial... OK
```

数据库现在恢复到其原始状态。如果您部署了这个版本，并且发现了一个错误，那么您可以撤销它！

[*Remove ads*](/account/join/)

### 处理特殊情况

当您将模型从一个应用程序转移到另一个应用程序时，一些 Django 特性可能需要特别注意。特别是，添加或修改数据库约束和使用**通用关系**都需要格外小心。

#### 修改约束

在实时系统上向包含数据的表添加约束可能是一项危险的操作。要添加约束，数据库必须首先验证它。在验证过程中，数据库获得了一个表上的锁，这可能会阻止其他操作，直到该过程完成。

有些约束，比如`NOT NULL`和`CHECK`，可能需要对表进行全面扫描，以验证新数据是否有效。其他约束，如`FOREIGN KEY`，需要用另一个表进行验证，这可能需要一些时间，具体取决于被引用表的大小。

#### 处理通用关系

如果你正在使用[通用关系](https://docs.djangoproject.com/en/3.0/ref/contrib/contenttypes/#generic-relations)，那么你可能需要一个额外的步骤。通用关系使用模型的主键和内容类型 ID 来引用*任何*模型表中的一行。旧模型和新模型没有相同的内容类型 ID，因此通用连接可能会中断。这有时会被忽视，因为数据库并不强制实现通用外键的完整性。

有两种方法可以处理泛型外键:

1.  将新模型的内容类型 ID 更新为旧模型的内容类型 ID。
2.  将任何引用表的内容类型 ID 更新为新模型的内容类型 ID。

无论您选择哪种方式，都要确保在部署到生产环境之前对其进行适当的测试。

### 总结:复制数据的利弊

通过复制数据将 Django 模型移动到另一个应用程序有其优点和缺点。以下是与这种方法相关的一些优点:

*   ORM 支持这一点:使用内置的迁移操作执行这一转换保证了适当的数据库支持。
*   **这是可逆的**:如果有必要，可以逆转这种迁移。

以下是这种方法的一些缺点:

*   **很慢**:复制大量数据需要时间。
*   **需要停机**:在将旧表中的数据复制到新表的过程中对其进行更改会导致数据在转换过程中丢失。为了防止这种情况发生，停机是必要的。
*   **同步数据库需要手动操作**:将数据加载到现有的表中需要同步序列和通用外键。

正如您将在接下来的小节中看到的，使用这种方法将 Django 模型移动到另一个应用程序比其他方法花费的时间要长得多。

## 最简单的方法:将新的 Django 模型引用到旧的表中

在前面的方法中，您将所有数据复制到新表中。迁移需要停机，并且可能需要很长时间才能完成，具体取决于要拷贝的数据量。

如果您不是复制数据，而是更改新模型来引用旧表，那会怎么样呢？

### 创建新模型

这一次，您将一次对模型进行所有的更改，然后让 Django 生成所有的迁移。

首先，从`catalog`应用程序中移除`Product`模型:

```py
--- a/store/catalog/models.py +++ b/store/catalog/models.py @@ -3,8 +3,3 @@ from django.db import models class Category(models.Model): name = models.CharField(max_length=100) - - -class Product(models.Model): -    name = models.CharField(max_length=100, db_index=True) -    category = models.ForeignKey(Category, on_delete=models.CASCADE)
```

您已经从`catalog`应用中移除了`Product`模型。现在将`Product`模型移动到新的`product`应用程序中:

```py
# store/product/models.py
from django.db import models

from catalog.models import Category

class Product(models.Model):
    name = models.CharField(max_length=100, db_index=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
```

现在`Product`模型已经存在于`product`应用程序中，您可以更改对旧`Product`模型的任何引用，以引用新的`Product`模型。在这种情况下，您需要将`sale`中的外键改为引用`product.Product`:

```py
--- a/store/sale/models.py +++ b/store/sale/models.py @@ -1,6 +1,6 @@ from django.db import models -from catalog.models import Product +from product.models import Product class Sale(models.Model): created = models.DateTimeField()
```

在继续生成迁移之前，您需要对新的`Product`模型做一个更小的更改:

```py
--- a/store/product/models.py +++ b/store/product/models.py @@ -5,3 +5,6 @@ from catalog.models import Category class Product(models.Model): name = models.CharField(max_length=100, db_index=True) category = models.ForeignKey(Category, on_delete=models.CASCADE) + +    class Meta: +        db_table = 'catalog_product'
```

Django 模型有一个`Meta`选项叫做 [`db_table`](https://docs.djangoproject.com/en/3.0/ref/models/options/#db-table) 。使用这个选项，您可以提供一个表名来代替 Django 生成的表名。当在现有数据库模式上设置 ORM 时，如果表名与 Django 的命名约定不匹配，那么最常用这个选项。

在这种情况下，您在`product`应用程序中设置表的名称，以引用`catalog`应用程序中现有的表。

要完成设置，请生成迁移:

```py
$ python manage.py makemigrations sale product catalog
Migrations for 'catalog':
 catalog/migrations/0002_remove_product_category.py
 - Remove field category from product
 catalog/migrations/0003_delete_product.py
 - Delete model Product
Migrations for 'product':
 product/migrations/0001_initial.py
 - Create model Product
Migrations for 'sale':
 sale/migrations/0002_auto_20200104_0724.py
 - Alter field product on sale
```

在您前进之前，使用 [`--plan`标志](https://docs.djangoproject.com/en/3.0/ref/django-admin/#cmdoption-migrate-plan)制定一个**迁移计划**:

```py
$ python manage.py migrate --plan
Planned operations:
catalog.0002_remove_product_category
 Remove field category from product
product.0001_initial
 Create model Product
sale.0002_auto_20200104_0724
 Alter field product on sale
catalog.0003_delete_product
 Delete model Product
```

该命令的输出列出了 Django 应用迁移的顺序。

[*Remove ads*](/account/join/)

### 消除对数据库的更改

这种方法的主要好处是，您实际上不需要对数据库进行任何更改，只需要对代码进行更改。要消除对数据库的更改，可以使用特殊的迁移操作 [`SeparateDatabaseAndState`](https://docs.djangoproject.com/en/3.0/ref/migration-operations/#django.db.migrations.operations.SeparateDatabaseAndState) 。

`SeparateDatabaseAndState`可用于修改 Django 在迁移过程中执行的操作。关于如何使用`SeparateDatabaseAndState`的更多信息，请查看[如何在 Django 中创建索引而不停机](https://realpython.com/create-django-index-without-downtime/)。

如果您查看 Django 生成的迁移的内容，那么您会看到 Django 创建了一个新模型并删除了旧模型。如果您执行这些迁移，那么数据将会丢失，并且表将被创建为空。为了避免这种情况，您需要确保 Django 在迁移过程中不会对数据库进行任何更改。

您可以通过将每个迁移操作包装在一个`SeparateDatabaseAndState`操作中来消除对数据库的更改。要告诉 Django 不要对数据库应用任何更改，可以将`db_operations`设置为空列表。

您计划重用旧表，所以您需要防止 Django 丢弃它。在删除模型之前，Django 将删除引用模型的字段。因此，首先，防止 Django 从`sale`到`product`丢弃外键:

```py
--- a/store/catalog/migrations/0002_remove_product_category.py +++ b/store/catalog/migrations/0002_remove_product_category.py @@ -10,8 +10,14 @@ class Migration(migrations.Migration): ] operations = [ -        migrations.RemoveField( -            model_name='product', -            name='category', +        migrations.SeparateDatabaseAndState( +            state_operations=[ +                migrations.RemoveField( +                    model_name='product', +                    name='category', +                ), +            ], +            # You're reusing the table, so don't drop it +            database_operations=[], ), ]
```

现在 Django 已经处理了相关的对象，它可以删除模型了。您想要保留`Product`表，所以要防止 Django 删除它:

```py
--- a/store/catalog/migrations/0003_delete_product.py +++ b/store/catalog/migrations/0003_delete_product.py @@ -11,7 +11,13 @@ class Migration(migrations.Migration): ] operations = [ -        migrations.DeleteModel( -            name='Product', -        ), +        migrations.SeparateDatabaseAndState( +            state_operations=[ +                migrations.DeleteModel( +                    name='Product', +                ), +            ], +            # You want to reuse the table, so don't drop it +            database_operations=[], +        ) ]
```

你用`database_operations=[]`阻止姜戈掉桌子。接下来，阻止 Django 创建新表:

```py
--- a/store/product/migrations/0001_initial.py +++ b/store/product/migrations/0001_initial.py @@ -13,15 +13,21 @@ class Migration(migrations.Migration): ] operations = [ -        migrations.CreateModel( -            name='Product', -            fields=[ -                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), -                ('name', models.CharField(db_index=True, max_length=100)), -                ('category', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='catalog.Category')), +        migrations.SeparateDatabaseAndState( +            state_operations=[ +                migrations.CreateModel( +                    name='Product', +                    fields=[ +                        ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), +                        ('name', models.CharField(db_index=True, max_length=100)), +                        ('category', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='catalog.Category')), +                    ], +                    options={ +                        'db_table': 'catalog_product', +                    }, +                ), ], -            options={ -                'db_table': 'catalog_product', -            }, -        ), +            # You reference an existing table +            database_operations=[], +        ) ]
```

这里，您使用了`database_operations=[]`来阻止 Django 创建新表。最后，您希望防止 Django 重新创建从`Sale`到新的`Product`模型的外键约束。因为您正在重用旧表，所以约束仍然存在:

```py
--- a/store/sale/migrations/0002_auto_20200104_0724.py +++ b/store/sale/migrations/0002_auto_20200104_0724.py @@ -12,9 +12,14 @@ class Migration(migrations.Migration): ] operations = [ -        migrations.AlterField( -            model_name='sale', -            name='product', -            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='product.Product'), +        migrations.SeparateDatabaseAndState( +            state_operations=[ +                migrations.AlterField( +                    model_name='sale', +                    name='product', +                    field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='product.Product'), +                ), +            ], +            database_operations=[], ), ]
```

现在您已经完成了迁移文件的编辑，请应用迁移:

```py
$ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, catalog, contenttypes, product, sale, sessions
Running migrations:
 Applying catalog.0002_remove_product_category... OK
 Applying product.0001_initial... OK
 Applying sale.0002_auto_20200104_0724... OK
 Applying catalog.0003_delete_product... OK
```

此时，您的新模型指向旧表。Django 没有对数据库做任何更改，所有的更改都是在代码中对 Django 的模型状态做的。但是在您称之为成功并继续前进之前，有必要确认新模型的状态与数据库的状态相匹配。

### 额外收获:对新模型进行更改

为了确保模型的状态与数据库的状态一致，尝试对新模型进行更改，并确保 Django 正确地检测到它。

`Product`模型在`name`字段上定义了一个索引。删除索引:

```py
--- a/store/product/models.py +++ b/store/product/models.py @@ -3,7 +3,7 @@ from django.db import models from catalog.models import Category class Product(models.Model): -    name = models.CharField(max_length=100, db_index=True) +    name = models.CharField(max_length=100) category = models.ForeignKey(Category, on_delete=models.CASCADE) class Meta:
```

您通过删除`db_index=True`删除了索引。接下来，生成迁移:

```py
$ python manage.py makemigrations
Migrations for 'product':
 product/migrations/0002_auto_20200104_0856.py
 - Alter field name on product
```

在继续之前，检查 Django 为这次迁移生成的 SQL:

```py
$ python manage.py sqlmigrate product 0002
BEGIN;
--
-- Alter field name on product
--
DROP INDEX IF EXISTS "catalog_product_name_924af5bc";
DROP INDEX IF EXISTS "catalog_product_name_924af5bc_like";
COMMIT;
```

太好了！Django 检测到旧索引，如前缀`"catalog_*"`所示。现在，您可以执行迁移了:

```py
$ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, catalog, contenttypes, product, sale, sessions
Running migrations:
 Applying product.0002_auto_20200104_0856... OK
```

确保您在数据库中获得了预期的结果:

```py
django_migration_test=#  \d  catalog_product
 Table "public.catalog_product"
 Column    |          Type          | Nullable |                   Default
-------------+------------------------+----------+---------------------------------------------
 id          | integer                | not null | nextval('catalog_product_id_seq'::regclass)
 name        | character varying(100) | not null |
 category_id | integer                | not null |
Indexes:
 "catalog_product_pkey" PRIMARY KEY, btree (id)
 "catalog_product_category_id_35bf920b" btree (category_id)
Foreign-key constraints:
 "catalog_product_category_id_35bf920b_fk_catalog_category_id"
 FOREIGN KEY (category_id) REFERENCES catalog_category(id)
 DEFERRABLE INITIALLY DEFERRED
Referenced by:
 TABLE "sale_sale" CONSTRAINT "sale_sale_product_id_18508f6f_fk_catalog_product_id"
 FOREIGN KEY (product_id) REFERENCES catalog_product(id)
 DEFERRABLE INITIALLY DEFERRED
```

成功！`name`列上的索引已被删除。

[*Remove ads*](/account/join/)

### 总结:更改模型参考的利弊

更改模型以引用另一个模型有其优点和缺点。以下是与这种方法相关的一些优点:

*   **很快**:这种方法不对数据库做任何改动，所以非常快。
*   **不需要停机**:这种方法不需要复制数据，因此可以在没有停机的情况下在活动系统上执行。
*   **这是可逆的**:如果有必要，可以逆转这种迁移。
*   ORM 支持这一点:使用内置的迁移操作执行这一转换保证了适当的数据库支持。
*   **它不需要与数据库**同步:使用这种方法，相关的对象，比如索引和序列，保持不变。

这种方法唯一的主要缺点是**它打破了命名惯例**。使用现有表格意味着表格仍将使用旧应用程序的名称。

请注意，这种方法比复制数据要简单得多。

## Django 方式:重命名表

在前面的示例中，您让新模型引用数据库中的旧表。结果，您打破了 Django 使用的命名约定。在这种方法中，您做相反的事情:让旧的表引用新的模型。

更具体地说，您创建了新的模型，并为它生成了一个迁移。然后，从 Django 创建的迁移中获取新表的名称，并使用特殊的迁移操作`AlterModelTable`将旧表重命名为新表的名称，而不是为新模型创建表。

### 创建新模型

就像之前一样，你首先创建一个新的`product`应用程序，一次性完成所有的更改。首先，从`catalog`应用中移除`Product`型号:

```py
--- a/store/catalog/models.py +++ b/store/catalog/models.py @@ -3,8 +3,3 @@ from django.db import models class Category(models.Model): name = models.CharField(max_length=100) - - -class Product(models.Model): -    name = models.CharField(max_length=100, db_index=True) -    category = models.ForeignKey(Category, on_delete=models.CASCADE)
```

您已经从`catalog`中删除了`Product`。接下来，将`Product`模型移动到新的`product`应用程序中:

```py
# store/product/models.py
from django.db import models

from catalog.models import Category

class Product(models.Model):
    name = models.CharField(max_length=100, db_index=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
```

`Product`模型现在存在于您的`product`应用程序中。现在将`Sale`中的外键改为引用`product.Product`:

```py
--- a/store/sale/models.py +++ b/store/sale/models.py @@ -1,6 +1,6 @@ from django.db import models -from catalog.models import Product +from product.models import Product class Sale(models.Model): created = models.DateTimeField() --- a/store/store/settings.py +++ b/store/store/settings.py @@ -40,6 +40,7 @@ INSTALLED_APPS = [ 'catalog', 'sale', +    'product', ]
```

接下来，让 Django 为您生成迁移:

```py
$ python manage.py makemigrations sale catalog product
Migrations for 'catalog':
 catalog/migrations/0002_remove_product_category.py
 - Remove field category from product
 catalog/migrations/0003_delete_product.py
 - Delete model Product
Migrations for 'product':
 product/migrations/0001_initial.py
 - Create model Product
Migrations for 'sale':
 sale/migrations/0002_auto_20200110_1304.py
 - Alter field product on sale
```

您希望防止 Django 删除该表，因为您打算对它进行重命名。

为了在`product`应用程序中获得`Product`模型的名称，为创建`Product`的迁移生成 SQL:

```py
$ python manage.py sqlmigrate product 0001
BEGIN;
--
-- Create model Product
--
CREATE TABLE "product_product" ("id" serial NOT NULL PRIMARY KEY, "name" varchar(100) NOT NULL, "category_id" integer NOT NULL); ALTER TABLE "product_product" ADD CONSTRAINT "product_product_category_id_0c725779_fk_catalog_category_id" FOREIGN KEY ("category_id") REFERENCES "catalog_category" ("id") DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX "product_product_name_04ac86ce" ON "product_product" ("name");
CREATE INDEX "product_product_name_04ac86ce_like" ON "product_product" ("name" varchar_pattern_ops);
CREATE INDEX "product_product_category_id_0c725779" ON "product_product" ("category_id");
COMMIT;
```

Django 在`product`应用中为`Product`模型生成的表的名称是`product_product`。

[*Remove ads*](/account/join/)

### 重命名旧表

既然已经为模型生成了名称 Django，就可以重命名旧表了。为了从`catalog`应用中删除`Product`模型，Django 创建了两个迁移:

1.  **`catalog/migrations/0002_remove_product_category`** 从表中删除外键。
2.  **`catalog/migrations/0003_delete_product`** 降将模式。

在重命名表之前，您希望防止 Django 将外键删除到`Category`:

```py
--- a/store/catalog/migrations/0002_remove_product_category.py +++ b/store/catalog/migrations/0002_remove_product_category.py @@ -10,8 +10,13 @@ class Migration(migrations.Migration): ] operations = [ -        migrations.RemoveField( -            model_name='product', -            name='category', +        migrations.SeparateDatabaseAndState( +            state_operations=[ +                migrations.RemoveField( +                    model_name='product', +                    name='category', +                ), +            ], +            database_operations=[], ), ]
```

使用将`database_operations`设置为空列表的`SeparateDatabaseAndState`可以防止 Django 删除该列。

Django 提供了一个特殊的迁移操作， [`AlterModelTable`](https://docs.djangoproject.com/en/3.0/ref/migration-operations/#altermodeltable) ，为一个模型重命名一个表。编辑删除旧表的迁移，并将表重命名为`product_product`:

```py
--- a/store/catalog/migrations/0003_delete_product.py +++ b/store/catalog/migrations/0003_delete_product.py @@ -11,7 +11,17 @@ class Migration(migrations.Migration): ] operations = [ -        migrations.DeleteModel( -            name='Product', -        ), +        migrations.SeparateDatabaseAndState( +            state_operations=[ +                migrations.DeleteModel( +                    name='Product', +                ), +            ], +            database_operations=[ +                migrations.AlterModelTable(  +                    name='Product',  +                    table='product_product',  +                ),  +            ], +        ) ]
```

您使用了`SeparateDatabaseAndState`和`AlterModelTable`来为 Django 提供不同的迁移操作，以便在数据库中执行。

接下来，您需要阻止 Django 为新的`Product`模型创建一个表。相反，您希望它使用您重命名的表。在`product`应用程序中对初始迁移进行以下更改:

```py
--- a/store/product/migrations/0001_initial.py +++ b/store/product/migrations/0001_initial.py @@ -13,12 +13,18 @@ class Migration(migrations.Migration): ] operations = [ -        migrations.CreateModel( -            name='Product', -            fields=[ -                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), -                ('name', models.CharField(db_index=True, max_length=100)), -                ('category', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='catalog.Category')), -            ], +        migrations.SeparateDatabaseAndState( +            state_operations=[ +                migrations.CreateModel( +                    name='Product', +                    fields=[ +                        ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), +                        ('name', models.CharField(db_index=True, max_length=100)), +                        ('category', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='catalog.Category')), +                    ], +                ), +            ], +            # Table already exists. See catalog/migrations/0003_delete_product.py +            database_operations=[], ), ]
```

迁移在 Django 的状态中创建了模型，但是由于行`database_operations=[]`，它没有在数据库中创建表。还记得你把老表改名为`product_product`的时候吗？通过将旧表重命名为 Django 为新模型生成的名称，可以强制 Django 使用旧表。

最后，您希望防止 Django 在`Sale`模型中重新创建外键约束:

```py
--- a/store/sale/migrations/0002_auto_20200110_1304.py +++ b/store/sale/migrations/0002_auto_20200110_1304.py @@ -12,9 +12,15 @@ class Migration(migrations.Migration): ] operations = [ -        migrations.AlterField( -            model_name='sale', -            name='product', -            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='product.Product'), -        ), +        migrations.SeparateDatabaseAndState( +            state_operations=[ +                migrations.AlterField( +                    model_name='sale', +                    name='product', +                    field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='product.Product'), +                ), +            ], +            # You're reusing an existing table, so do nothing +            database_operations=[], +        ) ]
```

您现在已经准备好运行迁移了:

```py
$ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, catalog, contenttypes, product, sale, sessions
Running migrations:
 Applying catalog.0002_remove_product_category... OK
 Applying product.0001_initial... OK
 Applying sale.0002_auto_20200110_1304... OK
 Applying catalog.0003_delete_product... OK
```

太好了！迁移成功。但在你继续前进之前，确保它可以被逆转:

```py
$ python manage.py migrate catalog 0001
Operations to perform:
 Target specific migration: 0001_initial, from catalog
Running migrations:
 Rendering model states... DONE
 Unapplying catalog.0003_delete_product... OK
 Unapplying sale.0002_auto_20200110_1304... OK
 Unapplying product.0001_initial... OK
 Unapplying catalog.0002_remove_product_category... OK
```

太神奇了！迁移是完全可逆的。

**注:** `AlterModelTable`一般比`RunSQL`更可取，原因有几个。

首先，`AlterModelTable`[能否处理基于模型名称的字段之间的多对多关系](https://github.com/django/django/blob/3.0.2/django/db/migrations/operations/models.py#L469)。使用`RunSQL`重命名表可能需要一些额外的工作。

此外，内置的迁移操作如`AlterModelTable`是数据库不可知的，而`RunSQL`不是。例如，如果您的应用程序需要在多个数据库引擎上工作，那么您可能会在编写与所有数据库引擎兼容的 SQL 时遇到一些麻烦。

[*Remove ads*](/account/join/)

### 加分:懂内省

Django ORM 是一个抽象层，它将 Python 类型转换成数据库表，反之亦然。例如，当您在`product`应用程序中创建模型`Product`时，Django 创建了一个名为`product_product`的表。除了表，ORM 还创建其他数据库对象，比如索引、约束、序列等等。Django 根据应用程序和模型的名称为所有这些对象命名。

为了更好地理解它的样子，请检查数据库中的表`catalog_category`:

```py
django_migration_test=#  \d  catalog_category
 Table "public.catalog_category"
 Column |          Type          | Nullable |                   Default
--------+------------------------+----------+----------------------------------------------
 id     | integer                | not null | nextval('catalog_category_id_seq'::regclass)
 name   | character varying(100) | not null |
Indexes:
 "catalog_category_pkey" PRIMARY KEY, btree (id)
```

该表是 Django 为应用程序`catalog`中的`Category`模型生成的，因此得名`catalog_category`。您还可以注意到其他数据库对象也有类似的命名约定。

*   **`catalog_category_pkey`** 指一个主键指标。
*   **`catalog_category_id_seq`** 是指为主键字段`id`生成值的序列。

接下来，检查您从`catalog`移动到`product`的`Product`模型的工作台:

```py
django_migration_test=#  \d  product_product
 Table "public.product_product"
 Column    |          Type          | Nullable |                   Default
-------------+------------------------+----------+---------------------------------------------
 id          | integer                | not null | nextval('catalog_product_id_seq'::regclass)
 name        | character varying(100) | not null |
 category_id | integer                | not null |
Indexes:
 "catalog_product_pkey" PRIMARY KEY, btree (id)
 "catalog_product_category_id_35bf920b" btree (category_id)
 "catalog_product_name_924af5bc" btree (name)
 "catalog_product_name_924af5bc_like" btree (name varchar_pattern_ops)
Foreign-key constraints:
 "catalog_product_category_id_35bf920b_fk_catalog_category_id"
 FOREIGN KEY (category_id)
 REFERENCES catalog_category(id)
 DEFERRABLE INITIALLY DEFERRED
```

乍一看，相关对象比较多。但是，仔细观察就会发现，相关对象的名称与表的名称并不一致。例如，表的名称是`product_product`，但是主键约束的名称是`catalog_product_pkey`。您从名为`catalog`的应用程序中复制了模型，这意味着迁移操作`AlterModelTable`不会改变*所有相关数据库对象的名称。*

为了更好地理解`AlterModelTable`是如何工作的，请查看这个迁移操作生成的 SQL:

```py
$ python manage.py sqlmigrate catalog 0003
BEGIN;
--
-- Custom state/database change combination
--
ALTER TABLE "catalog_product" RENAME TO "product_product";
COMMIT;
```

这表明 **`AlterModelTable`只重命名了表格**。如果是这种情况，那么如果您试图对与这些对象的表相关的数据库对象之一进行更改，会发生什么情况呢？姜戈能够应对这些变化吗？

要找到答案，请尝试删除`Product`模型中字段`name`的索引:

```py
--- a/store/product/models.py +++ b/store/product/models.py @@ -3,5 +3,5 @@ from django.db import models from catalog.models import Category class Product(models.Model): -    name = models.CharField(max_length=100, db_index=True) +    name = models.CharField(max_length=100, db_index=False) category = models.ForeignKey(Category, on_delete=models.CASCADE)
```

接下来，生成迁移:

```py
$ python manage.py makemigrations
Migrations for 'product':
 product/migrations/0002_auto_20200110_1426.py
 - Alter field name on product
```

命令成功了，这是一个好迹象。现在检查生成的 SQL:

```py
$ python manage.py sqlmigrate product 0002
BEGIN;
--
-- Alter field name on product
--
DROP INDEX IF EXISTS "catalog_product_name_924af5bc";
DROP INDEX IF EXISTS "catalog_product_name_924af5bc_like";
COMMIT;
```

生成的 SQL 命令删除索引`catalog_product_name_924af5bc`。Django 能够检测到现有的索引，即使它与表名不一致。这被称为**内省**。

ORM 内部使用自省，所以你不会找到太多关于它的文档。每个[数据库后端](https://github.com/django/django/tree/3.0.2/django/db/backends)包含一个[自省模块](https://github.com/django/django/blob/3.0.2/django/db/backends/postgresql/introspection.py)，它可以根据数据库对象的属性来识别它们。自检模块通常会使用数据库提供的元数据表。使用自省，ORM 可以操纵对象，而不依赖于命名约定。这就是 Django 能够检测要删除的索引名称的方法。

[*Remove ads*](/account/join/)

### 总结:重命名表的利弊

重命名表有其优点和缺点。以下是与这种方法相关的一些优点:

*   **很快**:这种方法只重命名数据库对象，所以非常快。
*   **不需要停机**:使用这种方法，数据库对象在被重命名时只被锁定一小段时间，因此可以在没有停机的情况下在活动系统上执行。
*   **这是可逆的**:如果有必要，可以逆转这种迁移。
*   ORM 支持这一点:使用内置的迁移操作执行这一转换保证了适当的数据库支持。

与这种方法相关的唯一潜在的缺点是**它打破了命名惯例**。只重命名表意味着其他数据库对象的名称将与 Django 的命名约定不一致。这可能会在直接使用数据库时造成一些混乱。但是，Django 仍然可以使用自省来识别和管理这些对象，所以这不是一个主要问题。

## 指南:选择最佳方法

在本教程中，您已经学习了如何以三种不同的方式将 Django 模型从一个应用程序移动到另一个应用程序。下面是本教程中描述的方法的比较:

| 公制的 | 复制数据 | 更改表格 | 重命名表格 |
| --- | --- | --- | --- |
| 快的 | 一千 | ✔️ | ✔️ |
| 无停机时间 | 一千 | ✔️ | ✔️ |
| 同步相关对象 | 一千 | ✔️ | ✔️ |
| 保留命名约定 | ✔️ | 一千 | ✔️ |
| 内置 ORM 支持 | ✔️ | ✔️ | ✔️ |
| 可逆的 | ✔️ | ✔️ | ✔️ |

**注意**:上表表明重命名表保留了 Django 的命名约定。虽然严格来说这并不正确，但是您在前面已经了解到 Django 可以使用内省来克服与这种方法相关的命名问题。

以上每种方法都有自己的优点和缺点。那么，您应该使用哪种方法呢？

根据一般经验，当您处理小表并且能够承受一些停机时间时，应该复制数据。否则，最好的办法是重命名该表，并引用新模型。

也就是说，每个项目都有自己独特的需求。您应该选择对您和您的团队最有意义的方法。

## 结论

阅读完本教程后，您将能够更好地根据您的具体用例、限制和需求，做出如何将 Django 模型迁移到另一个应用程序的正确决定。

**在本教程中，您已经学习了:**

*   如何将 Django 模型从一个应用程序移动到另一个应用程序
*   如何使用 Django 迁移 CLI 的**高级功能**，如`sqlmigrate`、`showmigrations`、`sqlsequencereset`
*   如何制定和检查**迁移计划**
*   如何使迁移可逆，以及如何**逆转迁移**
*   什么是内省以及 Django 如何在迁移中使用它

要深入了解，请查看完整的[数据库教程](https://realpython.com/tutorials/databases/)和 [Django 教程](https://realpython.com/tutorials/django/)。**********