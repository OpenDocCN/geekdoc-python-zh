# 如何在 Django 中创建索引而不停机

> 原文：<https://realpython.com/create-django-index-without-downtime/>

在任何软件项目中，管理数据库迁移都是一个巨大的挑战。幸运的是，从 1.7 版本开始， [Django](https://realpython.com/get-started-with-django-1/) 提供了一个内置的迁移框架。该框架在管理数据库变化方面非常强大和有用。但是框架提供的灵活性需要一些妥协。为了理解 Django 迁移的局限性，您将处理一个众所周知的问题:在 Django 中创建一个索引，并且不停机。

在本教程中，您将学习:

*   Django 如何以及何时产生新的迁移
*   如何检查 Django 生成的执行迁移的命令
*   如何安全地修改迁移以满足您的需求

这篇中级教程是为已经熟悉 Django 迁移的读者设计的。关于这个主题的介绍，请查看 [Django 迁移:初级读本](https://realpython.com/django-migrations-a-primer/)。

**免费奖励:** ，您可以用它们来加深您的 Python web 开发技能。

## Django 迁移中创建索引的问题

当应用程序存储的数据增长时，通常需要进行的一个常见更改是添加索引。索引用于加快查询速度，让你的应用程序感觉更快、响应更快。

在大多数数据库中，添加索引需要表上的排他锁。创建索引时，一个排他锁会阻止数据修改(DML)操作，如`UPDATE`、`INSERT`和`DELETE`。

当执行某些操作时，数据库会隐式获取锁。例如，当用户登录你的应用时，Django 会更新`auth_user`表中的`last_login`字段。要执行更新，数据库必须首先获得该行的锁。如果该行当前被另一个连接锁定，那么您可能会得到一个[数据库异常](https://docs.djangoproject.com/en/2.1/ref/exceptions/#database-exceptions)。

当需要在迁移期间保持系统可用时，锁定表可能会带来问题。表越大，创建索引所需的时间就越长。创建索引的时间越长，系统不可用或对用户无响应的时间就越长。

一些数据库供应商提供了一种在不锁定表的情况下创建索引的方法。例如，要在 PostgreSQL 中创建索引而不锁定表，可以使用 [`CONCURRENTLY`](https://www.postgresql.org/docs/current/sql-createindex.html) 关键字:

```py
CREATE  INDEX  CONCURRENTLY  ix  ON  table  (column);
```

在 Oracle 中，有一个 [`ONLINE`](https://docs.oracle.com/en/database/oracle/oracle-database/18/sqlrf/CREATE-INDEX.html) 选项允许在创建索引时对表进行 DML 操作:

```py
CREATE  INDEX  ix  ON  table  (column)  ONLINE;
```

在生成迁移时，Django 不会使用这些特殊的关键字。按原样运行迁移将使数据库获得表上的排他锁，并在创建索引时阻止 DML 操作。

同时创建索引有一些注意事项。提前了解特定于数据库后端的问题非常重要。例如，[PostgreSQL](https://www.postgresql.org/docs/current/sql-createindex.html#SQL-CREATEINDEX-CONCURRENTLY)中的一个警告是，并发创建索引需要更长时间，因为它需要额外的表扫描。

在本教程中，您将使用 Django 迁移在大型表上创建索引，而不会导致任何停机。

**注意:**要学习本教程，建议您使用 PostgreSQL 后端、Django 2.x 和 Python 3。

也可以使用其他数据库后端。在使用 PostgreSQL 独有的 SQL 特性的地方，更改 SQL 以匹配您的数据库后端。

[*Remove ads*](/account/join/)

## 设置

你将在名为`app`的应用中使用一个虚构的`Sale`模型。在现实生活中，像`Sale`这样的模型是数据库中的主表，它们通常会非常大，存储大量数据:

```py
# models.py

from django.db import models

class Sale(models.Model):
    sold_at = models.DateTimeField(
        auto_now_add=True,
    )
    charged_amount = models.PositiveIntegerField()
```

要创建表，请生成初始迁移并应用它:

```py
$ python manage.py makemigrations
Migrations for 'app':
 app/migrations/0001_initial.py
 - Create model Sale

$ python manage migrate
Operations to perform:
 Apply all migrations: app
Running migrations:
 Applying app.0001_initial... OK
```

一段时间后，销售表变得非常大，用户开始抱怨速度慢。在监控数据库时，您注意到许多查询都使用了`sold_at`列。为了加快速度，您决定在列上需要一个索引。

要在`sold_at`上添加索引，您需要对模型进行以下更改:

```py
# models.py

from django.db import models

class Sale(models.Model):
    sold_at = models.DateTimeField(
        auto_now_add=True,
 db_index=True,    )
    charged_amount = models.PositiveIntegerField()
```

如果您照原样运行这个迁移，那么 Django 将在表上创建索引，并且在索引完成之前它将被锁定。在一个非常大的表上创建索引可能需要一段时间，并且您希望避免停机。

在具有小数据集和很少连接的本地开发环境中，这种迁移可能感觉是瞬间的。但是，在具有许多并发连接的大型数据集上，获取锁和创建索引可能需要一段时间。

在接下来的步骤中，您将修改 Django 创建的迁移，以便在不导致任何停机的情况下创建索引。

## 假移民

第一种方法是手动创建索引。您将生成迁移，但实际上并不打算让 Django 应用它。相反，您将在数据库中手动运行 SQL，然后让 Django 认为迁移已经完成。

首先，生成迁移:

```py
$ python manage.py makemigrations --name add_index_fake
Migrations for 'app':
 app/migrations/0002_add_index_fake.py
 - Alter field sold_at on sale
```

使用 [`sqlmigrate`命令](https://docs.djangoproject.com/en/2.1/ref/django-admin/#django-admin-sqlmigrate)查看 Django 将用于执行该迁移的 SQL:

```py
$ python manage.py sqlmigrate app 0002

BEGIN;
--
-- Alter field sold_at on sale
--
CREATE INDEX "app_sale_sold_at_b9438ae4" ON "app_sale" ("sold_at");
COMMIT;
```

您希望在不锁定表的情况下创建索引，因此需要修改命令。添加`CONCURRENTLY`关键字并在数据库中执行:

```py
app=#  CREATE  INDEX  CONCURRENTLY  "app_sale_sold_at_b9438ae4" ON  "app_sale"  ("sold_at"); CREATE INDEX
```

请注意，您执行了不带`BEGIN`和`COMMIT`部分的命令。省略这些关键字将在没有数据库事务的情况下执行命令。我们将在本文后面讨论数据库事务。

执行该命令后，如果您尝试应用迁移，将会出现以下错误:

```py
$ python manage.py migrate

Operations to perform:
 Apply all migrations: app
Running migrations:
 Applying app.0002_add_index_fake...Traceback (most recent call last):
 File "venv/lib/python3.7/site-packages/django/db/backends/utils.py", line 85, in _execute
 return self.cursor.execute(sql, params)

psycopg2.ProgrammingError: relation "app_sale_sold_at_b9438ae4" already exists
```

Django 抱怨说索引已经存在，所以它不能继续迁移。您刚刚在数据库中直接创建了索引，所以现在您需要让 Django 认为已经应用了迁移。

**如何伪造移民**

Django 提供了一种内置的方法来将迁移标记为已执行，而不是实际执行它们。要使用此选项，请在应用迁移时设置`--fake`标志:

```py
$ python manage.py migrate --fake
Operations to perform:
 Apply all migrations: app
Running migrations:
 Applying app.0002_add_index_fake... FAKED
```

Django 这次没有出错。事实上，Django 并没有真正应用任何迁移。它只是将它标记为已执行(或`FAKED`)。

以下是伪造迁移时需要考虑的一些问题:

*   **手动命令必须等同于 Django 生成的 SQL:**你需要确保你执行的命令等同于 Django 生成的 SQL。使用`sqlmigrate`生成 SQL 命令。如果命令不匹配，那么可能会导致数据库和模型状态之间的不一致。

*   **其他未申请的迁移也会造假:**当你有多个未申请的迁移时，都会造假。在应用迁移之前，务必确保只有您想要伪造的迁移未被应用。否则，您可能会以不一致而告终。另一个选项是指定您想要伪造的确切迁移。

*   **需要直接访问数据库:**需要在数据库中运行 SQL 命令。这并不总是一个选项。此外，在生产数据库中直接执行命令是危险的，应该尽可能避免。

*   **自动化部署流程可能需要调整:**如果您[自动化部署流程](https://realpython.com/automating-django-deployments-with-fabric-and-ansible/)(使用 CI、CD 或其他自动化工具)，那么您可能需要更改流程以模拟迁移。这并不总是可取的。

**清理**

在进入下一节之前，您需要将数据库恢复到初始迁移后的状态。为此，请迁移回初始迁移:

```py
$ python manage.py migrate 0001
Operations to perform:
 Target specific migration: 0001_initial, from app
Running migrations:
 Rendering model states... DONE
 Unapplying app.0002_add_index_fake... OK
```

Django 没有应用第二次迁移中所做的更改，所以现在也可以安全地删除文件了:

```py
$ rm app/migrations/0002_add_index_fake.py
```

为了确保您做的一切都是正确的，请检查迁移:

```py
$ python manage.py showmigrations app
app
 [X] 0001_initial
```

已应用初始迁移，没有未应用的迁移。

[*Remove ads*](/account/join/)

## 在迁移中执行原始 SQL

在上一节中，您直接在数据库中执行 SQL 并伪造了迁移。这就完成了工作，但是还有一个更好的解决方案。

Django 提供了一种使用 [`RunSQL`](https://docs.djangoproject.com/en/2.1/ref/migration-operations/#runsql) 在迁移中执行原始 SQL 的方法。让我们尝试使用它，而不是直接在数据库中执行命令。

首先，生成一个新的空迁移:

```py
$ python manage.py makemigrations app --empty --name add_index_runsql
Migrations for 'app':
 app/migrations/0002_add_index_runsql.py
```

接下来，编辑迁移文件并添加一个`RunSQL`操作:

```py
# migrations/0002_add_index_runsql.py

from django.db import migrations, models

class Migration(migrations.Migration):
    atomic = False

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.RunSQL(
 'CREATE INDEX "app_sale_sold_at_b9438ae4" ' 'ON "app_sale" ("sold_at");', ), ]
```

运行迁移时，您将获得以下输出:

```py
$ python manage.py migrate
Operations to perform:
 Apply all migrations: app
Running migrations:
 Applying app.0002_add_index_runsql... OK
```

这看起来不错，但有一个问题。让我们再次尝试生成迁移:

```py
$ python manage.py makemigrations --name leftover_migration
Migrations for 'app':
 app/migrations/0003_leftover_migration.py
 - Alter field sold_at on sale
```

Django 再次产生了同样的迁移。为什么会这样？

**清理**

在我们回答这个问题之前，您需要清理并撤消您对数据库所做的更改。从删除最后一次迁移开始。它未被应用，因此可以安全地删除:

```py
$ rm app/migrations/0003_leftover_migration.py
```

接下来，列出`app`应用的迁移:

```py
$ python manage.py showmigrations app
app
 [X] 0001_initial
 [X] 0002_add_index_runsql
```

第三次迁移没有了，但是应用了第二次迁移。您希望在初始迁移后立即恢复状态。尝试像您在上一节中所做的那样迁移回初始迁移:

```py
$ python manage.py migrate app 0001
Operations to perform:
 Target specific migration: 0001_initial, from app
Running migrations:
 Rendering model states... DONE
 Unapplying app.0002_add_index_runsql...Traceback (most recent call last):

NotImplementedError: You cannot reverse this operation
```

Django 无法逆转迁移。

[*Remove ads*](/account/join/)

## 反向迁移操作

为了逆转迁移，Django 对每个操作执行相反的操作。在这种情况下，与添加索引相反的是删除索引。正如您已经看到的，当迁移是可逆的时，您可以取消应用它。就像您可以在 Git 中使用`checkout`一样，如果您对早期的迁移执行`migrate`,您可以逆转迁移。

许多内置迁移操作已经定义了反向操作。例如，添加字段的相反操作是删除相应的列。创建模型的相反操作是删除相应的表。

有些迁移操作是不可逆的。例如，移除字段或删除模型没有相反的操作，因为一旦应用了迁移，数据就消失了。

在上一节中，您使用了`RunSQL`操作。当您尝试反转迁移时，遇到了错误。根据该错误，迁移中的某个操作无法反转。Django 默认情况下不能反转原始 SQL。因为 Django 不知道操作执行了什么，所以它不能自动生成相反的动作。

**如何使迁移可逆**

要使迁移可逆，其中的所有操作都必须可逆。不可能逆转部分迁移，因此单个不可逆操作将使整个迁移不可逆。

要使`RunSQL`操作可逆，您必须提供 SQL 以便在操作可逆时执行。反向 SQL 在`reverse_sql`参数中提供。

与添加索引相反的操作是删除索引。要使您的迁移可逆，请提供`reverse_sql`来删除索引:

```py
# migrations/0002_add_index_runsql.py

from django.db import migrations, models

class Migration(migrations.Migration):
    atomic = False

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.RunSQL(
            'CREATE INDEX "app_sale_sold_at_b9438ae4" '
            'ON "app_sale" ("sold_at");',

            reverse_sql='DROP INDEX "app_sale_sold_at_b9438ae4";',
 ),    ]
```

现在尝试反向迁移:

```py
$ python manage.py showmigrations app
app
 [X] 0001_initial
 [X] 0002_add_index_runsql

$ python manage.py migrate app 0001
Operations to perform:
 Target specific migration: 0001_initial, from app
Running migrations:
 Rendering model states... DONE
 Unapplying app.0002_add_index_runsql... OK 
$ python manage.py showmigrations app
app
 [X] 0001_initial
 [ ] 0002_add_index_runsql
```

第二次迁移逆转，指数被 Django 掉了。现在可以安全地删除迁移文件了:

```py
$ rm app/migrations/0002_add_index_runsql.py
```

提供`reverse_sql`总是个好主意。在撤销原始 SQL 操作不需要任何操作的情况下，您可以使用特殊标记`migrations.RunSQL.noop`将操作标记为可撤销:

```py
migrations.RunSQL(
    sql='...',  # Your forward SQL here
 reverse_sql=migrations.RunSQL.noop, ),
```

## 了解模型状态和数据库状态

在您之前使用`RunSQL`手工创建索引的尝试中，Django 一次又一次地生成相同的迁移，即使索引是在数据库中创建的。要理解 Django 为什么这样做，首先需要理解 Django 如何决定何时生成新的迁移。

### 当 Django 生成新的迁移时

在生成和应用迁移的过程中，Django 在数据库状态和模型状态之间进行同步。例如，当您向模型中添加一个字段时，Django 会向表中添加一列。当您从模型中删除一个字段时，Django 会从表中删除该列。

为了在模型和数据库之间同步，Django 维护一个表示模型的状态。为了将数据库与模型同步，Django 生成迁移操作。迁移操作转换成可以在数据库中执行的特定于供应商的 SQL。当执行所有迁移操作时，数据库和模型应该是一致的。

为了获得数据库的状态，Django 汇总了过去所有迁移的操作。当迁移的聚合状态与模型状态不一致时，Django 会生成一个新的迁移。

在前面的示例中，您使用原始 SQL 创建了索引。Django 不知道您创建了索引，因为您没有使用熟悉的迁移操作。

当 Django 汇总所有的迁移并与模型的状态进行比较时，它发现缺少一个索引。这就是为什么即使手动创建了索引，Django 仍然认为它丢失了，并为它生成了一个新的迁移。

[*Remove ads*](/account/join/)

### 如何在迁移中分离数据库和状态

由于 Django 不能按照您想要的方式创建索引，所以您希望提供自己的 SQL，但仍然让 Django 知道是您创建的。

换句话说，您需要在数据库中执行一些操作，并为 Django 提供迁移操作来同步其内部状态。为此，Django 为我们提供了一个名为 [`SeparateDatabaseAndState`](https://docs.djangoproject.com/en/2.1/ref/migration-operations/#separatedatabaseandstate) 的特殊迁移操作。这种操作并不广为人知，应该保留给像这样的特殊情况。

编辑迁移比从头开始编写迁移要容易得多，所以从生成迁移开始，通常的方式是:

```py
$ python manage.py makemigrations --name add_index_separate_database_and_state

Migrations for 'app':
 app/migrations/0002_add_index_separate_database_and_state.py
 - Alter field sold_at on sale
```

这是 Django 生成的迁移内容，和以前一样:

```py
# migrations/0002_add_index_separate_database_and_state.py

from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sale',
            name='sold_at',
            field=models.DateTimeField(
                auto_now_add=True,
                db_index=True,
            ),
        ),
    ]
```

Django 在字段`sold_at`上生成了一个`AlterField`操作。该操作将创建一个索引并更新状态。我们希望保留这个操作，但是提供一个不同的命令在数据库中执行。

再次使用 Django 生成的 SQL 来获得命令:

```py
$ python manage.py sqlmigrate app 0002
BEGIN;
--
-- Alter field sold_at on sale
--
CREATE INDEX "app_sale_sold_at_b9438ae4" ON "app_sale" ("sold_at");
COMMIT;
```

在适当的位置添加`CONCURRENTLY`关键字:

```py
CREATE  INDEX  CONCURRENTLY  "app_sale_sold_at_b9438ae4" ON  "app_sale"  ("sold_at");
```

接下来，编辑迁移文件并使用`SeparateDatabaseAndState`来提供修改后的 SQL 命令以供执行:

```py
# migrations/0002_add_index_separate_database_and_state.py

from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [

        migrations.SeparateDatabaseAndState(

            state_operations=[
                migrations.AlterField(
                    model_name='sale',
                    name='sold_at',
                    field=models.DateTimeField(
                        auto_now_add=True,
                        db_index=True,
                    ),
                ),
            ],

            database_operations=[
                migrations.RunSQL(sql="""
 CREATE INDEX CONCURRENTLY "app_sale_sold_at_b9438ae4"
 ON "app_sale" ("sold_at");
 """, reverse_sql="""
 DROP INDEX "app_sale_sold_at_b9438ae4";
 """),
            ],
        ),

    ],
```

迁移操作`SeparateDatabaseAndState`接受两个操作列表:

1.  **state_operations** 是应用于内部模型状态的操作。它们不会影响数据库。
2.  **database_operations** 是应用于数据库的操作。

你在`state_operations`中保留了 Django 生成的原始操作。当使用`SeparateDatabaseAndState`时，这是你通常想要做的。请注意，`db_index=True`参数被提供给了该字段。这个迁移操作将让 Django 知道字段上有一个索引。

您使用了 Django 生成的 SQL 并添加了`CONCURRENTLY`关键字。您使用了特殊动作 [`RunSQL`](https://docs.djangoproject.com/en/2.1/ref/migration-operations/#runsql) 来执行迁移中的原始 SQL。

如果您尝试运行迁移，您将获得以下输出:

```py
$ python manage.py migrate app
Operations to perform:
 Apply all migrations: app
Running migrations:
 Applying app.0002_add_index_separate_database_and_state...Traceback (most recent call last):
 File "/venv/lib/python3.7/site-packages/django/db/backends/utils.py", line 83, in _execute
 return self.cursor.execute(sql)
psycopg2.InternalError: CREATE INDEX CONCURRENTLY cannot run inside a transaction block
```

[*Remove ads*](/account/join/)

## 非原子迁移

在 SQL 中，`CREATE`、`DROP`、`ALTER`、`TRUNCATE`操作被称为**数据定义语言** (DDL)。在支持事务性 DDL，[的数据库中，比如 PostgreSQL](https://wiki.postgresql.org/wiki/Transactional_DDL_in_PostgreSQL:_A_Competitive_Analysis#Transactional_DDL) ，Django 默认在数据库事务内部执行迁移。但是，根据上面的错误，PostgreSQL 不能在事务块中同时创建索引。

为了能够在迁移中同时创建索引，您需要告诉 Django 不要在数据库事务中执行迁移。为此，通过将`atomic`设置为`False`，将迁移标记为[非原子](https://docs.djangoproject.com/en/2.1/howto/writing-migrations/#non-atomic-migrations):

```py
# migrations/0002_add_index_separate_database_and_state.py

from django.db import migrations, models

class Migration(migrations.Migration):
 atomic = False 
    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [

        migrations.SeparateDatabaseAndState(

            state_operations=[
                migrations.AlterField(
                    model_name='sale',
                    name='sold_at',
                    field=models.DateTimeField(
                        auto_now_add=True,
                        db_index=True,
                    ),
                ),
            ],

            database_operations=[
                migrations.RunSQL(sql="""
 CREATE INDEX CONCURRENTLY "app_sale_sold_at_b9438ae4"
 ON "app_sale" ("sold_at");
 """,
                reverse_sql="""
 DROP INDEX "app_sale_sold_at_b9438ae4";
 """),
            ],
        ),

    ],
```

将迁移标记为非原子后，您可以运行迁移:

```py
$ python manage.py migrate app
Operations to perform:
 Apply all migrations: app
Running migrations:
 Applying app.0002_add_index_separate_database_and_state... OK
```

您刚刚执行了迁移，没有造成任何停机。

以下是使用`SeparateDatabaseAndState`时需要考虑的一些问题:

*   **数据库操作必须等同于状态操作:**数据库和模型状态之间的不一致会导致很多麻烦。一个好的起点是将 Django 生成的操作保存在`state_operations`中，并编辑`sqlmigrate`的输出以在`database_operations`中使用。

*   **非原子迁移在出现错误时无法回滚:**如果迁移过程中出现错误，您将无法回滚。您必须回滚迁移或手动完成迁移。将非原子迁移中执行的操作保持在最低限度是一个好主意。如果您在迁移中有其他操作，请将它们移到新的迁移中。

*   **迁移可能是特定于供应商的:**Django 生成的 SQL 是特定于项目中使用的数据库后端的。它可能适用于其他数据库后端，但这不能保证。如果您需要支持多个数据库后端，您需要对这种方法进行一些调整。

## 结论

您以一个大表和一个问题开始了本教程。你想让你的应用程序对你的用户来说更快，你想在不给他们造成任何停机的情况下做到这一点。

到本教程结束时，您已经成功地生成并安全地修改了一个 Django 迁移来实现这个目标。在这个过程中，您处理了不同的问题，并使用迁移框架提供的内置工具成功地克服了这些问题。

在本教程中，您学习了以下内容:

*   Django 迁移如何使用模型和数据库状态在内部工作，以及何时生成新的迁移
*   如何使用`RunSQL`动作在迁移中执行定制 SQL
*   什么是可逆迁移，以及如何使`RunSQL`操作可逆
*   什么是原子迁移，以及如何根据您的需要更改默认行为
*   如何在 Django 中安全地执行复杂的迁移

模型和数据库状态之间的分离是一个重要的概念。一旦您理解了它以及如何利用它，您就可以克服内置迁移操作的许多限制。想到的一些用例包括添加已经在数据库中创建的索引，以及为 DDL 命令提供特定于供应商的参数。*****