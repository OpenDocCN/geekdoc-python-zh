# 用 Alembic 迁移 SQLAlchemy 数据库

> 原文：<https://www.pythoncentral.io/migrate-sqlalchemy-databases-alembic/>

## 蒸馏器

Alembic 是 SQLAlchemy 的一个轻量级数据库迁移工具。它是由 SQLAlchemy 的作者创建的，并且已经成为在 SQLAlchemy 支持的数据库上执行迁移的事实上的标准工具。

## SQLAlchemy 中的数据库迁移

数据库迁移通常会更改数据库的模式，例如添加列或约束、添加表或更新表。它通常使用封装在事务中的原始 SQL 来执行，以便在迁移过程中出现问题时可以回滚。在本文中，我们将使用一个示例数据库来演示如何为 SQLAlchemy 数据库编写 Alembic 迁移脚本。

为了迁移一个 SQLAlchemy 数据库，我们为计划的迁移添加一个 Alembic 迁移脚本，执行迁移，更新模型定义，然后开始在迁移的模式下使用数据库。这些步骤听起来很多，但是它们做起来非常简单，这将在下一节中进行说明。

## 示例数据库模式

让我们创建一个 SQLAlchemy 数据库，其中包含一个*部门*和一个*雇员*表。

```py

import os
from sqlalchemy 导入列，DateTime，String，Integer，ForeignKey，func 
 from sqlalchemy.orm 导入关系，back ref
from sqlalchemy . ext . declarative import declarative _ base
Base = declarative_base()
类 Department(Base):
_ _ tablename _ _ = ' Department '
id = Column(Integer，primary _ key = True)
name = Column(String)
类 Employee(Base):
_ _ tablename _ _ = ' Employee '
id = Column(Integer，primary _ key = True)
name = Column(String)
hired _ on = Column(DateTime，default=func.now())
db _ name = ' alem BIC _ sample . SQLite '
如果 OS . path . exists(db _ name):
OS . remove(db _ name)
从 sqlalchemy 导入创建引擎
引擎=创建引擎(' sqlite:///' + db_name)
从 sqlalchemy.orm 导入 session maker
session = session maker()
session . configure(bind = engine)
base . metadata . create _ all(engine)

```

在创建了数据库 *alembic_sample.sqlite* 之后，我们意识到我们忘记了在`Employee`和`Department`之间添加一个*多对多*关系。

## 移民

我们选择使用 alembic 迁移数据库，而不是直接更改模式，然后从头开始重新创建数据库。为此，我们安装 alembic，初始化 alembic 环境，编写迁移脚本来添加链接表，执行迁移，然后使用更新的模型定义再次访问数据库。

```py

$ alembic init alembic

Creating directory /home/vagrant/python2-workspace/pythoncentral/sqlalchemy_series/alembic/alembic ... done

Creating directory /home/vagrant/python2-workspace/pythoncentral/sqlalchemy_series/alembic/alembic/versions ... done

Generating /home/vagrant/python2-workspace/pythoncentral/sqlalchemy_series/alembic/alembic/env.pyc ... done

Generating /home/vagrant/python2-workspace/pythoncentral/sqlalchemy_series/alembic/alembic.ini ... done

Generating /home/vagrant/python2-workspace/pythoncentral/sqlalchemy_series/alembic/alembic/script.py.mako ... done

Generating /home/vagrant/python2-workspace/pythoncentral/sqlalchemy_series/alembic/alembic/env.py ... done

Generating /home/vagrant/python2-workspace/pythoncentral/sqlalchemy_series/alembic/alembic/README ... done

Please edit configuration/connection/logging settings in '/home/vagrant/python2-workspace/pythoncentral/sqlalchemy_series/alembic/alembic.ini' before proceeding.
$ vim alembic.ini #将以“sqlalchemy.url”开头的行更改为“sqlalchemy . URL = SQLite:///alem BIC _ sample . SQLite”
$ alem BIC current
INFO[alem BIC . migration]Context impl SQLiteImpl。
 INFO [alembic.migration]将采用非事务性 DDL。
SQLite 的当前版本:///alembic_sample.sqlite:无
$ alem BIC revision-m " add department _ employee _ link "
生成/home/vagger/python 2-workspace/python central/sqlalchemy _ series/alem BIC/alem BIC/versions/1da 977 FD 3e 6 e _ add _ department _ employee _ link . py...完成的
$ alem BIC upgrade head
INFO[alem BIC . migration]Context impl SQL item pl。
 INFO [alembic.migration]将采用非事务性 DDL。
信息【alembic.migration】运行升级无- > 1da977fd3e6e，添加部门 _ 员工 _ 链接
$ alem BIC current
INFO[alem BIC . migration]Context impl SQLiteImpl。
 INFO [alembic.migration]将采用非事务性 DDL。
当前版本为 SQLite:///alem BIC _ sample . SQLite:None->1da 977 FD 3 E6 e(负责人)，添加 department_employee_link 

```

迁移脚本如下:
【python】
“”
添加部门 _ 员工 _ 链接

修订 ID: 1da977fd3e6e
修订:无
创建日期:2014-10-23 22:38:42.894194

'''

#版本标识符，由 Alembic 使用。
修订版= '1da977fd3e6e'
下一版=无

从 alembic 导入操作
将 sqlalchemy 导入为 sa

def upgrade():
op . create _ table(
' department _ employee _ link '，
sa。列(
'部门标识'，服务协议。
整数，撒。ForeignKey('department.id ')，primary_key=True
)，
sa。列(
'雇员标识'，服务协议。整数，
sa。ForeignKey('employee.id ')，primary_key=True
)
)

定义降级():
op.drop_table(
'部门 _ 员工 _ 链接'
)

现在数据库 *alembic_sample.sqlite* 已经升级，我们可以使用一段更新的模型代码来访问升级后的数据库。

```py

import os
从 sqlalchemy 导入列、日期时间、字符串、整数、外键、func 
从 sqlalchemy.orm 导入关系
从 sqlalchemy.ext.declarative 导入 declarative_base
Base = declarative_base()
class Department(Base):
_ _ tablename _ _ = ' Department '
id = Column(Integer，primary _ key = True)
name = Column(String)
employees = relationship(
' Employee '，
secondary = ' Department _ Employee _ link '
)
class Employee(Base):
_ _ tablename _ _ = ' Employee '
id = Column(Integer，primary _ key = True)
name = Column(String)
hired _ on = Column(DateTime，default = func . now())
departments = relationship(
Department，
secondary = ' Department _ Employee _ link '
)
class DepartmentEmployeeLink(Base):
_ _ tablename _ _ = ' department _ employee _ link '
department _ id = Column(Integer，ForeignKey('department.id ')，primary_key=True)
employee _ id = Column(Integer，ForeignKey('employee.id ')，primary _ key = True)
db_name = 'alembic_sample.sqlite '
从 sqlalchemy 导入创建引擎
引擎=创建引擎(' sqlite:///' + db_name)
从 sqlalchemy.orm 导入 session maker
session = session maker()
session . configure(bind = engine)
base . metadata . bind = engine
s = session()
IT = Department(name = ' IT ')
Financial = Department(name = ' Financial ')
s . add(IT)
s . add(Financial)
Cathy = Employee(name = ' Cathy ')
Marry = Employee(name = ' Marry ')
John = Employee(财务)
```

注意，我们并没有删除数据库 *alembic_sample.sqlite* ，而是执行了一个迁移来添加一个链接表。迁移后，关系`Department.employees`和`Employee.departments`按预期工作。

## 摘要

由于 *Alembic* 是专门为 SQLAlchemy 构建的轻量级数据库迁移工具，它允许您重用相同类型的数据库模型 API 来执行简单的迁移。然而，它不是一个万能的工具。对于特定于数据库的迁移，比如在 PostgreSQL 中添加触发器函数，仍然需要原始的 DDL 语句。