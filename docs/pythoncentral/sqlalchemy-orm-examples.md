# SQLAlchemy ORM 示例

> 原文：<https://www.pythoncentral.io/sqlalchemy-orm-examples/>

## ORM 摘要

在[以前的一篇文章](https://www.pythoncentral.io/overview-sqlalchemys-expression-language-orm-queries/ "Overview of SQLAlchemy’s Expression Language and ORM Queries")中，我们简要地浏览了一个带有两个表`department`和`employee`的示例数据库，其中一个部门可以有多个雇员，一个雇员可以属于任意数量的部门。我们使用了几个代码片段来展示 SQLAlchemy 表达式语言的强大功能，并展示如何编写 ORM 查询。

在本文中，我们将更详细地了解 SQLAlchemy 的 ORM，并找出如何更有效地使用它来解决现实世界中的问题。

## 部门和员工

我们将继续使用前一篇文章中的 department-employee 作为本文中的示例数据库。我们还将向每个表中添加更多的列，以使我们的示例更加有趣。

```py

from sqlalchemy import Column, DateTime, String, Integer, ForeignKey, func

from sqlalchemy.orm import relationship, backref

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
类 Department(Base):
_ _ tablename _ _ = ' Department '
id = Column(Integer，primary _ key = True)
name = Column(String)
class Employee(Base):
_ _ tablename _ _ = ' Employee '
id = Column(Integer，primary _ key = True)
name = Column(String)
# Use default = func . now()将雇员的默认雇佣时间
 #设置为创建
 # Employee 记录的当前时间
 hired_on = Column(DateTime，default = func . now())
Department _ id = Column(Integer，foreign key(' Department . id ')
#雇员的当前时间
从 sqlalchemy 导入 create _ engine
engine = create _ engine(' SQLite:///ORM _ in _ detail . SQLite ')
从 sqlalchemy.orm 导入 session maker
session = session maker()
session . configure(bind = engine)
base . metadata . create _ all(engine)

```

注意，我们对 employee 表做了两处修改:1。我们插入了一个新列‘hired _ on ’,它是一个日期时间列，存储雇员被雇用的时间和，2。我们在关系`Employee.department`的`backref`中插入了一个关键字参数‘cascade ’,其值为 delete，all’。cascade 允许 SQLAlchemy 在部门本身被删除时自动删除该部门的雇员。

现在让我们写几行代码来使用新的表定义。

```py

>>> d = Department(name="IT")

>>> emp1 = Employee(name="John", department=d)

>>> s = session()

>>> s.add(d)

>>> s.add(emp1)

>>> s.commit()

>>> s.delete(d)  # Deleting the department also deletes all of its employees.

>>> s.commit()

>>> s.query(Employee).all()

[]

```

让我们创建另一个雇员来测试我们新的日期时间列“hired_on”:

```py

>>> emp2 = Employee(name="Marry")

>>> emp2.hired_on

>>> s.add(emp2)

>>> emp2.hired_on

>>> s.commit()

>>> emp2.hired_on

datetime.datetime(2014, 3, 24, 2, 3, 46)

```

你注意到这个小片段有些奇怪吗？既然`Employee.hired_on`被定义为默认值`func.now()`，那么`emp2.hired_on`在被创建后怎么会是`None` *？*

答案在于 SQLAlchemy 是如何处理`func.now()`的。`func`生成 SQL 函数表达式。`func.now()`在 SQL 中直译为 *now()* :

```py

>>> print func.now()

now()

>>> from sqlalchemy import select

>>> rs = s.execute(select([func.now()]))

>>> rs.fetchone()

(datetime.datetime(2014, 3, 24, 2, 9, 12),)

```

正如您所看到的，通过 SQLAlchemy 数据库会话对象执行`func.now()`函数会根据我们机器的时区给出当前的日期时间。

在继续下一步之前，让我们删除`department`表和`employee`表中的所有记录，这样我们可以稍后从一个干净的数据库开始。

```py

>>> for department in s.query(Department).all():

...     s.delete(department)

...

>>> s.commit()

>>> s.query(Department).count()

0

>>> s.query(Employee).count()

0

```

## 更多 ORM 查询

让我们继续编写查询，以便更加熟悉 ORM API。首先，我们在两个部门“IT”和“财务”中插入几个雇员。

```py

IT = Department(name="IT")

Financial = Department(name="Financial")

john = Employee(name="John", department=IT)

marry = Employee(name="marry", department=Financial)

s.add(IT)

s.add(Financial)

s.add(john)

s.add(marry)

s.commit()

cathy = Employee(name="Cathy", department=Financial)

s.add(cathy)

s.commit()

```

假设我们想找到名字以“C”开头的所有雇员，我们可以使用`startswith()`来实现我们的目标:

```py

>>>s.query(Employee).filter(Employee.name.startswith("C")).one().name

u'Cathy'

```

让查询变得更加困难的是，假设我们想要查找姓名以“C”开头并且也在财务部门工作的所有雇员，我们可以使用一个连接查询:

```py

>>> s.query(Employee).join(Employee.department).filter(Employee.name.startswith('C'), Department.name == 'Financial').all()[0].name

u'Cathy'

```

如果我们想搜索在某个日期之前雇用的员工，该怎么办？我们可以在 filter 子句中使用普通的 datetime 比较运算符。

```py

>>> from datetime import datetime

# Find all employees who will be hired in the future

>>> s.query(Employee).filter(Employee.hired_on > func.now()).count()

0

# Find all employees who have been hired in the past

>>> s.query(Employee).filter(Employee.hired_on < func.now()).count()

3

```

## 部门和员工之间的多对多

到目前为止，一个`Department`可以有多个`Employees`，一个`Employee`最多属于一个`Department`。因此，`Department`和`Employee`之间是一对多的关系。如果一个`Employee`可以属于任意数量的`Department`呢？我们如何处理多对多的关系？

为了处理`Department`和`Employee`之间的多对多关系，我们将创建一个新的关联表“department_employee_link ”,其中外键列指向`Department`和`Employee`。我们还需要从`Department`中删除`backref`定义，因为我们将在`Employee`中插入一个多对多`relationship`。

```py

import os
from sqlalchemy 导入列，DateTime，String，Integer，ForeignKey，func 
 from sqlalchemy.orm 导入关系，back ref
from sqlalchemy . ext . declarative import declarative _ base
Base = declarative _ Base()
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
hired _ on = Column(
DateTime，
default = func . now())
departments = relationship(
Department，
secondary = ' Department _ Employee _ link '
)
class DepartmentEmployeeLink(Base):
_ _ tablename _ _ = ' department _ employee _ link '
department _ id = Column(Integer，ForeignKey('department.id ')，primary _ key = True)
employee _ id = Column(Integer，ForeignKey('employee.id ')，primary_key=True) 

```

请注意，`DepartmentEmployeeLink`中的所有列“部门标识”和“员工标识”被组合在一起形成了表`department_employee_link`的主键，而类`Department`和类`Employee`中的`relationship`参数有一个指向关联表的附加关键字参数“次要”。

一旦我们定义了我们的模型，我们就可以按以下方式使用它们:

```py

>>> fp = 'orm_in_detail.sqlite'

>>> # Remove the existing orm_in_detail.sqlite file

>>> if os.path.exists(fp):

...     os.remove(fp)

...

>>> from sqlalchemy import create_engine

>>> engine = create_engine('sqlite:///orm_in_detail.sqlite')

>>>

>>> from sqlalchemy.orm import sessionmaker

>>> session = sessionmaker()

>>> session.configure(bind=engine)

>>> Base.metadata.create_all(engine)

>>>

>>> s = session()

>>> IT = Department(name="IT")

>>> Financial = Department(name="Financial")

>>> cathy = Employee(name="Cathy")

>>> marry = Employee(name="Marry")

>>> john = Employee(name="John")

>>> cathy.departments.append(Financial)

>>> Financial.employees.append(marry)

>>> john.departments.append(IT)

>>> s.add(IT)

>>> s.add(Financial)

>>> s.add(cathy)

>>> s.add(marry)

>>> s.add(john)

>>> s.commit()

>>> cathy.departments[0].name

u'Financial'

>>> marry.departments[0].name

u'Financial'

>>> john.departments[0].name

u'IT'

>>> IT.employees[0].name

u'John'

```

注意，我们使用`Employee.departments.append()`将一个`Department`添加到一个`Employee`的部门列表中。

要查找 IT 部门的员工列表，无论他们是否属于其他部门，我们可以使用`relationship.any()`函数。

```py

>>> s.query(Employee).filter(Employee.departments.any(Department.name == 'IT')).all()[0].name

u'John'

```

另一方面，要查找 John 是其雇员之一的部门列表，我们可以使用相同的函数。

```py

>>> s.query(Department).filter(Department.employees.any(Employee.name == 'John')).all()[0].name

u'IT'

```

## 总结和提示

在本文中，我们深入研究了 SQLAlchemy 的 ORM 库，并编写了更多的查询来探索 API。请注意，当您想要将删除从外键引用的对象级联到引用对象时，您可以在引用对象的外键定义的`backref`中指定`cascade='all,delete'`(如示例关系`Employee.department`中所示)。