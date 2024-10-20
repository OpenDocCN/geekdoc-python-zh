# SQLAlchemy 表达式语言和 ORM 查询概述

> 原文：<https://www.pythoncentral.io/overview-sqlalchemys-expression-language-orm-queries/>

## 概观

在上一篇文章中，我们对 SQLAlchemy 和其他 Python ORMs 进行了比较。在本文中，我们将深入了解 SQLAlchemy 的 ORM 和表达式语言，并使用一个示例来展示它们强大的 API 和易于理解的 Python 结构。

SQLAlchemy ORM 不仅提供了将数据库概念映射到 Python 空间的方法，还提供了方便的 Python 查询 API。使用 ORM 在 SQLAlchemy 数据库中查找东西是令人愉快的，因为一切都很简单，查询结果和查询参数都以 Python 对象的形式返回。

SQLAlchemy 表达式语言为程序员提供了一个使用 Python 结构编写“SQL 语句”的系统。这些结构被建模为尽可能地类似底层数据库的结构，同时对用户隐藏了各种数据库后端之间的差异。尽管这些构造旨在用一致的结构表示后端之间的等价概念，但是它们并没有隐藏有用的后端特定的特性。因此，表达式语言为程序员提供了一种编写后端中立表达式的方法，同时允许程序员利用特定的后端特性，如果他们真的想这样做的话。

表达式语言补充了对象关系映射器。ORM 提供了将数据库概念映射到 Python 空间的抽象使用模式，其中模型用于映射表，关系用于通过关联表进行多对多映射，通过外键进行一对一映射，而表达式语言用于直接表示数据库中更原始的结构，而没有意见。

## 部门和员工的例子

我们用一个例子来说明如何在有两个表`department`和`employee`的数据库中使用表达式语言。一个`department`有很多个`employees`，而一个`employee`最多属于一个`department`。因此，数据库可以设计如下:

```py

>>> from sqlalchemy import Column, String, Integer, ForeignKey

>>> from sqlalchemy.orm import relationship, backref

>>> from sqlalchemy.ext.declarative import declarative_base

>>>

>>>

>>> Base = declarative_base()

>>>

>>>

>>> class Department(Base):

...     __tablename__ = 'department'

...     id = Column(Integer, primary_key=True)

...     name = Column(String)

...

>>>

>>> class Employee(Base):

...     __tablename__ = 'employee'

...     id = Column(Integer, primary_key=True)

...     name = Column(String)

...     department_id = Column(Integer, ForeignKey('department.id'))

...     department = relationship(Department, backref=backref('employees', uselist=True))

...

>>>

>>> from sqlalchemy import create_engine

>>> engine = create_engine('sqlite:///')

>>>

>>> from sqlalchemy.orm import sessionmaker

>>> session = sessionmaker()

>>> session.configure(bind=engine)

>>> Base.metadata.create_all(engine)

```

在本例中，我们创建了一个内存 sqlite 数据库，其中包含两个表“department”和“employee”。列' employee.department_id '是列' department.id '的外键，关系' department.employees '包括该部门的所有雇员。为了测试我们的设置，我们可以简单地插入几个示例记录，并使用 SQLAlchemy 的 ORM 查询它们:

```py

>>> john = Employee(name='john')

>>> it_department = Department(name='IT')

>>> john.department = it_department

>>> s = session()

>>> s.add(john)

>>> s.add(it_department)

>>> s.commit()

>>> it = s.query(Department).filter(Department.name == 'IT').one()

>>> it.employees

[]

>>> it.employees[0].name

u'john'

```

如你所见，我们在 IT 部门安插了一个叫约翰的人。

现在让我们使用表达式语言执行相同类型的查询:

```py

>>> from sqlalchemy import select

>>> find_it = select([Department.id]).where(Department.name == 'IT')

>>> rs = s.execute(find_it)

>>> rs
> > > rs.fetchone() 
 (1，)
>>>RS . fetchone()#查询只返回一个结果，所以多得到一个都不返回。
>>>RS . fetchone()#由于前一个 fetchone()返回 None，所以获取更多会导致结果关闭异常
 Traceback(最近一次调用 last): 
 File " "，第 1 行，在
File "/Users/xiaonuogantan/python 2-workspace/lib/python 2.7/site-packages/sqlalchemy/engine/result . py "，第 790 行，在 fetchone 
 self.cursor，self . context)
File "/Users/Users_ fetchone _ impl()
File "/Users/xiaonuogantan/python 2-workspace/lib/python 2.7/site-packages/sqlalchemy/engine/result . py "，第 700 行，in _fetchone_impl 
 self。_ non _ result()
File "/Users/xiaonuogantan/python 2-workspace/lib/python 2.7/site-packages/sqlalchemy/engine/result . py "，第 724 行，in _non_result 
 raise exc。ResourceClosedError("此结果对象已关闭。")
sqlalchemy . exc . resourceclosederror:此结果对象已关闭。
>>>find _ John = select([employee . id])。其中(employee . department _ id = = 1)
>>>RS = s . execute(find _ John)
> > > rs.fetchone() #员工约翰的 ID 
 (1)，
>>>RS . fetchone()

```

由于表达式语言提供了模仿后端中立 SQL 的低级 Python 结构，这感觉上几乎等同于以 Python 方式编写实际的 SQL。

## 部门和员工之间的多对多

在我们之前的例子中，很简单，一个雇员最多属于一个部门。如果一名员工可能属于多个部门会怎样？难道一个外键不足以代表这种关系吗？

是的，一个外键是不够的。为了对`department`和`employee`之间的多对多关系建模，我们创建了一个新的关联表，它有两个外键，一个指向“department.id ”,另一个指向“employee.id”。

```py

>>> from sqlalchemy import Column, String, Integer, ForeignKey

>>> from sqlalchemy.orm import relationship, backref

>>> from sqlalchemy.ext.declarative import declarative_base

>>>

>>>

>>> Base = declarative_base()

>>>

>>>

>>> class Department(Base):

...     __tablename__ = 'department'

...     id = Column(Integer, primary_key=True)

...     name = Column(String)

...     employees = relationship('Employee', secondary='department_employee')

...

>>>

>>> class Employee(Base):

...     __tablename__ = 'employee'

...     id = Column(Integer, primary_key=True)

...     name = Column(String)

...     departments = relationship('Department', secondary='department_employee')

...

>>>

>>> class DepartmentEmployee(Base):

...     __tablename__ = 'department_employee'

...     department_id = Column(Integer, ForeignKey('department.id'), primary_key=True)

...     employee_id = Column(Integer, ForeignKey('employee.id'), primary_key=True)

...

>>> from sqlalchemy import create_engine

>>> engine = create_engine('sqlite:///')

>>> from sqlalchemy.orm import sessionmaker

>>> session = sessionmaker()

>>> session.configure(bind=engine)

>>> Base.metadata.create_all(engine)

>>>

>>> s = session()

>>> john = Employee(name='john')

>>> s.add(john)

>>> it_department = Department(name='IT')

>>> it_department.employees.append(john)

>>> s.add(it_department)

>>> s.commit()

```

在前面的例子中，我们创建了一个带有两个外键的关联表。这个关联表' department_employee '链接' department '和' employee ',关系`Department.employees`和`Employee.departments`是表之间的多对多映射。请注意，实现这一点的“魔术”是我们传递给`Department`和`Employee`模型类中的`relationship()`函数的参数`secondary`。

我们可以使用以下查询测试我们的设置:

```py

>>> john = s.query(Employee).filter(Employee.name == 'john').one()

>>> john.departments

[]

>>> john.departments[0].name

u'IT'

>>> it = s.query(Department).filter(Department.name == 'IT').one()

>>> it.employees

[]

>>> it.employees[0].name

u'john'

```

现在让我们在数据库中再插入一名员工和另一个部门:

```py

>>> marry = Employee(name='marry')

>>> financial_department = Department(name='financial')

>>> financial_department.employees.append(marry)

>>> s.add(marry)

>>> s.add(financial_department)

>>> s.commit()

```

要查找 IT 部门的所有员工，我们可以用 ORM 编写:

```py

>>> s.query(Employee).filter(Employee.departments.any(Department.name == 'IT')).one().name

u'john'

```

或者表达语言:

```py

>>> find_employees = select([DepartmentEmployee.employee_id]).select_from(Department.__table__.join(DepartmentEmployee)).where(Department.name == 'IT')

>>> rs = s.execute(find_employees)

>>> rs.fetchone()

(1,)

>>> rs.fetchone()

```

现在，让我们将员工 marry 分配到 IT 部门，这样她将属于两个部门。

```py

>>> s.refresh(marry)

>>> s.refresh(it)

>>> it.employees

[]

>>> it.employees.append(marry)

>>> s.commit()

>>> it.employees

[, ]

```

为了找到 marry，即属于至少两个部门的所有雇员，我们在 ORM 查询中使用`group_by`和`having`:

```py

>>> from sqlalchemy import func

>>> s.query(Employee).join(Employee.departments).group_by(Employee.id).having(func.count(Department.id) > 1).one().name

```

类似于 ORM 查询，我们也可以在表达式语言查询中使用`group_by`和`having`:

```py

>>> find_marry = select([Employee.id]).select_from(Employee.__table__.join(DepartmentEmployee)).group_by(Employee.id).having(func.count(DepartmentEmployee.department_id) > 1)

>>> s.execute(find_marry)
> > > RS = _
>>>RS . fetchall()
[(2，)] 

```

当然，一定要记得在完成后关闭数据库会话。

```py

>>> s.close()

```

## 总结和提示

在本文中，我们使用了一个带有两个主表和一个关联表的示例数据库来演示如何用 SQLAlchemy 的 ORM 和表达式语言编写查询。作为一个精心设计的 API，编写查询就像编写普通的 Python 代码一样简单。由于表达式语言提供了比 ORM 更低级的 API，所以用表达式语言编写查询感觉更像是用 DBAPI(如 psycopg2 和 Python-MySQL)编写查询。然而，低级 API 提供的表达式语言比 ORM 更灵活，其查询可以映射到 Python 中的`selectable` SQL 视图，这在我们的查询变得越来越复杂时非常有用。在以后的文章中，我们将进一步探索如何利用表达式语言使编写复杂的查询变得愉快而不是痛苦。