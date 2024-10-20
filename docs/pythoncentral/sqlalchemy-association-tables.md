# SQLAlchemy 关联表

> 原文：<https://www.pythoncentral.io/sqlalchemy-association-tables/>

## 关联表

在我们之前的文章中，我们使用关联表来建模表之间的`many-to-many`关系，比如`Department`和`Employee`之间的关系。在本文中，我们将更深入地研究关联表的概念，看看我们如何使用它来进一步解决更复杂的问题。

## 部门员工链接和额外数据

在上一篇文章中，我们创建了以下 SQLAlchemy 模型:

```py

import os
from sqlalchemy 导入列，DateTime，String，Integer，ForeignKey，func 
 from sqlalchemy.orm 导入关系，back ref
from sqlalchemy . ext . declarative import declarative _ base
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
department _ id = Column(Integer，ForeignKey('department.id ')，primary _ key = True)
employee _ id = Column(Integer，ForeignKey('employee.id ')，primary_key=True) 

```

请注意，`DepartmentEmployeeLink`类包含两个外键列，足以模拟`Department`和`Employee`之间的多对多关系。现在我们再添加一列`extra_data`和两个关系`department`和`employee`。

```py

import os
from sqlalchemy 导入列，DateTime，String，Integer，ForeignKey，func 
 from sqlalchemy.orm 导入关系，back ref
from sqlalchemy . ext . declarative import declarative _ base
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
_ _ tablename _ _ = ' Department _ Employee _ link '
Department _ id = Column(Integer，ForeignKey('department.id ')，primary _ key = True)
Employee _ id = Column(Integer，ForeignKey('employee.id ')，primary _ key = True)
extra _ data = Column(String(256))
Department = relationship(Department，backref = backref(" Employee _ assoc "))
Employee = relationship(Employee，
```

通过在`DepartmentEmployeeLink`关联模型上增加一个额外的列和两个额外的关系，我们可以存储更多的信息，并且可以更加自由地使用这些信息。例如，假设我们有一个在 IT 部门兼职的员工约翰，我们可以将字符串“兼职”插入到列`extra_data`中，并创建一个`DepartmentEmployeeLink`对象来表示这种关系。

```py

>>> fp = 'orm_in_detail.sqlite'

>>> if os.path.exists(fp):

...     os.remove(fp)

...

>>> from sqlalchemy import create_engine

>>> engine = create_engine('sqlite:///association_tables.sqlite')

>>>

>>> from sqlalchemy.orm import sessionmaker

>>> session = sessionmaker()

>>> session.configure(bind=engine)

>>> Base.metadata.create_all(engine)

>>>

>>>

>>> IT = Department(name="IT")

>>> John = Employee(name="John")

>>> John_working_part_time_at_IT = DepartmentEmployeeLink(department=IT, employee=John, extra_data='part-time')

>>> s = session()

>>> s.add(John_working_part_time_at_IT)

>>> s.commit()

```

然后，我们可以通过查询 IT 部门或`DepartmentEmployeeLink`模型来找到 John。

```py

>>> IT.employees[0].name

u'John'

>>> de_link = s.query(DepartmentEmployeeLink).join(Department).filter(Department.name == 'IT').one()

>>> de_link.employee.name

u'John'

>>> de_link = s.query(DepartmentEmployeeLink).filter(DepartmentEmployeeLink.extra_data == 'part-time').one()

>>> de_link.employee.name

u'John'

```

最后，使用关系`Department.employees`添加 IT 员工仍然有效，如前一篇文章所示:

```py

>>> Bill = Employee(name="Bill")

>>> IT.employees.append(Bill)

>>> s.add(Bill)

>>> s.commit()

```

## 链接与 Backref 的关系

到目前为止，我们在`relationship`定义中使用的一个常见关键字参数是`backref`。一个`backref`是将第二个`relationship()`放置到目标表上的常见快捷方式。例如，下面的代码通过在`Post.owner`上指定一个`backref`将第二个`relationship()`“帖子”放到`user`表格上:

```py

class User(Base):

    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)

    name = Column(String(256))
类 Post(Base):
_ _ tablename _ _ = ' Post '
id = Column(Integer，primary _ key = True)
owner _ id = Column(Integer，foreign key(' User . id ')
owner = relationship(User，backref = backref(' Post '，uselist=True)) 

```

这相当于以下定义:

```py

class User(Base):

    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)

    name = Column(String(256))

    posts = relationship("Post", back_populates="owner")
类 Post(Base):
_ _ tablename _ _ = ' Post '
id = Column(Integer，primary _ key = True)
owner _ id = Column(Integer，foreign key(' User . id ')
owner = relationship(User，back _ populated = " posts ")

```

现在我们在`User`和`Post`之间有了一个`one-to-many`关系。我们可以通过以下方式与这两个模型进行交互:

```py

>>> s = session()

>>> john = User(name="John")

>>> post1 = Post(owner=john)

>>> post2 = Post(owner=john)

>>> s.add(post1)

>>> s.add(post2)

>>> s.commit()

>>> s.refresh(john)

>>> john.posts

[, ]

>>> john.posts[0].owner
> > > John . posts[0]. owner . name
u ' John '

```

## 一对一

在模型之间创建`one-to-one`关系与创建`many-to-one`关系非常相似。通过在`backref()`中将`uselist`参数的值修改为`False`，我们强制数据库模型以`one-to-one`关系相互映射。

```py

class User(Base):

    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)

    name = Column(String(256))
类地址(Base):
_ _ tablename _ _ = ' Address '
id = Column(Integer，primary _ key = True)
Address = Column(String(256))
User _ id = Column(Integer，foreign key(' User . id '))
User = relationship(' User '，backref=backref('address '，uselist=False)) 

```

然后，我们可以按以下方式使用模型:

```py

>>> s = session()

>>> john = User(name="John")

>>> home_of_john = Address(address="1234 Park Ave", user=john)

>>> s.add(home_of_john)

>>> s.commit()

>>> s.refresh(john)

>>> john.address.address

u'1234 Park Ave'

>>> john.address.user.name

u'John'

>>> s.close()

```

## 关系更新级联

在关系数据库中，参照完整性保证当`one-to-many`或`many-to-many`关系中被引用对象的主键改变时，引用主键的引用对象的外键也将改变。但是，对于不支持参照完整性的数据库，如关闭了参照完整性选项的 SQLite 或 MySQL，更改被引用对象的主键值不会触发引用对象的更新。在这种情况下，我们可以使用`relationship`或`backref`中的`passive_updates`标志来通知数据库执行额外的 SELECT 和 UPDATE 语句，这些语句将更新引用对象的外键的值。

在下面的例子中，我们在`User`和`Address`之间构造了一个`one-to-many`关系，并且没有在关系中指定`passive_updates`标志。数据库后端是 SQLite。

```py

class User(Base):

    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)

    name = Column(String(256))
类地址(Base):
_ _ tablename _ _ = ' Address '
id = Column(Integer，primary _ key = True)
Address = Column(String(256))
User _ id = Column(Integer，foreign key(' User . id '))
User = relationship(
' User '，backref=backref('addresses '，uselist=True) 
 ) 

```

然后，当我们改变一个`User`对象的主键值时，它的`Address`对象的`user_id`外键值将不会改变。因此，当你想再次访问一个`address`的`user`对象时，你会得到一个`AttributeError`。

```py

>>> s = session()

>>> john = User(name='john')

>>> home_of_john = Address(address='home', user=john)

>>> office_of_john = Address(address='office', user=john)

>>> s.add(home_of_john)

>>> s.add(office_of_john)

>>> s.commit()

>>> s.refresh(john)

>>> john.id

1

>>> john.id = john.id + 1

>>> s.commit()

>>> s.refresh(home_of_john)

>>> s.refresh(office_of_john)

>>> home_of_john.user.name

Traceback (most recent call last):

  File "", line 1, in

AttributeError: 'NoneType' object has no attribute 'name'

>>> s.close()

```

如果我们在`Address`模型中指定了`passive_updates`标志，那么我们可以更改`john`的主键，并期望 SQLAlchemy 发出额外的 SELECT 和 UPDATE 语句来保持`home_of_john.user`和`office_of_john.user`是最新的。

```py

class User(Base):

    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)

    name = Column(String(256))
类地址(Base):
_ _ tablename _ _ = ' Address '
id = Column(Integer，primary _ key = True)
Address = Column(String(256))
User _ id = Column(Integer，foreign key(' User . id '))
User = relationship(
' User '，backref=backref('addresses '，uselist=True，passive_updates=False) 
 ) 

```

```py

>>> s = session()

>>> john = User(name='john')

>>> home_of_john = Address(address='home', user=john)

>>> office_of_john = Address(address='office', user=john)

>>> s.add(home_of_john)

>>> s.add(office_of_john)

>>> s.commit()

>>> s.refresh(john)

>>> john.id

1

>>> john.id = john.id + 1

>>> s.commit()

>>> s.refresh(home_of_john)

>>> s.refresh(office_of_john)

>>> home_of_john.user.name

u'john'

>>> s.close()

```

## 摘要

在本文中，我们将深入探讨 SQLAlchemy 的关联表和关键字参数`backref`。理解这两个概念背后的机制对于完全掌握复杂的连接查询通常是至关重要的，这将在以后的文章中展示。