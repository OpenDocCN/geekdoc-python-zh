# Flask 扩展系列(四)–SQLAlchemy

熟悉 Java 的朋友们一定使用过 Hibernate 或 MyBatis 吧，这类的框架称为对象关系映射 ORM 框架，它将对数据库的操作从繁琐的 SQL 语言执行简化为对象的操作。Python 中也有类似的 ORM 框架，叫 SQLAlchemy。本篇我们将介绍 Flask 中支持 SQLAlchemy 框架的第三方扩展，Flask-SQLAlchemy。

### 系列文章

*   Flask 扩展系列(一)–Restful
*   Flask 扩展系列(二)–Mail
*   Flask 扩展系列(三)–国际化 I18N 和本地化 L10N
*   Flask 扩展系列(四)–SQLAlchemy
*   Flask 扩展系列(五)–MongoDB
*   Flask 扩展系列(六)–缓存
*   Flask 扩展系列(七)–表单
*   Flask 扩展系列(八)–用户会话管理
*   Flask 扩展系列(九)–HTTP 认证
*   Flask 扩展系列–自定义扩展

### 安装和启用

在阅读此文之前，强烈建议读者先了解[SQLAlchemy 的基本知识](http://docs.sqlalchemy.org/)。

我们依然通过 pip 安装：

```py
$ pip install Flask-SQLAlchemy
```

PyPI 自动会将其所依赖的 SQLAlchemy 包装上。我们可以采用下面的方法初始化一个 Flask-SQLAlchemy 的实例：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db/users.db'
db = SQLAlchemy(app)

```

应用配置项”SQLALCHEMY_DATABASE_URI”指定了 SQLAlchemy 所要操作的数据库的连接字符串，本文中我们使用 SQLite3，连接字符串以”sqlite:///”开头，后面的”db/users.db”表示数据库文件是当前位置下 db 子目录中的”users.db”文件。

### 定义模型

一个模型即对应数据库中的一个表，这里我们来定义一个用户模型：

```py
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True)
    age = db.Column(db.Integer)

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return '<User %r>' % self.name

```

模型类必须继承”db.Model”, db 即上一节的”db = SQLAlchemy(app)”，上例中的 User 模型将自动映射到数据库中的”user”表。User 模型中定义了三个属性：

1.  “id”：整型主键
2.  “name”：最大长度为 50 的字符串，且值唯一
3.  “age”：整型

这三个属性将分别对应”user”表中”id”主键, “name”和”age”字段。写好”__init__()”和”__repr__”()方法，我们的模型就定义完成了。现在你就可以通过下面的代码来创建数据库和表：

```py
    db.create_all()

```

让我们来验证下，”user”表是否创建成功。首先打开数据库文件：

```py
$ sqlite3 db/users.db
```

查询下”user”表的 schema：

```py
sqlite> .schema user
```

你应该可以看到下面的信息：

```py
CREATE TABLE user (
	id INTEGER NOT NULL,
	name VARCHAR(50),
	age INTEGER,
	PRIMARY KEY (id),
	UNIQUE (name)
);

```

另外，你可以通过”db.drop_all()”方法删除所有的表，不过数据库文件将会被保留。

### 添加数据

数据表创建完后，让我们添加些数据进去：

```py
    db.session.add(User('Michael', 18))
    db.session.add(User('Tom', 21))
    db.session.add(User('Jane', 17))
    db.session.commit()

```

一定要记得调用”db.session.commit()”提交事务，不然数据不会保存到数据库中。我们无需指定每条记录的”id”主键值，数据库会自动使用自增的数值作为主键。

### 查询数据

每个数据模型都有”query”接口可以用来查询模型所对应的表的记录。比如，查询”user”表中的所有记录：

```py
    users = User.query.all()

```

返回的 users 是一个列表，其中每个元素都是一个 User 类型的对象，对应于”user”表中的一条记录。该方法相当于执行了 SQL 语句：

```py
SELECT * FROM user

```

“query”接口拥有丰富的方法，这里列举一些常用的：

1.  “filter_by()”方法，对查询结果过滤，参数必须是键值对”key=value”

```py
    # WHERE name='Tom'
    users = User.query.filter_by(name='Tom')
    # WHERE name='Tom' AND age=17
    users = User.query.filter_by(name='Jane', age=17)

```

效果相当于使用了 WHERE 子句，多个键值对用逗号分割。

*   “filter()”方法，对查询结果过滤，比”filter_by()”方法更强大，参数是布尔表达式

```py
    # WHERE age<20
    users = User.query.filter(User.age<20)
    # WHERE name LIKE 'J%' AND age<20
    users = User.query.filter(User.name.startswith('J'), User.age<20)

```

多个查询条件用逗号分割。

*   “first()”方法，取返回列表中的第一个元素，当我们只查询一条记录时非常有用

```py
    user = User.query.filter_by(name='Michael').first()

```

*   “order_by()”方法，排序

```py
    from sqlalchemy import desc

    # ORDER BY name
    user = User.query.order_by(User.name)
    # ORDER BY age DESC, name
    user = User.query.order_by(desc(User.age), User.name)

```

*   “limit()”和”offset()”方法，分页

```py
    # LIMIT 10 OFFSET 10
    user = User.query.limit(10).offset(10)

```

等同于 MySQL 中的 LIMIT 和 OFFSET，上例中我们从第 11 条记录开始取，并最多只取 10 条。

*   “slice(start, stop)”，分页

```py
    # LIMIT 2 OFFSET 1
    user = User.query.slice(1, 3)

```

从 start 位置开始取记录，到 stop 位置前结束。本质上来说，SQLAlchemy 会将其翻译成 LIMIT/OFFSET 语句来实现，上例中的”slice(1, 3)”等同于”LIMIT 2 OFFSET 1″。

### 更新数据

在添加数据时，我们使用了”add()”方法，其实它一样可以用来更新数据：

```py
    user = User.query.filter_by(name='Tom').first()
    if user is not None:
        user.age += 1
        db.session.add(user)
        db.session.commit()

```

SQLAlchemy 会自动判断，如果对象对应的记录已存在，就更新而不是添加。

SQLAlchemy 还支持批量更新，比如我们要将所有岁数小于 20 的人都加 1 岁：

```py
    User.query.filter(User.age<20).update({'age': User.age+1})
    db.session.commit()

```

更新完后，别忘了提交事务。

### 删除数据

只需调用”delete()”方法即可，传入的参数是对应数据库中记录的对象。记得同”add()”一样，要调用”commit()”来提交事务：

```py
    user = User.query.filter_by(name='Michael').first()
    if user is not None:
        db.session.delete(user)
        db.session.commit()

```

### 一对多关系

现在让我们再添加一个模型，成绩单。每个用户对于不同的课程，会有不同的分数，这样用户同成绩单之前就是一对多的关系。怎么在模型类的定义中体现这个一对多关系呢。保持 User 类不变，现在让我们添加一个 Score 类：

```py
from datetime import datetime

class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course = db.Column(db.String(50))
    assess_date = db.Column(db.DateTime)
    score = db.Column(db.Float)
    is_pass = db.Column(db.Boolean)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship('User', backref=db.backref('scores', lazy='dynamic'))

    def __init__(self, course, score, user, assess_date=None):
        self.course = course
        self.score = score
        self.is_pass = (score >= 60)
        if assess_date is None:
            assess_date = datetime.now()
        self.assess_date = assess_date
        self.user = user

    def __repr__(self):
        return '<Course %r of User %r>' % (self.course, self.user.name)

```

Score 模型中有这些属性：

1.  “id”：整型主键
2.  “course”：最大长度为 50 的字符串
3.  “assess_date”：日期时间类型
4.  “score”：浮点型
5.  “is_pass”：布尔型

分别对应数据库”score”表中”id”主键, “course”, “access_date”, “score”和”is_pass”字段。另外，它还有两个属性：

6.  “user_id”：整型外键，对应于”user”表的主键”id”
7.  “user”：User 对象

“user_id”字段声明了外键，也就相当于声明了”user”表同”score”表的一对多关系。”user”属性并不是数据表中的字段，它使用了”db.relationship()”方法，使得我们可以通过”Score.user”访问当前 score 记录的 user 对象，它的第一个参数”User”就表明了对应的对象模型是 User。而第二个参数”backref”定义了从 User 模型反向引用 Score 模型的方法，上例中，我们就可以用”User.scores”获取当前 user 对象所有的 score 记录，它是一个列表。”db.backref()”方法的”lazy”参数决定了在 User 对象中什么时候加载其 scores 列表的值，延迟加载可以提高性能，并避免内存的浪费，”lazy”参数的选择可以[参阅这里](http://flask-sqlalchemy.pocoo.org/2.1/models/#one-to-many-relationships)。

现在查询下”score”表的 schema，你会看到下面的结果：

```py
CREATE TABLE score (
	id INTEGER NOT NULL,
	course VARCHAR(50),
	assess_date DATETIME,
	score FLOAT,
	is_pass BOOLEAN,
	user_id INTEGER,
	PRIMARY KEY (id),
	CHECK (is_pass IN (0, 1)),
	FOREIGN KEY(user_id) REFERENCES user (id)
);

```

让我们添加些 score 记录：

```py
    user = User.query.filter_by(name='Tom').first()
    if user is not None:
        db.session.add(Score('Math', 80.5, user))
        db.session.add(Score('Politics', 58, user))
    user = User.query.filter_by(name='Jane').first()
    if user is not None:
        db.session.add(Score('Math', 88, user))
    db.session.commit()

```

然后试试通过”User.scores”查询某个用户的成绩：

```py
def scores(name):
    user = User.query.filter_by(name=name).first()
    if user is not None:
        for score in user.scores:
            print 'Name "%s" course "%s", score is %s' % (name, score.course, score.score)

```

对于多对多关系，大家可以创建一个单独的关系表，然后每个表同这个关系表都是一对多的关系。或者大家可以参考[官方文档](http://flask-sqlalchemy.pocoo.org/2.1/models/#many-to-many-relationships)上的例子来实现多对多关系。

#### 更多参考资料

[SQLAlchemy 的官方文档](http://docs.sqlalchemy.org/)
[Flask-SQLAlchemy 的官方文档](http://flask-sqlalchemy.pocoo.org/)
[Flask-SQLAlchemy 的源码](https://github.com/mitsuhiko/flask-sqlalchemy/)

本篇的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ext4.html)