# 使用 Flask-Login 通过 Flask 进行用户管理

> 原文：<https://realpython.com/using-flask-login-for-user-management-with-flask/>

*以下是[杰夫·克努普](http://www.jeffknupp.com/)、[编写惯用 Python](http://www.jeffknupp.com/writing-idiomatic-python-ebook/) 的作者的客座博文。杰夫目前正在 Kickstarter 上开展一个[活动，将这本书改编成视频系列——看看吧！](https://www.kickstarter.com/projects/1219760486/a-writing-idiomatic-python-video-series-watch-and)*

* * *

几个月前，我厌倦了用来卖书的数字商品支付服务，决定自己写书。两个小时后，[公牛](http://www.github.com/jeffknupp/bull/)诞生了。这是一个使用 Flask 和 Python 编写的小应用程序，它被证明是实现的一个极好的选择。它从最基本的功能开始:客户可以在一个 Stripe JavaScript 弹出窗口中输入他们的详细信息，`bull`会记录他们的电子邮件地址，并为购买创建一个唯一的 id，然后将用户与他们购买的内容相关联。

它工作得非常好。而在此之前，潜在客户不仅要输入他们的全名和地址(这两个我都没用过)，他们还必须*在我的支付处理器网站*上创建一个账户。我不确定由于复杂的结账过程我损失了多少销售额，但我确定这是一笔好交易。在 bull 上，从点击图书销售页面上的“立即购买”按钮到真正阅读图书的时间大约是 10 秒钟。顾客喜欢它。

我也很喜欢它，但原因略有不同:由于`bull`在我的网络服务器上运行，我可以获得比让客户到第三方网站付款更丰富的分析。这为一系列新的可能性打开了大门:A/B 测试、分析报告、定制销售报告。我很兴奋。

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

## 添加用户

我决定，至少，我希望`bull`能够显示一个“销售概述”页面，其中包含基本的销售数据:交易信息、一段时间内的销售图表等。为了做到这一点(以安全的方式)，我需要向我的小 Flask 应用程序添加身份验证和授权。不过，有益的是，我只需要支持一个被授权查看报告的*单个*“管理员”用户。

幸运的是，通常情况下，已经有一个第三方包来处理这个问题。 [Flask-login](https://flask-login.readthedocs.org/en/latest/) 是一个 Flask 扩展，支持用户认证。所需要的只是一个`User`模型和一些简单的函数。让我们看看需要什么。

[*Remove ads*](/account/join/)

### `User`型号

`bull`已经在使用 [Flask-sqlalchemy](http://pythonhosted.org/Flask-SQLAlchemy/) 来创建`purchase`和`product`模型，分别捕获关于销售和产品的信息。Flask-login 需要一个具有以下属性的`User`模型:

*   有一个`is_authenticated()`方法，如果用户提供了有效的凭证，该方法将返回`True`
*   有一个`is_active()`方法，如果用户的帐户是活动的，该方法返回`True`
*   有一个`is_anonymous()`方法，如果当前用户是匿名用户，该方法返回`True`
*   有一个`get_id()`方法，给定一个`User`实例，该方法返回该对象的唯一 ID

虽然 Flask-login 提供了一个`UserMixin`类，该类提供了所有这些的默认实现，但我只是像这样定义了所有需要的东西:

```py
from flask.ext.sqlalchemy import SQLAlchemy

db = SQLAlchemy()

#...

class User(db.Model):
    """An admin user capable of viewing reports.

 :param str email: email address of user
 :param str password: encrypted password for the user

 """
    __tablename__ = 'user'

    email = db.Column(db.String, primary_key=True)
    password = db.Column(db.String)
    authenticated = db.Column(db.Boolean, default=False)

    def is_active(self):
        """True, as all users are active."""
        return True

    def get_id(self):
        """Return the email address to satisfy Flask-Login's requirements."""
        return self.email

    def is_authenticated(self):
        """Return True if the user is authenticated."""
        return self.authenticated

    def is_anonymous(self):
        """False, as anonymous users aren't supported."""
        return False
```

### `user_loader`

Flask-login 还要求您定义一个“user_loader”函数，给定一个用户 ID，该函数返回相关的用户对象。

**简单:**

```py
@login_manager.user_loader
def user_loader(user_id):
    """Given *user_id*, return the associated User object.

 :param unicode user_id: user_id (email) user to retrieve

 """
    return User.query.get(user_id)
```

`@login_manager.user_loader`部分告诉 Flask-login 如何加载给定 id 的用户。我把这个函数放在定义了我的所有路线的文件中，因为它就用在这里。

### `/reports`终点

现在，我可以创建一个需要认证的`/reports`端点。该端点的代码如下所示:

```py
@bull.route('/reports')
@login_required
def reports():
    """Run and display various analytics reports."""
    products = Product.query.all()
    purchases = Purchase.query.all()
    purchases_by_day = dict()
    for purchase in purchases:
        purchase_date = purchase.sold_at.date().strftime('%m-%d')
        if purchase_date not in purchases_by_day:
            purchases_by_day[purchase_date] = {'units': 0, 'sales': 0.0}
        purchases_by_day[purchase_date]['units'] += 1
        purchases_by_day[purchase_date]['sales'] += purchase.product.price
    purchase_days = sorted(purchases_by_day.keys())
    units = len(purchases)
    total_sales = sum([p.product.price for p in purchases])

    return render_template(
        'reports.html',
        products=products,
        purchase_days=purchase_days,
        purchases=purchases,
        purchases_by_day=purchases_by_day,
        units=units,
        total_sales=total_sales)
```

您会注意到大部分代码与身份验证无关，这正是它应该有的样子。由于装饰器的原因，该函数假设用户已经通过了身份验证，因此有权查看这些数据。只有一个问题:用户如何被“认证”？

### 登录和注销

当然是通过一个`/login`端点！`/login`和`/logout`都很简单，几乎可以从 Flask-login 文档中一字不差地提取出来:

```py
@bull.route("/login", methods=["GET", "POST"])
def login():
    """For GET requests, display the login form. 
 For POSTS, login the current user by processing the form.

 """
    print db
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.get(form.email.data)
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                user.authenticated = True
                db.session.add(user)
                db.session.commit()
                login_user(user, remember=True)
                return redirect(url_for("bull.reports"))
    return render_template("login.html", form=form)

@bull.route("/logout", methods=["GET"])
@login_required
def logout():
    """Logout the current user."""
    user = current_user
    user.authenticated = False
    db.session.add(user)
    db.session.commit()
    logout_user()
    return render_template("logout.html")
```

您会注意到，一旦验证了数据库中的`User`对象，我们就会更新它们。这是因为从一个请求到下一个请求，每次都会创建一个新的`User`对象实例，所以我们需要一个地方来存储用户已经验证过的信息。注销也是如此。

### 创建管理员用户

与 Django 的`manage.py`非常相似，我需要一种方法来创建一个具有正确登录凭证的管理员用户。我不能只手动向数据库添加一行，因为密码是以加盐散列的形式存储的(而不是纯文本，从安全角度来看，纯文本是愚蠢的)。为此，我创建了下面的脚本`create_user.py`:

```py
#!/usr/bin/env python
"""Create a new admin user able to view the /reports endpoint."""
from getpass import getpass
import sys

from flask import current_app
from bull import app, bcrypt
from bull.models import User, db

def main():
    """Main entry point for script."""
    with app.app_context():
        db.metadata.create_all(db.engine)
        if User.query.all():
            print 'A user already exists! Create another? (y/n):',
            create = raw_input()
            if create == 'n':
                return

        print 'Enter email address: ',
        email = raw_input()
        password = getpass()
        assert password == getpass('Password (again):')

        user = User(
            email=email, 
            password=bcrypt.generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        print 'User added.'

if __name__ == '__main__':
    sys.exit(main())
```

最后，我有了一种方法来隔离销售站点的一部分，为管理员用户显示销售数据。在我的例子中，我只需要一个用户，但是 Flask-login 显然同时支持许多用户。

[*Remove ads*](/account/join/)

## 烧瓶生态系统

我能够快速地将这个功能添加到站点中，这说明了 Flask 扩展的丰富生态系统。最近，我想创建一个 web 应用程序，其中包括一个论坛。Django 有各种各样的复杂的论坛应用程序，您可以通过大量的努力来使用它们，但是没有一个能很好地与我选择的认证应用程序一起工作；这两个应用程序没有理由耦合在一起，但却是这样。

另一方面，Flask 使得将正交应用程序组合成更大、更复杂的应用程序变得容易，就像在函数式语言中组合函数一样。以[烧瓶论坛](https://github.com/akprasad/flask-forum)为例。它在创建论坛时使用了以下 Flask 扩展:

*   flask-数据库管理管理员
*   烧瓶-用于资产管理的资产
*   用于调试和分析的 Flask-DebugToolbar。
*   论坛帖子的减价
*   基本命令的脚本
*   flask-认证安全性
*   用于数据库查询的 Flask-SQLAlchemy
*   用于表单的 Flask-WTF

有了这么长的列表，几乎令人惊讶的是，所有的应用程序都能够协同工作，而不会相互依赖(或者更确切地说，如果您来自 Django，这是令人惊讶的)，但是 Flask 扩展通常遵循“做好一件事”的 Unix 哲学。我还没有遇到过我会认为“臃肿”的烧瓶扩展。

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

## 总结

虽然我的用例及实现相当简单，但 Flask 的伟大之处在于它让简单的事情变得简单。如果某件事看起来应该很容易做，不需要花太多时间，用 Flask，这通常是真的。我能够在不到一个小时的时间里在我的支付处理器中添加一个经过认证的管理部分，而且没有任何魔法。我知道所有的东西是如何工作和组合在一起的。

这是一个强大的概念:没有魔法。虽然许多 web 框架为开发人员创建应用程序所需要做的事情如此之少而自豪，但他们没有意识到开发人员只理解自己编写和使用的内容，他们隐藏这么多内容可能对开发人员不利。Flask 将其全部公开，并在此过程中允许 Django 着手创建的应用程序的功能组合。**