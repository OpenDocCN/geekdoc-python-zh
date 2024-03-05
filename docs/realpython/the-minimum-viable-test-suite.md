# 最小可行测试套件

> 原文：<https://realpython.com/the-minimum-viable-test-suite/>

在上一篇[帖子](https://realpython.com/handling-email-confirmation-in-flask/)中，我们详细介绍了如何在用户注册期间验证电子邮件地址。

这次，我们将添加单元和集成测试(耶！)添加到我们使用[烧瓶测试](http://pythonhosted.org/Flask-Testing/)扩展的应用程序中，涵盖了最重要的特性。这种类型的测试被称为最小可行测试(或[基于风险的测试](http://en.wikipedia.org/wiki/Risk-based_testing))，旨在围绕应用程序的特性测试高风险功能。

> 你错过了第一个邮件吗？从[项目回购](https://github.com/realpython/flask-registration/releases/tag/v1)中抓取代码，快速上手。

## 单元和集成测试–已定义

对于那些测试新手来说，[测试你的应用程序是至关重要的](https://realpython.com/python-testing/),因为“未经测试的应用程序很难改进现有的代码，未经测试的应用程序的开发人员往往会变得相当偏执。如果一个应用程序有自动化测试，你可以安全地进行修改，并立即知道是否有任何问题”([来源](http://flask.pocoo.org/docs/0.10/testing/))。

单元测试本质上是测试孤立的代码单元——即单个函数——以确保实际输出与预期输出相同。在许多情况下，由于您经常需要进行外部 API 调用或接触数据库，单元测试可能会严重依赖于模仿假数据。通过模拟测试，它们可能运行得更快，但也可能效率更低，更难维护。因此，除非万不得已，否则我们不会使用模拟；相反，我们将根据需要读写数据库。

请记住，当一个数据库在一个特定的测试中被触及时，从技术上来说，它是一个集成测试，因为测试本身并没有被隔离到一个特定的单元。此外，如果您通过 Flask 应用程序运行您的测试，使用测试助手 test [client](http://werkzeug.pocoo.org/docs/0.10/test/#werkzeug.test.Client) ，它们也被认为是集成测试。

[*Remove ads*](/account/join/)

## 开始使用

通常很难决定如何开始测试应用程序。这个问题的一个解决方案是从终端用户功能的角度来考虑你的应用:

1.  未注册用户必须注册才能访问该应用程序。
2.  用户注册后，一封确认电子邮件会发送给用户，他们被认为是“未确认”用户。
3.  未经确认的用户可以登录，但他们会立即被重定向到一个页面，提醒他们在访问应用程序之前通过电子邮件确认他们的帐户。
4.  确认后，用户可以完全访问该网站，在那里他们可以查看主页，在个人资料页面上更新他们的密码，并注销。

如开始所述，我们将编写足够的测试来覆盖这个主要功能。测试很难；我们非常清楚这一点，所以如果你只是热衷于编写一些测试，那么测试什么是最重要的。这与通过 [coverage.py](http://nedbatchelder.com/code/coverage/) 进行的覆盖测试(我们将在本系列的下一篇文章中详细介绍)一起，将使构建一个健壮的测试套件变得更加容易。

## 设置

激活 virtualenv，然后确保设置了以下环境变量:

```py
$ export APP_SETTINGS="project.config.DevelopmentConfig"
$ export APP_MAIL_USERNAME="foo"
$ export APP_MAIL_PASSWORD="bar"
```

然后运行当前的测试套件:

```py
$ python manage.py test
test_app_is_development (test_config.TestDevelopmentConfig) ... ok
test_app_is_production (test_config.TestProductionConfig) ... ok
test_app_is_testing (test_config.TestTestingConfig) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.003s

OK
```

这些测试只是测试配置和环境变量。它们应该相当简单。

为了扩展套件，我们需要从一个有组织的结构开始，以保持一切整洁。由于应用程序已经围绕蓝图构建，让我们对测试套件做同样的事情。因此，在“测试”目录中创建两个新的测试文件——*test _ main . py*和*test _ user . py*——并在每个文件中添加以下代码:

```py
import unittest
from flask.ext.login import current_user
from project.util import BaseTestCase

#
# Tests go here
#

if __name__ == '__main__':
    unittest.main()
```

> 注意:你也可以围绕测试类型来组织你的测试——单元、集成、功能等..

## 第 1 部分–主要蓝图

查看 *views.py* 文件中的代码(在“project/main”文件夹中)，以及最终用户工作流，我们可以看到我们只需要测试主路由`/`需要用户登录。所以将下面的代码添加到 *test_main.py* 中:

```py
def test_main_route_requires_login(self):
    # Ensure main route requires a logged in user.
    response = self.client.get('/', follow_redirects=True)
    self.assertTrue(response.status_code == 200)
    self.assertTemplateUsed('user/login.html')
```

这里，我们断言响应状态代码是`200`，并且使用了正确的模板。运行测试套件。所有 4 项测试都应该通过。

## 第 2 部分–用户蓝图

在这个蓝图中还有更多的东西要做，所以需要的测试要密集得多。本质上，我们需要测试视图，因此，我们将相应地分解我们的测试套件。别担心，我会指导你的。让我们创建两个类来确保我们的测试在逻辑上是分开的。

将以下代码添加到 *test_user.py* 中，这样我们就可以开始测试所需的许多功能。

```py
class TestUserForms(BaseTestCase):
    pass

class TestUserViews(BaseTestCase):
    pass
```

[*Remove ads*](/account/join/)

### 表格

拥有一个用户注册是一个基于登录的程序的核心概念，没有它，我们就有了一扇“打开的门”去解决问题。这必须按设计运行。因此，按照用户工作流程，让我们从**注册表单**开始。将这段代码添加到`TestUserForms()`类中。

```py
def test_validate_success_register_form(self):
    # Ensure correct data validates.
    form = RegisterForm(
        email='new@test.test',
        password='example', confirm='example')
    self.assertTrue(form.validate())

def test_validate_invalid_password_format(self):
    # Ensure incorrect data does not validate.
    form = RegisterForm(
        email='new@test.test',
        password='example', confirm='')
    self.assertFalse(form.validate())

def test_validate_email_already_registered(self):
    # Ensure user can't register when a duplicate email is used
    form = RegisterForm(
        email='test@user.com',
        password='just_a_test_user',
        confirm='just_a_test_user'
    )
    self.assertFalse(form.validate())
```

在这些测试中，我们确保表单根据输入的数据通过或未通过验证。将它与“项目/用户”文件夹中的 *forms.py* 文件进行比较。在最后一个测试中，我们只是从 *util.py* 文件中的`BaseTestCase`方法注册了同一个用户。

当我们测试表单时，让我们继续测试**登录表单**:

```py
def test_validate_success_login_form(self):
    # Ensure correct data validates.
    form = LoginForm(email='test@user.com', password='just_a_test_user')
    self.assertTrue(form.validate())

def test_validate_invalid_email_format(self):
    # Ensure invalid email format throws error.
    form = LoginForm(email='unknown', password='example')
    self.assertFalse(form.validate())
```

最后，让我们测试一下更改密码表单:

```py
def test_validate_success_change_password_form(self):
    # Ensure correct data validates.
    form = ChangePasswordForm(password='update', confirm='update')
    self.assertTrue(form.validate())

def test_validate_invalid_change_password(self):
    # Ensure passwords must match.
    form = ChangePasswordForm(password='update', confirm='unknown')
    self.assertFalse(form.validate())

def test_validate_invalid_change_password_format(self):
    # Ensure invalid email format throws error.
    form = ChangePasswordForm(password='123', confirm='123')
    self.assertFalse(form.validate())
```

确保添加所需的导入:

```py
from project.user.forms import RegisterForm, \
    LoginForm, ChangePasswordForm
```

然后运行测试！

```py
$ python manage.py test
test_app_is_development (test_config.TestDevelopmentConfig) ... ok
test_app_is_production (test_config.TestProductionConfig) ... ok
test_app_is_testing (test_config.TestTestingConfig) ... ok
test_main_route_requires_login (test_main.TestMainViews) ... ok
test_validate_email_already_registered (test_user.TestUserForms) ... ok
test_validate_invalid_change_password (test_user.TestUserForms) ... ok
test_validate_invalid_change_password_format (test_user.TestUserForms) ... ok
test_validate_invalid_email_format (test_user.TestUserForms) ... ok
test_validate_invalid_password_format (test_user.TestUserForms) ... ok
test_validate_success_change_password_form (test_user.TestUserForms) ... ok
test_validate_success_login_form (test_user.TestUserForms) ... ok
test_validate_success_register_form (test_user.TestUserForms) ... ok

----------------------------------------------------------------------
Ran 12 tests in 1.656s
```

对于表单测试，我们基本上只是实例化表单并调用 validate 函数，该函数将触发所有验证，包括我们的自定义验证，并返回一个[布尔值](https://realpython.com/python-or-operator/)，指示表单数据是否确实有效。

测试完表单后，让我们继续查看视图…

### 视图

登录和查看档案是安全的关键部分，所以我们要确保这是彻底的测试。

`login`:

```py
def test_correct_login(self):
    # Ensure login behaves correctly with correct credentials.
    with self.client:
        response = self.client.post(
            '/login',
            data=dict(email="test@user.com", password="just_a_test_user"),
            follow_redirects=True
        )
        self.assertTrue(response.status_code == 200)
        self.assertTrue(current_user.email == "test@user.com")
        self.assertTrue(current_user.is_active())
        self.assertTrue(current_user.is_authenticated())
        self.assertTemplateUsed('main/index.html')

def test_incorrect_login(self):
    # Ensure login behaves correctly with incorrect credentials.
    with self.client:
        response = self.client.post(
            '/login',
            data=dict(email="not@correct.com", password="incorrect"),
            follow_redirects=True
        )
        self.assertTrue(response.status_code == 200)
        self.assertIn(b'Invalid email and/or password.', response.data)
        self.assertFalse(current_user.is_active())
        self.assertFalse(current_user.is_authenticated())
        self.assertTemplateUsed('user/login.html')
```

`profile`:

```py
def test_profile_route_requires_login(self):
    # Ensure profile route requires logged in user.
    self.client.get('/profile', follow_redirects=True)
    self.assertTemplateUsed('user/login.html')
```

添加所需的导入:

```py
from project import db
from project.models import User
```

`register`和`resend_confirmation`:

在编写测试来覆盖`register`和`resend_confirmation`视图之前，先看一下[代码](https://github.com/realpython/flask-registration/blob/v1/project/user/views.py)。注意我们是如何利用 *email.py* 文件中的`send_email()`函数来发送确认邮件的。我们真的想发送这封邮件吗，还是应该用一个模仿库来伪造它？即使我们发送了邮件，如果不使用 Selenium 在浏览器中调出实际的收件箱，也很难断言实际的邮件出现在虚拟收件箱中。因此，让我们模拟发送电子邮件，我们将在后续文章中处理。

`confirm/<token>`:

```py
def test_confirm_token_route_requires_login(self):
    # Ensure confirm/<token> route requires logged in user.
    self.client.get('/confirm/blah', follow_redirects=True)
    self.assertTemplateUsed('user/login.html')
```

像最后两个视图一样，这个视图的其余部分可能会被嘲笑，因为需要生成一个确认令牌。然而，我们可以使用 *token.py* 文件中的实用函数生成一个令牌，`generate_confirmation_token()`:

```py
def test_confirm_token_route_valid_token(self):
    # Ensure user can confirm account with valid token.
    with self.client:
        self.client.post('/login', data=dict(
            email='test@user.com', password='just_a_test_user'
        ), follow_redirects=True)
        token = generate_confirmation_token('test@user.com')
        response = self.client.get('/confirm/'+token, follow_redirects=True)
        self.assertIn(b'You have confirmed your account. Thanks!', response.data)
        self.assertTemplateUsed('main/index.html')
        user = User.query.filter_by(email='test@user.com').first_or_404()
        self.assertIsInstance(user.confirmed_on, datetime.datetime)
        self.assertTrue(user.confirmed)

def test_confirm_token_route_invalid_token(self):
    # Ensure user cannot confirm account with invalid token.
    token = generate_confirmation_token('test@test1.com')
    with self.client:
        self.client.post('/login', data=dict(
            email='test@user.com', password='just_a_test_user'
        ), follow_redirects=True)
        response = self.client.get('/confirm/'+token, follow_redirects=True)
        self.assertIn(
            b'The confirmation link is invalid or has expired.',
            response.data
        )
```

添加导入:

```py
import datetime
from project.token import generate_confirmation_token, confirm_token
```

然后进行测试。一个应该失败:

```py
Ran 18 tests in 4.666s

FAILED (failures=1)
```

本次测试失败:`test_confirm_token_route_invalid_token()`。为什么？因为视图中有一个错误:

```py
@user_blueprint.route('/confirm/<token>')
@login_required
def confirm_email(token):
    try:
        email = confirm_token(token)
    except:
        flash('The confirmation link is invalid or has expired.', 'danger')
    user = User.query.filter_by(email=email).first_or_404()
    if user.confirmed:
        flash('Account already confirmed. Please login.', 'success')
    else:
        user.confirmed = True
        user.confirmed_on = datetime.datetime.now()
        db.session.add(user)
        db.session.commit()
        flash('You have confirmed your account. Thanks!', 'success')
    return redirect(url_for('main.home'))
```

怎么了?

现在,`flash`调用——例如,`flash('The confirmation link is invalid or has expired.', 'danger')`——不会导致函数退出，所以即使令牌无效，它也会进入 if/else 并确认用户。*这就是你写测试的原因。*

让我们重写函数:

```py
@user_blueprint.route('/confirm/<token>')
@login_required
def confirm_email(token):
    if current_user.confirmed:
        flash('Account already confirmed. Please login.', 'success')
        return redirect(url_for('main.home'))
    email = confirm_token(token)
    user = User.query.filter_by(email=current_user.email).first_or_404()
    if user.email == email:
        user.confirmed = True
        user.confirmed_on = datetime.datetime.now()
        db.session.add(user)
        db.session.commit()
        flash('You have confirmed your account. Thanks!', 'success')
    else:
        flash('The confirmation link is invalid or has expired.', 'danger')
    return redirect(url_for('main.home'))
```

再次运行测试。18 个都应该通过。

如果令牌过期会发生什么？写一个测试。

```py
def test_confirm_token_route_expired_token(self):
    # Ensure user cannot confirm account with expired token.
    user = User(email='test@test1.com', password='test1', confirmed=False)
    db.session.add(user)
    db.session.commit()
    token = generate_confirmation_token('test@test1.com')
    self.assertFalse(confirm_token(token, -1))
```

再次运行测试:

```py
$ python manage.py test
test_app_is_development (test_config.TestDevelopmentConfig) ... ok
test_app_is_production (test_config.TestProductionConfig) ... ok
test_app_is_testing (test_config.TestTestingConfig) ... ok
test_main_route_requires_login (test_main.TestMainViews) ... ok
test_validate_email_already_registered (test_user.TestUserForms) ... ok
test_validate_invalid_change_password (test_user.TestUserForms) ... ok
test_validate_invalid_change_password_format (test_user.TestUserForms) ... ok
test_validate_invalid_email_format (test_user.TestUserForms) ... ok
test_validate_invalid_password_format (test_user.TestUserForms) ... ok
test_validate_success_change_password_form (test_user.TestUserForms) ... ok
test_validate_success_login_form (test_user.TestUserForms) ... ok
test_validate_success_register_form (test_user.TestUserForms) ... ok
test_confirm_token_route_expired_token (test_user.TestUserViews) ... ok
test_confirm_token_route_invalid_token (test_user.TestUserViews) ... ok
test_confirm_token_route_requires_login (test_user.TestUserViews) ... ok
test_confirm_token_route_valid_token (test_user.TestUserViews) ... ok
test_correct_login (test_user.TestUserViews) ... ok
test_incorrect_login (test_user.TestUserViews) ... ok
test_profile_route_requires_login (test_user.TestUserViews) ... ok

----------------------------------------------------------------------
Ran 19 tests in 5.306s

OK
```

[*Remove ads*](/account/join/)

## 反射

这可能是停下来反思的好时机，尤其是因为我们关注的是最小测试。还记得我们的核心特征吗？

1.  未注册用户必须注册才能访问该应用程序。
2.  用户注册后，会发送一封确认邮件——他们被认为是“未确认”用户。
3.  未经确认的用户可以登录，但他们会立即被重定向到一个页面，提醒他们在访问应用程序之前通过电子邮件确认他们的帐户。
4.  确认后，用户可以完全访问该网站，在那里他们可以查看主页，在个人资料页面上更新他们的密码，并注销。

我们是否涵盖了所有这些内容？让我们看看:

> *“未注册用户必须注册才能访问应用程序”*:

*   `test_main_route_requires_login`
*   `test_validate_email_already_registered`
*   `test_validate_invalid_email_format`
*   `test_validate_invalid_password_format`
*   `test_validate_success_register_form`

> *“用户注册后，会发送一封确认电子邮件，他们被视为‘未确认’用户”*
> 
> 并且:
> 
> *“未经确认的用户可以登录，但他们会立即被重定向到一个页面，提醒他们在访问应用程序之前通过电子邮件确认帐户”*:

*   `test_validate_success_login_form`
*   `test_confirm_token_route_expired_token`
*   `test_confirm_token_route_invalid_token`
*   `test_confirm_token_route_requires_login`
*   `test_confirm_token_route_valid_token`
*   `test_correct_login`
*   `test_incorrect_login`
*   `test_profile_route_requires_login`

> *“确认后，用户可以完全访问该网站，在那里他们可以查看主页，在个人资料页面上更新密码，并注销”*:

*   `test_validate_invalid_change_password`
*   `test_validate_invalid_change_password_format`
*   `test_validate_success_change_password_form`

在上面的测试中，我们直接测试了表单，然后*还*为视图创建了测试(这使用了*很多*与表单测试相同的代码)。这种方法的利弊是什么？当我们进行覆盖测试时，我们会解决这个问题。

## 下次

这个帖子到此为止。在接下来的几篇文章中，我们将-

1.  模拟`user`蓝图中的以下全部或部分功能，以完成单元/集成测试- `register()`和`resend_confirmation()`
2.  通过 [coverage.py](http://nedbatchelder.com/code/coverage/) 添加覆盖测试，以帮助确保我们的代码库得到充分的测试。
3.  通过使用 Selenium 添加功能测试来扩展测试套件。

测试愉快！***