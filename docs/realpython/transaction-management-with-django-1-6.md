# 用 Django 1.6 进行交易管理

> 原文：<https://realpython.com/transaction-management-with-django-1-6/>

如果您曾经花了很多时间在 Django 数据库事务管理上，您就会知道这有多么令人困惑。在过去，文档提供了相当多的深度，但是只有通过构建和实验才能理解。

有太多的[装修工](https://realpython.com/primer-on-python-decorators/)可以一起工作，比如`commit_on_success`、`commit_manually`、`commit_unless_managed`、`rollback_unless_managed`、`enter_transaction_management`、`leave_transaction_management`等等。幸运的是，有了 Django 1.6，这一切都不复存在了。你现在真的只需要知道几个函数。我们一会儿就会谈到这些。首先，我们将讨论以下主题:

*   **什么是事务管理？**
*   Django 1.6 之前的事务管理有什么问题？

在进入之前:

*   **Django 1.6 中关于事务管理的正确之处是什么？**

然后处理一个详细的例子:

*   **条纹示例**
*   **交易**
*   **推荐方式**
*   **使用装饰器**
*   **每个 HTTP 请求的事务**
*   **保存点**
*   **嵌套事务**

## 什么是交易？

根据 [SQL-92](http://www.contrib.andrew.cmu.edu/~shadow/sql/sql1992.txt) ，“一个 SQL 事务(有时简称为“事务”)是 SQL 语句的一个执行序列，它在恢复方面是原子的”。换句话说，所有的 [SQL](https://realpython.com/python-sql-libraries/) 语句被一起执行和提交。同样，回滚时，所有语句一起回滚。

例如:

```py
# START
note = Note(title="my first note", text="Yay!")
note = Note(title="my second note", text="Whee!")
address1.save()
address2.save()
# COMMIT
```

所以事务是数据库中的一个工作单元。并且该单个工作单元由开始事务和随后的提交或显式回滚来划分。

[*Remove ads*](/account/join/)

## Django 1.6 之前的事务管理有什么问题？

为了全面回答这个问题，我们必须解决在数据库、客户程序库和 Django 中如何处理事务。

### 数据库

数据库中的每条语句都必须在事务中运行，即使事务只包含一条语句。

大多数数据库都有一个`AUTOCOMMIT`设置，默认情况下通常设置为 True。这个`AUTOCOMMIT`包装事务中的每个语句，如果语句成功，就立即提交该语句。当然，你可以手动调用类似于`START_TRANSACTION`的东西，这将暂时中止`AUTOCOMMIT`，直到你调用`COMMIT_TRANSACTION`或`ROLLBACK`。

*然而，这里的要点是`AUTOCOMMIT`设置在每个语句*之后应用隐式提交。

### 客户端库

然后是 Python **客户端库**，比如 sqlite3 和 mysqldb，它们允许 Python 程序与数据库本身接口。这种库遵循一套关于如何访问和查询数据库的标准。该标准 DB API 2.0 在 [PEP 249](http://www.python.org/dev/peps/pep-0249/) 中有描述。虽然这可能会导致一些稍微枯燥的阅读，但重要的是 PEP 249 声明数据库`AUTOCOMMIT`在默认情况下应该*关闭*。

这显然与数据库中发生的事情相冲突:

*   SQL 语句总是必须在事务中运行，数据库通常通过`AUTOCOMMIT`为您打开事务。
*   但是，根据 PEP 249，这种情况应该不会发生。
*   客户端库必须反映数据库中发生的事情，但是由于默认情况下它们不允许打开`AUTOCOMMIT`,所以它们只是将 SQL 语句包装在一个事务中，就像数据库一样。

好吧。多陪我一会儿。

### 姜戈

进入[姜戈](https://realpython.com/get-started-with-django-1/)。 **Django** 对于交易管理也有话要说。在 Django 1.5 和更早的版本中，Django 基本上运行一个打开的事务，并在您向数据库写入数据时自动提交该事务。所以每次您调用类似于`model.save()`或`model.update()`的东西时，Django 都会生成适当的 SQL 语句并提交事务。

同样在 Django 1.5 和更早的版本中，建议使用`TransactionMiddleware`将事务绑定到 HTTP 请求。每个请求都有一个事务。如果响应返回没有异常，Django 将提交事务，但是如果您的视图函数抛出错误，将调用`ROLLBACK`。这实际上关闭了`AUTOCOMMIT`。如果您想要标准的、数据库级的自动提交风格的事务管理，您必须自己管理事务——通常通过在您的视图函数上使用事务装饰器，比如`@transaction.commit_manually`或`@transaction.commit_on_success`。

深呼吸。或者两个。

### 这是什么意思？

是的，那里有很多正在进行的事情，结果是大多数开发人员只是想要标准的数据库级自动提交——这意味着事务留在幕后，做他们的事情，直到您需要手动调整它们。

## Django 1.6 中关于事务管理的哪些内容是正确的？

现在，欢迎来到 Django 1.6。尽最大努力忘记我们刚刚谈到的一切，只需记住在 Django 1.6 中，您使用数据库`AUTOCOMMIT`并在需要时手动管理事务。本质上，我们有一个简单得多的模型，它基本上完成了数据库最初设计的功能。

理论够了。我们编码吧。

[*Remove ads*](/account/join/)

## 条纹示例

这里我们有这个示例视图函数，它处理注册用户并调用 Stripe 进行信用卡处理。

```py
def register(request):
    user = None
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():

            customer = Customer.create("subscription",
              email = form.cleaned_data['email'],
              description = form.cleaned_data['name'],
              card = form.cleaned_data['stripe_token'],
              plan="gold",
            )

            cd = form.cleaned_data
            try:
                user = User.create(cd['name'], cd['email'], cd['password'],
                   cd['last_4_digits'])

                if customer:
                    user.stripe_id = customer.id
                    user.save()
                else:
                    UnpaidUsers(email=cd['email']).save()

            except IntegrityError:
                form.addError(cd['email'] + ' is already a member')
            else:
                request.session['user'] = user.pk
                return HttpResponseRedirect('/')

    else:
      form = UserForm()

    return render_to_response(
        'register.html',
        {
          'form': form,
          'months': range(1, 12),
          'publishable': settings.STRIPE_PUBLISHABLE,
          'soon': soon(),
          'user': user,
          'years': range(2011, 2036),
        },
        context_instance=RequestContext(request)
    )
```

这个视图首先调用`Customer.create`，它实际上调用 Stripe 来处理信用卡处理。然后我们创建一个新用户。如果我们从 Stripe 得到响应，我们就用`stripe_id`更新新创建的客户。如果我们找不到客户(Stripe 已关闭),我们将在新创建的客户电子邮件的`UnpaidUsers`表中添加一个条目，这样我们可以要求他们稍后重试他们的信用卡详细信息。

这个想法是，即使 Stripe 关闭了，用户仍然可以注册并开始使用我们的网站。我们稍后会再次向他们询问信用卡信息。

> 我知道这可能是一个有点做作的例子，如果必须的话，我不会用这种方式实现这样的功能，但目的是演示事务。

前进。考虑事务，记住 Django 1.6 默认为我们的数据库提供了`AUTOCOMMIT`行为，让我们再看一下数据库相关的代码。

```py
cd = form.cleaned_data
try:
    user = User.create(
        cd['name'], cd['email'], 
        cd['password'], cd['last_4_digits'])

    if customer:
        user.stripe_id = customer.id
        user.save()
    else:
        UnpaidUsers(email=cd['email']).save()

except IntegrityError:
    # ...
```

你能发现任何问题吗？嗯，如果`UnpaidUsers(email=cd['email']).save()`行失败了会发生什么？

你将有一个用户，在系统中注册，系统认为已经验证了他们的信用卡，但实际上他们并没有验证信用卡。

我们只想要两种结果中的一种:

1.  用户被创建(在数据库中)并有一个`stripe_id`。
2.  用户被创建(在数据库中)并且没有一个`stripe_id`，并且在`UnpaidUsers`表中生成一个具有相同电子邮件地址的相关行。

这意味着我们希望两个独立的数据库语句要么都提交，要么都回滚。谦逊交易的完美案例。

首先，[让我们编写一些测试](https://realpython.com/python-testing/)来验证事情是否按照我们想要的方式运行。

```py
@mock.patch('payments.models.UnpaidUsers.save', side_effect = IntegrityError)
def test_registering_user_when_strip_is_down_all_or_nothing(self, save_mock):

    #create the request used to test the view
    self.request.session = {}
    self.request.method='POST'
    self.request.POST = {'email' : 'python@rocks.com',
                         'name' : 'pyRock',
                         'stripe_token' : '...',
                         'last_4_digits' : '4242',
                         'password' : 'bad_password',
                         'ver_password' : 'bad_password',
                        }

    #mock out stripe  and ask it to throw a connection error
    with mock.patch('stripe.Customer.create', side_effect =
                    socket.error("can't connect to stripe")) as stripe_mock:

        #run the test
        resp = register(self.request)

        #assert there is no record in the database without stripe id.
        users = User.objects.filter(email="python@rocks.com")
        self.assertEquals(len(users), 0)

        #check the associated table also didn't get updated
        unpaid = UnpaidUsers.objects.filter(email="python@rocks.com")
        self.assertEquals(len(unpaid), 0)
```

测试顶部的装饰器是一个模拟，当我们试图保存到`UnpaidUsers`表时，它将[抛出一个‘integrity error’](https://realpython.com/python-exceptions/)。

这是为了回答“如果`UnpaidUsers(email=cd['email']).save()`线出现故障会怎么样？”下一段代码只是创建一个模拟会话，其中包含注册函数所需的适当信息。然后,`with mock.patch`迫使系统认为条带关闭……最后我们进行测试。

```py
resp = register(self.request)
```

上面的代码只是调用了我们的注册视图函数，并传入了被模仿的请求。然后，我们只需检查以确保表没有被更新:

```py
#assert there is no record in the database without stripe_id.
users = User.objects.filter(email="python@rocks.com")
self.assertEquals(len(users), 0)

#check the associated table also didn't get updated
unpaid = UnpaidUsers.objects.filter(email="python@rocks.com")
self.assertEquals(len(unpaid), 0)
```

因此，如果我们运行测试，它应该会失败:

```py
======================================================================
FAIL: test_registering_user_when_strip_is_down_all_or_nothing (tests.payments.testViews.RegisterPageTests)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "/Users/j1z0/.virtualenvs/django_1.6/lib/python2.7/site-packages/mock.py", line 1201, in patched
 return func(*args, **keywargs)
 File "/Users/j1z0/Code/RealPython/mvp_for_Adv_Python_Web_Book/tests/payments/testViews.py", line 266, in test_registering_user_when_strip_is_down_all_or_nothing
 self.assertEquals(len(users), 0)
AssertionError: 1 != 0

----------------------------------------------------------------------
```

很好。说起来似乎很可笑，但那正是我们想要的。记住:我们在这里练习 TDD。错误消息告诉我们，用户确实被存储在数据库中——这正是我们不想要的，因为他们没有付费！

拯救交易…

[*Remove ads*](/account/join/)

## 交易

在 Django 1.6 中，实际上有几种创建事务的方法。

我们来看几个。

### 推荐方式

根据 Django 1.6 [文件](https://docs.djangoproject.com/en/1.6/topics/db/transactions/):

> Django 提供了一个 API 来控制数据库事务。[…]原子性是数据库事务的定义属性。atomic 允许我们创建一个代码块，在其中保证数据库的原子性。如果代码块成功完成，更改将提交到数据库。如果出现异常，更改将被回滚。

Atomic 既可以用作装饰器，也可以用作 context_manager。因此，如果我们将它用作上下文管理器，注册函数中的代码将如下所示:

```py
from django.db import transaction

try:
    with transaction.atomic():
        user = User.create(
            cd['name'], cd['email'], 
            cd['password'], cd['last_4_digits'])

        if customer:
            user.stripe_id = customer.id
            user.save()
        else:
            UnpaidUsers(email=cd['email']).save()

except IntegrityError:
    form.addError(cd['email'] + ' is already a member')
```

注意第`with transaction.atomic()`行。该块中的所有代码都将在一个事务中执行。因此，如果我们重新运行我们的测试，他们都应该通过！请记住，事务是一个单一的工作单元，所以当`UnpaidUsers`调用失败时，上下文管理器中的所有内容都会一起回滚。

### 使用装饰器

我们还可以尝试添加 atomic 作为装饰器。

```py
@transaction.atomic():
def register(request):
    # ...snip....

    try:
        user = User.create(
            cd['name'], cd['email'], 
            cd['password'], cd['last_4_digits'])

        if customer:
            user.stripe_id = customer.id
            user.save()
        else:
                UnpaidUsers(email=cd['email']).save()

    except IntegrityError:
        form.addError(cd['email'] + ' is already a member')
```

如果我们重新运行我们的测试，它们将会失败，并出现与我们之前相同的错误。

这是为什么呢？为什么事务没有正确回滚？原因是因为`transaction.atomic`正在寻找某种异常，我们捕捉到了那个错误(即在我们的 try except 块中的`IntegrityError`，所以`transaction.atomic`从未发现它，因此标准的`AUTOCOMMIT`功能接管了它。

当然，移除 try except 将导致异常被抛出调用链，并且很可能在其他地方爆发。所以我们也不能那么做。

因此，技巧是将原子上下文管理器放在 try except 块中，这是我们在第一个解决方案中所做的。再次查看正确的代码:

```py
from django.db import transaction

try:
    with transaction.atomic():
        user = User.create(
            cd['name'], cd['email'], 
            cd['password'], cd['last_4_digits'])

        if customer:
            user.stripe_id = customer.id
            user.save()
        else:
            UnpaidUsers(email=cd['email']).save()

except IntegrityError:
    form.addError(cd['email'] + ' is already a member')
```

当`UnpaidUsers`触发`IntegrityError`时，`transaction.atomic()`上下文管理器将捕获它并执行回滚。当我们的代码在异常处理程序中执行时，(即`form.addError`行)，回滚将完成，如果需要，我们可以安全地进行数据库调用。还要注意，无论上下文管理器的最终结果如何，在`transaction.atomic()`上下文管理器之前或之后的任何数据库调用都不会受到影响。

### 每个 HTTP 请求的事务

Django 1.6(像 1.5 一样)也允许您以“每个请求一个事务”的模式操作。在这种模式下，Django 会自动将视图函数包装在一个事务中。如果函数抛出异常，Django 将回滚事务，否则将提交事务。

要设置它，你必须在数据库配置中为你想要的每个数据库设置`ATOMIC_REQUEST`为真。因此，在我们的“settings.py”中，我们进行了如下更改:

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(SITE_ROOT, 'test.db'),
        'ATOMIC_REQUEST': True,
    }
}
```

实际上，这就像你把装饰器放在我们的视图函数上一样。所以它不符合我们的目的。

然而值得注意的是，使用`ATOMIC_REQUESTS`和`@transaction.atomic`装饰器，仍然可以在这些错误从视图中抛出后捕获/处理它们。为了捕捉这些错误，你必须实现一些定制的中间件，或者你可以覆盖 [urls.hadler500](https://docs.djangoproject.com/en/dev/topics/http/urls/#error-handling) 或者制作一个[500.html 模板](https://docs.djangoproject.com/en/dev/topics/http/views/#the-500-server-error-view)。

[*Remove ads*](/account/join/)

## 保存点

尽管事务是原子性的，但它们可以进一步分解成保存点。将保存点视为部分事务。

因此，如果您有一个需要四个 SQL 语句来完成的事务，您可以在第二个语句之后创建一个保存点。一旦创建了保存点，即使第 3 或第 4 条语句失败，您也可以进行部分回滚，去掉第 3 和第 4 条语句，但保留前两条。

因此，这基本上就像将一个事务分割成更小的轻量级事务，允许您进行部分回滚或提交。

> 但是请记住，如果主事务回滚到哪里(可能是因为一个`IntegrityError`被引发而没有被捕获，那么所有保存点也将回滚)。

让我们看一个保存点如何工作的例子。

```py
@transaction.atomic()
def save_points(self,save=True):

    user = User.create('jj','inception','jj','1234')
    sp1 = transaction.savepoint()

    user.name = 'starting down the rabbit hole'
    user.stripe_id = 4
    user.save()

    if save:
        transaction.savepoint_commit(sp1)
    else:
        transaction.savepoint_rollback(sp1)
```

这里，整个函数都在一个事务中。创建新用户后，我们创建一个保存点，并获取对该保存点的引用。接下来的三个陈述-

```py
user.name = 'starting down the rabbit hole'
user.stripe_id = 4
user.save()
```

-不是现有保存点的一部分，因此它们有可能成为下一个`savepoint_rollback`或`savepoint_commit`的一部分。在使用`savepoint_rollback`的情况下，行`user = User.create('jj','inception','jj','1234')`仍然会提交给数据库，即使其余的更新不会提交。

换句话说，以下两个测试描述了保存点的工作方式:

```py
def test_savepoint_rollbacks(self):

    self.save_points(False)

    #verify that everything was stored
    users = User.objects.filter(email="inception")
    self.assertEquals(len(users), 1)

    #note the values here are from the original create call
    self.assertEquals(users[0].stripe_id, '')
    self.assertEquals(users[0].name, 'jj')

def test_savepoint_commit(self):
    self.save_points(True)

    #verify that everything was stored
    users = User.objects.filter(email="inception")
    self.assertEquals(len(users), 1)

    #note the values here are from the update calls
    self.assertEquals(users[0].stripe_id, '4')
    self.assertEquals(users[0].name, 'starting down the rabbit hole')
```

同样，在我们提交或回滚保存点之后，我们可以继续在同一个事务中工作。并且该工作不会受到前一个保存点的结果的影响。

例如，如果我们这样更新我们的`save_points`函数:

```py
@transaction.atomic()
def save_points(self,save=True):

    user = User.create('jj','inception','jj','1234')
    sp1 = transaction.savepoint()

    user.name = 'starting down the rabbit hole'
    user.save()

    user.stripe_id = 4
    user.save()

    if save:
        transaction.savepoint_commit(sp1)
    else:
        transaction.savepoint_rollback(sp1)

    user.create('limbo','illbehere@forever','mind blown',
           '1111')
```

无论调用的是`savepoint_commit`还是`savepoint_rollback`,都将成功创建“中间状态”用户。除非有其他原因导致整个事务回滚。

## 嵌套交易

除了用`savepoint()`、`savepoint_commit`和`savepoint_rollback`手动指定保存点之外，创建一个嵌套事务将自动为我们创建一个保存点，并在我们遇到错误时回滚。

进一步扩展我们的例子，我们得到:

```py
@transaction.atomic()
def save_points(self,save=True):

    user = User.create('jj','inception','jj','1234')
    sp1 = transaction.savepoint()

    user.name = 'starting down the rabbit hole'
    user.save()

    user.stripe_id = 4
    user.save()

    if save:
        transaction.savepoint_commit(sp1)
    else:
        transaction.savepoint_rollback(sp1)

    try:
        with transaction.atomic():
            user.create('limbo','illbehere@forever','mind blown',
                   '1111')
            if not save: raise DatabaseError
    except DatabaseError:
        pass
```

这里我们可以看到，在处理完保存点之后，我们使用了`transaction.atomic`上下文管理器来封装我们的“limbo”用户的创建。当调用该上下文管理器时，它实际上创建了一个保存点(因为我们已经在一个事务中了)，该保存点将在退出上下文管理器时被提交或回滚。

因此，以下两个测试描述了它们的行为:

```py
 def test_savepoint_rollbacks(self):

    self.save_points(False)

    #verify that everything was stored
    users = User.objects.filter(email="inception")
    self.assertEquals(len(users), 1)

    #savepoint was rolled back so we should have original values
    self.assertEquals(users[0].stripe_id, '')
    self.assertEquals(users[0].name, 'jj')

    #this save point was rolled back because of DatabaseError
    limbo = User.objects.filter(email="illbehere@forever")
    self.assertEquals(len(limbo),0)

def test_savepoint_commit(self):
    self.save_points(True)

    #verify that everything was stored
    users = User.objects.filter(email="inception")
    self.assertEquals(len(users), 1)

    #savepoint was committed
    self.assertEquals(users[0].stripe_id, '4')
    self.assertEquals(users[0].name, 'starting down the rabbit hole')

    #save point was committed by exiting the context_manager without an exception
    limbo = User.objects.filter(email="illbehere@forever")
    self.assertEquals(len(limbo),1)
```

所以实际上，您可以使用`atomic`或`savepoint`在事务中创建保存点。使用`atomic`,您不必担心提交/回滚，就像使用`savepoint`一样，您可以完全控制何时提交/回滚。

[*Remove ads*](/account/join/)

## 结论

如果您以前使用过 Django 事务的早期版本，您会发现事务模型要简单得多。另外，默认打开`AUTOCOMMIT`是 Django 和 Python 都引以为豪的“正常”默认的一个很好的例子。对于许多系统来说，你不需要直接处理事务，让`AUTOCOMMIT`做它的工作就行了。但是如果你这样做了，希望这篇文章能给你提供像专家一样管理 Django 交易所需的信息。*****