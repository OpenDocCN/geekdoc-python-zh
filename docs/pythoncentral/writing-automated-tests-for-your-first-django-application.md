# 为您的第一个 Django 应用程序编写自动化测试

> 原文：<https://www.pythoncentral.io/writing-automated-tests-for-your-first-django-application/>

## 测试，测试和更多的测试

作为一名软件程序员，你经常听到别人谈论测试是任何项目中最重要的组成部分之一。当有适当的测试覆盖时，软件项目通常会成功。而当很少或没有时，它经常失败。你可能想知道:到底什么是测试？为什么大家都在不断强调它的重要性？

测试是检查代码正确性和完整性的简单程序或迷你程序。一些测试检查软件项目的微小细节- *当一个 POST 方法被调用时，一个特定的 Django 模型会被更新吗？*，而其他人检查软件的整体运行——*代码是否正确执行了我关于客户订单提交的业务逻辑？*。不管多小，每个测试都很重要，因为它告诉你你的代码是否有问题。尽管对你的代码进行 100%的测试覆盖是相当困难的，并且需要付出大量的努力，但是你应该总是尽可能地用测试覆盖你的代码。

总体而言，测试:

*   **节省您的时间**，因为它们允许您测试应用程序，而无需一直手动运行代码。
*   **帮助您验证和阐明软件需求**，因为它们迫使您思考手头的问题，并编写适当的测试来证明解决方案可行。
*   让你的代码更健壮，对其他人更有吸引力因为他们能让任何读者看到你的代码在测试中被证明是正确运行的。
*   **帮助团队一起工作**,因为他们允许队友通过编写代码测试来验证彼此的代码。

## 编写您的第一个自动化 Django 测试

在我们现有的应用程序 *myblog* 的索引视图中，我们将返回用户在不到两天后发布的最新帖子。索引视图的代码附在下面:

```py

def index(request):

    two_days_ago = datetime.utcnow() - timedelta(days=2)

    recent_posts = m.Post.objects.filter(created_at__gt=two_days_ago).all()

    context = Context({

        'post_list': recent_posts

    })

    return render(request, 'index.html', context)

```

这个视图中有一个小 bug。你能找到它吗？

似乎我们假设我们网站中的所有帖子都是过去“发布”的，即`Post.created_at`比`timezone.now()`早。然而，很有可能用户提前准备了一篇文章，并希望在未来的某个日期发布它。显然，当前代码也将返回那些未来的帖子。这可以在下面的代码片段中得到验证:

```py

>>> m.Post.objects.all().delete()

>>> import datetime

>>> from django.utils import timezone

>>> from myblog import models as m

>>> future_post = m.Post(content='Future Post',

>>>                      created_at=timezone.now() + datetime.timedelta(days=10))

>>> future_post.save()

>>> two_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=2)

>>> recent_posts = m.Post.objects.filter(created_at__gt=two_days_ago).all()

# recent_posts contain future_post, which is wrong.

>>> recent_posts[0].content

u'Future Post'

```

在我们继续修复视图中的 bug 之前，让我们暂停一下，写一个测试来暴露这个 bug。首先，我们将一个新方法`recent_posts()`添加到模型`Post`中，这样我们就可以从视图中提取不正确的代码:

```py

import datetime
从 django.db 导入模型作为 m 
从 django.utils 导入时区
类 Post(m . Model):
content = m . CharField(max _ length = 256)
created _ at = m . datetime field(' datetime created ')
@ class method
def recent _ posts(cls):
two _ days _ ago = time zone . now()-datetime . time delta(days = 2)
return post . objects . filter(created _ at _ _ gt = two _ days _ ago)

```

然后，我们修改索引视图的代码，以使用来自模型`Post`的`recent_posts()`方法:

```py

def index(request):

    recent_posts = m.Post.recent_posts()

    context = Context({

        'post_list': recent_posts

    })

    return render(request, 'index.html', context)

```

现在我们将下面的代码添加到`myblog/tests.py`中，这样我们可以运行它来测试我们代码的行为:

```py

import datetime
从 django.utils 导入时区
从 django.test 导入测试用例
从我的博客导入模型作为 m
类 PostModelTests(test case):
def setUp(self):
' ' '从未来创建帖子‘
超级(PostModelTests，self)。setUp()
self . Future _ Post = m . Post(
content = ' Future Post '，created _ at = time zone . now()+datetime . time delta(days = 10))
self . Future _ Post . save()
def tearDown(self): 
' ' '从未来删除帖子。''
超级(PostModelTests，self)。tear down()
m . Post . objects . get(content = ' Future Post ')。删除()
def test _ recent _ posts _ not _ including _ future _ posts(self):
' ' ' m . post . recent _ posts()不应该返回未来的帖子。''
recent _ posts = m . post . recent _ posts()
self . assert notin(self . future _ post，recent_posts) 

```

在这个测试用例中，我们想要验证未来的帖子不包含在从`m.Post.recent_posts()`返回的帖子列表中。现在，您可以通过以下方式运行测试:

```py

$ python manage.py test

Creating test database for alias 'default'...

....................................................

======================================================================

FAIL: test_recent_posts_not_including_future_posts (myblog.tests.PostModelTests)

m.Post.recent_posts() should not return posts from the future.

----------------------------------------------------------------------

Traceback (most recent call last):

  File "/Users/user/python2-workspace/pythoncentral/django_series/article7/myblog/myblog/tests.py", line 23, in test_recent_posts_not_including_future_posts

    self.assertNotIn(self.future_post, recent_posts)

AssertionError:  unexpectedly found in []
- 
在 11.877 秒内进行了 483 次测试
失败(失败=1，跳过=1，预期失败=1) 
销毁别名“默认”的测试数据库...

```

由于来自未来的帖子在从`recent_posts()`返回的列表中，并且我们的测试抱怨了它，我们肯定知道在我们的代码中有一个 bug。

## 修复我们的测试用例错误

我们可以通过确保在`recent_posts()`的查询中`m.Post.created_at`早于`timezone.now()`来轻松修复这个错误:

```py

class Post(m.Model):

    content = m.CharField(max_length=256)

    created_at = m.DateTimeField('datetime created')
@ class method
def recent _ posts(cls):
now = time zone . now()
two _ days _ ago = now-datetime . time delta(days = 2)
return post . objects . \
filter(created _ at _ _ gt = two _ days _ ago)。\ 
过滤器(created_at__lt=now) 

```

现在，您可以重新运行测试，它应该会在没有警告的情况下通过:

```py

$ python manage.py test

Creating test database for alias 'default'...

.................................................................................................................................................s.....................................................................................................................................x...........................................................................................................................................................................................................

----------------------------------------------------------------------

Ran 483 tests in 12.725s
OK (skipped=1，expected failures=1) 
销毁别名“default”的测试数据库...

```

## 自动化测试案例总结和提示

在本文中，我们学习了如何为我们的第一个 Django 应用程序编写自动化测试。因为编写测试是最好的软件工程实践之一，它总是有回报的。这可能看起来违背直觉，因为您必须编写更多的代码来实现相同的功能，但是测试将在将来节省您的大量时间。

当编写 Django 应用程序时，我们将测试代码放入`tests.py`中，并通过运行`$ python manage.py test`来运行它们。如果有任何测试没有通过，Django 会将错误报告给我们，这样我们就可以相应地修复任何 bug。如果所有测试都通过了，那么 Django 显示没有错误，我们可以非常自信地说我们的代码工作正常。因此，对代码进行适当的测试覆盖是编写高质量软件的最佳方式之一。