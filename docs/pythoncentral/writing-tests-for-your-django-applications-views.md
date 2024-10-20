# 为 Django 应用程序的视图编写测试

> 原文：<https://www.pythoncentral.io/writing-tests-for-your-django-applications-views/>

## 测试 Django 应用程序的视图

在我们的[上一篇文章](https://www.pythoncentral.io/writing-automated-tests-for-your-first-django-application/ "Writing Automated Tests for Your First Django Application")中，我们学习了如何为 Django 应用程序编写自动化测试，包括编写一个简单的测试来验证模型方法`m.Post.recent_posts()`的行为，并修复方法`recent_posts()`返回未来帖子的错误。

在本文中，我们将学习如何为像`myblog.views.post_detail(request, post_id)`这样的视图编写测试。为了做到这一点，我们将利用 Django 的测试客户端来模拟用户对视图执行操作。

## Django 测试客户端

在接下来的代码片段中，我们将使用`django.test.utils.setup_test_environment`和`django.test.client.Client`来模拟用户与网站视图的交互。

```py

>>> from django.test.utils import setup_test_environment

>>> # Set up a test environment so that response.context becomes available

>>> setup_test_environment()

>>> from django.test.client import Client

>>> client = Client()

>>> from django.test.client import Client

>>> client = Client()

>>> # GET the root path of our server

>>> response = client.get('/')

>>> # Inspect the status_code of our server's response

>>> response.status_code

200

>>> # Print the content of our server's response

>>> print(response.content)
过去两天没有帖子。
>>>从 myblog 导入 models as m
>>>from django . utils 导入 timezone 
 > > > #创建一个新的帖子以便下一个响应返回它
>>>m . Post . objects . Create(content = ' New Post '，created _ at = time zone . now())
>>>
>>>#打印我们服务器的新响应的内容，其中包括新的内容
 '\n

```

*   \n [新帖子](https://www.pythoncentral.io/post/1/)\ n2013 年 8 月 14 日晚上 9:12\ n

\n \n

\n\n'

在前面的代码片段中，`response.context`是一个字典，包含了与 Django 服务器的当前请求-响应生命周期相关的所有键和值的信息。除非我们调用`setup_test_environment()`使当前 shell 成为测试环境，否则它不可用。这种设置测试环境的调用在`tests.py`中是不必要的，因为 Django 已经在`python manage.py test`中构建好了。

## 编写测试代码以验证 Post 详细视图

我们现有的帖子详细信息视图根据 URL 内部传递的 *Post.id* 值返回一个帖子:

```py

# myblog/urls.py

urlpatterns = patterns('',

...

    url(r'^post/(?P\d+)/detail.html$',

        'myblog.views.post_detail', name='post_detail'),

...
# myblog/views.py 在 post/1/detail . html
def post _ detail(request，post _ id):
try:
post = m . post . objects . get(PK = post _ id)
除 m . post . doesnotexist:
raise http 404
return render(request，' post/detail.html '，{'post': post}) 

```

现在，我们可以编写测试用例来验证当 URL 中提供了`post_id`时,`post_detail`视图确实返回了一篇文章，并且当文章不存在时引发 404 错误。将以下代码插入`myblog/tests.py`:

```py

import sys

from django.core.urlresolvers import reverse

# The function 'reverse' resolves a view name and its arguments into a path

# which can be passed into the method self.client.get(). We use the 'reverse'

# method here in order to avoid writing hard-coded URLs inside tests.
类 PostDetailViewTests(test case):
def setUp(self):
super(PostDetailViewTests，self)。setUp()
self . Post = m . Post . objects . create(content = ' New Post '，created_at=timezone.now())
def tear down(self):
super(PostDetailViewTests，self)。tearDown() 
 self.post.delete()
def test _ post_detail _ success(self):
response = self . client . get(reverse(' Post _ detail '，args=(self.post.id，))
# Assert self . Post 实际上是由 Post _ detail 视图返回的
self . Assert equal(response . status _ code，200)
self . Assert contains(response，' New Post ')
def test _ post_detail _ 404(self):
response = self . client . get(reverse(' post _ detail '，args=(sys.maxint，))
 #断言 post _ detail 视图为不存在的帖子返回 404
self . Assert equal(response . status _ code，404) 

```

## 测试 Django 视图摘要

在本文中，我们学习了如何为 Django 应用程序中的视图编写自动化测试。我们利用 Django 的测试客户端来模拟对服务器的`GET`请求，并检查返回的响应，以确保它们满足不同的用例。