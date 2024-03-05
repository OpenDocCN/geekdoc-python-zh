# Django 重定向的最终指南

> 原文：<https://realpython.com/django-redirects/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Django 重定向**](/courses/django-redirects/)

当您使用 [Django 框架](https://realpython.com/tutorials/django/)构建 Python web 应用程序时，您将不得不在某个时候将用户从一个 URL 重定向到另一个 URL。

在本指南中，您将了解到关于 HTTP 重定向以及如何在 [Django](https://realpython.com/get-started-with-django-1/) 中处理它们的所有知识。在本教程结束时，您将:

*   能够重定向用户从一个网址到另一个网址
*   了解临时重定向和永久重定向的区别
*   使用重定向时避免常见的陷阱

本教程假设您熟悉 Django 应用程序的基本构件，比如[视图](https://docs.djangoproject.com/en/2.1/topics/http/views/)和 [URL 模式](https://docs.djangoproject.com/en/2.1/topics/http/urls/)。

## Django 重定向:一个超级简单的例子

在 Django 中，通过从视图中返回一个实例`HttpResponseRedirect`或`HttpResponsePermanentRedirect`，将用户重定向到另一个 URL。最简单的方法是使用模块`django.shortcuts`中的函数 [`redirect()`](https://docs.djangoproject.com/en/2.1/topics/http/shortcuts/#redirect) 。这里有一个例子:

```py
# views.py
from django.shortcuts import redirect

def redirect_view(request):
    response = redirect('/redirect-success/')
    return response
```

在你的视图中用一个 URL 调用`redirect()`。它将返回一个`HttpResponseRedirect`类，然后从视图中返回。

与任何其他视图一样，返回重定向的视图必须添加到您的`urls.py`中:

```py
# urls.py
from django.urls import path

from .views import redirect_view

urlpatterns = [
    path('/redirect/', redirect_view)
    # ... more URL patterns here
]
```

假设这是 Django 项目的主`urls.py`，URL `/redirect/`现在重定向到`/redirect-success/`。

为了避免对 URL 进行硬编码，您可以使用视图或 URL 模式或模型的名称来调用`redirect()`,以避免对重定向 URL 进行硬编码。您还可以通过传递关键字参数`permanent=True`来创建永久重定向。

这篇文章可以到此结束，但是它很难被称为“Django 重定向的终极指南”我们一会儿将仔细研究一下`redirect()`函数，并深入了解 HTTP 状态代码和不同的`HttpRedirectResponse`类的本质细节，但是让我们后退一步，从一个基本问题开始。

[*Remove ads*](/account/join/)

## 为什么重定向

您可能想知道为什么您首先要将用户重定向到一个不同的 URL。为了理解重定向的意义，看看 Django 本身是如何将重定向合并到框架默认提供的特性中的:

*   当您没有登录并请求一个需要认证的 URL 时，比如 Django admin，Django 会将您重定向到登录页面。
*   当您成功登录时，Django 会将您重定向到您最初请求的 URL。
*   当您使用 Django admin 更改您的密码时，您会被重定向到一个页面，显示更改成功。
*   当您在 Django admin 中创建一个对象时，Django 会将您重定向到对象列表。

没有重定向的替代实现会是什么样子？如果用户必须登录才能查看页面，您可以简单地显示一个类似“单击此处登录”的页面这是可行的，但是对用户来说不方便。

像 http://bit.ly 这样的网址缩写是重定向派上用场的另一个例子:你在浏览器的地址栏中键入一个短网址，然后被重定向到一个长而笨拙的网址页面。

**注意:**如果你想建立一个你自己的网址缩写器，那么看看[用 FastAPI 和 Python 建立一个网址缩写器](https://realpython.com/build-a-python-url-shortener-with-fastapi/)。

在其他情况下，重定向不仅仅是为了方便。重定向是引导用户浏览 web 应用程序的重要工具。在执行了某种有副作用的操作(比如创建或删除一个对象)后，最好重定向到另一个 URL，以防止意外执行该操作两次。

表单处理就是使用重定向的一个例子，在表单处理中，用户在成功提交表单后被重定向到另一个 URL。下面是一个代码示例，它说明了通常如何处理表单:

```py
 1from django import forms
 2from django.http import HttpResponseRedirect
 3from django.shortcuts import redirect, render
 4
 5def send_message(name, message):
 6    # Code for actually sending the message goes here
 7
 8class ContactForm(forms.Form):
 9    name = forms.CharField()
10    message = forms.CharField(widget=forms.Textarea)
11
12def contact_view(request):
13    # The request method 'POST' indicates
14    # that the form was submitted
15    if request.method == 'POST':  # 1 16        # Create a form instance with the submitted data
17        form = ContactForm(request.POST)  # 2 18        # Validate the form
19        if form.is_valid():  # 3 20            # If the form is valid, perform some kind of
21            # operation, for example sending a message
22            send_message(
23                form.cleaned_data['name'],
24                form.cleaned_data['message']
25            )
26            # After the operation was successful,
27            # redirect to some other page
28            return redirect('/success/')  # 4 29    else:  # 5 30        # Create an empty form instance
31        form = ContactForm()
32
33    return render(request, 'contact_form.html', {'form': form})
```

该视图的目的是显示和处理允许用户发送消息的联系人表单。让我们一步一步地跟随它:

1.  首先，视图查看请求方法。当用户访问连接到这个视图的 URL 时，浏览器执行一个`GET`请求。

2.  如果用一个`POST`请求调用视图，那么`POST`数据被用来实例化一个`ContactForm`对象。

3.  如果表单有效，表单数据被传递给`send_message()`。这个函数与上下文无关，因此这里没有显示。

4.  发送消息后，视图返回一个到 URL `/success/`的重定向。这是我们感兴趣的步骤。为了简单起见，URL 在这里是硬编码的。稍后您将看到如何避免这种情况。

5.  如果视图接收到一个`GET`请求(或者，准确地说，任何不是`POST`请求的请求)，它会创建一个`ContactForm`的实例，并使用`django.shortcuts.render()`来呈现`contact_form.html`模板。

如果用户现在点击重载，只有`/success/` URL 被重载。如果没有重定向，重新加载页面会重新提交表单并发送另一条消息。

## 幕后:HTTP 重定向如何工作

现在你知道为什么重定向有意义了，但是它们是如何工作的呢？让我们快速回顾一下，当您在 web 浏览器的地址栏中输入 URL 时会发生什么。

### HTTP 快速入门

让我们假设您已经创建了一个 Django 应用程序，它有一个处理路径`/hello/`的“Hello World”视图。您正在用 Django 开发服务器运行您的应用程序，所以完整的 URL 是`http://127.0.0.1:8000/hello/`。

当您在浏览器中输入该 URL 时，它会连接到 IP 地址为[的服务器上的端口`8000`，并发送一个路径为`/hello/`的 HTTP `GET`请求。服务器回复一个 HTTP 响应。](https://realpython.com/python-ipaddress-module/)

HTTP 是基于文本的，所以在客户机和服务器之间来回查看相对容易。您可以使用带有选项`--include`的命令行工具 [`curl`](https://curl.haxx.se/docs/manpage.html) 来查看完整的 HTTP 响应，包括标头，如下所示:

```py
$ curl --include http://127.0.0.1:8000/hello/
HTTP/1.1 200 OK
Date: Sun, 01 Jul 2018 20:32:55 GMT
Server: WSGIServer/0.2 CPython/3.6.3
Content-Type: text/html; charset=utf-8
X-Frame-Options: SAMEORIGIN
Content-Length: 11

Hello World
```

如您所见，HTTP 响应以包含状态代码和状态消息的状态行开始。状态行后面是任意数量的 HTTP 头。空行表示头的结束和响应体的开始，响应体包含服务器想要发送的实际数据。

[*Remove ads*](/account/join/)

### HTTP 重定向状态代码

重定向响应看起来像什么？让我们假设路径`/redirect/`由`redirect_view()`处理，如前所示。如果你用`curl`访问`http://127.0.0.1:8000/redirect/`，你的控制台看起来像这样:

```py
$ curl --include http://127.0.0.1:8000/redirect/
HTTP/1.1 302 Found
Date: Sun, 01 Jul 2018 20:35:34 GMT
Server: WSGIServer/0.2 CPython/3.6.3
Content-Type: text/html; charset=utf-8
Location: /redirect-success/
X-Frame-Options: SAMEORIGIN
Content-Length: 0
```

这两个回答可能看起来很相似，但是有一些关键的区别。重定向:

*   返回不同的状态代码(`302`对`200`)
*   包含一个带有相对 URL 的`Location`标题
*   以空行结束，因为重定向响应的主体是空的

主要区别在于状态代码。HTTP 标准的规范说明如下:

> 302(已找到)状态代码表示目标资源暂时位于不同的 URI 下。由于重定向有时可能会被更改，所以客户端应该继续为将来的请求使用有效的请求 URI。服务器应该在响应中生成一个位置头字段，其中包含不同 URI 的 URI 引用。用户代理可以使用位置字段值进行自动重定向。([来源](https://tools.ietf.org/html/rfc7231#section-6.4))

换句话说，每当服务器发送一个状态码`302`，它就对客户机说，“嘿，现在，你要找的东西可以在这个地方找到。”

规范中的一个关键短语是“可以使用位置字段值进行自动重定向”这意味着你不能强迫客户端加载另一个网址。客户端可以选择等待用户确认，或者决定根本不加载 URL。

现在您知道了重定向只是一个带有`3xx`状态代码和`Location`报头的 HTTP 响应。这里的关键要点是，HTTP 重定向和任何旧的 HTTP 响应一样，但是有一个空的主体、3xx 状态代码和一个`Location`头。

就是这样。我们稍后将把它与 Django 联系起来，但是首先让我们来看看在那个`3xx`状态代码范围中的两种类型的重定向，看看为什么它们对 web 开发很重要。

### 临时与永久重定向

HTTP 标准指定了几个重定向状态代码，都在`3xx`范围内。两个最常见的状态代码是`301 Permanent Redirect`和`302 Found`。

状态代码`302 Found`表示临时重定向。一个临时的重定向写道:“目前，你要找的东西可以在这个地址找到。”把它想象成一个商店的招牌，上面写着，“我们的商店目前正在装修。请去我们在拐角处的另一家商店。”因为这只是暂时的，你下次去购物的时候会检查原始地址。

**注意:**在 HTTP 1.0 中，状态代码 302 的消息是`Temporary Redirect`。在 HTTP 1.1 中消息被更改为`Found`。

顾名思义，永久重定向应该是永久的。永久重定向告诉浏览器，“你要找的东西已经不在这个地址了。它现在在这个新地址，再也不会在旧地址了。”

永久重定向就像一个商店招牌，上面写着:“我们搬家了。我们的新店就在附近。”这种改变是永久性的，所以下次你想去商店的时候，你会直接去新的地址。

**注意:**永久重定向可能会产生意想不到的后果。在使用永久重定向之前，请完成本指南，或者直接跳到“永久重定向是永久的”一节

浏览器在处理重定向时的行为类似:当 URL 返回一个永久的重定向响应时，这个响应被缓存。下次浏览器遇到旧的 URL 时，它会记住重定向并直接请求新的地址。

缓存重定向可以节省不必要的请求，并带来更好更快的用户体验。

此外，临时和永久重定向之间的区别与搜索引擎优化相关。

[*Remove ads*](/account/join/)

## Django 中的重定向

现在您知道了重定向只是一个带有`3xx`状态代码和`Location`报头的 HTTP 响应。

您可以自己从一个常规的`HttpResponse`对象构建这样一个响应:

```py
def hand_crafted_redirect_view(request):
  response = HttpResponse(status=302)
  response['Location'] = '/redirect/success/'
  return response
```

这个解决方案在技术上是正确的，但是它涉及到相当多的输入。

### `HTTPResponseRedirect`类

使用`HttpResponse`的子类`HttpResponseRedirect`可以节省一些输入。只需用要重定向到的 URL 作为第一个参数实例化该类，该类将设置正确的状态和位置头:

```py
def redirect_view(request):
  return HttpResponseRedirect('/redirect/success/')
```

您可以在 Python shell 中使用`HttpResponseRedirect`类来看看您得到了什么:

>>>

```py
>>> from django.http import HttpResponseRedirect
>>> redirect = HttpResponseRedirect('/redirect/success/')
>>> redirect.status_code
302
>>> redirect['Location']
'/redirect/success/'
```

还有一个用于永久重定向的类，它被恰当地命名为`HttpResponsePermanentRedirect`。它的工作原理与`HttpResponseRedirect`相同，唯一的区别是它有一个状态码`301 (Moved Permanently)`。

**注意:**在上面的例子中，重定向 URL 是硬编码的。对 URL 进行硬编码是一种不好的做法:如果 URL 发生了变化，您必须搜索所有代码并修改所有出现的内容。让我们解决这个问题！

你可以使用 [`django.urls.reverse()`](https://docs.djangoproject.com/en/2.1/ref/urlresolvers/#reverse) 来创建一个 URL，但是有一个更方便的方法，你将在下一节看到。

### `redirect()`功能

为了让您的生活更轻松，Django 提供了您已经在简介中看到的多功能快捷功能: [`django.shortcuts.redirect()`](https://docs.djangoproject.com/en/2.1/topics/http/shortcuts/#redirect) 。

您可以通过以下方式调用此函数:

*   一个模型实例，或任何其他对象，用 [`get_absolute_url()`](https://docs.djangoproject.com/en/2.1/ref/models/instances/#get-absolute-url) 方法
*   URL 或视图名称以及位置和/或关键字参数
*   一个网址

它将采取适当的步骤将参数转换成 URL 并返回一个`HTTPResponseRedirect`。如果你传递了`permanent=True`，它将返回一个`HttpResponsePermanentRedirect`的实例，导致一个永久的重定向。

这里有三个例子来说明不同的使用案例:

1.  传递模型:

    ```py
    from django.shortcuts import redirect

    def model_redirect_view(request):
        product = Product.objects.filter(featured=True).first()
        return redirect(product)` 
    ```

    `redirect()`将调用`product.get_absolute_url()`并将结果作为重定向目标。如果给定的类，在这个例子中是`Product`，没有`get_absolute_url()`方法，那么这个类将会以`TypeError`失败。

2.  传递 URL 名称和参数:

    ```py
    from django.shortcuts import redirect

    def fixed_featured_product_view(request):
        ...
        product_id = settings.FEATURED_PRODUCT_ID
        return redirect('product_detail', product_id=product_id)` 
    ```

    `redirect()`将尝试使用其给定的参数来反转一个 URL。此示例假设您的 URL 模式包含如下模式:

    ```py
    path('/product/<product_id>/', 'product_detail_view', name='product_detail')` 
    ```

3.  传递 URL:

    ```py
    from django.shortcuts import redirect

    def featured_product_view(request):
        return redirect('/products/42/')` 
    ```

    `redirect()`会将任何包含`/`或`.`的字符串视为 URL，并将其用作重定向目标。

[*Remove ads*](/account/join/)

### `RedirectView`基于类的视图

如果你有一个视图除了返回一个重定向什么也不做，你可以使用基于类的视图 [`django.views.generic.base.RedirectView`](https://docs.djangoproject.com/en/2.1/ref/class-based-views/base/#redirectview) 。

您可以通过各种属性来定制`RedirectView`以满足您的需求。

如果该类有一个`.url`属性，它将被用作重定向 URL。字符串格式占位符被替换为 URL 中的命名参数:

```py
# urls.py
from django.urls import path
from .views import SearchRedirectView

urlpatterns = [
    path('/search/<term>/', SearchRedirectView.as_view())
]

# views.py
from django.views.generic.base import RedirectView

class SearchRedirectView(RedirectView):
  url = 'https://google.com/?q=%(term)s'
```

URL 模式定义了一个参数`term`，该参数在`SearchRedirectView`中用于构建重定向 URL。应用程序中的路径`/search/kittens/`会将您重定向到`https://google.com/?q=kittens`。

您也可以在您的`urlpatterns`中将关键字参数`url`传递给`as_view()`，而不是子类化`RedirectView`来覆盖`url`属性:

```py
#urls.py
from django.views.generic.base import RedirectView

urlpatterns = [
    path('/search/<term>/',
         RedirectView.as_view(url='https://google.com/?q=%(term)s')),
]
```

您也可以覆盖`get_redirect_url()`以获得完全自定义的行为:

```py
from random import choice
from django.views.generic.base import RedirectView

class RandomAnimalView(RedirectView):

     animal_urls = ['/dog/', '/cat/', '/parrot/']
     is_permanent = True

     def get_redirect_url(*args, **kwargs):
        return choice(self.animal_urls)
```

这个基于类的视图重定向到一个从`.animal_urls`中随机选取的 URL。

`django.views.generic.base.RedirectView`提供了更多的定制挂钩。以下是完整的列表:

*   `.url`

    如果设置了此属性，它应该是一个带有要重定向到的 URL 的字符串。如果它包含像`%(name)s`这样的字符串格式占位符，它们会使用传递给视图的关键字参数来扩展。

*   `.pattern_name`

    如果设置了此属性，它应该是要重定向到的 URL 模式的名称。传递给视图的任何位置和关键字参数都用来反转 URL 模式。

*   `.permanent`

    如果这个属性是`True`，视图返回一个永久的重定向。默认为`False`。

*   `.query_string`

    如果该属性为`True`，视图会将任何提供的查询字符串附加到重定向 URL。如果是默认的`False`，查询字符串将被丢弃。

*   `get_redirect_url(*args, **kwargs)`

    这个方法负责构建重定向 URL。如果这个方法返回`None`，视图返回一个`410 Gone`状态。

    默认实现首先检查`.url`。它将`.url`视为“旧式”[格式字符串](https://realpython.com/python-string-formatting/)，使用传递给视图的任何命名 URL 参数来扩展任何命名格式说明符。

    如果`.url`未置位，则检查`.pattern_name`是否置位。如果是，它就用它来用接收到的任何位置和关键字参数来反转 URL。

    您可以通过覆盖此方法以任何方式更改该行为。只要确保它返回一个包含 URL 的字符串。

注意:基于类的视图是一个强大的概念，但是有点难以理解。与常规的基于函数的视图不同，在常规的基于函数的视图中，跟踪代码流相对简单，而基于类的视图由复杂的混合和基类层次结构组成。

理解基于类的视图类的一个很好的工具是网站 [Classy Class-Based Views](http://ccbv.co.uk/) 。

您可以用这个简单的基于函数的视图实现上面例子中的`RandomAnimalView`的功能:

```py
from random import choice
from django.shortcuts import redirect

def random_animal_view(request):
    animal_urls = ['/dog/', '/cat/', '/parrot/']
    return redirect(choice(animal_urls))
```

正如您所看到的，基于类的方法没有提供任何明显的好处，同时增加了一些隐藏的复杂性。这就提出了一个问题:什么时候应该使用`RedirectView`？

如果你想在你的`urls.py`中直接添加一个重定向，使用`RedirectView`是有意义的。但是如果你发现自己在重写`get_redirect_url`，基于功能的视图可能更容易理解，对未来的增强也更灵活。

## 高级用法

一旦你知道你可能想要使用`django.shortcuts.redirect()`，重定向到一个不同的 URL 是非常简单的。但是有几个高级用例并不明显。

[*Remove ads*](/account/join/)

### 用重定向传递参数

有时，您希望将一些参数传递给要重定向到的视图。最佳选择是在重定向 URL 的查询字符串中传递数据，这意味着重定向到如下 URL:

```py
http://example.com/redirect-path/?parameter=value
```

让我们假设您想从`some_view()`重定向到`product_view()`，但是传递一个可选参数`category`:

```py
from django.urls import reverse
from urllib.parse import urlencode

def some_view(request):
    ...
    base_url = reverse('product_view')  # 1 /products/
    query_string =  urlencode({'category': category.id})  # 2 category=42
    url = '{}?{}'.format(base_url, query_string)  # 3 /products/?category=42
    return redirect(url)  # 4

def product_view(request):
    category_id = request.GET.get('category')  # 5
    # Do something with category_id
```

本例中的代码相当密集，所以让我们一步一步来看:

1.  首先，使用`django.urls.reverse()`获取映射到`product_view()`的 URL。

2.  接下来，您必须构建查询字符串。那是问号后面的部分。建议使用`urllib.urlparse.urlencode()`来实现，因为它会对任何特殊字符进行适当的编码。

3.  现在你得用问号把`base_url`和`query_string`连起来。格式字符串很适合这种情况。

4.  最后，将`url`传递给`django.shortcuts.redirect()`或重定向响应类。

5.  在您的重定向目标`product_view()`中，参数将在`request.GET`字典中可用。参数可能会丢失，所以您应该使用`requests.GET.get('category')`而不是`requests.GET['category']`。前者在参数不存在时返回`None`，而后者会引发一个异常。

**注意:**确保验证从查询字符串中读取的任何数据。看起来这些数据在您的控制之下，因为您创建了重定向 URL。

实际上，重定向可能会被用户操纵，像任何其他用户输入一样，不能被信任。如果没有适当的验证，[攻击者可能会获得未经授权的访问](https://www.owasp.org/index.php/Top_10-2017_A5-Broken_Access_Control)。

### 特殊重定向代码

Django 为状态代码`301`和`302`提供 HTTP 响应类。这些应该涵盖了大多数用例，但是如果您必须返回状态代码`303`、`307`或`308`，您可以非常容易地创建自己的响应类。简单地子类化`HttpResponseRedirectBase`并覆盖`status_code`属性:

```py
class HttpResponseTemporaryRedirect(HttpResponseRedirectBase):
    status_code = 307
```

或者，您可以使用`django.shortcuts.redirect()`方法创建一个响应对象并更改返回值。当您有了想要重定向到的视图或 URL 或模型的名称时，这种方法是有意义的:

```py
def temporary_redirect_view(request):
    response = redirect('success_view')
    response.status_code = 307
    return response
```

**注意:**在`3xx`范围内其实还有一个状态码为`HttpResponseNotModified`的第三类，状态码为`304`。这表明内容 URL 没有改变，客户端可以使用缓存的版本。

有人可能会说`304 Not Modified`响应重定向到 URL 的缓存版本，但这有点牵强。因此，它不再列在 HTTP 标准的[“重定向 3xx”部分](https://tools.ietf.org/html/rfc7231#section-6.4)中。

## 陷阱

### 无法重定向的重定向

`django.shortcuts.redirect()`的简单可能具有欺骗性。该函数本身并不执行重定向:它只是返回一个重定向响应对象。您必须从您的视图中(或在中间件中)返回这个响应对象。否则，不会发生重定向。

但是即使你知道仅仅调用`redirect()`是不够的，也很容易通过简单的重构将这个 bug 引入到一个工作的应用程序中。这里有一个例子来说明。

让我们假设您正在建立一个商店，并且有一个负责展示产品的视图。如果该产品不存在，请重定向至主页:

```py
def product_view(request, product_id):
    try:
        product = Product.objects.get(pk=product_id)
    except Product.DoesNotExist:
        return redirect('/')
    return render(request, 'product_detail.html', {'product': product})
```

现在，您想要添加第二个视图来显示某个产品的客户评论。它还应该重定向到不存在的产品的主页，所以作为第一步，您将这个功能从`product_view()`提取到一个助手函数`get_product_or_redirect()`:

```py
def get_product_or_redirect(product_id):
    try:
        return Product.objects.get(pk=product_id)
    except Product.DoesNotExist:
        return redirect('/')

def product_view(request, product_id):
    product = get_product_or_redirect(product_id)
    return render(request, 'product_detail.html', {'product': product})
```

不幸的是，重构之后，重定向不再有效。

你能发现错误吗？ 显示/隐藏

`redirect()`的结果从`get_product_or_redirect()`返回，但是`product_view()`不返回。相反，它被传递给模板。

根据您在`product_detail.html`模板中如何使用`product`变量，这可能不会导致错误消息，而只是显示空值。

[*Remove ads*](/account/join/)

### 无法停止重定向的重定向

在处理重定向时，您可能会意外地创建一个重定向循环，让 URL A 返回一个指向 URL B 的重定向，URL B 返回一个到 URL A 的重定向，依此类推。大多数 HTTP 客户端会检测到这种重定向循环，并在多次请求后显示一条错误消息。

不幸的是，这种错误很难发现，因为在服务器端一切看起来都很好。除非您的用户抱怨这个问题，否则唯一可能出错的迹象是，您已经从一个客户端收到了许多请求，这些请求都导致了快速连续的重定向响应，但是没有状态为`200 OK`的响应。

下面是重定向循环的一个简单示例:

```py
def a_view(request):
    return redirect('another_view')

def another_view(request):
    return redirect('a_view')
```

这个例子说明了原理，但是它过于简单了。您在现实生活中遇到的重定向循环可能更难发现。让我们看一个更详细的例子:

```py
def featured_products_view(request):
    featured_products = Product.objects.filter(featured=True)
    if len(featured_products == 1):
        return redirect('product_view', kwargs={'product_id': featured_products[0].id})
    return render(request, 'featured_products.html', {'product': featured_products})

def product_view(request, product_id):
    try:
        product = Product.objects.get(pk=product_id, in_stock=True)
    except Product.DoesNotExist:
        return redirect('featured_products_view')
    return render(request, 'product_detail.html', {'product': product})
```

`featured_products_view()`获取所有特色产品，换句话说，`.featured`设置为`True`的`Product`实例。如果只有一个特色产品，它会直接重定向到`product_view()`。否则，它将使用`featured_products` queryset 呈现一个模板。

从上一节来看,`product_view`看起来很熟悉，但是它有两个小的不同:

*   视图试图获取库存的`Product`，通过将`.in_stock`设置为`True`来表示。
*   如果没有库存产品，视图将重定向到`featured_products_view()`。

这种逻辑运作良好，直到你的商店成为其自身成功的受害者，你目前拥有的特色产品脱销。如果你将`.in_stock`设置为`False`，但是忘记将`.featured`也设置为`False`，那么任何访问你的`feature_product_view()`的访问者都将陷入重定向循环。

没有防弹的方法来防止这种错误，但是一个好的起点是检查您重定向到的视图是否使用重定向本身。

### 永久重定向是永久的

永久重定向可能就像糟糕的纹身:它们当时看起来可能是一个好主意，但一旦你意识到它们是一个错误，就很难摆脱它们。

当浏览器收到对 URL 的永久重定向响应时，它会无限期地缓存该响应。将来任何时候你请求旧的 URL，浏览器都不会加载它，而是直接加载新的 URL。

说服浏览器加载一个曾经返回永久重定向的 URL 可能相当棘手。谷歌 Chrome 在缓存重定向方面尤其积极。

为什么这会是一个问题？

假设您想用 Django 构建一个 web 应用程序。您在`myawesomedjangowebapp.com`注册您的域名。作为第一步，你在`https://myawesomedjangowebapp.com/blog/`安装一个博客应用程序来建立一个发布邮件列表。

您在`https://myawesomedjangowebapp.com/`的网站主页仍在建设中，因此您重定向到`https://myawesomedjangowebapp.com/blog/`。你决定使用永久重定向，因为你听说永久重定向被缓存，缓存使事情更快，越快越好，因为速度是谷歌搜索结果排名的一个因素。

事实证明，你不仅是一个伟大的开发者，还是一个有才华的作家。你的博客变得受欢迎，你的发布邮件列表也在增长。几个月后，你的应用程序就准备好了。现在它有了一个闪亮的主页，你终于删除了重定向。

您向您庞大的发布邮件列表发送一封带有特殊折扣代码的公告电子邮件。你靠在椅背上，等待注册通知滚滚而来。

让你感到恐惧的是，你的邮箱塞满了困惑的访问者的信息，他们想访问你的应用程序，但总是被重定向到你的博客。

发生了什么事？当重定向到`https://myawesomedjangowebapp.com/blog/`仍然有效时，您的博客读者已经访问了`https://myawesomedjangowebapp.com/`。因为这是一个永久的重定向，它被缓存在他们的浏览器中。

当他们点击你发布公告邮件中的链接时，他们的浏览器根本不会检查你的新主页，而是直接进入你的博客。你没有庆祝你的成功发布，而是忙着指导你的用户如何摆弄`chrome://net-internals`来重置他们浏览器的缓存。

当在本地机器上开发时，永久重定向的永久性质也会伤害到你。让我们倒回到你为 myawesomedjangowebapp.com 实施那个决定性的永久重定向的时刻。

您启动开发服务器并打开`http://127.0.0.1:8000/`。如你所愿，你的应用将你的浏览器重定向到`http://127.0.0.1:8000/blog/`。对您的工作感到满意，您停止开发服务器，去吃午饭。

你带着满满的肚子回来，准备处理一些客户的工作。客户希望对他们的主页进行一些简单的更改，因此您加载客户的项目并启动开发服务器。

等等，这是怎么回事？主页坏了，现在返回一个 404！由于下午的低迷，过了一会儿你才注意到你被重定向到了`http://127.0.0.1:8000/blog/`，这在客户的项目中并不存在。

对于浏览器来说，URL `http://127.0.0.1:8000/`现在服务于一个完全不同的应用程序并不重要。对浏览器来说，重要的是这个 URL 曾经返回一个永久重定向到`http://127.0.0.1:8000/blog/`。

这个故事告诉我们，你应该只在你不打算再使用的 URL 上使用永久重定向。永久重定向是存在的，但是你必须意识到它们的后果。

即使你确信你真的需要一个永久的重定向，先实现一个临时的重定向也是一个好主意，只有当你 100%确定一切正常时才切换到它的永久的表亲。

[*Remove ads*](/account/join/)

### 未经验证的重定向会危及安全性

从安全的角度来看，重定向是一种相对安全的技术。攻击者无法通过重定向攻击网站。毕竟，重定向只是重定向到一个 URL，攻击者只需在浏览器的地址栏中键入即可。

但是，如果您使用某种类型的用户输入，如 URL 参数，而没有作为重定向 URL 进行适当的验证，这可能会被攻击者滥用来进行网络钓鱼攻击。这种重定向被称为[开放或未验证重定向](https://cwe.mitre.org/data/definitions/601.html)。

对于重定向到从用户输入中读取的 URL，有一些合理的使用案例。一个主要的例子是 Django 的登录视图。它接受一个 URL 参数`next`，该参数包含用户登录后被重定向到的页面的 URL。要在登录后将用户重定向到他们的个人资料，URL 可能如下所示:

```py
https://myawesomedjangowebapp.com/login/?next=/profile/
```

Django 确实验证了`next`参数，但是让我们假设它没有验证。

未经验证，攻击者可以创建一个 URL，将用户重定向到他们控制的网站，例如:

```py
https://myawesomedjangowebapp.com/login/?next=https://myawesomedjangowebapp.co/profile/
```

网站`myawesomedjangowebapp.co`可能会显示一条错误消息，并欺骗用户再次输入他们的凭据。

避免开放重定向的最佳方式是在构建重定向 URL 时不使用任何用户输入。

如果您不能确定 URL 对于重定向是安全的，您可以使用函数`django.utils.http.is_safe_url()`来验证它。docstring 很好地解释了它的用法:

> `is_safe_url(url, host=None, allowed_hosts=None, require_https=False)`
> 
> 如果 url 是安全重定向(即它不指向不同的主机并使用安全方案)，则返回`True`。总是在空 url 上返回`False`。如果`require_https`是`True`，那么只有‘https’将被认为是有效的方案，而不是默认为`False`的‘http’和‘https’。([来源](https://github.com/django/django/blob/53a3d2b2454ff9a612a376f58bb7c61733f82d12/django/utils/http.py#L280))

让我们看一些例子。

相对 URL 被认为是安全的:

>>>

```py
>>> # Import the function first.
>>> from django.utils.http import is_safe_url
>>> is_safe_url('/profile/')
True
```

指向另一台主机的 URL 通常被认为是不安全的:

>>>

```py
>>> is_safe_url('https://myawesomedjangowebapp.com/profile/')
False
```

如果在`allowed_hosts`中提供了主机，则指向另一个主机的 URL 被认为是安全的:

>>>

```py
>>> is_safe_url('https://myawesomedjangowebapp.com/profile/',
...             allowed_hosts={'myawesomedjangowebapp.com'})
True
```

如果参数`require_https`是`True`，使用`http`方案的 URL 被认为是不安全的:

>>>

```py
>>> is_safe_url('http://myawesomedjangowebapp.com/profile/',
...             allowed_hosts={'myawesomedjangowebapp.com'},
...             require_https=True)
False
```

[*Remove ads*](/account/join/)

## 总结

关于 Django 的 HTTP 重定向指南到此结束。恭喜:现在您已经接触到了重定向的各个方面，从 HTTP 协议的底层细节到 Django 中处理它们的高层方式。

您了解了 HTTP 重定向在幕后是什么样子，不同的状态代码是什么，以及永久重定向和临时重定向有什么不同。这些知识并不是 Django 特有的，对于任何语言的 web 开发都是有价值的。

现在可以用 Django 执行重定向了，要么使用重定向响应类`HttpResponseRedirect`和`HttpResponsePermanentRedirect`，要么使用方便的函数`django.shortcuts.redirect()`。您看到了一些高级用例的解决方案，并且知道如何避开常见的陷阱。

如果你有任何关于 HTTP 重定向的问题，请在下面留下评论，同时，祝你重定向愉快！

## 参考文献

*   [姜戈文档:`django.http.HttpResponseRedirect`](https://docs.djangoproject.com/en/2.1/ref/request-response/#django.http.HttpResponseRedirect)
*   [姜戈文档:`django.shortcuts.render()`](https://docs.djangoproject.com/en/2.1/topics/http/shortcuts/#redirect)
*   [姜戈文档:`django.views.generic.base.RedirectView`](https://docs.djangoproject.com/en/2.1/ref/class-based-views/base/#redirectview)
*   [RFC 7231:超文本传输协议(HTTP/1.1):语义和内容- 6.4 重定向 3xx](https://tools.ietf.org/html/rfc7231#section-6.4)
*   [CWE-601: URL 重定向到不受信任的站点(‘打开重定向’)](http://cwe.mitre.org/data/definitions/601.html)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Django 重定向**](/courses/django-redirects/)**********