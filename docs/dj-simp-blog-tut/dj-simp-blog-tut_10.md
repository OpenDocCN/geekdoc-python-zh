# 归档, AboutMe 和标签分类

> 这一章节说的东西都是一些知识回顾,

# 归档

归档就是列出当前博客中所有的文章, 并且能够显示时间, 很容易的可以写出对应的 view 和模板来

在 my_blog/my_blog/view 下新建归档 view

```py
def archives(request) :
    try:
        post_list = Article.objects.all()
    except Article.DoesNotExist :
        raise Http404
    return render(request, 'archives.html', {'post_list' : post_list, 
                                            'error' : False}) 
```

在 my_blog/templates 新建模板`archives.html`

```py
{% extends "base.html" %}

{% block content %}
<div class="posts">
    {% for post in post_list %}
        <section class="post">
            <header class="post-header">
                <h2 class="post-title"><a href="{% url "detail" id=post.id %}">{{ post.title }}</a></h2>

                    <p class="post-meta">
                        Time:  <a class="post-author" href="#">{{ post.date_time |date:"Y /m /d"}}</a> <a class="post-category post-category-js" href="{% url "search_tag" tag=post.category %}">{{ post.category }}</a>
                    </p>
            </header>
        </section>
    {% endfor %}
</div><!-- /.blog-post -->
{% endblock %} 
```

并在 my_blog/my_blog/usls.py 中添加对应 url 配置

```py
from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'my_blog.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', 'article.views.home', name = 'home'),
    url(r'^(?P<id>\d+)/$', 'article.views.detail', name='detail'),
    url(r'^archives/$', 'article.views.archives', name = 'archives'),
) 
```

# AboutMe

这个就不多说了

在 my_blog/my_blog/view.py 下添加新的逻辑

```py
def about_me(request) :
    return render(request, 'aboutme.html') 
```

在 my_blog/template 下新建模板 aboutme.html, 内容如下, 大家可以自定义自己喜欢的简介

```py
{% extends "base.html" %}
{% load custom_markdown %}

{% block content %}
<div class="posts">
        <p> About Me 正在建设中 </p>
</div><!-- /.blog-post -->
{% endblock %} 
```

并在 my_blog/my_blog/usls.py 中添加对应 url 配置

```py
from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'my_blog.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', 'article.views.home', name = 'home'),
    url(r'^(?P<id>\d+)/$', 'article.views.detail', name='detail'),
    url(r'^archives/$', 'article.views.archives', name = 'archives'),
    url(r'^aboutme/$', 'article.views.about_me', name = 'about_me'),
) 
```

# 标签分类

实现功能: 点击对应的标签按钮, 会跳转到一个新的页面, 这个页面是所有相关标签的文章的罗列

只需要在在 my_blog/my_blog/view.py 下添加新的逻辑

```py
def search_tag(request, tag) :
    try:
        post_list = Article.objects.filter(category__iexact = tag) #contains
    except Article.DoesNotExist :
        raise Http404
    return render(request, 'tag.html', {'post_list' : post_list}) 
```

可以看成是对 tag 的查询操作, 通过传入对应点击的 tag, 然后对 tag 进行查询

在对应的有 tag 的 html 网页中修改代码

```py
{% extends "base.html" %}

{% load custom_markdown %}
{% block content %}
<div class="posts">
    {% for post in post_list %}
        <section class="post">
            <header class="post-header">
                <h2 class="post-title"><a href="{% url "detail" id=post.id %}">{{ post.title }}</a></h2>

                    <p class="post-meta">
                        Time:  <a class="post-author" href="#">{{ post.date_time |date:"Y M d"}}</a> <a class="post-category post-category-js" href="{% url "search_tag" tag=post.category %}">{{ post.category|title }}</a>
                    </p>
            </header>

                <div class="post-description">
                    <p>
                        {{ post.content|custom_markdown }}
                    </p>
                </div>
                <a class="pure-button" href="{% url "detail" id=post.id %}">Read More >>> </a>
        </section>
    {% endfor %}
</div><!-- /.blog-post -->
{% endblock %} 
```

仔细看这一句`<a class="post-category post-category-js" href="{% url "search_tag" tag=post.category %}">{{ post.category|title }}</a>`. 其中标签对超链接已经发生改变, 这是在对标签就行点击时, 会将标签作为参数, 传入到对应的 view 中执行逻辑, 然后进行网页跳转...

并在 my_blog/my_blog/usls.py 中添加对应 url 配置

```py
from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'my_blog.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', 'article.views.home', name = 'home'),
    url(r'^(?P<id>\d+)/$', 'article.views.detail', name='detail'),
    url(r'^archives/$', 'article.views.archives', name = 'archives'),
    url(r'^aboutme/$', 'article.views.about_me', name = 'about_me'),
    url(r'^tag(?P<tag>\w+)/$', 'article.views.search_tag', name = 'search_tag'),
) 
```

现在在浏览器中输入[`127.0.0.1:8000/`](http://127.0.0.1:8000/), 点击对应的归档或者 ABOUT ME 或者标签按钮可以看到对应的效果