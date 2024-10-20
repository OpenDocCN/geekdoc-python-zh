# Python 的 Django vs Ruby on Rails

> 原文：<https://www.pythoncentral.io/python-django-vs-ruby-on-rails/>

## Python vs Ruby

Ruby 是一种动态的、反射的、面向对象的通用编程语言，它是在 20 世纪 90 年代中期设计和开发的。与将代码可读性看得比什么都重要的 Python 相比，Ruby 背后的哲学是程序员应该拥有编写简洁紧凑代码的灵活性、自由和权力。

Python 和 Ruby 之间最重要的区别在于，Python 的哲学要求程序员明确定义几乎所有东西，而 Ruby 允许程序员用从其他组件继承的隐式行为编写代码。例如，在 Ruby 中，实例方法中的`self`是隐式定义的，而 Python 要求`self`是实例方法中声明的第一个参数。

```py

class Hello

  def hello

    'hello'

  end

  def call_hello

    hello()

  end

end

```

```py

class Hello(object):

  def hello(self):

    return 'hello'

  def call_hello(self):

    return self.hello()

```

在上面的示例代码中，类`Hello`包含两个实例方法`hello()`和`call_hello()`。注意，Ruby 对`call_hello()`的定义只是调用了`hello()`方法，而没有像 Python 的定义那样引用`self.hello()`。

## Ruby on Rails 概述

Ruby on Rails 是一个开源的 web 应用框架，运行在 Ruby 编程语言之上。就像 Django 一样，Rails 允许程序员编写与后端数据库对话的 web 应用程序来检索数据，并将数据呈现在客户端的模板中。

不出所料，Ruby 隐含的哲学也影响了 Rails 的设计方式。Django 强迫程序员显式地配置项目，并在代码中表达她的意图，与此不同，Rails 提供了许多程序员可以依赖的现成的隐式默认约定。与 Django 不同，Django 通常很灵活，不强制使用一种方式来做事情，Rails 认为做某些事情只有一种最好的方式，因此程序员很难修改 Rails 代码的逻辑和行为。

### 约定胜于配置

Rails 最重要的哲学是约定胜于配置(CoC)，这意味着 Rails 项目有预定义的布局和合理的默认值。所有组件，如模型、控制器、静态 CSS 和 JavaScript 文件都位于标准子目录中，您只需将自己的实现文件放入这些目录中，Rails 就会自动获取它们。在 Django 中，您经常需要指定组件文件的位置。

CoC 是开发人员的一大胜利，因为它节省了反复输入相同配置代码的大量时间。然而，如果您想要定制您的项目的配置，您必须学习相当多的关于 Rails 的知识，以便在不破坏整个项目的情况下更改配置。

### 模型-视图-控制器和休息

Rails 是一个模型-视图-控制器(MVC)全栈 web 框架，这意味着控制器从模型中调用函数，并将数据返回给视图。尽管许多 web 框架也是基于 MVC 的，但 Rails 是独一无二的，因为它支持开箱即用的完整 REST 协议。使用标准的 HTTP 动词(如 GET、PUT、DELETE 和 POST)以统一的方式访问和处理所有模型对象。

## 为什么选择 Ruby on Rails

像 Django 一样，Rails 也使用*模型*来描述数据库表，使用*控制器*从*模型*中检索数据，并将数据返回给*视图*，视图最终将数据呈现为 HTTP 响应。为了将*控制器*映射到传入的请求，程序员在配置文件中指定路由。

自从 2005 年首次发布以来，Rails 在 web 程序员中越来越受欢迎。它易于使用和理解的技术栈和隐含的 CoC 哲学似乎允许敏捷的 web 程序员比其他框架更快地实现应用程序。因此，Rails 是 Django 的一个很好的替代方案。

## Ruby on Rails 和 Django 的区别

到目前为止，Rails 看起来几乎和 Django 一样。因为 Django 的模型-模板-视图可以被视为模型-视图-控制器的变体，其中 Django 视图是 MVC 控制器，而 Django 模板是 MVC 视图，所以 Django 和 Rails 都支持 MVC 并提供通用的 web 框架功能。你可能会开始怀疑:Rails 和 Django 有什么区别吗？

最重要的区别是 Rails 推广了*约定胜于配置*的理念，这意味着开发者只需要指定应用程序中非常规的部分。例如，如果有一个名为`Employee`的*模型*类，那么如果开发人员没有提供一个显式的表名，那么数据库中相应的表将被命名为`employees`。如果开发人员希望表有不同的名称，这是非传统的，那么他或她需要在*模型*文件中为`Employee`显式指定表名。

Rails 运行在 Ruby 之上，Ruby 重视表现性并提供大量隐式行为，而 Django 运行在 Python 之上，后者重视显式性，可能会更冗长。例如，Rails 模型在每个*模型*对象上公开了一个隐式方法`find_by`，该方法允许程序员通过*模型*属性找到数据库记录。而 Django 并没有在其*模型*上公开这样的隐式方法。

除了像 Django 的`django-admin.py createproject`这样的常规项目生成脚本，Rails 还提供了一个强大的脚手架脚本，允许您直接从命令行生成一个新的控制器。

```py

$ rails generate controller welcome index

create  app/controllers/welcome_controller.rb

 route  get "welcome/index"

invoke  erb

create    app/views/welcome

create    app/views/welcome/index.html.erb

invoke  test_unit

create    test/controllers/welcome_controller_test.rb

invoke  helper

create    app/helpers/welcome_helper.rb

invoke    test_unit

create      test/helpers/welcome_helper_test.rb

invoke  assets

invoke    coffee

create      app/assets/javascripts/welcome.js.coffee

invoke    scss

create      app/assets/stylesheets/welcome.css.scss

```

上面的命令为您创建了几个文件和一个 URL 路由。大多数时候，你只需要在`app/controllers/welcome_controller.rb`的控制器实现文件和`app/views/welcome/index.html.erb`的视图实现文件中插入自定义逻辑。

与 Django 不同，Django 具有内置的用户认证和授权支持，Rails 不提供开箱即用的认证服务。幸运的是，有很多第三方认证库可用。你可以在 [rails_authentication](https://www.ruby-toolbox.com/categories/rails_authentication.html) 查看它们。

Rails 的一个缺点是它的 API 比 Django API 变化更频繁。因此，Rails 通常不保留向后兼容性，而 Django 很少更改其 API 并保持向后兼容性。

## 托管支持和开发工具

由于不可否认的受欢迎程度，Django 和 Rails 都有优秀的主机提供商，比如 Heroku 和 T2 Rackspace。当然，如果您想自己管理服务器，您可以选择 VPS 提供商来托管您的 Django 或 Rails 应用程序。

如果你遇到了关于 Rails 的问题，官方网站[rubyonrails.org](https://rubyonrails.org/)提供了很好的资源在文档中查找信息，你也可以在社区论坛中提问。

## 总结和提示

Rails 和 Django 都是由高效编程语言支持的有效的 web 框架。归根结底，选择完全是主观和个人的，没有谁比谁更好。如果你要去一家使用这两种框架之一的公司工作，我建议和那家公司的开发人员谈谈，选择一个几乎所有人都感到兴奋的框架。