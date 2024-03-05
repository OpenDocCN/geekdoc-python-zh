# 使用 Angular 4 和 Flask 进行用户验证

> 原文：<https://realpython.com/user-authentication-with-angular-4-and-flask/>

在本教程中，我们将演示如何用 Angular 4 和 Flask 设置基于令牌的认证(通过 JSON Web 令牌)。

*主要依赖关系*:

1.  角度 v [4.2.4](https://github.com/angular/angular/releases/tag/4.2.4) (通过角度 v [1.3.2](https://github.com/angular/angular-cli/releases/tag/v1.3.2)
2.  烧瓶 v [0.12](http://flask.pocoo.org/docs/0.12/changelog/#version-0-12)
3.  Python v [3.6.2](https://www.python.org/downloads/release/python-362/)

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

## 授权工作流

以下是完整的用户认证过程:

1.  客户端登录，凭据被发送到服务器
2.  如果凭证正确，服务器将生成一个令牌，并将其作为响应发送给客户端
3.  客户端接收令牌并将其存储在本地存储中
4.  然后，客户端在请求标头中的后续请求上向服务器发送令牌

[*Remove ads*](/account/join/)

## 项目设置

从全局安装[角度控制器](https://cli.angular.io/)开始:

```py
$ npm install -g @angular/cli@1.3.2
```

然后生成一个新的 Angular 4 项目样板文件:

```py
$ ng new angular4-auth
```

安装完依赖项后启动应用程序:

```py
$ cd angular4-auth
$ ng serve
```

编译和构建您的应用程序可能需要一两分钟。完成后，导航到 [http://localhost:4200](http://localhost:4200) 以确保应用程序启动并运行。

在您最喜欢的代码编辑器中打开项目，然后浏览代码:

```py
├── e2e
│   ├── app.e2e-spec.ts
│   ├── app.po.ts
│   └── tsconfig.e2e.json
├── karma.conf.js
├── package.json
├── protractor.conf.js
├── src
│   ├── app
│   │   ├── app.component.css
│   │   ├── app.component.html
│   │   ├── app.component.spec.ts
│   │   ├── app.component.ts
│   │   └── app.module.ts
│   ├── assets
│   ├── environments
│   │   ├── environment.prod.ts
│   │   └── environment.ts
│   ├── favicon.ico
│   ├── index.html
│   ├── main.ts
│   ├── polyfills.ts
│   ├── styles.css
│   ├── test.ts
│   ├── tsconfig.app.json
│   ├── tsconfig.spec.json
│   └── typings.d.ts
├── tsconfig.json
└── tslint.json
```

简而言之，客户端代码位于“src”文件夹中，而 Angular 应用程序本身位于“app”文件夹中。

注意 *app.module.ts* 中的`AppModule`。这用于引导 Angular 应用程序。`@NgModule`装饰器获取元数据，让 Angular 知道如何运行应用程序。我们在本教程中创建的所有东西都将被添加到这个对象中。

在继续下一步之前，确保你已经很好地掌握了应用程序的结构。

> **注:**刚入门 Angular 4？查看 [Angular Style 指南](https://angular.io/guide/styleguide)，因为从 CLI 生成的 app 遵循该指南推荐的结构，以及 [Angular4Crud 教程](https://github.com/mjhea0/angular4-crud/blob/master/tutorial.md)。

您是否注意到 CLI 初始化了一个新的 Git repo？这部分是可选的，但是创建一个新的 Github 存储库并更新 remote 是个好主意:

```py
$ git remote set-url origin <newurl>
```

现在，让我们连接一个新的[组件](https://angular.io/guide/architecture#components) …

## 认证组件

首先，使用 CLI 生成一个新的登录组件:

```py
$ ng generate component components/login
```

这将设置组件文件和文件夹，甚至将其连接到 *app.module.ts* 。接下来，让我们将 *login.component.ts* 文件修改如下:

```py
import  {  Component  }  from  '@angular/core'; @Component({ selector:  'login', templateUrl:  './login.component.html', styleUrls:  ['./login.component.css'] }) export  class  LoginComponent  { test:  string  =  'just a test'; }
```

如果你以前没有使用过 [TypeScript](https://www.typescriptlang.org/) ，那么这段代码可能对你来说很陌生。TypeScript 是 [JavaScript](https://realpython.com/python-vs-javascript/) 的静态类型超集，编译成普通 JavaScript，它是构建 Angular 4 应用程序的事实上的编程语言。

在 Angular 4 中，我们通过用一个`@Component`装饰器包装一个配置对象来定义一个*组件*。通过导入我们需要的类，我们可以在包之间共享代码；在这种情况下，我们从`@angular/core`包中导入`Component`。`LoginComponent`类是组件的控制器，我们使用`export`操作符使它可供其他类导入。

将以下 HTML 添加到*login.component.html*中:

```py
<h1>Login</h1>

<p>{{test}}</p>
```

接下来，通过 *app.module.ts* 文件中的 [RouterModule](https://angular.io/api/router/RouterModule) 配置路由:

```py
import  {  BrowserModule  }  from  '@angular/platform-browser'; import  {  NgModule  }  from  '@angular/core'; import  {  RouterModule  }  from  '@angular/router'; import  {  AppComponent  }  from  './app.component'; import  {  LoginComponent  }  from  './components/login/login.component'; @NgModule({ declarations:  [ AppComponent, LoginComponent, ], imports:  [ BrowserModule, RouterModule.forRoot([ {  path:  'login',  component:  LoginComponent  } ]) ], providers:  [], bootstrap:  [AppComponent] }) export  class  AppModule  {  }
```

通过将*app.component.html*文件中的所有 HTML 替换为`<router-outlet>`标签来完成启用路由:

```py
<router-outlet></router-outlet>
```

在您的终端中运行`ng serve`，如果您还没有运行的话，然后导航到[http://localhost:4200/log in](http://localhost:4200/login)。如果一切顺利，您应该会看到`just a test`文本。

[*Remove ads*](/account/join/)

## 自举风格

为了快速添加一些样式，更新*index.html*，添加[引导](http://getbootstrap.com/)，并将`<app-root></app-root>`包装在[容器](http://getbootstrap.com/css/#overview-container)中:

```py
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Angular4Auth</title>
  <base href="/">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" type="image/x-icon" href="favicon.ico">
  <link rel="stylesheet" href="//maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
</head>
<body>
  <div class="container">
    <app-root></app-root>
  </div>
</body>
</html>
```

你应该看到应用程序自动重新加载，只要你保存。

## 授权服务

接下来，让我们创建一个全局[服务](https://angular.io/guide/architecture#services)来处理用户登录、注销和注册:

```py
$ ng generate service services/auth
```

编辑 *auth.service.ts* ，使其具有以下代码:

```py
import  {  Injectable  }  from  '@angular/core'; @Injectable() export  class  AuthService  { test():  string  { return  'working'; } }
```

还记得 Angular 1 中提供者是如何工作的吗？它们是存储单一状态的全局对象。当提供程序中的数据发生变化时，任何注入该提供程序的对象都会收到更新。在 Angular 4 中，提供者保留了它们的特殊行为，它们是用`@Injectable`装饰器定义的。

### 健全性检查

在向`AuthService`添加任何重要内容之前，让我们确保服务本身被正确连接。为此，在 *login.component.ts* 中注入服务并调用`test()`方法:

```py
import  {  Component,  OnInit  }  from  '@angular/core'; import  {  AuthService  }  from  '../../services/auth.service'; @Component({ selector:  'login', templateUrl:  './login.component.html', styleUrls:  ['./login.component.css'] }) export  class  LoginComponent  implements  OnInit  { test:  string  =  'just a test'; constructor(private  auth:  AuthService)  {} ngOnInit():  void  { console.log(this.auth.test()); } }
```

我们引入了一些新概念和关键词。`constructor()`函数是一个特殊的方法，我们用它来建立一个类的新实例。`constructor()`是我们传递该类所需的任何参数的地方，包括我们想要注入的任何提供者(即`AuthService`)。在 TypeScript 中，我们可以用`private`关键字对外界隐藏变量。在构造函数中传递一个`private`变量是在类中定义它，然后将参数值赋给它的捷径。请注意`auth`变量在传递给构造函数后如何被`this`对象访问。

我们实现了`OnInit`接口，以确保我们明确定义了一个`ngOnInit()`函数。实现`OnInit`确保我们的组件将在第一次变更检测检查后被调用。该函数在组件首次初始化时调用一次，使其成为配置依赖于其他角度类的数据的理想位置。

与自动添加的组件不同，服务必须在`@NgModule`上手动导入和配置。因此，要让它工作，你还必须导入 *app.module.ts* 中的`AuthService`，并将其添加到`providers`:

```py
import  {  BrowserModule  }  from  '@angular/platform-browser'; import  {  NgModule  }  from  '@angular/core'; import  {  RouterModule  }  from  '@angular/router'; import  {  AppComponent  }  from  './app.component'; import  {  LoginComponent  }  from  './components/login/login.component'; import  {  AuthService  }  from  './services/auth.service'; @NgModule({ declarations:  [ AppComponent, LoginComponent, ], imports:  [ BrowserModule, RouterModule.forRoot([ {  path:  'login',  component:  LoginComponent  } ]) ], providers:  [AuthService], bootstrap:  [AppComponent] }) export  class  AppModule  {  }
```

运行服务器，然后导航到[http://localhost:4200/log in](http://localhost:4200/login)。您应该看到`working`被记录到 JavaScript 控制台。

### 用户登录

要处理用户登录，请像这样更新`AuthService`:

```py
import  {  Injectable  }  from  '@angular/core'; import  {  Headers,  Http  }  from  '@angular/http'; import  'rxjs/add/operator/toPromise'; @Injectable() export  class  AuthService  { private  BASE_URL:  string  =  'http://localhost:5000/auth'; private  headers:  Headers  =  new  Headers({'Content-Type':  'application/json'}); constructor(private  http:  Http)  {} login(user):  Promise<any>  { let  url:  string  =  `${this.BASE_URL}/login`; return  this.http.post(url,  user,  {headers:  this.headers}).toPromise(); } }
```

我们借助一些内置的 Angular 类，`Headers`和`Http`，来处理我们对服务器的 AJAX 调用。

同样，更新 *app.module.ts* 文件来导入`HttpModule`。

```py
import  {  BrowserModule  }  from  '@angular/platform-browser'; import  {  NgModule  }  from  '@angular/core'; import  {  RouterModule  }  from  '@angular/router'; import  {  HttpModule  }  from  '@angular/http'; import  {  AppComponent  }  from  './app.component'; import  {  LoginComponent  }  from  './components/login/login.component'; import  {  AuthService  }  from  './services/auth.service'; @NgModule({ declarations:  [ AppComponent, LoginComponent, ], imports:  [ BrowserModule, HttpModule, RouterModule.forRoot([ {  path:  'login',  component:  LoginComponent  } ]) ], providers:  [AuthService], bootstrap:  [AppComponent] }) export  class  AppModule  {  }
```

这里，我们使用 Http 服务向`/user/login`端点发送一个 AJAX 请求。这将返回一个承诺对象。

> **注意:**确保从`LoginComponent`组件上移除`console.log(this.auth.test());`。

[*Remove ads*](/account/join/)

### 用户注册

让我们继续添加注册用户的功能，这类似于让用户登录。更新*src/app/services/auth . service . ts*，注意`register`方法:

```py
import  {  Injectable  }  from  '@angular/core'; import  {  Headers,  Http  }  from  '@angular/http'; import  'rxjs/add/operator/toPromise'; @Injectable() export  class  AuthService  { private  BASE_URL:  string  =  'http://localhost:5000/auth'; private  headers:  Headers  =  new  Headers({'Content-Type':  'application/json'}); constructor(private  http:  Http)  {} login(user):  Promise<any>  { let  url:  string  =  `${this.BASE_URL}/login`; return  this.http.post(url,  user,  {headers:  this.headers}).toPromise(); } register(user):  Promise<any>  { let  url:  string  =  `${this.BASE_URL}/register`; return  this.http.post(url,  user,  {headers:  this.headers}).toPromise(); } }
```

现在，为了测试这一点，我们需要设置一个后端…

## 服务器端设置

对于服务器端，我们将使用上一篇博文中完成的项目，[使用 Flask](https://realpython.com/token-based-authentication-with-flask/) 进行基于令牌的认证。您可以从 [flask-jwt-auth](https://github.com/realpython/flask-jwt-auth) 存储库中查看代码。

> **注意:**随意使用自己的服务器，只要确保更新`AuthService`中的`baseURL`即可。

在新的终端窗口中克隆项目结构:

```py
$ git clone https://github.com/realpython/flask-jwt-auth
```

按照[自述文件](https://github.com/realpython/flask-jwt-auth/blob/master/README.md)中的指示建立项目，[确保在继续](https://realpython.com/python-testing/)之前通过测试。一旦完成，用`python manage.py runserver`运行服务器，它将监听端口 5000。

## 健全性检查

为了测试，更新`LoginComponent`以使用服务中的`login`和`register`方法:

```py
import  {  Component,  OnInit  }  from  '@angular/core'; import  {  AuthService  }  from  '../../services/auth.service'; @Component({ selector:  'login', templateUrl:  './login.component.html', styleUrls:  ['./login.component.css'] }) export  class  LoginComponent  implements  OnInit  { test:  string  =  'just a test'; constructor(private  auth:  AuthService)  {} ngOnInit():  void  { let  sampleUser:  any  =  { email:  'michael@realpython.com'  as  string, password:  'michael'  as  string }; this.auth.register(sampleUser) .then((user)  =>  { console.log(user.json()); }) .catch((err)  =>  { console.log(err); }); this.auth.login(sampleUser).then((user)  =>  { console.log(user.json()); }) .catch((err)  =>  { console.log(err); }); } }
```

在浏览器中刷新[http://localhost:4200/log in](http://localhost:4200/login),在用户登录后，您应该在 JavaScript 控制台中看到一个成功，标记为:

```py
{ "auth_token":  "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1M…jozfQ.bPNQb3C98yyNe0LDyl1Bfkp0Btn15QyMxZnBoE9RQMI", "message":  "Successfully logged in.", "status":  "success" }
```

## 授权登录

更新*login.component.html*:

```py
<div class="row">
  <div class="col-md-4">
    <h1>Login</h1>
    <hr><br>
    <form (ngSubmit)="onLogin()" novalidate>
     <div class="form-group">
       <label for="email">Email</label>
       <input type="text" class="form-control" id="email" placeholder="enter email" [(ngModel)]="user.email" name="email" required>
     </div>
     <div class="form-group">
       <label for="password">Password</label>
       <input type="password" class="form-control" id="password" placeholder="enter password" [(ngModel)]="user.password" name="password" required>
     </div>
     
    </form>
  </div>
</div>
```

请注意表格。我们在每个表单输入上使用了`[(ngModel)]`指令来捕获控制器中的那些值。同样，当表单被提交时，`ngSubmit`指令通过触发`onLogin()`方法来处理事件。

现在，让我们更新组件代码，添加`onLogin()`:

```py
import  {  Component  }  from  '@angular/core'; import  {  AuthService  }  from  '../../services/auth.service'; import  {  User  }  from  '../../models/user'; @Component({ selector:  'login', templateUrl:  './login.component.html', styleUrls:  ['./login.component.css'] }) export  class  LoginComponent  { user:  User  =  new  User(); constructor(private  auth:  AuthService)  {} onLogin():  void  { this.auth.login(this.user) .then((user)  =>  { console.log(user.json()); }) .catch((err)  =>  { console.log(err); }); } }
```

如果您运行 Angular web 服务器，您应该会在浏览器中看到错误`Cannot find module '../../models/user'`。在我们的代码工作之前，我们需要创建一个`User`模型。

```py
$ ng generate class models/user
```

更新 *src/app/models/user.ts* :

```py
export  class  User  { constructor(email?:  string,  password?:  string)  {} }
```

我们的`User`模型有两个属性，`email`和`password`。`?`字符是一个特殊的操作符，表示用显式的`email`和`password`值初始化`User`是可选的。这相当于 Python 中的以下类:

```py
class User(object):
    def __init__(self, email=None, password=None):
        self.email = email
        self.password = password
```

不要忘记更新 *auth.service.ts* 以使用新对象。

```py
import  {  Injectable  }  from  '@angular/core'; import  {  Headers,  Http  }  from  '@angular/http'; import  {  User  }  from  '../models/user'; import  'rxjs/add/operator/toPromise'; @Injectable() export  class  AuthService  { private  BASE_URL:  string  =  'http://localhost:5000/auth'; private  headers:  Headers  =  new  Headers({'Content-Type':  'application/json'}); constructor(private  http:  Http)  {} login(user:  User):  Promise<any>  { let  url:  string  =  `${this.BASE_URL}/login`; return  this.http.post(url,  user,  {headers:  this.headers}).toPromise(); } register(user:  User):  Promise<any>  { let  url:  string  =  `${this.BASE_URL}/register`; return  this.http.post(url,  user,  {headers:  this.headers}).toPromise(); } }
```

最后一件事。我们需要导入 *app.module.ts* 文件中的`FormsModule`。

```py
import  {  BrowserModule  }  from  '@angular/platform-browser'; import  {  NgModule  }  from  '@angular/core'; import  {  RouterModule  }  from  '@angular/router'; import  {  HttpModule  }  from  '@angular/http'; import  {  FormsModule  }  from  '@angular/forms'; import  {  AppComponent  }  from  './app.component'; import  {  LoginComponent  }  from  './components/login/login.component'; import  {  AuthService  }  from  './services/auth.service'; @NgModule({ declarations:  [ AppComponent, LoginComponent, ], imports:  [ BrowserModule, HttpModule, FormsModule, RouterModule.forRoot([ {  path:  'login',  component:  LoginComponent  } ]) ], providers:  [AuthService], bootstrap:  [AppComponent] }) export  class  AppModule  {  }
```

因此，当提交表单时，我们捕获电子邮件和密码，并将它们传递给服务上的`login()`方法。

用这个测试一下-

*   电子邮件:`michael@realpython.com`
*   密码:`michael`

同样，您应该在 javaScript 控制台中看到一个成功的标记。

[*Remove ads*](/account/join/)

## 授权寄存器

就像登录功能一样，我们需要添加一个组件来注册用户。首先生成一个新的寄存器组件:

```py
$ ng generate component components/register
```

更新*src/app/components/register/register . component . html*:

```py
<div class="row">
  <div class="col-md-4">
    <h1>Register</h1>
    <hr><br>
    <form (ngSubmit)="onRegister()" novalidate>
     <div class="form-group">
       <label for="email">Email</label>
       <input type="text" class="form-control" id="email" placeholder="enter email" [(ngModel)]="user.email" name="email" required>
     </div>
     <div class="form-group">
       <label for="password">Password</label>
       <input type="password" class="form-control" id="password" placeholder="enter password" [(ngModel)]="user.password" name="password" required>
     </div>
     
    </form>
  </div>
</div>
```

然后，更新*src/app/components/register/register . component . ts*如下:

```py
import  {  Component  }  from  '@angular/core'; import  {  AuthService  }  from  '../../services/auth.service'; import  {  User  }  from  '../../models/user'; @Component({ selector:  'register', templateUrl:  './register.component.html', styleUrls:  ['./register.component.css'] }) export  class  RegisterComponent  { user:  User  =  new  User(); constructor(private  auth:  AuthService)  {} onRegister():  void  { this.auth.register(this.user) .then((user)  =>  { console.log(user.json()); }) .catch((err)  =>  { console.log(err); }); } }
```

向 *app.module.ts* 文件添加一个新的路由处理程序:

```py
RouterModule.forRoot([ {  path:  'login',  component:  LoginComponent  }, {  path:  'register',  component:  RegisterComponent  } ])
```

通过注册一个新用户来测试它！

## 本地存储

接下来，让我们通过将*src/app/components/log in/log in . component . ts*中的`console.log(user.json());`替换为`localStorage.setItem('token', user.data.token);`,将令牌添加到本地存储中进行持久化:

```py
onLogin():  void  { this.auth.login(this.user) .then((user)  =>  { localStorage.setItem('token',  user.json().auth_token); }) .catch((err)  =>  { console.log(err); }); }
```

在*src/app/components/register/register . component . ts*内做同样的操作:

```py
onRegister():  void  { this.auth.register(this.user) .then((user)  =>  { localStorage.setItem('token',  user.json().auth_token); }) .catch((err)  =>  { console.log(err); }); }
```

只要该令牌存在，就可以认为用户已经登录。而且，当用户需要发出 AJAX 请求时，可以使用这个令牌。

> **注意:**除了令牌，您还可以将用户 id 和电子邮件添加到本地存储。您只需要更新服务器端，以便在用户登录时发送回该信息。

测试一下。登录后，确保令牌存在于本地存储中。

## 用户状态

为了测试登录持久性，我们可以添加一个新的视图来验证用户是否登录以及令牌是否有效。

将以下方法添加到`AuthService`:

```py
ensureAuthenticated(token):  Promise<any>  { let  url:  string  =  `${this.BASE_URL}/status`; let  headers:  Headers  =  new  Headers({ 'Content-Type':  'application/json', Authorization:  `Bearer ${token}` }); return  this.http.get(url,  {headers:  headers}).toPromise(); }
```

记下`Authorization: 'Bearer ' + token`。这被称为[承载模式](https://security.stackexchange.com/questions/108662/why-is-bearer-required-before-the-token-in-authorization-header-in-a-http-re)，它随请求一起发送。在服务器上，我们只是检查`Authorization`头，然后检查令牌是否有效。你能在服务器端找到这段代码吗？

然后，生成一个新的状态组件:

```py
$ ng generate component components/status
```

创建 HTML 模板，*src/app/components/status/status . component . HTML*:

```py
<div class="row">
  <div class="col-md-4">
    <h1>User Status</h1>
    <hr><br>
    <p>Logged In? {{isLoggedIn}}</p>
  </div>
</div>
```

并更改*src/app/components/status/status . component . ts*中的组件代码:

```py
import  {  Component,  OnInit  }  from  '@angular/core'; import  {  AuthService  }  from  '../../services/auth.service'; @Component({ selector:  'status', templateUrl:  './status.component.html', styleUrls:  ['./status.component.css'] }) export  class  StatusComponent  implements  OnInit  { isLoggedIn:  boolean  =  false; constructor(private  auth:  AuthService)  {} ngOnInit():  void  { const  token  =  localStorage.getItem('token'); if  (token)  { this.auth.ensureAuthenticated(token) .then((user)  =>  { console.log(user.json()); if  (user.json().status  ===  'success')  { this.isLoggedIn  =  true; } }) .catch((err)  =>  { console.log(err); }); } } }
```

最后，向 *app.module.ts* 文件添加一个新的路由处理程序:

```py
RouterModule.forRoot([ {  path:  'login',  component:  LoginComponent  }, {  path:  'register',  component:  RegisterComponent  }, {  path:  'status',  component:  StatusComponent  } ])
```

准备测试了吗？登录，然后导航到[http://localhost:4200/status](http://localhost:4200/status)。如果本地存储中有令牌，您应该会看到:

```py
{ "message":  "Signature expired. Please log in again.", "status":  "fail" }
```

为什么？嗯，如果你在服务器端深入挖掘，你会发现这个令牌在 *project/server/models.py* 中只有效 5 秒钟:

```py
def encode_auth_token(self, user_id):
    """
 Generates the Auth Token
 :return: string
 """
    try:
        payload = {
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=0, seconds=5),
            'iat': datetime.datetime.utcnow(),
            'sub': user_id
        }
        return jwt.encode(
            payload,
            app.config.get('SECRET_KEY'),
            algorithm='HS256'
        )
    except Exception as e:
        return e
```

将此更新为 1 天:

```py
'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1, seconds=0)
```

然后再测试一次。您现在应该会看到类似这样的内容:

```py
{ "data":  { "admin":  false, "email":  "michael@realpython.com", "registered_on":  "Sun, 13 Aug 2017 17:21:52 GMT", "user_id":  4 }, "status":  "success" }
```

最后，让我们在用户成功注册或登录后重定向到状态页面。更新*src/app/components/log in/log in . component . ts*像这样:

```py
import  {  Component  }  from  '@angular/core'; import  {  Router  }  from  '@angular/router'; import  {  AuthService  }  from  '../../services/auth.service'; import  {  User  }  from  '../../models/user'; @Component({ selector:  'login', templateUrl:  './login.component.html', styleUrls:  ['./login.component.css'] }) export  class  LoginComponent  { user:  User  =  new  User(); constructor(private  router:  Router,  private  auth:  AuthService)  {} onLogin():  void  { this.auth.login(this.user) .then((user)  =>  { localStorage.setItem('token',  user.json().auth_token); this.router.navigateByUrl('/status'); }) .catch((err)  =>  { console.log(err); }); } }
```

然后更新*src/app/components/register/register . component . ts*:

```py
import  {  Component  }  from  '@angular/core'; import  {  Router  }  from  '@angular/router'; import  {  AuthService  }  from  '../../services/auth.service'; import  {  User  }  from  '../../models/user'; @Component({ selector:  'register', templateUrl:  './register.component.html', styleUrls:  ['./register.component.css'] }) export  class  RegisterComponent  { user:  User  =  new  User(); constructor(private  router:  Router,  private  auth:  AuthService)  {} onRegister():  void  { this.auth.register(this.user) .then((user)  =>  { localStorage.setItem('token',  user.json().auth_token); this.router.navigateByUrl('/status'); }) .catch((err)  =>  { console.log(err); }); } }
```

测试一下！

[*Remove ads*](/account/join/)

## 路线限制

现在，所有的路线都是开放的；因此，无论用户是否登录，他们都可以访问每条路线。如果用户未登录，则应限制某些路由，而如果用户登录，则应限制其他路由:

1.  `/` -没有限制
2.  `/login` -登录时受限
3.  `/register` -登录时受限
4.  `/status` -未登录时受限

为了实现这一点，根据您是希望将用户引导到`status`视图还是`login`视图，向每条路线添加`EnsureAuthenticated`或`LoginRedirect`。

首先创建两个新服务:

```py
$ ng generate service services/ensure-authenticated
$ ng generate service services/login-redirect
```

替换*确保-认证.服务. ts* 文件中的代码如下:

```py
import  {  Injectable  }  from  '@angular/core'; import  {  CanActivate,  Router  }  from  '@angular/router'; import  {  AuthService  }  from  './auth.service'; @Injectable() export  class  EnsureAuthenticated  implements  CanActivate  { constructor(private  auth:  AuthService,  private  router:  Router)  {} canActivate():  boolean  { if  (localStorage.getItem('token'))  { return  true; } else  { this.router.navigateByUrl('/login'); return  false; } } }
```

并替换*log in-redirect . service . ts*中的代码，如下所示:

```py
import  {  Injectable  }  from  '@angular/core'; import  {  CanActivate,  Router  }  from  '@angular/router'; import  {  AuthService  }  from  './auth.service'; @Injectable() export  class  LoginRedirect  implements  CanActivate  { constructor(private  auth:  AuthService,  private  router:  Router)  {} canActivate():  boolean  { if  (localStorage.getItem('token'))  { this.router.navigateByUrl('/status'); return  false; } else  { return  true; } } }
```

最后，更新 *app.module.ts* 文件以导入和配置新服务:

```py
import  {  BrowserModule  }  from  '@angular/platform-browser'; import  {  NgModule  }  from  '@angular/core'; import  {  RouterModule  }  from  '@angular/router'; import  {  HttpModule  }  from  '@angular/http'; import  {  FormsModule  }  from  '@angular/forms'; import  {  AppComponent  }  from  './app.component'; import  {  LoginComponent  }  from  './components/login/login.component'; import  {  AuthService  }  from  './services/auth.service'; import  {  RegisterComponent  }  from  './components/register/register.component'; import  {  StatusComponent  }  from  './components/status/status.component'; import  {  EnsureAuthenticated  }  from  './services/ensure-authenticated.service'; import  {  LoginRedirect  }  from  './services/login-redirect.service'; @NgModule({ declarations:  [ AppComponent, LoginComponent, RegisterComponent, StatusComponent, ], imports:  [ BrowserModule, HttpModule, FormsModule, RouterModule.forRoot([ { path:  'login', component:  LoginComponent, canActivate:  [LoginRedirect] }, { path:  'register', component:  RegisterComponent, canActivate:  [LoginRedirect] }, { path:  'status', component:  StatusComponent, canActivate: [EnsureAuthenticated] } ]) ], providers:  [ AuthService, EnsureAuthenticated, LoginRedirect ], bootstrap:  [AppComponent] }) export  class  AppModule  {  }
```

请注意我们是如何将我们的服务添加到新的 route 属性中的。路由系统使用`canActivate`数组中的服务来确定是否显示请求的 URL 路径。如果路线有`LoginRedirect`并且用户已经登录，那么他们将被重定向到`status`视图。如果用户试图访问需要认证的 URL，包含`EnsureAuthenticated`服务会将用户重定向到`login`视图。

最后测试一次。

## 下一步是什么？

在本教程中，我们经历了使用 JSON Web 令牌向 Angular 4 + Flask 应用程序添加身份验证的过程。

下一步是什么？

尝试使用以下端点将 Flask 后端切换到不同的 web 框架，如 Django 或 Bottle:

*   `/auth/register`
*   `/auth/login`
*   `/auth/logout`
*   `/auth/user`

如果您想了解如何使用 Flask 构建完整的 Python web 应用程序，请查看此视频系列:

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

在下面添加问题和/或评论。从 [angular4-auth](https://github.com/realpython/angular4-auth) repo 中抓取最终代码。*****