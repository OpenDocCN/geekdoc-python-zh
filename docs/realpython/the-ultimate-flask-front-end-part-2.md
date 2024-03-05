# 终极烧瓶前端–第 2 部分

> 原文：<https://realpython.com/the-ultimate-flask-front-end-part-2/>

欢迎来到第 2 部分！同样，这也是我们要讨论的内容:**让我们看看小而强大的 JavaScript UI 库 [ReactJS](http://facebook.github.io/react/) 在构建一个基本的 web 应用程序时的表现。**

这款应用由 Python 3 和后端的 [Flask 框架](http://flask.pocoo.org/)以及前端的 [React](https://facebook.github.io/react/) 提供支持。另外我们会用到 [gulp.js](http://gulpjs.com/) (任务运行器) [bower](http://bower.io/) (前端包管理器) [Browserify](http://browserify.org/) (JavaScript 依赖捆绑器)。

*   [第 1 部分–入门](/the-ultimate-flask-front-end/)
*   **第 2 部分——开发动态搜索工具(当前)**

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

*更新:*

*   05/22/2016:升级至 React 最新版本(v [15.0.1](http://facebook.github.io/react/blog/2016/04/08/react-v15.0.1.html) )。

## 反应–第二轮

Hello World 很棒，但是让我们创建一个更有趣的东西——一个动态搜索工具。这个[组件](http://facebook.github.io/react/docs/reusable-components.html)用于根据用户输入过滤出一个条目列表。

> 需要密码吗？从[回购](https://github.com/realpython/ultimate-flask-front-end/tags)中抢过来。下载 [v2.2](https://github.com/realpython/ultimate-flask-front-end/releases/tag/v2.2) 从我们离开第 1 部分的地方开始。

将以下代码添加到*project/static/scripts/jsx/main . js*中:

```py
var  DynamicSearch  =  React.createClass({ // sets initial state getInitialState:  function(){ return  {  searchString:  ''  }; }, // sets state, triggers render method handleChange:  function(event){ // grab value form input box this.setState({searchString:event.target.value}); console.log("scope updated!"); }, render:  function()  { var  countries  =  this.props.items; var  searchString  =  this.state.searchString.trim().toLowerCase(); // filter countries list by value from input box if(searchString.length  >  0){ countries  =  countries.filter(function(country){ return  country.name.toLowerCase().match(  searchString  ); }); } return  ( <div> <input  type="text"  value={this.state.searchString}  onChange={this.handleChange}  placeholder="Search!"  /> <ul> {  countries.map(function(country){  return  <li>{country.name}  </li> }) } </ul> </div> ) } }); // list of countries, defined with JavaScript object literals var  countries  =  [ {"name":  "Sweden"},  {"name":  "China"},  {"name":  "Peru"},  {"name":  "Czech Republic"}, {"name":  "Bolivia"},  {"name":  "Latvia"},  {"name":  "Samoa"},  {"name":  "Armenia"}, {"name":  "Greenland"},  {"name":  "Cuba"},  {"name":  "Western Sahara"},  {"name":  "Ethiopia"}, {"name":  "Malaysia"},  {"name":  "Argentina"},  {"name":  "Uganda"},  {"name":  "Chile"}, {"name":  "Aruba"},  {"name":  "Japan"},  {"name":  "Trinidad and Tobago"},  {"name":  "Italy"}, {"name":  "Cambodia"},  {"name":  "Iceland"},  {"name":  "Dominican Republic"},  {"name":  "Turkey"}, {"name":  "Spain"},  {"name":  "Poland"},  {"name":  "Haiti"} ]; ReactDOM.render( <DynamicSearch  items={  countries  }  />, document.getElementById('main') );
```

**到底怎么回事？**

我们创建了一个名为`DynamicSearch`的组件，当输入框中的值改变时，它会更新 DOM。这是如何工作的？当在输入框中添加或删除一个值时，调用`handleChange()`函数，然后通过`setState()`更新状态。这个方法然后调用`render()`函数来重新呈现组件。

这里的关键要点是，状态变化只发生在组件内部。

测试一下:

[//jsfiddle.net/mjhea0/2qn7ktq3/embedded/result,html/](//jsfiddle.net/mjhea0/2qn7ktq3/embedded/result,html/)

由于我们将 JSX 代码添加到了一个外部文件中，我们需要在浏览器之外用 Gulp 触发从 JSX 到普通 JavaScript 的转换。

[*Remove ads*](/account/join/)

## 一饮而尽

[Gulp](http://gulpjs.com/) 是一个强大的任务运行器/构建工具，可用于自动化转换过程。我们还将使用它来监视我们代码的变更(*project/static/scripts/jsx/main . js*)，并基于这些变更自动创建新的构建。

### 初始化

和 bower 一样，可以用 npm 安装 gulp。全局安装，然后添加到 *package.json* 文件:

```py
$ npm install -g gulp
$ npm install --save-dev gulp
```

在项目的根目录下添加一个 *gulpfile.js* :

```py
// requirements var  gulp  =  require('gulp'); // tasks gulp.task('default',  function()  { console.log("hello!"); });
```

这个文件告诉 gulp 运行哪些任务，以及以什么顺序运行它们。您可以看到我们的`default`任务将字符串`hello!`记录到控制台。你可以通过运行`gulp`来运行这个任务。您应该会看到类似这样的内容:

```py
$ gulp
[08:54:47] Using gulpfile ~/gulpfile.js
[08:54:47] Starting 'default'...
hello!
[08:54:47] Finished 'default' after 148 μs
```

我们需要安装以下 gulp 插件:

*   [德尔](https://github.com/sindresorhus/del)
*   [大口尺寸](https://github.com/sindresorhus/gulp-size)
*   [大口浏览器](https://www.npmjs.com/package/gulp-browser)
*   [反应化](https://www.npmjs.com/package/reactify)

为此，只需更新 *package.json* 文件:

```py
{ "name":  "ultimate-flask-front-end", "version":  "1.0.0", "description":  "", "main":  "index.js", "scripts":  { "test":  "echo \"Error: no test specified\" && exit 1" }, "repository":  { "type":  "git", "url":  "git+https://github.com/realpython/ultimate-flask-front-end.git" }, "author":  "", "license":  "ISC", "bugs":  { "url":  "https://github.com/realpython/ultimate-flask-front-end/issues" }, "homepage":  "https://github.com/realpython/ultimate-flask-front-end#readme", "devDependencies":  { "bower":  "^1.7.9", "del":  "^2.2.0", "gulp":  "^3.9.1", "gulp-browser":  "^2.1.4", "gulp-size":  "^2.1.0", "reactify":  "^1.1.1" } }
```

然后运行`npm install`安装插件。

> 您可以在“node_modules”目录中看到这些已安装的插件。确保将该目录包含在您的*中。gitignore* 文件。

最后更新 *gulpfile.js* :

```py
// requirements var  gulp  =  require('gulp'); var  gulpBrowser  =  require("gulp-browser"); var  reactify  =  require('reactify'); var  del  =  require('del'); var  size  =  require('gulp-size'); // tasks gulp.task('transform',  function  ()  { // add task }); gulp.task('del',  function  ()  { // add task }); gulp.task('default',  function()  { console.log("hello!"); });
```

## 任务

现在让我们添加一些任务，从`transform`开始，处理从 JSX 到 JavaScript 的转换过程。

### 第一个任务—`transform`

```py
gulp.task('transform',  function  ()  { var  stream  =  gulp.src('./project/static/scripts/jsx/*.js') .pipe(gulpBrowser.browserify({transform:  ['reactify']})) .pipe(gulp.dest('./project/static/scripts/js/')) .pipe(size()); return  stream; });
```

`task()`函数有两个参数——任务名和一个匿名函数:

*   定义源目录，
*   将 JSX 文件通过浏览器传输到 JSX 转换器
*   指定目标目录，并且
*   计算创建的文件的大小。

Gulp 利用管道[来传输](http://nodejs.org/api/stream.html)数据进行处理。在抓取源文件( *main.js* )之后，该文件被“输送”到`browserify()`函数进行转换/绑定。然后，这些经过转换和捆绑的代码将与`size()`函数一起被“输送”到目标目录。

> 对管道和溪流感到好奇？查看[这个](https://github.com/substack/stream-handbook)优秀的资源。

准备好快速测试了吗？更新*index.html*:

```py
<!DOCTYPE html>
<html>
  <head lang="en">
    <meta charset="UTF-8">
    <title>Flask React</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- styles -->
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='bower_components/bootstrap/dist/css/bootstrap.min.css') }}">
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/style.css') }}">
  </head>
  <body>
    <div class="container">
      <h1>Flask React</h1>
      <br>
      <div id="main"></div>
    </div>
    <!-- scripts -->
    <script src="{{ url_for('static', filename='bower_components/react/react.min.js') }}"></script>
    <script src="{{ url_for('static', filename='bower_components/react/react-dom.min.js') }}"></script>
    <script src="{{ url_for('static', filename='scripts/js/main.js') }}"></script>
  </body>
</html>
```

然后更新`default`任务:

```py
gulp.task('default',  function  ()  { gulp.start('transform'); });
```

测试:

```py
$ gulp
[08:58:39] Using gulpfile /gulpfile.js
[08:58:39] Starting 'default'...
[08:58:39] Starting 'transform'...
[08:58:39] Finished 'default' after 12 ms
[08:58:40] all files 1.99 kB
[08:58:40] Finished 'transform' after 181 ms
```

你注意到倒数第二行了吗？这是`size()`函数的结果。换句话说，新创建的 JavaScript 文件(转换后)，*project/static/scripts/js/main . js*，大小为 1.99 kB。

启动 Flask 服务器，导航到 [http://localhost:5000/](http://localhost:5000/hello) 。你应该可以在搜索框中看到所有的国家。[测试](https://realpython.com/python-testing/)功能。此外，如果您在 Chrome Developer Tools 中打开 JavaScript 控制台，您会看到每次范围发生变化时都会记录的字符串`scope updated!`——它来自于`DynamicSearch`组件中的`handleChange()`函数。

[*Remove ads*](/account/join/)

### 第二个任务—`clean`*

```py
gulp.task('del',  function  ()  { return  del(['./project/static/scripts/js']); });
```

当这个任务运行时，我们获取源目录——`transform`任务的结果——然后运行`del()`函数来删除目录及其内容。这是一个好主意，在每一个新的构建之前运行它，以确保你的开始是全新的和干净的。

尝试运行`gulp del`。这应该会删除“项目/静态/脚本/js”。让我们将它添加到我们的`default`任务中，以便它在转换之前自动运行:

```py
gulp.task('default',  ['del'],  function  ()  { gulp.start('transform'); });
```

在继续之前，请务必对此进行测试。

### 第三个任务—`watch`

最后，最后一次更新`default`任务，添加在任何*发生变化时自动重新运行`transform`任务的能力。“项目/静态/脚本/jsx”目录下的 js* 文件:

```py
gulp.task('default',  ['del'],  function  ()  { gulp.start('transform'); gulp.watch('./project/static/scripts/jsx/*.js',  ['transform']); });
```

打开一个新的终端窗口，导航到项目根目录，运行`gulp`生成一个新的构建并激活`watcher`功能。在另一个窗口中运行`sh run.sh`来运行 Flask 服务器。您的应用程序应该正在运行。现在如果你注释掉*项目/static/scripts/jsx/main.js* 文件中的所有代码，这将触发`transform`函数。刷新浏览器以查看更改。完成后，请确保恢复更改。

想更进一步吗？查看一下 [Livereload](https://github.com/vohof/gulp-livereload) 插件。

## 结论

下面是给*project/static/CSS/style . CSS*添加一些自定义样式后的最终结果:

[![Flask React dynamic search](img/2227d5bc1338ce8556fc97d0c9e9d77c.png)](https://files.realpython.com/media/flask-react-dynamic-search.1575d97d167e.png)

请务必查看官方[文档](http://facebook.github.io/react/)以及优秀的[教程](http://facebook.github.io/react/docs/tutorial.html)以获得更多关于 React 的信息。

从[回购](https://github.com/realpython/ultimate-flask-front-end)中抓取代码。如果你想更深入地研究 Flask，看看如何用它从头开始构建一个完整的 web 应用程序，一定要看看这个视频系列:

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

如果您有任何问题或发现任何错误，请在下面评论。还有，你还想看什么？如果人们感兴趣，我们可以增加第三部分。下面评论。**