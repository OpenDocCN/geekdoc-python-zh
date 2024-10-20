# 如何用 Python 构建货币转换器

> 原文：<https://www.askpython.com/python/build-currency-converter-python>

你曾经尝试过提供一种产品，一种服务，或者只是想用不同的货币显示价格吗？那么你就知道提供最新的、准确的汇率有多难。

这就是货币兑换 API 的用武之地。外汇 API 帮助您处理您的外汇汇率转换。在这个例子中，我们将看看如何使用 Flask web 框架和一些用于前端样式的 Javascript 将货币 API 集成到一个简单的 Python 应用程序中，以便您可以构建自己的货币转换器。

## 如何用 Python 创建货币转换器的分步指南

首先，我们将建立我们的开发堆栈:

*   [Python 3](https://www.python.org/downloads/) ( > 3.7)
*   [烧瓶](https://flask.palletsprojects.com/en/2.1.x/installation/)
*   [Javascript](https://nodejs.org/en/) (节点)
*   纱线(npm 安装-全球纱线)
*   轻快地
*   顺风 CSS
*   postcss
*   自动贴合
*   免费的[currencyapi.com](https://currencyapi.com/)API 密钥

### 步骤 1:初始化我们的前端项目

首先，我们需要在开发工作区中初始化一个 Vite 项目:

```py
yarn create vite currency-converter --template vanilla

```

### 步骤 2:样式设置(可选)

样式是可选的，但是如果您选择遵循这个步骤，我们推荐使用 Tailwind CSS。Autoprefixer & postcss 进一步实现了流畅的开发体验。因此，我们需要安装这些软件包:

```py
yarn add -D tailwindcss postcss autoprefixer

```

现在我们可以初始化顺风。这将创建一个配置文件(tailwind.config.js):

```py
npx tailwindcss init

```

我们现在需要修改这个新创建的配置，以便与我们的 Vite 项目设置一起工作:

```py
module.exports = {
 content: [
   './main.js',
   './index.html',
 ],
 theme: {
   extend: {},
 },
 plugins: [],
}

```

要包含 Tailwind CSS，请在 style.css 的顶部添加以下代码行:

```py
@tailwind base;
@tailwind components;
@tailwind utilities;

```

接下来，我们需要在 postcss 的根目录下创建一个名为 postcss.config.js 的配置文件。因此，我们补充:

```py
module.exports = {
	plugins: [
    	require('tailwindcss'),
    	require('autoprefixer')
	]
}

```

### 第三步:开始邀请

我们现在可以在开发模式下启动 vite，通过热重装来提供我们的文件:

```py
yarn dev

```

### 步骤 4:准备我们的 HTML

接下来，我们要修改默认的登录页面。为此，我们打开 index.html 并构建一个表单。我们将需要以下要素:

*   我们输入的包装器
*   本方基础货币的输入:
*   基础货币选择`<select id= currency >`提交按钮一个响应容器下面是我们实现 index.html 的样子:

```py
<!DOCTYPE html> <html lang="en"> <head> <meta charset="UTF-8" /> <link rel="icon" type="image/svg+xml" href="favicon.svg" /> <meta name="viewport" content="width=device-width, initial-scale=1.0" /> <title>Currency converter example</title> </head> <body class="bg-gradient-to-b from-cyan-800 to-slate-800 min-h-screen py-5"> <form id="currency_converter" class="mx-auto w-full max-w-sm bg-white shadow rounded-md p-5 space-y-3 text-sm"> <div class="flex items-center space-x-5"> <label for="base_currency_input">Amount:</label> <input type="tel" id="base_currency_input" name="base_currency_input" placeholder="1" value="" class="grow border-slate-300 border rounded-md py-2 px-4 text-sm" required /> </div> <div class="flex items-center space-x-5"> <label for="currency">Currency:</label> <select name="currency" id="currency" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"> <option selected value="USD">USD</option> <option value="EUR">EUR</option> <option value="CHF">CHF</option> </select> </div> <button type="submit" class="bg-slate-800 text-white rounded-md py-2 px-4 mx-auto relative block w-full">Convert </button> </form> <div id="result" class="mx-auto my-5 w-full max-w-sm bg-white shadow rounded-md relative overflow-hidden text-sm empty:hidden divide-y divide-dotted divide-slate-300"> </div> <script type="module" src="/main.js"></script> </body> </html>
```

步骤 5:用 JavaScript 处理表单提交

在 main.js 中，我们将货币金额(` base_currency_input `)和货币(` base_currency `)发送到后端。我们将在响应中收到所有货币和相应值的列表。

```py
import './style.css' const currencyConverter = document.getElementById('currency_converter'); const baseCurrencyInput = document.getElementById('base_currency_input'); const baseCurrency = document.getElementById('currency'); const resultContainer = document.getElementById('result'); currencyConverter.addEventListener('submit', (e) => { e.preventDefault(); fetch(`http://localhost:6001/convert?` + new URLSearchParams({ 'base_currency_input': baseCurrencyInput.value, 'currency': baseCurrency.value })) .then(response => response.json()) .then(data => { var result = '<div class="space-y-1 px-5 py-3 border-2 rounded-md">'; for (let entry of data) { result += `<div class="flex items-baseline justify-between"><span class="font-medium">${entry.code}:</span><span>${entry.value}</span></div>`; } resultContainer.innerHTML = result; }); }); 
```

步骤 6:准备后端应用程序

现在，我们创建一个新文件夹，即“货币转换器”文件夹中的“后端应用程序”:  注意:命令对 macOS/Linux 有效；对于 Windows，请在这里勾选。

```py
mkdir backend-application cd backend-application python3 –m venv venv . venv/bin/activate pip install Flask currencyapicom 
```

步骤 7:创建后端应用程序

在最后一步中，我们只需添加一个名为“main.py”的新文件:

```py
from flask import Flask, request, jsonify from currencyapicom import Client from config import CURRENCYAPI_KEY app = Flask(__name__) @app.route("/convert", methods=['GET']) def convert(): currency_input = request.args.get('base_currency_input', '') currency = request.args.get('currency', 'USD') if currency_input and currency in ['USD', 'EUR', 'CHF']: api_client = Client(CURRENCYAPI_KEY) response = api_client.latest(currency) response = jsonify([{'value': response['data'][x]['value'] * float(currency_input), 'code': x} for x in response['data'].keys()]) response.headers.add("Access-Control-Allow-Origin", "*") return response
```

我们可以用几个简单的命令运行应用程序(我们将端口绑定到 6001 以避免与其他应用程序冲突):

```py
export FLASK_APP=main flask run –port 6001 
```

在最后一步，我们需要创建一个“config.py”文件，包括 currencyapi.com API 密钥。您可以免费获得它，并在文档中了解更多关于 API 的信息。就是这样！  最新和准确通过这几个步骤，您现在可以使用 Python 构建自己的货币转换器，并以不同的货币显示准确和最新的价格。货币转换器有许多使用案例；无论您是将它用于电子商务商店、分析还是电子表格，我们都希望本教程能指导您完成这一过程。