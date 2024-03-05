# 使用 AWS Chalice 构建无服务器 Python 应用程序

> 原文：<https://realpython.com/aws-chalice-serverless-python/>

发布一个 web 应用程序通常需要在一台或多台服务器上运行您的代码。在这种模式下，您最终需要设置监控、配置和扩展服务器的流程。虽然这看起来工作得很好，但是以自动化的方式处理 web 应用程序的所有后勤工作减少了大量的人工开销。输入无服务器。

使用[无服务器架构](https://aws.amazon.com/lambda/serverless-architectures-learn-more/)，您不需要管理服务器。相反，您只需要将代码或可执行包发送到执行它的平台。并不是真的没有服务器。服务器确实存在，但是开发者不需要担心它们。

AWS 推出了 [Lambda Services](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html) ，这是一个平台，使开发人员能够简单地在特定的运行时环境中执行他们的代码。为了使平台易于使用，许多社区围绕它提出了一些非常好的框架，以使无服务器应用程序成为一个工作解决方案。

**本教程结束时，你将能够**:

*   讨论无服务器架构的优势
*   探索 Chalice，一个 Python 无服务器框架
*   为真实世界的使用案例构建成熟的无服务器应用
*   部署到 Amazon Web Services (AWS) Lambda
*   比较纯函数和 Lambda 函数

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## AWS 圣杯入门

[Chalice](https://github.com/aws/chalice/) ，一个由 AWS 开发的 Python 无服务器微框架，使您能够快速启动和部署一个工作的无服务器应用程序，该应用程序可以使用 AWS Lambda 根据需要自行伸缩。

[*Remove ads*](/account/join/)

### 为什么是圣杯？

对于习惯于 Flask web 框架的 Python 开发人员来说，在构建和发布您的第一个应用程序方面，Chalice 应该是轻而易举的事情。受到 Flask 的高度启发，Chalice 在定义服务应该是什么样的以及最终制作相同的可执行包方面保持了相当的极简主义。

理论够了！让我们从一个基本的`hello-world`应用开始，开始我们的无服务器之旅。

### 项目设置

在进入 Chalice 之前，您将在本地机器上设置一个工作环境，这将为您完成本教程的其余部分做好准备。

首先，创建并激活一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)并安装 Chalice:

```py
$ python3.6 -m venv env
$ source env/bin/activate
(env)$ pip install chalice
```

遵循我们关于 [Pipenv 包装工具](https://realpython.com/pipenv-guide/)的全面指南。

**注意:** Chalice 带有一个用户友好的 CLI，可以轻松使用您的无服务器应用程序。

现在您已经在虚拟环境中安装了 Chalice，让我们使用 Chalice CLI 来生成一些样板代码:

```py
(env)$ chalice new-project
```

出现提示时输入项目名称，然后按 return 键。以该名称创建一个新目录:

```py
<project-name>/
|
├── .chalice/
│   └── config.json
|
├── .gitignore
├── app.py
└── requirements.txt
```

看看圣杯代码库有多简约。一个`.chalice`目录、`app.py`和`requirements.txt`是启动和运行一个无服务器应用程序所需要的全部。让我们在本地机器上快速运行这个应用程序。

Chalice CLI 包含非常棒的实用函数，允许您执行大量操作，从本地运行到在 Lambda 环境中部署。

### 在本地构建和运行

您可以通过使用 Chalice 的`local`实用程序在本地运行该应用程序来模拟它:

```py
(env)$ chalice local
Serving on 127.0.0.1:8000
```

默认情况下，Chalice 运行在端口 8000 上。我们现在可以通过发出一个 [curl 请求](http://www.codingpedia.org/ama/how-to-test-a-rest-api-from-command-line-with-curl/)到`http://localhost:8000/`来检查索引路径:

```py
$ curl -X GET http://localhost:8000/
{"hello": "world"}
```

现在，如果我们看一下`app.py`，我们可以体会到 Chalice 允许您构建无服务器服务的简单性。所有复杂的东西都由装饰者处理:

```py
from chalice import Chalice
app = Chalice(app_name='serverless-sms-service')

@app.route('/')
def index():
    return {'hello': 'world'}
```

**注意**:我们还没有命名我们的应用`hello-world`，因为我们将在同一个应用上建立我们的短信服务。

现在，让我们继续在 AWS Lambda 上部署我们的应用程序。

[*Remove ads*](/account/join/)

### 在 AWS Lambda 上部署

Chalice 使部署您的无服务器应用程序完全不费力。使用`deploy`实用程序，您可以简单地指示 Chalice 部署并创建一个 Lambda 函数，该函数可以通过 REST API 访问。

在我们开始部署之前，我们需要确保我们有 AWS 证书，通常位于`~/.aws/config`。该文件的内容如下所示:

```py
[default] aws_access_key_id=<your-access-key-id> aws_secret_access_key=<your-secret-access-key> region=<your-region>
```

有了 AWS 凭证，让我们从一个命令开始部署过程:

```py
(env)$ chalice deploy
Creating deployment package.
Updating policy for IAM role: hello-world-dev
Creating lambda function: hello-world-dev
Creating Rest API
Resources deployed:
 - Lambda ARN: arn:aws:lambda:ap-south-1:679337104153:function:hello-world-dev
 - Rest API URL: https://fqcdyzvytc.execute-api.ap-south-1.amazonaws.com/api/
```

**注意**:上面代码片段中生成的 ARN 和 API URL 会因用户而异。

哇！是的，启动和运行您的无服务器应用程序真的很容易。要进行验证，只需在生成的 Rest API URL 上发出 curl 请求:

```py
$ curl -X GET https://fqcdyzvytc.execute-api.ap-south-1.amazonaws.com/api/
{"hello": "world"}
```

通常情况下，这就是你启动和运行无服务器应用程序所需的全部内容。您还可以转到 AWS 控制台，查看在 Lambda service 部分下创建的 Lambda 函数。每个 Lambda 服务都有一个惟一的 REST API 端点，可以在任何 web 应用程序中使用。

接下来，您将开始使用 Twilio 作为 SMS 服务提供商来构建您的无服务器 SMS Sender 服务。

## 构建无服务器手机短信服务

部署了一个基本的`hello-world`应用程序后，让我们继续构建一个可以与日常 web 应用程序一起使用的更真实的应用程序。在本节中，您将构建一个完全无服务器的 SMS 发送应用程序，只要输入参数正确，它可以插入任何系统并按预期工作。

为了发送短信，我们将使用 [Twilio](https://www.twilio.com) ，一个开发者友好的短信服务。在开始使用 Twilio 之前，我们需要考虑一些先决条件:

*   创建一个账户，获得`ACCOUNT_SID`和`AUTH_TOKEN`。
*   获得一个手机号码，这是在 Twilio 免费提供的小测试的东西。
*   使用`pip install twilio`在我们的虚拟环境中安装`twilio`包。

检查完以上所有先决条件后，您就可以开始使用 Twilio 的 Python 库构建您的 SMS 服务客户端了。让我们从克隆[库](https://github.com/realpython/materials/pull/16)并创建一个新的特性分支开始:

```py
$ git clone <project-url>
$ cd <project-dir>
$ git checkout tags/1.0 -b twilio-support
```

现在对`app.py`做如下修改，使它从一个简单的`hello-world`应用程序发展到支持 Twilio 服务。

首先，让我们包括所有的导入语句:

```py
from os import environ as env

# 3rd party imports
from chalice import Chalice, Response
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Twilio Config
ACCOUNT_SID = env.get('ACCOUNT_SID')
AUTH_TOKEN = env.get('AUTH_TOKEN')
FROM_NUMBER = env.get('FROM_NUMBER')
TO_NUMBER = env.get('TO_NUMBER')
```

接下来，您将封装 Twilio API 并使用它来发送 SMS:

```py
app = Chalice(app_name='sms-shooter')

# Create a Twilio client using account_sid and auth token
tw_client = Client(ACCOUNT_SID, AUTH_TOKEN)

@app.route('/service/sms/send', methods=['POST'])
def send_sms():
    request_body = app.current_request.json_body
    if request_body:
        try:
            msg = tw_client.messages.create(
                from_=FROM_NUMBER,
                body=request_body['msg'],
                to=TO_NUMBER)

            if msg.sid:
                return Response(status_code=201,
                                headers={'Content-Type': 'application/json'},
                                body={'status': 'success',
                                      'data': msg.sid,
                                      'message': 'SMS successfully sent'})
            else:
                return Response(status_code=200,
                                headers={'Content-Type': 'application/json'},
                                body={'status': 'failure',
                                      'message': 'Please try again!!!'})
        except TwilioRestException as exc:
            return Response(status_code=400,
                            headers={'Content-Type': 'application/json'},
                            body={'status': 'failure',
                                  'message': exc.msg})
```

在上面的代码片段中，您只需使用`ACCOUNT_SID`和`AUTH_TOKEN`创建一个 Twilio 客户端对象，并使用它在`send_sms`视图下发送消息。`send_sms`是一个基本功能，使用 Twilio 客户端的 API 将 SMS 发送到指定的目的地。在继续下一步之前，让我们尝试一下并在本地机器上运行它。

[*Remove ads*](/account/join/)

### 在本地构建和运行

现在，您可以使用`local`实用程序在您的机器上运行您的应用程序，并验证一切正常:

```py
(env)$ chalice local
```

现在，使用特定的有效负载向`http://localhost:8000/service/sms/send`发出 curl POST 请求，并在本地测试应用程序:

```py
$ curl -H "Content-Type: application/json" -X POST -d '{"msg": "hey mate!!!"}' http://localhost:8000/service/sms/send
```

上述请求答复如下:

```py
{ "status":  "success", "data":  "SM60f11033de4f4e39b1c193025bcd5cd8", "message":  "SMS successfully sent" }
```

响应表明消息已成功发送。现在，让我们继续在 AWS Lambda 上部署应用程序。

### 在 AWS Lambda 上部署

正如在前面的部署部分中所建议的，您只需要发出以下命令:

```py
(env)$ chalice deploy
Creating deployment package.
Updating policy for IAM role: sms-shooter-dev
Creating lambda function: sms-shooter-dev
Creating Rest API
Resources deployed:
 - Lambda ARN: arn:aws:lambda:ap-south-1:679337104153:function:sms-shooter-dev
 - Rest API URL: https://qtvndnjdyc.execute-api.ap-south-1.amazonaws.com/api/
```

**注意**:上面的命令成功了，您在输出中有了您的 API URL。现在在测试 URL 时，API 抛出一条错误消息。**哪里出了问题？**

根据 AWS [Lambda 日志](https://www.dropbox.com/s/ectzx2std57toaf/Screenshot%202018-11-08%20at%208.35.18%20PM.png?dl=0)，没有找到或者安装`twilio`包，所以你需要告诉 Lambda 服务安装[依赖项](https://realpython.com/courses/managing-python-dependencies/)。为此，您需要添加`twilio`作为对`requirements.txt`的依赖:

```py
twilio==6.18.1
```

其他包比如 Chalice 及其依赖项不应该包含在`requirements.txt`中，因为它们不是 Python 的 WSGI 运行时的一部分。相反，我们应该维护一个`requirements-dev.txt`，它只适用于开发环境，包含所有与 Chalice 相关的依赖项。要了解更多，请查看[这期 GitHub](https://github.com/aws/chalice/issues/803)。

一旦所有的包依赖项都被排序，您需要确保所有的环境变量都被附带，并在 Lambda 运行时被正确设置。为此，您必须以如下方式在`.chalice/config.json`中添加所有环境变量:

```py
{ "version":  "2.0", "app_name":  "sms-shooter", "stages":  { "dev":  { "api_gateway_stage":  "api", "environment_variables":  { "ACCOUNT_SID":  "<your-account-sid>", "AUTH_TOKEN":  "<your-auth-token>", "FROM_NUMBER":  "<source-number>", "TO_NUMBER":  "<destination-number>" } } } }
```

现在我们可以部署了:

```py
Creating deployment package.
Updating policy for IAM role: sms-shooter-dev
Updating lambda function: sms-shooter-dev
Updating rest API
Resources deployed:
 - Lambda ARN: arn:aws:lambda:ap-south-1:679337104153:function:sms-shooter-dev
 - Rest API URL: https://fqcdyzvytc.execute-api.ap-south-1.amazonaws.com/api/
```

通过向生成的 API 端点发出 curl 请求来进行健全性检查:

```py
$ curl -H "Content-Type: application/json" -X POST -d '{"msg": "hey mate!!!"}' https://fqcdyzvytc.execute-api.ap-south-1.amazonaws.com/api/service/sms/send
```

上述请求如预期的那样响应:

```py
{ "status":  "success", "data":  "SM60f11033de4f4e39b1c193025bcd5cd8", "message":  "SMS successfully sent" }
```

现在，你有一个完全无服务器的短信发送服务启动和运行。由于该服务的前端是一个 REST API，因此它可以作为一个可伸缩、安全和可靠的即插即用特性在其他应用程序中使用。

[*Remove ads*](/account/join/)

## 重构

最后，我们将重构我们的 SMS 应用程序，使其不完全包含`app.py`中的所有业务逻辑。相反，我们将遵循圣杯规定的最佳实践，并抽象出`chalicelib/`目录下的业务逻辑。

让我们从创建一个新分支开始:

```py
$ git checkout tags/2.0 -b sms-app-refactor
```

首先，在项目的根目录下创建一个名为`chalicelib/`的新目录，并创建一个名为`sms.py`的新文件:

```py
(env)$ mkdir chalicelib
(env)$ touch chalicelib/sms.py
```

通过对`app.py`进行抽象，用 SMS 发送逻辑更新上面创建的`chalicelib/sms.py`:

```py
from os import environ as env
from twilio.rest import Client

# Twilio Config
ACCOUNT_SID = env.get('ACCOUNT_SID')
AUTH_TOKEN = env.get('AUTH_TOKEN')
FROM_NUMBER = env.get('FROM_NUMBER')
TO_NUMBER = env.get('TO_NUMBER')

# Create a twilio client using account_sid and auth token
tw_client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send(payload_params=None):
    """ send sms to the specified number """
    msg = tw_client.messages.create(
        from_=FROM_NUMBER,
        body=payload_params['msg'],
        to=TO_NUMBER)

    if msg.sid:
        return msg
```

上面的代码片段只接受输入参数，并根据需要进行响应。现在，为了实现这一点，我们还需要对`app.py`进行修改:

```py
# Core imports
from chalice import Chalice, Response
from twilio.base.exceptions import TwilioRestException

# App level imports
from chalicelib import sms

app = Chalice(app_name='sms-shooter')

@app.route('/')
def index():
    return {'hello': 'world'}

@app.route('/service/sms/send', methods=['POST'])
def send_sms():
    request_body = app.current_request.json_body
    if request_body:
        try:
            resp = sms.send(request_body)
            if resp:
                return Response(status_code=201,
                                headers={'Content-Type': 'application/json'},
                                body={'status': 'success',
                                      'data': resp.sid,
                                      'message': 'SMS successfully sent'})
            else:
                return Response(status_code=200,
                                headers={'Content-Type': 'application/json'},
                                body={'status': 'failure',
                                      'message': 'Please try again!!!'})
        except TwilioRestException as exc:
            return Response(status_code=400,
                            headers={'Content-Type': 'application/json'},
                            body={'status': 'failure',
                                  'message': exc.msg})
```

在上面的代码片段中，所有的 SMS 发送逻辑都是从`chalicelib.sms`模块调用的，这使得视图层在可读性方面更加清晰。这种抽象允许您添加更复杂的业务逻辑，并根据需要定制功能。

## 健全性检查

重构我们的代码后，让我们确保它按预期运行。

### 在本地构建和运行

使用`local`实用程序再次运行应用程序:

```py
(env)$ chalice local
```

提出 curl 请求并验证。完成后，继续部署。

### 在 AWS Lambda 上部署

一旦您确定一切正常，现在就可以部署您的应用了:

```py
(env)$ chalice deploy
```

像往常一样，该命令成功执行，您可以验证端点。

[*Remove ads*](/account/join/)

## 结论

您现在知道如何执行以下操作:

*   根据最佳实践，使用 AWS Chalice 构建一个无服务器应用程序
*   在 Lambda 运行时环境中部署您的工作应用程序

底层的 Lambda 服务类似于纯函数，它在一组输入/输出上有一定的行为。开发精确的 Lambda 服务允许更好的测试、可读性和原子性。因为 Chalice 是一个极简框架，所以您可以只关注业务逻辑，剩下的工作就交给您了，从部署到 IAM 策略生成。这一切都只需要一个命令部署！

此外，Lambda 服务主要侧重于繁重的 CPU 处理，并按照单位时间内的请求数量以自我管理的方式进行扩展。使用无服务器架构允许你的代码库更像 SOA(面向服务的架构)。在 AWS 的生态系统中使用其他能够很好地插入 Lambda 功能的产品会更加强大。*****