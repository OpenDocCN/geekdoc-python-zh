# Python 的请求库(指南)

> 原文：<https://realpython.com/python-requests/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python**](/courses/python-requests/) 制作 HTTP 请求

[`requests`](https://requests.readthedocs.io/en/latest/) 库是在 Python 中进行 HTTP 请求的事实上的标准。它将发出请求的复杂性抽象在一个漂亮、简单的 API 后面，这样您就可以专注于与服务交互和在应用程序中使用数据。

在整篇文章中，您将看到一些`requests`必须提供的最有用的特性，以及如何针对您可能遇到的不同情况定制和优化这些特性。您还将学习如何有效地使用`requests`,以及如何防止对外部服务的请求降低应用程序的速度。

**在本教程中，您将学习如何:**

*   使用最常见的 HTTP 方法发出请求
*   **使用查询字符串和消息体定制**您的请求的标题和数据
*   **检查**来自您的请求和响应的数据
*   发出**个经过验证的**个请求
*   **配置**您的请求，以帮助防止您的应用程序备份或变慢

尽管我已经尽力包含您理解本文中的特性和示例所需的尽可能多的信息，但我确实假设您对 HTTP 有一个*非常* [的基本常识。也就是说，无论如何，你仍然可以很好地理解。](https://www.w3schools.com/tags/ref_httpmethods.asp)

既然已经解决了这个问题，让我们深入研究一下，看看如何在您的应用程序中使用`requests`!

***参加测验:****通过我们的交互式“HTTP 请求与请求库”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-requests/)

## `requests` 入门

让我们从安装`requests`库开始。为此，请运行以下命令:

```py
$ pip install requests
```

如果您更喜欢使用 [Pipenv](https://realpython.com/pipenv-guide/) 来管理 [Python 包](https://realpython.com/python-modules-packages/)，您可以运行以下代码:

```py
$ pipenv install requests
```

一旦安装了`requests`，您就可以在您的应用程序中使用它了。导入`requests`看起来像这样:

```py
import requests
```

现在你已经设置好了，是时候开始你的`requests`之旅了。你的第一个目标是学习如何提出`GET`请求。

[*Remove ads*](/account/join/)

## 获取请求

[HTTP 方法](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol#Request_methods)，比如`GET`和`POST`，决定了在发出 HTTP 请求时你试图执行的动作。除了`GET`和`POST`之外，在本教程的后面部分，您还会用到其他几种常见的方法。

最常见的 HTTP 方法之一是`GET`。`GET`方法表明您正试图从指定的资源中获取或检索数据。要发出一个`GET`请求，调用`requests.get()`。

为了测试这一点，您可以通过使用以下 URL 调用`get()`向 GitHub 的[根 REST API](https://developer.github.com/v3/#root-endpoint) 发出`GET`请求:

>>>

```py
>>> requests.get('https://api.github.com')
<Response [200]>
```

恭喜你！你已经提出了第一个请求。让我们更深入地研究一下该请求的响应。

## 回应

一个`Response`是检查请求结果的强大对象。让我们再次发出同样的请求，但是这次将返回值存储在一个[变量](https://realpython.com/python-variables/)中，这样您可以更仔细地查看它的属性和行为:

>>>

```py
>>> response = requests.get('https://api.github.com')
```

在这个例子中，您捕获了`get()`的返回值，它是`Response`的一个实例，并将它存储在一个名为`response`的变量中。您现在可以使用`response`来查看关于您的`GET`请求的结果的大量信息。

### 状态代码

您可以从`Response`收集的第一点信息是状态代码。状态代码通知您请求的状态。

例如，`200 OK`状态意味着您的请求成功，而`404 NOT FOUND`状态意味着您要寻找的资源没有找到。还有[许多其他可能的状态代码](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes)也可以让你具体了解你的请求发生了什么。

通过访问`.status_code`，您可以看到服务器返回的状态代码:

>>>

```py
>>> response.status_code
200
```

`.status_code`返回了一个`200`，这意味着您的请求成功了，服务器响应了您所请求的数据。

有时，您可能希望使用这些信息在代码中做出决策:

```py
if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')
```

按照这个逻辑，如果服务器返回一个`200`状态码，你的程序将[打印](https://realpython.com/python-print/) `Success!`。如果结果是一个`404`，你的程序将打印`Not Found`。

`requests`为您进一步简化这一过程。如果您在条件表达式中使用一个`Response`实例，如果状态代码在`200`和`400`之间，它将计算为`True`，否则为`False`。

因此，您可以通过重写`if`语句来简化最后一个示例:

```py
if response:
    print('Success!')
else:
    print('An error has occurred.')
```

**技术细节:**这个[真值测试](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)之所以成为可能，是因为 [`__bool__()`是`Response`上的重载方法](https://realpython.com/operator-function-overloading/#making-your-objects-truthy-or-falsey-using-bool)。

这意味着`Response`的默认行为已被重新定义，以在确定对象的真值时考虑状态代码。

请记住，该方法是*而不是*验证状态代码是否等于`200`。其原因是在`200`到`400`范围内的其他状态码，如`204 NO CONTENT`和`304 NOT MODIFIED`，在提供一些可行响应的意义上也被认为是成功的。

例如，`204`告诉您响应成功，但是消息体中没有要返回的内容。

因此，确保只有当您想知道请求是否总体上成功时，才使用这种方便的简写方式，然后，如果必要的话，根据状态代码适当地处理响应。

假设您不想在`if`语句中检查响应的状态代码。相反，如果请求不成功，您希望引发一个异常。您可以使用`.raise_for_status()`来完成此操作:

```py
import requests
from requests.exceptions import HTTPError

for url in ['https://api.github.com', 'https://api.github.com/invalid']:
    try:
        response = requests.get(url)

        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  # Python 3.6
    except Exception as err:
        print(f'Other error occurred: {err}')  # Python 3.6
    else:
        print('Success!')
```

如果您调用`.raise_for_status()`，将会为某些状态代码产生一个`HTTPError`。如果状态代码指示请求成功，程序将继续运行，不会引发异常。

**延伸阅读:**如果你不熟悉 Python 3.6 的 [f-strings](https://realpython.com/python-f-strings/) ，我鼓励你利用它们，因为它们是简化格式化字符串的好方法。

现在，您已经非常了解如何处理从服务器返回的响应的状态代码。然而，当您发出一个`GET`请求时，您很少只关心响应的状态代码。通常，你想看到更多。接下来，您将看到如何查看服务器在响应正文中发回的实际数据。

[*Remove ads*](/account/join/)

### 内容

`GET`请求的响应通常在消息体中包含一些有价值的信息，称为有效负载。使用`Response`的属性和方法，您可以查看各种不同格式的有效载荷。

要在 [`bytes`](https://realpython.com/python-strings/) 中查看响应的内容，可以使用`.content`:

>>>

```py
>>> response = requests.get('https://api.github.com')
>>> response.content
b'{"current_user_url":"https://api.github.com/user","current_user_authorizations_html_url":"https://github.com/settings/connections/applications{/client_id}","authorizations_url":"https://api.github.com/authorizations","code_search_url":"https://api.github.com/search/code?q={query}{&page,per_page,sort,order}","commit_search_url":"https://api.github.com/search/commits?q={query}{&page,per_page,sort,order}","emails_url":"https://api.github.com/user/emails","emojis_url":"https://api.github.com/emojis","events_url":"https://api.github.com/events","feeds_url":"https://api.github.com/feeds","followers_url":"https://api.github.com/user/followers","following_url":"https://api.github.com/user/following{/target}","gists_url":"https://api.github.com/gists{/gist_id}","hub_url":"https://api.github.com/hub","issue_search_url":"https://api.github.com/search/issues?q={query}{&page,per_page,sort,order}","issues_url":"https://api.github.com/issues","keys_url":"https://api.github.com/user/keys","notifications_url":"https://api.github.com/notifications","organization_repositories_url":"https://api.github.com/orgs/{org}/repos{?type,page,per_page,sort}","organization_url":"https://api.github.com/orgs/{org}","public_gists_url":"https://api.github.com/gists/public","rate_limit_url":"https://api.github.com/rate_limit","repository_url":"https://api.github.com/repos/{owner}/{repo}","repository_search_url":"https://api.github.com/search/repositories?q={query}{&page,per_page,sort,order}","current_user_repositories_url":"https://api.github.com/user/repos{?type,page,per_page,sort}","starred_url":"https://api.github.com/user/starred{/owner}{/repo}","starred_gists_url":"https://api.github.com/gists/starred","team_url":"https://api.github.com/teams","user_url":"https://api.github.com/users/{user}","user_organizations_url":"https://api.github.com/user/orgs","user_repositories_url":"https://api.github.com/users/{user}/repos{?type,page,per_page,sort}","user_search_url":"https://api.github.com/search/users?q={query}{&page,per_page,sort,order}"}'
```

虽然`.content`允许您访问响应有效负载的原始字节，但是您通常会希望使用字符编码(如 [UTF-8](https://en.wikipedia.org/wiki/UTF-8) )将它们转换成一个[字符串](https://realpython.com/python-data-types/)。`response`将在您访问`.text`时为您完成:

>>>

```py
>>> response.text
'{"current_user_url":"https://api.github.com/user","current_user_authorizations_html_url":"https://github.com/settings/connections/applications{/client_id}","authorizations_url":"https://api.github.com/authorizations","code_search_url":"https://api.github.com/search/code?q={query}{&page,per_page,sort,order}","commit_search_url":"https://api.github.com/search/commits?q={query}{&page,per_page,sort,order}","emails_url":"https://api.github.com/user/emails","emojis_url":"https://api.github.com/emojis","events_url":"https://api.github.com/events","feeds_url":"https://api.github.com/feeds","followers_url":"https://api.github.com/user/followers","following_url":"https://api.github.com/user/following{/target}","gists_url":"https://api.github.com/gists{/gist_id}","hub_url":"https://api.github.com/hub","issue_search_url":"https://api.github.com/search/issues?q={query}{&page,per_page,sort,order}","issues_url":"https://api.github.com/issues","keys_url":"https://api.github.com/user/keys","notifications_url":"https://api.github.com/notifications","organization_repositories_url":"https://api.github.com/orgs/{org}/repos{?type,page,per_page,sort}","organization_url":"https://api.github.com/orgs/{org}","public_gists_url":"https://api.github.com/gists/public","rate_limit_url":"https://api.github.com/rate_limit","repository_url":"https://api.github.com/repos/{owner}/{repo}","repository_search_url":"https://api.github.com/search/repositories?q={query}{&page,per_page,sort,order}","current_user_repositories_url":"https://api.github.com/user/repos{?type,page,per_page,sort}","starred_url":"https://api.github.com/user/starred{/owner}{/repo}","starred_gists_url":"https://api.github.com/gists/starred","team_url":"https://api.github.com/teams","user_url":"https://api.github.com/users/{user}","user_organizations_url":"https://api.github.com/user/orgs","user_repositories_url":"https://api.github.com/users/{user}/repos{?type,page,per_page,sort}","user_search_url":"https://api.github.com/search/users?q={query}{&page,per_page,sort,order}"}'
```

因为从`bytes`到`str`的解码需要一个编码方案，所以如果你不指定编码方案，`requests`会根据响应的[头](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers)来猜测[编码](https://docs.python.org/3/howto/unicode.html#encodings)。您可以通过在访问`.text`之前设置`.encoding`来提供显式编码:

>>>

```py
>>> response.encoding = 'utf-8' # Optional: requests infers this internally
>>> response.text
'{"current_user_url":"https://api.github.com/user","current_user_authorizations_html_url":"https://github.com/settings/connections/applications{/client_id}","authorizations_url":"https://api.github.com/authorizations","code_search_url":"https://api.github.com/search/code?q={query}{&page,per_page,sort,order}","commit_search_url":"https://api.github.com/search/commits?q={query}{&page,per_page,sort,order}","emails_url":"https://api.github.com/user/emails","emojis_url":"https://api.github.com/emojis","events_url":"https://api.github.com/events","feeds_url":"https://api.github.com/feeds","followers_url":"https://api.github.com/user/followers","following_url":"https://api.github.com/user/following{/target}","gists_url":"https://api.github.com/gists{/gist_id}","hub_url":"https://api.github.com/hub","issue_search_url":"https://api.github.com/search/issues?q={query}{&page,per_page,sort,order}","issues_url":"https://api.github.com/issues","keys_url":"https://api.github.com/user/keys","notifications_url":"https://api.github.com/notifications","organization_repositories_url":"https://api.github.com/orgs/{org}/repos{?type,page,per_page,sort}","organization_url":"https://api.github.com/orgs/{org}","public_gists_url":"https://api.github.com/gists/public","rate_limit_url":"https://api.github.com/rate_limit","repository_url":"https://api.github.com/repos/{owner}/{repo}","repository_search_url":"https://api.github.com/search/repositories?q={query}{&page,per_page,sort,order}","current_user_repositories_url":"https://api.github.com/user/repos{?type,page,per_page,sort}","starred_url":"https://api.github.com/user/starred{/owner}{/repo}","starred_gists_url":"https://api.github.com/gists/starred","team_url":"https://api.github.com/teams","user_url":"https://api.github.com/users/{user}","user_organizations_url":"https://api.github.com/user/orgs","user_repositories_url":"https://api.github.com/users/{user}/repos{?type,page,per_page,sort}","user_search_url":"https://api.github.com/search/users?q={query}{&page,per_page,sort,order}"}'
```

如果您看一看响应，您会发现它实际上是序列化的 JSON 内容。要获得一个字典，您可以从`.text`中获取`str`，并使用 [`json.loads()`](https://realpython.com/python-json/#deserializing-json) 对其进行反序列化。然而，完成这项任务的一个更简单的方法是使用`.json()`:

>>>

```py
>>> response.json()
{'current_user_url': 'https://api.github.com/user', 'current_user_authorizations_html_url': 'https://github.com/settings/connections/applications{/client_id}', 'authorizations_url': 'https://api.github.com/authorizations', 'code_search_url': 'https://api.github.com/search/code?q={query}{&page,per_page,sort,order}', 'commit_search_url': 'https://api.github.com/search/commits?q={query}{&page,per_page,sort,order}', 'emails_url': 'https://api.github.com/user/emails', 'emojis_url': 'https://api.github.com/emojis', 'events_url': 'https://api.github.com/events', 'feeds_url': 'https://api.github.com/feeds', 'followers_url': 'https://api.github.com/user/followers', 'following_url': 'https://api.github.com/user/following{/target}', 'gists_url': 'https://api.github.com/gists{/gist_id}', 'hub_url': 'https://api.github.com/hub', 'issue_search_url': 'https://api.github.com/search/issues?q={query}{&page,per_page,sort,order}', 'issues_url': 'https://api.github.com/issues', 'keys_url': 'https://api.github.com/user/keys', 'notifications_url': 'https://api.github.com/notifications', 'organization_repositories_url': 'https://api.github.com/orgs/{org}/repos{?type,page,per_page,sort}', 'organization_url': 'https://api.github.com/orgs/{org}', 'public_gists_url': 'https://api.github.com/gists/public', 'rate_limit_url': 'https://api.github.com/rate_limit', 'repository_url': 'https://api.github.com/repos/{owner}/{repo}', 'repository_search_url': 'https://api.github.com/search/repositories?q={query}{&page,per_page,sort,order}', 'current_user_repositories_url': 'https://api.github.com/user/repos{?type,page,per_page,sort}', 'starred_url': 'https://api.github.com/user/starred{/owner}{/repo}', 'starred_gists_url': 'https://api.github.com/gists/starred', 'team_url': 'https://api.github.com/teams', 'user_url': 'https://api.github.com/users/{user}', 'user_organizations_url': 'https://api.github.com/user/orgs', 'user_repositories_url': 'https://api.github.com/users/{user}/repos{?type,page,per_page,sort}', 'user_search_url': 'https://api.github.com/search/users?q={query}{&page,per_page,sort,order}'}
```

`.json()`返回值的`type`是一个字典，所以可以通过键访问对象中的值。

您可以对状态代码和消息体做很多事情。但是，如果需要更多信息，比如关于响应本身的元数据，就需要查看响应的头。

### 标题

响应头可以为您提供有用的信息，比如响应负载的内容类型和缓存响应的时间限制。要查看这些标题，请访问`.headers`:

>>>

```py
>>> response.headers
{'Server': 'GitHub.com', 'Date': 'Mon, 10 Dec 2018 17:49:54 GMT', 'Content-Type': 'application/json; charset=utf-8', 'Transfer-Encoding': 'chunked', 'Status': '200 OK', 'X-RateLimit-Limit': '60', 'X-RateLimit-Remaining': '59', 'X-RateLimit-Reset': '1544467794', 'Cache-Control': 'public, max-age=60, s-maxage=60', 'Vary': 'Accept', 'ETag': 'W/"7dc470913f1fe9bb6c7355b50a0737bc"', 'X-GitHub-Media-Type': 'github.v3; format=json', 'Access-Control-Expose-Headers': 'ETag, Link, Location, Retry-After, X-GitHub-OTP, X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, X-OAuth-Scopes, X-Accepted-OAuth-Scopes, X-Poll-Interval, X-GitHub-Media-Type', 'Access-Control-Allow-Origin': '*', 'Strict-Transport-Security': 'max-age=31536000; includeSubdomains; preload', 'X-Frame-Options': 'deny', 'X-Content-Type-Options': 'nosniff', 'X-XSS-Protection': '1; mode=block', 'Referrer-Policy': 'origin-when-cross-origin, strict-origin-when-cross-origin', 'Content-Security-Policy': "default-src 'none'", 'Content-Encoding': 'gzip', 'X-GitHub-Request-Id': 'E439:4581:CF2351:1CA3E06:5C0EA741'}
```

返回一个类似字典的对象，允许你通过键访问头值。例如，要查看响应负载的内容类型，您可以访问`Content-Type`:

>>>

```py
>>> response.headers['Content-Type']
'application/json; charset=utf-8'
```

不过，这个类似字典的 headers 对象有一些特殊之处。HTTP 规范将头定义为不区分大小写，这意味着我们能够访问这些头，而不用担心它们的大小写:

>>>

```py
>>> response.headers['content-type']
'application/json; charset=utf-8'
```

无论您使用键`'content-type'`还是`'Content-Type'`，您都会得到相同的值。

现在，你已经了解了关于`Response`的基本知识。您已经看到了它最有用的属性和方法。让我们后退一步，看看当您定制您的`GET`请求时，您的响应是如何变化的。

[*Remove ads*](/account/join/)

## 查询字符串参数

定制`GET`请求的一种常见方式是通过 URL 中的[查询字符串](https://en.wikipedia.org/wiki/Query_string)参数传递值。要使用`get()`完成这项工作，您需要将数据传递给`params`。例如，您可以使用 GitHub 的[搜索](https://developer.github.com/v3/search/) API 来查找`requests`库:

```py
import requests

# Search GitHub's repositories for requests
response = requests.get(
    'https://api.github.com/search/repositories',
 params={'q': 'requests+language:python'}, )

# Inspect some attributes of the `requests` repository
json_response = response.json()
repository = json_response['items'][0]
print(f'Repository name: {repository["name"]}')  # Python 3.6+
print(f'Repository description: {repository["description"]}')  # Python 3.6+
```

通过将字典`{'q': 'requests+language:python'}`传递给`.get()`的`params`参数，您能够修改从搜索 API 返回的结果。

您可以像刚才那样以字典的形式将`params`传递给`get()`，或者以元组列表的形式传递:

>>>

```py
>>> requests.get(
...     'https://api.github.com/search/repositories',
...     params=[('q', 'requests+language:python')],
... )
<Response [200]>
```

您甚至可以将这些值作为`bytes`:

>>>

```py
>>> requests.get(
...     'https://api.github.com/search/repositories',
...     params=b'q=requests+language:python',
... )
<Response [200]>
```

查询字符串对于参数化`GET`请求很有用。您还可以通过添加或修改发送的邮件头来自定义您的请求。

## 请求标题

要定制头，可以使用`headers`参数将 HTTP 头的字典传递给`get()`。例如，通过在`Accept`标题中指定`text-match`媒体类型，您可以更改之前的搜索请求，以在结果中突出显示匹配的搜索词:

```py
import requests

response = requests.get(
    'https://api.github.com/search/repositories',
    params={'q': 'requests+language:python'},
 headers={'Accept': 'application/vnd.github.v3.text-match+json'}, )

# View the new `text-matches` array which provides information
# about your search term within the results
json_response = response.json()
repository = json_response['items'][0]
print(f'Text matches: {repository["text_matches"]}')
```

`Accept`头告诉服务器您的应用程序可以处理什么类型的内容。在本例中，由于您希望匹配的搜索词被高亮显示，所以您使用了头值`application/vnd.github.v3.text-match+json`，这是一个专有的 GitHub `Accept`头，其中的内容是一种特殊的 JSON 格式。

在学习更多定制请求的方法之前，让我们通过探索其他 HTTP 方法来拓宽视野。

## 其他 HTTP 方法

除了`GET`，其他流行的 HTTP 方法还有`POST`、`PUT`、`DELETE`、`HEAD`、`PATCH`和`OPTIONS`。`requests`为每个 HTTP 方法提供了一个与`get()`相似的方法:

>>>

```py
>>> requests.post('https://httpbin.org/post', data={'key':'value'})
>>> requests.put('https://httpbin.org/put', data={'key':'value'})
>>> requests.delete('https://httpbin.org/delete')
>>> requests.head('https://httpbin.org/get')
>>> requests.patch('https://httpbin.org/patch', data={'key':'value'})
>>> requests.options('https://httpbin.org/get')
```

每个函数调用都使用相应的 HTTP 方法向`httpbin`服务发出请求。对于每种方法，您可以像以前一样检查它们的响应:

>>>

```py
>>> response = requests.head('https://httpbin.org/get')
>>> response.headers['Content-Type']
'application/json'

>>> response = requests.delete('https://httpbin.org/delete')
>>> json_response = response.json()
>>> json_response['args']
{}
```

每个方法的头、响应体、状态代码等等都在`Response`中返回。接下来，您将仔细查看`POST`、`PUT`和`PATCH`方法，并了解它们与其他请求类型的不同之处。

[*Remove ads*](/account/join/)

## 消息正文

根据 HTTP 规范，`POST`、`PUT`和不太常见的`PATCH`请求通过消息体传递数据，而不是通过查询字符串中的参数。使用`requests`，您将把有效载荷传递给相应函数的`data`参数。

`data`采用字典、元组列表、字节或类似文件的对象。您将希望使您在请求正文中发送的数据适应您正在交互的服务的特定需求。

例如，如果您的请求的内容类型是`application/x-www-form-urlencoded`，您可以将表单数据作为字典发送:

>>>

```py
>>> requests.post('https://httpbin.org/post', data={'key':'value'})
<Response [200]>
```

您还可以将相同数据作为元组列表发送:

>>>

```py
>>> requests.post('https://httpbin.org/post', data=[('key', 'value')])
<Response [200]>
```

但是，如果需要发送 JSON 数据，可以使用`json`参数。当您通过`json`传递 JSON 数据时，`requests`将序列化您的数据并为您添加正确的`Content-Type`头。

【httpbin.org】[是`requests`](https://httpbin.org/)[的作者肯尼斯·雷兹](https://realpython.com/interview-kenneth-reitz/)创造的一个伟大资源。这是一个接受测试请求并以请求数据进行响应的服务。例如，您可以使用它来检查一个基本的`POST`请求:

>>>

```py
>>> response = requests.post('https://httpbin.org/post', json={'key':'value'})
>>> json_response = response.json()
>>> json_response['data']
'{"key": "value"}'
>>> json_response['headers']['Content-Type']
'application/json'
```

您可以从响应中看到，当您发送请求数据和报头时，服务器收到了它们。`requests`也以`PreparedRequest`的形式向您提供这些信息。

## 检查您的请求

当您发出请求时，`requests`库会在将请求发送到目的服务器之前准备好请求。请求准备包括诸如验证头和序列化 JSON 内容之类的事情。

您可以通过访问`.request`来查看`PreparedRequest`:

>>>

```py
>>> response = requests.post('https://httpbin.org/post', json={'key':'value'})
>>> response.request.headers['Content-Type']
'application/json'
>>> response.request.url
'https://httpbin.org/post'
>>> response.request.body
b'{"key": "value"}'
```

检查`PreparedRequest`可以让您访问关于请求的各种信息，比如有效负载、URL、头、认证等等。

到目前为止，您已经发出了许多不同类型的请求，但是它们都有一个共同点:它们都是对公共 API 的未经验证的请求。您可能遇到的许多服务都希望您以某种方式进行身份验证。

## 认证

身份验证有助于服务了解您是谁。通常，您可以通过服务定义的`Authorization`头或自定义头传递数据，从而向服务器提供您的凭证。到目前为止，您看到的所有请求函数都提供了一个名为`auth`的参数，它允许您传递凭证。

需要认证的 API 的一个例子是 GitHub 的[认证用户](https://developer.github.com/v3/users/#get-the-authenticated-user) API。此端点提供关于已验证用户的配置文件的信息。要向认证用户 API 发出请求，您可以将您的 GitHub 用户名和密码以元组的形式传递给`get()`:

>>>

```py
>>> from getpass import getpass
>>> requests.get('https://api.github.com/user', auth=('username', getpass()))
<Response [200]>
```

如果您在元组中传递给`auth`的凭证有效，则请求成功。如果您尝试在没有凭证的情况下发出这个请求，您会看到状态代码是`401 Unauthorized`:

>>>

```py
>>> requests.get('https://api.github.com/user')
<Response [401]>
```

当您将用户名和密码以元组的形式传递给`auth`参数时，`requests`正在使用 HTTP 的[基本访问认证方案](https://en.wikipedia.org/wiki/Basic_access_authentication)来应用凭证。

因此，您可以通过使用`HTTPBasicAuth`传递显式的基本认证凭证来发出相同的请求:

>>>

```py
>>> from requests.auth import HTTPBasicAuth
>>> from getpass import getpass
>>> requests.get(
...     'https://api.github.com/user',
...     auth=HTTPBasicAuth('username', getpass())
... )
<Response [200]>
```

尽管基本身份验证不需要明确，但您可能希望使用另一种方法进行身份验证。`requests`提供其他现成的认证方法，如`HTTPDigestAuth`和`HTTPProxyAuth`。

您甚至可以提供自己的身份验证机制。为此，您必须首先创建一个`AuthBase`的子类。然后，您实现`__call__()`:

```py
import requests
from requests.auth import AuthBase

class TokenAuth(AuthBase):
    """Implements a custom authentication scheme."""

    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        """Attach an API token to a custom auth header."""
        r.headers['X-TokenAuth'] = f'{self.token}'  # Python 3.6+
        return r

requests.get('https://httpbin.org/get', auth=TokenAuth('12345abcde-token'))
```

在这里，您的定制`TokenAuth`机制接收一个令牌，然后将该令牌包含在您的请求的`X-TokenAuth`头中。

糟糕的身份验证机制会导致安全漏洞，所以除非服务出于某种原因需要定制的身份验证机制，否则您总是希望使用像 Basic 或 OAuth 这样可靠的身份验证方案。

当您考虑安全性时，让我们考虑使用`requests`处理 SSL 证书。

[*Remove ads*](/account/join/)

## SSL 证书验证

任何时候你试图发送或接收的数据都是敏感的，安全是很重要的。通过 HTTP 与安全站点通信的方式是使用 SSL 建立加密连接，这意味着验证目标服务器的 SSL 证书至关重要。

好消息是`requests`默认为你做这件事。但是，在某些情况下，您可能希望改变这种行为。

如果您想要禁用 SSL 证书验证，您可以将`False`传递给请求函数的`verify`参数:

>>>

```py
>>> requests.get('https://api.github.com', verify=False)
InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
 InsecureRequestWarning)
<Response [200]>
```

甚至在您提出不安全的请求时发出警告，帮助您保护数据安全！

**注意:** [`requests`使用一个名为`certifi`](https://requests.readthedocs.io/en/latest/user/advanced/#ca-certificates) 的包来提供认证机构。这让`requests`知道它可以信任哪些权威。因此，您应该经常更新`certifi`以尽可能保证您的连接安全。

## 性能

当使用`requests`时，尤其是在生产应用程序环境中，考虑性能影响很重要。超时控制、会话和重试限制等功能可以帮助您保持应用程序平稳运行。

### 超时

当您向外部服务发出内联请求时，您的系统需要等待响应才能继续。如果应用程序等待响应的时间太长，对服务的请求可能会备份，用户体验可能会受到影响，或者后台作业可能会挂起。

默认情况下，`requests`将无限期地等待响应，因此您应该总是指定一个超时持续时间来防止这些事情发生。要设置请求的超时，请使用`timeout`参数。`timeout`可以是一个整数或浮点数，表示超时前等待响应的秒数:

>>>

```py
>>> requests.get('https://api.github.com', timeout=1)
<Response [200]>
>>> requests.get('https://api.github.com', timeout=3.05)
<Response [200]>
```

在第一个请求中，请求将在 1 秒钟后超时。在第二个请求中，请求将在 3.05 秒后超时。

[您还可以将一个元组](https://requests.readthedocs.io/en/latest/user/advanced/#timeouts)传递给`timeout`，第一个元素是连接超时(它允许客户端建立到服务器的连接的时间)，第二个元素是读取超时(一旦您的客户端建立连接，它将等待响应的时间):

>>>

```py
>>> requests.get('https://api.github.com', timeout=(2, 5))
<Response [200]>
```

如果请求在 2 秒内建立连接，并在连接建立后的 5 秒内收到数据，则响应将像以前一样返回。如果请求超时，那么该函数将引发一个`Timeout`异常:

```py
import requests
from requests.exceptions import Timeout

try:
    response = requests.get('https://api.github.com', timeout=1)
except Timeout:
    print('The request timed out')
else:
    print('The request did not time out')
```

您的程序可以捕捉到`Timeout`异常并做出相应的响应。

### 会话对象

到目前为止，你一直在处理高级的`requests`API，比如`get()`和`post()`。这些函数是当你发出请求时发生的事情的抽象。它们隐藏了实现细节，比如如何管理连接，这样您就不必担心它们了。

在这些抽象的下面是一个名为`Session`的类。如果您需要微调对如何发出请求的控制或者提高请求的性能，您可能需要直接使用一个`Session`实例。

会话用于跨请求保存参数。例如，如果希望在多个请求中使用相同的身份验证，可以使用会话:

```py
import requests
from getpass import getpass

# By using a context manager, you can ensure the resources used by
# the session will be released after use
with requests.Session() as session:
    session.auth = ('username', getpass())

    # Instead of requests.get(), you'll use session.get()
    response = session.get('https://api.github.com/user')

# You can inspect the response just like you did before
print(response.headers)
print(response.json())
```

每次使用`session`发出请求时，一旦使用认证凭证对其进行了初始化，凭证将被持久化。

会话的主要性能优化以持久连接的形式出现。当您的应用程序使用`Session`连接到服务器时，它会将该连接保存在连接池中。当您的应用程序想要再次连接到同一个服务器时，它将重用池中的连接，而不是建立一个新的连接。

[*Remove ads*](/account/join/)

### 最大重试次数

当请求失败时，您可能希望应用程序重试同一请求。但是，默认情况下，`requests`不会这样做。为了应用这个功能，您需要实现一个定制的[传输适配器](https://requests.readthedocs.io/en/latest/user/advanced/#transport-adapters)。

传输适配器允许您为正在交互的每个服务定义一组配置。例如，假设您希望所有对`https://api.github.com`的请求在最终引发`ConnectionError`之前重试三次。您将构建一个传输适配器，设置它的`max_retries`参数，并将其挂载到现有的`Session`:

```py
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError

github_adapter = HTTPAdapter(max_retries=3)

session = requests.Session()

# Use `github_adapter` for all requests to endpoints that start with this URL
session.mount('https://api.github.com', github_adapter)

try:
    session.get('https://api.github.com')
except ConnectionError as ce:
    print(ce)
```

当您挂载`HTTPAdapter`、`github_adapter`、`session`、`session`时，将会按照其配置对 https://api.github.com 发出每一个请求。

超时、传输适配器和会话是为了保持代码的高效和应用程序的弹性。

## 结论

在学习 Python 强大的`requests`库的过程中，你已经走了很长的路。

您现在能够:

*   使用各种不同的 HTTP 方法发出请求，例如`GET`、`POST`和`PUT`
*   通过修改标题、身份验证、查询字符串和消息正文来自定义您的请求
*   检查您发送给服务器的数据和服务器发回给您的数据
*   使用 SSL 证书验证
*   使用`max_retries`、`timeout`、会话和传输适配器有效地使用`requests`

因为您学习了如何使用`requests`，所以您已经准备好探索 web 服务的广阔世界，并使用它们提供的迷人数据构建令人惊叹的应用程序。

***参加测验:****通过我们的交互式“HTTP 请求与请求库”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-requests/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python**](/courses/python-requests/) 制作 HTTP 请求**********