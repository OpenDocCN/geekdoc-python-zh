# 如何在 Python 中使用 Vimeo API

> 原文：<https://www.pythonforbeginners.com/api/how-to-use-the-vimeo-api-in-python>

## 概观

```py
 In this post we will be looking on how to use the Vimeo API in Python. 
```

## 关于 Vimeo API

```py
 Vimeo offers an API which lets us integrate with their site and build applications
on top of their data.

Vimeo provides a "[Simple API](https://developer.vimeo.com/apis/simple "simple_api")" and an "[Advanced API](https://developer.vimeo.com/apis/advanced "advanced_api")".

To perform authenticated read/write requests on videos, users, groups, channels,
albums, or upload, you will have to use the Advanced API.

It uses OAuth for authentication, so you will first need to sign up.

When using the Simple API, you don’t need to register or authenticate your app.

One of the drawback with using the Simple API is that it’s limited to public data
and is read-only. 

The response limits in the Simple API include up to 20 items per page.

Vimeos API provides different Response formats, and they all return the exact
same data. 
```

## 入门指南

```py
 After looking at the website of Vimeo, I found out that I can start out with this
URL: http://vimeo.com/api/v2/video/video_id.output

the video_id is the ID of the video you want information for.

the output specifies which type that we want (json, xml) 
```

## 常见 Vimeo 请求

```py
 Making a User Request
Making a Video Request
Making an Activity Request
Making a Group Request
Making a Channel Request
Making an Album Request 
```

## 使用 Vimeo API 获取数据

```py
 This script will show how you can set up the script to make a Video request from
Vimeo. 
```

```py
import requests

import json

r = requests.get("http://vimeo.com/api/v2/video/48082757.json")

r.text

data = json.loads(r.text)

'do something with the data'

```

```py
 Vimeo has been kind enough to provide some API examples for us which can be found
on their GitHub repository. [https://github.com/vimeo/vimeo-api-examples](https://github.com/vimeo/vimeo-api-examples "github_api_vimeo") 
```

##### 更多阅读

```py
 [http://developer.vimeo.com/apis/](https://developer.vimeo.com/apis/ "vimeo_api") 
```