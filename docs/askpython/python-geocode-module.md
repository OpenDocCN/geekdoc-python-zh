# 使用 Python 地理编码模块反向查找邮政编码

> 原文：<https://www.askpython.com/python-modules/python-geocode-module>

在本文中，我们将使用地理编码模块对不同的邮政编码进行反向查找。整个代码非常简单，所以我们将快速浏览每一部分。

## 地理编码模块有什么帮助？

Python 地理编码模块旨在处理地理数据，并帮助我们将不同的数据配对和匹配在一起。通过 pgeocode 模块，我们可以获取并表示使用邮政编码信息的地区或区域相关信息。

因此，我们可以将该模块用于我们的目的。

**在本文中，我们将讨论 pgeocode 模块的一些功能，如下所述:**

*   通过邮政编码获得的国家/地区数据
*   邮政编码之间的差异
*   来自邮政编码的多区域数据

## 地理代码企业应用程序 API

如果您正在为商业应用程序寻找邮政编码地址查找 API，我会推荐您看一看 [ZipCodeBase](https://zipcodebase.com/) 。它们提供了 200 多个国家的邮政编码数据，如果您的应用程序能够满足全球用户的需求，这将是一件好事。他们有各种 API，[邮政编码半径搜索 API](https://app.zipcodebase.com/documentation#radius) 对于查找半径内的邮政编码非常有用。他们的文档是一流的，最好的部分是他们的免费计划，让我们快速开始并尝试这项服务。

### 1.从邮政编码获取地区数据

让我们看看如何从作为输入提供的不同邮政编码中获取地区数据。我们可以使用地理编码模块轻松获取国家代码、州名等。

**地理编码语法:**

```py
pgeocode.Nominatim(country)
query_postal_code(postal code)

```

当 query_post_code()方法处理邮政编码时，names()方法允许我们查询国家名称。

**举例:**

让我们举一个例子，看看这在现实生活中是如何工作的。我们将查询国家代码“US”和一个随机的邮政编码，并查看地理编码模块显示的数据:

```py
import pgeocode
data = pgeocode.Nominatim('US')
print(data.query_postal_code("95014"))

```

**输出:**

```py
postal_code             95014
country_code               US
place_name          Cupertino
state_name         California
state_code                 CA
county_name       Santa Clara
county_code                85
community_name            NaN
community_code            NaN
latitude               37.318
longitude            -122.045
accuracy                    4
Name: 0, dtype: object

```

如您所见，我们收到了大量关于我们查询的国家和邮政编码输入的数据。

### 2.两个邮政编码之间的地理距离

地理编码模块提供的另一个很酷的特性是能够找到两个邮政编码之间的地理距离。借助 Geodistance()方法，我们也可以做到这一点。让我们首先用有问题的国家初始化这个方法，然后输入我们要查找的两个邮政编码之间的地理距离。

```py
import pgeocode
data = pgeocode.GeoDistance('fr')
print(data.query_postal_code("75013", "69006"))

```

**输出:**

```py
391.1242610965041

```

## 结论

许多程序确实需要我们处理地理位置和邮政编码，而地理编码库使得处理如此复杂的数据变得非常容易，否则就需要一个巨大的数据集。

我们希望您喜欢了解这个主题。我们已经介绍了另一个处理地理位置的模块，即[地理位置模块](https://www.askpython.com/python/python-geopy-to-find-geocode-of-an-address)。

**参考资料:** [官方文件](https://pypi.org/project/pgeocode/)