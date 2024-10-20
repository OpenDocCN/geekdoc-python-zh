# 使用 Python 获取 GPS EXIF 数据

> 原文：<https://www.blog.pythonlibrary.org/2021/01/13/getting-gps-exif-data-with-python/>

您知道可以使用 Python 编程语言从 JPG 图像文件中获取 EXIF 数据吗？您可以使用 Pillow 来实现这一点，Pillow 是 Python 图像库的友好分支。如果你想的话，你可以在这个网站上阅读一篇文章。

以下是从 JPG 文件中获取常规 EXIF 数据的示例代码:

```py
# exif_getter.py

from PIL import Image
from PIL.ExifTags import TAGS

def get_exif(image_file_path):
    exif_table = {}
    image = Image.open(image_file_path)
    info = image.getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        exif_table[decoded] = value
    return exif_table

if __name__ == "__main__":
    exif = get_exif("bridge.JPG")
    print(exif)

```

该代码使用以下图像运行:

![Mile long bridge](img/9102a1b6313f58245af252adae4f59ae.png)

在本文中，您将关注如何从图像中提取 GPS 标签。这些是特殊的 EXIF 标签，只有在拍摄照片的相机打开了其位置信息时才会出现。你也可以事后在电脑上添加 GPS 标签。

例如，我在杰斯特公园的这张照片上添加了 GPS 标签，杰斯特公园位于伊利诺伊州的格兰杰:

![](img/85bbc5431a7518d06b220de35505ac27.png)

要访问这些标记，您需要使用前面的代码示例并做一些小的调整:

```py
# gps_exif_getter.py

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif(image_file_path):
    exif_table = {}
    image = Image.open(image_file_path)
    info = image.getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        exif_table[decoded] = value

    gps_info = {}
    for key in exif_table['GPSInfo'].keys():
        decode = GPSTAGS.get(key,key)
        gps_info[decode] = exif_table['GPSInfo'][key]

    return gps_info

if __name__ == "__main__":
    exif = get_exif("jester.jpg")
    print(exif)

```

要访问 GPS 标签，您需要从 PIL.ExifTags 导入 GPS tags。如果存在，那么你就可以提取 GPS 标签。

运行此代码时，您应该会看到以下输出:

```py
{'GPSLatitudeRef': 'N',
 'GPSLatitude': (41.0, 47.0, 2.17),
 'GPSLongitudeRef': 'W',
 'GPSLongitude': (93.0, 46.0, 42.09)}
```

您可以获取这些信息，并使用 Python 加载 Google 地图，或者使用流行的 GIS 相关 Python 库。

#### 相关阅读

*   [使用 Python 获取照片元数据(EXIF)](https://www.blog.pythonlibrary.org/2010/03/28/getting-photo-metadata-exif-using-python/)

*   [给图像浏览器添加 EXIF 浏览器](https://www.blog.pythonlibrary.org/2010/04/10/adding-an-exif-viewer-to-the-image-viewer/)