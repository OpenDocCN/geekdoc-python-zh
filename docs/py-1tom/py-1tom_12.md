# 图片处理

# PIL

# PIL

## 实例联系

### 图片转 ASC II 码

原理：

[`www.jave.de/image2ascii/algorithms.html`](http://www.jave.de/image2ascii/algorithms.html)

示例代码：[`github.com/kxxoling/image2ascii`](https://github.com/kxxoling/image2ascii)

### 图片相似度计算

### 图片相似度计算&索引

# QRCode

# QRCode

QR 码（全称为快速响应矩阵码；英语：Quick Response Code）是二维条码的一种， 于 1994 年由日本 DENSO WAVE 公司发明。QR 码使用四种标准化编码模式（数字，字母数字， 二进制和 Kanji）来存储数据。

## Python QRCode 库

Python 可以安装 [qrcode](https://github.com/lincolnloop/python-qrcode) 库以获取 QR Code 生成的支持。

### 安装

qrcode 库依赖于 Python Image Library（PIL），不过 PIL 已经停止更新，而且其设计并不符合 Python 规范。 因此推荐使用 PIL 的继任者 Pillow 代替。 安装 PIL 或者 Pillow 需要先安装 C 语言 PIL 库，各操作系统各有不同。 Python PIL 或者 Pillow 的安装可以通过 `pip` 或者 `easy_install`：

```
pip install pillow 
```

或者

```
pip install PIL 
```

安装 Python QRCode：

```
pip install qrcode 
```

### 使用 qrcode

QRCode 库提供两种调用提供方式——Python 库和系统命令(qr)。

#### qr 命令

`qr` 命令的使用如下：

```
qr some_word[ > some_iamge.png] 
```

qrcode 会根据文字的长度自动选择合适的 QRCode 版本

#### Python API

Python 下使用 qrcode 库：

```
import qrcode
qr = qrcode.QRCode(
    version=1,                                              # QR Code version，1-4
    error_correction=qrcode.constants.ERROR_CORRECT_L,      # 错误纠正等级 L、M、Q、H 四等，默认是 M
    box_size=10,                                            # QR Code 图片的大小，单位是像素
    border=4,                                               # QR Code 的边框，单位是像素，默认 4
)
qr.add_data('Some data')            # 想要添加到 QR Code 中的内容
qr.make(fit=True)

img = qr.make_image() 
```

默认输出格式是 JPG，生成 SVG 需要设定 `image_factory` 参数：

```
import qrcode
import qrcode.image.svg

if method == 'basic':
    # Simple factory, just a set of rects.
    factory = qrcode.image.svg.SvgImage
elif method == 'fragment':
    # Fragment factory (also just a set of rects)
    factory = qrcode.image.svg.SvgFragmentImage
else:
    # Combined path factory, fixes white space that may occur when zooming
    factory = qrcode.image.svg.SvgPathImage

img = qrcode.make('Some data here', image_factory=factory) 
```

如果需要 PNG 支持，还需要安装第三支持：

```
pip install git+git://github.com/ojii/pymaging.git#egg=pymaging
pip install git+git://github.com/ojii/pymaging-png.git#egg=pymaging-png 
```

依旧是设定 `image_factory` 设置输出格式为 PNG：

```
import qrcode
from qrcode.image.pure import PymagingImage
img = qrcode.make('Some data here', image_factory=PymagingImage) 
```

# 几种图片转字符算法介绍

# 图片转字符原理

图片转字符通常分以下几种：

1.  黑白算法

1.  灰度算法

1.  边际追踪／界定算法

## 黑白算法

黑白算法最简单

低保真度解法：

```
##### ####   ## #####
  #   #     #     #
  #   ##    #     #
  #   #      #    #
  #   #### ##     # 
```

高保真度解法：

```
88888 8888  d8b 88888
  8   8    ]b     8
  8   88    Yb    8
  8   8    o d8   8
  8   8888  YP    8 
```

低保真度解法仅仅简单地通过字符 `#` 和空白符表示图形区域地颜色， 高保真算法则使用 12 个字符对应一个 `2*2` 的区域，对照表如下：

| 模式： |
| --- |

```
..
..
```

|

```
.X
..
```

|

```
X.
..
```

|

```
..
X.
```

|

```
..
.X
```

|

```
XX
..
```

|

```
..
XX
```

|

```
X.
X.
```

|

```
.X
.X
```

|

```
.X
XX
```

|

```
XX
.X
```

|

```
XX
X.
```

|

```
XX
XX
```

|

| 字符： |
| --- |

```
`
```

|

```
'
```

|

```
.
```

|

```
,
```

|

```
"
```

|

```
_
```

|

```
(
```

|

```
)
```

|

```
J
```

|

```
L
```

|

```
7
```

|

```
P
```

|

```
8
```

|

## 灰度算法

灰度算法的基本目标是���求转换后的每个字符亮度都尽可能接近该区域原始图像的亮度， 因此在文字渐渐缩小或者与人眼的距离越来越远时，看起来越发接近图片。 边缘、边框以及其它高对比度的地方应该使用更加契合其结构的字符。

灰度算法需要使用实现亮度－字符的转换，大多数具体实现都采用 1 个像素对 1 个字符的转换， “state of the art”算法使用 4 个像素对应一个字符，并且能够提供更清晰的结果。

## 边际追踪／界定算法

边际界定算法看起更加复杂，但其实不然！

全文主要翻译自 [java.de](http://www.jave.de/image2ascii/algorithms.html)

# 验证码破解

# 验证码的原理与破解

[常见验证码的弱点与验证码识别](http://drops.wooyun.org/tips/141)
