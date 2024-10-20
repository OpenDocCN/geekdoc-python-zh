# 在 Python 2.x 中为您的图像添加水印

> 原文：<https://www.pythoncentral.io/watermark-images-python-2x/>

The module we use in this recipe to resize an image with Python is `PIL`. At the time of writing, it is only available for Python 2.x. Also, if you wish to do other things with images, checkout our article on [how to resize an image with Python](https://www.pythoncentral.io/resize-image-python-batch/ "Resize an Image with Python").

当您拍摄照片并将其发布到互联网上时，添加水印来防止和阻止未经授权的复制或图像盗窃通常是很方便的。

下面是一个简单的 Python 脚本，它使用`PIL`模块给你的图像添加水印。它使用系统字体给图像添加可见的文本水印。

```py

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

import os, sys
FONT = 'Arial.ttf' 

```

我们从导入`PIL`模块开始，加载 Truetype 字体‘arial . TTF’。默认情况下，它会在同一个文件夹中搜索字体，然后查看你的字体目录(例如 C:/Windows/Fonts)

```py

def add_watermark(in_file, text, out_file='watermark.jpg', angle=23, opacity=0.25):

```

我们定义了一个函数`add_watermark`并声明了一些默认参数。

*   `in_file`–输入文件名。
*   `text`–水印文本。
*   `out_file`–输出文件名(默认:watermark.jpg)。
*   `angle`–水印的角度(默认:23 度)。
*   `opacity`–不透明度(默认值:0.25)

```py

img = Image.open(in_file).convert('RGB')

watermark = Image.new('RGBA', img.size, (0,0,0,0))

```

首先，我们打开输入文件并创建一个相似尺寸的水印图像。两个文件都需要处于`RGB`模式，因为我们正在使用*阿尔法*通道。

```py

size = 2

n_font = ImageFont.truetype(FONT, size)

n_width, n_height = n_font.getsize(text)

```

从字体大小 2 开始，我们创建文本并获得文本的宽度和高度。

```py

while (n_width+n_height < watermark.size[0]):

    size += 2

    n_font = ImageFont.truetype(FONT, size)

    n_width, n_height = n_font.getsize(text)

```

通过增加字体大小，我们搜索不超过图像尺寸(宽度)的文本长度。

```py

draw = ImageDraw.Draw(watermark, 'RGBA')

draw.text(((watermark.size[0] - n_width) / 2,

          (watermark.size[1] - n_height) / 2),

          text, font=n_font)

```

使用正确的字体大小，我们使用 header 部分中声明的系统字体将文本绘制到水印图像的中心。

```py

watermark = watermark.rotate(angle, Image.BICUBIC)

```

然后我们使用`Image.BICUBIC`(算法)近似旋转图像(默认为 23 度)。

```py

alpha = watermark.split()[3]

alpha = ImageEnhance.Brightness(alpha).enhance(opacity)

watermark.putalpha(alpha)

```

在 alpha 通道上，我们通过默认值 0.25 来降低水印的不透明度(例如:降低亮度和对比度)。(注意:值 1 返回原始图像)。

```py

Image.composite(watermark, img, watermark).save(out_file, 'JPEG')

```

最后，我们将水印重新合并到原始图像中，并保存为一个新的 JPEG 文件。

整个代码如下:

```py

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

import os, sys
字体= 'Arial.ttf '
def add_watermark(in_file，text，out_file='watermark.jpg '，angle=23，opacity = 0.25):
img = image . open(in _ file)。convert(' RGB ')
watermark = image . new(' RGBA '，img.size，(0，0，0，0))
size = 2
n _ FONT = image FONT . truetype(FONT，size) 
 n_width，n _ height = n _ FONT . getsize(text)
while n _ width+n _ height<watermark . size[0]:
size+= 2
n _ FONT = image FONT . truetype(FONT，size)
n _ nDraw(水印，' RGBA ')
draw . text(((watermark . size[0]-n _ width)/2，
(watermark . size[1]-n _ height)/2)，
 text，font = n _ font)
watermark = watermark . rotate(角度，图像。双三次)
alpha = watermark . split()【3】
alpha = image enhance。亮度(alpha)。enhance(不透明度)
watermark . put alpha(alpha)
image . composite(水印，img，水印)。保存(输出文件，“JPEG”)
if _ _ name _ _ = ' _ _ main _ _ ':
if len(sys . argv)<3:
sys . exit('用法:% s<input-image><text><output-image>' \
'<angle><opacity>' % OS . path . basename(sys . argv[0])
add _ watermark(* sys . argv[1:])【T4
```