## Freezing 的多种方式

打包你的代码 是指把你的库或工具分发给其他开发者。

在 Linux 一个冻结的待选物是 创建一个 Linux 分发 包 <packaging-for-linux-distributions-ref> (e.g.对于 Debian 或 Ubuntu 是 .deb 文件， 而对于 Red Hat 与 SuSE 是.rpm 文件)

待处理

完善 “冻结你的代码” 部分（stub）。

## 比较

各解决方案的平台/特性支持性

| Solution | Windows | Linux | OS X | Python 3 | License | One-file mode | Zipfile import | Eggs | pkg_resources support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bbFreeze | yes | yes | yes | no | MIT | no | yes | yes | yes |
| py2exe | yes | no | no | yes | MIT | yes | yes | no | no |
| pyInstaller | yes | yes | yes | yes | GPL | yes | no | yes | no |
| cx_Freeze | yes | yes | yes | yes | PSF | no | yes | yes | no |
| py2app | no | no | yes | yes | MIT | no | yes | yes | yes |

注解

从 Linux 到 Windows 的冻结只有 PyInstaller 支持， [其余的](http://stackoverflow.com/questions/2950971/cross-compiling-a-python-script-on-linux-into-a-windows-executable#comment11890276_2951046) [http://stackoverflow.com/questions/2950971/cross-compiling-a-python-script-on-linux-into-a-windows-executable#comment11890276_2951046]。

注解

所有解决方案需要目前机器上安装了 MS Visual C++ dll。除了 py2app 以外。 只有 Pyinstaller 创建了可以自足运行的 exe 文件，其绑定了 dll，可以传递 `--onefile` to `Configure.py`。

## Windows

### bbFreeze

前置要求是安装 Python, Setuptools 以及 pywin32 的依赖项。

待处理

补充更多简单的生成 .exe 的步骤。

### py2exe

前置要求是安装了 Python on Windows。

1.  下载并且安装 [`sourceforge.net/projects/py2exe/files/py2exe/`](http://sourceforge.net/projects/py2exe/files/py2exe/)
2.  编写 `setup.py` ([配置选项清单](http://www.py2exe.org/index.cgi/ListOfOptions) [http://www.py2exe.org/index.cgi/ListOfOptions]):

```py
from distutils.core import setup
import py2exe

setup(
    windows=[{'script': 'foobar.py'}],
) 
```

3.  (可选) [包含图标](http://www.py2exe.org/index.cgi/CustomIcons) [http://www.py2exe.org/index.cgi/CustomIcons]
4.  (可选) [单文件模式](http://stackoverflow.com/questions/112698/py2exe-generate-single-executable-file#113014) [http://stackoverflow.com/questions/112698/py2exe-generate-single-executable-file#113014]
5.  生成 :file: .exe 到 `dist` 目录:

```py
$ python setup.py py2exe 
```

6.提供 Microsoft Visual C 运行时 DLL。两个选项: [在目标机器全局安装 dll](https://www.microsoft.com/en-us/download/details.aspx?id=29) [https://www.microsoft.com/en-us/download/details.aspx?id=29] 或者 [与.exe 一起分发 dll](http://www.py2exe.org/index.cgi/Tutorial#Step52) [http://www.py2exe.org/index.cgi/Tutorial#Step52]。

### PyInstaller

前置是安装 Python, Setuptools 以及 pywin32 依赖项.

*   [更多的简单教程](http://bojan-komazec.blogspot.com/2011/08/how-to-create-windows-executable-from.html) [http://bojan-komazec.blogspot.com/2011/08/how-to-create-windows-executable-from.html]
*   [手册](http://www.pyinstaller.org/export/d3398dd79b68901ae1edd761f3fe0f4ff19cfb1a/project/doc/Manual.html?format=raw) [http://www.pyinstaller.org/export/d3398dd79b68901ae1edd761f3fe0f4ff19cfb1a/project/doc/Manual.html?format=raw]

## OS X

### py2app

### PyInstaller

## Linux

### bbFreeze

### PyInstaller © 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.