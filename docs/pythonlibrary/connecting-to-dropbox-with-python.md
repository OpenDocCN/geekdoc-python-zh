# 使用 Python 连接到 Dropbox

> 原文：<https://www.blog.pythonlibrary.org/2013/07/17/connecting-to-dropbox-with-python/>

昨天偶然发现了 Dropbox 的 [Python API](https://www.dropbox.com/developers/core/start/python) 。我最终使用他们的教程设计了一个简单的类来访问我的 Dropbox。你需要下载他们的 dropbox 模块，或者使用“pip install dropbox”来安装。您还需要注册一个密钥和密码。一旦有了这些，您就需要命名您的应用程序并选择您的访问级别。那你应该可以走了！

现在我们准备开始写一些代码。这是我想到的:

```py

import dropbox
import os
import sys
import webbrowser

from configobj import ConfigObj

########################################################################
class DropObj(object):
    """
    Dropbox object that can access your dropbox folder,
    as well as download and upload files to dropbox
    """

    #----------------------------------------------------------------------
    def __init__(self, filename=None, path='/'):
        """Constructor"""
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.filename = filename
        self.path = path
        self.client = None

        config_path = os.path.join(self.base_path, "config.ini")
        if os.path.exists(config_path):
            try:
                cfg = ConfigObj(config_path)
            except IOError:
                print "ERROR opening config file!"
                sys.exit(1)
            self.cfg_dict = cfg.dict()
        else:
            print "ERROR: config.ini not found! Exiting!"
            sys.exit(1)

        self.connect()

    #----------------------------------------------------------------------
    def connect(self):
        """
        Connect and authenticate with dropbox
        """
        app_key = self.cfg_dict["key"]
        app_secret = self.cfg_dict["secret"]

        access_type = "dropbox"
        session = dropbox.session.DropboxSession(app_key,
                                                 app_secret,
                                                 access_type)

        request_token = session.obtain_request_token()

        url = session.build_authorize_url(request_token)
        msg = "Opening %s. Please make sure this application is allowed before continuing."
        print msg % url
        webbrowser.open(url)
        raw_input("Press enter to continue")
        access_token = session.obtain_access_token(request_token)

        self.client = dropbox.client.DropboxClient(session)

    #----------------------------------------------------------------------
    def download_file(self, filename=None, outDir=None):
        """
        Download either the file passed to the class or the file passed
        to the method
        """

        if filename:
            fname = filename
            f, metadata = self.client.get_file_and_metadata("/" + fname)
        else:
            fname = self.filename
            f, metadata = self.client.get_file_and_metadata("/" + fname)

        if outDir:
            dst = os.path.join(outDir, fname)
        else:
            dst = fname

        with open(fname, "w") as fh:
            fh.write(f.read())

        return dst, metadata

    #----------------------------------------------------------------------
    def get_account_info(self):
        """
        Returns the account information, such as user's display name,
        quota, email address, etc
        """
        return self.client.account_info()

    #----------------------------------------------------------------------
    def list_folder(self, folder=None):
        """
        Return a dictionary of information about a folder
        """
        if folder:
            folder_metadata = self.client.metadata(folder)
        else:
            folder_metadata = self.client.metadata("/")
        return folder_metadata

    #----------------------------------------------------------------------
    def upload_file(self):
        """
        Upload a file to dropbox, returns file info dict
        """
        try:
            with open(self.filename) as fh:
                path = os.path.join(self.path, self.filename)
                res = self.client.put_file(path, fh)
                print "uploaded: ", res
        except Exception, e:
            print "ERROR: ", e

        return res

if __name__ == "__main__":
    drop = DropObj("somefile.txt")

```

我把我的密钥和秘密放在一个配置文件中，如下所示:

```py

key = someKey
secret = secret

```

然后我使用 configobj 将这些信息提取到一个 Python 字典中。我试图找到一种方法来缓存请求令牌，但我总是得到一个关于我的令牌过期的错误，所以这个脚本总是会弹出一个浏览器窗口，要求您“允许”您的应用程序访问您的 Dropbox。一旦你连接上了，客户端就被实例化了，你可以提取关于你的 Dropbox 的各种信息。例如，你可以在你的 Dropbox 中获得你的账户信息或者任何文件夹的元数据。我还创建了和 **upload_file** 方法来允许轻松上传文件。我可能应该这样做，这样你也可以传递一个文件给这个方法，但是那要等到版本 2。 **download_file** 方法遵循 Dropbox 支持的官方方法，然而我下载的每个文件最终都被破坏了。

无论如何，我认为这是一个有趣的 API，希望你会发现这个小脚本也很有帮助。