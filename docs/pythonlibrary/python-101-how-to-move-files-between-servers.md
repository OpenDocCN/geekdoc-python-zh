# Python 101:如何在服务器之间移动文件

> 原文：<https://www.blog.pythonlibrary.org/2012/10/26/python-101-how-to-move-files-between-servers/>

如果您做过大量的系统管理工作，那么您会知道有时您必须编写脚本来在服务器之间移动文件。我并不是一个真正的系统管理员，但是无论如何我必须在我的一些程序中做这样的事情。Python 有几个提供这种能力的第三方包。我们将看看如何用依赖于 [PyCrypto](https://www.dlitz.net/software/pycrypto/) 的 [paramiko](http://pypi.python.org/pypi/paramiko/1.8.0) 来做这件事(或者从 [PyPI](http://pypi.python.org/pypi/pycrypto/2.6) 下载 PyCrypto)。

### 编写代码

假设你有前面提到的所有第三方包，我们就可以开始编码了。为了使事情超级简单，我们将只使用 paramiko 作为我们的第一个例子。以下代码大致基于我在工作中使用的一些代码。我们来看看吧！

```py
import paramiko

########################################################################
class SSHConnection(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, host, username, password, port=22):
        """Initialize and setup connection"""
        self.sftp = None
        self.sftp_open = False

        # open SSH Transport stream
        self.transport = paramiko.Transport((host, port))

        self.transport.connect(username=username, password=password)

    #----------------------------------------------------------------------
    def _openSFTPConnection(self):
        """
        Opens an SFTP connection if not already open
        """
        if not self.sftp_open:
            self.sftp = paramiko.SFTPClient.from_transport(self.transport)
            self.sftp_open = True

    #----------------------------------------------------------------------
    def get(self, remote_path, local_path=None):
        """
        Copies a file from the remote host to the local host.
        """
        self._openSFTPConnection()        
        self.sftp.get(remote_path, local_path)        

    #----------------------------------------------------------------------
    def put(self, local_path, remote_path=None):
        """
        Copies a file from the local host to the remote host
        """
        self._openSFTPConnection()
        self.sftp.put(local_path, remote_path)

    #----------------------------------------------------------------------
    def close(self):
        """
        Close SFTP connection and ssh connection
        """
        if self.sftp_open:
            self.sftp.close()
            self.sftp_open = False
        self.transport.close()

if __name__ == "__main__":
    host = "myserver"
    username = "mike"
    pw = "dingbat!"

    origin = '/home/mld/projects/ssh/random_file.txt'
    dst = '/home/mdriscoll/random_file.txt'

    ssh = SSHConnection(host, username, pw)
    ssh.put(origin, dst)
    ssh.close()

```

让我们花点时间来分解一下。在我们类的 *__init__* 中，我们至少需要传入参数。在这个例子中，我们传递给它我们的主机，用户名和密码。然后我们打开一个 SSH 传输流对象。接下来，我们调用我们的 **put** 方法将文件从我们的机器发送到服务器。如果你想下载一个文件，参见**获取**方法。最后，我们调用我们的 **close** 方法来关闭我们的连接。您会注意到，在 put 和 get 方法中，我们让它们调用一个半私有的方法来检查我们的 SFTPClient 是否已初始化，如果没有，它将继续创建它。

### 包扎

Paramiko 让这一切变得简单。我强烈推荐阅读 Jesse 关于这个主题的旧文章(链接如下),因为他有更多的细节。我很好奇其他人用什么包来实现 ssh 和 scp，所以请在评论中给我留下一些建议。我听到了一些关于布料的好消息。

### 进一步阅读

*   Jesse Noller 的使用 Paramiko 的 SSH 编程