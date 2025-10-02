# Testing 测试

# Testing with Paste and Nose

## Problem

You want to test your web.py application.

## Solution

```py
from paste.fixture import TestApp
from nose.tools import *
from code import app

class TestCode():
    def test_index(self):
        middleware = []
        app = TestApp(app.wsgifunc(*middleware))
        r = app.get('/')
        assert_equal(r.status, 200)
        r.mustcontain('Hello, world!') 
```

## Background

This example makes use of the Paste and Nose libraries. [Paste](http://pythonpaste.org/) lets you throw test requests at your application, and adds some helpful [custom methods to the response objects](http://pythonpaste.org/webtest/#the-response-object), such as mustcontain(), seen above. [Nose](http://somethingaboutorange.com/mrl/projects/nose/) makes writing and running your tests dead simple. When run from the base of your tree, it automatically finds and runs anything which is named like a test, adding necessary modules to your PYTHONPATH. This gives you the flexibility to run your tests from other directories, as well. Another benefit of Nose is that you no longer need to have every test class inherit from unittest.TestCase. Many more details are outlined on the project page.

## Explanation

This code resides in a file called test_code.py. The directory layout of the application looks like this:

```py
./
code.py
./test
    test_code.py 
```

Most of the code example above should be fairly self-explanatory. From our main module, code, we import app, which is defined in the usual way:

```py
app = web.application(urls, globals()) 
```

To set up the test, we pass its wsgifunc() to Paste's TestApp, as you have already seen in the example.

```py
app = TestApp(app.wsgifunc(*middleware)) 
```

assert_equal() is one of the methods provided by nose's utils, and works just like unittest's assertEqual().

## Setting Up the Test Environment

In order to avoid kicking off web.py's webserver when we run our tests, a change is required to the line which calls run(). It normally looks something like this:

```py
if __name__ == "__main__": app.run() 
```

We can define an environment variable, such as WEBPY_ENV=test, when we run our tests. In that case, the above line becomes the following:

```py
import os

def is_test():
    if 'WEBPY_ENV' in os.environ:
        return os.environ['WEBPY_ENV'] == 'test'

if (not is_test()) and __name__ == "__main__": app.run() 
```

Then, it's simply a matter of running nosetests like so:

```py
WEBPY_ENV=test nosetests 
```

The is_test() function comes in handy for other things, such as doing conditional database commits to avoid test database pollution.

# RESTful doctesting using app.request

```py
## !/usr/bin/env python

"""
RESTful web.py testing

usage: python webapp.py 8080 [--test]

>>> req = app.request('/mathematicians', method='POST')
>>> req.status
'400 Bad Request'

>>> name = {'first': 'Beno\xc3\xaet', 'last': 'Mandelbrot'}
>>> data = urllib.urlencode(name)
>>> req = app.request('/mathematicians', method='POST', data=data)
>>> req.status
'201 Created'
>>> created_path = req.headers['Location']
>>> created_path
'/mathematicians/b-mandelbrot'
>>> fn = '<h1 class=fn>{0} {1}</h1>'.format(name['first'], name['last'])
>>> assert fn in app.request(created_path).data

"""

import doctest
import urllib
import sys

import web

paths = (
  '/mathematicians(/)?', 'Mathematicians',
  '/mathematicians/([a-z])-([a-z]{2,})', 'Mathematician'
)
app = web.application(paths, globals())

dbname = {True: 'test', False: 'production'}[sys.argv[-1] == '--test']
db = {} # db = web.database(..., db='math_{0}'.format(dbname))

class Mathematicians:

  def GET(self, slash=False):
    """list all mathematicians and form to create new one"""
    if slash:
        raise web.seeother('/mathematicians')
    mathematicians = db.items() # db.select(...)
    return web.template.Template("""$def with (mathematicians)
      <!doctype html>
      <html>
      <head>
        <meta charset=utf-8>
        <title>Mathematicians</title>
      </head>
      <body>
        <h1>Mathematicians</h1>
        $if mathematicians:
          <ul class=blogroll>
            $for path, name in mathematicians:
              <li class=vcard><a class="fn url"
              href=/mathematicians/$path>$name.first $name.last</a></li>
          </ul>
        <form action=/mathematicians method=post>
          <label>First <input name=first type=text></label>
          <label>Last <input name=last type=text></label>
          <input type=submit value=Add>
        </form>
      </body>
      </html>""")(mathematicians)

  def POST(self, _):
    """create new mathematician"""
    name = web.input('first', 'last')
    key = '{0}-{1}'.format(name.first[0].lower(), name.last.lower())
    name.first, name.last = name.first.capitalize(), name.last.capitalize()
    db[key] = name # db.insert(...)
    path = '/mathematicians/{0}'.format(key)
    web.ctx.status = '201 Created'
    web.header('Location', path)
    return web.template.Template("""$def with (path, name)
      <!doctype html>
      <html>
      <head>
        <meta charset=utf-8>
        <title>Profile Created</title>
      </head>
      <body>
        <p>Profile created for <a href=$path>$name.first $name.last</a>.</p>
      </body>
      </html>""")(path, name)

class Mathematician:

  def GET(self, first_initial, last_name):
    """display mathematician"""
    key = '{0}-{1}'.format(first_initial, last_name)
    try:
        mathematician = db[key] # db.select(...)
    except KeyError:
        raise web.notfound()
    return web.template.Template("""$def with (name)
      <!doctype html>
      <html>
      <head>
        <meta charset=utf-8>
        <title>$name.first $name.last</title>
      </head>
      <body class=vcard>
        <p><a href=/mathematicians rel=up>Mathematicians</a> &#x25B8;</p>
        <h1 class=fn>$name.first $name.last</h1>
      </body>
      </html>""")(mathematician)

if __name__ == "__main__":
  if sys.argv[-1] == '--test':
    doctest.testmod()
  else:
    app.run() 
```