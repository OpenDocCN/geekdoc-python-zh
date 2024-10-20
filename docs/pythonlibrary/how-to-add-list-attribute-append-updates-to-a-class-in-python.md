# How to Override a List Attribute's Append() Method in Python

> 原文：<https://www.blog.pythonlibrary.org/2021/11/10/how-to-add-list-attribute-append-updates-to-a-class-in-python/>

I had a use-case where I needed to create a class that had an attribute that was a Python list. Seems simple, right? The part that made it complicated is that I needed to do something special when anything was appended to that attribute. Watching for attribute changes don't work the same way for lists as it would for a string.

I tried a lot of different solutions, but most of them either didn't work or made the code really hard to understand.

Finally, someone on the Real Python Slack channel mentioned sub-classing **collections.UserList** and over-riding the **append()** method so that it executed a callback whenever that list object was appended to.

Here is a very simplified version of the code:

```py
import datetime

from collections import UserList
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

class ListWithCallback(UserList):
    """
    Create a class that emulates a Python list and supports a callback
    """
    def __init__(self, callback: Callable, *args: Tuple, **kwargs: Dict
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.callback = callback

    def append(self, item) -> None:
        super().append(item)
        self.callback()

@dataclass
class Media:
    channels: List = field(default_factory=list)

    def update(self) -> None:
        now = datetime.datetime.today()
        print(f"{now:%B %d - %H:%m:%S}")

    def __post_init__(self) -> None:
        self.channels = ListWithCallback(self.update)  # type: ignore

if __name__ == "__main__":
    import time
    impl = Media()
    impl.channels.append("Blah")
    time.sleep(2)
    impl.channels.append("Blah")

```

The main class of interest here is **Media**, which is a data class. It has a single attribute, **channels**, which is a Python **list**. To make this work, you use **__post_init__()** to set **channels** to an instance of your custom class, **ListWithCallback**. This class takes in a function or method to call when an item is appended to your special list.

In this case, you call **Media**'s **update()** method whenever an item is appended. To test that this functionality works, you import the **time** module at the bottom of the code and append two strings to the list with a **sleep()** between them.

## Wrapping Up

If you ever find yourself needing to subclass a Python built-in, check out Python's **collections** module. It has several classes that are usually recommended over directly subclassing from the built-ins themselves. In this case, you used **collections.UserList**.

Subclassing from the **collections** module is straightforward. You will learn a lot and your code may even be better because you did that.