"""

This is the entrance file to the import chain of MLgebra.

Every basic import is done here. Also, the main purpose of
this file is to declare exceptions.

The only import done here is 'vectorgebra'. The reason is
because, we will also import vectorgebra's exceptions.
Since this is the file for exceptions, it needs to be done
here.

Another important notice is that, every other import that
vectorgebra does, is also done here. This includes 'typing',
'random', etc.

"""

from vectorgebra import *

class ConfigError(Exception):

    def __init__(self, hint: str = ""):
        super().__init__(f"Model configuration invalid{': ' + hint if hint else ''}")

class FileStructureError(Exception):

    def __init__(self, hint: str = ""):
        super().__init__(f"File structured incorrectly{': ' + hint if hint else ''}")
