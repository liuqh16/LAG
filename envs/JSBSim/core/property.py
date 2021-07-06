from collections import namedtuple
from gym.spaces import Box, Discrete

"""

A class to wrap and extend the Property object implemented in JSBSim

"""

Property = namedtuple("Property", "name_jsbsim description min max access spaces clipped update")
Property.__new__.__defaults__ = (None, None, float("-inf"), float("+inf"), "RW", Box, True, None)

CustomProperty = namedtuple("CustomProperty", "name_jsbsim description min max access spaces clipped read write")
CustomProperty.__new__.__defaults__ = (None, None, float("-inf"), float("+inf"), "RW", Box, False, None, None)
