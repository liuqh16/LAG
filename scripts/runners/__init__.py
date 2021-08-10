# Deal with module import error
import os
import sys
filepath = os.path.realpath(__file__)
rootpath = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
sys.path.append(rootpath)