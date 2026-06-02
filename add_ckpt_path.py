import sys
import os
import os.path as path


def add_path_to_da3(ckpt):
    HERE_PATH = os.path.dirname(os.path.abspath(ckpt))
    # workaround for sibling import
    sys.path.insert(0, HERE_PATH)
    HERE_PATH2 = os.path.join(HERE_PATH, 'src')
    sys.path.insert(0, HERE_PATH2)

add_path_to_da3(__file__)