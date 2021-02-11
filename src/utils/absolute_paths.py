import os

def absolute_paths(folder):
    for dir,_,filenames in os.walk(folder):
        for f in filenames:
            yield os.path.abspath(os.path.join(dir, f))