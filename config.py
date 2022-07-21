import site
import os
import sys
from distutils.sysconfig import get_python_lib

site_directory = get_python_lib()

package_directory = os.getcwd()

filepath = os.path.join(site_directory,"GraphCodes.pth")

print(site_directory)
print(package_directory)
print(filepath)

f = open(filepath,"w")
f.write(package_directory)