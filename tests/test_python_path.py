import sys
print(sys.executable)
import os
import sys
import site

print("\nPython environment information:")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Prefix: {sys.prefix}")
print(f"Base prefix: {sys.base_prefix}")
print(f"Site packages: {site.getsitepackages()}")
print(f"User site packages: {site.getusersitepackages()}")
print(f"Environment PATH: {os.getenv('PATH')}")
