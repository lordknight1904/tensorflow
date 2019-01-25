# # import io
# # import os
# # import sys
# # from shutil import rmtree
# from setuptools import find_packages, setup \
#     # ,Command
#
# # Package meta-data.
# NAME = 'Tensorflow'
# DESCRIPTION = 'Tensorflow project'
# URL = 'https://github.com/'
# EMAIL = 'ngoak@islab.snu.ac.kr'
# AUTHOR = 'Khoa Anh Ngo'
# REQUIRES_PYTHON = '3.6.0'
# VERSION = None
#
# # What packages are required for this module to be executed?
# REQUIRED = [
#     'tensorflow-gpu',
# ]
#
# # What packages are optional?
# # EXTRAS = {
# #     # 'fancy feature': ['django'],
# # }
#
#
# setup(
#     name=NAME,
#     description=DESCRIPTION,
#     # long_description=long_description,
#     # long_description_content_type='text/markdown',
#     author=AUTHOR,
#     author_email=EMAIL,
#     python_requires=REQUIRES_PYTHON,
#     url=URL,
#     packages=find_packages(exclude=('tests',)),
#     # If your package is a single module, use this instead of 'packages':
#     # py_modules=['mypackage'],
#
#     # entry_points={
#     #     'console_scripts': ['mycli=mymodule:cli'],
#     # },
#     install_requires=REQUIRED,
#     # extras_require=EXTRAS,
#     include_package_data=True,
# )