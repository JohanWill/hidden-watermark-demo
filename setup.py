from setuptools import setup, find_packages

name = "eye"
version = "0.1"
description = "数字水印的嵌入和提取，图像比对"
install_requires = ['pymysql','DBUtils','apscheduler','numpy','pywavelets','scipy','cos-python-sdk-v5']


with open("README.md","r",encoding='utf-8') as f:
    long_desc = f.readlines()

setup(
    name=name,
    version=version,
    python_requires='>=3.6',
    description=description,
    install_requires=install_requires,
    packages=[
        "src","src.mark","src.font","src.compare"
    ],
    include_package_data=True,
    # packages=find_packages(exclude=['tests','tests.*']),
    entry_points={
        'console_scripts':[
            'eye = src.eye:main'
        ]
    },

    # long_description=long_desc,
    # long_description_content_type="text/markdown",

    # package_data={
    #     # If any package contains *.txt or *.rst files, include them:
    #     # '': ['*.txt', '*.rst'],
    #     # And include any *.msg files found in the 'hello' package, too:
    #     'monitor': ['*.*'],
    # },
)

