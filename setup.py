from setuptools import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='rodeo',
        version='0.0.0',
        description='Implementation of the rodeo multidimensional KDE bandwidth estimation method.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/austinschneider/the-rodeo',
        author='Austin Schneider',
        author_email='physics.schneider@gmail.com',
        license='L-GPL-2.1',
        packages=['rodeo'],
        package_data={'rodeo': [
            'resources/*',
            ]},
        include_package_data=True,
        zip_safe=False)
