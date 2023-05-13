from setuptools import setup

setup(
    name='fgr',
    version='1.0',
    author='Tomer Roditi, Gal Babayof, Aaron Gerston (X-trodes LTD)',
    author_email='aarong@xtrodes.com, tomerroditi1@gmail.com',
    packages=['fgr', 'streamer'],
    include_package_data=True,
    license='GNU GPLv3',
    long_description=open('readme.md').read(),
    url="https://github.com/tomerroditi/Fingers-Gesture-Recognition.git",
    install_requires=open('requirements.txt').read()
)
