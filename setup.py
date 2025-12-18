from setuptools import setup

setup(
    name='my_package',
    version='0.1',
    packages=['src'],
    install_requires=[
        'common_ml @ git+https://github.com/eluv-io/common-ml.git#egg=common_ml',
        'nemo_toolkit[all]',
        'deepmultilingualpunctuation==1.0.1',
        'spacy==3.7.2',
        'dacite',
        'setproctitle'
    ]
)
