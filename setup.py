from setuptools import setup

setup(
    name='my_package',
    version='0.1',
    packages=['src'],
    install_requires=[
        'common_ml @ git+ssh://git@github.com/qluvio/common-ml.git#egg=common_ml',
        'nemo_toolkit[all]',
        'quick_test_py @ git+https://github.com/elv-nickB/quick_test_py.git#egg=quick_test_py',
        'ollama',
        'cuda-python>=12.3'
    ]
)
