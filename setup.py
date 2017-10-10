import os

from setuptools import setup


with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    required = f.read().splitlines()

setup(
    name='sentiment-analysis-rd',
    version='1.1.0',
    packages=['loader', 'message', 'preprocessing', 'vectorizer'],
    install_requires=required,
    extras_require={},
    include_package_data=True,
    license='MIT License',
    description='A sentiment analysis library',
    long_description='A sentiment analysis library',
    url='https://github.com/lukawskikacper/sentiment-analysis-rd',
    author='Kacper Łukawski',
    author_email='kacper.lukawski@codete.com',
    entry_points={},
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
