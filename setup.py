from setuptools import setup, find_packages

setup(
    name='any-tokenizers',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=[
        # 필요한 의존성 패키지들 나열
    ]
)