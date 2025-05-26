from setuptools import setup, find_packages

setup(
    name='partinstruct',
    version='0.1.0',
    description='PartInstruct: A Part-level Instruction Following benchmark for Fine-grained Robot Manipulation',
    author='Yifan Yin; Zhengtao Han',
    author_email='yyin34@jhu.edu; zhan47@jhu.edu',
    packages=find_packages(),
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License'
    ]
)
