import setuptools
import os


here = os.path.abspath(os.path.dirname(__file__))

install_requires = [
    "Flask==1.1.2",
    "flask-restplus==0.13.0",
    "pandas==1.2.4",
    "scikit-learn==0.24.1",
    "numpy==1.19.5",
    "waitress==1.3.1",
    "Werkzeug==0.16.1"
]
setuptools.setup(
    name='PredMntec CbV',
    version='1.0',
    include_package_data=True,
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    # packages=['data', 'main', 'test'],
    url='',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Data Science',

    ],
    license='',
    author='Capgemini',
    author_email='',
    description='Predicting the maintenance for the CbV',
    python_requires=">=3.6"
)