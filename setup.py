from setuptools import setup

if __name__ == "__main__":
    setup(
        name='pyspark_pipes',
        version='0.1',
        description='Helper functions for building complex Spark ML pipelines',
        long_description=None,
        url='https://github.com/daniel-acuna/pyspark_pipes',
        author='Daniel E. Acuna',
        author_email='deacuna@syr.edu',
        license='(c) 2016, 2017, 2018 Daniel E. Acuna',
        install_requires=['pyspark'],
        packages=['pyspark_pipes'],
        package_data={}
    )