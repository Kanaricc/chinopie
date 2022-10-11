import setuptools

with open('README.md','r') as fh:
    long_description=fh.read()

setuptools.setup(
    name='chinopie',
    version='0.1.1',
    author='Cheng Chen',
    author_email='iovo7c@gmail.com',
    description='Chino Pie is a deep learning helper.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
)