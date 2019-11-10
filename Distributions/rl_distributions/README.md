This package contains utilities for basic distribution based on Udacity's Data Science nano degree

The following command can be used to generate a package file for uploading to Pypi / test pypi
cd [Package file dir]
python setup.py sdist
pip install twine

# commands to upload to the pypi test repository
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
pip install --index-url https://test.pypi.org/simple/ [package name]

# command to upload to the pypi repository
twine upload dist/*
pip install [package name]