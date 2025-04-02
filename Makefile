.PHONY: test prepforbuild artifacts uploadbuild release


# Prepare the build environment
prepforbuild:
	pip install --upgrade twine setuptools wheel

# Build artifacts (source and wheel)
artifacts: test
	python3 setup.py sdist bdist_wheel

# Upload artifacts to PyPI
uploadbuild: artifacts
	python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

# Full release process: prepare, build, and upload
release: prepforbuild artifacts uploadbuild
	@echo "Release completed successfully!"