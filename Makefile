.PHONY: clean test prepforbuild artifacts uploadbuild release

# Prior to execution update the versions in setup.py and pyproject.toml

# Remove any existing build, dist, or egg-info folders
clean:
	rm -rf build dist *.egg-info

# Prepare the build environment
prepforbuild:
	pip install --upgrade twine setuptools wheel

# Build artifacts (source and wheel)
artifacts: test
	python3 setup.py sdist bdist_wheel

# Upload artifacts to PyPI
uploadbuild: artifacts
	python3 -m twine upload dist/*

# Full release process: prepare, build, and upload
release: clean prepforbuild artifacts uploadbuild
	@echo "Release completed successfully!"