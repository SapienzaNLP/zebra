"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py
To create the package for pypi.
1. Run `make pre-release` (or `make pre-patch` for a patch release) then run `make fix-copies` to fix the index of the
   documentation.
2. Run Tests for Amazon Sagemaker. The documentation is located in `./tests/sagemaker/README.md`, otherwise @philschmid.
3. Unpin specific versions from setup.py that use a git install.
4. Commit these changes with the message: "Release: VERSION"
5. Add a tag in git to mark the release: "git tag VERSION -m 'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master
6. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
7. Check that everything looks correct by uploading the package to the pypi test server:
   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers
8. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
9. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.
10. Run `make post-release` (or `make post-patch` for a patch release).
"""
from collections import defaultdict

import setuptools


def parse_requirements_file(
    path, allowed_extras: set = None, include_all_extra: bool = True
):
    requirements = []
    extras = defaultdict(list)
    find_links = []
    with open(path) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git",
                req,
            )
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            if (
                line.startswith("-f")
                or line.startswith("--find-links")
                or line.startswith("--index-url")
            ):
                find_links.append(line.split(" ", maxsplit=1)[-1].strip())
                continue

            req, *needed_by = line.split("# needed by:")
            req = fix_url_dependencies(req.strip())
            if needed_by:
                for extra in needed_by[0].strip().split(","):
                    extra = extra.strip()
                    if allowed_extras is not None and extra not in allowed_extras:
                        raise ValueError(f"invalid extra '{extra}' in {path}")
                    extras[extra].append(req)
                if include_all_extra and req not in extras["all"]:
                    if "gpu" in extra:
                        extras["all-gpu"].append(req)
                    elif "cpu" in extra:
                        extras["all-cpu"].append(req)
                    else:
                        extras["all"].append(req)
            else:
                requirements.append(req)
    return requirements, extras, find_links


# allowed_extras = {"all", "all-cpu", "all-gpu", "onnx", "onnx-gpu", "serve", "dev", "faiss"}

# Load requirements.
install_requirements, extras, find_links = parse_requirements_file(
    "requirements.txt", #allowed_extras=allowed_extras
)

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here, so we don't import allennlp whilst setting up.
VERSION = {}  # type: ignore
with open("zebra/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zebra-qa",
    version=VERSION["VERSION"],
    author="Francesco Molfese",
    author_email="molfesefrancesco@gmail.com",
    description="ZEBRA: Zero-Shot Example-Based Retrieval Augmentation for Commonsense Question Answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sapienzanlp/zebra",
    keywords="NLP Sapienza sapienzanlp deep learning transformer pytorch retriever question answering commonsense large language models",
    packages=setuptools.find_packages(),
    include_package_data=True,
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
    license_files=["LICENSE"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=install_requirements,
    extras_require=extras,
    python_requires=">=3.10",
)
