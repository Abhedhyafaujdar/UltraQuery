from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
setup(
    name="UltraQuery",
    version="0.0.3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=["pandas","matplotlib"],
    license="MIT",
    author="Mayank Chaudhary,Krishna Agarwal,Abhedhya Faujdar",
    author_email="mayankchaudhary92197@gmail.com,krishnaiit7@gmail.com,fabhedhya@gmail.com",
    entry_points={
        'console_scripts':[
            'ultraquery=UltraQuery_core.cli:main',
        ],
    },
    python_requires=">=3.10"
)