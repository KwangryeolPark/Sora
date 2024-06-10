import setuptools

setuptools.setup(
    name="sora",
    version="1.0.0",
    author="Kwangryeol Park",
    author_email="pkr7098@gmail.com",
    description="SoRA PEFT",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.13.1',
        'scikit-learn',
        'transformers',
        'opendelta @ git+https://github.com/KwangryeolPark/OpenDelta.git'
    ]
)