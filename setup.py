import setuptools

setuptools.setup(
    name='hm',
    version='0.1',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'voxelmorph',
    ],
    include_package_data=True,
    package_data={
        'hm': ['*.h5'],
        'hm': ['*.npy'],
    },
)
