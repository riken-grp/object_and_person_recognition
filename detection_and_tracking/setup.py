from glob import glob

from setuptools import setup

package_name = 'detection_and_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yasutomo57jp',
    maintainer_email='yasutomo57jp@gmail.com',
    description='A 3D object detection and tracking using a realsense',
    license='AGPL-3.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'detection_and_tracking_node = {package_name}.detection_and_tracking_node:main',
            f'pose_estimation_node = {package_name}.pose_estimation_node:main',
        ],
    },
)
