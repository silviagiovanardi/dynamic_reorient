from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'dynamic_reorient'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.xml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'srdf'), glob('srdf/*.srdf')),
        (os.path.join('share', package_name, 'meshes', 'robotiq'), glob('meshes/robotiq/*.STL')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='',
    description='Dynamic Pick and Reorient with UR5',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pick_reorient_node = dynamic_reorient.pick_reorient_node:main',
            'pose_estimator = dynamic_reorient.pose_estimator:main',
        ],
    },
)
