from setuptools import setup
package_name = 'person_perception'

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
    maintainer='tombae',
    maintainer_email='tombae@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'yolov8_node = person_perception.yolov8_node:main',
                'debug_node =   person_perception.debug_node:main',
                'tracking_node = person_perception.tracking_node:main',
                'image_publisher = person_perception.image_publisher:main',
                'video_publisher = person_perception.video_publisher:main',
                'number_detector = person_perception.number_detector:main',
                'image_filter = person_perception.image_filter:main',
        ],
    },
)