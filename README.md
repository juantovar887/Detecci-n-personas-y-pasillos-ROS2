# Deteccion personas y pasillos ROS2
Sistema de percepción para la detección de personas y pasillos objetivo, creditos por la implementación de YOLOv8 en ROS2 al usuario @mgonzs13, cualquier duda sobre la implementación dirigirse a su repositorio: https://github.com/mgonzs13/yolov8_ros

Para la ejecución del paquete primero se debe llamar al nodo publicador ya sea offline/online, para ello ejecuta el siguiente comando:
    ros2 run person_perception video_publisher (ruta donde se encuentra el video)
    ros2 run person_perception image_publisher (publica directo de la camara)

En una nueva terminal lanzamos todo el paquete, el cual tiene por default como parametro de imagen de entrada (input_image) el topic de image_publisher, y clase objetivo (class) class1. Para lanzar el paquete con video y otras clases, lo hacemos
de la siguiente manera:
    ros2 launch perception_launch person_perception_launch.py input_image:=/video class:=class2 (la clase que busques en el frame)
    ros2 launch perception_launch person_perception_launch.py class:=class2 (la clase que busques en el frame) 'Trabajando con imagenes online'

Para visulizar los resultados usa Rviz2 y busca el topic hall_image.
