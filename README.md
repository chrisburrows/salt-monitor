# salt-monitor
Simple script to analyse the level of salt in my water softener using computer vision.

Uses the [ESP32 Camera Web Server](https://github.com/easytarget/esp32-cam-webserver) to grab images from my water softener and 
analyses the image using [OpenCV](https://opencv.org/) looking for the red line indicating the salt level is low. Publishes status and images to MQTT.

