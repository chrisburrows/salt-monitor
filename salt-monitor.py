#!/usr/bin/python3
#
# Send WoL packets controlled by MQTT messages.
#
# Chris Burrows
# 8 July 2021

import os
import argparse
import logging.handlers
import sys
from datetime import datetime
import time
import json

import cv2
import requests
import netifaces
import cv2 as cv
import numpy as np
import math
import paho.mqtt.client as mqtt

DEBUG = os.getenv("DEBUG", "false").lower() in ['true', 'enabled', 'yes']

# HSV low / high filter thresholds
LOW_HSV = (0,64,0)
HIGH_HSV = (16, 168, 255)

BOUNDING_BOX_TOP_LEFT = (150, 400)
BOUNDING_BOX_BOTTOM_RIGHT = (775, 600)

# default update is 6 hours
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "21600"))

MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt.local")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER", "mqtt")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "password")
MQTT_BASE_TOPIC = "salt"

MQTT_CHECKED_TIME_TOPIC = "{base}/last-checked-time".format(base=MQTT_BASE_TOPIC)
MQTT_SALT_NEEDED_TOPIC = "{base}/needed".format(base=MQTT_BASE_TOPIC)
MQTT_SALT_CAMERA_PROBLEM_TOPIC = "{base}/camera-problem".format(base=MQTT_BASE_TOPIC)
MQTT_SALT_CAMERA_IMAGE_TOPIC = "{base}/image".format(base=MQTT_BASE_TOPIC)
MQTT_SALT_CAMERA_DETECTION_IMAGE_TOPIC = "{base}/detection-image".format(base=MQTT_BASE_TOPIC)

HA_ICON_SALT_NEEDED = "mdi:align-vertical-top"
HA_ICON_UPDATE_TIME = "mdi:clock-outline"

CAMERA_RETRY_COUNT = int(os.getenv("CAMERA_RETRY_COUNT", "10"))
CAMERA_IP = os.getenv("CAMERA_IP", "192.168.200.206")
CAMERA_RESOLUTION = 10 # 1024x768
CAMERA_QUALITY = 5 # High
CAMERA_LAMP = 100 # Full on
camera_url = "http://{ip}".format(ip=CAMERA_IP)
camera_control_url = "{url}/control".format(url=camera_url)

LOG_FILENAME = '/var/log/salt-monitor.log'
LOG_FILENAME = 'salt-monitor.log'


mqtt_connected = False

def get_default_interface():
    """Get the interface to the default gateway"""
    return list(netifaces.gateways()['default'].values())[0][1]


def get_mac_address():
    """Get MAC address from active interface"""
    return netifaces.ifaddresses(get_default_interface())[netifaces.AF_LINK][0]['addr']


def check_publish_result(result):
    """Check repsonse from publishing"""
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
      log.error("Error publishing message {rc}".format(rc=result.rc))


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    """Callback to handle connection to MQTT broker"""

    log.info("MQTT: connected to broker with result code " + str(rc))

    if rc == 0:
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.

        client.subscribe(MQTT_BASE_TOPIC + "/command")
        check_publish_result(client.publish(MQTT_BASE_TOPIC + "/status", payload="online", retain=True))

        publish_home_assistant_discovery(client)

        mqtt_connected = True
    else:
        log.error("Failed to correctly login to MQTT broker")
        client.disconnect()
        time.sleep(5)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    """Callback to handle received messages"""

    log.debug("MQTT: Message " + msg.topic + " = " + str(msg.payload, "UTF-8"))


def on_log(client, userdata, level, buf):
    """Callback to handle logging"""

    if debug:
        log.debug("MQTT: Log " + buf )


def publish_home_assistant_discovery(client):
    """Publish discovery for the sensors"""

    device = {
        "connections": [("mac", get_mac_address())],
        "identifiers": get_mac_address(),
        "name": "Salt monitor",
        "suggested_area": "kitchen"
    }

    log.info("MQTT: publishing Home Assistant discovery data")
    payload = {
        "name": "Salt monitor last check time",
        "device": device,
        "state_topic": MQTT_CHECKED_TIME_TOPIC,
        "availability_topic": "{base}/status".format(base=MQTT_BASE_TOPIC),
        "payload_available": "online",
        "payload_not_available": "offline",
        "unique_id": "salt-last-checked-time",
        "icon": HA_ICON_UPDATE_TIME
    }
    discovery_topic = "homeassistant/sensor/salt/last-checked-time/config"
    check_publish_result(client.publish(discovery_topic, payload=json.dumps(payload), retain=True))

    payload = {
        "name": "Salt needed",
        "device": device,
        "state_topic": MQTT_SALT_NEEDED_TOPIC,
        "availability_topic": "{base}/status".format(base=MQTT_BASE_TOPIC),
        "payload_available": "online",
        "payload_not_available": "offline",
        "unique_id": "salt-needed",
        "icon": HA_ICON_SALT_NEEDED
    }
    discovery_topic = "homeassistant/binary_sensor/salt/salt-needed/config"
    check_publish_result(client.publish(discovery_topic, payload=json.dumps(payload), retain=True))

    payload = {
        "name": "Salt camera problem",
        "device": device,
        "state_topic": MQTT_SALT_CAMERA_PROBLEM_TOPIC,
        "availability_topic": "{base}/status".format(base=MQTT_BASE_TOPIC),
        "payload_available": "online",
        "payload_not_available": "offline",
        "unique_id": "salt-camera-problem",
        "device_class": "problem",
        "icon": HA_ICON_SALT_NEEDED
    }
    discovery_topic = "homeassistant/binary_sensor/salt/camera-problem/config"
    check_publish_result(client.publish(discovery_topic, payload=json.dumps(payload), retain=True))

    payload = {
        "name": "Salt camera image",
        "device": device,
        "topic": MQTT_SALT_CAMERA_IMAGE_TOPIC,
        "availability_topic": "{base}/status".format(base=MQTT_BASE_TOPIC),
        "payload_available": "online",
        "payload_not_available": "offline",
        "unique_id": "salt-camera-image",
        "icon": HA_ICON_SALT_NEEDED
    }
    discovery_topic = "homeassistant/camera/salt/config"
    check_publish_result(client.publish(discovery_topic, payload=json.dumps(payload), retain=True))

    payload = {
        "name": "Salt camera detection image",
        "device": device,
        "topic": MQTT_SALT_CAMERA_DETECTION_IMAGE_TOPIC,
        "availability_topic": "{base}/status".format(base=MQTT_BASE_TOPIC),
        "payload_available": "online",
        "payload_not_available": "offline",
        "unique_id": "salt-camera-detection-image",
        "icon": HA_ICON_SALT_NEEDED
    }
    discovery_topic = "homeassistant/camera/salt-detection/config"
    check_publish_result(client.publish(discovery_topic, payload=json.dumps(payload), retain=True))


def wait_for_mqtt_connection(client):
    """Wait until MQTT connects"""
    log.info("Waiting for MQTT to connect")
    while not mqtt_connected:
        client.loop(0.25)
        log.debug("Waiting for MQTT to connect...")
        

def update_check_time_sensor(client):
    """Update the sensor with the salt checked time"""
    log.info("MQTT: update salt checked time: {x}".format(x=datetime.now().isoformat()))
    check_publish_result(client.publish(MQTT_CHECKED_TIME_TOPIC, payload=datetime.now().isoformat(), retain=True))


def update_salt_camera_problem_sensor(client, problem):
    """Update the camera problem sensor"""
    log.info("MQTT: update salt camera problem sensor: {x}".format(x=problem))
    check_publish_result(client.publish(MQTT_SALT_CAMERA_PROBLEM_TOPIC, payload="ON" if problem else "OFF", retain=True))


def update_salt_needed_sensor(client, salt_needed):
    """Update the sensor with the salt status"""
    log.info("MQTT: update salt needed sensor: {x}".format(x=salt_needed))
    check_publish_result(client.publish(MQTT_SALT_NEEDED_TOPIC, payload="ON" if salt_needed else "OFF", retain=True))


def update_salt_image(client, image):
    """Update the camera image"""
    log.info("MQTT: update salt camera image sensor")
    check_publish_result(client.publish(MQTT_SALT_CAMERA_IMAGE_TOPIC, payload=convert_img_to_jpeg(image), retain=True))


def update_salt_detection_image(client, image):
    """Update the camera detection image"""
    log.info("MQTT: update salt camera detection image sensor")
    check_publish_result(client.publish(MQTT_SALT_CAMERA_DETECTION_IMAGE_TOPIC, payload=convert_img_to_jpeg(image), retain=True))


def send_camera_setting(setting, value):
    """Send a setting to the camera"""
    log.debug("CAMERA: setting camera parameter {var} = {val}".format(var=setting, val=value))

    r = requests.get(camera_control_url, params={'var': setting, 'val': value})
    if r.status_code == 200:
        return
    raise RuntimeError("Camera setting failed: {err}".format(err=r.text))


def convert_img_to_jpeg(img):
    """Convert ndarray format image into a JPEG byte array"""
    _, jpeg = cv2.imencode('.jpeg', img)
    return jpeg.tobytes()


def configure_camera():
    """Send config settings to camera"""
    send_camera_setting('lamp', '0')
    send_camera_setting('autolamp', 0)
    send_camera_setting('framesize', CAMERA_RESOLUTION)
    send_camera_setting('quality', CAMERA_QUALITY)

    send_camera_setting('contrast', 0)
    send_camera_setting('brightness', 0)
    send_camera_setting('saturation', 0)
    send_camera_setting('awb', 1)
    send_camera_setting('hmirror', 0)
    send_camera_setting('vflip', 0)

def turn_on_lamp():
    """Turn on camera light"""
    log.debug("CAMERA: turning on lamp")
    send_camera_setting('lamp', CAMERA_LAMP)


def turn_off_lamp():
    """Turn on camera light"""
    log.debug("CAMERA: turning off lamp")
    send_camera_setting('lamp', '0')


def get_camera_image():
    """Get a salt image from the camera"""

    log.debug("CAMERA: fetching image")
    configure_camera()

    turn_on_lamp()
    # we take multiple images in sequence to give the AGC a chance to
    # settle on good values before keeping the final image to work with
    for i in range(CAMERA_RETRY_COUNT):
        time.sleep(0.5)
        r = requests.get("{url}/capture".format(url=camera_url))
    turn_off_lamp()
    if r.status_code == 200:
        raw_image = np.array(bytearray(r.content), dtype=np.uint8)
        return cv.imdecode(raw_image, -1)
    raise RuntimeError("Failed to capture image")


def get_file_image(filename):
    """Get an image from file"""
    log.debug("CAMERA: Loading image from file {file}".format(file=filename))
    image = cv.imread(filename)
    return image


def get_image():
    """Fetch an image to analyse"""
    if args.file is not None:
        return get_file_image(args.file)
    return get_camera_image()
    
def line_in_box(box_tl, box_br, line):
    """Check if a line in wholly within a bounding box"""
    if line[0] < box_tl[0] or line[2] < box_tl[0] or line[0] > box_br[0] or line[2] > box_br[0]:
        return False

    if line[1] < box_tl[1] or line[3] < box_tl[1] or line[1] > box_br[1] or line[3] > box_br[1]:
        return False

    return True

def check_salt_needed(src):
    """Check salt status by looking for the re-stock red line"""

    if debug:
        cv.imshow('Salt Camera Image', src)
        cv.waitKey(0)

    # filter to smooth some noise
    image = cv.GaussianBlur(src,(5,5),cv.BORDER_DEFAULT)

    if debug:
        cv.imshow('Salt Camera Image', image)
        cv.waitKey(0)

    # convert colour space to HSV
    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # filter image by threshold produces a binary image
    thresh = cv.inRange(img_hsv, LOW_HSV, HIGH_HSV)

    if debug:
        cv.imshow('Salt Camera Image', thresh)
        cv.waitKey(0)

    # detect edges in the image
    edges = cv.Canny(thresh, 30, 200, apertureSize=3, L2gradient=True)

    if debug:
        cv.imshow('Salt Camera Image', edges)
        cv.waitKey(0)


    # look for lines
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=200, maxLineGap=75)

    # draw bounding box where we look for lines... this must include the red line guage
    cv.line(src, BOUNDING_BOX_TOP_LEFT, (BOUNDING_BOX_BOTTOM_RIGHT[0], BOUNDING_BOX_TOP_LEFT[1]), (0, 0, 255), 2)
    cv.line(src, (BOUNDING_BOX_BOTTOM_RIGHT[0], BOUNDING_BOX_TOP_LEFT[1]), BOUNDING_BOX_BOTTOM_RIGHT, (0, 0, 255), 2)
    cv.line(src, BOUNDING_BOX_TOP_LEFT, (BOUNDING_BOX_TOP_LEFT[0], BOUNDING_BOX_BOTTOM_RIGHT[1]), (0, 0, 255), 2)
    cv.line(src, (BOUNDING_BOX_TOP_LEFT[0], BOUNDING_BOX_BOTTOM_RIGHT[1]), BOUNDING_BOX_BOTTOM_RIGHT, (0, 0, 255), 2)

    if debug:
        cv.imshow('Salt Camera Image', src)
        cv.waitKey(0)

    # if we find a line at an angle to the horizontal between -4 and +4 degrees
    # we'll assume this is the red need salt line
    is_salt_needed = False
    if lines is not None:
        for line in lines:
            if line_in_box(BOUNDING_BOX_TOP_LEFT, BOUNDING_BOX_BOTTOM_RIGHT, line[0]):
                for x1,y1,x2,y2 in line:
                    cv.line(src, (x1,y1), (x2,y2), (0,255,0), 2)

                    if x2 != x1:
                        angle = math.atan((y2 - y1) / (x2 - x1)) * 90 / np.pi
                        log.debug("IMAGE: line at angle: {angle}".format(angle=angle))
                        if 4 > angle > -4:
                            log.debug("IMAGE: Salt required line at angle: {angle}".format(angle=angle))
                            is_salt_needed = True
        if debug:
            cv.imshow('Salt Camera Image', src)
            cv.waitKey(0)
    else:
        log.debug("IMAGE: No salt threshold line detected")

    #picture = cv.imencode('.jpg', src)
    #return (is_salt_needed, picture[1].tobytes())
    return (is_salt_needed, src)


def run_on_test_image(filename):
    """Run a simple analysis on a test image file"""
    print("Loading test image from {filename}".format(filename=filename))
    image = cv.imread(filename)
    salt_needed, detection_jpeg = check_salt_needed(image)
    print("Salt needed: {salt_needed}".format(salt_needed=salt_needed))
    cv.destroyAllWindows()


# parse command line to see what we're meant to be doing
parser = argparse.ArgumentParser(description="Salt Monitor")
parser.add_argument("--camera", action="store", default=CAMERA_IP, help="IP address of camera (ESP32 webserver)")
parser.add_argument("--file", action="store", help="Name of a test image")
parser.add_argument("--debug", action="store_true", default=DEBUG, help="Enable debug mode")
parser.add_argument("--test", action="store_true", default=False, help="Disable MQTT reporting")

args = parser.parse_args()

debug = args.debug
mqtt_enabled = not args.test
camera_url = "http://{ip}".format(ip=args.camera)
camera_control_url = "{url}/control".format(url=camera_url)

# setup logging
log = logging.getLogger()
stderr_handler = logging.StreamHandler()
formatter = logging.Formatter('{asctime} {levelname:8s} {message}', style='{')
stderr_handler.setFormatter(formatter)

log.addHandler(stderr_handler)
log.setLevel(logging.DEBUG)
log.info("+-----------------+")
log.info("|   Starting up   |")
log.info("+-----------------+")
log.info("")

client = None
try:
    # Initialise connect to MQTT broker
    if mqtt_enabled:
        client_id = "salt-monitor-{mac}".format(mac=get_mac_address())
        client = mqtt.Client(client_id=client_id)
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_log = on_log
        client.will_set(MQTT_BASE_TOPIC + "/status", payload="offline", retain=True)

    while(True):

        try:
            if client is not None:
                log.info("MQTT: connecting to broker {ip}:{port}".format(ip=MQTT_BROKER, port=MQTT_PORT))
                client.connect(MQTT_BROKER, port=MQTT_PORT, keepalive=60)
                client.loop_start()

            while(True):
                img = get_image()

                log.debug("IMAGE: writing camera image file")
                cv.imwrite('salt.jpg', img)
                log.debug("IMAGE: Analysing image")
                salt_needed, detection_img = check_salt_needed(img.copy())

                log.debug("IMAGE: writing analysed detection image file")
                cv.imwrite('detection.jpg', detection_img)

                if client is not None:
                    log.debug("MQTT: updating state")
                    update_salt_needed_sensor(client, salt_needed)
                    update_check_time_sensor(client)
                    update_salt_image(client, img)
                    update_salt_detection_image(client, detection_img)
                    update_salt_camera_problem_sensor(client, False)

                log.debug("Waiting until next update")
                time.sleep(UPDATE_INTERVAL)

        except RuntimeError as e:
            log.error("CAMERA: API error: {err}".format(err=str(e)))
            if client is not None:
                update_salt_camera_problem_sensor(client, True)
            time.sleep(30)

        except ConnectionRefusedError:
            log.error("MQTT: Failed to connect to broker on {ip}:{port}".format(ip=MQTT_BROKER, port=MQTT_PORT))
            time.sleep(30)

        except requests.exceptions.ConnectionError as e:
            log.error("CAMERA: API error: {err}".format(err=str(e)))
            if client is not None:
                update_salt_camera_problem_sensor(client, True)
            time.sleep(30)

except KeyboardInterrupt:
    log.info("Interrupted... shutting down")
    
# mark us offline and disconnect
log.info("MQTT: Publishing offline status")

if client is not None:
    check_publish_result(client.publish(MQTT_BASE_TOPIC + "/status", payload="offline"))

time.sleep(3)
log.info("MQTT: disconnecting")

if client is not None:
    client.loop_stop()
    client.disconnect()

