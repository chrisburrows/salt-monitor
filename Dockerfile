FROM python:3.9
LABEL maintainer="chris.burrows1965@gmail.com"

RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install opencv-python-headless

RUN pip install requests netifaces paho-mqtt

COPY salt-monitor.py /usr/local/bin
RUN chmod 755 /usr/local/bin/salt-monitor.py

CMD ["python", "/usr/local/bin/salt-monitor.py"]


