import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import sys
import time
from main.model import start_training_task
from main.utils import ping_host, print_log
from main.client_fl import *

if __name__ == "__main__":
    if len(sys.argv) != '':
        print("Usage: python client.py [client_id]")
        sys.exit(1)

    client_id = "client_" + sys.argv[1]
    print(client_id)
    time.sleep(5)
    fl_client = FLClient(client_id, "192.168.1.119")
    fl_client.start()
