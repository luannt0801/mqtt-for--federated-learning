from main.client_fl import *
from main.utils import *

import paho.mqtt.client as client
import sys
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import time


if __name__ == "__main__":

    broker_name = "192.168.1.119"
    # port_mqtt = 1883

    client_id = "client_" + sys.argv[1]
    print(client_id)
    time.sleep(5)

    client_fl = Client(broker_name, client_id)
    
    # client_fl.connect(broker_name, port=port_mqtt, keepalive=3600)

    # client_fl.on_connect
    # client_fl.on_disconnect
    # client_fl.on_message
    # client_fl.on_subscribe

    client_fl.message_callback_add("dynamicFL/model/"+client_id, client_fl.handle_model)
    client_fl.message_callback_add("dynamicFL/model/all_client", client_fl.handle_model)

    client_fl.loop_start()

    client_fl.subscribe(topic="dynamicFL/model/"+client_id)
    client_fl.subscribe(topic="dynamicFL/model/all_client")

    client_fl.subscribe(topic="dynamicFL/req/"+client_id)

    client_fl.publish(topic="dynamicFL/join", payload=client_id)
    print_log(f"{client_id} joined dynamicFL/join of {broker_name}")

    client_fl._thread.join()
    time.sleep(30)
    print_log("client exits")