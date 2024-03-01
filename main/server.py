import time
import paho.mqtt.client as mqtt

from handle_server import *

broker_host = "10.130.9.133"
broker_port = 1883
time_to_live = 60

server = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
server.connect(broker_host, broker_port, time_to_live)
server.loop_start()
server.on_publish = on_publish

unacked_publish = set()
server.user_data_set(unacked_publish)



while len(unacked_publish):
    time.sleep(0.1)

server.disconnect()
server.loop_stop()


