import time
import paho.mqtt.client as mqtt

from handle_server import *


server = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
server.connect("192.168.1.140", 1883)
server.loop_start()
server.on_publish = on_publish

unacked_publish = set()
server.user_data_set(unacked_publish)

# # Our application produce some messages
# msg_info = server.publish("dulieu", "nguyenthanhluan1", qos=1)
# unacked_publish.add(msg_info.mid)

# msg_info2 = server.publish("dulieu", "nguyenthanhluan2 xin chao", qos=0)
# unacked_publish.add(msg_info2.mid)

# Wait for all message to be published
# while len(unacked_publish):
#     time.sleep(0.1)

# # Due to race-condition described above, the following way to wait for all publish is safer
# msg_info.wait_for_publish()
# msg_info2.wait_for_publish()

while len(unacked_publish):
    time.sleep(0.1)

server.disconnect()
server.loop_stop()


