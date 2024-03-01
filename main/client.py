import paho.mqtt.client as mqtt

from handle_client import *

broker_host = "10.130.9.133"
broker_port = 1883
time_to_live = 60

topic = "dulieu"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
# client.on_connect = on_connect(topic,)
client.on_connect = on_connect
client.on_message = on_message
client.on_subscribe = on_subcribe
client.on_unsubscribe = on_unsubcribe

client.user_data_set([])
client.connect(broker_host, broker_port)
client.loop_forever()
print(f"Nhan 10 messgae: {client.user_data_get()}")