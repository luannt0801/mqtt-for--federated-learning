from paho.mqtt.client import Client as MqttClient
from main.utils import *

class Client_fl(MqttClient):
    def __init__(self, broker_address, client_id):
        super().__init__(client_id)
        self.on_connect = self._on_connect
        self.on_disconnect = self._on_disconnect
        self.on_message = self._on_message
        self.on_subscribe = self._on_subscribe
        self.connect(broker_address, port=1883, keepalive=3600)
        self.broker_name = broker_address
        self.client_id = client_id

    def _on_connect(self, client, userdata, flags, rc):
        print_log("do _on_conncet")
        print_log("Connected with result code " + str(rc))
        client.subscribe("dynamicFL/join")

    def _on_disconnect(self, client, userdata, rc):
        print_log("Disconnected with result code " + str(rc))
        client.reconnect()

    def _on_message(self, client, userdata, msg):
        print(f"received msg from {msg.topic}")
        topic = msg.topic
        if topic == "dynamicFL/join":
            self.handle_join(client, userdata, msg)
        elif "dynamicFL/res" in topic:
            tmp = topic.split("/")
            this_client_id = tmp[2]
            self.handle_res(this_client_id, msg)
            
    def _on_subscribe(self, userdata, mid, granted_qos):
        print_log("Subscribed: " + str(mid) + " " + str(granted_qos))


test = Client_fl("192.168.101.246", 'test')
test.connect("192.168.101.246", 1883, keepalive=3600)
test.on_connect = test._on_connect
test.loop_start()