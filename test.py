# from paho.mqtt.client import Client as MqttClient
# from main.utils import *

# class Client_fl(MqttClient):
#     def __init__(self, broker_address, client_id):
#         super().__init__(client_id)
#         self.on_connect = self._on_connect
#         self.on_disconnect = self._on_disconnect
#         self.on_message = self._on_message
#         self.on_subscribe = self._on_subscribe
#         self.connect(broker_address, port=1883, keepalive=3600)
#         self.broker_name = broker_address
#         self.client_id = client_id

#     def _on_connect(self, client, userdata, flags, rc):
#         print_log("do _on_conncet")
#         print_log("Connected with result code " + str(rc))
#         client.subscribe("dynamicFL/join")

#     def _on_disconnect(self, client, userdata, rc):
#         print_log("Disconnected with result code " + str(rc))
#         client.reconnect()

#     def _on_message(self, client, userdata, msg):
#         print(f"received msg from {msg.topic}")
#         topic = msg.topic
#         if topic == "dynamicFL/join":
#             self.handle_join(client, userdata, msg)
#         elif "dynamicFL/res" in topic:
#             tmp = topic.split("/")
#             this_client_id = tmp[2]
#             self.handle_res(this_client_id, msg)
            
#     def _on_subscribe(self, userdata, mid, granted_qos):
#         print_log("Subscribed: " + str(mid) + " " + str(granted_qos))


# test = Client_fl("192.168.101.246", 'test')
# test.connect("192.168.101.246", 1883, keepalive=3600)
# test.on_connect = test._on_connect
# test.loop_start()

# from paho.mqtt.client import Client as MqttClient
# import logging

# class MyMqttClient(MqttClient):
#     def __init__(self, client_id="", clean_session=True, userdata=None, protocol=4, transport="tcp"):
#         super().__init__(client_id, clean_session, userdata, protocol, transport)
#         self.on_connect = self._on_connect  # Thiết lập callback khi kết nối
#         self.on_message = self._on_message  # Thiết lập callback khi nhận tin nhắn

#     def connect_to_broker(self, broker_address, port=1883):
#         logging.info(f"Connecting to broker {broker_address} on port {port}...")
#         self.connect(broker_address, port)

#     def _on_connect(self, client, userdata, flags, rc):
#         if rc == 0:
#             logging.info("Connected to broker successfully")
#             # Subscribe vào topic khi kết nối thành công
#             self.subscribe("file")
#         else:
#             print("error")
#             logging.error(f"Failed to connect to broker with return code {rc}")

#     def _on_message(self, client, userdata, msg):
#         logging.info(f"Received message: {msg.payload.decode()} on topic {msg.topic}")

# # Sử dụng class MyMqttClient
# if __name__ == "__main__":
#     broker_address = "192.168.101.246"
#     # client_id = "luan"
    
#     # Khởi tạo đối tượng MyMqttClient
#     mqtt_client = MyMqttClient()

#     # Kết nối đến broker và subscribe vào topic
#     mqtt_client.connect_to_broker(broker_address)

#     # Lặp vô hạn để duy trì kết nối và xử lý tin nhắn
#     mqtt_client.loop_forever()

import paho.mqtt.client as mqtt

class MyMQTTClient(mqtt.Client):
    def __init__(self, client_id="", clean_session=True, userdata=None, protocol=mqtt.MQTTv311):
        super().__init__(client_id, clean_session, userdata, protocol)
        
        # Set callbacks
        self.on_connect = self.on_connect_callback
        self.on_message = self.on_message_callback
        
    def on_connect_callback(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")
            # Subscribe to topic when connected
            client.subscribe("$SYS/#")
        else:
            print("Connection failed")
    
    def on_message_callback(self, client, userdata, message):
        print("Received message:", str(message.payload.decode("utf-8")))

def main():
    # Create an instance of MyMQTTClient
    mqtt_client = MyMQTTClient()
    
    # Connect to MQTT broker
    mqtt_client.connect("mqtt.eclipseprojects.io", 1883, 60)
    
    # Start the loop
    mqtt_client.loop_forever()

if __name__ == "__main__":
    main()
