import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import json
import numpy as np

# import paho.mqtt.client as client
from paho.mqtt.client import Client as MqttClient

from collections import OrderedDict
from main.model import start_training_task
from main.utils import *

start_line = 0
start_benign = 0
start_main_dga = 0
num_line = 20
num_file = 1
count = 0
alpha = 0.6
arr_num_line = [134, 279, 590, 111, 157, 196, 109, 659, 100, 126, 185, 145, 274, 264, 239, 89, 189,
                     206, 89, 145, 106, 825, 143, 134, 603, 114, 123, 374, 119, 124, 715, 376, 128, 101,
                       114, 249, 85, 224, 280, 69, 149, 62, 427, 130, 102, 102, 104, 116, 67, 139] 
    
class Client_fl(MqttClient):
    def __init__(self, broker_address, port_mqtt, client_id):
        super().__init__(client_id)
        self.on_connect = self._on_connect
        self.on_disconnect = self._on_disconnect
        self.on_message = self._on_message
        self.on_subscribe = self._on_subscribe
        self.broker_name = broker_address
        self.client_id = client_id

    def _on_connect(self, userdata, flags, rc):
        print_log("Connected with result code "+str(rc))

    def _on_disconnect(self, userdata, rc):
        print_log("Disconnected with result code "+str(rc))
        #reconnect
        self.reconnect()

    def _on_message(self, userdata, msg):
        print_log(f"on_message {self.client_id.decode()}")
        print_log("RECEIVED msg from " + msg.topic)
        topic = msg.topic
        if topic == "dynamicFL/req/"+self.client_id:
            self.handle_cmd(self, userdata, msg)

    def _on_subscribe(self, mosq, obj, mid, granted_qos):
        print_log("Subscribed: " + str(mid) + " " + str(granted_qos))

    def do_evaluate_connection(self):
        print_log("doing ping")
        client_id = self.client_id
        result = ping_host(self.broker_name)
        result["client_id"] = client_id
        result["task"] = "EVA_CONN"
        publish.single(topic="dynamicFL/res/"+client_id, payload=json.dumps(result), hostname=self.broker_name)
        print_log(f"publish to topic dynamicFL/res/{client_id}")
        return result
    
    def do_evaluate_data(self):
        pass

    def do_train_non_iid(self):
        global alpha
        global start_line
        global start_benign
        global start_main_dga
        
        print_log(f"start training")
        client_id = self.client_id
        result = start_training_task(start_line, start_main_dga, start_benign, arr_num_line, count, alpha)
        start_line = start_line + arr_num_line[count-1]
        start_main_dga = start_main_dga + 1584
        start_benign = start_benign + 1980
        # Convert tensors to numpy arrays
        result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
        payload = {
            "task": "TRAIN",
            "weight": result_np
        }
        self.publish(topic="dynamicFL/res/"+client_id, payload=json.dumps(payload))
        print_log(f"end training")

    def do_train(self):
        global start_line
        global start_benign
        global start_main_dga
        global arr_num_line
        global count
        
        print_log(f"start training")
        client_id = self.client_id
        print(f"quantity main dga in round {int(alpha*arr_num_line[count]*10)}")
        print(f"quantity 9 dga in round {int((1-alpha)*arr_num_line[count])}")
        result = start_training_task(start_line, start_main_dga, start_benign, arr_num_line, count, alpha)
        count += 1
        start_main_dga = start_main_dga + (int(alpha*arr_num_line[count-1]*10))
        start_line = start_line + (int((1-alpha)*arr_num_line[count-1]))
        start_benign = start_benign + int((arr_num_line[count-1] * 10))
        print("Luan check: ", start_benign)
        result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
        payload = {
            "task": "TRAIN",
            "weight": result_np
        }
        self.publish(topic="dynamicFL/res/"+client_id, payload=json.dumps(payload))
        print_log(f"end training")

    def do_test(self):
        pass

    def do_update_model(self):
        pass

    def do_stop_client(self):
        print_log("stop client")
        self.loop_stop()

    def handle_task(self, msg):
        task_name = msg.payload.decode("utf-8")
        if task_name == "EVA_CONN":
            self.do_evaluate_connection()
        elif task_name == "EVA_DATA":
            self.do_evaluate_data()
        elif task_name == "TRAIN":
            self.do_train()
        elif task_name == "TEST":
            self.do_test()
        elif task_name == "UPDATE":
            self.do_update_model()
        elif task_name == "REJECTED":
            self.do_add_errors()
        elif task_name == "STOP":
            self.do_stop_client()
        else:
            print_log(f"Command {task_name} is not supported")

    def join_dFL_topic(self):
        client_id = self.client_id
        self.publish(topic="dynamicFL/join", payload=client_id)
        print_log(f"{client_id} joined dynamicFL/join of {self.broker_name}")

    def do_add_errors(self, client_id):
        publish.single(topic="dynamicFL/errors", payload=client_id, hostname=self.broker_name, client_id=client_id)

    def wait_for_model(self):
        msg = subscribe.simple("dynamicFL/model", hostname=self.broker_name)
        fo = open("mymodel.pt", "wb")
        fo.write(msg.payload)
        fo.close()
        print_log(f"{self.client_id} write model to mymodel.pt")

    def handle_cmd(self, client, userdata, msg):
        print_log("wait for cmd")
        self.handle_task(msg)
        print_log(f"{self.client_id} finished task {msg.payload.decode()}")

    def handle_model(self, client, userdata, msg):
        print_log("receive model")
        f = open("newmode.pt","wb")
        f.write(msg.payload)
        f.close()
        print_log("done write model")
        result = {
            "client_id": self.client_id,
            "task": "WRITE_MODEL"
        }
        self.publish(topic="dynamicFL/res/"+self.client_id, payload=json.dumps(result))