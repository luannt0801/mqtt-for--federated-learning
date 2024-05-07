import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import sys
import os
import time
import json
from main.model import start_training_task
from main.utils import ping_host, print_log

class FLClient:
    def __init__(self, client_id, broker_name):
        self.client_id = client_id
        self.broker_name = broker_name
        self.start_line = 0
        self.start_benign = 0
        self.start_main_dga = 0
        self.count = 0
        self.alpha = 0.6
        self.arr_num_line = [134, 279, 590, 111, 157, 196, 109, 659, 100, 126, 185, 145, 274, 264, 239, 89, 189, 206, 89, 145, 106, 825, 143, 134, 603, 114, 123, 374, 119, 124, 715, 376, 128, 101, 114, 249, 85, 224, 280, 69, 149, 62, 427, 130, 102, 102, 104, 116, 67, 139]
        self.num_line = self.arr_num_line[self.count]

        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe

    def on_connect(self, client, userdata, flags, rc):
        print_log(f"Connected with result code {rc}")
        self.join_dFL_topic()

    def on_disconnect(self, client, userdata, rc):
        print_log(f"Disconnected with result code {rc}")
        # Reconnect
        client.reconnect()

    def on_message(self, client, userdata, msg):
        print_log(f"on_message {client._client_id.decode()}")
        print_log(f"RECEIVED msg from {msg.topic}")
        topic = msg.topic
        if topic == "dynamicFL/req/"+self.client_id:
            self.handle_cmd(msg)
        elif topic == "dynamicFL/model/all_client":
            self.handle_model(msg)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print_log(f"Subscribed: {mid} {granted_qos}")

    def do_evaluate_connection(self):
        print_log("doing ping")
        result = ping_host(self.broker_name)
        result["client_id"] = self.client_id
        result["task"] = "EVA_CONN"
        self.client.publish(topic="dynamicFL/res/"+self.client_id, payload=json.dumps(result))
        print_log(f"Published to topic dynamicFL/res/{self.client_id}")
        return result

    def do_train(self):
        print_log("start training")
        client_id = self.client_id
        result = start_training_task(self.start_line, self.start_main_dga, self.start_benign, self.arr_num_line, self.count, self.alpha)
        self.count += 1
        self.start_main_dga += int(self.alpha * self.arr_num_line[self.count-1] * 10)
        self.start_line += int((1 - self.alpha) * self.arr_num_line[self.count-1])
        self.start_benign += int(self.arr_num_line[self.count-1] * 10)
        print_log(f"Luan check: {self.start_benign}")
        result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
        payload = {
            "task": "TRAIN",
            "weight": result_np
        }
        self.client.publish(topic="dynamicFL/res/"+client_id, payload=json.dumps(payload))
        print_log("end training")

    def join_dFL_topic(self):
        self.client.publish(topic="dynamicFL/join", payload=self.client_id)
        print_log(f"{self.client_id} joined dynamicFL/join of {self.broker_name}")

    def handle_cmd(self, msg):
        print_log("wait for cmd")
        self.handle_task(msg)

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

    def do_evaluate_data(self):
        pass

    def do_test(self):
        pass

    def do_update_model(self):
        pass

    def do_stop_client(self):
        print_log("stop client")
        self.client.loop_stop()

    def do_add_errors(self):
        publish.single(topic="dynamicFL/errors", payload=self.client_id, hostname=self.broker_name, client_id=self.client_id)

    def wait_for_model(self):
        msg = subscribe.simple("dynamicFL/model", hostname=self.broker_name)
        with open("mymodel.pt", "wb") as fo:
            fo.write(msg.payload)
        print_log(f"{self.client_id} write model to mymodel.pt")

    def handle_model(self, client, userdata, msg):
        print_log("receive model")
        with open("newmode.pt", "wb") as f:
            f.write(msg.payload)
        print_log("done write model")
        result = {
            "client_id": self.client_id,
            "task": "WRITE_MODEL"
        }
        self.client.publish(topic="dynamicFL/res/"+self.client_id, payload=json.dumps(result))

    def message_callback_add(self, sub, callback):
        """Register a message callback for a specific topic.
        Messages that match 'sub' will be passed to 'callback'. Any
        non-matching messages will be passed to the default on_message
        callback.

        Call multiple times with different 'sub' to define multiple topic
        specific callbacks.

        Topic specific callbacks may be removed with
        message_callback_remove()."""
        if callback is None or sub is None:
            raise ValueError("sub and callback must both be defined.")

        with self._callback_mutex:
            self._on_message_filtered[sub] = callback

    def start(self):
        self.client.connect(self.broker_name, port=1883, keepalive=3600)
        self.client.message_callback_add("dynamicFL/model/all_client", self.handle_model)
        self.client.loop_start()

        self.client.subscribe(topic="dynamicFL/model/all_client")
        self.client.subscribe(topic="dynamicFL/req/"+self.client_id)

        self.client.publish(topic="dynamicFL/join", payload=self.client_id)
        print_log(f"{self.client_id} joined dynamicFL/join of {self.broker_name}")

        self.client._thread.join()
        print_log("client exits")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py [client_id]")
        sys.exit(1)

    client_id = "client_" + sys.argv[1]
    print(client_id)
    time.sleep(5)
    fl_client = FLClient(client_id, "192.168.1.119")
    fl_client.start()
