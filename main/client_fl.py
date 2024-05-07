import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import json
from main.model import start_training_task
from main.utils import ping_host, print_log
from main.add_config import client_config

class FLClient:
    def __init__(self, client_id, broker_name):
        self.client_id = client_id
        self.broker_name = broker_name
        self.start_line = client_config["start_line"]
        self.start_benign = client_config["start_benign"]
        self.start_main_dga = client_config["start_main_dga"]
        self.count = client_config["count"]
        self.alpha = client_config["alpha"]
        self.arr_num_line = client_config["arr_num_line"]
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