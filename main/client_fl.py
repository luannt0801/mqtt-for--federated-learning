from main.utils import *
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import paho.mqtt.client
import os
import time
import json
import numpy as np
import math
from collections import OrderedDict
from main.model import start_training_task

#from model_api.src.ml_api import start_line


broker_name = "192.168.1.119"
#broker_name = "192.168.10.128"

#global start_line
start_line = 0
start_benign = 0
start_main_dga = 0
#percent_main_dga = 0.8
#1 round
#total_data_dgas = 1980
num_line = 20
num_file = 1
# chi tinh main dga nhung so luong bengin van = dgas
#len_dga_types = 1
count = 0
alpha = 0.6
arr_num_line = [134, 279, 590, 111, 157, 196, 109, 659, 100, 126, 185, 145, 274, 264, 239, 89, 189, 206, 89, 145, 106, 825, 143, 134, 603, 114, 123, 374, 119, 124, 715, 376, 128, 101, 114, 249, 85, 224, 280, 69, 149, 62, 427, 130, 102, 102, 104, 116, 67, 139] 

num_line = arr_num_line[count]

class client_luan:
    def on_connect(client, userdata, flags, rc):
        print_log("Connected with result code "+str(rc))

    def on_disconnect(client, userdata, rc):
        print_log("Disconnected with result code "+str(rc))
        #reconnect
        client.reconnect()

    def on_message(client, userdata, msg):
        print_log(f"on_message {client._client_id.decode()}")
        print_log("RECEIVED msg from " + msg.topic)
        topic = msg.topic
        if topic == "dynamicFL/req/"+client_id:
            handle_cmd(client, userdata, msg)


    def on_subscribe(client, userdata, mid, granted_qos):
        print_log("Subscribed: " + str(mid) + " " + str(granted_qos))

    def do_evaluate_connection(client):
        print_log("doing ping")
        client_id = client._client_id.decode("utf-8")
        result = ping_host(broker_name)
        result["client_id"] = client_id
        result["task"] = "EVA_CONN"
        client.publish(topic="dynamicFL/res/"+client_id, payload=json.dumps(result))
        print_log(f"publish to topic dynamicFL/res/{client_id}")
        return result
    def do_evaluate_data():
        pass

    def do_train_non_iid(client):
        global alpha
        global start_line
        global start_benign
        global start_main_dga
        
        print_log(f"start training")
        client_id = client._client_id.decode("utf-8")
        # print(start_line)
        # print(start_bengin)
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
        client.publish(topic="dynamicFL/res/"+client_id, payload=json.dumps(payload))
        print_log(f"end training")

    def do_train(client):
        
        global start_line
        global start_benign
        global start_main_dga
        global arr_num_line
        global count
        
        print_log(f"start training")
        # print(f" start line dga {start_line}")
        print(f"quantity main dga in round {int(alpha*arr_num_line[count]*10)}")
        print(f"quantity 9 dga in round {int((1-alpha)*arr_num_line[count])}")
        client_id = client._client_id.decode("utf-8")
        #print(start_line)
        #print(start_bengin)
        result = start_training_task(start_line, start_main_dga, start_benign, arr_num_line, count, alpha)
        count += 1
        # alpha
        start_main_dga = start_main_dga + (int(alpha*arr_num_line[count-1]*10))
        start_line = start_line + (int((1-alpha)*arr_num_line[count-1]))
        # ko co alpha
        # start_line = start_line + (arr_num_line[count-1])
        start_benign = start_benign + int((arr_num_line[count-1] * 10))
        print("Luan check: ", start_benign)
        result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
        payload = {
            "task": "TRAIN",
            "weight": result_np
        }
        client.publish(topic="dynamicFL/res/"+client_id, payload=json.dumps(payload))
        print_log(f"end training")

    def do_test():
        pass

    def do_update_model():
        pass

    def do_stop_client(client):
        print_log("stop client")
        client.loop_stop()

    def handle_task(msg, client):
        task_name = msg.payload.decode("utf-8")
        if task_name == "EVA_CONN":
            do_evaluate_connection(client)
        elif task_name == "EVA_DATA":
            do_evaluate_data(client)
        elif task_name == "TRAIN":
            do_train(client)
        elif task_name == "TEST":
            do_test(client)
        elif task_name == "UPDATE":
            do_update_model(client)
        elif task_name == "REJECTED":
            do_add_errors(client)
        elif task_name == "STOP":
            do_stop_client(client)
        else:
            print_log(f"Command {task_name} is not supported")
            
    def join_dFL_topic(client):
        client_id = client._client_id.decode("utf-8")
        client.publish(topic="dynamicFL/join", payload=client_id)
        print_log(f"{client_id} joined dynamicFL/join of {broker_name}")

    def do_add_errors(client_id):
        publish.single(topic="dynamicFL/errors", payload=client_id, hostname=broker_name, client_id=client_id)

    def wait_for_model(client_id):
        msg = subscribe.simple("dynamicFL/model", hostname=broker_name)
        fo = open("mymodel.pt", "wb")
        fo.write(msg.payload)
        fo.close()
        print_log(f"{client_id} write model to mymodel.pt")

    def handle_cmd(client, userdata, msg):
        print_log("wait for cmd")
        client_id = client._client_id.decode("utf-8")
        handle_task(msg, client)
        print_log(f"{client_id} finished task {msg.payload.decode()}")

    def handle_model(client, userdata, msg):
        print_log("receive model")
        f = open("newmode.pt","wb")
        f.write(msg.payload)
        f.close()
        print_log("done write model")
        client_id = client._client_id.decode("utf-8")
        result = {
            "client_id": client_id,
            "task": "WRITE_MODEL"
        }
        client.publish(topic="dynamicFL/res/"+client_id, payload=json.dumps(result))
    