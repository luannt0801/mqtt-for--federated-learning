import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe

# import paho.mqtt.client as client
from paho.mqtt.client import Client as MqttClient

import time
import threading
import json
import logging

import torch
from collections import OrderedDict

from main.utils import *

client_dict = {}
client_trainres_dict = {}
NUM_DEVICE = 0
n_round = 0
NUM_ROUND = 10
time_between_two_round = 30

logger = logging.getLogger()

class Server(MqttClient):
    def __init__(self, broker_address, port_mqtt, client_id):
        super().__init__(client_id)
        # self.on_connect = self._on_connect
        # self.on_disconnect = self._on_disconnect
        # self.on_message = self._on_message
        # self.on_subscribe = self._on_subscribe

        # self.broker_address = broker_address

        # self.client_id = client_id

        self.client_dict = {}
        self.client_trainres_dict = {}
        self.NUM_ROUND = 50
        self.NUM_DEVICE = 2
        self.time_between_two_round = 10
        self.round_state = "finished"
        self.n_round = 0

    # check connect to broker return result code
    def _on_connect(self, userdata, flags, rc):
        if rc == '':
            print_log("Connect fault")
        else:
            print_log("Connected with result code "+str(rc))
        self.subscribe("dynamicFL/join")

    # while disconnect reconnect
    def _on_disconnect(self, userdata, rc):
        print_log("Disconnected with result code "+str(rc))
        self.reconnect()

    # handle message receive from client
    def _on_message(self, userdata, msg):
        print(f"received msg from {msg.topic}")
        topic = msg.topic
        if topic == "dynamicFL/join": # topic is join --> handle_join
            self.handle_join(self, userdata, msg)
        elif "dynamicFL/res" in topic:
            tmp = topic.split("/")
            this_client_id = tmp[2]
            self.handle_res(this_client_id, msg)

    def _on_subscribe(self, mosq, obj, mid, granted_qos):
        print_log("Subscribed: " + str(mid) + " " + str(granted_qos))

    # send task to client
    def send_task(self, task_name, this_client_id):
        print_log("publish to " + "dynamicFL/req/"+this_client_id)
        self.publish(topic="dynamicFL/req/"+this_client_id, payload=task_name)

    # send model to client
    def send_model(self, path, this_client_id):
        with open(path, "rb") as f:
            data = f.read()
        self.publish(topic="dynamicFL/model/all_client", payload=data)

    # do aggregated Model
    def aggregated_models(self):
        sum_state_dict = OrderedDict()
        for client_id, state_dict in self.client_trainres_dict.items():
            for key, value in state_dict.items():
                if key in sum_state_dict:
                    sum_state_dict[key] = sum_state_dict[key] + torch.tensor(value, dtype=torch.float32)
                else:
                    sum_state_dict[key] = torch.tensor(value, dtype=torch.float32)
        num_models = len(self.client_trainres_dict)
        avg_state_dict = OrderedDict((key, value / num_models) for key, value in sum_state_dict.items())
        torch.save(avg_state_dict, f'model_round_{self.n_round}.pt')
        torch.save(avg_state_dict, "saved_model/LSTMModel.pt")
        self.client_trainres_dict.clear()

    # check ping packet loss
    def handle_pingres(self, this_client_id, msg):
        ping_res = json.loads(msg.payload)
        this_client_id = ping_res["client_id"]
        if ping_res["packet_loss"] == 0.0:
            print_log(f"{this_client_id} is a good client")
            state = self.client_dict[this_client_id]["state"]
            print_log(f"state {this_client_id}: {state}, round: {self.n_round}")
            if state == "joined" or state == "trained":
                self.client_dict[this_client_id]["state"] = "eva_conn_ok"
                count_eva_conn_ok = sum(1 for client_info in self.client_dict.values() if client_info["state"] == "eva_conn_ok")
                if count_eva_conn_ok == self.NUM_DEVICE:
                    print_log("publish to dynamicFL/model/all_client")
                    self.send_model("saved_model/LSTMModel.pt", self, this_client_id)

    #
    def handle_trainres(self, this_client_id, msg):
        payload = json.loads(msg.payload.decode())
        self.client_trainres_dict[this_client_id] = payload["weight"]
        state = self.client_dict[this_client_id]["state"]
        if state == "model_recv":
            self.client_dict[this_client_id]["state"] = "trained"

    def handle_update_writemodel(self, this_client_id, msg):
        state = self.client_dict[this_client_id]["state"]
        if state == "eva_conn_ok":
            self.client_dict[this_client_id]["state"] = "model_recv"
            self.send_task("TRAIN", self.client, this_client_id)
            count_model_recv = sum(1 for client_info in self.client_dict.values() if client_info["state"] == "model_recv")
            if count_model_recv == self.NUM_DEVICE:
                print_log(f"Waiting for training round {self.n_round} from client...")
                logger.info(f"Start training round {self.n_round} ...")

    def handle_res(self, this_client_id, msg):
        data = json.loads(msg.payload)
        cmd = data["task"]
        if cmd == "EVA_CONN":
            print_log(f"{this_client_id} complete task EVA_CONN")
            self.handle_pingres(this_client_id, msg)
        elif cmd == "TRAIN":
            print_log(f"{this_client_id} complete task TRAIN")
            self.handle_trainres(this_client_id, msg)
        elif cmd == "WRITE_MODEL":
            print_log(f"{this_client_id} complete task WRITE_MODEL")
            self.handle_update_writemodel(this_client_id, msg)

    def handle_join(self, userdata, msg):
        this_client_id = msg.payload.decode("utf-8")
        print("joined from"+" "+this_client_id)
        self.client_dict[this_client_id] = {"state": "joined"}
        self.subscribe(topic="dynamicFL/res/"+this_client_id)

    # start round
    def start_round(self):
        self.n_round = self.n_round + 1
        print_log(f"server start round {self.n_round}")
        round_state = "started"
        for client_i in self.client_dict:
            self.send_task("EVA_CONN", self.client, client_i)
        while len(self.client_trainres_dict) != self.NUM_DEVICE:
            time.sleep(1)
        time.sleep(1)
        self.end_round()

    def handle_next_round_duration(self):
        while len(self.client_trainres_dict) < self.NUM_DEVICE:
            time.sleep(1)

    def end_round(self):
        logger.info(f"server end round {self.n_round}")
        print_log(f"server end round {self.n_round}")
        round_state = "finished"
        if self.n_round < self.NUM_ROUND:
            self.handle_next_round_duration()
            self.aggregated_models()
            t = threading.Timer(self.time_between_two_round, self.start_round)
            t.start()
        else:
            self.aggregated_models()
            for c in self.client_dict:
                self.send_task("STOP", self.client, c)
                print_log(f"send task STOP {c}")
            logger.info(f"Stop all!")
            self.client.loop_stop()

