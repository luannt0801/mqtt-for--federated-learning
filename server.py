import json
import paho.mqtt.client as client
import logging
import datetime
from main.utils import *
from main.server_fl import *


LOG_DIR = 'model_api/src/logs'
LOG_FILE = f"model_api/src/logs/app-{datetime.today().strftime('%Y-%m-%d')}.log"

    
if __name__ == "__main__":
   
   #  NUM_ROUND = 50
   #  NUM_DEVICE = 10
   #  global global_model
   #  client_dict = {}
   #  client_trainres_dict = {}
   #  #round_duration = 50
   #  time_between_two_round = 10
   #  round_state = "finished"
   #  n_round = 0

    broker_name = "192.168.101.246"
    port_mqtt = 1883

    server = Server(broker_name, port_mqtt, 'server')
    server.connect(broker_name, port=port_mqtt, keepalive=3600)
    print("do on connect")
    server.on_connect = server._on_connect
    server.on_disconnect
    server.on_message
    server.on_subscribe
    server.loop_start()
    server.subscribe(topic = "dynamicFL/join")
    print_log(f"server sub to dynamicFL/join of {broker_name}")
    print_log("server is waiting for clients to join the topic ...")

    while (NUM_DEVICE > len(client_dict)):
       time.sleep(1)

    server.start_round()
    server._thread.join()
    time.sleep(10)
    print_log("server exits")
    

    
