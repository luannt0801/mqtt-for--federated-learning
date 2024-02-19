import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import paho.mqtt.client
import os
import time
import json

### address broker
broker_host= "emqx.io"
broker_name= "broker.emqx.io"

def do_evaluate_connection(client):
    print_log("doing Ping")