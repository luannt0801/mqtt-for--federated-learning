from main.client_fl import *
from main.utils import *
import paho.mqtt.client as client
from main.client_fl import *
import sys


if __name__ == "__main__":
    # here

    broker_name = "192.168.101.246"
    port_mqtt = 1883

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

    num_line = arr_num_line[count]
    client_id = "client_" + sys.argv[1]
    client_fl = Client_fl(broker_name, port_mqtt, client_id)
    print(client_id)
    time.sleep(5)

    client_fl._on_connect
    client_fl._on_disconnect
    client_fl._on_message
    client_fl._on_subscribe

    client_fl.message_callback_add("dynamicFL/model/"+client_id, client_fl.handle_model)
    client_fl.message_callback_add("dynamicFL/model/all_client", client_fl.handle_model)

    client_fl.loop_start

    client_fl.subscribe(topic="dynamicFL/model/"+client_id)
    client_fl.subscribe(topic="dynamicFL/model/all_client")

    client_fl.subscribe(topic="dynamicFL/req/"+client_id)

    client_fl.publish(topic="dynamicFL/join", payload=client_id)
    print_log(f"{client_id} joined dynamicFL/join of {broker_name}")

    client_fl._thread.join()
    #time.sleep(10)
    print_log("client exits")

