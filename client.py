from main.client_fl import *
import paho.mqtt.client as client
import sys


if __name__ == "__main__":
    #start_line 
    client_id = "client_" + sys.argv[1]
    print(client_id)
    # sleep to load data 
    time.sleep(5)
    fl_client = client.Client(client_id=client_id)
    fl_client.connect(broker_name, port=1883, keepalive=3600)
    fl_client.on_connect = on_connect
    fl_client.on_disconnect = on_disconnect
    fl_client.on_message = on_message
    fl_client.on_subscribe = on_subscribe

    #fl_client.message_callback_add("dynamicFL/model/"+client_id, handle_model)
    fl_client.message_callback_add("dynamicFL/model/all_client", handle_model)
    
    fl_client.loop_start()

    #fl_client.subscribe(topic="dynamicFL/model/"+client_id)
    fl_client.subscribe(topic="dynamicFL/model/all_client")

    fl_client.subscribe(topic="dynamicFL/req/"+client_id)

    fl_client.publish(topic="dynamicFL/join", payload=client_id)
    print_log(f"{client_id} joined dynamicFL/join of {broker_name}")

    fl_client._thread.join()
    #time.sleep(10)
    print_log("client exits")