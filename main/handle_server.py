import time
import paho.mqtt.client as mqtt
 
def on_publish(server, userdata,mid , reason_code, properties):
    try:
        userdata.remove(mid)
    except KeyError:
        print("on_publish() is called with a mid not present in unacked_publish")
        print("This is due to an unavoidable race-condition:")
        print("* publish() return the mid of the message sent.")
        print("* mid from publish() is added to unacked_publish by the main thread")
        print("* on_publish() is called by the loop_start thread")
        print("While unlikely (because on_publish() will be called after a network round-trip),")
        print(" this is a race-condition that COULD happen")
        print("")
        print("The best solution to avoid race-condition is using the msg_info from publish()")
        print("We could also try using a list of acknowledged mid rather than removing from pending list,")
        print("but remember that mid could be re-used !")

## Start here

def send_task(task_name, client, this_client_id):
    # publish.single(topic="dynamicFL/cmd", payload=task_name, hostname=broker_name, client_id=server_id)
    print_log("publish to " + "dynamicFL/req/"+this_client_id)
    client.publish(topic="dynamicFL/req/"+this_client_id, payload=task_name)
def send_model(path, client, this_client_id):
    f = open(path, "rb")
    data = f.read()
    f.close()
    print_log(len(data))
    print_log("publish to " + "dynamicFL/model/"+this_client_id)
    client.publish(topic="dynamicFL/model/"+this_client_id, payload=data)

def on_connect(client, userdata, flags, rc):
    print_log("Connected with result code "+str(rc))

def on_message(client, userdata, msg):
    print(f"received msg from {msg.topic}")
    topic = msg.topic
    if topic == "dynamicFL/join":
        handle_join(client, userdata, msg)
    elif "dynamicFL/res" in topic:
        tmp = topic.split("/")
        this_client_id = tmp[2]
        handle_res(this_client_id, msg)

def handle_res(this_client_id, msg):
    # this_client_id = this_client_id.decode("utf-8")
    data = json.loads(msg.payload)
    cmd = data["task"]
    if cmd == "EVA_CONN":
        handle_pingres(this_client_id, msg)
    elif cmd == "TRAIN":
        handle_trainres(this_client_id, msg)
    elif cmd == "WRITE_MODEL":
        handle_update_writemodel(this_client_id, msg)

def handle_join(client, userdata, msg):
    this_client_id = msg.payload.decode("utf-8")
    print("joined from"+" "+this_client_id)
    client_dict[this_client_id] = {
        "state": "joined"
    }
    server.subscribe(topic="dynamicFL/res/"+this_client_id)
    # print(client_dict)
    

def handle_pingres(this_client_id, msg):
    print(msg.topic+" "+str(msg.payload.decode()))
    # ping_res = msg.payload.decode("utf-8")
    ping_res = json.loads(msg.payload)
    this_client_id = ping_res["client_id"]
    # print(ping_res["clientid"])
    if ping_res["packet_loss"] == 0.0:
        print_log(f"{this_client_id} is a good client")
        state = client_dict[this_client_id]["state"]
        print(state)
        if state == "joined" or state == "trained":
            client_dict[this_client_id]["state"] = "eva_conn_ok"
            send_model("saved_model/FashionMnist.pt", server, this_client_id)
        # time.sleep(10)
        # send_task("TRAIN", client)
        # start_time = threading.Timer(5, send_task, args=["EVA_CONN", client])
        # print_log("server wait for newround")
        # start_time.start()
    # print(client_dict)

def handle_trainres(this_client_id, msg):
    #print("trainres"+" "+str(msg.payload))
    print("Trainres")
    payload = json.loads(msg.payload.decode())
    
    client_trainres_dict[this_client_id] = payload["weight"]
    state = client_dict[this_client_id]["state"]
    if state == "model_recv":
        client_dict[this_client_id]["state"] = "trained"

    print("done train!")
    #print(client_trainres_dict)
    # print(client_dict)
    
def handle_update_writemodel(this_client_id, msg):
    state = client_dict[this_client_id]["state"]
    if state == "eva_conn_ok":
        client_dict[this_client_id]["state"] = "model_recv"
        send_task("TRAIN", server, this_client_id)
    # print(client_dict)

def start_round():
    global n_round
    n_round = n_round + 1
    print_log(f"server start round {n_round}")
    round_state = "started"
    for client_i in client_dict:
        send_task("EVA_CONN", server, client_i)
    t = threading.Timer(round_duration, end_round)
    t.start()

def do_aggregate():
    #aggregated_models(client_trainres_dict, )
    print("do_aggregate")

def handle_next_round_duration():
    if len(client_trainres_dict) < len(client_dict):
        time_between_two_round = time_between_two_round + 10

def end_round():
    global n_round
    print_log(f"server end round {n_round}")
    round_state = "finished"
    if n_round <= NUM_ROUND:
        handle_next_round_duration()
        do_aggregate()
        t = threading.Timer(time_between_two_round, start_round)
        t.start()
    else:
        for c in client_dict:
            send_task("STOP", server, c)

def on_subscribe(mosq, obj, mid, granted_qos):
    print_log("Subscribed: " + str(mid) + " " + str(granted_qos))