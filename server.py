import paho.mqtt.client as mqtt
import json
import numpy as np

# Thiết lập thông tin kết nối MQTT
MQTT_BROKER = 'broker.emqx.io'
MQTT_PORT = 1883
TOPIC_REGISTER = "federated_learning/register"
TOPIC_MODEL_UPDATE = "federated_learning/model_update"

# Đăng ký thiết bị và mô hình cục bộ
device_info = {
    "device_id": "client_1",
    "model_type": "linear_regression",
}

local_model = np.array([1.0, 2.0])  # Mô hình đơn giản, ví dụ linear regression

# Thiết lập client MQTT
client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Đăng ký thiết bị khi kết nối thành công
    client.subscribe(TOPIC_MODEL_UPDATE)
    client.publish(TOPIC_REGISTER, json.dumps(device_info))

def on_message(client, userdata, msg):
    global local_model
    print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")
    
    # Xử lý thông điệp nhận được (cập nhật mô hình cục bộ)
    update_data = json.loads(msg.payload.decode())
    local_model += np.array(update_data["model_update"])


if __name__ == '__main__':

    client.on_connect = on_connect # sub vào topic model và gửi thông tin về đăng nhập
    client.on_message = on_message # nhận msg được gửi về từ model updates 

    # Kết nối đến MQTT broker
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    # Bắt đầu vòng lặp để duy trì kết nối
    client.loop_start()

    # Giả sử quá trình đào tạo cục bộ ở đây, sau đó gửi thông điệp cập nhật mô hình
    while True:
        # Thực hiện quá trình đào tạo cục bộ ở đây

        # Gửi thông điệp cập nhật mô hình lên MQTT broker
        update_data = {"device_id": device_info["device_id"], "model_update": local_model.tolist()}
        client.publish(TOPIC_MODEL_UPDATE, json.dumps(update_data))

    # Lưu ý: Bạn cũng cần thêm các phần xử lý lỗi và cơ chế chấp nhận bảo mật trong mã nguồn của bạn.
