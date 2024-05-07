import yaml
import argparse

# parser = argparse.ArgumentParser(description='Test Algorithms.')
# parser.add_argument('--yamlfile', default=None, type=str, help='Configuration file.')

# args = parser.parse_args()
path = "D:\\FIL 2024\\MQTT_FL\\Fl_mqtt_Config.yaml"
with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    # parse the default setting
server_config = config['Server']
client_config = config['Client']
