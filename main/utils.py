import os
import time
from .color import color
from pythonping import ping

def find_color(color_):
    if color_ == "red":
        return color.RED
    elif color_ == "yellow":
        return color.YELLOW
    elif color_ == "green":
        return color.GREEN
    elif color_ == "cyan":
        return color.CYAN
    elif color_ == "purple":
        return color.PURPLE
    elif color_ == "blue":
        return color.BLUE
    else:
        return ""

def cur_time_str():
    cur_time = time.localtime()
    cur_time_string = "[" + time.strftime("%H:%M:%S", cur_time) + "]"
    return cur_time_string

def print_log(line, color_ = "", show_time = True):
    if type(line) == str:
        color_str = find_color(color_)
        if show_time == True:
            print(cur_time_str(), end=" ")
        else:
            line = "           " + line
        print(color_str + line + color.END)
    else:
        print(line)

def int_to_ubyte(num):
    return num.to_bytes(1, "big", signed = False)

def int_to_Nubyte(num, N):
    return num.to_bytes(N, "big", signed = False)
    
def choose_file_in_folder_by_order(folder, file_order):
    dir_name = folder
    # Get list of all files in a given directory sorted by name
    list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(dir_name, x)),
                        os.listdir(dir_name) ) )
    return list_of_files[file_order]

def ping_host(host):
    ping_result = ping(target=host, count=10, timeout=2)
    return {
        'host': host,
        'avg_latency': ping_result.rtt_avg_ms,
        'min_latency': ping_result.rtt_min_ms,
        'max_latency': ping_result.rtt_max_ms,
        'packet_loss': ping_result.packet_loss
    }

