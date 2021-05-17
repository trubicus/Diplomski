import socket

port = 5555
buffer = 1024
#self_ip = "192.168.43.128"
self_ip = "192.168.0.137"
data_filter = [0, 0, 3]

def init_client(srv_port):
    client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    client_socket.bind((self_ip, srv_port))
    return client_socket

client = []

client.append(init_client(port))
#client.append(init_client(port + 1))
# client.append(init_client(port + 2))


print("UDP client up")
signal = []
while(True):
    sample = []
    for phone in client:
        bytes_adress_pair = phone.recvfrom(buffer)
        data = []
        message = bytes_adress_pair[0].decode().split(",")
        for i in message:
            data.append(i.strip())

        # filter koji mice podatke bez ziroskopa (prvih nekoliko mjerenja)
        if "4" not in data:
            continue

        for i in data_filter:
            data.pop(i)
        

        if len(data) != 0:
            sample.append(data[:6])
            
    if len(sample) != 0:
        signal.append(sample)

    print(signal)
        