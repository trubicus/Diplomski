import socket
import matplotlib.pyplot as plt
import numpy as np


port = 5555
buffer = 1024
#self_ip = "192.168.43.128"
self_ip = "192.168.0.137"
data_filter = [0, 0, 3]

plt.style.use('ggplot')

def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.0001):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1


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

# Plot
size = 100
x_vec = np.linspace(0, 1, size+1)[0:-1]
y_vec = np.linspace(-5, 5, size+1)[0:-1]
line1=[]

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

    if len(sample) == 0:
        continue

    y_vec[-1] = float(sample[0][3])
    line1 = live_plotter(x_vec, y_vec, line1)
    y_vec = np.append(y_vec[1:], 0.0)

    print(signal)

        