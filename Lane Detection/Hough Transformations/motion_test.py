import socket
import time

last_time = 0

def udp_sendCmd(message, esp32_ip="192.168.82.10", esp32_port=12345):
    """Sends a UDP message to the ESP32."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode(), (esp32_ip, esp32_port))
    sock.close()
    print(f"Sent: {message}")

while True:
    n = input('Enter the cmd(f, b, l, r): ')
    if time.time() - last_time > 1.5:
        udp_sendCmd(n)
        last_time = time.time()
    else:
        print(f'Last command was sent before {time.time() - last_time} seconds')
