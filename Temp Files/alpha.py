import socket
import keyboard  # Import keyboard library

ip = "192.168.82.10"  # ESP32 Server IPwawawawawaawawdwaawdawds
udp_port = 8888
speed = 130

def send_udp_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(command.encode(), (ip, udp_port))  # Send command
        response, _ = sock.recvfrom(1024)  # Receive response
        return response.decode()

def control_loop():
    print("Use W/A/S/D for movement, SPACE to stop, and Q to quit.")

    while True:
        if keyboard.is_pressed('w'):
            response = send_udp_command(f"f:{speed}")
            print("Forward:", response)
        elif keyboard.is_pressed('s'):
            response = send_udp_command(f"s")
            print("Backward:", response)
        elif keyboard.is_pressed('a'):
            response = send_udp_command(f"l:{speed}")
            print("Left:", response)
        elif keyboard.is_pressed('d'):
            response = send_udp_command(f"r:{speed}")
            print("Right:", response)
        elif keyboard.is_pressed('space'):
            response = send_udp_command("b{speed}")
            print("Stop:", response)
        elif keyboard.is_pressed('q'):
            print("Exiting...")
            break

if __name__ == "__main__":
    control_loop()
