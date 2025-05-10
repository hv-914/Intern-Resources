import socket

ip = "192.168.82.10"  # ESP32 Server IP
udp_port = 8888 

speed = 150
def send_udp_command(command):
    
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(command.encode(), (ip, udp_port))  # Send command
        response, _ = sock.recvfrom(1024)  # Receive response
        return response.decode()

# Example usage
if __name__ == "__main__":

    while True:
        cmd = input("Enter command (f/b/l/r/s or q to quit): ")
        if cmd == 'q':
            print("Exiting...")
            break
        
        cmd = f"{cmd}:{speed}"
        response = send_udp_command(cmd)
        print(f"ESP32: {response}")
