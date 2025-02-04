import socket
import pickle


# Function to receive counting data over socket
def receive_counting_data(client_socket):
    try:
        while True:
            # Receive serialized counting data from server
            counting_data_bytes = client_socket.recv(4096)

            # Check if data is received
            if not counting_data_bytes:
                print("Connection closed by server")
                break

            # Deserialize counting data using pickle
            counting_data = pickle.loads(counting_data_bytes)

            # Process the counting data as needed
            print("Counting data:", counting_data)
    except Exception as e:
        print("Error receiving counting data:", e)


# Create a socket to connect to the server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    try:
        # Connect to the server (assumed running on localhost)
        client_socket.connect(('localhost', 12345))

        print("Connected to server")

        # Call the function to receive counting data
        receive_counting_data(client_socket)
    except Exception as e:
        print("Error connecting to server:", e)
