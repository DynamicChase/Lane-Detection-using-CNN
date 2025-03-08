import serial
import time
import re

# Set up the serial connection (change port accordingly)
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Updated for Ubuntu

total_distance = 0  # Total distance traveled in meters
previous_time = time.time()

def extract_speed(data):
    print("data",data)
    match = re.search(r'KMPH\s*=\s*([0-9]+(?:\.[0-9]+)?)', data)    
    
    # print(match)
    if match:
        return float(match.group(1))
    return 0.0

try:
    while True:
        line = ser.readline().decode('utf-8').strip()  # Read and decode serial data
        if line:
            speed_kmph = extract_speed(line)  # Extract speed in km/h
            # if(speed_kmph < 5):
            #     speed_kmph=0
            speed_mps = speed_kmph   # Convert km/h to m/s
            
            
            current_time = time.time()
            time_elapsed = current_time - previous_time  # Time interval in seconds
            previous_time = current_time
            
            distance = speed_mps * time_elapsed  # Distance = Speed * Time
            total_distance += distance  # Accumulate distance
            
            print(f"Speed: {speed_mps:.2f} m/s, Distance Traveled: {total_distance:.2f} m")
            
        time.sleep(0.1)  # Wait 500ms before next reading

except KeyboardInterrupt:
    print("Stopped by user")
    ser.close()
