import signal
import time

# Define the handler for the timeout
def handler(signum, frame):
    print("Timeout!")
    raise Exception("end of time")

# Set the alarm for 5 seconds
signal.signal(signal.SIGALRM, handler)
signal.alarm(5)  # Set the timeout to 5 seconds

try:
    # Place your code here
    while True:
        time.sleep(1)
        print("Running...")
except Exception as e:
    print("Stopped: ", e)
finally:
    signal.alarm(0)  # Disable the alarm
