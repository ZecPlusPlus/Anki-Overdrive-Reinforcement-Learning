import random
import time
import logging
from drive import Overdrive
logging.basicConfig(level=logging.DEBUG)

class RandomDriver:
    def __init__(self, car):
        self.car = car
        self.car.changeSpeed(500, 500)
        self.running = False

    def start(self):
        
        self.running = True
        while self.running:
            command = random.choice(['change_speed', 'change_lane_right', 'change_lane_left', 'do_uturn'])
            if command == 'change_speed':
                speed = random.randint(800, 2000)
                accel = random.randint(800, 2000)
                logging.info(f"Changing speed to {speed} with acceleration {accel}")
                self.car.changeSpeed(speed, accel)
            elif command == 'change_lane_right':
                speed = random.randint(0, 1000)
                accel = random.randint(0, 1000)
                logging.info(f"Changing lane to the right with speed {speed} and acceleration {accel}")
                self.car.changeLaneRight(speed, accel)
            elif command == 'change_lane_left':
                speed = random.randint(0, 1000)
                accel = random.randint(0, 1000)
                logging.info(f"Changing lane to the left with speed {speed} and acceleration {accel}")
                self.car.changeLaneLeft(speed, accel)
            time.sleep(random.uniform(2, 5.0))  # Random sleep between 0.5 and 2 seconds

    def stop(self):
        self.running = False


if __name__ == "__main__":
    car = Overdrive("CB:76:55:B9:54:67")  # Replace with your car's MAC address Other Car: CB:76:55:B9:54:67
    driver = RandomDriver(car)
    try:
        driver.start()
    except KeyboardInterrupt:
        driver.stop()
