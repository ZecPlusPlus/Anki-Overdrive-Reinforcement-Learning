import logging
import threading
import time
from drive import Overdrive

lock = threading.Lock()


class CustomOverdrive(Overdrive):
    def __init__(self, addr):
        super().__init__(addr)
        self.last_piece_time = None
        self.last_piece = None
        self.piece_minimum_interval = 0.4  # Minimum time interval between pieces in seconds
        self.piece_timestamp = {}  # Dictionary to store the timestamp of each piece to avoid rapid triggers

    def locationChangeCallback(self, addr, location, piece, offset, speed, clockwise):
        current_time = time.time()
        # print("Location from " + addr + " : " + "Piece=" + str(piece) + " Location=" + str(location) + " Clockwise=" + str(clockwise) + " Speed=" + str(speed))
        # Check if the piece has changed and if the time since last trigger is greater than minimum interval
        with lock:
            if self.last_piece != piece and (piece not in self.piece_timestamp or current_time - self.piece_timestamp[piece] > self.piece_minimum_interval):
                if self.last_piece_time is not None:
                    piece_time = current_time - self.last_piece_time
                    #print(f"Time to reach piece {piece},{offset}: {piece_time:.2f} seconds.")
                self.last_piece_time = current_time
                self.last_piece = piece
                self.piece_timestamp[piece] = current_time  # Update the timestamp for the current piece


def execute_commands_from_file(car, filename):
    with open(filename, 'r') as file:
        commands = file.readlines()
    k=0
    for command in commands:
        k += 1
        print(k,"Green Car")
        command = command.strip().lower()
        if command == "end":
            car.disconnect()
            print("Disconnected from the car.")
            break
        elif command.startswith("speed"):
            try:
                _, speed = command.split()
                speed = int(speed)
                accel = 700  # Fixed acceleration value
                car.changeSpeed(speed, accel)
                print(f"Speed changed to {speed} with acceleration {accel}.")
            except ValueError:
                print("Invalid command format. Use: speed <speed>")
        elif command == "right":
            car.changeLaneRight(500, 200)
            print("Changed lane to the right.")
        elif command == "left":
            car.changeLaneLeft(500, 200)
            print("Changed lane to the left.")
        elif command == "uturn":
            car.doUturn()  # Execute U-turn
            print("Performed a U-turn.")
        elif command == "stop":
            car.changeSpeed(0, 0)
            print("Car stopped.")
        elif command.startswith("offset"):
            try:
                _, offset = command.split()
                speed = 500
                accel = 500
                offset = float(offset)
                car.changeLane(speed, accel, offset)
                print(f"Lane changed with speed {speed}, acceleration {accel}, and offset {offset}.")
            except ValueError:
                print("Invalid command format. Use: offset <speed> <accel> <offset>")
        else:
            print("Unknown command.")
        time.sleep(15)


def main():
    addr = "DC:7E:B8:5F:BF:46"  # CB:76:55:B9:54:67, CF:45:33:60:24:69 Mac-Adress of your AnkiCar -> sudo bluetoothctl -> Scan on -> Look for Anki car(Normally smth with with Anki... Drive 0')
    car = CustomOverdrive(addr)

    # Register the callback
    car.setLocationChangeCallback(car.locationChangeCallback)

    # Start the car with an initial speed and acceleration
    initial_speed = 250
    initial_accel = 200
    car.changeSpeed(initial_speed, initial_accel)
    #äprint(f"Car started with speed {initial_speed} and acceleration {initial_accel}.")

    command_file = "./commands.txt"  # Path to the command file

    # Start the command execution from the file in a separate thread
    command_thread = threading.Thread(target=execute_commands_from_file, args=(car, command_file))
    command_thread.start()

    # Main loop to keep the program running
    try:
        while command_thread.is_alive():
            command_thread.join(1)
    except KeyboardInterrupt:
        car.disconnect()
        print("Disconnected from the car due to keyboard interrupt.")


if __name__ == "__main__":
    main()
