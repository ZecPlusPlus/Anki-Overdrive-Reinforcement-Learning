from drive import Overdrive
import threading
import time
# Variable to track the current piece
current_piece = None
# Create a lock for synchronizing access to current_piece
lock = threading.Lock()
last_piece_time = None

def time_to_next_piece(addr, location, offset, piece, speed, clockwise):
    global current_piece, last_piece_time
    current_time = time.perf_counter()
    
    with lock:
        if piece != current_piece:
            if current_piece is not None:
                time_taken = current_time - last_piece_time
                print(f"Time to reach piece {piece}: {time_taken:.6f} seconds.")
                
            print("Next piece reached!:", offset, piece, location, speed)
            current_piece = piece
            last_piece_time = current_time

    # Print out addr, piece ID, location ID of the vehicle, this print everytime when location changed
    # print("Location from " + addr + " : " + "Piece=" + str(piece) + " Location=" + str(location) + " Clockwise=" + str(clockwise))

car = Overdrive("CB:76:55:B9:54:67")
car.setLocationChangeCallback(time_to_next_piece)  
 # Switch to next right lane with speed = 1000, acceleration = 1000
input("Press Enter to exit\n")  # Hold the program so it won't end abruptly
