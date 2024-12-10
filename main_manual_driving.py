import holoocean, cv2, os, numpy as np
from pynput import keyboard

is_paused = False
was_paused = False
save = False

pressed_keys = set()

force = 100
latest_pixels = None

captured_images_directory = "captured_images"
stream_left_dir = 'stream/left'
stream_right_dir = 'stream/right'

image_count = 0
stream_count = 0

GT_positions = []
GT_poses = []

def on_press(key):
    global is_paused, save
    if hasattr(key, 'char'):
        if key.char == 'p':
            is_paused = True  # Pause the simulation when 'p' is pressed
        if key.char == 's' and is_paused: # Save the image when 's' is pressed and sim is paused
            save = True
        pressed_keys.add(key.char)

def on_release(key):
    if hasattr(key, 'char'):
        if key.char in pressed_keys:
            pressed_keys.remove(key.char)

#Setting up the keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys: # up
        command[0:4] += val
    if 'k' in keys: # down
        command[0:4] -= val
    if 'j' in keys: # turn left
        command[[4,7]] += val
        command[[5,6]] -= val
    if 'l' in keys: # turn right
        command[[4,7]] -= val
        command[[5,6]] += val

    if 'w' in keys: # forward
        command[4:8] += 10 * val
    if 's' in keys: # backward
        command[4:8] -= 10 * val
    if 'a' in keys: # move left
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys: # move right
        command[[4,6]] -= val
        command[[5,7]] += val

    return command

commands = []
tick_count = 0
sim_count = 0
#Using the prebuild OpenWater world with HoveringAUV as agent
with holoocean.make("OpenWater-HoveringCamera") as env:
    while True:
        if 'q' in pressed_keys:  #Exit the loop when 'q' is pressed
            break

        if not is_paused:
            command = parse_keys(pressed_keys, force)
            commands.append(command)
            env.act("auv0", command)
            state = env.tick()
            sim_count += 1
            
            if "LeftCamera" in state and "PoseSensor" in state:
                tick_count += 1

                if tick_count % 5 == 0:
                    
                    leftCam = state["LeftCamera"]

                    # Display live cameras
                    cv2.imshow("Left Camera Live Stream", leftCam)
                    cv2.waitKey(1)

                    # Save each frame from both cameras
                    # left_image_filename = os.path.join(stream_left_dir, f"left_{stream_count + 1}.jpg")
                    
                    #cv2.imwrite(left_image_filename, leftCam[:, :, :3])  # Save left camera image
                    #print('Saved image:', left_image_filename)
                    #stream_count += 1
                    
                    #Extract the GT pose every time we take an image
                    pose = state["PoseSensor"]
                    GT_poses.append(pose)
                    p = pose[:3, 3]
                    GT_positions.append(p)
                
        if is_paused:
            if not was_paused:  # Print "Simulation is paused" only once when paused
                print("Simulation is paused")
                was_paused = True  # Set the flag to True after printing the message

            if save and leftCam is not None:  # Save the latest captured image from left camera
                image_filename = os.path.join(captured_images_directory, f"image_{image_count + 1}.jpg")
                cv2.imwrite(image_filename, leftCam[:, :, 0:3])  # Save the image as a .jpg file
                print(f"Image {image_count + 1} saved to {image_filename}")
                image_count += 1  # Increment the image count

                save = False  # Reset the save flag

        if 'r' in pressed_keys:  # Resume the simulation when 'r' is pressed
            is_paused = False
            was_paused = False  # Reset the flag when resuming the simulation
            print("Simulation resumed")

listener.stop()
cv2.destroyAllWindows()

np.savetxt("commands/testplan3.txt", np.array(commands), fmt="%.6f") # CHANGE THIS WHEN YOU RUN A NEW TEST