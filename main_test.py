import holoocean, cv2, os, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

latest_pixels = None

# Running test 0
image_dir = 'images/0'
commands_file = "commands/testplan0.txt"
commands = np.loadtxt(commands_file) 

image_count = 0
stream_count = 0

GT_positions = []
GT_poses = []

commands = []
tick_count = 0
sim_count = 0

#Using the prebuild OpenWater world with HoveringAUV as agent
with holoocean.make("OpenWater-HoveringCamera") as env:
    while True:

        if sim_count < len(commands):
            command = commands[sim_count]
        else:
            break

        env.act("auv0", command)
        state = env.tick()
        sim_count += 1
        
        if "LeftCamera" in state and "PoseSensor" in state:
            tick_count += 1

            if tick_count % 5 == 0:
                leftCam = state["LeftCamera"] # Left Camera
                depthSensor = state["DepthSensor"] # Measure depth
                GT_pose = state["PoseSensor"] # Extract te GT pose

                # Save each frame from both cameras
                image_filename = os.path.join(image_dir, f"{stream_count:06d}.png")
                
                cv2.imwrite(image_filename, leftCam[:, :, :3])  # Save left camera image
                print('Saved image:', image_filename)
                stream_count += 1
                
                GT_poses.append(GT_pose) # Save the GT poses
                GT_position = GT_pose[:3, 3]
                GT_positions.append(GT_position)

cv2.destroyAllWindows()

# Saving the GT poses for the given testplan 
np.savetxt("GT_poses/2.txt", [pose.flatten() for pose in GT_poses], fmt = "%.6f") # CHANGE THIS WHEN RUNNING A NEW TEST

# Plotting GT
GT_positions = np.array(GT_positions)
GT_x = GT_positions[:, 0]
GT_y = GT_positions[:, 1]
GT_z = GT_positions[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(GT_x, GT_y, GT_z, label = 'GT') # GT

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
plt.legend()
plt.show()