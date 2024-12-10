import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_poses(filepath):
    poses = []
    with open(filepath, 'r') as file:
        for line in file:
            flat_pose = np.fromstring(line.strip(), sep=' ')
            pose_matrix = flat_pose.reshape((4, 4))
            poses.append(pose_matrix)
    return poses # returns a list of NumPy 4x4 arrays.

def load_images(filepath):
    files = os.listdir(filepath)
    files.sort()
    all_image_paths = [os.path.join(filepath, i) for i in files]
    images_list_grayscale = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in all_image_paths]
    return images_list_grayscale

def transf_matrix(R, t):
    T = np.eye(4, dtype=np.float64) # Initializing the transfer matrix 4x4 with 1 on diagonal
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def get_matches(index, images, ratio_thresh, k = 1):
    orb = cv2.ORB_create(10000)

    # Compute keypoints and descriptors for previous and current images
    kp1, des1 = orb.detectAndCompute(images[index - k], None)
    kp2, des2 = orb.detectAndCompute(images[index], None)

    # Check if descriptors are valid
    if des1 is None or des2 is None:
        # Return empty arrays if descriptors are None
        return np.array([]), np.array([])

    # Binary descriptors (e.g., ORB), use LSH index parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,     # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)

    # Create FLANN-based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform KNN matching with k=2
    matches = flann.knnMatch(des1, des2, k = 2)

    # Lowe's ratio test to find out the bad matches
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return q1, q2

def get_pose(q1, q2, K):
    E, _ = cv2.findEssentialMat(q1, q2, K, method=cv2.RANSAC, prob=0.999, threshold = 1)
    _, R, t, _ = cv2.recoverPose(E, q1, q2, focal = 256, pp = (256, 256))

    return transf_matrix(R, np.squeeze(t))

K = np.array([[256,     0,      256],
              [0,       256,    256],
              [0,       0,      1]])

# Selecting test 0
GT_poses = load_poses("GT_poses/0.txt")
images = load_images("images/0")

GT_path = []
estimated_path = []
ratio_tresh = 1
scale_factor = 1

for i in range(len(images)):
    GT_pose = GT_poses[i]

    if i == 0:
        cur_pose = GT_pose
        GT_path.append((GT_pose[0, 3], GT_pose[1, 3], GT_pose[2, 3])) # update GT path
    else:
        q1, q2 = get_matches(i, images, ratio_tresh)

        transf = get_pose(q1, q2, K)

        transf[:3, 3] *= scale_factor # scale the estimated translation

        cur_pose = np.dot(cur_pose, transf) # update cur_pose
        GT_path.append((GT_poses[i][0, 3], GT_poses[i][1, 3], GT_poses[i][2, 3]))  # update GT path

    estimated_path.append((cur_pose[0, 3], cur_pose[1, 3], cur_pose[2, 3]))

# Separating the x, y, and z coordinates
est_x = [point[0] for point in estimated_path]
est_y = [point[1] for point in estimated_path]
est_z = [point[2] for point in estimated_path]

GT_x = [point[0] for point in GT_path]
GT_y = [point[1] for point in GT_path]
GT_z = [point[2] for point in GT_path]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the estimated and GT trajectory
ax.plot(est_x, est_y, est_z, label="Estimated Path", marker = 'o')  # Add markers to show points
ax.plot(GT_x, GT_y, GT_z, label="GT Path", marker = 'o')  # Add markers to show points

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

ax.legend()
plt.show()
