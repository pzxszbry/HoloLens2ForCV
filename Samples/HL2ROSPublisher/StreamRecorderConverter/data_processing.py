from distutils import extension
import os
from turtle import width
import cv2
import numpy as np
import tarfile
from glob import glob
import multiprocessing
import open3d as o3d


from hand_defs import HandJointIndex

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

folders_extensions = [
    ("PV", "bytes"),
    ("Depth AHaT", "[0-9].pgm"),
    ("Depth Long Throw", "[0-9].pgm"),
]


def extract_tar_file(filename, outpath):
    with tarfile.open(filename) as tar:
        tar.extractall(outpath)


def check_framerates(capture_path):
    HundredsOfNsToMilliseconds = 1e-4
    MillisecondsToSeconds = 1e-3

    def get_avg_delta(timestamps):
        deltas = [
            (timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))
        ]
        return np.mean(deltas)

    for (img_folder, img_ext) in folders_extensions:
        base_folder = os.path.join(capture_path, img_folder)
        paths = sorted(glob(os.path.join(base_folder, f"*{img_ext}")))
        timestamps = [int(path.stem) for path in paths]
        if len(timestamps):
            avg_delta = get_avg_delta(timestamps) * HundredsOfNsToMilliseconds
            print(
                "Average {} delta: {:.3f}ms, fps: {:.3f}".format(
                    img_folder, avg_delta, 1 / (avg_delta * MillisecondsToSeconds)
                )
            )

    head_hat_stream_path = capture_path.glob("*eye.csv")
    try:
        head_hat_stream_path = next(head_hat_stream_path)
        timestamps = load_head_hand_eye_data(str(head_hat_stream_path))[0]
        hh_avg_delta = get_avg_delta(timestamps) * HundredsOfNsToMilliseconds
        print(
            "Average hand/head delta: {:.3f}ms, fps: {:.3f}".format(
                hh_avg_delta, 1 / (hh_avg_delta * MillisecondsToSeconds)
            )
        )
    except StopIteration:
        pass


def match_timestamp(target, all_timestamps):
    return np.argmin([abs(x - target) for x in all_timestamps])


def load_lut(lut_filename):
    with open(lut_filename, mode="rb") as f:
        lut = np.frombuffer(f.read(), dtype=np.float32)
        lut = np.reshape(lut, (-1, 3))
    return lut


def load_head_hand_eye_data(csv_path):
    joint_count = HandJointIndex.Count.value

    data = np.loadtxt(csv_path, delimiter=",")

    n_frames = len(data)
    timestamps = np.zeros(n_frames)
    head_transs = np.zeros((n_frames, 3))

    left_hand_transs = np.zeros((n_frames, joint_count, 3))
    left_hand_transs_available = np.ones(n_frames, dtype=bool)
    right_hand_transs = np.zeros((n_frames, joint_count, 3))
    right_hand_transs_available = np.ones(n_frames, dtype=bool)

    # origin (vector, homog) + direction (vector, homog) + distance (scalar)
    gaze_data = np.zeros((n_frames, 9))
    gaze_available = np.ones(n_frames, dtype=bool)

    for i_frame, frame in enumerate(data):
        timestamps[i_frame] = frame[0]
        # head
        head_transs[i_frame, :] = frame[1:17].reshape((4, 4))[:3, 3]
        # left hand
        left_hand_transs_available[i_frame] = frame[17] == 1
        left_start_id = 18
        for i_j in range(joint_count):
            j_start_id = left_start_id + 16 * i_j
            j_trans = frame[j_start_id : j_start_id + 16].reshape((4, 4))[:3, 3]
            left_hand_transs[i_frame, i_j, :] = j_trans
        # right hand
        right_hand_transs_available[i_frame] = (
            frame[left_start_id + joint_count * 4 * 4] == 1
        )
        right_start_id = left_start_id + joint_count * 4 * 4 + 1
        for i_j in range(joint_count):
            j_start_id = right_start_id + 16 * i_j
            j_trans = frame[j_start_id : j_start_id + 16].reshape((4, 4))[:3, 3]
            right_hand_transs[i_frame, i_j, :] = j_trans

        assert j_start_id + 16 == 851
        gaze_available[i_frame] = frame[851] == 1
        gaze_data[i_frame, :4] = frame[852:856]
        gaze_data[i_frame, 4:8] = frame[856:860]
        gaze_data[i_frame, 8] = frame[860]

    return (
        timestamps,
        head_transs,
        left_hand_transs,
        left_hand_transs_available,
        right_hand_transs,
        right_hand_transs_available,
        gaze_data,
        gaze_available,
    )


def project_on_pv(points, pv_img, pv2world_transform, focal_length, principal_point):
    height, width, _ = pv_img.shape

    homog_points = np.hstack((points, np.ones(len(points)).reshape((-1, 1))))
    world2pv_transform = np.linalg.inv(pv2world_transform)
    points_pv = (world2pv_transform @ homog_points.T).T[:, :3]

    intrinsic_matrix = np.array(
        [
            [focal_length[0], 0, width - principal_point[0]],
            [0, focal_length[1], principal_point[1]],
            [0, 0, 1],
        ]
    )
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    xy, _ = cv2.projectPoints(points_pv, rvec, tvec, intrinsic_matrix, None)
    xy = np.squeeze(xy)
    xy[:, 0] = width - xy[:, 0]
    xy = np.floor(xy).astype(int)

    rgb = np.zeros_like(points)
    width_check = np.logical_and(0 <= xy[:, 0], xy[:, 0] < width)
    height_check = np.logical_and(0 <= xy[:, 1], xy[:, 1] < height)
    valid_ids = np.where(np.logical_and(width_check, height_check))[0]

    z = points_pv[valid_ids, 2]
    xy = xy[valid_ids, :]

    depth_image = np.zeros((height, width))
    for i, p in enumerate(xy):
        depth_image[p[1], p[0]] = z[i]

    colors = pv_img[xy[:, 1], xy[:, 0], :]
    rgb[valid_ids, :] = colors[:, ::-1] / 255.0

    return rgb, depth_image


def project_on_depth(points, rgb, intrinsic_matrix, width, height):
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    xy, _ = cv2.projectPoints(points, rvec, tvec, intrinsic_matrix, None)
    xy = np.squeeze(xy)
    xy = np.around(xy).astype(int)

    width_check = np.logical_and(0 <= xy[:, 0], xy[:, 0] < width)
    height_check = np.logical_and(0 <= xy[:, 1], xy[:, 1] < height)
    valid_ids = np.where(np.logical_and(width_check, height_check))[0]
    xy = xy[valid_ids, :]

    z = points[valid_ids, 2]
    depth_image = np.zeros((height, width))
    image = np.zeros((height, width, 3))
    rgb = rgb[valid_ids, :]
    rgb = rgb[:, ::-1]
    for i, p in enumerate(xy):
        depth_image[p[1], p[0]] = z[i]
        image[p[1], p[0]] = rgb[i]

    image = image * 255.0

    return image, depth_image


def get_pv_width_and_height(pv_info_path):
    with open(pv_info_path, "r") as f:
        (_, _, width, height) = f.readline().strip().split(",")
    return (int(width), int(height))


def write_bytes_to_jpg(bytes_path, width, height, save_path=None):
    print(".", end="", flush=True)
    if save_path:
        file_name = os.path.split(bytes_path.replace("bytes", "jpg"))[-1]
        out_path = os.path.join(save_path, file_name)
    else:
        out_path = bytes_path.replace("bytes", "jpg")
    if os.path.exists(out_path):
        return
    with open(bytes_path, "rb") as f:
        img = np.frombuffer(f.read(), dtype=np.uint8)
    img = img.reshape((height, width, 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(out_path, img)


def convert_pgm_to_png(pgm_path, save_path=None):
    print(".", end="", flush=True)
    if save_path:
        file_name = os.path.split(pgm_path.replace("pgm", "png"))[-1]
        out_path = os.path.join(save_path, file_name)
    else:
        out_path = pgm_path.replace("pgm", "png")
    if os.path.exists(out_path):
        return
    img = cv2.imread(pgm_path, -1)
    cv2.imwrite(out_path, img)


def convert_images(sensor_name, record_path, depth_path_suffix=""):
    save_path = os.path.join(record_path, "processed_data", sensor_name)
    os.makedirs(save_path, exist_ok=True)
    if sensor_name == "PV":
        if os.path.exists(os.path.join(record_path, sensor_name)):
            pv_path = glob(os.path.join(record_path, "*_pv.txt"))[0]
            width, height = get_pv_width_and_height(pv_path)
            pv_imgs = sorted(glob(os.path.join(record_path, sensor_name, "*.bytes")))
            print(f"  * Processing {sensor_name} images")
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            for pv_img in pv_imgs:
                p.apply_async(write_bytes_to_jpg, (pv_img, width, height, save_path))
            p.close()
            p.join()
            print()

    if sensor_name in ["Depth Long Throw", "Depth AHaT"]:
        if os.path.exists(os.path.join(record_path, sensor_name)):
            depth_imgs = sorted(
                glob(
                    os.path.join(
                        record_path, sensor_name, f"*[0-9]{depth_path_suffix}.pgm"
                    )
                )
            )
            if len(depth_imgs) > 0:
                print(f"  * Processing {sensor_name} images")
                for depth_img in depth_imgs:
                    convert_pgm_to_png(depth_img, save_path)
                print()


######################
def extract_timestamp(path):
    timesstamp = os.path.split(path)[-1][:-4]
    return int(timesstamp)


def load_extrinsics(extrinsics_path):
    mtx = np.loadtxt(extrinsics_path, delimiter=",").reshape((4, 4)).astype(np.float32)
    return mtx


def load_rig2world_transforms(path):
    transforms = {}
    with open(path, "r") as f:
        for line in f.readlines():
            words = line.strip().split(",")
            timestamp = int(words[0])
            transform = np.array(words[1:]).reshape((4, 4)).astype(np.float32)
            transforms[timestamp] = transform
    return transforms


def load_pv_data(pv_info_path):
    with open(pv_info_path) as f:
        lines = f.readlines()

    pv_data = {}

    # The first line contains info about the intrinsics.
    # The following lines (one per frame) contain timestamp, focal length and transform PVtoWorld
    n_frames = len(lines) - 1
    pv_data["total_frames"] = n_frames
    frame_timestamps = np.zeros(n_frames, dtype=np.longlong)
    focal_lengths = np.zeros((n_frames, 2))
    pv2world_transforms = np.zeros((n_frames, 4, 4))

    ppx, ppy, width, height = lines[0].strip().split(",")
    pv_data["width"] = int(width)
    pv_data["height"] = int(height)
    pv_data["ppx"] = float(ppx)
    pv_data["ppy"] = float(ppy)

    for frame_idx, frame in enumerate(lines[1:]):
        # Row format is
        # timestamp, focal length (2), transform PVtoWorld (4x4)
        words = frame.strip().split(",")
        frame_timestamps[frame_idx] = int(words[0])
        focal_lengths[frame_idx, 0] = float(words[1])
        focal_lengths[frame_idx, 1] = float(words[2])
        pv2world_transforms[frame_idx] = (
            np.array(words[3:20]).astype(np.float32).reshape((4, 4))
        )

    pv_data["timestamps"] = frame_timestamps
    pv_data["focal_lengths"] = focal_lengths
    pv_data["pv2world_transforms"] = pv2world_transforms

    return pv_data


def transform_cam_point_to_world(points, rig2cam, rig2world):
    cam2world_transform = np.matmul(rig2world, np.linalg.inv(rig2cam))
    homog_points = np.hstack((points, np.ones((points.shape[0], 1))))
    world_points = np.matmul(cam2world_transform, homog_points.T)
    return world_points.T[:, :3], cam2world_transform


def get_points_in_cam_space(img, lut):
    img = np.tile(img.flatten().reshape((-1, 1)), (1, 3))
    points = img * lut
    remove_ids = np.where(np.sqrt(np.sum(points**2, axis=1)) < 1e-6)[0]
    points = np.delete(points, remove_ids, axis=0)
    points /= 1000.0
    return points


def save_ply(save_path, points, rgb=None, cam2world_transform=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.estimate_normals()
    if cam2world_transform is not None:
        # Camera center
        camera_center = (cam2world_transform) @ np.array([0, 0, 0, 1])
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(
            pcd, camera_center[:3]
        )

    o3d.io.write_point_cloud(save_path, pcd)


def save_single_point_cloud(
    shared_dict,
    depth_image,
    record_path,
    save_path,
    rig2cam,
    rig2world_transforms,
    # cam2world,
    lut,
    has_pv,
    pv_info,
    save_in_cam_space=False,
    discard_no_rgb=False,
):
    suffix = "_cam" if save_in_cam_space else ""
    save_file_name = os.path.split(depth_image)[-1][:-4] + f"{suffix}.ply"
    out_path = os.path.join(save_path, save_file_name)

    print(".", end="", flush=True)

    # extract the timestamp for this frame
    timestamp = extract_timestamp(depth_image)
    img = cv2.imread(depth_image, -1)
    height, width = img.shape
    assert len(lut) == width * height

    # get xyz points in camera space
    points = get_points_in_cam_space(img, lut)

    if save_in_cam_space:
        save_ply(out_path, points, rgb=None)
    else:
        if rig2world_transforms and (timestamp in rig2world_transforms):
            # if we have the transform from rig to world for this frame,
            # then put the point clouds in world space
            rig2world = rig2world_transforms[timestamp]
            # print('Transform found for timestamp %s' % timestamp)
            xyz, cam2world_transform = transform_cam_point_to_world(
                points, rig2cam, rig2world
            )

            rgb = None
            if has_pv:
                # if we have pv, get vertex colors
                # get the pv frame which is closest in time
                pv_timestamps = pv_info["timestamps"]
                pv2world_transforms = pv_info["pv2world_transforms"]
                focal_lengths = pv_info["focal_lengths"]
                principal_point = np.array([pv_info["ppx"], pv_info["ppy"]]).astype(
                    np.float32
                )
                target_id = match_timestamp(timestamp, pv_timestamps)
                pv_ts = pv_timestamps[target_id]
                rgb_path = os.path.join(record_path, f"processed_data/PV/{pv_ts}.jpg")
                assert os.path.exists(rgb_path)
                pv_img = cv2.imread(rgb_path)

                # Project from depth to pv going via world space
                rgb, depth = project_on_pv(
                    xyz,
                    pv_img,
                    pv2world_transforms[target_id],
                    focal_lengths[target_id],
                    principal_point,
                )

                # Project depth on virtual pinhole camera and save corresponding
                # rgb image inside <workspace>/pinhole_projection folder
                # if not disable_project_pinhole:
                #     # Create virtual pinhole camera
                #     scale = 1
                #     width = 320 * scale
                #     height = 288 * scale
                #     focal_length = 200 * scale
                #     intrinsic_matrix = np.array(
                #         [
                #             [focal_length, 0, width / 2.0],
                #             [0, focal_length, height / 2.0],
                #             [0, 0, 1.0],
                #         ]
                #     )
                #     rgb_proj, depth = project_on_depth(
                #         points, rgb, intrinsic_matrix, width, height
                #     )

                #     # Save depth image
                #     depth_proj_folder = pinhole_folder / "depth" / f"{pv_ts}.png"
                #     depth_proj_path = str(depth_proj_folder)[:-4] + f"{suffix}_proj.png"
                #     depth = (depth * DEPTH_SCALING_FACTOR).astype(np.uint16)
                #     cv2.imwrite(depth_proj_path, (depth).astype(np.uint16))

                #     # Save rgb image
                #     rgb_proj_folder = pinhole_folder / "rgb" / f"{pv_ts}.png"
                #     rgb_proj_path = str(rgb_proj_folder)[:-4] + f"{suffix}_proj.png"
                #     cv2.imwrite(rgb_proj_path, rgb_proj)

                #     # Save virtual pinhole information inside calibration.txt
                #     intrinsic_path = pinhole_folder / Path("calibration.txt")
                #     intrinsic_list = [
                #         intrinsic_matrix[0, 0],
                #         intrinsic_matrix[1, 1],
                #         intrinsic_matrix[0, 2],
                #         intrinsic_matrix[1, 2],
                #     ]
                #     with open(str(intrinsic_path), "w") as p:
                #         p.write(
                #             f"{intrinsic_list[0]} \
                #                 {intrinsic_list[1]} \
                #                 {intrinsic_list[2]} \
                #                 {intrinsic_list[3]} \n"
                #         )

                #     # Create rgb and depth paths
                #     rgb_parts = Path(rgb_proj_path).parts[2:]
                #     rgb_tmp = Path(rgb_parts[-2]) / Path(rgb_parts[-1])
                #     depth_parts = Path(depth_proj_path).parts[2:]
                #     depth_tmp = Path(depth_parts[-2]) / Path(depth_parts[-1])

                #     # Compute camera center
                #     camera_center = cam2world_transform @ np.array([0, 0, 0, 1])

                #     # Save depth, rgb, camera center, extrinsics inside shared dictionary
                #     shared_dict[path.stem] = [
                #         depth_tmp,
                #         rgb_tmp,
                #         camera_center[:3],
                #         cam2world_transform,
                #     ]

            if discard_no_rgb:
                colored_points = rgb[:, 0] > 0
                xyz = xyz[colored_points]
                rgb = rgb[colored_points]
            save_ply(out_path, xyz, rgb, cam2world_transform)
            # print('Saved %s' % output_path)
        else:
            print("Transform not found for timestamp %s" % timestamp)


def save_point_clouds(
    record_path,
    sensor_name,
    has_pv=False,
    save_in_cam_space=False,
    discard_no_rgb=False,
):
    print("  * Saving point clouds")
    save_path = os.path.join(record_path, "processed_data", "point_clouds", sensor_name)
    os.makedirs(save_path, exist_ok=True)

    calib = f"{sensor_name}_lut.bin"
    extrinsics = f"{sensor_name}_extrinsics.txt".format(sensor_name)
    rig2world = f"{sensor_name}_rig2world.txt"
    bin_path = os.path.join(record_path, calib)
    rig2cam_path = os.path.join(record_path, extrinsics)
    rig2world_path = os.path.join(record_path, rig2world)

    if has_pv:
        pv_info_path = glob(os.path.join(record_path, "*_pv.txt"))[0]
        pv_info = load_pv_data(pv_info_path)

    # lookup table to extract xyz from depth
    lut = load_lut(bin_path)
    # from camera to rig space transformation (fixed value)
    rig2cam = load_extrinsics(rig2cam_path)
    # from rig to world transformations (one per frame)
    rig2world_transforms = load_rig2world_transforms(rig2world_path)

    depth_imgs = sorted(
        glob(os.path.join(record_path, "processed_data", sensor_name, "*.png"))
    )

    # Create shared dictionary to save odometry and file list
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    multiprocess_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for depth_img in depth_imgs:
        multiprocess_pool.apply_async(
            save_single_point_cloud(
                shared_dict,
                depth_img,
                record_path,
                save_path,
                rig2cam,
                rig2world_transforms,
                lut,
                has_pv,
                pv_info,
                save_in_cam_space,
                discard_no_rgb,
            )
        )
    multiprocess_pool.close()
    multiprocess_pool.join()


def process_all(record_path, project_hand_eye=False):
    print("#" * 80)
    print(f"==> Processing recording folder {os.path.split(record_path)[-1]}...")

    has_pv = False
    sensor_names = []
    for tar_fname in sorted(glob(os.path.join(record_path, "*.tar"))):
        sensor_name = os.path.split(tar_fname)[-1][:-4]
        sensor_names.append(sensor_name)
        print(f"  * Extracting {sensor_name}")
        tar_output = os.path.join(record_path, sensor_name)
        os.makedirs(tar_output, exist_ok=True)
        extract_tar_file(tar_fname, tar_output)
        if sensor_name == "PV" and len(glob(os.path.join(record_path, "*_pv.txt"))) > 0:
            # pv_info_path = glob(os.path.join(record_path, "*_pv.txt"))[0]
            has_pv = True
        convert_images(sensor_name, record_path)
    for sensor_name in sensor_names:
        if sensor_name in ["Depth Long Throw", "Depth AHaT"]:
            save_point_clouds(
                record_path,
                sensor_name,
                has_pv,
                save_in_cam_space=False,
                discard_no_rgb=False,
            )


if __name__ == "__main__":
    recording_folder = os.path.join(CURR_DIR, "./Recordings/2022-08-02-101904")
    # recording_folder = os.path.join(CURR_DIR, "./Recordings/2022-08-02-105257")

    process_all(recording_folder)
