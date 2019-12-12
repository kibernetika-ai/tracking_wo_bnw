import argparse
import logging
import os
import time
from os import path as osp

import motmetrics as mm
from torchvision.transforms import ToTensor

mm.lap.default_solver = 'lap'

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor import utils


LOG = logging.getLogger(__name__)


tracktor = {
    "name": "Tracktor++",
    "module_name": "MOT17",
    "desription": None,
    "seed": 12345,
    "network": "fpn",
    "interpolate": False,
    "write_images": False,
    "dataset": "mot17_train_FRCNN17",
    "frame_split": [
        0.0,
        1.0
    ],
    "tracker": {
        "detection_person_thresh": 0.1,
        "regression_person_thresh": 0.3,
        "detection_nms_thresh": 0.3,
        "regression_nms_thresh": 0.6,
        "motion_model": {
            "enabled": False,
            "n_steps": 1,
            "center_only": True
        },
        "public_detections": False,
        "max_features_num": 10,
        "do_align": False,
        "warp_mode": "cv2.MOTION_EUCLIDEAN",
        "number_of_iterations": 100,
        "termination_eps": 1e-05,
        "do_reid": True,
        "inactive_patience": 10,
        "reid_sim_threshold": 2.0,
        "reid_iou_threshold": 0.2
    }
}
# imagenet_stats
mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


colors = get_spaced_colors(100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source',
        required=True
    )
    parser.add_argument(
        '--detection-path',
        required=True
    )
    parser.add_argument(
        '--reid-path',
        required=True
    )
    parser.add_argument(
        '--output',
        required=True
    )
    parser.add_argument(
        '--every-frame',
        type=int,
        default=1,
    )

    return parser.parse_args()


def draw_boxes(frame, frame_id, results):
    # Result structure:
    # {
    #   track_id1: {
    #     frame_id1: [x1, y1, x2, y2],
    #     frame_id2: [...],
    #     ...
    #   },
    #   track_id2: { ... },
    #   ...
    # }
    # results = utils.interpolate(results)
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 1.2
    for track_id, track in results.items():
        if frame_id in track:
            box = np.array(track[frame_id]).astype(int)
            cv2.rectangle(
                frame,
                (box[0], box[1]),
                (box[2], box[3]),
                colors[track_id % len(colors)][::-1],
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                str(track_id),
                (box[0] + 10, box[1] + 40),
                font, size, (0, 0, 0),
                thickness=4,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                str(track_id),
                (box[0] + 10, box[1] + 40),
                font, size, (50, 50, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    return frame


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logging.root.setLevel(logging.INFO)

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        LOG.info('-' * 50)
        LOG.info('Enabling CUDA')
        LOG.info('-' * 50)

    device = torch.device('cuda' if is_cuda else 'cpu')
    # sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    if is_cuda:
        torch.cuda.manual_seed(tracktor['seed'])
        torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    LOG.info("Initializing object detector.")

    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect_state_dict = torch.load(args.detection_path, map_location=device)
    obj_detect.load_state_dict(obj_detect_state_dict)

    obj_detect.eval()
    if is_cuda:
        obj_detect.cuda()

    # LOG.info('Load detection model...')
    # obj_detect = load_object_detection_driver(args.detection_path)
    # LOG.info('Done.')

    # reid
    LOG.info("Initializing reidentification network.")
    reid_network = resnet50(pretrained=False, output_dim=128)
    reid_network.load_state_dict(torch.load(args.reid_path, map_location=device))
    reid_network.eval()
    if is_cuda:
        reid_network.cuda()

    # tracktor
    if 'oracle' in tracktor:
        tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    time_total = 0
    tracker.reset()

    start = time.time()
    vc = cv2.VideoCapture(args.source)

    frame_count = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    video_output = args.output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vc.get(cv2.CAP_PROP_FPS)
    each_frame = args.every_frame
    writer = cv2.VideoWriter(video_output, fourcc, fps / each_frame, frameSize=(width, height))

    LOG.info(f"Tracking: {args.source}")
    frame_id = 0
    frame_num = 0
    results = {}
    try:
        while True:
            frame_num += 1
            if frame_num % each_frame == 0:
                ret, frame = vc.read()
                if not ret:
                    break
            else:
                vc.grab()
                continue

            frame_id += 1
            if frame_id % 50 == 0:
                LOG.info(f'Processing frame {frame_id}')

            if frame_count * tracktor['frame_split'][0] <= frame_id <= frame_count * tracktor['frame_split'][1]:
                rgb_frame = frame[:, :, ::-1]

                torch_frame = F.to_tensor(rgb_frame.copy())
                torch_frame = torch_frame.expand([1, *torch_frame.shape])
                if is_cuda:
                    torch_frame = torch_frame.cuda()

                torch_blob = {
                    'img': torch_frame
                }
                tracker.step(torch_blob, frame)

                # __import__('ipdb').set_trace()
                results = tracker.results
                output = draw_boxes(frame, frame_id - 1, results=results)
                writer.write(output)
    except KeyboardInterrupt:
        LOG.info('Stopping.')

    writer.release()

    time_total += time.time() - start

    LOG.info(f"Tracks found: {len(results)}")
    LOG.info(f"Runtime for {args.source}: {time.time() - start :.1f} s.")

    if tracktor['interpolate']:
        results = utils.interpolate(results)

    # if tracktor['write_images']:
    #     utils.plot_sequence(results, seq, osp.join(output_dir, args.source))

    LOG.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
             f"{time_total:.1f} s ({frame_id / time_total:.1f} Hz)")
    # if mot_accums:
    #     evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt], generate_overall=True)


def norm(img):
    img = img.transpose([2, 0, 1])
    img = img / 255.0
    return (img - mean) / std


if __name__ == '__main__':
    main()
