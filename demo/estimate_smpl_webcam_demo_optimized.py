import os
import os.path as osp
import shutil
import time
import warnings
from argparse import ArgumentParser
from pathlib import Path
import socket
import pickle
import cv2
import json
import mmcv
import numpy as np
import torch
from json import JSONEncoder

from mmhuman3d.apis import (
    feature_extract,
    inference_image_based_model,
    inference_video_based_model,
    init_model,
)
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_hmr
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.data.data_converters.humman import HuMManConverter
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.utils.demo_utils import (
    extract_feature_sequence,
    get_speed_up_interval,
    prepare_frames,
    process_mmdet_results,
    process_mmtracking_results,
    smooth_process,
    speed_up_interpolate,
    speed_up_process,
)
from mmhuman3d.utils.ffmpeg_utils import array_to_images
from mmhuman3d.utils.transforms import rotmat_to_aa

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    from mmtrack.apis import inference_mot
    from mmtrack.apis import init_model as init_tracking_model
    has_mmtrack = True
except (ImportError, ModuleNotFoundError):
    has_mmtrack = False
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_tracking_result(args, frames, tracking_model):
    max_track_id = 0
    max_instance = 0
    result_list = []
    frame_id_list = []

    def process_frame(i, frame):
        mmtracking_results = inference_mot(tracking_model, frame, frame_id=i)
        result, track_id, instance_num = process_mmtracking_results(mmtracking_results, max_track_id)
        return i, result, track_id, instance_num

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, i, frame) for i, frame in enumerate(frames)]
        
        for future in as_completed(futures):
            i, result, track_id, instance_num = future.result()
            result_list.append(result)
            frame_id_list.append(i)
            if instance_num > max_instance:
                max_instance = instance_num
            if track_id > max_track_id:
                max_track_id = track_id

    return result_list, frame_id_list, max_instance, max_track_id


def multi_person_with_mmtracking(args, frames_iter, mesh_model, extractor, source_index, tracking_model):
    result_list, frame_id_list, max_instance, max_track_id = get_tracking_result(args, frames_iter, tracking_model)

    frame_num = len(frame_id_list)
    verts = np.zeros([frame_num, max_track_id + 1, 6890, 3])
    pred_cams = np.zeros([frame_num, max_track_id + 1, 3])
    bboxes_xyxy = np.zeros([frame_num, max_track_id + 1, 5])
    smpl_poses = np.zeros([frame_num, max_track_id + 1, 24, 3, 3])
    smpl_betas = np.zeros([frame_num, max_track_id + 1, 10])

    if args.speed_up_type:
        speed_up_interval = get_speed_up_interval(args.speed_up_type)
        speed_up_frames = (frame_num - 1) // speed_up_interval * speed_up_interval

    track_ids_lists = []

    # Optimize by processing frames in batches
    def process_batch(frame_ids, results):
        batch_results = []
        for i, (frame_id, result) in enumerate(zip(frame_ids, results)):
            if mesh_model.cfg.model.type == 'VideoBodyModelEstimator':
                if args.speed_up_type:
                    warnings.warn(
                        'Video based models do not support speed up. '
                        'By default we will inference with original speed.',
                        UserWarning)
                feature_results_seq = extract_feature_sequence(
                    result_list, frame_idx=i, causal=True, seq_len=16, step=1)
                mesh_results = inference_video_based_model(
                    mesh_model,
                    extracted_results=feature_results_seq,
                    with_track_id=True)
            elif mesh_model.cfg.model.type == 'ImageBodyModelEstimator':
                if args.speed_up_type and i % speed_up_interval != 0 and i <= speed_up_frames:
                    mesh_results = [{
                        'bbox': np.zeros((5)),
                        'camera': np.zeros((3)),
                        'smpl_pose': np.zeros((24, 3, 3)),
                        'smpl_beta': np.zeros((10)),
                        'vertices': np.zeros((6890, 3)),
                        'keypoints_3d': np.zeros((17, 3)),
                        'track_id': r['track_id']
                    } for r in result]
                else:
                    mesh_results = inference_image_based_model(
                        mesh_model,
                        frames_iter[frame_id],
                        result,
                        bbox_thr=args.bbox_thr,
                        format='xyxy')
            else:
                raise Exception(f'{mesh_model.cfg.model.type} is not supported yet')
            batch_results.append((i, mesh_results))
        return batch_results

    with ThreadPoolExecutor() as executor:
        batch_size = 5
        futures = [executor.submit(process_batch, frame_id_list[i:i + batch_size], result_list[i:i + batch_size])
                   for i in range(0, len(frame_id_list), batch_size)]
        
        for future in as_completed(futures):
            batch_results = future.result()
            for i, mesh_results in batch_results:
                track_ids = []
                for mesh_result in mesh_results:
                    instance_id = mesh_result['track_id']
                    bboxes_xyxy[i, instance_id] = mesh_result['bbox']
                    pred_cams[i, instance_id] = mesh_result['camera']
                    verts[i, instance_id] = mesh_result['vertices']
                    smpl_betas[i, instance_id] = mesh_result['smpl_beta']
                    smpl_poses[i, instance_id] = mesh_result['smpl_pose']
                    track_ids.append(instance_id)
                track_ids_lists.append(track_ids)

    del mesh_model
    del extractor
    torch.cuda.empty_cache()

    if args.speed_up_type:
        if smpl_poses.shape[0] > 1:
            smpl_poses = speed_up_process(torch.tensor(smpl_poses).to(args.device.lower()), args.speed_up_type)
            selected_frames = np.arange(0, len(frames_iter), speed_up_interval)
            smpl_poses, smpl_betas, pred_cams, bboxes_xyxy = speed_up_interpolate(
                selected_frames, speed_up_frames, smpl_poses, smpl_betas, pred_cams, bboxes_xyxy)
        else:
            print("Warning: Not enough frames for speed up processing.")


    if args.smooth_type is not None:
        smpl_poses = smooth_process(smpl_poses.reshape(frame_num, -1, 24, 9), smooth_type=args.smooth_type).reshape(frame_num, -1, 24, 3, 3)
        verts = smooth_process(verts, smooth_type=args.smooth_type)
        pred_cams = smooth_process(pred_cams[:, np.newaxis], smooth_type=args.smooth_type).reshape(frame_num, -1, 3)

    if smpl_poses.shape[2:] == (24, 3, 3):
        smpl_poses = rotmat_to_aa(smpl_poses)
    elif smpl_poses.shape[2:] == (24, 3):
        smpl_poses = smpl_poses
    else:
        raise Exception(f'Wrong shape of `smpl_pose`: {smpl_poses.shape}')

    if args.output is not None:
        body_pose_, global_orient_, smpl_betas_, verts_, pred_cams_, bboxes_xyxy_, image_path_, frame_id_, person_id_ = \
            [], [], [], [], [], [], [], [], []
        human_data = HumanData()
        frames_folder = osp.join(args.output, 'images')
        os.makedirs(frames_folder, exist_ok=True)
        array_to_images(np.array(frames_iter)[frame_id_list], output_folder=frames_folder)

        for i, img_i in enumerate(sorted(os.listdir(frames_folder))):
            if i < len(track_ids_lists):
                for person_i in track_ids_lists[i]:
                    body_pose_.append(smpl_poses[i][person_i][1:])
                    global_orient_.append(smpl_poses[i][person_i][:1])
                    smpl_betas_.append(smpl_betas[i][person_i])
                    verts_.append(verts[i][person_i])
                    pred_cams_.append(pred_cams[i][person_i])
                    bboxes_xyxy_.append(bboxes_xyxy[i][person_i])
                    image_path_.append(os.path.join('images', img_i))
                    person_id_.append(person_i)
                    frame_id_.append(frame_id_list[i])
            else:
                print(f"경고: 인덱스 {i}는 track_ids_lists의 범위를 벗어납니다.")

        smpl = {}
        smpl['body_pose'] = np.array(body_pose_).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(global_orient_).reshape((-1, 3))
        smpl['betas'] = np.array(smpl_betas_).reshape((-1, 10))
        human_data['smpl'] = smpl
        human_data['verts'] = verts_
        human_data['pred_cams'] = pred_cams_
        human_data['bboxes_xyxy'] = bboxes_xyxy_
        human_data['image_path'] = image_path_
        human_data['person_id'] = person_id_
        human_data['frame_id'] = frame_id_
        human_data.dump(osp.join(args.output, f'inference_result_multi{source_index}.npz'))

    


def main(args):

    # prepare input
    #frames_iter = prepare_frames(args.input_path)
    vid_cap = cv2.VideoCapture("rtsp://172.22.48.1:8554/webcam.h264")
    vid_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    if not vid_cap.isOpened():
        print("Failed to open webcam.")
        return
    
    source_index = 1
    mesh_model, extractor = \
        init_model(args.mesh_reg_config, args.mesh_reg_checkpoint,
                   device=args.device.lower())
    
    tracking_model = init_tracking_model(
        args.tracking_config, None, device=args.device.lower())
    
    start = time.time()
    while True:
        #frame args.fps만큼 모으기
        frames_iter=[]
        for _ in range(args.fps):
            ret, frame = vid_cap.read()
            frames_iter.append(frame)
            if ret:
                cv2.imshow("test", frame)
       
            # 프레임 처리 코드 추가 가능
        source_index += 1 
        
        

        if args.multi_person_demo:
            multi_person_with_mmtracking(args, frames_iter, mesh_model, extractor, source_index, tracking_model)
        else:
            raise ValueError('Only supports single_person_demo or multi_person_demo')
        
        end = time.time()
        print(end - start)

        start = end

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        'mesh_reg_config',
        type=str,
        default=None,
        help='Config file for mesh regression')
    parser.add_argument(
        'mesh_reg_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for mesh regression')
    parser.add_argument(
        '--single_person_demo',
        action='store_true',
        help='Single person demo with MMDetection')
    parser.add_argument('--det_config', help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        '--det_cat_id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--multi_person_demo',
        action='store_true',
        help='Multi person demo with MMTracking')
    parser.add_argument('--tracking_config', help='Config file for tracking')

    parser.add_argument(
        '--body_model_dir',
        type=str,
        default='data/body_models/',
        help='Body models file path')
    parser.add_argument(
        '--input_path', type=str, default=None, help='Input path')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='directory to save output result file')
    parser.add_argument(
        '--show_path',
        type=str,
        default=None,
        help='directory to save rendered images or video')
    parser.add_argument(
        '--render_choice',
        type=str,
        default='hq',
        help='Render choice parameters')
    parser.add_argument(
        '--palette', type=str, default='segmentation', help='Color theme')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.99,
        help='Bounding box score threshold')
    parser.add_argument(
        '--draw_bbox',
        action='store_true',
        help='Draw a bbox for each detected instance')
    parser.add_argument(
        '--smooth_type',
        type=str,
        default=None,
        help='Smooth the data through the specified type.'
        'Select in [oneeuro,gaus1d,savgol].')
    parser.add_argument(
        '--speed_up_type',
        type=str,
        default=None,
        help='Speed up data processing through the specified type.'
        'Select in [deciwatch].')
    parser.add_argument(
        '--focal_length', type=float, default=5000., help='Focal lenght')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda:0',
        help='device used for testing')
    parser.add_argument(
        '--fps',
        type=int,
        default=17,
        help='Set fps')
    args = parser.parse_args()

    if args.single_person_demo:
        assert has_mmdet, 'Please install mmdet to run the demo.'
        assert args.det_config is not None
        assert args.det_checkpoint is not None

    if args.multi_person_demo:
        assert has_mmtrack, 'Please install mmtrack to run the demo.'
        assert args.tracking_config is not None

    main(args)


