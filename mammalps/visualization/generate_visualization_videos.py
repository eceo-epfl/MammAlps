"""
Generates annotated videos with bounding boxes, labels, and other metadata for visualization purposes.

Usage:
    python generate_visualization_videos.py <input_video_folder> <input_json_folder> <output_video_folder> [--color_by <type>] [--skip_existing]

Arguments:
    input_video_folder: Path to the folder containing input videos.
    input_json_folder: Path to the folder containing JSON dense annotations.
    output_video_folder: Path to the folder where annotated videos will be saved.
    --color_by: Specifies the annotation type (action, action2, activity, individual). Default is "action".
    --skip_existing: Skips processing if the output video already exists.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm.auto import tqdm

activity_classes = {
    "foraging": 0,
    "camera_reaction": 1,
    "vigilance": 2,
    "courtship": 3,
    "unknown": 4,
    "escaping": 5,
    "grooming": 6,
    "marking": 7,
    "resting": 8,
    "chasing": 9,
    "playing": 10,
    "none": 11,
}

action_classes = {
    "grazing": 0,
    "standing_head_down": 1,
    "drinking": 2,
    "sniffing": 3,
    "walking": 4,
    "running": 5,
    "jumping": 6,
    "standing_head_up": 7,
    "looking_at_camera": 8,
    "shaking_fur": 9,
    "scratching_body": 10,
    "scratching_hoof": 11,
    "scratching_antlers": 12,
    "bathing": 13,
    "laying": 14,
    "vocalizing": 15,
    "defecating": 16,
    "urinating": 17,
    "unknown": 18,
    "fighting": 19,
    "none": 20,
    "rutting": 21,
}


def supervision_from_json(frame_detection_results, color_by="action") -> sv.Detections:
    """
    Converts JSON detection results for a single frame into a `sv.Detections` object.

    Args:
        frame_detection_results (dict): Detection results for a single frame in JSON format.
        color_by (str): Specifies the attribute to use for coloring the bounding boxes
                        (e.g., "action", "action2", "activity", "individual").

    Returns:
        sv.Detections: A `Detections` object containing bounding box coordinates, tracker IDs,
                       class IDs, and additional metadata (e.g., species, activity, actions).
    """
    xyxy_coord = np.array(
        [detection["bbox"] for detection in frame_detection_results["detections"]]
    )
    track_id = np.array(
        [
            int(detection["track_id"]) if "track_id" in detection.keys() else 0
            for detection in frame_detection_results["detections"]
        ]
    )
    if color_by == "action":
        class_id = np.array(
            [
                (
                    int(action_classes[detection["attributes"]["action"]])
                    if "attributes" in detection.keys()
                    else 999
                )
                for detection in frame_detection_results["detections"]
            ]
        )
    elif color_by == "action2":
        class_id = np.array(
            [
                (
                    int(action_classes[detection["attributes"]["action2"]])
                    if "attributes" in detection.keys()
                    else 999
                )
                for detection in frame_detection_results["detections"]
            ]
        )
    elif color_by == "activity":
        class_id = np.array(
            [
                (
                    int(action_classes[detection["attributes"]["activity"]])
                    if "attributes" in detection.keys()
                    else 999
                )
                for detection in frame_detection_results["detections"]
            ]
        )
    elif color_by == "individual":
        class_id == track_id
    activity = np.array(
        [
            (
                detection["attributes"]["activity"]
                if "attributes" in detection.keys()
                else "none"
            )
            for detection in frame_detection_results["detections"]
        ]
    )
    action_1 = np.array(
        [
            (
                detection["attributes"]["action"]
                if "attributes" in detection.keys()
                else "none"
            )
            for detection in frame_detection_results["detections"]
        ]
    )
    action_2 = np.array(
        [
            (
                detection["attributes"]["action2"]
                if (
                    "attributes" in detection.keys()
                    and "action2" in detection["attributes"]
                )
                else "none"
            )
            for detection in frame_detection_results["detections"]
        ]
    )
    species = np.array(
        [
            (
                detection["attributes"]["species"]
                if "attributes" in detection.keys()
                else "none"
            )
            for detection in frame_detection_results["detections"]
        ]
    )

    # Sort by track id for visualization
    idx = np.argsort(track_id)
    track_id = track_id[idx]
    class_id = class_id[idx]
    xyxy_coord = xyxy_coord[idx, :]
    data = {
        "species": species[idx].tolist(),
        "activity": activity[idx].tolist(),
        "action_1": action_1[idx].tolist(),
        "action_2": action_2[idx].tolist(),
    }

    detections = sv.Detections(
        xyxy=xyxy_coord, tracker_id=track_id, class_id=class_id, data=data
    )

    return detections


def process_video(
    source_video_path: str,
    detections_path: str,
    target_video_path: str,
    color_by: str,
) -> None:
    """
    Processes a single video by annotating it with detection results and saving the output.

    Args:
        source_video_path (str): Path to the input video file.
        detections_path (str): Path to the JSON file containing detection results.
        target_video_path (str): Path to save the annotated video.
        color_by (str): Specifies the attribute to use for coloring the bounding boxes
                        (e.g., "action", "action2", "activity", "individual").

    Returns:
        None
    """

    color_lookup = sv.ColorLookup("class")

    box_annotator = sv.BoundingBoxAnnotator(
        color_lookup=color_lookup, thickness=2
    )  # BondingBox annotator instance
    box_fill_annotator = sv.ColorAnnotator(color_lookup=color_lookup, opacity=0.2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.8, color_lookup=color_lookup, text_thickness=1
    )  # Label annotator instance
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=1
    )  # for generating frames from video
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with open(detections_path, "r") as f:
        detection_results = json.load(f)["frames"]

    frame_idx = 0
    with sv.VideoSink(
        target_path=target_video_path, video_info=video_info, codec="mp4v"
    ) as sink:
        for frame in tqdm(frame_generator):
            if len(detection_results[frame_idx]["detections"]) > 0:
                detections = supervision_from_json(
                    detection_results[frame_idx], color_by=color_by
                )

                # Prepare labels
                labels = []
                for tracker_id in range(len(detections.tracker_id)):
                    labels.append(
                        " | ".join(
                            str(detections.data[l][tracker_id])
                            for l in detections.data
                            if detections.data[l][tracker_id] != "none"
                        ).replace("_", " ")
                    )

                # Annotating detection boxes
                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(), detections=detections
                )
                annotated_frame = box_fill_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )

                # Annotating labels
                annotated_label_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
            else:
                annotated_label_frame = frame.copy()

            annotated_label_frame = cv2.putText(
                annotated_label_frame,
                str(frame_idx),
                org=(10, 20),
                color=(255, 255, 255),
                fontFace=1,
                fontScale=2,
            )
            sink.write_frame(frame=annotated_label_frame)

            frame_idx += 1


def process_batch_video_folder(options):
    """
    Processes a folder of videos for detection and tracking, generating annotated videos.

    Args:
        options (argparse.Namespace): Parsed command-line arguments containing:
            - input_video_folder (str): Path to the folder containing input videos.
            - input_json_folder (str): Path to the folder containing JSON detection results.
            - output_video_folder (str): Path to save the annotated videos.
            - color_by (str): Attribute to use for coloring bounding boxes.
            - skip_existing (bool): Whether to skip processing videos that already have output files.

    Returns:
        None
    """

    ## Check inputs
    assert os.path.isdir(options.input_video_folder), "{} is not a folder".format(
        options.input_video_folder
    )

    assert os.path.isdir(options.input_json_folder), "{} is not a folder".format(
        options.input_json_folder
    )

    assert (
        options.output_video_folder is not None
    ), "{} You must specify an output folder".format(options.output_video_folder)

    ## Get the paths of all videos in provided folder
    input_videos_full_paths = list(Path(options.input_video_folder).rglob("*.mp4"))

    ## Process each video individually
    for input_video_file in tqdm(
        input_videos_full_paths,
        leave=True,
        position=0,
        total=len(input_videos_full_paths),
    ):
        rel_path = input_video_file.relative_to(options.input_video_folder)
        json_input_file = Path(options.input_json_folder) / rel_path.with_suffix(
            ".json"
        )
        if not json_input_file.exists():
            print(f"Skipping {input_video_file.stem} as no tracks annotation available")
            continue
        video_output_file = Path(options.output_video_folder) / rel_path
        os.makedirs(video_output_file.parent, exist_ok=True)
        if video_output_file.exists() and options.skip_existing:
            continue

        process_video(
            str(input_video_file),
            str(json_input_file),
            str(video_output_file),
            options.color_by,
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("input_video_folder", type=str, help="video folder to process")

    parser.add_argument(
        "input_json_folder", type=str, default=None, help="input json folder"
    )

    parser.add_argument(
        "output_video_folder", type=str, default=None, help="annotated videos output folder"
    )

    parser.add_argument(
        "--color_by",
        choices=["action", "action2", "activity", "individual"],
        type=str,
        default="action",
        help="Color of the animal tracks",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Don't save if the video already exists",
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    process_batch_video_folder(args)


if __name__ == "__main__":
    main()
