import os


def env_path(name, default):
    return os.environ.get(name, default)


def annotation_dir():
    return env_path('PDESTRE_ANNOTATION_DIR', 'data/P_Destre/annotation')


def video_dir():
    return env_path('PDESTRE_VIDEO_DIR', 'data/P_Destre/videos')


def frame_output_dir():
    return env_path('PDESTRE_FRAME_OUTPUT_DIR', 'data/P_Destre/frames')


def detection_output_dir():
    return env_path(
        'PDESTRE_DETECTION_OUTPUT_DIR',
        'outputs/visible_PDestre_bottom_exp_out',
    )


def cam_output_dir():
    return env_path(
        'PDESTRE_CAM_OUTPUT_DIR',
        'outputs/visible_PDestre_head_exp_out',
    )
