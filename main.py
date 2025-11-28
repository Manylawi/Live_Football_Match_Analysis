from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator, draw_camera_movement
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import argparse
import os

# Ensure the argument here matches the variable used inside (video_path)
def main(video_path):
    # 1. Setup Dynamic Paths based on input filename
    filename = os.path.basename(video_path)
    video_name = os.path.splitext(filename)[0]
    
    track_stub_path = f'stubs/track_stubs_{video_name}.pkl'
    camera_stub_path = f'stubs/camera_movement_stub_{video_name}.pkl'
    output_video_path = f'output_videos/output_{video_name}.mp4'

    # Ensure output directories exist
    os.makedirs('stubs', exist_ok=True)
    os.makedirs('output_videos', exist_ok=True)

    # 2. Get Video FPS dynamically
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Processing: {video_path} | FPS: {input_fps}")

    # Read Video
    video_frames = read_video(video_path)
    if len(video_frames) == 0:
        print("Error: No frames read from video.")
        return

    # Resize frames to 720p
    resized_frames = [cv2.resize(f, (1280, 720)) for f in video_frames]

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(resized_frames,
                                       read_from_stub=True,
                                       stub_path=track_stub_path)
    tracker.add_position_to_tracks(tracks)

    # Camera Movement
    camera_estimator = CameraMovementEstimator(resized_frames[0])
    camera_movement_per_frame = camera_estimator.get_camera_movement(
        resized_frames,
        read_from_stub=True,
        stub_path=camera_stub_path
    )
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and Distance Estimator
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(resized_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(resized_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # Initialize Video Writer
    height, width, _ = resized_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, input_fps, (width, height))

    # Process and write each frame
# Process and write each frame
# Process and write each frame
    for frame_num, frame in enumerate(resized_frames):
        # 1. Prepare data for the current frame wrapped in a list [ ]
        #    This creates a "batch of 1" so the functions don't crash
        current_frame_tracks = {k: [v[frame_num]] for k, v in tracks.items()}

# 2. Draw object tracks
        frame = tracker.draw_annotations(
            [frame], 
            current_frame_tracks, 
            team_ball_control[:frame_num+1] # <--- Pass history from start (0) to current (frame_num)
        )[0]
        
        # 3. Draw camera movement
        frame = draw_camera_movement([frame], [camera_movement_per_frame[frame_num]])[0]
        
        # 4. Draw speed & distance
        #    We pass 'current_frame_tracks' which has the list wrapper [ ]
        speed_estimator.draw_speed_and_distance([frame], current_frame_tracks)

        # 5. Write frame
        out.write(frame)

        # Free memory
        del frame

    out.release()
    print(f"Video processing complete! Saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process football video for tracking.")
    parser.add_argument('--video', type=str, required=True, help="Path to the input video file")
    
    args = parser.parse_args()
    
    # We pass args.video (the value) into main(video_path)
    main(args.video)