import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pickle
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator, draw_camera_movement
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

# --- Page Config ---
st.set_page_config(page_title="Football Analysis AI", layout="wide")
st.title("⚽ Football Analysis System")
st.markdown("Upload a video to track players, estimate speed, and analyze possession.")

# --- Cache the Model ---
@st.cache_resource
def load_tracker_model():
    return Tracker('models/best.pt')

tracker = load_tracker_model()

# --- Helper: Save Uploaded File ---
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling file: {e}")
        return None

# --- Main Processing Function ---
def process_video(video_path, file_name):
    st.info("Reading video frames... (This might take a moment)")
    
    # 1. Read Video
    video_frames = read_video(video_path)
    if not video_frames:
        st.error("Could not read video frames.")
        return None

    # Resize frames to 720p
    resized_frames = [cv2.resize(f, (1280, 720)) for f in video_frames]
    
    # --- FIX: Generate Dynamic Stub Paths ---
    # We use the file_name to create a unique stub for this specific video.
    # This prevents the "IndexError" when switching between videos of different lengths.
    clean_name = os.path.splitext(file_name)[0]
    track_stub_path = f'stubs/track_stubs_{clean_name}.pkl'
    camera_stub_path = f'stubs/camera_movement_stub_{clean_name}.pkl'
    
    # Ensure directories exist
    os.makedirs('stubs', exist_ok=True)
    os.makedirs('output_videos', exist_ok=True)

    # --- Tracking ---
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("Detecting and Tracking Objects...")
    tracks = tracker.get_object_tracks(
        resized_frames,
        read_from_stub=True, 
        stub_path=track_stub_path
    )
    
    # Safety Check: If the loaded stub has more frames than the current video, 
    # it's invalid. We must re-run detection.
    if len(tracks['players']) != len(resized_frames):
        st.warning("Cached data mismatch. Re-running detection...")
        tracks = tracker.get_object_tracks(
            resized_frames,
            read_from_stub=False, # Force re-run
            stub_path=track_stub_path
        )

    tracker.add_position_to_tracks(tracks)
    progress_bar.progress(20)

    # --- Camera Movement ---
    status_text.text("Estimating Camera Movement...")
    camera_estimator = CameraMovementEstimator(resized_frames[0])
    camera_movement_per_frame = camera_estimator.get_camera_movement(
        resized_frames,
        read_from_stub=True,
        stub_path=camera_stub_path
    )
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    progress_bar.progress(40)

    # --- View Transformer ---
    status_text.text("Transforming View...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # --- Interpolate Ball ---
    status_text.text("Interpolating Ball Positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    progress_bar.progress(50)

    # --- Speed & Distance ---
    status_text.text("Calculating Speed & Distance...")
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    # --- Team Assignment ---
    status_text.text("Assigning Teams...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(resized_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Safety check to ensure we don't go out of bounds
            if frame_num < len(resized_frames):
                team = team_assigner.get_player_team(resized_frames[frame_num], track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    progress_bar.progress(70)

    # --- Ball Acquisition ---
    status_text.text("Calculating Ball Possession...")
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
    progress_bar.progress(80)

    # --- Video Generation ---
    status_text.text("Rendering Output Video...")
    height, width, _ = resized_frames[0].shape
    
    # We save to a temp file
    output_video_path = tempfile.mktemp(suffix=".mp4")
    
    # Define codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = len(resized_frames)
    
    for frame_num, frame in enumerate(resized_frames):
        # Update progress for rendering
        pct = 80 + int((frame_num / total_frames) * 20)
        progress_bar.progress(min(pct, 100))
        
        # Prepare batch-of-1
        current_frame_tracks = {k: [v[frame_num]] for k, v in tracks.items()}
        
        # Draw
        frame = tracker.draw_annotations(
            [frame], 
            current_frame_tracks, 
            team_ball_control[:frame_num+1]
        )[0]
        
        frame = draw_camera_movement([frame], [camera_movement_per_frame[frame_num]])[0]
        speed_estimator.draw_speed_and_distance([frame], current_frame_tracks)

        out.write(frame)

    out.release()
    progress_bar.progress(100)
    status_text.text("Processing Complete!")
    
    return output_video_path

# --- UI Layout ---
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.subheader("Input Video")
    st.video(uploaded_file)
    
    if st.button("Process Video"):
        # Save temp file
        temp_path = save_uploaded_file(uploaded_file)
        
        if temp_path:
            with st.spinner('AI is processing the video...'):
                # Pass the uploaded filename so we can create a unique stub
                output_path = process_video(temp_path, uploaded_file.name)
            
            if output_path:
                st.subheader("Output Video")
                try:
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                    st.video(video_bytes)
                    
                    st.download_button(
                        label="Download Processed Video",
                        data=video_bytes,
                        file_name="football_analysis_output.mp4",
                        mime="video/mp4"
                    )
                except Exception as e:
                    st.error(f"Error displaying video: {e}")
                
            # Cleanup temp files
            os.remove(temp_path)