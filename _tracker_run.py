import sys, os
sys.path.insert(0, "C:\\Users\\rajku\\OneDrive\\Desktop\\Dezzex-stable\\3d twin\\backend")
from person_tracker_bbox2 import track_persons
track_persons(
    video_path    = "C:\\Users\\rajku\\OneDrive\\Desktop\\Dezzex-stable\\3d twin\\backend\\uploads\\1780563281053_emp_head_count1_output.mp4",
    floor_polygon = [[340.33,387.54],[1138.69,231.8],[1699.34,710.49],[1243.1,1079],[394.43,1079]],
    room_size_ft  = [39.3701,39.3701],
    camera_id     = "CAM-001",
    save_video    = False,
    yolo_model    = "C:\\Users\\rajku\\OneDrive\\Desktop\\Dezzex-stable\\3d twin\\backend\\yolo26n.pt",
)
