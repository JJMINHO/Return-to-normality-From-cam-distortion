import cv2
import numpy as np
from datetime import datetime

VIDEO_FILE = 'chessboard.mp4'
CALIB_FILE = 'calib_result.npz'

try:
    with np.load(CALIB_FILE) as data:
        K = data['K']
        dist = data['dist']
except FileNotFoundError:
    print("calib_result.npz 파일이 없습니다. 캘리브레이션을 먼저 실행하세요.")
    exit()

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print("동영상 파일을 찾을 수 없습니다.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), 1, (width, height))

map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, new_K, (width, height), cv2.CV_32FC1)

scale = 0.5 if width > 1000 else 1.0
out_width = int(width * 2 * scale)
out_height = int(height * scale)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('demo_result.mp4', fourcc, fps, (out_width, out_height))

print("데모 영상 재생 및 저장을 시작합니다.")
print(" - 전체 영상은 'demo_result.mp4'로 자동 저장됩니다.")
print(" - 화면 캡처를 원하시면 's' 또는 'S'를, 종료하시려면 'q'를 누르세요.")
# ------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    cv2.putText(frame, 'Original', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(undistorted_frame, 'Undistorted', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    combined = np.hstack((frame, undistorted_frame))
    display_frame = cv2.resize(combined, None, fx=scale, fy=scale)

    cv2.imshow('Distortion Correction Demo', display_frame)

    out.write(display_frame)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') or key == ord('S'):
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f'demo_capture_{time_str}.jpg'
        cv2.imwrite(img_name, display_frame)
        print(f"📸 캡처 완료: {img_name}")

cap.release()
out.release()
cv2.destroyAllWindows()