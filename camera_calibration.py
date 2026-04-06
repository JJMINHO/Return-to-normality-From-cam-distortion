import cv2
import numpy as np

VIDEO_FILE = 'chessboard.mp4'
PATTERN_SIZE = (8, 6)  # 체스보드 내부 코너 개수 (가로, 세로)
SQUARE_SIZE = 0.032  # 체스보드 한 칸의 크기 (단위: m, 예: 25mm = 0.025)
# ---------------------------------------------------

# 3D 공간상의 실제 코너 좌표 생성 (Z=0)
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 실제 3D 점들을 저장
imgpoints = []  # 이미지 상의 2D 점들을 저장

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print("동영상 파일을 찾을 수 없습니다.")
    exit()

frame_count = 0
success_frames = 0

print("왜곡계수 측정 중...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # 연산 속도와 이미지 다양성 확보를 위해 매 5프레임마다 샘플링
    if frame_count % 5 != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]

    # 체스보드 코너 찾기
    ret_corners, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

    if ret_corners:
        # 서브픽셀 단위로 코너 위치 정밀 보정
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)
        success_frames += 1

        cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners2, ret_corners)
        cv2.imshow('Finding Corners', frame)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

# 2. 카메라 캘리브레이션 수행
if success_frames > 0:
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("\n" + "=" * 40)
    print("[캘리브레이션 결과]")
    print(f"RMS Error: {rms:.4f}")
    print("\nCamera Matrix (K):")
    print(camera_matrix)
    print(f" - fx: {camera_matrix[0, 0]:.4f}")
    print(f" - fy: {camera_matrix[1, 1]:.4f}")
    print(f" - cx: {camera_matrix[0, 2]:.4f}")
    print(f" - cy: {camera_matrix[1, 2]:.4f}")
    print("\nDistortion Coefficients:")
    print(dist_coefs.ravel())
    print("=" * 40 + "\n")

    # 파라미터 파일로 저장
    np.savez('calib_result.npz', K=camera_matrix, dist=dist_coefs)
    print("파라미터가 'calib_result.npz' 파일로 저장되었습니다.")
else:
    print("체스보드를 찾지 못했습니다.")