import cv2
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

print(cap)
while True:
    ret, frame = cap.read()

    if not ret:
        print("cant read frame")
        break

    sift = cv2.SIFT_create(200)
    keypoint, descriptor = sift.detectAndCompute(frame, None)
    frame_with_keypoints = cv2.drawKeypoints(
        frame, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # print(frame_with_keypoints.shape)
    cv2.imshow("frame", frame_with_keypoints)
    out.write(frame_with_keypoints)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
