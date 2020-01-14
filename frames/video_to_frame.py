import cv2
import os


def video_to_frame(file_path, fps):
    """
    Converts a video file to many jpeg images
    :param file_path: The path to the video file
    :param fps: The desired frames per second for the outputted images
    :param out_path: The path where the images will be written to
    """

    cap = cv2.VideoCapture(file_path)


    basename = os.path.basename(file_path)
    name, ext = os.path.splitext(basename)
    out_path = "frames/" + name


    # Check to see desired fps <= video's fps
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > video_fps:
        print("Error: Desired fps > video's fps: {}".format(video_fps))
        exit(1)

    frame_increment = round(video_fps / fps)

    # Create output folder to store frames
    try:
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    except OSError:
        print("Error: Could not create directory for output")
        exit(1)

    # Save frames of video at desired fps, named after frame number from original video
    current_frame = 0
    while True:
        success, frame = cap.read()

        if not success:
            break

        f_name = out_path + "/frame_" + str(current_frame) + ".jpg"
        cv2.imwrite(f_name, frame)

        current_frame += frame_increment
        cap.set(1, current_frame)

    cap.release()
    cv2.destroyAllWindows()


#video_to_frame("2019 United States Grand Prix - Sunday Race.mp4", 1)

