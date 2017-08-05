# ref: https://stackoverflow.com/questions/14440400/creating-a-video-using-opencv-2-4-0-in-python
# ref: http://www.xavierdupre.fr/blog/2016-03-30_nojs.html
def make_video(images, outvid="movie.avi", fps=5, size=None,
               is_color=True, format="MJPG"):
    """
    Create a video from a list of images.

    @param      images      list of images to use in the video
    @param      outvid      output video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

import csv, os
center_images = []
left_images = []
right_images = []
path = './training_data/track2_set0/IMG/'
with open('./training_data/track2_set0/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if 'IMG' not in row[0]:
            continue
        center_images.append(path + row[0].strip().split('/')[-1])
        left_images.append(path + row[1].strip().split('/')[-1])
        right_images.append(path + row[2].strip().split('/')[-1])
        if len(center_images) > 1000:
            break
movie_path = './training_data/track2_set0/movie/'
make_video(center_images, movie_path + "center_myData.avi", fps=24, format="MJPG")
make_video(left_images, movie_path + "left_myData.avi", fps=24, format="MJPG")
make_video(right_images, movie_path + "right_myData.avi", fps=24, format="MJPG")