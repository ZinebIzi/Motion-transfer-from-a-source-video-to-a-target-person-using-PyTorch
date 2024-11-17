import cv2
import os

class VideoReader:
    """ Helper class for video utilities """
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename) #object in OpenCV is a class that provides an interface for capturing video from a file, camera, or any video stream. It allows you to open, read, and manipulate video streams frame by frame.
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # retrieves the total number of frames in the video
        #cap.get(propId): Retrieves various properties of the video, like frame width, height, frame count, and more.
        self._current_frame = 0

    def readFrame(self):
        """ Read a frame """
        #reads and returns a single frame from the video.
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            # ret : boolean value indicating whether the frame was successfully read (True) or not (False).
            # frame : the actual frame read from the video
            if ret is False or frame is None:
                return None
            self._current_frame += 1
        else:
            return None
        return frame

    def readNFrames(self, num_frames=1):
        """ Read n frames """
        frames_list = []
        for _ in range(num_frames):
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret is False or frame is None:
                    return None
                frames_list.append(frame)
                self._current_frame += 1
            else:
                return None
        return frames_list

    def isOpened(self):
        """ Check is video capture is opened """
        return self.cap.isOpened()

    def getFrameWidth(self):
        """ Get width of a frame """
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def getFrameHeight(self):
        """ Get height of a frame """
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def getVideoFps(self):
        """ Get Frames per second of video """
        return self.cap.get(cv2.CAP_PROP_FPS)

    def getCurrentFrame(self):
        """ Get current frame of video being read """
        return self._current_frame

    def getTotalFrames(self):
        """ Get total frames of a video """
        return self._total_frames

    def release(self):
        """ Release video capture """
        #close the video capture
        self.cap.release()

    def __del__(self):
        #method is invoked to clean up resources before the object is removed from memory.
        self.release()


if __name__ == '__main__':
    print("Current Working Directory:", os.getcwd())
    video = VideoReader("data/taichi1.mp4")
    for i in range(video.getTotalFrames()):
        image = video.readFrame()
        cv2.imshow('Image', image) # function to display the current frame in a window named 'Image'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            #If the user presses 'q' while the video is being played, the loop breaks, and the video stops playing.
            break
    cv2.destroyAllWindows()