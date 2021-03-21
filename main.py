import time 
#
import torch 
import imageio
import PIL
import numpy as np
import cv2 

from models.experimental import attempt_load
from detect import detect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_vid = "test1"
input_folder = "./data/inputs"
output_folder = "./data/outputs"
vid = imageio.get_reader(f"{input_folder}/{test_vid}.mp4",  'ffmpeg')
weights = "yolov5x.pt"  # yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt
input_size = 416
model = attempt_load(weights, map_location=device)
save_image = False

def main():
    f = 0
    while True:
        t = Timer()
        try:
            im = vid.get_data(f)
            array_im = cv2.resize(im, (input_size, input_size))
            pillow_im = PIL.Image.fromarray(array_im)
        except:
            break

        results = detect(model, device, array_im, input_size)
        results = results[:, 0:-2]
        
        if save_image:
            for box in results:
                imdraw = PIL.ImageDraw.Draw(pillow_im)
                imdraw.rectangle(box, fill=None, outline=None, width=2)
            pillow_im.save(f"{output_folder}/temp_images/{f}.jpg")
        
        t.print_delta()
        f += 1
        

class Timer():
    def __init__(self):
        self.start_time = time.time()

    def print_delta(self):
        print(time.time() - self.start_time)

if __name__ == "__main__":
    main()
