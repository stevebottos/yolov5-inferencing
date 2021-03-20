import time 

import torch 
import imageio
import PIL
import numpy as np

from models.experimental import attempt_load
from detect import detect

device = torch.device("cpu") #  torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_vid = "test1"
input_folder = "./data/inputs"
output_folder = "./data/outputs"
vid = imageio.get_reader(f"{input_folder}/{test_vid}.mp4",  'ffmpeg')
weights = "yolov5s.pt"  # yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt
input_size = 320
model = attempt_load(weights, map_location=device)

def main():
    f = 0
    while True:
        s = time.time()
        try:
            im = vid.get_data(f)
        except:
            break

        im = PIL.Image.fromarray(im)
        im = im.resize([input_size, input_size])
        im_asarray = np.asarray(im)
        weights =  "yolov5s.pt" # ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']
        results = detect(model, weights, device, im_asarray)
        
        for r in results:
            box= r[:-1]
            imdraw = PIL.ImageDraw.Draw(im)
            imdraw.rectangle(box, fill=None, outline=None, width=2)

        im.save(f"{output_folder}/temp_images/{f}.jpg")
        f += 1
        print(time.time() - s)


if __name__ == "__main__":
    main()