import cv2
import numpy as np
import torch
from concern.config import Configurable, Config
import argparse
import math
import copy
import time

def init_torch_tensor():
    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
    return device

def init_model(experiment, device):
    model = experiment.structure.builder.build(device)
    return model

def resume(model, path):
    states = torch.load(
        path, map_location=device)
    model.load_state_dict(states, strict=False)

def resize_image(img):
    height, width, _ = img.shape
    if height < width:
        new_height = 736
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = width
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def load_image(img):
    img = resize_image(img)
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    img -= RGB_MEAN
    img /= 255.
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return img

parser = argparse.ArgumentParser(description='Text Recognition Training')
parser.add_argument('exp', default="experiments/seg_detector/panel_resnet18_deform_thre.yaml", type=str)
parser.add_argument('--source', default='test_v5.mp4', type=str)
parser.add_argument('--weights', default='weights/final', type=str)

args = parser.parse_args()
args = vars(args)
args = {k: v for k, v in args.items() if v is not None}

conf = Config()
experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
experiment_args.update(cmd=args)
experiment = Configurable.construct_class_from_config(experiment_args)

points = []
batch = dict()
device = init_torch_tensor()
model = init_model(experiment, device)
resume(model, path=args['weights'])
model.eval()

def mouse_callback(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        
        points.append([x/int(frame.shape[1]), y/int(frame.shape[0])])
        batch['center_points'] = torch.as_tensor([points]).to('cuda:0')


# Video   
cap = cv2.VideoCapture(args['source'])    
# cap = cv2.VideoCapture(0)     
while cap.isOpened():               
    ret, frame = cap.read()        
    if ret:                        
        frame_copy = copy.deepcopy(frame).astype('float32')
        batch['image'] = load_image(frame_copy).to(device)
        batch['shape'] = [[frame.shape[0], frame.shape[1]]]
        cv2.setMouseCallback('frame', mouse_callback)

        if 'center_points' in batch.keys():
            with torch.no_grad():
                t0 = time.time()
                pred = model.forward(batch, training=False)
                x = (1/(time.time()-t0))
                fps = "%.2f fps"%x
                # points = batch['center_points'].cpu().numpy()
                cv2.putText(frame, fps, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                for i in range(len(points)):
                    cv2.circle(frame, (int(points[i][0]), int(points[i][1])), 3, (0, 0, 255), 2)
                output = experiment.structure.representer.represent(batch, pred, is_output_polygon=False) 
                boxes, _ = output
                boxes = boxes[0]
                for box in boxes:
                    box = np.array(box).astype(np.int32).reshape(-1, 2)
                    cv2.polylines(frame, [box], True, (0, 255, 0), 2)
        cv2.imshow('frame', frame)  
        key = cv2.waitKey(25)       
        if key == ord('q'):         
            cap.release()           
            break
    else:
        cap.release()
cv2.destroyAllWindows()



