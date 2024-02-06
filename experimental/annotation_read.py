import numpy as np
import xml.etree.ElementTree as ET
import cv2

def addmask(canvas: np.ndarray, mask: dict) -> np.ndarray:
    top = mask["top"]
    height = mask["height"]
    width = mask["width"]
    decoded_mask = mask["mask"]

    print(f"Canvas shape: {canvas.shape}")
    print(f"Mask shape: {decoded_mask.shape}")
    
    canvas[top[0]:top[0]+height, top[1]:top[1]+width, :] = decoded_mask
    
    return canvas

def calculate_optical_flow(prev_frame, current_frame, curr_mask):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 1080, 3, 5, 1.2, 0)

    total_mean_flow_x = np.mean(flow[:, :, 0])
    total_mean_flow_y = np.mean(flow[:, :, 1])

    pan_factor = np.sqrt(total_mean_flow_x**2 + total_mean_flow_y**2)
    zoom_factor = np.sqrt(1 / (1 + pan_factor))

    flow_x_values = []
    flow_y_values = []
   
    top, left = curr_mask["top"]
    
    for y in range(curr_mask["mask"].shape[0]):
        for x in range(curr_mask["mask"].shape[1]):
            if flow[y + top, x + left] == 1:
                flow_x_values.append(flow[y + top, x + left, 0])
                flow_x_values.append(flow[y + top, x + left, 1])

    green_mean_flow_x = np.mean(flow_x_values)
    green_mean_flow_y = np.mean(flow_y_values)

    return pan_factor, zoom_factor, green_mean_flow_x, green_mean_flow_y

def calc_ball_displacements(box_data: list):
    for i in range(1, len(box_data)):
        displacement_y = box_data[i]["center"][0] - box_data[i-1]["center"][0]
        displacement_x = box_data[i]["center"][1] - box_data[i-1]["center"][1]

        displacement_area = (box_data[i]["area"]) - (box_data[i-1]["area"])

        speed = ((abs(displacement_x) + abs(displacement_y)))
        
        print((displacement_y, displacement_x, displacement_area, speed))

def rle2mask(rle: list, left: int, top: int, width: int, height: int, img_width: int, img_height: int, label: str):
    decoded = [0] * (width * height)
    decoded_idx = 0

    value = 0

    previous_count = rle[0]
    
    for i, v in enumerate(rle):
        if (v < previous_count) and (value == 0):
            decoded[decoded_idx:decoded_idx+v] = [1] * v
        else:
            decoded[decoded_idx:decoded_idx+v] = [value] * v
        decoded_idx += v
        value = abs(value - 1)

        previous_count = v

    decoded = np.array(decoded, dtype=np.uint8)
    decoded = decoded.reshape((height, width))

    return decoded


tree = ET.parse("./annotations/annotations.xml")

root = tree.getroot()

box_data = []

for image in root.findall('image'):
    if int(image.attrib["id"]) > 28:
        break

    img_width = int(image.attrib["width"])
    img_height = int(image.attrib["height"])

    img_name = image.attrib["name"]

    canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    for mask in image.findall("mask"):
        mask_attr = mask.attrib
        
        label = mask_attr["label"]

        rle = [int(count) for count in mask_attr["rle"].split(", ")]
        left = int(mask_attr["left"])
        top = int(mask_attr["top"])
        width = int(mask_attr["width"])
        height = int(mask_attr["height"])

        mask = rle2mask(rle, left, top, width, height, img_width, img_height, label)
        
        if label == "green":
            green = {
                "mask": mask,
                "top": (top, left),
                "width": width,
                "height": height
            }
        elif label == "hole":
            hole = {
                "mask": mask,
                "top": (top, left),
                "width": width,
                "height": height
            }

    for box in image.findall("box"):
        box_attr = box.attrib

        xtl = int(float(box_attr["xtl"]))
        ytl = int(float(box_attr["ytl"]))
        xbr = int(float(box_attr["xbr"]))
        ybr = int(float(box_attr["ybr"]))

        ball = np.zeros((ybr-ytl, xbr-xtl, 3), dtype=np.uint8)

        ball = np.stack((ball[:,:,0], ball[:,:,1], np.ones((ybr-ytl, xbr-xtl), dtype=np.uint8)), axis=-1)

        radius = xbr - xtl

        center_x = int((xtl + xbr) / 2)
        center_y = int((ytl + ybr) / 2)

        width = xbr-xtl
        height = ybr-ytl

        box_data.append({
            "center": (center_y, center_x),
            "area": width * height,
        })

        ball = {
            "mask": ball,
            "top": (ytl, xtl),
            "width": (xbr-xtl),
            "height": (ybr-ytl)
        }
