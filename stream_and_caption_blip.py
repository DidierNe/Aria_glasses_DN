import cv2
import numpy as np
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
import argparse
import sys
import torch
# === Load BLIP ===
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",
                                          use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP loaded.")


torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # accelerate for apple sillicon
model = model.to(torch_device)

# === Observer class to receive frames ===
class StreamingObserver:
    def __init__(self):
        self.last_image = None
        self.last_caption_time = 0
        self.cooldown = 3  # seconds
        self.caption = "Waiting for image..."

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord): # called from aria SDK aria.sdk.BaseStreamingClientObserver
        if record.camera_id == aria.CameraId.Rgb:   # be sure to have the right camera stream
            self.last_image = np.rot90(image, -1)  # Rotate to correct orientation
            self.maybe_caption()                    #call method below to generate caption related to the image

    def maybe_caption(self):    #maybe not optimal but in order to not have to generate a caption every frame check time
        now = time.time()
        if self.last_image is not None and now - self.last_caption_time >= self.cooldown:   #new caption every 3 sec
            self.caption = self.generate_caption(self.last_image)
            self.last_caption_time = now
            print("Caption from BLIP:", self.caption)

    def generate_caption(self, np_img: np.ndarray) -> str: # return the caption
        # Resize image for BLIP
        resized = cv2.resize(np_img, (384, 384)) #BLIP format , maybe check quality and algo of resize
        image = Image.fromarray(resized)
######QUESTION for rapidity do we need a RGB one or monochromatic could be as good and faster?
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
 #preprocess image for model --->'pixel_values': tensor of shape (1, 3, 384, 384)
        with torch.no_grad():  # ✅ Optional but recommended
            output = model.generate(**inputs)   #generate a sequence of token IDs (numbers that map to words)
        return processor.decode(output[0], skip_special_tokens=True)
# === Parse command line args ===
parser = argparse.ArgumentParser()
parser.add_argument(
    "--interface",
    type=str,
    required=True,
    choices=["usb", "wifi"],
    help="Connection type: usb or wifi",
)
# parser.add_argument(
#     "--device-ip",
#     type=str,
#     help="IP address of the Aria glasses (required for wifi)",
# )
args = parser.parse_args()

# === Optional device connection via DeviceClient ===
if args.interface == "wifi":
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    # if args.device_ip:
    #     client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    device = device_client.connect()
    streaming_manager = device.streaming_manager

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = "profile18"  # default profile
    if args.interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb


    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    streaming_manager.start_streaming()
    print("Streaming started over Wi-Fi.")

# === Aria client setup ===
print(" Initializing Aria streaming client...")
aria.set_log_level(aria.Level.Info)                             # set function/log used for getting errors and others info in the terminal directly
streaming_client = aria.StreamingClient()                       #Creates the client then we subscibe for a stream 

config = streaming_client.subscription_config                     #Retrieves the current configuration ---> then we modify, next line
config.subscriber_data_type = aria.StreamingDataType.Rgb        # what stream we want to receive, here test the front so RGB one
config.message_queue_size[aria.StreamingDataType.Rgb] = 1          #avoid lag ---> Keep only the most recent frame. Drop older ones if too slow.

# below chat gpt tips for securities subscription config, "Use temporary, per-session certificates that are automatically generated when run"
options = aria.StreamingSecurityOptions()
options.use_ephemeral_certs = True
config.security_options = options

streaming_client.subscription_config = config #Applies the security settings you just defined to the subscription config, FINALIZES what we will subscribe when started



# Instantiates your custom observer class (which must define on_image_received).
# This is where your own logic for handling each new frame lives:
# Displaying it
# Processing it
# Captioning it, etc
observer = StreamingObserver()              #class define above


streaming_client.set_streaming_client_observer(observer) ##Attaches your observer to the client
streaming_client.subscribe()                               #Starts the live data stream 
print("✅ Connected to Aria. Streaming started.")

# === OpenCV display loop ===
cv2.namedWindow("Aria RGB + Caption", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Aria RGB + Caption", 640, 480)

try:
    while True:
        if observer.last_image is not None:
            frame = observer.last_image.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Draw caption on image
            cv2.putText(
                frame,
                observer.caption,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("Aria RGB + Caption", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    streaming_client.unsubscribe()
    cv2.destroyAllWindows()
    print(" exit")
# command for parser: python stream_and_caption_blip.py --interface wifi 
