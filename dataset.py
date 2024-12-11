import os
from lumaai import LumaAI
import time
# read .env file
from dotenv import load_dotenv
load_dotenv()


# delete the existing files
import shutil
shutil.rmtree("in", ignore_errors=True)
shutil.rmtree("gt", ignore_errors=True)

os.makedirs("in", exist_ok=True)
os.makedirs("gt", exist_ok=True)

client = LumaAI(
    auth_token=os.environ["LUMA_TOKEN"],
)

# video_path = "smokes/" + os.listdir("smokes")[0]
for video_path in os.listdir("smokes"):
  import numpy as np
  from PIL import Image, ImageDraw, ImageFilter


  def soft_edge(size):

    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    edge_width = min(width, height) // 10

    for i in range(edge_width):
        draw.rectangle([i, i, width - i - 1, height - i - 1], fill=i * (255 // edge_width))

    blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_width / 2))
    transparency_array = np.array(blurred_mask)

    return transparency_array/transparency_array.max()




  generation = client.generations.image.create(
    prompt="image of forest",
  )
  completed = False
  while not completed:
    generation = client.generations.get(id=generation.id)
    if generation.state == "completed":
      completed = True
    elif generation.state == "failed":
      raise RuntimeError(f"Generation failed: {generation.failure_reason}")
    print("Dreaming")
    time.sleep(2)

  image_url = generation.assets.image

  # download image_url into PIL
  from PIL import Image
  import requests
  from io import BytesIO

  response = requests.get(image_url)
  img = Image.open(BytesIO(response.content))
  background = img

  import cv2
  import random
  masks = []
  frames = []
  blended = []


  slience_frames = random.randint(5, 10) * 30
  for i in range(slience_frames):
    masks.append(np.zeros((512, 512), dtype=np.uint8))
    top = random.randint(0, 3)
    left = random.randint(0, 3)
    background = img.resize((512, 512)).convert("L")
    background = background.crop((left, top, left + 500, top + 500))
    background = background.resize((512, 512))
    background = Image.fromarray((np.array(background) * random.uniform(0.95, 1)).astype("uint8"))
    
    blended.append(background)
    
  video = cv2.VideoCapture("smokes/"+video_path)
  print(f"Processing {video_path}, with frame counts: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}")
  size = random.randint(128, 256)
  locx = random.randint(0, 512 - size)
  locy = random.randint(0, 512 - size) 
  while video.isOpened():
    background = img.resize((512, 512)).convert("L")
    ret, frame = video.read()
    if not ret:
      break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (512, 512))

    gray = Image.fromarray(gray).resize((size, size))
    
    white = Image.new('RGB', (512, 512), (255, 255, 255))

    canvas = Image.new('L', (512, 512), (0))
    softedge_mask = Image.fromarray((soft_edge((size, size)) * 255).astype("uint8"))
    # softedge_mask = Image.new('L', (512, 512), (0))
    # softedge_mask.paste(softedge_mask, (100, 100))
    canvas.paste(gray, (locx, locy), mask=softedge_mask)
    gray = np.array(canvas)
    mask = gray > 0.1 * 255
    gray = Image.fromarray(gray.astype("uint8"))
    relocated_mask = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    frames.append(gray)
    background.paste(white, (locx, locy), mask=Image.fromarray((np.array(gray)//2 * mask).astype("uint8")))
    relocated_mask.paste(white, (locx, locy), mask=Image.fromarray((np.array(gray) * mask).astype("uint8")))
    # relocated_mask = mask
    # randomly crop
    top = random.randint(0, 3)
    left = random.randint(0, 3)
    background = background.crop((left, top, left + 500, top + 500))
    background = background.resize((512, 512))
    background = Image.fromarray((np.array(background) * random.uniform(0.95, 1)).astype("uint8"))
    mask = Image.fromarray(((np.array(relocated_mask) > (0)) * 255).astype("uint8"))
    mask = mask.resize((512, 512))
    masks.append(mask)
    blended.append(background)


  video_writer = cv2.VideoWriter("in/"+video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))
  icount = 0
  for i in range(len(blended)):
    video_writer.write(np.array(blended[i])[:,:,None].repeat(3, axis=2))
    icount+=1
  video_writer.release()

  video_writer = cv2.VideoWriter("gt/"+video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))
  jcount = 0
  for i in range(len(blended)):
    video_writer.write(np.array(masks[i])[:,:,None].repeat(3, axis=2))
    # print(np.array(masks[i]).shape)
    jcount+=1
  print(icount, jcount)
  video_writer.release() 
  video.release()
  print("Finished processing", video_path)
