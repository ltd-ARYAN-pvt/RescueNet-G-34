from json import load
from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
import numpy as np
import io
import torch
from PIL import Image
import cv2
import tempfile
import torch
from PIL import Image
from CycleGans.generator import Generator
import CycleGans.config as config

app = Flask(__name__)
# CORS(app)

gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
state_dict=torch.load(config.CHECKPOINT_GEN_Z)["state_dict"]
gen_Z.load_state_dict(state_dict)
gen_Z.eval()


@app.route('/upload', methods=['POST'])
def upload():
    # print(request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    # print("file end",file.filename.lower().endswith(('.png', '.jpg', '.jpeg','.mp4')))
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img=load_image(io.BytesIO(img_bytes))
        # img = Image.open(io.BytesIO(img_bytes))
        # img = img.convert('RGB')

        #--> For this example, we just return the mirror image
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        img_transformed = transform_image(img)
        #--> Yaha par model lana hai

        output_image = Image.fromarray(img_transformed)
        output_image.save('output.png')

        return send_file('output.png', mimetype='image/png')
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file.flush()
    
    if file and file.filename.lower().endswith(('.mp4')):
        # Process the video
        video = cv2.VideoCapture(temp_file.name)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(total_frames):
            ret, frame = video.read()
            frame = transform_image(frame)
            mirrored_frame = transform_image(frame)
            frames.append(mirrored_frame)
        video.release()

        # Save the processed video to a file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            output_video.write(frame)
        output_video.release()

        # Return the processed video to the frontend
        return send_file('output.mp4', as_attachment=True)
    else:
        return jsonify({'error': 'Nither image or video'}), 400

def load_image(image_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    augmentations = config.transforms(image=image, image0=image)
    image = augmentations["image"]
    return image.to(config.DEVICE)

def transform_image(input_image):

    with torch.no_grad():
        output_image = gen_Z(input_image)
    output_image = (output_image + 1) / 2
    output_image = output_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    output_image = (output_image * 255).astype(np.uint8)
    return output_image


if __name__ == '__main__':
    app.run(debug=True)