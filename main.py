import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends
from pydantic import BaseModel
import torch
from PIL import Image
import io
from utils.maze_gen import MazeGeneratorONNX
import uvicorn
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from data import Maze

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # List of allowed origins (frontend domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ImageRequest(BaseModel):
    frame1: str  # base64 string
    frame2: str  # base64 string

class EntrancePoint(BaseModel):
    x: int
    y: int

class MazeRequest(BaseModel):
    initialGrid: list
    size: int
    entrances: list[EntrancePoint]
    path: list
    startCell: dict
    firstMoveCell: dict
    entrancePositions: list[list[int]]

generator = MazeGeneratorONNX(onnx_model_path="denoiser_denoise.onnx", device="cpu")


API_KEY = "your-secret-api-key-here"  # In production, use an environment variable
API_KEY_NAME = "X-API-Key"

# Add the security scheme
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Add this function to verify the API key
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header is None:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="API Key header is missing"
        )
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )
    return api_key_header

@app.post("/generate_sequence/")
async def generate_sequence(
    request: MazeRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Create maze from the request data
        maze = Maze(request.size, request.size)
        maze.grid = np.array(request.initialGrid)
        
        # Create single maze image with path and entrances highlighted
        maze_img = maze.create_maze_frame(
            highlight_cells=[request.startCell, request.firstMoveCell],
            target_size=(32, 32),
            entrances=request.entrances
        )
        
        # Convert PIL image to tensor and reshape to match expected dimensions
        img_tensor = torch.tensor(np.array(maze_img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        initial_frames = img_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1, 1)  # Shape: [1, 2, 3, 32, 32]
        
        # Generate sequence
        generated_frames = generator.generate_sequence(initial_frames)

        # Convert generated frames to list of base64 images
        output_frames = []
        for frame in generated_frames:
            frame = frame.squeeze(0).permute(1, 2, 0).numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            base64_str = image_to_base64(pil_image)
            output_frames.append(base64_str)

        return {"frames": output_frames}
    except Exception as e:
        print(f"Error in generate_sequence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def base64_to_image(base64_str):
    try:
        # Remove potential data URL prefix
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Decode base64 string to bytes
        img_data = base64.b64decode(base64_str)
        
        # Try to open the image
        img = Image.open(io.BytesIO(img_data))
        return img
    except Exception as e:
        raise ValueError(f"Invalid image data. Please ensure you're sending a valid base64-encoded image. Error: {str(e)}")

# Optional: If you need to return images as base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)