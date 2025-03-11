import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends
from pydantic import BaseModel
from PIL import Image
import io
from utils.maze_gen import MazeGeneratorONNX
import uvicorn
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from utils.maze_inference import Maze
from utils.solver_inference import Solver

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

class MazeRequest(BaseModel):
    initialGrid: list
    size: int
    start: list[int]  # [y, x] coordinates of start position
    firstStep: list[int]  # [y, x] coordinates of first step
    end: list[int]  # [y, x] coordinates of end position

generator = MazeGeneratorONNX(onnx_model_path="denoiser_best.onnx")


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


@app.get("/health")
async def health():
    return {"message": "OK"}

@app.post("/generate_sequence/")
async def generate_sequence(
    request: MazeRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Create maze from the request data
        maze = Maze(request.size, request.size)
        maze.grid = np.array(request.initialGrid)
        
        # Set entrance and exit for the solver to use
        maze.entrance = (request.start[0], request.start[1])
        maze.exit = (request.end[0], request.end[1])
        
        maze.grid[maze.entrance] = 0
        maze.grid[maze.exit] = 0
        
        # Use the solver to find the solution path
        solution_path = Solver.solve(maze)
        
        if solution_path:
            num_frames = min(len(solution_path) + 2, 30) # not more than 30 frames
        else:
            # Fallback if no solution found
            num_frames = 10
            
        # Log the path length and number of frames
        print(f"Solution path length: {len(solution_path) if solution_path else 0}")
        print(f"Generating {num_frames} frames for animation")
        
        # Convert positions to the format expected by create_maze_frame
        first_move_cell = {"x": request.firstStep[1], "y": request.firstStep[0]}
        start_cell = {"x": request.start[1], "y": request.start[0]}
        end_cell = {"x": request.end[1], "y": request.end[0]}
        
        # Create maze frames with path and entrances highlighted
        maze_images = maze.create_maze_frame(
            target_size=(64, 64),
            start_cell=start_cell,
            end_cell=end_cell,
            first_move_cell=first_move_cell
        )
        
        # Convert images to numpy arrays and normalize
        img_arrays = [
            (np.array(img).astype(np.float32) / 255.0 - 0.5) / 0.5
            for img in maze_images
        ]

        # Transpose to channels-first format and add batch dimension
        img_arrays_transposed = [img.transpose(2, 0, 1) for img in img_arrays]  # [H,W,C] -> [C,H,W]
        initial_frames = np.stack(img_arrays_transposed, axis=0)[np.newaxis, ...]  # Shape: [1, 2, 3, 64, 64]

        # Generate sequence
        generated_frames = generator.generate_sequence(initial_frames, num_frames=num_frames)

        # Process the results
        output_frames = []
        # Process initial frames
        for i in range(initial_frames.shape[1]):
            frame = initial_frames[0, i]  # Get frame [C,H,W]
            frame = frame.transpose(1, 2, 0)  # Convert to [H,W,C]
            frame = (frame * 0.5 + 0.5) * 255.0  # Denormalize
            frame = frame.clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            output_frames.append(image_to_base64(pil_image))

        # Process generated frames
        for frame in generated_frames:
            # Convert from [C,H,W] to [H,W,C]
            frame = frame.transpose(1, 2, 0)
            frame = (frame * 0.5 + 0.5) * 255.0  # Denormalize
            frame = frame.clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            output_frames.append(image_to_base64(pil_image))

        # Log total number of frames returned
        print(f"Total frames returned: {len(output_frames)}")
        
        return {"frames": output_frames, "path_length": len(solution_path) if solution_path else 0, "num_frames": num_frames}
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