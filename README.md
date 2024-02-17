# Advertisement Asset Generation Pipeline

This project aims to generate advertisement assets using deep learning models and algorithms. It takes textual descriptions of advertisement frames as input and generates corresponding images, which are then combined into a storyboard.

## Folder Structure

The folder structure of the project is organized as follows:
/
│
├── app/
│ ├── algorithm.py
│ ├── utils.py
│ ├── image_generator.py
│ └── storyboard_visualizer.py
│
├── assets/
│ └── [Generated images and storyboards]
│
├── main.py
└── README.md

- `app/`: Contains the main modules for image generation, blending algorithms, and storyboard visualization.
- `assets/`: Directory to store the generated images and storyboards.
- `main.py`: Main script to orchestrate the advertisement asset generation pipeline.
- `README.md`: Documentation file explaining the project and usage instructions.

## Usage

To use the advertisement asset generation pipeline, follow these steps:

1. Ensure that Python and the required dependencies are installed.
2. Update the input dictionaries in `main.py` with your advertisement frame descriptions.
3. Run `main.py` to generate the advertisement assets.
4. The generated images and storyboards will be saved in the `assets/` directory.

## Dependencies

The project relies on the following dependencies:

- Python 3.10
- replicate
- Pillow
- requests

Install the dependencies using the following command:

```bash
pip install -r requirements.txt


