from typing import List
from image_generator import ImageGenerator
from algorithm import blend_assets
from storyboard_visualizer import StoryBoard

def main(frame_descriptions: dict):
    # Initialize the image generator
    image_generator = ImageGenerator()

    # Generate images based on frame descriptions
    generated_images = image_generator.generate_images(frame_descriptions)

    # Blend generated assets using the algorithm
    blended_assets = blend_assets(generated_images)

    # Generate frames from the blended assets
    image_composer = ImageComposer(600, 600, blended_assets)
    generated_frames = image_composer.generate_frames()

    # Combine generated frames horizontally
    combined_image = StoryBoard.combine_images_horizontally(generated_frames)

    # Display the combined image
    combined_image.show()

if __name__ == "__main__":
    # Example frame descriptions
    frame_descriptions = {
        "frame_1": {
            "Background": "A high-resolution 3D Coca-Cola bottle center-screen, bubbles rising to the top, transitioning into a sleek DJ turntable with a vinyl record that has the Coke Studio logo.",
            "CTA Button": "'Mix Your Beat' in bold, playful font pulsating to the rhythm of a subtle background beat, positioned at the bottom of the screen."
        },
        "frame_2": {
            "Interactive Elements": "A digital beat mixer interface with vibrant, touch-responsive Latin American instrument icons like congas, claves, and maracas, each activating a unique sound layer.",
            "Background": "A dynamic, abstract representation of sound waves that move in sync with the user's interactions."
        },
        "frame_3": {
            "Background": "A dynamic, abstract representation of sound waves that move in sync with the user's interactions.",
            "Animation": "A kaleidoscope of colors that dance across the screen, with each beat added, symbolizing the fusion of cultures and music.",
            "CTA Button": "A 'Play Your Mix' button that pulses like a heartbeat, encouraging users to share their creation."
        }
    }

    # Call the main function with frame descriptions
    main(frame_descriptions)
