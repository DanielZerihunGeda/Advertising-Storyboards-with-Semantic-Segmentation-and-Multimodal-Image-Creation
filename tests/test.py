import unittest
from main import generate_advertisement_storyboard

class TestAdvertisementStoryboardGeneration(unittest.TestCase):
    def test_storyboard_generation(self):
        # Define test input data
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
    },
    "explanation": "This variation emphasizes the joy and interactivity of music mixing, with each frame building on the last to create a crescendo of engagement. The 3D bottle-to-turntable animation captures attention, the interactive beat mixer sustains engagement, and the vibrant animations encourage sharing, aligning with the campaign's objectives of engagement and message recall."
}

        # Call the function to generate the advertisement storyboard
        generated_storyboard = generate_advertisement_storyboard(frame_descriptions)

        # Perform assertions to check if the generated storyboard is valid
        # You can add more specific assertions based on your project requirements

        # Check if the generated storyboard is a list
        self.assertIsInstance(generated_storyboard, list)

        # Check if each item in the generated storyboard is an image file
        for image_file in generated_storyboard:
            self.assertTrue(image_file.endswith('.png'))

if __name__ == '__main__':
    unittest.main()
    