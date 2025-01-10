from dotenv import load_dotenv
import google.generativeai as genai
import os
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


genai.configure(api_key=GEMINI_API_KEY)

system_ins = """
You are a specialized assistant designed to break down complex instructions into simple, atomic actions. Given a sentence containing a detailed instruction, you must identify each simple, actionable step and return them as separate entries in a structured JSON format. Each step should be a single, independent action, such as "move," "turn," or "pick," and the output should be organized in a list.

Example Input: "To make the humanoid robot pick up an object, move forward, turn left, bend the arm, extend the hand, grip the object, lift the arm, move backward, and place the object down."

Expected Output:

json
Copy code
{
  "steps": [
    "Move forward",
    "Turn left",
    "Bend the arm",
    "Extend the hand",
    "Grip the object",
    "Lift the arm",
    "Move backward",
    "Place the object down"
  ]
}
"""
# prompt = """
# Move forward 2 meters, rotate 90 degrees to the right, then move forward 1.5 meters to reach the object. Once the object is within reach, extend your arm and grasp it with a 10 N force to ensure a secure grip. Afterward, rotate 180 degrees, move backward 1.5 meters, and then move forward 2 meters to the target location, placing the object precisely within a 10 cm radius of the designated spot.
# """
prompt = """
Move forward 2 meters, ensuring steady movement along the path. Upon completion, rotate 90 degrees to the right to align with the object. Then, move forward 1.5 meters, being mindful of your surroundings to avoid obstacles. As you approach, reach out with your arm and grasp the object with a controlled 10 N force to secure it firmly. Once you've established a stable hold, rotate 180 degrees, making sure the arm position is comfortable and precise. Move backward 1.5 meters, maintaining balance, and proceed forward 2 meters to the target location. Gently place the object within a 10 cm radius of the designated spot, ensuring careful handling throughout the task to avoid any sudden shifts or errors.
"""

model = genai.GenerativeModel("gemini-1.5-flash",system_instruction=system_ins)
response = model.generate_content(prompt)
print(response.text)