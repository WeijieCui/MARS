from openai import OpenAI

key = 'sk-bcea760b4a8b4e8e9cdcf5fdbcc18548'
client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            'role': 'system',
            'content': """You simulate an image robot and act as the brain of the robot to provide command support and
             answer visual questions. The robot is responsible for viewing the same image multiple times. 
             Please only answer the command. The image module is responsible for reading the image content and
              returning the information to you. However, due to the limited capabilities of the image module, 
              it can only view targets of comparable size. But you can use commands to adjust the model's visual area
               and zoom to discover more content. You need to determine whether the information meets the problem.
                If it does, output the command: "stop", and answer the question.
                 Otherwise, output other commands to guide the image module
                 to see more targets clearly. The commands include: 
                 ["left_top", "left_bottom", "right_bottom", "right_top", "zoom_in", "zoom_out", "stop"]. 
                 There is a picture of size 20 * 20.
                 The question is: How many red planes are there in the picture and where are they."""
        },
        {"role": "user", "content": """[{"object":"plane", "x": 3,"y": 3, "confidence": "0.3", "color": "green"},
                                    {"object":"car", "x": 8,"y": 3, "confidence": "0.3", "color": "blue"}]"""},
        {"role": "system", "content": "zoom_in"},
        {"role": "user", "content": """[{"object":"plane", "x": 3,"y": 3, "confidence": "0.8", "color": "green"},
                                    {"object":"plane", "x": 5,"y": 5, "confidence": "0.3", "color": "red"},
                                    {"object":"car", "x": 8,"y": 3, "confidence": "0.7", "color": "blue"}]"""},
        # {"role": "system", "content": "zoom_in"},
        # {"role": "user", "content": '[{"object":"plane", "x": 10,"y": 10, "confidence": "0.6", "color": "red"}]'},
        # {"role": "system", "content": "right_bottom"},
        # {"role": "user", "content": '[{"object":"plane", "x": 15,"y": 8, "confidence": "0.6", "color": "red"}]'},
        # {"role": "system", "content": """stop
        #                                 There are 3 red planes in the picture located at:
        #                                 - (5, 5)
        #                                 - (10, 10)
        #                                 - (15, 8)"""},
    ],
    stream=False
)

print(response.choices[0].message.content)
