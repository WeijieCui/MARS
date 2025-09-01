from openai import OpenAI

API_KEY = 'sk-bcea760b4a8b4e8e9cdcf5fdbcc18548'
BASE_URL = "https://api.deepseek.com"


class DeepSeekChat:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def extract_context(self, question):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    'role': 'system',
                    'content': """Please help me to extract the objects from the question.
                    The objects should within the list: [plane, ship, storage-tank, baseball-diamond,
                    tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle,
                    small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool].
                    Only to return the name of objects in a list using [].
                    The question is: {}""".format(question)
                }
            ],
            stream=False
        )
        content = response.choices[0].message.content
        return content[content.index('[')+1:content.index((']'))].replace('"', '').replace(' ', '').split(',')

    def next_action(self, context):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    'role': 'system',
                    'content': """Please help me to extract the objects from the question.
                    The objects should within the list: [plane, ship, storage tank, baseball diamond,
                    tennis court, basketball court, ground track field, harbor, bridge, large vehicle,
                    small vehicle, helicopter, roundabout, soccer ball field, swimming pool].
                    Only to include the name of objects.
                    The context is: {}""".format(context)
                }
            ],
            stream=False
        )
        return response.choices[0].message.content

    def answer(self, question, observation):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    'role': 'system',
                    'content': """Please help me to extract the objects from the question.
                    The objects should within the list: [plane, ship, storage tank, baseball diamond,
                    tennis court, basketball court, ground track field, harbor, bridge, large vehicle,
                    small vehicle, helicopter, roundabout, soccer ball field, swimming pool].
                    Only to include the name of objects.
                    The question is: {}.
                    The observations are: {}""".format(question, observation)
                }
            ],
            stream=False
        )
        return response.choices[0].message.content
