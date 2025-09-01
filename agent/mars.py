from agent.envs2 import Env
from language_model.deepseek_chat import DeepSeekChat
from utils.data_loader import get_data_loader
from visual_model.yolo import Yolo11


class Mars:
    def __init__(self, language_model, visual_model, coordinate_model):
        self.lm = language_model
        self.vm = visual_model
        self.cm = coordinate_model
        self.env = None

    def answer(self, image, question, ground_truth):
        target_classes = self.lm.extract_context(question)
        self.env = Env(image, self.vm, target_classes, ground_truth)
        done = False
        observation = None
        while not done:
            action = self.cm.next_action(self.env)
            new_state, reward, done, observation = self.env.step(action)
        return self.lm.answer(question, observation)


def main():
    mars = Mars(DeepSeekChat(), Yolo11(), DeepSeekChat())
    dataloader = get_data_loader('../data/train', batch_size=1)
    for batch_idx, (images, labels) in enumerate(dataloader):
        for image, label in zip(images, labels):
            for question in label['questions']:
                result = mars.answer(images[0], question, {
                    "boxes": label['boxes'],
                    'labels': label['labels']
                })
                print(result)


if __name__ == '__main__':
    main()
