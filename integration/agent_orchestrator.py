# integration/agent_orchestrator.py
class MARSAgent:
    def __init__(self, vm, rl_agent, llm):
        self.vm = vm
        self.rl_agent = rl_agent
        self.llm = llm

    def execute_query(self, image, question):
        # Step 1: LLM parses question
        context = self.llm.parse_question(question)
        # Output: {'entities': ['harbour'], 'task': 'count'}

        # Step 2: RL-guided exploration
        env = RSExplorationEnv(image, self.vm, context['entities'])
        obs = env.reset()
        for _ in range(MAX_STEPS):
            action = self.rl_agent.predict(obs)
            obs, _, done, _ = env.step(action)
            if done: break

        # Step 3: LLM answer generation
        detections = env.get_aggregated_detections()
        answer = self.llm.generate_answer(question, detections)
        return answer, detections

if __name__ == '__main__':
    agent = MARSAgent()