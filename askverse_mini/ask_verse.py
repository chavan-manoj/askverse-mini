import logging

logger = logging.getLogger(__name__)

class AskVerse:
    def __init__(self, askEnsemble, askAPIs):
        self.askEnsemble = askEnsemble
        self.askAPIs = askAPIs

    def initialize(self):
        self.askEnsemble.initialize()
        self.askAPIs.initialize()
        
    def ask(self, question: str):
        answer = self.askAPIs.ask(question)
        if not answer or answer.get("sources") is None or len(answer["sources"]) == 0:
            answer = self.askEnsemble.ask(question)

        return answer
