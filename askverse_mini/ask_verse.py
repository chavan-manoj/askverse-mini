import logging

logger = logging.getLogger(__name__)

class AskVerseSimple:
    """
    AskVerseSimple is simplistic implementation that just delegates the 
    questions to APIs, and if APIs cannot answer then delegates it to Ensemble search.
    """

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

class AskVerse:
    """
    AskVerse is a smart AI system that identifies which tools/APIs are suitable, and execute only required ones.
    """
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