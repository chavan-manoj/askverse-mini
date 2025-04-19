from askverse_mini.ask_ensemble import AskEnsemble
from askverse_mini.document_processor import DocumentProcessor
from askverse_mini.ask_wiki import AskWiki
from askverse_mini.ask_tavily import AskTavily
from askverse_mini.ask_docs import AskDocs
from askverse_mini.ask_arxiv import AskArxiv

document_processor = None
def setup_document_processor():
    global document_processor
    if document_processor is None:
        document_processor = DocumentProcessor()
        document_processor.setup("pdfs")
    return document_processor

askverse_systems = {}
def setup_askverse_system(system: str):
    global askverse_systems
    if system in askverse_systems:
        return askverse_systems[system]

    if system == "wiki":
        askverse_system = AskWiki()
    elif system == "tavily":
        askverse_system = AskTavily()
    elif system == "arxiv":
        askverse_system = AskArxiv()
    elif system == "docs":
        askverse_system = AskDocs(document_processor=setup_document_processor())
    elif system == "ensemble":
        askverse_system = AskEnsemble(document_processor=setup_document_processor())
    
    askverse_system.initialize()
    askverse_systems[system] = askverse_system
    return askverse_system