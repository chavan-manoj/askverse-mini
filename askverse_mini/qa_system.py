"""
Main QA system module implementing the multi-agent architecture
"""

import os
import logging
from typing import Dict, Any, List, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AgentState(BaseModel):
    """State for the agent workflow"""
    messages: List[Any]

class AskVerse:
    """Main QA system implementing multi-agent architecture"""
    
    def __init__(self):
        """Initialize the QA system"""
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
        self.agent_llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
        self.str_output_parser = StrOutputParser()
        self.tools = []
        self.graph = None
        
    def initialize(self, document_processor: DocumentProcessor = None):
        """
        Initialize the system with document processor and tools
        
        Args:
            document_processor: Optional DocumentProcessor instance
        """
        # Set up web search tool
        web_search = TavilySearchResults(max_results=4)
        self.tools.append(web_search)
        
        # Set up document retriever if provided
        if document_processor:
            retriever_tool = create_retriever_tool(
                document_processor.get_retriever(),
                "retrieve_document_answers",
                "Extensive information from the loaded documents."
            )
            self.tools.append(retriever_tool)
            
        # Set up the workflow graph
        self._setup_workflow()
        
    def _setup_workflow(self):
        """Set up the multi-agent workflow graph"""
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        retrieve = ToolNode(self.tools)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("improve", self._improve_node)
        workflow.add_node("generate", self._generate_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            }
        )
        
        workflow.add_conditional_edges("retrieve", self._score_documents)
        workflow.add_edge("generate", END)
        workflow.add_edge("improve", "agent")
        
        # Compile the graph
        self.graph = workflow.compile()
        
    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent node for decision making"""
        messages = state.messages
        
        # Add system message to guide the agent
        system_message = SystemMessage(content="""You are a helpful AI assistant that can use various tools to find and provide information.
        Your goal is to help answer questions by using the available tools to gather relevant information.
        Always try to use the tools to get more information before providing an answer.
        If you have enough information to answer the question, you can end the conversation.
        If you need more information, use the appropriate tool to get it.""")
        
        # Combine messages with system message
        full_messages = [system_message] + messages
        
        # Bind tools to LLM
        llm = self.agent_llm.bind_tools(self.tools)
        
        # Get response
        logger.info("Agent processing question...")
        response = llm.invoke(full_messages)
        logger.info(f"Agent response: {response}")
        
        return {"messages": [response]}
        
    def _improve_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for improving queries"""
        messages = state.messages
        question = messages[0].content
        
        msg = [
            HumanMessage(content=f"""
                Look at the input and try to reason about the underlying semantic intent / meaning.
                
                Here is the initial question:
                -------
                {question}
                -------
                
                Formulate an improved question:
                """)
        ]
        
        response = self.llm.invoke(msg)
        return {"messages": [response]}
        
    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for generating final answers"""
        messages = state.messages
        question = messages[0].content
        docs = messages[-1].content
        
        # Generation prompt
        generation_prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer
            the question. If you don't know the answer, just say
            that you don't know. Provide a thorough description to
            fully answer the question, utilizing any relevant
            information you find.

            Question: {question}
            Context: {context}

            Answer:"""
        )
        
        # Chain
        rag_chain = generation_prompt | self.llm | self.str_output_parser
        
        # Run
        logger.info("Generating final answer...")
        response = rag_chain.invoke({"context": docs, "question": question})
        logger.info(f"Generated answer: {response}")
        
        # Create AIMessage with the response
        ai_message = AIMessage(content=response)
        return {"messages": [ai_message]}
        
    def _score_documents(self, state: AgentState) -> Literal["generate", "improve"]:
        """Score retrieved documents for relevance"""
        messages = state.messages
        last_message = messages[-1]
        question = messages[0].content
        docs = last_message.content
        
        # Scoring model
        class Scoring(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")
            
        # LLM with tool and validation
        llm_with_tool = self.llm.with_structured_output(Scoring)
        
        # Prompt
        prompt = PromptTemplate(
            template="""
            You are assessing relevance of a retrieved document to a user question with a binary grade.
            
            Here is the retrieved document:
            {context}
            
            Here is the user question: {question}
            
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
            
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            """,
            input_variables=["context", "question"],
        )
        
        # Chain
        chain = prompt | llm_with_tool
        
        # Score
        logger.info("Scoring documents for relevance...")
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score
        logger.info(f"Document relevance score: {score}")
        
        return "generate" if score == "yes" else "improve"
        
    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer
        
        Args:
            question: The question to ask
            
        Returns:
            str: The generated answer
            
        Raises:
            ValueError: If the system is not initialized or if no answer is generated
        """
        if not self.graph:
            raise ValueError("System not initialized. Call initialize() first.")
            
        # Create initial state
        initial_state = AgentState(messages=[HumanMessage(content=question)])
        
        # Run the graph
        final_answer = None
        for output in self.graph.stream(initial_state):
            logger.info(f"Processing output: {output}")
            
            # Handle nested output structure
            if isinstance(output, dict):
                # Check for 'generate' key
                if 'generate' in output and isinstance(output['generate'], dict):
                    messages = output['generate'].get('messages', [])
                    if messages:
                        last_message = messages[-1]
                        if isinstance(last_message, AIMessage):
                            final_answer = last_message.content
                            logger.info(f"Found answer: {final_answer}")
                        elif isinstance(last_message, str):
                            final_answer = last_message
                            logger.info(f"Found string answer: {final_answer}")
                # Check for direct messages
                elif 'messages' in output:
                    messages = output['messages']
                    if messages:
                        last_message = messages[-1]
                        if isinstance(last_message, AIMessage):
                            final_answer = last_message.content
                            logger.info(f"Found answer: {final_answer}")
                        elif isinstance(last_message, str):
                            final_answer = last_message
                            logger.info(f"Found string answer: {final_answer}")
                
        if final_answer is None:
            raise ValueError("No answer was generated. Please try rephrasing your question.")
            
        return final_answer 