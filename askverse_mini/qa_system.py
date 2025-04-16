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
from pydantic.v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Import ragas for evaluation
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall
)
from ragas import evaluate
from datasets import Dataset

from askverse_mini.document_processor import DocumentProcessor

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AgentState(BaseModel):
    """State for the agent workflow"""
    messages: List[Any]
    metrics: Dict[str, float] = Field(default_factory=dict)
    retrieved_docs: List[str] = Field(default_factory=list)
    question: str = ""

class AskVerse:
    """Main QA system implementing multi-agent architecture"""
    
    def __init__(self):
        """Initialize the QA system"""
        model_name = os.getenv("MODEL_NAME")
        self.llm = ChatOpenAI(model_name=model_name, temperature=0, streaming=True)
        self.agent_llm = ChatOpenAI(model_name=model_name, temperature=0, streaming=True)
        self.str_output_parser = StrOutputParser()
        self.tools = []
        self.graph = None
        
    def initialize(self, document_processor: DocumentProcessor = None, use_web_search: bool = True, retriever_kind: str = "ensemble", production_mode: bool = True):
        """
        Initialize the system with document processor and tools
        
        Args:
            document_processor: Optional DocumentProcessor instance
            use_web_search: Whether to use web search tool
            retriever_kind: Type of retriever to use (ensemble, dense, sparse)
        """
        if use_web_search:
            # Set up web search tool
            web_search = TavilySearchResults(max_results=4)
            self.tools.append(web_search)
        
        # Set up document retriever if provided
        if document_processor:
            self.document_processor = document_processor  # Store for later use
            self.retriever_kind = retriever_kind  # Store for direct access later
            retriever_tool = create_retriever_tool(
                document_processor.get_retriever(retriever_kind),
                "retrieve_document_answers",
                "Extensive information from the loaded documents."
            )
            self.tools.append(retriever_tool)
            
        # Set up the workflow graph
        if production_mode:
            self._setup_production_workflow()
        else:
            self._setup_workflow()
        
    def _setup_production_workflow(self):
        """Set up the multi-agent productionworkflow graph"""
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        retrieve = ToolNode(self.tools)
        workflow.add_node("retrieve", retrieve)
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
        
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compile the graph
        self.graph = workflow.compile()
    
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
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("evaluate_rag", self._evaluate_rag_node)
        
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
        workflow.add_edge("generate", "evaluate")
        workflow.add_edge("evaluate", "evaluate_rag")
        workflow.add_edge("evaluate_rag", END)
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
        
        # Save the initial question if this is the first message
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            state.question = messages[0].content
        
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
        
        # Store retrieved documents for RAG evaluation
        state.retrieved_docs.append(docs)
        
        # Generation prompt
        generation_prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer
            the question. If you don't know the answer, just say
            that you don't know. Provide a thorough description to
            fully answer the question, utilizing any relevant
            information you find.

            IMPORTANT: If the information in the context is outdated or from a previous year,
            explicitly mention this in your answer. For example:
            "According to the 2023 report..." or "Based on 2023 data..."

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
        return {"messages": [ai_message], "retrieved_docs": docs}
        
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
        
    def _evaluate_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for evaluating answer quality"""
        messages = state.messages
        question = messages[0].content
        answer = messages[-1].content
        
        # Evaluation model
        class Evaluation(BaseModel):
            relevance: float = Field(description="Relevance score between 0 and 1")
            completeness: float = Field(description="Completeness score between 0 and 1")
            coherence: float = Field(description="Coherence score between 0 and 1")
            
        # LLM with tool and validation
        llm_with_tool = self.llm.with_structured_output(Evaluation)
        
        # Prompt
        prompt = PromptTemplate(
            template="""
            You are evaluating the quality of an answer to a question. Rate the answer on three dimensions with specific criteria:

            CRITICAL: First, check if the question mentions a specific year (e.g., 2025). If it does, and the answer contains information from a different year (e.g., 2023), you MUST:
            1. Give a maximum relevance score of 0.6
            2. Give a maximum completeness score of 0.6
            3. Deduct additional points if the year difference is not acknowledged in the answer

            1. Relevance (0-1):
               - 0.0-0.3: Answer is completely off-topic, irrelevant, or contains outdated information without acknowledging it
               - 0.4-0.6: Answer is somewhat related but contains outdated information (even if acknowledged) or misses key aspects
               - 0.7-0.8: Answer addresses most aspects of the question with current information
               - 0.9-1.0: Answer perfectly matches the question's intent with current information
               STRICT RULE: If the question asks about a specific year and the answer contains information from a different year, maximum score is 0.6

            2. Completeness (0-1):
               - 0.0-0.3: Answer is missing critical information or only contains outdated information
               - 0.4-0.6: Answer covers basic information but lacks details or contains outdated information
               - 0.7-0.8: Answer provides most required information with current data
               - 0.9-1.0: Answer is comprehensive and complete with current information
               STRICT RULE: If the question asks about a specific year and the answer contains information from a different year, maximum score is 0.6

            3. Coherence (0-1):
               - 0.0-0.3: Answer is disorganized and hard to follow
               - 0.4-0.6: Answer has some structure but could be clearer
               - 0.7-0.8: Answer is well-structured and mostly clear
               - 0.9-1.0: Answer is perfectly organized and crystal clear

            Question: {question}
            Answer: {answer}

            IMPORTANT RULES:
            1. If the question mentions a specific year and the answer contains information from a different year:
               - Maximum relevance score: 0.6
               - Maximum completeness score: 0.6
               - Additional deduction if year difference is not acknowledged
            2. A perfect score (1.0) is ONLY possible if:
               - The information is current and matches the question's timeframe
               - The answer fully addresses the question
               - The information is complete and up-to-date
            3. Be extremely critical of temporal relevance - deduct points for any outdated information

            Provide scores between 0 and 1 for each dimension based on these criteria.
            Be critical and honest in your evaluation.
            """,
            input_variables=["question", "answer"],
        )
        
        # Chain
        chain = prompt | llm_with_tool
        
        # Evaluate
        logger.info("Evaluating answer quality...")
        evaluation = chain.invoke({"question": question, "answer": answer})
        
        # Calculate average score
        avg_score = (evaluation.relevance + evaluation.completeness + evaluation.coherence) / 3
        
        # Create metrics dictionary
        metrics = {
            "relevance": evaluation.relevance,
            "completeness": evaluation.completeness,
            "coherence": evaluation.coherence,
            "average_score": avg_score
        }
        
        state.metrics.update(metrics)
        
        return {
            "messages": messages,
            "metrics": metrics
        }
    
    def _evaluate_rag_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for evaluating RAG metrics using RAGAS"""
        messages = state.messages
        question = state.question if state.question else messages[0].content
        answer = messages[-1].content if isinstance(messages[-1], AIMessage) else messages[-1].content
        
        # Get the retrieved context
        context = " ".join(state.retrieved_docs) if state.retrieved_docs else ""
        
        try:
            # Create test data in the format required by RAGAS
            test_data = {
                "question": [question],
                "answer": [answer],
                "contexts": [[context]],
                "reference": [answer]  # Using answer as reference since we don't have ground truth
            }
            
            # Convert to Dataset format required by Ragas
            dataset = Dataset.from_dict(test_data)
            
            # Use the existing LLM instance
            logger.info(f"Using LLM model for RAGAS evaluation: {self.llm.model_name}")
            
            # Compute RAG metrics
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    context_entity_recall
                ],
                llm=self.llm
            )
            
            # Extract scores from the result
            rag_metrics = {}
            for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "context_entity_recall"]:
                metric_scores = result[metric_name]
                # Calculate average if it's a list of scores
                if isinstance(metric_scores, list):
                    rag_metrics[metric_name] = sum(metric_scores) / len(metric_scores)
                else:
                    rag_metrics[metric_name] = float(metric_scores)
            
            # Calculate average score
            average_score = sum(rag_metrics.values()) / len(rag_metrics)
            rag_metrics["average_score"] = average_score
            
            # Update state metrics
            state.metrics.update(rag_metrics)
            
            # Format answer with all metrics
            formatted_answer = f"""
{answer}

Quality Metrics:
- Faithfulness: {rag_metrics.get('faithfulness', 0):.4f}
- Answer Relevancy: {rag_metrics.get('answer_relevancy', 0):.4f}
- Context Precision: {rag_metrics.get('context_precision', 0):.4f}
- Context Recall: {rag_metrics.get('context_recall', 0):.4f}
- Context Entity Recall: {rag_metrics.get('context_entity_recall', 0):.4f}
- Average Score: {rag_metrics.get('average_score', 0):.4f}
"""
            
        except Exception as e:
            logger.error(f"Error evaluating RAG metrics: {str(e)}")
            rag_metrics = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "context_entity_recall": 0.0,
                "average_score": 0.0,
                "error": str(e)
            }
            
            # Update state metrics
            state.metrics.update(rag_metrics)
            
            # Format answer with error message
            formatted_answer = f"""
{answer}

Quality Metrics: Error calculating metrics - {str(e)}
"""
        
        return {
            "messages": [AIMessage(content=formatted_answer)],
            "metrics": state.metrics
        }
        
    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer
        
        Args:
            question: The question to ask
            
        Returns:
            str: The generated answer with quality metrics
            
        Raises:
            ValueError: If the system is not initialized or if no answer is generated
        """
        if not self.graph:
            raise ValueError("System not initialized. Call initialize() first.")
            
        # Create initial state
        initial_state = AgentState(messages=[HumanMessage(content=question)], question=question)
        
        # Run the graph
        final_answer = None
        metrics = {}
        for output in self.graph.stream(initial_state):
            logger.info(f"Processing output: {output}")
            
            # Handle nested output structure
            if isinstance(output, dict):
                # Check for 'evaluate_rag' key first since it contains the final formatted answer
                if 'evaluate_rag' in output and isinstance(output['evaluate_rag'], dict):
                    messages = output['evaluate_rag'].get('messages', [])
                    if messages:
                        last_message = messages[-1]
                        if isinstance(last_message, AIMessage):
                            final_answer = last_message.content
                            logger.info(f"Found evaluated answer with RAG metrics: {final_answer}")
                            if 'metrics' in output['evaluate_rag']:
                                metrics = output['evaluate_rag']['metrics']
                # Check for 'evaluate' key
                elif 'evaluate' in output and isinstance(output['evaluate'], dict):
                    messages = output['evaluate'].get('messages', [])
                    if messages:
                        last_message = messages[-1]
                        if isinstance(last_message, AIMessage):
                            final_answer = last_message.content
                            logger.info(f"Found evaluated answer: {final_answer}")
                            if 'metrics' in output['evaluate']:
                                metrics = output['evaluate']['metrics']
                # Check for 'generate' key
                elif 'generate' in output and isinstance(output['generate'], dict):
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

    def ask_with_metrics(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer
        
        Args:
            question: The question to ask
            
        Returns:
            str: The generated answer with quality metrics
            
        Raises:
            ValueError: If the system is not initialized or if no answer is generated
        """
        if not self.graph:
            raise ValueError("System not initialized. Call initialize() first.")
            
        # Create initial state
        initial_state = AgentState(messages=[HumanMessage(content=question)], question=question)
        
        # Run the graph
        final_answer = None
        for output in self.graph.stream(initial_state):
            logger.info(f"Processing output: {output}")
            
            # Handle nested output structure
            if isinstance(output, dict):
                # Check for 'evaluate_rag' key first since it contains the final formatted answer
                if 'generate' in output and isinstance(output['generate'], dict):
                    messages = output['generate'].get('messages', [])
                    retrieved_docs = output['generate'].get('retrieved_docs', "")
                    if messages:
                        last_message = messages[-1]
                        if isinstance(last_message, AIMessage):
                            final_answer = {"content": last_message.content, "retrieved_docs": retrieved_docs}
                            logger.info(f"Found answer: {final_answer}")

        if final_answer is None:
            raise ValueError("No answer was generated. Please try rephrasing your question.")
            
        return final_answer
            
