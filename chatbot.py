from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import logging
from retrieval import SimpleRAG, RAGRetriever
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    original_query: str
    chat_history: List
    rewritten_query: str
    classification: str
    response: str


class AgenticRAG:
    
    def __init__(self, retriever, llm=None, openai_api_key=None, model_name="gpt-4o-mini"):
        # Set OpenAI API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
        self.llm = llm or ChatOpenAI(model=model_name)
        self.simple_rag = SimpleRAG(retriever, self.llm)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        self.rewrite_prompt = PromptTemplate(
            template="Given the chat history and current question, rewrite the question to be standalone.\n\nChat History: {chat_history}\nCurrent Question: {query}\nStandalone Question:",
            input_variables=["chat_history", "query"]
        )
        self.rewrite_chain = LLMChain(llm=self.llm, prompt=self.rewrite_prompt)
        
        self.classify_prompt = PromptTemplate(
            template="""You are a query classifier. Classify each query into exactly one category.

GREETING: Social interactions, pleasantries, thanks, goodbyes - queries that don't need document information
DOCUMENT: Questions that require searching through documents for specific information

Examples:
Query: "Hello" -> GREETING
Query: "Hi there!" -> GREETING  
Query: "How are you?" -> GREETING
Query: "Thanks" -> GREETING
Query: "Thank you" -> GREETING
Query: "Goodbye" -> GREETING
Query: "What is Phil's phone number?" -> DOCUMENT
Query: "Tell me about the data protection law" -> DOCUMENT
Query: "Who works at PwC?" -> DOCUMENT
Query: "Show me contact information" -> DOCUMENT

Query: {query}
Classification:""",
            input_variables=["query"]
        )
        self.classify_chain = LLMChain(llm=self.llm, prompt=self.classify_prompt)
        
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
    
    def _rewrite_node(self, state):
        logger.info(f"🔄 REWRITE NODE - Original: '{state['original_query']}'")
        
        chat_history = state.get("chat_history", [])
        
        if not chat_history:
            rewritten_query = state["original_query"]
            logger.info(f"   No history - keeping original query")
        else:
            history_str = "\n".join([f"Human: {msg.content}" if hasattr(msg, 'content') else str(msg) for msg in chat_history[-6:]])
            rewritten_query = self.rewrite_chain.run(
                chat_history=history_str,
                query=state["original_query"]
            ).strip()
            logger.info(f"   Rewritten with history")
        
        logger.info(f"   Rewritten: '{rewritten_query}'")
        return {"rewritten_query": rewritten_query}
    
    def _classify_node(self, state):
        logger.info(f"🎯 CLASSIFY NODE - Query: '{state['rewritten_query']}'")
        
        result = self.classify_chain.run(query=state["rewritten_query"])
        classification = result.strip().lower()
        
        # Extract just the classification word
        if "greeting" in classification:
            classification = "greeting"
        elif "document" in classification:
            classification = "document"
        else:
            logger.info(f"   Invalid classification '{classification}' - defaulting to 'document'")
            classification = "document"
        
        logger.info(f"   Classification: '{classification}'")
        return {"classification": classification}
    
    def _greeting_node(self, state):
        logger.info(f"👋 GREETING NODE - Processing: '{state['rewritten_query']}'")
        
        greeting_prompt = PromptTemplate(
            template="Respond to this greeting in a friendly, helpful way. Keep it brief and offer assistance.\n\nGreeting: {query}\nResponse:",
            input_variables=["query"]
        )
        greeting_chain = LLMChain(llm=self.llm, prompt=greeting_prompt)
        
        response = greeting_chain.run(query=state["rewritten_query"])
        logger.info(f"   Generated response: '{response.strip()}'")
        return {"response": response.strip()}
    
    def _rag_node(self, state):
        logger.info(f"📚 RAG NODE - Processing: '{state['original_query']}'")
        
        result = self.simple_rag.ask(state["original_query"])
        logger.info(f"   RAG response generated")
        return {"response": result["answer"]}
    
    def _route_query(self, state):
        route = state["classification"]
        logger.info(f"🔀 ROUTING to: '{route}' node")
        return route
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("rewrite", self._rewrite_node)
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("greeting", self._greeting_node)
        workflow.add_node("rag", self._rag_node)
        
        workflow.set_entry_point("rewrite")
        
        workflow.add_edge("rewrite", "classify")
        
        workflow.add_conditional_edges(
            "classify",
            self._route_query,
            {
                "greeting": "greeting",
                "document": "rag"
            }
        )
        
        workflow.add_edge("greeting", END)
        workflow.add_edge("rag", END)
        
        return workflow
    
    def ask(self, question: str):
        chat_history = self.memory.chat_memory.messages
        
        result = self.app.invoke({
            "original_query": question,
            "chat_history": chat_history
        })
        
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(result["response"])
        
        return result["response"]
    
    def clear_memory(self):
        self.simple_rag.clear_memory()
        self.memory.clear()