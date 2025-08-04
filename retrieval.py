from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage
import os


class RAGRetriever(BaseRetriever):
    rag_system: Any
    top_k: int = 10
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        results = self.rag_system.search(query, top_k=self.top_k)
        
        docs = []
        for result in results:
            filename = result.get('filename', 'Unknown')
            page_number = result.get('page_number', 'Unknown')
            original_text = result.get('text', '')
            
            enhanced_content = f"{original_text}\n\n[Source: {filename}, Page: {page_number}]"
            
            metadata = {
                'rank': result.get('rank'),
                'score': result.get('score'),
                'filename': filename,
                'page_number': page_number,
                'chunk_id': result.get('chunk_id'),
                'chunk_type': result.get('chunk_type'),
                'text_length': result.get('text_length'),
                'original_text': result.get('original_text')
            }
            doc = Document(page_content=enhanced_content, metadata=metadata)
            docs.append(doc)
        
        return docs


class SimpleRAG:
    
    def __init__(self, retriever, llm=None, openai_api_key=None, model_name="o4-mini"):
        self.retriever = retriever
        
        # Set OpenAI API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = llm or ChatOpenAI(model=model_name)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.prompt = PromptTemplate(
            template="""You are a precise information retrieval assistant that ONLY answers questions using the exact information provided in the context. You must follow these strict rules:

CRITICAL RULES:
1. ONLY use information explicitly stated in the provided context
2. NEVER generate, infer, or assume information not directly present in the context
3. If the answer is not in the context, you MUST say "No answer found"
4. Every answer must cite the exact source using the template below
5. Keep you output in markdown only format.

Greetings Type of queries:
- if you are getting greetings type of query then deal with it accordingly.
- Like hello, hi, thank you, goodbye etc type of questions should be answered accordingly.
- For greeting type of queries do not mention source

ANSWER TEMPLATE (use exactly this format):
[Your answer based solely on context]

Source: [filename] -> page: [page number]

For multiple sources:
[Your answer]

Source: [filename1] -> page: [page number], [filename2] -> page: [page number]

EXAMPLES:

Example 1 (Information found in context):
Question: "What is the company's revenue for 2023?"
Response:
ABC Corp reported revenue of $45 million in 2023.

Source: annual_report.pdf -> page: 12

Example 2 (Information NOT in context):
Question: "What is the company's profit margin?"
Response:
I don't know - this information is not available in the provided context.


Example 3 (Partial information):
Question: "What are the company's revenue and expenses for 2023?"
Response:
The company's revenue for 2023 was $45 million. However, I don't know the expenses as this information is not available in the provided context.

Source: annual_report.pdf -> page: 12

INSTRUCTIONS:
- Read the entire context carefully before answering
- Quote or paraphrase ONLY what is explicitly stated
- Do not combine information from your general knowledge
- If asked about information not in the context, always respond with "I don't know - this information is not available in the provided context"
- Be concise but include all relevant information from the context
- If the previous conversation contains relevant context, you may reference it, but still cite sources
- Always use lowercase "answer:" and "source:" in your response

Context from documents:
{context}

Previous conversation:
{chat_history}

Current question: {question}

Answer:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            verbose=False
        )
    
    def ask(self, question: str):
        response = self.chain({"question": question})
        return {
            "answer": response["answer"],
            "sources": response["source_documents"],
            "chat_history": response["chat_history"]
        }
    
    def clear_memory(self):
        self.memory.clear()