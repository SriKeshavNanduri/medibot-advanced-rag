import ast
import os
import re
import threading
import logging
import warnings
import urllib.parse
from dotenv import load_dotenv

from qdrant_client.http import models as rest
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_classic.chains import create_sql_query_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_community.utilities import SQLDatabase
from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from api.rbac import get_allowed_collections_for_role

# 1. Suppress library logging and warnings
logging.getLogger("semantic_router").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, "creds.env"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

reranker = CrossEncoderReranker(
    model=cross_encoder,
    top_n=3
)

sql_route = Route(
    name="sql_rag",
    utterances=[
        "What is the equipment name for ticket 1045?",
        "Find the patient name associated with claim ID 8821",
        "Show the status of ticket number 4410",
        "Get the resolution note for this specific issue id",
        "Which insurer is tied to claim ID 5521?",
        "Who is the patient for claim code 203?",
        "Identify the campus location for ticket 990",
        "Show me all maintenance tickets where status is open",
        "List all claims submitted by the cardiology department",
        "Find tickets assigned to the main campus location",
        "Filter all records matching fault code F-402",
        "Extract active issues raised by Arjun Desai",
        "Show me claims filed under Blue Cross or Aetna",
        "Give me a list of pending review status insurance filings",
        "What is the total claimed amount across all departments?",
        "Calculate the total approved amount for insurance last year",
        "Count the total maintenance issues raised for MRI machines",
        "Sum up the total policy payout approved for inpatient treatments",
        "Average time taken to resolve equipment faults",
        "What is the maximum claimed amount filed this month?",
        "What is the total volume of entries submitted in the last 30 days?",
        "Compute the difference between claimed amount and approved amount",
        "Give me a breakdown of issue types by campus location",
        "List the total number of issues raised by staff this month",
        "Show numerical metrics on approved claim costs grouped by insurer",
        "Which campus has the highest number of broken equipment reports?",
        "What is the most frequent diagnosis code used in our claims?",
        "What is the most common fault code for our equipment?",
        "Rank the departments by total approved insurance amounts",
        "List all resolved tickets between yesterday and today",
        "How many dental claims were submitted during last quarter?",
        "Show records raised after January 1st",
        "Find tickets that were resolved within the same day of being raised"
    ]
)

qdrant_route = Route(
    name="qdrant_rag",
    utterances=[
        "What is the company policy regarding sick leave eligibility?",
        "Explain the step-by-step process to apply for maternity leave",
        "Read me the code of conduct guidelines for hospital staff",
        "What are the official rules on medical document extensions?",
        "Where can I find the standard handbook protocols for clinical care?",
        "Explain our organizational guidelines for emergency coverage",
        "What does the policy manual say about long term health benefits?",
        "Show me the instructional guide text for submitting claims manually"
    ]
)

encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

router = SemanticRouter(
    encoder=encoder, 
    routes=[sql_route, qdrant_route],
    top_k=1,
    auto_sync="local"
)
router.sync(sync_mode="local")

my_postgres_pwd = os.getenv("POSTGRES_PWD")
db_name = os.getenv("DB_NAME")
safe_password = urllib.parse.quote_plus(str(my_postgres_pwd))

_SQL_URI = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/{db_name}"

SQL_DENIED_TABLES = {
    os.getenv("USERS_TABLE", "app_users"),
    os.getenv("CONTEXT_TABLE") or "medibot_chat_history",
    "chunk_history",
}


def _build_sql_database() -> SQLDatabase:
    _log = logging.getLogger(__name__)
    try:
        all_tables = set(SQLDatabase.from_uri(_SQL_URI).get_usable_table_names())
    except Exception as exc:
        _log.error("Could not reflect the database schema: %s", exc)
        raise

    allowed = sorted(all_tables - SQL_DENIED_TABLES)
    if not allowed:
        raise RuntimeError(
            f"No tables left for SQL RAG after excluding {sorted(SQL_DENIED_TABLES)}. "
            f"Found: {sorted(all_tables)}"
        )

    _log.info(
        "SQL RAG scope: %s (excluded: %s)",
        allowed,
        sorted(SQL_DENIED_TABLES & all_tables),
    )
    return SQLDatabase.from_uri(_SQL_URI, include_tables=allowed)


db = _build_sql_database()

POSTGRES_CONNECTION_STRING = f"postgresql://postgres:{safe_password}@localhost:5432/{db_name}"
TABLE_NAME = os.getenv("CONTEXT_TABLE") or "medibot_chat_history"

_VECTORSTORE = None
_VECTORSTORE_LOCK = threading.Lock()


def get_vectorstore() -> QdrantVectorStore:
    global _VECTORSTORE
    if _VECTORSTORE is None:
        with _VECTORSTORE_LOCK:
            if _VECTORSTORE is None:
                dense_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
                sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25", batch_size=256)
                _VECTORSTORE = QdrantVectorStore.from_existing_collection(
                    embedding=dense_embeddings,
                    sparse_embedding=sparse_embeddings,
                    url=os.getenv("LOCAL_QDRANT_URL"),
                    collection_name=os.getenv("COLLECTION_NAME"),
                    retrieval_mode=RetrievalMode.HYBRID,
                )
    return _VECTORSTORE


def get_postgres_session_history(session_id: str) -> PostgresChatMessageHistory:
    return PostgresChatMessageHistory(
        connection_string=POSTGRES_CONNECTION_STRING,
        session_id=session_id,
        table_name=TABLE_NAME
    )


class DirectConversationalRAG:
    """Standard, runnable-free Conversational RAG pipeline."""

    def __init__(self, compressed_retriever, llm: ChatGoogleGenerativeAI, user_role: str, allowed_collections: list[str] | None = None):
        self.compressed_retriever = compressed_retriever
        self.llm = llm
        self.user_role = user_role
        self.allowed_collections = allowed_collections or []
        self.system_prompt_template = (
            "You are a helpful Medibot customer support assistant.\n"
            "Answer the customer's question using ONLY the information provided in the context below.\n"
            "If the answer is not in the context, say \"I don't have that information.\"\n"
            "Keep answers concise and friendly.\n\n"
            "Context:\n{context}"
        )

    def _retrieve(self, query: str, chat_hist: list) -> list:
        logger = logging.getLogger("medibot.retrieval")
        logger.info("[DB History Read] Retrieved raw chat history messages count: %d", len(chat_hist))

        search_query = query
        if chat_hist:
            recent_turns = []
            for msg in chat_hist[-4:]:
                role_label = "User" if type(msg).__name__ == "HumanMessage" else "Assistant"
                recent_turns.append(f"{role_label}: {getattr(msg, 'content', '')}")
            history_context_str = " ".join(recent_turns)
            search_query = f"{history_context_str} Current Question: {query}"
            logger.info("[Retrieval] Combined search query with chat history context: '%s'", search_query)
        else:
            logger.info("[Retrieval] Executing hybrid search for query: '%s'", query)

        docs = self.compressed_retriever.invoke(search_query)
        # logger.info("[Retrieval] Retrieved documents before score filtering %s", docs)
        return docs

    def _generate_answer(self, query: str, chat_hist: list, docs: list) -> str:
        # Document stuffing logic
        context_text = "\n\n".join(doc.page_content for doc in docs)
        system_content = self.system_prompt_template.format(context=context_text)

        messages = [SystemMessage(content=system_content)]
        messages.extend(chat_hist)
        messages.append(HumanMessage(content=query))

        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            answer = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        else:
            answer = str(content)
        return answer


    def _cross_check(self, answer: str, query: str) -> bool:
        # Use Pydantic structured output for reliable boolean evaluation
        class Evaluation(BaseModel):
            is_valid: bool = Field(description="True if the answer fully addresses the user question, False otherwise.")

        try:
            structured_llm = self.llm.with_structured_output(Evaluation)
            prompt = (
                f"Evaluate whether the generated answer fully and accurately addresses the user's question.\n\n"
                f"User Question: {query}\n"
                f"Generated Answer: {answer}"
            )
            result = structured_llm.invoke(prompt)
            if isinstance(result, Evaluation):
                return result.is_valid
            elif isinstance(result, dict):
                return bool(result.get("is_valid", False))
            return False
        except Exception:
            # Fallback to chain with StrOutputParser if structured output is unsupported by the model version
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a strict evaluator. Respond with ONLY 'yes' or 'no'."),
                ("human", "Question: {query}\nAnswer: {answer}\nDoes the answer fully address the question? Respond with 'yes' or 'no'.")
            ])
            chain = prompt_template | self.llm | StrOutputParser()
            res = chain.invoke({"query": query, "answer": answer})
            return res.strip().lower() == "yes"

    def invoke_function(self, inputs: dict,) -> dict:
        query = inputs["input"]

        # Resolve chat history

        session_id = inputs.get("session_id")
        history_mgr = get_postgres_session_history(session_id)
        chat_hist = history_mgr.messages

        if len(chat_hist)==0   :
            chat_hist = []

        # 1. Retrieve
        docs = self._retrieve(query, chat_hist)

        if len(docs) == 0:
            allowed_str = ", ".join(self.allowed_collections) if self.allowed_collections else "your permitted sources"
            answer = f"I don't have that information. As a {self.user_role}, you can access only your respective sources ({allowed_str})."
            return {
                "answer": answer,
                "context": [],
            }

        # 2. Answer
        answer = self._generate_answer(query, chat_hist, docs)

        #3. Cross checking if answer fulfills the user query, if not then return a default answer 
        cross_check = self._cross_check(answer, query) 

        if not cross_check:
            allowed_str = ", ".join(self.allowed_collections) if self.allowed_collections else "your permitted sources"
            answer = f"I don't have that information. As a {self.user_role}, you can access only your respective sources ({allowed_str})."
            return {
                "answer": answer,
                "context": docs
            }


        # 4. Handle Postgres history persistence if state wrapper is enabled
        
        history_mgr = get_postgres_session_history(session_id)
        history_mgr.add_user_message(query)
        history_mgr.add_ai_message(answer)

        return {
            "answer": answer,
            "context": docs
        }


def setup_conversational_rag(
    user_role: str,
    allowed_collections: list[str] | None = None,
) -> DirectConversationalRAG:
    
    load_dotenv(os.path.join(_HERE, "creds.env"))
    if "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    vectorstore = get_vectorstore()

    must_conditions = [
        rest.FieldCondition(
            key="metadata.access_roles",
            match=rest.MatchAny(any=[user_role]),
        )
    ]
    if allowed_collections:
        must_conditions.append(
            rest.FieldCondition(
                key="metadata.collection",
                match=rest.MatchAny(any=list(allowed_collections)),
            )
        )
    qdrant_filter = rest.Filter(must=must_conditions)

    hybrid_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10, "filter": qdrant_filter}
    )

    compressed_retriever = ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
        base_compressor=reranker
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

    return DirectConversationalRAG(
        compressed_retriever=compressed_retriever,
        llm=llm,
        user_role=user_role,
        allowed_collections=allowed_collections,
    )


def _clean_sql(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^SQLQuery:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def _row_count(db_result) -> int | None:
    if not db_result:
        return 0
    try:
        parsed = ast.literal_eval(db_result)
    except (ValueError, SyntaxError):
        return None
    return len(parsed) if isinstance(parsed, (list, tuple)) else None


def smart_router_agent(
    question: str,
    session_id: str,
    chain: DirectConversationalRAG,
    role: str ,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"\nUser: {question}")

    route_result = router(question)

    detected_route = route_result.name
    confidence_score = getattr(route_result, "similarity_score", None) 

    if detected_route is None: 
        detected_route = "qdrant_rag"

    if verbose:
        print("DEBUG Raw Route Result:", repr(route_result))
        if confidence_score is not None:
            print("confidence_score", confidence_score)

        print(f"[Router Decision] -> Detected Intent Route: {detected_route}")

    def qdrant_rag_path():
        
        result = chain.invoke_function({"input": question, "session_id": session_id} ) 

        documents = result.get("context") or []

        if verbose:
            print(f"Bot: {result['answer']}")
            print("-" * 40)
            print("Sources Used:")
            for doc in documents:
                print(f" {doc.metadata.get('source_document', 'unknown')}")
            print("-" * 40)

        return {
            "route": "qdrant_rag",
            "answer": result["answer"],
            "documents": documents,
            "sql_query": None,
            "sql_result": None,
            "row_count": None,
            "confidence": confidence_score,
        }
    
    if detected_route == "qdrant_rag":
        return qdrant_rag_path()

    if detected_route == "sql_rag" and confidence_score is not None and confidence_score >= 0.55:
        try:
            if verbose:
                print("Directing query to SQL RAG pipeline...")

            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
            
            # Text-to-SQL generation


            write_query_chain = create_sql_query_chain(llm, db)
            raw_sql = write_query_chain.invoke({"question": question})
            sql_query = _clean_sql(raw_sql)

            prompt_template = ChatPromptTemplate.from_messages([
                            ("system", "you are an sql expert and a strict evaluator. You respond with table name based on sql_query. You respond with ONLY 'claims', 'maintenance_tickets'"),
                            ("human", "Question: {sql_query}\n which table sql query is referring to? Respond with 'claims' or 'maintenance_tickets'.")
                        ])

            chain = prompt_template | llm | StrOutputParser()

            res = chain.invoke({"query": sql_query})

            table_name =  res.strip().lower() 

            if table_name =='claims' and role not in ['billing_executive', 'admin']:
            
                allowed_str = ", ".join(get_allowed_collections_for_role(role)) if get_allowed_collections_for_role(role) else "your permitted sources"
                restricted_answer = f"I don't have that information. As a {role}, you can access only your respective sources ({allowed_str})."
                return {
                                "route": "sql_rag",
                                "answer": restricted_answer,
                                "documents": [],
                                "sql_query": sql_query,
                                "sql_result": db_result,
                                "row_count": 0,
                                "confidence": confidence_score,
                            }

            if table_name =='maintenance_tickets' and role not in ['technician', 'admin']:
                allowed_str = ", ".join(get_allowed_collections_for_role(role)) if get_allowed_collections_for_role(role) else "your permitted sources"
                restricted_answer = f"I don't have that information. As a {role}, you can access only your respective sources ({allowed_str})."
                return {
                                "route": "sql_rag",
                                "answer": restricted_answer,
                                "documents": [],
                                "sql_query": sql_query,
                                "sql_result": db_result,
                                "row_count": 0,
                                "confidence": confidence_score,
                            }
            
            if verbose:
                print(f"[SQL RAG] Generated SQL Query: {sql_query}")

            # SQL execution
            db_result = db.run(sql_query)
            
            if verbose:
                print(f"[SQL RAG] Query Result: {db_result}")

            # Answer synthesis without LCEL pipes
            prompt_content = (
                f"Based on the user's question: '{question}', the SQL query: '{sql_query}', "
                f"and the result: '{db_result}', formulate a friendly, natural language answer."
            )
            response = llm.invoke([HumanMessage(content=prompt_content)])
            content = response.content
            if isinstance(content, list):
                final_answer = "".join(item.get("text", "") for item in content if isinstance(item, dict))
            else:
                final_answer = str(content)

            if verbose:
                print(f"Bot: {final_answer}")
                print("-" * 40)

            return {
                "route": "sql_rag",
                "answer": final_answer,
                "documents": [],
                "sql_query": sql_query,
                "sql_result": db_result,
                "row_count": _row_count(db_result),
                "confidence": confidence_score,
            }
        except Exception:
            return qdrant_rag_path()
    else: 
        return qdrant_rag_path()