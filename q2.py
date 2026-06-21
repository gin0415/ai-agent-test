import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using only the context below. "
     "If it doesn't cover the question, say you don't know.\n\n{context}"),
    ("human", "{question}"),
])

def build_kb(docs_dir: str) -> Chroma:
    docs = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)
    return Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))

def answer(kb: Chroma, question: str) -> str:
    context = "\n\n".join(d.page_content for d in kb.similarity_search(question, k=4))
    chain = PROMPT | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return chain.invoke({"context": context, "question": question}).content

if __name__ == "__main__":
    docs_dir, question = sys.argv[1], sys.argv[2]
    print(answer(build_kb(docs_dir), question))
