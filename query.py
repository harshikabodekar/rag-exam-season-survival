from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
main = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, max_retries=2)
backup1 = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, max_retries=2)
backup2 = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, max_retries=2)
llm = main.with_fallbacks([backup1, backup2])

PROMPT = """You are an exam-prep assistant. Answer ONLY using the notes below.
If the answer is not in the notes, say exactly: "not in your notes".
Keep it short and clear.

NOTES:
{context}

QUESTION: {question}"""

while True:
    question = input("\nask (or type q to quit): ")
    if question.strip().lower() == "q":
        break

    results = db.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in results)

    try:
        answer = llm.invoke(PROMPT.format(context=context, question=question))
        text = answer.text
    except Exception:
        print("\ngemini is busy right now, try again in a minute")
        continue

    print("\n" + text)

    if "not in your notes" not in text.lower():
        print("\nsources:")
        seen = set()
        for doc in results:
            name = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", 0) + 1
            if (name, page) not in seen:
                seen.add((name, page))
                print(f"  - {name}, page {page}")