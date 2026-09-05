"""
RAGAS Retriever Evaluation for Agentic Trading Chatbot
=======================================================
Evaluates the Pinecone retriever using RAGAS metrics:
- Context Precision  : Are retrieved chunks relevant to the question?
- Context Recall     : Did retriever find all relevant chunks?

Questions are auto-generated from the actual document:
    notebook/stock_market_investing_guide.docx

Usage:
    cd "Agentic Trading Chatbot"
    python -m evaluations.ragas_eval
"""

import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pinecone import Pinecone
from utils.model_loaders import ModelLoader
from utils.config_loader import load_config
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI as LangchainChatOpenAI

# ── Config ────────────────────────────────────────────────────────────────────

config = load_config()
model_loader = ModelLoader()

DOCX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebook",
    "stock_market_investing_guide.docx"
)

NUM_QUESTIONS = 20  # number of questions to auto-generate

# ── Step 1: Load Document ─────────────────────────────────────────────────────

def load_document():
    print("\n[1/4] Loading document...")
    loader = Docx2txtLoader(DOCX_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"      Loaded {len(chunks)} chunks from document")
    return chunks

# ── Step 2: Auto-generate QA pairs from document chunks ──────────────────────

def generate_qa_pairs(chunks):
    print(f"\n[2/4] Auto-generating {NUM_QUESTIONS} QA pairs from document...")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Pick evenly spaced chunks to cover the whole document
    step = max(1, len(chunks) // NUM_QUESTIONS)
    selected_chunks = [chunks[i * step] for i in range(NUM_QUESTIONS)]

    qa_pairs = []

    for i, chunk in enumerate(selected_chunks):
        prompt = f"""Based on the following text from a stock market investing guide, 
generate generate ONE difficult, paraphrased question 
that is not directly copied from the text and may require reasoning that can be answered using ONLY this text.
Then provide the answer based ONLY on this text.

Text:
{chunk.page_content}

Respond in this exact JSON format with no extra text:
{{
    "question": "your question here",
    "answer": "your answer here"
}}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            # Clean response and parse JSON
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            qa = json.loads(content)
            qa["context"] = chunk.page_content
            qa_pairs.append(qa)
            print(f"      ✅ [{i+1}/{NUM_QUESTIONS}] Q: {qa['question'][:70]}...")
        except Exception as e:
            print(f"      ⚠  [{i+1}/{NUM_QUESTIONS}] Skipped — {str(e)[:50]}")
            continue

    print(f"      Generated {len(qa_pairs)} QA pairs successfully")
    return qa_pairs

# ── Step 3: Retrieve contexts from Pinecone ───────────────────────────────────

def retrieve_contexts(qa_pairs):
    print("\n[3/4] Retrieving contexts from Pinecone...")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    vector_store = PineconeVectorStore(
        index=pc.Index(config["vector_db"]["index_name"]),
        embedding=model_loader.load_embeddings()
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config["retriever"]["top_k"]}
    )

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        retrieved_chunks = retriever.invoke(question)
        retrieved_texts = [doc.page_content for doc in retrieved_chunks]

        questions.append(question)
        answers.append(qa["answer"])
        contexts.append(retrieved_texts)
        ground_truths.append(qa["context"])

        print(f"      [{i+1}/{len(qa_pairs)}] Retrieved {len(retrieved_texts)} chunks for: {question[:60]}...")

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

# ── Step 4: Run RAGAS Evaluation ──────────────────────────────────────────────

def run_ragas_evaluation(ragas_data):
    print("\n[4/4] Running RAGAS evaluation...")

    dataset = Dataset.from_dict(ragas_data)

    evaluator_llm = LangchainLLMWrapper(LangchainChatOpenAI(model="gpt-4o-mini"))

    results = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
        ]
    )

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("   RAGAS RETRIEVER EVALUATION")
    print("   Agentic Trading Chatbot")
    print("="*60)
    print(f"   Document : stock_market_investing_guide.docx")
    print(f"   Index    : {config['vector_db']['index_name']}")
    print(f"   Top K    : {config['retriever']['top_k']}")
    print("="*60)

    # Step 1 — Load document
    chunks = load_document()

    # Step 2 — Auto-generate QA pairs
    qa_pairs = generate_qa_pairs(chunks)

    if not qa_pairs:
        print("\n❌ No QA pairs generated. Check OpenAI API key.")
        return

    # Step 3 — Retrieve contexts
    ragas_data = retrieve_contexts(qa_pairs)

    # Step 4 — Run RAGAS
    results = run_ragas_evaluation(ragas_data)

    # ── Print Results ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("   RAGAS EVALUATION RESULTS")
    print("="*60)

    df = results.to_pandas()

    prec_col = [c for c in df.columns if "precision" in c.lower()][0]
    rec_col  = [c for c in df.columns if "recall" in c.lower()][0]
    context_prec = df[prec_col].mean()
    context_rec  = df[rec_col].mean()

    print(f"   Context Precision : {round(context_prec, 4)}")
    print(f"   Context Recall    : {round(context_rec, 4)}")
    print("="*60)

    # ── Interpretation ─────────────────────────────────────────────────────────
    print("\n   INTERPRETATION")
    print("-"*60)

    if context_prec >= 0.7:
        print(f"   ✅ Context Precision {round(context_prec, 2)} — Retrieved chunks are highly relevant")
    elif context_prec >= 0.4:
        print(f"   ⚠  Context Precision {round(context_prec, 2)} — Some irrelevant chunks being retrieved")
    else:
        print(f"   ❌ Context Precision {round(context_prec, 2)} — Retriever returning mostly irrelevant chunks")

    if context_rec >= 0.7:
        print(f"   ✅ Context Recall {round(context_rec, 2)} — Retriever finding most relevant content")
    elif context_rec >= 0.4:
        print(f"   ⚠  Context Recall {round(context_rec, 2)} — Retriever missing some relevant chunks")
    else:
        print(f"   ❌ Context Recall {round(context_rec, 2)} — Retriever missing most relevant content")

    print("="*60)

    # ── Suggestions ────────────────────────────────────────────────────────────
    print("\n   SUGGESTIONS TO IMPROVE")
    print("-"*60)
    if context_prec < 0.7:
        print("   → Lower score_threshold in config.yaml (try 0.3)")
        print("   → Reduce chunk_size for more precise chunks")
    if context_rec < 0.7:
        print("   → Increase top_k in config.yaml (try 10-15)")
        print("   → Increase chunk_overlap for better coverage")
    if context_prec >= 0.7 and context_rec >= 0.7:
        print("   → Retriever is performing well! No changes needed.")
    print("="*60)

    # ── Save Results ───────────────────────────────────────────────────────────
    output = {
        "summary": {
            "context_precision": round(context_prec, 4),
            "context_recall": round(context_rec, 4),
        },
        "per_question": df.to_dict(orient="records")
    }

    output_path = os.path.join(os.path.dirname(__file__), "ragas_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n   Results saved to evaluations/ragas_results.json\n")
    return output

if __name__ == "__main__":
    main()
