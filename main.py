import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

app = FastAPI(title="Dzyon AI - Embedding Service")

# Carregar variáveis de ambiente do Easypanel
# Carregar variáveis de ambiente removendo espaços em branco acidentais (.strip())
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_KEY", "")
SUPABASE_KEY = SUPABASE_KEY.strip()
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Variáveis de ambiente do Supabase não configuradas!")

# Inicializar o cliente do Supabase e o Modelo de Embedding local
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print(f"Carregando modelo de embedding: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Modelo carregado com sucesso!")


class KMInput(BaseModel):
    erp_record_id: str
    title: str
    summary: str
    product: str
    module: str
    category: str
    raw_text: str  # Esse campo receberá o 'answer' bruto vindo do Progress


def parse_progress_text(text: str):
    """Extrai o conteúdo entre [CAUSA]=...| e [SOLUCAO]=...|"""
    causa_match = re.search(r"\[CAUSA\]=(.*?)(?=\|\[|$)", text, re.DOTALL)
    solucao_match = re.search(r"\[SOLUCAO\]=(.*?)(?=\|\[|$)", text, re.DOTALL)

    causa = causa_match.group(1).strip() if causa_match else ""
    solucao = solucao_match.group(1).strip() if solucao_match else ""

    # Se vier sem tags, limpa caracteres comuns e trata como texto geral
    if not causa and not solucao:
        clean_text = text.replace("|", "\n").strip()
        return clean_text, [("Conteúdo Geral", clean_text)]

    clean_text = f"CAUSA:\n{causa}\n\nSOLUÇÃO:\n{solucao}".strip()

    chunks_to_create = []
    if causa:
        chunks_to_create.append(("Causa", causa))
    if solucao:
        chunks_to_create.append(("Solução", solucao))

    return clean_text, chunks_to_create


@app.post("/embed")
async def process_and_embed(data: KMInput):
    try:
        # 1. Parsear o texto bruto do ERP
        clean_text, sections = parse_progress_text(data.raw_text)

        # 2. Inserir na tabela ai_sources (Documento Pai)
        source_data = {
            "source_type": "km",
            "erp_record_id": data.erp_record_id,
            "erp_table_name": "km-doc-ms",
            "title": data.title,
            "summary": data.summary,
            "product": data.product,
            "module": data.module,
            "category": data.category,
            "raw_text": data.raw_text,
            "clean_text": clean_text,
        }

        source_response = supabase.table("ai_sources").insert(source_data).execute()
        if not source_response.data:
            raise HTTPException(status_code=500, detail="Falha ao salvar ai_sources.")

        source_id = source_response.data[0]["id"]

        # 3. Processar cada seção extraída (Gerar Chunks e Embeddings)
        for idx, (section_title, section_content) in enumerate(sections):
            if not section_content.strip():
                continue

            # Inserir o fragmento na tabela ai_chunks
            chunk_data = {
                "source_id": source_id,
                "chunk_index": idx,
                "chunk_title": section_title,
                "content": section_content,
                "product": data.product,
                "module": data.module,
                "category": data.category,
                "char_count": len(section_content),
            }

            chunk_response = supabase.table("ai_chunks").insert(chunk_data).execute()
            if not chunk_response.data:
                continue

            chunk_id = chunk_response.data[0]["id"]

            # Gerar o vetor numérico localmente usando o BGE-Small
            embedding_vector = model.encode(section_content).tolist()

            # Inserir o vetor na tabela ai_embeddings
            embedding_data = {
                "chunk_id": chunk_id,
                "model_name": MODEL_NAME,
                "embedding": embedding_vector,
            }
            supabase.table("ai_embeddings").insert(embedding_data).execute()

        return {
            "status": "success",
            "ai_source_id": source_id,
            "chunks_processed": len(sections),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}
