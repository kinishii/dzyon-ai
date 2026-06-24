import os
import re
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Dzyon AI - Embedding Service")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_KEY", "")
SUPABASE_KEY = SUPABASE_KEY.strip()
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Variáveis de ambiente do Supabase não configuradas!")

REST_URL = SUPABASE_URL.rstrip("/") + "/rest/v1"

print(f"Carregando modelo de embedding: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Modelo carregado com sucesso!")


def supabase_insert(table: str, data: dict) -> dict:
    """Insere registro no Supabase via REST API e retorna o primeiro registro inserido."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    resp = requests.post(f"{REST_URL}/{table}", json=data, headers=headers, timeout=15)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Supabase insert error ({table}): {resp.status_code} {resp.text[:200]}")
    if not resp.json():
        raise RuntimeError(f"Supabase insert returned empty ({table})")
    return resp.json()[0]


class KMInput(BaseModel):
    erp_record_id: str = ""
    title: str = ""
    summary: str = ""
    product: str = ""
    module: str = ""
    category: str = ""
    raw_text: str = ""


def parse_progress_text(text: str):
    causa_match = re.search(r"\[CAUSA\]=(.*?)(?=\|\[|$)", text, re.DOTALL)
    solucao_match = re.search(r"\[SOLU[CÇ][AÃ]O\]=(.*?)(?=\|\[|$)", text, re.DOTALL)

    causa = causa_match.group(1).strip() if causa_match else ""
    solucao = solucao_match.group(1).strip() if solucao_match else ""

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
    print(f"[DZYON] Recebido: erp_record_id={data.erp_record_id!r} title={data.title!r} raw_text={data.raw_text[:100]!r}")
    try:
        if not data.raw_text:
            raise HTTPException(status_code=400, detail="raw_text vazio")
        clean_text, sections = parse_progress_text(data.raw_text)

        source_data = {
            "source_type": "kb_article",
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

        source = supabase_insert("ai_sources", source_data)
        source_id = source["id"]

        chunks_ok = 0
        for idx, (section_title, section_content) in enumerate(sections):
            if not section_content.strip():
                continue

            chunk_data = {
                "source_id": source_id,
                "chunk_index": idx,
                "chunk_title": section_title,
                "content": section_content,
                "product": data.product,
                "module": data.module,
                "category": data.category,
            }

            try:
                chunk = supabase_insert("ai_chunks", chunk_data)
            except RuntimeError as e:
                continue

            chunk_id = chunk["id"]
            embedding_vector = model.encode(section_content).tolist()

            # Schema exige 1536 dimensoes (OpenAI ada-002 padrao)
            # BGE-small gera 512 — padding com zeros para 1536
            EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
            current_dim = len(embedding_vector)
            if current_dim < EMBED_DIM:
                embedding_vector += [0.0] * (EMBED_DIM - current_dim)

            embedding_data = {
                "chunk_id": chunk_id,
                "model_name": MODEL_NAME,
                "embedding": embedding_vector,
            }
            supabase_insert("ai_embeddings", embedding_data)
            chunks_ok += 1

        return {
            "status": "success",
            "ai_source_id": source_id,
            "chunks_processed": chunks_ok,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/echo")
async def echo(data: KMInput):
    return {"received": data.model_dump() if hasattr(data, "model_dump") else data.dict()}


@app.post("/raw-debug")
async def raw_debug(request: Request):
    """Debug: retorna o body CRU como recebido, antes de qualquer validacao."""
    body = await request.body()
    headers = dict(request.headers)
    print(f"[RAW-DEBUG] Headers: {headers}")
    print(f"[RAW-DEBUG] Body ({len(body)} bytes): {body[:500]!r}")
    return {
        "content_type": headers.get("content-type", ""),
        "body_length": len(body),
        "body_preview": body[:500].decode("utf-8", errors="replace"),
        "body_hex": body[:100].hex(),
    }
