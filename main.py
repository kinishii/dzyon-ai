import json
import os
import re
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
from typing import Optional

app = FastAPI(title="Dzyon AI - Embedding Service")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_KEY", "")
SUPABASE_KEY = SUPABASE_KEY.strip()
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Variáveis de ambiente do Supabase não configuradas!")

REST_URL = SUPABASE_URL.rstrip("/") + "/rest/v1"

EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

print(f"Carregando modelo de embedding: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Modelo carregado com sucesso!")


def _sanitize_json_body_bytes(body: bytes) -> bytes:
    """Torna JSON montado no Progress parseavel: controles -> espaco, \\ -> /."""
    if not body:
        return body
    out = bytearray(len(body))
    for i, b in enumerate(body):
        if b < 32 or b == 127:
            out[i] = 32  # espaco
        elif b == 92:  # backslash \
            out[i] = 47  # /
        else:
            out[i] = b
    return bytes(out)


def _parse_km_input(body: bytes) -> "KMInput":
    """Sanitize + json.loads no endpoint.

    O @app.middleware('http') do FastAPI (BaseHTTPMiddleware) NAO reinsere o body
    de forma confiavel — o teste real ZYON00003 ainda 422ava com LF no summary
    mesmo apos o middleware 'SANITIZE'. Parse manual evita isso.
    """
    cleaned = _sanitize_json_body_bytes(body)
    if cleaned != body:
        print(f"[SANITIZE] embed/echo: {len(body)}b -> {len(cleaned)}b")
    try:
        text = cleaned.decode("utf-8")
    except UnicodeDecodeError:
        text = cleaned.decode("latin-1")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[EMBED] JSON invalido pos-sanitize: {e} preview={text[:500]!r}")
        raise HTTPException(status_code=422, detail=f"JSON invalido: {e}") from e
    try:
        return KMInput(**payload)
    except ValidationError as e:
        print(f"[EMBED] payload invalido: {e.errors()}")
        raise HTTPException(status_code=422, detail=e.errors()) from e


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """Loga body cru + erros Pydantic em todo 422 (ex.: JSON invalido do Progress)."""
    body = b""
    try:
        body = await request.body()
    except Exception as e:
        body = f"<body unread: {e}>".encode()

    client = request.client.host if request.client else "?"
    ctype = request.headers.get("content-type", "")
    preview = body[:1000].decode("utf-8", errors="replace")
    hex_head = body[:64].hex() if body else ""

    print("=" * 72)
    print(f"[422] {client} {request.method} {request.url.path}")
    print(f"[422] content-type={ctype!r} content-length={request.headers.get('content-length')}")
    print(f"[422] body_bytes={len(body)} hex64={hex_head}")
    print(f"[422] body_preview={preview!r}")
    print(f"[422] errors={exc.errors()}")
    print("=" * 72)

    return JSONResponse(status_code=422, content={"detail": exc.errors()})


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


def supabase_delete(table: str, match_column: str, match_value: str):
    """Deleta registros no Supabase usando filtro de coluna."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    url = f"{REST_URL}/{table}?{match_column}=eq.{match_value}"
    resp = requests.delete(url, headers=headers, timeout=15)
    if resp.status_code not in (200, 204):
        print(f"Supabase delete warning ({table}): {resp.status_code} {resp.text[:200]}")


class KMInput(BaseModel):
    erp_record_id: str = ""
    title: str = ""
    summary: str = ""
    product: str = ""
    module: str = ""
    category: str = ""
    raw_text: str = ""
    force_update: bool = False
    erp_internal_id: str = ""


class FeedbackInput(BaseModel):
    question: str
    is_useful: bool
    rating: Optional[int] = None
    response: Optional[str] = None
    source_cited: Optional[str] = None
    km_ids: Optional[str] = None


@app.post("/feedback")
async def save_feedback(data: FeedbackInput):
    try:
        supabase_insert("ai_feedback", {
            "question": data.question,
            "is_useful": data.is_useful,
            "rating": data.rating,
            "response": data.response,
            "source_cited": data.source_cited,
            "km_ids": data.km_ids
        })
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_progress_text(text: str):
    causa_match = re.search(r"\[CAUSA\]=(.*?)(?=\|\s*\[|$)", text, re.DOTALL)
    solucao_match = re.search(r"\[SOLU[CÇ][AÃ]O\]=(.*?)(?=\|\s*\[|$)", text, re.DOTALL)

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
async def process_and_embed(request: Request):
    data = _parse_km_input(await request.body())
    print(f"[DZYON] Recebido: erp_record_id={data.erp_record_id!r} force_update={data.force_update} internal_id={data.erp_internal_id!r}")
    try:
        if not data.raw_text:
            raise HTTPException(status_code=400, detail="raw_text vazio")
            
        if data.force_update:
            print(f"[DZYON] Deletando registro antigo (se existir): {data.erp_record_id}")
            supabase_delete("ai_sources", "erp_record_id", data.erp_record_id)
            
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
            "metadata": {"internal_id": data.erp_internal_id}
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

    except HTTPException:
        raise
    except Exception as e:
        detail = str(e)
        if "23505" in detail and "already exists" in detail:
            import json
            erp_id = data.erp_record_id
            try:
                headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
                get_url = f"{REST_URL}/ai_sources?erp_record_id=eq.{erp_id}&select=id,title"
                existente = requests.get(get_url, headers=headers, timeout=10)
                source_id = existente.json()[0]["id"] if existente.status_code == 200 and existente.json() else None
            except Exception:
                source_id = None
            return {
                "status": "duplicate_skipped",
                "message": f"KM {erp_id} ja existe no banco. Nada foi duplicado.",
                "ai_source_id": source_id,
                "chunks_processed": 0,
            }
        raise HTTPException(status_code=500, detail=detail)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}


def embed_text(text: str) -> list:
    vec = model.encode(text).tolist()
    if len(vec) < EMBED_DIM:
        vec += [0.0] * (EMBED_DIM - len(vec))
    return vec


class SearchRequest(BaseModel):
    query: str
    product: Optional[str] = None
    module: Optional[str] = None
    top_k: int = 10
    threshold: float = 0.5


class EmbedQueryRequest(BaseModel):
    text: str


@app.post("/search")
async def search_kms(req: SearchRequest):
    try:
        query_vec = embed_text(req.query)

        rpc_url = f"{REST_URL}/rpc/match_chunks"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "query_embedding": query_vec,
            "match_count": req.top_k,
            "match_threshold": req.threshold,
        }
        if req.product:
            payload["filter_product"] = req.product
        if req.module:
            payload["filter_module"] = req.module

        resp = requests.post(rpc_url, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Supabase RPC error: {resp.text[:300]}")

        results = resp.json()

        # [NOVO] Injeta o metadata (com o id_interno) vindo da tabela ai_sources nos resultados
        source_ids = list(set([r["source_id"] for r in results if "source_id" in r]))
        if source_ids:
            try:
                # Monta a string no formato: in.("id1","id2")
                ids_str = ",".join([f'"{sid}"' for sid in source_ids])
                get_url = f"{REST_URL}/ai_sources?id=in.({ids_str})&select=id,metadata"
                meta_resp = requests.get(get_url, headers=headers, timeout=10)
                
                if meta_resp.status_code == 200:
                    meta_dict = {row["id"]: row.get("metadata", {}) for row in meta_resp.json()}
                    for r in results:
                        sid = r.get("source_id")
                        if sid in meta_dict:
                            r["metadata"] = meta_dict[sid]
            except Exception as e:
                print("Erro ao anexar metadados:", e)

        return {"results": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed-query")
async def embed_query(req: EmbedQueryRequest):
    try:
        vec = embed_text(req.text)
        return {"embedding": vec, "dimensions": len(vec), "model": MODEL_NAME}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/echo")
async def echo(request: Request):
    data = _parse_km_input(await request.body())
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
