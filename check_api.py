"""S0: confere se o .env está configurado e se a chave da OpenAI responde.

Uso: python check_api.py
"""
import sys

import config


def main() -> int:
    if not config.OPENAI_API_KEY:
        print("ERRO: OPENAI_API_KEY não definida. Copie .env.example para .env e preencha a chave.")
        return 1
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        OpenAIEmbeddings(model=config.EMBEDDING_MODEL).embed_query("teste")
        reply = ChatOpenAI(model=config.CHAT_MODEL, temperature=0).invoke("Responda apenas: ok")
    except Exception as exc:  # erro de rede, chave inválida, cota etc.
        print(f"ERRO ao chamar a API da OpenAI: {exc}")
        return 1
    print(f"OK: embeddings ({config.EMBEDDING_MODEL}) e chat ({config.CHAT_MODEL}) respondendo -> {reply.content!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
