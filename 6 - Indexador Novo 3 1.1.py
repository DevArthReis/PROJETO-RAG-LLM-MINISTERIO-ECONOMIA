import os
import json
import asyncio
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

# Certifique-se de que as stopwords e o tokenizador estão disponíveis
nltk.download('stopwords')
nltk.download('punkt')

# Definir as variáveis de ambiente
os.environ['OPENAI_API_KEY'] = "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"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Função de pré-processamento de perguntas
def preprocess_question(question):
    question = question.lower().translate(str.maketrans('', '', '.,;:!?()[]'))
    stop_words = set(stopwords.words('portuguese'))
    tokens = word_tokenize(question)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = RSLPStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    processed_question = ' '.join(stemmed_tokens)
    return processed_question

# Configuração do node parser com o chunk_size desejado
node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=100)

# Configurar o modelo de embeddings da OpenAI
openai_embedding_model = "text-embedding-ada-002"
embed_model = OpenAIEmbedding(model_name=openai_embedding_model)

# Configurações do modelo LLM
llm = OpenAI(temperature=0, model="gpt-4o-mini")

# Diretório raiz onde estão as pastas com os arquivos a serem indexados
diretorio_raiz =  Path(r'C:\Users\arthu\RESUMIDOR\FEITO - 3.5')

def process_documents(pasta, node_parser, llm, embed_model):
    """
    Processa documentos em uma pasta específica, cria o índice e salva-o.
    Esta função é síncrona e será executada em um executor separado.
    """
    if not any(pasta.iterdir()):
        print(f"Nenhum documento encontrado em {pasta}. Pulando...")
        return

    print(f"Processando documentos em {pasta}")
    try:
        documents = SimpleDirectoryReader(str(pasta)).load_data()

        index = VectorStoreIndex.from_documents(
            documents=documents,
            llm=llm,
            embed_model=embed_model,
            node_parser=node_parser
        )

        pasta_destino_indices = pasta / "indices"
        pasta_destino_indices.mkdir(exist_ok=True)

        index.storage_context.persist(persist_dir=str(pasta_destino_indices))
        print(f"Índices salvos em {pasta_destino_indices}")
    except Exception as e:
        print(f"Erro ao processar documentos em {pasta}: {e}")

async def process_documents_async(pasta, node_parser, llm, embed_model, loop, semaphore):
    """
    Função assíncrona que processa documentos utilizando um semáforo para limitar a concorrência.
    """
    async with semaphore:
        try:
            await loop.run_in_executor(None, process_documents, pasta, node_parser, llm, embed_model)
        except asyncio.CancelledError:
            print(f"Tarefa cancelada para a pasta {pasta}")
        except Exception as e:
            print(f"Erro assíncrono ao processar documentos em {pasta}: {e}")

async def main():
    """
    Função principal que gerencia o processamento assíncrono de múltiplas pastas.
    """
    max_concurrent_tasks = 5  # Ajuste conforme necessário
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    loop = asyncio.get_running_loop()
    tarefas = []

    for pasta in diretorio_raiz.iterdir():
        if pasta.is_dir():
            tarefa = asyncio.create_task(
                process_documents_async(pasta, node_parser, llm, embed_model, loop, semaphore)
            )
            tarefas.append(tarefa)

    try:
        await asyncio.gather(*tarefas)
        print("Todos os documentos foram processados e os índices criados.")
    except asyncio.CancelledError:
        print("Processamento cancelado.")
    except Exception as e:
        print(f"Ocorreu um erro durante a execução das tarefas: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Processamento interrompido pelo usuário.")
