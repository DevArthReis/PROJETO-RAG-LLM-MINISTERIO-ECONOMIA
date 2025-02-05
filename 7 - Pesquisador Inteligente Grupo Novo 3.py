import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor.llm_rerank import LLMRerank

# Configuração da chave da API da OpenAI
os.environ["OPENAI_API_KEY"] = ""
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Definir eventos personalizados
class RetrieverEvent(StartEvent):
    """Resultado da recuperação"""
    nodes: List[NodeWithScore]

class RerankEvent(StartEvent):
    """Resultado do re-ranking"""
    nodes: List[NodeWithScore]

# Definir o fluxo de trabalho
class RAGWorkflow(Workflow):
    def __init__(self, **kwargs):
        super().__init__(timeout=300, **kwargs)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        """Recuperar nós relevantes para a consulta."""
        query = ev.get("query")
        index = ev.get("index")

        if not query or not index:
            return None

        print(f"Consultando o índice com: {query}")
        await ctx.set("query", query)

        retriever = index.as_retriever(similarity_top_k=10)  # Recuperar 10 nós
        nodes = await retriever.aretrieve(query)
        print(f"Recuperados {len(nodes)} nós.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        """Reordenar os nós recuperados."""
        ranker = LLMRerank(
            choice_batch_size=5,  # Reordenar 5 nós
            top_n=5,              # Selecionar 5 nós após o re-ranking
            llm=OpenAI(model="gpt-4o-mini", temperature=0)
        )
        query = await ctx.get("query", default=None)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=query
        )
        print(f"Nós reordenados para {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Gerar a resposta final com no mínimo 4 nós."""
        llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        summarizer = CompactAndRefine(llm=llm, streaming=False, verbose=False)
        query = await ctx.get("query", default=None)

        # Garantir que pelo menos 4 nós sejam passados
        nodes_to_use = ev.nodes[:5]  # Passar no máximo 5 nós após reordenação

        response = await summarizer.asynthesize(query, nodes=nodes_to_use)
        return StopEvent(result=response)


async def executar_workflow(w: RAGWorkflow, pergunta: str, index) -> str:
    """Executa o workflow para uma pergunta específica e retorna a resposta."""
    result = await w.run(query=pergunta, index=index)
    response = result.response
    return str(response).strip() if response else "Nenhuma resposta clara fornecida."

# Lista de perguntas padrão
async def executar_workflow(w: RAGWorkflow, pergunta: str, index) -> str:
    """Executa o workflow para uma pergunta específica e retorna a resposta."""
    result = await w.run(query=pergunta, index=index)
    response = result.response
    return str(response).strip() if response else "Nenhuma resposta clara fornecida."

# Lista de perguntas padrão com solicitação de FGTS
perguntas = [
    """Verifique todo o documento e me informe os valores de  Total Salário Líquido,Salário Base, FGTS e  VALE TRANSPORTES. 
    Os nomes são: 
    ALAN RAMOS NAVEGANTES
    ALESSANDRA SANTOS LEMOS
    ALINE AGUIAR SILVEIRA
    (... lista continua ...)
    
    Me retorne as informações no seguinte formato:
    Nome: <Nome da pessoa>
    Total Salário Líquido: <Valor>
    Salário Base: <Valor>
    FGTS: <Valor>
    Vale-Transporte: <Valor>
    """
]

async def processar_pasta(pasta: Path, pasta_destino: Path):
    pasta_indices = pasta / "indices"
    
    if not pasta_indices.exists() or not pasta_indices.is_dir():
        print(f"Pasta 'indices' não encontrada em: {pasta}")
        return

    storage_context = StorageContext.from_defaults(persist_dir=str(pasta_indices))
    index = load_index_from_storage(storage_context)
    print(f"Índice carregado de {pasta_indices}")

    w = RAGWorkflow()
    respostas = {}

    for pergunta in perguntas:
        try:
            resposta = await executar_workflow(w, pergunta, index)
            respostas[pergunta] = resposta
            print(f"Resposta processada para pasta {pasta.name}")
        except Exception as e:
            print(f"Erro ao processar pergunta para {pasta.name}: {str(e)}")
            respostas[pergunta] = f"Erro: {str(e)}"

    filename_json = pasta_destino / f"{pasta.name}.json"
    with open(filename_json, 'w', encoding='utf-8') as f:
        json.dump(respostas, f, ensure_ascii=False, indent=4)

    print(f"Respostas salvas em: {filename_json}")

async def main():
    # Caminhos atualizados conforme a nova localização
    pasta_matriz = Path(r'C:\Users\arthu\RESUMIDOR\FEITO - 3.5')
    pasta_destino = Path(r'C:\Users\arthu\RESUMIDOR\RESPOSTAS')
    pasta_destino.mkdir(parents=True, exist_ok=True)

    # Processar todas as pastas na matriz
    tasks = []
    for pasta in pasta_matriz.iterdir():
        if pasta.is_dir():
            tasks.append(processar_pasta(pasta, pasta_destino))
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())