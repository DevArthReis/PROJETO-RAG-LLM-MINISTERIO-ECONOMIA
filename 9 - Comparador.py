import pandas as pd
from pathlib import Path

def comparar_planilhas(caminho_resumo: Path, caminho_verdade: Path, caminho_saida: Path):
    # Carregar as planilhas
    df_resumo = pd.read_excel("C:\Users\arthu\RESUMIDOR\RESPOSTAS")
    df_verdade = pd.read_excel("C:\Users\arthu\RESUMIDOR\Verdade")

    # Verificar se as colunas necessárias existem
    if "Nome" not in df_resumo.columns or "Salário Base" not in df_resumo.columns:
        raise ValueError("A planilha de resumo não contém as colunas necessárias ('Nome' e 'Salário Base').")
    
    if "Nome" not in df_verdade.columns or "Salário Base" not in df_verdade.columns:
        raise ValueError("A planilha de verdade não contém as colunas necessárias ('Nome' e 'Salário Base').")

    # Mesclar as planilhas com base no nome
    df_comparacao = pd.merge(
        df_resumo[["Nome", "Salário Base"]],
        df_verdade[["Nome", "Salário Base"]],
        on="Nome",
        suffixes=("_Resumo", "_Verdade")
    )

    # Calcular a diferença entre os valores
    df_comparacao["Diferença"] = df_comparacao["Salário Base_Resumo"].astype(float) - df_comparacao["Salário Base_Verdade"].astype(float)

    # Salvar a planilha de comparação
    df_comparacao.to_excel(caminho_saida, index=False)
    print(f"Planilha de comparação salva em: {caminho_saida}")

def main():
    # Definir os caminhos das planilhas
    pasta_respostas = Path(r'C:\Users\arthu\RESUMIDOR\RESPOSTAS')
    pasta_verdade = Path(r'C:\Users\arthu\RESUMIDOR\Verdade')
    caminho_resumo = pasta_respostas / "resumo.xlsx"
    caminho_verdade = pasta_verdade / "planilha.xlsx"
    caminho_saida = pasta_respostas / "comparacao.xlsx"

    # Verificar se as planilhas existem
    if not caminho_resumo.exists():
        print(f"Planilha de resumo não encontrada: {caminho_resumo}")
        return
    
    if not caminho_verdade.exists():
        print(f"Planilha de verdade não encontrada: {caminho_verdade}")
        return

    # Comparar as planilhas
    comparar_planilhas(caminho_resumo, caminho_verdade, caminho_saida)

if __name__ == "__main__":
    main()