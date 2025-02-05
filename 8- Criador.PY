import os
import json
import asyncio
import pandas as pd
from pathlib import Path

async def json_para_excel(pasta_destino: Path):
    """Converte todos os arquivos JSON na pasta de destino em uma única planilha Excel."""
    all_data = []
    
    for json_file in pasta_destino.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            for pergunta, resposta in data.items():
                # Parseando a resposta para capturar Nome, Salário, FGTS, Vale-Transporte
                linhas = resposta.strip().split("\n")
                temp_dict = {"Arquivo": json_file.stem}
                
                for linha in linhas:
                    if linha.startswith("Nome:"):
                        # Se já houver um nome no temp_dict, adicionamos ao all_data
                        if "Nome" in temp_dict:
                            all_data.append(temp_dict)
                            temp_dict = {"Arquivo": json_file.stem}
                        temp_dict["Nome"] = linha.replace("Nome:", "").strip()
                    elif linha.startswith("Total Salário Líquido:"):
                        temp_dict["Total Salário Líquido"] = linha.replace("Total Salário Líquido:", "").strip()
                    elif linha.startswith("FGTS:"):
                        temp_dict["FGTS"] = linha.replace("FGTS:", "").strip()
                    elif linha.startswith("Salário Base:"):
                        temp_dict["Salário Base"] = linha.replace("Salário Base:", "").strip()
                    elif linha.startswith("Vale-Transporte:"):                          
                        temp_dict["Vale-Transporte"] = linha.replace("Vale-Transporte:", "").strip()
                    
                     
                
                # Adiciona a última pessoa processada
                if "Nome" in temp_dict:
                    all_data.append(temp_dict)
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Remover colunas não desejadas (se existirem)
        colunas_para_remover = ["Valor das Horas", "Horas Trabalhadas", "Dias Úteis"]
        for coluna in colunas_para_remover:
            if coluna in df.columns:
                df.drop(columns=[coluna], inplace=True)
        
        excel_path = pasta_destino / "resumo.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"Planilha salva em: {excel_path}")
    else:
        print("Nenhum dado encontrado para gerar a planilha.")

async def main():
    pasta_destino = Path(r'C:\Users\arthu\RESUMIDOR\RESPOSTAS')
    await json_para_excel(pasta_destino)

if __name__ == "__main__":
    asyncio.run(main())

