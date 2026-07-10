"""Gera respostas de um modelo (fp8, offline vLLM) para as golden questions da Fase 2."""
if __name__ == "__main__":
    import sys, json
    from vllm import LLM, SamplingParams

    MODEL = sys.argv[1]
    OUT = sys.argv[2]
    GOLDEN = [
        {"consulta": "Como funciona o vale-pedágio obrigatório no transporte de cargas?", "fontes": ["6024/2023"]},
        {"consulta": "Regras para o transporte rodoviário internacional de cargas", "fontes": ["6038/2024"]},
        {"consulta": "O que é o Registro Nacional do Agente Transportador de cargas?", "fontes": ["5990/2022"]},
        {"consulta": "Quais documentos são exigidos no transporte de produtos perigosos?", "fontes": ["5232/2016"]},
        {"consulta": "Como são calculados os pisos mínimos de frete por eixo carregado?", "fontes": ["5867/2020"]},
        {"consulta": "Regulamento das concessões rodoviárias federais, segunda norma", "fontes": ["6000/2022"]},
        {"consulta": "Parcelamento de débitos não inscritos em dívida ativa", "fontes": ["5830/2018"]},
        {"consulta": "Regulamento do transporte rodoviário coletivo interestadual de passageiros", "fontes": ["5998/2022"]},
        {"consulta": "Delegação de competências da diretoria colegiada da ANTT", "fontes": ["5818/2018"]},
        {"consulta": "Programa de regularização de débitos não tributários PRD", "fontes": ["5386/2017"]},
    ]
    SYS = "Responda sobre a regulação da ANTT, citando a resolução."
    llm = LLM(model=MODEL, quantization="fp8", max_model_len=2048,
              gpu_memory_utilization=0.80, enforce_eager=True)
    convs = [[{"role": "system", "content": SYS}, {"role": "user", "content": g["consulta"]}] for g in GOLDEN]
    outs = llm.chat(convs, SamplingParams(max_tokens=256, temperature=0.0))
    ans = [{"consulta": g["consulta"], "fontes": g["fontes"], "resposta": o.outputs[0].text.strip()}
           for g, o in zip(GOLDEN, outs)]
    json.dump(ans, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("SALVO", OUT, "n=", len(ans))
    print("DONE_OFFLINE")
