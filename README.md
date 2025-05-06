# Classificador de Flores com KNN

Esta aplicação é um exemplo simples de um classificador de flores usando o algoritmo KNN aplicado ao dataset Iris. A aplicação é construída com Python e Flask, e exibe tanto a predição para novos valores quanto as métricas de desempenho (acurácia, precisão e recall) para os conjuntos de treinamento e teste. Além disso, o front-end apresenta gráficos da matriz de confusão e da superfície de decisão, e a interface conta com uma logo da UFCA centralizada no topo.

## Pré-requisitos

- Python 3.x instalado
- Pip (gerenciador de pacotes do Python)

## Como Ambientar o Projeto

1. **Clone o repositório ou extraia os arquivos do projeto**  
   Certifique-se de que o diretório do projeto contenha, pelo menos:
   - `back.py`
   - `front.html` (dentro de um diretório `templates`, se o Flask estiver configurado para isso)
   - `requirements.txt` com as dependências necessárias

2. **Criar o ambiente virtual**

   No diretório do projeto, execute:
   ```bash
   python -m venv venv
   ```

3. **Ativar o ambiente virtual**

   - No Windows (PowerShell):
     ```bash
     .\venv\Scripts\Activate
     ```
   - No Linux/MacOS:
     ```bash
     source venv/bin/activate
     ```

4. **Instalar as dependências**

   Com o ambiente virtual ativo, execute:
   ```bash
   pip install -r requirements.txt
   ```

## Executando a Aplicação

Após instalar as dependências, inicie o servidor Flask executando:
```bash
python back.py
```

O servidor ficará disponível em [http://127.0.0.1:5000](http://127.0.0.1:5000). Abra este endereço em seu navegador para utilizar a aplicação.

## Estrutura do Projeto

A estrutura básica do projeto é a seguinte:

```
KNN-back-front-sample/
├── templates/
│   └── front.html       # Front-end da aplicação
├── venv/                # Ambiente virtual (gerado)
├── back.py              # Back-end da aplicação (Flask)
├── requirements.txt     # Lista de dependências do projeto
└── README.md            # Este arquivo
```

## Uso da Aplicação

- **Treino:** Clique no botão "Treino" para treinar o classificador usando o dataset Iris.
- **Teste:** Clique no botão "Teste" para visualizar as métricas de desempenho do modelo (tanto para o conjunto de treinamento quanto para o teste), além dos gráficos da matriz de confusão e da superfície de decisão.
- **Teste Novo Valor:** Preencha os campos com as medidas de uma nova amostra e clique em "Enviar Valores" para obter a predição.

---

Este projeto serve como exemplo básico para aprendizado e demonstração do uso do Flask e do KNN. Sinta-se à vontade para expandir ou modificar conforme necessário.

Este projeto faz parte do desafio prático de aprendizagem de máquina, cujo objetivo é aplicar conceitos teóricos e práticos para criar uma solução real. Desenvolvido em equipe, o sistema integra front-end e back-end, permitindo a classificação de íris (flores) utilizando diferentes algoritmos de machine learning. Foram incluídas visualizações interativas (dashboards) que comparam o desempenho dos modelos e uma interface customizada, destacando a identidade e criatividade da equipe.

Funcionalidades
Integração Inicial: Clonagem do repositório base e inclusão de todos os membros da equipe.

Modelos Comparativos: Implementação do algoritmo KNN e outros modelos (por exemplo, Árvores de Decisão, Random Forest, SVM) para análise comparativa dos resultados.

Dashboards Interativos: Visualização dos dados e dos resultados por meio de gráficos e tabelas.

Interface Customizada: Layout exclusivo com elementos de design (botões, barras, imagens e cores) para otimizar a experiência do usuário.
