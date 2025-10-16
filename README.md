# Simulação e Previsão de Atendimento com Rede Neural

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## Descrição
Este projeto simula um sistema de atendimento do tipo M/M/c, onde estudantes chegam aleatoriamente para realizar pagamentos ou tratar de documentos. Utiliza uma **rede neural** para prever o tamanho da fila e recomendar o número ideal de atendentes.  

O objetivo é estudar padrões de congestionamento e otimizar recursos humanos no atendimento, fornecendo um ambiente de análise científica baseado em simulação e Machine Learning.

## Estrutura do Projeto

- `atendimento_nn/`
  - `.gitignore` — Ficheiros e pastas a ignorar pelo Git
  - `data/` — Dados simulados e gerados
  - `models/` — Modelos treinados (Keras/TensorFlow)
  - `src/` — Código fonte
    - `__init__.py`
    - `simulate.py` — Simulação do sistema M/M/c
    - `prepare_data.py` — Preparação do dataset para ML
    - `train_model.py` — Treino da rede neural
    - `recommend.py` — Função de recomendação de atendentes
  - `main.py` — Script principal de execução
  - `requirements.txt` — Dependências do Python
  - `README.md` — Este ficheiro

## Requisitos

- Python 3.10 ou superior
- Bibliotecas Python:
  - numpy
  - pandas
  - tensorflow
  - tqdm

Instala as dependências com:

```bash
pip install -r requirements.txt

git clone <URL_DO_REPOSITORIO>
cd atendimento_nn

python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

pip install -r requirements.txt

python main.py
