# PredictQueue / PQ (Sistema de Otimização de Atendimento com Rede Neural)

## Visão Geral do Projecto
O projeto implementa um **sistema inteligente de recomendação de atendimento**, baseado em **simulação de filas do tipo M/M/c** e redes neurais, que permite aos utilizadores **planearem o seu tempo de forma eficiente**.  

Nota: M/M/c - Modelo matemático que simula sistemas de atendimento com múltiplos serviços, onde chegadas e atendimentos seguem distribuições exponenciais, permitindo estudar filas, tempos de espera e utilização de recursos

O sistema é projetado para cenários onde:
- Existem múltiplos serviços (pagamentos, documentação, consultas, etc.);
- Os utilizadores chegam aleatoriamente ao sistema, gerando filas;
- Cada utilizador quer **minimizar o tempo de espera** ou aproveitar o tempo de espera realizando outras tarefas.

---

## Objectivo
- **Simular o comportamento de um sistema de atendimento multi-serviço** com múltiplos estudantes e atendentes.
- **Prever o tamanho das filas em tempo real** utilizando uma rede neural.
- **Recomendar o número ideal de atendentes** ou horários ideais de chegada, otimizando recursos humanos.
- **Permitir ao utilizador gerir melhor o seu tempo**, evitando filas desnecessárias e aproveitando períodos de espera para outras atividades.

---

## Como Funciona o Sistema
1. **Simulação Multiagente**
   - Os civis chegam aleatoriamente para realizar diferentes serviços.
   - Cada civil tem um tipo de serviço e uma duração estimada de atendimento.
   - Atendentes atendem os civis disponíveis; cada atendente tem um tempo médio de serviço.
   - O sistema regista o estado da fila e a utilização dos atendentes ao longo do tempo.

2. **Rede Neural Preditiva**
   - Uma **LSTM** observa o histórico do sistema (tamanho das filas, carga de atendentes, chegada de civis).
   - Aprende padrões temporais e consegue prever **tamanho da fila futuro**.

3. **Recomendação Inteligente**
   - Baseada nas previsões da rede neural:
     - Recomenda o **número ideal de atendentes activos** para minimizar filas.
     - Sugere ao civil **o melhor momento para se dirigir ao atendimento**, evitando tempo perdido.
   - Permite aos civis **realizar outras tarefas** enquanto a fila está congestionada.

4. **Relatórios Analíticos**
   - Geração de PDF.

---

## Benefícios
- Redução do tempo médio de espera para os civis.
- Melhor utilização dos atendentes, evitando sobrecarga ou ociosidade.
- Planeamento dinâmico de atendimento com base em dados simulados e predição.
- Experiência científica e analítica para estudo de filas e otimização de recursos humanos.

---

## Dependências
- **Python 3.9+** – linguagem de programação usada para implementar toda a simulação e o treino da rede neural.  
- **numpy** – manipulação eficiente de arrays e operações matemáticas, fundamental para cálculos da simulação.  
- **pandas** – gestão e análise de dados em tabelas (DataFrames), usado para armazenar histórico de filas e resultados.  
- **tensorflow** – framework de machine learning para construir, treinar e usar a rede neural LSTM.  
- **matplotlib / seaborn** – geração de gráficos e visualizações dos resultados da simulação e métricas.  
- **fpdf2 ou reportlab** – criação de relatórios PDF com gráficos, estatísticas e recomendações do sistema.

Instalação rápida:
```bash
pip install -r requirements.txt
