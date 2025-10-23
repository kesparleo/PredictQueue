# 🤖 Detector de Texto Gerado por IA com TensorFlow

Um aplicativo Python com interface gráfica que usa Deep Learning (TensorFlow/Keras) para analisar textos e estimar a probabilidade de terem sido gerados por inteligência artificial.

## 📋 Características

- **Modelo de Deep Learning**: Rede neural LSTM bidirecional treinada com TensorFlow
- Interface gráfica intuitiva com Tkinter
- Análise baseada em:
  - Embeddings de texto processados por rede neural
  - Detecção de padrões linguísticos típicos de IA
  - Identificação de conectivos formais excessivos
  - Análise de uniformidade e repetição
- Carregamento de arquivos .txt
- Relatório detalhado com estatísticas e indicadores específicos
- Visualização com barra de progresso colorida

## 🔧 Requisitos

Instale as dependências necessárias:

\`\`\`bash
pip install tensorflow numpy
\`\`\`

**Nota**: Tkinter geralmente já vem instalado com Python. Se não estiver disponível:
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **macOS**: Já incluído com Python
- **Windows**: Já incluído com Python

## 🚀 Como usar

1. Execute o script:
\`\`\`bash
python scripts/ai_detector.py
\`\`\`

2. Aguarde o modelo TensorFlow carregar (alguns segundos na primeira execução)

3. Carregue um arquivo de texto ou cole o texto diretamente na área de texto

4. Clique em "Analisar com IA" para ver os resultados

## 📊 Interpretação dos resultados

- **0-30%**: Provavelmente escrito por humano (alta confiança)
- **30-50%**: Incerto - características mistas (baixa confiança)
- **50-70%**: Possivelmente gerado por IA (média confiança)
- **70-100%**: Provavelmente gerado por IA (alta confiança)

## 🧠 Sobre o Modelo

O detector usa uma arquitetura de rede neural com:
- Camada de Embedding (128 dimensões)
- LSTM Bidirecional (64 unidades)
- Global Max Pooling
- Camadas Dense com Dropout para regularização
- Ativação Sigmoid para classificação binária

O modelo é treinado automaticamente ao iniciar com dados sintéticos que simulam padrões de texto humano vs IA.

## 📝 Exemplos incluídos

- `exemplo_texto_humano.txt` - Texto informal com características humanas
- `exemplo_texto_ia.txt` - Texto formal com padrões típicos de IA

## ⚠️ Limitações

- Este é um modelo treinado com dados sintéticos limitados
- A precisão pode variar dependendo do tipo, idioma e qualidade do texto
- Não deve ser usado como única fonte de verificação
- Textos muito curtos (< 20 caracteres) não podem ser analisados adequadamente

## 🔬 Tecnologias

- **TensorFlow/Keras**: Framework de Deep Learning
- **NumPy**: Processamento numérico
- **Tkinter**: Interface gráfica
- **Python 3.7+**: Linguagem de programação
