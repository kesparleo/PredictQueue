import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class AITextDetectorML:
    def __init__(self):
        self.max_words = 5000
        self.max_len = 200
        self.model = None
        self.tokenizer = None
        self._build_and_train_model()
        
    def _build_and_train_model(self):
        print("Inicializando modelo TensorFlow...")
        
        # Criar tokenizer
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=self.max_words,
            oov_token='<OOV>'
        )
        
        # Gerar dados de treino sintéticos
        ai_texts, human_texts = self._generate_training_data()
        all_texts = ai_texts + human_texts
        labels = [1] * len(ai_texts) + [0] * len(human_texts)
        
        # Treinar tokenizer
        self.tokenizer.fit_on_texts(all_texts)
        
        # Converter textos para sequências
        sequences = self.tokenizer.texts_to_sequences(all_texts)
        padded = keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding='post',
            truncating='post'
        )
        
        # Construir modelo
        self.model = keras.Sequential([
            layers.Embedding(self.max_words, 128, input_length=self.max_len),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Treinar modelo
        X = np.array(padded)
        y = np.array(labels)
        
        self.model.fit(
            X, y,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        print("Modelo treinado com sucesso!")
    
    def _generate_training_data(self):
        """Gera dados sintéticos para treino"""
        # Textos típicos de IA (formais, estruturados, com conectivos)
        ai_texts = [
            "É importante notar que a inteligência artificial tem revolucionado diversos setores. Além disso, as aplicações práticas são vastas. Portanto, podemos concluir que o futuro é promissor.",
            "No contexto atual, observamos que a tecnologia avança rapidamente. Consequentemente, as empresas precisam se adaptar. Vale ressaltar que a inovação é fundamental.",
            "A análise dos dados demonstra claramente que há uma tendência crescente. Ademais, os resultados são consistentes. Dessa forma, podemos inferir conclusões importantes.",
            "Furthermore, it is essential to understand the implications. Moreover, the research indicates significant progress. Therefore, we can conclude with confidence.",
            "The implementation of this solution requires careful consideration. Additionally, the benefits are substantial. Consequently, stakeholders should prioritize this initiative.",
            "In this context, we observe several key factors. Furthermore, the data supports our hypothesis. Thus, we can proceed with the proposed approach.",
            "É fundamental destacar que os resultados obtidos são significativos. Além disso, a metodologia empregada foi rigorosa. Portanto, as conclusões são confiáveis.",
            "No âmbito desta discussão, cabe ressaltar diversos aspectos relevantes. Ademais, as evidências corroboram nossa análise. Dessa maneira, podemos avançar com segurança.",
            "The systematic approach yields consistent results. Moreover, the framework is robust. Therefore, implementation can proceed as planned.",
            "It is worth noting that the parameters are well-defined. Additionally, the metrics demonstrate effectiveness. Consequently, the model performs optimally.",
        ]
        
        # Textos típicos humanos (informais, variados, com erros ocasionais)
        human_texts = [
            "Cara, ontem eu vi um negócio muito doido! Tipo assim, tava andando na rua e do nada... nossa, nem sei explicar direito haha",
            "Bom, eu acho que isso não faz muito sentido sabe? Tipo, já tentei fazer desse jeito antes e não deu certo. Mas sei lá, vai que agora funciona né",
            "Olha só, vou te contar uma coisa: esse lance de IA tá ficando cada vez mais louco. Mas tem hora que eu fico pensando... será que é tudo isso mesmo?",
            "So yeah, I was thinking about this the other day and... idk, it's kinda weird right? Like, why would anyone do that lol",
            "Honestly? I'm not sure what to think anymore. Things have been pretty crazy lately and I just... yeah.",
            "Mano do céu, que dia! Acordei tarde, perdi o ônibus, cheguei atrasado no trabalho. Enfim, só mais um dia normal na minha vida caótica kkk",
            "Então né, eu tava conversando com meu amigo outro dia e ele falou uma parada interessante. Não lembro exatamente o que era mas... ah, deixa pra lá",
            "You know what really grinds my gears? When people don't use their turn signals. Like seriously?? It's not that hard!",
            "Acabei de ver aquele filme que todo mundo tava falando. Meh, achei meio overrated. Mas tem gente que amou, vai entender...",
            "I mean, sure, it could work. But also... couldn't it not work? That's what I'm worried about tbh",
        ]
        
        # Expandir dataset com variações
        ai_expanded = []
        human_expanded = []
        
        for text in ai_texts:
            ai_expanded.append(text)
            # Adicionar variações
            ai_expanded.append(text + " " + ai_texts[np.random.randint(0, len(ai_texts))])
        
        for text in human_texts:
            human_expanded.append(text)
            # Adicionar variações
            human_expanded.append(text + " " + human_texts[np.random.randint(0, len(human_texts))])
        
        return ai_expanded, human_expanded
    
    def analyze_text(self, text):
        """Analisa o texto usando o modelo TensorFlow"""
        if not text or len(text.strip()) < 20:
            return 0, [], "Texto muito curto para análise"
        
        # Preparar texto para o modelo
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=self.max_len,
            padding='post',
            truncating='post'
        )
        
        # Fazer predição
        prediction = self.model.predict(padded, verbose=0)[0][0]
        ai_probability = float(prediction) * 100
        
        # Análise adicional para identificar partes suspeitas
        suspicious_parts = self._identify_suspicious_parts(text)
        
        # Gerar relatório
        report = self._generate_report(ai_probability, text)
        
        return ai_probability, suspicious_parts, report
    
    def _identify_suspicious_parts(self, text):
        """Identifica partes específicas que parecem geradas por IA"""
        suspicious = []
        
        # Conectivos formais típicos de IA
        ai_connectors = [
            'além disso', 'portanto', 'no entanto', 'consequentemente',
            'é importante notar', 'vale ressaltar', 'cabe destacar',
            'neste contexto', 'dessa forma', 'assim sendo',
            'furthermore', 'moreover', 'however', 'therefore',
            'additionally', 'consequently'
        ]
        
        text_lower = text.lower()
        found_connectors = []
        
        for connector in ai_connectors:
            if connector in text_lower:
                count = text_lower.count(connector)
                found_connectors.append(f"'{connector}' ({count}x)")
        
        if found_connectors:
            suspicious.append(f"Conectivos formais detectados: {', '.join(found_connectors[:5])}")
        
        # Verificar uniformidade de sentenças
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) >= 3:
            lengths = [len(s.split()) for s in sentences]
            avg_len = np.mean(lengths)
            std_len = np.std(lengths)
            
            if std_len < 3 and avg_len > 10:
                suspicious.append(f"Sentenças muito uniformes (média: {avg_len:.1f} palavras, desvio: {std_len:.1f})")
        
        # Verificar padrões repetitivos
        if len(sentences) >= 3:
            starts = [s.split()[0].lower() if s.split() else '' for s in sentences]
            start_counter = Counter(starts)
            most_common = start_counter.most_common(1)[0]
            
            if most_common[1] > len(sentences) * 0.3:
                suspicious.append(f"Padrão repetitivo: {most_common[1]} sentenças começam com '{most_common[0]}'")
        
        return suspicious
    
    def _generate_report(self, ai_probability, text):
        """Gera relatório detalhado da análise"""
        report = "=== Análise com TensorFlow/Keras ===\n\n"
        
        # Estatísticas do texto
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        report += f"Estatísticas do texto:\n"
        report += f"• Total de palavras: {len(words)}\n"
        report += f"• Total de sentenças: {len(sentences)}\n"
        report += f"• Média de palavras por sentença: {len(words)/max(len(sentences), 1):.1f}\n\n"
        
        # Interpretação do modelo
        report += f"Resultado do modelo neural:\n"
        report += f"• Probabilidade de IA: {ai_probability:.2f}%\n"
        report += f"• Probabilidade humana: {100-ai_probability:.2f}%\n\n"
        
        # Interpretação
        if ai_probability < 30:
            interpretation = "✓ Provavelmente escrito por humano"
            confidence = "Alta confiança"
        elif ai_probability < 50:
            interpretation = "? Incerto - características mistas"
            confidence = "Baixa confiança"
        elif ai_probability < 70:
            interpretation = "⚠ Possivelmente gerado por IA"
            confidence = "Média confiança"
        else:
            interpretation = "✗ Provavelmente gerado por IA"
            confidence = "Alta confiança"
        
        report += f"Interpretação: {interpretation}\n"
        report += f"Confiança: {confidence}\n"
        
        return report


class AIDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Texto IA - TensorFlow")
        self.root.geometry("950x750")
        self.root.configure(bg='#f0f0f0')
        
        self.detector = None
        self.current_text = ""
        
        self._create_widgets()
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa o modelo em background"""
        self.status_label.config(text="⏳ Carregando modelo TensorFlow...")
        self.root.update()
        
        try:
            self.detector = AITextDetectorML()
            self.status_label.config(text="✓ Modelo carregado e pronto!")
            self.load_btn.config(state='normal')
        except Exception as e:
            self.status_label.config(text=f"✗ Erro ao carregar modelo: {str(e)}")
    
    def _create_widgets(self):
        # Título
        title_frame = tk.Frame(self.root, bg='#1a237e', pady=15)
        title_frame.pack(fill='x')
        
        title = tk.Label(
            title_frame,
            text="🤖 Detector de IA com TensorFlow",
            font=('Arial', 18, 'bold'),
            bg='#1a237e',
            fg='white'
        )
        title.pack()
        
        subtitle = tk.Label(
            title_frame,
            text="Análise de texto usando Deep Learning",
            font=('Arial', 10),
            bg='#1a237e',
            fg='#bbdefb'
        )
        subtitle.pack()
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        # Status do modelo
        self.status_label = tk.Label(
            main_frame,
            text="⏳ Inicializando...",
            font=('Arial', 10),
            bg='#fff3cd',
            fg='#856404',
            pady=8,
            relief='solid',
            borderwidth=1
        )
        self.status_label.pack(fill='x', pady=(0, 15))
        
        # Botões de ação
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=(0, 15))
        
        self.load_btn = tk.Button(
            button_frame,
            text="📁 Carregar Arquivo",
            command=self.load_file,
            font=('Arial', 11),
            bg='#1976d2',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief='flat',
            state='disabled'
        )
        self.load_btn.pack(side='left', padx=(0, 10))
        
        self.analyze_btn = tk.Button(
            button_frame,
            text="🔍 Analisar com IA",
            command=self.analyze,
            font=('Arial', 11),
            bg='#388e3c',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief='flat',
            state='disabled'
        )
        self.analyze_btn.pack(side='left')
        
        # Área de texto
        text_label = tk.Label(
            main_frame,
            text="Texto para análise:",
            font=('Arial', 11, 'bold'),
            bg='#f0f0f0'
        )
        text_label.pack(anchor='w', pady=(0, 5))
        
        self.text_area = scrolledtext.ScrolledText(
            main_frame,
            height=10,
            font=('Consolas', 10),
            wrap='word',
            relief='solid',
            borderwidth=1
        )
        self.text_area.pack(fill='both', expand=True, pady=(0, 15))
        self.text_area.bind('<KeyRelease>', self._on_text_change)
        
        # Resultado
        result_frame = tk.Frame(main_frame, bg='white', relief='solid', borderwidth=1)
        result_frame.pack(fill='x', pady=(0, 15))
        
        self.result_label = tk.Label(
            result_frame,
            text="Probabilidade de ser IA: ---%",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#1a237e',
            pady=15
        )
        self.result_label.pack()
        
        # Barra de progresso
        self.progress = ttk.Progressbar(
            result_frame,
            length=500,
            mode='determinate',
            style='Custom.Horizontal.TProgressbar'
        )
        self.progress.pack(pady=(0, 15))
        
        # Configurar estilo da barra
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            'Custom.Horizontal.TProgressbar',
            thickness=30,
            troughcolor='#e3f2fd',
            background='#d32f2f'
        )
        
        # Relatório detalhado
        report_label = tk.Label(
            main_frame,
            text="Relatório detalhado:",
            font=('Arial', 11, 'bold'),
            bg='#f0f0f0'
        )
        report_label.pack(anchor='w', pady=(0, 5))
        
        self.report_area = scrolledtext.ScrolledText(
            main_frame,
            height=10,
            font=('Consolas', 9),
            wrap='word',
            relief='solid',
            borderwidth=1,
            bg='#fafafa'
        )
        self.report_area.pack(fill='both', expand=True)
        self.report_area.config(state='disabled')
    
    def _on_text_change(self, event=None):
        """Habilita botão de análise quando há texto"""
        if self.detector is None:
            return
            
        text = self.text_area.get('1.0', 'end-1c').strip()
        if text:
            self.analyze_btn.config(state='normal')
        else:
            self.analyze_btn.config(state='disabled')
    
    def load_file(self):
        """Carrega arquivo de texto"""
        filename = filedialog.askopenfilename(
            title="Selecione um arquivo de texto",
            filetypes=[
                ("Arquivos de texto", "*.txt"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.text_area.delete('1.0', 'end')
                    self.text_area.insert('1.0', content)
                    self.current_text = content
                    self.analyze_btn.config(state='normal')
            except Exception as e:
                self._show_error(f"Erro ao carregar arquivo: {str(e)}")
    
    def analyze(self):
        """Analisa o texto usando TensorFlow"""
        text = self.text_area.get('1.0', 'end-1c').strip()
        
        if not text:
            self._show_error("Por favor, insira ou carregue um texto para análise.")
            return
        
        if self.detector is None:
            self._show_error("Modelo ainda não foi carregado. Aguarde...")
            return
        
        # Mostrar que está processando
        self.analyze_btn.config(text="⏳ Analisando...", state='disabled')
        self.root.update()
        
        try:
            # Realiza análise com TensorFlow
            score, suspicious_parts, report = self.detector.analyze_text(text)
            
            # Atualiza interface
            self.result_label.config(text=f"Probabilidade de ser IA: {score:.1f}%")
            self.progress['value'] = score
            
            # Muda cor baseado no score
            if score < 30:
                color = '#4caf50'  # Verde
            elif score < 50:
                color = '#ff9800'  # Laranja
            elif score < 70:
                color = '#ff5722'  # Laranja escuro
            else:
                color = '#d32f2f'  # Vermelho
            
            style = ttk.Style()
            style.configure('Custom.Horizontal.TProgressbar', background=color)
            
            # Atualiza relatório
            self.report_area.config(state='normal')
            self.report_area.delete('1.0', 'end')
            
            full_report = report
            
            if suspicious_parts:
                full_report += "\n\nIndicadores suspeitos encontrados:\n"
                for i, part in enumerate(suspicious_parts, 1):
                    full_report += f"{i}. {part}\n"
            
            self.report_area.insert('1.0', full_report)
            self.report_area.config(state='disabled')
            
        except Exception as e:
            self._show_error(f"Erro durante análise: {str(e)}")
        
        finally:
            self.analyze_btn.config(text="🔍 Analisar com IA", state='normal')
    
    def _show_error(self, message):
        """Mostra mensagem de erro"""
        error_window = tk.Toplevel(self.root)
        error_window.title("Erro")
        error_window.geometry("450x150")
        error_window.configure(bg='white')
        
        label = tk.Label(
            error_window,
            text=message,
            font=('Arial', 11),
            bg='white',
            wraplength=400
        )
        label.pack(pady=30)
        
        btn = tk.Button(
            error_window,
            text="OK",
            command=error_window.destroy,
            bg='#1976d2',
            fg='white',
            padx=30,
            pady=5,
            relief='flat',
            cursor='hand2'
        )
        btn.pack()


def main():
    root = tk.Tk()
    app = AIDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()