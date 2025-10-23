import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import re
from collections import Counter
import math

class AITextDetector:
    def __init__(self):
        # Palavras comuns em textos gerados por IA
        self.ai_indicators = [
            'além disso', 'portanto', 'no entanto', 'consequentemente',
            'é importante notar', 'vale ressaltar', 'cabe destacar',
            'neste contexto', 'dessa forma', 'assim sendo',
            'furthermore', 'moreover', 'however', 'therefore',
            'it is important to note', 'additionally', 'consequently'
        ]
        
    def analyze_text(self, text):
        """Analisa o texto e retorna a probabilidade de ser gerado por IA"""
        if not text or len(text.strip()) < 50:
            return 0, [], "Texto muito curto para análise"
        
        sentences = self._split_sentences(text)
        paragraphs = text.split('\n\n')
        
        # Critérios de análise
        scores = []
        suspicious_parts = []
        
        # 1. Uniformidade no tamanho das sentenças
        uniformity_score, uniform_parts = self._check_sentence_uniformity(sentences)
        scores.append(uniformity_score)
        suspicious_parts.extend(uniform_parts)
        
        # 2. Uso excessivo de conectivos/transições
        connector_score, connector_parts = self._check_connectors(text, sentences)
        scores.append(connector_score)
        suspicious_parts.extend(connector_parts)
        
        # 3. Repetição de estruturas
        repetition_score, rep_parts = self._check_repetition(sentences)
        scores.append(repetition_score)
        suspicious_parts.extend(rep_parts)
        
        # 4. Perfeição gramatical excessiva
        perfection_score = self._check_perfection(text)
        scores.append(perfection_score)
        
        # 5. Falta de erros/variação humana
        human_variation_score = self._check_human_variation(text)
        scores.append(human_variation_score)
        
        # Calcula a média ponderada
        final_score = sum(scores) / len(scores)
        
        return min(100, final_score), suspicious_parts, self._generate_report(scores)
    
    def _split_sentences(self, text):
        """Divide o texto em sentenças"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _check_sentence_uniformity(self, sentences):
        """Verifica se as sentenças têm tamanho muito uniforme"""
        if len(sentences) < 3:
            return 0, []
        
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        # Textos humanos têm mais variação
        if std_dev < 3 and avg_length > 10:
            score = 30
            parts = [f"Sentenças muito uniformes (desvio: {std_dev:.1f})"]
            return score, parts
        
        return 0, []
    
    def _check_connectors(self, text, sentences):
        """Verifica uso excessivo de conectivos"""
        text_lower = text.lower()
        connector_count = 0
        found_connectors = []
        
        for indicator in self.ai_indicators:
            count = text_lower.count(indicator.lower())
            if count > 0:
                connector_count += count
                found_connectors.append(f"'{indicator}' ({count}x)")
        
        # Se mais de 20% das sentenças têm conectivos formais
        connector_ratio = connector_count / max(len(sentences), 1)
        
        if connector_ratio > 0.2:
            score = min(35, connector_ratio * 100)
            parts = [f"Conectivos formais excessivos: {', '.join(found_connectors[:5])}"]
            return score, parts
        
        return 0, []
    
    def _check_repetition(self, sentences):
        """Verifica padrões repetitivos"""
        if len(sentences) < 3:
            return 0, []
        
        # Verifica início de sentenças
        starts = [s.split()[0].lower() if s.split() else '' for s in sentences]
        start_counter = Counter(starts)
        most_common = start_counter.most_common(1)[0]
        
        if most_common[1] > len(sentences) * 0.3:
            score = 25
            parts = [f"Muitas sentenças começam com '{most_common[0]}' ({most_common[1]}x)"]
            return score, parts
        
        return 0, []
    
    def _check_perfection(self, text):
        """Verifica se o texto é 'perfeito demais'"""
        # Textos humanos geralmente têm alguns erros ou inconsistências
        # IA tende a ser muito consistente
        
        # Verifica pontuação perfeita
        sentences = self._split_sentences(text)
        if len(sentences) > 5:
            # Verifica se todas as sentenças têm comprimento razoável
            all_reasonable = all(5 < len(s.split()) < 50 for s in sentences)
            if all_reasonable:
                return 20
        
        return 0
    
    def _check_human_variation(self, text):
        """Verifica falta de variação humana natural"""
        # Textos humanos têm mais variação em pontuação, espaçamento, etc.
        
        # Verifica uso de pontuação variada
        punctuation = re.findall(r'[,;:—\-()]', text)
        if len(punctuation) < len(text) / 100:  # Muito pouca pontuação variada
            return 15
        
        return 0
    
    def _generate_report(self, scores):
        """Gera relatório detalhado"""
        labels = [
            "Uniformidade das sentenças",
            "Uso de conectivos formais",
            "Padrões repetitivos",
            "Perfeição gramatical",
            "Falta de variação humana"
        ]
        
        report = "Análise detalhada:\n\n"
        for label, score in zip(labels, scores):
            report += f"• {label}: {score:.1f}%\n"
        
        return report


class AIDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Texto Gerado por IA")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        self.detector = AITextDetector()
        self.current_text = ""
        
        self._create_widgets()
    
    def _create_widgets(self):
        # Título
        title_frame = tk.Frame(self.root, bg='#2c3e50', pady=15)
        title_frame.pack(fill='x')
        
        title = tk.Label(
            title_frame,
            text="🤖 Detector de Texto Gerado por IA",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title.pack()
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        # Botões de ação
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=(0, 15))
        
        self.load_btn = tk.Button(
            button_frame,
            text="📁 Carregar Arquivo",
            command=self.load_file,
            font=('Arial', 11),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief='flat'
        )
        self.load_btn.pack(side='left', padx=(0, 10))
        
        self.analyze_btn = tk.Button(
            button_frame,
            text="🔍 Analisar Texto",
            command=self.analyze,
            font=('Arial', 11),
            bg='#2ecc71',
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
            height=12,
            font=('Arial', 10),
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
            fg='#2c3e50',
            pady=20
        )
        self.result_label.pack()
        
        # Barra de progresso
        self.progress = ttk.Progressbar(
            result_frame,
            length=400,
            mode='determinate',
            style='Custom.Horizontal.TProgressbar'
        )
        self.progress.pack(pady=(0, 20))
        
        # Configurar estilo da barra
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            'Custom.Horizontal.TProgressbar',
            thickness=25,
            troughcolor='#ecf0f1',
            background='#e74c3c'
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
            height=8,
            font=('Arial', 9),
            wrap='word',
            relief='solid',
            borderwidth=1,
            bg='#fafafa'
        )
        self.report_area.pack(fill='both', expand=True)
        self.report_area.config(state='disabled')
    
    def _on_text_change(self, event=None):
        """Habilita botão de análise quando há texto"""
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
        """Analisa o texto"""
        text = self.text_area.get('1.0', 'end-1c').strip()
        
        if not text:
            self._show_error("Por favor, insira ou carregue um texto para análise.")
            return
        
        # Realiza análise
        score, suspicious_parts, report = self.detector.analyze_text(text)
        
        # Atualiza interface
        self.result_label.config(text=f"Probabilidade de ser IA: {score:.1f}%")
        self.progress['value'] = score
        
        # Muda cor baseado no score
        if score < 30:
            color = '#2ecc71'  # Verde
            interpretation = "Provavelmente escrito por humano"
        elif score < 60:
            color = '#f39c12'  # Laranja
            interpretation = "Incerto - pode ter partes geradas por IA"
        else:
            color = '#e74c3c'  # Vermelho
            interpretation = "Provavelmente gerado por IA"
        
        style = ttk.Style()
        style.configure('Custom.Horizontal.TProgressbar', background=color)
        
        # Atualiza relatório
        self.report_area.config(state='normal')
        self.report_area.delete('1.0', 'end')
        
        full_report = f"Interpretação: {interpretation}\n\n{report}"
        
        if suspicious_parts:
            full_report += "\n\nIndicadores encontrados:\n"
            for part in suspicious_parts:
                full_report += f"• {part}\n"
        
        self.report_area.insert('1.0', full_report)
        self.report_area.config(state='disabled')
    
    def _show_error(self, message):
        """Mostra mensagem de erro"""
        error_window = tk.Toplevel(self.root)
        error_window.title("Erro")
        error_window.geometry("400x150")
        error_window.configure(bg='white')
        
        label = tk.Label(
            error_window,
            text=message,
            font=('Arial', 11),
            bg='white',
            wraplength=350
        )
        label.pack(pady=30)
        
        btn = tk.Button(
            error_window,
            text="OK",
            command=error_window.destroy,
            bg='#3498db',
            fg='white',
            padx=30,
            pady=5
        )
        btn.pack()


def main():
    root = tk.Tk()
    app = AIDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
