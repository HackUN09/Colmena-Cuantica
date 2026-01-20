import torch
import torch.nn.functional as F
import requests
import os

class NLPSentimentEngine:
    """
    Motor de Procesamiento de Lenguaje Natural (NLP) basado en FinBERT.
    Puede funcionar como CLIENTE (proxy) o como MOTOR LOCAL (BERT).
    """
    def __init__(self, service_url=None, force_local=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.force_local = force_local
        # Si no se fuerza local, intentamos usar el microservicio
        self.service_url = service_url or os.environ.get("NLP_SERVICE_URL")
        
        if self.force_local or not self.service_url:
            print(f"[NLP-ENGINE] Inicializando modelo FinBERT local en {self.device}...")
            try:
                from transformers import BertTokenizer, BertForSequenceClassification, pipeline
                model_name = "yiyanghkust/finbert-tone"
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.model = BertForSequenceClassification.from_pretrained(model_name).to(self.device)
                self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)
                self.local_mode = True
                print("[NLP-ENGINE] Modelo cargado con éxito.")
            except Exception as e:
                print(f"[NLP-ENGINE] Error cargando modelo local: {e}. El sistema funcionará en modo resiliente.")
                self.local_mode = False
        else:
            print(f"[NLP-CLIENT] Operando como proxy hacia: {self.service_url}")
            self.local_mode = False

    def get_sentiment_vector(self, text_list):
        """
        Retorna: [Positivo, Negativo, Neutral]
        """
        if not text_list:
            return torch.tensor([0.0, 0.0, 1.0], device=self.device)

        if self.local_mode:
            # Procesamiento LOCAL con BERT
            try:
                results = self.nlp(text_list)
                # Promediar sentimientos del batch
                scores = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}
                for res in results:
                    scores[res['label']] += res['score']
                
                n = len(results)
                vector = [scores["Positive"]/n, scores["Negative"]/n, scores["Neutral"]/n]
                return torch.tensor(vector, device=self.device)
            except Exception as e:
                print(f"[NLP-ENGINE] Error en inferencia local: {e}")
                return torch.tensor([0.0, 0.0, 1.0], device=self.device)
        else:
            # Procesamiento REMOTO via Microservicio
            try:
                response = requests.post(f"{self.service_url}/analyze", json=text_list, timeout=60)
                if response.status_code == 200:
                    return torch.tensor(response.json(), device=self.device)
                return torch.tensor([0.0, 0.0, 1.0], device=self.device)
            except:
                return torch.tensor([0.0, 0.0, 1.0], device=self.device)

if __name__ == "__main__":
    # Test
    engine = NLPSentimentEngine()
    news = [
        "Bitcoin price hits new all-time high as institutional adoption grows",
        "Regulatory concerns weigh on crypto markets following new SEC statement",
        "Ethereum network upgrade successfully implemented without issues"
    ]
    
    print("Analizando sentimiento de noticias...")
    vec = engine.get_sentiment_vector(news)
    print(f"Vector de Sentimiento: {vec.cpu().numpy()}")
    print(f"Predominancia: {['Neutral', 'Positivo', 'Negativo'][torch.argmax(vec).item()]}")
