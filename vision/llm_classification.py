import json
import ollama


class LLMClassifier:
    def __init__(self):
        # Load configuration from config_llm.json
        with open('config_llm.json', 'r') as f:
            self.config = json.load(f)
    
    def transform_image(self):
        pass
    
    def configure_network(self):
        # Initialize Ollama client
        self.client = ollama.Client()
        
        # Check if Gemma3 vision model is already downloaded
        try:
            # Try to list models to see if gemma3:vision exists
            models = self.client.list()
            model_exists = any('gemma3:vision' in model['name'] for model in models['models'])
            
            if model_exists:
                print("Gemma3 vision model already downloaded, using existing model.")
                self.model = 'gemma3:vision'
            else:
                print("Downloading Gemma3 vision model...")
                self.model = self.client.pull('gemma3:vision')
                
        except Exception as e:
            print(f"Error checking for existing model: {e}")
            print("Downloading Gemma3 vision model...")
            self.model = self.client.pull('gemma3:vision')
    
    def inference(self):
        pass


def main():
    llm_classifier = LLMClassifier()


if __name__ == "__main__":
    main()
