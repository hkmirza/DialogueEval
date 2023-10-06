import random
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BlenderbotSmallForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel, AutoModelForSeq2SeqLM,
)
from tqdm.auto import tqdm
from pathlib import Path
path = lambda p: Path(p).absolute().resolve()


class PersonaManager:
    def __init__(self):
        '''
        Please put "personas.txt" and "utils.py" in the same directory.
        You can also set `persona_file` as the absolute path to "degraded_random_responses_filtered.txt" file.
        '''
        persona_file = path(__file__).parent.joinpath("personas.txt")
        with open(persona_file) as f:
            self.all_personas = f.read().strip().splitlines()
        
    def get_persona(self, seed):
        # seed is randomly generated between 0~1000, where any seed > 500 means no persona
        if seed > 500:
            return []
        random.seed(int(seed))
        return random.sample(self.all_personas, k=5)
    
    def get_single_persona(self):
        random.seed()
        return random.choice(self.all_personas)


class DialogueModels:
    def __init__(self):
        modelclass_dict = {
            "qc": QcModel,
            "bart": PersonaChatBART,
            "dialogpt": PersonaChatGPT,
            'vanilla_dialogpt': VanillaDialoGPT,
            'vanilla_blenderbot_small': VanillaBlenderbotSmall,
        }
        self.models = {}
        pbar = tqdm(modelclass_dict.items())
        for modelname,modelclass in pbar:
            pbar.set_description(f"Loading {modelclass.__name__}")
            self.models[modelname] = modelclass()
        
        self.modelnames = list(self.models.keys())

    def get_response(self,data):
        model = data['model']
        response = self.models[model].response(
            data['user_input'],
            data['history'],
            data['personas'],
            
        )
        return response



class QcModel:
    def __init__(self):
        '''
        Please put "degraded_random_responses_filtered.txt" and "utils.py" in the same directory.
        You can also set `qc_response_file` as the absolute path to "degraded_random_responses_filtered.txt" file.
        '''
        qc_response_file = path(__file__).parent.joinpath("degraded_random_responses_filtered.txt")
        with open(qc_response_file) as f:
            self.all_qc_responses = f.read().strip().splitlines()
        
    def response(self, *args, **kwarg):
        random.seed() # reset seed 
        return random.choice(self.all_qc_responses)


class PersonaChatBART:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bart-base-en-persona-chat")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("DeepPavlov/bart-base-en-persona-chat")
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
        self.bos = self.tokenizer.bos_token
    
    def preprocess_persona(self,personas):
        return " ".join(personas)
  
    def preprocess_dial_history(self,dial_history):
        return " EOS ".join(dial_history)
        
    def response(self,user_input,dial_history,personas):
        persona_txt = self.preprocess_persona(personas)
        history_txt = self.preprocess_dial_history(dial_history + [user_input]) # PersonaChatBART put the user input together with dialogue history
        full_input_txt = f"{self.bos} [CONTEXT] {history_txt} [KNOWLEDGE] {persona_txt} {self.eos}"

        full_input_ids = self.tokenizer.encode(full_input_txt,return_tensors='pt')
        conversation_ids = self.model.generate(full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.batch_decode(conversation_ids, skip_special_tokens=True)[0]
        return bot_respond

class PersonaChatGPT:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")
        self.model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
        
    def preprocess_persona(self,personas):
        # personas_fact = [f'Fact: {p}' for p in personas]
        personas_fact = [f'{p}{self.eos}' for p in personas]
        full_persona = ''.join(['<|p2|>'] + personas_fact + ['<|sep|>'] + ['<|start|>'])
        return full_persona
        
    def preprocess_dial_history(self,dial_history):
        return self.eos.join(dial_history)
    
    def response(self,user_input,dial_history,personas):
        persona_txt = self.preprocess_persona(personas)
        history_txt = self.preprocess_dial_history(dial_history)
        full_input_txt = f"{persona_txt}{history_txt}{self.eos}{user_input}{self.eos}"
        full_input_ids = self.tokenizer.encode(full_input_txt,return_tensors='pt')
        
        conversation_ids = self.model.generate(full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.decode(conversation_ids[:, full_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_respond

class VanillaDialoGPT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
    
    def preprocess_persona(self,personas):
        return " ".join(personas)
    
    def preprocess_dial_history(self,dial_history):
        return " ".join(dial_history)
    
    def response(self,user_input,dial_history,personas):
        persona_txt = self.preprocess_persona(personas)
        history_txt = self.preprocess_dial_history(dial_history)
        full_input_txt = f"{persona_txt}{self.eos}{history_txt}{self.eos}{user_input}{self.eos}"
        full_input_ids = self.tokenizer.encode(full_input_txt,return_tensors='pt')
        conversation_ids = self.model.generate(full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.decode(conversation_ids[:, full_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_respond

class VanillaBlenderbotSmall:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
        self.bos = self.tokenizer.bos_token
    
    def preprocess_persona(self,personas):
        return "</s> <s>".join(personas)
  
    def preprocess_dial_history(self,dial_history):
        return "</s> <s>".join(dial_history)
    
    def response(self,user_input,dial_history,personas):
        persona_txt = self.preprocess_persona(personas)
        history_txt = self.preprocess_dial_history(dial_history)
        full_input_txt = f"{self.bos}{persona_txt}{self.eos}{history_txt}{self.eos}{user_input}{self.eos}"
        full_input_ids = self.tokenizer([full_input_txt],return_tensors='pt')
        conversation_ids = self.model.generate(**full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.batch_decode(conversation_ids, skip_special_tokens=True)[0]
        return bot_respond