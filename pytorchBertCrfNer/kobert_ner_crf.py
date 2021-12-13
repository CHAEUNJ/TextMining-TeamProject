from __future__ import absolute_import, division, print_function, unicode_literals
import json
import pickle
import torch

from model.net import KobertCRF
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Tokenizer
from data_utils.pad_sequence import keras_pad_fn

from gluonnlp.data import SentencepieceTokenizer
from pathlib import Path


class DecoderFromNamedEntitySequence:
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def decode_ner_data(self, list_input_ids, list_pred_ids):

        ##### Convert ids -> token, tag #####
        list_input_token = self.tokenizer.decode_token_ids(list_input_ids)[0]
        list_input_token = [word.replace("[UNK]", "") if "[UNK]" in word else word for word in list_input_token]
        list_pred_tag = [self.index_to_ner[pred_id] for pred_id in list_pred_ids[0]]
        #print("Input_token\n", list_input_token, "\n")
        #print("Preg_tag\n", list_pred_tag, "\n")


        ##### Extract ner word from list_pred_tag #####
        #list_ner_word = []
        dict_ner_word = {}
        list_ner_filter = ["LOC", "ORG", "POH"]

        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for idx, pred_tag in enumerate(list_pred_tag):

        ##### NER_TAG starts with B #####
            if "B-" in pred_tag:
                entity_tag = pred_tag[-3:]

                ##### prev entity tag is not O #####
                if prev_entity_tag != entity_tag and prev_entity_tag != "" and prev_entity_tag in list_ner_filter:
                    edited_word = entity_word.replace("▁", "")

                    ##### Check duplicated in list_ner_word #####
                    #cnt_flag = [(data_idx, 1) for data_idx, data in enumerate(list_ner_word) if data["word"] == edited_word]

                    #if cnt_flag:
                    #    list_ner_word[cnt_flag[0][0]]["count"] += 1
                    #else:
                    #    list_ner_word.append({"word": edited_word, "tag": prev_entity_tag, "count": 1})
                    if edited_word in dict_ner_word:
                        dict_ner_word[edited_word] += 1
                    else:
                        dict_ner_word[edited_word] = 1

                entity_word = list_input_token[idx]
                prev_entity_tag = entity_tag

            ##### NER_TAG starts with I #####
            elif "I-" + entity_tag in pred_tag:
                entity_word += list_input_token[idx]

            ##### NER_TAG O tag #####
            else:
                if entity_word != "" and entity_tag != "" and entity_tag in list_ner_filter:
                    edited_word = entity_word.replace("▁", "")

                    ##### Check duplicated in list_ner_word #####
                    #cnt_flag = [(data_idx, 1) for data_idx, data in enumerate(list_ner_word) if data["word"] == edited_word]

                    #if cnt_flag:
                    #    list_ner_word[cnt_flag[0][0]]["count"] += 1
                    #else:
                    #    list_ner_word.append({"word": edited_word, "tag": entity_tag, "count": 1})
                    if edited_word in dict_ner_word:
                        dict_ner_word[edited_word] += 1
                    else:
                        dict_ner_word[edited_word] = 1

                entity_word, entity_tag, prev_entity_tag = "", "", ""

        return dict_ner_word


class KoBERT_NER_CRF:

    def __init__(self):
        ###### Set model path & config #####
        model_dir = Path('./pytorchBertCrfNer/experiments/base_model_with_crf_val')
        # print("model_dir\n", model_dir, "\n")
        model_config = Config(json_path=model_dir / 'config.json')
        # print("model_config\n", model_config.dict, "\n")

        ###### Set tokenizer #####
        ptr_tokenizer = SentencepieceTokenizer("./pytorchBertCrfNer/ptr_lm_model/tokenizer_78b3253a26.model")
        # print("ptr_tokenizer.tokens\n", ptr_tokenizer.tokens, "\n")
        # print("len(ptr_tokenizer.tokens)\n", len(ptr_tokenizer.tokens), "\n")

        ###### Load vocab : token -> idx #####
        with open(model_dir / "vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        # print("len(vocab)\n", len(vocab), "\n")

        ###### Load tokenizer #####
        self.tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn,
                                   maxlen=model_config.dict['maxlen'])

        ###### Load ner_to_index, Set index_to_ner #####
        with open(model_dir / "ner_to_index.json", 'rb') as f:
            ner_to_index = json.load(f)
            index_to_ner = {v: k for k, v in ner_to_index.items()}
        # print("ner_to_index\n", ner_to_index, "\n")
        # print("index_to_ner\n", index_to_ner, "\n")

        ##### Set KoBERT CRF model #####
        self.model = KobertCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

        ##### Load model checkpoint (Trained model) #####
        model_dict = self.model.state_dict()
        checkpoint = torch.load("./pytorchBertCrfNer/experiments/base_model_with_crf_val/best-epoch-12-step-1000-acc-0.960.bin",
                                map_location=torch.device('cpu'))

        convert_keys = {}
        for k, v in checkpoint['model_state_dict'].items():
            new_key_name = k.replace("module.", '')
            if new_key_name not in model_dict:
                print("{} is not int model_dict".format(new_key_name))
                continue
            convert_keys[new_key_name] = v
        # print(convert_keys)

        ##### Set model from checkpoint #####
        device = torch.device('cpu')
        self.model.load_state_dict(convert_keys, strict=False)
        self.model.to(device)
        self.model.eval()

        ##### Set Decoder instance #####
        self.decoder = DecoderFromNamedEntitySequence(tokenizer=self.tokenizer, index_to_ner=index_to_ner)

    def ko_ner_crf(self, input_text):
        ##### Convert input -> ids -> tensor #####
        list_of_input_ids = self.tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
        x_input = torch.tensor(list_of_input_ids).long()

        ##### X input to model #####
        list_of_pred_ids = self.model(x_input)

        ##### Convert ids -> text #####
        return self.decoder.decode_ner_data(list_of_input_ids, list_of_pred_ids)

        #y_output = self.decoder.decode_ner_data(list_of_input_ids, list_of_pred_ids)
        #print("Input_ids\n", list_of_input_ids, "\n")
        #print("Pred_ids\n", list_of_pred_ids, "\n")
        #print(y_output)  # input_token

