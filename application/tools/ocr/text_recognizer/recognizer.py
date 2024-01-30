import os
import yaml
import torch
import numpy as np
from collections import OrderedDict

import torch.utils.data
import torch.nn.functional as F

from .model import Model
from .utils.img_proc import custom_mean
from .utils.collate import AlignCollate
from .utils.general import AttrDict, CTCLabelConverter, ListDataset

class Recognizer:
    def __init__(self, recognizer_path, workers = 1, decoder = 'greedy', beamWidth = 5) -> None:
        self.device = 'cpu'
        self.quantize = True
        model_path = ''
        for model_name in os.listdir(recognizer_path):
            if model_name.endswith('.pth'):
                print (model_name)
                model_path = os.path.join(recognizer_path, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError("model not found")
        config_path = os.path.join(recognizer_path, model_name.replace('.pth', '.yaml'))
        if not os.path.exists(config_path):
            raise FileNotFoundError("config yaml not found")

        self.get_recognizer(model_path, config_path)
        self.model.eval()
        self.decoder = decoder
        self.beamWidth = beamWidth
        self.workers = workers

    def recognizer_predict(self, test_loader, batch_max_length):
        result = []
        with torch.no_grad():
            for image_tensors in test_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)
                # For max length prediction
                length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(self.device)

                preds = self.model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)

                ######## filter ignore_char, rebalance
                preds_prob = F.softmax(preds, dim=2)
                preds_prob = preds_prob.cpu().detach().numpy()
                preds_prob[:,:,[]] = 0.
                pred_norm = preds_prob.sum(axis=2)
                preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1)
                preds_prob = torch.from_numpy(preds_prob).float().to(self.device)

                if self.decoder == 'greedy':
                    # Select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds_prob.max(2)
                    preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)
                elif self.decoder == 'beamsearch':
                    k = preds_prob.cpu().detach().numpy()
                    preds_str = self.converter.decode_beamsearch(k, beamWidth=self.beamWidth)
                elif self.decoder == 'wordbeamsearch':
                    k = preds_prob.cpu().detach().numpy()
                    preds_str = self.converter.decode_wordbeamsearch(k, beamWidth=self.beamWidth)

                preds_prob = preds_prob.cpu().detach().numpy()
                values = preds_prob.max(axis=2)
                indices = preds_prob.argmax(axis=2)
                preds_max_prob = []
                for v,i in zip(values, indices):
                    max_probs = v[i!=0]
                    if len(max_probs)>0:
                        preds_max_prob.append(max_probs)
                    else:
                        preds_max_prob.append(np.array([0]))

                for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                    confidence_score = custom_mean(pred_max_prob)
                    result.append([pred, confidence_score])

        return result

    def get_recognizer(self, model_path, config_path, separator_list={}):
        
        dict_list = {"en" : os.path.join("languages/en.txt")}

        with open(config_path, 'r', encoding="utf8") as stream:
            opt = yaml.safe_load(stream)
        opt = AttrDict(opt)

        character = opt.number + opt.symbol + opt.lang_char
        self.imgH = opt.imgH

        self.converter = CTCLabelConverter(character, separator_list, dict_list)
        num_class = len(self.converter.character)

        self.model = Model(opt=opt, num_class=num_class)

        if self.device == 'cpu':
            state_dict = torch.load(model_path, map_location=self.device)
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key[7:]
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict)
            if self.quantize:
                try:
                    torch.quantization.quantize_dynamic(self.model, dtype=torch.qint8, inplace=True)
                except:
                    pass
        else:
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def get_text(self, imgW, image_list, batch_size=1, contrast_ths=0.1, adjust_contrast=0.5):
        batch_max_length = int(imgW/10)

        coord = [item[0] for item in image_list]
        img_list = [item[1] for item in image_list]

        test_loader = torch.utils.data.DataLoader(ListDataset(img_list), 
                                                  batch_size=batch_size, 
                                                  shuffle=False,
                                                  num_workers=int(self.workers), 
                                                  collate_fn= AlignCollate(imgH=self.imgH, imgW=imgW, keep_ratio_with_pad=True), 
                                                  pin_memory=True)

        # predict first round
        result1 = self.recognizer_predict(test_loader, batch_max_length)

        # predict second round
        low_confident_idx = [i for i,item in enumerate(result1) if (item[1] < contrast_ths)]
        if len(low_confident_idx) > 0:
            img_list2 = [img_list[i] for i in low_confident_idx]

            test_loader = torch.utils.data.DataLoader(ListDataset(img_list2),
                                                      batch_size=batch_size, 
                                                      shuffle=False, 
                                                      num_workers=int(self.workers), 
                                                      collate_fn=AlignCollate(imgH=self.imgH, imgW=imgW, keep_ratio_with_pad=True, adjust_contrast=adjust_contrast), 
                                                      pin_memory=True)
            
            result2 = self.recognizer_predict(test_loader, batch_max_length)

        result = []
        for i, zipped in enumerate(zip(coord, result1)):
            box, pred1 = zipped
            if i in low_confident_idx:
                pred2 = result2[low_confident_idx.index(i)]
                if pred1[1]>pred2[1]:
                    result.append( (box, pred1[0], pred1[1]) )
                else:
                    result.append( (box, pred2[0], pred2[1]) )
            else:
                result.append( (box, pred1[0], pred1[1]) )

        return result

