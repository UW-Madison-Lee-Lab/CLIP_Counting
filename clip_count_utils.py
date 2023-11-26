import torch
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm
from requests.exceptions import Timeout
spacy_nlp = spacy.load("en_core_web_sm")

OBJ_NAMES = {
    'dogs':"dogs", 
    'lions':"lions", 
    'chairs':"chairs", 
    'laptops':"laptops",
    'home_cat':"cats", 
    'outside_cats':"cats", 
    'cartoon_cats':"cats",
    'goats':'goats',
    'cows':'cows', 
    'cherries':'cherries', 
    'roses':'roses', 
    'boats':'boats',
}
NUMBER_WORDS_SUB = [
        "two", "three", "four", "five",
    ]
NUMBER_WORDS = [
    "two", "three", "four", "five",
    "six", "seven", "eight", "nine",
    'ten'
]
SUB_NUMBER_RANGE=[2,3,4,5]

def project_tensor_B_onto_A(A, B):

    # Ensure the tensors have the correct shape
    assert A.shape == B.shape, "Tensors must have same shape"

    # Initialize the resulting tensor
    proj_B_on_A = torch.zeros_like(A)

    # Project each vector of B onto A
    for i in range(A.shape[1]):
        a = A[0,i,:]
        b = B[0,i,:]

        dot_product_b_a = torch.dot(b, a)
        dot_product_a_a = torch.dot(a, a)

        projection = (dot_product_b_a / dot_product_a_a) * a
        proj_B_on_A[0,i,:] = projection

    return proj_B_on_A

def get_prompt_embeds(model,input_ids,attention_mask,normalize=True):
    text_outputs = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                output_attentions=model.config.output_attentions,
                output_hidden_states=model.config.output_hidden_states,
                return_dict=model.config.use_return_dict,
            )
    text_embeds = text_outputs[1] #torch.Size([9, 512])
    text_embeds = model.text_projection(text_embeds) #torch.Size([9, 512])
    if normalize:
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return text_embeds

def text2embedding(text,model,processor,device,normalize):
    inputs = processor(text=text, images=None, return_tensors="pt", padding=True)
    if inputs["input_ids"].shape[1] > 77:
        last_id = inputs["input_ids"][:,[-1]]
        inputs["input_ids"] = torch.concat([inputs["input_ids"][:,:76],last_id],dim=1)
        last_mask = inputs["attention_mask"][:,[-1]]
        inputs["attention_mask"] = torch.concat([inputs["attention_mask"][:,:76],last_mask],dim=1)
    return get_prompt_embeds(
        model=model,
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        normalize=normalize,
    )# torch.Size([k, 512])

def get_target_rep(target_object,target_aug_sentences,model,processor,device="cpu",normalize=True):
    target_object_prompt_embeds = text2embedding(target_object,model,processor,device,normalize)
    target_aug_text_embeds = text2embedding(target_aug_sentences,model,processor,device,normalize)

    return target_object_prompt_embeds,target_aug_text_embeds


def get_ref_difference(ref_aug_sentences,ref_object,model,processor,device="cpu",normalize=True):
    ref_prompt_multi = text2embedding(ref_aug_sentences,model,processor,device,normalize)
    ref_prompt_single = text2embedding(ref_object,model,processor,device,normalize)
    ref_diff=ref_prompt_multi-ref_prompt_single # torch.Size([9, 512])
    return ref_diff,ref_prompt_single

def apply_reff_diff(start,end,ref_diff,factor,linear_shift,start_with_target_with_num):
        if linear_shift:
            if not start_with_target_with_num:
                merged_text_embeds=factor*(start+ref_diff)+(1-factor)*end
            else: 
                merged_text_embeds = end + factor * ref_diff
        else:
            raise NotImplementedError
        return merged_text_embeds

def get_image_embeds(model,pixel_values,device="cpu"):
    vision_outputs = model.vision_model(
            pixel_values=pixel_values.to(device),
            output_attentions=model.config.output_attentions,
            output_hidden_states=model.config.output_hidden_states,
            return_dict=model.config.use_return_dict,
        )
    image_embeds = vision_outputs[1]
    image_embeds = model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    return image_embeds

def get_logits(model,text_embeds,image_embeds):
    # print(model.device,text_embeds.device,image_embeds.device)
    logit_scale = model.logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    logits_per_image = logits_per_text.t()

    return logits_per_text,logits_per_image


def run_on_my_data_img_retrievel(model,processor,target_data,target,ref,normalize,device,factor,sample_size=10,num_classes=4,linear_shift=True,start_with_target_with_num=False,normalize_before_scoring=False):
    ref_aug_sentences=[f"{word} {OBJ_NAMES[ref]}" for word in NUMBER_WORDS[:num_classes]]
    target_aug_sentences=[f"{word} {OBJ_NAMES[target]}" for word in NUMBER_WORDS[:num_classes]]

    ref_diff,ref_prompt_single=get_ref_difference(
        ref_aug_sentences=ref_aug_sentences,
        ref_object=[OBJ_NAMES[ref]],
        model=model,
        processor=processor,
        device=device,
        normalize=normalize
    )
    target_text_sample={}
    target_text_sample["target_obj_embeds"],target_text_sample["target_obj_aug_embeds"]=get_target_rep(
        target_object=[OBJ_NAMES[target]],
        target_aug_sentences=target_aug_sentences,
        model=model,
        processor=processor,
        device=device,
        normalize=normalize
    )
    ref_diff_projection = (torch.mm(ref_diff, ref_prompt_single.t()) / torch.mm(ref_prompt_single, ref_prompt_single.t()).squeeze()) * ref_prompt_single
    ref_diff_projection_2 = (torch.sum(ref_diff-ref_diff_projection * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True)/torch.sum(target_text_sample["target_obj_aug_embeds"] * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True))*target_text_sample["target_obj_aug_embeds"]
    ref_diff = ref_diff - ref_diff_projection - ref_diff_projection_2 
    merged_text_embeds = apply_reff_diff(target_text_sample["target_obj_embeds"],target_text_sample["target_obj_aug_embeds"],ref_diff,factor,linear_shift,start_with_target_with_num)
        
    # if normalize_before_scoring:
    merged_text_embeds=merged_text_embeds/merged_text_embeds.norm(p=2,dim=-1,keepdim=True)

    all_probs_per_class=[]
    for text_num in range(2,num_classes+2):
        all_logits_per_text = []
        merged_text_embeds_ = merged_text_embeds[text_num-2]
        for number in range(2,num_classes+2):
            
            # print(merged_text_embeds_.size())
            selected_data = target_data[number][:sample_size]
            for sample in selected_data:
                pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"] # torch.Size([1, 3, 224, 224])
                image_embeds = get_image_embeds(
                    model=model,
                    pixel_values=pixel_values.to(device),
                    device=device
                )
                logits_per_text,_= get_logits(model,merged_text_embeds_,image_embeds)
                all_logits_per_text.append(logits_per_text.item())
            torch.cuda.empty_cache()

        all_probs_per_class.append(torch.nn.functional.softmax(torch.tensor(all_logits_per_text).float(),dim=0).reshape(num_classes,sample_size).sum(dim=1).numpy().tolist())
    
    return all_probs_per_class

def run_on_my_data_clf(model,processor,target_data,target,ref,normalize,device,sample_size,num_classes=4,linear_shift=True,start_with_target_with_num=False):
    ref_aug_sentences=[f"{word} {OBJ_NAMES[ref]}" for word in NUMBER_WORDS[:num_classes]]
    target_aug_sentences=[f"{word} {OBJ_NAMES[target]}" for word in NUMBER_WORDS[:num_classes]]

    ref_diff,ref_prompt_single=get_ref_difference(
        ref_aug_sentences=ref_aug_sentences,
        ref_object=[OBJ_NAMES[ref]],
        model=model,
        processor=processor,
        device=device,
        normalize=normalize
    )
    target_text_sample={}
    target_text_sample["target_obj_embeds"],target_text_sample["target_obj_aug_embeds"]=get_target_rep(
        target_object=[OBJ_NAMES[target]],
        target_aug_sentences=target_aug_sentences,
        model=model,
        processor=processor,
        device=device,
        normalize=normalize
    )

    ref_diff_projection = (torch.mm(ref_diff, ref_prompt_single.t()) / torch.mm(ref_prompt_single, ref_prompt_single.t()).squeeze()) * ref_prompt_single
    ref_diff_projection_2 = (torch.mm(ref_diff-ref_diff_projection, target_text_sample["target_obj_embeds"].t()) / torch.mm(target_text_sample["target_obj_embeds"], target_text_sample["target_obj_embeds"].t()).squeeze()) * target_text_sample["target_obj_embeds"]
    # ref_diff_projection_2 = (torch.sum(ref_diff-ref_diff_projection * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True)/torch.sum(target_text_sample["target_obj_aug_embeds"] * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True))*target_text_sample["target_obj_aug_embeds"]
    ref_diff = ref_diff - ref_diff_projection - ref_diff_projection_2 #+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)

    merged_text_embeds = apply_reff_diff(target_text_sample["target_obj_aug_embeds"],target_text_sample["target_obj_aug_embeds"],ref_diff,1,linear_shift,start_with_target_with_num)

    # if normalize_before_scoring:
    merged_text_embeds=merged_text_embeds/merged_text_embeds.norm(p=2,dim=-1,keepdim=True)

    flat_predictions=[]
    flat_labels=[]
    for number in range(2,num_classes+2):
        if sample_size is None:
            selected_data = target_data[number]
        else:
            selected_data = target_data[number][:sample_size]
        predictions=[]
        for sample in selected_data:
            pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"] # torch.Size([1, 3, 224, 224])
            image_embeds = get_image_embeds(
                model=model,
                pixel_values=pixel_values.to(device),
                device=device
            )
            _,logits_per_image= get_logits(model,merged_text_embeds,image_embeds)
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label prob
            predictions.append(torch.argmax(probs).item()+2)
        flat_predictions+=predictions
        flat_labels+=[number]*len(predictions)

    acc=np.mean(np.array(flat_predictions)==np.array(flat_labels)).round(4)*100

    return flat_predictions,flat_labels,acc

def run_countbench_sample(sample,model,processor,normalize,factor,num_classes,ref_obj,\
                      linear_shift=True,device="cpu",use_ref_with_context=False,start_with_target_with_num=True,\
                        use_target_obj_with_context=False,use_target_aug_sent_with_context=True):
    if not use_target_obj_with_context:
        start=sample["target_obj_embeds"].to(device)
    else:
        start=sample["target_obj_embeds_with_context"].to(device)
    if not use_target_aug_sent_with_context:
        end=sample["target_obj_aug_embeds"][:num_classes].to(device)
    else:
        end=sample["target_obj_aug_embeds_with_context"][:num_classes].to(device)
    
    if factor == 0:
        merged_text_embeds=end
    else:
        if use_ref_with_context:
            ref_diff_per_sample,ref_prompt_single = get_ref_difference(
                ref_aug_sentences=[ele.replace(sample["target_obj"],ref_obj) for ele in sample["target_obj_aug_with_context"]][:num_classes],
                ref_object=sample["target_obj_with_context"].replace(sample["target_obj"],ref_obj),
                model=model,
                processor=processor,
                device=device,
                normalize=normalize
            )
        else:
            ref_diff_per_sample,ref_prompt_single= get_ref_difference(
                ref_aug_sentences=[f"{number} {ref_obj}" for number in NUMBER_WORDS[:num_classes]],
                ref_object=[ref_obj],
                model=model,
                processor=processor,
                device=device,
                normalize=normalize
            )
        ref_diff_projection = (torch.mm(ref_diff_per_sample, ref_prompt_single.t()) / torch.mm(ref_prompt_single, ref_prompt_single.t()).squeeze()) * ref_prompt_single
        ref_diff_aligned_B = (torch.sum(ref_diff_per_sample-ref_diff_projection * end, dim=1, keepdim=True)/torch.sum(end * end, dim=1, keepdim=True))*end

        ref_diff_per_sample = ref_diff_per_sample - ref_diff_projection - ref_diff_aligned_B #+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
        merged_text_embeds = apply_reff_diff(start,end,ref_diff_per_sample,factor,linear_shift,start_with_target_with_num)
        
    merged_text_embeds=merged_text_embeds/merged_text_embeds.norm(p=2,dim=-1,keepdim=True)
    merged_text_embeds=merged_text_embeds[:num_classes]
    return merged_text_embeds

def contains_word(sentence, word_list):
    words = sentence.lower().split()
    for word in word_list:
        if word.lower() in words:
            return True, word
    return False, None

def sentence_augmentation(sentence):
    sentence=sentence.lower()

    new_sentences = []
    obj_with_nums = []
    object_name = ""
    sentence_no_num=None

    contains,word=contains_word(sentence, NUMBER_WORDS)
    if contains:
        doc = spacy_nlp(sentence.split(word)[1])
        for np_ in doc.noun_chunks:
            if np_.text.strip() is not None:
                object_name = np_.text.strip()
            break
        if object_name!="":
            if (sentence.split(word)[1].split(object_name)[0].strip()!=""):
                object_name=sentence.split(word)[1].split(object_name)[0].strip()+" "+object_name
        for number_word in NUMBER_WORDS:
            new_sentences.append(sentence.replace(word, number_word))
            obj_with_nums.append(f"{number_word} {object_name}")
        sentence_no_num=sentence.replace(f"{word} ","")

    return new_sentences,object_name,sentence_no_num,obj_with_nums


def countbench_streaming_data(sample,model,processor,device="cpu",number=None,normalize=True):
    if (number is not None) and (sample["number"] != number):
        return None
    if sample["image"] is None:
        try:
            image = Image.open(requests.get(sample["image_url"], stream=True,timeout=2).raw)
        except Timeout:
            print("timeout")
            return None
        except:
            # print(f"Error loading {sample['image_url']}")
            return None
    else:
        image = sample["image"]
    target_obj_aug_with_context,target_obj,target_obj_with_context,target_obj_aug = sentence_augmentation(sample["text"])

    target_obj_embeds,target_obj_aug_embeds=get_target_rep(
        target_object=target_obj,
        target_aug_sentences=target_obj_aug,
        model=model,
        processor=processor,
        normalize=normalize,
        device=device,
    )
    target_obj_embeds_with_context,target_obj_aug_embeds_with_context=get_target_rep(
        target_object=target_obj_with_context,
        target_aug_sentences=target_obj_aug_with_context,
        model=model,
        processor=processor,
        normalize=normalize,
        device=device,
    )

    pixel_values=processor(text=None, images=image, return_tensors="pt", padding=True)["pixel_values"] # torch.Size([1, 3, 224, 224])
    image_embeds = get_image_embeds(
        model=model,
        pixel_values=pixel_values,
        device=device,
    ) 

    return {
            "number":sample['number'],
            "target_obj":target_obj,
            "target_obj_embeds":target_obj_embeds,
            "target_obj_aug":target_obj_aug,
            "target_obj_aug_embeds":target_obj_aug_embeds,
            "target_obj_with_context":target_obj_with_context,
            "target_obj_embeds_with_context":target_obj_embeds_with_context,
            "target_obj_aug_with_context":target_obj_aug_with_context,
            "target_obj_aug_embeds_with_context":target_obj_aug_embeds_with_context,
            "image_embeds":image_embeds,
        }

def run_on_countbench(model,processor,normalize,factor,my_count_bench_dataset,num_classes,ref_type,ref_obj=None,constant_ref_diff=None,ll_layer=None,\
                      linear_shift=True,sample_size=48,device="cpu",use_ref_with_context=False,start_with_target_with_num=False,\
                        use_target_obj_with_context=False,use_target_aug_sent_with_context=False,ref_obj_list=None,
                        use_context_for_similarity=False,normalize_sim=False,normalize_before_scoring=False,train_set=None,number=None):
    print("ref_type",ref_type)
    
    
    predictions = []
    if my_count_bench_dataset is not None:
        for i, sample in enumerate(my_count_bench_dataset[:sample_size]):
            merged_text_embeds = run_countbench_sample(sample,model,processor,normalize,factor,num_classes,ref_obj,\
                      linear_shift=linear_shift,device=device,use_ref_with_context=use_ref_with_context,start_with_target_with_num=start_with_target_with_num,\
                        use_target_obj_with_context=use_target_obj_with_context,use_target_aug_sent_with_context=use_target_aug_sent_with_context)
                
            # if normalize_before_scoring:
            _,logits_per_image=get_logits(model,merged_text_embeds,sample["image_embeds"].to(device))
            predictions.append(torch.argmax(logits_per_image,dim=1).item()+2)
    else:
        for i, raw_sample in tqdm(enumerate(train_set)):
            sample = countbench_streaming_data(raw_sample,model,processor,device,number,normalize)
            if sample is not None:
                merged_text_embeds = run_countbench_sample(sample,model,processor,normalize,factor,num_classes,ref_obj,\
                      linear_shift=linear_shift,device=device,use_ref_with_context=use_ref_with_context,start_with_target_with_num=start_with_target_with_num,\
                        use_target_obj_with_context=use_target_obj_with_context,use_target_aug_sent_with_context=use_target_aug_sent_with_context)
                    
                # if normalize_before_scoring:
                _,logits_per_image=get_logits(model,merged_text_embeds,sample["image_embeds"].to(device))
                predictions.append(torch.argmax(logits_per_image,dim=1).item()+2)
        
    return predictions