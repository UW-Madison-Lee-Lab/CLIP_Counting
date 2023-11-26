
import torch,diffusers,os
from clip_count_utils import *
from tqdm import tqdm

DEMO_EXPS = {
    "chair":{
        "count":"four",
        "single":"An old building with ruined walls and antique pink armchairs",
        "multiple":"An old building with ruined walls and four antique pink armchairs",
    },
    "spoon":{
        "count":"two",
        "single":"vintage silver plate tablespoons, serving spoon set of 1847 Rogers Ambassador pattern",
        "multiple":"vintage silver plate tablespoons, serving spoon set of two 1847 Rogers Ambassador pattern",
    },
    "lion":{
        "count":"three",
        "single":"lions",
        "multiple": "three lions",
    }
}

REF_OBJECTS = ['dogs','cats']
FACTORS=[0.2,0.4,0.6,0.8,1]
NUM_IMGS = 10
RANDOM_SEEDS=[42,1024,38] 

def reproduce_stable_diffusion_results(local_directory,pretrained_model_name,device):
    
    data_type=torch.float32
    sd_gen_dir = os.path.join(local_directory, "sd_gen_imgs/")
    if not os.path.exists(sd_gen_dir):
        os.makedirs(sd_gen_dir)    
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(pretrained_model_name,torch_dtype=data_type).to(device)

    for ref_obj in tqdm(REF_OBJECTS, desc="Processing objects"):  # Add progress bar for objects
        ref_single_prompt_embeds,_ = pipe.encode_prompt(
                prompt=[ref_obj],
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
                )
        
        for target_obj in tqdm(DEMO_EXPS, desc="Processing experiments", leave=False):  # Nested progress bar for experiments
            ref_multiple_prompt_embeds,_ = pipe.encode_prompt(
                prompt=[ref_obj+" "+DEMO_EXPS[target_obj]["count"]],
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
                )
            taget_single_prompt_embeds,_ = pipe.encode_prompt(
                prompt=[DEMO_EXPS[target_obj]["single"]],
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
                )
            taget_multiple_prompt_embeds,_ = pipe.encode_prompt(
                prompt=[DEMO_EXPS[target_obj]["multiple"]],
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
                )
            
            difference = ref_multiple_prompt_embeds-ref_single_prompt_embeds
            ref_diff_projection = project_tensor_B_onto_A(ref_single_prompt_embeds, difference)
            ref_diff_projection_2 = project_tensor_B_onto_A(taget_single_prompt_embeds, difference-ref_diff_projection)
            difference = difference - ref_diff_projection - ref_diff_projection_2
            
            for random_seed in tqdm(RANDOM_SEEDS, desc="Processing seeds", leave=False):  # Nested progress bar for seeds
                for factor in FACTORS:
                    new_prompmts=taget_multiple_prompt_embeds + factor * difference
                    generator = torch.Generator(device).manual_seed(random_seed)
                    results_new = pipe(prompt_embeds=new_prompmts,guidance_scale=7.5,num_inference_steps=50,num_images_per_prompt=NUM_IMGS,generator=generator)
                    for img_idx,img in enumerate(results_new.images):
                        filename = os.path.join(sd_gen_dir, f"{ref_obj}_guide_{target_obj}_seed{random_seed}_factor{factor}_img{img_idx}.jpg")
                        img.save(filename)
                    
                generator = torch.Generator(device).manual_seed(random_seed)
                results_org = pipe(prompt=taget_multiple_prompt_embeds,guidance_scale=7.5,num_inference_steps=50,num_images_per_prompt=NUM_IMGS,generator=generator).images
                for img_idx,img in enumerate(results_org.images):
                    filename = os.path.join(sd_gen_dir, f"{target_obj}_org_seed{random_seed}_img{img_idx}.jpg")
                    img.save(filename)
        
    