# from environment import *
from utils import *
from clip_count_utils import *
import os

# model_name= "openai/clip-vit-base-patch32" # "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32" "openai/clip-vit-base-patch16"
# model_name="openai/clip-vit-large-patch14"
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cuda"
# model = CLIPModel.from_pretrained(model_name).to(device)
# model.requires_grad=False
# processor = CLIPProcessor.from_pretrained(model_name)


def image_retrievel(model_name,ref_obj,sample_size,augmented_data,local_directory,device="cpu"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.requires_grad=False
    processor = CLIPProcessor.from_pretrained(model_name)

    normalize=False
    # sample_size = len(augmented_data['dogs'][2])
    num_classes = 4
    linear_shift=True
    task = 'image_retrievel'
    start_with_target_with_num = True


    all_probs_by_factors = []
    all_mean_probs_by_factors = []
    all_probs_by_target = []
    for factor in [0,1]: # output original results and results after applying our method
        # all_probs_by_target = []
        all_mean_probs_by_target = []
        for target in augmented_data.keys():
            all_probs_per_class=run_on_my_data_img_retrievel(
                model=model,
                processor=processor,
                target_data= augmented_data[target],
                target=target,
                ref=ref_obj,
                normalize=normalize,
                device=device,
                factor=factor,
                sample_size=sample_size,
                num_classes=num_classes,
                linear_shift=linear_shift,
                start_with_target_with_num=start_with_target_with_num)
            mean_prob = np.mean([all_probs_per_class[i][i] for i in range(len(all_probs_per_class))])
            all_mean_probs_by_target.append(mean_prob)
            # all_probs_by_target.append(all_probs_per_class)
        # all_probs_by_factors.append(all_probs_by_target)
        all_mean_probs_by_factors.append(all_mean_probs_by_target)

    # pb_pd = pd.DataFrame(all_probs_by_factors,columns=list(augmented_data.keys()))
    # pb_pd.index = factors_list[1:]
    # pb_pd.to_csv(f"csv/final/{fn}")

    mean_pb_pd = pd.DataFrame(all_mean_probs_by_factors,columns=list(augmented_data.keys()))
    mean_pb_pd.index = [[ele]*len(all_mean_probs_by_target) for ele in [0,1]]
    mean_pb_pd["average"] = np.array(all_mean_probs_by_factors).mean(axis=1)
    mean_pb_pd.to_csv(os.path.join(local_directory,get_file_name(task,model_name,ref_obj,data_name="custom_data",num_classes=num_classes)))


def img_clf(model_name,ref_obj,sample_size,augmented_data,local_directory,device="cpu"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.requires_grad=False
    processor = CLIPProcessor.from_pretrained(model_name)
    normalize=False
    num_classes = 4
    linear_shift=True
    factors_list = [0,0.2]
    task = 'img_clf'
    start_with_target_with_num = True

    acc_by_factors=[]
    for factor in factors_list:
        acc_by_target=[]
        for target in augmented_data.keys():
            _,_,acc=run_on_my_data(
                model=model,
                processor=processor,
                target_data= augmented_data[target],
                target=target,
                ref=ref_obj,
                normalize=normalize,
                device=device,
                factor=factor,
                sample_size=sample_size,
                num_classes=num_classes,
                linear_shift=linear_shift,
                start_with_target_with_num=start_with_target_with_num)
            acc_by_target.append(acc)
        acc_by_factors.append(acc_by_target)
    acc_pd = pd.DataFrame(np.array(acc_by_factors),columns=list(augmented_data.keys()))
    acc_pd.index = factors_list
    acc_pd["average"] = np.array(acc_by_factors).mean(axis=1)
    acc_pd.to_csv(os.path.join(local_directory,get_file_name(task,model_name,ref_obj,data_name="",num_classes="")))

