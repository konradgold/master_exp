from fourm.models.generate import GenerationSampler, build_chained_generation_schedules
from run_generation import get_args, load_model
from fourm.models.fm import FM
import torch
from PIL import Image
import torchvision.transforms as T
from tokenizers import Tokenizer
from fourm.utils.plotting_utils import decode_text
import cv2
import random

def load_video_frame(video_path, video_part=None, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_part is None:
        frame_index = random.randint(0, total_frames - 1)
    elif video_part < 1:
        frame_index = int(video_part * total_frames)
    else:
        frame_index = min(int(video_part), total_frames - 1)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_index} from video")
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(img).unsqueeze(0).view(1, 3, 224, 224).to(device)

def load_image(path, device):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),   # adjust to what 4M expects
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # adjust if needed
    ])

    return transform(img).unsqueeze(0).view(1, 3, 224, 224).to(device)

def main(config_path: str):
    args = get_args(['-c', config_path])
    print(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schedule = build_chained_generation_schedules(
            cond_domains= ['rgb@224'],
            target_domains = ['sam_instance', 'caption'],
            tokens_per_target = [200, 200],
            temp_schedules = ['constant','constant'],
            temps = [1., 1.0],
            autoregression_schemes=['roar','autoregressive'],
            decoding_steps=[1, 1],
            token_decoding_schedules=['top_p','top_p'],
            cfg_scales = [2.,2.],
            cfg_schedules = ['constant', 'constant'],
            cfg_grow_conditioning=True,
        )
    print(schedule)
    
    image_tensor = load_video_frame("/Users/konradgoldenbaum/Downloads/valid/action_5/clip_1.mp4",0.25, device)
    image_tensor2 = load_video_frame("/Users/konradgoldenbaum/Downloads/valid/action_5/clip_1.mp4",0.75, device)
    image_tensor3 = load_video_frame("/Users/konradgoldenbaum/Downloads/valid/action_5/clip_1.mp4",0.25, device)
    image_tensor4 = load_video_frame("/Users/konradgoldenbaum/Downloads/valid/action_5/clip_1.mp4",0.75, device)

    #image_tensor11 = torch.cat([image_tensor1, image_tensor2], dim=3)
    #image_tensor12 = torch.cat([image_tensor3, image_tensor4], dim=3)
    #image_tensor = torch.cat([image_tensor11, image_tensor12], dim=2)

    model = load_model(model_id=args.model, model_class=FM, device= device)
    sampler = GenerationSampler(
        model
    )
    mod_dict = {
        'rgb@224': {
            'tensor': image_tensor,
            'input_mask': torch.full((1,196), False, dtype=torch.bool, device=device),
            'target_mask': torch.full((1,196), True, dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.full((1,196), False, dtype=torch.bool, device=device)
        },
        'sam_instance': {
            'tensor': torch.full((1,256), 0, dtype=torch.int32, device=device),
            'input_mask': torch.full((1,256), True, dtype=torch.bool, device=device),
            'target_mask': torch.full((1,256), False, dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.full((1,256), False, dtype=torch.bool, device=device)
        },
        'caption': {
            'tensor': torch.full((1,256), 0, dtype=torch.int32, device=device),
            'input_mask': torch.full((1,256), True, dtype=torch.bool, device=device),
            'target_mask': torch.full((1,256), False, dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.full((1,256), False, dtype=torch.bool, device=device)
        }
    }

    mod_dict['sam_instance']['tensor'][:,:2] = 5
    mod_dict['caption']['tensor'][:,:2] = 5
    text_tokenizer = Tokenizer.from_file("/Volumes/KG1TB/Developement/master/master_exp/ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json")
    
    result = sampler.generate(
        mod_dict=mod_dict,
        schedule=schedule,
        top_p=0.5,
        text_tokenizer=text_tokenizer,  # often you need to pass the tokenizer used during training
    )

    decoded = decode_text(mod_dict=result, key='caption', text_tokenizer=text_tokenizer)
    print(decoded)


main("/Volumes/KG1TB/Developement/master/master_exp/ml-4m/cfgs/default/generation/models/4m-b_mod21+sr_4m-l_mod7.yaml")

