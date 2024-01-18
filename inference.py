import torch
from time import strftime
import os, sys
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from third_part.GFPGAN.gfpgan import GFPGANer
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
import warnings

warnings.filterwarnings("ignore")


def mkdir_file_dir(file_dir):
    if not os.path.exists(file_dir):
        # 如果不存在则创建文件夹
        os.mkdir(file_dir)
        print(f"文件夹{file_dir}创建成功！")
    else:
        print(f"文件夹{file_dir}已存在！")


def main(args):
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    current_code_path = sys.argv[0]
    current_root_path = os.path.split(current_code_path)[0]
    os.environ['TORCH_HOME'] = os.path.join(current_root_path, args.checkpoint_dir)

    path_of_lm_croper = os.path.join(current_root_path, args.checkpoint_dir, 'shape_predictor_68_face_landmarks.dat')
    path_of_net_recon_model = os.path.join(current_root_path, args.checkpoint_dir, 'epoch_20.pth')
    dir_of_BFM_fitting = os.path.join(current_root_path, args.checkpoint_dir, 'BFM_Fitting')
    wav2lip_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'wav2lip.pth')

    audio2pose_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'auido2pose_00140-model.pth')
    audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')

    audio2exp_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'auido2exp_00300-model.pth')
    audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

    free_view_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'facevid2vid_00189-model.pth.tar')

    mapping_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'mapping_00109-model.pth.tar')
    facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender_still.yaml')

    # init model
    preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, args.device)
    audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, audio2exp_checkpoint, audio2exp_yaml_path,
                                 wav2lip_checkpoint, args.device)
    animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, facerender_yaml_path, args.device)

    restorer_model = GFPGANer(model_path='checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean',
                              channel_multiplier=2, bg_upsampler=None)
    enhancer_model = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False,
                                     sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1,
                                     device=args.device)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    # input_path, save_dir
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        input_path=args.source_video_or_image,
        save_dir=first_frame_dir,
        crop_or_resize='crop',
        source_image_flag=False
    )
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return
    # audio2ceoff
    batch = get_data(first_coeff_path, args.driven_audio, args.device)
    coeff_path = audio_to_coeff.generate(batch, save_dir)
    # coeff2video
    data = get_facerender_data(
        source_video_or_image=args.source_video_or_image,
        coeff_path=coeff_path,
        pic_path=crop_pic_path,
        first_coeff_path=first_coeff_path,
        audio_path=args.driven_audio,
        batch_size=args.batch_size,
        size=args.size,
        device=args.device,
        input_yaw_list=args.input_yaw,
        input_pitch_list=args.input_pitch,
        input_roll_list=args.input_roll
    )
    tmp_path, new_audio_path, return_path = animate_from_coeff.generate(
        x=data,
        video_save_dir=save_dir,
        pic_path=args.source_video_or_image,
        crop_info=crop_info,
        restorer_model=restorer_model,
        enhancer_model=enhancer_model,
        enhancer_region=args.enhancer_region,
        enhancer=args.enhancer,
        background_enhancer=args.background_enhancer,
        preprocess=args.preprocess
    )
    torch.cuda.empty_cache()
    if args.use_DAIN:
        import paddle
        from src.dain_model import dain_predictor
        paddle.enable_static()
        mkdir_file_dir(args.dian_output)
        predictor_dian = dain_predictor.DAINPredictor(args.dian_output, weight_path=args.DAIN_weight,
                                                      time_step=args.time_step,
                                                      remove_duplicates=args.remove_duplicates)
        frames_path, temp_video_path = predictor_dian.run(tmp_path)
        paddle.disable_static()
        save_path = return_path[:-4] + '_dain.mp4'
        command = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (temp_video_path, new_audio_path, save_path)
        os.system(command)
    os.remove(tmp_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--driven_audio",
                        default='/mk_data/znn/mkios_shuziren/origin_sadtalker/examples/my_test/测试语音1.mp3',
                        help="path to driven audio")
    parser.add_argument("--source_video_or_image",
                        # default='/mk_data/znn/mkios_shuziren/origin_sadtalker/examples/my_test/立体模特.mp4',
                        default='/mk_data/znn/mkios_shuziren/origin_sadtalker/examples/my_test/测试2.jpg',
                        help="path to source image or video")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    # todo batch size如果不等于1会出现合成的人脸徐晃，无法做到加速
    parser.add_argument("--batch_size", type=int, default=1, help="the batch size of facerender")
    parser.add_argument("--enhancer_region", type=str, default='lip', help="enhaner region:[none,lip,face] \
                                                                      none:do not enhance; \
                                                                      lip:only enhance lip region \
                                                                      face: enhance (skin nose eye brow lip) region")
    parser.add_argument("--size", type=int, default=256, help="the image size of the facerender")
    parser.add_argument("--preprocess", default='full', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" )

    parser.add_argument('--enhancer', type=str, default="gfpgan", help="Face enhancer, [none,gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer', type=str, default="realesrgan",
                        help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", help="是否使用cpu进行推理")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument("--use_DAIN", dest="use_DAIN", default=False, help="Depth-Aware Video Frame Interpolation")
    parser.add_argument('--DAIN_weight', type=str, default='./checkpoints/DAIN_weight', help='Path to model weight')
    parser.add_argument('--dian_output', type=str, default='./dian_output', help='output dir')
    parser.add_argument('--time_step', type=float, default=0.5, help='choose the time steps')
    parser.add_argument('--remove_duplicates', action='store_true', default=True,
                        help='whether to remove duplicated frames')

    args = parser.parse_args()
    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"
    main(args)
