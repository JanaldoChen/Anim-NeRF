import os
import cv2
import argparse
import shutil
import subprocess

def crop_image(img, rect):
    x, y, w, h = rect

    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1]-(x+w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0]-(y+h)) if y + h >= img.shape[0] else 0

    if img.shape[2] == 4:
        color = [0, 0, 0, 0]
    else:
        color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    x = x + left
    y = y + top

    return new_img[y:(y+h),x:(x+w),:]

def video_to_images(vid_file, frame_freq=1, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = os.path.join('/tmp', os.path.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-r', f'{frame_freq}',
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')


def main(args):

    video_file = args.vid_file
    output_folder = args.output_folder

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    video_to_images(video_file, frame_freq=args.frame_freq, img_folder=output_folder, return_info=True)
    
    filenamelist = os.listdir(output_folder)
    filenamelist.sort()
    for fn in filenamelist:
        img_path = os.path.join(output_folder, fn)
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        LH, LW = args.img_wh[1], args.img_wh[0]
        rect = [W//2+args.offsets[1]-LW//2, H//2+args.offsets[0]-LH//2, LW, LH]
        new_img = crop_image(img, rect)
        cv2.imwrite(img_path, new_img)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str,
                        help='input video path')
    parser.add_argument('--frame_freq', type=int, default=20,
                        help='skip time per frame')
    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--offsets', nargs="+", type=int, default=[0, 0],
                        help='offsets about the center')
    
    args = parser.parse_args()

    main(args)