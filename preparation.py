import os, shutil, argparse, time
import cv2 as cv
import utils.opencv

""" Script for removing black description (information of the microscope) from images """

def main():
    parser = argparse.ArgumentParser(description="Get substrates without black description")
    parser.add_argument('source_dir', type=str, help='Input dir with files from CVAT')
    parser.add_argument('dest_dir', type=str, help='Output dir with .tif images of cropped substrates')
    args = parser.parse_args()
    
    raw_support_data = args.source_dir
    substrate_dir = args.dest_dir

    if not os.path.isdir(substrate_dir):
        os.mkdir(substrate_dir)

    start = time.time()

    # Collect photos
    for file in os.listdir(raw_support_data):
        if file[-3:] == 'tif':
            shutil.copyfile(f'{raw_support_data}/{file}', f'{substrate_dir}/{file}')
    
    # Crop photos
    for idx, file in enumerate(os.listdir(substrate_dir)):
        image = cv.imread(f'{substrate_dir}/{file}', 0)
        if idx == 0:
            crop_size = utils.opencv.get_size_for_crop(image)
        image = utils.opencv.delete_description(image, crop_size)
        cv.imwrite(f'{substrate_dir}/{file}', image)
    
    print("Successfully !\nTotal time: {0:.4f} sec".format(time.time() - start))

if __name__ == "__main__":
    main()