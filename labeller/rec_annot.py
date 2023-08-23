import os
import cv2
import matplotlib.pyplot as plt
import time

input_folder = input("Enter input images folder path: ")
txt_file_path = 'marker/'+input_folder+'_preds.txt'
updated_file_path = 'corrected_labels/'+input_folder+'.txt'
base_image_folder = 'rec_crops/'+input_folder+'_crops'
# base_image_folder = input("Enter crops image folder path: ")

with open(txt_file_path, 'r') as fl:
    data = fl.readlines()

data_copy = data.copy()

with open(txt_file_path, 'w') as w:
    with open(updated_file_path, 'a') as f:
        for line in data:
            img_name = line.strip().split('\t')[0]
            text = line.strip().split('\t')[1]
            img_path = os.path.join(base_image_folder, img_name)
            img = cv2.imread(img_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            scale = 3.0
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            plt.imshow(resized_image)
            plt.title(text)
            plt.show(block=False)
            # cv2.imshow('Image', resized_image)
            print('-------------------------------------------------------------------------------')
            print(text)
            
            new_txt = input("Enter correct text(just press enter to continue without changing): ")
            
            if not new_txt:
                # cv2.destroyAllWindows()
                f.write(line)
                plt.close()
                # time.sleep(1)
            elif new_txt=='remove':
                data_copy.pop(0)
                plt.close()
                continue
            elif new_txt=='quit':
                # time.sleep(5)
                break
            else:
                # cv2.destroyAllWindows()
                text = new_txt
                print(text)
                # f.write(img_name + '\t' + f"\"{text}\"" + '\n')
                try:
                    f.write(img_name + '\t' + f"{text}" + '\n')
                except Exception as e:
                    print(f"An error occurred while writing to the file: {e}")
                # time.sleep(1)
                plt.close()

            try:
                # print("--------------------------")
                # print("Before pop:", data_copy[0])  # Print the first element before popping
                data_copy.pop(0)
                # print("After pop:", data_copy[0])  # Print the first element after popping
                # print("--------------------------")
            except IndexError:
                print("Data list is empty.")
    w.writelines(data_copy)