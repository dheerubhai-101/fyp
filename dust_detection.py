import cv2 as cv
from skimage.feature import graycomatrix, graycoprops
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = graycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature


import GLCM.fast_glcm as fast_glcm
from skimage import data

if __name__ == '__main__':
    img = cv.imread('D:\FYP\Images\Imgdirty_1045_1.jpg')
    window_name = 'image'
    
    # Using cv2.imshow() method
    # Displaying the image
    cv.imshow(window_name, img)
    
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv.waitKey(0)
    
    # closing all open windows
    cv.destroyAllWindows()

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(hsv[:,:,0].shape)
    hue = hsv[:,:,0]
    # cv.imshow(window_name, hue)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imshow(window_name, hsv[:,:,1])
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imshow(window_name, hsv[:,:,2])
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    glcm_contrast_h = fast_glcm.fast_glcm_contrast(hue, angle=0)
    glcm_contrast_v = fast_glcm.fast_glcm_contrast(hue, angle=90)
    glcm_homogeneity = fast_glcm.fast_glcm_homogeneity(hue)
    glcm_entropy = fast_glcm.fast_glcm_entropy(hue)
    glcm_asm = fast_glcm.fast_glcm_ASM(hue)
    glcm_dissimilarity = fast_glcm.fast_glcm_dissimilarity(hue)

    print(glcm_homogeneity.shape)
    
    flattened_contrast = glcm_contrast_h.flatten()
    print(flattened_contrast.shape)
    

    # plt.imshow(glcm_contrast_h, cmap="gray")
    # plt.tight_layout()
    # plt.show()

    # plt.imshow(glcm_homogeneity, cmap="gray")
    # plt.tight_layout()
    # plt.show()

    # plt.imshow(glcm_entropy, cmap="gray")
    # plt.tight_layout()
    # plt.show()

    # # plt.imshow(glcm_asm, cmap="gray")
    # # plt.tight_layout()
    # # plt.show()

    # plt.imshow(glcm_dissimilarity, cmap="gray")
    # plt.tight_layout()
    # plt.show()

    # plt.plot(glcm_contrast_h)
    # plt.show()


    



# ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
# properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

# glcm_all_agls = []
# # for img, label in zip(imgs, labels): 
# #     glcm_all_agls.append(
# #             calc_glcm_all_agls(img, 
# #                                 label, 
# #                                 props=properties)
# #                             )

# glcm_all_agls.append(
#             calc_glcm_all_agls(gray, 
#                                 1, 
#                                 props=properties)
#                             )
# # features = calc_glcm_all_agls(gray, 1, props=properties)


# columns = []
# angles = ['0', '45', '90','135']
# for name in properties :
#     for ang in angles:
#         columns.append(name + "_" + ang)
        
# columns.append("label")


# # Create the pandas DataFrame for GLCM features data
# glcm_df = pd.DataFrame(glcm_all_agls, 
#                       columns = columns)
# print(glcm_df.head(1))