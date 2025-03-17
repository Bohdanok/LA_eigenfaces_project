import numpy as np
# import matplotlib.pyplot as plt
import cv2 as cv

NUM_OF_FACES = 40
NUM_OF_IMAGES_PER_HUMAN_ENTRY = 10

read_face = cv.imread("pics/BOhdan_test_photo.jpg")

def prepare_the_enviroment(faces_target_path:str, faces_path:str):
    faces = np.load(faces_path)
    faces_target = np.load(faces_target_path)

    # print(f"{faces[0] = }")
    # print(f"{faces_target[0] = }")

    return faces, faces_target

    # cv.imshow("Faces regular", faces[0])
    # cv.waitKey(0)
    # cv.destroyAllWindows()

def display_normed_image(win_name:str, im:np.array, L2:float):
    """Display normed image. It is [0, 1]"""
    im = im * L2
    # print(f"{im = }")
    cv.imshow(win_name, im)
    cv.waitKey(0)
    cv.destroyAllWindows()

def preprocess_image(image:np.array):
    """
    Make the image brightness invariant, which is sufficient
     in Eigenfaces(that is why we do not use Gaussian blur)
    
    In the dataset all images are 64x64, so no resizing here
    
    """
    L2 = np.linalg.norm(image)

    if L2 != 0:
        normalized_image = image / L2
    else:
        normalized_image = image
    
    return normalized_image
    # print(f"{L2 = }")

    # display_normed_image("Normed image", normalized_image, L2)
    # cv.imshow("Faces regular", image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()



def main():
    faces, faces_targets = prepare_the_enviroment("pics/archive(1)/olivetti_faces_target.npy", "pics/archive(1)/olivetti_faces.npy")

    faces_matrices_dict = {i:[] for i in range(0, NUM_OF_FACES)}

    for person_starting_index in range(0, NUM_OF_FACES):
        


        # There are 10 learning images per object (M)
        for image_per_person in range(0, NUM_OF_IMAGES_PER_HUMAN_ENTRY):

            L2 = np.linalg.norm(faces[person_starting_index + image_per_person]) # for debugging, delete afterwards

            image = preprocess_image(faces[person_starting_index + image_per_person])

            # compute mean feature vector

            mean_feature_vector = np.mean(image, axis=1)

            # normalize the distribution of points of N dimensional space

            mean_subtucted_image = np.array([np.array([image[j][i] - mean_feature_vector[j] for j in range(0, image.shape[1])]) for i in range(0, image.shape[0])])

            # display_normed_image("Normalized face", mean_subtucted_image, L2)

            faces_matrices_dict[person_starting_index].append(mean_subtucted_image)
        faces_matrices_dict[person_starting_index] = np.array(faces_matrices_dict[person_starting_index])

        # Find R - covariance matrix

        R = np.cov(faces_matrices_dict[person_starting_index], rowvar=False)

        # SVD R = U * S * U^T

        U, S, U_T = np.linalg.svd(R)

        




    print("Hi there")


if __name__ == "__main__":
    main()