import numpy as np
from eigenfaces_model import EigenfacesModel

def split_data(faces, faces_target, test_percentage):
    faces_dict = {}
    for i, img in zip(faces_target, faces):
        faces_dict[i] = faces_dict.get(i, []) + [img]

    faces_train = []
    faces_train_target = []
    faces_test = []
    faces_test_target = []

    for person_id, person_faces in faces_dict.items():
        TEST_FACES_COUNT = int(len(person_faces) * test_percentage)
        TRAIN_FACES_COUNT = len(person_faces) - TEST_FACES_COUNT
        faces_test.extend(person_faces[:TEST_FACES_COUNT])
        faces_test_target.extend([person_id] * TEST_FACES_COUNT)
        faces_train.extend(person_faces[TEST_FACES_COUNT:])
        faces_train_target.extend([person_id] * TRAIN_FACES_COUNT)

    return np.array(faces_train), np.array(faces_train_target), np.array(faces_test), np.array(faces_test_target)


FACES_PATH, FACES_TARGET_PATH = "pics/archive(1)/olivetti_faces.npy", "pics/archive(1)/olivetti_faces_target.npy"

faces = np.load(FACES_PATH)
faces_target = np.load(FACES_TARGET_PATH)


faces_train, faces_train_target, faces_test, faces_test_target = split_data(faces, faces_target, 0.2)

model = EigenfacesModel()

model.train(faces_train, faces_train_target, 0.95)
acc = model.test(faces_test, faces_test_target)

print("accuracy:", acc)
