'''face recognition model'''
import numpy as np
from svd_lab import svd

class EigenfacesModel:
    def __init__(self):
        self.faces = None
        self.face_labels = None
        self.best_eigenfaces = None
        self.mean_face_vector = None
        self.weight_list = None
        self.threshold = None

    def compute_threshold(self):
        max_length = float("-inf")
        person_maximum = float("-inf")

        faces_dict = {}
        for i, img in zip(self.face_labels, self.weight_list):
            faces_dict[i] = faces_dict.get(i, []) + [img]

        face_lengthes = []

        for _, person_projected_faces in faces_dict.items():
            person_maximum = float("inf")

            length = len(person_projected_faces)

            for i in range(length):

                for j in range(i + 1, length):
                    projections_difference = np.array(person_projected_faces[i]) - np.array(person_projected_faces[j])
                    face_length = np.linalg.norm(projections_difference)
                    person_maximum = min(person_maximum, face_length)

                face_lengthes.append(person_maximum)
                max_length = max(max_length, person_maximum)

        face_lengthes.sort()
        precentile_index = max(int(0.95 * len(face_lengthes) - 1), 0)
        # print("lengthes", face_lengthes[precentile_index:])

        return face_lengthes[precentile_index]



    def train(self, faces, face_labels, confidence_level = 0.95):
        self.faces = faces
        self.face_labels = face_labels

        faces_as_vectors = np.array([matrix.flatten() for matrix in faces])
        mean_face_vector = np.mean(faces_as_vectors, axis=0)

        faces_matrix_a = np.vstack(faces_as_vectors)
        normed_faces = faces_matrix_a - mean_face_vector

        self.mean_face_vector = mean_face_vector

        eigenvectors, eigenvalues = self.get_eigenvectors_and_eigenvalues(normed_faces)

        eigenfaces = self.convert_to_eigenfaces(eigenvectors, normed_faces)
        k = self.num_of_best_eigenvalues(eigenvalues, confidence_level)

        self.best_eigenfaces = eigenfaces[:k]

        self.weight_list = self.get_weight_list(normed_faces)
        self.threshold = self.compute_threshold()
        # self.threshold = 150


    def test(self, faces_test, face_test_labels):
        testing_results = {
            "False": {"positive": 0, "negative": 0},
            "True": {"positive": 0, "negative": 0}
        }
        for label, face in zip(face_test_labels, faces_test):
            lprediciton = self.predict(face)
            true_prediction = label == lprediciton
            # print(label, lprediciton)
            face_in_dataset = lprediciton != -1
            if face_in_dataset:
                testing_results[str(true_prediction)]["positive"] += 1
            else:
                testing_results[str(true_prediction)]["negative"] += 1
        # print(testing_results)
        total_correct = testing_results["True"]["positive"] + testing_results["True"]["negative"]
        total_incorrect = testing_results["False"]["positive"] +testing_results["False"]["negative"]
        accuracy = total_correct / (total_correct + total_incorrect)
        precision = testing_results["True"]["positive"] / (testing_results["True"]["positive"] + testing_results["False"]["positive"])
        recall = testing_results["True"]["positive"] / (testing_results["True"]["positive"] + testing_results["False"]["negative"])
        f1_score = 2 * precision * recall / (precision + recall)
        metrics = accuracy, precision, recall, f1_score
        return [round(metric, 2) for metric in metrics]


    def predict(self, face_to_classify) -> int:
        mean_substructed_face = face_to_classify.flatten() - self.mean_face_vector
        weighted_prediction = []

        for eigenface in self.best_eigenfaces:
            weighted_prediction.append(np.dot(mean_substructed_face, eigenface))

        val_face_dist = float("inf")
        ind = -1

        for num, image in enumerate(self.weight_list):
            val = np.linalg.norm(np.array(weighted_prediction) - image)
            if (val_face_dist > val):
                val_face_dist = val
                ind = num

        # print(val_face_dist, self.threshold)
        return self.face_labels[ind] if val_face_dist < self.threshold else -1


    def get_eigenvectors_and_eigenvalues(self, normed_faces):
        covariance_matrix = np.matmul(normed_faces, normed_faces.transpose())

        U, eigenvalues, *_ = svd(covariance_matrix, 50)

        return U, eigenvalues


    def convert_to_eigenfaces(self, eigenvectors, normed_faces):

        eigenfaces = []
        NUMBER_OF_FACES = normed_faces.shape[0]

        for idx_l in range(NUMBER_OF_FACES):
            sum_ = 0

            for idx_k in range(NUMBER_OF_FACES):
                sum_ += eigenvectors[:, idx_l][idx_k] * normed_faces[idx_k, :]

            eigenfaces.append(sum_)

        return np.array(eigenfaces)


    def num_of_best_eigenvalues(self, eigenvalues, confidence_level):

        sum_of_eigenvalues = sum(eigenvalues)

        sum_k_eigenvalues, k = 0, 0

        for num, i in enumerate(eigenvalues):
            sum_k_eigenvalues += i

            if (sum_k_eigenvalues / sum_of_eigenvalues > confidence_level):
                k = num
                break
        else:
            k = len(eigenvalues)

        return k
    

    def get_weight_list(self, normed_faces):

        NUMBER_OF_FACES = normed_faces.shape[0]

        weight_list = []
        for i in range(NUMBER_OF_FACES):
            image_list = []
            for eigenface in self.best_eigenfaces:
                image_list.append(np.dot(eigenface, normed_faces[i]))
            weight_list.append(np.array(image_list))

        return weight_list
