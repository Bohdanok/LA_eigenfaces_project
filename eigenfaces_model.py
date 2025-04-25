import numpy as np

class EigenfacesModel:


    def __init__(self):
        self.faces = None
        self.face_labels = None
        self.best_eigenfaces = None
        self.mean_face_vector = None
        self.weight_list = None
        self.threshold = None



    def train(self, faces, face_labels, confidence_level = 0.95):

        self.faces = faces
        self.face_labels = face_labels

        faces_as_vectors = np.array([matrix.flatten() for matrix in faces])
        mean_face_vector = np.mean(faces_as_vectors, axis=0)

        faces_matrix_A = np.vstack(faces_as_vectors)
        normed_faces = faces_matrix_A - mean_face_vector

        self.mean_face_vector = mean_face_vector

        eigenvectors, eigenvalues = self.get_eigenvectors_and_eigenvalues(normed_faces)

        eigenfaces = eigenvectors

        # eigenfaces = self.convert_to_eigenfaces(eigenvectors, normed_faces)
        k = self.num_of_best_eigenvalues(eigenvalues, confidence_level)

        self.best_eigenfaces = eigenfaces[:k]

        self.weight_list = self.get_weight_list(normed_faces)


    def compute_threshold(self):
    
        max_length = float("-inf")
        person_maximum = float("-inf")

        faces_dict = {}
        for i, img in zip(self.face_labels, self.weight_list):
            faces_dict[i] = faces_dict.get(i, []) + [img]

        for _, person_projected_faces in faces_dict.items():
            person_maximum = float("-inf")

            length = len(person_projected_faces)
            
            for i in range(length):

                for j in range(i + 1, length):
                        
                    face_length = np.linalg.norm(np.array(person_projected_faces[i]) - np.array(person_projected_faces[j]))
                    person_maximum = max(person_maximum, face_length)

                max_length = max(max_length, person_maximum)

        return max_length * 0.8


    def test(self, faces_test, face_test_labels):
        correct_count = 0
        for label, face in zip(face_test_labels, faces_test):
            correct_count += label == self.predict(face)
        return correct_count / faces_test.shape[0]


    def predict(self, face_to_classify) -> int:
        mean_substructed_face = face_to_classify.flatten() - self.mean_face_vector
        weighted_prediction = []

        for eigenface in self.best_eigenfaces:
            # print(f'{i} : {np.dot(mean_substructed_face, np_eigenfaces[i])}')
            weighted_prediction.append(np.dot(mean_substructed_face, eigenface))

        length_list = []
        val_face_dist = float("inf")
        ind = -1

        for num, image in enumerate(self.weight_list):
            val = np.linalg.norm(np.array(weighted_prediction) - image)
            if (val_face_dist > val):
                val_face_dist = val
                ind = num
            # val_face_dist = min(val, val_face_dist)
            length_list.append(val)

        return self.face_labels[ind]


    def get_eigenvectors_and_eigenvalues(self, normed_faces):
        covariance_matrix = np.matmul(normed_faces.transpose(), normed_faces)

        V, U, *_ = np.linalg.svd(covariance_matrix)

        return V, U


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
