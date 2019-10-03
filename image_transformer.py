from PIL import Image
import numpy as np
import os

class imageTransformer:
    """
    imageTransformer is class to manipulate the raw input images coming from the cameras.
    The function 'apply' can be used to apply random rotation, zoom to the image and fill the extra 
    background with random patterns. (Tested but not used currently)

    The function 'apply_homography' has a precalculated and hard coded transformation for each of the cameras,
    which will be applied on all of the images. It can also apply this transformation with noise for regulation purposes.

    Validation: The main function at the end of this file can be executed to see the before and
    after effects of the transformations.
    """
    def __init__(self, patterns_folder_path=None, read_patterns=False):
        self.patterns_folder_path = patterns_folder_path
        
        if read_patterns:
            self.patterns = [name for name in os.listdir(patterns_folder_path)]
            self.num_patterns = len(self.patterns)

        self.upleft_X_offset = [10.0/224, 10.0/224, 10.0/224]
        self.upright_X_offset= [0/224, 10.0/224, 0.0/224]
        self.bottomright_X_offset = [25.0/224, 25.0/224, 18.0/224]
        self.bottomleft_X_offset = [40.0/224, 30.0/224, 30.0/224]

    def apply_homography(self, im, camera_id):
        im = self.homography(im, camera_id)
        return im

    def apply(self, im, camera_id):
        im = self.homography(im, camera_id)

        rand_index = np.random.randint(self.num_patterns, size=1)[0]
        pattern = Image.open(os.path.join(self.patterns_folder_path, self.patterns[rand_index]))
        pattern = self.paste_in_middle(im, pattern)

        rand_deg = np.random.choice(range(-180,180,10), size=1)[0]
        pattern = self.rotate(pattern, rand_deg)

        zoom_offset = (pattern.width - im.width) / 2 - np.random.randint(-7, 7, size=1)[0]
        final= self.zoom_to_fit(pattern, zoom_offset)

        return final

    def homography(self, im, camera_id):
        width, height = im.size
        coeffs = self.find_coeffs(
                [(0, 0), (width, 0), (width, height), (0, height)],
                [(self.upleft_X_offset[camera_id] * width, 0), (width - self.upright_X_offset[camera_id] * width, 0), 
                (width - self.bottomright_X_offset[camera_id] * width, height), (self.bottomleft_X_offset[camera_id] * width, height)])

        # start = time.time()
        im_transformed = im.transform((width, height), Image.PERSPECTIVE, coeffs,
                Image.BICUBIC)
        return im_transformed
    
    def paste_in_middle(self, source, dest):
        position = ((dest.width - source.width)/2, (dest.height - source.height)/2)
        dest.paste(source, position)
        return dest
    
    def rotate(self, im, degree):
        im_rotated = im.rotate(int(degree), resample=Image.BICUBIC)
        return im_rotated
    
    def zoom_to_fit(self, im, zoom_offset):
        w, h = im.size
        coeffs_zoom = self.find_coeffs(
                [(0, 0), (w, 0), (w, h), (0, h)],
                [(0 + zoom_offset, 0 + zoom_offset), (w - zoom_offset, 0 + zoom_offset), (w - zoom_offset, h - zoom_offset), (0 + zoom_offset, h - zoom_offset)])

        im_zoomed = im.transform((w, h), Image.PERSPECTIVE, coeffs_zoom, Image.BICUBIC)
        return im_zoomed

    def find_coeffs(self, pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

if __name__ == '__main__':
    IT = imageTransformer()
    im = Image.open("/home/d3gan/development/datasets/record/sth_sth_224/test.jpg")
    im.show()
    result = IT.apply_homography(im, 1)
    result.show()