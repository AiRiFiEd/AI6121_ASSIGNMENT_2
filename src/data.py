import os
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import cv2
import uuid

class Image(object):
    _ALLOWED_EXTENSIONS = ['JPG', 'JPEG', 'PNG']
    def __init__(self) -> None:
        self.name = ''
        self.original_image = None
        self.keypoints = tuple()
        self.descriptors = np.empty((0,0))
        self.annotated_image = np.empty((0,0))
        self.uid = str(uuid.uuid4())

    def from_file(self, filepath: str) -> None:
        self.original_image = cv2.imread(filepath)

    def sift(self) -> None:
        sifter = cv2.xfeatures2d.SIFT_create()
        self.keypoints, self.descriptors = sifter.detectAndCompute(self.original_image, None)
        self.annotated_image = np.copy(self.original_image)
        cv2.drawKeypoints(
            self.gray, self.keypoints, self.annotated_image
        )
    
    def _show(self, type_: str = 'original') -> None:
        pass

    def save(self, filepath: str, type_: str = 'original') -> None:
        if type_ == 'original':
            cv2.imwrite(filepath, self.original_image)     
        elif type_ == 'annotated':
            cv2.imwrite(filepath, self.annotated_image)   

    @property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

class Matcher(object):
    #https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    #https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    #https://kushalvyas.github.io/stitching.html
    def __init__(self, method: str) -> None:
        self.method = method
    
    def match(self, descriptor_1: np.ndarray, descriptor_2: np.ndarray) -> Tuple:
        pass

class MatcherFlannBased(Matcher):
    def __init__(self, index_params: Dict, search_params: Dict, k: int = 2) -> None:
        super().__init__('flann')
        self.index_params = index_params
        self.search_params = search_params
        self.k = k
        self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        self.matches = tuple()

    def match(self, descriptor_1: np.ndarray, descriptor_2: np.ndarray) -> Tuple:
        self.matches = self.matcher.knnMatch(descriptor_1, descriptor_2, self.k)
        return self.matches

class MatcherBruteForce(Matcher):
    def __init__(self, norm_type: str = 'norm_l2', cross_check: bool = False, k: int = 2) -> None:
        self.norm_type = norm_type
        self.cross_check = cross_check
        self.k = k                     

        self.matcher = cv2.BFMatcher(self._enum, self.cross_check)
        self.matches = tuple() 

    @property
    def _enum(self) -> int:
        if self.norm_type == 'norm_l2':
            return cv2.NORM_L2
        elif self.norm_type =='norm_inf':
            return cv2.NORM_INF
        elif self.norm_type =='norm_l1':
            return cv2.NORM_L1  
        elif self.norm_type =='norm_l2sqr':
            return cv2.NORM_L2SQR    
        elif self.norm_type =='norm_hamming':
            return cv2.NORM_HAMMING
        elif self.norm_type =='norm_hamming2':
            return cv2.NORM_HAMMING2                         

    def match(self, descriptor_1: np.ndarray, descriptor_2: np.ndarray) -> Tuple:
        self.matches = self.matcher.knnMatch(descriptor_1, descriptor_2, self.k)
        return self.matches               

class Homographer(object):
    def __init__(self, method: str, minimum_match_count: int) -> None:
        self.method = method
        self.minimum_match_count = minimum_match_count

    def has_minimum_match(self, good_matches: List) -> bool:
        return len(good_matches) > self.minimum_match_count

    def find_homography(self, source_points: np.ndarray,
        destination_points: np.ndarray) -> Tuple:
        pass

class HomographerRansac(Homographer):
    def __init__(self, minimum_match_count: int = 10, 
            reprojection_threshold: int=3) -> None:
        super().__init__('ransac', minimum_match_count)
        self.reprojection_threshold = reprojection_threshold
        self.matrix = np.empty((0,0))
        self.mask = np.empty((0,0))
    
    def find_homography(self, source_points: np.ndarray, 
        destination_points: np.ndarray) -> Tuple:
        self.matrix, self.mask = cv2.findHomography(source_points, 
                                    destination_points, cv2.RANSAC,
                                    self.reprojection_threshold)
        return self.matrix, self.mask

class HomographerLMeDS(Homographer):
    def __init__(self, minimum_match_count: int = 10) -> None:
        super().__init__('lmeds', minimum_match_count)
        self.matrix = np.empty((0,0))
        self.mask = np.empty((0,0))

    def find_homography(self, source_points: np.ndarray, 
        destination_points: np.ndarray) -> Tuple:
        self.matrix, self.mask = cv2.findHomography(source_points, 
                                    destination_points, cv2.LMEDS)
        return self.matrix, self.mask        

# class Detector(object):
#     def __init__(self, method: str = 'sift') -> None:
#         self.method = method.strip().lower()
#         self.detector = self._get_detector(self.method)
    
#     def _get_detector(self, method: str) -> Any:
#         if method == 'sift':
#             return cv2.xfeatures2d.SIFT_create()

#     def detect(self, image: np.ndarray):
#         return self.detector.detectAndCompute(image, None)

class ImagePairs(object):
    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.pairs = defaultdict(list)
        self.good_matches = defaultdict(list)
        self.annotated_match = defaultdict(lambda: np.empty((0,0)))
        self.homograph = defaultdict(list) # matrix, mask, transformed_perspective
        self.stitched = defaultdict(lambda: np.empty((0,0)))
        self.uid = uuid.uuid4()
    
    def load_images(self) -> bool:
        pairs = defaultdict(list)
        for name in os.listdir(self.directory):
            n = name.split('.')[0].split('_')[1]
            filepath = os.path.join(
                self.directory, name
            )
            img = Image()
            img.from_file(filepath)
            self.pairs[n].append(img)

    # def save(self, directory: str, type_: str = 'annotated_match') -> None:
    #     if type_ == 'annotated_match':
    #         for pair_id, annotation in self.annotated_match.items():

    #         cv2.imwrite(filepath, self.original_image)     


    def detect_features_and_annotate(self, pair_id: str='') -> None:
        if pair_id:
            images = self.pairs[pair_id]
            for img in images:
                img.sift()
        else:
            for pair_id, images in self.pairs.items():
                for img in images:
                    img.sift()

    def show_annotations(self, pair_id: str='',
            output_directory: str = '') -> None:
        if pair_id:
            images = self.pair[pair_id]
            fig, axes = plt.subplot(1,2)
            for i in range(len(images)):
                axes[i].imshow(images[i].annotated_image)
            if output_directory:
                filepath = os.path.join(
                    output_directory, 'annotated_image_{pair_id}_{uid}.png'.format(
                        pair_id=pair_id, uid=self.uid
                    )
                )
                fig.savefig(filepath)
        else:
            for pair_id, images in self.pairs.items():
                fig, axes = plt.subplots(1, 2)
                for i in range(len(images)):
                    axes[i].imshow(images[i].annotated_image)
                if output_directory:
                    filepath = os.path.join(
                        output_directory, 'annotated_image_{pair_id}_{uid}.png'.format(
                            pair_id=pair_id, uid=self.uid
                        )
                    )
                    fig.savefig(filepath)

    def match_descriptors_and_annotate(self, matcher: Matcher, threshold: float = 0.7, pair_id: str = ''):
        if pair_id:
            images = self.pair[pair_id]
            descriptor_1 = images[0].descriptors
            descriptor_2 = images[1].descriptors
            matches = matcher.match(descriptor_1, descriptor_2)
            tmp = []
            for m,n in matches:
                if m.distance < threshold * n.distance:
                    tmp.append(m)
            self.good_matches[pair_id] = tmp
            annotation = np.empty((max(images[0].shape[0], images[1].shape[0]), images[0].shape[1]+images[1].shape[1], 3), dtype=np.uint8)
            cv2.drawMatches(images[0], images[0].keypoints, 
                images[1], images[1].keypoints, 
                self.good_matches[pair_id], 
                annotation, 
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            ) 
            self.annotated_match[pair_id] = annotation

        else:
            for pair_id, images in self.pairs.items():
                # currently only handles stitching of 2 images
                descriptor_1 = images[0].descriptors
                descriptor_2 = images[1].descriptors
                matches = matcher.match(descriptor_1, descriptor_2)
                tmp = []
                for m,n in matches:
                    if m.distance < threshold * n.distance:
                        tmp.append(m)
                self.good_matches[pair_id] = tmp
                annotation = np.empty((max(images[0].original_image.shape[0], images[1].original_image.shape[0]), 
                    images[0].original_image.shape[1]+images[1].original_image.shape[1], 3), dtype=np.uint8)
                cv2.drawMatches(images[0].original_image, images[0].keypoints, 
                    images[1].original_image, images[1].keypoints, 
                    self.good_matches[pair_id], 
                    annotation, 
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                ) 
                self.annotated_match[pair_id] = annotation

    def show_matching_annotations(self, pair_id: str = '',
            output_directory:str='') -> None:
        if pair_id:
            fig, axes = plt.subplots(1, 1, figsize=(16,8))
            axes.imshow(cv2.cvtColor(self.annotated_match[pair_id], cv2.COLOR_BGR2RGB))
            if output_directory:
                filepath = os.path.join(
                    output_directory, 'annotated_match_{pair_id}_{uid}.png'.format(
                        pair_id=pair_id, uid=self.uid
                    )
                )
                fig.savefig(filepath)

        else:            
            for pair_id, good_match in self.annotated_match.items():
                fig, axes = plt.subplots(1, 1, figsize=(16,8))
                axes.imshow(cv2.cvtColor(good_match, cv2.COLOR_BGR2RGB))
                if output_directory:
                    filepath = os.path.join(
                        output_directory, 'annotated_match_{pair_id}_{uid}.png'.format(
                            pair_id=pair_id, uid=self.uid
                        )
                    )
                    fig.savefig(filepath)                
                



    def find_homography(self, homographer: Homographer,
        pair_id: str = '') -> np.ndarray:
        if pair_id:
            good_matches = self.good_matches[pair_id]
            if homographer.has_minimum_match(good_matches):
                source_points = np.float32(
                    [self.pairs[pair_id][0].keypoints[m.queryIdx].pt 
                        for m in good_matches]
                ).reshape(-1,1,2)
                destination_points = np.float32(
                    [self.pairs[pair_id][1].keypoints[m.trainIdx].pt
                        for m in good_matches]
                ).reshape(-1,1,2)
                matrix, mask = homographer.find_homography(
                    source_points, destination_points
                )
                matches_mask = mask.ravel().tolist()
                h, w = self.pairs[pair_id][0].gray.shape
                points = np.float32(
                    [ [0,0], [0, h-1], [w-1, h-1], [w-1,0]]
                ).reshape(-1,1,2)
                transformed_perspective =  cv2.perspectiveTransform(points, matrix)
                self.homograph[pair_id] = [matrix, matches_mask, transformed_perspective]
        else:
            for pair_id, good_matches in self.good_matches.items():
                if homographer.has_minimum_match(good_matches):
                    source_points = np.float32(
                        [self.pairs[pair_id][0].keypoints[m.queryIdx].pt 
                            for m in good_matches]
                    ).reshape(-1,1,2)
                    
                    destination_points = np.float32(
                        [self.pairs[pair_id][1].keypoints[m.trainIdx].pt
                            for m in good_matches]
                    ).reshape(-1,1,2)
                    matrix, mask = homographer.find_homography(
                        source_points, destination_points
                    )
                    matches_mask = mask.ravel().tolist()
                    h, w = self.pairs[pair_id][0].gray.shape
                    points = np.float32(
                        [ [0,0], [0, h-1], [w-1, h-1], [w-1,0]]
                    ).reshape(-1,1,2)
                    transformed_perspective =  cv2.perspectiveTransform(points, matrix)
                    self.homograph[pair_id] = [matrix, matches_mask, transformed_perspective]
                else:
                    raise ValueError('number of good matches for image pair {pair_id} is below threshold'.format(pair_id=pair_id))
                
       
    def show_homography(self, pair_id: str = '',
        output_directory: str = '') -> None:
        if pair_id:
            fig, axes = plt.subplots(1, 1, figsize=(16,8))
            homography_transformation = self.homograph[pair_id]
            image_2 = self.pairs[pair_id][1].original_image
            image_2 = cv2.polylines(
                image_2, [np.int32(homography_transformation[2])], True, 255, 3, cv2.LINE_AA
            )
            params = {
                'matchColor' : (0, 255, 0),
                'singlePointColor' : None,
                'matchesMask' : homography_transformation[1],
                'flags' : 2
            } 
            image_3 = cv2.drawMatches(
                self.pairs[pair_id][0].original_image,
                self.pairs[pair_id][0].keypoints,
                self.pairs[pair_id][1].original_image,
                self.pairs[pair_id][1].keypoints,   
                self.good_matches[pair_id],
                None,
                **params                                
            )
            axes.imshow(cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB))
            if output_directory:
                filepath = os.path.join(
                    output_directory, 'homography_{pair_id}_{uid}.png'.format(
                        pair_id=pair_id, uid=self.uid
                    )
                )
                fig.savefig(filepath)


        else:            
            for pair_id, homography_transformation in self.homograph.items():
                fig, axes = plt.subplots(1, 1, figsize=(16,8))
                image_2 = self.pairs[pair_id][1].original_image
                image_2 = cv2.polylines(
                    image_2, [np.int32(homography_transformation[2])], True, 255, 3, cv2.LINE_AA
                )
                params = {
                    'matchColor' : (0, 255, 0),
                    'singlePointColor' : None,
                    'matchesMask' : homography_transformation[1],
                    'flags' : 2
                } 
                image_3 = cv2.drawMatches(
                    self.pairs[pair_id][0].original_image,
                    self.pairs[pair_id][0].keypoints,
                    self.pairs[pair_id][1].original_image,
                    self.pairs[pair_id][1].keypoints,   
                    self.good_matches[pair_id],
                    None,
                    **params                                
                )
                axes.imshow(cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB))
                if output_directory:
                    filepath = os.path.join(
                        output_directory, 'homography_{pair_id}_{uid}.png'.format(
                            pair_id=pair_id, uid=self.uid
                        )
                    )
                    fig.savefig(filepath)

    def stitch(self, alpha: float = 0.25, pair_id: str = '') -> None:
        if pair_id:
            homography_transformation = self.homograph[pair_id]
            result = cv2.warpPerspective(
                self.pairs[pair_id][0].original_image,
                homography_transformation[0],
                (self.pairs[pair_id][1].original_image.shape[1], 
                    self.pairs[pair_id][1].original_image.shape[0])
            )
            stitched_image = cv2.addWeighted(result, alpha, self.pairs[pair_id][1].original_image, 1-alpha, 0)
            self.stitched[pair_id] = stitched_image
        else:
            for pair_id, homography_transformation in self.homograph.items():
                result = cv2.warpPerspective(
                    self.pairs[pair_id][0].original_image,
                    homography_transformation[0],
                    (self.pairs[pair_id][1].original_image.shape[1], 
                        self.pairs[pair_id][1].original_image.shape[0])
                )
                stitched_image = cv2.addWeighted(result, alpha, self.pairs[pair_id][1].original_image, 1-alpha, 0)
                self.stitched[pair_id] = stitched_image

    def show_stitched(self, pair_id: str = '',
            output_directory: str = '') -> None:
        if pair_id:
            fig, axes = plt.subplots(len(self.stitched), 1, figsize=(16,8)) 
            axes.imshow(cv2.cvtColor(self.stitched[pair_id], cv2.COLOR_BGR2RGB))    
            if output_directory:
                filepath = os.path.join(
                    output_directory, 'stitched_{pair_id}_{uid}.png'.format(
                        pair_id=pair_id, uid=self.uid
                    )
                )
                fig.savefig(filepath)               
        else:
            for pair_id, stitched_image in self.stitched.items():
                fig, axes = plt.subplots(1, 1, figsize=(16,8))            
                axes.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
                if output_directory:
                    filepath = os.path.join(
                        output_directory, 'stitched_{pair_id}_{uid}.png'.format(
                            pair_id=pair_id, uid=self.uid
                        )
                    )
                    fig.savefig(filepath)  
