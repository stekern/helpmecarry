#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Erlend Ekern <dev@ekern.me>
#
# Distributed under terms of the MIT license.

"""
A ROS node that looks for square-shaped, bright colored bags and
publishes the approximate location of the bag handle on a topic
"""

import rospy
import message_filters
import image_geometry
import cv2
import numpy as np
from helpmecarry.msg import Bag3D
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped


# The bag's bounding box is cropped on every side according to this percentage
# and the distance to the bag depth is averaged from this cropped area
BOUNDING_BOX_OFFSET = 0.25
# The lower and upper boundary of the width/height-ratio of the bag
RECTANGLE_RATIO_BOUNDARIES = (0.70, 0.90)
# The height of the handle in terms of the height of the bag
BAG_HEIGHT_HANDLE_RATIO = 1/float(3)

# Camera topics
RGB_TOPIC = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
DEPTH_TOPIC = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'
CAM_INFO_TOPIC = '/hsrb/head_rgbd_sensor/rgb/camera_info'

# Lower and upper thresholds for color thresholding
COLOR_BOUNDARIES = {
        'BLUE': ([30, 50, 30], [220, 190, 100]),
        'RED': ([20, 20, 90], [60, 60, 255]),
        'YELLOW': ([0, 120, 120], [150, 255, 255]),
        'GREEN': ([10, 120, 80], [70, 240, 150]),
        }
# Set to True if you would like to do color thresholding on every color in
# the COLOR_BOUNDARIES dictionary
USE_MULTIPLE_MASKS = False
# The color to threshold on if USE_MULTIPLE_MASKS is False
DEFAULT_COLOR = 'YELLOW'


class BagDetector(object):

    def __init__(self):
        rospy.init_node('bag_detector', anonymous=True)
        self.bridge = CvBridge()

        # RGB and depth image must arrive within 1 second of each other
        rgb_sub = message_filters.Subscriber(RGB_TOPIC, Image)
        depth_sub = message_filters.Subscriber(DEPTH_TOPIC, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 1)
        self.ts.registerCallback(self.callback)

        # The camera model is used to convert image pixels to 3D rays
        cam_info = rospy.wait_for_message(CAM_INFO_TOPIC, CameraInfo, timeout=None)
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(cam_info)

        # Topic to publish bag location on
        self.bag_pub = rospy.Publisher('bag_location', Bag3D, queue_size=10)


    def callback(self, raw_rgb_image, raw_depth_image):
        # Convert rgb image to openCV image
        rgb_image = self.bridge.imgmsg_to_cv2(raw_rgb_image, 'bgr8')

        # Convert depth image to openCV image
        depth_image = self.bridge.imgmsg_to_cv2(raw_depth_image, 'passthrough')

        # Apply color mask
        color_filtered = self.apply_color_mask(rgb_image)
        cv2.imshow('Color filtered', color_filtered)
        
        # Apply saturation filter
        saturation_filtered = self.apply_saturation_filter(color_filtered)
        cv2.imshow('Saturation filtered', saturation_filtered)

        # Find the contours
        contours, _ = cv2.findContours(saturation_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Only keep the 10 largest contours
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

        # Draw detected objects
        image_draw = self.get_image_with_objects(contours, rgb_image, depth_image)

        # Display the image
        cv2.imshow('Detected bags', image_draw)
        cv2.waitKey(3)

    def get_cropped_bounding_box(
        self,
        bounding_box,
        width,
        height,
        offset=BOUNDING_BOX_OFFSET
    ):
        """
        Returns the coordinates of a cropped bounding box
        """
        xmin, ymin = bounding_box.min(axis=0)
        xmax, ymax = bounding_box.max(axis=0)

        xmin += int(width * offset)
        xmax -= int(width * offset)
        ymin += int(height * offset)
        ymax -= int(height * offset)

        return ymin, ymax, xmin, xmax

    def apply_color_mask(self, image):
        if USE_MULTIPLE_MASKS:
            mask = self.get_combined_color_mask(image)
        else:
            # Only use a yellow color mask by default
            lower, upper = COLOR_BOUNDARIES[DEFAULT_COLOR]
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")

            # Apply color mask
            mask = cv2.inRange(image, lower, upper)

	color_filtered = cv2.bitwise_and(image, image, mask = mask)

        # Apply median blur to reduce noise
        blurred = cv2.medianBlur(color_filtered, 11)

        return blurred

    def apply_saturation_filter(self, image):
        # Convert image to hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)

        # Do a threshold on the saturation
        threshold = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Smooth the image to remove noise
        blurred = cv2.medianBlur(threshold, 5)

        return blurred

    def publish_bag_information(self, center, point3d):
        bag = Bag3D()

        bag_handle = PointStamped()
        bag_handle.header.frame_id = self.camera_model.tfFrame()
        bag_handle.header.stamp = rospy.Time.now()
        bag_handle.point.x = point3d[0]
        bag_handle.point.y = point3d[1]
        bag_handle.point.z = point3d[2]

        bag.bag_handle = bag_handle
        bag.center_x, bag.center_y = center

        self.bag_pub.publish(bag)

    def get_image_with_objects(self, contours, rgb_image, depth_image):
        draw_image = rgb_image.copy()
        for c in contours:
            # Compute the center of the contour
            M = cv2.moments(c)
            if M['m00'] == 0.0: continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Skip contours that do not have four corners
            if self.get_corners(c) != 4: continue

            # Get the rotated rectangle surrounding the contour
            rect = cv2.minAreaRect(c)
            center = rect[0]
            width, height = rect[1]
            angle = rect[2]

            # Correct the width, height and angle
            # See https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
            if width > height:
                angle += 90
                height, width = rect[1]

            ratio = float(width)/height

            # Skip contours where width/height-ratio is off
            if not (RECTANGLE_RATIO_BOUNDARIES[0] < ratio < RECTANGLE_RATIO_BOUNDARIES[1]): continue

            bounding_box = cv2.cv.BoxPoints(rect)
            bounding_box = np.int0(bounding_box)
            bounding_box_corners = tuple(map(tuple, bounding_box))
            ymin, ymax, xmin, xmax = self.get_cropped_bounding_box(bounding_box, width, height)

            # Draw the corners of the rotated bounding box
            for corner in bounding_box_corners:
                cv2.circle(draw_image, corner, 3, (255, 0, 0), -1)

            # Crop the bounding box and estimate the depth
            cropped_bounding_box = depth_image[ymin:ymax, xmin:xmax]
            if cropped_bounding_box.shape[0]:
                depth = np.nanmean(cropped_bounding_box)
            else: continue

            # Draw the cropped bounding box
            cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            # Draw the center of the contour and the depth of the object
            cv2.circle(draw_image, (cX, cY), 4, (255, 255, 255), -1)
            cv2.putText(draw_image, str(depth), (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Sort the corners by the y-values
            bounding_box_corners = sorted(bounding_box_corners, key=lambda x: x[1])
            # Take the average of the two smallest y-values
            min_avg_y = sum(x[1] for x in bounding_box_corners[:2])/2
            # Use the ratio between the bag height and the handle position
            # to calculate y-value of bag handle
            bag_handle_y = int(min_avg_y - height * BAG_HEIGHT_HANDLE_RATIO)
            bag_handle = (cX, bag_handle_y)

            point3d = self.convert_2d_point_to_3d_point(bag_handle, depth)
            cv2.circle(draw_image, bag_handle, 4, (255, 255, 255), -1)

            # A detection has been made. Publish the necessary information
            self.publish_bag_information((cX, cY), point3d)

        return draw_image

    def convert_2d_point_to_3d_point(self, pixel, depth):
        # Convert a pixel with depth to a 3d point
        ray = np.array(self.camera_model.projectPixelTo3dRay(pixel))
        point = ray * depth
        return point

    def get_combined_color_mask(self, image):
        masks = []
        for key, value in COLOR_BOUNDARIES.iteritems():
            lower = np.array(value[0], dtype='uint8')
            upper = np.array(value[1], dtype='uint8')
            mask = cv2.inRange(image, lower, upper)
            masks.append(mask)

        return self._combine_masks(masks)

    def _combine_masks(self, masks):
        if len(masks) == 2:
            return cv2.bitwise_or(masks[0], masks[1])
        return cv2.bitwise_or(masks[0], self._combine_masks(masks[1::]))

    def get_corners(self, c):
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
        # Return number of detected corners
        return len(approx)


if __name__ == '__main__':
    BagDetector()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
