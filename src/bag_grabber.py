#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Erlend Ekern <dev@ekern.me>
#
# Distributed under terms of the MIT license.

"""
A ROS node that subscribes to the position of bag handles and
tries to move towards the bag and pick it up
"""

import tf
import rospy
import math
import hsrb_interface as hsrb
import numpy as np
from helpmecarry.msg import Bag3D
from geometry_msgs.msg import Twist

# Name of the joints
ARM_LIFT_JOINT = 'arm_lift_joint'
ARM_FLEX_JOINT = 'arm_flex_joint'
ARM_ROLL_JOINT = 'arm_roll_joint'
WRIST_FLEX_JOINT = 'wrist_flex_joint'
WRIST_ROLL_JOINT = 'wrist_roll_joint'
HEAD_PAN_JOINT = 'head_pan_joint'
HEAD_TILT_JOINT = 'head_tilt_joint'

# Topics for the joints
BAG_LOCATION_TOPIC = '/bag_location'
VELOCITY_TOPIC = '/hsrb/command_velocity'

# Constants used for centering the bag in the middle of the camera
IMAGE_WIDTH = 640
CENTERING_THRESHOLD = 30
LINEAR_VELOCITY = 0.04

# Constant for activating different phases of the task
DEPTH_THRESHOLDS = (0.8, 1.2)

# How many samples to do of bag position before initiating arm movement
NUM_SAMPLES = 5

# This constant is multiplied by the current depth to control the number of
# seconds the robot can move towards the bag at a time
TIMEOUT_MULTIPLIER = 2.0

# The highest the ARM_LIFT_JOINT can be raised
MAX_ARM_LIFT = 0.69

# Neutral position
NEUTRAL_ARM_FLEX_ANGLE = math.radians(-90)
NEUTRAL_WRIST_ROLL_ANGLE = math.radians(90)
NEUTRAL_WRIST_FLEX_ANGLE = math.radians(-75) # Slightly inclined wrist

# Grab position
GRAB_WRIST_FLEX_ANGLE = math.radians(12) # Wrist inclination after claw is inside bag handle
GRAB_ARM_LIFT_INCREASE = 0.1 # Meters to lift hand after claw is inside bag handle
GRAB_FORWARD_VELOCITY = 0.2 # Speed used to move horizontally for placing bag in center
GRAB_BACKWARDS_VELOCITY = -0.1 # Speed used to back up if robot is too close to bag

# Angles for the gripper joint
GRIPPER_CLOSED = 0.0
GRIPPER_OPENED = 1.2

# Misc. delays
SLEEP_DELAY = 2.0 # Number of seconds to wait between control commands that might interfere 
TF_DELAY = 4.0 # Number of seconds to wait for transformation between two frames

# Frames 
HAND_FRAME = 'hand_palm_link'
MAP_FRAME = 'map'


class BagGrabber(object):
    def __init__(self):
        # Set up the robot
        self.robot = hsrb.Robot()
        self.whole_body = self.robot.get('whole_body')
        self.gripper = self.robot.get('gripper', self.robot.Items.END_EFFECTOR)
        self.omni_base = self.robot.get('omni_base')

        self.neutral_position = {
            ARM_LIFT_JOINT: 0.0,
            ARM_FLEX_JOINT: 0.0,
            ARM_ROLL_JOINT: 0.0,
            WRIST_FLEX_JOINT: NEUTRAL_WRIST_FLEX_ANGLE,
            WRIST_ROLL_JOINT: NEUTRAL_WRIST_ROLL_ANGLE,
            HEAD_PAN_JOINT: 0.0,
            HEAD_TILT_JOINT: 0.0,
        }

        #self.neutral_position = {
        #    HEAD_PAN_JOINT: 0.0,
        #    HEAD_TILT_JOINT: 0.0,
        #}

        rospy.loginfo('Moving robot to neutral position')
        self.whole_body.move_to_joint_positions(self.neutral_position)
        self.gripper.command(GRIPPER_OPENED)

        self.samples = []

        # Set flag for first part of movement
        self.in_position = False

        # Set flag for second and final part of movement
        self.finished = False

        # Setup TF listener
        self.listener = tf.TransformListener()

        # Subscribe to bag location topic
        self.bag_location_sub = rospy.Subscriber(BAG_LOCATION_TOPIC, Bag3D, self.callback, queue_size=1)

        # Velocity publisher is used for simple sideways movement
        self.velocity_pub = rospy.Publisher(VELOCITY_TOPIC, Twist, queue_size=10)


    def is_target_in_center(self, point2d):
        center = IMAGE_WIDTH/2
        x, y = point2d
        if (x > center + CENTERING_THRESHOLD or x < center - CENTERING_THRESHOLD):
            return False
        return True

    def center_target(self, point2d, depth):
        center = IMAGE_WIDTH/2
        x, y = point2d
        twist = Twist()
        # Velocity is multiplied by depth to move slower when close to bag
        if x > center + CENTERING_THRESHOLD:
            twist.linear.y = -LINEAR_VELOCITY * depth
        elif x < center - CENTERING_THRESHOLD:
            twist.linear.y = LINEAR_VELOCITY * depth
        self.velocity_pub.publish(twist)

    def transform_point(self, source, target, point):
        self.listener.waitForTransform(source, target, rospy.Time.now(), rospy.Duration(TF_DELAY))
        while not rospy.is_shutdown():
            try:
                self.listener.waitForTransform(source, target, rospy.Time.now(), rospy.Duration(TF_DELAY))
                trans = self.listener.transformPoint(target, point)
                return trans
            except:
                continue

    def get_updated_arm_lift(self, change):
        # whole_body.joint_state.position is an array containing the current
        # state of all of the joints. Index 1 belongs to the ARM_LIFT_JOINT
        curr_arm_lift = self.whole_body.joint_state.position[1]
        new_arm_lift = curr_arm_lift + change
        if new_arm_lift > MAX_ARM_LIFT:
            new_arm_lift = MAX_ARM_LIFT
        elif new_arm_lift < 0:
            new_arm_lift = 0
        return new_arm_lift

    def callback(self, bag):
        bag_handle = bag.bag_handle
        center = (bag.center_x, bag.center_y)
        cam_frame = bag_handle.header.frame_id
        depth = bag_handle.point.z

        if self.finished:
            self.robot.close()
        elif not self.is_target_in_center(center) and not self.finished:
            rospy.loginfo('Centering bag along y-axis')
            self.center_target(center, depth)
        elif not self.finished:
            if depth > DEPTH_THRESHOLDS[1]:
                trans = self.transform_point(cam_frame, MAP_FRAME, bag_handle) 
                x = trans.point.x
                y = trans.point.y

                # Get the current yaw of the robot
                yaw = self.omni_base.pose[2]

                # Set the timeout for the movement
                timeout = TIMEOUT_MULTIPLIER * depth

                rospy.loginfo('Moving the robot towards x: %.4f, y: %.4f for %.4f seconds' % (x, y, timeout))

                # Move straight towards the object
                try:
                    self.omni_base.go(x, y, yaw, timeout=timeout)
                except hsrb.exceptions.MobileBaseError:
                    rospy.logerr('Movement towards bag timed out')
                    pass

                # A delay is added here to avoid interference between 
                # the previous and next control command
                rospy.sleep(SLEEP_DELAY)

                # Reset position to get more accurate readings
                self.whole_body.move_to_joint_positions(self.neutral_position)

            elif depth < DEPTH_THRESHOLDS[0]:
                twist = Twist()
                twist.linear.x = GRAB_BACKWARDS_VELOCITY
                self.velocity_pub.publish(twist)
                rospy.loginfo('Robot is too close to bag. Backing up')
                #rospy.sleep(SLEEP_DELAY)

            elif DEPTH_THRESHOLDS[0] < depth < DEPTH_THRESHOLDS[1]:
                rospy.loginfo('Setting robot to neutral position')

                # A delay is added here to avoid interference between 
                # the previous and next control command
                rospy.sleep(SLEEP_DELAY)

                self.whole_body.move_to_joint_positions(self.neutral_position)

                trans = self.transform_point(cam_frame, HAND_FRAME, bag_handle)
                # Perform multiple readings to improve accuracy
                self.samples.append((trans.point.x, trans.point.y, trans.point.z))
                if len(self.samples) >= NUM_SAMPLES:
                    rospy.loginfo('Moving arm to the bag handle')
                    try:
                        x = np.mean([s[0] for s in self.samples])
                        y = np.mean([s[1] for s in self.samples])
                        z = np.mean([s[2] for s in self.samples])

                        self.whole_body.move_end_effector_pose(hsrb.geometry.pose(x=x, y=y, z=z), HAND_FRAME)

                        rospy.loginfo('Picking up the bag handle')
                        twist = Twist()
                        twist.linear.x = GRAB_FORWARD_VELOCITY
                        self.velocity_pub.publish(twist)

                        # A delay is added here to avoid interference between 
                        # the previous and next control command
                        rospy.sleep(SLEEP_DELAY)

                        arm_lift = self.get_updated_arm_lift(GRAB_ARM_LIFT_INCREASE)
                        self.whole_body.move_to_joint_positions({
                            ARM_LIFT_JOINT: arm_lift,
                            WRIST_FLEX_JOINT: GRAB_WRIST_FLEX_ANGLE
                            })

                        self.finished = True

                        rospy.loginfo('Bag should be picked up. Shutting down ...')
                    except hsrb.exceptions.MotionPlanningError:
                        rospy.logerr('Failed to plan arm movement')


if __name__ == '__main__':
    BagGrabber()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
