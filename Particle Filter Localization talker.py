#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

#import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math
import numpy as np
def talker():
    count = 0
    MARKERS_MAX = 100
    topic = '/visualization_marker_array'
    publisher = rospy.Publisher(topic, MarkerArray,queue_size=10)

    rospy.init_node('register')

    markerArray = MarkerArray()
    #landmark=np.zeros([10,3],dtype=float)
    landmark=[[-8.985,0,0],
              [-3.597,0,0],
              [0,1.337,0],
              [-.439,5.227,0],
              [-13.810,3.284,0],
              [-13.810,6.426,0],
              [-13.810,9.547,0],
              [-9.577,10.180,0],
              [-3.542,10.180,0],
              [-0,9.537,0]
                ]
    #print landmark[0]
    # count=0
    # MARKERS_MAX=
    while not rospy.is_shutdown():

       # ... here I get the data I want to plot into a vector called trans
       for m in range (0,10):
           marker = Marker()
           marker.header.frame_id = "/odom"
           marker.type = marker.SPHERE
           marker.action = marker.ADD
           marker.scale.x = 0.2
           marker.scale.y = 0.2
           marker.scale.z = 0.2
           marker.color.a = 1.0
           marker.color.a = 1.0
           marker.color.r = 1.0
           marker.color.g = 1.0
           marker.color.b = 0.0   
           marker.pose.orientation.w = 1.0
       
           marker.pose.position.x = landmark[m][0]+1.590#+.4390   ##starting position offset
           marker.pose.position.y = landmark[m][1]-1.337-0.04145  ##starting position offset
           marker.pose.position.z = landmark[m][2]
           marker.id=m
           # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
           if(count > MARKERS_MAX):
               markerArray.markers.pop(0)
           # else:
           #  count += 1
           markerArray.markers.append(marker)

       # id = 0
       # for m in markerArray.markers:
       #     m.id = id
       #     id += 1  

        # Publish the MarkerArray
       publisher.publish(markerArray)
       count += 1

       rospy.sleep(1)       
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
