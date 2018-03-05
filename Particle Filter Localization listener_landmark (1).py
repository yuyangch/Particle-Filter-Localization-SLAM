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

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic

import rospy
import tf
#import math
from nav_msgs.msg import Odometry
import numpy as np
import random



from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from apriltags_ros.msg import AprilTagDetectionArray
from apriltags_ros.msg import AprilTagDetection
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg
import message_filters
import scipy
import scipy.stats
import random
########################INITIALIZATION####################################################

#subsample counters######################################################################
subsample_interval= 5 
subsample_counter= 0
Observation_subsample_interval=2
Observation_subsample_counter=0


#Initialize Particles and weights####################################################################
Particle_numbers=400
Particles=np.zeros([Particle_numbers,3],dtype=float)
#print Particles[0]
#Particles1=np.zeros([0,3],dtype=float)

#Particles1.vstack([1,1,1])
#Particles12=np.vstack((Particles1,Particles[0]))

#Particles12=np.vstack((Particles12,Particles[0]))
#print Particles12
#print Particles12
BestParticle=[0,0,0]
Particles_weight=np.zeros([Particle_numbers,1],dtype=float)
Particles_weight[:,0]=1.0/Particle_numbers

##this is the starting angle from odometer
starting_quaternion = (
    0,
    0,
    -0.00113446377045,
    0.999999356496)
euler = tf.transformations.euler_from_quaternion(starting_quaternion)
Particles[:,2]=euler[2]

#current and previous data from /odom##################################################

Odom_current=Odometry() 
Odom_current.pose.pose.orientation.z=-0.00113446377045
Odom_current.pose.pose.orientation.w=0.999999356496
Odom_previous=Odom_current
Odom_best=Odometry() 
Odom_best.pose.pose.orientation.z=-0.00113446377045
Odom_best.pose.pose.orientation.w=0.999999356496
##################motion model variance parameters alpha1-alpha4 #####################################
alpha=[.002,.02,.02,.002]

#######################Marker array######################################################
markerArray = MarkerArray()
#############################land mark global position####################################
#addded empty cells so landmark[m] matches tag ID information
landmark=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[-8.985,0,0],
          [-3.597,0,0],
          [0,1.337,0],
          [-.439,5.227,0],[],[],[],[],[],[],
          [-13.810,3.284,0],
          [-13.810,6.426,0],
          [-13.810,9.547,0],
          [-9.577,10.180,0],
          [-3.542,10.180,0],
          [-0,9.537,0]
            ]

# def beam(Z,Range):
#     #scipy.stats.norm(Range, .25).pdf(Z)
#     return scipy.stats.norm(Range, .25).cdf(Z+.1)-scipy.stats.norm(Range, .25).cdf(Z-.1) # find the P(Z|range) with ~N(range,.25). and use Z+-.01 to fi


#----------------------------------------------------------------------------------------#
def motion_model(Particles,relative_motion):
    #simply add gaussian noise to odometry motion model, reference to the text book
    global Particle_numbers
    global alpha 
    delta_rot1=relative_motion[0]
    delta_trans=relative_motion[1]
    delta_rot2=relative_motion[2]

    for m in range (0,Particle_numbers):
        delta_rot1_hat=delta_rot1-random.gauss(0,alpha[0]*delta_rot1**2+alpha[1]*delta_trans**2)
        delta_trans_hat=delta_trans-random.gauss(0,alpha[2]*delta_trans**2+alpha[3]*delta_rot1**2+alpha[3]*delta_rot2**2)
        delta_rot2_hat=delta_rot2-random.gauss(0,alpha[0]*delta_rot1**2+alpha[1]*delta_trans**2)
        Particles[m,0]=Particles[m,0]+delta_trans_hat*np.math.cos(Particles[m,2]+delta_rot1_hat)
        Particles[m,1]=Particles[m,1]+delta_trans_hat*np.math.sin(Particles[m,2]+delta_rot1_hat)
        Particles[m,2]=Particles[m,2]+delta_rot1_hat+delta_rot2_hat
    return Particles

def sensor_beam_model(observed_landmark_worldframe,predicted_landmark_worldframe):
    #assume independence of noise in x,and y direction. thus P=px*py
    x_predicted=predicted_landmark_worldframe[0,0]
    y_predicted=predicted_landmark_worldframe[1,0]
    x_observed=observed_landmark_worldframe[0,0]
    y_observed=observed_landmark_worldframe[1,0]
    x_variance=1.1
    y_variance=1.8
    gaussian_cdf_discretization_parameter=.08
    #print y_predicted
    Px=scipy.stats.norm(x_predicted, x_variance).cdf(x_observed+gaussian_cdf_discretization_parameter)-scipy.stats.norm(x_predicted, x_variance).cdf(x_observed-gaussian_cdf_discretization_parameter)
    Py=scipy.stats.norm(y_predicted, y_variance).cdf(y_observed+gaussian_cdf_discretization_parameter)-scipy.stats.norm(y_predicted, y_variance).cdf(y_observed-gaussian_cdf_discretization_parameter)


    return Px*Py


def low_varaince_resampling():
    #reference to textbook p110
    global Particles
    global Particles_weight
    global Particle_numbers
    Resampled_Particles=np.zeros([0,3],dtype=float)
    r=random.uniform(0,1.0/Particle_numbers)
    c=Particles_weight[0,0]
    i=0
    for m in range(0,Particle_numbers):
        U=r+(m)*1.0/Particle_numbers
        while (U>c):
            i=i+1
            c=c+Particles_weight[i,0]
        Resampled_Particles=np.vstack((Resampled_Particles,Particles[i]))
        #Resampled_Particles.append(Particles[i])






    #print Resampled_Particles    
    return Resampled_Particles



def callback1(data):
    global subsample_interval
    global subsample_counter
    global BestParticle
    subsample_counter+=1
    if(subsample_counter%subsample_interval==0):
        global Odom_previous
        global Odom_current
        global Particles    
        subsample_counter=0
        Odom_previous=Odom_current
        Odom_current=data

        Particles_copy=Particles

        previous_quaternion = (
            Odom_previous.pose.pose.orientation.x,
            Odom_previous.pose.pose.orientation.y,
            Odom_previous.pose.pose.orientation.z,
            Odom_previous.pose.pose.orientation.w)
        previous_euler = tf.transformations.euler_from_quaternion(previous_quaternion)
        
        current_quaternion = (
            Odom_current.pose.pose.orientation.x,
            Odom_current.pose.pose.orientation.y,
            Odom_current.pose.pose.orientation.z,
            Odom_current.pose.pose.orientation.w)
        current_euler = tf.transformations.euler_from_quaternion(current_quaternion)
        BestParticle[0]=Odom_current.pose.pose.position.x
        BestParticle[1]=Odom_current.pose.pose.position.y
        BestParticle[2]=current_euler[2]

        delta_rot1=np.math.atan2(Odom_current.pose.pose.position.y-Odom_previous.pose.pose.position.y,Odom_current.pose.pose.position.x-Odom_previous.pose.pose.position.x)-previous_euler[2]
        delta_trans=np.math.sqrt((Odom_current.pose.pose.position.y-Odom_previous.pose.pose.position.y)**2+(Odom_current.pose.pose.position.x-Odom_previous.pose.pose.position.x)**2)
        delta_rot2=current_euler[2]-previous_euler[2]-delta_rot1
        relative_motion=[delta_rot1,delta_trans,delta_rot2]
                 
        if (relative_motion[0]!=0 or relative_motion[1]!=0 or relative_motion[1]!=0):
            Particles=motion_model(Particles_copy,relative_motion)
            BestParticle[0]=Particles[:,0].sum()/float(Particle_numbers)
            BestParticle[1]=Particles[:,1].sum()/float(Particle_numbers)
            BestParticle[2]=Particles[:,2].sum()/float(Particle_numbers)

            Odom_best.pose.pose.position.x=BestParticle[0]
            Odom_best.pose.pose.position.y=BestParticle[1]
            Odom_best.pose.pose.position.z=0
            best_euler=[0,0,BestParticle[2]]
            best_quaternion=tf.transformations.quaternion_from_euler(0,0,BestParticle[2])
            Odom_best.pose.pose.orientation.x=best_quaternion[0]
            Odom_best.pose.pose.orientation.y=best_quaternion[1]
            Odom_best.pose.pose.orientation.z=best_quaternion[2]
            Odom_best.pose.pose.orientation.w=best_quaternion[3]
            cloud_points=np.zeros([Particle_numbers,3],dtype=float)
            cloud_points[:,0]=Particles[:,0]
            cloud_points[:,1]=Particles[:,1]
            #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data)
            #rospy.loginfo('Particles %s', Particles)

            ##################Publishing Odom massages #####################################################
            pub = rospy.Publisher('/odom_subsampled', Odometry, queue_size=500)
            pub.publish(data)


            Odom_best.header=std_msgs.msg.Header()
            Odom_best.header.frame_id='odom'
            Odom_best.header.stamp= rospy.Time.now()
            pub_best=rospy.Publisher('/odom_subsampled_maximum_likelyhood', Odometry, queue_size=500)
            pub_best.publish(Odom_best)
            #pub2= rospy.Publisher('/MCL', Odometry, queue_size=10)
            #pub2.publish(data)



            ################publishing point cloud massages ################################################
            pcl_pub = rospy.Publisher("/my_pcl_topic", PointCloud2,queue_size=500)

            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'odom'
            scaled_polygon_pcl = pcl2.create_cloud_xyz32(header, cloud_points)
            pcl_pub.publish(scaled_polygon_pcl)        
                    
    #return 1;
def callback2(data):
    #global Observation_subsample_interval
    #global Observation_subsample_counter

    #Observation_subsample_counter+=1
    #if(subsample_counter%Observation_subsample_interval==0):
    

    global Odom_previous
    global Odom_current
    global BestParticle
    global markerArray
    global Particle_numbers
    global Particles
    global Particles_weight
    if data.detections:
        ######################################################Part 1, publish landmark estimated land mark position referencing to esimated body frame###############
    #     print "empty"
    # else:
        #print data.detections[0].pose.pose
        #print len(data.detections)
        x=data.detections[0].pose.pose.position.x
        y=data.detections[0].pose.pose.position.y
        z=data.detections[0].pose.pose.position.z
        X=np.matrix([[x],[y],[z]])
        R=np.matrix([[0,0,1],[-1,0,0],[0,-1,0]])
        t=np.matrix([[-0.0870],[0.0125],[0.2972]])
        landmark_inbody_frame=R*X+t
        # current_quaternion = (
        #     BestParticle.pose.pose.orientation.x,
        #     BestParticle.pose.pose.orientation.y,
        #     BestParticle.pose.pose.orientation.z,
        #     BestParticle.pose.pose.orientation.w)
        # current_euler = tf.transformations.euler_from_quaternion(current_quaternion)
        R=np.matrix([[np.cos(BestParticle[2]),-np.sin(BestParticle[2])],[np.sin(BestParticle[2]),np.cos(BestParticle[2])]])
        nonstandardframe_R=np.matrix([[0,1],[-1,0]])
        #nonstandardframe_R=np.matrix([[1,0],[0,1]])
        # print landmark_inbody_frame
        # print landmark_inbody_frame[0,0]
        # print landmark_inbody_frame[1,0]
        landmark_inbody_frame_short=np.matrix([[landmark_inbody_frame[0,0]],[landmark_inbody_frame[1,0]]])
        rotated_landmark_inbody_frame_short=nonstandardframe_R.T*R*nonstandardframe_R*landmark_inbody_frame_short
        translated_landmark_inbody_frame_short=rotated_landmark_inbody_frame_short+np.matrix([[BestParticle[0]],[BestParticle[1]]])
        #translated_landmark_inbody_frame_short=landmark_inbody_frame_short+np.matrix([[BestParticle[0]],[BestParticle[1]]])

        #print data.detections[0].id
        #print translated_landmark_inbody_frame_short


        count = 0
        MARKERS_MAX = 10000
        topic = '/visualization_landmark_onthemove'
        publisher = rospy.Publisher(topic, MarkerArray,queue_size=100)

        #rospy.init_node('register')

        
        #landmark=np.zeros([10,3],dtype=float)
        # landmark=[[-8.985,0,0],
        #           [-3.597,0,0],
        #           [0,1.337,0],
        #           [-.439,5.227,0],
        #           [-13.810,3.284,0],
        #           [-13.810,6.426,0],
        #           [-13.810,9.547,0],
        #           [-9.577,10.180,0],
        #           [-3.542,10.180,0],
        #           [-0,9.537,0]
        #             ]
        #print landmark[0]
        # count=0
        # MARKERS_MAX=
        if(1):

           # ... here I get the data I want to plot into a vector called trans
           
           marker = Marker()
           marker.header.frame_id = "/odom"
           marker.type = marker.SPHERE
           marker.action = marker.ADD
           marker.scale.x = 0.2
           marker.scale.y = 0.2
           marker.scale.z = 0.2
           marker.color.a = 1.0
           marker.color.a = 1.0
           marker.color.r = 0.0
           marker.color.g = 1.0
           marker.color.b = 0.0   
           marker.pose.orientation.w = 1.0
       
           marker.pose.position.x = translated_landmark_inbody_frame_short[0,0]
           marker.pose.position.y = translated_landmark_inbody_frame_short[1,0]
           marker.pose.position.z = 0
           #marker.id=+1
           # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
           if(count > MARKERS_MAX):
               markerArray.markers.pop(0)
           # else:
           #  count += 1
           markerArray.markers.append(marker)

           id = 0
           for m in markerArray.markers:
               m.id = id
               id += 1  

            # Publish the MarkerArray
           publisher.publish(markerArray)
           count += 1
        #subsample_counter=0
            

        ######################################################Part2 Assigning weights to the weighting vector and resample################
        for tag_order_n in range (0,len(data.detections)):
            if (data.detections[tag_order_n].id != 18 and landmark[data.detections[tag_order_n].id]):
                x=data.detections[0].pose.pose.position.x
                y=data.detections[0].pose.pose.position.y
                z=data.detections[0].pose.pose.position.z
                X=np.matrix([[x],[y],[z]])
                R1=np.matrix([[0,0,1],[-1,0,0],[0,-1,0]])
                t=np.matrix([[-0.0870],[0.0125],[0.2972]])
                landmark_inbody_frame=R1*X+t
                # current_quaternion = (
                #     BestParticle.pose.pose.orientation.x,
                #     BestParticle.pose.pose.orientation.y,
                #     BestParticle.pose.pose.orientation.z,
                #     BestParticle.pose.pose.orientation.w)
                # current_euler = tf.transformations.euler_from_quaternion(current_quaternion)
                
                nonstandardframe_R=np.matrix([[0,1],[-1,0]])
                #nonstandardframe_R=np.matrix([[1,0],[0,1]])
                # print landmark_inbody_frame
                # print landmark_inbody_frame[0,0]
                # print landmark_inbody_frame[1,0]
                landmark_inbody_frame_short=np.matrix([[landmark_inbody_frame[0,0]],[landmark_inbody_frame[1,0]]])


                for m in range (0,Particle_numbers):
                    R2=np.matrix([[np.cos(Particles[m,2]),-np.sin(Particles[m,2])],[np.sin(Particles[m,2]),np.cos(Particles[m,2])]])
                    rotated_landmark_inbody_frame_short=nonstandardframe_R.T*R2*nonstandardframe_R*landmark_inbody_frame_short
                    translated_landmark_world_frame_short=rotated_landmark_inbody_frame_short+np.matrix([[Particles[m,0]],[Particles[m,1]]])


                    translated_landmark_world_frame_short_predicted=np.matrix([[landmark[data.detections[tag_order_n].id][0]+1.590],[landmark[data.detections[tag_order_n].id][1]-1.337-0.04145]])
                    if(np.math.isnan(Particles_weight[m,0])):
                        print "warning nan detected"
                    #print Particles_weight[m][0]

                    Particles_weight[m,0]=sensor_beam_model(translated_landmark_world_frame_short,translated_landmark_world_frame_short_predicted)

                Particles_weight=Particles_weight/Particles_weight.sum()
                
                Resampled_Particles=low_varaince_resampling()

                Particles=Resampled_Particles
                BestParticle[0]=Particles[:,0].sum()/float(Particle_numbers)
                BestParticle[1]=Particles[:,1].sum()/float(Particle_numbers)
                BestParticle[2]=Particles[:,2].sum()/float(Particle_numbers)
                # print Particles[:,0].sum()/float(Particle_numbers)
                # print Particles[:,1].sum()/float(Particle_numbers)
                # print Particles[:,2].sum()/float(Particle_numbers)
                #BestParticle=[]
                Odom_best.pose.pose.position.x=BestParticle[0]
                Odom_best.pose.pose.position.y=BestParticle[1]
                Odom_best.pose.pose.position.z=0
                best_euler=[0,0,BestParticle[2]]
                best_quaternion=tf.transformations.quaternion_from_euler(0,0,BestParticle[2])
                Odom_best.pose.pose.orientation.x=best_quaternion[0]
                Odom_best.pose.pose.orientation.y=best_quaternion[1]
                Odom_best.pose.pose.orientation.z=best_quaternion[2]
                Odom_best.pose.pose.orientation.w=best_quaternion[3]

                Odom_best.header=std_msgs.msg.Header()
                Odom_best.header.frame_id='odom'
                Odom_best.header.stamp= rospy.Time.now()
                pub_best=rospy.Publisher('/odom_subsampled_maximum_likelyhood', Odometry, queue_size=100)
                pub_best.publish(Odom_best)




def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('/tag_detections', AprilTagDetectionArray, callback2)
    rospy.Subscriber('/odom', Odometry, callback1)
    #observation_apriltag=message_filters.Subscriber('/tag_detections', AprilTagDetectionArray)
    #motion_odom=message_filters.Subscriber('/odom', Odometry)

    #ts = message_filters.TimeSynchronizer([motion_odom, observation_apriltag], 500)
    #ts = message_filters.TimeSynchronizer([motion_odom, observation_apriltag], 500)
    #ts.registerCallback(callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

# def MCL():
#     return 1



# def measure():
#     return 1

# def resample():
#     return 1



if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass