from visualization_msgs.msg import MarkerArray
# bag = rosbag.Bag('check16.bag')
# for topic, msg, t in bag.read_messages(topics=['/apriltags/detections']):
# 	print msg
# bag.close()
import rosbag
import sys
import math
import tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
from nav_msgs.msg import Odometry

import rospy
import tf
#import math
from nav_msgs.msg import Odometry
import numpy as np
import random
import rosbag
from visualization_msgs.msg import Marker
#from visualization_msgs.msg import MarkerArray
#from apriltags.msg import MarkerArray
# bag = rosbag.Bag('/home/yuyangch/Desktop/output11.bag')
# for topic, msg, t in bag.read_messages(topics=['/apriltags/detections']):
#   print msg
# bag.close()

from apriltags_ros.msg import AprilTagDetectionArray
from apriltags_ros.msg import AprilTagDetection
#from apriltags_ros.msg import AprilTagDetections
#from apriltags import AprilTagDetections
#import apriltags
import apriltags_ros
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg
import message_filters
import scipy
import scipy.stats
import random
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
from math import atan2








############Fast SLAM###########
subsample_interval=50 
subsample_counter= 0
Observation_subsample_interval=1
Observation_subsample_counter=0

Particle_numbers=30
particle_initialization_x_variance=.1
particle_initialization_y_variance=.1
particle_initialization_theta_variance=.05
Particles=np.zeros([Particle_numbers,3],dtype=float)

for m in range(0,Particle_numbers):
    Particles[m,0]=random.gauss(0,particle_initialization_x_variance)
    Particles[m,1]=random.gauss(0,particle_initialization_y_variance)
    Particles[m,2]=random.gauss(0,particle_initialization_theta_variance)

Particle_Landmark_u=np.zeros([Particle_numbers,2],dtype=float)

Particle_Landmark_Sigma=np.zeros([Particle_numbers,2,2],dtype=float)    

BestParticle=[0,0,0]
BestParticle[0]=Particles[:,0].sum()/float(Particle_numbers)
BestParticle[1]=Particles[:,1].sum()/float(Particle_numbers)
BestParticle[2]=Particles[:,2].sum()/float(Particle_numbers)
#print BestParticle

Particles_weight=np.zeros([Particle_numbers,1],dtype=float)
Particles_weight[:,0]=1.0/Particle_numbers

Odom_current=Odometry() 
# Odom_current.pose.pose.orientation.z=-0.00113446377045
# Odom_current.pose.pose.orientation.w=0.999999356496
Odom_previous=Odom_current
Odom_best=Odometry() 
# Odom_best.pose.pose.orientation.z=-0.00113446377045
# Odom_best.pose.pose.orientation.w=0.999999356496
alpha=[.001,.01,.01,.001]

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

landmark_numbers=30

landmark_seen=np.zeros([landmark_numbers,1],dtype=float)
landmark_corresponsding_tag=landmark_seen=np.zeros([landmark_numbers,1],dtype=float)

first_land_mark_discovered=0

########################################################EKF parameters######################
#---------------------H----------------
H=np.zeros([2,2],dtype=float)
H[0,0]=-1
H[1,1]=-1

#---------------------Q----------------

#APriltagReadingXVariance=.35*12
#APriltagReadingYVariance=.6*24

APriltagReadingXVariance=.7
APriltagReadingYVariance=1.5
# r_Variance=np.sqrt(APriltagReadingXVariance+APriltagReadingYVariance)
# phi_Variance=np.math.atan2(APriltagReadingYVariance,APriltagReadingXVariance)

r_Variance=1.5
#phi_Variance=2.3
phi_Variance=2.3

Q_initial=np.diag([1.0,1.0])
# print Q_initial
# print Q_initial[0,0]
# print Q_initial[0,0]*APriltagReadingXVariance
Q_initial[0,0]=Q_initial[0,0]*r_Variance
Q_initial[1,1]=Q_initial[1,1]*phi_Variance
# print Q_initial

#---------------------Sigma----------------

Sigma_initial=inv(H)*Q_initial*np.transpose(inv(H))
New_Sigma_array=np.zeros([Particle_numbers,2,2],dtype=float)

for i in range(0,Particle_numbers):
    New_Sigma_array[i]=Sigma_initial
#default_importance_weight=


##########################################Robot Movement Parameters############################
RobotStopped=0



# def beam(Z,Range):
#     #scipy.stats.norm(Range, .25).pdf(Z)
#     return scipy.stats.norm(Range, .25).cdf(Z+.1)-scipy.stats.norm(Range, .25).cdf(Z-.1) # find the P(Z|range) with ~N(range,.25). and use Z+-.01 to fi

#####
#########################################################################################
Turtlebot_position_recording=[]
#Turtlebot_position_recording=np.zeros
AprilTag_poistion_recording=[]
#AprilTag_poistion_recording
Turtlebotdata_saved=0
AprilTagdata_saved=0





def motion_model(Particles,relative_motion):
    #simply add gaussian noise to odometry motion model, reference to the text book
    global Particle_numbers
    global alpha 
    delta_rot1=relative_motion[0]
    delta_trans=relative_motion[1]
    delta_rot2=relative_motion[2]
    #print Particles
    for m in range (0,Particle_numbers):
        delta_rot1_hat=delta_rot1-random.gauss(0,alpha[0]*delta_rot1**2+alpha[1]*delta_trans**2)
        delta_trans_hat=delta_trans-random.gauss(0,alpha[2]*delta_trans**2+alpha[3]*delta_rot1**2+alpha[3]*delta_rot2**2)
        delta_rot2_hat=delta_rot2-random.gauss(0,alpha[0]*delta_rot1**2+alpha[1]*delta_trans**2)
        Particles[m,0]=Particles[m,0]+delta_trans_hat*np.math.cos(Particles[m,2]+delta_rot1_hat)
        Particles[m,1]=Particles[m,1]+delta_trans_hat*np.math.sin(Particles[m,2]+delta_rot1_hat)
        Particles[m,2]=Particles[m,2]+delta_rot1_hat+delta_rot2_hat
    return Particles




def low_varaince_resampling():
    #reference to textbook p110
    global Particles
    global Particles_weight
    global Particle_numbers
    global Particle_Landmark_u
    global Particle_Landmark_Sigma
    #print Particles

    Resampled_Particles=np.zeros([0,3],dtype=float)

    Resampled_Particle_Landmark_u=np.zeros([0,np.shape(Particle_Landmark_u[0])[0]],dtype=float)
    Resampled_Particle_Landmark_Sigma=np.zeros([0,np.shape(Particle_Landmark_Sigma[0])[0],np.shape(Particle_Landmark_Sigma[0])[1]],dtype=float)

    r=random.uniform(0,1.0/Particle_numbers)
    c=Particles_weight[0,0]
    i=0
    for m in range(0,Particle_numbers):
        U=r+(m)*1.0/Particle_numbers
        while (U>c):
            i=i+1
            c=c+Particles_weight[i,0]
        Resampled_Particles=np.vstack((Resampled_Particles,Particles[i]))
        # print np.shape(Particle_Landmark_u[i])
        # print np.shape(Resampled_Particle_Landmark_u)
        Resampled_Particle_Landmark_u=np.vstack((Resampled_Particle_Landmark_u,Particle_Landmark_u[i]))
        #print Resampled_Particle_Landmark_u
        # print np.shape(Particle_Landmark_Sigma[i])
        # print np.shape(Resampled_Particle_Landmark_Sigma)
        # print Particle_Landmark_Sigma[i]
        # print Resampled_Particle_Landmark_Sigma
        # print Particle_Landmark_u[i]
        # print Particle_Landmark_Sigma[i]
        Resampled_Particle_Landmark_Sigma=np.vstack((Resampled_Particle_Landmark_Sigma,[Particle_Landmark_Sigma[i]]))
        #Resampled_Particles.append(Particles[i])
    #print Resampled_Particles
    #print Resampled_Particle_Landmark_u
    #print Resampled_Particle_Landmark_Sigma    
    Particles=Resampled_Particles
    Particle_Landmark_u=Resampled_Particle_Landmark_u
    Particle_Landmark_Sigma=Resampled_Particle_Landmark_Sigma
    # k=Particle_Landmark_u[0,:]
    # for i in range (1,Particle_numbers):
    #     k=k+Particle_Landmark_u[i,:]
    # k=k/float(Particle_numbers)
    # Bestlandmark=k
    #print Bestlandmark
    #print Particle_Landmark_Sigma
    #print Particle_Landmark_u
    # print np.shape(Particles)
    # print np.shape(Particle_Landmark_u)
    # print np.shape(Particle_Landmark_Sigma)


    #print Resampled_Particles    
    return
def callback1(data):
    global subsample_interval
    global subsample_counter
    global BestParticle
    global Turtlebot_position_recording
    global RobotStopped

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
        # BestParticle[0]=Odom_current.pose.pose.position.x
        # BestParticle[1]=Odom_current.pose.pose.position.y
        # BestParticle[2]=current_euler[2]

        delta_rot1=np.math.atan2(Odom_current.pose.pose.position.y-Odom_previous.pose.pose.position.y,Odom_current.pose.pose.position.x-Odom_previous.pose.pose.position.x)-previous_euler[2]
        delta_trans=np.math.sqrt((Odom_current.pose.pose.position.y-Odom_previous.pose.pose.position.y)**2+(Odom_current.pose.pose.position.x-Odom_previous.pose.pose.position.x)**2)
        delta_rot2=current_euler[2]-previous_euler[2]-delta_rot1

        relative_motion=[delta_rot1,delta_trans,delta_rot2]
        if (delta_trans>1):
            print 'warning long travel'
        #print relative_motion          
        if (relative_motion[0]>0.001 or relative_motion[1]>0.001 or relative_motion[2]>0.01):
            RobotStopped=0
            Particles=motion_model(Particles_copy,relative_motion)
            BestParticle[0]=Particles[:,0].sum()/float(Particle_numbers)
            BestParticle[1]=Particles[:,1].sum()/float(Particle_numbers)
            BestParticle[2]=Particles[:,2].sum()/float(Particle_numbers)
            # Temp_Recording=np.zeros([1,5],dtype=float)
            # Temp_Recording[0,0]=Odom_current.header.stamp.secs
            # Temp_Recording[0,1]=Odom_current.header.stamp.nsecs
            # Temp_Recording[0,2]=BestParticle[0]
            # Temp_Recording[0,3]=BestParticle[1]
            # Temp_Recording[0,4]=BestParticle[2]

            # Turtlebot_position_recording=np.vstack((Turtlebot_position_recording,Temp_Recording))
            
            # Turtlebot_position_recording.append(Temp_Recording)

            # Odom_best.pose.pose.position.x=BestParticle[0]
            # Odom_best.pose.pose.position.y=BestParticle[1]
            # Odom_best.pose.pose.position.z=0
            # best_euler=[0,0,BestParticle[2]]
            # best_quaternion=tf.transformations.quaternion_from_euler(0,0,BestParticle[2])
            # Odom_best.pose.pose.orientation.x=best_quaternion[0]
            # Odom_best.pose.pose.orientation.y=best_quaternion[1]
            # Odom_best.pose.pose.orientation.z=best_quaternion[2]
            # Odom_best.pose.pose.orientation.w=best_quaternion[3]
            # cloud_points=np.zeros([Particle_numbers,3],dtype=float)
            # cloud_points[:,0]=Particles[:,0]
            # cloud_points[:,1]=Particles[:,1]
            # #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data)
            # #rospy.loginfo('Particles %s', Particles)

            # ##################Publishing Odom massages #####################################################
            # pub = rospy.Publisher('/odom_subsampled', Odometry, queue_size=5000)
            # pub.publish(data)


            # Odom_best.header=std_msgs.msg.Header()
            # Odom_best.header.frame_id='odom'
            # Odom_best.header.stamp= rospy.Time.now()
            # pub_best=rospy.Publisher('/odom_subsampled_maximum_likelyhood', Odometry, queue_size=5000)
            # pub_best.publish(Odom_best)
            # #pub2= rospy.Publisher('/MCL', Odometry, queue_size=10)
            # #pub2.publish(data)



            # ################publishing point cloud massages ################################################
            # pcl_pub = rospy.Publisher("/my_pcl_topic", PointCloud2,queue_size=5000)

            # header = std_msgs.msg.Header()
            # header.stamp = rospy.Time.now()
            # header.frame_id = 'odom'
            # scaled_polygon_pcl = pcl2.create_cloud_xyz32(header, cloud_points)
            # pcl_pub.publish(scaled_polygon_pcl)        
        else:
            RobotStopped=1            
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
    global Turtlebot_position_recording
    global AprilTag_poistion_recording
    global AprilTagdata_saved
    global Turtlebotdata_saved
    global Q_initial
    #global H
    #global Sigma_initial
    #global New_Sigma_array
    global Particle_Landmark_u
    global Particle_Landmark_Sigma
    global first_land_mark_discovered
    global RobotStopped


    if data.detections:
        #print Particles      

        ######################################################Part1 Assigning weights to the weighting vector and resample,also update the landmark related arrays################
        #for tag_order_n in range (0,len(data.detections)):
        for tag_order_n in range (0,1):

            if (data.detections[tag_order_n].id != 18  and landmark[data.detections[tag_order_n].id]):         #if tag detected

                New_u_array=np.zeros([Particle_numbers,2],dtype=float) #create a new u (mean poistion of land mark) array :nx2
                New_Sigma_array=np.zeros([Particle_numbers,2,2],dtype=float)

                if(landmark_seen[data.detections[tag_order_n].id]==0):    #####Landmark previously not observered
                    
                    landmark_seen[data.detections[tag_order_n].id]=1     #mark landmark as observered

                    #New_u_array=np.zeros([Particle_numbers,2],dtype=float)#create a new u (mean poistion of land mark) array :nx2
                    #New_Sigma_array=np.zeros([Particle_numbers,2,2],dtype=float)#create a new Sigma (Covariance of land mark position) array :nX2X2

                    x=data.detections[tag_order_n].pose.pose.position.x
                    y=data.detections[tag_order_n].pose.pose.position.y
                    z=data.detections[tag_order_n].pose.pose.position.z
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
                    
                    #nonstandardframe_R=np.matrix([[0,1],[-1,0]])
                    nonstandardframe_R=np.matrix([[1,0],[0,1]])
                    #nonstandardframe_R=np.matrix([[0,1],[1,0]])
                    # print landmark_inbody_frame
                    # print landmark_inbody_frame[0,0]
                    # print landmark_inbody_frame[1,0]
                    landmark_inbody_frame_short=np.matrix([[landmark_inbody_frame[0,0]],[landmark_inbody_frame[1,0]]]) # get the landmark position in the body frame
                    


                    for m in range (0,Particle_numbers):
                        R2=np.matrix([[np.cos(Particles[m,2]),-np.sin(Particles[m,2])],[np.sin(Particles[m,2]),np.cos(Particles[m,2])]])    #get the landmark position in world frame
                        rotated_landmark_inbody_frame_short=nonstandardframe_R.T*R2*nonstandardframe_R*landmark_inbody_frame_short
                        translated_landmark_world_frame_short=rotated_landmark_inbody_frame_short+np.matrix([[Particles[m,0]],[Particles[m,1]]])
                        #print nonstandardframe_R.T
                        New_u_array[m,0]=translated_landmark_world_frame_short[0,0] #save the newly observed landmark position in the new u array
                        New_u_array[m,1]=translated_landmark_world_frame_short[1,0] 

                        x_diff_hat=translated_landmark_world_frame_short[0,0]-Particles[m,0]
                        y_diff_hat=translated_landmark_world_frame_short[1,0]-Particles[m,1]
                        q_hat=(x_diff_hat)**2+(y_diff_hat)**2
                        r_t_hat=np.sqrt((x_diff_hat)**2+(y_diff_hat)**2)
                        phi_t_hat=np.math.atan2(y_diff_hat,x_diff_hat)-Particles[m,2]

                        z_t_hat=np.matrix([[r_t_hat],[phi_t_hat]])    #prediction                      

                        H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[-float(y_diff_hat)/q_hat,float(x_diff_hat)/q_hat]]) 
                        #H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[float(x_diff_hat)/q_hat,-float(y_diff_hat)/q_hat]])

                        #H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[-float(x_diff_hat)/q_hat,float(y_diff_hat)/q_hat]])
                        #H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[float(x_diff_hat)/q_hat,float(y_diff_hat)/q_hat]])
                        #H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[float(x_diff_hat)/q_hat,float(y_diff_hat)/q_hat]])
                        #H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[-float(x_diff_hat)/q_hat,-float(y_diff_hat)/q_hat]])
                        #H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[float(y_diff_hat)/q_hat,-float(x_diff_hat)/q_hat]]) 
                        #print pinv(H_prediction)
                        Sigma_initial=inv(H_prediction)*Q_initial*np.transpose(inv(H_prediction))
                        New_Sigma_array[m]=Sigma_initial                        

                        #translated_landmark_world_frame_short_predicted=np.matrix([[landmark[data.detections[tag_order_n].id][0]+1.590],[landmark[data.detections[tag_order_n].id][1]-1.337-0.04145]])
                        if(np.math.isnan(Particles_weight[m,0])):
                            print "warning nan detected"
                        #print Particles_weight[m][0]

                        #Particles_weight[m,0]=sensor_beam_model(translated_landmark_world_frame_short,translated_landmark_world_frame_short_predicted)

                    #------------------------EKF array concatenation------------------------------
                    if (first_land_mark_discovered==0):
                        first_land_mark_discovered=1
                        #print first_land_mark_discovered
                        Particle_Landmark_Sigma=New_Sigma_array
                        Particle_Landmark_u=New_u_array
                        landmark_corresponsding_tag[data.detections[tag_order_n].id]=1 #first landmark observed,record
                        k=Particle_Landmark_u[0,:]
                        for i in range (1,Particle_numbers):
                            k=k+Particle_Landmark_u[i,:]
                        k=k/float(Particle_numbers)
                        Bestlandmark=k
                        #print Bestlandmark
                        #print Particle_Landmark_Sigma
                        # print Particle_Landmark_u
                    else:    
                        Particle_Landmark_Sigma=np.hstack((Particle_Landmark_Sigma,New_Sigma_array))
                        Particle_Landmark_u=np.hstack((Particle_Landmark_u,New_u_array))

                        landmark_corresponsding_tag[data.detections[tag_order_n].id]=len(Particle_Landmark_u[0])/2    #record tag's observed order
                        #print landmark_corresponsding_tag
                        # print np.shape(Particles)
                        # print np.shape(Particle_Landmark_u)
                        # print np.shape(Particle_Landmark_Sigma)
                        # print Particle_Landmark_Sigma
                        # print Particle_Landmark_u
                        k=Particle_Landmark_u[0,:]
                        for i in range (1,Particle_numbers):
                            k=k+Particle_Landmark_u[i,:]
                        k=k/float(Particle_numbers)
                        Bestlandmark=k
                        #print Bestlandmark
                        #print Particle_Landmark_Sigma
                        # print Particle_Landmark_u
                        #print landmark_corresponsding_tag
                else:
                    if (RobotStopped==0):                                        #####Landmark previously observered        
                        x=data.detections[tag_order_n].pose.pose.position.x
                        y=data.detections[tag_order_n].pose.pose.position.y
                        z=data.detections[tag_order_n].pose.pose.position.z
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
                        
                        #nonstandardframe_R=np.matrix([[0,1],[-1,0]])
                        nonstandardframe_R=np.matrix([[1,0],[0,1]])
                        #nonstandardframe_R=np.matrix([[0,1],[1,0]])
                        # print landmark_inbody_frame
                        # print landmark_inbody_frame[0,0]
                        # print landmark_inbody_frame[1,0]
                        landmark_inbody_frame_short=np.matrix([[landmark_inbody_frame[0,0]],[landmark_inbody_frame[1,0]]])
                        phi_t=np.math.atan2(landmark_inbody_frame[1,0],landmark_inbody_frame[0,0])
                        #print phi_t
                        r_t=np.sqrt((landmark_inbody_frame[1,0])**2+(landmark_inbody_frame[0,0])**2)
                        z_t=np.matrix([[r_t],[phi_t]])
                        #print Particle_Landmark_Sigma
                        #print Particle_Landmark_u
                        #--------------------------------update the u and covariance for each particle---------------
                        for m in range (0,Particle_numbers):

                            # R2=np.matrix([[np.cos(Particles[m,2]),-np.sin(Particles[m,2])],[np.sin(Particles[m,2]),np.cos(Particles[m,2])]])
                            # rotated_landmark_inbody_frame_short=nonstandardframe_R.T*R2*nonstandardframe_R*landmark_inbody_frame_short
                            # translated_landmark_world_frame_short=rotated_landmark_inbody_frame_short+np.matrix([[Particles[m,0]],[Particles[m,1]]])

                            # # x_diff=translated_landmark_world_frame_short[0,0]-Particles[m,0]
                            # # y_diff=translated_landmark_world_frame_short[1,0]-Particles[m,1]
                            # q=(x_diff)**2+(y_diff)**2
                            # r_t=np.sqrt((x_diff)**2+(y_diff)**2)
                            # print r_t
                            #phi_t=np.math.atan2(y_diff,x_diff)-Particles[m,2]
                            #print phi_t
                            #z_t=np.matrix([[r_t],[phi_t]])  #measurement 

                            #H_t=np.matrix([[-()/np.sqrt(q)],[],[]])


                            indexofthelandmark_in_u_array=int(landmark_corresponsding_tag[data.detections[tag_order_n].id]-1)*2   #retrieve index in u array

                            translated_landmark_world_frame_short_predicted=np.matrix([[Particle_Landmark_u[m,indexofthelandmark_in_u_array]],[Particle_Landmark_u[m,indexofthelandmark_in_u_array+1]]])

                            #print translated_landmark_world_frame_short_predicted

                            x_diff_hat=translated_landmark_world_frame_short_predicted[0,0]-Particles[m,0]
                            y_diff_hat=translated_landmark_world_frame_short_predicted[1,0]-Particles[m,1]

                            q_hat=(x_diff_hat)**2+(y_diff_hat)**2
                            r_t_hat=np.sqrt((x_diff_hat)**2+(y_diff_hat)**2)
                            phi_t_hat=np.math.atan2(y_diff_hat,x_diff_hat)-Particles[m,2]

                            z_t_hat=np.matrix([[r_t_hat],[phi_t_hat]])    #prediction
                            #print z_t-z_t_hat                      

                            H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[-float(y_diff_hat)/q_hat,float(x_diff_hat)/q_hat]])
                            #H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[float(x_diff_hat)/q_hat,-float(y_diff_hat)/q_hat]])
                            #H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[-float(x_diff_hat)/q_hat,float(y_diff_hat)/q_hat]])
                            #H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[float(x_diff_hat)/q_hat,float(y_diff_hat)/q_hat]])

                            #=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[-float(x_diff_hat)/q_hat,-float(y_diff_hat)/q_hat]])
                            #H_prediction=np.matrix([[float(x_diff_hat)/np.sqrt(q_hat),float(y_diff_hat)/np.sqrt(q_hat)],[float(y_diff_hat)/q_hat,-float(x_diff_hat)/q_hat]]) 

                            indexofthelandmark_in_Sigma_array=int(landmark_corresponsding_tag[data.detections[tag_order_n].id]-1)*2 #retrieve index in Sigma array
                            #indexofthelandmark_in_Sigma_array=int(landmark_corresponsding_tag[data.detections[tag_order_n].id]-1)*2 #retrieve index in Sigma array
                            OldSigma=Particle_Landmark_Sigma[m][indexofthelandmark_in_Sigma_array:indexofthelandmark_in_Sigma_array+2]
                            #print OldSigma

                            Q_measurement=Q_initial
                            #print Q_measurement
                            #print H_prediction
                            #print Q_initial
                            Q=H_prediction*OldSigma*np.transpose(H_prediction)+Q_measurement
                            #print Q
                            #print Q

                            K=OldSigma*np.transpose(H_prediction)*inv(Q)
                            if(det(Q)<10**(-9)):
                                print "warning ubstable Q"
                            # z_t=np.zeros([2,1],dtype=float)
                            # z_t[0,0]=translated_landmark_world_frame_short_predicted[0,0]
                            # z_t[1,0]=translated_landmark_world_frame_short_predicted[1,0]
                            # z_predict=np.zeros([2,1],dtype=float)

                            # old_u=np.zeros([2,1],dtype=float)s

                            Innovation=z_t-z_t_hat

                            new_u=translated_landmark_world_frame_short_predicted+K*(Innovation)


                            Particle_Landmark_u[m,indexofthelandmark_in_u_array]=new_u[0,0]
                            Particle_Landmark_u[m,indexofthelandmark_in_u_array+1]=new_u[1,0]

                            new_sigma=(np.diag([1.0,1.0]-K*H_prediction))*OldSigma

                            Particle_Landmark_Sigma[m,indexofthelandmark_in_Sigma_array:(indexofthelandmark_in_Sigma_array+2),:]= new_sigma

                            new_exp=np.transpose(Innovation)*inv(Q)*Innovation
                            #print new_exp
                            #print np.transpose(new_exp[0,0])
                            #print (abs(2*np.pi*Q)**(-.5))*np.exp((-.5)*(np.transpose(new_exp[0,0])))

                            Particles_weight[m,0]=(det(2*np.pi*Q)**(-.5))*np.exp((-.5)*((new_exp[0,0])))

                            if(np.math.isnan(Particles_weight[m,0])):
                                print "warning nan detected"
                            #print Particles_weight[m][0]

                            # New_u_array[m,0]=translated_landmark_world_frame_short[0,0] #save the newly observed landmark position in the new u array
                            # New_u_array[m,1]=translated_landmark_world_frame_short[1,0] 
                            #Particles_weight[m,0]=sensor_beam_model(translated_landmark_world_frame_short,translated_landmark_world_frame_short_predicted)
                        #-------------------------------------update the best position-------------------------------    
                        Particles_weight=Particles_weight/Particles_weight.sum()

                        
                        low_varaince_resampling()

                        Particles_weight[:,0]=1.0/Particle_numbers
                        #Particles=Resampled_Particles
                        BestParticle[0]=Particles[:,0].sum()/float(Particle_numbers)
                        BestParticle[1]=Particles[:,1].sum()/float(Particle_numbers)
                        BestParticle[2]=Particles[:,2].sum()/float(Particle_numbers)
                        #-----------------------------------------Record the updated best position----------------------
                        # Temp_Recording=np.zeros([1,5],dtype=float)
                        # Temp_Recording[0,0]=data.detections[0].pose.header.stamp.secs
                        # Temp_Recording[0,1]=data.detections[0].pose.header.stamp.nsecs
                        # Temp_Recording[0,2]=BestParticle[0]
                        # Temp_Recording[0,3]=BestParticle[1]
                        # Temp_Recording[0,4]=BestParticle[2]
                        # #Turtlebot_position_recording=np.vstack((Turtlebot_position_recording,Temp_Recording))

                        # Turtlebot_position_recording.append(Temp_Recording)



                        # if ((Temp_Recording[0,0]>1487260100-600) and Turtlebotdata_saved==0):
                        #     np.save('/home/yuyangch/Turtlebot_position', Turtlebot_position_recording)
                        #     Turtlebotdata_saved=1
                        #     print "saved_Turtlebot"




                        # #---------------------publish the new best position-------------------------------------------    
                        # # print Particles[:,0].sum()/float(Particle_numbers)
                        # # print Particles[:,1].sum()/float(Particle_numbers)
                        # # print Particles[:,2].sum()/float(Particle_numbers)
                        # #BestParticle=[]
                        # Odom_best.pose.pose.position.x=BestParticle[0]
                        # Odom_best.pose.pose.position.y=BestParticle[1]
                        # Odom_best.pose.pose.position.z=0
                        # best_euler=[0,0,BestParticle[2]]
                        # best_quaternion=tf.transformations.quaternion_from_euler(0,0,BestParticle[2])
                        # Odom_best.pose.pose.orientation.x=best_quaternion[0]
                        # Odom_best.pose.pose.orientation.y=best_quaternion[1]
                        # Odom_best.pose.pose.orientation.z=best_quaternion[2]
                        # Odom_best.pose.pose.orientation.w=best_quaternion[3]

                        # Odom_best.header=std_msgs.msg.Header()
                        # Odom_best.header.frame_id='odom'
                        # Odom_best.header.stamp= rospy.Time.now()
                        # pub_best=rospy.Publisher('/odom_subsampled_maximum_likelyhood', Odometry, queue_size=5000)
                        # pub_best.publish(Odom_best)





        ###############################################Part 2 -save and publish the observed landmark ,reference to the updated best prediction   ################         
        # x=data.detections[0].pose.pose.position.x
        # y=data.detections[0].pose.pose.position.y
        # z=data.detections[0].pose.pose.position.z
        # X=np.matrix([[x],[y],[z]])
        # R=np.matrix([[0,0,1],[-1,0,0],[0,-1,0]])
        # t=np.matrix([[-0.0870],[0.0125],[0.2972]])
        # landmark_inbody_frame=R*X+t
        # # current_quaternion = (
        # #     BestParticle.pose.pose.orientation.x,
        # #     BestParticle.pose.pose.orientation.y,
        # #     BestParticle.pose.pose.orientation.z,
        # #     BestParticle.pose.pose.orientation.w)
        # # current_euler = tf.transformations.euler_from_quaternion(current_quaternion)
        # R=np.matrix([[np.cos(BestParticle[2]),-np.sin(BestParticle[2])],[np.sin(BestParticle[2]),np.cos(BestParticle[2])]])
        # nonstandardframe_R=np.matrix([[0,1],[-1,0]])
        # #nonstandardframe_R=np.matrix([[1,0],[0,1]])
        # #nonstandardframe_R=np.matrix([[0,1],[1,0]])
        # # print landmark_inbody_frame
        # # print landmark_inbody_frame[0,0]
        # # print landmark_inbody_frame[1,0]
        # landmark_inbody_frame_short=np.matrix([[landmark_inbody_frame[0,0]],[landmark_inbody_frame[1,0]]])
        # rotated_landmark_inbody_frame_short=nonstandardframe_R.T*R*nonstandardframe_R*landmark_inbody_frame_short
        # translated_landmark_world_frame_short=rotated_landmark_inbody_frame_short+np.matrix([[BestParticle[0]],[BestParticle[1]]])
        # #translated_landmark_inbody_frame_short=landmark_inbody_frame_short+np.matrix([[BestParticle[0]],[BestParticle[1]]])

        # #print data.detections[0].id
        # #print translated_landmark_inbody_frame_short


        # count = 0
        # MARKERS_MAX = 10000
        # topic = '/visualization_landmark_onthemove'
        # publisher = rospy.Publisher(topic, MarkerArray,queue_size=500)

        # #rospy.init_node('register')

        
        # #landmark=np.zeros([10,3],dtype=float)
        # # landmark=[[-8.985,0,0],
        # #           [-3.597,0,0],
        # #           [0,1.337,0],
        # #           [-.439,5.227,0],
        # #           [-13.810,3.284,0],
        # #           [-13.810,6.426,0],
        # #           [-13.810,9.547,0],
        # #           [-9.577,10.180,0],
        # #           [-3.542,10.180,0],
        # #           [-0,9.537,0]
        # #             ]
        # #print landmark[0]
        # # count=0
        # # MARKERS_MAX=
        # Temp_landmark_Recording=np.zeros([1,5],dtype=float)
        # Temp_landmark_Recording[0,0]=data.detections[tag_order_n].pose.header.stamp.secs
        # Temp_landmark_Recording[0,1]=data.detections[tag_order_n].pose.header.stamp.nsecs
        # Temp_landmark_Recording[0,2]=data.detections[tag_order_n].id
        # Temp_landmark_Recording[0,3]=translated_landmark_world_frame_short[0,0]
        # Temp_landmark_Recording[0,4]=translated_landmark_world_frame_short[1,0]

        

        # #AprilTag_poistion_recording=np.vstack((AprilTag_poistion_recording,Temp_landmark_Recording))
        # AprilTag_poistion_recording.append(Temp_landmark_Recording)
                            
        # if ((Temp_landmark_Recording[0,0]>1487260100-600) and  AprilTagdata_saved==0):
        #     np.save('/home/yuyangch/AprilTagposition', AprilTag_poistion_recording)
        #     AprilTagdata_saved=1
        #     print "saved_AprilTag"
        

        # # ... here I get the data I want to plot into a vector called trans

        # marker = Marker()
        # marker.header.frame_id = "/odom"
        # marker.type = marker.SPHERE
        # marker.action = marker.ADD
        # marker.scale.x = 0.2
        # marker.scale.y = 0.2
        # marker.scale.z = 0.2
        # marker.color.a = 1.0
        # marker.color.a = 1.0
        # marker.color.r = 0.0
        # marker.color.g = 1.0
        # marker.color.b = 0.0   
        # marker.pose.orientation.w = 1.0

        # marker.pose.position.x = translated_landmark_world_frame_short[0,0]
        # marker.pose.position.y = translated_landmark_world_frame_short[1,0]
        # marker.pose.position.z = 0
        # #marker.id=+1
        # # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
        # if(count > MARKERS_MAX):
        #    markerArray.markers.pop(0)
        # # else:
        # #  count += 1
        # markerArray.markers.append(marker)

        # id = 0
        # for m in markerArray.markers:
        #    m.id = id
        #    id += 1  

        # # Publish the MarkerArray
        # publisher.publish(markerArray)
        # count += 1
        # #subsample_counter=0

################################################Naive PLot#################################
bag = rosbag.Bag('./CSE668_1.bag')
d=np.zeros((3062,1))
tag_global_x = []
tag_global_y = []
TAG = []
x_position = []
y_position = []
theta = []
cnt = 0



for topic, msg, t in bag.read_messages(topics=['/odom', '/tag_detections']):
	#cnt +=1
	#if cnt>5000:
		#break
	#else:
	    if topic == '/odom':
	    	#print t,'yo'

	        x_r = msg.pose.pose.position.x 
	        y_r = msg.pose.pose.position.y 
		x_position.append(x_r)
		y_position.append(y_r)
	        #degree = msg.pose.pose.orientation.z
	        #print x_r
		x = msg.pose.pose.orientation.x
		y = msg.pose.pose.orientation.y
		z = msg.pose.pose.orientation.z
		w = msg.pose.pose.orientation.w
		#print x
		(raw, pitch, yaw) = tf.transformations.euler_from_quaternion([x,y,z,w]) 
		theta.append(yaw)
	    if topic == '/tag_detections':
	    	#print t,'yoyo'
	        if len(msg.detections) == 0:
		    continue
		# print msg
		#print msg.markers[0].id
		if msg.detections[0].id >0:

		        x_tag = msg.detections[0].pose.pose.position.x
		        z_tag = msg.detections[0].pose.pose.position.z   

		        x_tag_calb = math.sqrt(x_tag**2+z_tag**2) * math.sin(math.atan(x_tag/z_tag) - yaw)
		        z_tag_calb = math.sqrt(x_tag**2+z_tag**2) * math.cos(math.atan(x_tag/z_tag) - yaw)
			
			tag_id = msg.detections[0].id
			TAG.append(tag_id)
			tag_global_x.append(x_r + z_tag_calb)
			tag_global_y.append(y_r - x_tag_calb)

bag.close()
#print TAG
colortable = ["#008080","#FF0000","#00FF00","#0000FF","#FFFF00","#00FFFF","#FF00FF","#C0C0C0","#808080","#800000","#808000","#008000","#800080","#008080","#000080","#FF8C00","#FFD700","#B8860B","#FFFF00"," 	#7CFC00","#7FFF00","#ADFF2F","#98FB98","#8FBC8F","#00FF7F","#66CDAA","#3CB371","#00CED1","#B0E0E6","#0000CD","#DA70D6","#FF69B4","#FFF8DC"]
colored=cm.rainbow(np.linspace(0,2,66))



for x in range(0,len(tag_global_x)):
       plt.scatter(tag_global_x[x],tag_global_y[x],color=colortable[TAG[x]])


plt.scatter(x_position,y_position)
plt.plot(x_position,y_position)
#plt.show()




##########################################Fast SLAM plot#########################################



bag = rosbag.Bag('./CSE668_1.bag')
d=np.zeros((3062,1))
tag_global_x = []
tag_global_y = []
TAG = []
x_position = []
y_position = []
theta = []
cnt = 0


for topic, msg, t in bag.read_messages(topics=['/odom', '/tag_detections']):
	#cnt +=1
	#if cnt>5000:
		#break
	#else:
	    if topic == '/odom':
	    	#print t,'yo'
	    	callback1(msg)
	    	x_r=BestParticle[0]
	    	y_r=BestParticle[1]
	        # x_r = msg.pose.pose.position.x 
	        # y_r = msg.pose.pose.position.y 
		x_position.append(x_r)
		y_position.append(y_r)
	        #degree = msg.pose.pose.orientation.z
	        #print x_r
		# x = msg.pose.pose.orientation.x
		# y = msg.pose.pose.orientation.y
		# z = msg.pose.pose.orientation.z
		# w = msg.pose.pose.orientation.w
		# #print x
		# (raw, pitch, yaw) = tf.transformations.euler_from_quaternion([x,y,z,w]) 
		yaw=BestParticle[2]
		theta.append(yaw)

	    if topic == '/tag_detections':
	    	#print t,'yoyo'
	        if len(msg.detections) == 0:
		    continue
		# print msg
		#print msg.markers[0].id
		if msg.detections[0].id >0:
			Observation_subsample_counter+=1
			if(Observation_subsample_counter%Observation_subsample_interval==0):

				Observation_subsample_counter=0	
				callback2(msg)
			        x_tag = msg.detections[0].pose.pose.position.x
			        z_tag = msg.detections[0].pose.pose.position.z   

			        x_tag_calb = math.sqrt(x_tag**2+z_tag**2) * math.sin(math.atan(x_tag/z_tag) - yaw)
			        z_tag_calb = math.sqrt(x_tag**2+z_tag**2) * math.cos(math.atan(x_tag/z_tag) - yaw)
				
				tag_id = msg.detections[0].id
				TAG.append(tag_id)

				# x_r=BestParticle[0]
		  #   	y_r=BestParticle[1]

				tag_global_x.append(x_r + z_tag_calb)
				tag_global_y.append(y_r - x_tag_calb)

bag.close()

Bestlandmark_Estimate=Particle_Landmark_u[0]

for x in range(0,len(Particle_Landmark_u[0])):
    Bestlandmark_Estimate[x]=Particle_Landmark_u[:,x].sum()/float(Particle_numbers)



for x in range(0,len(tag_global_x)):
    plt.scatter(tag_global_x[x],tag_global_y[x],color=colortable[TAG[x]])

for x in range (0,len(Bestlandmark_Estimate),2):
    plt.scatter(Bestlandmark_Estimate[x],Bestlandmark_Estimate[x+1],color='Black',s=30)



#plt.scatter(tag_global_x,tag_global_y,color='green')
plt.scatter(x_position,y_position,color='green')
plt.plot(x_position,y_position,color='green')
plt.show()