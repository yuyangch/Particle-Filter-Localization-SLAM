# Particle-Filter-Localization-SLAM

Particle filter Localization result:
[Video](https://www.youtube.com/watch?v=LlDtm2JpKg0&t=677s)



3/14/2018 update:

Q:How are particles updated based on observations?
A:at each observation, each particle's weight is updated based it's location in the gaussian distribution of the observation. Then, Low Variance Resampling(a stochastic process, based on the book Probablistic Robotics) is performed to resample the particles, After the Resampling, each particle's weight is returned to 1
