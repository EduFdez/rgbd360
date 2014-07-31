------------------------------------------
              - RGBD360 -
------------------------------------------

This project integrates the functionality to do image acquisition, localization and mapping using an omnidirectional RGB-D sensor developed in INRIA Sophia-Antipolis by the team LAGADIC, and with the collaboration of the University of Malaga. This functionality comprises: reading and serializing the data streaming from the omnidirectional RGB-D sensor; registering frames based on a compact planar description of the scene (PbMap http://www.mrpt.org/pbmap); loop closure detection; performing human-guided semi-automatic labelization of the scene; PbMap-based hybrid SLAM (i.e. metric-topological-semantic) with the omnidirectional RGB-D sensor moving freely with 6 DoF, or in planar movement with 3 DoF. Also, some visualization tools are provided to show the results from the above applications.

The documentation of this project can be found in 'doc/html/index.html'
