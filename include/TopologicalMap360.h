/*
 *  Copyright (c) 2012, Universidad de Málaga - Grupo MAPIR and
 *                      INRIA Sophia Antipolis - LAGADIC Team
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of the holder(s) nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *  Author: Eduardo Fernandez-Moral
 */

#ifndef TOPOLOGICALMAP360_H
#define TOPOLOGICALMAP360_H

#include "Map360.h"
#include <mrpt/graphs/CGraphPartitioner.h> // Topological partitioning

/*! This class is used for the Map360's topological arrangement by cutting a graph of nodes (keyframes) linked by common
 *  observations (SSO = Sensed-Space-Overlap). Partitioner is the main function of this class, which produces a new
 *  partitioning of the current neighbourhood.
 */
class TopologicalMap360
{
private:

    /*! Reference to the Map */
    Map360 &Map;

    //  /*! Keyframe registration object */
    //  RegisterRGBD360 registerer;

    /*! SSO matrices coupling adjacent topological areas */
    std::map<unsigned, std::map<unsigned, mrpt::math::CMatrix> > mmNeigSSO;  // A sub-map also contains the SSO matrix w.r.t. its neighbors (with bigger index)

public:

    //  /*! Mutex to syncrhronize eventual changes in the map topological arrangement */
    //  boost::mutex topologicalMutex;

    /*! SSO matrices of the different topological areas */
    std::vector<mrpt::math::CMatrix> vSSO;

    /*! Constructor */
    TopologicalMap360(Map360 &map) :
        Map(map)
    {

    }

    /*! Destructor */
    ~TopologicalMap360()
    {

    }

    /*! Add new connection between two KFs */
    void addConnection(unsigned kf1_global, unsigned kf2_global, float &sso)
    {
        // Search the ordination of the keyframes in their respective submaps
        unsigned kf1_local_ord = std::distance(Map.vsAreas[Map.vpSpheres[kf1_global]->node].begin(), Map.vsAreas[Map.vpSpheres[kf1_global]->node].find(kf1_global) );
        unsigned kf2_local_ord = std::distance(Map.vsAreas[Map.vpSpheres[kf2_global]->node].begin(), Map.vsAreas[Map.vpSpheres[kf2_global]->node].find(kf2_global) );
        if(Map.vpSpheres[kf1_global]->node == Map.vpSpheres[kf2_global]->node) // The KFs belong to the same local map
        {
            vSSO[Map.vpSpheres[kf1_global]->node](kf1_local_ord, kf2_local_ord) = vSSO[Map.vpSpheres[kf1_global]->node](kf2_local_ord, kf1_local_ord) = sso;
        }
        else
        {
            Map.vsNeighborAreas[Map.vpSpheres[kf1_global]->node].insert( Map.vpSpheres[kf2_global]->node );
            Map.vsNeighborAreas[Map.vpSpheres[kf2_global]->node].insert( Map.vpSpheres[kf1_global]->node );

            unsigned node1 = Map.vpSpheres[kf1_global]->node < Map.vpSpheres[kf2_global]->node ? Map.vpSpheres[kf1_global]->node : Map.vpSpheres[kf2_global]->node;
            unsigned node2 = Map.vpSpheres[kf1_global]->node > Map.vpSpheres[kf2_global]->node ? Map.vpSpheres[kf1_global]->node : Map.vpSpheres[kf2_global]->node;
            mmNeigSSO[node1][node2](kf1_local_ord, kf2_local_ord) = mmNeigSSO[node1][node2](kf2_local_ord, kf1_local_ord) = sso;
        }
    }

    /*! Return a symmetric matrix with SSO coeficients of sensed-space-overlap (SSO) between the frames in the vicinity */
    mrpt::math::CMatrix getVicinitySSO(std::set<unsigned> &vicinityCurrentNode)
    {
        //    std::cout << "getVicinitySSO...\m";
        unsigned vicinitySize = 0;
        std::map<unsigned, unsigned> vicinitySSO_pos;    // Keeps the vicinitySSO position of each gMapMM's neighbor sub-map
        for( std::set<unsigned>::iterator node = vicinityCurrentNode.begin(); node != vicinityCurrentNode.end(); node++ )
        {
            vicinitySSO_pos[*node] = vicinitySize;
            vicinitySize += Map.vsAreas[*node].size();
        }

        mrpt::math::CMatrix vicinitySSO(vicinitySize,vicinitySize);
        vicinitySSO.zeros();    // Is this necessary?
        //    std::cout << "SSO_0\n" << vicinitySSO << std::endl << std::endl;
        for( std::set<unsigned>::iterator node = vicinityCurrentNode.begin(); node != vicinityCurrentNode.end(); node++ )
        {
            //    std::cout << "SSO_\n" << vSSO[*node] << std::endl << std::endl;
            vicinitySSO.insertMatrix( vicinitySSO_pos[*node], vicinitySSO_pos[*node], vSSO[*node] ); // Intruduce diagonal block SSO
            //std::cout << "SSO_1\n" << vicinitySSO << std::endl << std::endl;
            // Intruduce off-diagonal SSO
            for( std::set<unsigned>::iterator neig = vicinityCurrentNode.begin(); (neig != vicinityCurrentNode.end() && *neig < *node); neig++ )
            {
                //    std::cout << "SSO_COUPLE\n" << mmNeigSSO[*neig][*node] << std::endl << std::endl;
                vicinitySSO.insertMatrix( vicinitySSO_pos[*neig], vicinitySSO_pos[*node], mmNeigSSO[*neig][*node] ); // Insert up-diagonal block
                vicinitySSO.insertMatrixTranspose( vicinitySSO_pos[*node], vicinitySSO_pos[*neig], mmNeigSSO[*neig][*node] ); // Insert down-diagonal block
            }
            //    std::cout << "SSO_2\n" << vicinitySSO << std::endl << std::endl;
        }
        //    ASSERT_( isZeroDiag(vicinitySSO) );
        //    ASSERT_( isSymmetrical(vicinitySSO) );
        //    std::cout << "SSO\n" << vicinitySSO << std::endl;
        return vicinitySSO;
    }


    /*! Update vicinity information */
    void ArrangeGraphSSO(const std::vector<mrpt::vector_uint> &parts,
                         mrpt::math::CMatrix &SSO,
                         const std::set<unsigned> &newNodes,
                         const std::set<unsigned> &previousNodes_,
                         std::map<unsigned, unsigned> &oldNodeStart)	// Update Map's vicinity & SSO matrices
    {
        //    std::cout << "ArrangeGraphSSO...\n";
        //
        std::set<unsigned> previousNodes = previousNodes_;
        previousNodes.erase(Map.currentArea);

        //      Eigen::MatrixXd SSO_reordered;
        mrpt::math::CMatrix SSO_reordered;
        std::vector<size_t> new_ordenation(0);
        // QUITAR
        //      std::cout << "Old and new ordination of kfs in SSO";
        for( unsigned count = 0; count < parts.size(); count++ )
        {
            std::cout << "\nGroup: ";
            for( unsigned count2 = 0; count2 < parts[count].size(); count2++ )
                std::cout << parts[count][count2] << "\t";
            new_ordenation.insert( new_ordenation.end(), parts[count].begin(), parts[count].end() );
        }
        std::cout << "\nNew order: ";
        for( unsigned count = 0; count < new_ordenation.size(); count++ )
            std::cout << new_ordenation[count] << "\t";
        std::cout << std::endl;

        //std::cout << "SSO \n" << SSO << std::endl;
        SSO.extractSubmatrixSymmetrical( new_ordenation, SSO_reordered ); // Update the own SSO matrices

        //std::cout << "Size most rep " << Map.vSelectedKFs.size() << " " << newNodes.size() << " SSO_reordered\n" << SSO_reordered << std::endl;
        Map.vSelectedKFs.resize(Map.vSelectedKFs.size() + newNodes.size()-previousNodes_.size());

        unsigned posNode1SSO = 0;
        // Update SSO of newNodes: a) Update internal SSO of nodes (diagonal blocks), b) Update connections of the newNodes
        for( std::set<unsigned>::iterator node1 = newNodes.begin(); node1 != newNodes.end(); node1++ )
        {
            std::cout << "Update node " << *node1 << std::endl;
            Map.vsNeighborAreas[*node1].insert( *node1 );

            // a) Update SSO of neighbors
            vSSO[*node1].setSize( Map.vsAreas[*node1].size(), Map.vsAreas[*node1].size() );
            SSO_reordered.extractMatrix( posNode1SSO, posNode1SSO, vSSO[*node1] ); // Update the own SSO matrices

            // Get the most representative keyframe
            float sum_sso, highest_sum_sso = 0;
            unsigned most_representativeKF;
            std::set<unsigned>::iterator row_id = Map.vsAreas[*node1].begin();
            //std::cout << "size vSSO " << *node1 << " " << vSSO[*node1].getRowCount() << " " << Map.vsAreas[*node1].size() << "\n" << vSSO[*node1] << std::endl;
            for(unsigned row=0; row < vSSO[*node1].getRowCount(); row++, row_id++)
            {
                //std::cout << "check " << row << " " << *row_id << std::endl;
                sum_sso = 0;
                for(unsigned col=0; col < vSSO[*node1].getRowCount(); col++)
                    sum_sso += vSSO[*node1](row,col);
                //std::cout << "sum_sso " << sum_sso << " highest_sum_sso " << highest_sum_sso << std::endl;
                if(sum_sso > highest_sum_sso)
                {
                    //std::cout << "assign " << row << " " << *row_id << std::endl;
                    most_representativeKF = *row_id;
                    highest_sum_sso = sum_sso;
                }
            }
            Map.vSelectedKFs[*node1] = most_representativeKF;
            std::cout << "Most representative KF " << *node1 << " " << most_representativeKF << "\n";

            //        std::cout << "Extracted SSO\n" << vSSO[*node1] << std::endl;

            unsigned posNode2SSO = posNode1SSO + Map.vsAreas[*node1].size();
            std::set<unsigned>::iterator node2 = node1;
            node2++;
            for( ; node2 != newNodes.end(); node2++ )
            {
                bool isNeighbor = false;
                for( unsigned frameN1 = 0; frameN1 < Map.vsAreas[*node1].size(); frameN1++ )
                    for( unsigned frameN2 = 0; frameN2 < Map.vsAreas[*node2].size(); frameN2++ )
                        if( SSO_reordered(posNode1SSO + frameN1, posNode2SSO + frameN2) > 0 ) // Graph connected
                        {
                            std::cout << "Neighbor submaps" << Map.vsAreas[*node1].size() << " " << Map.vsAreas[*node2].size() << "\n";
                            Map.vsNeighborAreas[*node1].insert( *node2 );
                            Map.vsNeighborAreas[*node2].insert( *node1 );

                            mrpt::math::CMatrix SSOconnectNeigs( Map.vsAreas[*node1].size(), Map.vsAreas[*node2].size() );
                            SSO_reordered.extractMatrix( posNode1SSO, posNode2SSO, SSOconnectNeigs ); // Update the own SSO matrices
                            std::cout << "SSOconnectNeigs\n" << SSOconnectNeigs << std::endl;

                            mmNeigSSO[*node1][*node2] = SSOconnectNeigs;

                            isNeighbor = true;
                            frameN1 = Map.vsAreas[*node1].size();    // Force exit of nested for loop
                            break;
                        }
                if(!isNeighbor)
                    if(Map.vsNeighborAreas[*node1].count(*node2) != 0 )
                    {
                        //        std::cout << "Entra con " << *node1 << " y " << *node2 << std::endl;
                        Map.vsNeighborAreas[*node1].erase(*node2);
                        mmNeigSSO[*node1].erase(*node2);
                        Map.vsNeighborAreas[*node2].erase(*node1);
                    }
                posNode2SSO += Map.vsAreas[*node2].size();
            }
            posNode1SSO += Map.vsAreas[*node1].size();
        }

        // Update the newNodes submaps' KFdistribution
        //      ArrangeKFdistributionSSO( newNodes );

        // Create list of 2nd order neighbors and make copy of the interSSO that are going to change
        std::map<unsigned, std::map<unsigned, mrpt::math::CMatrix> > prevNeigSSO;
        for( std::set<unsigned>::iterator node = previousNodes.begin(); node != previousNodes.end(); node++ )
            for( std::map<unsigned, mrpt::math::CMatrix>::iterator neigNeig = mmNeigSSO[*node].begin(); neigNeig != mmNeigSSO[*node].end(); neigNeig++ )
                if( newNodes.count(neigNeig->first) == 0 ) // If neigNeig is not in the vicinity
                {
                    if(neigNeig->first < *node)
                    {
                        prevNeigSSO[neigNeig->first][*node] = mmNeigSSO[neigNeig->first][*node];
                        mmNeigSSO[neigNeig->first].erase(*node);
                    }
                    else
                    {
                        prevNeigSSO[neigNeig->first][*node] = mmNeigSSO[neigNeig->first][*node].transpose();
                        mmNeigSSO[*node].erase(neigNeig->first);
                    }
                    Map.vsNeighborAreas[neigNeig->first].erase(*node);
                    Map.vsNeighborAreas[*node].erase(neigNeig->first);
                    std::cout << "Copy 2nd order relation between 2nd " << neigNeig->first << " and " << *node << " whose size is " << Map.vsAreas[*node].size() << std::endl;
                }

        std::cout << "Update the SSO interconnections with 2nd order neighbors\n";
        // Update the SSO interconnections with 2nd order neighbors (the 2nd order neighbors is referred by the map's first element, and the 1st order is referred in its second element)
        //  std::map<unsigned, std::map<unsigned, mrpt::math::CMatrix> > newInterSSO;
        for( std::map<unsigned, std::map<unsigned, mrpt::math::CMatrix> >::iterator neig2nd = prevNeigSSO.begin(); neig2nd != prevNeigSSO.end(); neig2nd++ )
        {
            std::cout << "Analyse neig " << neig2nd->first << std::endl;
            std::map<unsigned, mrpt::math::CMatrix> &interSSO2nd = neig2nd->second;
            mrpt::math::CMatrix subCol( Map.vsAreas[neig2nd->first].size(), 1 );

            // Search for all relations of current neighbors with previous neighbors
            for( std::map<unsigned, mrpt::math::CMatrix>::iterator neig1st = interSSO2nd.begin(); neig1st != interSSO2nd.end(); neig1st++ )
            {
                std::cout << " with neighbor " << neig1st->first << std::endl;
                // Check for non-zero elements
                for( unsigned KF_pos1st = 0; KF_pos1st < Map.vsAreas[neig1st->first].size(); KF_pos1st++ )
                {
                    for( unsigned KF_pos2nd = 0; KF_pos2nd < Map.vsAreas[neig2nd->first].size(); KF_pos2nd++ )
                    {
                        if( KF_pos2nd >= neig1st->second.getRowCount() || KF_pos1st >= neig1st->second.getColCount() )
                            assert(false);
                        //                std::cout << "Matrix dimensions " << neig1st->second.getRowCount() << "x" << neig1st->second.getColCount() << " trying to access " << KF_pos2nd << " " << KF_pos1st << std::endl;
                        if( neig1st->second(KF_pos2nd,KF_pos1st) > 0 )
                        {
                            // Extract subCol
                            neig1st->second.extractMatrix(0,KF_pos1st,subCol);

                            // Search where is now that KF to insert the extracted column
                            unsigned KF_oldPos = oldNodeStart[neig1st->first] + KF_pos1st;
                            bool found = false;
                            for( unsigned new_node = 0; new_node < parts.size(); new_node++ ) // Search first in the same node
                                for( unsigned KF_newPos = 0; KF_newPos < parts[new_node].size(); KF_newPos++ ) // Search first in the same node
                                {
                                    if( KF_oldPos == KF_newPos )
                                    {
                                        Map.vsNeighborAreas[neig2nd->first].insert(neig1st->first);
                                        Map.vsNeighborAreas[neig1st->first].insert(neig2nd->first);

                                        if( neig2nd->first < neig1st->first )
                                        {
                                            if( mmNeigSSO[neig2nd->first].count(neig1st->first) == 0 )
                                                mmNeigSSO[neig2nd->first][neig1st->first].zeros( Map.vsAreas[neig2nd->first].size(), Map.vsAreas[neig1st->first].size() );

                                            mmNeigSSO[neig2nd->first][neig1st->first].insertMatrix( 0, KF_newPos, subCol );
                                        }
                                        else
                                        {
                                            if( mmNeigSSO[neig1st->first].count(neig2nd->first) == 0 )
                                                mmNeigSSO[neig1st->first][neig2nd->first].zeros( Map.vsAreas[neig1st->first].size(), Map.vsAreas[neig2nd->first].size() );

                                            mmNeigSSO[neig1st->first][neig2nd->first].insertMatrix( KF_newPos, 0, subCol.transpose() );
                                        }
                                        //                interSSO2nd[neig1st->first].insertMatrix( 0, KF_newPos, subCol );
                                        new_node = parts.size();
                                        found = true;
                                        break;
                                    }
                                }
                            assert(found);
                            break;
                        }
                    }
                }
            }
            std::cout << "InterSSO calculated\n";

        }

        std::cout << "Finish ArrangeGraphSSO\n";
    }


    /*! Arrange the parts given by the partitioning function */
    void RearrangePartition( std::vector<mrpt::vector_uint> &parts )
    {
        std::cout << "RearrangePartition...\n";
        std::vector<mrpt::vector_uint> parts_swap;
        parts_swap.resize( parts.size() );
        for( unsigned int countPart1 = 0; countPart1 < parts_swap.size(); countPart1++ )	// First, we arrange the parts regarding its front element (for efficiency in the next step)
        {
            unsigned int smallestKF = 0;
            for( unsigned int countPart2 = 1; countPart2 < parts.size(); countPart2++ )
                if( parts[smallestKF].front() > parts[countPart2].front() )
                    smallestKF = countPart2;
            parts_swap[countPart1] = parts[smallestKF];
            parts.erase( parts.begin() + smallestKF );
        }
        parts = parts_swap;
    }


    /*! Calculate Partitions of the current map(s). Arrange KFs and points of updated partition */
    void Partitioner()
    {
        std::cout << "Partitioner...\n";
        mrpt::utils::CTicTac time;	// Clock to measure performance times
        time.Tic();

        unsigned numPrevNeigNodes = Map.vsNeighborAreas[Map.currentArea].size();
        std::vector<mrpt::vector_uint> parts;  // Vector of vectors to keep the KFs index of the different partitions (submaps)
        parts.reserve( numPrevNeigNodes+5 );	// We reserve enough memory for the eventual creation of more than one new map	//Preguntar a JL -> Resize o reserve? // Quitar // Necesario?
        //    std::cout << "getVicinitySSO(Map.vsNeighborAreas[Map.currentArea])...\n";
        //    std::cout << "Map.currentArea " << Map.currentArea << ". Neigs:";
        //    for( std::set<unsigned>::iterator node = Map.vsNeighborAreas[Map.currentArea].begin(); node != Map.vsNeighborAreas[Map.currentArea].end(); node++ )
        //      std::cout << " " << *node;
        //    std::cout << std::endl;

        mrpt::math::CMatrix SSO = getVicinitySSO(Map.vsNeighborAreas[Map.currentArea]);
        //std::cout << "SSO\n" << SSO << std::endl;
        if(SSO.getRowCount() < 3)
            return;

        mrpt::graphs::CGraphPartitioner<mrpt::math::CMatrix>::RecursiveSpectralPartition(SSO, parts, 0.4, false, true, true, 3);
        std::cout << "Time RecursiveSpectralPartition " << time.Tac()*1000 << "ms" << std::endl;

        int numberOfNewMaps = parts.size() - numPrevNeigNodes;
        std::cout <<"numPrevNeigNodes " << numPrevNeigNodes << " numberOfNewMaps " << numberOfNewMaps << std::endl;

        if( numberOfNewMaps == 0 ) //|| !ClosingLoop ) // Restructure the map only if there isn't a loop closing right now
            return;

        RearrangePartition( parts ); // Arrange parts ordination to reduce computation in next steps

        //    { mrpt::synch::CCriticalSectionLocker csl(&CS_RM_T); // CRITICAL SECTION Tracker
        std::cout << "Rearrange map CS\n";

        //    std::cout << "Traza1.b\n";
        if( numberOfNewMaps > 0 )
        {
            //	// Show previous KF* ordination
            std::cout << "\nParts:\n";
            for( unsigned counter_part=0; counter_part < parts.size(); counter_part++ )	// Search for the new location of the KF
            {
                for( unsigned counter_KFpart=0; counter_KFpart < parts[counter_part].size(); counter_KFpart++)
                    std::cout << parts[counter_part][counter_KFpart] << " ";
            }

            //    std::cout << "Traza1.2\n";

            std::vector<unsigned> prevDistributionSSO;
            std::map<unsigned, unsigned> oldNodeStart;
            std::map<unsigned, std::set<unsigned> > vsFramesInPlace_prev;
            unsigned posInSSO = 0;
            for( std::set<unsigned>::iterator node = Map.vsNeighborAreas[Map.currentArea].begin(); node != Map.vsNeighborAreas[Map.currentArea].end(); node++ )
            {
                vsFramesInPlace_prev[*node] = Map.vsAreas[*node];
                oldNodeStart[*node] = posInSSO;
                posInSSO += Map.vsAreas[*node].size();
                for( std::set<unsigned>::iterator frame = Map.vsAreas[*node].begin(); frame != Map.vsAreas[*node].end(); frame++ )
                    prevDistributionSSO.push_back(*frame);
            }

            //    std::vector<unsigned> new_DistributionSSO;

            std::set<unsigned> previousNodes = Map.vsNeighborAreas[Map.currentArea];
            std::set<unsigned> newNodes = previousNodes;

            for( int NumNewMap = 0; NumNewMap < numberOfNewMaps; NumNewMap++ )	// Should new maps be created?
            {
                newNodes.insert(Map.vsAreas.size());
                mmNeigSSO[Map.vsAreas.size()] = std::map<unsigned, mrpt::math::CMatrix>();
                Map.vsNeighborAreas.push_back( std::set<unsigned>() );
                Map.vsAreas.push_back( std::set<unsigned>() );
                mrpt::math::CMatrix sso(parts[previousNodes.size()+NumNewMap].size(),parts[previousNodes.size()+NumNewMap].size());
                vSSO.push_back(sso);

                // Set the local reference system of this new node
                //      unsigned size_append = 0;
            }

//            std::cout << "Traza1.3a\n";

            //        std::cout << "previousNodes: ";
            //        for( std::set<unsigned>::iterator node = previousNodes.begin(); node != previousNodes.end(); node++ )
            //          std::cout << *node << " ";
            //        std::cout << "Map.vsNeighborAreas[Map.currentArea]: ";
            //        for( std::set<unsigned>::iterator node = Map.vsNeighborAreas[Map.currentArea].begin(); node != Map.vsNeighborAreas[Map.currentArea].end(); node++ )
            //          std::cout << *node << " ";
            //        std::cout << std::endl;

            //      std::cout << "Traza1.4\n";
            // Rearrange the KF in the maps acording to the new ordination
            unsigned counter_SSO_KF = 0;
            unsigned actualFrame = prevDistributionSSO[counter_SSO_KF];
            //        std::vector<unsigned> insertedKFs(parts.size(), 0);
            unsigned counterNode = 0;
            for( std::set<unsigned>::iterator node = previousNodes.begin(); node != previousNodes.end(); node++ )// For all the maps of the previous step
            {
                //      std::cout << "Traza1.5a\n";
                std::cout << "NumNode: " << *node << " previous KFs: " << Map.vsAreas[*node].size() << std::endl;
                //      unsigned counter_KFsamePart = insertedKFs[counterNode];??
                unsigned counter_KFsamePart = 0;
                for( std::set<unsigned>::iterator itFrame = vsFramesInPlace_prev[*node].begin(); itFrame != vsFramesInPlace_prev[*node].end(); itFrame++ )	//For all the keyframes of this map (except for the those transferred in this step
                {
                    std::cout << "Check KF " << *itFrame << "th of map " << *node << " which has KFs " << Map.vsAreas[*node].size() << std::endl;
                    while ( counter_KFsamePart < parts[counterNode].size() && counter_SSO_KF > parts[counterNode][counter_KFsamePart] )
                        ++counter_KFsamePart;	// It takes counter_KFpart to the position where it can find its corresponding KF index (increase lightly the efficiency)

                    if( counter_KFsamePart < parts[counterNode].size() && counter_SSO_KF == parts[counterNode][counter_KFsamePart] )	//Check if the KF has switched to another map (For efficiency)
                    {
                        //          new_DistributionSSO.push_back(actualFrame);
                        ++counter_KFsamePart;
                        std::cout << "KF " << counter_SSO_KF << std::endl;
                    }
                    else	// KF switches
                    {
                        bool found = false;
                        std::set<unsigned>::iterator neigNode = newNodes.begin();
                        for( unsigned counter_part=0; counter_part < parts.size(); counter_part++, neigNode++ )	// Search for the new location of the KF
                        {
                            std::cout << "counter_part " << counter_part << " neigNode " << *neigNode << std::endl;
                            if( counter_part == counterNode )
                            { continue;}	// Do not search in the same map
                            for( unsigned counter_KFpart = 0; counter_KFpart < parts[counter_part].size(); counter_KFpart++)
                            {
                                if( counter_SSO_KF < parts[counter_part][counter_KFpart] )  // Skip part when the parts[counter_part][counter_KFpart] is higher than counter_SSO_KF
                                    break;
                                if( counter_SSO_KF == parts[counter_part][counter_KFpart] )	// Then transfer KF* and its points to the corresponding map
                                {
                                    std::cout << "aqui1\n";
                                    found = true;

                                    //                SE3<> se3NewFromOld = foundMap.se3MapfromW * thisMap.se3MapfromW.inverse();
                                    Map.vsAreas[*node].erase(actualFrame);
                                    Map.vsAreas[*neigNode].insert(actualFrame);
                                    Map.vpSpheres[actualFrame]->node = *neigNode;
                                    //                Map.vpSpheres[actualFrame]->node = *neigNode;

                                    counter_part = parts.size();	// This forces to exit both for loops
                                    break;
                                }// End if
                            }// End for
                        }// End for
                        if(!found)
                        {
                            std::cout << "\n\nWarning1: KF lost\n Looking for KF " << counter_SSO_KF << "\nParts:\n";
                            for( unsigned counter_part=0; counter_part < parts.size(); counter_part++ )	// Search for the new location of the KF
                            {
                                for( unsigned counter_KFpart=0; counter_KFpart < parts[counter_part].size(); counter_KFpart++)
                                    std::cout << parts[counter_part][counter_KFpart] << " ";
                                //                  std::cout << " insertedKFs " << insertedKFs[counter_part] << std::endl;
                            }
                            assert(false);
                        }
                    } // End else
                    ++counter_SSO_KF;
                    actualFrame = prevDistributionSSO[counter_SSO_KF];
                } // End for
                ++counterNode;
            } // End for

            for( std::set<unsigned>::iterator node = newNodes.begin(); node != newNodes.end(); node++ )// For all the maps of the previous step
                std::cout << "Node " << *node << "numKFs " << Map.vsAreas[*node].size() << std::endl;

            ArrangeGraphSSO( parts, SSO, newNodes, previousNodes, oldNodeStart );	// Update neighbors and SSO matrices of the surrounding maps

            // Search for the part containing the last KF to set it as gMap
            unsigned nextCurrentNode, highestKF = 0;
            int part_count=0;
            for( std::set<unsigned>::iterator node = newNodes.begin(); node != newNodes.end(); node++, part_count++ )
            {
                std::cout << "Last of " << *node << " is " << parts[part_count].back() << std::endl;
                if( parts[part_count].back() > highestKF )
                {
                    nextCurrentNode = *node;
                    highestKF = parts[part_count].back();
                }
            }
            std::cout << "nextCurrentNode " << nextCurrentNode << std::endl;
            Map.currentArea = nextCurrentNode;

            // Update current pose

            std::cout << "critical section " << time.Tac()*1000 << " ms\n";

        } // End if( numberOfNewMaps > 0 )
        //    std::cout << "Map Rearranged CS\n";

    }
};

#endif
