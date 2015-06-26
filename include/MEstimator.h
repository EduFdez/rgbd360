/*
 *  Copyright (c) 2015,   INRIA Sophia Antipolis - LAGADIC Team
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
 * Author: Eduardo Fernandez-Moral
 */

#ifndef M_ESTIMATOR_H
#define M_ESTIMATOR_H

/*! Robust estimators. This class implements: Huber, Tukey, t-Student.
 */
class MEstimator
{
public:
    /*! Huber weight for robust estimation. */
    template<typename T>
    inline T weightHuber(const T & error)//, const T &scale)
    {
        //        assert(!std::isnan(error) && !std::isnan(scale))
        T weight = (T)1;
        const T scale = 1.345;
        T error_abs = fabs(error);
        if(error_abs < scale){//std::cout << "weight One\n";
            return weight;}

        weight = scale / error_abs;
        //std::cout << "weight " << weight << "\n";
        return weight;
    }

    /*! Huber weight for robust estimation. */
    template<typename T>
    inline T weightHuber_sqrt(const T & error)//, const T &scale)
    {
        //        assert(!std::isnan(error) && !std::isnan(scale))
        T weight = (T)1;
        const T scale = 1.345;
        T error_abs = fabs(error);
        if(error_abs < scale){//std::cout << "weight One\n";
            return weight;}

        weight = sqrt(scale / error_abs);
        //std::cout << "weight " << weight << "\n";
        return weight;
    }

    /*! Tukey weight for robust estimation. */
    template<typename T>
    inline T weightTukey(const T & error)//, const T &scale)
    {
        T weight = (T)0.;
        const T scale = 4.685;
        T error_abs = fabs(error);
        if(error_abs > scale)
            return weight;

        T error_scale = error_abs/scale;
        T w_aux = 1 - error_scale*error_scale;
        weight = w_aux*w_aux;
        return weight;
    }

    /*! T-distribution weight for robust estimation. This is computed following the paper: "Robust Odometry Estimation for RGB-D Cameras" Kerl et al. ICRA 2013 */
    template<typename T>
    inline T weightTDist(const T & error, const T & stdDev, const T & nu)//, const T &scale)
    {
        T err_std = error / stdDev;
        T weight = nu+1 / (nu + err_std*err_std);
        return weight;
    }

    /*! Compute the standard deviation of the T-distribution following the paper: "Robust Odometry Estimation for RGB-D Cameras" Kerl et al. ICRA 2013 */
    template<typename T>
    inline T stdDev_TDist(const std::vector<T> & v_error, const T & stdDev, const T & nu)//, const T &scale)
    {
        std::vector<T> &v_error2( v_error.size() );
        for(size_t i=0; i < v_error.size(); ++i)
            v_error2[i] = v_error[i]*v_error[i];

        int it = 0;
        int max_iterations = 5;
        T diff_convergence = 1e-3;
        T diff_var = 100;
        T variance_prev = 10000;
        while (diff_var > diff_convergence && it < max_iterations)
        {
            T variance = 0.f;
            for(size_t i=0; i < v_error.size(); ++i)
                variance += v_error2[i]*weightTDist(v_error[i]);
            variance /= v_error.size();
            diff_var = fabs(variance_prev - variance);
            variance_prev = variance;
            ++it;
        }
        return sqrt(variance_prev);
    }

    template<typename T>
    inline T weightMEstimator( T error_scaled)
    {
        //std::cout << " error_scaled " << error_scaled << "weightHuber(error_scaled) " << weightHuber(error_scaled) << "\n";
        return weightHuber(error_scaled);
        //return weightTukey(error_scaled);
        //return weightTDist(error_scaled,dev,5);
    }

};

#endif
