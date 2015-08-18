/*
 *  Copyright (c) 2013, Universidad de Málaga  - Grupo MAPIR
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
 * Author: Eduardo Fernandez-Moral
 */

#include <warp.h>


inline Pixel GetPixel(const Image* img, float x, float y)
{
 int px = (int)x; // floor of x
 int py = (int)y; // floor of y
 const int stride = img->width;
 const Pixel* p0 = img->data + px + py * stride; // pointer to first pixel

 // load the four neighboring pixels
 const Pixel& p1 = p0[0 + 0 * stride];
 const Pixel& p2 = p0[1 + 0 * stride];
 const Pixel& p3 = p0[0 + 1 * stride];
 const Pixel& p4 = p0[1 + 1 * stride];

 // Calculate the weights for each pixel
 float fx = x - px;
 float fy = y - py;
 float fx1 = 1.0f - fx;
 float fy1 = 1.0f - fy;

 int w1 = fx1 * fy1 * 256.0f;
 int w2 = fx  * fy1 * 256.0f;
 int w3 = fx1 * fy  * 256.0f;
 int w4 = fx  * fy  * 256.0f;

 // Calculate the weighted sum of pixels (for each color channel)
 int outr = p1.r * w1 + p2.r * w2 + p3.r * w3 + p4.r * w4;
 int outg = p1.g * w1 + p2.g * w2 + p3.g * w3 + p4.g * w4;
 int outb = p1.b * w1 + p2.b * w2 + p3.b * w3 + p4.b * w4;
 int outa = p1.a * w1 + p2.a * w2 + p3.a * w3 + p4.a * w4;

 return Pixel(outr >> 8, outg >> 8, outb >> 8, outa >> 8);
}


inline Pixel GetPixelSSE3(const Image<Pixel>* img, float x, float y)
{
 const int stride = img->width;
 const Pixel* p0 = img->data + (int)x + (int)y * stride; // pointer to first pixel

 // Load the data (2 pixels in one load)
 __m128i p12 = _mm_loadl_epi64((const __m128i*)&p0[0 * stride]);
 __m128i p34 = _mm_loadl_epi64((const __m128i*)&p0[1 * stride]);

 __m128 weight = CalcWeights(x, y);

 // convert RGBA RGBA RGBA RGAB to RRRR GGGG BBBB AAAA (AoS to SoA)
 __m128i p1234 = _mm_unpacklo_epi8(p12, p34);
 __m128i p34xx = _mm_unpackhi_epi64(p1234, _mm_setzero_si128());
 __m128i p1234_8bit = _mm_unpacklo_epi8(p1234, p34xx);

 // extend to 16bit
 __m128i pRG = _mm_unpacklo_epi8(p1234_8bit, _mm_setzero_si128());
 __m128i pBA = _mm_unpackhi_epi8(p1234_8bit, _mm_setzero_si128());

 // convert weights to integer
 weight = _mm_mul_ps(weight, CONST_256);
 __m128i weighti = _mm_cvtps_epi32(weight); // w4 w3 w2 w1
         weighti = _mm_packs_epi32(weighti, weighti); // 32->2x16bit

 //outRG = [w1*R1 + w2*R2 | w3*R3 + w4*R4 | w1*G1 + w2*G2 | w3*G3 + w4*G4]
 __m128i outRG = _mm_madd_epi16(pRG, weighti);
 //outBA = [w1*B1 + w2*B2 | w3*B3 + w4*B4 | w1*A1 + w2*A2 | w3*A3 + w4*A4]
 __m128i outBA = _mm_madd_epi16(pBA, weighti);

 // horizontal add that will produce the output values (in 32bit)
 __m128i out = _mm_hadd_epi32(outRG, outBA);
 out = _mm_srli_epi32(out, 8); // divide by 256

 // convert 32bit->8bit
 out = _mm_packus_epi32(out, _mm_setzero_si128());
 out = _mm_packus_epi16(out, _mm_setzero_si128());

 // return
 return _mm_cvtsi128_si32(out);
}

