#include <stdio.h>
void my_int_func(int x)
{
    printf( "%d\n", x );
}


int main()
{
    void (*foo)(int);
    foo = &my_int_func;

    /* call my_int_func (note that you do not need to write (*foo)(2) ) */
    foo( 2 );
    /* but if you want to, you may */
    (*foo)( 2 );

    return 0;
}

//#include <immintrin.h>
//#include <iostream>
//using namespace std;

//int main() {
//    float out[8];
//    float a[8] = { 0.0,1.0,2.0,3.0,4.0,5.0,6.0,7};
//    __m256 test =  _mm256_load_ps(&a[0]);
//    cout << "" << endl; // prints
//    return 0;
//}

//// sort algorithm example
//#include <iostream>     // std::cout
//#include <algorithm>    // std::sort
//#include <vector>       // std::vector
////#include <Eigen/Core>
//#include "/usr/local/include/eigen3/Eigen/Core"

//using namespace std;

//int main () {

//#if _SSE2
//    std::cout << __SSE2__ << " _SSE2 \n";
//#endif

//#if _AVX
//    std::cout << " _AVX \n";
//#endif


//    const size_t n_pts = 1200;
//    Eigen::MatrixXf xyz = Eigen::MatrixXf::Ones(n_pts,3);
//    Eigen::MatrixXf xyz2;
//    xyz2.resize(xyz.rows(),xyz.cols());
//    __m256 _const2 = _mm256_set1_ps(2.f);
//    std::vector<int> idx_(n_pts);

//    float *_x = &xyz(0,0);
//    float *_x_out = &xyz2(0,0);
//    int *_ind = &idx_[0];

//    cout << " alignment 32 x " << (((unsigned long)_x & 31) == 0) << " \n";
//    cout << " alignment 32 x " << (((unsigned long)_x_out & 31) == 0) << " \n";
//    cout << " alignment 16 ind " << (((unsigned long)_ind & 15) == 0) << " \n";
//    cout << " alignment 32 ind " << (((unsigned long)_ind & 31) == 0) << " \n";

//    for(size_t r=0;r<n_pts; r+=8)
//    {
//        __m256 block_xyz = _mm256_load_ps(_x+r);
//        __m256 block_x = _mm256_mul_ps( _const2, block_xyz );
//        _mm256_store_ps(_x_out+r, block_x);
//    }

//    for(size_t r=0;r<20; r++)
//    {
//        cout << "val " << xyz(r,0) << " " << xyz2(r,0) << endl;
//    }

//  std::vector<float> v(4, 0.2f);
//  v.push_back(0.3f);
//  v[2] = 0.1f;

//  vector<size_t> idx(v.size());
//  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

//  sort(idx.begin(), idx.end(),
//       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

//  std::cout << "ordering:";
//  for (std::vector<size_t>::iterator it=idx.begin(); it!=idx.end(); ++it)
//    std::cout << ' ' << *it;
//  std::cout << '\n';

//  return 0;
//}

//bool myfunction (int i,int j) { return (i<j); }

//struct myclass {
//  bool operator() (int i,int j) { return (i<j);}
//} myobject;

/////*! Sort a vector and retrieve the indexes of teh sorted values.*/
////std::vector<size_t> sort_indexes__(const std::vector<float> & v)
////{
////  // initialize original index locations
////  std::vector<size_t> idx(v.size());
////  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

////  // sort indexes based on comparing values in v
////  std::sort( idx.begin(), idx.end(), [&v](size_t i1, size_t i2) -> bool {return (v[i1]) > (v[i2]);} );

////  return idx;
////}

//vector<size_t> sort_indexes(const vector<T> &v) {

//  // initialize original index locations
//  vector<size_t> idx(v.size());
//  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

//  // sort indexes based on comparing values in v
//  sort(idx.begin(), idx.end(),
//       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

//  return idx;
//}

//std::vector<size_t> ordered(std::vector<float> const& values) {
//    std::vector<size_t> indices(values.size());
//    std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));

//    std::sort(
//        indices.begin(), indices.end(),
//        [&](size_t a, size_t b) { return values[a] < values[b]; }
//    );
//    return indices;
//}

//int main () {
//  int myints[] = {32,71,12,45,26,80,53,33};
//  std::vector<int> myvector (myints, myints+8);               // 32 71 12 45 26 80 53 33

//  // using default comparison (operator <):
//  std::sort (myvector.begin(), myvector.begin()+4);           //(12 32 45 71)26 80 53 33

//  // using function as comp
//  std::sort (myvector.begin()+4, myvector.end(), myfunction); // 12 32 45 71(26 33 53 80)

//  // using object as comp
//  std::sort (myvector.begin(), myvector.end(), myobject);     //(12 26 32 33 45 53 71 80)

//  // print out content:
//  std::cout << "myvector contains:";
//  for (std::vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
//    std::cout << ' ' << *it;
//  std::cout << '\n';

//  std::vector<float> w(4, 0.2f);
//  w.push_back(0.3f);

//  std::vector<size_t> idx = sort_indexes(w);

//  return 0;
//}


//// sort algorithm example
//#include <iostream>     // std::cout
//#include <algorithm>    // std::sort
//#include <vector>       // std::vector
//#include <math.h>

//using namespace std;

////vector<size_t> sort_indexes(const vector<float> &v) {

////  vector<size_t> idx(v.size());
////  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

////  sort(idx.begin(), idx.end(),
////       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

////  return idx;
////}

//typedef std::pair<size_t,float> mypair;
//bool comparator ( const mypair& l, const mypair& r)
//   { return (fabs(l.second) > fabs(r.second)); }

//int main () {

//  std::vector<float> v(4, 0.2f);
//  v.push_back(0.3f);
//  v[2] = 0.1f;

//  //    std::vector<mypair> idx(v.size());
//  //    for (size_t i = 0; i != idx.size(); ++i)
//  //    {
//  //        idx[i].first = i;
//  //        idx[i].second = v[i];
//  //    }
//  //    std::sort( idx.begin(), idx.end(), comparator );
//  //    std::cout << "ordering:";
//  //    for (std::vector<mypair>::iterator it=idx.begin(); it!=idx.end(); ++it)
//  //      std::cout << ' ' << it->first << ":" << it->second;
//  //    std::cout << '\n';

//  vector<size_t> idx(v.size());
//  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

//  sort(idx.begin(), idx.end(),
//       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

////  //std::vector<size_t> idx = sort_indexes(v);

////  // print out content:
////  std::cout << "ordering:";
////  for (std::vector<size_t>::iterator it=idx.begin(); it!=idx.end(); ++it)
////    std::cout << ' ' << *it;
////  std::cout << '\n';

//  return 0;
//}
