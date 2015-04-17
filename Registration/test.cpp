//// sort algorithm example
//#include <iostream>     // std::cout
//#include <algorithm>    // std::sort
//#include <vector>       // std::vector

//using namespace std;

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


// sort algorithm example
#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <math.h>

using namespace std;

//vector<size_t> sort_indexes(const vector<float> &v) {

//  vector<size_t> idx(v.size());
//  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

//  sort(idx.begin(), idx.end(),
//       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

//  return idx;
//}

typedef std::pair<size_t,float> mypair;
bool comparator ( const mypair& l, const mypair& r)
   { return (fabs(l.second) > fabs(r.second)); }

int main () {

    std::vector<float> w(4, 0.2f);
    w.push_back(0.3f);
    w[2] = 0.1f;

    std::vector<mypair> idx(w.size());
    for (size_t i = 0; i != idx.size(); ++i)
    {
        idx[i].first = i;
        idx[i].second = w[i];
    }
    std::sort( idx.begin(), idx.end(), comparator );
    std::cout << "ordering:";
    for (std::vector<mypair>::iterator it=idx.begin(); it!=idx.end(); ++it)
      std::cout << ' ' << it->first << ":" << it->second;
    std::cout << '\n';

//  std::vector<float> v(4, 0.2f);
//  v.push_back(0.3f);

//  vector<size_t> idx(v.size());
//  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

//  sort(idx.begin(), idx.end(),
//       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

//  //std::vector<size_t> idx = sort_indexes(v);

//  // print out content:
//  std::cout << "ordering:";
//  for (std::vector<size_t>::iterator it=idx.begin(); it!=idx.end(); ++it)
//    std::cout << ' ' << *it;
//  std::cout << '\n';

  return 0;
}
