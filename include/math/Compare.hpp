#ifndef COMPARE_HPP
#define COMPARE_HPP

namespace CGF{
  template<class T>
  class Compare{
  public:
    static bool less(const T& a, const T& b){
      return a<b;
    }

    static bool equal(const T& a, const T& b){
      return a==b;
    }
  };
}

#endif/*COMPARE_HPP*/
