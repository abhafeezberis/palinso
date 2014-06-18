#ifndef LIST_HPP
#define LIST_HPP

#include "core/cgfdefs.hpp"
#include <ostream>

namespace CGF{

  /*Double linked list*/
  template<class T>
  class ListNode;

  template<class T>
  class List;

  template<class T>
  class Allocator{
  public:
    static Allocator* instance(){
      if(inst == 0){
        inst = new Allocator;
      }
      return inst;
    }

    T* allocate(){
      if(used == size){
        extendStorage();
      }
      int idx = index[used++];
      return &(data[idx]);
    }
  protected:
    ~Allocator(){
      PRINT_FUNCTION;
      delete [] data;
      delete [] index;
    }

  private:
    void extendStorage(){
    }

    Allocator(){
      PRINT_FUNCTION;
      message("singleton constructor");
      size = 1000;
      used = 0;

      data = new T[size];
      index = new int[size];
      for(int i=0;i<size;i++){
        index[i] = i;
      }
    }

    T* data;
    int* index;
    int size;
    int used;
    static Allocator<T>* inst;
  };


  template<class T>
  class Iterator{
  public:
    ListNode<T>* ptr;

    /*Dereference operator*/
    T& operator*(){
      return ptr->data;
    }

    /*Pointer operator*/
    T* operator->(){
      return &(ptr->data);
    }

    /*Post increment*/
    Iterator operator++(int i){
      Iterator it;
      it.ptr = ptr;
      ptr = ptr->next;
      return it;
    }

    /*Post decrement*/
    Iterator operator--(int i){
      Iterator it;
      it.ptr = ptr;
      ptr = ptr->prev;
      return it;
    }

    /*Pre increment*/
    Iterator& operator++(){
      ptr = ptr->next;
      return *this;
    }

    /*Pre decrement*/
    Iterator& operator--(){
      ptr = ptr->prev;
      return *this;
    }

    /*Equality*/
    bool operator==(const Iterator& i)const{
      return ptr == i.ptr;
    }

    /*Inequality*/
    bool operator!=(const Iterator& i)const{
      return ptr != i.ptr;
    }

    /*Next item*/
    Iterator next(){
      Iterator i;
      i.ptr = ptr->next;
      return i;
    }

    /*Previous item*/
    Iterator prev(){
      Iterator i;
      i.ptr = ptr->prev;
      return i;
    }
  };

  template<class T>
  class ListNode{
  public:
    ListNode():next(0),prev(0){
    }

    ~ListNode(){
    }

    ListNode* next;
    ListNode* prev;
    T data;
  protected:
    ListNode(const ListNode& l);
    ListNode& operator=(const ListNode& l);
  };

  template<class T>
  class List{
  public:
    List(){
      first = new ListNode<T>;
      last  = new ListNode<T>;
      first->next = last;
      last->prev  = first;
      sz = 0;
    }

    ~List(){
      clear();
      delete first;
      delete last;
    }

    List(const List& c){
      first = new ListNode<T>;
      last  = new ListNode<T>;
      first->next = last;
      last->prev  = first;
      sz = 0;

      Iterator<T> it = c.begin();
      Iterator<T> e = c.end();
      while(it != e){
        append(*it++);
      }
    }

    List& operator=(const List& c){
      if(this == &c){
        message("Self assignment");
      }else{
        clear();
        Iterator<T> it = c.begin();
        Iterator<T> e = c.end();
        while(it != e){
          append(*it++);
        }
      }
      return *this;
    }

    Iterator<T> begin() const{
      Iterator<T> i;
      i.ptr = first->next;
      return i;
    }

    Iterator<T> end() const{
      Iterator<T> i;
      i.ptr = last;
      return i;
    }

    void clear(){
      while(first->next != last){
        ListNode<T>* next = first->next->next;
        delete first->next;
        first->next = next;
        first->next->prev = first;
      }
      sz = 0;
    }

    void append(const T& t){
      ListNode<T>* n = new ListNode<T>;
      ListNode<T>* current = last->prev;
      current->next = n;
      n->prev = current;
      n->data = t;
      last->prev = n;
      n->next = last;
      sz++;
    }

    void append(List<T>& l){
      Iterator<T> iter = l.begin();
      while(l.size() != 0){
        append(*iter);
        l.remove2(iter);
      }
    }

    void appendByValue(T t){
      ListNode<T>* n = new ListNode<T>;
      ListNode<T>* current = last->prev;
      current->next = n;
      n->prev = current;
      n->data = t;
      last->prev = n;
      n->next = last;
      sz++;
    }

    /*Removes the current node and advances to the next node*/
    /*If used in an iteration, do not increase the iterator after removal*/
    void remove2(Iterator<T>& it){
      ListNode<T>* curr = it.ptr;
      ListNode<T>* prev = it.ptr->prev;
      ListNode<T>* next = it.ptr->next;

      next->prev = prev;
      prev->next = next;

      delete curr;
      sz--;
      it.ptr = next;
    }

    /*Removes the current node and advances to the previous node*/
    /*If used in an iteration, always increase the iterator after removal*/
    void remove(Iterator<T>& it){
      ListNode<T>* curr = it.ptr;
      ListNode<T>* prev = it.ptr->prev;
      ListNode<T>* next = it.ptr->next;

      error("deprecated, will be removed");

      next->prev = prev;
      prev->next = next;

      delete curr;
      sz--;
      it.ptr = prev;
    }

    void removeAt(int i){
      ListNode<T>* current = first->next;
      int index = 0;

      if(i>=sz)
        return;

      while(index != i){
        current = current->next;
        index++;
      }
      ListNode<T>* prev = current->prev;
      ListNode<T>* next = current->next;

      prev->next = next;
      next->prev = prev;

      delete current;
      sz--;
    }

    int size()const{
      return sz;
    }

    T& operator[](int idx) const{
      cgfassert(idx >=0 && idx < sz);

      ListNode<T>* current = first->next;
      int index = 0;
      while(index != idx){
        current = current->next;
        index++;
      }
      return current->data;
    }

    void removeDuplicates(){
      unique();
    }

    void unique(){
      if(sz == 0 || sz == 1){
        return;
      }

      sort();

      /*List has at least 2 elements*/

      ListNode<T>* curr = first->next;
      ListNode<T>* next = curr->next;

      while(curr != last && next != last){
        if(curr->data == next->data){
          /*duplicate found, remove current*/
          ListNode<T>* prev = curr->prev;
          prev->next = next;
          next->prev = prev;
          delete curr;
          curr = next;
          next = curr->next;
          sz--;
        }else{
          curr = next;
          next = next->next;
        }
      }
    }

    void sort(){
      /*Mergesort*/
      if(sz == 0 || sz == 1){
        return;
      }

      int insize = 1;
      int nmerges, psize, qsize, i;

      ListNode<T>* list = first->next;

      list->prev = 0;
      first->next = 0;
      cgfassert(list!=0);

      /*Remove link to end*/
      last->prev->next = 0;
      last->prev = 0;

      ListNode<T>* p;
      ListNode<T>* q;

      ListNode<T>* e;
      ListNode<T>* tail;

      while(1){
        p = list;
        list = 0;
        tail = 0;

        nmerges = 0;

        while(p){
          nmerges++;
          q = p;
          psize = 0;
          for(i=0;i<insize;i++){
            psize++;
            q = q->next;
            if(!q)
              break;
          }

          qsize = insize;

          while(psize > 0 || (qsize > 0 && q)){
            if(psize == 0){
              e = q; q = q->next; qsize--;
            }else if(qsize == 0 || !q){
              e = p; p = p->next; psize--;
            }else if(p->data < q->data){
              e = p; p = p->next; psize--;
            }else{
              e = q; q = q->next; qsize--;
            }

            if(tail){
              tail->next = e;
            }else{
              list = e;
            }
            if(true){
              e->prev = tail;
            }
            tail = e;
          }
          p = q;
        }
        tail->next = 0;

        if(nmerges <= 1){
          first->next = list;
          list->prev = first;

          while(list->next){
            list = list->next;
          }
          list->next = last;
          last->prev = list;
          return;
        }
        insize *= 2;
      }
    }
    template<class U> friend inline std::ostream& operator<<(std::ostream& os,
                                                             const List<U>& l);

    friend class Iterator<T>;
  protected:
    ListNode<T>* first;
    ListNode<T>* last;
    int sz;
  };

  template<class U>
  inline std::ostream& operator<<(std::ostream& os, const List<U>& l){
    Iterator<U> it = l.begin();
    while(it != l.end()){
      os << *it++ << ',';
    }
    return os;
  }
}

#endif/*LIST_HPP*/
