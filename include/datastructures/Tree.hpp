#ifndef TREE_HPP
#define TREE_HPP
#include <iostream>
#include "core/cgfdefs.hpp"
#include <stdlib.h>
#include "math/Compare.hpp"
#include <string.h>
#include <datastructures/List.hpp>
/*AVL tree*/

namespace CGF{
  template<class T>
  class Tree;

  template<class T>
  class TreeIterator;

  template<class T>
  class TreeReduce;

  template<class T>
  class TreeNode{
  public:
    TreeNode(T& t, int idx):parent(0), l_child(0), r_child(0),
                            data(t), index(idx), balance(0), depth(0){

    }

    ~TreeNode(){
      if(l_child){
        delete l_child;
        l_child = 0;
      }

      if(r_child){
        delete r_child;
        r_child = 0;
      }

      parent = 0;
    }
  protected:
    void computeDepth(){
      int left_depth = 0;
      int right_depth = 0;

      if(l_child){
        left_depth = l_child->depth;
      }
      if(r_child){
        right_depth = r_child->depth;
      }

      depth = 1+MAX(left_depth, right_depth);
      balance = left_depth - right_depth;
    }

    TreeNode<T>* parent;
    TreeNode<T>* l_child;
    TreeNode<T>* r_child;
    T data;
    int index;
    int balance;
    int depth;

    friend class Tree<T>;
    friend class TreeIterator<T>;
    friend class TreeReduce<T>;
  };

  template<class T>
  class TreeIterator{
  public:
    TreeIterator(){
    }

    TreeIterator(TreeNode<T>* p):ptr(p){
    }

    T& operator*(){
      return ptr->data;
    }

    T* operator->(){
      return &(ptr->data);
    }

    /*Post increment*/
    TreeIterator operator++(int i){
      TreeIterator it;
      it.ptr = ptr;
      next();
      return it;
    }

    /*Post decrement*/
    TreeIterator operator--(int i){
      TreeIterator it;
      it.ptr = ptr;
      prev();
      return it;
    }

    /*Pre increment*/
    TreeIterator& operator++(){
      next();
      return *this;
    }

    /*Pre decrement*/
    TreeIterator& operator--(){
      prev();
      return *this;
    }

    /*Equality*/
    bool operator==(const TreeIterator& i)const{
      return ptr == i.ptr;
    }

    /*Equality*/
    bool operator!=(const TreeIterator& i)const{
      return ptr != i.ptr;
    }

  protected:
    TreeNode<T>* ptr;

    void next(){
      if(ptr->r_child){
        /*Find left most child*/
        ptr = ptr->r_child;
        while(ptr->l_child){
          ptr = ptr->l_child;
        }
        /*Arrived at left most child*/
      }else{
        /*Go to parent node if this node is a left child of the parent*/
        if(ptr->parent){
          if(ptr->parent->l_child == ptr){
            ptr = ptr->parent;
          }else if(ptr->parent->r_child == ptr){
            /*Find first parent node for which
              the current node is a left child*/
            while(ptr->parent){
              if(ptr->parent->r_child == ptr){
                ptr = ptr->parent;
              }else if(ptr->parent->l_child == ptr){
                ptr = ptr->parent;
                return;
              }
            }
            ptr = 0;
          }
        }else{
          ptr = 0;
        }
      }
    }

    void prev(){
      //error();
      message("pointer = %p", ptr);
      if(ptr->l_child){
        /*Find right most child*/
        ptr = ptr->l_child;
        while(ptr->r_child){
          ptr = ptr->r_child;
        }
        /*Arrived at left most child*/
      }else{
        /*Go to parent node if this node is a right child of the parent*/
        if(ptr->parent){
          if(ptr->parent->r_child == ptr){
            ptr = ptr->parent;
          }else if(ptr->parent->l_child == ptr){
            /*Find first parent node for which
              the current lode is a left child*/
            while(ptr->parent){
              if(ptr->parent->l_child == ptr){
                ptr = ptr->parent;
              }else if(ptr->parent->r_child == ptr){
                ptr = ptr->parent;
                return;
              }
            }
            ptr = 0;
          }
        }else{
          ptr = 0;
        }
      }
    }

    friend class Tree<T>;
  };

  template<class T>
  class Tree{
  public:
    Tree(bool(*le)(const T&, const T&) = 0,
         bool(*eq)(const T&, const T&) = 0):root(0), beginNode(0),
                                            updateBegin(true),
                                            order(0), sz(0){
      l_less = le;
      l_equal = eq;
    }

    Tree(const List<T>& list):root(0), beginNode(0),
                              updateBegin(true),
                              order(0), sz(0){
      l_less = 0;
      l_equal = 0;

      Iterator<T> lit = list.begin();
      while(lit != list.end()){
        insert(*lit++);
      }
    }

    ~Tree(){
      if(root){
        delete root;
      }
    }

    static const int undefinedIndex = -1;

    /*Implicitly advances the iterator to the next node*/
    void remove(TreeIterator<T>& it){
      TreeNode<T>* node = it.ptr;

      if(node->l_child != 0 &&
         node->r_child != 0){
        /*Node will be reused and will contain the next logical value,
          no need to update the iterator*/
      }else{
        /*Node will be deleted, advance iterator to next*/
        it.next();
      }

      deleteNode2(node);
    }

    /*Removes the node having value t.*/
    void remove(T& t){
      TreeNode<T>* node = findNode(t);
      if(node == 0){
        return;
      }
      deleteNode2(node);
    }

    int size()const{
      return sz;
    }

    int getDepth(){
      return getDepth(root);
    }

    void getBalance(TreeNode<T>* node)const{
      if(node){
        message("node = %p, depth = %d, balance = %d",
                node, node->depth, node->balance);

        getBalance(node->l_child);
        getBalance(node->r_child);
      }
    }

    int getBalance()const{
      if(root){
        getBalance(root);
        return root->balance;
      }
      return 0;
    }

    void uniqueInsert(T& t, int index = 0){
      int findex = findIndex(t);

      if(findex == undefinedIndex){
        insert(t, index);
      }
    }

    void insertc(T t, int index = 0){
      T tt = t;
      insert(tt, index);
    }

    /*AVL insert*/
    void insert(T& t, int index = 0){
      sz++;
      if(beginNode && (updateBegin == false)){
        /*beginNode is valid*/
        if(l_less){
          if(l_less(t, beginNode->data)){
            /*New value is smaller than begin, update beginnode*/
            updateBegin = true;
          }
        }else{
          if(Compare<T>::less(t, beginNode->data)){
            /*New value is smaller than begin, update beginnode*/
            updateBegin = true;
          }
        }
      }

      if(root == 0){
        root = new TreeNode<T>(t, index);
        updateDepth(root, false);
        return;
      }

      TreeNode<T>* curr = root;
      if(l_less){
        while(curr){
          if(l_less(t, curr->data)){
            /*Traverse left*/
            if(curr->l_child){
              curr = curr->l_child;
            }else{
              /*Left child does not exist, create*/
              curr->l_child = new TreeNode<T>(t, index);
              curr->l_child->parent = curr;
              updateDepth(curr->l_child, false);
              return;
            }
          }else{
            /*Traverse right*/
            if(curr->r_child){
              curr = curr->r_child;
            }else{
              curr->r_child = new TreeNode<T>(t, index);
              curr->r_child->parent = curr;
              updateDepth(curr->r_child, false);
              return;
            }
          }
        }
      }else{
        while(curr){
          if(Compare<T>::less(t, curr->data)){
            /*Traverse left*/
            if(curr->l_child){
              curr = curr->l_child;
            }else{
              /*Left child does not exist, create*/
              curr->l_child = new TreeNode<T>(t, index);
              curr->l_child->parent = curr;
              updateDepth(curr->l_child, false);
              return;
            }
          }else{
            /*Traverse right*/
            if(curr->r_child){
              curr = curr->r_child;
            }else{
              curr->r_child = new TreeNode<T>(t, index);
              curr->r_child->parent = curr;
              updateDepth(curr->r_child, false);
              return;
            }
          }
        }
      }
    }

    TreeIterator<T> begin(){
      if(updateBegin){
        /*Find left most item*/
        TreeNode<T>* curr = root;
        if(curr == 0){
          return end();
        }

        while(curr->l_child){
          curr = curr->l_child;
        }
        /*Current node is left most*/
        TreeIterator<T> it(curr);

        //updateBegin = false;

        beginNode = curr;
        return it;
      }else{
        return TreeIterator<T>(beginNode);
      }
    }

    TreeIterator<T> end(){
      TreeIterator<T> it(0);
      return it;
    }


    const TreeIterator<T> begin()const{
      if(updateBegin){
        /*Find left most item*/
        TreeNode<T>* curr = root;
        if(curr == 0){
          return end();
        }

        while(curr->l_child){
          curr = curr->l_child;
        }
        /*Current node is left most*/
        beginNode = curr;
        //updateBegin = false;
        const TreeIterator<T> it(curr);
        return it;
      }else{
        const TreeIterator<T> it(beginNode);
        return it;
      }
    }

    const TreeIterator<T> end()const{
      const TreeIterator<T> it(0);
      return it;
    }

    TreeIterator<T> find(const T& t)const{
      TreeNode<T>* f = findNode(t);
      if(f==0){
        return end();
      }

      return TreeIterator<T>(f);
    }

    int findIndex(const T& t)const{
      TreeNode<T>* f = findNode(t);
      if(f == 0)
        return undefinedIndex;
      else
        return f->index;
    }

    void traverse(){
      ltraverse(root);
    }

    void clear(){
      if(root){
        delete root;
        root = 0;
      }
      sz = 0;
    }

    void consistent(){
      message("Consistency check");
      consistent(root, 0);
      message("End consistency check");
    }

    template<class U> friend inline std::ostream& operator<<(std::ostream& os,
                                                             const Tree<U>& t);

  public:
    Tree(const Tree<T>& t){
      *this = t;
    }

    Tree<T>& operator=(const Tree<T>& t){
      if(this != & t){
        TreeIterator<T> it = t.begin();
        while(it != t.end()){
          T val = it.ptr->data;
          int index = it.ptr->index;
          insert(val, index);
        }
      }
      return *this;
    }

  protected:
    /*User defined compare functions*/
    bool(*l_less )(const T&, const T&);
    bool(*l_equal)(const T&, const T&);

    int getDepth(TreeNode<T>* node){
      return node->depth;
    }

    void consistent(TreeNode<T>* node, TreeNode<T>* parent){
      if(node != 0){
        cgfassert(node->parent == parent);
        if(node->parent != parent){
          error("inconsistent tree");
        }
        consistent(node->l_child, node);
        consistent(node->r_child, node);
      }
    }


#if 0
    void computeDepth(TreeNode<T>* node, int depth = 0){
      if(node){
        computeDepth(node->l_child, depth+1);
        computeDepth(node->r_child, depth+1);
      }

      if(node){
        node->computeDepth();
        message("node level = %d, depth = %d, balance = %d",
                depth, node->depth, node->balance);
        if(node->l_child){
          message("left depth = %d", node->l_child->depth);
        }
        if(node->r_child){
          message("right depth = %d", node->r_child->depth);
        }
      }
    }
#endif

    /*If cont == false, we can skip the traversal to the
      root. Otherwise, we must continue the traversal.*/
    void updateDepth(TreeNode<T>* node, bool deleting, bool cont = false){
      node->computeDepth();

      if(deleting && node->balance == 0){
        cont = true;
      }

      cgfassert(node->balance >= -2 && node->balance <= 2);

      if(!cont){
        /*We can stop the traversal*/
        if(!deleting){
          if(node->l_child != 0 || node->r_child != 0){
            if(node->balance == 0){
              return;
            }
          }
        }else{
          if(node->l_child != 0 && node->r_child != 0){
            if(node->balance == 1 || node->balance == -1){
              return;
            }
          }
        }
      }

      if(node->balance == 2){
        /*Left subtree is deeper*/
        TreeNode<T>* P = node;
        TreeNode<T>* L = node->l_child;

        if(L->balance == -1){
          /*Left-right case*/
          rotL(L);
          rotR(P);
        }else if(L->balance == 1){
          /*Left-left case*/
          rotR(P);
        }else{
          if(deleting){
            rotR(P);
          }else{
            error();
          }
        }
        /*Update parent*/
        updateDepth(P, deleting, cont);
      }else if(node->balance == -2){
        /*Right subtree is deeper*/
        TreeNode<T>* P = node;
        TreeNode<T>* R = node->r_child;

        if(R->balance == -1){
          /*Right-tight case*/
          rotL(P);
        }else if(R->balance == 1){
          /*Left-right case*/
          rotR(R);
          rotL(P);
        }else{
          if(deleting){
            rotL(P);
          }else{
            error();
          }
        }
        /*Update parent*/
        updateDepth(P, deleting, cont);
      }else if(node->parent){
        /*Update parent*/
        updateDepth(node->parent, deleting, cont);
      }
    }

    TreeNode<T>* rotL(TreeNode<T>* node){
      TreeNode<T>* right = node->r_child;
      TreeNode<T>* parent = node->parent;
      TreeNode<T>* tmp;
      int dir = 0;
      if(parent){
        if(parent->l_child == node){
          dir = -1;
        }else{
          dir = 1;
        }
      }

      tmp = node->r_child->l_child;
      right->l_child = node;
      node->r_child = tmp;
      if(tmp){
        tmp->parent = node;
      }

      node->parent = right;
      right->parent = parent;

      /*Update parent*/
      if(dir == 0){
        /*Root*/
        root = right;
      }else if(dir == -1){
        parent->l_child = right;
      }else{
        parent->r_child = right;
      }

      /*Update depth/balance info*/
      if(node)node->computeDepth();
      if(right)right->computeDepth();
      if(parent)parent->computeDepth();

      return right;
    }

    TreeNode<T>* rotR(TreeNode<T>* node){
      TreeNode<T>* left  = node->l_child;
      TreeNode<T>* parent = node->parent;
      TreeNode<T>* tmp;
      int dir = 0;
      if(parent){
        if(parent->l_child == node){
          dir = -1;
        }else{
          dir = 1;
        }
      }

      tmp = node->l_child->r_child;
      left->r_child = node;
      node->l_child = tmp;
      if(tmp){
        tmp->parent = node;
      }

      node->parent = left;
      left->parent = parent;

      /*Update parent*/
      if(dir == 0){
        /*Root*/
        root = left;
      }else if(dir == -1){
        parent->l_child = left;
      }else{
        parent->r_child = left;
      }

      /*Update depth/balance info*/
      if(node)node->computeDepth();
      if(left)left->computeDepth();
      if(parent)parent->computeDepth();

      return left;
    }

    void deleteNode(TreeNode<T>* node){
      if(node == beginNode){
        updateBegin = true;
      }

      if(node->l_child == 0 && node->r_child == 0){
        if(node == root){
          root = 0;
        }else{
          /*Remove this node*/
          TreeNode<T>* parent = node->parent;
          if(parent->l_child == node){
            parent->l_child = 0;
          }
          if(parent->r_child == node){
            parent->r_child = 0;
          }
        }
        delete node;
        return;
      }else if(node->l_child == 0 && node->r_child != 0){
        /*Left child is NULL. Replace current node with right child.*/
        TreeNode<T>* parent = node->parent;
        if(parent != 0){
          if(parent->l_child == node){
            parent->l_child = node->r_child;
            node->r_child->parent = parent;
          }else if(parent->r_child == node){
            parent->r_child = node->r_child;
            node->r_child->parent = parent;
          }
        }else{
          /*Node == Root*/
          root = node->r_child;
          root->parent = 0;
        }
        node->r_child = 0;

        delete node;

        return;
      }else if(node->l_child != 0 && node->r_child == 0){
        /*Right child is NULL. Replace current node with left child.*/
        TreeNode<T>* parent = node->parent;
        if(parent != 0){
          if(parent->l_child == node){
            parent->l_child = node->l_child;
            node->l_child->parent = parent;
          }else if(parent->r_child == node){
            parent->r_child = node->l_child;
            node->l_child->parent = parent;
          }
        }else{
          /*Node == Root*/
          root = node->l_child;
          root->parent = 0;
        }
        node->l_child = 0;

        order = 1 - order;

        delete node;

        return;
      }
    }

    void deleteNode2(TreeNode<T>* node){
      TreeNode<T>* parent = node->parent;

      sz--;

      if(node->l_child == 0 && node->r_child == 0){
        deleteNode(node);
        if(parent)updateDepth(parent, true);
      }else if(node->l_child == 0 && node->r_child != 0){
        deleteNode(node);
        if(parent)updateDepth(parent, true);
      }else if(node->l_child != 0 && node->r_child == 0){
        deleteNode(node);
        if(parent)updateDepth(parent, true);
      }else{
        cgfassert(node->l_child != 0 && node->r_child != 0);
        /*Both childs are not NULL*/
        TreeNode<T>* currentNode=0;

        if(order == 0){
          /*Find current node in-order successor*/
          currentNode = node->r_child;
          while(currentNode->l_child){
            currentNode = currentNode->l_child;
          }

          cgfassert(currentNode->l_child == 0);
        }else{
          /*Find current node in-order predecessor*/
          currentNode = node->l_child;
          while(currentNode->r_child){
            currentNode = currentNode->r_child;
          }

          cgfassert(currentNode->r_child == 0);
        }

        node->data = currentNode->data;
        node->index = currentNode->index;

        parent = currentNode->parent;

        deleteNode(currentNode);

        if(parent){
          updateDepth(parent, true);
        }
        //return;
      }
    }

    TreeNode<T>* findNode(const T& t)const{
      TreeNode<T>* curr = root;

      if(l_equal && l_less){
        while(curr){
          if(l_equal(t, curr->data)){
            return curr;
          }else{
            if(l_less(t, curr->data)){
              if(curr->l_child){
                curr = curr->l_child;
              }else{
                return 0;
              }
            }else{
              if(curr->r_child){
                curr = curr->r_child;
              }else{
                return 0;
              }
            }
          }
        }
      }else{
        while(curr){
          if(Compare<T>::equal(t, curr->data)){
            return curr;
          }else{
            if(Compare<T>::less(t, curr->data)){
              if(curr->l_child){
                curr = curr->l_child;
              }else{
                return 0;
              }
            }else{
              if(curr->r_child){
                curr = curr->r_child;
              }else{
                return 0;
              }
            }
          }
        }
      }
      return 0;
    }

    void ltraverse(TreeNode<T>* subtree){
      if(subtree == 0){
        return;
      }

      ltraverse(subtree->l_child);
      message("[%d], %d || t = %p, p = %p, l = %p, r = %p, depth = %d, bal = %d",
              subtree->index, subtree->data, subtree,
              subtree->parent, subtree->l_child, subtree->r_child,
              subtree->depth, subtree->balance);
      ltraverse(subtree->r_child);
    }

    TreeNode<T>* root;
    mutable TreeNode<T>* beginNode;
    bool updateBegin;
#if 0
    TreeNode<T>* index;
    int last_direction;
#endif
    int order;
    int sz;

    friend class TreeReduce<T>;
  };

#if 0
  template<class U>
  inline void getSearchTree(Tree<U>& t, List<U>& list){
    Iterator<U> it = list.begin();
    while(it != list.end()){
      t.insert(*it++,1);
    }
  }
#endif

  template<class U>
  inline std::ostream& operator<<(std::ostream& os, const Tree<U>& t){
    TreeIterator<U> it = t.begin();
    while(it != t.end()){
      os << *it++ << ',';
    }
    return os;
  }

  template<class T>
  class SumReduce{
  public:
    static void reduction(T* r, T* a, T* b, T* c){
      if(b!=0 && c!=0){
        *r = (*a + *b) + *c;
      }else if(b==0 && c!=0){
        *r = *a + *c;
      }else if(c==0 && b!=0){
        *r = *a + *b;
      }else{
        *r = *a;
      }
    }
  };

  template<class T>
  class TreeReduce{
  public:
    TreeReduce(const Tree<T>* _tree,
               void (*_function)(T*, T*, T*, T*)):tree(_tree),
                                                  function(_function){
    }

    T reduce(){
      T result;
      memset(&result, 0, sizeof(T));
      reduce(&result, tree->root);

      return result;
    }
  protected:
    void reduce(T* a, TreeNode<T>* node){
      T b;
      T c;

      memset(&b, 0, sizeof(T));
      memset(&c, 0, sizeof(T));

      if(node == 0){
        *a = b;
      }else{
        reduce(&b, node->l_child);
        reduce(&c, node->r_child);
        T* pb = &b;
        T* pc = &c;
        if(node->l_child == 0){
          pb = 0;
        }
        if(node->r_child == 0){
          pc = 0;
        }
        (*function)(a, &node->data, pb, pc);
      }
    }

    const Tree<T>* tree;
    void (*function)(T*, T*, T*, T*);
  };
}

#endif/*TREE_HPP*/
