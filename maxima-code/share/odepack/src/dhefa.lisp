;;; Compiled by f2cl version:
;;; ("f2cl1.l,v 95098eb54f13 2013/04/01 00:45:16 toy $"
;;;  "f2cl2.l,v 95098eb54f13 2013/04/01 00:45:16 toy $"
;;;  "f2cl3.l,v 96616d88fb7e 2008/02/22 22:19:34 rtoy $"
;;;  "f2cl4.l,v 96616d88fb7e 2008/02/22 22:19:34 rtoy $"
;;;  "f2cl5.l,v 95098eb54f13 2013/04/01 00:45:16 toy $"
;;;  "f2cl6.l,v 1d5cbacbb977 2008/08/24 00:56:27 rtoy $"
;;;  "macros.l,v 1409c1352feb 2013/03/24 20:44:50 toy $")

;;; Using Lisp CMU Common Lisp snapshot-2013-11 (20E Unicode)
;;; 
;;; Options: ((:prune-labels nil) (:auto-save t) (:relaxed-array-decls t)
;;;           (:coerce-assigns :as-needed) (:array-type ':array)
;;;           (:array-slicing t) (:declare-common nil)
;;;           (:float-format single-float))

(in-package "ODEPACK")


(defun dhefa (a lda n ipvt info job)
  (declare (type (array f2cl-lib:integer4 (*)) ipvt)
           (type (f2cl-lib:integer4) job info n lda)
           (type (array double-float (*)) a))
  (f2cl-lib:with-multi-array-data
      ((a double-float a-%data% a-%offset%)
       (ipvt f2cl-lib:integer4 ipvt-%data% ipvt-%offset%))
    (prog ((j 0) (k 0) (km1 0) (kp1 0) (l 0) (nm1 0) (t$ 0.0d0))
      (declare (type (double-float) t$)
               (type (f2cl-lib:integer4) nm1 l kp1 km1 k j))
      (if (> job 1) (go label80))
      (setf info 0)
      (setf nm1 (f2cl-lib:int-sub n 1))
      (if (< nm1 1) (go label70))
      (f2cl-lib:fdo (k 1 (f2cl-lib:int-add k 1))
                    ((> k nm1) nil)
        (tagbody
          (setf kp1 (f2cl-lib:int-add k 1))
          (setf l
                  (f2cl-lib:int-sub
                   (f2cl-lib:int-add
                    (idamax 2
                     (f2cl-lib:array-slice a-%data%
                                           double-float
                                           (k k)
                                           ((1 lda) (1 *))
                                           a-%offset%)
                     1)
                    k)
                   1))
          (setf (f2cl-lib:fref ipvt-%data% (k) ((1 *)) ipvt-%offset%) l)
          (if
           (= (f2cl-lib:fref a-%data% (l k) ((1 lda) (1 *)) a-%offset%) 0.0d0)
           (go label40))
          (if (= l k) (go label10))
          (setf t$ (f2cl-lib:fref a-%data% (l k) ((1 lda) (1 *)) a-%offset%))
          (setf (f2cl-lib:fref a-%data% (l k) ((1 lda) (1 *)) a-%offset%)
                  (f2cl-lib:fref a-%data% (k k) ((1 lda) (1 *)) a-%offset%))
          (setf (f2cl-lib:fref a-%data% (k k) ((1 lda) (1 *)) a-%offset%) t$)
         label10
          (setf t$
                  (/ -1.0d0
                     (f2cl-lib:fref a-%data%
                                    (k k)
                                    ((1 lda) (1 *))
                                    a-%offset%)))
          (setf (f2cl-lib:fref a-%data%
                               ((f2cl-lib:int-add k 1) k)
                               ((1 lda) (1 *))
                               a-%offset%)
                  (*
                   (f2cl-lib:fref a-%data%
                                  ((f2cl-lib:int-add k 1) k)
                                  ((1 lda) (1 *))
                                  a-%offset%)
                   t$))
          (f2cl-lib:fdo (j kp1 (f2cl-lib:int-add j 1))
                        ((> j n) nil)
            (tagbody
              (setf t$
                      (f2cl-lib:fref a-%data%
                                     (l j)
                                     ((1 lda) (1 *))
                                     a-%offset%))
              (if (= l k) (go label20))
              (setf (f2cl-lib:fref a-%data% (l j) ((1 lda) (1 *)) a-%offset%)
                      (f2cl-lib:fref a-%data%
                                     (k j)
                                     ((1 lda) (1 *))
                                     a-%offset%))
              (setf (f2cl-lib:fref a-%data% (k j) ((1 lda) (1 *)) a-%offset%)
                      t$)
             label20
              (daxpy (f2cl-lib:int-sub n k) t$
               (f2cl-lib:array-slice a-%data%
                                     double-float
                                     ((+ k 1) k)
                                     ((1 lda) (1 *))
                                     a-%offset%)
               1
               (f2cl-lib:array-slice a-%data%
                                     double-float
                                     ((+ k 1) j)
                                     ((1 lda) (1 *))
                                     a-%offset%)
               1)
             label30))
          (go label50)
         label40
          (setf info k)
         label50
         label60))
     label70
      (setf (f2cl-lib:fref ipvt-%data% (n) ((1 *)) ipvt-%offset%) n)
      (if (= (f2cl-lib:fref a-%data% (n n) ((1 lda) (1 *)) a-%offset%) 0.0d0)
          (setf info n))
      (go end_label)
     label80
      (setf nm1 (f2cl-lib:int-sub n 1))
      (if (<= nm1 1) (go label105))
      (f2cl-lib:fdo (k 2 (f2cl-lib:int-add k 1))
                    ((> k nm1) nil)
        (tagbody
          (setf km1 (f2cl-lib:int-sub k 1))
          (setf l (f2cl-lib:fref ipvt-%data% (km1) ((1 *)) ipvt-%offset%))
          (setf t$ (f2cl-lib:fref a-%data% (l n) ((1 lda) (1 *)) a-%offset%))
          (if (= l km1) (go label90))
          (setf (f2cl-lib:fref a-%data% (l n) ((1 lda) (1 *)) a-%offset%)
                  (f2cl-lib:fref a-%data% (km1 n) ((1 lda) (1 *)) a-%offset%))
          (setf (f2cl-lib:fref a-%data% (km1 n) ((1 lda) (1 *)) a-%offset%) t$)
         label90
          (setf (f2cl-lib:fref a-%data% (k n) ((1 lda) (1 *)) a-%offset%)
                  (+ (f2cl-lib:fref a-%data% (k n) ((1 lda) (1 *)) a-%offset%)
                     (*
                      (f2cl-lib:fref a-%data%
                                     (k km1)
                                     ((1 lda) (1 *))
                                     a-%offset%)
                      t$)))
         label100))
     label105
      (setf info 0)
      (setf l
              (f2cl-lib:int-sub
               (f2cl-lib:int-add
                (idamax 2
                 (f2cl-lib:array-slice a-%data%
                                       double-float
                                       (nm1 nm1)
                                       ((1 lda) (1 *))
                                       a-%offset%)
                 1)
                nm1)
               1))
      (setf (f2cl-lib:fref ipvt-%data% (nm1) ((1 *)) ipvt-%offset%) l)
      (if (= (f2cl-lib:fref a-%data% (l nm1) ((1 lda) (1 *)) a-%offset%) 0.0d0)
          (go label140))
      (if (= l nm1) (go label110))
      (setf t$ (f2cl-lib:fref a-%data% (l nm1) ((1 lda) (1 *)) a-%offset%))
      (setf (f2cl-lib:fref a-%data% (l nm1) ((1 lda) (1 *)) a-%offset%)
              (f2cl-lib:fref a-%data% (nm1 nm1) ((1 lda) (1 *)) a-%offset%))
      (setf (f2cl-lib:fref a-%data% (nm1 nm1) ((1 lda) (1 *)) a-%offset%) t$)
     label110
      (setf t$
              (/ -1.0d0
                 (f2cl-lib:fref a-%data%
                                (nm1 nm1)
                                ((1 lda) (1 *))
                                a-%offset%)))
      (setf (f2cl-lib:fref a-%data% (n nm1) ((1 lda) (1 *)) a-%offset%)
              (* (f2cl-lib:fref a-%data% (n nm1) ((1 lda) (1 *)) a-%offset%)
                 t$))
      (setf t$ (f2cl-lib:fref a-%data% (l n) ((1 lda) (1 *)) a-%offset%))
      (if (= l nm1) (go label120))
      (setf (f2cl-lib:fref a-%data% (l n) ((1 lda) (1 *)) a-%offset%)
              (f2cl-lib:fref a-%data% (nm1 n) ((1 lda) (1 *)) a-%offset%))
      (setf (f2cl-lib:fref a-%data% (nm1 n) ((1 lda) (1 *)) a-%offset%) t$)
     label120
      (setf (f2cl-lib:fref a-%data% (n n) ((1 lda) (1 *)) a-%offset%)
              (+ (f2cl-lib:fref a-%data% (n n) ((1 lda) (1 *)) a-%offset%)
                 (* t$
                    (f2cl-lib:fref a-%data%
                                   (n nm1)
                                   ((1 lda) (1 *))
                                   a-%offset%))))
      (go label150)
     label140
      (setf info nm1)
     label150
      (setf (f2cl-lib:fref ipvt-%data% (n) ((1 *)) ipvt-%offset%) n)
      (if (= (f2cl-lib:fref a-%data% (n n) ((1 lda) (1 *)) a-%offset%) 0.0d0)
          (setf info n))
      (go end_label)
     end_label
      (return (values nil nil nil nil info nil)))))

(in-package #-gcl #:cl-user #+gcl "CL-USER")
#+#.(cl:if (cl:find-package '#:f2cl) '(and) '(or))
(eval-when (:load-toplevel :compile-toplevel :execute)
  (setf (gethash 'fortran-to-lisp::dhefa fortran-to-lisp::*f2cl-function-info*)
          (fortran-to-lisp::make-f2cl-finfo
           :arg-types '((array double-float (*)) (fortran-to-lisp::integer4)
                        (fortran-to-lisp::integer4)
                        (array fortran-to-lisp::integer4 (*))
                        (fortran-to-lisp::integer4)
                        (fortran-to-lisp::integer4))
           :return-values '(nil nil nil nil fortran-to-lisp::info nil)
           :calls '(fortran-to-lisp::daxpy fortran-to-lisp::idamax))))

