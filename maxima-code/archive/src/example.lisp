;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; This version of $example implements the following
;;; changes/improvements in the original version of example: 1)
;;; It handles %TH(2) correctly; 2) It makes effort to protect
;;; user-defined functions, variables, labels and arrays from
;;; being overwritten by an example; while protecting variables
;;; is quite straightforward, protecting functions is quite
;;; involved; it is done by moving the value of the property
;;; 'mprops' in a symbol property list to a property with a name
;;; generated by gensym; this happens before the examples are
;;; evaluated; afterwards the value of the property 'mprops' is
;;; restored; 3) rules and letrules are not being protected; it
;;; would be more complicated to make this feature work sanely;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(in-package "MAXIMA")
(defmspec $example (item &optional (file
				 (merge-pathnames "manual.demo"
						  $describe_documentation))
		      &aux tmp-name
		      )
  (and (symbolp file) (setq file (stripdollar file)))
  (or (probe-file file)
      (return-from $example "Please supply a file name as the second arg"))
  (and (symbolp item) (setq item (symbol-name item))
       (setq item (subseq item 1))
       (with-open-file
	(st file)
	(sloop with tem
	       while (setq tem (read-char st nil))
	       do
	       (cond ((and (eql tem #\&)
			   (eql (setq tem (read-char st nil)) #\&))
		      (cond
		       ((and (symbolp (setq tem (read st nil)))
			     (string-search item (symbol-name tem)))
			(format t "~%Examples for ~a :~%" tem)
			;; This code fulls maxima into thinking that it just
			;; started, by resetting the values of the special
			;; variables $labels and $linenum to their initial
			;; values. They will be reset just after $example
			;; is done. The d-labels will also not be disturbed
			;; by calling example.
			;;
			;; Hide the definitions of user functions.
			(setq tmp-name 
			      (hide-maxima-props
			       (mapcar #'caar (cdr $functions))))
			(unwind-protect
			    (progv
			     ;; Protect the user labels, variables and functions
			     ;; from being overwritten.
			     (append '($linenum
				       $labels
				       $values
				       $functions
				       $arrays
				       $%)
				     (cdr $labels)
				     (cdr $values)
				     (cdr $arrays))
			     (list 1
				   '((mlist simp))
				   '((mlist simp))
				   '((mlist simp))
				   '((mlist simp)))
			     ;; Run the example.
			     (sloop until
				    (or (null (setq tem (peek-char nil st nil)))
					(eql tem #\&))
				    for expr = (mread st nil)
				    do
				    (let ($display2d) (displa (third  expr)))
				    (let ((c-label (makelabel $inchar))
					  (d-label (makelabel $outchar)))
				      (set c-label (third expr))
				      (format t "<~d>==>" $linenum)
				      (displa (setq $% (meval* (third  expr))))
				      (terpri )
				      (set d-label $%)
				      (incf $linenum)
				      ))
			     ;; Clean-up time. Make all symbols used in
			     ;; the example unbound.
			     (mapc #'makunbound
				   (append
				    (cdr $labels)
				    (cdr $values)
				    (cdr $arrays))))
			  ;; Restore the defintions of functions.
			  (unhide-maxima-props
			   (mapcar #'caar (cdr $functions))
			   tmp-name))))))))))


(defun hide-maxima-props (symbols
			  &aux tmp-name)
  ;; Rename the property mprops, under which the function
  ;; definition e.t.c. is stored, to tmp-name.
  (setq tmp-name (gensym))
  (dolist (symbol symbols)
	  (putprop symbol (get symbol 'mprops) tmp-name)
	  (remprop symbol 'mprops))
  ;; Return the temporary name of the property.
  tmp-name)

(defun unhide-maxima-props (symbols tmp-name)
  ;; Undo the action of hide-maxima-props.
  (dolist (symbol symbols)
	  (putprop symbol (get symbol tmp-name) 'mprops)
	  (remprop symbol tmp-name)))
