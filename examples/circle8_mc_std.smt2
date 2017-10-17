(set-logic QF_NRA)

(define-fun max ((x Real) (y Real)) Real (ite (>= x y) x y))

; Training seed = 3912243200

; NN variables

(declare-fun y_0 () Real)
(declare-fun y_1 () Real)
(declare-fun var_1_0 () Real)
(declare-fun var_1_1 () Real)
(declare-fun var_1_2 () Real)
(declare-fun var_1_3 () Real)
(declare-fun var_1_4 () Real)
(declare-fun var_1_5 () Real)
(declare-fun var_1_6 () Real)
(declare-fun var_1_7 () Real)
(declare-fun var_2_0 () Real)
(declare-fun var_2_1 () Real)
(declare-fun var_2_2 () Real)
(declare-fun var_2_3 () Real)
(declare-fun var_2_4 () Real)
(declare-fun var_2_5 () Real)
(declare-fun var_2_6 () Real)
(declare-fun var_2_7 () Real)
(declare-fun x_0 () Real)
(declare-fun x_1 () Real)

; NN assertions

(assert (= var_2_0 (+ -0.1973019540309906 (* -0.1587744504213333 x_0) (* -1.7523682117462158 x_1))))
(assert (= var_2_1 (+ 1.1218498945236206 (* 0.941836416721344 x_0) (* -0.2134438008069992 x_1))))
(assert (= var_2_2 (+ -0.4321659207344055 (* -0.38892117142677307 x_0) (* 2.5521466732025146 x_1))))
(assert (= var_2_3 (+ 1.4073632955551147 (* 1.0150442123413086 x_0) (* -0.14200188219547272 x_1))))
(assert (= var_2_4 (+ -0.18212860822677612 (* 1.941652536392212 x_0) (* -0.07564631849527359 x_1))))
(assert (= var_2_5 (+ -0.1788042187690735 (* 1.9927451610565186 x_0) (* -0.08603068441152573 x_1))))
(assert (= var_2_6 (+ -0.8070051074028015 (* 0.9721934199333191 x_0) (* 0.9846165180206299 x_1))))
(assert (= var_2_7 (+ -0.23280419409275055 (* 0.026348650455474854 x_0) (* -1.5873267650604248 x_1))))
(assert (= var_1_0 (max 0.0 var_2_0)))
(assert (= var_1_1 (max 0.0 var_2_1)))
(assert (= var_1_2 (max 0.0 var_2_2)))
(assert (= var_1_3 (max 0.0 var_2_3)))
(assert (= var_1_4 (max 0.0 var_2_4)))
(assert (= var_1_5 (max 0.0 var_2_5)))
(assert (= var_1_6 (max 0.0 var_2_6)))
(assert (= var_1_7 (max 0.0 var_2_7)))
(assert (= y_0 (+ -0.741036057472229 (* 1.6723042726516724 var_1_0) (* -1.6030690670013428 var_1_1) (* 1.105102777481079 var_1_2) (* -1.2502785921096802 var_1_3) (* 1.519044041633606 var_1_4) (* 1.9651744365692139 var_1_5) (* 1.5097137689590454 var_1_6) (* 1.5603405237197876 var_1_7))))
(assert (= y_1 (+ 0.741036057472229 (* -1.510440468788147 var_1_0) (* 0.9256947040557861 var_1_1) (* -1.565413236618042 var_1_2) (* 2.162538766860962 var_1_3) (* -2.0473792552948 var_1_4) (* -1.8674323558807373 var_1_5) (* -1.3817808628082275 var_1_6) (* -0.631747841835022 var_1_7))))

; Radius test - variables

(declare-fun sq_radius () Real)

; Radius test - assertions

(assert (= sq_radius (+ (* x_0 x_0) (* x_1 x_1))))

(assert (and (<= -1 x_0)   ; restrict input to box
             (<= x_0 1)
             (<= -1 x_1)
             (<= x_1 1)))

; Find all misclassified points
(assert (or
    (and (> y_0 y_1)  ; classified as outside
        (< sq_radius 0.918))   ; 0.919 -> sat
    (and (< y_0 y_1)  ; classified as inside
        (> sq_radius 1.244)))) ; 1.243 -> sat

(check-sat)
(exit)
