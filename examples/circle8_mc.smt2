(set-logic QF_NRA)

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

(assert (= var_2_0 (+ -0.19730195403099060 (* -0.15877445042133331 x_0) (* -1.75236821174621582 x_1))))
(assert (= var_2_1 (+  1.12184989452362061 (*  0.94183641672134399 x_0) (* -0.21344380080699921 x_1))))
(assert (= var_2_2 (+ -0.43216592073440552 (* -0.38892117142677307 x_0) (*  2.55214667320251465 x_1))))
(assert (= var_2_3 (+  1.40736329555511475 (*  1.01504421234130859 x_0) (* -0.14200188219547272 x_1))))
(assert (= var_2_4 (+ -0.18212860822677612 (*  1.94165253639221191 x_0) (* -0.07564631849527359 x_1))))
(assert (= var_2_5 (+ -0.17880421876907349 (*  1.99274516105651855 x_0) (* -0.08603068441152573 x_1))))
(assert (= var_2_6 (+ -0.80700510740280151 (*  0.97219341993331909 x_0) (*  0.98461651802062988 x_1))))
(assert (= var_2_7 (+ -0.23280419409275055 (*  0.02634865045547485 x_0) (* -1.58732676506042480 x_1))))
(assert (= var_1_0 (max 0.0 var_2_0)))
(assert (= var_1_1 (max 0.0 var_2_1)))
(assert (= var_1_2 (max 0.0 var_2_2)))
(assert (= var_1_3 (max 0.0 var_2_3)))
(assert (= var_1_4 (max 0.0 var_2_4)))
(assert (= var_1_5 (max 0.0 var_2_5)))
(assert (= var_1_6 (max 0.0 var_2_6)))
(assert (= var_1_7 (max 0.0 var_2_7)))
(assert (= y_0 (+ -0.74103605747222900 (*  1.67230427265167236 var_1_0) (* -1.60306906700134277 var_1_1) (*  1.10510277748107910 var_1_2) (* -1.25027859210968018 var_1_3) (*  1.51904404163360596 var_1_4) (*  1.96517443656921387 var_1_5) (*  1.50971376895904541 var_1_6) (*  1.56034052371978760 var_1_7))))
(assert (= y_1 (+  0.74103605747222900 (* -1.51044046878814697 var_1_0) (*  0.92569470405578613 var_1_1) (* -1.56541323661804199 var_1_2) (*  2.16253876686096191 var_1_3) (* -2.04737925529479980 var_1_4) (* -1.86743235588073730 var_1_5) (* -1.38178086280822754 var_1_6) (* -0.63174784183502197 var_1_7))))

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
