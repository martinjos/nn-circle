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

; Test seed = 989102080

; Assertion for test data

(assert (or
 (and (= x_0  0.45286175608634949) (= x_1  0.73088616132736206)
  (or (< y_0 -0.98216591937637332) (> y_0 -0.98216391937637326)
      (< y_1  0.83952899114990231) (> y_1  0.83953099114990237)))
 (and (= x_0  0.56065165996551514) (= x_1  0.20367322862148285)
  (or (< y_0 -2.58789019448852553) (> y_0 -2.58788819448852525)
      (< y_1  2.89562888281249986) (> y_1  2.89563088281250014)))
 (and (= x_0 -0.12850362062454224) (= x_1  0.27331551909446716)
  (or (< y_0 -3.45139365060424819) (> y_0 -3.45139165060424791)
      (< y_1  3.79730029241943345) (> y_1  3.79730229241943373)))
 (and (= x_0  0.58691769838333130) (= x_1  0.38161677122116089)
  (or (< y_0 -1.88178055150604240) (> y_0 -1.88177855150604256)
      (< y_1  2.05705995695495591) (> y_1  2.05706195695495619)))
 (and (= x_0 -0.02370956353843212) (= x_1 -0.19188530743122101)
  (or (< y_0 -3.98318104608154311) (> y_0 -3.98317904608154283)
      (< y_1  4.58660931723022447) (> y_1  4.58661131723022475)))
 (and (= x_0  0.74927568435668945) (= x_1  0.37059012055397034)
  (or (< y_0 -1.09985570771789543) (> y_0 -1.09985370771789559)
      (< y_1  1.24727578298950204) (> y_1  1.24727778298950187)))
 (and (= x_0  0.41629746556282043) (= x_1  0.03005343116819859)
  (or (< y_0 -3.21874003274536147) (> y_0 -3.21873803274536119)
      (< y_1  3.59678168432617174) (> y_1  3.59678368432617201)))
 (and (= x_0  0.69624716043472290) (= x_1  0.31551167368888855)
  (or (< y_0 -1.62180356843566886) (> y_0 -1.62180156843566903)
      (< y_1  1.83713252680206307) (> y_1  1.83713452680206291)))
 (and (= x_0  0.89959955215454102) (= x_1  0.73994660377502441)
  (or (< y_0  1.34832937853240975) (> y_0  1.34833137853240959)
      (< y_1 -1.60711185796356193) (> y_1 -1.60710985796356209)))
 (and (= x_0 -0.13854703307151794) (= x_1  0.61972790956497192)
  (or (< y_0 -2.26211790902709975) (> y_0 -2.26211590902709947)
      (< y_1  2.20158548490905748) (> y_1  2.20158748490905776)))))

(check-sat)
(exit)
