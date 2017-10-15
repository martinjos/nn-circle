; Radius test - variables

(declare-fun sq_radius () Real)

; Radius test - assertions

(assert (= sq_radius (+ (^ x_0 2) (^ x_1 2))))

(assert (and (<= -1 x_0)   ; restrict input to box
             (<= x_0 1)
             (<= -1 x_1)
             (<= x_1 1)))

; Find all misclassified points
(assert (or
    (and (> y_0 y_1)  ; classified as outside
        (< sq_radius 0.9))
    (and (< y_0 y_1)  ; classified as inside
        (> sq_radius 1.2))))
