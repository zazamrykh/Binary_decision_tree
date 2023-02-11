# Binary_decision_tree
Make binary decision tree. Branching criteria - tree depth. Make split using Gini formula.
Output of the program:
                        -> class : 0
                    -> 17
                        -> class : 1
                -> 17
                    -> 0
            -> 42
                        -> class : 1
                    -> 1
                        -> class : 1
                -> 25
                        -> class : 0
                    -> 24
                        -> class : 0
        -> 45
                        -> class : 1
                    -> 2
                        -> class : 1
                -> 2
                    -> 0
            -> 3
                        -> class : 1
                    -> 1
                        -> class : 1
                -> 1
                    -> 0
    -> 50
                        -> class : 1
                    -> 4
                        -> class : 1
                -> 4
                    -> 0
            -> 4
                -> 0
        -> 5
                        -> class : 1
                    -> 1
                        -> class : 1
                -> 1
                    -> 0
            -> 1
                -> 0
-> 60
                        -> class : 1
                    -> 6
                        -> class : 1
                -> 6
                    -> 0
            -> 6
                -> 0
        -> 6
            -> 0
    -> 10
                        -> class : 1
                    -> 4
                        -> class : 1
                -> 4
                    -> 0
            -> 4
                -> 0
        -> 4
            -> 0
Tree overfitted and remember all train selection
Accuracy at tree with depth = 7:	0.95
Make new tree using all selection:
Accuracy:	0.6948051948051948
Try this method on other dataset
Accuracy:	0.9818181818181818

Process finished with exit code 0
