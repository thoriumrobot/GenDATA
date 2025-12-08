/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, numeric_literal

import org.checkerframework.checker.index.qual.NonNegative;

public class ErrorMessageCheck {

    void method3(@NonNegative int size, @NonNegative int value) {
        this.size = size;
        this.vDown = new int[this.size];
        vDown[value + 1] = 10;
    }
}
