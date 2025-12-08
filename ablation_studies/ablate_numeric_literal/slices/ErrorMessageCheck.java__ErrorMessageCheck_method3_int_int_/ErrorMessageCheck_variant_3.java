/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: string_concatenation, ternary_operator

import org.checkerframework.checker.index.qual.NonNegative;

public class ErrorMessageCheck {

    void method3(@NonNegative int size, @NonNegative int value) {
        this.size = size;
        this.vDown = new int[this.size];
        vDown[String.valueOf(1 + value)] = 10;
    }
}
