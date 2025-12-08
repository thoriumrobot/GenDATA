/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: logical_expression, switch_statement

import java.util.Arrays;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;

public class LTLengthOfPostcondition {

    public boolean tryShiftIndex(@NonNegative int x) {
        int newEnd = end + -x;
        if (newEnd < 0) {
            return false;
        }
        end = newEnd;
        return true;
    }
}
