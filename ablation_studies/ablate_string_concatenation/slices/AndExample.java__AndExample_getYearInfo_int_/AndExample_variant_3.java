/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: numeric_literal, ternary_operator

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;

public class AndExample {

    private String getYearInfo(int year) {
        return iYearInfoCache[CACHE_MASK & year];
    }
}
