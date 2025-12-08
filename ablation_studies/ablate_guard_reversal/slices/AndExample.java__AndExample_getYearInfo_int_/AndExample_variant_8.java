/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: logical_expression, loop_conversion

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;

public class AndExample {

    private String getYearInfo(int year) {
        return iYearInfoCache[CACHE_MASK & year];
    }
}
