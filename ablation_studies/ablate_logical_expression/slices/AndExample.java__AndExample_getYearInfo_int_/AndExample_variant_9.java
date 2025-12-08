/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_guard_reversal, attempted_mathematical_expression

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;

public class AndExample {

    private String getYearInfo(int year) {
        return iYearInfoCache[year & CACHE_MASK];
    }
}
