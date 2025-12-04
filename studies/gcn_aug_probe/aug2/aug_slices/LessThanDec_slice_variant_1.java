/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

import org.checkerframework.checker.index.qual.LessThan;

    @Positive
public class LessThanDec {
    @Positive
  private static @IndexOrLow("#1") @LessThan("#4") int lastIndexOf(
    @Positive
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    for (int i = end - 1; i >= start; i--) {
    @Positive
      if (array[i] == target) {
    @Positive
        return i;
    @Positive
      }
    @Positive
    }
    @Positive
    return -1;
    @Positive
