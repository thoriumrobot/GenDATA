/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

import org.checkerframework.checker.index.qual.*;

    @Positive
public class SizeVsLength {

    @Positive
  public int[] getArray(@NonNegative int size) {
    @Positive
    int[] values = new int[size];
    @Positive
    for (int i = 0; i < size; i++) {
    @Positive
      values[i] = 22;
    @Positive
    }
    @Positive
    return values;
    @Positive
  }
    @Positive
}
