/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  private void m(BitSet b) {
    @Positive
    b.set(b.nextClearBit(0));
    // next set bit does not have to exist
    // :: error: (argument)
    @Positive
    b.clear(b.nextSetBit(0));
    @Positive
    @GTENegativeOne int i = b.nextSetBit(0);

    @Positive
    @GTENegativeOne int j = b.previousClearBit(-1);
    @Positive
    @GTENegativeOne int k = b.previousSetBit(-1);
    @Positive
  }
