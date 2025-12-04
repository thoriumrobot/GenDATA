/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void method3(@NonNegative int size, @NonNegative int value) {
    @Positive
    this.size = size;
    @Positive
    this.vDown = new int[this.size];
    // :: error: (array.access.unsafe.high)
    @Positive
    vDown[1 + value] = 10;
    @Positive
  }
