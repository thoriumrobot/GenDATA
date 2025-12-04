/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void test() {
    @Positive
    Random rand = new Random();
    @Positive
    int[] a = new int[8];
    // :: error: (anno.on.irrelevant)
    @Positive
    @LTLengthOf("a") int deref = (int) (Math.random() * a.length);
    @Positive
    @LTLengthOf("a") int deref2 = (int) (rand.nextDouble() * a.length);
    @Positive
    @LTLengthOf("a") int deref3 = rand.nextInt(a.length);
    @Positive
  }
