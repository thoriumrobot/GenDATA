/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void test(String arglist, @IndexFor("#1") int pos) {
    @Positive
    int semi_pos = arglist.indexOf(";");
    @Positive
    if (semi_pos == -1) {
    @Positive
      throw new Error("Malformed arglist: " + arglist);
    @Positive
    }
    @Positive
    arglist.substring(pos, semi_pos + 1);
    // :: error: (argument)
    @Positive
    arglist.substring(pos, semi_pos + 2);
    @Positive
  }
