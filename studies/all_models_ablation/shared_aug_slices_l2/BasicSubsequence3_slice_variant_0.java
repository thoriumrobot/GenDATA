/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

  int[] array;

    @Positive
  int[] array2;

    @Positive
  final @IndexFor("array") int start;

    @Positive
  final @IndexOrHigh("array") int end;

    @Positive
  private BasicSubsequence3(@IndexFor("array") int s, @IndexOrHigh("array") int e) {
    @Positive
    start = s;
    @Positive
    end = e;
    @Positive
  }

    @Positive
