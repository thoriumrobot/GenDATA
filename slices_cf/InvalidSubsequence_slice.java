public class InvalidSubsequence {
  // :: error: (flowexpr.parse.error) :: error: (not.final)
  @HasSubsequence(subsequence = "banana", from = "this.from", to = "this.to")
  int[] a;

  // :: error: (flowexpr.parse.error) :: error: (not.final)
  @HasSubsequence(subsequence = "this", from = "banana", to = "this.to")
  int[] b;

  // :: error: (flowexpr.parse.error) :: error: (not.final)
  @HasSubsequence(subsequence = "this", from = "this.from", to = "banana")
  int[] c;

  // :: error: (not.final)
  @HasSubsequence(subsequence = "this", from = "this.from", to = "10")
  int[] e;

    @NonNegative
  @IndexFor("a") @LessThan("to") int from;

    @NonNegative
