
    @NonNegative
  final @IndexFor("array") int start;

    @NonNegative
  final @IndexOrHigh("array") int end;

    @NonNegative
  private BasicSubsequence3(@IndexFor("array") int s, @IndexOrHigh("array") int e) {
    start = s;
    end = e;
  }

    @NonNegative
  void testStartIndex(@IndexFor("array") @LessThan("this.end") int x) {
    @NonNegative
    @IndexFor("this") int y = x - start;
  }

    @NonNegative
  void testViewpointAdaption(@IndexFor("array2") @LessThan("end") int x) {
