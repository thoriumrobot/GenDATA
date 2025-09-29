
    @NonNegative
  final @IndexFor("array") int start;

    @NonNegative
  final @IndexOrHigh("array") int end;

    @NonNegative
  private BasicSubsequence2(@IndexFor("array") int s, @IndexOrHigh("array") int e) {
    start = s;
    end = e;
  }

    @NonNegative
  void testStartIndex(@IndexFor("this") int x) {
    @NonNegative
    @IndexFor("array") int y = x + start;
  }

    @NonNegative
  void testViewpointAdaption(@IndexFor("this") int x) {
