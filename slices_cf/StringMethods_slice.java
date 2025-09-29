    @Positive
  void testCharAt(String s, int i) {
    // ::  error: (argument)
    @Positive
    s.charAt(i);
    // ::  error: (argument)
    @Positive
    s.codePointAt(i);

    @Positive
    if (i >= 0 && i < s.length()) {
    @Positive
      s.charAt(i);
    @Positive
      s.codePointAt(i);
    @Positive
    }
    @Positive
  }

    @Positive
  void testCodePointBefore(String s) {
    // ::  error: (argument)
    @Positive
    s.codePointBefore(0);

    @Positive
    if (s.length() > 0) {
    @Positive
    @Positive
    @Positive
    @Positive
    @Positive
    @Positive
      s.codePointBefore(s.length());
    @Positive
    }
    @Positive
  }
