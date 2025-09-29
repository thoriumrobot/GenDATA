    @NonNegative
  void testCharAt(CharSequence cs, int i, @IndexFor("#1") int j) {
    cs.charAt(j);
    cs.subSequence(j, j);
    // :: error: (argument)
    cs.charAt(i);
    // :: error: (argument)
    cs.subSequence(i, j);
  }

    @NonNegative
  void testAppend(Appendable app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
    app.append(cs, i, i);
    // :: error: (argument)
    app.append(cs, 1, 2);
  }

    @NonNegative
  void testAppend(StringWriter app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
    app.append(cs, i, i);
    // :: error: (argument)
    app.append(cs, 1, 2);
  }
