  void testAppend(Appendable app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
    app.append(cs, i, i);
    // :: error: (argument)
    app.append(cs, 1, 2);
  }

  void testAppend(StringWriter app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
    app.append(cs, i, i);
    // :: error: (argument)
    app.append(cs, 1, 2);
  }
