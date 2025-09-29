    @NonNegative
  void test(String arglist, @IndexFor("#1") int pos) {
    @NonNegative
    int semi_pos = arglist.indexOf(";");
    if (semi_pos == -1) {
      throw new Error("Malformed arglist: " + arglist);
    }
    arglist.substring(pos, semi_pos + 1);
    // :: error: (argument)
    arglist.substring(pos, semi_pos + 2);
  }
