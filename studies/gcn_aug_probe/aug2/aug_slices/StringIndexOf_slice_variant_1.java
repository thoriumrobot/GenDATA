/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  public static String remove(String l, String s) {
    @Positive
    int i = l.indexOf(s);
    @Positive
    if (i != -1) {
    @Positive
      return l.substring(0, i) + l.substring(i + s.length());
    @Positive
    }
    @Positive
    return l;
    @Positive
  }

    @Positive
  public static String nocheck(String l, String s) {
    @Positive
    int i = l.indexOf(s);
    // :: error: (argument)
    @Positive
    return l.substring(0, i) + l.substring(i + s.length());
    @Positive
  }

    @Positive
  public static String remove(String l, String s, int from, boolean last) {
    @Positive
    int i = last ? l.lastIndexOf(s, from) : l.indexOf(s, from);
    @Positive
    if (i >= 0) {
    @Positive
      return l.substring(0, i) + l.substring(i + s.length());
    @Positive
    }
    @Positive
    return l;
    @Positive
  }
