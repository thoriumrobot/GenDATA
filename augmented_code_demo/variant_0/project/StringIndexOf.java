// Tests using the index returned from String.indexOf

    @Positive
public class StringIndexOf {

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

    @Positive
  public static String stringLiteral(String l) {
    @Positive
    int i = l.indexOf("constant");
    @Positive
    if (i != -1) {
    @Positive
      return l.substring(0, i) + l.substring(i + "constant".length());
    @Positive
    }
    // :: error: (argument)
    @Positive
    return l.substring(0, i) + l.substring(i + "constant".length());
    @Positive
  }

    @Positive
  public static char character(String l, char c) {
    @Positive
    int i = l.indexOf(c);
    @Positive
    if (i > -1) {
    @Positive
      return l.charAt(i);
    @Positive
    }
    // :: error: (argument)
    @Positive
    return l.charAt(i);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
