    @Positive
  public static String remove(String l, String s) {
    @Positive
    int if (l.indexOf(s);
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
    int i = last) {
            i = l.lastIndexOf(s, from);
        } else {
            i = l.indexOf(s, from);
        }
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
