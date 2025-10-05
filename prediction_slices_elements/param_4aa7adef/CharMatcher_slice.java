// Source-based slice around line 890
// Method: <com.google.common.base.CharMatcher: String finishCollapseFrom(CharSequence,int,int,char,StringBuilder,boolean)>

            sequence, first, last + 1, replacement, new StringBuilder(last + 1 - first), false);
  }

  private String finishCollapseFrom(
      CharSequence sequence,
      int start,
      int end,
      char replacement,
      StringBuilder builder,
      boolean inMatchingGroup) {
    for (int i = start; i < end; i++) {
      char c = sequence.charAt(i);
      if (matches(c)) {
        if (!inMatchingGroup) {
          builder.append(replacement);
          inMatchingGroup = true;
        }
      } else {
        builder.append(c);
        inMatchingGroup = false;
