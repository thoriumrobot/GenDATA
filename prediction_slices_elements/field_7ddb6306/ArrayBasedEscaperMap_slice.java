// Source-based slice around line 82
// Method: com.google.common.escape.ArrayBasedEscaperMap.EMPTY_REPLACEMENT_ARRAY

    char[][] replacements = new char[max + 1][];
    for (Character c : map.keySet()) {
      replacements[c] = map.get(c).toCharArray();
    }
    return replacements;
  }

  // Immutable empty array for when there are no replacements.
  @SuppressWarnings("ConstantCaseForConstants") // An empty array is a constant.
  private static final char[][] EMPTY_REPLACEMENT_ARRAY = new char[0][0];
}
