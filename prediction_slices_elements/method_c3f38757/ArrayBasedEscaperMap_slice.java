// Source-based slice around line 67
// Method: <com.google.common.escape.ArrayBasedEscaperMap: char[][] createReplacementArray(Map)>

  // Returns the non-null array of replacements for fast lookup.
  char[][] getReplacementArray() {
    return replacementArray;
  }

  // Creates a replacement array from the given map. The returned array is a
  // linear lookup table of replacement character sequences indexed by the
  // original character value.
  @VisibleForTesting
  static char[][] createReplacementArray(Map<Character, String> map) {
    checkNotNull(map); // GWT specific check (do not optimize)
    if (map.isEmpty()) {
      return EMPTY_REPLACEMENT_ARRAY;
    }
    char max = max(map.keySet());
    char[][] replacements = new char[max + 1][];
    for (Character c : map.keySet()) {
      replacements[c] = map.get(c).toCharArray();
    }
    return replacements;
