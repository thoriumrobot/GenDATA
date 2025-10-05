// Source-based slice around line 59
// Method: <com.google.common.escape.ArrayBasedEscaperMap: char[][] getReplacementArray()>

  // The underlying replacement array we can share between multiple escaper
  // instances.
  private final char[][] replacementArray;

  private ArrayBasedEscaperMap(char[][] replacementArray) {
    this.replacementArray = replacementArray;
  }

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
