// Source-based slice around line 52
// Method: com.google.common.escape.ArrayBasedEscaperMap.replacementArray

   *
   * @param replacements a map of characters to their escaped representations
   */
  public static ArrayBasedEscaperMap create(Map<Character, String> replacements) {
    return new ArrayBasedEscaperMap(createReplacementArray(replacements));
  }

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

