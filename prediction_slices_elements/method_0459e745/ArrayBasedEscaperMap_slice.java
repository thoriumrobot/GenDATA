// Source-based slice around line 46
// Method: <com.google.common.escape.ArrayBasedEscaperMap: ArrayBasedEscaperMap create(Map)>

 */
@GwtCompatible
public final class ArrayBasedEscaperMap {
  /**
   * Returns a new ArrayBasedEscaperMap for creating ArrayBasedCharEscaper or
   * ArrayBasedUnicodeEscaper instances.
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
