// Source-based slice around line 32
// Method: com.google.common.base.SmallCharMatcher.containsZero

 * An immutable version of CharMatcher for smallish sets of characters that uses a hash table with
 * linear probing to check for matches.
 *
 * @author Christopher Swenson
 */
@GwtIncompatible // no precomputation is done in GWT
final class SmallCharMatcher extends NamedFastMatcher {
  static final int MAX_SIZE = 1023;
  private final char[] table;
  private final boolean containsZero;
  private final long filter;

  private SmallCharMatcher(char[] table, long filter, boolean containsZero, String description) {
    super(description);
    this.table = table;
    this.filter = filter;
    this.containsZero = containsZero;
  }

  private static final int C1 = 0xcc9e2d51;
