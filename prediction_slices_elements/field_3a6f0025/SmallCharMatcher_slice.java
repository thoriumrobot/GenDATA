// Source-based slice around line 33
// Method: com.google.common.base.SmallCharMatcher.filter

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
  private static final int C2 = 0x1b873593;
