// Source-based slice around line 36
// Method: com.google.common.io.PatternFilenameFilter.pattern

 * and immutable.
 *
 * @author Apple Chow
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
public final class PatternFilenameFilter implements FilenameFilter {

  private final Pattern pattern;

  /**
   * Constructs a pattern file name filter object.
   *
   * @param patternStr the pattern string on which to filter file names
   * @throws PatternSyntaxException if pattern compilation fails (runtime)
   */
  public PatternFilenameFilter(String patternStr) {
    this(Pattern.compile(patternStr));
  }
