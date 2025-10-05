// Source-based slice around line 79
// Method: com.google.common.net.InternetDomainName.DOTS_MATCHER

 * versions.
 *
 * @author Catherine Berry
 * @since 5.0
 */
@GwtCompatible
@Immutable
public final class InternetDomainName {

  private static final CharMatcher DOTS_MATCHER = CharMatcher.anyOf(".\u3002\uFF0E\uFF61");
  private static final Splitter DOT_SPLITTER = Splitter.on('.');
  private static final Joiner DOT_JOINER = Joiner.on('.');

  /**
   * Value of {@link #publicSuffixIndex()} or {@link #registrySuffixIndex()} which indicates that no
   * relevant suffix was found.
   */
  private static final int NO_SUFFIX_FOUND = -1;

  /**
