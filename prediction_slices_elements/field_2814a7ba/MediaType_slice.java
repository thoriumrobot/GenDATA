// Source-based slice around line 82
// Method: com.google.common.net.MediaType.TOKEN_MATCHER

 */
@GwtCompatible
@Immutable
public final class MediaType {
  private static final String CHARSET_ATTRIBUTE = "charset";
  private static final ImmutableListMultimap<String, String> UTF_8_CONSTANT_PARAMETERS =
      ImmutableListMultimap.of(CHARSET_ATTRIBUTE, Ascii.toLowerCase(UTF_8.name()));

  /** Matcher for type, subtype and attributes. */
  private static final CharMatcher TOKEN_MATCHER =
      ascii()
          .and(javaIsoControl().negate())
          .and(CharMatcher.isNot(' '))
          .and(CharMatcher.noneOf("()<>@,;:\\\"/[]?="));

  private static final CharMatcher QUOTED_TEXT_MATCHER = ascii().and(CharMatcher.noneOf("\"\\\r"));

  /*
   * This matches the same characters as linear-white-space from RFC 822, but we make no effort to
   * enforce any particular rules with regards to line folding as stated in the class docs.
