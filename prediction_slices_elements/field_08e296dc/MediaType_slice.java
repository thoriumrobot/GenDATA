// Source-based slice around line 78
// Method: com.google.common.net.MediaType.UTF_8_CONSTANT_PARAMETERS

 * "_UTF_8" suffix. To get a version without a character set, use {@link #withoutParameters}.
 *
 * @since 12.0
 * @author Gregory Kick
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
