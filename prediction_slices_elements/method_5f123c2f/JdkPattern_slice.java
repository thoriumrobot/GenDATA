// Source-based slice around line 33
// Method: <com.google.common.base.JdkPattern: CommonMatcher matcher(CharSequence)>

@GwtIncompatible
final class JdkPattern extends CommonPattern implements Serializable {
  private final Pattern pattern;

  JdkPattern(Pattern pattern) {
    this.pattern = Preconditions.checkNotNull(pattern);
  }

  @Override
  public CommonMatcher matcher(CharSequence t) {
    return new JdkMatcher(pattern.matcher(t));
  }

  @Override
  public String pattern() {
    return pattern.pattern();
  }

  @Override
  public int flags() {
