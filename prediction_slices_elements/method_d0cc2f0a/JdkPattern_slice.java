// Source-based slice around line 43
// Method: <com.google.common.base.JdkPattern: int flags()>

    return new JdkMatcher(pattern.matcher(t));
  }

  @Override
  public String pattern() {
    return pattern.pattern();
  }

  @Override
  public int flags() {
    return pattern.flags();
  }

  @Override
  public String toString() {
    return pattern.toString();
  }

  private static final class JdkMatcher extends CommonMatcher {
    final Matcher matcher;
