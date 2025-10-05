// Source-based slice around line 48
// Method: <com.google.common.base.JdkPattern: String toString()>

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

    JdkMatcher(Matcher matcher) {
      this.matcher = Preconditions.checkNotNull(matcher);
    }

