// Source-based slice around line 38
// Method: <com.google.common.base.JdkPattern: String pattern()>

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
    return pattern.flags();
  }

  @Override
  public String toString() {
