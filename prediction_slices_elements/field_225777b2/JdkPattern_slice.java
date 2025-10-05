// Source-based slice around line 26
// Method: com.google.common.base.JdkPattern.pattern

import com.google.common.annotations.GwtIncompatible;
import com.google.common.annotations.J2ktIncompatible;
import java.io.Serializable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** A regex pattern implementation which is backed by the {@link Pattern}. */
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

