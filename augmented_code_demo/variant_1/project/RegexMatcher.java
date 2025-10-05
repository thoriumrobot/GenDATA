// Test case for Issue panacekcz#8:
// https://github.com/panacekcz/checker-framework/issues/8

    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class RegexMatcher {
    @Positive
  static void m(String p, String s) {
    @Positive
    Matcher matcher = Pattern.compile(p).matcher(s);
    // The following line cannot be used as a test, because the relation of matcher to p is not
    // tracked, so the upper bound is not known.

    // s.substring(matcher.start(), matcher.end());

    @Positive
    @NonNegative int i;
    @Positive
    i = matcher.start();
    @Positive
    i = matcher.end();
    // :: error: (assignment)
    @Positive
    i = matcher.start(1);
    // :: error: (assignment)
    @Positive
    i = matcher.end(1);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
