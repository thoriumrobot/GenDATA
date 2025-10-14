// Test case for Issue panacekcz#16:
// https://github.com/panacekcz/checker-framework/issues/16

    @Positive
import java.util.StringTokenizer;

    @Positive
public class StringTokenizerMinLen {
    @Positive
  void test(String str, String delim, boolean returnDelims) {
    @Positive
    StringTokenizer st = new StringTokenizer(str, delim, returnDelims);
    @Positive
    while (st.hasMoreTokens()) {
    @Positive
      String token = st.nextToken();
    @Positive
      char c = token.charAt(0);
    @Positive
    }
    @Positive
  }
    @Positive
}
