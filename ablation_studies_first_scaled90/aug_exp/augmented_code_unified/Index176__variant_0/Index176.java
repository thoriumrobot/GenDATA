// Test case for Issue 176:
// https://github.com/kelloggm/checker-framework/issues/176

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;

    @Positive
public class Index176 {
    @Positive
  void test(String arglist, @IndexFor("#1") int pos) {
    @Positive
    int semi_pos = arglist.indexOf(";");
    @Positive
    if (semi_pos == -1) {
    @Positive
      throw new Error("Malformed arglist: " + arglist);
    @Positive
    }
    @Positive
    arglist.substring(pos, semi_pos + 1);
    // :: error: (argument)
    @Positive
    arglist.substring(pos, semi_pos + 2);
    @Positive
  }
    @Positive
}
