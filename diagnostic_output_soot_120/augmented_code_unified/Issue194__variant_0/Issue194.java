// Test case for kelloggm 194
// https://github.com/kelloggm/checker-framework/issues/194

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LengthOf;
    @Positive
import org.checkerframework.checker.index.qual.SameLen;

    @Positive
public class Issue194 {
    @Positive
  class Custom {
    @Positive
    public @LengthOf("this") int length() {
    @Positive
      throw new RuntimeException();
    @Positive
    }

    @Positive
    public Object get(@IndexFor("this") int i) {
    @Positive
      return null;
    @Positive
    }

    @Positive
    void call() {
    @Positive
      length();
    @Positive
    }
    @Positive
  }

    @Positive
  public boolean m(Custom a, Custom b) {
    @Positive
    if (a.length() != b.length()) {
    @Positive
      return false;
    @Positive
    }
    @Positive
    for (int i = 0; i < a.length(); ++i) {
    @Positive
      if (a.get(i) != b.get(i)) {
    @Positive
        return false;
    @Positive
      }
    @Positive
    }
    @Positive
    return true;
    @Positive
  }

    @Positive
  public void m2(Custom a, Custom b) {
    @Positive
    if (a.length() != b.length()) {
    @Positive
      return;
    @Positive
    }
    @Positive
  }
    @Positive
}
