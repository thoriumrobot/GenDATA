// Test case for Issue 166:
// https://github.com/kelloggm/checker-framework/issues/166

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;

    @Positive
public class Index166 {

    @Positive
  public void testMethodInvocation() {
    @Positive
    requiresIndex("012345", 5);
    // :: error: (argument)
    @Positive
    requiresIndex("012345", 6);
    @Positive
  }

    @Positive
  public void requiresIndex(String str, @IndexFor("#1") int index) {}
    @Positive
}

// CFWR semantic augmentation - variant 1
