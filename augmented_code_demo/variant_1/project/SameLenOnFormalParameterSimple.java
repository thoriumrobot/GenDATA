// Test case for issue #2434: http://tinyurl.com/cfissue/2434

    @Positive
import org.checkerframework.checker.index.qual.SameLen;

    @Positive
public class SameLenOnFormalParameterSimple {
    @Positive
  public void requiresSameLen1(String x1, @SameLen("#1") String y1) {}

    @Positive
  public void m1(@SameLen("#2") String a1, String b1) {
    @Positive
    requiresSameLen1(a1, b1);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
