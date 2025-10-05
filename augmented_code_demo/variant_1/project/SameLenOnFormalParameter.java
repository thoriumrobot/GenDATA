// Test case for issue #2434: http://tinyurl.com/cfissue/2434

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SameLenOnFormalParameter {
    @Positive
  public void requiresSameLen1(String x1, @SameLen("#1") String y1) {}

    @Positive
  public void requiresSameLen2(@SameLen("#2") String x2, String y2) {}

    @Positive
  public void m1(@SameLen("#2") String a1, String b1) {
    @Positive
    requiresSameLen1(a1, b1);
    @Positive
  }

    @Positive
  public void m2(@SameLen("#2") String a2, String b2) {
    @Positive
    @SameLen("a2") String b22 = b2;
    @Positive
    requiresSameLen1(a2, b22);
    @Positive
  }

    @Positive
  public void m3(@SameLen("#2") String a3, String b3) {
    @Positive
    @SameLen("b3") String a2 = a3;
    @Positive
    @SameLen("a3") String b32 = b3;
    @Positive
    requiresSameLen1(a3, b32);
    @Positive
  }

    @Positive
  public void m4(@SameLen("#2") String a4, String b4) {
    @Positive
    requiresSameLen2(a4, b4);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
