    @Positive
import org.checkerframework.checker.index.qual.SameLen;
    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
public class PrivateSameLen {

    @Positive
  private @SameLen("#1") String getSameLenString(String in) {
    @Positive
    return in;
    @Positive
  }

    @Positive
  private void test() {
    @Positive
    String in = "foo";
    @Positive
    @SameLen("this.getSameLenString(in)") String myStr = getSameLenString(in);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
