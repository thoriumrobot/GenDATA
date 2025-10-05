    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SameLenWithObjects {

    @Positive
  class SimpleCollection {
    @Positive
    Object[] var_infos;
    @Positive
  }

    @Positive
  static final class Invocation1 {
    @Positive
    SimpleCollection sc;
    @Positive
    Object @SameLen({"vals1", "this.sc.var_infos"}) [] vals1;

    @Positive
    void format1() {
    @Positive
      for (int j = 0; j < vals1.length; j++) {
    @Positive
        System.out.println(sc.var_infos[j]);
    @Positive
      }
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
