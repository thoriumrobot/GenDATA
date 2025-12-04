    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class MinLenIndexFor {
    @Positive
  int @MinLen(2) [] arrayLen2 = {0, 1, 2};

    @Positive
  void test(@IndexFor("this.arrayLen2") int i) {
    @Positive
    int j = arrayLen2[i];
    @Positive
    int j2 = arrayLen2[1];
    @Positive
  }

    @Positive
  void callTest(int x) {
    @Positive
    test(0);
    @Positive
    test(1);
    // :: error: (argument)
    @Positive
    test(2);
    // :: error: (argument)
    @Positive
    test(3);
    @Positive
    test(arrayLen2.length - 1);
    @Positive
  }

    @Positive
  int @MinLen(4) [] arrayLen4 = {0, 1, 2, 4, 5};

    @Positive
  void test2(@IndexOrHigh("this.arrayLen4") int i) {
    @Positive
    if (i > 0) {
    @Positive
      int j = arrayLen4[i - 1];
    @Positive
    }
    @Positive
    int j2 = arrayLen4[1];
    @Positive
  }

    @Positive
  void callTest2(int x) {
    @Positive
    test2(0);
    @Positive
    test2(1);
    @Positive
    test2(2);
    @Positive
    test2(4);
    // :: error: (argument)
    @Positive
    test2(5);
    @Positive
    test2(arrayLen4.length);
    @Positive
  }
    @Positive
}
