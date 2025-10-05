    @Positive
import java.util.Arrays;
    @Positive
import java.util.List;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

// @skip-test until we bring list support back

    @Positive
public class ArrayAsList {

    @Positive
  public static void toList(Integer @MinLen(10) [] arg) {
    @Positive
    System.out.println("Integer: " + list.size());
    @Positive
  }

    @Positive
  public static void toList2(int @MinLen(10) [] arg2) {
    // :: error: (assignment)
    @Positive
    System.out.println("int: " + list.size());

    @Positive
  }

    @Positive
  public static void toList3() {
    @Positive
    System.out.println("args: " + list.size());
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
