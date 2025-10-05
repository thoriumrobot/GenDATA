    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class EnumValues {

    @Positive
  public static enum Direction {
    @Positive
    NORTH,
    @Positive
    SOUTH,
    @Positive
    EAST,
    @Positive
    WEST
    @Positive
  };

    @Positive
  public static void enumValues() {
    @Positive
    Direction @ArrayLen(4) [] arr4 = Direction.values();
    @Positive
    Direction[] arr = Direction.values();
    @Positive
    Direction a = arr[0];
    @Positive
    Direction b = arr[1];
    @Positive
    Direction c = arr[2];
    @Positive
    Direction d = arr[3];
    // :: error: (array.access.unsafe.high.constant)
    @Positive
    Direction e = arr[4];
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
