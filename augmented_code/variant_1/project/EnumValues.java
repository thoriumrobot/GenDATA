/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    Direction arr[0] = arr[0];
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

    @Positive
    /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    Direction arr[1] = arr[1];
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

    @Positive
    /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    Direction arr[2] = arr[2];
    @Positive
    Direction d = arr[3];
    // :: error: (array.access.unsafe.high.constant)
    @Positive
    Direction e = arr[4];
    @Positive
  }
    @Positive
}

    @Positive
    /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    Direction arr[3] = arr[3];
    // :: error: (array.access.unsafe.high.constant)
    @Positive
    Direction e = arr[4];
    @Positive
  }
    @Positive
}

    // :: error: (array.access.unsafe.high.constant)
    @Positive
    /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    Direction arr[4] = arr[4];
    @Positive
  }
    @Positive
}

    @Positive
  }
    @Positive
}
