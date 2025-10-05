/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import java.lang.reflect.Array;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class ReflectArray {

    @Positive
  void testNewInstance(int i) {
    // :: error: (argument)
    @Positive
    Array.newInstance(Object.class, i);
    @Positive
    if (i >= 0) {
    @Positive
      Array.newInstance(Object.class, i);
    @Positive
    }
    @Positive
  }

    @Positive
  void testFor(Object a) {
    @Positive
    for (int i = 0; i < Array.getLength(a); ++i) {
    @Positive
      Array.setInt(a, i, 1 + Array.getInt(a, i));
    @Positive
    }
    @Positive
  }

    @Positive
  void testMinLen(Object @MinLen(1) [] a) {
    @Positive
    Array.get(a, 0);
    // :: error: (argument)
    @Positive
    Array.get(a, 1);
    @Positive
  }
    @Positive
}
