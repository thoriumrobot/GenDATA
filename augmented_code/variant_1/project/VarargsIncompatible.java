/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
public class VarargsIncompatible {

    @Positive
  public static void test(int[] arr) {
    @Positive
    help(arr);
    @Positive
  }

    @Positive
  public static <T> void help(T... arr) {}
    @Positive
}
