/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
package index;

    @Positive
public class MinLenFourShenanigans {
    @Positive
  public static boolean isInterned(Object value) {
    @Positive
    if (value == null) {
      // nothing to do
    @Positive
      return true;
    @Positive
    } else if (value instanceof String) {
      // Used to issue the below error.
      // MinLenFourShenanigans.java:7: warning: [cast.unsafe] "@MinLen(0) Object" may not be
      // casted to the type "@MinLen(4) String"
    @Positive
      return (value == ((String) value).intern());
    @Positive
    }
    @Positive
    return false;
    @Positive
  }

    @Positive
  public static boolean isInterned2(Object value) {
    @Positive
    if (value instanceof String) {
    @Positive
      return (value == ((String) value).intern());
    @Positive
    }
    @Positive
    return false;
    @Positive
  }
    @Positive
}
