    @Positive
public class ComputeConst {

    @Positive
  public static int hash(long l) {
    // If possible, use the value itself.
    @Positive
    if (l >= Integer.MIN_VALUE && l <= Integer.MAX_VALUE) {
    @Positive
      return (int) l;
    @Positive
    }

    @Positive
    return Long.hashCode(l);
    @Positive
  }
    @Positive
}
