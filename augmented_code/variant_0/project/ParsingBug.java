/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
public class ParsingBug {
    @Positive
  void test() {
    @Positive
    String[] saOrig = new String[] {"foo", "bar"};
    @Positive
    Object o1 = do_things((Object) saOrig);
    @Positive
  }

    @Positive
  Object do_things(Object o) {
    @Positive
    return o;
    @Positive
  }
    @Positive
}
