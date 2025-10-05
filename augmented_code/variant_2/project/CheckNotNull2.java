/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
public class CheckNotNull2<T extends Object> {
    @Positive
  T checkNotNull(T ref) {
    @Positive
    return ref;
    @Positive
  }

    @Positive
  void test(T ref) {
    @Positive
    checkNotNull(ref);
    @Positive
  }
    @Positive
}
