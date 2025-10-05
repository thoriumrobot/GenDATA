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

// CFWR semantic augmentation - variant 1
