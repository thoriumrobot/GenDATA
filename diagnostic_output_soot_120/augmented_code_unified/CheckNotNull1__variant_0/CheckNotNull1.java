    @Positive
public class CheckNotNull1 {
    @Positive
  <T extends Object> T checkNotNull(T ref) {
    @Positive
    return ref;
    @Positive
  }

    @Positive
  <S extends Object> void test(S ref) {
    @Positive
    checkNotNull(ref);
    @Positive
  }
    @Positive
}
