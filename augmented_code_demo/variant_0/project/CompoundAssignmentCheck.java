    @Positive
public class CompoundAssignmentCheck {
    @Positive
  void test() {
    @Positive
    int a = 9;
    @Positive
    a += 5;
    @Positive
    a -= 2;
    @Positive
    int[] arr5 = new int[a]; // LBC shouldn't warn here.
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
