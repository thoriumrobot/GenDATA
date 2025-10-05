    @Positive
public class PreAndPostDec {

    @Positive
  void pre1(int[] args) {
    @Positive
    int ii = 0;
    @Positive
    while ((ii < args.length)) {
      // :: error: (array.access.unsafe.high)
    @Positive
      int m = args[++ii];
    @Positive
    }
    @Positive
  }

    @Positive
  void pre2(int[] args) {
    @Positive
    int ii = 0;
    @Positive
    while ((ii < args.length)) {
    @Positive
      ii++;
      // :: error: (array.access.unsafe.high)
    @Positive
      int m = args[ii];
    @Positive
    }
    @Positive
  }

    @Positive
  void post1(int[] args) {
    @Positive
    int ii = 0;
    @Positive
    while ((ii < args.length)) {
    @Positive
      int m = args[ii++];
    @Positive
    }
    @Positive
  }

    @Positive
  void post2(int[] args) {
    @Positive
    int ii = 0;
    @Positive
    while ((ii < args.length)) {
    @Positive
      int m = args[ii];
    @Positive
      ii++;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
