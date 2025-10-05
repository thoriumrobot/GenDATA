    @Positive
public class SwitchDataflowRefinement {

    @Positive
  void readInfo(String[] parts) {

    @Positive
    if (parts.length >= 1) {
    @Positive
      Integer.parseInt(parts[0]);
    @Positive
    }

    @Positive
    switch (parts.length) {
    @Positive
      case 1:
    @Positive
        Integer.parseInt(parts[0]);
    @Positive
        break;
    @Positive
    }

    @Positive
    switch (parts.length) {
    @Positive
      case 0:
    @Positive
        break;
    @Positive
      default:
    @Positive
        Integer.parseInt(parts[0]);
    @Positive
        break;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
