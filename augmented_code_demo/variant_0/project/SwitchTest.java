    @Positive
import org.checkerframework.common.value.qual.IntVal;

    @Positive
public class SwitchTest {

    @Positive
  public String findSlice_unordered(String[] vis) {
    @Positive
    switch (vis.length) {
    @Positive
      case 1:
    @Positive
        @IntVal(1) int x = vis.length;
    @Positive
        return vis[0];
    @Positive
      case 2:
    @Positive
        return vis[0] + vis[1];
    @Positive
      case 3:
    @Positive
        return vis[0] + vis[1] + vis[2];
    @Positive
      default:
    @Positive
        throw new RuntimeException("Bad length " + vis.length);
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
