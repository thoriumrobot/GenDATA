/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.common.value.qual.IntVal;

    @Positive
public class SwitchTest {

    @Positive
  public String findSlice_unordered(String[] vis) {
    @Positive
    if (vis.length == 1) {
            @Positive
@IntVal(1) int x = vis.length;
@Positive
return vis[0];
@Positive
        } else if (vis.length == 2) {
            @Positive
return vis[0] + vis[1];
@Positive
        } else if (vis.length == 3) {
            @Positive
return vis[0] + vis[1] + vis[2];
@Positive
        } else else {
            @Positive
throw new RuntimeException("Bad length " + vis.length);
@Positive
        }
    @Positive
  }
    @Positive
}
