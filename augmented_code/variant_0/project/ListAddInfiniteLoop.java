/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// @skip-test until we bring list support back

    @Positive
public class ListAddInfiniteLoop {

    @Positive
  void ListLoop(List<Integer> list) {
    @Positive
    while (true) {
    @Positive
      list.add(4);
    @Positive
    }
    @Positive
  }
    @Positive
}
