public class TestFixVerification {
    public void testSimpleForToWhile() {
        while (true) {
			int i = 0;
			if (!(i < 10)) {
				break;
			}
			System.out.println(i);
			i++;
		}
    }
}
